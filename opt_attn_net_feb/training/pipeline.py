from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import gc
import shutil

import numpy as np
import pandas as pd
import torch
import optuna
from optuna.trial import Trial

from ..data.datasets import MILTrainDataset, MILExportDataset
from ..data.collate import collate_train, collate_export
from ..data.exports import export_leaderboard_attention
from .builders import DataLoaderBuilder, LoaderConfig, MILModelBuilder
from .configs import HPOConfig
from .loss_config import compute_gamma, compute_lam, compute_posw_clips
from .search_space import search_space
from .trainer import make_trainer_gpu, eval_best_epoch
from ..utils.ops import (
    set_all_seeds,
    pos_weight_per_task,
    fit_standardizer,
    apply_standardizer,
    make_weighted_sampler,
)
from ..utils.data_io import align_by_id


def objective_mil_cv(
    trial: Trial,
    X2d_scaled: np.ndarray,
    y_cls: np.ndarray,
    w_cls: np.ndarray,
    y_abs: np.ndarray,
    m_abs: np.ndarray,
    w_abs: np.ndarray,
    y_fluo: np.ndarray,
    m_fluo: np.ndarray,
    w_fluo: np.ndarray,
    ids: List[str],
    folds_info,
    starts: np.ndarray,
    counts: np.ndarray,
    id2pos: Dict[str, int],
    Xinst_sorted: np.ndarray,
    seed: int,
    max_epochs: int,
    patience: int,
    accelerator: str,
    devices: int,
    num_workers: int,
    pin_memory: bool,
    precision: str,
    ckpt_root: Path,
) -> float:
    p = search_space(trial)
    cfg = HPOConfig.from_params(p)
    max_instances = 0

    scores: List[float] = []
    fold_detail: Dict[str, Any] = {}

    # Objective is fixed to macro_plus_min to jointly optimize mean AP and weakest task AP.
    objective_mode = str(cfg.objective.mode)
    min_w = float(cfg.objective.min_w)

    device = torch.device("cuda" if torch.cuda.is_available() and accelerator in ("gpu", "cuda") else "cpu")
    loader_builder = DataLoaderBuilder(LoaderConfig(num_workers=int(num_workers), pin_memory=bool(pin_memory)))

    for step, (tr, va, f) in enumerate(folds_info):
        set_all_seeds(seed + 5000 * f + trial.number)

        lam = compute_lam(cfg.loss, y_train=y_cls[tr])
        posw = pos_weight_per_task(y_cls[tr], clip=compute_posw_clips(cfg.loss))
        gamma_t = compute_gamma(cfg.loss)

        # aux standardization fit on TRAIN indices only
        mu_abs, sd_abs = fit_standardizer(y_abs, m_abs, tr)
        mu_f, sd_f = fit_standardizer(y_fluo, m_fluo, tr)
        y_abs_sc = apply_standardizer(y_abs, mu_abs, sd_abs)
        y_fluo_sc = apply_standardizer(y_fluo, mu_f, sd_f)

        ids_tr = [ids[i] for i in tr]
        ids_va = [ids[i] for i in va]

        ds_tr = MILTrainDataset(
            ids_tr,
            X2d_scaled[tr],
            y_cls[tr],
            w_cls[tr],
            y_abs_sc[tr],
            m_abs[tr],
            w_abs[tr],
            y_fluo_sc[tr],
            m_fluo[tr],
            w_fluo[tr],
            starts,
            counts,
            id2pos,
            Xinst_sorted,
            max_instances=max_instances,
            seed=seed + f,
        )
        ds_va = MILTrainDataset(
            ids_va,
            X2d_scaled[va],
            y_cls[va],
            w_cls[va],
            y_abs_sc[va],
            m_abs[va],
            w_abs[va],
            y_fluo_sc[va],
            m_fluo[va],
            w_fluo[va],
            starts,
            counts,
            id2pos,
            Xinst_sorted,
            max_instances=max_instances,
            seed=seed + 999 + f,
        )

        sampler = make_weighted_sampler(
            y_cls[tr],
            rare_mult=float(cfg.sampler.rare_oversample_mult),
            rare_prev_thr=float(cfg.sampler.rare_prev_thr),
            sample_weight_cap=float(cfg.sampler.sample_weight_cap),
        )

        dl_tr = loader_builder.train_loader(
            ds_tr,
            batch_size=int(cfg.runtime.batch_size),
            sampler=sampler,
            collate_fn=collate_train,
        )
        dl_va = loader_builder.eval_loader(
            ds_va,
            batch_size=min(128, int(cfg.runtime.batch_size)),
            collate_fn=collate_train,
        )

        model = MILModelBuilder.build(
            config=cfg,
            mol_dim=int(X2d_scaled.shape[1]),
            inst_dim=int(Xinst_sorted.shape[1]),
            pos_weight=posw,
            gamma=gamma_t,
            lam=lam,
        )

        fold_ckpt_dir = ckpt_root / f"mil_trial{trial.number}_fold{f}"
        fold_ckpt_dir.mkdir(parents=True, exist_ok=True)

        trainer, ckpt_cb = make_trainer_gpu(
            max_epochs=max_epochs,
            patience=patience,
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            accumulate_grad_batches=int(cfg.runtime.accumulate_grad_batches),
            ckpt_dir=str(fold_ckpt_dir),
            trial=trial,
        )
        trainer.fit(model, dl_tr, dl_va)

        epochs_trained = int(trainer.current_epoch) + 1

        best_path = ckpt_cb.best_model_path
        best_epoch = None
        if best_path and Path(best_path).exists():
            ckpt = torch.load(best_path, map_location="cpu")
            best_epoch = int(ckpt.get("epoch", -1))
            model.load_state_dict(ckpt["state_dict"], strict=True)

        best_macro, best_aps = eval_best_epoch(model, dl_va, device=device)
        best_min = float(np.min(best_aps))
        fold_score = float((1.0 - min_w) * float(best_macro) + min_w * best_min)

        print(
            f"[MIL-TASK-ATTN] trial={trial.number} fold={f} trained_epochs={epochs_trained} best_epoch={best_epoch} "
            f"best_macro_ap={best_macro:.6f} min_ap={best_min:.6f} aps={best_aps} score={fold_score:.6f} mode={objective_mode} min_w={min_w:.2f}"
        )

        scores.append(fold_score)
        fold_detail[str(f)] = {
            "trained_epochs": epochs_trained,
            "best_epoch": best_epoch,
            "macro_ap_best_epoch": float(best_macro),
            "min_ap_best_epoch": best_min,
            "score": fold_score,
            "objective_mode": objective_mode,
            "min_w": float(min_w),
            "ap_task0": float(best_aps[0]),
            "ap_task1": float(best_aps[1]),
            "ap_task2": float(best_aps[2]),
            "ap_task3": float(best_aps[3]),
            "accumulate_grad_batches": int(cfg.runtime.accumulate_grad_batches),
        }

        try:
            shutil.rmtree(fold_ckpt_dir, ignore_errors=True)
        except Exception:
            pass

        del trainer, model, dl_tr, dl_va, ds_tr, ds_va, sampler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        trial.report(float(np.mean(scores)), step=step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    trial.set_user_attr("fold_detail", fold_detail)
    return float(np.mean(scores))


def save_study_artifacts(outdir: Path, study: optuna.Study, prefix: str):
    df_trials = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
    df_trials.to_csv(outdir / f"{prefix}_trials.csv", index=False)
    best = dict(study.best_params)
    best["best_value_macro_ap_cv"] = float(study.best_value)
    (outdir / f"{prefix}_best_params.json").write_text(json.dumps(best, indent=2))


def save_best_fold_metrics(outdir: Path, prefix: str, fold_metrics: Dict[str, Any]):
    (outdir / f"{prefix}_best_fold_metrics.json").write_text(json.dumps(fold_metrics, indent=2))


def drop_ids_without_bags(
    ids: List[str],
    X2d: np.ndarray,
    df_part: pd.DataFrame,
    id2pos: Dict[str, int],
) -> Tuple[List[str], np.ndarray, pd.DataFrame]:
    mask = np.array([(i in id2pos) for i in ids], dtype=bool)
    if mask.all():
        return ids, X2d, df_part
    df2 = df_part.loc[mask].reset_index(drop=True)
    ids2 = df2["ID"].astype(str).tolist()
    return ids2, X2d[mask], df2


def train_best_and_export(
    outdir: Path,
    df_full: pd.DataFrame,
    best_params: Dict[str, Any],
    args,
    X2d_file_ids: List[str],
    X2d_file: np.ndarray,
    starts: np.ndarray,
    counts: np.ndarray,
    id2pos: Dict[str, int],
    Xinst_sorted: np.ndarray,
    conf_sorted: np.ndarray,
    num_workers: int,
    pin_memory: bool,
):
    cfg = HPOConfig.from_params(
        best_params,
        fallback_lambda_power=1.0,
        fallback_lam_floor=0.25,
        fallback_lam_ceil=6.0,
        fallback_pos_weight_clip=50.0,
    )

    df_tr = df_full[df_full[args.split_col] == "train"].copy().reset_index(drop=True)
    df_lb = df_full[df_full[args.split_col] == args.leaderboard_split].copy().reset_index(drop=True)
    if len(df_lb) == 0:
        raise ValueError(f"No rows with split == '{args.leaderboard_split}'")

    ids_tr = df_tr[args.id_col].astype(str).tolist()
    ids_lb = df_lb[args.id_col].astype(str).tolist()

    X2d_tr = align_by_id(X2d_file_ids, X2d_file, ids_tr)
    X2d_lb = align_by_id(X2d_file_ids, X2d_file, ids_lb)

    # drop IDs without bags
    ids_tr, X2d_tr, df_tr = drop_ids_without_bags(ids_tr, X2d_tr, df_tr.rename(columns={args.id_col: "ID"}), id2pos)
    ids_lb, X2d_lb, df_lb = drop_ids_without_bags(ids_lb, X2d_lb, df_lb.rename(columns={args.id_col: "ID"}), id2pos)

    # targets/weights
    from ..utils.ops import coerce_binary_labels, build_task_weights, build_aux_targets_and_masks, build_aux_weights

    y_tr = coerce_binary_labels(df_tr)
    w_tr = build_task_weights(df_tr)
    y_abs_tr, m_abs_tr, y_fluo_tr, m_fluo_tr = build_aux_targets_and_masks(df_tr)
    w_abs_tr, w_fluo_tr = build_aux_weights(df_tr)

    y_lb = coerce_binary_labels(df_lb)
    w_lb = build_task_weights(df_lb)
    y_abs_lb, m_abs_lb, y_fluo_lb, m_fluo_lb = build_aux_targets_and_masks(df_lb)
    w_abs_lb, w_fluo_lb = build_aux_weights(df_lb)

    # aux standardization fit on train
    tr_idx = np.arange(len(df_tr), dtype=np.int64)
    mu_abs, sd_abs = fit_standardizer(y_abs_tr, m_abs_tr, tr_idx)
    mu_f, sd_f = fit_standardizer(y_fluo_tr, m_fluo_tr, tr_idx)
    y_abs_tr_sc = apply_standardizer(y_abs_tr, mu_abs, sd_abs)
    y_abs_lb_sc = apply_standardizer(y_abs_lb, mu_abs, sd_abs)
    y_fluo_tr_sc = apply_standardizer(y_fluo_tr, mu_f, sd_f)
    y_fluo_lb_sc = apply_standardizer(y_fluo_lb, mu_f, sd_f)

    # loss weights
    lam = compute_lam(
        cfg.loss,
        y_train=y_tr,
        fallback_lambda_power=1.0,
        fallback_lam_floor=0.25,
        fallback_lam_ceil=6.0,
    )
    posw = pos_weight_per_task(y_tr, clip=compute_posw_clips(cfg.loss, fallback_clip=50.0))
    gamma_t = compute_gamma(cfg.loss)

    # datasets + loaders
    ds_tr = MILTrainDataset(
        ids_tr,
        X2d_tr,
        y_tr,
        w_tr,
        y_abs_tr_sc,
        m_abs_tr,
        w_abs_tr,
        y_fluo_tr_sc,
        m_fluo_tr,
        w_fluo_tr,
        starts,
        counts,
        id2pos,
        Xinst_sorted,
        max_instances=0,
        seed=int(args.seed),
    )
    ds_lb = MILTrainDataset(
        ids_lb,
        X2d_lb,
        y_lb,
        w_lb,
        y_abs_lb_sc,
        m_abs_lb,
        w_abs_lb,
        y_fluo_lb_sc,
        m_fluo_lb,
        w_fluo_lb,
        starts,
        counts,
        id2pos,
        Xinst_sorted,
        max_instances=0,
        seed=int(args.seed) + 999,
    )

    sampler_tr = make_weighted_sampler(
        y_tr,
        rare_mult=float(cfg.sampler.rare_oversample_mult),
        rare_prev_thr=float(cfg.sampler.rare_prev_thr),
        sample_weight_cap=float(cfg.sampler.sample_weight_cap),
    )

    loader_builder = DataLoaderBuilder(LoaderConfig(num_workers=int(num_workers), pin_memory=bool(pin_memory)))
    dl_tr = loader_builder.train_loader(
        ds_tr,
        batch_size=int(cfg.runtime.batch_size),
        sampler=sampler_tr,
        collate_fn=collate_train,
    )
    dl_val = loader_builder.eval_loader(
        ds_lb,
        batch_size=min(128, int(cfg.runtime.batch_size)),
        collate_fn=collate_train,
    )

    model = MILModelBuilder.build(
        config=cfg,
        mol_dim=int(X2d_tr.shape[1]),
        inst_dim=int(Xinst_sorted.shape[1]),
        pos_weight=posw,
        gamma=gamma_t,
        lam=lam,
    )

    final_dir = outdir / "final_best_train_vs_leaderboard"
    final_dir.mkdir(parents=True, exist_ok=True)

    trainer, ckpt_cb = make_trainer_gpu(
        max_epochs=int(args.max_epochs),
        patience=int(args.patience),
        accelerator=str(args.nn_accelerator),
        devices=int(args.nn_devices),
        precision=str(args.precision),
        accumulate_grad_batches=int(cfg.runtime.accumulate_grad_batches),
        ckpt_dir=str(final_dir),
        trial=None,
    )
    trainer.fit(model, dl_tr, dl_val)

    best_path = ckpt_cb.best_model_path
    if best_path and Path(best_path).exists():
        ckpt = torch.load(best_path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"], strict=True)
        print(f"[FINAL] loaded best ckpt: {best_path}")

    # Evaluate best checkpoint on leaderboard (validation) split
    dev_eval = torch.device("cuda" if torch.cuda.is_available() and str(args.nn_accelerator) in ("gpu", "cuda") else "cpu")
    macro_ap_lb, aps_lb = eval_best_epoch(model, dl_val, device=dev_eval)
    eval_json = {
        "macro_ap": float(macro_ap_lb),
        "ap_task0": float(aps_lb[0]),
        "ap_task1": float(aps_lb[1]),
        "ap_task2": float(aps_lb[2]),
        "ap_task3": float(aps_lb[3]),
    }
    (final_dir / "leaderboard_eval.json").write_text(json.dumps(eval_json, indent=2))
    print(f"[FINAL] leaderboard eval: macro_ap={macro_ap_lb:.6f} aps={aps_lb}")

    # export attention weights on leaderboard
    export_ds = MILExportDataset(
        ids_lb,
        X2d_lb,
        starts=starts,
        counts=counts,
        id2pos=id2pos,
        Xinst_sorted=Xinst_sorted,
        conf_sorted=conf_sorted,
        max_instances=0,
        seed=int(args.seed) + 123,
    )
    export_dl = loader_builder.eval_loader(
        export_ds,
        batch_size=min(64, int(cfg.runtime.batch_size)),
        collate_fn=collate_export,
    )

    dev = torch.device("cuda" if torch.cuda.is_available() and str(args.nn_accelerator) in ("gpu", "cuda") else "cpu")
    out_path = Path(args.attn_out) if args.attn_out else (outdir / "leaderboard_attn.parquet")
    export_leaderboard_attention(model, export_dl, device=dev, out_path=out_path)


__all__ = [
    "objective_mil_cv",
    "save_study_artifacts",
    "save_best_fold_metrics",
    "drop_ids_without_bags",
    "train_best_and_export",
]
