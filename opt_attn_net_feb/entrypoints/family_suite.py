from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import gc
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from ..data.collate import collate_train
from ..data.datasets import MILTrainDataset
from ..training.builders import DataLoaderBuilder, LoaderConfig, MILModelBuilder
from ..training.configs import HPOConfig
from ..training.execution import (
    CVRunConfig,
    MILCVData,
    MILFoldTrainer,
    StudyConfig,
    TrainerSystemConfig,
    _resolve_device,
)
from ..training.loss_config import compute_gamma, compute_lam, compute_posw_clips
from ..training.search_space import search_space
from ..training.trainer import LightningTrainerConfig, LightningTrainerFactory, ModelEvaluator
from ..utils.constants import TASK_COLS
from ..utils.data_io import align_by_id, load_2d, load_labels
from ..utils.instances import build_instance_index, load_and_merge_instances
from ..utils.metrics import ap_per_task, roc_auc_per_task
from ..utils.ops import (
    apply_standardizer,
    build_aux_targets_and_masks,
    build_aux_weights,
    build_bitmask_group_definition,
    build_task_weights,
    coerce_binary_labels,
    fit_standardizer,
    fold_indices,
    make_balanced_batch_sampler,
    make_bitmask_sample_weights,
    make_weighted_sampler,
    maybe_set_torch_fast_flags,
    pos_weight_per_task,
    set_all_seeds,
)
from ..utils.progress import log_event, log_step

FAMILY_CHOICES: tuple[str, ...] = ("catboost_st", "mt_2d", "mt_2d3d", "mt_3d")


@dataclass(frozen=True)
class FamilySuiteConfig:
    labels: str
    feat2d_scaled: str
    feat2d_raw: str | None
    feat3d_scaled: str
    feat3d_qm_scaled: str
    study_dir: str
    id_col: str
    conf_col: str
    split_col: str
    fold_col: str
    use_splits: tuple[str, ...]
    leaderboard_split: str
    max_epochs: int
    patience: int
    trials: int
    seed: int
    nn_accelerator: str
    nn_devices: int
    precision: str
    num_workers: int
    cpu_workers: int
    pin_memory: bool
    run_hpo: bool
    hpo_only: bool
    pruner_warmup_steps: int
    model_families: tuple[str, ...]
    calibration_method: str
    best_params_dir: str | None
    catboost_hpo_parallel_tasks: int


@dataclass(frozen=True)
class _MILFamilyData:
    family: str
    ids_train: List[str]
    ids_lb: List[str]
    X2d_train: np.ndarray
    X2d_lb: np.ndarray
    y_cls_train: np.ndarray
    y_cls_lb: np.ndarray
    w_cls_train: np.ndarray
    w_cls_lb: np.ndarray
    y_abs_train: np.ndarray
    m_abs_train: np.ndarray
    w_abs_train: np.ndarray
    y_abs_lb: np.ndarray
    m_abs_lb: np.ndarray
    w_abs_lb: np.ndarray
    y_fluo_train: np.ndarray
    m_fluo_train: np.ndarray
    w_fluo_train: np.ndarray
    y_fluo_lb: np.ndarray
    m_fluo_lb: np.ndarray
    w_fluo_lb: np.ndarray
    folds_info: Sequence[Tuple[np.ndarray, np.ndarray, int]]
    starts: np.ndarray
    counts: np.ndarray
    id2pos: Dict[str, int]
    Xinst_sorted: np.ndarray
    conf_sorted: np.ndarray
    inst_geom_dim: int
    inst_qm_dim: int


def _clip_prob(p: np.ndarray) -> np.ndarray:
    return np.clip(np.nan_to_num(np.asarray(p, dtype=np.float64), nan=0.5, posinf=1.0, neginf=0.0), 1e-6, 1 - 1e-6)


def _df_by_ids(df: pd.DataFrame, *, id_col: str, ids: Sequence[str]) -> pd.DataFrame:
    # Keep first row for duplicate IDs to preserve one-label-per-ID contract.
    d = df.drop_duplicates(subset=[id_col], keep="first").copy()
    d[id_col] = d[id_col].astype(str)
    pos = {str(x): i for i, x in enumerate(d[id_col].tolist())}
    idx = [pos[str(i)] for i in ids if str(i) in pos]
    return d.iloc[idx].reset_index(drop=True)


def _metric_table(
    *,
    y_true: np.ndarray,
    p_pred: np.ndarray,
    w_cls: np.ndarray,
) -> pd.DataFrame:
    aps = ap_per_task(y_true, p_pred, w_cls=w_cls, weighted_tasks=(0, 1))
    aucs = roc_auc_per_task(y_true, p_pred, w_cls=w_cls, weighted_tasks=(0, 1))
    rows: List[Dict[str, Any]] = []
    nlls: List[float] = []
    briers: List[float] = []
    for t, task in enumerate(TASK_COLS):
        yt = y_true[:, t].astype(int)
        pt = _clip_prob(p_pred[:, t])
        sw = np.asarray(w_cls[:, t], dtype=float) if t in (0, 1) else None
        try:
            nll = float(log_loss(yt, pt, sample_weight=sw, labels=[0, 1]))
        except Exception:
            nll = float("nan")
        try:
            brier = float(brier_score_loss(yt, pt, sample_weight=sw))
        except Exception:
            brier = float("nan")
        nlls.append(nll)
        briers.append(brier)
        rows.append(
            {
                "task": str(task),
                "pr_auc": float(aps[t]),
                "roc_auc": float(aucs[t]),
                "nll": nll,
                "brier": brier,
            }
        )
    rows.append(
        {
            "task": "macro",
            "pr_auc": float(np.mean(aps)),
            "roc_auc": float(np.mean(aucs)),
            "nll": float(np.nanmean(nlls)),
            "brier": float(np.nanmean(briers)),
        }
    )
    return pd.DataFrame(rows)


def _save_family_best_params(*, outdir: Path, family: str, best_params: Mapping[str, Any], best_value: float) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{family}.json"
    payload = {
        "family": str(family),
        "best_params": dict(best_params),
        "best_value_macro_ap_cv": float(best_value),
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def _save_catboost_task_best_params(
    *,
    outdir: Path,
    task_idx: int,
    best_params: Mapping[str, Any],
    best_value: float,
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"catboost_st_t{int(task_idx)}.json"
    payload = {
        "family": "catboost_st",
        "task_idx": int(task_idx),
        "task_name": str(TASK_COLS[int(task_idx)]),
        "best_params": dict(best_params),
        "best_value_pr_auc_cv": float(best_value),
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def _load_catboost_task_best_params(*, outdir: Path, task_idx: int) -> Dict[str, Any]:
    path = outdir / f"catboost_st_t{int(task_idx)}.json"
    if not path.exists():
        raise FileNotFoundError(f"CatBoost best params file not found for task {task_idx}: {path}")
    payload = json.loads(path.read_text())
    if isinstance(payload, dict) and isinstance(payload.get("best_params"), dict):
        return dict(payload["best_params"])
    if isinstance(payload, dict):
        return dict(payload)
    raise ValueError(
        f"Invalid CatBoost best params JSON for task {task_idx}: expected object, got {type(payload).__name__}"
    )


def _load_family_best_params(*, outdir: Path, family: str) -> Dict[str, Any]:
    path = outdir / f"{family}.json"
    if not path.exists():
        raise FileNotFoundError(f"Best params file not found for family '{family}': {path}")
    payload = json.loads(path.read_text())
    if isinstance(payload, dict) and isinstance(payload.get("best_params"), dict):
        return dict(payload["best_params"])
    if isinstance(payload, dict):
        return dict(payload)
    raise ValueError(f"Invalid best params JSON for '{family}': expected object, got {type(payload).__name__}")


class _MILMacroCrossValidator:
    def __init__(self, *, data: MILCVData, run_config: CVRunConfig):
        self.data = data
        self.run_config = run_config

    def evaluate_trial(self, trial: optuna.Trial) -> float:
        log_event("START", "family.hpo.trial.evaluate", trial=int(trial.number))
        params = search_space(trial)
        # Keep fixed objective weighting across MIL-family trials.
        params["min_w"] = 0.4
        cfg = HPOConfig.from_params(params)

        fold_runner = MILFoldTrainer(
            trial=trial,
            hpo_config=cfg,
            data=self.data,
            run_config=self.run_config,
        )
        fold_scores: List[float] = []
        fold_detail: Dict[str, Any] = {}
        for step, (tr, va, fold_id) in enumerate(self.data.folds_info):
            _unused_score, detail = fold_runner.run_fold(
                train_idx=np.asarray(tr, dtype=np.int64),
                val_idx=np.asarray(va, dtype=np.int64),
                fold_id=int(fold_id),
            )
            fold_macro = float(detail.get("macro_pr_auc_best_epoch", detail.get("macro_ap_best_epoch", 0.0)))
            fold_scores.append(fold_macro)
            detail["objective_macro_ap"] = float(fold_macro)
            fold_detail[str(fold_id)] = detail
            trial.report(float(np.mean(fold_scores)), step=int(step))
            if trial.should_prune():
                raise optuna.TrialPruned()
        mean_score = float(np.mean(fold_scores))
        trial.set_user_attr("fold_detail", fold_detail)
        log_event("DONE", "family.hpo.trial.evaluate", trial=int(trial.number), mean_score=f"{mean_score:.6f}")
        return mean_score


def _catboost_search_space(trial: optuna.Trial, *, task_idx: int) -> Dict[str, Any]:
    # Prevalence-aware clipping ranges for scale_pos_weight:
    # t0~5.6%, t1~1.5%, t2~16.7%, t3~0.24%.
    if int(task_idx) == 0:
        posw_lo, posw_hi = 8.0, 35.0
    elif int(task_idx) == 1:
        posw_lo, posw_hi = 30.0, 120.0
    elif int(task_idx) == 2:
        posw_lo, posw_hi = 2.0, 12.0
    else:  # t3
        posw_lo, posw_hi = 120.0, 500.0
    p = {
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-2, 2e-1, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 2, 80),
        "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
        "rsm": trial.suggest_float("rsm", 0.5, 1.0),
        "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli"]),
        "pos_weight_clip": trial.suggest_float("pos_weight_clip", posw_lo, posw_hi, log=True),
    }
    if p["bootstrap_type"] == "Bayesian":
        p["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 10.0)
    else:
        p["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
    return p


def _catboost_common_params(*, params: Mapping[str, Any], seed: int, threads: int, scale_pos_weight: float) -> Dict[str, Any]:
    cb = dict(
        iterations=4000,
        loss_function="Logloss",
        eval_metric="PRAUC",
        depth=int(params["depth"]),
        learning_rate=float(params["learning_rate"]),
        l2_leaf_reg=float(params["l2_leaf_reg"]),
        min_data_in_leaf=int(params["min_data_in_leaf"]),
        random_strength=float(params["random_strength"]),
        rsm=float(params["rsm"]),
        bootstrap_type=str(params["bootstrap_type"]),
        random_seed=int(seed),
        scale_pos_weight=float(scale_pos_weight),
        verbose=False,
        task_type="CPU",
        thread_count=int(max(1, threads)),
        allow_writing_files=False,
    )
    if "bagging_temperature" in params:
        cb["bagging_temperature"] = float(params["bagging_temperature"])
    if "subsample" in params:
        cb["subsample"] = float(params["subsample"])
    return cb


def _calibrate_task(y: np.ndarray, p: np.ndarray, method: str) -> tuple[np.ndarray, Dict[str, Any]]:
    yb = np.asarray(y, dtype=int).reshape(-1)
    pp = _clip_prob(p).reshape(-1)
    if int(np.unique(yb).size) < 2:
        return pp.astype(np.float32), {"kind": "identity_single_class"}
    if method == "isotonic":
        ir = IsotonicRegression(out_of_bounds="clip")
        out = ir.fit_transform(pp, yb)
        return np.asarray(out, dtype=np.float32), {
            "kind": "isotonic",
            "x_thresholds": [float(x) for x in ir.X_thresholds_.tolist()],
            "y_thresholds": [float(x) for x in ir.y_thresholds_.tolist()],
        }
    if method == "temperature":
        logits = np.log(pp / (1.0 - pp))
        best_t = 1.0
        best_nll = float("inf")
        for t in np.exp(np.linspace(np.log(0.25), np.log(4.0), 96)):
            q = 1.0 / (1.0 + np.exp(-logits / float(t)))
            q = _clip_prob(q)
            try:
                nll = float(log_loss(yb, q, labels=[0, 1]))
            except Exception:
                continue
            if nll < best_nll:
                best_nll = nll
                best_t = float(t)
        out = 1.0 / (1.0 + np.exp(-logits / best_t))
        return np.asarray(_clip_prob(out), dtype=np.float32), {"kind": "temperature", "temperature": float(best_t)}
    # default platt
    x = np.log(pp / (1.0 - pp)).reshape(-1, 1)
    lr = LogisticRegression(max_iter=1000, solver="lbfgs")
    lr.fit(x, yb)
    out = lr.predict_proba(x)[:, 1]
    return np.asarray(_clip_prob(out), dtype=np.float32), {
        "kind": "platt",
        "coef": float(lr.coef_[0, 0]),
        "intercept": float(lr.intercept_[0]),
    }


def _run_mil_final_train_and_predict(
    *,
    cfg: FamilySuiteConfig,
    family_data: _MILFamilyData,
    best_params: Mapping[str, Any],
    outdir: Path,
) -> pd.DataFrame:
    set_all_seeds(int(cfg.seed))
    hpo_cfg = HPOConfig.from_params(
        dict(best_params),
        fallback_lambda_power=1.0,
        fallback_lam_floor=0.25,
        fallback_lam_ceil=6.0,
        fallback_pos_weight_clip=50.0,
    )
    lam = compute_lam(hpo_cfg.loss, y_train=family_data.y_cls_train)
    posw = pos_weight_per_task(
        family_data.y_cls_train,
        clip=compute_posw_clips(hpo_cfg.loss),
    )
    gamma_t = compute_gamma(hpo_cfg.loss)

    mu_abs, sd_abs = fit_standardizer(
        np.concatenate([family_data.y_abs_train, family_data.y_abs_lb], axis=0),
        np.concatenate([family_data.m_abs_train, family_data.m_abs_lb], axis=0),
        np.arange(len(family_data.y_abs_train), dtype=np.int64),
    )
    mu_f, sd_f = fit_standardizer(
        np.concatenate([family_data.y_fluo_train, family_data.y_fluo_lb], axis=0),
        np.concatenate([family_data.m_fluo_train, family_data.m_fluo_lb], axis=0),
        np.arange(len(family_data.y_fluo_train), dtype=np.int64),
    )
    y_abs_tr_sc = apply_standardizer(family_data.y_abs_train, mu_abs, sd_abs)
    y_abs_lb_sc = apply_standardizer(family_data.y_abs_lb, mu_abs, sd_abs)
    y_fluo_tr_sc = apply_standardizer(family_data.y_fluo_train, mu_f, sd_f)
    y_fluo_lb_sc = apply_standardizer(family_data.y_fluo_lb, mu_f, sd_f)

    w_cls_tr = np.asarray(family_data.w_cls_train, dtype=np.float32).copy()
    if bool(hpo_cfg.sampler.use_bitmask_loss_weight):
        bitmask_w = make_bitmask_sample_weights(
            family_data.y_cls_train,
            alpha=float(hpo_cfg.sampler.bitmask_weight_alpha),
            cap=float(hpo_cfg.sampler.bitmask_weight_cap),
        )
        w_cls_tr = (w_cls_tr * bitmask_w.reshape(-1, 1)).astype(np.float32)

    ds_tr = MILTrainDataset(
        family_data.ids_train,
        family_data.X2d_train,
        family_data.y_cls_train,
        w_cls_tr,
        y_abs_tr_sc,
        family_data.m_abs_train,
        family_data.w_abs_train,
        y_fluo_tr_sc,
        family_data.m_fluo_train,
        family_data.w_fluo_train,
        family_data.starts,
        family_data.counts,
        family_data.id2pos,
        family_data.Xinst_sorted,
        max_instances=0,
        seed=int(cfg.seed) + 7,
    )
    ds_lb = MILTrainDataset(
        family_data.ids_lb,
        family_data.X2d_lb,
        family_data.y_cls_lb,
        family_data.w_cls_lb,
        y_abs_lb_sc,
        family_data.m_abs_lb,
        family_data.w_abs_lb,
        y_fluo_lb_sc,
        family_data.m_fluo_lb,
        family_data.w_fluo_lb,
        family_data.starts,
        family_data.counts,
        family_data.id2pos,
        family_data.Xinst_sorted,
        max_instances=0,
        seed=int(cfg.seed) + 17,
    )

    loader_builder = DataLoaderBuilder(
        LoaderConfig(num_workers=int(cfg.num_workers), pin_memory=bool(cfg.pin_memory and torch.cuda.is_available()))
    )
    if bool(hpo_cfg.sampler.use_balanced_batch_sampler):
        batch_sampler = make_balanced_batch_sampler(
            family_data.y_cls_train,
            batch_size=int(hpo_cfg.runtime.batch_size),
            rare_mult=float(hpo_cfg.sampler.rare_oversample_mult),
            rare_target_prev=float(hpo_cfg.sampler.rare_target_prev),
            sample_weight_cap=float(hpo_cfg.sampler.sample_weight_cap),
            batch_pos_fraction=float(hpo_cfg.sampler.batch_pos_fraction),
            min_pos_per_batch=int(hpo_cfg.sampler.min_pos_per_batch),
            enforce_bitmask_quota=bool(hpo_cfg.sampler.enforce_bitmask_quota),
            quota_t450_per_256=int(hpo_cfg.sampler.quota_t450_per_256),
            quota_fgt480_per_256=int(hpo_cfg.sampler.quota_fgt480_per_256),
            quota_multi_per_256=int(hpo_cfg.sampler.quota_multi_per_256),
            rare_prev_thr=hpo_cfg.sampler.rare_prev_thr,
            seed=int(cfg.seed) + 1000,
        )
        dl_tr = loader_builder.train_loader(
            ds_tr,
            batch_size=int(hpo_cfg.runtime.batch_size),
            batch_sampler=batch_sampler,
            collate_fn=collate_train,
        )
    else:
        sampler = make_weighted_sampler(
            family_data.y_cls_train,
            rare_mult=float(hpo_cfg.sampler.rare_oversample_mult),
            rare_target_prev=float(hpo_cfg.sampler.rare_target_prev),
            sample_weight_cap=float(hpo_cfg.sampler.sample_weight_cap),
            rare_prev_thr=hpo_cfg.sampler.rare_prev_thr,
        )
        dl_tr = loader_builder.train_loader(
            ds_tr,
            batch_size=int(hpo_cfg.runtime.batch_size),
            sampler=sampler,
            collate_fn=collate_train,
        )
    dl_lb = loader_builder.eval_loader(
        ds_lb,
        batch_size=min(128, int(hpo_cfg.runtime.batch_size)),
        collate_fn=collate_train,
    )

    bitmask_group_top_ids, bitmask_group_class_weight = build_bitmask_group_definition(
        family_data.y_cls_train,
        top_k=int(hpo_cfg.loss.bitmask_group_top_k),
        class_weight_alpha=float(hpo_cfg.loss.bitmask_group_weight_alpha),
        class_weight_cap=float(hpo_cfg.loss.bitmask_group_weight_cap),
    )
    model = MILModelBuilder.build(
        config=hpo_cfg,
        mol_dim=int(family_data.X2d_train.shape[1]),
        inst_dim=int(family_data.Xinst_sorted.shape[1]),
        inst_geom_dim=int(family_data.inst_geom_dim),
        inst_qm_dim=int(family_data.inst_qm_dim),
        pos_weight=posw,
        gamma=gamma_t,
        lam=lam,
        bitmask_group_top_ids=bitmask_group_top_ids,
        bitmask_group_class_weight=bitmask_group_class_weight,
    )

    family_dir = outdir / family_data.family
    family_dir.mkdir(parents=True, exist_ok=True)
    trainer_cfg = LightningTrainerConfig(
        max_epochs=int(cfg.max_epochs),
        patience=int(cfg.patience),
        accelerator=str(cfg.nn_accelerator),
        devices=int(cfg.nn_devices),
        precision=str(cfg.precision),
        accumulate_grad_batches=int(hpo_cfg.runtime.accumulate_grad_batches),
        save_checkpoint=True,
        save_weights_only=True,
    )
    trainer, ckpt_cb = LightningTrainerFactory(trainer_cfg).build(
        ckpt_dir=str(family_dir),
        trial=None,
    )
    trainer.fit(model, dl_tr, dl_lb)
    if ckpt_cb is not None:
        best_path = ckpt_cb.best_model_path
        if best_path and Path(best_path).exists():
            ckpt = torch.load(best_path, map_location="cpu")
            model.load_state_dict(ckpt["state_dict"], strict=True)

    eval_device = _resolve_device(cfg.nn_accelerator)
    evaluator = ModelEvaluator(device=eval_device)
    macro_ap, aps, macro_auc, aucs = evaluator.eval_best_epoch(model, dl_lb)
    eval_json = {
        "macro_pr_auc": float(macro_ap),
        "macro_roc_auc": float(macro_auc),
        "pr_aucs": [float(x) for x in aps],
        "roc_aucs": [float(x) for x in aucs],
    }
    (family_dir / "leaderboard_eval.json").write_text(json.dumps(eval_json, indent=2))

    model.eval()
    model.to(eval_device)
    preds: List[np.ndarray] = []
    with torch.no_grad():
        for batch in dl_lb:
            x2d, x3d, kpm, *_ = batch
            x2d = x2d.to(eval_device, non_blocking=True)
            x3d = x3d.to(eval_device, non_blocking=True)
            kpm = kpm.to(eval_device, non_blocking=True)
            logits, _, _ = model(x2d, x3d, kpm, return_attn=False)
            p = torch.sigmoid(torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)).detach().cpu().numpy()
            preds.append(p.astype(np.float32))
    P = np.concatenate(preds, axis=0)
    Y = np.asarray(family_data.y_cls_lb, dtype=np.int64)
    W = np.asarray(family_data.w_cls_lb, dtype=np.float32)

    df_pred = pd.DataFrame({"ID": [str(x) for x in family_data.ids_lb]})
    for t in range(4):
        df_pred[f"p_t{t}"] = P[:, t]
        df_pred[f"y_t{t}"] = Y[:, t]
        df_pred[f"w_t{t}"] = W[:, t]
        df_pred[f"pred_t{t}"] = (P[:, t] >= 0.5).astype(int)
    metrics = _metric_table(y_true=Y, p_pred=P, w_cls=W)
    metrics.to_csv(outdir / f"leaderboard_metrics_{family_data.family}.csv", index=False)
    df_pred.to_csv(outdir / f"leaderboard_preds_{family_data.family}.csv", index=False)

    del trainer, model, dl_tr, dl_lb, ds_tr, ds_lb
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return df_pred


def _prepare_family_inputs(
    cfg: FamilySuiteConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, List[str], np.ndarray, List[str], np.ndarray, Dict[str, _MILFamilyData]]:
    df_full = load_labels(cfg.labels, id_col=cfg.id_col)
    df_full[cfg.id_col] = df_full[cfg.id_col].astype(str)
    df_full[cfg.split_col] = df_full[cfg.split_col].astype(str)

    df_train = df_full[df_full[cfg.split_col].isin(set(cfg.use_splits))].copy().reset_index(drop=True)
    df_lb = df_full[df_full[cfg.split_col] == str(cfg.leaderboard_split)].copy().reset_index(drop=True)
    if len(df_train) == 0:
        raise ValueError(f"No rows in labels match use_splits={cfg.use_splits}")
    if len(df_lb) == 0:
        raise ValueError(f"No rows with split == '{cfg.leaderboard_split}'")

    ids_2d_file, X2d_file = load_2d(cfg.feat2d_scaled, id_col=cfg.id_col)
    if cfg.feat2d_raw:
        ids_2d_cat_file, X2d_cat_file = load_2d(cfg.feat2d_raw, id_col=cfg.id_col)
        cat_source = "raw"
    else:
        ids_2d_cat_file, X2d_cat_file = ids_2d_file, X2d_file
        cat_source = "scaled_fallback"
    ids_train_all = df_train[cfg.id_col].astype(str).tolist()
    ids_lb_all = df_lb[cfg.id_col].astype(str).tolist()
    _ = align_by_id(ids_2d_file, X2d_file, ids_train_all)
    _ = align_by_id(ids_2d_file, X2d_file, ids_lb_all)
    _ = align_by_id(ids_2d_cat_file, X2d_cat_file, ids_train_all)
    _ = align_by_id(ids_2d_cat_file, X2d_cat_file, ids_lb_all)
    log_event(
        "INFO",
        "family_suite.prepare_data.features_2d",
        mil_source="scaled",
        catboost_source=cat_source,
    )

    allowed = set(ids_train_all) | set(ids_lb_all)
    ids_conf, conf_ids, Xinst, inst_meta = load_and_merge_instances(
        cfg.feat3d_scaled,
        cfg.feat3d_qm_scaled,
        allowed_ids=allowed,
        id_col=cfg.id_col,
        conf_col=cfg.conf_col,
        return_meta=True,
    )
    _, starts, counts, id2pos, Xinst_sorted, conf_sorted = build_instance_index(ids_conf, conf_ids, Xinst)

    def _make_mil_family(family: str) -> _MILFamilyData:
        if family == "mt_2d":
            ids_tr = ids_train_all
            ids_lb = ids_lb_all
            X2d_tr = align_by_id(ids_2d_file, X2d_file, ids_tr)
            X2d_lb = align_by_id(ids_2d_file, X2d_file, ids_lb)
            union_ids = list(dict.fromkeys([str(x) for x in (ids_tr + ids_lb)]))
            starts_loc = np.arange(len(union_ids), dtype=np.int64)
            counts_loc = np.ones(len(union_ids), dtype=np.int64)
            id2pos_loc = {str(mid): int(i) for i, mid in enumerate(union_ids)}
            Xinst_loc = np.zeros((len(union_ids), 1), dtype=np.float32)
            conf_loc = np.array(["dummy"] * len(union_ids), dtype=object)
            geom_dim = 0
            qm_dim = 0
        elif family == "mt_3d":
            ids_tr = [i for i in ids_train_all if i in id2pos]
            ids_lb = [i for i in ids_lb_all if i in id2pos]
            X2d_tr = np.zeros((len(ids_tr), 0), dtype=np.float32)
            X2d_lb = np.zeros((len(ids_lb), 0), dtype=np.float32)
            starts_loc, counts_loc, id2pos_loc, Xinst_loc, conf_loc = starts, counts, id2pos, Xinst_sorted, conf_sorted
            geom_dim = int(inst_meta["geom_dim"])
            qm_dim = int(inst_meta["qm_dim"])
        else:  # mt_2d3d
            ids_tr = [i for i in ids_train_all if i in id2pos]
            ids_lb = [i for i in ids_lb_all if i in id2pos]
            X2d_tr = align_by_id(ids_2d_file, X2d_file, ids_tr)
            X2d_lb = align_by_id(ids_2d_file, X2d_file, ids_lb)
            starts_loc, counts_loc, id2pos_loc, Xinst_loc, conf_loc = starts, counts, id2pos, Xinst_sorted, conf_sorted
            geom_dim = int(inst_meta["geom_dim"])
            qm_dim = int(inst_meta["qm_dim"])

        df_tr = _df_by_ids(df_train, id_col=cfg.id_col, ids=ids_tr)
        df_lb_f = _df_by_ids(df_lb, id_col=cfg.id_col, ids=ids_lb)
        folds = sorted(df_tr[cfg.fold_col].dropna().astype(int).unique().tolist())
        folds_info = fold_indices(df_tr, cfg.fold_col, folds)

        y_cls_tr = coerce_binary_labels(df_tr)
        y_cls_lb = coerce_binary_labels(df_lb_f)
        w_cls_tr = build_task_weights(df_tr)
        w_cls_lb = build_task_weights(df_lb_f)
        y_abs_tr, m_abs_tr, y_fluo_tr, m_fluo_tr = build_aux_targets_and_masks(df_tr)
        y_abs_lb, m_abs_lb, y_fluo_lb, m_fluo_lb = build_aux_targets_and_masks(df_lb_f)
        w_abs_tr, w_fluo_tr = build_aux_weights(df_tr)
        w_abs_lb, w_fluo_lb = build_aux_weights(df_lb_f)

        return _MILFamilyData(
            family=str(family),
            ids_train=[str(x) for x in ids_tr],
            ids_lb=[str(x) for x in ids_lb],
            X2d_train=np.asarray(X2d_tr, dtype=np.float32),
            X2d_lb=np.asarray(X2d_lb, dtype=np.float32),
            y_cls_train=np.asarray(y_cls_tr, dtype=np.int64),
            y_cls_lb=np.asarray(y_cls_lb, dtype=np.int64),
            w_cls_train=np.asarray(w_cls_tr, dtype=np.float32),
            w_cls_lb=np.asarray(w_cls_lb, dtype=np.float32),
            y_abs_train=np.asarray(y_abs_tr, dtype=np.float32),
            m_abs_train=np.asarray(m_abs_tr, dtype=bool),
            w_abs_train=np.asarray(w_abs_tr, dtype=np.float32),
            y_abs_lb=np.asarray(y_abs_lb, dtype=np.float32),
            m_abs_lb=np.asarray(m_abs_lb, dtype=bool),
            w_abs_lb=np.asarray(w_abs_lb, dtype=np.float32),
            y_fluo_train=np.asarray(y_fluo_tr, dtype=np.float32),
            m_fluo_train=np.asarray(m_fluo_tr, dtype=bool),
            w_fluo_train=np.asarray(w_fluo_tr, dtype=np.float32),
            y_fluo_lb=np.asarray(y_fluo_lb, dtype=np.float32),
            m_fluo_lb=np.asarray(m_fluo_lb, dtype=bool),
            w_fluo_lb=np.asarray(w_fluo_lb, dtype=np.float32),
            folds_info=folds_info,
            starts=np.asarray(starts_loc, dtype=np.int64),
            counts=np.asarray(counts_loc, dtype=np.int64),
            id2pos={str(k): int(v) for k, v in id2pos_loc.items()},
            Xinst_sorted=np.asarray(Xinst_loc, dtype=np.float32),
            conf_sorted=np.asarray(conf_loc),
            inst_geom_dim=int(geom_dim),
            inst_qm_dim=int(qm_dim),
        )

    fam_data = {f: _make_mil_family(f) for f in ("mt_2d", "mt_2d3d", "mt_3d")}
    return df_train, df_lb, ids_2d_file, X2d_file, ids_2d_cat_file, X2d_cat_file, fam_data


def run_family_suite(args: Any) -> None:
    cfg = FamilySuiteConfig(
        labels=str(args.labels),
        feat2d_scaled=str(args.feat2d_scaled),
        feat2d_raw=(None if getattr(args, "feat2d_raw", None) in (None, "") else str(args.feat2d_raw)),
        feat3d_scaled=str(args.feat3d_scaled),
        feat3d_qm_scaled=str(args.feat3d_qm_scaled),
        study_dir=str(args.study_dir),
        id_col=str(args.id_col),
        conf_col=str(args.conf_col),
        split_col=str(args.split_col),
        fold_col=str(args.fold_col),
        use_splits=tuple(str(x) for x in args.use_splits),
        leaderboard_split=str(args.leaderboard_split),
        max_epochs=int(args.max_epochs),
        patience=int(args.patience),
        trials=int(args.trials),
        seed=int(args.seed),
        nn_accelerator=str(args.nn_accelerator),
        nn_devices=int(args.nn_devices),
        precision=str(args.precision),
        num_workers=int(args.num_workers),
        cpu_workers=int(args.cpu_workers),
        pin_memory=bool(args.pin_memory),
        run_hpo=bool(args.run_hpo),
        hpo_only=bool(args.hpo_only),
        pruner_warmup_steps=int(args.pruner_warmup_steps),
        model_families=tuple(str(x) for x in (args.model_families or FAMILY_CHOICES)),
        calibration_method=str(args.calibration_method),
        best_params_dir=(None if args.best_params_dir is None else str(args.best_params_dir)),
        catboost_hpo_parallel_tasks=int(getattr(args, "catboost_hpo_parallel_tasks", 1)),
    )
    invalid = [x for x in cfg.model_families if x not in FAMILY_CHOICES]
    if invalid:
        raise ValueError(f"Unknown model family names: {invalid}. Allowed: {list(FAMILY_CHOICES)}")

    set_all_seeds(int(cfg.seed))
    maybe_set_torch_fast_flags()
    outdir = Path(cfg.study_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    best_params_dir = Path(cfg.best_params_dir) if cfg.best_params_dir else (outdir / "best_params")
    best_params_dir.mkdir(parents=True, exist_ok=True)
    cal_dir = outdir / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)

    with log_step("family_suite.prepare_data", families=list(cfg.model_families)):
        df_train, df_lb, ids_2d_file, X2d_file, ids_2d_cat_file, X2d_cat_file, mil_families = _prepare_family_inputs(
            cfg
        )

    best_params: Dict[str, Any] = {}
    if cfg.run_hpo:
        for family in cfg.model_families:
            if family == "catboost_st":
                try:
                    from catboost import CatBoostClassifier  # noqa: F401
                except Exception as exc:
                    raise RuntimeError("catboost is required for family 'catboost_st'.") from exc
                folds = sorted(df_train[cfg.fold_col].dropna().astype(int).unique().tolist())
                folds_info = fold_indices(df_train, cfg.fold_col, folds)
                ids_tr = df_train[cfg.id_col].astype(str).tolist()
                X2d_tr = align_by_id(ids_2d_cat_file, X2d_cat_file, ids_tr)
                y_tr = coerce_binary_labels(df_train)
                w_tr = build_task_weights(df_train)
                catboost_params_by_task: Dict[str, Dict[str, Any]] = {}
                best_values_by_task: Dict[str, float] = {}
                task_parallel = int(max(1, min(4, int(cfg.catboost_hpo_parallel_tasks))))
                threads_per_task = int(max(1, int(cfg.cpu_workers) // task_parallel))
                log_event(
                    "INFO",
                    "family.hpo.catboost.parallel_config",
                    task_parallel=int(task_parallel),
                    cpu_workers=int(cfg.cpu_workers),
                    threads_per_task=int(threads_per_task),
                )

                def _run_catboost_task_hpo(task_idx: int) -> tuple[str, Dict[str, Any], float]:
                    study_name = f"catboost_st_t{int(task_idx)}"

                    def objective(trial: optuna.Trial, task_idx_inner: int = int(task_idx)) -> float:
                        from catboost import CatBoostClassifier

                        p = _catboost_search_space(trial, task_idx=int(task_idx_inner))
                        fold_pr_scores: List[float] = []
                        fold_roc_scores: List[float] = []
                        for fold_step, (tr, va, fold_id) in enumerate(folds_info):
                            pos = float(y_tr[tr, task_idx_inner].sum())
                            neg = float(len(tr) - pos)
                            spw = min(neg / max(pos, 1.0), float(p["pos_weight_clip"]))
                            cb = CatBoostClassifier(
                                **_catboost_common_params(
                                    params=p,
                                    seed=int(cfg.seed) + 97 * int(task_idx_inner) + 7919 * int(trial.number),
                                    threads=int(threads_per_task),
                                    scale_pos_weight=float(spw),
                                )
                            )
                            sw = (w_tr[tr, task_idx_inner] if task_idx_inner in (0, 1) else None)
                            cb.fit(
                                X2d_tr[tr],
                                y_tr[tr, task_idx_inner],
                                sample_weight=sw,
                                eval_set=(X2d_tr[va], y_tr[va, task_idx_inner]),
                                use_best_model=True,
                                early_stopping_rounds=200,
                                verbose=False,
                            )
                            p_va = cb.predict_proba(X2d_tr[va])[:, 1]
                            p_va = np.clip(np.nan_to_num(p_va, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0)
                            sw_va = (w_tr[va, task_idx_inner] if task_idx_inner in (0, 1) else None)
                            try:
                                ap_t = float(average_precision_score(y_tr[va, task_idx_inner], p_va, sample_weight=sw_va))
                            except ValueError:
                                ap_t = 0.0
                            try:
                                roc_t = float(roc_auc_score(y_tr[va, task_idx_inner], p_va, sample_weight=sw_va))
                            except ValueError:
                                roc_t = 0.5
                            fold_pr_scores.append(ap_t)
                            fold_roc_scores.append(roc_t)
                            trial.report(float(np.mean(fold_pr_scores)), step=int(fold_step))
                            if trial.should_prune():
                                raise optuna.TrialPruned()
                            log_event(
                                "INFO",
                                "family.hpo.catboost.trial.fold",
                                task_idx=int(task_idx_inner),
                                trial=int(trial.number),
                                fold_id=int(fold_id),
                                pr_auc=f"{ap_t:.6f}",
                                roc_auc=f"{roc_t:.6f}",
                            )
                        mean_pr = float(np.mean(fold_pr_scores)) if fold_pr_scores else 0.0
                        mean_roc = float(np.mean(fold_roc_scores)) if fold_roc_scores else 0.5
                        trial.set_user_attr("cv_pr_auc", mean_pr)
                        trial.set_user_attr("cv_roc_auc", mean_roc)
                        log_event(
                            "INFO",
                            "family.hpo.catboost.trial.done",
                            task_idx=int(task_idx_inner),
                            trial=int(trial.number),
                            cv_pr_auc=f"{mean_pr:.6f}",
                            cv_roc_auc=f"{mean_roc:.6f}",
                        )
                        return mean_pr

                    storage = f"sqlite:///{(outdir / f'{study_name}.sqlite3').as_posix()}"
                    study = optuna.create_study(
                        study_name=study_name,
                        direction="maximize",
                        sampler=optuna.samplers.TPESampler(seed=int(cfg.seed) + 113 * int(task_idx)),
                        pruner=optuna.pruners.PercentilePruner(
                            percentile=25.0,
                            n_startup_trials=10,
                            n_warmup_steps=int(cfg.pruner_warmup_steps),
                        ),
                        storage=storage,
                        load_if_exists=True,
                    )
                    with log_step("family.hpo.catboost.optimize", task_idx=int(task_idx), n_trials=int(cfg.trials)):
                        study.optimize(objective, n_trials=int(cfg.trials), gc_after_trial=True)
                    pd.DataFrame(
                        study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
                    ).to_csv(outdir / f"{study_name}_trials.csv", index=False)

                    bp_t = dict(study.best_params)
                    saved_path = _save_catboost_task_best_params(
                        outdir=best_params_dir,
                        task_idx=int(task_idx),
                        best_params=bp_t,
                        best_value=float(study.best_value),
                    )
                    log_event(
                        "INFO",
                        "family.hpo.catboost.best_params.saved",
                        task_idx=int(task_idx),
                        path=str(saved_path),
                        best_value=f"{float(study.best_value):.6f}",
                    )
                    return f"task_{int(task_idx)}", bp_t, float(study.best_value)

                if int(task_parallel) <= 1:
                    for t in range(4):
                        key, bp_t, best_v = _run_catboost_task_hpo(int(t))
                        catboost_params_by_task[key] = bp_t
                        best_values_by_task[key] = float(best_v)
                else:
                    with ThreadPoolExecutor(max_workers=int(task_parallel)) as pool:
                        futures = [pool.submit(_run_catboost_task_hpo, int(t)) for t in range(4)]
                        for fut in as_completed(futures):
                            key, bp_t, best_v = fut.result()
                            catboost_params_by_task[key] = bp_t
                            best_values_by_task[key] = float(best_v)
                summary_path = _save_family_best_params(
                    outdir=best_params_dir,
                    family=family,
                    best_params=catboost_params_by_task,
                    best_value=float(np.mean([v for v in best_values_by_task.values()] or [0.0])),
                )
                log_event(
                    "INFO",
                    "family.hpo.best_params.saved",
                    family=str(family),
                    path=str(summary_path),
                )
                best_params[family] = catboost_params_by_task
                continue

            data = mil_families[family]
            cv_data = MILCVData(
                X2d_scaled=data.X2d_train,
                y_cls=data.y_cls_train,
                w_cls=data.w_cls_train,
                y_abs=data.y_abs_train,
                m_abs=data.m_abs_train,
                w_abs=data.w_abs_train,
                y_fluo=data.y_fluo_train,
                m_fluo=data.m_fluo_train,
                w_fluo=data.w_fluo_train,
                ids=data.ids_train,
                folds_info=data.folds_info,
                starts=data.starts,
                counts=data.counts,
                id2pos=data.id2pos,
                Xinst_sorted=data.Xinst_sorted,
                inst_geom_dim=int(data.inst_geom_dim),
                inst_qm_dim=int(data.inst_qm_dim),
            )
            run_cfg = CVRunConfig(
                seed=int(cfg.seed),
                trainer=TrainerSystemConfig(
                    max_epochs=int(cfg.max_epochs),
                    patience=int(cfg.patience),
                    accelerator=str(cfg.nn_accelerator),
                    devices=int(cfg.nn_devices),
                    precision=str(cfg.precision),
                ),
                loader=LoaderConfig(
                    num_workers=int(cfg.num_workers),
                    pin_memory=bool(cfg.pin_memory and torch.cuda.is_available()),
                ),
                ckpt_root=(outdir / f"_tmp_best_ckpts_{family}"),
            )
            run_cfg.ckpt_root.mkdir(parents=True, exist_ok=True)
            cv = _MILMacroCrossValidator(data=cv_data, run_config=run_cfg)
            study_name = f"{family}_mil"
            storage = f"sqlite:///{(outdir / f'{study_name}.sqlite3').as_posix()}"
            study = optuna.create_study(
                study_name=study_name,
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=int(cfg.seed)),
                pruner=optuna.pruners.PercentilePruner(
                    percentile=25.0,
                    n_startup_trials=10,
                    n_warmup_steps=int(cfg.pruner_warmup_steps),
                ),
                storage=storage,
                load_if_exists=True,
            )
            with log_step("family.hpo.mil.optimize", family=family, n_trials=int(cfg.trials)):
                study.optimize(cv.evaluate_trial, n_trials=int(cfg.trials), gc_after_trial=True)
            pd.DataFrame(
                study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
            ).to_csv(outdir / f"{study_name}_trials.csv", index=False)
            bp = dict(study.best_params)
            bp["min_w"] = 0.4
            best_params[family] = bp
            _save_family_best_params(
                outdir=best_params_dir,
                family=family,
                best_params=bp,
                best_value=float(study.best_value),
            )
            log_event("INFO", "family.hpo.best_params.saved", family=str(family), path=str(best_params_dir / f"{family}.json"))
    else:
        for family in cfg.model_families:
            if family == "catboost_st":
                try:
                    best_params[family] = {
                        f"task_{int(t)}": _load_catboost_task_best_params(outdir=best_params_dir, task_idx=int(t))
                        for t in range(4)
                    }
                except FileNotFoundError:
                    # Backward compatibility with older shared CatBoost params format.
                    shared = _load_family_best_params(outdir=best_params_dir, family=family)
                    best_params[family] = {f"task_{int(t)}": dict(shared) for t in range(4)}
                    log_event(
                        "WARN",
                        "family.best_params.catboost.shared_fallback",
                        path=str(best_params_dir / f"{family}.json"),
                    )
            else:
                best_params[family] = _load_family_best_params(outdir=best_params_dir, family=family)

    if cfg.hpo_only:
        log_event("INFO", "family_suite.hpo_only.completed", n_families=int(len(cfg.model_families)))
        return

    pred_tables: Dict[str, pd.DataFrame] = {}
    for family in cfg.model_families:
        if family == "catboost_st":
            from catboost import CatBoostClassifier

            ids_tr = df_train[cfg.id_col].astype(str).tolist()
            ids_lb = df_lb[cfg.id_col].astype(str).tolist()
            X2d_tr = align_by_id(ids_2d_cat_file, X2d_cat_file, ids_tr)
            X2d_lb = align_by_id(ids_2d_cat_file, X2d_cat_file, ids_lb)
            y_tr = coerce_binary_labels(df_train)
            y_lb = coerce_binary_labels(df_lb)
            w_tr = build_task_weights(df_train)
            w_lb = build_task_weights(df_lb)
            catboost_task_params = dict(best_params[family])
            pred = np.zeros((len(ids_lb), 4), dtype=np.float64)
            for t in range(4):
                if f"task_{int(t)}" not in catboost_task_params:
                    raise KeyError(
                        f"Missing CatBoost params for task_{t}. "
                        "Expected task-wise JSONs: catboost_st_t0..catboost_st_t3."
                    )
                p = dict(catboost_task_params[f"task_{int(t)}"])
                pos = float(y_tr[:, t].sum())
                neg = float(len(y_tr) - pos)
                spw = min(neg / max(pos, 1.0), float(p.get("pos_weight_clip", 100.0)))
                cb = CatBoostClassifier(
                    **_catboost_common_params(
                        params=p,
                        seed=int(cfg.seed) + 97 * int(t),
                        threads=max(1, int(cfg.cpu_workers)),
                        scale_pos_weight=float(spw),
                    )
                )
                sw = (w_tr[:, t] if t in (0, 1) else None)
                cb.fit(
                    X2d_tr,
                    y_tr[:, t],
                    sample_weight=sw,
                    eval_set=(X2d_tr, y_tr[:, t]),
                    use_best_model=False,
                    verbose=False,
                )
                pred[:, t] = cb.predict_proba(X2d_lb)[:, 1]
            df_pred = pd.DataFrame({"ID": [str(x) for x in ids_lb]})
            for t in range(4):
                df_pred[f"p_t{t}"] = pred[:, t]
                df_pred[f"y_t{t}"] = y_lb[:, t]
                df_pred[f"w_t{t}"] = w_lb[:, t]
                df_pred[f"pred_t{t}"] = (pred[:, t] >= 0.5).astype(int)
            df_pred.to_csv(outdir / f"leaderboard_preds_{family}.csv", index=False)
            _metric_table(y_true=y_lb, p_pred=pred, w_cls=w_lb).to_csv(
                outdir / f"leaderboard_metrics_{family}.csv", index=False
            )
            pred_tables[family] = df_pred
            continue

        df_pred = _run_mil_final_train_and_predict(
            cfg=cfg,
            family_data=mil_families[family],
            best_params=best_params[family],
            outdir=outdir,
        )
        pred_tables[family] = df_pred

    # Calibration stage (leaderboard-fitted, post-hoc).
    calibrated: Dict[str, pd.DataFrame] = {}
    for family, dfp in pred_tables.items():
        arr = dfp.copy()
        y = np.stack([arr[f"y_t{t}"].to_numpy(dtype=np.int64) for t in range(4)], axis=1)
        p = np.stack([arr[f"p_t{t}"].to_numpy(dtype=np.float64) for t in range(4)], axis=1)
        p_cal = np.zeros_like(p, dtype=np.float32)
        params_by_task: Dict[str, Any] = {}
        for t in range(4):
            p_cal[:, t], t_params = _calibrate_task(y[:, t], p[:, t], cfg.calibration_method)
            params_by_task[str(t)] = t_params
        for t in range(4):
            arr[f"p_cal_t{t}"] = p_cal[:, t]
        arr.to_csv(outdir / f"leaderboard_preds_{family}_calibrated.csv", index=False)
        calibrated[family] = arr
        (cal_dir / f"{family}_calib.json").write_text(
            json.dumps(
                {
                    "family": str(family),
                    "method": str(cfg.calibration_method),
                    "task_params": params_by_task,
                },
                indent=2,
            )
        )
        w_lb = np.stack(
            [arr[f"w_t{t}"].to_numpy(dtype=np.float32) if f"w_t{t}" in arr.columns else np.ones(len(arr), dtype=np.float32) for t in range(4)],
            axis=1,
        )
        _metric_table(
            y_true=y,
            p_pred=p_cal,
            w_cls=w_lb,
        ).to_csv(outdir / f"leaderboard_metrics_{family}_calibrated.csv", index=False)

    # Blend calibrated probabilities on common leaderboard IDs.
    fams = list(calibrated.keys())
    common_ids = set(calibrated[fams[0]]["ID"].astype(str).tolist())
    for f in fams[1:]:
        common_ids &= set(calibrated[f]["ID"].astype(str).tolist())
    common_ids = sorted(common_ids)
    if len(common_ids) == 0:
        raise RuntimeError("No common leaderboard IDs across selected families for blending.")

    y_bl = None
    w_bl = None
    x_by_family: Dict[str, np.ndarray] = {}
    for f in fams:
        d = calibrated[f].copy()
        d["ID"] = d["ID"].astype(str)
        d = d.set_index("ID").loc[common_ids].reset_index(drop=False)
        x_by_family[f] = np.stack([d[f"p_cal_t{t}"].to_numpy(dtype=np.float64) for t in range(4)], axis=1)
        if y_bl is None:
            y_bl = np.stack([d[f"y_t{t}"].to_numpy(dtype=np.int64) for t in range(4)], axis=1)
            w_bl = np.stack(
                [d[f"w_t{t}"].to_numpy(dtype=np.float32) if f"w_t{t}" in d.columns else np.ones(len(d), dtype=np.float32) for t in range(4)],
                axis=1,
            )
    assert y_bl is not None
    assert w_bl is not None

    blend_pred = np.zeros_like(y_bl, dtype=np.float64)
    blend_weights: Dict[str, Any] = {"families": fams, "tasks": {}}
    for t in range(4):
        X_task = np.column_stack([x_by_family[f][:, t] for f in fams]).astype(np.float64)
        y_task = y_bl[:, t].astype(int)
        if int(np.unique(y_task).size) < 2:
            w = np.ones(len(fams), dtype=np.float64) / float(len(fams))
            p_task = np.clip(np.dot(X_task, w), 1e-6, 1 - 1e-6)
            blend_pred[:, t] = p_task
            blend_weights["tasks"][str(t)] = {
                "kind": "uniform_single_class",
                "weights": {f: float(w[i]) for i, f in enumerate(fams)},
                "intercept": 0.0,
            }
            continue
        lr = LogisticRegression(max_iter=2000, solver="lbfgs")
        lr.fit(X_task, y_task)
        p_task = lr.predict_proba(X_task)[:, 1]
        blend_pred[:, t] = p_task
        blend_weights["tasks"][str(t)] = {
            "kind": "logistic_stacking",
            "weights": {f: float(lr.coef_[0, i]) for i, f in enumerate(fams)},
            "intercept": float(lr.intercept_[0]),
        }

    blend_df = pd.DataFrame({"ID": common_ids})
    for t in range(4):
        blend_df[f"p_blend_t{t}"] = blend_pred[:, t]
        blend_df[f"y_t{t}"] = y_bl[:, t]
        blend_df[f"pred_t{t}"] = (blend_pred[:, t] >= 0.5).astype(int)
    blend_df.to_csv(outdir / "leaderboard_preds_blend.csv", index=False)
    (outdir / "blend_weights.json").write_text(json.dumps(blend_weights, indent=2))

    _metric_table(
        y_true=y_bl,
        p_pred=blend_pred,
        w_cls=w_bl,
    ).to_csv(outdir / "leaderboard_metrics_blend.csv", index=False)
    log_event("DONE", "family_suite.run", outdir=str(outdir), n_families=int(len(fams)))


__all__ = ["run_family_suite", "FAMILY_CHOICES"]
