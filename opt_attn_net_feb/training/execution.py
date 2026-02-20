from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
import gc
import json
import shutil

import numpy as np
import optuna
import pandas as pd
import torch
from optuna.trial import Trial

from ..data.collate import collate_export, collate_train
from ..data.datasets import MILExportDataset, MILTrainDataset
from ..data.exports import export_leaderboard_attention
from ..utils.data_io import align_by_id
from ..utils.ops import (
    apply_standardizer,
    build_aux_targets_and_masks,
    build_aux_weights,
    build_task_weights,
    coerce_binary_labels,
    fit_standardizer,
    make_weighted_sampler,
    pos_weight_per_task,
    set_all_seeds,
)
from .builders import DataLoaderBuilder, LoaderConfig, MILModelBuilder
from .configs import HPOConfig
from .loss_config import compute_gamma, compute_lam, compute_posw_clips
from .search_space import search_space
from .trainer import LightningTrainerConfig, LightningTrainerFactory, ModelEvaluator


def _resolve_device(accelerator: str) -> torch.device:
    return torch.device(
        "cuda" if torch.cuda.is_available() and str(accelerator) in ("gpu", "cuda") else "cpu"
    )


@dataclass(frozen=True)
class TrainerSystemConfig:
    max_epochs: int
    patience: int
    accelerator: str
    devices: int
    precision: str


@dataclass(frozen=True)
class CVRunConfig:
    seed: int
    trainer: TrainerSystemConfig
    loader: LoaderConfig
    ckpt_root: Path


@dataclass(frozen=True)
class StudyConfig:
    outdir: Path
    study_name: str
    n_trials: int
    seed: int
    direction: str = "maximize"
    pruner_warmup_steps: int = 1


@dataclass(frozen=True)
class FinalTrainConfig:
    seed: int
    trainer: TrainerSystemConfig
    loader: LoaderConfig
    attn_out: str | None = None


@dataclass(frozen=True)
class MILCVData:
    X2d_scaled: np.ndarray
    y_cls: np.ndarray
    w_cls: np.ndarray
    y_abs: np.ndarray
    m_abs: np.ndarray
    w_abs: np.ndarray
    y_fluo: np.ndarray
    m_fluo: np.ndarray
    w_fluo: np.ndarray
    ids: List[str]
    folds_info: Sequence[Tuple[np.ndarray, np.ndarray, int]]
    starts: np.ndarray
    counts: np.ndarray
    id2pos: Dict[str, int]
    Xinst_sorted: np.ndarray


@dataclass(frozen=True)
class MILFinalData:
    df_full: pd.DataFrame
    id_col: str
    split_col: str
    leaderboard_split: str
    X2d_file_ids: List[str]
    X2d_file: np.ndarray
    starts: np.ndarray
    counts: np.ndarray
    id2pos: Dict[str, int]
    Xinst_sorted: np.ndarray
    conf_sorted: np.ndarray


def drop_ids_without_bags(
    *,
    ids: List[str],
    X2d: np.ndarray,
    df_part: pd.DataFrame,
    id2pos: Dict[str, int],
    id_col: str,
) -> Tuple[List[str], np.ndarray, pd.DataFrame]:
    mask = np.array([(i in id2pos) for i in ids], dtype=bool)
    if mask.all():
        return ids, X2d, df_part
    df2 = df_part.loc[mask].reset_index(drop=True)
    ids2 = df2[id_col].astype(str).tolist()
    return ids2, X2d[mask], df2


class MILFoldTrainer:
    """Runs one CV fold end-to-end with typed config and data contracts."""

    def __init__(
        self,
        *,
        trial: Trial,
        hpo_config: HPOConfig,
        data: MILCVData,
        run_config: CVRunConfig,
    ):
        self.trial = trial
        self.hpo_config = hpo_config
        self.data = data
        self.run_config = run_config
        self.loader_builder = DataLoaderBuilder(run_config.loader)
        self.eval_device = _resolve_device(run_config.trainer.accelerator)

    def run_fold(
        self,
        *,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        fold_id: int,
    ) -> Tuple[float, Dict[str, Any]]:
        cfg = self.hpo_config

        set_all_seeds(int(self.run_config.seed) + 5000 * int(fold_id) + int(self.trial.number))

        lam = compute_lam(cfg.loss, y_train=self.data.y_cls[train_idx])
        posw = pos_weight_per_task(
            self.data.y_cls[train_idx],
            clip=compute_posw_clips(cfg.loss),
        )
        gamma_t = compute_gamma(cfg.loss)

        mu_abs, sd_abs = fit_standardizer(self.data.y_abs, self.data.m_abs, train_idx)
        mu_f, sd_f = fit_standardizer(self.data.y_fluo, self.data.m_fluo, train_idx)
        y_abs_sc = apply_standardizer(self.data.y_abs, mu_abs, sd_abs)
        y_fluo_sc = apply_standardizer(self.data.y_fluo, mu_f, sd_f)

        ids_tr = [self.data.ids[i] for i in train_idx]
        ids_va = [self.data.ids[i] for i in val_idx]

        ds_tr = MILTrainDataset(
            ids_tr,
            self.data.X2d_scaled[train_idx],
            self.data.y_cls[train_idx],
            self.data.w_cls[train_idx],
            y_abs_sc[train_idx],
            self.data.m_abs[train_idx],
            self.data.w_abs[train_idx],
            y_fluo_sc[train_idx],
            self.data.m_fluo[train_idx],
            self.data.w_fluo[train_idx],
            self.data.starts,
            self.data.counts,
            self.data.id2pos,
            self.data.Xinst_sorted,
            max_instances=0,
            seed=int(self.run_config.seed) + int(fold_id),
        )
        ds_va = MILTrainDataset(
            ids_va,
            self.data.X2d_scaled[val_idx],
            self.data.y_cls[val_idx],
            self.data.w_cls[val_idx],
            y_abs_sc[val_idx],
            self.data.m_abs[val_idx],
            self.data.w_abs[val_idx],
            y_fluo_sc[val_idx],
            self.data.m_fluo[val_idx],
            self.data.w_fluo[val_idx],
            self.data.starts,
            self.data.counts,
            self.data.id2pos,
            self.data.Xinst_sorted,
            max_instances=0,
            seed=int(self.run_config.seed) + 999 + int(fold_id),
        )

        sampler = make_weighted_sampler(
            self.data.y_cls[train_idx],
            rare_mult=float(cfg.sampler.rare_oversample_mult),
            rare_prev_thr=float(cfg.sampler.rare_prev_thr),
            sample_weight_cap=float(cfg.sampler.sample_weight_cap),
        )

        dl_tr = self.loader_builder.train_loader(
            ds_tr,
            batch_size=int(cfg.runtime.batch_size),
            sampler=sampler,
            collate_fn=collate_train,
        )
        dl_va = self.loader_builder.eval_loader(
            ds_va,
            batch_size=min(128, int(cfg.runtime.batch_size)),
            collate_fn=collate_train,
        )

        model = MILModelBuilder.build(
            config=cfg,
            mol_dim=int(self.data.X2d_scaled.shape[1]),
            inst_dim=int(self.data.Xinst_sorted.shape[1]),
            pos_weight=posw,
            gamma=gamma_t,
            lam=lam,
        )

        fold_ckpt_dir = self.run_config.ckpt_root / f"mil_trial{self.trial.number}_fold{fold_id}"
        fold_ckpt_dir.mkdir(parents=True, exist_ok=True)

        trainer_cfg = LightningTrainerConfig(
            max_epochs=int(self.run_config.trainer.max_epochs),
            patience=int(self.run_config.trainer.patience),
            accelerator=str(self.run_config.trainer.accelerator),
            devices=int(self.run_config.trainer.devices),
            precision=str(self.run_config.trainer.precision),
            accumulate_grad_batches=int(cfg.runtime.accumulate_grad_batches),
        )
        trainer, ckpt_cb = LightningTrainerFactory(trainer_cfg).build(
            ckpt_dir=str(fold_ckpt_dir),
            trial=self.trial,
        )
        trainer.fit(model, dl_tr, dl_va)

        epochs_trained = int(trainer.current_epoch) + 1

        best_path = ckpt_cb.best_model_path
        best_epoch = None
        if best_path and Path(best_path).exists():
            ckpt = torch.load(best_path, map_location="cpu")
            best_epoch = int(ckpt.get("epoch", -1))
            model.load_state_dict(ckpt["state_dict"], strict=True)

        evaluator = ModelEvaluator(device=self.eval_device)
        best_macro, best_aps = evaluator.eval_best_epoch(model, dl_va)
        best_min = float(np.min(best_aps))

        min_w = float(cfg.objective.min_w)
        fold_score = float((1.0 - min_w) * float(best_macro) + min_w * best_min)

        print(
            f"[MIL-TASK-ATTN] trial={self.trial.number} fold={fold_id} trained_epochs={epochs_trained} "
            f"best_epoch={best_epoch} best_macro_ap={best_macro:.6f} min_ap={best_min:.6f} "
            f"aps={best_aps} score={fold_score:.6f} mode={cfg.objective.mode} min_w={min_w:.2f}"
        )

        detail = {
            "trained_epochs": epochs_trained,
            "best_epoch": best_epoch,
            "macro_ap_best_epoch": float(best_macro),
            "min_ap_best_epoch": best_min,
            "score": fold_score,
            "objective_mode": str(cfg.objective.mode),
            "min_w": min_w,
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

        return fold_score, detail


class MILCrossValidator:
    """Optuna objective implementation with explicit cross-validation class flow."""

    def __init__(self, *, data: MILCVData, run_config: CVRunConfig):
        self.data = data
        self.run_config = run_config

    def evaluate_trial(self, trial: Trial) -> float:
        params = search_space(trial)
        cfg = HPOConfig.from_params(params)
        if str(cfg.objective.mode) != "macro_plus_min":
            raise ValueError(
                f"Unsupported objective_mode={cfg.objective.mode}; only macro_plus_min is allowed."
            )

        fold_runner = MILFoldTrainer(
            trial=trial,
            hpo_config=cfg,
            data=self.data,
            run_config=self.run_config,
        )

        scores: List[float] = []
        fold_detail: Dict[str, Any] = {}
        for step, (tr, va, fold_id) in enumerate(self.data.folds_info):
            fold_score, detail = fold_runner.run_fold(
                train_idx=np.asarray(tr, dtype=np.int64),
                val_idx=np.asarray(va, dtype=np.int64),
                fold_id=int(fold_id),
            )
            scores.append(float(fold_score))
            fold_detail[str(fold_id)] = detail

            trial.report(float(np.mean(scores)), step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        trial.set_user_attr("fold_detail", fold_detail)
        return float(np.mean(scores))


class StudyArtifactsWriter:
    @staticmethod
    def save_study_artifacts(*, outdir: Path, study: optuna.Study, prefix: str) -> None:
        df_trials = study.trials_dataframe(
            attrs=("number", "value", "state", "params", "user_attrs")
        )
        df_trials.to_csv(outdir / f"{prefix}_trials.csv", index=False)
        best = dict(study.best_params)
        best["best_value_macro_ap_cv"] = float(study.best_value)
        (outdir / f"{prefix}_best_params.json").write_text(json.dumps(best, indent=2))

    @staticmethod
    def save_best_fold_metrics(
        *,
        outdir: Path,
        prefix: str,
        fold_metrics: Dict[str, Any],
    ) -> None:
        (outdir / f"{prefix}_best_fold_metrics.json").write_text(
            json.dumps(fold_metrics, indent=2)
        )


class MILStudyRunner:
    """Owns Optuna study creation, optimization, and artifact persistence."""

    def __init__(self, *, config: StudyConfig, cross_validator: MILCrossValidator):
        self.config = config
        self.cross_validator = cross_validator

    def _make_storage(self) -> str:
        return f"sqlite:///{(self.config.outdir / f'{self.config.study_name}.sqlite3').as_posix()}"

    def run(self) -> optuna.Study:
        self.config.outdir.mkdir(parents=True, exist_ok=True)
        sampler = optuna.samplers.TPESampler(seed=int(self.config.seed))
        pruner = optuna.pruners.MedianPruner(
            n_warmup_steps=int(self.config.pruner_warmup_steps)
        )
        study = optuna.create_study(
            direction=str(self.config.direction),
            sampler=sampler,
            pruner=pruner,
            study_name=str(self.config.study_name),
            storage=self._make_storage(),
            load_if_exists=True,
        )
        study.optimize(
            self.cross_validator.evaluate_trial,
            n_trials=int(self.config.n_trials),
            gc_after_trial=True,
            catch=(RuntimeError, ValueError, FloatingPointError),
        )
        StudyArtifactsWriter.save_study_artifacts(
            outdir=self.config.outdir,
            study=study,
            prefix=self.config.study_name,
        )
        StudyArtifactsWriter.save_best_fold_metrics(
            outdir=self.config.outdir,
            prefix=self.config.study_name,
            fold_metrics=study.best_trial.user_attrs.get("fold_detail", {}),
        )
        print(f"[HPO] best macro AP (CV mean) = {study.best_value:.6f}")
        return study


class MILFinalTrainer:
    """Trains best HPO config and exports leaderboard attention outputs."""

    def __init__(self, *, config: FinalTrainConfig):
        self.config = config
        self.loader_builder = DataLoaderBuilder(config.loader)
        self.eval_device = _resolve_device(config.trainer.accelerator)

    def run(self, *, outdir: Path, best_params: Dict[str, Any], data: MILFinalData) -> None:
        cfg = HPOConfig.from_params(
            best_params,
            fallback_lambda_power=1.0,
            fallback_lam_floor=0.25,
            fallback_lam_ceil=6.0,
            fallback_pos_weight_clip=50.0,
        )

        df_tr = data.df_full[data.df_full[data.split_col] == "train"].copy().reset_index(drop=True)
        df_lb = (
            data.df_full[data.df_full[data.split_col] == data.leaderboard_split]
            .copy()
            .reset_index(drop=True)
        )
        if len(df_lb) == 0:
            raise ValueError(f"No rows with split == '{data.leaderboard_split}'")

        ids_tr = df_tr[data.id_col].astype(str).tolist()
        ids_lb = df_lb[data.id_col].astype(str).tolist()

        X2d_tr = align_by_id(data.X2d_file_ids, data.X2d_file, ids_tr)
        X2d_lb = align_by_id(data.X2d_file_ids, data.X2d_file, ids_lb)

        ids_tr, X2d_tr, df_tr = drop_ids_without_bags(
            ids=ids_tr,
            X2d=X2d_tr,
            df_part=df_tr,
            id2pos=data.id2pos,
            id_col=data.id_col,
        )
        ids_lb, X2d_lb, df_lb = drop_ids_without_bags(
            ids=ids_lb,
            X2d=X2d_lb,
            df_part=df_lb,
            id2pos=data.id2pos,
            id_col=data.id_col,
        )

        y_tr = coerce_binary_labels(df_tr)
        w_tr = build_task_weights(df_tr)
        y_abs_tr, m_abs_tr, y_fluo_tr, m_fluo_tr = build_aux_targets_and_masks(df_tr)
        w_abs_tr, w_fluo_tr = build_aux_weights(df_tr)

        y_lb = coerce_binary_labels(df_lb)
        w_lb = build_task_weights(df_lb)
        y_abs_lb, m_abs_lb, y_fluo_lb, m_fluo_lb = build_aux_targets_and_masks(df_lb)
        w_abs_lb, w_fluo_lb = build_aux_weights(df_lb)

        tr_idx = np.arange(len(df_tr), dtype=np.int64)
        mu_abs, sd_abs = fit_standardizer(y_abs_tr, m_abs_tr, tr_idx)
        mu_f, sd_f = fit_standardizer(y_fluo_tr, m_fluo_tr, tr_idx)
        y_abs_tr_sc = apply_standardizer(y_abs_tr, mu_abs, sd_abs)
        y_abs_lb_sc = apply_standardizer(y_abs_lb, mu_abs, sd_abs)
        y_fluo_tr_sc = apply_standardizer(y_fluo_tr, mu_f, sd_f)
        y_fluo_lb_sc = apply_standardizer(y_fluo_lb, mu_f, sd_f)

        lam = compute_lam(
            cfg.loss,
            y_train=y_tr,
            fallback_lambda_power=1.0,
            fallback_lam_floor=0.25,
            fallback_lam_ceil=6.0,
        )
        posw = pos_weight_per_task(y_tr, clip=compute_posw_clips(cfg.loss, fallback_clip=50.0))
        gamma_t = compute_gamma(cfg.loss)

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
            data.starts,
            data.counts,
            data.id2pos,
            data.Xinst_sorted,
            max_instances=0,
            seed=int(self.config.seed),
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
            data.starts,
            data.counts,
            data.id2pos,
            data.Xinst_sorted,
            max_instances=0,
            seed=int(self.config.seed) + 999,
        )

        sampler_tr = make_weighted_sampler(
            y_tr,
            rare_mult=float(cfg.sampler.rare_oversample_mult),
            rare_prev_thr=float(cfg.sampler.rare_prev_thr),
            sample_weight_cap=float(cfg.sampler.sample_weight_cap),
        )

        dl_tr = self.loader_builder.train_loader(
            ds_tr,
            batch_size=int(cfg.runtime.batch_size),
            sampler=sampler_tr,
            collate_fn=collate_train,
        )
        dl_val = self.loader_builder.eval_loader(
            ds_lb,
            batch_size=min(128, int(cfg.runtime.batch_size)),
            collate_fn=collate_train,
        )

        model = MILModelBuilder.build(
            config=cfg,
            mol_dim=int(X2d_tr.shape[1]),
            inst_dim=int(data.Xinst_sorted.shape[1]),
            pos_weight=posw,
            gamma=gamma_t,
            lam=lam,
        )

        final_dir = outdir / "final_best_train_vs_leaderboard"
        final_dir.mkdir(parents=True, exist_ok=True)

        trainer_cfg = LightningTrainerConfig(
            max_epochs=int(self.config.trainer.max_epochs),
            patience=int(self.config.trainer.patience),
            accelerator=str(self.config.trainer.accelerator),
            devices=int(self.config.trainer.devices),
            precision=str(self.config.trainer.precision),
            accumulate_grad_batches=int(cfg.runtime.accumulate_grad_batches),
        )
        trainer, ckpt_cb = LightningTrainerFactory(trainer_cfg).build(
            ckpt_dir=str(final_dir),
            trial=None,
        )
        trainer.fit(model, dl_tr, dl_val)

        best_path = ckpt_cb.best_model_path
        if best_path and Path(best_path).exists():
            ckpt = torch.load(best_path, map_location="cpu")
            model.load_state_dict(ckpt["state_dict"], strict=True)
            print(f"[FINAL] loaded best ckpt: {best_path}")

        evaluator = ModelEvaluator(device=self.eval_device)
        macro_ap_lb, aps_lb = evaluator.eval_best_epoch(model, dl_val)
        eval_json = {
            "macro_ap": float(macro_ap_lb),
            "ap_task0": float(aps_lb[0]),
            "ap_task1": float(aps_lb[1]),
            "ap_task2": float(aps_lb[2]),
            "ap_task3": float(aps_lb[3]),
        }
        (final_dir / "leaderboard_eval.json").write_text(json.dumps(eval_json, indent=2))
        print(f"[FINAL] leaderboard eval: macro_ap={macro_ap_lb:.6f} aps={aps_lb}")

        export_ds = MILExportDataset(
            ids_lb,
            X2d_lb,
            starts=data.starts,
            counts=data.counts,
            id2pos=data.id2pos,
            Xinst_sorted=data.Xinst_sorted,
            conf_sorted=data.conf_sorted,
            max_instances=0,
            seed=int(self.config.seed) + 123,
        )
        export_dl = self.loader_builder.eval_loader(
            export_ds,
            batch_size=min(64, int(cfg.runtime.batch_size)),
            collate_fn=collate_export,
        )
        out_path = Path(self.config.attn_out) if self.config.attn_out else (outdir / "leaderboard_attn.parquet")
        export_leaderboard_attention(model, export_dl, device=self.eval_device, out_path=out_path)


__all__ = [
    "TrainerSystemConfig",
    "CVRunConfig",
    "StudyConfig",
    "FinalTrainConfig",
    "MILCVData",
    "MILFinalData",
    "MILFoldTrainer",
    "MILCrossValidator",
    "MILStudyRunner",
    "MILFinalTrainer",
    "StudyArtifactsWriter",
    "drop_ids_without_bags",
]
