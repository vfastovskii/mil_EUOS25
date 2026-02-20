from __future__ import annotations

"""
Class-based CLI orchestration for MIL HPO/training pipeline.

Public package exports live in `opt_attn_net_feb.__init__`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence
import gc
import json
import os
import shutil
import time

import numpy as np
import pandas as pd
import torch

from ..training.builders import LoaderConfig
from ..training.execution import (
    CVRunConfig,
    FinalTrainConfig,
    MILCVData,
    MILCrossValidator,
    MILFinalData,
    MILFinalTrainer,
    MILStudyRunner,
    StudyConfig,
    TrainerSystemConfig,
)
from ..utils.constants import WEIGHT_COLS
from ..utils.data_io import align_by_id, load_2d, load_labels
from ..utils.instances import build_instance_index, load_and_merge_instances
from ..utils.ops import (
    build_aux_targets_and_masks,
    build_aux_weights,
    build_task_weights,
    coerce_binary_labels,
    fold_indices,
    maybe_set_torch_fast_flags,
    set_all_seeds,
)


@dataclass(frozen=True)
class CLIDataPathsConfig:
    labels: str
    feat2d_scaled: str
    feat3d_scaled: str
    feat3d_qm_scaled: str
    study_dir: str


@dataclass(frozen=True)
class CLIColumnsConfig:
    id_col: str
    conf_col: str
    split_col: str
    fold_col: str


@dataclass(frozen=True)
class CLISplitsConfig:
    use_splits: tuple[str, ...]
    folds: tuple[int, ...] | None
    leaderboard_split: str


@dataclass(frozen=True)
class CLIRuntimeConfig:
    max_epochs: int
    patience: int
    trials: int
    seed: int
    nn_accelerator: str
    nn_devices: int
    precision: str
    num_workers: int
    pin_memory: bool


@dataclass(frozen=True)
class CLIExportConfig:
    export_leaderboard_attn: bool
    attn_out: str | None


@dataclass(frozen=True)
class PipelineConfig:
    data_paths: CLIDataPathsConfig
    columns: CLIColumnsConfig
    splits: CLISplitsConfig
    runtime: CLIRuntimeConfig
    export: CLIExportConfig


@dataclass(frozen=True)
class PipelineEnvironment:
    outdir: Path
    num_workers: int
    pin_memory: bool


@dataclass(frozen=True)
class PreparedHPOData:
    df_full: pd.DataFrame
    ids_2d_file: List[str]
    X2d_file: np.ndarray
    cv_data: MILCVData


class PipelineConfigFactory:
    @staticmethod
    def from_args(args) -> PipelineConfig:
        return PipelineConfig(
            data_paths=CLIDataPathsConfig(
                labels=str(args.labels),
                feat2d_scaled=str(args.feat2d_scaled),
                feat3d_scaled=str(args.feat3d_scaled),
                feat3d_qm_scaled=str(args.feat3d_qm_scaled),
                study_dir=str(args.study_dir),
            ),
            columns=CLIColumnsConfig(
                id_col=str(args.id_col),
                conf_col=str(args.conf_col),
                split_col=str(args.split_col),
                fold_col=str(args.fold_col),
            ),
            splits=CLISplitsConfig(
                use_splits=tuple(str(x) for x in args.use_splits),
                folds=(None if args.folds is None else tuple(map(int, args.folds))),
                leaderboard_split=str(args.leaderboard_split),
            ),
            runtime=CLIRuntimeConfig(
                max_epochs=int(args.max_epochs),
                patience=int(args.patience),
                trials=int(args.trials),
                seed=int(args.seed),
                nn_accelerator=str(args.nn_accelerator),
                nn_devices=int(args.nn_devices),
                precision=str(args.precision),
                num_workers=int(args.num_workers),
                pin_memory=bool(args.pin_memory),
            ),
            export=CLIExportConfig(
                export_leaderboard_attn=bool(args.export_leaderboard_attn),
                attn_out=args.attn_out,
            ),
        )


class PipelineEnvironmentFactory:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def prepare(self, *, argv: Any | None) -> PipelineEnvironment:
        set_all_seeds(int(self.config.runtime.seed))
        maybe_set_torch_fast_flags()

        outdir = Path(self.config.data_paths.study_dir)
        outdir.mkdir(parents=True, exist_ok=True)

        run_meta = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seed": int(self.config.runtime.seed),
            "nn_accelerator": str(self.config.runtime.nn_accelerator),
            "nn_devices": int(self.config.runtime.nn_devices),
            "precision": str(self.config.runtime.precision),
            "patience": int(self.config.runtime.patience),
            "argv": " ".join([str(x) for x in (argv if argv is not None else os.sys.argv)]),
            "weight_cols": WEIGHT_COLS,
            "model": "MILTaskAttnMixerWithAux (task-specific attention queries)",
        }
        (outdir / "run_meta.json").write_text(json.dumps(run_meta, indent=2))

        num_workers = self._resolve_num_workers()
        pin_memory = bool(self.config.runtime.pin_memory) and torch.cuda.is_available()
        return PipelineEnvironment(outdir=outdir, num_workers=num_workers, pin_memory=pin_memory)

    def _resolve_num_workers(self) -> int:
        if int(self.config.runtime.num_workers) >= 0:
            return int(self.config.runtime.num_workers)
        cpus = int(os.environ.get("SLURM_CPUS_PER_TASK") or 0)
        if cpus <= 0:
            cpus = os.cpu_count() or 0
        return max(2, min(23, cpus - 2)) if cpus >= 4 else 0


class HPODataBuilder:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def build(self) -> PreparedHPOData:
        c = self.config.columns
        p = self.config.data_paths
        s = self.config.splits

        df_full = load_labels(p.labels, id_col=c.id_col)
        df_full[c.id_col] = df_full[c.id_col].astype(str)
        df_full[c.split_col] = df_full[c.split_col].astype(str)

        df_hpo = df_full[df_full[c.split_col].isin(s.use_splits)].copy().reset_index(drop=True)
        if len(df_hpo) == 0:
            raise ValueError(f"No rows in labels match use_splits={s.use_splits}")

        ids_hpo = df_hpo[c.id_col].astype(str).tolist()
        ids_2d_file, X2d_file = load_2d(p.feat2d_scaled, id_col=c.id_col)
        X2d_hpo = align_by_id(ids_2d_file, X2d_file, ids_hpo)

        y_cls = coerce_binary_labels(df_hpo)
        w_cls = build_task_weights(df_hpo)
        y_abs, m_abs, y_fluo, m_fluo = build_aux_targets_and_masks(df_hpo)
        w_abs, w_fluo = build_aux_weights(df_hpo)

        folds = self._resolve_folds(df_hpo)
        folds_info = fold_indices(df_hpo, c.fold_col, folds)

        ids_conf_hpo, conf_ids_hpo, Xinst_hpo = load_and_merge_instances(
            p.feat3d_scaled,
            p.feat3d_qm_scaled,
            allowed_ids=set(ids_hpo),
            id_col=c.id_col,
            conf_col=c.conf_col,
        )
        _, starts_hpo, counts_hpo, id2pos_hpo, Xinst_sorted_hpo, _ = build_instance_index(
            ids_conf_hpo,
            conf_ids_hpo,
            Xinst_hpo,
        )

        have_bag_mask = np.array([(i in id2pos_hpo) for i in ids_hpo], dtype=bool)
        if not have_bag_mask.all():
            missing = int((~have_bag_mask).sum())
            examples = [ids_hpo[i] for i in np.where(~have_bag_mask)[0][:10]]
            print(
                f"[WARN] Dropping {missing} HPO IDs with 0 conformers after merge. Examples: {examples}"
            )

            df_hpo = df_hpo.loc[have_bag_mask].reset_index(drop=True)
            ids_hpo = df_hpo[c.id_col].astype(str).tolist()
            X2d_hpo = X2d_hpo[have_bag_mask]
            y_cls = y_cls[have_bag_mask]
            w_cls = w_cls[have_bag_mask]
            y_abs = y_abs[have_bag_mask]
            m_abs = m_abs[have_bag_mask]
            y_fluo = y_fluo[have_bag_mask]
            m_fluo = m_fluo[have_bag_mask]
            w_abs = w_abs[have_bag_mask]
            w_fluo = w_fluo[have_bag_mask]
            folds = sorted(df_hpo[c.fold_col].dropna().astype(int).unique().tolist())
            folds_info = fold_indices(df_hpo, c.fold_col, folds)

        print(f"[DATA-HPO] n_ids={len(ids_hpo)} | X2d_dim={X2d_hpo.shape[1]}")
        print(
            f"[DATA-HPO] n_conf={Xinst_sorted_hpo.shape[0]} | inst_dim={Xinst_sorted_hpo.shape[1]}"
        )

        cv_data = MILCVData(
            X2d_scaled=X2d_hpo,
            y_cls=y_cls,
            w_cls=w_cls,
            y_abs=y_abs,
            m_abs=m_abs,
            w_abs=w_abs,
            y_fluo=y_fluo,
            m_fluo=m_fluo,
            w_fluo=w_fluo,
            ids=ids_hpo,
            folds_info=folds_info,
            starts=starts_hpo,
            counts=counts_hpo,
            id2pos=id2pos_hpo,
            Xinst_sorted=Xinst_sorted_hpo,
        )
        return PreparedHPOData(
            df_full=df_full,
            ids_2d_file=ids_2d_file,
            X2d_file=X2d_file,
            cv_data=cv_data,
        )

    def _resolve_folds(self, df_hpo: pd.DataFrame) -> Sequence[int]:
        if self.config.splits.folds is None:
            return sorted(df_hpo[self.config.columns.fold_col].dropna().astype(int).unique().tolist())
        return list(self.config.splits.folds)


class MILPipelineOrchestrator:
    def __init__(self, *, config: PipelineConfig, argv: Any | None):
        self.config = config
        self.argv = argv

    def run(self) -> None:
        env = PipelineEnvironmentFactory(self.config).prepare(argv=self.argv)
        hpo_data = HPODataBuilder(self.config).build()

        print(
            "[DATALOADER] "
            f"num_workers={env.num_workers} pin_memory={env.pin_memory} "
            f"precision={self.config.runtime.precision}"
        )
        ckpt_root = env.outdir / "_tmp_best_ckpts"
        ckpt_root.mkdir(parents=True, exist_ok=True)

        study = self._run_hpo(env=env, hpo_data=hpo_data, ckpt_root=ckpt_root)
        self._run_final_if_needed(env=env, hpo_data=hpo_data, study=study)

        try:
            shutil.rmtree(ckpt_root, ignore_errors=True)
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def _run_hpo(self, *, env: PipelineEnvironment, hpo_data: PreparedHPOData, ckpt_root: Path):
        run_cfg = CVRunConfig(
            seed=int(self.config.runtime.seed),
            trainer=TrainerSystemConfig(
                max_epochs=int(self.config.runtime.max_epochs),
                patience=int(self.config.runtime.patience),
                accelerator=str(self.config.runtime.nn_accelerator),
                devices=int(self.config.runtime.nn_devices),
                precision=str(self.config.runtime.precision),
            ),
            loader=LoaderConfig(
                num_workers=int(env.num_workers),
                pin_memory=bool(env.pin_memory),
            ),
            ckpt_root=ckpt_root,
        )
        cross_validator = MILCrossValidator(data=hpo_data.cv_data, run_config=run_cfg)
        study_runner = MILStudyRunner(
            config=StudyConfig(
                outdir=env.outdir,
                study_name="multimodal_mil_aux_gpu",
                n_trials=int(self.config.runtime.trials),
                seed=int(self.config.runtime.seed),
            ),
            cross_validator=cross_validator,
        )
        return study_runner.run()

    def _run_final_if_needed(self, *, env: PipelineEnvironment, hpo_data: PreparedHPOData, study):
        if not self.config.export.export_leaderboard_attn:
            return

        c = self.config.columns
        p = self.config.data_paths
        split = self.config.splits
        df_full = hpo_data.df_full

        allowed_final = set(
            df_full[df_full[c.split_col].isin(["train", split.leaderboard_split])][c.id_col]
            .astype(str)
            .tolist()
        )
        ids_conf_all, conf_ids_all, Xinst_all = load_and_merge_instances(
            p.feat3d_scaled,
            p.feat3d_qm_scaled,
            allowed_ids=allowed_final,
            id_col=c.id_col,
            conf_col=c.conf_col,
        )
        _, starts_all, counts_all, id2pos_all, Xinst_sorted_all, conf_sorted_all = build_instance_index(
            ids_conf_all,
            conf_ids_all,
            Xinst_all,
        )

        final_data = MILFinalData(
            df_full=df_full,
            id_col=c.id_col,
            split_col=c.split_col,
            leaderboard_split=split.leaderboard_split,
            X2d_file_ids=hpo_data.ids_2d_file,
            X2d_file=hpo_data.X2d_file,
            starts=starts_all,
            counts=counts_all,
            id2pos=id2pos_all,
            Xinst_sorted=Xinst_sorted_all,
            conf_sorted=conf_sorted_all,
        )
        final_cfg = FinalTrainConfig(
            seed=int(self.config.runtime.seed),
            trainer=TrainerSystemConfig(
                max_epochs=int(self.config.runtime.max_epochs),
                patience=int(self.config.runtime.patience),
                accelerator=str(self.config.runtime.nn_accelerator),
                devices=int(self.config.runtime.nn_devices),
                precision=str(self.config.runtime.precision),
            ),
            loader=LoaderConfig(
                num_workers=int(env.num_workers),
                pin_memory=bool(env.pin_memory),
            ),
            attn_out=self.config.export.attn_out,
        )
        MILFinalTrainer(config=final_cfg).run(
            outdir=env.outdir,
            best_params=dict(study.best_params),
            data=final_data,
        )


def _parse_args(argv: Any | None = None):
    import argparse

    ap = argparse.ArgumentParser()

    ap.add_argument("--labels", required=True)
    ap.add_argument("--feat2d_scaled", required=True)
    ap.add_argument("--feat3d_scaled", required=True)
    ap.add_argument("--feat3d_qm_scaled", required=True)
    ap.add_argument("--study_dir", required=True)

    ap.add_argument("--id_col", default="ID")
    ap.add_argument("--conf_col", default="conf_id")
    ap.add_argument("--split_col", default="split")
    ap.add_argument("--fold_col", default="cv_fold")

    ap.add_argument("--use_splits", nargs="+", default=["train"])
    ap.add_argument("--folds", nargs="+", type=int, default=None)

    ap.add_argument("--max_epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--trials_mil", type=int, default=None)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nn_accelerator", default="gpu")
    ap.add_argument("--nn_devices", type=int, default=1)
    ap.add_argument("--precision", default="16-mixed")
    ap.add_argument("--num_workers", type=int, default=-1)
    ap.add_argument("--pin_memory", action="store_true")

    ap.add_argument("--export_leaderboard_attn", action="store_true")
    ap.add_argument("--leaderboard_split", default="leaderboard")
    ap.add_argument("--attn_out", default=None)

    # Accepted for compatibility; pipeline is MIL-only now.
    ap.add_argument("--do_mil", action="store_true")

    return ap.parse_args(argv)


def _normalize_compat_args(args) -> None:
    if args.trials_mil is not None:
        args.trials = int(args.trials_mil)


def main(argv: Any | None = None) -> None:
    args = _parse_args(argv)
    _normalize_compat_args(args)
    config = PipelineConfigFactory.from_args(args)
    MILPipelineOrchestrator(config=config, argv=argv).run()


if __name__ == "__main__":
    import sys as _sys

    main(_sys.argv[1:])
