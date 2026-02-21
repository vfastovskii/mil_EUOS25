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
    """
    Configuration class to store paths for various CLI data files.

    This immutable data class holds the paths for labels, 2D features, scaled
    3D features, QM-scaled 3D features, and the study directory. It is designed
    to ensure consistency and immutability, helping to organize and standardize
    file path configuration for a CLI application.

    Attributes:
        labels: The path to the labels file.
        feat2d_scaled: The path to the scaled 2D feature file.
        feat3d_scaled: The path to the scaled 3D feature file.
        feat3d_qm_scaled: The path to the QM-scaled 3D feature file.
        study_dir: The directory path where the study files are stored.
    """
    labels: str
    feat2d_scaled: str
    feat3d_scaled: str
    feat3d_qm_scaled: str
    study_dir: str


@dataclass(frozen=True)
class CLIColumnsConfig:
    """
    Configuration class to define column names for CLI operations.

    This class is used to specify the column names required for various CLI
    functionalities, such as identifiers, configuration details, data splits,
    and folds. It is immutable due to the use of the `@dataclass(frozen=True)`
    decorator, ensuring that once initialized, its fields cannot be modified.

    Attributes:
        id_col: The name of the column used for identifying entries.
        conf_col: The name of the column used to store configuration details.
        split_col: The name of the column used for specifying data splits.
        fold_col: The name of the column used to denote data folds.
    """
    id_col: str
    conf_col: str
    split_col: str
    fold_col: str


@dataclass(frozen=True)
class CLISplitsConfig:
    """
    Configuration for CLI splits.

    This dataclass encapsulates configuration details for CLI-based
    splits used in a data processing or machine learning workflow. It
    defines the splits to be used, folds for training/testing, and the
    leaderboard split for evaluation. The class is immutable.

    Attributes:
    use_splits (tuple[str, ...]): The dataset splits that are specified
        for use.
    folds (tuple[int, ...] | None): Optional. Specifies the folds for
        cross-validation or other purposes. Can be None if not
        applicable.
    leaderboard_split (str): Name of the dataset split designated for
        leaderboard evaluation or final assessment.
    """
    use_splits: tuple[str, ...]
    folds: tuple[int, ...] | None
    leaderboard_split: str


@dataclass(frozen=True)
class CLIRuntimeConfig:
    """
    Represents the configuration for running a CLI-based machine learning training workflow.

    This class is used to encapsulate runtime configuration settings for a machine learning
    workflow. These settings include parameters for training control, hardware preferences,
    and data loading behaviors. The configuration is immutable, and the specified parameters
    serve as directives for how the workflow should proceed.

    Attributes:
        max_epochs: Maximum number of epochs for training the model.
        patience: Number of epochs to wait for improvement before early stopping.
        trials: Number of trials to perform in hyperparameter optimization.
        seed: Seed value for reproducibility of results.
        nn_accelerator: The accelerator to use, such as "cpu", "gpu", or other supported
                        hardware.
        nn_devices: The number of devices to allocate for neural network training.
        precision: Precision format for training, such as "32-bit" or "16-bit".
        num_workers: Number of subprocesses to use for data loading.
        pin_memory: Whether to load data into pinned memory for faster transfer to device.
    """
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
    """
    Configuration class for CLI Export.

    Represents the necessary configuration settings for exporting data
    via the command-line interface. This class utilizes immutability to
    ensure the configuration's integrity during runtime.

    Attributes:
        attn_out: Specifies the output location or parameter for attention
            mechanism output. It could be a valid path or None.
    """
    attn_out: str | None


@dataclass(frozen=True)
class CLIHPOControlConfig:
    """
    Controls whether HPO is executed or precomputed params are loaded.

    Attributes:
        run_hpo: If True, run Optuna CV optimization.
        best_params_json: Optional path to JSON file with best params in the
            same format as `<study_name>_best_params.json`.
    """
    run_hpo: bool
    best_params_json: str | None


@dataclass(frozen=True)
class PipelineConfig:
    """
    Represents configuration for a processing pipeline.

    This class holds the various configuration settings required to manage the
    operation of a processing pipeline. These settings are categorized into
    several groups, such as paths for data inputs and outputs, configuration for
    data columns, splitting of data, runtime behaviors, and export configurations.
    The configuration values are immutable and encapsulated for use across the
    pipeline processes.

    Attributes:
        data_paths: Configuration for data input and output paths.
        columns: Configuration for data column usage and behavior.
        splits: Configuration for data splitting (e.g., training/testing splits).
        runtime: Configuration for runtime behaviors and settings.
        export: Configuration for exporting the results of the pipeline.
        hpo: Controls if optimization is run or loaded from file.
    """
    data_paths: CLIDataPathsConfig
    columns: CLIColumnsConfig
    splits: CLISplitsConfig
    runtime: CLIRuntimeConfig
    export: CLIExportConfig
    hpo: CLIHPOControlConfig


@dataclass(frozen=True)
class PipelineEnvironment:
    """
    Represents the environment configuration for a pipeline.

    This class is used to store settings related to the runtime environment
    of a pipeline. It includes directories, processing configurations, and
    memory management settings.

    Attributes:
        outdir: Path object representing the output directory for pipeline-related
            data and results.
        num_workers: Integer specifying the number of workers to be used for
            parallel processing tasks within the pipeline.
        pin_memory: Boolean indicating whether memory pinning is enabled for
            operations, typically useful in data loading for enhanced
            performance.
    """
    outdir: Path
    num_workers: int
    pin_memory: bool


@dataclass(frozen=True)
class PreparedHPOData:
    """
    Encapsulates prepared data for hyperparameter optimization.

    This class is designed to store and manage data that has been preprocessed
    and prepared for hyperparameter optimization. It is intended to ensure
    consistency and organization of the required datasets and files during the
    optimization workflow.

    Attributes:
    ----------
    df_full : pd.DataFrame
        The complete DataFrame containing all the relevant data for processing
        and analysis.
    ids_2d_file : List[str]
        A list of identifiers related to 2D file data.
    X2d_file : np.ndarray
        A NumPy array representing 2D file data used in the process.
    cv_data : MILCVData
        Cross-validation data utilized in model training and evaluation.
    """
    df_full: pd.DataFrame
    ids_2d_file: List[str]
    X2d_file: np.ndarray
    cv_data: MILCVData


class PipelineConfigFactory:
    """
    Factory class for creating PipelineConfig instances from command-line arguments.

    Provides a method to generate a fully populated PipelineConfig object based on
    the provided command-line arguments. This simplifies the process of configuring
    a pipeline by directly converting arguments into structured configurations
    such as paths, columns, splits, runtime settings, and export options.

    Methods:
        from_args (static): Creates a PipelineConfig object from parsed
        command-line arguments.
    """
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
                attn_out=args.attn_out,
            ),
            hpo=CLIHPOControlConfig(
                run_hpo=bool(args.run_hpo),
                best_params_json=(
                    None if args.best_params_json is None else str(args.best_params_json)
                ),
            ),
        )


class PipelineEnvironmentFactory:
    """
    PipelineEnvironmentFactory is responsible for configuring and preparing the pipeline environment.

    This class serves as a factory for constructing and configuring the pipeline environment.
    It ensures that necessary directories are created, runtime seeds and configurations are set,
    and environment metadata is logged appropriately. The main objective of this class is to
    facilitate ease of pipeline environment setup while adhering to the provided configuration.

    Attributes:
        config: PipelineConfig
            Configuration object that contains runtime and data path settings.
    """
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
            "run_hpo": bool(self.config.hpo.run_hpo),
            "best_params_json": self.config.hpo.best_params_json,
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
    """
    Class used for building HPO (Hyperparameter Optimization) data.

    This class is responsible for processing input data, applying transformations, and
    assembling all necessary components for HPO experiments. It performs steps such as
    data loading, filtering, target preparation, weight computation, and dataset segmentation.
    The main output is structured data compatible with HPO pipelines.

    Attributes:
        config: PipelineConfig
            Configuration object containing paths, column names, and split information
            required for building the HPO data.

    Methods:
        build():
            Constructs and returns the complete PreparedHPOData object by processing
            input data and preparing all required features and targets.
        _resolve_folds(df_hpo: pd.DataFrame) -> Sequence[int]:
            Resolves data folds based on configuration or directly from the input dataframe.
    """
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
    """Orchestrates the execution of a MIL (Multiple Instance Learning) pipeline.

    This class manages the high-level process of running the pipeline, including
    the configuration, the handling of preparatory steps, the execution of hyperparameter
    optimization, and the final training phase of the MIL pipeline.

    This orchestrator is designed to work with a provided pipeline configuration and
    optional command-line arguments. It ensures that the proper environment is set up,
    data is prepared, and the necessary components for the pipeline are executed in the
    correct sequence.

    Attributes:
        config: PipelineConfig
            The configuration object that holds all necessary settings for the pipeline.
        argv: Any | None
            Optional command-line arguments passed to the pipeline.

    Methods:
        run() -> None:
            Executes the main operations of the pipeline including environment setup,
            hyperparameter optimization (HPO), final training, and cleanup.
    """
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
        ckpt_root: Path | None = None

        if bool(self.config.hpo.run_hpo):
            ckpt_root = env.outdir / "_tmp_best_ckpts"
            ckpt_root.mkdir(parents=True, exist_ok=True)
            study = self._run_hpo(env=env, hpo_data=hpo_data, ckpt_root=ckpt_root)
            best_params = dict(study.best_params)
        else:
            best_params = self._load_best_params(outdir=env.outdir)

        self._run_final(env=env, hpo_data=hpo_data, best_params=best_params)

        if ckpt_root is not None:
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

    def _load_best_params(self, *, outdir: Path) -> dict[str, Any]:
        explicit = self.config.hpo.best_params_json
        if explicit is not None:
            path = Path(explicit)
        else:
            path = outdir / "multimodal_mil_aux_gpu_best_params.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Best-params file not found: {path}. "
                "Run with --run_hpo or provide --best_params_json <file>."
            )
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object in {path}, got {type(payload).__name__}")
        print(f"[HPO] using precomputed params from: {path}")
        return dict(payload)

    def _run_final(self, *, env: PipelineEnvironment, hpo_data: PreparedHPOData, best_params: dict[str, Any]):
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
            best_params=dict(best_params),
            data=final_data,
        )


def _parse_args(argv: Any | None = None):
    """
    Parses and processes command-line arguments for configuring training and evaluation.

    The function uses `argparse` to define and interpret various command-line arguments
    required for launching and configuring a machine learning training pipeline. It
    supports parameters for data paths, training settings, splits, cross-validation,
    and experiment-specific controls. Deprecated and compatibility-related flags are
    also included for backward compatibility.

    Args:
        argv: Optional specification of command-line arguments. If None, defaults
              to `sys.argv`.

    Returns:
        Namespace: A namespace populated with the parsed arguments where the associated
                   flags and options can be accessed as attributes.
    """
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
    ap.add_argument(
        "--run_hpo",
        action="store_true",
        help="Run Optuna CV optimization before final train.",
    )
    ap.add_argument(
        "--best_params_json",
        default=None,
        help=(
            "Path to precomputed best params JSON (pipeline format). "
            "Used when --run_hpo is not set."
        ),
    )

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nn_accelerator", default="gpu")
    ap.add_argument("--nn_devices", type=int, default=1)
    ap.add_argument("--precision", default="16-mixed")
    ap.add_argument("--num_workers", type=int, default=-1)
    ap.add_argument("--pin_memory", action="store_true")

    # Deprecated compatibility flag; final train+leaderboard stage now always runs.
    ap.add_argument("--export_leaderboard_attn", action="store_true")
    ap.add_argument("--leaderboard_split", default="leaderboard")
    ap.add_argument("--attn_out", default=None)

    # Accepted for compatibility; pipeline is MIL-only now.
    ap.add_argument("--do_mil", action="store_true")

    return ap.parse_args(argv)


def _normalize_compat_args(args) -> None:
    """
    Normalizes compatibility arguments for trials.

    This function converts the 'trials_mil' argument in the provided object
    to an integer and assigns it to the 'trials' attribute if 'trials_mil'
    is not None. It modifies the object in place and does not return any
    value.

    Parameters:
        args (Any): The object containing trials_mil and potentially other attributes.

    Returns:
        None
    """
    if args.trials_mil is not None:
        args.trials = int(args.trials_mil)


def main(argv: Any | None = None) -> None:
    """
    The `main` function serves as the entry point for the execution of the pipeline
    orchestration process. It initializes the required configurations and manages
    the execution of a pipeline by invoking the orchestrator.

    Args:
        argv (Any | None): Command-line arguments provided by the user or defaulted
            to None if not provided.

    Raises:
        None

    Returns:
        None
    """
    args = _parse_args(argv)
    _normalize_compat_args(args)
    config = PipelineConfigFactory.from_args(args)
    MILPipelineOrchestrator(config=config, argv=argv).run()


if __name__ == "__main__":
    import sys as _sys

    main(_sys.argv[1:])
