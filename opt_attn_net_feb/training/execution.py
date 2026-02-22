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
    build_bitmask_group_definition,
    build_aux_targets_and_masks,
    build_aux_weights,
    build_task_weights,
    coerce_binary_labels,
    fit_standardizer,
    make_balanced_batch_sampler,
    make_bitmask_sample_weights,
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
    """
    Resolves the device to be used for computation based on the availability of CUDA
    and the specified accelerator type.

    Parameters:
    accelerator (str): The desired accelerator type. It can be 'gpu', 'cuda', or other
                       strings indicating the accelerator preference.

    Returns:
    torch.device: The resolved computation device, either 'cuda' if available and
                  requested, or 'cpu' otherwise.
    """
    return torch.device(
        "cuda" if torch.cuda.is_available() and str(accelerator) in ("gpu", "cuda") else "cpu"
    )


@dataclass(frozen=True)
class TrainerSystemConfig:
    """
    A configuration class for defining system parameters for a trainer.

    This class provides a structured way to define and store configuration settings
    for training systems. The configuration settings include values for controlling
    training duration, hardware utilization, and computational precision.

    Attributes:
        max_epochs: int
            The maximum number of epochs allowed for training.
        patience: int
            The number of epochs to wait for improvement before applying early stopping.
        accelerator: str
            The type of hardware accelerator to use for training, such as 'cpu' or 'gpu'.
        devices: int
            The number of devices available for distributed training.
        precision: str
            The precision to use during training, typically 'float32', 'bfloat16', etc.
    """
    max_epochs: int
    patience: int
    accelerator: str
    devices: int
    precision: str


@dataclass(frozen=True)
class CVRunConfig:
    """
    Configuration for a cross-validation run.

    Encapsulates the settings required for executing a single run of a
    cross-validation process, including random seed for reproducibility,
    training system configurations, data loading specifications, and the
    root path for storing checkpoints.

    Attributes:
    seed: Random seed for ensuring reproducibility of experiments. Type: int.
    trainer: Configuration for the training system. Type: TrainerSystemConfig.
    loader: Configuration for the data loader. Type: LoaderConfig.
    ckpt_root: Path to the root directory for storing checkpoints. Type: Path.
    """
    seed: int
    trainer: TrainerSystemConfig
    loader: LoaderConfig
    ckpt_root: Path


@dataclass(frozen=True)
class StudyConfig:
    """
    Represents configuration for a study.

    This class defines the parameters required to configure and execute a study. It
    includes details such as the output directory, study name, number of trials,
    direction of optimization, random seed, and pruning configuration. The dataclass
    is immutable to ensure the configuration cannot be altered once initialized.

    Attributes:
    outdir (Path): The output directory where study results will be stored.
    study_name (str): The name of the study.
    n_trials (int): The number of trials to execute in the study.
    seed (int): The random seed for reproducibility.
    direction (str, optional): The optimization direction, either "maximize" or
    "minimize". Defaults to "maximize".
    pruner_kind (str, optional): Pruner type, either "percentile" or "median".
        Defaults to "percentile".
    pruner_warmup_steps (int, optional): The number of warmup steps before pruning
        trials. Defaults to 8.
    pruner_startup_trials (int, optional): Number of full trials to run before
        enabling pruning decisions. Defaults to 10.
    pruner_percentile (float, optional): Percentile threshold used by
        PercentilePruner. Lower values are less aggressive. Defaults to 25.0.
    """
    outdir: Path
    study_name: str
    n_trials: int
    seed: int
    direction: str = "maximize"
    pruner_kind: str = "percentile"
    pruner_warmup_steps: int = 8
    pruner_startup_trials: int = 10
    pruner_percentile: float = 25.0


@dataclass(frozen=True)
class FinalTrainConfig:
    """
    Configuration data class for the final training setup.

    Represents the configuration needed for initiating the final training process,
    including parameters for seeding, trainer configuration, data loader setup, and
    optional attention output management.
    """
    seed: int
    trainer: TrainerSystemConfig
    loader: LoaderConfig
    attn_out: str | None = None


@dataclass(frozen=True)
class MILCVData:
    """
    MILCVData serves as a container for data related to Multiple Instance Learning (MIL)
    with cross-validation support. The class aggregates data components including features,
    labels, weights, IDs, fold information, and metadata for handling MIL-specific datasets.

    The class is designed to hold and organize scaled features, classification and regression
    labels, sample weights, and additional attributes required for MIL analysis. It also
    includes fold-specific information and mappings that facilitate the corresponding
    cross-validation workflows.

    Attributes:
        X2d_scaled (np.ndarray): Scaled 2D feature array.
        y_cls (np.ndarray): Classification labels.
        w_cls (np.ndarray): Weights associated with classification labels.
        y_abs (np.ndarray): Absolute regression labels.
        m_abs (np.ndarray): Mask for absolute regression.
        w_abs (np.ndarray): Weights associated with absolute regression labels.
        y_fluo (np.ndarray): Fluorescence-related regression labels.
        m_fluo (np.ndarray): Mask for fluorescence-related regression.
        w_fluo (np.ndarray): Weights associated with fluorescence-related regression labels.
        ids (List[str]): Instance or observation identifiers.
        folds_info (Sequence[Tuple[np.ndarray, np.ndarray, int]]): Cross-validation fold
            information consisting of training indices, validation indices, and the fold
            integer identifier.
        starts (np.ndarray): Start indices for MIL bag-level data.
        counts (np.ndarray): Instance counts corresponding to MIL bags.
        id2pos (Dict[str, int]): Mapping from instance/ID to positional index.
        Xinst_sorted (np.ndarray): Instance-level feature array sorted for efficient
            processing.
    """
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
    """
    Represents a container for finalized multi-instance learning (MIL) input data.

    This class is designed to organize, store, and maintain information about
    processed data required for multi-instance learning tasks. The attributes
    define the structure of input data and metadata, which are critical for
    handling and processing MIL-specific datasets.

    Attributes:
        df_full (pd.DataFrame): Full dataset DataFrame, including all data points.
        id_col (str): Column name in `df_full` representing unique IDs for
            instances or bags.
        split_col (str): Column name in `df_full` defining data splits (e.g.,
            train/test/validation).
        leaderboard_split (str): Identifier for a specific split designated for
            leaderboard or evaluation purposes.
        X2d_file_ids (List[str]): List of file IDs corresponding to 2D features,
            intended for MIL-specific feature representation.
        X2d_file (np.ndarray): 2D features in the form of a NumPy array, typically
            used for MIL representation.
        starts (np.ndarray): Array of start indices for instances or bags within
            the dataset, mapping to their corresponding positions.
        counts (np.ndarray): Array of counts or lengths indicating how many
            instances are associated with each bag or unique ID.
        id2pos (Dict[str, int]): Dictionary mapping unique IDs to their positions
            within the dataset for fast access and lookup.
        Xinst_sorted (np.ndarray): NumPy array of sorted instance-level features,
            ensuring organized MIL feature representation.
        conf_sorted (np.ndarray): NumPy array of sorted confidences, aligning with
            `Xinst_sorted` for consistent ordering.
    """
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
    """
    Filters out IDs that do not have corresponding bags (positions) in the id2pos mapping.

    This function takes a list of IDs, a 2D array, a DataFrame, and a mapping of IDs to
    positions. It filters the input data to retain only those entries in the list of IDs
    that are present in the id2pos dictionary. If all IDs are present in the id2pos
    mapping, the input data is returned as-is.

    Parameters:
    ids: List[str]
        A list of IDs to be filtered.
    X2d: np.ndarray
        A 2D numpy array corresponding to the provided IDs. The array will be filtered
        based on the IDs that have corresponding positions in the id2pos mapping.
    df_part: pd.DataFrame
        A DataFrame containing data corresponding to the IDs. Rows in the DataFrame
        will be filtered based on the IDs that have corresponding positions in id2pos.
    id2pos: Dict[str, int]
        A dictionary mapping IDs to positions. Only IDs present in this dictionary will
        be retained in the filtered output.
    id_col: str
        The name of the column in the DataFrame that contains the IDs.

    Returns:
    Tuple[List[str], np.ndarray, pd.DataFrame]
        A tuple containing the filtered list of IDs, the filtered 2D numpy array, and
        the filtered DataFrame. Rows and elements in the outputs correspond only to
        IDs that exist in the id2pos mapping.
    """
    mask = np.array([(i in id2pos) for i in ids], dtype=bool)
    if mask.all():
        return ids, X2d, df_part
    df2 = df_part.loc[mask].reset_index(drop=True)
    ids2 = df2[id_col].astype(str).tolist()
    return ids2, X2d[mask], df2


class MILFoldTrainer:
    """
    Manages training over multiple Instance Learning (MIL) data folds.

    This class orchestrates the MIL training process for a given fold of data
    evaluation under a trial-specific hyperparameter optimization (HPO) configuration
    and cross-validation setup. The main goal is to tune the model using relevant
    parameters and obtain optimized scores while evaluating for each data fold.
    The fold-specific results and configurations can subsequently be used for
    model comparison and selection.

    Attributes:
        trial: Specific trial being executed during HPO search.
        hpo_config: Configuration instance containing hyperparameters for the training.
        data: MILCVData instance encapsulating training data and metadata.
        run_config: Configuration for the runtime setup, including training parameters.

    Methods:
        run_fold:
            Executes the training and evaluation for a specific fold of the data.
    """

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

        w_cls_tr = np.asarray(self.data.w_cls[train_idx], dtype=np.float32).copy()
        if bool(cfg.sampler.use_bitmask_loss_weight):
            bitmask_w = make_bitmask_sample_weights(
                self.data.y_cls[train_idx],
                alpha=float(cfg.sampler.bitmask_weight_alpha),
                cap=float(cfg.sampler.bitmask_weight_cap),
            )
            w_cls_tr = (w_cls_tr * bitmask_w.reshape(-1, 1)).astype(np.float32)

        ids_tr = [self.data.ids[i] for i in train_idx]
        ids_va = [self.data.ids[i] for i in val_idx]

        ds_tr = MILTrainDataset(
            ids_tr,
            self.data.X2d_scaled[train_idx],
            self.data.y_cls[train_idx],
            w_cls_tr,
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

        if bool(cfg.sampler.use_balanced_batch_sampler):
            batch_sampler = make_balanced_batch_sampler(
                self.data.y_cls[train_idx],
                batch_size=int(cfg.runtime.batch_size),
                rare_mult=float(cfg.sampler.rare_oversample_mult),
                rare_target_prev=float(cfg.sampler.rare_target_prev),
                sample_weight_cap=float(cfg.sampler.sample_weight_cap),
                batch_pos_fraction=float(cfg.sampler.batch_pos_fraction),
                min_pos_per_batch=int(cfg.sampler.min_pos_per_batch),
                enforce_bitmask_quota=bool(cfg.sampler.enforce_bitmask_quota),
                quota_t450_per_256=int(cfg.sampler.quota_t450_per_256),
                quota_fgt480_per_256=int(cfg.sampler.quota_fgt480_per_256),
                quota_multi_per_256=int(cfg.sampler.quota_multi_per_256),
                rare_prev_thr=cfg.sampler.rare_prev_thr,
                seed=int(self.run_config.seed) + 1000 * int(fold_id) + int(self.trial.number),
            )
            dl_tr = self.loader_builder.train_loader(
                ds_tr,
                batch_size=int(cfg.runtime.batch_size),
                batch_sampler=batch_sampler,
                collate_fn=collate_train,
            )
            sampler = batch_sampler
        else:
            sampler = make_weighted_sampler(
                self.data.y_cls[train_idx],
                rare_mult=float(cfg.sampler.rare_oversample_mult),
                rare_target_prev=float(cfg.sampler.rare_target_prev),
                sample_weight_cap=float(cfg.sampler.sample_weight_cap),
                rare_prev_thr=cfg.sampler.rare_prev_thr,
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

        bitmask_group_top_ids, bitmask_group_class_weight = build_bitmask_group_definition(
            self.data.y_cls[train_idx],
            top_k=int(cfg.loss.bitmask_group_top_k),
            class_weight_alpha=float(cfg.loss.bitmask_group_weight_alpha),
            class_weight_cap=float(cfg.loss.bitmask_group_weight_cap),
        )

        model = MILModelBuilder.build(
            config=cfg,
            mol_dim=int(self.data.X2d_scaled.shape[1]),
            inst_dim=int(self.data.Xinst_sorted.shape[1]),
            pos_weight=posw,
            gamma=gamma_t,
            lam=lam,
            bitmask_group_top_ids=bitmask_group_top_ids,
            bitmask_group_class_weight=bitmask_group_class_weight,
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
            save_checkpoint=False,
            save_weights_only=True,
        )
        trainer, ckpt_cb = LightningTrainerFactory(trainer_cfg).build(
            ckpt_dir=str(fold_ckpt_dir),
            trial=self.trial,
        )
        trainer.fit(model, dl_tr, dl_va)

        epochs_trained = int(trainer.current_epoch) + 1

        best_epoch = None
        if ckpt_cb is not None:
            best_path = ckpt_cb.best_model_path
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
    """
    Provides functionality for multi-instance learning cross-validation.

    This class is designed to handle the cross-validation process for multi-instance
    learning (MIL) tasks. It takes in a dataset and configuration details, and
    performs the necessary evaluation by dividing the dataset into folds, training
    and validating on these folds, and computing performance scores. It integrates
    hyperparameter optimization (HPO) and mode selection for evaluating different
    configurations.

    Attributes:
        data (MILCVData): The dataset and cross-validation folding information.
        run_config (CVRunConfig): Configuration details for running the cross-validation.
    """

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
    """
    Provides methods for saving study artifacts and metrics.

    This class defines static methods to save Optuna study artifacts such as trials
    information and best parameters in structured formats, along with capabilities
    to save fold-specific metrics in a consistent manner.
    """
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
    """
    MILStudyRunner is a class for managing and executing hyperparameter optimization
    (HPO) studies using the Optuna framework.

    This class is designed to facilitate the creation and execution of machine
    learning cross-validation studies. It handles study storage, sampler and pruner
    configuration, and manages the optimization trials and results storage. Users can
    apply this to systematically tune hyperparameters for machine learning models.

    Attributes:
        config (StudyConfig): Configuration for the study.
        cross_validator (MILCrossValidator): Object that evaluates individual trials.
    """

    def __init__(self, *, config: StudyConfig, cross_validator: MILCrossValidator):
        self.config = config
        self.cross_validator = cross_validator

    def _make_storage(self) -> str:
        return f"sqlite:///{(self.config.outdir / f'{self.config.study_name}.sqlite3').as_posix()}"

    def run(self) -> optuna.Study:
        self.config.outdir.mkdir(parents=True, exist_ok=True)
        sampler = optuna.samplers.TPESampler(seed=int(self.config.seed))
        if str(self.config.pruner_kind).lower() == "median":
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=int(self.config.pruner_startup_trials),
                n_warmup_steps=int(self.config.pruner_warmup_steps),
            )
        else:
            # Default: less aggressive than median pruning for sparse multitask AP.
            pruner = optuna.pruners.PercentilePruner(
                percentile=float(self.config.pruner_percentile),
                n_startup_trials=int(self.config.pruner_startup_trials),
                n_warmup_steps=int(self.config.pruner_warmup_steps),
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
    """
    Represents a MIL (Multiple Instance Learning) final trainer.

    This class is responsible for configuring, constructing datasets, initializing models,
    and managing the training process for MIL tasks. It uses the provided configurations
    and data to execute a training pipeline, including data preparation, loader creation,
    model building, and trainer setup. The class handles auxiliary target creation,
    standardization, and sampling strategies to improve the robustness of training.
    It ensures resources such as data loaders, model instances, and trainer configurations
    are optimally utilized for evaluation and comparison between training and leaderboard
    datasets.

    Attributes:
        config (FinalTrainConfig): Configuration object containing parameters for training,
            including dataset loader settings, trainer behavior, and other necessary options.
        loader_builder (DataLoaderBuilder): Builder instance used to create train and
            evaluation loaders based on the specified data configuration.
        eval_device: The device resolved for evaluation tasks, based on trainer configuration.

    Methods:
        run(outdir: Path, best_params: Dict[str, Any], data: MILFinalData) -> None:
            Executes the full training pipeline. Prepares datasets, applies auxiliary data
            transformations, builds data loaders, initializes the model, and trains it using
            a configured trainer. Saves the best model checkpoint for further evaluation.
    """

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

        if bool(cfg.sampler.use_bitmask_loss_weight):
            bitmask_w_tr = make_bitmask_sample_weights(
                y_tr,
                alpha=float(cfg.sampler.bitmask_weight_alpha),
                cap=float(cfg.sampler.bitmask_weight_cap),
            )
            w_tr = (np.asarray(w_tr, dtype=np.float32) * bitmask_w_tr.reshape(-1, 1)).astype(
                np.float32
            )

        bitmask_group_top_ids, bitmask_group_class_weight = build_bitmask_group_definition(
            y_tr,
            top_k=int(cfg.loss.bitmask_group_top_k),
            class_weight_alpha=float(cfg.loss.bitmask_group_weight_alpha),
            class_weight_cap=float(cfg.loss.bitmask_group_weight_cap),
        )

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

        if bool(cfg.sampler.use_balanced_batch_sampler):
            sampler_tr = make_balanced_batch_sampler(
                y_tr,
                batch_size=int(cfg.runtime.batch_size),
                rare_mult=float(cfg.sampler.rare_oversample_mult),
                rare_target_prev=float(cfg.sampler.rare_target_prev),
                sample_weight_cap=float(cfg.sampler.sample_weight_cap),
                batch_pos_fraction=float(cfg.sampler.batch_pos_fraction),
                min_pos_per_batch=int(cfg.sampler.min_pos_per_batch),
                enforce_bitmask_quota=bool(cfg.sampler.enforce_bitmask_quota),
                quota_t450_per_256=int(cfg.sampler.quota_t450_per_256),
                quota_fgt480_per_256=int(cfg.sampler.quota_fgt480_per_256),
                quota_multi_per_256=int(cfg.sampler.quota_multi_per_256),
                rare_prev_thr=cfg.sampler.rare_prev_thr,
                seed=int(self.config.seed) + 4242,
            )
            dl_tr = self.loader_builder.train_loader(
                ds_tr,
                batch_size=int(cfg.runtime.batch_size),
                batch_sampler=sampler_tr,
                collate_fn=collate_train,
            )
        else:
            sampler_tr = make_weighted_sampler(
                y_tr,
                rare_mult=float(cfg.sampler.rare_oversample_mult),
                rare_target_prev=float(cfg.sampler.rare_target_prev),
                sample_weight_cap=float(cfg.sampler.sample_weight_cap),
                rare_prev_thr=cfg.sampler.rare_prev_thr,
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
            bitmask_group_top_ids=bitmask_group_top_ids,
            bitmask_group_class_weight=bitmask_group_class_weight,
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
            # Keep a single best checkpoint for the final run only.
            save_checkpoint=True,
            save_weights_only=True,
        )
        trainer, ckpt_cb = LightningTrainerFactory(trainer_cfg).build(
            ckpt_dir=str(final_dir),
            trial=None,
        )
        trainer.fit(model, dl_tr, dl_val)

        best_epoch = None
        best_ckpt_path = None
        if ckpt_cb is not None:
            best_path = ckpt_cb.best_model_path
            if best_path and Path(best_path).exists():
                ckpt = torch.load(best_path, map_location="cpu")
                best_epoch = int(ckpt.get("epoch", -1))
                best_ckpt_path = str(best_path)
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
            "best_epoch": best_epoch,
            "best_ckpt_path": best_ckpt_path,
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
        out_path = Path(self.config.attn_out) if self.config.attn_out else (outdir / "leaderboard_attn.csv")
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
