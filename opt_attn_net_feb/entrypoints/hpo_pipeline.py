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
    FinalExplainabilityConfig,
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
from ..utils.progress import log_event, log_step


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
    feat3d_raw: str | None
    feat3d_qm_raw: str | None
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
    cpu_workers: int
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
        hpo_only: If True, run only HPO and skip final training/evaluation.
        best_params_json: Optional path to JSON file with best params in the
            same format as `<study_name>_best_params.json`.
        pruner_warmup_steps: Number of validation-report steps ignored before
            Optuna pruning activates.
    """
    run_hpo: bool
    hpo_only: bool
    best_params_json: str | None
    pruner_warmup_steps: int


@dataclass(frozen=True)
class CLIExplainabilityConfig:
    """CLI controls for Chem-ACE and Lambda-Vol integration."""

    run_chem_ace: bool
    run_lambda_vol: bool
    run_concept_rl: bool
    run_concept_rl_ablation: bool
    curated_smiles_col: str
    chem_ace_conformer_sdf: str | None
    chem_ace_sdf_conf_id_prop: str
    chem_ace_output_dir: str | None
    chem_ace_db_uri: str | None
    chem_ace_max_ids: int
    chem_ace_max_confs_per_id: int
    chem_ace_local_radii: tuple[int, ...]
    chem_ace_patch_cap_per_mol: int
    chem_ace_target_total_patches: int
    chem_ace_embed_dim_2d: int
    chem_ace_embed_dim_3d_geom: int
    chem_ace_embed_dim_3d_qm: int
    chem_ace_context_dim: int
    chem_ace_context_alpha: float
    chem_ace_qm_gating: bool
    chem_ace_max_2d_dim: int
    chem_ace_max_3dqm_dim: int
    chem_ace_persist_patch_embeddings: bool
    chem_ace_top_concepts: int
    chem_ace_infer_max_distance: float
    run_activity_calibration: bool
    activity_calibration_min_concept_support: int
    activity_calibration_min_tag_support: int
    activity_calibration_prior_strength: float
    activity_calibration_min_w: float
    activity_calibration_task_weight: float
    activity_calibration_bitmask_weight: float
    activity_calibration_bitmask_min_count: int
    activity_calibration_bitmask_exclude_zero: bool
    activity_calibration_mix_base: float
    activity_calibration_keep_threshold: float
    activity_calibration_min_confidence: float
    activity_calibration_max_confidence: float
    activity_calibration_ratio_cap: float
    activity_calibration_fallback_top1_if_empty: bool
    chem_ace_use_advanced_geom_topology: bool
    chem_ace_advanced_geom_topology_max_patches: int
    chem_ace_advanced_geom_topology_min_atoms: int
    chem_ace_advanced_geom_topology_max_torsion_paths: int
    chem_ace_advanced_geom_use_convex_hull: bool
    chem_ace_advanced_geom_use_persistent_homology: bool
    chem_ace_advanced_geom_persistence_max_atoms: int
    chem_ace_use_orca_descriptors: bool
    chem_ace_orca_descriptors_path: str | None
    chem_ace_orca_conf_id_col: str
    chem_ace_orca_mol_id_col: str
    chem_ace_orca_descriptor_cols: tuple[str, ...]
    chem_ace_orca_min_vectors_for_tagging: int
    chem_ace_orca_z_threshold: float
    chem_ace_use_pmapper_signatures: bool
    chem_ace_pmapper_tol: int
    chem_ace_pmapper_tol_alt: int
    chem_ace_strict_rerank: bool
    chem_ace_strict_rerank_layer_name: str
    chem_ace_strict_rerank_top_rows_per_task: int
    chem_ace_strict_rerank_batch_size: int
    chem_ace_strict_rerank_weight: float
    lambda_vol_output_dir: str | None
    lambda_vol_db_uri: str | None
    lambda_vol_layer_name: str
    lambda_vol_top_concepts: int
    lambda_vol_monitor_max_samples: int
    lambda_vol_tcav_repeats: int
    lambda_vol_random_counterexamples: int
    lambda_vol_min_concept_samples: int
    lambda_vol_tcav_holdout_fraction: float
    lambda_vol_tcav_holdout_min_samples: int
    lambda_vol_tcav_significance_alpha: float
    lambda_vol_tcav_bonferroni_m: int
    lambda_vol_run_ricci: bool
    lambda_vol_ricci_edge_keep_quantile: float
    lambda_vol_ricci_min_edge_weight: float
    lambda_vol_ricci_top_k_per_node: int
    lambda_vol_ricci_flow_steps: int
    lambda_vol_ricci_flow_step_size: float
    lambda_vol_ricci_use_flow_as_coupling: bool
    lambda_vol_ricci_coupling_strength: float
    concept_rl_top_k_per_task: int
    concept_rl_min_pos_coverage: float
    concept_rl_min_pos_hits: int
    concept_rl_min_lift: float
    concept_rl_overlap_weight: float
    concept_rl_global_weight: float
    concept_rl_lift_weight: float
    concept_rl_general_top_k: int
    concept_rl_min_multi_active_count: int
    concept_rl_require_conf_support: bool
    concept_rl_min_conf_pos_coverage: float
    concept_rl_init_scale: float
    concept_rl_max_scale: float
    concept_rl_policy_lr: float
    concept_rl_policy_sigma: float
    concept_rl_reward_alignment_w: float
    concept_rl_reward_min_ap_w: float
    concept_rl_baseline_momentum: float
    concept_rl_negative_penalty: float


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
    explainability: CLIExplainabilityConfig


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
    cpu_workers: int
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
                feat3d_raw=(None if args.feat3d_raw is None else str(args.feat3d_raw)),
                feat3d_qm_raw=(None if args.feat3d_qm_raw is None else str(args.feat3d_qm_raw)),
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
                cpu_workers=int(args.cpu_workers),
                pin_memory=bool(args.pin_memory),
            ),
            export=CLIExportConfig(
                attn_out=args.attn_out,
            ),
            hpo=CLIHPOControlConfig(
                run_hpo=bool(args.run_hpo),
                hpo_only=bool(args.hpo_only),
                best_params_json=(
                    None if args.best_params_json is None else str(args.best_params_json)
                ),
                pruner_warmup_steps=int(args.pruner_warmup_steps),
            ),
            explainability=CLIExplainabilityConfig(
                run_chem_ace=bool(args.run_chem_ace),
                run_lambda_vol=bool(args.run_lambda_vol),
                run_concept_rl=bool(args.run_concept_rl),
                run_concept_rl_ablation=bool(args.run_concept_rl_ablation),
                curated_smiles_col=str(args.curated_smiles_col),
                chem_ace_conformer_sdf=(
                    None if args.chem_ace_conformer_sdf is None else str(args.chem_ace_conformer_sdf)
                ),
                chem_ace_sdf_conf_id_prop=str(args.chem_ace_sdf_conf_id_prop),
                chem_ace_output_dir=(
                    None if args.chem_ace_output_dir is None else str(args.chem_ace_output_dir)
                ),
                chem_ace_db_uri=(
                    None if args.chem_ace_db_uri is None else str(args.chem_ace_db_uri)
                ),
                chem_ace_max_ids=int(args.chem_ace_max_ids),
                chem_ace_max_confs_per_id=int(args.chem_ace_max_confs_per_id),
                chem_ace_local_radii=tuple(int(x) for x in args.chem_ace_local_radii),
                chem_ace_patch_cap_per_mol=int(args.chem_ace_patch_cap_per_mol),
                chem_ace_target_total_patches=int(args.chem_ace_target_total_patches),
                chem_ace_embed_dim_2d=int(args.chem_ace_embed_dim_2d),
                chem_ace_embed_dim_3d_geom=int(args.chem_ace_embed_dim_3d_geom),
                chem_ace_embed_dim_3d_qm=int(args.chem_ace_embed_dim_3d_qm),
                chem_ace_context_dim=int(args.chem_ace_context_dim),
                chem_ace_context_alpha=float(args.chem_ace_context_alpha),
                chem_ace_qm_gating=bool(args.chem_ace_qm_gating),
                chem_ace_max_2d_dim=int(args.chem_ace_max_2d_dim),
                chem_ace_max_3dqm_dim=int(args.chem_ace_max_3dqm_dim),
                chem_ace_persist_patch_embeddings=bool(args.chem_ace_persist_patch_embeddings),
                chem_ace_top_concepts=int(args.chem_ace_top_concepts),
                chem_ace_infer_max_distance=float(args.chem_ace_infer_max_distance),
                run_activity_calibration=bool(args.run_activity_calibration),
                activity_calibration_min_concept_support=int(args.activity_calibration_min_concept_support),
                activity_calibration_min_tag_support=int(args.activity_calibration_min_tag_support),
                activity_calibration_prior_strength=float(args.activity_calibration_prior_strength),
                activity_calibration_min_w=float(args.activity_calibration_min_w),
                activity_calibration_task_weight=float(args.activity_calibration_task_weight),
                activity_calibration_bitmask_weight=float(args.activity_calibration_bitmask_weight),
                activity_calibration_bitmask_min_count=int(args.activity_calibration_bitmask_min_count),
                activity_calibration_bitmask_exclude_zero=bool(args.activity_calibration_bitmask_exclude_zero),
                activity_calibration_mix_base=float(args.activity_calibration_mix_base),
                activity_calibration_keep_threshold=float(args.activity_calibration_keep_threshold),
                activity_calibration_min_confidence=float(args.activity_calibration_min_confidence),
                activity_calibration_max_confidence=float(args.activity_calibration_max_confidence),
                activity_calibration_ratio_cap=float(args.activity_calibration_ratio_cap),
                activity_calibration_fallback_top1_if_empty=bool(
                    args.activity_calibration_fallback_top1_if_empty
                ),
                chem_ace_use_advanced_geom_topology=bool(args.chem_ace_use_advanced_geom_topology),
                chem_ace_advanced_geom_topology_max_patches=int(
                    args.chem_ace_advanced_geom_topology_max_patches
                ),
                chem_ace_advanced_geom_topology_min_atoms=int(
                    args.chem_ace_advanced_geom_topology_min_atoms
                ),
                chem_ace_advanced_geom_topology_max_torsion_paths=int(
                    args.chem_ace_advanced_geom_topology_max_torsion_paths
                ),
                chem_ace_advanced_geom_use_convex_hull=bool(
                    args.chem_ace_advanced_geom_use_convex_hull
                ),
                chem_ace_advanced_geom_use_persistent_homology=bool(
                    args.chem_ace_advanced_geom_use_persistent_homology
                ),
                chem_ace_advanced_geom_persistence_max_atoms=int(
                    args.chem_ace_advanced_geom_persistence_max_atoms
                ),
                chem_ace_use_orca_descriptors=bool(args.chem_ace_use_orca_descriptors),
                chem_ace_orca_descriptors_path=(
                    None
                    if args.chem_ace_orca_descriptors_path is None
                    else str(args.chem_ace_orca_descriptors_path)
                ),
                chem_ace_orca_conf_id_col=str(args.chem_ace_orca_conf_id_col),
                chem_ace_orca_mol_id_col=str(args.chem_ace_orca_mol_id_col),
                chem_ace_orca_descriptor_cols=tuple(str(x) for x in args.chem_ace_orca_descriptor_cols),
                chem_ace_orca_min_vectors_for_tagging=int(args.chem_ace_orca_min_vectors_for_tagging),
                chem_ace_orca_z_threshold=float(args.chem_ace_orca_z_threshold),
                chem_ace_use_pmapper_signatures=bool(args.chem_ace_use_pmapper_signatures),
                chem_ace_pmapper_tol=int(args.chem_ace_pmapper_tol),
                chem_ace_pmapper_tol_alt=int(args.chem_ace_pmapper_tol_alt),
                chem_ace_strict_rerank=bool(args.chem_ace_strict_rerank),
                chem_ace_strict_rerank_layer_name=str(args.chem_ace_strict_rerank_layer_name),
                chem_ace_strict_rerank_top_rows_per_task=int(args.chem_ace_strict_rerank_top_rows_per_task),
                chem_ace_strict_rerank_batch_size=int(args.chem_ace_strict_rerank_batch_size),
                chem_ace_strict_rerank_weight=float(args.chem_ace_strict_rerank_weight),
                lambda_vol_output_dir=(
                    None if args.lambda_vol_output_dir is None else str(args.lambda_vol_output_dir)
                ),
                lambda_vol_db_uri=(
                    None if args.lambda_vol_db_uri is None else str(args.lambda_vol_db_uri)
                ),
                lambda_vol_layer_name=str(args.lambda_vol_layer_name),
                lambda_vol_top_concepts=int(args.lambda_vol_top_concepts),
                lambda_vol_monitor_max_samples=int(args.lambda_vol_monitor_max_samples),
                lambda_vol_tcav_repeats=int(args.lambda_vol_tcav_repeats),
                lambda_vol_random_counterexamples=int(args.lambda_vol_random_counterexamples),
                lambda_vol_min_concept_samples=int(args.lambda_vol_min_concept_samples),
                lambda_vol_tcav_holdout_fraction=float(args.lambda_vol_tcav_holdout_fraction),
                lambda_vol_tcav_holdout_min_samples=int(args.lambda_vol_tcav_holdout_min_samples),
                lambda_vol_tcav_significance_alpha=float(args.lambda_vol_tcav_significance_alpha),
                lambda_vol_tcav_bonferroni_m=int(args.lambda_vol_tcav_bonferroni_m),
                lambda_vol_run_ricci=bool(args.lambda_vol_run_ricci),
                lambda_vol_ricci_edge_keep_quantile=float(args.lambda_vol_ricci_edge_keep_quantile),
                lambda_vol_ricci_min_edge_weight=float(args.lambda_vol_ricci_min_edge_weight),
                lambda_vol_ricci_top_k_per_node=int(args.lambda_vol_ricci_top_k_per_node),
                lambda_vol_ricci_flow_steps=int(args.lambda_vol_ricci_flow_steps),
                lambda_vol_ricci_flow_step_size=float(args.lambda_vol_ricci_flow_step_size),
                lambda_vol_ricci_use_flow_as_coupling=bool(args.lambda_vol_ricci_use_flow_as_coupling),
                lambda_vol_ricci_coupling_strength=float(args.lambda_vol_ricci_coupling_strength),
                concept_rl_top_k_per_task=int(args.concept_rl_top_k_per_task),
                concept_rl_min_pos_coverage=float(args.concept_rl_min_pos_coverage),
                concept_rl_min_pos_hits=int(args.concept_rl_min_pos_hits),
                concept_rl_min_lift=float(args.concept_rl_min_lift),
                concept_rl_overlap_weight=float(args.concept_rl_overlap_weight),
                concept_rl_global_weight=float(args.concept_rl_global_weight),
                concept_rl_lift_weight=float(args.concept_rl_lift_weight),
                concept_rl_general_top_k=int(args.concept_rl_general_top_k),
                concept_rl_min_multi_active_count=int(args.concept_rl_min_multi_active_count),
                concept_rl_require_conf_support=bool(args.concept_rl_require_conf_support),
                concept_rl_min_conf_pos_coverage=float(args.concept_rl_min_conf_pos_coverage),
                concept_rl_init_scale=float(args.concept_rl_init_scale),
                concept_rl_max_scale=float(args.concept_rl_max_scale),
                concept_rl_policy_lr=float(args.concept_rl_policy_lr),
                concept_rl_policy_sigma=float(args.concept_rl_policy_sigma),
                concept_rl_reward_alignment_w=float(args.concept_rl_reward_alignment_w),
                concept_rl_reward_min_ap_w=float(args.concept_rl_reward_min_ap_w),
                concept_rl_baseline_momentum=float(args.concept_rl_baseline_momentum),
                concept_rl_negative_penalty=float(args.concept_rl_negative_penalty),
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
        with log_step(
            "pipeline.prepare_environment",
            seed=int(self.config.runtime.seed),
            study_dir=str(self.config.data_paths.study_dir),
        ):
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
                "num_workers": int(self.config.runtime.num_workers),
                "cpu_workers": int(self.config.runtime.cpu_workers),
                "run_hpo": bool(self.config.hpo.run_hpo),
                "hpo_only": bool(self.config.hpo.hpo_only),
                "best_params_json": self.config.hpo.best_params_json,
                "pruner_warmup_steps": int(self.config.hpo.pruner_warmup_steps),
                "feat3d_raw": self.config.data_paths.feat3d_raw,
                "feat3d_qm_raw": self.config.data_paths.feat3d_qm_raw,
                "run_chem_ace": bool(self.config.explainability.run_chem_ace),
                "run_lambda_vol": bool(self.config.explainability.run_lambda_vol),
                "run_concept_rl": bool(self.config.explainability.run_concept_rl),
                "run_concept_rl_ablation": bool(self.config.explainability.run_concept_rl_ablation),
                "lambda_vol_run_ricci": bool(self.config.explainability.lambda_vol_run_ricci),
                "curated_smiles_col": str(self.config.explainability.curated_smiles_col),
                "argv": " ".join([str(x) for x in (argv if argv is not None else os.sys.argv)]),
                "weight_cols": WEIGHT_COLS,
                "model": "MILTaskAttnMixerWithAux (task-specific attention queries)",
            }
            run_meta_path = outdir / "run_meta.json"
            run_meta_path.write_text(json.dumps(run_meta, indent=2))
            log_event("INFO", "pipeline.run_meta_written", path=str(run_meta_path))

            num_workers = self._resolve_num_workers()
            cpu_workers = self._resolve_cpu_workers()
            if cpu_workers > 0:
                try:
                    torch.set_num_threads(int(cpu_workers))
                except Exception:
                    pass
                try:
                    torch.set_num_interop_threads(int(max(1, min(8, cpu_workers // 2))))
                except Exception:
                    pass
            pin_memory = bool(self.config.runtime.pin_memory) and torch.cuda.is_available()
            log_event(
                "INFO",
                "pipeline.dataloader_runtime",
                num_workers=int(num_workers),
                cpu_workers=int(cpu_workers),
                pin_memory=bool(pin_memory),
                precision=str(self.config.runtime.precision),
            )
            return PipelineEnvironment(
                outdir=outdir,
                num_workers=num_workers,
                cpu_workers=cpu_workers,
                pin_memory=pin_memory,
            )

    def _resolve_num_workers(self) -> int:
        if int(self.config.runtime.num_workers) >= 0:
            return int(self.config.runtime.num_workers)
        cpus = int(os.environ.get("SLURM_CPUS_PER_TASK") or 0)
        if cpus <= 0:
            cpus = os.cpu_count() or 0
        return max(2, min(23, cpus - 2)) if cpus >= 4 else 0

    def _resolve_cpu_workers(self) -> int:
        if int(self.config.runtime.cpu_workers) >= 0:
            return int(self.config.runtime.cpu_workers)
        cpus = int(os.environ.get("SLURM_CPUS_PER_TASK") or 0)
        if cpus <= 0:
            cpus = os.cpu_count() or 0
        if cpus <= 0:
            return 1
        # Reserve some CPU for dataloader workers and system activity.
        if int(self.config.runtime.num_workers) >= 0:
            reserve = int(self.config.runtime.num_workers)
        else:
            reserve = max(2, min(23, cpus - 2)) if cpus >= 4 else 0
        return max(1, int(cpus - reserve - 1))


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

        with log_step("hpo_data.build", labels=str(p.labels), use_splits=list(s.use_splits)):
            with log_step("hpo_data.load_labels"):
                df_full = load_labels(p.labels, id_col=c.id_col)
                df_full[c.id_col] = df_full[c.id_col].astype(str)
                df_full[c.split_col] = df_full[c.split_col].astype(str)
                log_event("INFO", "hpo_data.labels_loaded", n_rows=int(len(df_full)))

            with log_step("hpo_data.filter_splits"):
                df_hpo = df_full[df_full[c.split_col].isin(s.use_splits)].copy().reset_index(drop=True)
                if len(df_hpo) == 0:
                    raise ValueError(f"No rows in labels match use_splits={s.use_splits}")
                log_event("INFO", "hpo_data.hpo_rows", n_rows=int(len(df_hpo)))

            with log_step("hpo_data.load_and_align_2d"):
                ids_hpo = df_hpo[c.id_col].astype(str).tolist()
                ids_2d_file, X2d_file = load_2d(p.feat2d_scaled, id_col=c.id_col)
                X2d_hpo = align_by_id(ids_2d_file, X2d_file, ids_hpo)
                log_event("INFO", "hpo_data.2d_ready", n_ids=int(len(ids_hpo)), dim_2d=int(X2d_hpo.shape[1]))

            with log_step("hpo_data.targets_and_weights"):
                y_cls = coerce_binary_labels(df_hpo)
                w_cls = build_task_weights(df_hpo)
                y_abs, m_abs, y_fluo, m_fluo = build_aux_targets_and_masks(df_hpo)
                w_abs, w_fluo = build_aux_weights(df_hpo)

            with log_step("hpo_data.resolve_folds"):
                folds = self._resolve_folds(df_hpo)
                folds_info = fold_indices(df_hpo, c.fold_col, folds)
                log_event("INFO", "hpo_data.folds", folds=list(map(int, folds)), n_folds=int(len(folds_info)))

            with log_step("hpo_data.load_and_merge_instances"):
                ids_conf_hpo, conf_ids_hpo, Xinst_hpo, inst_meta_hpo = load_and_merge_instances(
                    p.feat3d_scaled,
                    p.feat3d_qm_scaled,
                    allowed_ids=set(ids_hpo),
                    id_col=c.id_col,
                    conf_col=c.conf_col,
                    return_meta=True,
                )
                _, starts_hpo, counts_hpo, id2pos_hpo, Xinst_sorted_hpo, _ = build_instance_index(
                    ids_conf_hpo,
                    conf_ids_hpo,
                    Xinst_hpo,
                )
                log_event(
                    "INFO",
                    "hpo_data.instances_ready",
                    n_conf=int(Xinst_sorted_hpo.shape[0]),
                    inst_dim=int(Xinst_sorted_hpo.shape[1]),
                    inst_geom_dim=int(inst_meta_hpo["geom_dim"]),
                    inst_qm_dim=int(inst_meta_hpo["qm_dim"]),
                    n_qm_cols=int(len(inst_meta_hpo.get("qm_cols", ()))),
                    n_ids_with_bags=int(len(id2pos_hpo)),
                )

            with log_step("hpo_data.drop_ids_without_bags_if_needed"):
                have_bag_mask = np.array([(i in id2pos_hpo) for i in ids_hpo], dtype=bool)
                if not have_bag_mask.all():
                    missing = int((~have_bag_mask).sum())
                    examples = [ids_hpo[i] for i in np.where(~have_bag_mask)[0][:10]]
                    log_event(
                        "WARN",
                        "hpo_data.dropping_ids_without_bags",
                        missing=missing,
                        examples=examples,
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

            with log_step("hpo_data.build_cv_container"):
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
                    inst_geom_dim=int(inst_meta_hpo["geom_dim"]),
                    inst_qm_dim=int(inst_meta_hpo["qm_dim"]),
                )
                log_event(
                    "INFO",
                    "hpo_data.summary",
                    n_ids=int(len(ids_hpo)),
                    dim_2d=int(X2d_hpo.shape[1]),
                    n_conf=int(Xinst_sorted_hpo.shape[0]),
                    inst_dim=int(Xinst_sorted_hpo.shape[1]),
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
        with log_step(
            "pipeline.run",
            run_hpo=bool(self.config.hpo.run_hpo),
            study_dir=str(self.config.data_paths.study_dir),
        ):
            with log_step("pipeline.prepare_environment"):
                env = PipelineEnvironmentFactory(self.config).prepare(argv=self.argv)
            with log_step("pipeline.build_hpo_data"):
                hpo_data = HPODataBuilder(self.config).build()

            log_event(
                "INFO",
                "pipeline.dataloader",
                num_workers=int(env.num_workers),
                cpu_workers=int(env.cpu_workers),
                pin_memory=bool(env.pin_memory),
                precision=str(self.config.runtime.precision),
            )
            ckpt_root: Path | None = None

            if bool(self.config.hpo.run_hpo):
                with log_step("pipeline.run_hpo"):
                    ckpt_root = env.outdir / "_tmp_best_ckpts"
                    ckpt_root.mkdir(parents=True, exist_ok=True)
                    study = self._run_hpo(env=env, hpo_data=hpo_data, ckpt_root=ckpt_root)
                    best_params = dict(study.best_params)
            else:
                with log_step("pipeline.load_best_params"):
                    best_params = self._load_best_params(outdir=env.outdir)

            if bool(self.config.hpo.hpo_only):
                log_event(
                    "INFO",
                    "pipeline.hpo_only.completed",
                    run_hpo=bool(self.config.hpo.run_hpo),
                    n_best_params=int(len(best_params)),
                )
            else:
                with log_step("pipeline.run_final"):
                    self._run_final(env=env, hpo_data=hpo_data, best_params=best_params)

            with log_step("pipeline.cleanup"):
                if ckpt_root is not None:
                    try:
                        shutil.rmtree(ckpt_root, ignore_errors=True)
                    except Exception:
                        pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

    def _run_hpo(self, *, env: PipelineEnvironment, hpo_data: PreparedHPOData, ckpt_root: Path):
        with log_step(
            "pipeline._run_hpo",
            n_trials=int(self.config.runtime.trials),
            max_epochs=int(self.config.runtime.max_epochs),
        ):
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
                    pruner_warmup_steps=int(self.config.hpo.pruner_warmup_steps),
                ),
                cross_validator=cross_validator,
            )
            return study_runner.run()

    def _load_best_params(self, *, outdir: Path) -> dict[str, Any]:
        with log_step("pipeline._load_best_params"):
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
            log_event("INFO", "pipeline.best_params_loaded", path=str(path), n_keys=int(len(payload)))
            return dict(payload)

    def _run_final(self, *, env: PipelineEnvironment, hpo_data: PreparedHPOData, best_params: dict[str, Any]):
        c = self.config.columns
        p = self.config.data_paths
        split = self.config.splits
        df_full = hpo_data.df_full

        with log_step("pipeline._run_final", leaderboard_split=str(split.leaderboard_split)):
            with log_step("pipeline.final.load_instances"):
                allowed_final = set(
                    df_full[df_full[c.split_col].isin(["train", split.leaderboard_split])][c.id_col]
                    .astype(str)
                    .tolist()
                )
                ids_conf_all, conf_ids_all, Xinst_all, inst_meta_all = load_and_merge_instances(
                    p.feat3d_scaled,
                    p.feat3d_qm_scaled,
                    allowed_ids=allowed_final,
                    id_col=c.id_col,
                    conf_col=c.conf_col,
                    return_meta=True,
                )
                _, starts_all, counts_all, id2pos_all, Xinst_sorted_all, conf_sorted_all = build_instance_index(
                    ids_conf_all,
                    conf_ids_all,
                    Xinst_all,
                )
                log_event(
                    "INFO",
                    "pipeline.final.instances_ready",
                    n_ids_with_bags=int(len(id2pos_all)),
                    n_conf=int(Xinst_sorted_all.shape[0]),
                    inst_dim=int(Xinst_sorted_all.shape[1]),
                    inst_geom_dim=int(inst_meta_all["geom_dim"]),
                    inst_qm_dim=int(inst_meta_all["qm_dim"]),
                    n_geom_cols=int(len(inst_meta_all.get("geom_cols", ()))),
                    n_qm_cols=int(len(inst_meta_all.get("qm_cols", ()))),
                )

                Xinst_sorted_raw_all = None
                conf_sorted_raw_all = None
                starts_raw_all = None
                counts_raw_all = None
                id2pos_raw_all = None
                inst_meta_raw_all = None
                raw_geom = self.config.data_paths.feat3d_raw
                raw_qm = self.config.data_paths.feat3d_qm_raw
                if (raw_geom is not None) and (raw_qm is not None):
                    with log_step("pipeline.final.load_instances_raw_for_semantics"):
                        ids_conf_raw, conf_ids_raw, Xinst_raw, inst_meta_raw = load_and_merge_instances(
                            str(raw_geom),
                            str(raw_qm),
                            allowed_ids=allowed_final,
                            id_col=c.id_col,
                            conf_col=c.conf_col,
                            return_meta=True,
                        )
                        _, starts_raw, counts_raw, id2pos_raw, Xinst_sorted_raw, conf_sorted_raw = build_instance_index(
                            ids_conf_raw,
                            conf_ids_raw,
                            Xinst_raw,
                        )
                        Xinst_sorted_raw_all = Xinst_sorted_raw
                        conf_sorted_raw_all = conf_sorted_raw
                        starts_raw_all = starts_raw
                        counts_raw_all = counts_raw
                        id2pos_raw_all = id2pos_raw
                        inst_meta_raw_all = inst_meta_raw
                        log_event(
                            "INFO",
                            "pipeline.final.instances_raw_ready",
                            n_ids_with_bags=int(len(id2pos_raw)),
                            n_conf=int(Xinst_sorted_raw.shape[0]),
                            inst_dim=int(Xinst_sorted_raw.shape[1]),
                            inst_geom_dim=int(inst_meta_raw["geom_dim"]),
                            inst_qm_dim=int(inst_meta_raw["qm_dim"]),
                            n_geom_cols=int(len(inst_meta_raw.get("geom_cols", ()))),
                            n_qm_cols=int(len(inst_meta_raw.get("qm_cols", ()))),
                        )
                elif (raw_geom is None) ^ (raw_qm is None):
                    log_event(
                        "WARN",
                        "pipeline.final.instances_raw_semantics_skipped",
                        reason="provide_both_raw_tables",
                        feat3d_raw=str(raw_geom),
                        feat3d_qm_raw=str(raw_qm),
                    )
                else:
                    log_event(
                        "INFO",
                        "pipeline.final.instances_raw_semantics_skipped",
                        reason="raw_tables_not_provided",
                    )

            with log_step("pipeline.final.build_config"):
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
                    inst_geom_dim=int(inst_meta_all["geom_dim"]),
                    inst_qm_dim=int(inst_meta_all["qm_dim"]),
                    inst_geom_cols=tuple(str(x) for x in inst_meta_all.get("geom_cols", ())),
                    inst_qm_cols=tuple(str(x) for x in inst_meta_all.get("qm_cols", ())),
                    starts_raw=starts_raw_all,
                    counts_raw=counts_raw_all,
                    id2pos_raw=id2pos_raw_all,
                    Xinst_sorted_raw=Xinst_sorted_raw_all,
                    conf_sorted_raw=conf_sorted_raw_all,
                    inst_geom_dim_raw=(
                        -1 if inst_meta_raw_all is None else int(inst_meta_raw_all["geom_dim"])
                    ),
                    inst_qm_dim_raw=(
                        -1 if inst_meta_raw_all is None else int(inst_meta_raw_all["qm_dim"])
                    ),
                    inst_geom_cols_raw=(
                        tuple()
                        if inst_meta_raw_all is None
                        else tuple(str(x) for x in inst_meta_raw_all.get("geom_cols", ()))
                    ),
                    inst_qm_cols_raw=(
                        tuple()
                        if inst_meta_raw_all is None
                        else tuple(str(x) for x in inst_meta_raw_all.get("qm_cols", ()))
                    ),
                )
                trainer_cfg = TrainerSystemConfig(
                    max_epochs=int(self.config.runtime.max_epochs),
                    patience=int(self.config.runtime.patience),
                    accelerator=str(self.config.runtime.nn_accelerator),
                    devices=int(self.config.runtime.nn_devices),
                    precision=str(self.config.runtime.precision),
                )
                loader_cfg = LoaderConfig(
                    num_workers=int(env.num_workers),
                    pin_memory=bool(env.pin_memory),
                )

                def _make_explainability_cfg(
                    *,
                    run_concept_rl: bool,
                    run_label: str | None,
                ) -> FinalExplainabilityConfig:
                    return FinalExplainabilityConfig(
                        run_chem_ace=bool(self.config.explainability.run_chem_ace),
                        run_lambda_vol=bool(self.config.explainability.run_lambda_vol),
                        curated_smiles_col=str(self.config.explainability.curated_smiles_col),
                        chem_ace_conformer_sdf=self.config.explainability.chem_ace_conformer_sdf,
                        chem_ace_sdf_conf_id_prop=str(self.config.explainability.chem_ace_sdf_conf_id_prop),
                        cpu_workers=int(env.cpu_workers),
                        chem_ace_output_dir=self._suffix_output_dir(
                            base_dir=self.config.explainability.chem_ace_output_dir,
                            suffix=run_label,
                        ),
                        chem_ace_db_uri=self._suffix_sqlite_uri(
                            base_uri=self.config.explainability.chem_ace_db_uri,
                            suffix=run_label,
                        ),
                        chem_ace_max_ids=int(self.config.explainability.chem_ace_max_ids),
                        chem_ace_max_confs_per_id=int(self.config.explainability.chem_ace_max_confs_per_id),
                        chem_ace_local_radii=tuple(self.config.explainability.chem_ace_local_radii),
                        chem_ace_patch_cap_per_mol=int(self.config.explainability.chem_ace_patch_cap_per_mol),
                        chem_ace_target_total_patches=int(self.config.explainability.chem_ace_target_total_patches),
                        chem_ace_embed_dim_2d=int(self.config.explainability.chem_ace_embed_dim_2d),
                        chem_ace_embed_dim_3d_geom=int(self.config.explainability.chem_ace_embed_dim_3d_geom),
                        chem_ace_embed_dim_3d_qm=int(self.config.explainability.chem_ace_embed_dim_3d_qm),
                        chem_ace_context_dim=int(self.config.explainability.chem_ace_context_dim),
                        chem_ace_context_alpha=float(self.config.explainability.chem_ace_context_alpha),
                        chem_ace_qm_gating=bool(self.config.explainability.chem_ace_qm_gating),
                        chem_ace_max_2d_dim=int(self.config.explainability.chem_ace_max_2d_dim),
                        chem_ace_max_3dqm_dim=int(self.config.explainability.chem_ace_max_3dqm_dim),
                        chem_ace_persist_patch_embeddings=bool(
                            self.config.explainability.chem_ace_persist_patch_embeddings
                        ),
                        chem_ace_top_concepts=int(self.config.explainability.chem_ace_top_concepts),
                        chem_ace_infer_max_distance=float(self.config.explainability.chem_ace_infer_max_distance),
                        run_activity_calibration=bool(self.config.explainability.run_activity_calibration),
                        activity_calibration_min_concept_support=int(
                            self.config.explainability.activity_calibration_min_concept_support
                        ),
                        activity_calibration_min_tag_support=int(
                            self.config.explainability.activity_calibration_min_tag_support
                        ),
                        activity_calibration_prior_strength=float(
                            self.config.explainability.activity_calibration_prior_strength
                        ),
                        activity_calibration_min_w=float(
                            self.config.explainability.activity_calibration_min_w
                        ),
                        activity_calibration_task_weight=float(
                            self.config.explainability.activity_calibration_task_weight
                        ),
                        activity_calibration_bitmask_weight=float(
                            self.config.explainability.activity_calibration_bitmask_weight
                        ),
                        activity_calibration_bitmask_min_count=int(
                            self.config.explainability.activity_calibration_bitmask_min_count
                        ),
                        activity_calibration_bitmask_exclude_zero=bool(
                            self.config.explainability.activity_calibration_bitmask_exclude_zero
                        ),
                        activity_calibration_mix_base=float(
                            self.config.explainability.activity_calibration_mix_base
                        ),
                        activity_calibration_keep_threshold=float(
                            self.config.explainability.activity_calibration_keep_threshold
                        ),
                        activity_calibration_min_confidence=float(
                            self.config.explainability.activity_calibration_min_confidence
                        ),
                        activity_calibration_max_confidence=float(
                            self.config.explainability.activity_calibration_max_confidence
                        ),
                        activity_calibration_ratio_cap=float(
                            self.config.explainability.activity_calibration_ratio_cap
                        ),
                        activity_calibration_fallback_top1_if_empty=bool(
                            self.config.explainability.activity_calibration_fallback_top1_if_empty
                        ),
                        chem_ace_use_advanced_geom_topology=bool(
                            self.config.explainability.chem_ace_use_advanced_geom_topology
                        ),
                        chem_ace_advanced_geom_topology_max_patches=int(
                            self.config.explainability.chem_ace_advanced_geom_topology_max_patches
                        ),
                        chem_ace_advanced_geom_topology_min_atoms=int(
                            self.config.explainability.chem_ace_advanced_geom_topology_min_atoms
                        ),
                        chem_ace_advanced_geom_topology_max_torsion_paths=int(
                            self.config.explainability.chem_ace_advanced_geom_topology_max_torsion_paths
                        ),
                        chem_ace_advanced_geom_use_convex_hull=bool(
                            self.config.explainability.chem_ace_advanced_geom_use_convex_hull
                        ),
                        chem_ace_advanced_geom_use_persistent_homology=bool(
                            self.config.explainability.chem_ace_advanced_geom_use_persistent_homology
                        ),
                        chem_ace_advanced_geom_persistence_max_atoms=int(
                            self.config.explainability.chem_ace_advanced_geom_persistence_max_atoms
                        ),
                        chem_ace_use_orca_descriptors=bool(
                            self.config.explainability.chem_ace_use_orca_descriptors
                        ),
                        chem_ace_orca_descriptors_path=self.config.explainability.chem_ace_orca_descriptors_path,
                        chem_ace_orca_conf_id_col=str(
                            self.config.explainability.chem_ace_orca_conf_id_col
                        ),
                        chem_ace_orca_mol_id_col=str(
                            self.config.explainability.chem_ace_orca_mol_id_col
                        ),
                        chem_ace_orca_descriptor_cols=tuple(
                            self.config.explainability.chem_ace_orca_descriptor_cols
                        ),
                        chem_ace_orca_min_vectors_for_tagging=int(
                            self.config.explainability.chem_ace_orca_min_vectors_for_tagging
                        ),
                        chem_ace_orca_z_threshold=float(
                            self.config.explainability.chem_ace_orca_z_threshold
                        ),
                        chem_ace_use_pmapper_signatures=bool(
                            self.config.explainability.chem_ace_use_pmapper_signatures
                        ),
                        chem_ace_pmapper_tol=int(self.config.explainability.chem_ace_pmapper_tol),
                        chem_ace_pmapper_tol_alt=int(self.config.explainability.chem_ace_pmapper_tol_alt),
                        chem_ace_strict_rerank=bool(
                            self.config.explainability.chem_ace_strict_rerank
                        ),
                        chem_ace_strict_rerank_layer_name=str(
                            self.config.explainability.chem_ace_strict_rerank_layer_name
                        ),
                        chem_ace_strict_rerank_top_rows_per_task=int(
                            self.config.explainability.chem_ace_strict_rerank_top_rows_per_task
                        ),
                        chem_ace_strict_rerank_batch_size=int(
                            self.config.explainability.chem_ace_strict_rerank_batch_size
                        ),
                        chem_ace_strict_rerank_weight=float(
                            self.config.explainability.chem_ace_strict_rerank_weight
                        ),
                        lambda_vol_output_dir=self._suffix_output_dir(
                            base_dir=self.config.explainability.lambda_vol_output_dir,
                            suffix=run_label,
                        ),
                        lambda_vol_db_uri=self._suffix_sqlite_uri(
                            base_uri=self.config.explainability.lambda_vol_db_uri,
                            suffix=run_label,
                        ),
                        lambda_vol_layer_name=str(self.config.explainability.lambda_vol_layer_name),
                        lambda_vol_top_concepts=int(self.config.explainability.lambda_vol_top_concepts),
                        lambda_vol_monitor_max_samples=int(self.config.explainability.lambda_vol_monitor_max_samples),
                        lambda_vol_tcav_repeats=int(self.config.explainability.lambda_vol_tcav_repeats),
                        lambda_vol_random_counterexamples=int(self.config.explainability.lambda_vol_random_counterexamples),
                        lambda_vol_min_concept_samples=int(self.config.explainability.lambda_vol_min_concept_samples),
                        lambda_vol_tcav_holdout_fraction=float(
                            self.config.explainability.lambda_vol_tcav_holdout_fraction
                        ),
                        lambda_vol_tcav_holdout_min_samples=int(
                            self.config.explainability.lambda_vol_tcav_holdout_min_samples
                        ),
                        lambda_vol_tcav_significance_alpha=float(
                            self.config.explainability.lambda_vol_tcav_significance_alpha
                        ),
                        lambda_vol_tcav_bonferroni_m=int(
                            self.config.explainability.lambda_vol_tcav_bonferroni_m
                        ),
                        lambda_vol_run_ricci=bool(self.config.explainability.lambda_vol_run_ricci),
                        lambda_vol_ricci_edge_keep_quantile=float(
                            self.config.explainability.lambda_vol_ricci_edge_keep_quantile
                        ),
                        lambda_vol_ricci_min_edge_weight=float(
                            self.config.explainability.lambda_vol_ricci_min_edge_weight
                        ),
                        lambda_vol_ricci_top_k_per_node=int(
                            self.config.explainability.lambda_vol_ricci_top_k_per_node
                        ),
                        lambda_vol_ricci_flow_steps=int(
                            self.config.explainability.lambda_vol_ricci_flow_steps
                        ),
                        lambda_vol_ricci_flow_step_size=float(
                            self.config.explainability.lambda_vol_ricci_flow_step_size
                        ),
                        lambda_vol_ricci_use_flow_as_coupling=bool(
                            self.config.explainability.lambda_vol_ricci_use_flow_as_coupling
                        ),
                        lambda_vol_ricci_coupling_strength=float(
                            self.config.explainability.lambda_vol_ricci_coupling_strength
                        ),
                        run_concept_rl=bool(run_concept_rl),
                        concept_rl_top_k_per_task=int(self.config.explainability.concept_rl_top_k_per_task),
                        concept_rl_min_pos_coverage=float(
                            self.config.explainability.concept_rl_min_pos_coverage
                        ),
                        concept_rl_min_pos_hits=int(self.config.explainability.concept_rl_min_pos_hits),
                        concept_rl_min_lift=float(self.config.explainability.concept_rl_min_lift),
                        concept_rl_overlap_weight=float(self.config.explainability.concept_rl_overlap_weight),
                        concept_rl_global_weight=float(self.config.explainability.concept_rl_global_weight),
                        concept_rl_lift_weight=float(self.config.explainability.concept_rl_lift_weight),
                        concept_rl_general_top_k=int(self.config.explainability.concept_rl_general_top_k),
                        concept_rl_min_multi_active_count=int(
                            self.config.explainability.concept_rl_min_multi_active_count
                        ),
                        concept_rl_require_conf_support=bool(
                            self.config.explainability.concept_rl_require_conf_support
                        ),
                        concept_rl_min_conf_pos_coverage=float(
                            self.config.explainability.concept_rl_min_conf_pos_coverage
                        ),
                        concept_rl_init_scale=float(self.config.explainability.concept_rl_init_scale),
                        concept_rl_max_scale=float(self.config.explainability.concept_rl_max_scale),
                        concept_rl_policy_lr=float(self.config.explainability.concept_rl_policy_lr),
                        concept_rl_policy_sigma=float(self.config.explainability.concept_rl_policy_sigma),
                        concept_rl_reward_alignment_w=float(
                            self.config.explainability.concept_rl_reward_alignment_w
                        ),
                        concept_rl_reward_min_ap_w=float(
                            self.config.explainability.concept_rl_reward_min_ap_w
                        ),
                        concept_rl_baseline_momentum=float(
                            self.config.explainability.concept_rl_baseline_momentum
                        ),
                        concept_rl_negative_penalty=float(
                            self.config.explainability.concept_rl_negative_penalty
                        ),
                    )

                def _make_final_cfg(
                    *,
                    run_concept_rl: bool,
                    attn_out: str | None,
                    run_label: str | None,
                ) -> FinalTrainConfig:
                    return FinalTrainConfig(
                        seed=int(self.config.runtime.seed),
                        trainer=trainer_cfg,
                        loader=loader_cfg,
                        attn_out=attn_out,
                        explainability=_make_explainability_cfg(
                            run_concept_rl=run_concept_rl,
                            run_label=run_label,
                        ),
                    )

            if not bool(self.config.explainability.run_concept_rl_ablation):
                with log_step("pipeline.final.train_and_eval"):
                    final_cfg = _make_final_cfg(
                        run_concept_rl=bool(self.config.explainability.run_concept_rl),
                        attn_out=self.config.export.attn_out,
                        run_label=None,
                    )
                    MILFinalTrainer(config=final_cfg).run(
                        outdir=env.outdir,
                        best_params=dict(best_params),
                        data=final_data,
                    )
                return

            with log_step("pipeline.final.train_and_eval_ablation"):
                run_specs = (
                    ("no_rl", False),
                    ("with_rl", True),
                )
                ablation_root = env.outdir / "ablation"
                ablation_root.mkdir(parents=True, exist_ok=True)
                summaries: list[dict[str, Any]] = []
                for run_label, run_concept_rl in run_specs:
                    run_outdir = ablation_root / run_label
                    run_outdir.mkdir(parents=True, exist_ok=True)
                    run_attn_out = self._with_suffix_path(
                        base_path=self.config.export.attn_out,
                        suffix=run_label,
                    )
                    final_cfg = _make_final_cfg(
                        run_concept_rl=bool(run_concept_rl),
                        attn_out=run_attn_out,
                        run_label=str(run_label),
                    )
                    with log_step(
                        "pipeline.final.ablation_run",
                        run_label=str(run_label),
                        run_concept_rl=bool(run_concept_rl),
                    ):
                        MILFinalTrainer(config=final_cfg).run(
                            outdir=run_outdir,
                            best_params=dict(best_params),
                            data=final_data,
                        )
                    summaries.append(
                        self._collect_ablation_run_summary(
                            run_outdir=run_outdir,
                            run_label=str(run_label),
                            run_concept_rl=bool(run_concept_rl),
                        )
                    )

                self._write_ablation_report(
                    outdir=env.outdir,
                    summaries=summaries,
                )

    @staticmethod
    def _with_suffix_path(*, base_path: str | None, suffix: str) -> str | None:
        if base_path is None:
            return None
        path = Path(base_path)
        if path.suffix:
            return str(path.with_name(f"{path.stem}_{suffix}{path.suffix}"))
        return f"{str(path)}_{suffix}"

    @staticmethod
    def _suffix_output_dir(*, base_dir: str | None, suffix: str | None) -> str | None:
        if base_dir is None or suffix is None or len(str(suffix)) == 0:
            return base_dir
        return str(Path(base_dir) / str(suffix))

    @staticmethod
    def _suffix_sqlite_uri(*, base_uri: str | None, suffix: str | None) -> str | None:
        if base_uri is None or suffix is None or len(str(suffix)) == 0:
            return base_uri
        prefix = "sqlite:///"
        if not str(base_uri).startswith(prefix):
            return base_uri
        path = Path(str(base_uri)[len(prefix):])
        if path.suffix:
            out = path.with_name(f"{path.stem}_{suffix}{path.suffix}")
        else:
            out = path.with_name(f"{path.name}_{suffix}")
        return f"{prefix}{out.as_posix()}"

    @staticmethod
    def _summarize_rl_policy(policy_path: Path) -> dict[str, Any]:
        payload = json.loads(policy_path.read_text())
        history = payload.get("history", [])
        if not isinstance(history, list) or len(history) == 0:
            return {
                "policy_history_path": str(policy_path),
                "n_epochs": 0,
                "policy_mean_start": None,
                "policy_mean_end": None,
                "policy_mean_abs_step": None,
                "policy_mean_sign_flip_rate": None,
                "policy_mean_tail_std": None,
                "policy_converged": None,
                "policy_likely_oscillating": None,
            }

        mu = np.asarray(
            [float(x.get("policy_mean_after", np.nan)) for x in history],
            dtype=np.float64,
        )
        mu = mu[np.isfinite(mu)]
        if mu.size == 0:
            return {
                "policy_history_path": str(policy_path),
                "n_epochs": int(len(history)),
                "policy_mean_start": None,
                "policy_mean_end": None,
                "policy_mean_abs_step": None,
                "policy_mean_sign_flip_rate": None,
                "policy_mean_tail_std": None,
                "policy_converged": None,
                "policy_likely_oscillating": None,
            }

        diff = np.diff(mu)
        mean_abs_step = float(np.mean(np.abs(diff))) if diff.size > 0 else 0.0
        if diff.size > 1:
            sign_flip_count = int(np.sum((diff[1:] * diff[:-1]) < 0.0))
            sign_flip_rate = float(sign_flip_count / max(1, diff.size - 1))
        else:
            sign_flip_rate = 0.0
        tail = mu[-min(5, int(mu.size)) :]
        tail_std = float(np.std(tail)) if tail.size > 0 else 0.0
        likely_oscillating = bool(sign_flip_rate > 0.50 and mean_abs_step > 0.003)
        converged = bool((not likely_oscillating) and tail_std <= 0.01)
        return {
            "policy_history_path": str(policy_path),
            "n_epochs": int(mu.size),
            "policy_mean_start": float(mu[0]),
            "policy_mean_end": float(mu[-1]),
            "policy_mean_abs_step": float(mean_abs_step),
            "policy_mean_sign_flip_rate": float(sign_flip_rate),
            "policy_mean_tail_std": float(tail_std),
            "policy_converged": bool(converged),
            "policy_likely_oscillating": bool(likely_oscillating),
        }

    def _collect_ablation_run_summary(
        self,
        *,
        run_outdir: Path,
        run_label: str,
        run_concept_rl: bool,
    ) -> dict[str, Any]:
        final_dir = run_outdir / "final_best_train_vs_leaderboard"
        eval_path = final_dir / "leaderboard_eval.json"
        if not eval_path.exists():
            raise FileNotFoundError(f"Missing final eval file for run '{run_label}': {eval_path}")
        eval_payload = json.loads(eval_path.read_text())

        pr_aucs = [float(eval_payload.get(f"pr_auc_task{i}", np.nan)) for i in range(4)]
        finite_pr_aucs = [x for x in pr_aucs if np.isfinite(x)]
        min_task_pr_auc = float(min(finite_pr_aucs)) if len(finite_pr_aucs) > 0 else float("nan")
        row: dict[str, Any] = {
            "run_label": str(run_label),
            "run_concept_rl": bool(run_concept_rl),
            "run_outdir": str(run_outdir),
            "final_dir": str(final_dir),
            "leaderboard_eval_path": str(eval_path),
            "macro_pr_auc": float(eval_payload.get("macro_pr_auc", np.nan)),
            "min_task_pr_auc": float(min_task_pr_auc),
            "macro_auc": float(eval_payload.get("macro_auc", np.nan)),
            "best_epoch": eval_payload.get("best_epoch"),
            "best_ckpt_path": eval_payload.get("best_ckpt_path"),
        }
        for i in range(4):
            row[f"pr_auc_task{i}"] = float(eval_payload.get(f"pr_auc_task{i}", np.nan))
            row[f"auc_task{i}"] = float(eval_payload.get(f"auc_task{i}", np.nan))

        policy_path = final_dir / "concept_rl_policy_history.json"
        if run_concept_rl and policy_path.exists():
            row.update(self._summarize_rl_policy(policy_path=policy_path))
        elif run_concept_rl:
            row.update(
                {
                    "policy_history_path": str(policy_path),
                    "n_epochs": 0,
                    "policy_mean_start": None,
                    "policy_mean_end": None,
                    "policy_mean_abs_step": None,
                    "policy_mean_sign_flip_rate": None,
                    "policy_mean_tail_std": None,
                    "policy_converged": None,
                    "policy_likely_oscillating": None,
                }
            )
        return row

    def _write_ablation_report(self, *, outdir: Path, summaries: list[dict[str, Any]]) -> None:
        if len(summaries) == 0:
            return
        csv_path = outdir / "final_concept_rl_ablation_comparison.csv"
        json_path = outdir / "final_concept_rl_ablation_comparison.json"
        md_path = outdir / "final_concept_rl_ablation_comparison.md"

        df = pd.DataFrame(summaries)
        df.to_csv(csv_path, index=False)

        by_label = {str(x.get("run_label")): x for x in summaries}
        no_rl = by_label.get("no_rl")
        with_rl = by_label.get("with_rl")
        delta_macro = None
        delta_min = None
        winner_macro = None
        winner_min = None
        if no_rl is not None and with_rl is not None:
            try:
                delta_macro = float(with_rl.get("macro_pr_auc", np.nan)) - float(
                    no_rl.get("macro_pr_auc", np.nan)
                )
            except Exception:
                delta_macro = None
            try:
                delta_min = float(with_rl.get("min_task_pr_auc", np.nan)) - float(
                    no_rl.get("min_task_pr_auc", np.nan)
                )
            except Exception:
                delta_min = None
            if delta_macro is not None and np.isfinite(delta_macro):
                winner_macro = "with_rl" if delta_macro >= 0.0 else "no_rl"
            if delta_min is not None and np.isfinite(delta_min):
                winner_min = "with_rl" if delta_min >= 0.0 else "no_rl"

        payload = {
            "runs": summaries,
            "comparison": {
                "delta_macro_pr_auc_with_rl_minus_no_rl": delta_macro,
                "delta_min_task_pr_auc_with_rl_minus_no_rl": delta_min,
                "winner_by_macro_pr_auc": winner_macro,
                "winner_by_min_task_pr_auc": winner_min,
            },
            "files": {
                "csv": str(csv_path),
                "markdown": str(md_path),
            },
        }
        json_path.write_text(json.dumps(payload, indent=2))

        md_lines = [
            "# Concept-RL Ablation Comparison",
            "",
            "| run | rl | macro_pr_auc | min_task_pr_auc | macro_auc | best_epoch | oscillating | converged |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in summaries:
            md_lines.append(
                "| "
                + f"{row.get('run_label')} | "
                + f"{int(bool(row.get('run_concept_rl')))} | "
                + f"{float(row.get('macro_pr_auc', np.nan)):.6f} | "
                + f"{float(row.get('min_task_pr_auc', np.nan)):.6f} | "
                + f"{float(row.get('macro_auc', np.nan)):.6f} | "
                + f"{row.get('best_epoch')} | "
                + f"{row.get('policy_likely_oscillating')} | "
                + f"{row.get('policy_converged')} |"
            )
        md_lines.extend(
            [
                "",
                "## Delta (with_rl - no_rl)",
                "",
                f"- macro_pr_auc: {delta_macro}",
                f"- min_task_pr_auc: {delta_min}",
                f"- winner_by_macro_pr_auc: {winner_macro}",
                f"- winner_by_min_task_pr_auc: {winner_min}",
            ]
        )
        md_path.write_text("\n".join(md_lines) + "\n")
        log_event(
            "INFO",
            "pipeline.final.ablation_report_written",
            csv_path=str(csv_path),
            json_path=str(json_path),
            markdown_path=str(md_path),
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
    ap.add_argument(
        "--feat2d_raw",
        default=None,
        help=(
            "Optional raw 2D feature table (same structure as --feat2d_scaled). "
            "Used by --run_family_suite for catboost_st only; MIL families keep --feat2d_scaled."
        ),
    )
    ap.add_argument("--feat3d_scaled", required=True)
    ap.add_argument("--feat3d_qm_scaled", required=True)
    ap.add_argument(
        "--feat3d_raw",
        default=None,
        help=(
            "Optional raw 3D-geometry table. "
            "If provided together with --feat3d_qm_raw, used for Chem-ACE semantic tagging."
        ),
    )
    ap.add_argument(
        "--feat3d_qm_raw",
        default=None,
        help=(
            "Optional raw 3D-quantum table. "
            "If provided together with --feat3d_raw, used for Chem-ACE semantic tagging."
        ),
    )
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
        "--run_family_suite",
        action="store_true",
        help=(
            "Run multi-family pipeline: "
            "catboost_st + mt_2d + mt_2d3d + mt_3d with calibration + blending."
        ),
    )
    ap.add_argument(
        "--model_families",
        nargs="+",
        default=["catboost_st", "mt_2d", "mt_2d3d", "mt_3d"],
        help="Model families for --run_family_suite.",
    )
    ap.add_argument(
        "--calibration_method",
        choices=["platt", "isotonic", "temperature"],
        default="platt",
        help="Post-hoc calibration method for family probabilities on leaderboard. Default: platt.",
    )
    ap.add_argument(
        "--skip_family_calibration",
        action="store_true",
        help=(
            "In --run_family_suite mode, skip post-hoc calibration and use identity probabilities "
            "for downstream outputs."
        ),
    )
    ap.add_argument(
        "--skip_family_explainability",
        action="store_true",
        help=(
            "In --run_family_suite mode, skip optional extended explainability integrations. "
            "Attention export for final 3D-bearing MIL models remains enabled by default."
        ),
    )
    ap.add_argument(
        "--skip_family_blending",
        action="store_true",
        help=(
            "In --run_family_suite mode, skip OOF/leaderboard blending and write single-model outputs only."
        ),
    )
    ap.add_argument(
        "--catboost_hpo_parallel_tasks",
        type=int,
        default=1,
        help=(
            "Parallel CatBoost task-HPO workers in --run_family_suite mode. "
            "Use 4 to optimize tasks t0..t3 concurrently."
        ),
    )
    ap.add_argument(
        "--blend_seed_ensemble_size",
        type=int,
        default=1,
        help=(
            "Number of seed replicas per model family for OOF/final predictions in --run_family_suite mode. "
            "Each replica is treated as a separate calibrated input to blending."
        ),
    )
    ap.add_argument(
        "--blend_seed_step",
        type=int,
        default=1000,
        help=(
            "Seed offset step between ensemble replicas in --run_family_suite mode. "
            "Replica r uses seed = seed + r * blend_seed_step."
        ),
    )
    ap.add_argument(
        "--best_params_dir",
        default=None,
        help=(
            "Directory with per-family best-params JSONs (<family>.json). "
            "If omitted, defaults to <study_dir>/best_params."
        ),
    )
    ap.add_argument(
        "--family_best_params_json",
        default=None,
        help=(
            "Optional JSON file with family-level best params overrides for --run_family_suite. "
            "Families present in this file skip HPO while other selected families can still be optimized. "
            "Supports either {'family_name': {...}} or saved payload style "
            "{'family': 'name', 'best_params': {...}}."
        ),
    )
    ap.add_argument(
        "--catboost_task_params_jsons",
        nargs="+",
        default=None,
        help=(
            "Optional 4 task-specific CatBoost params JSON files "
            "(each containing task_idx + best_params). "
            "When provided, catboost_st HPO is skipped and these params are used."
        ),
    )
    ap.add_argument(
        "--catboost_final_iterations_jsons",
        nargs="+",
        default=None,
        help=(
            "Optional 4 task-specific CatBoost iteration-selection JSON files "
            "(each containing task_idx + selected_iterations). "
            "When provided, these selected iterations override CV-derived CatBoost "
            "selected iterations for final refit."
        ),
    )
    ap.add_argument(
        "--mt_2d_params_json",
        default=None,
        help=(
            "Optional best-params JSON override for mt_2d in --run_family_suite mode. "
            "When provided, mt_2d HPO is skipped and these params are used."
        ),
    )
    ap.add_argument(
        "--mt_2d3d_params_json",
        default=None,
        help=(
            "Optional best-params JSON override for mt_2d3d in --run_family_suite mode. "
            "When provided, mt_2d3d HPO is skipped and these params are used."
        ),
    )
    ap.add_argument(
        "--mt_3d_params_json",
        default=None,
        help=(
            "Optional best-params JSON override for mt_3d in --run_family_suite mode. "
            "When provided, mt_3d HPO is skipped and these params are used."
        ),
    )
    ap.add_argument(
        "--mt_2d_final_epochs_json",
        default=None,
        help=(
            "Optional final-epochs JSON override for mt_2d in --run_family_suite mode. "
            "Expected payload with selected_epochs (e.g., final_epochs_mt_2d.json)."
        ),
    )
    ap.add_argument(
        "--mt_2d3d_final_epochs_json",
        default=None,
        help=(
            "Optional final-epochs JSON override for mt_2d3d in --run_family_suite mode. "
            "Expected payload with selected_epochs (e.g., final_epochs_mt_2d3d.json)."
        ),
    )
    ap.add_argument(
        "--mt_3d_final_epochs_json",
        default=None,
        help=(
            "Optional final-epochs JSON override for mt_3d in --run_family_suite mode. "
            "Expected payload with selected_epochs (e.g., final_epochs_mt_3d.json)."
        ),
    )
    ap.add_argument(
        "--run_hpo",
        action="store_true",
        help="Run Optuna CV optimization before final train.",
    )
    ap.add_argument(
        "--hpo_only",
        action="store_true",
        help="Run Optuna CV optimization only and skip final train/eval.",
    )
    ap.add_argument(
        "--best_params_json",
        default=None,
        help=(
            "Path to precomputed best params JSON (pipeline format). "
            "Used when --run_hpo is not set."
        ),
    )
    ap.add_argument(
        "--pruner_warmup_steps",
        type=int,
        default=8,
        help="Warmup validation-report steps before Optuna pruner can prune trials.",
    )

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nn_accelerator", default="gpu")
    ap.add_argument("--nn_devices", type=int, default=1)
    ap.add_argument("--precision", default="16-mixed")
    ap.add_argument("--num_workers", type=int, default=-1)
    ap.add_argument(
        "--cpu_workers",
        type=int,
        default=-1,
        help="CPU worker budget for CPU-bound pipeline stages; -1 auto-resolves from node CPUs.",
    )
    ap.add_argument("--pin_memory", action="store_true")

    # Deprecated compatibility flag; final train+leaderboard stage now always runs.
    ap.add_argument("--export_leaderboard_attn", action="store_true")
    ap.add_argument("--leaderboard_split", default="leaderboard")
    ap.add_argument("--attn_out", default=None)

    # Accepted for compatibility; pipeline is MIL-only now.
    ap.add_argument("--do_mil", action="store_true")

    # Explainability integrations: Chem-ACE + Lambda-Vol
    ap.add_argument("--run_chem_ace", action="store_true")
    ap.add_argument("--run_lambda_vol", action="store_true")
    ap.add_argument("--run_concept_rl", action="store_true")
    ap.add_argument(
        "--run_concept_rl_ablation",
        action="store_true",
        help=(
            "Run final stage twice with identical params/seed: "
            "baseline (no RL) and RL-enabled, then export comparison."
        ),
    )
    ap.add_argument("--curated_smiles_col", default="curated_SMILES")
    ap.add_argument(
        "--chem_ace_conformer_sdf",
        default=None,
        help=(
            "Path to precomputed conformer SDF. Entries are matched by conf_id; "
            "missing conformers are skipped for 3D patching."
        ),
    )
    ap.add_argument(
        "--chem_ace_sdf_conf_id_prop",
        default="conf_id",
        help="SDF property name storing conf_id (fallback: molecule name field).",
    )
    ap.add_argument("--chem_ace_output_dir", default=None)
    ap.add_argument("--chem_ace_db_uri", default=None)
    ap.add_argument(
        "--chem_ace_max_ids",
        type=int,
        default=0,
        help="Max molecule IDs for Chem-ACE (0 means use all available IDs in scope).",
    )
    ap.add_argument(
        "--chem_ace_max_confs_per_id",
        type=int,
        default=0,
        help="Max conformers per molecule for Chem-ACE (<=0 means use all conformers).",
    )
    ap.add_argument(
        "--chem_ace_local_radii",
        nargs="+",
        type=int,
        default=[1],
        help="Local-subgraph radii for 2D patching.",
    )
    ap.add_argument(
        "--chem_ace_patch_cap_per_mol",
        type=int,
        default=0,
        help="Per-molecule patch cap (<=0 enables dynamic auto-cap).",
    )
    ap.add_argument(
        "--chem_ace_target_total_patches",
        type=int,
        default=1200000,
        help="Target total patches when auto-cap is enabled (<=0 disables auto-cap).",
    )
    ap.add_argument(
        "--chem_ace_embed_dim_2d",
        type=int,
        default=64,
        help="Hybrid Chem-ACE concept embedding width for 2D modality.",
    )
    ap.add_argument(
        "--chem_ace_embed_dim_3d_geom",
        type=int,
        default=64,
        help="Hybrid Chem-ACE concept embedding width for 3D geometry modality.",
    )
    ap.add_argument(
        "--chem_ace_embed_dim_3d_qm",
        type=int,
        default=64,
        help="Hybrid Chem-ACE concept embedding width for 3D QM modality.",
    )
    ap.add_argument(
        "--chem_ace_context_dim",
        type=int,
        default=16,
        help="Context projection width used by hybrid patch embeddings.",
    )
    ap.add_argument(
        "--chem_ace_context_alpha",
        type=float,
        default=0.2,
        help="Weight of context branch in hybrid local+context patch embeddings.",
    )
    ap.add_argument(
        "--chem_ace_qm_gating",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable geometry-gated local QM contribution in 3D QM patch embeddings.",
    )
    ap.add_argument(
        "--chem_ace_max_2d_dim",
        type=int,
        default=0,
        help="[Deprecated] Legacy feature-projection cap. Kept for CLI compatibility only.",
    )
    ap.add_argument(
        "--chem_ace_max_3dqm_dim",
        type=int,
        default=0,
        help="[Deprecated] Legacy feature-projection cap. Kept for CLI compatibility only.",
    )
    ap.add_argument(
        "--chem_ace_persist_patch_embeddings",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Persist per-patch embedding files and DB rows. "
            "Disabled by default to avoid massive disk usage."
        ),
    )
    ap.add_argument(
        "--chem_ace_top_concepts",
        type=int,
        default=0,
        help="Top concepts kept from Chem-ACE by support (<=0 keeps all discovered concepts).",
    )
    ap.add_argument(
        "--chem_ace_infer_max_distance",
        type=float,
        default=-1.0,
        help=(
            "Max L2 distance for nearest-centroid assignment on infer scope "
            "(<=0 disables distance gating)."
        ),
    )
    ap.add_argument(
        "--run_activity_calibration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Calibrate semantic tag confidence from train-scope activity labels "
            "(task + 16-bitmask supervision)."
        ),
    )
    ap.add_argument("--activity_calibration_min_concept_support", type=int, default=12)
    ap.add_argument("--activity_calibration_min_tag_support", type=int, default=24)
    ap.add_argument("--activity_calibration_prior_strength", type=float, default=32.0)
    ap.add_argument("--activity_calibration_min_w", type=float, default=0.40)
    ap.add_argument("--activity_calibration_task_weight", type=float, default=0.70)
    ap.add_argument("--activity_calibration_bitmask_weight", type=float, default=0.30)
    ap.add_argument("--activity_calibration_bitmask_min_count", type=int, default=20)
    ap.add_argument(
        "--activity_calibration_bitmask_exclude_zero",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    ap.add_argument("--activity_calibration_mix_base", type=float, default=0.60)
    ap.add_argument("--activity_calibration_keep_threshold", type=float, default=0.55)
    ap.add_argument("--activity_calibration_min_confidence", type=float, default=0.05)
    ap.add_argument("--activity_calibration_max_confidence", type=float, default=0.99)
    ap.add_argument("--activity_calibration_ratio_cap", type=float, default=8.0)
    ap.add_argument(
        "--activity_calibration_fallback_top1_if_empty",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    ap.add_argument(
        "--chem_ace_use_advanced_geom_topology",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable advanced geometry/topology descriptors from conformer coordinates.",
    )
    ap.add_argument("--chem_ace_advanced_geom_topology_max_patches", type=int, default=3000)
    ap.add_argument("--chem_ace_advanced_geom_topology_min_atoms", type=int, default=4)
    ap.add_argument("--chem_ace_advanced_geom_topology_max_torsion_paths", type=int, default=96)
    ap.add_argument(
        "--chem_ace_advanced_geom_use_convex_hull",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use convex-hull based surface/cavity proxies when scipy is available.",
    )
    ap.add_argument(
        "--chem_ace_advanced_geom_use_persistent_homology",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use ripser-based H1 persistence features when ripser is available.",
    )
    ap.add_argument("--chem_ace_advanced_geom_persistence_max_atoms", type=int, default=48)
    ap.add_argument(
        "--chem_ace_use_orca_descriptors",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable optional external ORCA descriptor table integration for semantics.",
    )
    ap.add_argument("--chem_ace_orca_descriptors_path", default=None)
    ap.add_argument("--chem_ace_orca_conf_id_col", default="conf_id")
    ap.add_argument("--chem_ace_orca_mol_id_col", default="ID")
    ap.add_argument(
        "--chem_ace_orca_descriptor_cols",
        nargs="+",
        default=[],
        help="Optional explicit ORCA descriptor columns; defaults to all numeric non-ID columns.",
    )
    ap.add_argument("--chem_ace_orca_min_vectors_for_tagging", type=int, default=8)
    ap.add_argument("--chem_ace_orca_z_threshold", type=float, default=0.50)
    ap.add_argument(
        "--chem_ace_use_pmapper_signatures",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Compute conformer-level pmapper pharmacophore signatures from SDF and "
            "attach signature diagnostics to attention exports."
        ),
    )
    ap.add_argument(
        "--chem_ace_pmapper_tol",
        type=int,
        default=0,
        help="Primary pmapper signature tolerance (0 uses exact/default signature).",
    )
    ap.add_argument(
        "--chem_ace_pmapper_tol_alt",
        type=int,
        default=5,
        help="Secondary pmapper signature tolerance for coarse grouping.",
    )
    ap.add_argument(
        "--chem_ace_strict_rerank",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run strict post-discovery re-embedding in trained mixer space for "
            "concept medoids + top-attention conformers and use scores for final reranking."
        ),
    )
    ap.add_argument(
        "--chem_ace_strict_rerank_layer_name",
        default="mixer_post_norm",
        help="Layer used for strict re-embedding (default: mixer_post_norm).",
    )
    ap.add_argument(
        "--chem_ace_strict_rerank_top_rows_per_task",
        type=int,
        default=256,
        help=(
            "Top-attention conformer rows per task used in strict pass; "
            "<=0 means all rows (can be expensive)."
        ),
    )
    ap.add_argument(
        "--chem_ace_strict_rerank_batch_size",
        type=int,
        default=256,
        help="Batch size for strict mixer-space re-embedding.",
    )
    ap.add_argument(
        "--chem_ace_strict_rerank_weight",
        type=float,
        default=0.35,
        help="Weight of strict mixer-space score in final concept explanation reranking.",
    )
    ap.add_argument("--lambda_vol_output_dir", default=None)
    ap.add_argument("--lambda_vol_db_uri", default=None)
    ap.add_argument("--lambda_vol_layer_name", default="mixer_post_norm")
    ap.add_argument(
        "--lambda_vol_top_concepts",
        type=int,
        default=0,
        help="Top concepts used by Lambda-Vol TCAV/pressure (<=0 uses all concepts from Chem-ACE).",
    )
    ap.add_argument("--lambda_vol_monitor_max_samples", type=int, default=512)
    ap.add_argument("--lambda_vol_tcav_repeats", type=int, default=2)
    ap.add_argument("--lambda_vol_random_counterexamples", type=int, default=96)
    ap.add_argument("--lambda_vol_min_concept_samples", type=int, default=8)
    ap.add_argument(
        "--lambda_vol_tcav_holdout_fraction",
        type=float,
        default=0.2,
        help="Holdout fraction for TCAV directional-derivative evaluation (<=0 disables holdout).",
    )
    ap.add_argument(
        "--lambda_vol_tcav_holdout_min_samples",
        type=int,
        default=16,
        help="Minimum holdout evaluation samples for TCAV when holdout is enabled.",
    )
    ap.add_argument(
        "--lambda_vol_tcav_significance_alpha",
        type=float,
        default=0.05,
        help="Significance alpha for repeat-level and pooled TCAV tests.",
    )
    ap.add_argument(
        "--lambda_vol_tcav_bonferroni_m",
        type=int,
        default=0,
        help="Bonferroni hypothesis count for TCAV significance (<=0 auto-uses number of concepts).",
    )
    ap.add_argument(
        "--lambda_vol_run_ricci",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    ap.add_argument("--lambda_vol_ricci_edge_keep_quantile", type=float, default=0.75)
    ap.add_argument("--lambda_vol_ricci_min_edge_weight", type=float, default=0.05)
    ap.add_argument("--lambda_vol_ricci_top_k_per_node", type=int, default=4)
    ap.add_argument("--lambda_vol_ricci_flow_steps", type=int, default=8)
    ap.add_argument("--lambda_vol_ricci_flow_step_size", type=float, default=0.12)
    ap.add_argument(
        "--lambda_vol_ricci_use_flow_as_coupling",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    ap.add_argument("--lambda_vol_ricci_coupling_strength", type=float, default=0.05)

    # Concept-RL guidance (final train only; relies on Chem-ACE concepts)
    ap.add_argument(
        "--concept_rl_top_k_per_task",
        type=int,
        default=8,
        help="Per-task cap of RL target concepts; <=0 means all passing concepts (no cap).",
    )
    ap.add_argument("--concept_rl_min_pos_coverage", type=float, default=0.02)
    ap.add_argument("--concept_rl_min_pos_hits", type=int, default=8)
    ap.add_argument("--concept_rl_min_lift", type=float, default=1.05)
    ap.add_argument("--concept_rl_overlap_weight", type=float, default=0.35)
    ap.add_argument("--concept_rl_global_weight", type=float, default=0.20)
    ap.add_argument("--concept_rl_lift_weight", type=float, default=0.25)
    ap.add_argument("--concept_rl_general_top_k", type=int, default=4)
    ap.add_argument("--concept_rl_min_multi_active_count", type=int, default=32)
    ap.add_argument(
        "--concept_rl_require_conf_support",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    ap.add_argument("--concept_rl_min_conf_pos_coverage", type=float, default=0.005)
    ap.add_argument("--concept_rl_init_scale", type=float, default=0.02)
    ap.add_argument("--concept_rl_max_scale", type=float, default=0.20)
    ap.add_argument("--concept_rl_policy_lr", type=float, default=0.05)
    ap.add_argument("--concept_rl_policy_sigma", type=float, default=0.02)
    ap.add_argument("--concept_rl_reward_alignment_w", type=float, default=0.25)
    ap.add_argument("--concept_rl_reward_min_ap_w", type=float, default=0.15)
    ap.add_argument("--concept_rl_baseline_momentum", type=float, default=0.90)
    ap.add_argument("--concept_rl_negative_penalty", type=float, default=0.15)

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
    if bool(args.hpo_only) and (not bool(args.run_hpo)):
        raise ValueError("--hpo_only requires --run_hpo")
    if bool(args.hpo_only):
        # HPO-only mode intentionally skips final-stage explainability pipeline.
        args.run_chem_ace = False
        args.run_lambda_vol = False
        args.run_concept_rl = False
        args.run_concept_rl_ablation = False
    if bool(args.run_lambda_vol) or bool(args.run_concept_rl) or bool(args.run_concept_rl_ablation):
        # Lambda-Vol and concept RL both rely on concept families from Chem-ACE.
        args.run_chem_ace = True


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
    with log_step("pipeline.main"):
        args = _parse_args(argv)
        _normalize_compat_args(args)
        if bool(getattr(args, "run_family_suite", False)):
            from .family_suite import run_family_suite

            log_event(
                "INFO",
                "pipeline.main.family_suite_dispatch",
                run_hpo=bool(args.run_hpo),
                hpo_only=bool(args.hpo_only),
                study_dir=str(args.study_dir),
                families=[str(x) for x in (args.model_families or [])],
                calibration_method=str(args.calibration_method),
                skip_family_explainability=bool(getattr(args, "skip_family_explainability", False)),
                skip_family_calibration=bool(getattr(args, "skip_family_calibration", False)),
                skip_family_blending=bool(getattr(args, "skip_family_blending", False)),
            )
            run_family_suite(args)
            return
        config = PipelineConfigFactory.from_args(args)
        log_event(
            "INFO",
            "pipeline.main.args",
            run_hpo=bool(config.hpo.run_hpo),
            hpo_only=bool(config.hpo.hpo_only),
            study_dir=str(config.data_paths.study_dir),
            num_workers=int(config.runtime.num_workers),
            cpu_workers=int(config.runtime.cpu_workers),
            run_chem_ace=bool(config.explainability.run_chem_ace),
            run_lambda_vol=bool(config.explainability.run_lambda_vol),
            run_concept_rl=bool(config.explainability.run_concept_rl),
            run_concept_rl_ablation=bool(config.explainability.run_concept_rl_ablation),
        )
        MILPipelineOrchestrator(config=config, argv=argv).run()


if __name__ == "__main__":
    import sys as _sys

    main(_sys.argv[1:])
