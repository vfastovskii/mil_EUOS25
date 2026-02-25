from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..data.collate import collate_export
from ..data.datasets import MILExportDataset
from ..explainability.chem_ace.cav import collect_gradients_for_layer, run_tcav_from_arrays
from ..explainability.chem_ace.config import (
    CAVConfig,
    ChemACEConfig,
    ConceptDiscoveryConfig,
    DatabaseConfig,
    EmbeddingConfig,
    LocalSubgraphPatchConfig,
    PatchGenerationConfig,
    Pharm3DPatchConfig,
    SemanticTaggingConfig,
)
from ..explainability.chem_ace.concepts.pipeline import ChemACEPipeline, MoleculeSource
from ..explainability.chem_ace.embedding.hooks import LayerActivationHook
from ..explainability.chem_ace.optional_deps import OptionalDependencyError, require_rdkit
from ..explainability.chem_ace.rules.rdkit_fragment_rules import (
    default_functional_rules_path,
    generate_fragment_rules_from_smiles,
    merge_rules,
)
from ..explainability.chem_ace.semantics.calibration import (
    ActivityAwareSemanticCalibrator,
    ActivityCalibrationConfig,
)
from ..explainability.chem_ace.types import (
    ConceptCandidate,
    ConceptMembership,
    ModelTaskAdapter,
    PatchEmbeddingRecord,
    PatchRecord,
)
from ..explainability.lambda_vol import LambdaVolConfig
from ..explainability.lambda_vol.config import ExportConfig as LambdaVolExportConfig
from ..explainability.lambda_vol.config import PolicyConfig, RicciConfig, StoreConfig, TrackerConfig
from ..explainability.lambda_vol.integrations import LambdaVolLightningCallback, LightningEpochFrames
from ..explainability.lambda_vol.monitor import LambdaVolMonitor
from ..training.builders import DataLoaderBuilder, LoaderConfig
from ..utils.constants import TASK_COLS
from ..utils.progress import log_event, log_step

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FinalExplainabilityConfig:
    """Controls Chem-ACE + Lambda-Vol integration in final optimized training."""

    run_chem_ace: bool = False
    run_lambda_vol: bool = False
    curated_smiles_col: str = "curated_SMILES"
    chem_ace_conformer_sdf: Optional[str] = None
    chem_ace_sdf_conf_id_prop: str = "conf_id"
    cpu_workers: int = 0

    chem_ace_output_dir: Optional[str] = None
    chem_ace_db_uri: Optional[str] = None
    chem_ace_max_ids: int = 0
    # <=0 means use all available conformers per molecule.
    chem_ace_max_confs_per_id: int = 0
    chem_ace_local_radii: tuple[int, ...] = (1,)
    # <= 0 enables dynamic cap by chem_ace_target_total_patches / n_molecules.
    chem_ace_patch_cap_per_mol: int = 0
    chem_ace_target_total_patches: int = 1200000
    # <=0 means use full available 2D feature dimension.
    chem_ace_max_2d_dim: int = 0
    # <=0 means use full available merged 3D+QM feature dimension.
    chem_ace_max_3dqm_dim: int = 0
    # If False, keep patch embeddings only in memory for this run (no per-patch npy/json, no DB rows).
    chem_ace_persist_patch_embeddings: bool = False
    chem_ace_top_concepts: int = 64
    # <=0 disables distance gating for inference-time nearest-centroid assignment.
    chem_ace_infer_max_distance: float = -1.0
    # Supervised semantic confidence calibration from train-scope activity labels.
    run_activity_calibration: bool = True
    activity_calibration_min_concept_support: int = 12
    activity_calibration_min_tag_support: int = 24
    activity_calibration_prior_strength: float = 32.0
    activity_calibration_min_w: float = 0.40
    activity_calibration_task_weight: float = 0.70
    activity_calibration_bitmask_weight: float = 0.30
    activity_calibration_bitmask_min_count: int = 20
    activity_calibration_bitmask_exclude_zero: bool = True
    activity_calibration_mix_base: float = 0.60
    activity_calibration_keep_threshold: float = 0.55
    activity_calibration_min_confidence: float = 0.05
    activity_calibration_max_confidence: float = 0.99
    activity_calibration_ratio_cap: float = 8.0
    activity_calibration_fallback_top1_if_empty: bool = True
    # Advanced geometry/topology semantics from conformer coordinates.
    chem_ace_use_advanced_geom_topology: bool = True
    chem_ace_advanced_geom_topology_max_patches: int = 3000
    chem_ace_advanced_geom_topology_min_atoms: int = 4
    chem_ace_advanced_geom_topology_max_torsion_paths: int = 96
    chem_ace_advanced_geom_use_convex_hull: bool = True
    chem_ace_advanced_geom_use_persistent_homology: bool = True
    chem_ace_advanced_geom_persistence_max_atoms: int = 48
    # Optional external ORCA descriptor table integration.
    chem_ace_use_orca_descriptors: bool = False
    chem_ace_orca_descriptors_path: Optional[str] = None
    chem_ace_orca_conf_id_col: str = "conf_id"
    chem_ace_orca_mol_id_col: str = "ID"
    chem_ace_orca_descriptor_cols: tuple[str, ...] = ()
    chem_ace_orca_min_vectors_for_tagging: int = 8
    chem_ace_orca_z_threshold: float = 0.50

    lambda_vol_output_dir: Optional[str] = None
    lambda_vol_db_uri: Optional[str] = None
    lambda_vol_layer_name: str = "mixer_post_norm"
    lambda_vol_top_concepts: int = 24
    lambda_vol_monitor_max_samples: int = 512
    lambda_vol_tcav_repeats: int = 2
    lambda_vol_random_counterexamples: int = 96
    lambda_vol_min_concept_samples: int = 8
    lambda_vol_run_ricci: bool = True
    lambda_vol_ricci_edge_keep_quantile: float = 0.75
    lambda_vol_ricci_min_edge_weight: float = 0.05
    lambda_vol_ricci_top_k_per_node: int = 4
    lambda_vol_ricci_flow_steps: int = 8
    lambda_vol_ricci_flow_step_size: float = 0.12
    lambda_vol_ricci_use_flow_as_coupling: bool = True
    lambda_vol_ricci_coupling_strength: float = 0.05

    # Concept RL guidance (applied during final training).
    run_concept_rl: bool = False
    concept_rl_top_k_per_task: int = 8
    concept_rl_min_pos_coverage: float = 0.02
    concept_rl_init_scale: float = 0.02
    concept_rl_max_scale: float = 0.20
    concept_rl_policy_lr: float = 0.05
    concept_rl_policy_sigma: float = 0.02
    concept_rl_reward_alignment_w: float = 0.25
    concept_rl_baseline_momentum: float = 0.90


@dataclass(frozen=True)
class ChemACEBundle:
    """Prepared Chem-ACE artifacts and concept mappings for downstream Lambda-Vol."""

    output_dir: str
    db_uri: str
    run_id: str
    concept_set_id: str
    concept_ids: tuple[str, ...]
    concept_metadata: Mapping[str, Mapping[str, Any]]
    concept_support: Mapping[str, int]
    concept_mol_map: Mapping[str, set[str]]
    concept_conf_map: Mapping[str, set[tuple[str, str]]]
    a_priori_tags_csv: Optional[str] = None
    a_priori_vs_concepts_csv: Optional[str] = None
    a_priori_tags_infer_csv: Optional[str] = None
    a_priori_vs_concepts_infer_csv: Optional[str] = None
    activity_calibrated_tags_csv: Optional[str] = None
    activity_calibration_summary_json: Optional[str] = None


def build_positive_concept_targets(
    *,
    config: FinalExplainabilityConfig,
    ids_train: Sequence[str],
    y_cls_train: np.ndarray,
    chem_bundle: ChemACEBundle,
) -> dict[int, tuple[str, ...]]:
    """
    Select target concept sets per task from concepts frequent in positive train samples.

    Returns mapping:
      task_index -> tuple(concept_id, ...)
    """
    if (not bool(config.run_concept_rl)) or len(ids_train) == 0:
        return {}

    y = np.asarray(y_cls_train, dtype=np.float32)
    if y.ndim != 2 or y.shape[1] != len(TASK_COLS):
        raise ValueError(
            f"y_cls_train must be [N,{len(TASK_COLS)}], got shape={tuple(y.shape)}"
        )

    ids = [str(x) for x in ids_train]
    concept_ids = [str(x) for x in chem_bundle.concept_ids]
    concept_mol_map = {
        str(k): {str(x) for x in v}
        for k, v in chem_bundle.concept_mol_map.items()
    }

    out: dict[int, tuple[str, ...]] = {}
    top_k = max(1, int(config.concept_rl_top_k_per_task))
    min_cov = float(max(0.0, config.concept_rl_min_pos_coverage))

    for ti in range(len(TASK_COLS)):
        pos_ids = [ids[i] for i in range(len(ids)) if float(y[i, ti]) > 0.5]
        pos_set = set(pos_ids)
        n_pos = max(1, len(pos_set))
        if len(pos_set) == 0:
            out[int(ti)] = tuple()
            continue

        scored: list[tuple[int, float, str]] = []
        for cid in concept_ids:
            hits = int(len(pos_set.intersection(concept_mol_map.get(cid, set()))))
            cov = float(hits) / float(n_pos)
            if cov >= min_cov:
                scored.append((hits, cov, str(cid)))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

        chosen = [cid for _, _, cid in scored[:top_k]]
        out[int(ti)] = tuple(chosen)

    logger.info(
        "Built concept RL targets",
        extra={
            "task_target_sizes": {int(k): int(len(v)) for k, v in out.items()},
            "top_k": int(top_k),
            "min_pos_coverage": float(min_cov),
        },
    )
    return out


class _MILTaskAdapter(ModelTaskAdapter):
    """Task adapter for MIL model gradients used in TCAV."""

    def __init__(self, model: torch.nn.Module, task_ids: Sequence[str]):
        self.model = model
        self.task_to_idx = {str(t): i for i, t in enumerate(task_ids)}

    def forward(self, model_input: Any) -> Any:
        if not isinstance(model_input, dict):
            raise TypeError("MIL task adapter expects dict model_input")
        return self.model(
            model_input["x2d"],
            model_input["x3d_pad"],
            model_input["key_padding_mask"],
            return_attn=False,
        )

    def get_task_scalar(self, model_output: Any, task_id: str) -> Any:
        logits = model_output[0] if isinstance(model_output, (tuple, list)) else model_output
        idx = int(self.task_to_idx[str(task_id)])
        return logits[:, idx].sum()


class MILLambdaVolFrameProvider:
    """Collects per-epoch TCAV/attention/task frames from the current MIL model."""

    def __init__(
        self,
        *,
        monitor_loader: DataLoader,
        concept_ids: Sequence[str],
        concept_mol_map: Mapping[str, set[str]],
        concept_conf_map: Mapping[str, set[tuple[str, str]]],
        task_ids: Sequence[str],
        layer_name: str,
        monitor_max_samples: int,
        tcav_repeats: int,
        tcav_random_counterexamples: int,
        tcav_min_concept_samples: int,
        seed: int,
    ) -> None:
        self.monitor_loader = monitor_loader
        self.concept_ids = tuple(str(x) for x in concept_ids)
        self.concept_mol_map = {str(k): set(v) for k, v in concept_mol_map.items()}
        self.concept_conf_map = {str(k): set(v) for k, v in concept_conf_map.items()}
        self.task_ids = tuple(str(x) for x in task_ids)
        self.layer_name = str(layer_name)
        self.monitor_max_samples = int(max(1, monitor_max_samples))
        self.tcav_repeats = int(max(1, tcav_repeats))
        self.tcav_random_counterexamples = int(max(8, tcav_random_counterexamples))
        self.tcav_min_concept_samples = int(max(2, tcav_min_concept_samples))
        self.seed = int(seed)

    def collect_epoch_frames(
        self,
        *,
        trainer: Any,
        pl_module: Any,
        epoch: int,
    ) -> LightningEpochFrames:
        model = pl_module
        model.eval()
        device = next(model.parameters()).device

        n_tasks = len(self.task_ids)
        n_concepts = len(self.concept_ids)

        support_sum = np.zeros((n_tasks, n_concepts), dtype=np.float64)
        prevalence_sum = np.zeros((n_tasks, n_concepts), dtype=np.float64)
        entropy_sum = np.zeros((n_tasks,), dtype=np.float64)
        witness_sum = np.zeros((n_tasks,), dtype=np.float64)

        sample_inputs: list[dict[str, torch.Tensor]] = []
        sample_mol_ids: list[str] = []
        sample_has_concept: list[np.ndarray] = []
        activations: list[np.ndarray] = []

        with torch.no_grad():
            for mol_ids, conf_pad, x2d, x3d, kpm in self.monitor_loader:
                x2d = x2d.to(device, non_blocking=True)
                x3d = x3d.to(device, non_blocking=True)
                kpm = kpm.to(device, non_blocking=True)

                with LayerActivationHook(model, self.layer_name) as hook:
                    logits, _, _, attn = model(x2d, x3d, kpm, return_attn=True)

                if hook.last_activation is None:
                    raise RuntimeError(f"No activation captured for layer '{self.layer_name}'")
                batch_act = _collapse_activation(hook.last_activation)

                attn_np = attn.detach().cpu().numpy()  # [B,T,N]
                kpm_np = kpm.detach().cpu().numpy().astype(bool)
                B, T, _N = attn_np.shape

                for b in range(B):
                    if len(sample_mol_ids) >= self.monitor_max_samples:
                        break
                    mol_id = str(mol_ids[b])
                    valid_mask = ~kpm_np[b]
                    L = int(valid_mask.sum())
                    if L <= 0:
                        continue
                    confs = [str(c) for c in conf_pad[b, :L].tolist()]

                    # Build per-task normalized attention over valid conformers.
                    attn_norm = np.zeros((T, L), dtype=np.float64)
                    for t in range(T):
                        w = attn_np[b, t, :L].astype(np.float64)
                        s = float(np.sum(w))
                        if (not np.isfinite(s)) or s <= 0.0:
                            w[:] = 1.0 / float(L)
                        else:
                            w /= s
                        attn_norm[t] = w
                        entropy_sum[t] += _normalized_entropy(w)
                        witness_sum[t] += float(np.max(w))

                    has_concept = np.zeros((n_concepts,), dtype=bool)
                    for ci, concept_id in enumerate(self.concept_ids):
                        mol_present = mol_id in self.concept_mol_map.get(concept_id, set())
                        has_concept[ci] = bool(mol_present)

                        conf_hits = [
                            i
                            for i, conf in enumerate(confs)
                            if (mol_id, str(conf)) in self.concept_conf_map.get(concept_id, set())
                        ]

                        prevalence_val = 1.0 if mol_present else 0.0
                        for t in range(n_tasks):
                            prevalence_sum[t, ci] += prevalence_val
                            if conf_hits:
                                support_sum[t, ci] += float(np.sum(attn_norm[t, conf_hits]))
                            elif mol_present:
                                support_sum[t, ci] += 1.0

                    sample_mol_ids.append(mol_id)
                    sample_has_concept.append(has_concept)
                    activations.append(batch_act[b].astype(np.float32))

                    sample_inputs.append(
                        {
                            "x2d": x2d[b : b + 1].detach().cpu(),
                            "x3d_pad": x3d[b : b + 1].detach().cpu(),
                            "key_padding_mask": kpm[b : b + 1].detach().cpu(),
                        }
                    )

                if len(sample_mol_ids) >= self.monitor_max_samples:
                    break

        n_samples = max(1, len(sample_mol_ids))
        attention_support = (support_sum / float(n_samples)).astype(np.float32)
        prevalence = (prevalence_sum / float(n_samples)).astype(np.float32)
        attention_entropy = (entropy_sum / float(n_samples)).astype(np.float32)
        witness_rate = (witness_sum / float(n_samples)).astype(np.float32)

        tcav_scores = np.zeros((n_tasks, n_concepts), dtype=np.float32)
        if len(sample_mol_ids) >= max(self.tcav_min_concept_samples * 2, 8):
            tcav_scores = self._compute_tcav_matrix(
                model=model,
                device=device,
                epoch=int(epoch),
                sample_inputs=sample_inputs,
                sample_has_concept=np.stack(sample_has_concept, axis=0),
                activation_matrix=np.stack(activations, axis=0),
            )

        concept_attention_df = _build_concept_attention_df(
            task_ids=self.task_ids,
            concept_ids=self.concept_ids,
            attention_support=attention_support,
            prevalence=prevalence,
        )
        tcav_df = _build_tcav_df(
            task_ids=self.task_ids,
            concept_ids=self.concept_ids,
            tcav_matrix=tcav_scores,
        )

        task_attention_df = pd.DataFrame(
            {
                "task_id": list(self.task_ids),
                "attention_entropy": [float(x) for x in attention_entropy.tolist()],
                "witness_rate": [float(x) for x in witness_rate.tolist()],
            }
        )

        task_metrics_df, context_covariates = _build_task_metric_frames(
            trainer=trainer,
            task_ids=self.task_ids,
            attention_entropy=attention_entropy,
            witness_rate=witness_rate,
        )

        return LightningEpochFrames(
            tcav_df=tcav_df,
            concept_attention_df=concept_attention_df,
            task_attention_df=task_attention_df,
            task_metrics_df=task_metrics_df,
            context_covariates=context_covariates,
        )

    def _compute_tcav_matrix(
        self,
        *,
        model: torch.nn.Module,
        device: torch.device,
        epoch: int,
        sample_inputs: Sequence[dict[str, torch.Tensor]],
        sample_has_concept: np.ndarray,
        activation_matrix: np.ndarray,
    ) -> np.ndarray:
        tcav_scores = np.zeros((len(self.task_ids), len(self.concept_ids)), dtype=np.float32)
        log_event(
            "INFO",
            "explainability.lambda_vol.tcav.compute.start",
            layer_name=str(self.layer_name),
            n_tasks=int(len(self.task_ids)),
            n_concepts=int(len(self.concept_ids)),
            concept_source="chem_ace_memberships_from_feature_fusion_2d3dqm",
        )

        adapter = _MILTaskAdapter(model=model, task_ids=self.task_ids)
        with torch.enable_grad():
            for ti, task_id in enumerate(self.task_ids):
                grads = collect_gradients_for_layer(
                    model=model,
                    layer_name=self.layer_name,
                    task_id=task_id,
                    adapter=adapter,
                    model_inputs=sample_inputs,
                    device=str(device),
                )
                if grads.ndim != 2 or grads.shape[0] != activation_matrix.shape[0]:
                    continue

                for ci, concept_id in enumerate(self.concept_ids):
                    mask = sample_has_concept[:, ci]
                    n_pos = int(mask.sum())
                    n_neg = int((~mask).sum())
                    if n_pos < self.tcav_min_concept_samples or n_neg < self.tcav_min_concept_samples:
                        continue

                    _cav_records, _tcav_records, summary = run_tcav_from_arrays(
                        run_id=f"lambda_vol_epoch_{int(epoch)}",
                        epoch=int(epoch),
                        concept_id=str(concept_id),
                        task_id=str(task_id),
                        layer_name=str(self.layer_name),
                        concept_embeddings=activation_matrix[mask],
                        random_pool_embeddings=activation_matrix[~mask],
                        target_gradients=grads,
                        config=CAVConfig(
                            classifier="logreg",
                            n_random_repeats=int(self.tcav_repeats),
                            random_counterexamples_per_repeat=int(self.tcav_random_counterexamples),
                            max_iter=1200,
                        ),
                        seed=int(self.seed + 1000 * int(epoch) + 10 * ti + ci),
                    )
                    tcav_scores[ti, ci] = float(summary.mean_sign_rate)
        model.zero_grad(set_to_none=True)
        valid_scores = int(np.isfinite(tcav_scores).sum())
        log_event(
            "INFO",
            "explainability.lambda_vol.tcav.compute.done",
            layer_name=str(self.layer_name),
            valid_scores=int(valid_scores),
        )
        return tcav_scores



def _prepare_dataset_functional_rules(
    *,
    df_full: pd.DataFrame,
    smiles_col: str,
    out_dir: Path,
    min_count: int = 10,
    min_prevalence: float = 0.0002,
) -> Optional[Path]:
    """
    Build RDKit fragment rules from dataset SMILES and merge with default functional rules.

    Returns path to merged rules JSON, or None when generation is not possible.
    """
    if smiles_col not in df_full.columns:
        return None

    smiles = (
        df_full[smiles_col]
        .dropna()
        .astype(str)
        .str.strip()
    )
    smiles = smiles[smiles.str.len() > 0].drop_duplicates().tolist()
    if len(smiles) == 0:
        log_event(
            "WARN",
            "explainability.chem_ace.functional_rules.no_smiles",
            smiles_col=str(smiles_col),
        )
        return None

    try:
        generated_rules, summary, fragment_stats = generate_fragment_rules_from_smiles(
            smiles_iter=smiles,
            min_count=int(min_count),
            min_prevalence=float(min_prevalence),
        )
    except Exception as exc:
        log_event(
            "WARN",
            "explainability.chem_ace.functional_rules.generation_failed",
            error=str(exc),
        )
        return None

    base_rules_path = default_functional_rules_path()
    payload_base: dict[str, Any] = {}
    base_rules: list[dict[str, Any]] = []
    if base_rules_path.exists():
        try:
            payload_base = json.loads(base_rules_path.read_text())
            base_rules = list(payload_base.get("rules", []))
        except Exception as exc:
            log_event(
                "WARN",
                "explainability.chem_ace.functional_rules.base_load_failed",
                path=str(base_rules_path),
                error=str(exc),
            )
            payload_base = {}
            base_rules = []

    merged_rules = merge_rules(base_rules=base_rules, generated_rules=generated_rules)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_rules = out_dir / "default_functional_group_rules.dataset.json"
    payload_out = {
        **payload_base,
        "rules": merged_rules,
        "rdkit_fragment_generation": {
            **summary.to_dict(),
            "smiles_col": str(smiles_col),
            "n_unique_smiles": int(len(smiles)),
            "source": "prepare_chem_ace_bundle",
        },
    }
    out_rules.write_text(json.dumps(payload_out, indent=2, sort_keys=False))

    out_stats = out_dir / "functional_group_fragment_stats.json"
    out_stats.write_text(
        json.dumps(
            {
                **summary.to_dict(),
                "smiles_col": str(smiles_col),
                "n_unique_smiles": int(len(smiles)),
                "fragments": fragment_stats,
            },
            indent=2,
            sort_keys=False,
        )
    )

    log_event(
        "INFO",
        "explainability.chem_ace.functional_rules.generated",
        path=str(out_rules),
        n_base_rules=int(len(base_rules)),
        n_generated_rules=int(summary.n_generated_rules),
        n_total_rules=int(len(merged_rules)),
        min_count=int(min_count),
        min_prevalence=float(min_prevalence),
        n_unique_smiles=int(len(smiles)),
    )
    return out_rules


def _semicolon_join(values: Iterable[str]) -> str:
    cleaned = sorted({str(v).strip() for v in values if str(v).strip()})
    return ";".join(cleaned)


def _molecule_a_priori_semantic_tags(
    *,
    mol: Any,
    semantic_tagger: Any,
) -> dict[str, list[str]]:
    """
    Compute molecule-level a priori tags from structure-only rule systems.

    This intentionally uses only functional SMARTS / RDKit fragment rules and SMARTS-RX
    patterns, without concept discovery or model activations.
    """
    functional_tags: set[str] = set()
    smartsrx_tags: set[str] = set()
    smartsrx_roles: set[str] = set()

    # Functional SMARTS rules.
    for rule, pattern in getattr(semantic_tagger, "_functional_patterns", ()):
        try:
            hit = bool(mol.HasSubstructMatch(pattern))
        except Exception:
            hit = False
        if hit:
            functional_tags.add(str(rule.tag))

    # Functional RDKit fragment-counter rules.
    frag_fns = getattr(semantic_tagger, "_functional_fragment_functions", {})
    if isinstance(frag_fns, Mapping) and len(frag_fns) > 0:
        for rule in getattr(semantic_tagger, "functional_rules", ()):
            fn_name = str(getattr(rule, "rdkit_fragment", "")).strip()
            if not fn_name:
                continue
            fn = frag_fns.get(fn_name)
            if fn is None:
                continue
            try:
                value = float(fn(mol))
            except Exception:
                value = 0.0
            if np.isfinite(value) and value > 0.0:
                functional_tags.add(str(rule.tag))

    # SMARTS-RX rule-level and role tags.
    for rule, pattern in getattr(semantic_tagger, "_smarts_rx_patterns", ()):
        try:
            hit = bool(mol.HasSubstructMatch(pattern))
        except Exception:
            hit = False
        if not hit:
            continue
        smartsrx_tags.add(str(rule.tag))
        role = str(getattr(rule, "role", "")).strip()
        if role:
            role_tag = role if role.startswith("rx_role_") else f"rx_role_{role}"
            smartsrx_roles.add(role_tag)

    all_tags = sorted(functional_tags.union(smartsrx_tags).union(smartsrx_roles))
    return {
        "all_tags": all_tags,
        "functional_tags": sorted(functional_tags),
        "smartsrx_tags": sorted(smartsrx_tags),
        "smartsrx_role_tags": sorted(smartsrx_roles),
    }


def _export_a_priori_and_concept_views(
    *,
    out_dir: Path,
    ids_discover: Sequence[str],
    ids_infer: Sequence[str],
    smiles_by_id: Mapping[str, str],
    molecules_by_id: Mapping[str, Any],
    semantic_tagger: Any,
    concept_mol_map: Mapping[str, set[str]],
    concept_metadata: Mapping[str, Mapping[str, Any]],
) -> dict[str, Optional[str]]:
    """
    Export molecule-level a priori tags and joined concept annotations.

    Files written:
    - a_priori_tags.csv
    - a_priori_vs_concepts.csv
    - a_priori_tags_infer_scope.csv
    - a_priori_vs_concepts_infer_scope.csv
    """
    all_ids = sorted(set(str(x) for x in ids_discover).union(str(x) for x in ids_infer))
    infer_ids_set = {str(x) for x in ids_infer}

    rows: list[dict[str, Any]] = []
    for mol_id in all_ids:
        mol = molecules_by_id.get(str(mol_id))
        if mol is None:
            continue
        tags = _molecule_a_priori_semantic_tags(mol=mol, semantic_tagger=semantic_tagger)
        rows.append(
            {
                "ID": str(mol_id),
                "scope": ("infer_scope" if str(mol_id) in infer_ids_set else "discover_train"),
                "curated_SMILES": str(smiles_by_id.get(str(mol_id), "")),
                "n_a_priori_tags": int(len(tags["all_tags"])),
                "a_priori_tags": _semicolon_join(tags["all_tags"]),
                "a_priori_functional_tags": _semicolon_join(tags["functional_tags"]),
                "a_priori_smartsrx_tags": _semicolon_join(tags["smartsrx_tags"]),
                "a_priori_smartsrx_role_tags": _semicolon_join(tags["smartsrx_role_tags"]),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    df_priors = pd.DataFrame(rows)
    priors_csv = out_dir / "a_priori_tags.csv"
    df_priors.to_csv(priors_csv, index=False)

    # Build molecule -> concepts index.
    mol_to_concepts: dict[str, list[str]] = {}
    for concept_id, mols in concept_mol_map.items():
        cid = str(concept_id)
        for mid in mols:
            m = str(mid)
            if m not in mol_to_concepts:
                mol_to_concepts[m] = []
            mol_to_concepts[m].append(cid)

    concept_rows: list[dict[str, Any]] = []
    for row in rows:
        mid = str(row["ID"])
        cids = sorted(set(mol_to_concepts.get(mid, [])))
        labels = [
            str(concept_metadata.get(cid, {}).get("label_auto") or cid)
            for cid in cids
        ]
        ctags: list[str] = []
        for cid in cids:
            tags = concept_metadata.get(cid, {}).get("tags", [])
            if isinstance(tags, (list, tuple)):
                ctags.extend([str(t) for t in tags if str(t).strip()])
        concept_rows.append(
            {
                **row,
                "n_concepts": int(len(cids)),
                "concept_ids": _semicolon_join(cids),
                "concept_labels": _semicolon_join(labels),
                "concept_tags": _semicolon_join(ctags),
            }
        )

    df_join = pd.DataFrame(concept_rows)
    join_csv = out_dir / "a_priori_vs_concepts.csv"
    df_join.to_csv(join_csv, index=False)

    priors_infer_csv: Optional[Path] = None
    join_infer_csv: Optional[Path] = None
    if len(infer_ids_set) > 0:
        df_priors_infer = df_priors[df_priors["ID"].astype(str).isin(infer_ids_set)].copy()
        priors_infer_csv = out_dir / "a_priori_tags_infer_scope.csv"
        df_priors_infer.to_csv(priors_infer_csv, index=False)

        df_join_infer = df_join[df_join["ID"].astype(str).isin(infer_ids_set)].copy()
        join_infer_csv = out_dir / "a_priori_vs_concepts_infer_scope.csv"
        df_join_infer.to_csv(join_infer_csv, index=False)

    log_event(
        "INFO",
        "explainability.chem_ace.a_priori_exports",
        n_rows_all=int(len(df_priors)),
        n_rows_infer=int(0 if priors_infer_csv is None else len(df_priors[df_priors["ID"].astype(str).isin(infer_ids_set)])),
        path_priors=str(priors_csv),
        path_join=str(join_csv),
        path_priors_infer=(None if priors_infer_csv is None else str(priors_infer_csv)),
        path_join_infer=(None if join_infer_csv is None else str(join_infer_csv)),
    )

    return {
        "a_priori_tags_csv": str(priors_csv),
        "a_priori_vs_concepts_csv": str(join_csv),
        "a_priori_tags_infer_csv": (None if priors_infer_csv is None else str(priors_infer_csv)),
        "a_priori_vs_concepts_infer_csv": (
            None if join_infer_csv is None else str(join_infer_csv)
        ),
    }


def _extract_binary_labels_for_ids(
    *,
    df_full: pd.DataFrame,
    id_col: str,
    ids: Sequence[str],
    task_cols: Sequence[str] = TASK_COLS,
) -> tuple[list[str], np.ndarray]:
    """Return (ids_aligned, y_binary) for requested IDs using first row per ID."""
    if len(ids) == 0:
        return [], np.zeros((0, len(task_cols)), dtype=np.int64)
    missing = [str(c) for c in task_cols if str(c) not in df_full.columns]
    if missing:
        raise ValueError(f"Missing task columns for activity calibration: {missing}")

    frame = df_full[[id_col, *[str(c) for c in task_cols]]].copy()
    frame[id_col] = frame[id_col].astype(str)
    frame = frame.drop_duplicates(subset=[id_col], keep="first")
    for c in task_cols:
        frame[str(c)] = pd.to_numeric(frame[str(c)], errors="coerce").fillna(0.0)

    id_to_y: dict[str, np.ndarray] = {}
    for row in frame.itertuples(index=False):
        rid = str(getattr(row, id_col))
        vals = np.asarray([getattr(row, str(c)) for c in task_cols], dtype=np.float32)
        id_to_y[rid] = (vals > 0.0).astype(np.int64)

    ordered_ids: list[str] = []
    y_rows: list[np.ndarray] = []
    for rid in ids:
        r = str(rid)
        yv = id_to_y.get(r)
        if yv is None:
            continue
        ordered_ids.append(r)
        y_rows.append(yv)
    y = (
        np.stack(y_rows, axis=0).astype(np.int64)
        if len(y_rows) > 0
        else np.zeros((0, len(task_cols)), dtype=np.int64)
    )
    n_missing = int(len(ids) - len(ordered_ids))
    log_event(
        "INFO",
        "explainability.chem_ace.activity_calibration.labels_ready",
        n_requested_ids=int(len(ids)),
        n_labels_ids=int(len(ordered_ids)),
        n_missing_ids=int(n_missing),
        n_tasks=int(len(task_cols)),
    )
    return ordered_ids, y


def _export_activity_calibrated_tags(
    *,
    out_dir: Path,
    rows: Sequence[Any],
    summary: Mapping[str, Any],
) -> tuple[Optional[str], Optional[str]]:
    """Persist activity-calibrated semantic tag rows and summary artifacts."""
    if len(rows) == 0:
        return None, None
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "concept_tags_calibrated.csv"
    json_path = out_dir / "concept_tags_calibration_summary.json"

    data = [
        {
            "concept_id": str(r.concept_id),
            "tag": str(r.tag),
            "provenance": str(r.provenance),
            "base_confidence": float(r.base_confidence),
            "calibrated_confidence": float(r.calibrated_confidence),
            "keep": int(bool(r.keep)),
            "concept_support": int(r.concept_support),
            "tag_support": int(r.tag_support),
            "concept_task_score": float(r.concept_task_score),
            "concept_bitmask_score": float(r.concept_bitmask_score),
            "tag_task_score": float(r.tag_task_score),
            "tag_bitmask_score": float(r.tag_bitmask_score),
            "selected_task": str(r.selected_task),
            "selected_task_ratio": float(r.selected_task_ratio),
            "selected_bitmask": int(r.selected_bitmask),
            "selected_bitmask_ratio": float(r.selected_bitmask_ratio),
        }
        for r in rows
    ]
    pd.DataFrame(data).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(dict(summary), indent=2, sort_keys=False))

    log_event(
        "INFO",
        "explainability.chem_ace.activity_calibration.exports",
        path_csv=str(csv_path),
        path_summary=str(json_path),
        n_rows=int(len(data)),
    )
    return str(csv_path), str(json_path)


def prepare_chem_ace_bundle(
    *,
    config: FinalExplainabilityConfig,
    outdir: Path,
    seed: int,
    df_full: pd.DataFrame,
    id_col: str,
    ids_scope: Sequence[str],
    ids_infer_scope: Optional[Sequence[str]],
    ids_2d_file: Sequence[str],
    X2d_file: np.ndarray,
    starts: np.ndarray,
    counts: np.ndarray,
    id2pos: Mapping[str, int],
    conf_sorted: np.ndarray,
    Xinst_sorted: np.ndarray,
    inst_geom_dim: int = -1,
    inst_qm_dim: int = -1,
    inst_geom_cols: Sequence[str] = (),
    inst_qm_cols: Sequence[str] = (),
) -> Optional[ChemACEBundle]:
    """
    Build Chem-ACE concepts and mappings with leakage-safe two-phase logic.

    Phase A (fit): discover concepts only from `ids_scope` patches.
    Phase B (infer): assign `ids_infer_scope` patches to frozen Phase A centroids.
    """
    if not bool(config.run_chem_ace):
        return None
    log_event(
        "START",
        "explainability.prepare_chem_ace_bundle",
        outdir=str(outdir),
        cpu_workers=int(max(0, config.cpu_workers)),
        n_scope_ids_input=int(len(ids_scope)),
        n_infer_scope_ids_input=int(len(ids_infer_scope or [])),
    )
    dim_2d_raw = int(X2d_file.shape[1]) if np.asarray(X2d_file).ndim == 2 else -1
    dim_inst_raw = int(Xinst_sorted.shape[1]) if np.asarray(Xinst_sorted).ndim == 2 else -1
    dim_2d_used = _resolve_effective_feature_dim(
        requested_dim=int(config.chem_ace_max_2d_dim),
        raw_dim=int(dim_2d_raw),
    )
    dim_3dqm_used = _resolve_effective_feature_dim(
        requested_dim=int(config.chem_ace_max_3dqm_dim),
        raw_dim=int(dim_inst_raw),
    )
    log_event(
        "INFO",
        "explainability.chem_ace.modalities",
        modalities="2d+3d+3d_qm",
        dim_2d_raw=int(dim_2d_raw),
        dim_3d_geom=(int(inst_geom_dim) if int(inst_geom_dim) > 0 else "unknown"),
        dim_3d_qm=(int(inst_qm_dim) if int(inst_qm_dim) > 0 else "unknown"),
        dim_3dqm_merged_raw=int(dim_inst_raw),
        dim_2d_cfg=int(config.chem_ace_max_2d_dim),
        dim_3dqm_cfg=int(config.chem_ace_max_3dqm_dim),
        dim_2d_used=int(dim_2d_used),
        dim_3dqm_used=int(dim_3dqm_used),
        n_geom_descriptor_cols=int(len(inst_geom_cols)),
        n_qm_descriptor_cols=int(len(inst_qm_cols)),
    )

    try:
        require_rdkit()
        from rdkit import Chem
    except OptionalDependencyError as exc:
        raise RuntimeError(
            "Chem-ACE requested but RDKit is unavailable. Install RDKit to enable --run_chem_ace"
        ) from exc

    smiles_col = str(config.curated_smiles_col)
    if smiles_col not in df_full.columns:
        raise ValueError(
            f"Chem-ACE requires column '{smiles_col}' in labels table"
        )

    ace_out_dir = Path(config.chem_ace_output_dir) if config.chem_ace_output_dir else (outdir / "chem_ace")
    ace_out_dir.mkdir(parents=True, exist_ok=True)
    db_uri = str(config.chem_ace_db_uri) if config.chem_ace_db_uri else f"sqlite:///{(ace_out_dir / 'chem_ace.sqlite3').as_posix()}"

    functional_rules_path: Optional[Path] = _prepare_dataset_functional_rules(
        df_full=df_full,
        smiles_col=smiles_col,
        out_dir=(ace_out_dir / "rules_autogen"),
        min_count=10,
        min_prevalence=0.0002,
    )

    local_radii = tuple(
        sorted({int(r) for r in config.chem_ace_local_radii if int(r) >= 0})
    )
    if len(local_radii) == 0:
        local_radii = (1,)
    log_event(
        "INFO",
        "explainability.chem_ace.patch_config",
        local_radii=",".join(str(x) for x in local_radii),
        patch_cap_per_mol=int(config.chem_ace_patch_cap_per_mol),
        target_total_patches=int(config.chem_ace_target_total_patches),
        persist_patch_embeddings=bool(config.chem_ace_persist_patch_embeddings),
    )
    log_event(
        "INFO",
        "explainability.chem_ace.activity_calibration.config",
        enabled=bool(config.run_activity_calibration),
        min_concept_support=int(config.activity_calibration_min_concept_support),
        min_tag_support=int(config.activity_calibration_min_tag_support),
        prior_strength=float(config.activity_calibration_prior_strength),
        min_w=float(config.activity_calibration_min_w),
        task_weight=float(config.activity_calibration_task_weight),
        bitmask_weight=float(config.activity_calibration_bitmask_weight),
        bitmask_min_count=int(config.activity_calibration_bitmask_min_count),
        bitmask_exclude_zero=bool(config.activity_calibration_bitmask_exclude_zero),
        mix_base=float(config.activity_calibration_mix_base),
        keep_threshold=float(config.activity_calibration_keep_threshold),
        ratio_cap=float(config.activity_calibration_ratio_cap),
    )
    log_event(
        "INFO",
        "explainability.chem_ace.advanced_geom_topology.config",
        enabled=bool(config.chem_ace_use_advanced_geom_topology),
        max_patches=int(config.chem_ace_advanced_geom_topology_max_patches),
        min_atoms=int(config.chem_ace_advanced_geom_topology_min_atoms),
        max_torsion_paths=int(config.chem_ace_advanced_geom_topology_max_torsion_paths),
        use_convex_hull=bool(config.chem_ace_advanced_geom_use_convex_hull),
        use_persistent_homology=bool(config.chem_ace_advanced_geom_use_persistent_homology),
        persistence_max_atoms=int(config.chem_ace_advanced_geom_persistence_max_atoms),
    )
    log_event(
        "INFO",
        "explainability.chem_ace.orca.config",
        enabled=bool(config.chem_ace_use_orca_descriptors),
        path=(None if config.chem_ace_orca_descriptors_path is None else str(config.chem_ace_orca_descriptors_path)),
        conf_id_col=str(config.chem_ace_orca_conf_id_col),
        mol_id_col=str(config.chem_ace_orca_mol_id_col),
        n_descriptor_cols=int(len(config.chem_ace_orca_descriptor_cols)),
        min_vectors=int(config.chem_ace_orca_min_vectors_for_tagging),
        z_threshold=float(config.chem_ace_orca_z_threshold),
    )

    ace_cfg = ChemACEConfig(
        run_name="chem_ace_final_pipeline",
        seed=int(seed),
        output_dir=str(ace_out_dir),
        cpu_workers=max(0, int(config.cpu_workers)),
        max_patches_per_molecule=int(config.chem_ace_patch_cap_per_mol),
        target_total_patches=int(config.chem_ace_target_total_patches),
        embedding=EmbeddingConfig(layer_name="feature_fusion_2d3dqm", strategy="masked_input"),
        discovery=ConceptDiscoveryConfig(),
        patch_generation=PatchGenerationConfig(
            local_subgraph=LocalSubgraphPatchConfig(radii=tuple(local_radii)),
            pharm3d=Pharm3DPatchConfig(enabled=True),
        ),
        semantics=SemanticTaggingConfig(
            functional_rules_path=(
                None if functional_rules_path is None else str(functional_rules_path)
            ),
            use_advanced_geom_topology=bool(config.chem_ace_use_advanced_geom_topology),
            advanced_geom_topology_max_patches=int(config.chem_ace_advanced_geom_topology_max_patches),
            advanced_geom_topology_min_atoms=int(config.chem_ace_advanced_geom_topology_min_atoms),
            advanced_geom_topology_max_torsion_paths=int(config.chem_ace_advanced_geom_topology_max_torsion_paths),
            advanced_geom_use_convex_hull=bool(config.chem_ace_advanced_geom_use_convex_hull),
            advanced_geom_use_persistent_homology=bool(config.chem_ace_advanced_geom_use_persistent_homology),
            advanced_geom_persistence_max_atoms=int(config.chem_ace_advanced_geom_persistence_max_atoms),
            use_orca_descriptors=bool(config.chem_ace_use_orca_descriptors),
            orca_descriptors_path=(
                None
                if config.chem_ace_orca_descriptors_path is None
                else str(config.chem_ace_orca_descriptors_path)
            ),
            orca_conf_id_col=str(config.chem_ace_orca_conf_id_col),
            orca_mol_id_col=str(config.chem_ace_orca_mol_id_col),
            orca_descriptor_cols=tuple(str(x) for x in config.chem_ace_orca_descriptor_cols),
            orca_min_vectors_for_tagging=int(config.chem_ace_orca_min_vectors_for_tagging),
            orca_z_threshold=float(config.chem_ace_orca_z_threshold),
        ),
        database=DatabaseConfig(uri=db_uri),
    )
    pipeline = ChemACEPipeline(config=ace_cfg)
    run_id = pipeline.start_run(task_ids=TASK_COLS)
    log_event("INFO", "explainability.chem_ace.run_started", run_id=str(run_id), db_uri=str(db_uri))

    ids_discover = sorted({str(x) for x in ids_scope})
    n_discover_before_limit = int(len(ids_discover))
    max_ids = int(config.chem_ace_max_ids)
    if max_ids > 0:
        ids_discover = ids_discover[:max_ids]
    ids_infer = sorted({str(x) for x in (ids_infer_scope or [])})
    discover_set = set(ids_discover)
    ids_infer = [x for x in ids_infer if x not in discover_set]
    log_event(
        "INFO",
        "explainability.chem_ace.scope_ids",
        n_discover_ids_input=int(n_discover_before_limit),
        n_discover_ids_selected=int(len(ids_discover)),
        n_infer_ids_input=int(len(ids_infer_scope or [])),
        n_infer_ids_selected=int(len(ids_infer)),
        chem_ace_max_ids=int(max_ids),
    )

    smiles_by_id = (
        df_full[[id_col, smiles_col]]
        .dropna(subset=[smiles_col])
        .drop_duplicates(subset=[id_col], keep="first")
        .set_index(id_col)[smiles_col]
        .astype(str)
        .to_dict()
    )

    x2d_by_id = {
        str(i): np.asarray(v, dtype=np.float32)
        for i, v in zip(ids_2d_file, X2d_file)
    }

    ids_for_features = sorted(set(ids_discover).union(ids_infer))
    conf_map, inst_map, inst_mean_map = _build_instance_feature_maps(
        ids=ids_for_features,
        starts=starts,
        counts=counts,
        id2pos=id2pos,
        conf_sorted=conf_sorted,
        Xinst_sorted=Xinst_sorted,
        max_confs_per_id=int(config.chem_ace_max_confs_per_id),
    )

    sdf_conformers_by_conf_id: dict[str, Any] = {}
    if config.chem_ace_conformer_sdf is not None:
        sdf_path = Path(str(config.chem_ace_conformer_sdf))
        if not sdf_path.exists():
            raise FileNotFoundError(f"Chem-ACE conformer SDF does not exist: {sdf_path}")
        with log_step("explainability.chem_ace.load_conformer_sdf", path=str(sdf_path)):
            sdf_conformers_by_conf_id, sdf_stats = _load_sdf_conformers_by_conf_id(
                sdf_path=sdf_path,
                conf_id_prop=str(config.chem_ace_sdf_conf_id_prop),
            )
        log_event(
            "INFO",
            "explainability.chem_ace.sdf_ready",
            path=str(sdf_path),
            loaded=int(sdf_stats["loaded"]),
            duplicates=int(sdf_stats["duplicates"]),
            skipped_none=int(sdf_stats["skipped_none"]),
            skipped_no_conf_id=int(sdf_stats["skipped_no_conf_id"]),
            skipped_no_conformer=int(sdf_stats["skipped_no_conformer"]),
        )

    def _build_molecules_for_ids(
        *,
        target_ids: Sequence[str],
        phase: str,
    ) -> tuple[list[MoleculeSource], dict[str, Any], dict[str, int]]:
        molecules_local: list[MoleculeSource] = []
        mols_by_id_local: dict[str, Any] = {}
        n_requested_confs = 0
        n_found_confs = 0
        n_skipped_missing_confs = 0
        n_skipped_incompatible_confs = 0
        n_mols_with_confs = 0

        for mol_id in target_ids:
            smi = smiles_by_id.get(mol_id)
            if smi is None:
                continue
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                continue
            mol = Chem.AddHs(mol)

            conf_labels = tuple(conf_map.get(mol_id, []))
            n_requested_confs += int(len(conf_labels))
            conf_ids: tuple[str, ...] = ()
            if len(conf_labels) > 0 and len(sdf_conformers_by_conf_id) > 0:
                merged_mol, kept_conf_labels, dropped_incompatible = _merge_sdf_conformers_for_molecule(
                    conf_ids=conf_labels,
                    sdf_conformers_by_conf_id=sdf_conformers_by_conf_id,
                )
                n_skipped_incompatible_confs += int(dropped_incompatible)
                n_found_confs += int(len(kept_conf_labels))
                n_skipped_missing_confs += int(
                    max(0, len(conf_labels) - len(kept_conf_labels) - dropped_incompatible)
                )
                if merged_mol is not None and len(kept_conf_labels) > 0:
                    mol = merged_mol
                conf_ids = tuple(str(x) for x in kept_conf_labels)
                if len(conf_ids) > 0:
                    n_mols_with_confs += 1
            elif len(conf_labels) > 0:
                n_skipped_missing_confs += int(len(conf_labels))

            molecules_local.append(MoleculeSource(mol_id=str(mol_id), mol=mol, conf_ids=conf_ids))
            mols_by_id_local[str(mol_id)] = mol

        stats = {
            "n_molecules": int(len(molecules_local)),
            "n_molecules_with_conformers": int(n_mols_with_confs),
            "requested_conformers": int(n_requested_confs),
            "found_in_sdf": int(n_found_confs),
            "skipped_missing": int(n_skipped_missing_confs),
            "skipped_incompatible": int(n_skipped_incompatible_confs),
        }
        log_event(
            "INFO",
            "explainability.chem_ace.conformers_ready",
            phase=str(phase),
            **stats,
        )
        return molecules_local, mols_by_id_local, stats

    molecules_train, molecules_by_id_train, _train_stats = _build_molecules_for_ids(
        target_ids=ids_discover,
        phase="discover_train",
    )
    if len(molecules_train) == 0:
        raise RuntimeError("Chem-ACE train-scope molecule set is empty; cannot continue")

    molecules_infer, molecules_by_id_infer, _infer_stats = _build_molecules_for_ids(
        target_ids=ids_infer,
        phase="infer_scope",
    )
    molecules_by_id: dict[str, Any] = {}
    molecules_by_id.update(molecules_by_id_train)
    molecules_by_id.update(molecules_by_id_infer)

    progress_extras_discover = {
        "phase": "discover_train",
        "modalities": "2d+3d+3d_qm",
        "dim_2d_used": int(dim_2d_used),
        "dim_3dqm_used": int(dim_3dqm_used),
    }
    if int(inst_geom_dim) > 0:
        progress_extras_discover["dim_3d_geom"] = int(inst_geom_dim)
    if int(inst_qm_dim) > 0:
        progress_extras_discover["dim_3d_qm"] = int(inst_qm_dim)

    with log_step(
        "explainability.chem_ace.generate_patches",
        phase="discover_train",
        n_molecules=int(len(molecules_train)),
    ):
        patches_train = pipeline.generate_patches(
            molecules=molecules_train,
            progress_extras=progress_extras_discover,
        )
    if len(patches_train) == 0:
        raise RuntimeError("Chem-ACE generated zero train patches; cannot continue")
    n_patches_train_2d = int(sum(1 for p in patches_train if p.conf_id is None))
    n_patches_train_3d = int(len(patches_train) - n_patches_train_2d)
    log_event(
        "INFO",
        "explainability.chem_ace.patches_ready",
        phase="discover_train",
        n_patches=int(len(patches_train)),
        n_patches_2d=int(n_patches_train_2d),
        n_patches_3d=int(n_patches_train_3d),
    )

    with log_step("explainability.chem_ace.embed_patches", phase="discover_train"):
        embeddings_train, descriptor_scaler = _build_feature_patch_embeddings(
            pipeline=pipeline,
            patches=patches_train,
            molecules_by_id=molecules_by_id,
            x2d_by_id=x2d_by_id,
            xinst_by_pair=inst_map,
            xinst_mean_by_id=inst_mean_map,
            max_2d_dim=int(dim_2d_used),
            max_3dqm_dim=int(dim_3dqm_used),
            fit_descriptor_scaler=True,
            descriptor_scaler=None,
            persist_embeddings=bool(config.chem_ace_persist_patch_embeddings),
            n_workers=max(0, int(config.cpu_workers)),
        )
    log_event(
        "INFO",
        "explainability.chem_ace.embeddings_ready",
        phase="discover_train",
        n_embeddings=int(len(embeddings_train)),
    )

    with log_step("explainability.chem_ace.discover_concepts", phase="discover_train"):
        concept_set, concept_set_id = pipeline.discover_and_store_concepts(
            run_id=run_id,
            embeddings=embeddings_train,
        )
    with log_step("explainability.chem_ace.tag_concepts", phase="discover_train"):
        tagging = pipeline.tag_and_store_concepts(
            concept_set=concept_set,
            patches=patches_train,
            molecules_by_id=molecules_by_id_train,
            inst_by_pair=inst_map,
            inst_mean_by_id=inst_mean_map,
            inst_geom_dim=int(inst_geom_dim),
            inst_qm_dim=int(inst_qm_dim),
            geom_feature_names=tuple(str(x) for x in inst_geom_cols),
            qm_feature_names=tuple(str(x) for x in inst_qm_cols),
        )

    activity_calibrated_tags_csv: Optional[str] = None
    activity_calibration_summary_json: Optional[str] = None
    concept_mol_map_train, concept_conf_map_train = _build_concept_membership_maps(
        concept_set=concept_set,
        patches=patches_train,
    )
    if bool(config.run_activity_calibration):
        calibration_rows: list[Any] = []
        calibration_summary: dict[str, Any] = {}
        with log_step("explainability.chem_ace.calibrate_semantics", phase="discover_train"):
            ids_calib, y_calib = _extract_binary_labels_for_ids(
                df_full=df_full,
                id_col=id_col,
                ids=ids_discover,
                task_cols=TASK_COLS,
            )
            if len(ids_calib) > 0 and len(tagging) > 0:
                calibrator = ActivityAwareSemanticCalibrator(
                    config=ActivityCalibrationConfig(
                        enabled=True,
                        min_concept_support=int(config.activity_calibration_min_concept_support),
                        min_tag_support=int(config.activity_calibration_min_tag_support),
                        prior_strength=float(config.activity_calibration_prior_strength),
                        min_w=float(config.activity_calibration_min_w),
                        task_weight=float(config.activity_calibration_task_weight),
                        bitmask_weight=float(config.activity_calibration_bitmask_weight),
                        bitmask_min_count=int(config.activity_calibration_bitmask_min_count),
                        bitmask_exclude_zero=bool(config.activity_calibration_bitmask_exclude_zero),
                        mix_base=float(config.activity_calibration_mix_base),
                        keep_threshold=float(config.activity_calibration_keep_threshold),
                        min_confidence=float(config.activity_calibration_min_confidence),
                        max_confidence=float(config.activity_calibration_max_confidence),
                        ratio_cap=float(config.activity_calibration_ratio_cap),
                        fallback_top1_if_empty=bool(config.activity_calibration_fallback_top1_if_empty),
                    ),
                    task_cols=TASK_COLS,
                )
                tagging, calibration_rows, calibration_summary = calibrator.calibrate(
                    tagging_results=tagging,
                    concept_mol_map_train=concept_mol_map_train,
                    ids_train=ids_calib,
                    y_train=y_calib,
                )
            else:
                calibration_summary = {
                    "enabled": True,
                    "reason": "empty_labels_or_tags",
                    "n_ids": int(len(ids_calib)),
                    "n_tagging_results": int(len(tagging)),
                }
                log_event(
                    "WARN",
                    "explainability.chem_ace.calibrate_semantics.skipped",
                    **calibration_summary,
                )

        if len(calibration_rows) > 0:
            # Keep base tags (already persisted) and append calibrated tags with explicit provenance.
            cal_tags = [tag for res in tagging for tag in res.tags]
            if len(cal_tags) > 0:
                with log_step(
                    "explainability.chem_ace.calibrate_semantics.persist_tags",
                    n_tags=int(len(cal_tags)),
                ):
                    pipeline.repository.upsert_tags(cal_tags)
            activity_calibrated_tags_csv, activity_calibration_summary_json = _export_activity_calibrated_tags(
                out_dir=ace_out_dir,
                rows=calibration_rows,
                summary=calibration_summary,
            )

    concept_mol_map_infer: dict[str, set[str]] = {}
    concept_conf_map_infer: dict[str, set[tuple[str, str]]] = {}
    n_patches_infer = 0
    n_embeddings_infer = 0
    n_inferred_memberships = 0

    if len(ids_infer) > 0 and len(concept_set.candidates) > 0:
        progress_extras_infer = {
            "phase": "infer_scope",
            "modalities": "2d+3d+3d_qm",
            "dim_2d_used": int(dim_2d_used),
            "dim_3dqm_used": int(dim_3dqm_used),
        }
        if int(inst_geom_dim) > 0:
            progress_extras_infer["dim_3d_geom"] = int(inst_geom_dim)
        if int(inst_qm_dim) > 0:
            progress_extras_infer["dim_3d_qm"] = int(inst_qm_dim)

        with log_step(
            "explainability.chem_ace.generate_patches",
            phase="infer_scope",
            n_molecules=int(len(molecules_infer)),
        ):
            patches_infer = pipeline.generate_patches(
                molecules=molecules_infer,
                progress_extras=progress_extras_infer,
            )
        n_patches_infer = int(len(patches_infer))
        if n_patches_infer > 0:
            n_patches_infer_2d = int(sum(1 for p in patches_infer if p.conf_id is None))
            n_patches_infer_3d = int(n_patches_infer - n_patches_infer_2d)
            log_event(
                "INFO",
                "explainability.chem_ace.patches_ready",
                phase="infer_scope",
                n_patches=int(n_patches_infer),
                n_patches_2d=int(n_patches_infer_2d),
                n_patches_3d=int(n_patches_infer_3d),
            )

            with log_step("explainability.chem_ace.embed_patches", phase="infer_scope"):
                embeddings_infer, _ = _build_feature_patch_embeddings(
                    pipeline=pipeline,
                    patches=patches_infer,
                    molecules_by_id=molecules_by_id,
                    x2d_by_id=x2d_by_id,
                    xinst_by_pair=inst_map,
                    xinst_mean_by_id=inst_mean_map,
                    max_2d_dim=int(dim_2d_used),
                    max_3dqm_dim=int(dim_3dqm_used),
                    fit_descriptor_scaler=False,
                    descriptor_scaler=descriptor_scaler,
                    persist_embeddings=bool(config.chem_ace_persist_patch_embeddings),
                    n_workers=max(0, int(config.cpu_workers)),
                )
            n_embeddings_infer = int(len(embeddings_infer))
            log_event(
                "INFO",
                "explainability.chem_ace.embeddings_ready",
                phase="infer_scope",
                n_embeddings=int(n_embeddings_infer),
            )

            with log_step("explainability.chem_ace.infer_memberships", phase="infer_scope"):
                inferred_memberships = _infer_memberships_to_frozen_centroids(
                    embeddings=embeddings_infer,
                    candidates=concept_set.candidates,
                    max_distance=float(config.chem_ace_infer_max_distance),
                )
            n_inferred_memberships = int(len(inferred_memberships))
            if n_inferred_memberships > 0 and hasattr(pipeline.repository, "upsert_memberships"):
                with log_step(
                    "explainability.chem_ace.persist_inferred_memberships",
                    n_memberships=int(n_inferred_memberships),
                ):
                    pipeline.repository.upsert_memberships(inferred_memberships)

            concept_mol_map_infer, concept_conf_map_infer = _build_membership_maps_from_memberships(
                memberships=inferred_memberships,
                patches=patches_infer,
            )
            log_event(
                "INFO",
                "explainability.chem_ace.infer_memberships_ready",
                n_memberships=int(n_inferred_memberships),
                n_concepts_hit=int(len(concept_mol_map_infer)),
                max_distance=float(config.chem_ace_infer_max_distance),
            )
        else:
            log_event(
                "WARN",
                "explainability.chem_ace.infer_scope.no_patches",
                n_molecules=int(len(molecules_infer)),
            )

    concept_mol_map = _merge_membership_maps(
        base=concept_mol_map_train,
        extra=concept_mol_map_infer,
    )
    concept_conf_map = _merge_membership_maps(
        base=concept_conf_map_train,
        extra=concept_conf_map_infer,
    )

    support_map = {str(c.concept_local_id): int(c.support) for c in concept_set.candidates}

    def _infer_modality(*, provenance: str, tag: str) -> str:
        p = str(provenance).strip().lower()
        t = str(tag).strip().lower()
        if ("qm" in p) or ("quantum" in p) or ("orca" in p):
            return "quantum"
        if ("geom" in p) or ("geometry" in p) or ("3d" in p):
            return "geometry"
        qm_tokens = (
            "homo",
            "lumo",
            "gap",
            "electrophil",
            "nucleophil",
            "dipole",
            "polariz",
            "electrostatic",
            "fukui",
            "charge-transfer",
            "frontier",
            "quantum",
            "excited-state",
            "oscillator",
            "singlet-triplet",
            "spin-orbit",
            "radiative",
            "nonradiative",
        )
        geom_tokens = (
            "planar",
            "non-planar",
            "twisted",
            "rigid",
            "flexible",
            "geometry",
            "torsion",
            "ring-strained",
            "shape",
            "surface/volume",
        )
        if any(tok in t for tok in qm_tokens):
            return "quantum"
        if any(tok in t for tok in geom_tokens):
            return "geometry"
        return "2d"

    tag_map: dict[str, dict[str, Any]] = {}
    for t in tagging:
        cid = str(t.concept_id)
        modality_scores = {"2d": 0.0, "geometry": 0.0, "quantum": 0.0}
        tags_by_modality = {"2d": [], "geometry": [], "quantum": []}
        tag_details: list[dict[str, Any]] = []
        for tag_obj in t.tags:
            tag_name = str(tag_obj.tag)
            provenance = str(tag_obj.provenance)
            confidence = float(getattr(tag_obj, "confidence", 0.0))
            modality = _infer_modality(provenance=provenance, tag=tag_name)
            modality_scores[modality] += max(0.0, confidence)
            tags_by_modality[modality].append(tag_name)
            tag_details.append(
                {
                    "tag": tag_name,
                    "confidence": float(confidence),
                    "provenance": provenance,
                    "modality": modality,
                }
            )

        total_score = float(sum(modality_scores.values()))
        if total_score <= 1e-12:
            modality_scores["2d"] = 1.0
            total_score = 1.0
        modality_weights = {
            k: float(max(0.0, v) / total_score)
            for k, v in modality_scores.items()
        }
        dominant_modality = max(modality_weights, key=lambda k: modality_weights[k])
        tags_all = [str(x.tag) for x in t.tags]
        tag_map[cid] = {
            "label_auto": str(t.label_auto),
            "tags": sorted(set(tags_all)),
            "tag_details": tag_details,
            "tags_by_modality": {
                k: sorted(set([str(x) for x in vals if str(x).strip()]))
                for k, vals in tags_by_modality.items()
            },
            "modality_weights": modality_weights,
            "dominant_modality": str(dominant_modality),
        }

    concept_metadata: dict[str, dict[str, Any]] = {}
    for cid, support in support_map.items():
        mols_train = concept_mol_map_train.get(cid, set())
        confs_train = concept_conf_map_train.get(cid, set())
        mols_total = concept_mol_map.get(cid, set())
        confs_total = concept_conf_map.get(cid, set())
        concept_metadata[cid] = {
            "support": int(support),
            "label_auto": tag_map.get(cid, {}).get("label_auto"),
            "tags": tag_map.get(cid, {}).get("tags", []),
            "tag_details": tag_map.get(cid, {}).get("tag_details", []),
            "tags_by_modality": tag_map.get(cid, {}).get("tags_by_modality", {}),
            "modality_weights": tag_map.get(cid, {}).get(
                "modality_weights",
                {"2d": 1.0, "geometry": 0.0, "quantum": 0.0},
            ),
            "dominant_modality": tag_map.get(cid, {}).get("dominant_modality", "2d"),
            "n_molecules_train": int(len(mols_train)),
            "n_conf_pairs_train": int(len(confs_train)),
            "n_molecules_total": int(len(mols_total)),
            "n_conf_pairs_total": int(len(confs_total)),
        }

    a_priori_paths = _export_a_priori_and_concept_views(
        out_dir=ace_out_dir,
        ids_discover=ids_discover,
        ids_infer=ids_infer,
        smiles_by_id=smiles_by_id,
        molecules_by_id=molecules_by_id,
        semantic_tagger=pipeline.semantic_tagger,
        concept_mol_map=concept_mol_map,
        concept_metadata=concept_metadata,
    )

    ordered_concepts = sorted(
        support_map.keys(),
        key=lambda cid: support_map[cid],
        reverse=True,
    )
    top_k = int(config.chem_ace_top_concepts)
    if top_k > 0:
        ordered_concepts = ordered_concepts[:top_k]

    summary = {
        "run_id": str(run_id),
        "concept_set_id": str(concept_set_id),
        "n_molecules_discover": int(len(molecules_train)),
        "n_molecules_infer": int(len(molecules_infer)),
        "n_patches_discover": int(len(patches_train)),
        "n_patches_infer": int(n_patches_infer),
        "n_embeddings_discover": int(len(embeddings_train)),
        "n_embeddings_infer": int(n_embeddings_infer),
        "n_inferred_memberships": int(n_inferred_memberships),
        "n_concepts": int(len(concept_set.candidates)),
        "selected_concepts": ordered_concepts,
        "a_priori_tags_csv": a_priori_paths.get("a_priori_tags_csv"),
        "a_priori_vs_concepts_csv": a_priori_paths.get("a_priori_vs_concepts_csv"),
        "a_priori_tags_infer_csv": a_priori_paths.get("a_priori_tags_infer_csv"),
        "a_priori_vs_concepts_infer_csv": a_priori_paths.get("a_priori_vs_concepts_infer_csv"),
        "activity_calibrated_tags_csv": activity_calibrated_tags_csv,
        "activity_calibration_summary_json": activity_calibration_summary_json,
    }
    (ace_out_dir / "chem_ace_pipeline_summary.json").write_text(json.dumps(summary, indent=2))

    logger.info(
        "Prepared Chem-ACE bundle",
        extra={
            "run_id": run_id,
            "concept_set_id": concept_set_id,
            "n_concepts": len(ordered_concepts),
            "n_patches": len(patches_train),
        },
    )
    log_event(
        "DONE",
        "explainability.prepare_chem_ace_bundle",
        run_id=str(run_id),
        concept_set_id=str(concept_set_id),
        n_concepts=int(len(ordered_concepts)),
    )

    return ChemACEBundle(
        output_dir=str(ace_out_dir),
        db_uri=db_uri,
        run_id=str(run_id),
        concept_set_id=str(concept_set_id),
        concept_ids=tuple(ordered_concepts),
        concept_metadata=concept_metadata,
        concept_support=support_map,
        concept_mol_map=concept_mol_map,
        concept_conf_map=concept_conf_map,
        a_priori_tags_csv=(
            None
            if a_priori_paths.get("a_priori_tags_csv") is None
            else str(a_priori_paths["a_priori_tags_csv"])
        ),
        a_priori_vs_concepts_csv=(
            None
            if a_priori_paths.get("a_priori_vs_concepts_csv") is None
            else str(a_priori_paths["a_priori_vs_concepts_csv"])
        ),
        a_priori_tags_infer_csv=(
            None
            if a_priori_paths.get("a_priori_tags_infer_csv") is None
            else str(a_priori_paths["a_priori_tags_infer_csv"])
        ),
        a_priori_vs_concepts_infer_csv=(
            None
            if a_priori_paths.get("a_priori_vs_concepts_infer_csv") is None
            else str(a_priori_paths["a_priori_vs_concepts_infer_csv"])
        ),
        activity_calibrated_tags_csv=activity_calibrated_tags_csv,
        activity_calibration_summary_json=activity_calibration_summary_json,
    )



def build_lambda_vol_callback(
    *,
    config: FinalExplainabilityConfig,
    outdir: Path,
    seed: int,
    monitor_loader: DataLoader,
    chem_bundle: ChemACEBundle,
) -> Optional[LambdaVolLightningCallback]:
    """Create Lambda-Vol Lightning callback bound to MIL monitoring providers."""
    if not bool(config.run_lambda_vol):
        return None
    log_event("START", "explainability.build_lambda_vol_callback")
    if not chem_bundle.concept_ids:
        logger.warning("Lambda-Vol requested but no Chem-ACE concepts are available")
        log_event("WARN", "explainability.build_lambda_vol_callback.no_concepts")
        return None

    top_k = int(config.lambda_vol_top_concepts)
    concept_ids = list(chem_bundle.concept_ids)
    if top_k > 0:
        concept_ids = concept_ids[:top_k]

    concept_metadata = {
        cid: dict(chem_bundle.concept_metadata.get(cid, {}))
        for cid in concept_ids
    }

    lv_out = Path(config.lambda_vol_output_dir) if config.lambda_vol_output_dir else (outdir / "lambda_vol")
    lv_out.mkdir(parents=True, exist_ok=True)
    lv_db_uri = str(config.lambda_vol_db_uri) if config.lambda_vol_db_uri else f"sqlite:///{(lv_out / 'lambda_vol.sqlite3').as_posix()}"

    lv_cfg = LambdaVolConfig(
        run_name="lambda_vol_final_pipeline",
        seed=int(seed),
        tracker=TrackerConfig(alpha=0.6, tcav_ema_beta=0.8, drift_clip=5.0),
        ricci=RicciConfig(
            enabled=bool(config.lambda_vol_run_ricci),
            edge_keep_quantile=float(config.lambda_vol_ricci_edge_keep_quantile),
            min_edge_weight=float(config.lambda_vol_ricci_min_edge_weight),
            top_k_per_node=int(config.lambda_vol_ricci_top_k_per_node),
            flow_steps=int(config.lambda_vol_ricci_flow_steps),
            flow_step_size=float(config.lambda_vol_ricci_flow_step_size),
            use_flow_as_concept_coupling=bool(config.lambda_vol_ricci_use_flow_as_coupling),
            coupling_strength=float(config.lambda_vol_ricci_coupling_strength),
        ),
        policy=PolicyConfig(enabled=True, auto_action=False),
        exporter=LambdaVolExportConfig(
            output_dir=str(lv_out),
            export_parquet=True,
            export_plotly_html=True,
            export_vtk=False,
            top_k_lattice=min(20, len(concept_ids)),
        ),
        store=StoreConfig(db_uri=lv_db_uri),
    )

    monitor = LambdaVolMonitor(
        config=lv_cfg,
        task_ids=tuple(TASK_COLS),
        concept_ids=tuple(concept_ids),
        concept_metadata=concept_metadata,
    )
    frame_provider = MILLambdaVolFrameProvider(
        monitor_loader=monitor_loader,
        concept_ids=tuple(concept_ids),
        concept_mol_map={cid: chem_bundle.concept_mol_map.get(cid, set()) for cid in concept_ids},
        concept_conf_map={cid: chem_bundle.concept_conf_map.get(cid, set()) for cid in concept_ids},
        task_ids=tuple(TASK_COLS),
        layer_name=str(config.lambda_vol_layer_name),
        monitor_max_samples=int(config.lambda_vol_monitor_max_samples),
        tcav_repeats=int(config.lambda_vol_tcav_repeats),
        tcav_random_counterexamples=int(config.lambda_vol_random_counterexamples),
        tcav_min_concept_samples=int(config.lambda_vol_min_concept_samples),
        seed=int(seed),
    )

    cb = LambdaVolLightningCallback(
        monitor=monitor,
        frame_provider=frame_provider,
        export_on_fit_end=True,
    )
    log_event(
        "DONE",
        "explainability.build_lambda_vol_callback",
        n_concepts=int(len(concept_ids)),
        outdir=str(lv_out),
    )
    return cb



def make_monitor_loader(
    *,
    ids: Sequence[str],
    x2d: np.ndarray,
    starts: np.ndarray,
    counts: np.ndarray,
    id2pos: Mapping[str, int],
    xinst_sorted: np.ndarray,
    conf_sorted: np.ndarray,
    batch_size: int,
    seed: int,
    loader_cfg: LoaderConfig,
) -> DataLoader:
    log_event(
        "START",
        "explainability.make_monitor_loader",
        n_ids=int(len(ids)),
        batch_size=int(batch_size),
    )
    ds = MILExportDataset(
        ids=[str(x) for x in ids],
        X2d=np.asarray(x2d, dtype=np.float32),
        starts=starts,
        counts=counts,
        id2pos=dict(id2pos),
        Xinst_sorted=np.asarray(xinst_sorted, dtype=np.float32),
        conf_sorted=np.asarray(conf_sorted),
        max_instances=0,
        seed=int(seed),
    )
    dl = DataLoaderBuilder(loader_cfg).eval_loader(
        ds,
        batch_size=int(max(1, batch_size)),
        collate_fn=collate_export,
    )
    log_event("DONE", "explainability.make_monitor_loader")
    return dl



def _build_instance_feature_maps(
    *,
    ids: Sequence[str],
    starts: np.ndarray,
    counts: np.ndarray,
    id2pos: Mapping[str, int],
    conf_sorted: np.ndarray,
    Xinst_sorted: np.ndarray,
    max_confs_per_id: int,
) -> tuple[dict[str, list[str]], dict[tuple[str, str], np.ndarray], dict[str, np.ndarray]]:
    conf_map: dict[str, list[str]] = {}
    inst_map: dict[tuple[str, str], np.ndarray] = {}
    inst_mean_map: dict[str, np.ndarray] = {}

    for mol_id in ids:
        pos = id2pos.get(str(mol_id))
        if pos is None:
            continue
        s = int(starts[int(pos)])
        c = int(counts[int(pos)])
        if c <= 0:
            continue

        confs = [str(x) for x in conf_sorted[s : s + c].tolist()]
        feats = np.asarray(Xinst_sorted[s : s + c], dtype=np.float32)

        if max_confs_per_id > 0:
            confs = confs[:max_confs_per_id]
            feats = feats[:max_confs_per_id]

        conf_map[str(mol_id)] = confs
        for conf_id, vec in zip(confs, feats):
            inst_map[(str(mol_id), str(conf_id))] = np.asarray(vec, dtype=np.float32)
        inst_mean_map[str(mol_id)] = np.asarray(feats.mean(axis=0), dtype=np.float32)

    return conf_map, inst_map, inst_mean_map


def _load_sdf_conformers_by_conf_id(
    *,
    sdf_path: Path,
    conf_id_prop: str,
) -> tuple[dict[str, Any], dict[str, int]]:
    from rdkit import Chem

    conf_to_mol: dict[str, Any] = {}
    stats = {
        "loaded": 0,
        "duplicates": 0,
        "skipped_none": 0,
        "skipped_no_conf_id": 0,
        "skipped_no_conformer": 0,
    }

    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    for mol in supplier:
        if mol is None:
            stats["skipped_none"] += 1
            continue
        if mol.GetNumConformers() <= 0:
            stats["skipped_no_conformer"] += 1
            continue

        conf_id: Optional[str] = None
        if conf_id_prop and mol.HasProp(str(conf_id_prop)):
            val = str(mol.GetProp(str(conf_id_prop))).strip()
            if val:
                conf_id = val
        if conf_id is None and mol.HasProp("_Name"):
            val = str(mol.GetProp("_Name")).strip()
            if val:
                conf_id = val

        if conf_id is None:
            stats["skipped_no_conf_id"] += 1
            continue
        if conf_id in conf_to_mol:
            stats["duplicates"] += 1
            continue

        conf_to_mol[conf_id] = mol
        stats["loaded"] += 1

    return conf_to_mol, stats


def _merge_sdf_conformers_for_molecule(
    *,
    conf_ids: Sequence[str],
    sdf_conformers_by_conf_id: Mapping[str, Any],
) -> tuple[Optional[Any], list[str], int]:
    from rdkit import Chem

    base_mol: Optional[Any] = None
    for conf_id in conf_ids:
        mol = sdf_conformers_by_conf_id.get(str(conf_id))
        if mol is not None and mol.GetNumConformers() > 0:
            base_mol = Chem.Mol(mol)
            break
    if base_mol is None:
        return None, [], 0

    base_mol.RemoveAllConformers()
    kept_conf_ids: list[str] = []
    conf_id_map: dict[str, int] = {}
    dropped_incompatible = 0
    n_atoms = int(base_mol.GetNumAtoms())

    for conf_id in conf_ids:
        mol = sdf_conformers_by_conf_id.get(str(conf_id))
        if mol is None or mol.GetNumConformers() <= 0:
            continue
        if int(mol.GetNumAtoms()) != n_atoms:
            dropped_incompatible += 1
            continue
        conf = Chem.Conformer(mol.GetConformer(0))
        new_idx = int(base_mol.AddConformer(conf, assignId=True))
        key = str(conf_id)
        kept_conf_ids.append(key)
        conf_id_map[key] = int(new_idx)

    if len(kept_conf_ids) == 0:
        return None, [], dropped_incompatible

    base_mol.SetProp("_chemace_conf_id_map", json.dumps(conf_id_map, sort_keys=True))
    return base_mol, kept_conf_ids, dropped_incompatible


def _build_feature_patch_embeddings(
    *,
    pipeline: ChemACEPipeline,
    patches: Sequence[PatchRecord],
    molecules_by_id: Mapping[str, Any],
    x2d_by_id: Mapping[str, np.ndarray],
    xinst_by_pair: Mapping[tuple[str, str], np.ndarray],
    xinst_mean_by_id: Mapping[str, np.ndarray],
    max_2d_dim: int,
    max_3dqm_dim: int,
    fit_descriptor_scaler: bool = False,
    descriptor_scaler: Any | None = None,
    persist_embeddings: bool = False,
    n_workers: int = 0,
) -> tuple[list[PatchEmbeddingRecord], Any | None]:
    emb_recs: list[PatchEmbeddingRecord] = []

    workers = max(0, int(n_workers))
    total_patches = int(len(patches))
    progress_every = 200000

    scaler_obj = descriptor_scaler
    if bool(fit_descriptor_scaler) and scaler_obj is None:
        scaler_obj = _fit_patch_descriptor_robust_scaler(
            patches=patches,
            molecules_by_id=molecules_by_id,
            x2d_by_id=x2d_by_id,
            max_fit_samples=200000,
        )

    def _compute_patch_vec(patch: PatchRecord) -> tuple[PatchRecord, np.ndarray, dict[str, int]] | None:
        mol_id = str(patch.mol_id)
        x2d = x2d_by_id.get(mol_id)
        if x2d is None:
            return None

        x3d = None
        if patch.conf_id is not None:
            x3d = xinst_by_pair.get((mol_id, str(patch.conf_id)))
        if x3d is None:
            x3d = xinst_mean_by_id.get(mol_id)
        if x3d is None:
            x3d = np.zeros((max(1, int(max_3dqm_dim)),), dtype=np.float32)

        mol = molecules_by_id.get(mol_id)
        desc_raw = _patch_descriptors(mol=mol, atom_indices=patch.atom_indices, patch_type=patch.patch_type)
        desc = _transform_patch_descriptors(desc_raw=desc_raw, scaler_obj=scaler_obj)

        v2d = _take_or_pad(np.asarray(x2d, dtype=np.float32), int(max_2d_dim))
        v3d = _take_or_pad(np.asarray(x3d, dtype=np.float32), int(max_3dqm_dim))
        vec = np.concatenate([v2d, v3d, desc], axis=0).astype(np.float32)
        md = {
            "d2": int(v2d.shape[0]),
            "d3qm": int(v3d.shape[0]),
            "ddesc": int(desc.shape[0]),
        }
        return patch, vec, md

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            computed_iter = ex.map(_compute_patch_vec, patches)
            for i, item in enumerate(computed_iter, start=1):
                if i == total_patches or (progress_every > 0 and (i % progress_every == 0)):
                    log_event(
                        "PROGRESS",
                        "explainability.chem_ace.embed_patches.compute",
                        done=f"{int(i)}/{int(total_patches)}",
                        pct=f"{(100.0 * i / float(max(1, total_patches))):.1f}",
                        cpu_workers=int(workers),
                    )
                if item is None:
                    continue
                patch, vec, md = item
                rec_md = {
                    "strategy": "feature_projection",
                    "d2": int(md["d2"]),
                    "d3qm": int(md["d3qm"]),
                    "ddesc": int(md["ddesc"]),
                }
                if bool(persist_embeddings):
                    rec = pipeline.embedding_cache.save(
                        patch=patch,
                        layer_name="feature_fusion_2d3dqm",
                        strategy="feature_projection",
                        vector=vec,
                        metadata=rec_md,
                    )
                else:
                    rec = PatchEmbeddingRecord(
                        patch_id=str(patch.patch_id),
                        layer_name="feature_fusion_2d3dqm",
                        strategy="feature_projection",
                        vector=np.asarray(vec, dtype=np.float32),
                        embedding_uri=None,
                        metadata=rec_md,
                    )
                emb_recs.append(rec)
    else:
        for i, patch in enumerate(patches, start=1):
            item = _compute_patch_vec(patch)
            if i == total_patches or (progress_every > 0 and (i % progress_every == 0)):
                log_event(
                    "PROGRESS",
                    "explainability.chem_ace.embed_patches.compute",
                    done=f"{int(i)}/{int(total_patches)}",
                    pct=f"{(100.0 * i / float(max(1, total_patches))):.1f}",
                    cpu_workers=int(workers),
                )
            if item is None:
                continue
            patch_rec, vec, md = item
            rec_md = {
                "strategy": "feature_projection",
                "d2": int(md["d2"]),
                "d3qm": int(md["d3qm"]),
                "ddesc": int(md["ddesc"]),
            }
            if bool(persist_embeddings):
                rec = pipeline.embedding_cache.save(
                    patch=patch_rec,
                    layer_name="feature_fusion_2d3dqm",
                    strategy="feature_projection",
                    vector=vec,
                    metadata=rec_md,
                )
            else:
                rec = PatchEmbeddingRecord(
                    patch_id=str(patch_rec.patch_id),
                    layer_name="feature_fusion_2d3dqm",
                    strategy="feature_projection",
                    vector=np.asarray(vec, dtype=np.float32),
                    embedding_uri=None,
                    metadata=rec_md,
                )
            emb_recs.append(rec)

    if bool(persist_embeddings):
        log_event(
            "START",
            "explainability.chem_ace.persist_patch_embeddings",
            n_embeddings=int(len(emb_recs)),
        )
        if hasattr(pipeline.repository, "upsert_patch_embeddings"):
            pipeline.repository.upsert_patch_embeddings(emb_recs)
        else:
            for rec in emb_recs:
                pipeline.repository.upsert_patch_embedding(rec)
        log_event(
            "DONE",
            "explainability.chem_ace.persist_patch_embeddings",
            n_embeddings=int(len(emb_recs)),
        )
    else:
        log_event(
            "INFO",
            "explainability.chem_ace.persist_patch_embeddings.skipped",
            reason="disabled",
            n_embeddings=int(len(emb_recs)),
        )

    if not emb_recs:
        raise RuntimeError("Chem-ACE feature projection produced zero patch embeddings")
    return emb_recs, scaler_obj



def _patch_descriptors(*, mol: Any, atom_indices: Sequence[int], patch_type: str) -> np.ndarray:
    if mol is None or len(atom_indices) == 0:
        base = np.zeros((10,), dtype=np.float32)
        return np.concatenate([base, _patch_type_one_hot(str(patch_type))], axis=0)

    atom_ids = [int(i) for i in atom_indices if 0 <= int(i) < int(mol.GetNumAtoms())]
    if not atom_ids:
        base = np.zeros((10,), dtype=np.float32)
        return np.concatenate([base, _patch_type_one_hot(str(patch_type))], axis=0)

    atoms = [mol.GetAtomWithIdx(i) for i in atom_ids]
    n = float(len(atom_ids))
    aromatic = float(sum(1 for a in atoms if a.GetIsAromatic())) / n
    hetero = float(sum(1 for a in atoms if a.GetAtomicNum() not in {1, 6})) / n
    charge_sum = float(sum(int(a.GetFormalCharge()) for a in atoms))

    atom_set = set(atom_ids)
    patch_bonds = []
    for b in mol.GetBonds():
        i = int(b.GetBeginAtomIdx())
        j = int(b.GetEndAtomIdx())
        if i in atom_set and j in atom_set:
            patch_bonds.append(b)

    n_bonds = max(1.0, float(len(patch_bonds)))
    conj_frac = float(sum(1 for b in patch_bonds if b.GetIsConjugated())) / n_bonds
    ring_frac = float(sum(1 for b in patch_bonds if b.IsInRing())) / n_bonds

    atomic_nums = np.asarray([float(a.GetAtomicNum()) for a in atoms], dtype=np.float32)
    deg = np.asarray([float(a.GetDegree()) for a in atoms], dtype=np.float32)

    base = np.asarray(
        [
            float(len(atom_ids)),
            aromatic,
            hetero,
            charge_sum,
            conj_frac,
            ring_frac,
            float(np.mean(atomic_nums)),
            float(np.std(atomic_nums)),
            float(np.mean(deg)),
            float(np.std(deg)),
        ],
        dtype=np.float32,
    )
    return np.concatenate([base, _patch_type_one_hot(str(patch_type))], axis=0)



def _patch_type_one_hot(patch_type: str) -> np.ndarray:
    keys = [
        "local_subgraph",
        "brics",
        "murcko",
        "murcko_framework",
        "pharm3d",
    ]
    out = np.zeros((len(keys),), dtype=np.float32)
    p = str(patch_type)
    for i, k in enumerate(keys):
        if p == k:
            out[i] = 1.0
    return out



def _take_or_pad(vec: np.ndarray, dim: int) -> np.ndarray:
    d = int(max(1, dim))
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    if v.shape[0] >= d:
        return v[:d]
    out = np.zeros((d,), dtype=np.float32)
    out[: v.shape[0]] = v
    return out



def _fit_patch_descriptor_robust_scaler(
    *,
    patches: Sequence[PatchRecord],
    molecules_by_id: Mapping[str, Any],
    x2d_by_id: Mapping[str, np.ndarray],
    max_fit_samples: int = 200000,
) -> Any | None:
    """Fit RobustScaler on train-scope patch descriptors using a deterministic prefix sample."""
    limit = int(max(0, max_fit_samples))
    if limit <= 0 or len(patches) == 0:
        return None

    try:
        from sklearn.preprocessing import RobustScaler
    except Exception as exc:
        log_event(
            "WARN",
            "explainability.chem_ace.descriptor_scaler.unavailable",
            reason="sklearn_import_failed",
            error=str(exc),
        )
        return None

    desc_rows: list[np.ndarray] = []
    for patch in patches:
        if len(desc_rows) >= limit:
            break
        mol_id = str(patch.mol_id)
        if mol_id not in x2d_by_id:
            continue
        mol = molecules_by_id.get(mol_id)
        desc = _patch_descriptors(mol=mol, atom_indices=patch.atom_indices, patch_type=patch.patch_type)
        desc_rows.append(np.asarray(desc, dtype=np.float32))

    if len(desc_rows) < 8:
        log_event(
            "WARN",
            "explainability.chem_ace.descriptor_scaler.skipped",
            reason="insufficient_samples",
            n_samples=int(len(desc_rows)),
        )
        return None

    mat = np.stack(desc_rows, axis=0)
    scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))
    scaler.fit(mat)
    log_event(
        "INFO",
        "explainability.chem_ace.descriptor_scaler.fitted",
        method="RobustScaler",
        fit_samples=int(mat.shape[0]),
        descriptor_dim=int(mat.shape[1]),
    )
    return scaler


def _transform_patch_descriptors(*, desc_raw: np.ndarray, scaler_obj: Any | None) -> np.ndarray:
    """Apply fitted RobustScaler to descriptor block when available."""
    desc = np.asarray(desc_raw, dtype=np.float32).reshape(1, -1)
    if scaler_obj is None:
        return desc.reshape(-1)
    try:
        out = scaler_obj.transform(desc)
        return np.asarray(out, dtype=np.float32).reshape(-1)
    except Exception as exc:
        log_event(
            "WARN",
            "explainability.chem_ace.descriptor_scaler.transform_failed",
            error=str(exc),
        )
        return desc.reshape(-1)


def _resolve_effective_feature_dim(*, requested_dim: int, raw_dim: int) -> int:
    """
    Resolve effective feature width for Chem-ACE feature projection.

    requested_dim > 0: use requested cap.
    requested_dim <= 0: auto-use full available raw width.
    """
    req = int(requested_dim)
    raw = int(raw_dim)
    if req > 0:
        return req
    if raw > 0:
        return raw
    return 1


def _infer_memberships_to_frozen_centroids(
    *,
    embeddings: Sequence[PatchEmbeddingRecord],
    candidates: Sequence[ConceptCandidate],
    max_distance: float,
    chunk_size: int = 4096,
) -> list[ConceptMembership]:
    if len(embeddings) == 0 or len(candidates) == 0:
        return []

    centroids = np.stack([np.asarray(c.centroid, dtype=np.float32) for c in candidates], axis=0)
    concept_ids = [str(c.concept_local_id) for c in candidates]
    c_sq = np.sum(np.square(centroids), axis=1).reshape(1, -1)

    out: list[ConceptMembership] = []
    n = int(len(embeddings))
    step = int(max(1, chunk_size))
    progress_every = 200000

    for start in range(0, n, step):
        end = min(n, start + step)
        batch = embeddings[start:end]
        x = np.stack([np.asarray(e.vector, dtype=np.float32) for e in batch], axis=0)
        x_sq = np.sum(np.square(x), axis=1, keepdims=True)
        d2 = x_sq + c_sq - 2.0 * np.matmul(x, centroids.T)
        np.maximum(d2, 0.0, out=d2)

        nn_idx = np.argmin(d2, axis=1)
        row_idx = np.arange(d2.shape[0])
        min_dist = np.sqrt(d2[row_idx, nn_idx]).astype(np.float32)

        for i, emb in enumerate(batch):
            dist = float(min_dist[i])
            if float(max_distance) > 0.0 and dist > float(max_distance):
                continue
            cid = concept_ids[int(nn_idx[i])]
            out.append(
                ConceptMembership(
                    concept_local_id=str(cid),
                    patch_id=str(emb.patch_id),
                    membership_score=float(1.0 / (1.0 + dist)),
                    distance_to_centroid=dist,
                )
            )

        done = int(end)
        if done == n or (progress_every > 0 and (done % progress_every == 0)):
            log_event(
                "PROGRESS",
                "explainability.chem_ace.infer_memberships.assign",
                done=f"{done}/{n}",
                pct=f"{(100.0 * done / float(max(1, n))):.1f}",
                assigned=int(len(out)),
            )

    return out



def _build_membership_maps_from_memberships(
    *,
    memberships: Sequence[ConceptMembership],
    patches: Sequence[PatchRecord],
) -> tuple[dict[str, set[str]], dict[str, set[tuple[str, str]]]]:
    patch_by_id = {str(p.patch_id): p for p in patches}
    concept_mols: dict[str, set[str]] = {}
    concept_confs: dict[str, set[tuple[str, str]]] = {}

    for m in memberships:
        cid = str(m.concept_local_id)
        patch = patch_by_id.get(str(m.patch_id))
        if patch is None:
            continue
        concept_mols.setdefault(cid, set()).add(str(patch.mol_id))
        if patch.conf_id is not None:
            concept_confs.setdefault(cid, set()).add((str(patch.mol_id), str(patch.conf_id)))

    return concept_mols, concept_confs



def _build_concept_membership_maps(
    *,
    concept_set: Any,
    patches: Sequence[PatchRecord],
) -> tuple[dict[str, set[str]], dict[str, set[tuple[str, str]]]]:
    return _build_membership_maps_from_memberships(
        memberships=list(concept_set.memberships),
        patches=patches,
    )



def _merge_membership_maps(
    *,
    base: Mapping[str, set[Any]],
    extra: Mapping[str, set[Any]],
) -> dict[str, set[Any]]:
    out: dict[str, set[Any]] = {str(k): set(v) for k, v in base.items()}
    for k, vals in extra.items():
        key = str(k)
        out.setdefault(key, set()).update(set(vals))
    return out



def _build_concept_attention_df(
    *,
    task_ids: Sequence[str],
    concept_ids: Sequence[str],
    attention_support: np.ndarray,
    prevalence: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ti, task_id in enumerate(task_ids):
        for ci, concept_id in enumerate(concept_ids):
            rows.append(
                {
                    "task_id": str(task_id),
                    "concept_id": str(concept_id),
                    "attention_support": float(attention_support[ti, ci]),
                    "prevalence": float(prevalence[ti, ci]),
                }
            )
    return pd.DataFrame(rows)



def _build_tcav_df(
    *,
    task_ids: Sequence[str],
    concept_ids: Sequence[str],
    tcav_matrix: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ti, task_id in enumerate(task_ids):
        for ci, concept_id in enumerate(concept_ids):
            rows.append(
                {
                    "task_id": str(task_id),
                    "concept_id": str(concept_id),
                    "tcav": float(tcav_matrix[ti, ci]),
                }
            )
    return pd.DataFrame(rows)



def _build_task_metric_frames(
    *,
    trainer: Any,
    task_ids: Sequence[str],
    attention_entropy: np.ndarray,
    witness_rate: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, float]]:
    cbm = getattr(trainer, "callback_metrics", {})

    def _m(name: str, default: float = 0.0) -> float:
        v = cbm.get(name, default)
        if torch.is_tensor(v):
            return float(v.detach().cpu().item())
        try:
            return float(v)
        except Exception:
            return float(default)

    train_loss = _m("train_loss", 0.0)
    val_macro = _m("val_macro_ap", 0.0)
    val_min = _m("val_min_ap", 0.0)

    task_rows: list[dict[str, Any]] = []
    for ti, task_id in enumerate(task_ids):
        val_t = _m(f"val_ap_{ti}", val_macro)
        train_metric = 1.0 / (1.0 + max(0.0, train_loss))
        task_rows.append(
            {
                "task_id": str(task_id),
                "train_metric": float(train_metric),
                "val_metric": float(val_t),
                "loss": float(train_loss),
                "calibration_error": 0.0,
            }
        )

    context = {
        "val_macro_ap": float(val_macro),
        "val_min_ap": float(val_min),
        "train_loss": float(train_loss),
        "attention_entropy_macro": float(np.mean(attention_entropy)) if attention_entropy.size else 0.0,
        "witness_rate_macro": float(np.mean(witness_rate)) if witness_rate.size else 0.0,
    }
    return pd.DataFrame(task_rows), context



def _collapse_activation(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().cpu().float().numpy()
    if arr.ndim == 1:
        return arr.reshape(1, -1).astype(np.float32)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim == 3:
        # [B,*,D] -> [B,D]
        b = int(arr.shape[0])
        return arr.reshape(b, -1, arr.shape[-1]).mean(axis=1).astype(np.float32)
    b = int(arr.shape[0])
    return arr.reshape(b, -1).astype(np.float32)



def _normalized_entropy(prob: np.ndarray) -> float:
    p = np.asarray(prob, dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0)
    h = -float(np.sum(p * np.log(p)))
    return float(h / max(np.log(p.shape[0]), 1e-12))


__all__ = [
    "ChemACEBundle",
    "FinalExplainabilityConfig",
    "MILLambdaVolFrameProvider",
    "build_positive_concept_targets",
    "build_lambda_vol_callback",
    "make_monitor_loader",
    "prepare_chem_ace_bundle",
]
