from __future__ import annotations

from dataclasses import dataclass
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
    PatchGenerationConfig,
    Pharm3DPatchConfig,
)
from ..explainability.chem_ace.concepts.pipeline import ChemACEPipeline, MoleculeSource
from ..explainability.chem_ace.embedding.hooks import LayerActivationHook
from ..explainability.chem_ace.optional_deps import OptionalDependencyError, require_rdkit
from ..explainability.chem_ace.types import ModelTaskAdapter, PatchEmbeddingRecord, PatchRecord
from ..explainability.lambda_vol import LambdaVolConfig
from ..explainability.lambda_vol.config import ExportConfig as LambdaVolExportConfig
from ..explainability.lambda_vol.config import PolicyConfig, StoreConfig, TrackerConfig
from ..explainability.lambda_vol.integrations import LambdaVolLightningCallback, LightningEpochFrames
from ..explainability.lambda_vol.monitor import LambdaVolMonitor
from ..training.builders import DataLoaderBuilder, LoaderConfig
from ..utils.constants import TASK_COLS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FinalExplainabilityConfig:
    """Controls Chem-ACE + Lambda-Vol integration in final optimized training."""

    run_chem_ace: bool = False
    run_lambda_vol: bool = False
    curated_smiles_col: str = "curated_SMILES"

    chem_ace_output_dir: Optional[str] = None
    chem_ace_db_uri: Optional[str] = None
    chem_ace_max_ids: int = 10000
    chem_ace_max_confs_per_id: int = 4
    chem_ace_max_2d_dim: int = 256
    chem_ace_max_3dqm_dim: int = 256
    chem_ace_top_concepts: int = 64

    lambda_vol_output_dir: Optional[str] = None
    lambda_vol_db_uri: Optional[str] = None
    lambda_vol_layer_name: str = "mixer_post_norm"
    lambda_vol_top_concepts: int = 24
    lambda_vol_monitor_max_samples: int = 512
    lambda_vol_tcav_repeats: int = 2
    lambda_vol_random_counterexamples: int = 96
    lambda_vol_min_concept_samples: int = 8


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
        return tcav_scores



def prepare_chem_ace_bundle(
    *,
    config: FinalExplainabilityConfig,
    outdir: Path,
    seed: int,
    df_full: pd.DataFrame,
    id_col: str,
    ids_scope: Sequence[str],
    ids_2d_file: Sequence[str],
    X2d_file: np.ndarray,
    starts: np.ndarray,
    counts: np.ndarray,
    id2pos: Mapping[str, int],
    conf_sorted: np.ndarray,
    Xinst_sorted: np.ndarray,
) -> Optional[ChemACEBundle]:
    """Build Chem-ACE concepts from curated SMILES + fused 2D/3D/QM features."""
    if not bool(config.run_chem_ace):
        return None

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

    ace_cfg = ChemACEConfig(
        run_name="chem_ace_final_pipeline",
        seed=int(seed),
        output_dir=str(ace_out_dir),
        embedding=EmbeddingConfig(layer_name="feature_fusion_2d3dqm", strategy="masked_input"),
        discovery=ConceptDiscoveryConfig(),
        patch_generation=PatchGenerationConfig(pharm3d=Pharm3DPatchConfig(enabled=False)),
        database=DatabaseConfig(uri=db_uri),
    )
    pipeline = ChemACEPipeline(config=ace_cfg)
    run_id = pipeline.start_run(task_ids=TASK_COLS)

    ids_unique = sorted({str(x) for x in ids_scope})
    max_ids = int(config.chem_ace_max_ids)
    if max_ids > 0:
        ids_unique = ids_unique[:max_ids]

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

    conf_map, inst_map, inst_mean_map = _build_instance_feature_maps(
        ids=ids_unique,
        starts=starts,
        counts=counts,
        id2pos=id2pos,
        conf_sorted=conf_sorted,
        Xinst_sorted=Xinst_sorted,
        max_confs_per_id=int(config.chem_ace_max_confs_per_id),
    )

    molecules: list[MoleculeSource] = []
    molecules_by_id: dict[str, Any] = {}
    for mol_id in ids_unique:
        smi = smiles_by_id.get(mol_id)
        if smi is None:
            continue
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        mol = Chem.AddHs(mol)
        conf_ids = tuple(conf_map.get(mol_id, []))
        molecules.append(MoleculeSource(mol_id=str(mol_id), mol=mol, conf_ids=conf_ids))
        molecules_by_id[str(mol_id)] = mol

    patches = pipeline.generate_patches(molecules=molecules)
    if len(patches) == 0:
        raise RuntimeError("Chem-ACE generated zero patches; cannot continue")

    embeddings = _build_feature_patch_embeddings(
        pipeline=pipeline,
        patches=patches,
        molecules_by_id=molecules_by_id,
        x2d_by_id=x2d_by_id,
        xinst_by_pair=inst_map,
        xinst_mean_by_id=inst_mean_map,
        max_2d_dim=int(config.chem_ace_max_2d_dim),
        max_3dqm_dim=int(config.chem_ace_max_3dqm_dim),
    )

    concept_set, concept_set_id = pipeline.discover_and_store_concepts(
        run_id=run_id,
        embeddings=embeddings,
    )
    tagging = pipeline.tag_and_store_concepts(
        concept_set=concept_set,
        patches=patches,
        molecules_by_id=molecules_by_id,
    )

    concept_mol_map, concept_conf_map = _build_concept_membership_maps(
        concept_set=concept_set,
        patches=patches,
    )
    support_map = {str(c.concept_local_id): int(c.support) for c in concept_set.candidates}
    tag_map = {
        str(t.concept_id): {
            "label_auto": str(t.label_auto),
            "tags": [str(x.tag) for x in t.tags],
        }
        for t in tagging
    }

    concept_metadata: dict[str, dict[str, Any]] = {}
    for cid, support in support_map.items():
        concept_metadata[cid] = {
            "support": int(support),
            "label_auto": tag_map.get(cid, {}).get("label_auto"),
            "tags": tag_map.get(cid, {}).get("tags", []),
            "n_molecules": int(len(concept_mol_map.get(cid, set()))),
            "n_conf_pairs": int(len(concept_conf_map.get(cid, set()))),
        }

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
        "n_molecules": int(len(molecules)),
        "n_patches": int(len(patches)),
        "n_embeddings": int(len(embeddings)),
        "n_concepts": int(len(concept_set.candidates)),
        "selected_concepts": ordered_concepts,
    }
    (ace_out_dir / "chem_ace_pipeline_summary.json").write_text(json.dumps(summary, indent=2))

    logger.info(
        "Prepared Chem-ACE bundle",
        extra={
            "run_id": run_id,
            "concept_set_id": concept_set_id,
            "n_concepts": len(ordered_concepts),
            "n_patches": len(patches),
        },
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
    if not chem_bundle.concept_ids:
        logger.warning("Lambda-Vol requested but no Chem-ACE concepts are available")
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

    return LambdaVolLightningCallback(
        monitor=monitor,
        frame_provider=frame_provider,
        export_on_fit_end=True,
    )



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
    return DataLoaderBuilder(loader_cfg).eval_loader(
        ds,
        batch_size=int(max(1, batch_size)),
        collate_fn=collate_export,
    )



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
) -> list[PatchEmbeddingRecord]:
    emb_recs: list[PatchEmbeddingRecord] = []

    for patch in patches:
        mol_id = str(patch.mol_id)
        x2d = x2d_by_id.get(mol_id)
        if x2d is None:
            continue

        x3d = None
        if patch.conf_id is not None:
            x3d = xinst_by_pair.get((mol_id, str(patch.conf_id)))
        if x3d is None:
            x3d = xinst_mean_by_id.get(mol_id)
        if x3d is None:
            x3d = np.zeros((max(1, int(max_3dqm_dim)),), dtype=np.float32)

        mol = molecules_by_id.get(mol_id)
        desc = _patch_descriptors(mol=mol, atom_indices=patch.atom_indices, patch_type=patch.patch_type)

        v2d = _take_or_pad(np.asarray(x2d, dtype=np.float32), int(max_2d_dim))
        v3d = _take_or_pad(np.asarray(x3d, dtype=np.float32), int(max_3dqm_dim))
        vec = np.concatenate([v2d, v3d, desc], axis=0).astype(np.float32)

        rec = pipeline.embedding_cache.save(
            patch=patch,
            layer_name="feature_fusion_2d3dqm",
            strategy="feature_projection",
            vector=vec,
            metadata={
                "strategy": "feature_projection",
                "d2": int(v2d.shape[0]),
                "d3qm": int(v3d.shape[0]),
                "ddesc": int(desc.shape[0]),
            },
        )
        pipeline.repository.upsert_patch_embedding(rec)
        emb_recs.append(rec)

    if not emb_recs:
        raise RuntimeError("Chem-ACE feature projection produced zero patch embeddings")
    return emb_recs



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



def _build_concept_membership_maps(
    *,
    concept_set: Any,
    patches: Sequence[PatchRecord],
) -> tuple[dict[str, set[str]], dict[str, set[tuple[str, str]]]]:
    patch_by_id = {str(p.patch_id): p for p in patches}
    concept_mols: dict[str, set[str]] = {}
    concept_confs: dict[str, set[tuple[str, str]]] = {}

    for m in concept_set.memberships:
        cid = str(m.concept_local_id)
        patch = patch_by_id.get(str(m.patch_id))
        if patch is None:
            continue
        concept_mols.setdefault(cid, set()).add(str(patch.mol_id))
        if patch.conf_id is not None:
            concept_confs.setdefault(cid, set()).add((str(patch.mol_id), str(patch.conf_id)))

    return concept_mols, concept_confs



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
    "build_lambda_vol_callback",
    "make_monitor_loader",
    "prepare_chem_ace_bundle",
]
