from __future__ import annotations

from dataclasses import asdict
import json
import logging
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .config import ExportConfig
from .detectors import summarize_alerts_markdown
from .optional_deps import has_plotly, has_pyarrow, has_pyvista
from .types import (
    AlertRecord,
    EpochConceptMetrics,
    EpochTaskMetrics,
    PressureRunArtifacts,
    RecommendationRecord,
    RicciEdgeMetrics,
    RicciTaskSummary,
)

logger = logging.getLogger(__name__)


class LambdaVolArtifactExporter:
    """Exports Lambda-Vol tensors, logs, and 3D visualization artifacts."""

    def __init__(self, *, config: ExportConfig, seed: int = 0) -> None:
        self.config = config
        self.seed = int(seed)

    def export(
        self,
        *,
        run_id: str,
        run_name: str,
        task_ids: Sequence[str],
        concept_ids: Sequence[str],
        concept_rows: Sequence[EpochConceptMetrics],
        task_rows: Sequence[EpochTaskMetrics],
        alerts: Sequence[AlertRecord],
        recommendations: Sequence[RecommendationRecord],
        concept_metadata: Optional[Mapping[str, Mapping[str, Any]]] = None,
        concept_coupling: Optional[np.ndarray] = None,
        ricci_edges: Optional[Sequence[RicciEdgeMetrics]] = None,
        ricci_task_summaries: Optional[Sequence[RicciTaskSummary]] = None,
    ) -> PressureRunArtifacts:
        out_dir = Path(self.config.output_dir).resolve() / str(run_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        concept_df = self._concept_rows_df(concept_rows)
        task_df = self._task_rows_df(task_rows)

        long_csv = out_dir / "concept_pressure_long.csv"
        concept_df.to_csv(long_csv, index=False)

        task_csv = out_dir / "task_metrics_long.csv"
        task_df.to_csv(task_csv, index=False)

        ricci_edge_df = self._ricci_edges_df(ricci_edges or ())
        ricci_summary_df = self._ricci_summary_df(ricci_task_summaries or ())
        ricci_edges_csv: Optional[Path] = None
        ricci_summary_csv: Optional[Path] = None
        ricci_flow_npz: Optional[Path] = None
        if not ricci_edge_df.empty:
            ricci_edges_csv = out_dir / "ricci_edges_long.csv"
            ricci_edge_df.to_csv(ricci_edges_csv, index=False)
            ricci_flow_npz = out_dir / "ricci_flow_tensors.npz"
            self._save_ricci_flow_npz(
                path=ricci_flow_npz,
                edge_df=ricci_edge_df,
                task_ids=task_ids,
                concept_ids=concept_ids,
            )
        if not ricci_summary_df.empty:
            ricci_summary_csv = out_dir / "ricci_task_summary.csv"
            ricci_summary_df.to_csv(ricci_summary_csv, index=False)

        long_parquet: Optional[Path] = None
        if bool(self.config.export_parquet) and has_pyarrow():
            long_parquet = out_dir / "concept_pressure_long.parquet"
            concept_df.to_parquet(long_parquet, index=False)
            task_df.to_parquet(out_dir / "task_metrics_long.parquet", index=False)

        tensors, index = self._build_tensors(
            task_ids=task_ids,
            concept_ids=concept_ids,
            concept_df=concept_df,
        )
        tensor_npz = out_dir / "concept_pressure_tensors.npz"
        self._save_tensor_npz(path=tensor_npz, tensors=tensors, task_ids=task_ids, concept_ids=concept_ids, epochs=index)

        concept_xyz = self._build_concept_xyz(tensors=tensors)
        xyz_df = pd.DataFrame(
            {
                "concept_id": list(concept_ids),
                "z1": concept_xyz[:, 0],
                "z2": concept_xyz[:, 1],
                "z3": concept_xyz[:, 2],
            }
        )
        xyz_csv = out_dir / "concept_xyz.csv"
        xyz_df.to_csv(xyz_csv, index=False)

        manifold_html_by_task: dict[str, str] = {}
        lattice_html = out_dir / "pressure_lattice.html"
        coupling_html: Optional[Path] = None

        lattice_data = self._build_lattice_frame(
            tensors=tensors,
            task_ids=task_ids,
            concept_ids=concept_ids,
            top_k=int(self.config.top_k_lattice),
        )
        lattice_data_csv = out_dir / "pressure_lattice_long.csv"
        lattice_data.to_csv(lattice_data_csv, index=False)

        if bool(self.config.export_plotly_html) and has_plotly():
            manifold_html_by_task = self._export_plotly_manifold(
                out_dir=out_dir,
                tensors=tensors,
                task_ids=task_ids,
                concept_ids=concept_ids,
                concept_xyz=concept_xyz,
            )
            self._export_plotly_lattice(
                out_html=lattice_html,
                lattice_df=lattice_data,
            )
            self._export_heatmaps(out_dir=out_dir, tensors=tensors, task_ids=task_ids, concept_ids=concept_ids)
            self._export_trend_dissipation_plot(out_dir=out_dir, concept_df=concept_df)
            if concept_coupling is not None:
                coupling_html = out_dir / "concept_coupling_3d.html"
                self._export_plotly_coupling(
                    out_html=coupling_html,
                    concept_ids=concept_ids,
                    concept_xyz=concept_xyz,
                    concept_coupling=concept_coupling,
                    tensors=tensors,
                )
        else:
            lattice_html.write_text("Plotly unavailable or export disabled.")

        alerts_json = out_dir / "alerts.json"
        alerts_json.write_text(json.dumps([asdict(a) for a in alerts], indent=2, sort_keys=True))

        alerts_md = out_dir / "alerts.md"
        alerts_md.write_text(summarize_alerts_markdown(alerts))

        recommendations_json = out_dir / "recommendations.json"
        recommendations_json.write_text(json.dumps([asdict(r) for r in recommendations], indent=2, sort_keys=True))

        trend_summary = self._trend_summary(concept_df=concept_df, task_df=task_df)
        trend_summary_path = out_dir / "trend_vs_dissipation.csv"
        trend_summary.to_csv(trend_summary_path, index=False)

        diagnostics_md = out_dir / "diagnostics_summary.md"
        diagnostics_md.write_text(self._diagnostics_markdown(task_df=task_df, alerts=alerts, recommendations=recommendations))

        vtk_path: Optional[Path] = None
        if bool(self.config.export_vtk) and has_pyvista():
            vtk_path = out_dir / "concept_pressure.vtp"
            self._export_vtk(
                out_path=vtk_path,
                tensors=tensors,
                task_ids=task_ids,
                concept_ids=concept_ids,
            )

        metadata = {
            "run_id": str(run_id),
            "run_name": str(run_name),
            "task_ids": [str(x) for x in task_ids],
            "concept_ids": [str(x) for x in concept_ids],
            "epochs": [int(e) for e in index],
            "concept_metadata": {
                str(k): dict(v) for k, v in (concept_metadata or {}).items()
            },
            "artifacts": {
                "tensor_npz": str(tensor_npz),
                "concept_long_csv": str(long_csv),
                "task_long_csv": str(task_csv),
                "lattice_long_csv": str(lattice_data_csv),
                "concept_xyz_csv": str(xyz_csv),
                "trend_vs_dissipation_csv": str(trend_summary_path),
                "alerts_json": str(alerts_json),
                "alerts_md": str(alerts_md),
                "recommendations_json": str(recommendations_json),
                "diagnostics_md": str(diagnostics_md),
                "heatmap_dir": str(out_dir / "heatmaps"),
                "ricci_edges_csv": (None if ricci_edges_csv is None else str(ricci_edges_csv)),
                "ricci_summary_csv": (None if ricci_summary_csv is None else str(ricci_summary_csv)),
                "ricci_flow_npz": (None if ricci_flow_npz is None else str(ricci_flow_npz)),
            },
        }
        metadata_json = out_dir / "metadata.json"
        metadata_json.write_text(json.dumps(metadata, indent=2, sort_keys=True))

        logger.info(
            "Exported Lambda-Vol artifacts",
            extra={
                "run_id": run_id,
                "output_dir": str(out_dir),
                "n_epochs": len(index),
                "n_alerts": len(alerts),
                "n_recommendations": len(recommendations),
                "n_ricci_edges": int(len(ricci_edge_df)),
                "n_ricci_task_summaries": int(len(ricci_summary_df)),
            },
        )

        return PressureRunArtifacts(
            tensor_npz=str(tensor_npz),
            long_csv=str(long_csv),
            long_parquet=(None if long_parquet is None else str(long_parquet)),
            metadata_json=str(metadata_json),
            manifold_html_by_task=manifold_html_by_task,
            lattice_html=str(lattice_html),
            coupling_html=(None if coupling_html is None else str(coupling_html)),
            alerts_json=str(alerts_json),
            alerts_md=str(alerts_md),
            recommendations_json=str(recommendations_json),
            vtk_path=(None if vtk_path is None else str(vtk_path)),
            ricci_edges_csv=(None if ricci_edges_csv is None else str(ricci_edges_csv)),
            ricci_summary_csv=(None if ricci_summary_csv is None else str(ricci_summary_csv)),
            ricci_flow_npz=(None if ricci_flow_npz is None else str(ricci_flow_npz)),
        )

    @staticmethod
    def _concept_rows_df(rows: Sequence[EpochConceptMetrics]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(
                columns=[
                    "epoch",
                    "task_id",
                    "concept_id",
                    "tcav",
                    "tcav_smoothed",
                    "delta_tcav",
                    "attention_support",
                    "prevalence",
                    "rho",
                    "drift",
                    "regime_label",
                    "regime_core",
                    "feedback",
                    "trend_loop",
                    "revert_loop",
                    "dissipation",
                    "context_term",
                ]
            )
        out = pd.DataFrame([asdict(r) for r in rows])
        out["regime_label"] = out["regime_label"].map(lambda x: getattr(x, "value", str(x)))
        return out.sort_values(["epoch", "task_id", "concept_id"]).reset_index(drop=True)

    @staticmethod
    def _task_rows_df(rows: Sequence[EpochTaskMetrics]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(
                columns=[
                    "epoch",
                    "task_id",
                    "attention_entropy",
                    "witness_rate",
                    "train_metric",
                    "val_metric",
                    "loss",
                    "calibration_error",
                    "concentration_entropy",
                    "concentration_gini",
                    "topk_mass",
                    "regime_label",
                ]
            )
        out = pd.DataFrame([asdict(r) for r in rows])
        out["regime_label"] = out["regime_label"].map(lambda x: getattr(x, "value", str(x)))
        if "extra" in out.columns:
            out["extra_json"] = out["extra"].map(
                lambda x: json.dumps(x if isinstance(x, dict) else {}, sort_keys=True)
            )
            out = out.drop(columns=["extra"])
        return out.sort_values(["epoch", "task_id"]).reset_index(drop=True)

    @staticmethod
    def _ricci_edges_df(rows: Sequence[RicciEdgeMetrics]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(
                columns=[
                    "epoch",
                    "task_id",
                    "concept_src",
                    "concept_dst",
                    "weight_raw",
                    "curvature",
                    "weight_flow",
                ]
            )
        return pd.DataFrame([asdict(r) for r in rows]).sort_values(
            ["epoch", "task_id", "concept_src", "concept_dst"]
        ).reset_index(drop=True)

    @staticmethod
    def _ricci_summary_df(rows: Sequence[RicciTaskSummary]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(
                columns=[
                    "epoch",
                    "task_id",
                    "n_nodes",
                    "n_edges",
                    "mean_curvature",
                    "std_curvature",
                    "min_curvature",
                    "max_curvature",
                    "negative_edge_fraction",
                    "strong_negative_edge_fraction",
                    "top_negative_src",
                    "top_negative_dst",
                    "top_negative_curvature",
                ]
            )
        return pd.DataFrame([asdict(r) for r in rows]).sort_values(
            ["epoch", "task_id"]
        ).reset_index(drop=True)

    @staticmethod
    def _save_ricci_flow_npz(
        *,
        path: Path,
        edge_df: pd.DataFrame,
        task_ids: Sequence[str],
        concept_ids: Sequence[str],
    ) -> None:
        if edge_df.empty:
            return

        epochs = tuple(sorted(int(x) for x in edge_df["epoch"].unique().tolist()))
        n_tasks = len(task_ids)
        n_concepts = len(concept_ids)
        n_epochs = len(epochs)

        t_to_idx = {str(t): i for i, t in enumerate(task_ids)}
        c_to_idx = {str(c): j for j, c in enumerate(concept_ids)}
        e_to_idx = {int(e): k for k, e in enumerate(epochs)}

        flow = np.zeros((n_tasks, n_concepts, n_concepts, n_epochs), dtype=np.float32)
        curv = np.zeros((n_tasks, n_concepts, n_concepts, n_epochs), dtype=np.float32)
        raw = np.zeros((n_tasks, n_concepts, n_concepts, n_epochs), dtype=np.float32)

        for row in edge_df.itertuples(index=False):
            ti = t_to_idx.get(str(row.task_id))
            ci = c_to_idx.get(str(row.concept_src))
            cj = c_to_idx.get(str(row.concept_dst))
            ei = e_to_idx.get(int(row.epoch))
            if ti is None or ci is None or cj is None or ei is None:
                continue
            wf = float(row.weight_flow)
            cr = float(row.curvature)
            wr = float(row.weight_raw)
            flow[ti, ci, cj, ei] = wf
            flow[ti, cj, ci, ei] = wf
            curv[ti, ci, cj, ei] = cr
            curv[ti, cj, ci, ei] = cr
            raw[ti, ci, cj, ei] = wr
            raw[ti, cj, ci, ei] = wr

        np.savez_compressed(
            path,
            ricci_weight_flow=flow,
            ricci_curvature=curv,
            ricci_weight_raw=raw,
            task_ids=np.asarray([str(x) for x in task_ids], dtype=object),
            concept_ids=np.asarray([str(x) for x in concept_ids], dtype=object),
            epochs=np.asarray([int(x) for x in epochs], dtype=np.int64),
        )

    @staticmethod
    def _build_tensors(
        *,
        task_ids: Sequence[str],
        concept_ids: Sequence[str],
        concept_df: pd.DataFrame,
    ) -> tuple[dict[str, np.ndarray], tuple[int, ...]]:
        if concept_df.empty:
            zeros = np.zeros((len(task_ids), len(concept_ids), 0), dtype=np.float32)
            return {
                "rho": zeros,
                "tcav": zeros,
                "tcav_smoothed": zeros,
                "delta_tcav": zeros,
                "attention_support": zeros,
                "prevalence": zeros,
                "drift": zeros,
                "regime_core": zeros,
                "feedback": zeros,
                "trend_loop": zeros,
                "revert_loop": zeros,
                "dissipation": zeros,
                "context_term": zeros,
            }, ()

        epochs = tuple(sorted(int(x) for x in concept_df["epoch"].unique().tolist()))
        t_to_idx = {str(t): i for i, t in enumerate(task_ids)}
        c_to_idx = {str(c): j for j, c in enumerate(concept_ids)}
        e_to_idx = {int(e): k for k, e in enumerate(epochs)}

        shape = (len(task_ids), len(concept_ids), len(epochs))
        tensors: dict[str, np.ndarray] = {
            key: np.zeros(shape, dtype=np.float32)
            for key in [
                "rho",
                "tcav",
                "tcav_smoothed",
                "delta_tcav",
                "attention_support",
                "prevalence",
                "drift",
                "regime_core",
                "feedback",
                "trend_loop",
                "revert_loop",
                "dissipation",
                "context_term",
            ]
        }

        for row in concept_df.itertuples(index=False):
            ti = t_to_idx.get(str(row.task_id))
            ci = c_to_idx.get(str(row.concept_id))
            ei = e_to_idx.get(int(row.epoch))
            if ti is None or ci is None or ei is None:
                continue
            for key in tensors:
                tensors[key][ti, ci, ei] = float(getattr(row, key))

        return tensors, epochs

    @staticmethod
    def _save_tensor_npz(
        *,
        path: Path,
        tensors: Mapping[str, np.ndarray],
        task_ids: Sequence[str],
        concept_ids: Sequence[str],
        epochs: Sequence[int],
    ) -> None:
        np.savez_compressed(
            path,
            **{k: np.asarray(v, dtype=np.float32) for k, v in tensors.items()},
            task_ids=np.asarray([str(x) for x in task_ids], dtype=object),
            concept_ids=np.asarray([str(x) for x in concept_ids], dtype=object),
            epochs=np.asarray([int(x) for x in epochs], dtype=np.int64),
        )

    def _build_concept_xyz(self, *, tensors: Mapping[str, np.ndarray]) -> np.ndarray:
        rho = np.asarray(tensors["rho"], dtype=np.float32)
        tcav = np.asarray(tensors["tcav"], dtype=np.float32)
        attn = np.asarray(tensors["attention_support"], dtype=np.float32)

        n_tasks, n_concepts, n_epochs = rho.shape
        if n_concepts == 0:
            return np.zeros((0, 3), dtype=np.float32)

        feats = []
        for ci in range(n_concepts):
            f = np.concatenate(
                [
                    rho[:, ci, :].reshape(-1),
                    tcav[:, ci, :].reshape(-1),
                    attn[:, ci, :].reshape(-1),
                ],
                axis=0,
            )
            feats.append(f)
        x = np.asarray(feats, dtype=np.float32)

        x = x - np.mean(x, axis=0, keepdims=True)
        if x.shape[0] == 1:
            return np.zeros((1, 3), dtype=np.float32)

        if x.shape[0] >= 4:
            umap_xyz = self._try_umap_3d(x)
            if umap_xyz is not None:
                return umap_xyz

        n_comp = max(1, min(3, x.shape[0], x.shape[1]))
        pca = PCA(n_components=n_comp, random_state=self.seed)
        z = pca.fit_transform(x)
        if n_comp < 3:
            pad = np.zeros((z.shape[0], 3 - n_comp), dtype=np.float32)
            z = np.concatenate([z.astype(np.float32), pad], axis=1)
        return z.astype(np.float32)

    def _try_umap_3d(self, x: np.ndarray) -> Optional[np.ndarray]:
        try:
            import umap  # type: ignore
        except Exception:
            return None

        try:
            nn = int(max(2, min(15, x.shape[0] - 1)))
            reducer = umap.UMAP(
                n_components=3,
                n_neighbors=nn,
                min_dist=0.05,
                metric="cosine",
                random_state=self.seed,
            )
            z = reducer.fit_transform(x)
            return np.asarray(z, dtype=np.float32)
        except Exception:
            logger.exception("UMAP projection failed; falling back to PCA")
            return None

    def _export_plotly_manifold(
        self,
        *,
        out_dir: Path,
        tensors: Mapping[str, np.ndarray],
        task_ids: Sequence[str],
        concept_ids: Sequence[str],
        concept_xyz: np.ndarray,
    ) -> dict[str, str]:
        import plotly.express as px

        rho = np.asarray(tensors["rho"], dtype=np.float32)
        tcav = np.asarray(tensors["tcav"], dtype=np.float32)
        prevalence = np.asarray(tensors["prevalence"], dtype=np.float32)
        attention_support = np.asarray(tensors["attention_support"], dtype=np.float32)
        drift = np.asarray(tensors["drift"], dtype=np.float32)

        n_tasks, n_concepts, n_epochs = rho.shape
        html_map: dict[str, str] = {}

        for ti in range(n_tasks):
            rows: list[dict[str, Any]] = []
            for ei in range(n_epochs):
                for ci in range(n_concepts):
                    rows.append(
                        {
                            "task_id": str(task_ids[ti]),
                            "concept_id": str(concept_ids[ci]),
                            "epoch": int(ei),
                            "z1": float(concept_xyz[ci, 0]),
                            "z2": float(concept_xyz[ci, 1]),
                            "z3": float(concept_xyz[ci, 2]),
                            "pressure": float(rho[ti, ci, ei]),
                            "tcav": float(tcav[ti, ci, ei]),
                            "prevalence": float(prevalence[ti, ci, ei]),
                            "attention_support": float(attention_support[ti, ci, ei]),
                            "drift": float(drift[ti, ci, ei]),
                        }
                    )
            frame = pd.DataFrame(rows)
            frame["size"] = np.clip(frame["prevalence"].to_numpy(dtype=np.float32), 0.01, 1.0)

            fig = px.scatter_3d(
                frame,
                x="z1",
                y="z2",
                z="z3",
                animation_frame="epoch",
                color="pressure",
                size="size",
                size_max=24,
                hover_name="concept_id",
                hover_data={
                    "task_id": True,
                    "tcav": ":.3f",
                    "prevalence": ":.3f",
                    "attention_support": ":.3f",
                    "drift": ":.3f",
                    "size": False,
                    "z1": False,
                    "z2": False,
                    "z3": False,
                },
                title=f"Concept Manifold (Task={task_ids[ti]})",
            )
            fig.update_layout(scene=dict(xaxis_title="z1", yaxis_title="z2", zaxis_title="z3"))

            html_path = out_dir / f"concept_manifold_{task_ids[ti]}.html"
            fig.write_html(html_path, include_plotlyjs="cdn")

            frame.to_csv(out_dir / f"concept_manifold_{task_ids[ti]}.csv", index=False)
            html_map[str(task_ids[ti])] = str(html_path)
        return html_map

    def _build_lattice_frame(
        self,
        *,
        tensors: Mapping[str, np.ndarray],
        task_ids: Sequence[str],
        concept_ids: Sequence[str],
        top_k: int,
    ) -> pd.DataFrame:
        rho = np.asarray(tensors["rho"], dtype=np.float32)
        drift = np.asarray(tensors["drift"], dtype=np.float32)
        tcav = np.asarray(tensors["tcav"], dtype=np.float32)

        n_tasks, n_concepts, n_epochs = rho.shape
        if n_concepts == 0 or n_epochs == 0:
            return pd.DataFrame(
                columns=[
                    "task_id",
                    "task_index",
                    "concept_id",
                    "concept_index",
                    "epoch",
                    "rho",
                    "drift",
                    "tcav",
                ]
            )

        mean_rho = np.mean(rho, axis=(0, 2))
        kk = int(max(1, min(top_k, n_concepts)))
        idx = np.argsort(mean_rho)[-kk:]

        rows: list[dict[str, Any]] = []
        for ei in range(n_epochs):
            for ti in range(n_tasks):
                for ci in idx:
                    rows.append(
                        {
                            "task_id": str(task_ids[ti]),
                            "task_index": int(ti),
                            "concept_id": str(concept_ids[ci]),
                            "concept_index": int(ci),
                            "epoch": int(ei),
                            "rho": float(rho[ti, ci, ei]),
                            "drift": float(drift[ti, ci, ei]),
                            "tcav": float(tcav[ti, ci, ei]),
                        }
                    )
        return pd.DataFrame(rows)

    def _export_plotly_lattice(self, *, out_html: Path, lattice_df: pd.DataFrame) -> None:
        import plotly.express as px

        if lattice_df.empty:
            out_html.write_text("No lattice data.")
            return

        lattice_df = lattice_df.copy()
        lattice_df["size"] = np.clip(np.abs(lattice_df["drift"].to_numpy(dtype=np.float32)), 0.01, 1.0)

        fig = px.scatter_3d(
            lattice_df,
            x="epoch",
            y="concept_index",
            z="task_index",
            color="rho",
            size="size",
            size_max=16,
            hover_name="concept_id",
            hover_data={
                "task_id": True,
                "rho": ":.3f",
                "drift": ":.3f",
                "tcav": ":.3f",
                "size": False,
            },
            title="Pressure Field Lattice (task, concept, epoch)",
        )
        fig.update_layout(
            scene=dict(
                xaxis_title="Epoch",
                yaxis_title="Concept Index",
                zaxis_title="Task Index",
            )
        )
        fig.write_html(out_html, include_plotlyjs="cdn")

    def _export_plotly_coupling(
        self,
        *,
        out_html: Path,
        concept_ids: Sequence[str],
        concept_xyz: np.ndarray,
        concept_coupling: np.ndarray,
        tensors: Mapping[str, np.ndarray],
    ) -> None:
        import plotly.graph_objects as go

        cmat = np.asarray(concept_coupling, dtype=np.float32)
        n = len(concept_ids)
        if cmat.shape != (n, n):
            out_html.write_text("Concept coupling shape mismatch.")
            return

        rho_last = np.asarray(tensors["rho"], dtype=np.float32)
        if rho_last.size > 0:
            node_value = np.mean(rho_last[:, :, -1], axis=0)
        else:
            node_value = np.zeros((n,), dtype=np.float32)

        thr = np.percentile(np.abs(cmat[np.triu_indices(n, k=1)]), 75) if n >= 2 else 0.0
        thr = float(max(thr, 1e-6))

        edge_x: list[float] = []
        edge_y: list[float] = []
        edge_z: list[float] = []
        edge_color: list[float] = []

        for i in range(n):
            for j in range(i + 1, n):
                w = float(cmat[i, j])
                if abs(w) < thr:
                    continue
                edge_x.extend([float(concept_xyz[i, 0]), float(concept_xyz[j, 0]), np.nan])
                edge_y.extend([float(concept_xyz[i, 1]), float(concept_xyz[j, 1]), np.nan])
                edge_z.extend([float(concept_xyz[i, 2]), float(concept_xyz[j, 2]), np.nan])
                edge_color.extend([w, w, w])

        edge_trace = go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode="lines",
            line=dict(color="rgba(120,120,120,0.4)", width=2),
            hoverinfo="none",
            name="coupling",
        )

        node_trace = go.Scatter3d(
            x=concept_xyz[:, 0],
            y=concept_xyz[:, 1],
            z=concept_xyz[:, 2],
            mode="markers+text",
            marker=dict(
                size=np.clip((node_value - np.min(node_value) + 1e-3) * 20.0, 5.0, 24.0),
                color=node_value,
                colorscale="Viridis",
                colorbar=dict(title="pressure(last)"),
            ),
            text=[str(c) for c in concept_ids],
            textposition="top center",
            name="concepts",
        )

        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title="Concept Coupling Spillover Graph",
            scene=dict(xaxis_title="z1", yaxis_title="z2", zaxis_title="z3"),
        )
        fig.write_html(out_html, include_plotlyjs="cdn")

    def _export_heatmaps(
        self,
        *,
        out_dir: Path,
        tensors: Mapping[str, np.ndarray],
        task_ids: Sequence[str],
        concept_ids: Sequence[str],
    ) -> None:
        import plotly.express as px

        heatmap_dir = out_dir / "heatmaps"
        heatmap_dir.mkdir(parents=True, exist_ok=True)

        rho = np.asarray(tensors["rho"], dtype=np.float32)
        n_tasks = len(task_ids)

        for ti in range(n_tasks):
            z = rho[ti]
            fig = px.imshow(
                z,
                x=[int(i) for i in range(z.shape[1])],
                y=[str(c) for c in concept_ids],
                labels={"x": "epoch", "y": "concept", "color": "rho"},
                title=f"Task {task_ids[ti]} pressure heatmap",
                aspect="auto",
            )
            fig.write_html(heatmap_dir / f"pressure_heatmap_{task_ids[ti]}.html", include_plotlyjs="cdn")

    def _export_trend_dissipation_plot(self, *, out_dir: Path, concept_df: pd.DataFrame) -> None:
        import plotly.express as px

        if concept_df.empty:
            return

        agg = (
            concept_df.groupby(["epoch", "task_id"], as_index=False)
            .agg(trend_loop=("trend_loop", "mean"), dissipation=("dissipation", "mean"))
            .sort_values(["task_id", "epoch"])
        )

        melt = pd.melt(
            agg,
            id_vars=["epoch", "task_id"],
            value_vars=["trend_loop", "dissipation"],
            var_name="component",
            value_name="value",
        )
        fig = px.line(
            melt,
            x="epoch",
            y="value",
            color="component",
            facet_row="task_id",
            title="Trend vs Dissipation",
        )
        fig.write_html(out_dir / "trend_vs_dissipation.html", include_plotlyjs="cdn")

    @staticmethod
    def _trend_summary(*, concept_df: pd.DataFrame, task_df: pd.DataFrame) -> pd.DataFrame:
        if concept_df.empty:
            return pd.DataFrame(
                columns=["task_id", "epoch", "trend_loop", "dissipation", "trend_over_dissipation"]
            )

        agg = (
            concept_df.groupby(["task_id", "epoch"], as_index=False)
            .agg(trend_loop=("trend_loop", "mean"), dissipation=("dissipation", "mean"))
            .sort_values(["task_id", "epoch"])
        )
        agg["trend_over_dissipation"] = agg["trend_loop"] / (agg["dissipation"].abs() + 1e-6)
        return agg

    @staticmethod
    def _diagnostics_markdown(
        *,
        task_df: pd.DataFrame,
        alerts: Sequence[AlertRecord],
        recommendations: Sequence[RecommendationRecord],
    ) -> str:
        lines = ["# Lambda-Vol Diagnostics", ""]
        if task_df.empty:
            lines.append("No task metrics available.")
        else:
            last_ep = int(task_df["epoch"].max())
            lines.append(f"Latest epoch: {last_ep}")
            lines.append("")
            for task_id, grp in task_df.groupby("task_id"):
                g = grp.sort_values("epoch")
                last = g.iloc[-1]
                lines.append(
                    (
                        f"- task={task_id}: val={float(last['val_metric']):.4f}, "
                        f"loss={float(last['loss']):.4f}, entropy={float(last['attention_entropy']):.4f}, "
                        f"topk_mass={float(last['topk_mass']):.4f}"
                    )
                )

        lines.append("")
        lines.append(f"Alerts: {len(alerts)}")
        lines.append(f"Recommendations: {len(recommendations)}")
        return "\n".join(lines)

    def _export_vtk(
        self,
        *,
        out_path: Path,
        tensors: Mapping[str, np.ndarray],
        task_ids: Sequence[str],
        concept_ids: Sequence[str],
    ) -> None:
        import pyvista as pv

        rho = np.asarray(tensors["rho"], dtype=np.float32)
        drift = np.asarray(tensors["drift"], dtype=np.float32)
        tcav = np.asarray(tensors["tcav"], dtype=np.float32)

        n_tasks, n_concepts, n_epochs = rho.shape
        pts: list[list[float]] = []
        rho_vals: list[float] = []
        drift_vals: list[float] = []
        tcav_vals: list[float] = []
        task_idx_vals: list[int] = []
        concept_idx_vals: list[int] = []
        epoch_vals: list[int] = []

        for ti in range(n_tasks):
            for ci in range(n_concepts):
                for ei in range(n_epochs):
                    pts.append([float(ti), float(ci), float(ei)])
                    rho_vals.append(float(rho[ti, ci, ei]))
                    drift_vals.append(float(drift[ti, ci, ei]))
                    tcav_vals.append(float(tcav[ti, ci, ei]))
                    task_idx_vals.append(int(ti))
                    concept_idx_vals.append(int(ci))
                    epoch_vals.append(int(ei))

        cloud = pv.PolyData(np.asarray(pts, dtype=np.float32))
        cloud["rho"] = np.asarray(rho_vals, dtype=np.float32)
        cloud["drift"] = np.asarray(drift_vals, dtype=np.float32)
        cloud["tcav"] = np.asarray(tcav_vals, dtype=np.float32)
        cloud["task_idx"] = np.asarray(task_idx_vals, dtype=np.int32)
        cloud["concept_idx"] = np.asarray(concept_idx_vals, dtype=np.int32)
        cloud["epoch"] = np.asarray(epoch_vals, dtype=np.int32)
        cloud.save(out_path)


__all__ = ["LambdaVolArtifactExporter"]
