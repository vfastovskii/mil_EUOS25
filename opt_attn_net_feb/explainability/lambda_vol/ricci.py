from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Mapping, Optional, Sequence

import numpy as np

from .config import RicciConfig
from .types import RicciEdgeMetrics, RicciTaskSummary

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RicciEpochOutput:
    """Ricci diagnostics and flow output for a single epoch."""

    edge_rows: list[RicciEdgeMetrics]
    task_summaries: list[RicciTaskSummary]
    flowed_similarity_by_task: np.ndarray  # [T, C, C]
    mean_flowed_similarity: np.ndarray  # [C, C]


class ConceptRicciFlowAnalyzer:
    """Builds concept graphs and applies discrete Ricci-flow-style reweighting."""

    def __init__(
        self,
        *,
        task_ids: Sequence[str],
        concept_ids: Sequence[str],
        config: RicciConfig,
    ) -> None:
        self.task_ids = tuple(str(x) for x in task_ids)
        self.concept_ids = tuple(str(x) for x in concept_ids)
        self.config = config

        if len(self.task_ids) == 0:
            raise ValueError("task_ids cannot be empty")
        if len(self.concept_ids) < 2:
            raise ValueError("concept_ids must contain at least 2 concepts")

    def analyze_epoch(
        self,
        *,
        epoch: int,
        rho: np.ndarray,
        attention_support: np.ndarray,
        prevalence: np.ndarray,
        tcav_history_by_task: Optional[Mapping[str, np.ndarray]] = None,
    ) -> RicciEpochOutput:
        """Compute per-task curvature diagnostics and flowed similarity graphs."""
        rho_m = np.asarray(rho, dtype=np.float32)
        attn_m = np.asarray(attention_support, dtype=np.float32)
        prev_m = np.asarray(prevalence, dtype=np.float32)
        expected_shape = (len(self.task_ids), len(self.concept_ids))
        if rho_m.shape != expected_shape:
            raise ValueError(f"rho shape mismatch: expected {expected_shape}, got {rho_m.shape}")
        if attn_m.shape != expected_shape or prev_m.shape != expected_shape:
            raise ValueError("attention_support/prevalence shape mismatch")

        flowed_stack = np.zeros((len(self.task_ids), len(self.concept_ids), len(self.concept_ids)), dtype=np.float32)
        edge_rows: list[RicciEdgeMetrics] = []
        summaries: list[RicciTaskSummary] = []

        for ti, task_id in enumerate(self.task_ids):
            tcav_hist = None
            if tcav_history_by_task is not None:
                tcav_hist = tcav_history_by_task.get(str(task_id))

            sim = self._build_similarity(
                rho=np.asarray(rho_m[ti], dtype=np.float32),
                attention=np.asarray(attn_m[ti], dtype=np.float32),
                prevalence=np.asarray(prev_m[ti], dtype=np.float32),
                tcav_history=tcav_hist,
            )
            curvature = self._forman_curvature(sim)
            sim_flow = self._run_ricci_flow(sim)

            flowed_stack[ti] = sim_flow.astype(np.float32)
            edge_rows.extend(
                self._build_edge_rows(
                    epoch=int(epoch),
                    task_id=str(task_id),
                    sim_raw=sim,
                    curvature=curvature,
                    sim_flow=sim_flow,
                )
            )
            summaries.append(
                self._build_summary(
                    epoch=int(epoch),
                    task_id=str(task_id),
                    sim_raw=sim,
                    curvature=curvature,
                )
            )

        mean_flowed = np.mean(flowed_stack, axis=0).astype(np.float32)
        return RicciEpochOutput(
            edge_rows=edge_rows,
            task_summaries=summaries,
            flowed_similarity_by_task=flowed_stack,
            mean_flowed_similarity=mean_flowed,
        )

    def _build_similarity(
        self,
        *,
        rho: np.ndarray,
        attention: np.ndarray,
        prevalence: np.ndarray,
        tcav_history: Optional[np.ndarray],
    ) -> np.ndarray:
        x_rho = np.maximum(np.asarray(rho, dtype=np.float64), 0.0)
        x_attn = np.maximum(np.asarray(attention, dtype=np.float64), 0.0)
        x_prev = np.maximum(np.asarray(prevalence, dtype=np.float64), 0.0)
        x_corr = self._tcav_abs_corr(tcav_history, n_concepts=len(self.concept_ids))

        sim_rho = np.sqrt(np.outer(x_rho, x_rho))
        sim_attn = np.sqrt(np.outer(x_attn, x_attn))
        sim_prev = np.sqrt(np.outer(x_prev, x_prev))

        sim = (
            float(self.config.w_rho) * sim_rho
            + float(self.config.w_attention) * sim_attn
            + float(self.config.w_prevalence) * sim_prev
            + float(self.config.w_tcav_corr) * x_corr
        )
        np.fill_diagonal(sim, 0.0)

        mx = float(np.max(sim))
        if mx > 1e-12:
            sim = sim / mx

        upper = sim[np.triu_indices(sim.shape[0], k=1)]
        positive = upper[upper > 0.0]
        if positive.size == 0:
            return np.zeros_like(sim, dtype=np.float32)

        q = float(np.clip(self.config.edge_keep_quantile, 0.0, 1.0))
        q_thr = float(np.quantile(positive, q))
        thr = max(float(self.config.min_edge_weight), q_thr)
        keep = sim >= thr

        k = int(max(0, self.config.top_k_per_node))
        if k > 0:
            n = sim.shape[0]
            kk = min(k, max(0, n - 1))
            if kk > 0:
                for i in range(n):
                    row = sim[i].copy()
                    row[i] = -np.inf
                    top_idx = np.argpartition(row, -kk)[-kk:]
                    keep[i, top_idx] = True

        keep = np.logical_or(keep, keep.T)
        sim = np.where(keep, sim, 0.0)
        np.fill_diagonal(sim, 0.0)
        return sim.astype(np.float32)

    @staticmethod
    def _tcav_abs_corr(history: Optional[np.ndarray], *, n_concepts: int) -> np.ndarray:
        if history is None:
            return np.zeros((n_concepts, n_concepts), dtype=np.float64)
        arr = np.asarray(history, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] != n_concepts:
            return np.zeros((n_concepts, n_concepts), dtype=np.float64)
        centered = arr - np.mean(arr, axis=0, keepdims=True)
        std = np.std(centered, axis=0)
        valid = std > 1e-12
        if not np.any(valid):
            return np.zeros((n_concepts, n_concepts), dtype=np.float64)
        z = np.zeros_like(centered, dtype=np.float64)
        z[:, valid] = centered[:, valid] / std[valid]
        corr = (z.T @ z) / float(max(1, arr.shape[0]))
        corr = np.nan_to_num(np.abs(corr), nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(corr, 0.0)
        return corr

    @staticmethod
    def _forman_curvature(sim: np.ndarray) -> np.ndarray:
        w = np.asarray(sim, dtype=np.float64)
        n = int(w.shape[0])
        curv = np.zeros_like(w, dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                wij = float(w[i, j])
                if wij <= 0.0:
                    continue

                term_i = 0.0
                for k in range(n):
                    if k == i or k == j:
                        continue
                    wik = float(w[i, k])
                    if wik > 0.0:
                        term_i += 1.0 / float(np.sqrt(max(wij * wik, 1e-12)))

                term_j = 0.0
                for k in range(n):
                    if k == i or k == j:
                        continue
                    wjk = float(w[j, k])
                    if wjk > 0.0:
                        term_j += 1.0 / float(np.sqrt(max(wij * wjk, 1e-12)))

                kappa = 2.0 - term_i - term_j
                curv[i, j] = kappa
                curv[j, i] = kappa
        return curv.astype(np.float32)

    def _run_ricci_flow(self, sim: np.ndarray) -> np.ndarray:
        if (not bool(self.config.flow_enabled)) or int(self.config.flow_steps) <= 0:
            return np.asarray(sim, dtype=np.float32)

        cur = np.asarray(sim, dtype=np.float64).copy()
        eps = float(max(1e-12, self.config.flow_eps))
        step = float(max(0.0, self.config.flow_step_size))
        if step <= 0.0:
            return cur.astype(np.float32)

        n = cur.shape[0]
        for _ in range(int(self.config.flow_steps)):
            kappa = self._forman_curvature(cur).astype(np.float64)
            for i in range(n):
                for j in range(i + 1, n):
                    wij = float(cur[i, j])
                    if wij <= 0.0:
                        continue
                    length = 1.0 / (wij + eps)
                    factor = float(np.clip(1.0 - step * float(kappa[i, j]), 0.05, 20.0))
                    length_new = length * factor
                    wij_new = 1.0 / (length_new + eps)
                    cur[i, j] = wij_new
                    cur[j, i] = wij_new

            mx = float(np.max(cur))
            if mx > 1e-12:
                cur = cur / mx
            np.fill_diagonal(cur, 0.0)
        return cur.astype(np.float32)

    def _build_edge_rows(
        self,
        *,
        epoch: int,
        task_id: str,
        sim_raw: np.ndarray,
        curvature: np.ndarray,
        sim_flow: np.ndarray,
    ) -> list[RicciEdgeMetrics]:
        out: list[RicciEdgeMetrics] = []
        n = len(self.concept_ids)
        for i in range(n):
            for j in range(i + 1, n):
                wij = float(sim_raw[i, j])
                if wij <= 0.0:
                    continue
                out.append(
                    RicciEdgeMetrics(
                        epoch=int(epoch),
                        task_id=str(task_id),
                        concept_src=str(self.concept_ids[i]),
                        concept_dst=str(self.concept_ids[j]),
                        weight_raw=wij,
                        curvature=float(curvature[i, j]),
                        weight_flow=float(sim_flow[i, j]),
                    )
                )
        return out

    def _build_summary(
        self,
        *,
        epoch: int,
        task_id: str,
        sim_raw: np.ndarray,
        curvature: np.ndarray,
    ) -> RicciTaskSummary:
        n = len(self.concept_ids)
        edge_mask = np.triu(sim_raw > 0.0, k=1)
        edge_idx = np.argwhere(edge_mask)
        vals = curvature[edge_mask]
        n_edges = int(vals.size)

        if n_edges == 0:
            return RicciTaskSummary(
                epoch=int(epoch),
                task_id=str(task_id),
                n_nodes=int(n),
                n_edges=0,
                mean_curvature=0.0,
                std_curvature=0.0,
                min_curvature=0.0,
                max_curvature=0.0,
                negative_edge_fraction=0.0,
                strong_negative_edge_fraction=0.0,
                top_negative_src=None,
                top_negative_dst=None,
                top_negative_curvature=0.0,
            )

        vals = np.asarray(vals, dtype=np.float64)
        neg_thr = float(self.config.negative_curvature_threshold)
        strong_thr = float(self.config.strong_negative_curvature_threshold)
        neg_frac = float(np.mean(vals < neg_thr))
        strong_frac = float(np.mean(vals < strong_thr))

        min_pos = int(np.argmin(vals))
        i, j = edge_idx[min_pos].tolist()
        return RicciTaskSummary(
            epoch=int(epoch),
            task_id=str(task_id),
            n_nodes=int(n),
            n_edges=int(n_edges),
            mean_curvature=float(np.mean(vals)),
            std_curvature=float(np.std(vals)),
            min_curvature=float(np.min(vals)),
            max_curvature=float(np.max(vals)),
            negative_edge_fraction=float(neg_frac),
            strong_negative_edge_fraction=float(strong_frac),
            top_negative_src=str(self.concept_ids[int(i)]),
            top_negative_dst=str(self.concept_ids[int(j)]),
            top_negative_curvature=float(vals[min_pos]),
        )


__all__ = ["ConceptRicciFlowAnalyzer", "RicciEpochOutput"]
