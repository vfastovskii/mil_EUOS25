from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Optional, Sequence

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
    """
    Build concept graph from co-activation and apply discrete Ricci-flow-style reweighting.

    Node scores (per task/concept/epoch):
      score = w_tcav * tcav_ema + w_attention * attn_ema (2D forces attention term to 0).

    Edge construction:
      - same modality: Jaccard/co-occurrence over sample-level concept activity.
      - cross modality: weighted co-activation using per-sample min/max overlap.
      - edge weight scales by sqrt(score_i * score_j).
    """

    _VALID_MODALITIES = {"2d", "3d_geom", "3d_qm"}

    def __init__(
        self,
        *,
        task_ids: Sequence[str],
        concept_ids: Sequence[str],
        concept_modalities: Sequence[str],
        config: RicciConfig,
    ) -> None:
        self.task_ids = tuple(str(x) for x in task_ids)
        self.concept_ids = tuple(str(x) for x in concept_ids)
        self.config = config
        if len(self.task_ids) == 0:
            raise ValueError("task_ids cannot be empty")
        if len(self.concept_ids) < 2:
            raise ValueError("concept_ids must contain at least 2 concepts")
        if len(concept_modalities) != len(self.concept_ids):
            raise ValueError("concept_modalities length mismatch")

        self.concept_modalities = tuple(
            (m if str(m) in self._VALID_MODALITIES else "2d")
            for m in (str(x) for x in concept_modalities)
        )

    def analyze_epoch(
        self,
        *,
        epoch: int,
        tcav_smoothed: np.ndarray,
        attention_support: np.ndarray,
        concept_activity_samples: Optional[np.ndarray] = None,
    ) -> RicciEpochOutput:
        """
        Compute per-task curvature diagnostics and flowed similarity graphs.

        Parameters
        ----------
        tcav_smoothed:
            [T, C] task-concept TCAV EMA matrix.
        attention_support:
            [T, C] task-concept attention support matrix.
        concept_activity_samples:
            Optional [T, N, C] sample-level concept activities.
            For 3D concepts this should be attention-derived mass; for 2D binary/continuous
            molecule-level concept activity.
        """
        tcav_m = np.asarray(tcav_smoothed, dtype=np.float32)
        attn_m = np.asarray(attention_support, dtype=np.float32)
        expected_shape = (len(self.task_ids), len(self.concept_ids))
        if tcav_m.shape != expected_shape:
            raise ValueError(f"tcav_smoothed shape mismatch: expected {expected_shape}, got {tcav_m.shape}")
        if attn_m.shape != expected_shape:
            raise ValueError(f"attention_support shape mismatch: expected {expected_shape}, got {attn_m.shape}")

        activity = None
        if concept_activity_samples is not None:
            arr = np.asarray(concept_activity_samples, dtype=np.float32)
            if arr.ndim == 3 and arr.shape[0] == expected_shape[0] and arr.shape[2] == expected_shape[1]:
                activity = np.maximum(arr, 0.0)

        flowed_stack = np.zeros((len(self.task_ids), len(self.concept_ids), len(self.concept_ids)), dtype=np.float32)
        edge_rows: list[RicciEdgeMetrics] = []
        summaries: list[RicciTaskSummary] = []

        for ti, task_id in enumerate(self.task_ids):
            act_t = None if activity is None else np.asarray(activity[ti], dtype=np.float32)  # [N, C]
            scores = self._node_scores(
                tcav=np.asarray(tcav_m[ti], dtype=np.float32),
                attention=np.asarray(attn_m[ti], dtype=np.float32),
                activity=act_t,
            )
            sim = self._build_similarity_from_coactivation(
                node_scores=scores,
                activity=act_t,
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

    def _node_scores(
        self,
        *,
        tcav: np.ndarray,
        attention: np.ndarray,
        activity: Optional[np.ndarray],
    ) -> np.ndarray:
        tc = np.maximum(np.asarray(tcav, dtype=np.float64), 0.0)
        att = np.maximum(np.asarray(attention, dtype=np.float64), 0.0)
        out = float(self.config.w_tcav) * tc
        for ci, mode in enumerate(self.concept_modalities):
            if mode != "2d":
                out[ci] += float(self.config.w_attention) * float(att[ci])
        mx = float(np.max(out))
        if mx > 1e-12:
            out = out / mx
            return out.astype(np.float32)

        # Fallback: if all node scores are zero, derive weak ranking from observed activity.
        if activity is not None and activity.ndim == 2 and activity.shape[1] == len(self.concept_ids):
            mean_act = np.maximum(np.mean(np.asarray(activity, dtype=np.float64), axis=0), 0.0)
            mx2 = float(np.max(mean_act))
            if mx2 > 1e-12:
                return (mean_act / mx2).astype(np.float32)
        return np.zeros((len(self.concept_ids),), dtype=np.float32)

    def _build_similarity_from_coactivation(
        self,
        *,
        node_scores: np.ndarray,
        activity: Optional[np.ndarray],
    ) -> np.ndarray:
        n_concepts = len(self.concept_ids)
        sim = np.zeros((n_concepts, n_concepts), dtype=np.float64)
        eps = 1e-12

        selected = self._selected_concepts(node_scores=node_scores)
        if selected.size < 2:
            return sim.astype(np.float32)

        if activity is None or activity.ndim != 2 or activity.shape[1] != n_concepts:
            # Fallback for providers without sample-level activity:
            # score-only sparse graph.
            idx = selected
            s = np.asarray(node_scores[idx], dtype=np.float64)
            outer = np.sqrt(np.outer(s, s))
            for ii, i in enumerate(idx):
                for jj, j in enumerate(idx):
                    if ii == jj:
                        continue
                    sim[i, j] = outer[ii, jj]
            np.fill_diagonal(sim, 0.0)
            return self._sparsify_similarity(sim.astype(np.float32))

        act = np.maximum(np.asarray(activity, dtype=np.float64), 0.0)[:, selected]  # [N, K]
        k = int(act.shape[1])
        if int(act.shape[0]) <= 0:
            return sim.astype(np.float32)

        # Focus same-modality co-occurrence on top-contributing samples.
        sample_keep_q = float(np.clip(getattr(self.config, "sample_keep_quantile", 0.5), 0.0, 1.0))
        if 0.0 < sample_keep_q < 1.0 and act.shape[0] > 4:
            s_sel = np.maximum(np.asarray(node_scores[selected], dtype=np.float64), 0.0)
            sample_score = act @ s_sel
            thr = float(np.quantile(sample_score, sample_keep_q))
            keep = sample_score >= thr
            if int(np.sum(keep)) >= 2:
                act = act[keep]

        # Normalize per concept to [0,1] for stable pairwise weighted overlap.
        scale = np.maximum(np.max(act, axis=0, keepdims=True), eps)
        act_n = act / scale
        active_b = act_n > 0.0

        for ii in range(k):
            i = int(selected[ii])
            si = float(node_scores[i])
            if si <= 0.0:
                continue
            for jj in range(ii + 1, k):
                j = int(selected[jj])
                sj = float(node_scores[j])
                if sj <= 0.0:
                    continue

                ai = act_n[:, ii]
                aj = act_n[:, jj]
                same_modality = self.concept_modalities[i] == self.concept_modalities[j]
                if same_modality:
                    bi = active_b[:, ii]
                    bj = active_b[:, jj]
                    union = float(np.sum(np.logical_or(bi, bj)))
                    if union <= 0.0:
                        continue
                    inter = float(np.sum(np.logical_and(bi, bj)))
                    jaccard = inter / union
                    cooccur = inter / float(max(1, act.shape[0]))
                    pair = 0.7 * jaccard + 0.3 * cooccur
                else:
                    # Cross-modality weighted co-activation:
                    # high when both concepts are active in the same molecules
                    # and 3D activity mass aligns with 2D concept presence.
                    num = float(np.mean(np.minimum(ai, aj)))
                    den = float(np.mean(np.maximum(ai, aj)))
                    if den <= eps:
                        continue
                    pair = num / den

                wij = float(pair) * float(np.sqrt(si * sj))
                if wij <= 0.0:
                    continue
                sim[i, j] = wij
                sim[j, i] = wij

        np.fill_diagonal(sim, 0.0)
        mx = float(np.max(sim))
        if mx > eps:
            sim = sim / mx
        return self._sparsify_similarity(sim.astype(np.float32))

    def _selected_concepts(self, *, node_scores: np.ndarray) -> np.ndarray:
        n_concepts = len(self.concept_ids)
        top_k = int(getattr(self.config, "node_top_k_per_task", 0))
        scores = np.asarray(node_scores, dtype=np.float64)
        if top_k <= 0 or top_k >= n_concepts:
            return np.arange(n_concepts, dtype=np.int64)
        k = max(2, top_k)
        idx = np.argpartition(scores, -k)[-k:]
        # keep deterministic ordering by descending score then index
        order = np.argsort(-scores[idx], kind="mergesort")
        return idx[order].astype(np.int64)

    def _sparsify_similarity(self, sim: np.ndarray) -> np.ndarray:
        out = np.asarray(sim, dtype=np.float32).copy()
        np.fill_diagonal(out, 0.0)
        upper = out[np.triu_indices(out.shape[0], k=1)]
        positive = upper[upper > 0.0]
        if positive.size == 0:
            return np.zeros_like(out, dtype=np.float32)

        q = float(np.clip(self.config.edge_keep_quantile, 0.0, 1.0))
        q_thr = float(np.quantile(positive, q))
        thr = max(float(self.config.min_edge_weight), q_thr)
        keep = out >= thr

        k = int(max(0, self.config.top_k_per_node))
        if k > 0:
            n = out.shape[0]
            kk = min(k, max(0, n - 1))
            if kk > 0:
                for i in range(n):
                    row = out[i].copy()
                    row[i] = -np.inf
                    top_idx = np.argpartition(row, -kk)[-kk:]
                    keep[i, top_idx] = True

        keep = np.logical_or(keep, keep.T)
        out = np.where(keep, out, 0.0)
        np.fill_diagonal(out, 0.0)
        return out.astype(np.float32)

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
