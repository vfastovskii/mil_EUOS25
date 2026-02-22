from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Mapping, Optional, Sequence

import numpy as np

from .config import DynamicsConfig
from .types import DynamicsDecomposition, RegimeLabel

logger = logging.getLogger(__name__)


@dataclass
class LinearConceptDynamicsModel:
    """Discrete Λ-Vol-inspired linear dynamics for concept pressure."""

    config: DynamicsConfig
    n_tasks: int
    n_concepts: int
    context_keys: tuple[str, ...] = ()
    task_coupling: Optional[np.ndarray] = None
    concept_coupling: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.n_tasks <= 0 or self.n_concepts <= 0:
            raise ValueError("n_tasks and n_concepts must be positive")
        if self.task_coupling is None:
            self.task_coupling = np.zeros((self.n_tasks, self.n_tasks), dtype=np.float32)
        if self.concept_coupling is None:
            self.concept_coupling = np.zeros((self.n_concepts, self.n_concepts), dtype=np.float32)

        self.task_coupling = np.asarray(self.task_coupling, dtype=np.float32)
        self.concept_coupling = np.asarray(self.concept_coupling, dtype=np.float32)
        if self.task_coupling.shape != (self.n_tasks, self.n_tasks):
            raise ValueError("task_coupling shape mismatch")
        if self.concept_coupling.shape != (self.n_concepts, self.n_concepts):
            raise ValueError("concept_coupling shape mismatch")

    def set_task_coupling(self, matrix: np.ndarray) -> None:
        """Set cross-task coupling matrix C."""
        m = np.asarray(matrix, dtype=np.float32)
        if m.shape != (self.n_tasks, self.n_tasks):
            raise ValueError("task coupling shape mismatch")
        self.task_coupling = m

    def set_concept_coupling(self, matrix: np.ndarray) -> None:
        """Set cross-concept coupling matrix S."""
        m = np.asarray(matrix, dtype=np.float32)
        if m.shape != (self.n_concepts, self.n_concepts):
            raise ValueError("concept coupling shape mismatch")
        self.concept_coupling = m

    def predict_next(
        self,
        *,
        rho: np.ndarray,
        context_covariates: Mapping[str, float],
        regime_label: RegimeLabel,
        prev_drift: Optional[np.ndarray] = None,
        running_mean_rho: Optional[np.ndarray] = None,
    ) -> DynamicsDecomposition:
        """Predict next rho and return contribution decomposition."""
        rho_t = np.asarray(rho, dtype=np.float32)
        if rho_t.shape != (self.n_tasks, self.n_concepts):
            raise ValueError(f"rho shape mismatch: expected {(self.n_tasks, self.n_concepts)}, got {rho_t.shape}")

        drift_prev = np.zeros_like(rho_t, dtype=np.float32) if prev_drift is None else np.asarray(prev_drift, dtype=np.float32)
        if drift_prev.shape != rho_t.shape:
            raise ValueError("prev_drift shape mismatch")

        rho_mean = rho_t if running_mean_rho is None else np.asarray(running_mean_rho, dtype=np.float32)
        if rho_mean.shape != rho_t.shape:
            raise ValueError("running_mean_rho shape mismatch")

        aq = float(self.config.regime_a.get(str(regime_label.value), 0.0))
        regime_core = aq * rho_t

        trend_loop = float(self.config.trend_coeff) * drift_prev
        revert_loop = -float(self.config.revert_coeff) * (rho_t - rho_mean)

        context_scalar = self._context_scalar(context_covariates)
        context_term = np.full_like(rho_t, fill_value=context_scalar, dtype=np.float32)

        task_fb = np.zeros_like(rho_t, dtype=np.float32)
        concept_fb = np.zeros_like(rho_t, dtype=np.float32)
        if bool(self.config.use_cross_task_coupling):
            task_fb = self.task_coupling @ rho_t
        if bool(self.config.use_cross_concept_coupling):
            concept_fb = rho_t @ self.concept_coupling
        feedback = task_fb + concept_fb

        lam = float(self.config.lambda_damping)
        phi = float(self.config.phi_state_damping)
        dissipation = lam * rho_t + phi * (rho_t ** 2)

        delta = regime_core + trend_loop + revert_loop + context_term + feedback - dissipation
        predicted_next = (rho_t + delta).astype(np.float32)

        return DynamicsDecomposition(
            regime_core=regime_core.astype(np.float32),
            feedback=feedback.astype(np.float32),
            trend_loop=trend_loop.astype(np.float32),
            revert_loop=revert_loop.astype(np.float32),
            dissipation=dissipation.astype(np.float32),
            context_term=context_term.astype(np.float32),
            predicted_next=predicted_next,
        )

    def _context_scalar(self, context_covariates: Mapping[str, float]) -> float:
        if not context_covariates:
            return 0.0

        if not self.context_keys:
            self.context_keys = tuple(sorted(str(k) for k in context_covariates.keys()))
        vals = np.asarray([float(context_covariates.get(k, 0.0)) for k in self.context_keys], dtype=np.float64)
        if vals.size == 0:
            return 0.0
        # bounded context projection for stability
        return float(self.config.context_scale * np.tanh(float(np.mean(vals))))

    @staticmethod
    def make_similarity_coupling(similarity: np.ndarray, strength: float = 0.05) -> np.ndarray:
        """Convert similarity matrix to zero-diagonal coupling matrix."""
        s = np.asarray(similarity, dtype=np.float32)
        if s.ndim != 2 or s.shape[0] != s.shape[1]:
            raise ValueError("similarity must be square")
        m = np.asarray(strength, dtype=np.float32) * s
        np.fill_diagonal(m, 0.0)
        return m.astype(np.float32)


__all__ = ["LinearConceptDynamicsModel"]
