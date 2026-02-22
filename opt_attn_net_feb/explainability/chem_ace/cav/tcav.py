from __future__ import annotations

from dataclasses import dataclass
import logging
from math import comb
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
try:
    import torch
except Exception:  # pragma: no cover - optional runtime dependency
    torch = None  # type: ignore

from ..config import CAVConfig
from ..types import CAVRecord, ModelTaskAdapter, TCAVRecord

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CAVFitResult:
    """Container for fitted CAV direction and train statistics."""

    cav_vector: np.ndarray
    intercept: float
    train_accuracy: float


@dataclass(frozen=True)
class TCAVSummary:
    """Aggregate summary across repeated CAV runs."""

    concept_id: str
    task_id: str
    layer_name: str
    epoch: int
    mean_sign_rate: float
    std_sign_rate: float
    mean_directional_derivative: float
    std_directional_derivative: float
    p_value_mean_sign_rate: Optional[float]
    n_repeats: int



def _fit_linear_separator(
    *,
    x_concept: np.ndarray,
    x_random: np.ndarray,
    classifier: str,
    seed: int,
    max_iter: int,
) -> CAVFitResult:
    x_pos = np.asarray(x_concept, dtype=np.float64)
    x_neg = np.asarray(x_random, dtype=np.float64)
    y_pos = np.ones((x_pos.shape[0],), dtype=np.int64)
    y_neg = np.zeros((x_neg.shape[0],), dtype=np.int64)

    x = np.concatenate([x_pos, x_neg], axis=0)
    y = np.concatenate([y_pos, y_neg], axis=0)

    if str(classifier).lower() == "svm":
        model = LinearSVC(random_state=int(seed), max_iter=int(max_iter))
        model.fit(x, y)
        coef = np.asarray(model.coef_[0], dtype=np.float64)
        intercept = float(model.intercept_[0])
        pred = model.predict(x)
    else:
        model = LogisticRegression(
            random_state=int(seed),
            max_iter=int(max_iter),
            solver="liblinear",
        )
        model.fit(x, y)
        coef = np.asarray(model.coef_[0], dtype=np.float64)
        intercept = float(model.intercept_[0])
        pred = model.predict(x)

    norm = float(np.linalg.norm(coef))
    if norm <= 1e-12:
        cav = coef
    else:
        cav = coef / norm
    acc = float(accuracy_score(y, pred))
    return CAVFitResult(cav_vector=cav.astype(np.float32), intercept=intercept, train_accuracy=acc)



def _binom_two_sided_p_value(k: int, n: int, p0: float = 0.5) -> float:
    """Exact two-sided binomial p-value without scipy dependency."""
    if n <= 0:
        return 1.0
    probs = np.asarray([comb(n, i) * (p0 ** i) * ((1.0 - p0) ** (n - i)) for i in range(n + 1)], dtype=np.float64)
    p_obs = probs[int(k)]
    return float(np.clip(probs[probs <= (p_obs + 1e-18)].sum(), 0.0, 1.0))



def directional_stats(cav_vector: np.ndarray, gradients: np.ndarray) -> tuple[float, float]:
    """Compute TCAV sign-rate and mean directional derivative from gradients."""
    cav = np.asarray(cav_vector, dtype=np.float64)
    g = np.asarray(gradients, dtype=np.float64)
    if g.ndim != 2:
        raise ValueError(f"Expected gradients shape [N,D], got {g.shape}")
    dd = g @ cav.reshape(-1)
    sign_rate = float(np.mean(dd > 0.0))
    return sign_rate, float(np.mean(dd))



def run_tcav_from_arrays(
    *,
    run_id: str,
    epoch: int,
    concept_id: str,
    task_id: str,
    layer_name: str,
    concept_embeddings: np.ndarray,
    random_pool_embeddings: np.ndarray,
    target_gradients: np.ndarray,
    config: CAVConfig,
    seed: int,
) -> tuple[list[CAVRecord], list[TCAVRecord], TCAVSummary]:
    """Run repeated CAV/TCAV from activation and gradient arrays."""
    rng = np.random.default_rng(int(seed))

    x_pos = np.asarray(concept_embeddings, dtype=np.float32)
    x_pool = np.asarray(random_pool_embeddings, dtype=np.float32)
    grads = np.asarray(target_gradients, dtype=np.float32)

    if x_pos.ndim != 2:
        raise ValueError(f"concept_embeddings must be [N,D], got {x_pos.shape}")
    if x_pool.ndim != 2:
        raise ValueError(f"random_pool_embeddings must be [N,D], got {x_pool.shape}")
    if grads.ndim != 2:
        raise ValueError(f"target_gradients must be [N,D], got {grads.shape}")
    if x_pos.shape[1] != x_pool.shape[1] or x_pos.shape[1] != grads.shape[1]:
        raise ValueError("Embedding and gradient dimensions must match")

    cav_records: list[CAVRecord] = []
    tcav_records: list[TCAVRecord] = []
    sign_rates: list[float] = []
    mean_dds: list[float] = []

    repeats = int(max(1, config.n_random_repeats))
    n_random = int(max(2, config.random_counterexamples_per_repeat))

    for repeat_idx in range(repeats):
        repeat_seed = int(seed) + int(repeat_idx)
        sample_idx = rng.choice(np.arange(x_pool.shape[0]), size=min(n_random, x_pool.shape[0]), replace=(x_pool.shape[0] < n_random))
        x_neg = x_pool[sample_idx]

        fit = _fit_linear_separator(
            x_concept=x_pos,
            x_random=x_neg,
            classifier=config.classifier,
            seed=repeat_seed,
            max_iter=config.max_iter,
        )

        sign_rate, mean_dd = directional_stats(fit.cav_vector, grads)
        n_pos = int(np.sum((grads @ fit.cav_vector.reshape(-1)) > 0.0))
        p_val = _binom_two_sided_p_value(n_pos, int(grads.shape[0]), p0=0.5)

        cav_records.append(
            CAVRecord(
                concept_id=concept_id,
                task_id=task_id,
                layer_name=layer_name,
                seed=repeat_seed,
                cav_vector=fit.cav_vector,
                intercept=float(fit.intercept),
                train_accuracy=float(fit.train_accuracy),
                metadata={
                    "repeat_idx": int(repeat_idx),
                    "n_concept": int(x_pos.shape[0]),
                    "n_random": int(x_neg.shape[0]),
                    "classifier": str(config.classifier),
                },
            )
        )

        tcav_records.append(
            TCAVRecord(
                run_id=str(run_id),
                epoch=int(epoch),
                concept_id=str(concept_id),
                task_id=str(task_id),
                layer_name=str(layer_name),
                seed=repeat_seed,
                tcav_sign_rate=float(sign_rate),
                tcav_mean_directional_derivative=float(mean_dd),
                n_samples=int(grads.shape[0]),
                p_value=float(p_val),
                metadata={"repeat_idx": int(repeat_idx)},
            )
        )

        sign_rates.append(float(sign_rate))
        mean_dds.append(float(mean_dd))

    summary = TCAVSummary(
        concept_id=str(concept_id),
        task_id=str(task_id),
        layer_name=str(layer_name),
        epoch=int(epoch),
        mean_sign_rate=float(np.mean(sign_rates)),
        std_sign_rate=float(np.std(sign_rates)),
        mean_directional_derivative=float(np.mean(mean_dds)),
        std_directional_derivative=float(np.std(mean_dds)),
        p_value_mean_sign_rate=float(np.mean([r.p_value for r in tcav_records if r.p_value is not None])) if tcav_records else None,
        n_repeats=int(len(sign_rates)),
    )
    return cav_records, tcav_records, summary



def _collapse_to_feature_vectors(t) -> np.ndarray:
    arr = t.detach().cpu().float().numpy()
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        return arr
    b = arr.shape[0]
    return arr.reshape(b, -1, arr.shape[-1]).mean(axis=1)



def collect_gradients_for_layer(
    *,
    model,
    layer_name: str,
    task_id: str,
    adapter: ModelTaskAdapter,
    model_inputs: Iterable[Any],
    device: str = "cpu",
) -> np.ndarray:
    """Collect d(task)/d(h) vectors for inputs, projected to feature axis."""
    if torch is None:
        raise RuntimeError("PyTorch is required for collect_gradients_for_layer")

    from ..embedding.hooks import LayerActivationHook

    dev = torch.device(device)
    model = model.to(dev)
    model.train(False)

    collected: list[np.ndarray] = []
    for model_input in model_inputs:
        model.zero_grad(set_to_none=True)
        if isinstance(model_input, dict):
            model_input = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in model_input.items()}
        elif isinstance(model_input, (tuple, list)):
            model_input = tuple(v.to(dev) if torch.is_tensor(v) else v for v in model_input)
        elif torch.is_tensor(model_input):
            model_input = model_input.to(dev)

        with LayerActivationHook(model, layer_name) as hook:
            output = adapter.forward(model_input)
            task_scalar = adapter.get_task_scalar(output, task_id)
            if torch.is_tensor(task_scalar) and task_scalar.ndim > 0:
                task_scalar = task_scalar.sum()
            if not torch.is_tensor(task_scalar):
                raise TypeError("adapter.get_task_scalar must return a torch.Tensor")
            if hook.last_activation is None:
                raise RuntimeError(f"No activation captured for layer '{layer_name}'")
            hook.last_activation.retain_grad()
            task_scalar.backward()
            grad = hook.last_activation.grad
            if grad is None:
                raise RuntimeError("Failed to collect activation gradients for TCAV")
            collected.append(_collapse_to_feature_vectors(grad))

    if not collected:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(collected, axis=0).astype(np.float32)


__all__ = [
    "CAVFitResult",
    "TCAVSummary",
    "collect_gradients_for_layer",
    "directional_stats",
    "run_tcav_from_arrays",
]
