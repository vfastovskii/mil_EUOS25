from __future__ import annotations

from dataclasses import dataclass, field
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
    """
    Represents the result of a Concept Activation Vector (CAV) fit process.

    This class encapsulates the results of fitting a CAV, including the
    CAV direction vector, the intercept term, and the accuracy of the CAV
    on the training dataset.
    """

    cav_vector: np.ndarray
    intercept: float
    train_accuracy: float


@dataclass(frozen=True)
class TCAVSummary:
    """
    Represents a summary of TCAV (Testing with Concept Activation Vectors) results.

    The class provides a structured representation of TCAV analysis outcome for a given
    concept and task under evaluation, including statistical metrics and associated
    metadata.

    Attributes:
        concept_id: Identifier for the concept analyzed.
        task_id: Identifier for the task related to the analysis.
        layer_name: Name of the neural network layer used for the analysis.
        epoch: Epoch number during which the evaluation is conducted.
        mean_sign_rate: Average sign rate computed for the analysis.
        std_sign_rate: Standard deviation of the sign rate.
        mean_directional_derivative: Average directional derivative value.
        std_directional_derivative: Standard deviation of the directional derivative.
        p_value_mean_sign_rate: Statistical p-value associated with the mean sign rate;
            optional.
        n_repeats: Number of repetitions for the statistical evaluation.
    """

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
    p_value_mean_sign_rate_bonferroni: Optional[float] = None
    is_significant_raw: bool = False
    is_significant_bonferroni: bool = False
    n_eval_samples: int = 0
    repeat_significant_raw_fraction: float = 0.0
    repeat_significant_bonferroni_fraction: float = 0.0
    details: Mapping[str, Any] = field(default_factory=dict)



def _fit_linear_separator(
    *,
    x_concept: np.ndarray,
    x_random: np.ndarray,
    classifier: str,
    seed: int,
    max_iter: int,
) -> CAVFitResult:
    """
    Fits a linear separator (either SVM or Logistic Regression) to distinguish between two groups
    of data: concept and random. Returns the fitted CAV (Concept Activation Vector), model intercept,
    and training accuracy.

    Sections for handling the linear classifier type (e.g., SVM or Logistic Regression) are included,
    with specific preprocessing and computation steps for each classifier type. The method
    normalizes the computed coefficients of the model to generate the CAV. Accuracy of the training is
    also computed by comparing predictions against the ground truth labels.

    Args:
        x_concept (np.ndarray): Positive samples belonging to the concept group.
        x_random (np.ndarray): Negative samples, considered random, for differentiation.
        classifier (str): Type of linear classifier to use ("svm" or "logistic").
        seed (int): Seed for the random state to ensure reproducibility.
        max_iter (int): Maximum number of iterations allowed for training the classifier.

    Returns:
        CAVFitResult: Object that encapsulates the fitted CAV (Concept Activation Vector),
        its intercept, and the training accuracy.

    Raises:
        ValueError: If the provided classifier type is not "svm" or "logistic".
    """
    x_pos = np.asarray(x_concept, dtype=np.float64)
    x_neg = np.asarray(x_random, dtype=np.float64)
    y_pos = np.ones((x_pos.shape[0],), dtype=np.int64)
    y_neg = np.zeros((x_neg.shape[0],), dtype=np.int64)

    x = np.concatenate([x_pos, x_neg], axis=0)
    y = np.concatenate([y_pos, y_neg], axis=0)

    clf = str(classifier).lower().strip()
    if clf == "svm":
        model = LinearSVC(random_state=int(seed), max_iter=int(max_iter))
        model.fit(x, y)
        coef = np.asarray(model.coef_[0], dtype=np.float64)
        intercept = float(model.intercept_[0])
        pred = model.predict(x)
    elif clf in {"logreg", "logistic", "logistic_regression"}:
        model = LogisticRegression(
            random_state=int(seed),
            max_iter=int(max_iter),
            solver="liblinear",
        )
        model.fit(x, y)
        coef = np.asarray(model.coef_[0], dtype=np.float64)
        intercept = float(model.intercept_[0])
        pred = model.predict(x)
    else:
        raise ValueError(f"Unsupported classifier '{classifier}'. Expected one of: svm, logreg")

    norm = float(np.linalg.norm(coef))
    if norm <= 1e-12:
        cav = coef
    else:
        cav = coef / norm
    acc = float(accuracy_score(y, pred))
    return CAVFitResult(cav_vector=cav.astype(np.float32), intercept=intercept, train_accuracy=acc)



def _binom_two_sided_p_value(k: int, n: int, p0: float = 0.5) -> float:
    """
    Calculate the two-sided p-value for a binomial test.

    This function computes the two-sided p-value for a binomial test, which is
    used to determine whether the number of successes in a sequence of n
    independent Bernoulli trials is consistent with a given probability of
    success under the null hypothesis.

    Parameters:
    k : int
        The observed number of successes.
    n : int
        The total number of trials (must be greater than 0 to perform the test).
    p0 : float, optional
        The hypothesized probability of success under the null hypothesis,
        with a default value of 0.5.

    Returns:
    float
        The two-sided p-value for the given binomial test. This value is
        clipped to lie within the range [0.0, 1.0].
    """
    if n <= 0:
        return 1.0
    probs = np.asarray([comb(n, i) * (p0 ** i) * ((1.0 - p0) ** (n - i)) for i in range(n + 1)], dtype=np.float64)
    p_obs = probs[int(k)]
    return float(np.clip(probs[probs <= (p_obs + 1e-18)].sum(), 0.0, 1.0))



def directional_stats(cav_vector: np.ndarray, gradients: np.ndarray) -> tuple[float, float]:
    """
    Calculates directional statistics based on the given Concept Activation Vector (CAV) and
    a set of gradients. The function computes the proportion of gradients having positive
    alignment towards the CAV (sign rate) and the mean projection of the gradients on the CAV.

    Parameters:
    cav_vector (np.ndarray): A Concept Activation Vector (CAV) used for directional alignment
                             calculation. Expected to be a 1-dimensional vector.
    gradients (np.ndarray): A 2-dimensional numpy array representing a set of gradients
                            with shape [N, D], where N is the number of gradient vectors,
                            and D is their dimensionality.

    Returns:
    tuple[float, float]: A tuple containing the following values:
                         - `sign_rate` (float): The proportion of gradients having a
                           positive alignment with the CAV.
                         - `mean_projection` (float): The mean projection value of the gradients
                           on the CAV.

    Raises:
    ValueError: If the gradients array does not have 2 dimensions.
    """
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
    """
    Executes the TCAV (Testing with Concept Activation Vectors) analysis using provided arrays and configuration.

    This function conducts TCAV analysis to understand the impact of conceptual directions on
    a target model's gradient. It involves calculating CAV (Concept Activation Vectors) and
    their respective statistics such as sign rates and directional derivatives. Results are
    stored in form of CAV records, TCAV records, and a summary of the TCAV analysis for the
    concepts, tasks, and layers being analyzed.

    Parameters:
    run_id: Unique identifier for the TCAV run.
    epoch: The epoch number at which TCAV is being calculated.
    concept_id: Identifier for the concept under analysis.
    task_id: Identifier for the task being studied.
    layer_name: Name of the neural network layer being analyzed.
    concept_embeddings: Numpy array of embeddings representing the concept in shape [N, D].
    random_pool_embeddings: Numpy array of random embeddings used as counterexamples in shape [N, D].
    target_gradients: Numpy array of gradients for the target variable in shape [N, D].
    config: Configuration object of type CAVConfig for TCAV analysis.
    seed: Random seed used for reproducibility.

    Returns:
    A tuple containing:
    1. List of CAVRecord objects representing CAV details for each repeat of the analysis.
    2. List of TCAVRecord objects representing TCAV analysis results for each repeat.
    3. A TCAVSummary object summarizing the analysis across all repeats.

    Raises:
    ValueError: If the dimensions of the provided embeddings or gradients do not match the
    expected format or are inconsistent with each other.
    """
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
    n_pos_per_repeat: list[int] = []
    pvals_per_repeat: list[float] = []
    pvals_bonf_per_repeat: list[float] = []
    n_sig_raw = 0
    n_sig_bonf = 0
    repeat_rows: list[dict[str, Any]] = []

    repeats = int(max(1, config.n_random_repeats))
    n_random = int(max(2, config.random_counterexamples_per_repeat))
    alpha = float(np.clip(float(getattr(config, "significance_alpha", 0.05)), 1e-12, 1.0))
    bonf_m = int(max(1, int(getattr(config, "bonferroni_n_hypotheses", 1))))

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
        p_val_bonf = float(min(1.0, float(p_val) * float(bonf_m)))
        sig_raw = bool(float(p_val) < alpha)
        sig_bonf = bool(float(p_val_bonf) < alpha)
        n_sig_raw += int(sig_raw)
        n_sig_bonf += int(sig_bonf)

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
                    "n_eval": int(grads.shape[0]),
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
                metadata={
                    "repeat_idx": int(repeat_idx),
                    "p_value_bonferroni": float(p_val_bonf),
                    "significant_raw": bool(sig_raw),
                    "significant_bonferroni": bool(sig_bonf),
                    "significance_alpha": float(alpha),
                    "bonferroni_n_hypotheses": int(bonf_m),
                },
            )
        )

        sign_rates.append(float(sign_rate))
        mean_dds.append(float(mean_dd))
        n_pos_per_repeat.append(int(n_pos))
        pvals_per_repeat.append(float(p_val))
        pvals_bonf_per_repeat.append(float(p_val_bonf))
        repeat_rows.append(
            {
                "repeat_idx": int(repeat_idx),
                "seed": int(repeat_seed),
                "n_concept_train": int(x_pos.shape[0]),
                "n_random_train": int(x_neg.shape[0]),
                "n_eval": int(grads.shape[0]),
                "tcav_sign_rate": float(sign_rate),
                "tcav_mean_directional_derivative": float(mean_dd),
                "p_value": float(p_val),
                "p_value_bonferroni": float(p_val_bonf),
                "significant_raw": bool(sig_raw),
                "significant_bonferroni": bool(sig_bonf),
            }
        )

    pooled_n = int(len(n_pos_per_repeat) * int(grads.shape[0]))
    pooled_k = int(sum(n_pos_per_repeat))
    pooled_p = _binom_two_sided_p_value(pooled_k, pooled_n, p0=0.5) if pooled_n > 0 else None
    pooled_p_bonf = (
        None
        if pooled_p is None
        else float(min(1.0, float(pooled_p) * float(bonf_m)))
    )
    repeat_sig_raw_frac = float(n_sig_raw / float(max(1, repeats)))
    repeat_sig_bonf_frac = float(n_sig_bonf / float(max(1, repeats)))

    summary = TCAVSummary(
        concept_id=str(concept_id),
        task_id=str(task_id),
        layer_name=str(layer_name),
        epoch=int(epoch),
        mean_sign_rate=float(np.mean(sign_rates)),
        std_sign_rate=float(np.std(sign_rates)),
        mean_directional_derivative=float(np.mean(mean_dds)),
        std_directional_derivative=float(np.std(mean_dds)),
        p_value_mean_sign_rate=(None if pooled_p is None else float(pooled_p)),
        n_repeats=int(len(sign_rates)),
        p_value_mean_sign_rate_bonferroni=pooled_p_bonf,
        is_significant_raw=(False if pooled_p is None else bool(float(pooled_p) < alpha)),
        is_significant_bonferroni=(
            False if pooled_p_bonf is None else bool(float(pooled_p_bonf) < alpha)
        ),
        n_eval_samples=int(grads.shape[0]),
        repeat_significant_raw_fraction=float(repeat_sig_raw_frac),
        repeat_significant_bonferroni_fraction=float(repeat_sig_bonf_frac),
        details={
            "significance_alpha": float(alpha),
            "bonferroni_n_hypotheses": int(bonf_m),
            "repeat_rows": repeat_rows,
            "mean_repeat_p_value": float(np.mean(pvals_per_repeat)) if len(pvals_per_repeat) > 0 else 1.0,
            "mean_repeat_p_value_bonferroni": (
                float(np.mean(pvals_bonf_per_repeat)) if len(pvals_bonf_per_repeat) > 0 else 1.0
            ),
            "n_repeats_significant_raw": int(n_sig_raw),
            "n_repeats_significant_bonferroni": int(n_sig_bonf),
        },
    )
    return cav_records, tcav_records, summary



def _collapse_to_feature_vectors(t) -> np.ndarray:
    """
    Converts a tensor to a 2D feature vector representation.

    This function accepts a tensor in various shapes and converts it to a
    2D array representation by collapsing dimensions appropriately. If the
    tensor has three or more dimensions, it is collapsed into a batch of
    mean feature vectors.

    Parameters:
        t (Tensor): Input tensor to be processed.

    Returns:
        np.ndarray: A 2D array representation of the input tensor. The result
        will be of shape (1, x) if the input was 1D, of shape (b, x) if the
        input was 3D or higher (collapsed along the extra dimensions), or
        unchanged if the input was already 2D.
    """
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
    """
    Collects gradients for a specified layer in the provided model using model inputs.

    This function is intended for gathering gradients to calculate Tensor Concept Activation
    Vectors (TCAV). It works by registering a hook on the requested layer to capture its
    activations, performing a forward pass using the model adapter, and then calculating
    gradients with respect to the model's input(s). The resulting gradients are collapsed
    into feature vectors for further analysis.

    Parameters
    ----------
    model: The model object on which the layer is present and gradients are to be
           collected.
    layer_name: str
        The name of the layer for which gradients need to be captured.
    task_id: str
        The task identifier used by the `adapter` to determine task-specific outputs.
    adapter: ModelTaskAdapter
        Adapter object to abstract operations like forwarding the model and obtaining
        task-specific scalars required for gradient calculation.
    model_inputs: Iterable[Any]
        An iterable containing the inputs to the model. Each input in the iterable can either
        be a dictionary, tuple, list, or tensor.
    device: str, optional
        The device on which the model and inputs should be placed during the gradient
        computation. Defaults to 'cpu'.

    Returns
    -------
    np.ndarray
        Concatenated gradients for the specified layer's activations. If no gradients
        could be collected, returns an empty numpy array with shape (0, 0) and dtype
        float32.

    Raises
    ------
    RuntimeError
        If PyTorch library is not available or the specified layer's activations could
        not be captured.
    TypeError
        If the task scalar obtained from the adapter is not a `torch.Tensor`.
    """
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
