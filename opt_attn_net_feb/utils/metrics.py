from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from sklearn.metrics import average_precision_score


def ap_per_task(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    w_cls: Optional[np.ndarray] = None,
    weighted_tasks: Tuple[int, ...] = (0, 1),
) -> List[float]:
    """Compute average precision per task with optional sample weights.

    - y_true: int array [B,4] with 0/1 labels
    - p_pred: float array [B,4] with probabilities in [0,1]
    - w_cls: optional float weights [B,4]; applied only to tasks in weighted_tasks
    - weighted_tasks: tasks indices for which to apply sample weights
    """
    y_true = y_true.astype(int)
    p_pred = np.nan_to_num(p_pred, nan=0.0, posinf=1.0, neginf=0.0)
    p_pred = np.clip(p_pred, 0.0, 1.0)

    out: List[float] = []
    for t in range(4):
        if y_true[:, t].sum() == 0:
            out.append(0.0)
            continue
        sw = None
        if w_cls is not None and t in weighted_tasks:
            sw = np.asarray(w_cls[:, t], dtype=float)
        try:
            out.append(float(average_precision_score(y_true[:, t], p_pred[:, t], sample_weight=sw)))
        except ValueError:
            out.append(0.0)
    return out
