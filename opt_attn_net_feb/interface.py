from __future__ import annotations

"""
Stable public interface for the Attention MIL package.

This module exposes public symbols used by downstream code and keeps a
stable `main()` entrypoint.
"""

from typing import Any

from .callbacks import OptunaPruningCallbackLocal  # noqa: F401
from .data import (  # noqa: F401
    MILExportDataset,
    MILTrainDataset,
    collate_export,
    collate_train,
    export_leaderboard_attention,
)
from .entrypoints.hpo_pipeline import main as _pipeline_main
from .losses import MultiTaskFocal  # noqa: F401
from .models import MILTaskAttnMixerWithAux, TaskAttentionPool  # noqa: F401
from .utils import (  # noqa: F401
    AUX_ABS_COLS,
    AUX_FLUO_BASE_COLS,
    NONFEAT_2D,
    NONFEAT_3D,
    NONFEAT_QM,
    TASK_COLS,
    WEIGHT_COLS,
)

__all__ = [
    "MILTaskAttnMixerWithAux",
    "TaskAttentionPool",
    "MultiTaskFocal",
    "MILTrainDataset",
    "MILExportDataset",
    "collate_train",
    "collate_export",
    "export_leaderboard_attention",
    "OptunaPruningCallbackLocal",
    "TASK_COLS",
    "AUX_ABS_COLS",
    "AUX_FLUO_BASE_COLS",
    "WEIGHT_COLS",
    "NONFEAT_2D",
    "NONFEAT_3D",
    "NONFEAT_QM",
    "main",
]


def main(argv: Any | None = None) -> None:
    """Package-level entrypoint delegating to the HPO pipeline."""
    _pipeline_main(argv)


if __name__ == "__main__":
    import sys as _sys

    main(_sys.argv[1:])
