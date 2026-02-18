from __future__ import annotations

"""
Stable public interface for the Attention MIL package.

This module exposes public symbols used by downstream code and keeps a
backward-compatible `main()` entrypoint.
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


def main(argv: Any | None = None) -> None:
    """Backward-compatible wrapper around the HPO pipeline entrypoint."""
    _pipeline_main(argv)


if __name__ == "__main__":
    import sys as _sys

    main(_sys.argv[1:])
