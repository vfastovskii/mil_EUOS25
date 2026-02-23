from __future__ import annotations

# Torch-dependent imports are optional in lightweight environments.
try:  # pragma: no cover
    from .datasets import MILTrainDataset, MILExportDataset  # noqa: F401
    from .collate import collate_train, collate_export  # noqa: F401
except Exception:  # pragma: no cover
    MILTrainDataset = None  # type: ignore
    MILExportDataset = None  # type: ignore
    collate_train = None  # type: ignore
    collate_export = None  # type: ignore
from .exports import export_leaderboard_attention, export_prediction_text_explanations  # noqa: F401
