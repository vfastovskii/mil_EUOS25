from __future__ import annotations

from importlib import import_module
from typing import Any

# Keep stable package-level names, but resolve them lazily so importing the package
# does not immediately require heavy optional deps (e.g., torch).
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "MILTaskAttnMixerWithAux": ("models", "MILTaskAttnMixerWithAux"),
    "TaskAttentionPool": ("models", "TaskAttentionPool"),
    "MultiTaskFocal": ("losses", "MultiTaskFocal"),
    "MILTrainDataset": ("data", "MILTrainDataset"),
    "MILExportDataset": ("data", "MILExportDataset"),
    "collate_train": ("data", "collate_train"),
    "collate_export": ("data", "collate_export"),
    "export_leaderboard_attention": ("data", "export_leaderboard_attention"),
    "export_prediction_text_explanations": ("data", "export_prediction_text_explanations"),
    "OptunaPruningCallbackLocal": ("callbacks", "OptunaPruningCallbackLocal"),
    "TASK_COLS": ("utils", "TASK_COLS"),
    "AUX_ABS_COLS": ("utils", "AUX_ABS_COLS"),
    "AUX_FLUO_BASE_COLS": ("utils", "AUX_FLUO_BASE_COLS"),
    "WEIGHT_COLS": ("utils", "WEIGHT_COLS"),
    "NONFEAT_2D": ("utils", "NONFEAT_2D"),
    "NONFEAT_3D": ("utils", "NONFEAT_3D"),
    "NONFEAT_QM": ("utils", "NONFEAT_QM"),
}

__all__ = [
    "MILTaskAttnMixerWithAux",
    "TaskAttentionPool",
    "MultiTaskFocal",
    "MILTrainDataset",
    "MILExportDataset",
    "collate_train",
    "collate_export",
    "export_leaderboard_attention",
    "export_prediction_text_explanations",
    "OptunaPruningCallbackLocal",
    "TASK_COLS",
    "AUX_ABS_COLS",
    "AUX_FLUO_BASE_COLS",
    "WEIGHT_COLS",
    "NONFEAT_2D",
    "NONFEAT_3D",
    "NONFEAT_QM",
]


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(str(name))
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, symbol = target
    mod = import_module(f".{module_name}", package=__name__)
    value = getattr(mod, symbol)
    globals()[name] = value
    return value
