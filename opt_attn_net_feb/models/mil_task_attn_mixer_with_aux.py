from __future__ import annotations

from .mil_task_attn_mixer import MILTaskAttnMixerWithAux as _MILTaskAttnMixerWithAux


class MILTaskAttnMixerWithAux(_MILTaskAttnMixerWithAux):
    """Backward-compatible import path wrapper."""


__all__ = ["MILTaskAttnMixerWithAux"]
