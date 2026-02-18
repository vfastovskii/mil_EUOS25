from __future__ import annotations

from .task_attention import TaskAttentionPool as _TaskAttentionPool


class TaskAttentionPool(_TaskAttentionPool):
    """Backward-compatible import path wrapper."""


__all__ = ["TaskAttentionPool"]
