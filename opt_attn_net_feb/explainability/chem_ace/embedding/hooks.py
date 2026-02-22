from __future__ import annotations

from contextlib import AbstractContextManager
import logging
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


def resolve_module(model: torch.nn.Module, layer_name: str) -> torch.nn.Module:
    """Resolve nested module by dotted name and return it."""
    parts = [p for p in str(layer_name).split(".") if p]
    module: torch.nn.Module = model
    for part in parts:
        if not hasattr(module, part):
            raise AttributeError(f"Layer path '{layer_name}' not found at '{part}'")
        module = getattr(module, part)
        if not isinstance(module, torch.nn.Module):
            raise TypeError(f"Resolved object for '{layer_name}' is not torch.nn.Module")
    return module


class LayerActivationHook(AbstractContextManager["LayerActivationHook"]):
    """Forward hook context manager for capturing activations at a module."""

    def __init__(self, model: torch.nn.Module, layer_name: str):
        self.model = model
        self.layer_name = layer_name
        self.handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self.last_activation: Optional[torch.Tensor] = None

    def __enter__(self) -> "LayerActivationHook":
        module = resolve_module(self.model, self.layer_name)

        def _hook(_module: torch.nn.Module, _inputs: Any, output: Any) -> None:
            if isinstance(output, (tuple, list)) and len(output) > 0:
                output = output[0]
            if torch.is_tensor(output):
                self.last_activation = output.detach()
            else:
                logger.debug(
                    "Non-tensor hook output encountered",
                    extra={"layer_name": self.layer_name, "type": str(type(output))},
                )
                self.last_activation = None

        self.handle = module.register_forward_hook(_hook)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


__all__ = ["LayerActivationHook", "resolve_module"]
