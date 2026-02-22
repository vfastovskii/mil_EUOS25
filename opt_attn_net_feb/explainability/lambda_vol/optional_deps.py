from __future__ import annotations

from importlib import import_module
from types import ModuleType


class OptionalDependencyError(RuntimeError):
    """Raised when an optional dependency is unavailable."""


def _import(name: str, hint: str) -> ModuleType:
    try:
        return import_module(name)
    except Exception as exc:  # pragma: no cover
        raise OptionalDependencyError(
            f"Optional dependency '{name}' is required. Install via: {hint}"
        ) from exc


def has_plotly() -> bool:
    try:
        import_module("plotly")
        return True
    except Exception:
        return False


def has_pyvista() -> bool:
    try:
        import_module("pyvista")
        return True
    except Exception:
        return False


def has_pyarrow() -> bool:
    try:
        import_module("pyarrow")
        return True
    except Exception:
        return False


def has_umap() -> bool:
    try:
        import_module("umap")
        return True
    except Exception:
        return False


def require_plotly() -> ModuleType:
    return _import("plotly", "pip install plotly")


def require_pyvista() -> ModuleType:
    return _import("pyvista", "pip install pyvista")


def require_sqlalchemy() -> ModuleType:
    return _import("sqlalchemy", "pip install sqlalchemy")


def require_umap() -> ModuleType:
    return _import("umap", "pip install umap-learn")


__all__ = [
    "OptionalDependencyError",
    "has_plotly",
    "has_pyarrow",
    "has_umap",
    "has_pyvista",
    "require_plotly",
    "require_pyvista",
    "require_sqlalchemy",
    "require_umap",
]
