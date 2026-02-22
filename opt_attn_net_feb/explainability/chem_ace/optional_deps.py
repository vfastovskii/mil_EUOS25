from __future__ import annotations

from importlib import import_module
from types import ModuleType


class OptionalDependencyError(RuntimeError):
    """Raised when an optional third-party dependency is unavailable."""


def _import_module(module_name: str, install_hint: str) -> ModuleType:
    try:
        return import_module(module_name)
    except Exception as exc:  # pragma: no cover - dependency runtime behavior
        raise OptionalDependencyError(
            f"Optional dependency '{module_name}' is required. Install via: {install_hint}"
        ) from exc


def has_rdkit() -> bool:
    """Return True when RDKit is importable."""
    try:
        import_module("rdkit")
        return True
    except Exception:
        return False


def has_hdbscan() -> bool:
    """Return True when hdbscan is importable."""
    try:
        import_module("hdbscan")
        return True
    except Exception:
        return False


def require_rdkit() -> ModuleType:
    """Import and return the RDKit root module or raise a clear error."""
    return _import_module("rdkit", "pip install rdkit")


def require_hdbscan() -> ModuleType:
    """Import and return hdbscan module or raise a clear error."""
    return _import_module("hdbscan", "pip install hdbscan")


def require_sqlalchemy() -> ModuleType:
    """Import and return sqlalchemy module or raise a clear error."""
    return _import_module("sqlalchemy", "pip install sqlalchemy")


__all__ = [
    "OptionalDependencyError",
    "has_hdbscan",
    "has_rdkit",
    "require_hdbscan",
    "require_rdkit",
    "require_sqlalchemy",
]
