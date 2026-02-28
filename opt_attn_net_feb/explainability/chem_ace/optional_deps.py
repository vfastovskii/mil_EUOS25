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


def has_openbabel_pybel() -> bool:
    """Return True when Open Babel pybel bindings are importable."""
    try:
        import_module("openbabel.pybel")
        return True
    except Exception:
        return False


def has_scipy() -> bool:
    """Return True when scipy is importable."""
    try:
        import_module("scipy")
        return True
    except Exception:
        return False


def has_ripser() -> bool:
    """Return True when ripser is importable."""
    try:
        import_module("ripser")
        return True
    except Exception:
        return False


def has_pmapper() -> bool:
    """Return True when pmapper is importable."""
    try:
        import_module("pmapper")
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


def require_openbabel_pybel() -> ModuleType:
    """Import and return Open Babel pybel module or raise a clear error."""
    return _import_module(
        "openbabel.pybel",
        "conda install -c conda-forge openbabel",
    )


def require_scipy() -> ModuleType:
    """Import and return scipy module or raise a clear error."""
    return _import_module("scipy", "pip install scipy")


def require_ripser() -> ModuleType:
    """Import and return ripser module or raise a clear error."""
    return _import_module("ripser", "pip install ripser")


def require_pmapper() -> ModuleType:
    """Import and return pmapper module or raise a clear error."""
    return _import_module("pmapper", "pip install pmapper")


__all__ = [
    "OptionalDependencyError",
    "has_hdbscan",
    "has_openbabel_pybel",
    "has_pmapper",
    "has_ripser",
    "has_rdkit",
    "has_scipy",
    "require_hdbscan",
    "require_openbabel_pybel",
    "require_pmapper",
    "require_ripser",
    "require_rdkit",
    "require_scipy",
    "require_sqlalchemy",
]
