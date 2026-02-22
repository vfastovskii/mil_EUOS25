from __future__ import annotations

from typing import Sequence


def run_hpo_main(argv: Sequence[str] | None = None) -> None:
    """Lazy wrapper for HPO pipeline entrypoint."""
    from .hpo_pipeline import main

    main(argv)


def run_lambda_vol_demo(argv: Sequence[str] | None = None) -> None:
    """Lazy wrapper for Lambda-Vol demo entrypoint."""
    from .lambda_vol_demo import main

    main(argv)


__all__ = ["run_hpo_main", "run_lambda_vol_demo"]
