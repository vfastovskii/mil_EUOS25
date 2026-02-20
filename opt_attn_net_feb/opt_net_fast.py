#!/usr/bin/env python3
from __future__ import annotations

"""Project CLI entrypoint for MIL HPO/training pipeline.

This wrapper delegates to:
    opt_attn_net_feb.entrypoints.hpo_pipeline.main
"""

import sys
from pathlib import Path


def _ensure_import_path() -> None:
    # Allow `python opt_net_fast.py` from project root without manual PYTHONPATH edits.
    project_dir = Path(__file__).resolve().parent
    project_parent = project_dir.parent
    if str(project_parent) not in sys.path:
        sys.path.insert(0, str(project_parent))


def main(argv: list[str] | None = None) -> None:
    _ensure_import_path()
    from opt_attn_net_feb.entrypoints.hpo_pipeline import main as pipeline_main

    pipeline_main(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
