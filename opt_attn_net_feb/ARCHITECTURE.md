# Package Architecture

This package is organized by responsibility and import stability.

## Directory Layout

```text
opt_attn_net_feb/
  callbacks/        # Training callbacks (Optuna pruning, etc.)
  data/             # Datasets, collate fns, export helpers
  entrypoints/      # CLI/orchestration entrypoints
  losses/           # Loss functions
  models/           # Model architectures and attention components
  train/            # Backward-compatible wrappers (legacy import paths)
  training/         # Core training/HPO implementation
  utils/            # Shared constants and utilities
  interface.py      # Stable public facade + backward-compatible main()
```

## Layer Boundaries

- `entrypoints/` may depend on `training/`, `data/`, `models/`, `utils/`.
- `training/` may depend on `models/`, `data/`, `losses/`, `utils/`.
- `models/`, `losses/`, `data/`, and `utils/` should avoid depending on `entrypoints/`.
- `train/` should remain thin re-export wrappers only.

## Compatibility Policy

- New code should import training code from `training.*`.
- Legacy imports from `train.*` remain functional via wrappers.
- External users should rely on `interface.py` or package-level exports for stable APIs.
