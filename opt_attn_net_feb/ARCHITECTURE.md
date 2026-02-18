# Package Architecture

This document describes both code layering and model design for the current
MIL attention system.

## 1) Code Layers

```text
opt_attn_net_feb/
  callbacks/        # Training callbacks
  data/             # Datasets, collate functions, export helpers
  entrypoints/      # CLI/orchestration
  losses/           # Loss implementations
  models/           # Neural architecture
  training/         # HPO, trainer construction, final training/export
  utils/            # Data/metrics/ops/constants helpers
  interface.py      # Stable facade entrypoint
```

Dependency direction:

- `entrypoints` -> `training`, `data`, `models`, `utils`
- `training` -> `models`, `data`, `losses`, `utils`
- `models`, `losses`, `data`, `utils` do not depend on `entrypoints`

Canonical imports:

- `opt_attn_net_feb.training`
- `opt_attn_net_feb.models`
- `opt_attn_net_feb.interface`

## 2) Model Architecture (`MILTaskAttnMixerWithAux`)

Primary implementation:

- `models/mil_task_attn_mixer/model.py`
- `models/task_attention/pool.py`
- `models/mil_task_attn_mixer/heads.py`

### 2.1 End-to-end flow

```text
Input:
  x2d        [B, F2]
  x3d_pad    [B, N, F3]
  kpm        [B, N]   (True = padding)

Branch A (molecule / 2D):
  x2d -> mol_enc MLP -> [B, mol_hidden]
      -> proj2d (Linear + LayerNorm) -> e2d [B, proj_dim]
      -> repeat per task -> e2d_rep [B, 4, proj_dim]

Branch B (instance / 3D):
  x3d_pad -> inst_enc MLP per instance -> [B, N, inst_hidden]
          -> LayerNorm
          -> TaskAttentionPool (4 learned queries, MHA)
          -> pooled_tasks [B, 4, inst_hidden]
          -> proj3d (Linear + LayerNorm) -> e3d [B, 4, proj_dim]

Fusion + task representation:
  concat(e2d_rep, e3d) -> [B, 4, 2*proj_dim]
  mixer MLP -> z_tasks [B, 4, mixer_hidden]

Heads:
  Classification heads (per-task linear):
    z_tasks -> logits [B, 4]

  Auxiliary shared heads (from mean task embedding):
    z_aux = mean(z_tasks over task axis) [B, mixer_hidden]
    z_aux -> abs_heads  -> abs_out  [B, 2]
    z_aux -> fluo_heads -> fluo_out [B, 4]

Optional explainability output:
  attention maps attn [B, 4, N] (mask-aware, renormalized)
```

### 2.2 Embedders

Both `mol_enc` and `inst_enc` use `utils.mlp.make_mlp`:

- Repeated block: `Linear -> LayerNorm -> Activation -> Dropout`
- Activation options: `GELU`, `SiLU`, `Mish`, `ReLU`, `LeakyReLU`
- Hidden width is constant across layers within each MLP

### 2.3 Pooling (task-specific attention)

`TaskAttentionPool` uses multi-query attention:

- Learned query tensor: `[1, n_tasks, dim]` (expanded to batch)
- `nn.MultiheadAttention(batch_first=True)` with `attn_heads`
- Per-head attention is averaged (when available), then:
  - padding positions masked to zero
  - renormalized so each task attention sums to 1 over valid instances

Constraint:

- `inst_hidden % attn_heads == 0` (enforced during HPO)

### 2.4 Multitask outputs

Task counts:

- Classification tasks: `4`
- Auxiliary absorbance heads: `2`
- Auxiliary fluorescence heads: `4`

Classification task names (`utils.constants.TASK_COLS`):

- `Transmittance_340`
- `Transmittance_450`
- `Fluorescence_340_450`
- `Fluorescence_more_than_480`

Auxiliary absorbance targets (`AUX_ABS_COLS`):

- `Transmittance_340_quantitative`
- `Transmittance_450_quantitative`

Auxiliary fluorescence is built from base columns `wl_pred_nm` and `qy_pred`,
expanded to 4 outputs to align with task structure.

## 3) Losses and Training Objective

Primary loss assembly: `models/mil_task_attn_mixer/training.py`

### 3.1 Classification loss (`MultiTaskFocal`)

`losses/multi_task_focal.py`:

- Base: `BCEWithLogits` with per-task `pos_weight`
- Focal factor: `(1 - pt) ^ gamma_t` per task
- Sample/task weighting via `w_cls`
- Returns per-task loss vector `[4]`

### 3.2 Regression losses (`reg_loss_weighted`)

`losses/regression.py`:

- Supports `smoothl1` or `mse`
- Applies boolean masks to ignore missing labels
- Uses per-output weights and stable denominator clamping

### 3.3 Total objective

For a batch:

- `per_task_cls = MultiTaskFocal(...)` -> `[4]`
- `loss_cls = mean(lam * per_task_cls)`
- `loss_abs = reg_loss_weighted(abs_out, y_abs, m_abs, w_abs, reg_loss_type)`
- `loss_fluo = reg_loss_weighted(fluo_out, y_fluo, m_fluo, w_fluo, reg_loss_type)`
- `loss_total = loss_cls + lambda_aux_abs * loss_abs + lambda_aux_fluo * loss_fluo`

Validation metrics:

- Per-task AP
- `val_macro_ap` (mean AP over 4 tasks)
- `val_min_ap` (worst-task AP)

## 4) Hyperparameters

HPO search space is defined in `training/hpo.py::search_space`.

Architecture/search parameters:

- `mol_hidden`: `{1024, 2048}`
- `mol_layers`: `[2, 5]`
- `mol_dropout`: `[0.10, 0.25]`
- `inst_hidden`: `{256, 512, 1024}`
- `inst_layers`: `[3, 5]`
- `inst_dropout`: `[0.05, 0.15]`
- `proj_dim`: `{512, 1024}`
- `attn_heads`: `{8, 16}`
- `attn_dropout`: `[0.05, 0.2]`
- `mixer_hidden`: `{512, 1024}`
- `mixer_layers`: `[3, 5]`
- `mixer_dropout`: `[0.05, 0.2]`
- `activation`: `{GELU, SiLU, Mish, ReLU, LeakyReLU}`

Optimization/training parameters:

- `lr`: `[8e-5, 8e-4]` (log scale)
- `weight_decay`: `[3e-6, 3e-4]` (log scale)
- `batch_size`: `{128, 256, 512}`
- `accumulate_grad_batches`: `{8, 16}`

Class imbalance / weighting parameters:

- `posw_clip_t0..t3`: each `[10, 200]` (log scale)
- `gamma_t0..t3`: each `[0, 4]`
- `lam_t0..t3`: each `[0.25, 3.0]` (log scale), then normalized/clipped
- `lam_floor`: `[0.25, 1.0]`
- `lam_ceil`: `[1.0, 3.5]`
- `rare_oversample_mult`: `[0, 200]`
- `rare_prev_thr`: `[0.005, 0.05]`
- `sample_weight_cap`: `[2.0, 20.0]`

Auxiliary loss parameters:

- `lambda_aux_abs`: `[0.05, 0.5]`
- `lambda_aux_fluo`: `[0.05, 0.5]`
- `reg_loss_type`: `{smoothl1, mse}`

HPO objective parameters:

- `objective_mode`: `{macro, macro_plus_min}`
- `min_w`: `[0.1, 0.6]` (used when `macro_plus_min`)

## 5) Training Behavior (non-search)

- Early stopping + checkpoint monitor: `val_macro_ap`
- Deterministic trainer setup is enabled
- Optuna pruning callback is integrated for CV trials
- Auxiliary targets are standardized using train-fold statistics only
- Final training stage retrains best config and can export leaderboard attention

## 6) Guardrails

- No compatibility alias modules for removed paths
- No duplicated implementations across layers
- No wildcard exports (`import *`) in package surfaces
