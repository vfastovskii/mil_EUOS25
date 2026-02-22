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
  __init__.py       # Stable package-level public API exports
  opt_net_fast.py   # Single project CLI entrypoint (root)
```

Dependency direction:

- `entrypoints` -> `training`, `data`, `models`, `utils`
- `training` -> `models`, `data`, `losses`, `utils`
- `models`, `losses`, `data`, `utils` do not depend on `entrypoints`

Canonical imports:

- `opt_attn_net_feb.training`
- `opt_attn_net_feb.models`
- `opt_attn_net_feb`

### 1.1 Configuration contracts

Large parameter groups are passed as typed dataclasses instead of ad-hoc dicts:

- Training-level config (`training/configs.py`):
  - `HPOConfig` with grouped sub-configs:
    - `BackboneConfig`
    - `HeadConfig`
    - `OptimizationConfig`
    - `RuntimeConfig`
    - `SamplerConfig`
    - `LossWeightingConfig`
    - `ObjectiveConfig`
- Model-level config (`models/multimodal_mil/configs.py`):
  - `MILModelConfig` with grouped sub-configs:
    - `MILBackboneConfig`
    - `MILPredictorConfig`
    - `MILOptimizationConfig`
    - `MILLossConfig`

Flow:

- `search_space` (flat Optuna dict) -> `HPOConfig.from_params(...)`
- `MILModelBuilder.build(config=HPOConfig, ...)` maps to `MILModelConfig`
- `MILTaskAttnMixerWithAux.from_config(...)` constructs the model from one structured object + loss tensors (`pos_weight`, `gamma`, `lam`)

### 1.2 Class-based training orchestration

Core orchestration now follows explicit classes and typed handoff objects:

- Entry pipeline (`entrypoints/hpo_pipeline.py`)
  - `PipelineConfigFactory` parses CLI args into `PipelineConfig`
  - `PipelineEnvironmentFactory` resolves output/runtime environment
  - `HPODataBuilder` builds `PreparedHPOData` and `MILCVData`
  - `MILPipelineOrchestrator` runs HPO and final export stages
- Execution layer (`training/execution.py`)
  - `MILCrossValidator` owns Optuna objective flow
  - `MILFoldTrainer` handles one CV fold train/eval cycle
  - `MILStudyRunner` owns Optuna study lifecycle + artifact writes
  - `MILFinalTrainer` runs best-config train and leaderboard export
- Trainer/eval infrastructure (`training/trainer.py`)
  - `LightningTrainerFactory` builds deterministic Lightning trainers
  - `ModelEvaluator` computes AP metrics from checkpoints

Config handoff chain:

- CLI args -> `PipelineConfig`
- `PipelineConfig` + prepared arrays -> `MILCVData`
- `MILCVData` + `CVRunConfig` -> `MILCrossValidator`
- Best params + `MILFinalData` + `FinalTrainConfig` -> `MILFinalTrainer`

## 2) Model Architecture (`MILTaskAttnMixerWithAux`)

Primary implementation:

- `models/multimodal_mil/model.py`
- `models/multimodal_mil/configs.py`
- `models/multimodal_mil/embedders.py`
- `models/multimodal_mil/embedder_mlp_v3_base.py`
- `models/multimodal_mil/aggregators.py`
- `models/multimodal_mil/predictors.py`
- `models/multimodal_mil/head_mlp_v3.py`
- `models/multimodal_mil/head_utils.py`
- `models/attention_pooling/pool.py`
- `models/multimodal_mil/heads.py`

### 2.0 Componentized structure (extensible by name)

Model assembly is now explicit and layered:

- `embedder`:
  - `2D` embedder registry in `models/multimodal_mil/embedders.py`
  - `3D` embedder registry in `models/multimodal_mil/embedders.py`
- `aggregator`:
  - registry in `models/multimodal_mil/aggregators.py`
  - current default: `task_attention_pool`
- `predictor`:
  - head-builder registry in `models/multimodal_mil/predictors.py`
  - current default: `mlp_v3`

Selection is name-based in model config:

- `mol_embedder_name`
- `inst_embedder_name`
- `aggregator_name`
- `predictor_name`

Canonical builder entrypoints:

- `build_2d_embedder` / `build_3d_embedder` -> `models/multimodal_mil/embedders.py`
- `build_aggregator` -> `models/multimodal_mil/aggregators.py`
- `build_predictor_heads` -> `models/multimodal_mil/predictors.py`
- mixer builder used by model -> `models/multimodal_mil/embedder_mlp_v3_base.py`

Concrete default implementations are also split per concern:

- shared V3 embedder implementation for both 2D and 3D -> `models/multimodal_mil/embedder_mlp_v3_base.py`
- predictor-head implementation -> `models/multimodal_mil/head_mlp_v3.py`
- head utilities (projection + apply helpers) -> `models/multimodal_mil/head_utils.py`

Compatibility facade:

- `models/multimodal_mil/heads.py` re-exports head symbols to keep import stability while implementation remains split.

Current defaults preserve existing behavior:

- `mol_embedder_name="mlp_v3_2d"`
- `inst_embedder_name="mlp_v3_3d"`
- `aggregator_name="task_attention_pool"`
- `predictor_name="mlp_v3"`

Design rule:

- 2D modality is molecule-level (`[B, F2]`) and does not use an aggregator.
- 3D modality is instance-level (`[B, N, F3]`) and must pass through an aggregator to produce per-task pooled representation.

Extension workflow:

- Add new 2D embedder:
  - implement a builder with signature `(input_dim, hidden_dim, layers, dropout, activation)`
  - register it via `register_2d_embedder("name", builder)`
- Add new 3D embedder:
  - implement a builder with the same signature
  - register it via `register_3d_embedder("name", builder)`
- Add new aggregator:
  - implement a builder with signature `(dim, n_heads, dropout, n_tasks, **kwargs)`
  - register it via `register_aggregator("name", builder)`
- Add new predictor family:
  - implement a builder with signature `(in_dim, count, activation, num_layers, dropout, stochastic_depth, fc2_gain_non_last)`
  - register it via `register_predictor("name", builder)`

Compatibility note:

- Legacy embedder value `mlp_v3` is accepted and resolved internally as:
  - 2D: `mlp_v3_2d`
  - 3D: `mlp_v3_3d`

### 2.1 End-to-end flow

```text
Input:
  x2d        [B, F2]
  x3d_pad    [B, N, F3]
  kpm        [B, N]   (True = padding)

Branch A (molecule / 2D):
  x2d -> 2D embedder (by name) -> [B, mol_hidden]
      -> LayerNorm (post-embedder normalization)
      -> proj2d (Linear + LayerNorm) -> e2d [B, proj_dim]
      -> repeat per task -> e2d_rep [B, 4, proj_dim]

Branch B (instance / 3D):
  x3d_pad -> 3D embedder (by name) per instance -> [B, N, inst_hidden]
          -> LayerNorm (post-embedder normalization, before aggregator)
          -> aggregator (by name; current TaskAttentionPool)
          -> pooled_tasks [B, 4, inst_hidden]
          -> LayerNorm (post-aggregator normalization)
          -> proj3d (Linear + LayerNorm) -> e3d [B, 4, proj_dim]

Fusion + task representation:
  concat(e2d_rep, e3d) -> [B, 4, 2*proj_dim]
  mixer residual MLP (V3-like, same logic family as embedders) -> z_tasks [B, 4, mixer_hidden]
  LayerNorm (post-mixer normalization, before predictors)

Heads:
  Classification heads (per-task residual MLP predictor):
    z_tasks -> logits [B, 4]

  Auxiliary shared heads (residual MLP predictors from mean task embedding):
    z_aux = mean(z_tasks over task axis) [B, mixer_hidden]
    z_aux -> abs_heads  -> abs_out  [B, 2]
    z_aux -> fluo_heads -> fluo_out [B, 4]

Optional explainability output:
  attention maps attn [B, 4, N] (mask-aware, renormalized)
```

### 2.2 Embedders

`mol_enc`, `inst_enc`, and `mixer` use `utils.mlp.make_residual_mlp_embedder_v3`
(`MLPEmbedderV3Like`):

- Input normalization + optional projection to hidden width
- Stack of residual FFN blocks with:
  - pre-norm layout
  - SwiGLU-style gated FFN
  - residual dropout
  - stochastic depth (DropPath) across depth
  - learnable residual scaling (with warmup for early blocks)
- Last block FF2 is zero-initialized for near-identity start
- Hidden width is constant across residual blocks

### 2.3 Pooling (task-specific attention)

`TaskAttentionPool` uses multi-query attention:

- Learned query tensor: `[1, n_tasks, dim]` (expanded to batch)
- `nn.MultiheadAttention(batch_first=True)` with `attn_heads`
- Pre-layer normalization before attention (V4-style)
- Optional query temperature scaling
- Per-head attention is averaged, then:
  - padding positions masked to zero
  - renormalized so each task attention sums to 1 over valid instances
- Pooling source can be configured (`inputs`, `normed_inputs`, or `attn_out`)
- Value projection can be tied to MHA V-projection (`tie_mha_v`)
- Supports top-k attention pooling and optional residual blend with mean pooling

Constraint:

- `inst_hidden % attn_heads == 0` (enforced during HPO)

### 2.4 Multitask outputs

Task counts:

- Classification tasks: `4`
- Auxiliary absorbance heads: `2`
- Auxiliary fluorescence heads: `4`
- Auxiliary bitmask-group head: `K+1` classes (`K` train-fold top masks + `other`)

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

Head architecture note:

- Heads are selected by predictor name (current: V3-like residual MLP predictors; not plain linear layers)
- Each head uses residual FFN blocks with SwiGLU, LayerNorm, DropPath, and learnable residual scaling
- Final scalar output per head is produced by a small-gain linear output layer
- Bitmask-group head (when enabled) uses the same V3 residual predictor style and outputs multi-class logits

## 3) Losses and Training Objective

Primary loss assembly: `models/multimodal_mil/training.py`

### 3.1 Classification loss (`MultiTaskFocal`)

`losses/multi_task_focal.py`:

- Base: `BCEWithLogits` with per-task `pos_weight`
- Focal factor: `(1 - pt) ^ gamma_t` per task
- Sample/task weighting via `w_cls`
- Returns per-task loss vector `[4]`

### 3.2 Regression losses (`reg_loss_weighted`)

`losses/regression.py`:

- Supports `mse`
- Applies boolean masks to ignore missing labels
- Uses per-output weights and stable denominator clamping

### 3.3 Total objective

For a batch:

- `per_task_cls = MultiTaskFocal(...)` -> `[4]`
- `loss_cls = mean(lam * per_task_cls)`
- `loss_abs = reg_loss_weighted(abs_out, y_abs, m_abs, w_abs, reg_loss_type)`
- `loss_fluo = reg_loss_weighted(fluo_out, y_fluo, m_fluo, w_fluo, reg_loss_type)`
- `loss_bitmask = CE(bitmask_logits, bitmask_group_targets, class_weight=bitmask_group_class_weight)` (optional)
- `loss_total = loss_cls + lambda_aux_abs * loss_abs + lambda_aux_fluo * loss_fluo + lambda_aux_bitmask * loss_bitmask`

Bitmask grouping protocol (no leakage):

- For each CV fold, grouping is built from train-fold labels only:
  select `top_k` most frequent bitmask IDs, map all remaining masks to `other`.
- The same train-derived mapping is used for that fold's train and validation steps.
- For final training, grouping is built from final-train split only and reused on leaderboard split.
- Group CE class weights are computed from train-fold grouped frequencies only.

Validation metrics:

- Per-task AP
- `val_macro_ap` (mean AP over 4 tasks)
- `val_min_ap` (worst-task AP)

## 4) Hyperparameters

HPO search space is defined in `training/search_space.py::search_space`.

### 4.1 Architecture capacity and regularization

- `mol_hidden` (`{128, 256}`): Hidden width of the 2D molecule encoder MLP; capped so effective V3 FFN widths stay <= 1024.
- `mol_layers` (`[2, 5]`): Depth of the 2D encoder; deeper models can learn richer nonlinear combinations but are more prone to optimization instability.
- `mol_dropout` (`[0.10, 0.25]`): Dropout probability in each 2D encoder block; higher values increase regularization and reduce overfitting risk.
- `inst_hidden` (`{128, 256}`): Hidden width of the 3D instance encoder; directly controls token embedding dimensionality entering attention.
- `inst_layers` (`[3, 5]`): Depth of the 3D encoder; affects expressiveness of conformation-level token features.
- `inst_dropout` (`[0.05, 0.15]`): Dropout in 3D encoder blocks; regularizes token features before attention pooling.
- `proj_dim` (`{256, 512}`): Common projection dimension for 2D and pooled 3D representations before fusion.
- `attn_heads` (`{8, 16, 32}`): Number of heads in task-attention pooling; must divide `inst_hidden`.
- `attn_dropout` (`[0.05, 0.2]`): Dropout inside multihead attention; regularizes per-task instance weighting.
- `mixer_hidden` (`{128, 256}`): Width of the fusion mixer MLP that maps concatenated 2D/3D task features to task embeddings.
- `mixer_layers` (`[3, 5]`): Depth of the fusion mixer; controls complexity of cross-modal feature interaction.
- `mixer_dropout` (`[0.05, 0.2]`): Dropout in fusion mixer blocks; regularizes final task representations.
- `mol_embedder_name` (fixed: `{mlp_v3_2d}`): 2D embedder implementation selected from registry.
- `inst_embedder_name` (fixed: `{mlp_v3_3d}`): 3D instance embedder implementation selected from registry.
- `aggregator_name` (fixed: `{task_attention_pool}`): 3D token aggregator selected from registry.
- `predictor_name` (fixed: `{mlp_v3}`): Predictor head family selected from registry.
- `head_num_layers` (`{2, 3, 4, 6}`): Shared residual predictor depth for all classification/aux heads.
- `head_dropout` (`[0.0, 0.2]`): Shared dropout inside residual predictor blocks across all heads.
- `head_stochastic_depth` (`[0.0, 0.1]`): Shared DropPath rate for residual predictor blocks in all heads.
- `head_fc2_gain_non_last` (`{1e-3, 3e-3, 1e-2}`): Shared non-last residual block `fc2` init gain in all heads (controls early optimization speed).
- `activation` (`{GELU, SiLU, Mish, ReLU, LeakyReLU}`): Nonlinearity selection passed to embedder and mixer blocks.

### 4.2 Optimization and effective batch size

- `lr` (`[8e-5, 8e-4]`, log): AdamW learning rate; primary control of convergence speed vs instability.
- `weight_decay` (`[3e-6, 3e-4]`, log): L2 regularization strength in AdamW; larger values can improve generalization but may underfit.
- `batch_size` (`{128, 256, 512}`): Per-step minibatch size; larger values reduce gradient noise but increase memory pressure.
- `accumulate_grad_batches` (`{8, 16}`): Gradient accumulation factor; increases effective batch size without increasing device memory footprint.

### 4.3 Class imbalance and task balancing

- Task index mapping for `t0..t3`:
  `t0=Transmittance_340`, `t1=Transmittance_450`,
  `t2=Fluorescence_340_450`, `t3=Fluorescence_more_than_480`.
- `posw_clip_t0` (`[12, 28]`, log): BCE positive-weight clip for `t0` (moderate imbalance, ~5.6% positives).
- `posw_clip_t1` (`[35, 90]`, log): BCE positive-weight clip for `t1` (strong imbalance, ~1.5% positives).
- `posw_clip_t2` (`[3, 10]`, log): BCE positive-weight clip for `t2` (least imbalanced task, ~16.7% positives).
- `posw_clip_t3` (`[90, 220]`, log): BCE positive-weight clip for `t3` (extreme rarity, ~0.24% positives), bounded for stability.
- `gamma_t0` (`[0.5, 2.0]`): Focal gamma for `t0`; moderate hard-example emphasis.
- `gamma_t1` (`[1.0, 3.0]`): Focal gamma for `t1`; stronger focus on rare positives.
- `gamma_t2` (`[0.0, 1.5]`): Focal gamma for `t2`; lighter focusing due to higher prevalence.
- `gamma_t3` (`[1.5, 4.0]`): Focal gamma for `t3`; strongest hard-example focus.
- `rare_oversample_mult` (`[2.0, 10.0]`): Multiplier for dynamic rarity score in sampler weights (`w = 1 + rare_oversample_mult * rarity` before clipping).
- `rare_target_prev` (`[0.06, 0.12]`): Target prevalence for rarity scoring; centered to boost rare endpoints without forcing oversampling of the ~16.7% task.
- `sample_weight_cap` (`[6.0, 9.0]`): Maximum per-sample sampler weight; tighter cap to avoid runaway repetition of very rare positives.
- `use_balanced_batch_sampler` (default: `True`): Enables batch-level balancing so training batches are not dominated by all-negative samples.
- `batch_pos_fraction` (default: `0.35`): Target fraction of positive samples per training batch (positives defined as any active endpoint).
- `min_pos_per_batch` (default: `1`): Hard lower bound on number of positives per batch when both positive and negative pools exist.
- `enforce_bitmask_quota` (default: `True`): Enables additional per-batch quotas for rare multitask patterns.
- `quota_t450_per_256` (default: `4`): Target minimum count of `t1`-positive samples per batch, scaled linearly with batch size from anchor `256`.
- `quota_fgt480_per_256` (default: `1`): Target minimum count of `t3`-positive samples per batch, scaled linearly with batch size from anchor `256`.
- `quota_multi_per_256` (default: `8`): Target minimum count of multi-positive samples (`sum_t y_it >= 2`) per batch, scaled from anchor `256`.
- `use_bitmask_loss_weight` (default: `True`): Enables train-fold bitmask-frequency weighting on `w_cls` to upweight rare endpoint combinations.
- `bitmask_weight_alpha` (default: `0.5`): Exponent in bitmask rarity weighting (`0` disables effect, larger values increase emphasis on rare patterns).
- `bitmask_weight_cap` (default: `3.0`): Upper bound for bitmask-frequency multiplier applied to `w_cls`.
- `lam_t0` (`[0.6, 1.6]`, log): Raw classification-loss multiplier prior for `t0`.
- `lam_t1` (`[1.0, 2.4]`, log): Raw classification-loss multiplier prior for `t1`.
- `lam_t2` (`[0.25, 0.9]`, log): Raw classification-loss multiplier prior for `t2`.
- `lam_t3` (`[1.8, 3.5]`, log): Raw classification-loss multiplier prior for `t3`.
- `lam_floor` (`[0.35, 0.85]`): Lower bound for normalized per-task lambda weights; prevents easier tasks from collapsing to near-zero weight but avoids forcing full uniformity.
- `lam_ceil` (`[1.30, 2.20]`): Upper bound for normalized per-task lambda weights; keeps rare-task emphasis strong without allowing unstable domination.
- `lambda_aux_bitmask` (`[0.02, 0.08]`): Weight of auxiliary bitmask-group CE loss.
- `bitmask_group_top_k` (default: `6`): Number of frequent bitmask IDs kept as explicit classes; remaining masks are merged into `other`.
- `bitmask_group_weight_alpha` (default: `0.5`): Exponent for grouped-class inverse-frequency weighting in bitmask CE.
- `bitmask_group_weight_cap` (default: `5.0`): Cap for grouped-class CE weight multiplier.

Lambda processing used in training:

- Raw `lam_t*` values are normalized by mean.
- Values are clipped by `lam_floor/lam_ceil`.
- Weights are renormalized by mean again.
- Final `lam` scales per-task focal loss before averaging.

Sampler rarity processing used in training:

Definitions:

- Let `y_it in {0,1}` be label of sample `i` for task `t` (`t=0..3`).
- Let `p_t = mean_i(y_it)` be task prevalence in the current train fold.
- Let `P = {i : sum_t y_it > 0}` be samples positive for at least one task.
- Let `N = {i : sum_t y_it = 0}` be all-negative samples.

Step 1: task rarity (auto mode):

- `r_t = clip((rare_target_prev - p_t) / rare_target_prev, 0, 1)`.
- Interpretation:
  `r_t = 0` means task is not rare w.r.t. the target.
  `r_t -> 1` means task is much rarer than target.

Step 2: per-sample rarity:

- `r_i = max_t(y_it * r_t)`.
- A sample is considered "more rare" if it is positive on a rarer task.

Step 3: rarity-aware sample weight:

- `w_i = clip(1 + rare_oversample_mult * r_i, 1, sample_weight_cap)`.
- This weight controls preference among positives during batch construction.

Step 4: batch-level positive quota (anti-all-negative guard):

- `target_pos = round(batch_size * batch_pos_fraction)`.
- `n_pos = clip(max(min_pos_per_batch, target_pos), low, high)`, where:
  `low = 1` if `|P|>0` else `0`;
  `high = batch_size - 1` if `|N|>0` else `batch_size`.
- `n_neg = batch_size - n_pos`.

Step 5: optional bitmask-aware sub-quotas inside positive draw:

- If `enforce_bitmask_quota=True`, reserve positive slots (up to available `n_pos`) for:
  `t3` positives (`quota_fgt480_per_256`), `t1` positives (`quota_t450_per_256`),
  and multi-positive samples (`quota_multi_per_256`), with per-batch quotas scaled by `batch_size / 256`.
- Priority order is rarest-first: `t3` -> `t1` -> multi-positive.
- Quota draws are with replacement and keep rarity-aware probabilities from `w_i`.

Step 6: stochastic drawing per batch:

- Draw remaining positive slots from `P` with replacement, probability proportional to `w_i`.
- Draw `n_neg` indices from `N` with replacement, uniformly.
- Concatenate and shuffle.

Step 7: repeat for each batch in epoch:

- Number of batches is `ceil(num_train_samples / batch_size)` (`drop_last=False`).
- Because draws use replacement, both positive balancing and rarity oversampling persist throughout the epoch.

What this guarantees:

- Batches are not dominated by all-negative samples when positives exist.
- Oversampling is still active:
  first at batch composition level (`n_pos` quota),
  then inside positive subset via rarity-aware probabilities (`w_i`) and optional bitmask quotas.
- Rare tasks are emphasized without hard task cutoffs.

Bitmask-frequency loss weighting (train fold only):

- Let `m_i` be integer bitmask of sample `i` (e.g., `[1,0,1,0] -> 5`).
- Let `count(m_i)` be train-fold frequency of bitmask `m_i`.
- Let `ref = median({count(k) | count(k) > 0})`.
- Additional sample multiplier:
  `u_i = clip((ref / count(m_i)) ^ bitmask_weight_alpha, 1, bitmask_weight_cap)`.
- Training classification weights become:
  `w_cls_train[i, t] = base_w_cls_train[i, t] * u_i`.
- Validation/leaderboard weights are not reweighted by bitmask frequency.

Legacy mode:

- If old params contain `rare_prev_thr`, sampler uses hard-threshold task rarity:
  task is rare iff `p_t < rare_prev_thr`.
  This is kept only for backward compatibility with previous studies.

Illustrative example for your prevalence profile:

- If fold prevalences are approximately
  `T340=0.056`, `T450=0.0147`, `F340450=0.1669`, `Fgt480=0.0024`
  and `rare_target_prev=0.10`, then:
  `r ~= [0.44, 0.85, 0.00, 0.98]`.
- With `rare_oversample_mult=6`, pre-cap weights for positives are roughly:
  `T340: 3.64`, `T450: 6.10`, `F340450: 1.00`, `Fgt480: 6.86`.
- So positives from `Fgt480` and `T450` are sampled much more often inside the positive quota.

### 4.4 Auxiliary weighting

- `lambda_aux_abs` (`[0.05, 0.5]`): Weight of absorbance auxiliary regression loss in total objective.
- `lambda_aux_fluo` (`[0.05, 0.5]`): Weight of fluorescence auxiliary regression loss in total objective.
- `lambda_aux_bitmask` (`[0.02, 0.08]`): Weight of bitmask-group auxiliary classification loss in total objective.
- `reg_loss_type` (fixed: `{mse}`): Regression criterion for auxiliary heads; currently constrained to MSE only in search space.

### 4.5 HPO objective control

- `objective_mode` (fixed: `macro_plus_min`): Trial score combines overall AP and worst-task AP to discourage neglecting harder tasks.
- `min_w` (`[0.1, 0.6]`): Weight of `min_ap` in score.

Per-fold metric definitions:

- `aps = [ap_task0, ap_task1, ap_task2, ap_task3]`
- `macro_ap = mean(aps)`
- `min_ap = min(aps)`

Per-fold objective:

- `fold_score = (1 - min_w) * macro_ap + min_w * min_ap`

Optuna trial objective across CV folds:

- `trial_value = mean(fold_score_fold0, fold_score_fold1, ..., fold_score_foldK)`

Notes:

- `fold_score` is what you see as `score=...` in fold logs.
- Printed `min_w` is rounded for display, while objective computation uses full precision.

### 4.6 Optuna pruning strategy

The CV-HPO stage uses delayed, less aggressive pruning by default:

- `pruner_kind` (default: `percentile`): selects Optuna pruner type.
- `pruner_warmup_steps` (default: `8`): no pruning decisions before this many reported steps.
- `pruner_startup_trials` (default: `10`): first trials run without pruning-based elimination.
- `pruner_percentile` (default: `25.0`): threshold for `PercentilePruner` (lower percentile = less aggressive pruning).

Behavior:

- If `pruner_kind == "percentile"`:
  use `optuna.pruners.PercentilePruner(percentile=25.0, n_startup_trials=10, n_warmup_steps=8)`.
- If `pruner_kind == "median"`:
  use `optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=8)`.

Rationale:

- Rare-task AP often stabilizes later, so earlier defaults could prune promising trials too soon.
- Longer warmup and startup windows reduce premature pruning while still controlling HPO cost.

## 5) Training Behavior (non-search)

- Early stopping + checkpoint monitor: `val_macro_ap`
- Deterministic trainer setup is enabled
- Optuna pruning callback is integrated for CV trials
- Auxiliary targets are standardized using train-fold statistics only
- Train dataloaders use `MultitaskBalancedBatchSampler` by default (`use_balanced_batch_sampler=True`)
- Train-fold `w_cls` can be bitmask-frequency reweighted (`use_bitmask_loss_weight=True` by default)
- Bitmask-group mapping and bitmask-group class weights are computed from train fold only (CV) / train split only (final)
- HPO runs only when CLI flag `--run_hpo` is provided
- Without `--run_hpo`, pipeline loads params from `--best_params_json`; if omitted, defaults to `<study_dir>/multimodal_mil_aux_gpu_best_params.json`
- Final training stage retrains best config and can export leaderboard attention

## 6) Guardrails

- No compatibility alias modules for removed paths
- No duplicated implementations across layers
- No wildcard exports (`import *`) in package surfaces
