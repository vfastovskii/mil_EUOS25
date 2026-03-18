# Architecture Specification

This document is the authoritative, code-aligned architecture spec for the current repository state.

Scope:
- Package: `opt_attn_net_feb/`
- Root wrapper entrypoint: `../opt_net_fast.py`
- Main entrypoint implementation: `entrypoints/hpo_pipeline.py`
- Family-suite implementation (dispatched by CLI flag): `entrypoints/family_suite.py`

Everything below is synchronized to the current code, including constants, defaults, heuristics, and fallback behavior.

---

## 1. System Overview

The system supports two training/evaluation modes:
- default MIL mode (single family)
- multi-family suite mode (`--run_family_suite`)

Core modeling target is 4-task classification with auxiliary regression heads, plus integrated explainability:
- Core model: `MILTaskAttnMixerWithAux`
- HPO: Optuna over CV on predefined train folds
- Final stage: train on split `train`, validate on split `leaderboard` (or custom `--leaderboard_split`)
- Explainability (optional):
  - Chem-ACE (concept discovery + semantic tagging)
  - Lambda-Vol (concept-pressure dynamics across epochs)
  - Concept-guided RL controller (final training only)

High-level flow (default MIL mode):
1. Parse CLI + build typed config objects.
2. Load labels + 2D + 3D/QM features.
3. Build HPO dataset from `--use_splits` and predefined fold column.
4. Optional HPO (`--run_hpo`): optimize CV objective.
5. Final run always executes:
   - train on split `train`
   - evaluate on split `leaderboard`
   - export leaderboard attention/predictions CSV
   - optional explainability outputs and optional concept-guided RL control.

High-level flow (`--run_family_suite` mode):
1. Build shared train/leaderboard data contracts once.
2. Optimize each selected family independently on train CV.
3. Save best params JSON per family.
4. Refit each family on train and evaluate on leaderboard.
5. Calibrate leaderboard probabilities per family.
6. Blend calibrated probabilities across families and report final metrics.

---

## 2. Code Layering and Dependency Direction

Repository layering:

```text
../opt_net_fast.py                   # root wrapper entrypoint
opt_attn_net_feb/
  callbacks/                         # Lightning callbacks (Optuna pruning)
  data/                              # Datasets, collate, export utilities
  entrypoints/                       # CLI + orchestration
  explainability/                    # Chem-ACE + Lambda-Vol
  losses/                            # focal + regression loss functions
  models/                            # multimodal MIL model + attention pooling
  training/                          # builders, configs, execution, trainer
  utils/                             # constants, IO, metrics, samplers, ops
  __init__.py                        # stable package-level API exports
```

Dependency direction (enforced by structure):
- `entrypoints -> training, data, models, utils, explainability`
- `training -> models, data, losses, utils, explainability`
- `models/losses/data/utils` do not depend on `entrypoints`
- `explainability` is modular and can be used independently or from training

Public API:
- Package-level exports in `__init__.py` are stable API surface for external users.
- The CLI wrapper `../opt_net_fast.py` delegates to `opt_attn_net_feb.entrypoints.hpo_pipeline.main`.

---

## 3. Configuration Architecture (Typed Contracts)

### 3.1 Training-level typed configs (`training/configs.py`)

`HPOConfig` groups:
- `BackboneConfig`
- `HeadConfig`
- `OptimizationConfig`
- `RuntimeConfig`
- `SamplerConfig`
- `LossWeightingConfig`
- `ObjectiveConfig`

Important defaults:
- Objective mode is fixed to `"macro_plus_min"`.
- Sampler defaults enable balanced batch sampling + bitmask weighting.
- Embedder names are explicit by modality:
  - 2D default: `mlp_v3_2d`
  - 3D default: `mlp_v3_3d`

### 3.2 Model-level typed configs (`models/multimodal_mil/configs.py`)

`MILModelConfig` groups:
- `MILBackboneConfig`
- `MILPredictorConfig`
- `MILOptimizationConfig`
- `MILLossConfig`

Conversion chain:
1. Flat Optuna params -> `HPOConfig.from_params(...)`
2. `MILModelBuilder.build(...)` maps `HPOConfig` -> `MILModelConfig`
3. `MILTaskAttnMixerWithAux.from_config(...)` builds Lightning module

Hard validation:
- `inst_hidden % attn_heads == 0` enforced in both search space pruning and config validation.

---

## 4. Data Contracts and Preprocessing

### 4.1 Required input files

CLI required:
- `--labels`
- `--feat2d_scaled`
- `--feat3d_scaled`
- `--feat3d_qm_scaled`
- `--study_dir`

Optional (semantic-only raw sources):
- `--feat3d_raw`
- `--feat3d_qm_raw`

Raw semantic tables are used only when both are provided.

### 4.2 Required columns

Label table (`utils/constants.py`):
- Task columns (`TASK_COLS`):
  - `Transmittance_340`
  - `Transmittance_450`
  - `Fluorescence_340_450`
  - `Fluorescence_more_than_480`
- Aux absorbance columns (`AUX_ABS_COLS`):
  - `Transmittance_340_quantitative`
  - `Transmittance_450_quantitative`
- Aux fluorescence base columns (`AUX_FLUO_BASE_COLS`):
  - `wl_pred_nm`
  - `qy_pred`
- Weight columns mapping (`WEIGHT_COLS`):
  - task0 -> `sample_weight_340`
  - task1 -> `sample_weight_450`
  - task2 -> unweighted
  - task3 -> unweighted

Default ID/split/fold columns:
- `--id_col ID`
- `--conf_col conf_id`
- `--split_col split`
- `--fold_col cv_fold`

### 4.3 Split semantics

- `--use_splits` controls which rows are used for HPO CV dataset construction.
- Final stage is independent of `--use_splits` and always uses:
  - train split: `split == "train"`
  - validation split: `split == --leaderboard_split` (default `leaderboard`)

### 4.4 Label and weight transforms

- Classification labels:
  - `coerce_binary_labels(df)`:
    - fill NaN with 0
    - cast to int
    - threshold `(y > 0)` -> `0/1`
- Task weights:
  - `build_task_weights(df)` loads mapped columns if present, else ones
  - clipped to `[0, +inf)`

### 4.5 Auxiliary target construction

`build_aux_targets_and_masks(df)`:
- `y_abs`: stacked 2 absorbance quantitative targets
- `m_abs`: finite mask
- `y_fbase`: stacked 2 fluorescence base targets (`wl_pred_nm`, `qy_pred`)
- `y_fluo4`: **constructed by duplication** `concat([y_fbase, y_fbase])` -> shape `(N,4)`
- `m_fluo4`: same duplication for masks

This duplication is a deliberate approximation to match 4 fluorescence auxiliary outputs.

`build_aux_weights(df)`:
- `w_abs = [sample_weight_340, sample_weight_450]`
- `w_fluo4 = repeat(w_ad, 4)`
- clipped to `[0, +inf)`

### 4.6 2D feature loading/alignment

`load_2d`:
- Reads CSV, validates ID column, logs duplicate-ID statistics.
- Feature columns = all columns except `NONFEAT_2D = {ID, curated_SMILES, split}`.

`align_by_id`:
- Exact ID lookup; raises on missing IDs.

### 4.7 3D+QM instance merge

`load_and_merge_instances`:
1. Load geometry and QM CSVs.
2. Optional ID filtering to allowed set.
3. QM cleanup:
   - drop rows with non-empty `error`
   - keep `status` in `{ok, success, 0, 1, true}` or NaN
4. Deduplicate repeated `(ID, conf_id)` rows by mean per feature block.
5. Inner join geometry and QM on `(ID, conf_id)` with `validate="one_to_one"`.
6. Final instance feature = horizontal concat `[geom_features, qm_features]`.

### 4.8 Bag index construction

`build_instance_index`:
- Stable sort by ID (`mergesort`) so bag slices are contiguous.
- Returns:
  - unique IDs
  - starts, counts per ID
  - `id2pos` lookup
  - sorted instance feature matrix
  - sorted conformer IDs

### 4.9 IDs without bags

Two safeguards:
- HPO stage: IDs with no conformers after merge are dropped; folds recomputed.
- Final stage: `drop_ids_without_bags` applied to both train and leaderboard sets.

---

## 5. Datasets, Collate, and Tensor Shapes

### 5.1 `MILTrainDataset`

Per item:
- `x2d`: `[F2]`
- `bag`: `[Ni, F3]`
- `y_cls`: `[4]`
- `w_cls`: `[4]`
- `y_abs`: `[2]`
- `m_abs`: `[2]` bool
- `w_abs`: `[2]`
- `y_fluo`: `[4]`
- `m_fluo`: `[4]` bool
- `w_fluo`: `[4]`

`max_instances=0` in pipeline means no bag truncation.

### 5.2 `collate_train`

Batch output:
- `x2d`: `[B, F2]`
- `x3d_pad`: `[B, Nmax, F3]`
- `kpm`: `[B, Nmax]` bool, `True` = padding
- labels/weights stacked across batch

### 5.3 `MILExportDataset` + `collate_export`

Export batch output:
- `mol_ids` list length `B`
- `conf_pad`: `[B, Nmax]` object (conformer IDs)
- `x2d`: `[B, F2]`
- `x3d_pad`: `[B, Nmax, F3]`
- `kpm`: `[B, Nmax]`

---

## 6. Model Architecture (`MILTaskAttnMixerWithAux`)

Implementation files:
- `models/multimodal_mil/model.py`
- `models/multimodal_mil/embedders.py`
- `models/multimodal_mil/aggregators.py`
- `models/multimodal_mil/predictors.py`
- `models/multimodal_mil/head_mlp_v3.py`
- `models/multimodal_mil/head_utils.py`
- `models/attention_pooling/pool.py`

### 6.1 Extensible component registries

- 2D embedder registry: `build_2d_embedder(name=...)`
- 3D embedder registry: `build_3d_embedder(name=...)`
- Aggregator registry: `build_aggregator(name=...)`
- Predictor registry: `build_predictor_heads(name=...)`

Default names:
- `mol_embedder_name="mlp_v3_2d"`
- `inst_embedder_name="mlp_v3_3d"`
- `aggregator_name="task_attention_pool"`
- `predictor_name="mlp_v3"`

Legacy aliases:
- 2D: `mlp_v3 -> mlp_v3_2d`
- 3D: `mlp_v3 -> mlp_v3_3d`

### 6.2 Forward pipeline and shapes

Inputs:
- `x2d`: `[B, F2]`
- `x3d_pad`: `[B, N, F3]`
- `key_padding_mask`: `[B, N]`, `True=PAD`

Branch A (2D molecule-level, optional when `mol_dim > 0`):
1. `mol_enc(x2d)` -> `[B, mol_hidden]`
2. `mol_post_embed_norm` (`LayerNorm(mol_hidden)`)
3. `proj2d = Linear(mol_hidden->proj_dim) + LayerNorm(proj_dim)` -> `e2d [B, proj_dim]`
4. Repeat per task: `e2d_rep = e2d.unsqueeze(1).expand(-1,4,-1)` -> `[B,4,proj_dim]`

Branch B1 (3D geometry instance-level, optional when `inst_geom_dim > 0`):
1. Split `x3d_pad` by configured dims into `x3d_geom [B,N,F3_geom]` and `x3d_qm [B,N,F3_qm]`.
2. Flatten geometry instances: `[B*N, F3_geom]`
3. `inst_geom_enc` -> `[B*N, inst_hidden]`
4. Reshape `[B,N,inst_hidden]`
5. `inst_geom_post_embed_norm` (`LayerNorm(inst_hidden)`)
6. `attn_pool_geom(tokens, kpm)` -> `pooled_geom [B,4,inst_hidden]`, optional `attn_geom [B,4,N]`
7. `agg_geom_post_norm` (`LayerNorm(inst_hidden)`)
8. `proj3d_geom` (`Linear+LayerNorm`) -> `e3d_geom [B,4,proj_dim]`

Branch B2 (3D quantum instance-level, optional when `inst_qm_dim > 0`):
1. Flatten quantum instances: `[B*N, F3_qm]`
2. `inst_qm_enc` -> `[B*N, inst_hidden]`
3. Reshape `[B,N,inst_hidden]`
4. `inst_qm_post_embed_norm` (`LayerNorm(inst_hidden)`)
5. `attn_pool_qm(tokens, kpm)` -> `pooled_qm [B,4,inst_hidden]`, optional `attn_qm [B,4,N]`
6. `agg_qm_post_norm` (`LayerNorm(inst_hidden)`)
7. `proj3d_qm` (`Linear+LayerNorm`) -> `e3d_qm [B,4,proj_dim]`

Attention compatibility note:
- If both 3D branches are present and `return_attn=True`, exported attention is `mean(attn_geom, attn_qm)`.
- If only one 3D branch is present, that branch attention is returned.

Fusion + mixer (dynamic by active modalities):
1. Build modality list `mix_parts` from enabled projections in `{2d, 3d_geom, 3d_qm}`.
2. Concat only active parts: `concat(mix_parts, dim=-1)` -> `[B,4,K*proj_dim]`, where `K = len(active_modalities)` and `K in {1,2,3}`.
3. Flatten task axis: `[B*4, K*proj_dim]`.
4. `mixer` residual MLP -> `[B*4, mixer_hidden]`.
5. Reshape `[B,4,mixer_hidden]`.
6. `mixer_post_norm` (`LayerNorm(mixer_hidden)`) -> `z_tasks`.

Heads:
- Classification: one head per task, input `z_tasks[:,t,:]` -> logits `[B,4]`
- Aux absorbance: two shared heads from `z_aux = mean(z_tasks, dim=1)` -> `[B,2]`
- Aux fluorescence: four shared heads from `z_aux` -> `[B,4]`
- Optional bitmask group head from `z_aux` -> `[B, n_groups]`

### 6.3 Why 2D embedding is repeated per task

2D branch learns one molecule representation per sample. Repetition does **not** create separate 2D encoders.
It broadcasts the same molecule context to each task-specific fusion slot, where it is combined with task-specific aggregated 3D geometry and 3D quantum contexts before task-specific heads.

So training remains end-to-end with one shared 2D encoder; task specificity is injected by:
- 3D task-query attention pooling
- task slot in fusion/mixer
- task-specific classification heads.

### 6.4 Embedder/mixer block logic (V3-like residual MLP)

Factory: `utils.mlp.make_residual_mlp_embedder_v3(...)` uses:
- expansion `2.0`
- pre-norm residual blocks
- gated FFN (SwiGLU-like)
- residual dropout `0.05`
- stochastic depth `0.05` (depth-scaled)
- learnable residual scale:
  - warmup init `0.01` for early blocks
  - main init `0.1` for last `2` blocks
  - applied with `tanh`
- last block FF2 zero-init (near-identity startup)
- inner dimension rounded to multiple of `64`

Used for:
- 2D embedder
- 3D embedder
- mixer

### 6.5 Aggregator logic (`TaskAttentionPool`)

Defaults and key options:
- MHA with `batch_first=True`
- learned task queries `q` shape `[1, n_tasks, dim]`
- `pre_ln` on tokens (`use_layer_norm=True`, `pre_layer_norm=True`)
- attention output average across heads -> `alpha [B,T,N]`
- `alpha` masked and renormalized over valid conformers
- optional threshold prune (`prune_below`) with argmax fallback if all pruned
- pooling source default `pool_from="normed_inputs"`
- value projection default `pool_v_mode="tie_mha_v"`
- optional top-k pooling (`topk_n`, strategies: `renorm|mean|sum|argmax`)
- optional residual blend with mean pooling
- optional query temperature (`use_temperature`, `temperature_init=0.3`)

Model guardrail:
- `aggregator_kwargs` cannot override reserved keys `{dim, n_heads, dropout, n_tasks}`.

### 6.6 Predictor head logic (V3-like)

Head class: `MLPPredictorV3Like`:
- stack of residual FFN blocks
- configurable `num_layers`, `dropout`, `stochastic_depth`, `fc2_gain_non_last`
- GLU enabled (`use_glu=True`)
- `input_layernorm=True`
- `output_dim=1` for each individual head

Head defaults in builder:
- expansion `2.0`
- `res_scale_init=0.1`
- `inner_multiple=64`
- `proj_gain=0.5`
- `head_dropout=0.0`

### 6.7 Bitmask auxiliary head

Enabled when:
- `lambda_aux_bitmask > 0`
- `bitmask_num_groups >= 2`

Construction:
- group IDs = `top_k` frequent bitmasks from fold-train + one `other`
- head output dim = `len(top_ids) + 1`
- targets from binary task vector -> integer bitmask -> mapped group ID
- CE class weights from fold-train group frequencies

---

## 7. Loss Functions and Training Objective

### 7.1 Classification loss (`MultiTaskFocal`)

Per-task focal BCE:
- `bce = BCEWithLogits(logits, targets, pos_weight, reduction=none)`
- `pt = sigmoid(logits)` matched to target class
- `focal = (1 - pt) ^ gamma_t`
- `loss = focal * bce`
- weighted reduction per task with `w_cls`:
  - `num_t = sum_i loss_it * w_it`
  - `den_t = sum_i w_it + 1e-6`
  - `per_task_loss_t = num_t / den_t`

Then `weighted_per_task = per_task_loss * lam_t` and:
- `loss_cls = mean_t(weighted_per_task_t)`

### 7.2 Regression losses (`reg_loss_weighted`)

For `abs_out` and `fluo_out`:
- mask invalid targets via `target_safe = where(mask, target, pred.detach())`
- per-element loss:
  - `mse`: `(pred-target_safe)^2`
  - or `smoothl1`
- weighted masked reduction:
  - `num = sum(per * w_eff)`
  - `den = sum(w_eff) + 1e-6`
  - mean across output channels

### 7.3 Bitmask auxiliary loss

If active:
- `loss_bitmask = CrossEntropy(bitmask_logits, bitmask_targets, class_weight)`
Else:
- `loss_bitmask = 0`

### 7.4 Total training loss

`loss_total = loss_cls + lambda_aux_abs*loss_abs + lambda_aux_fluo*loss_fluo + lambda_aux_bitmask*loss_bitmask`

---

## 8. Task Weighting, Pos Weights, and Imbalance Logic

### 8.1 Lambda task weighting (`lam`)

Two modes:
1. Explicit per-task lambdas (`lam_t0..lam_t3`) provided:
   - normalize by mean
   - clip each task to `[lam_floor, lam_ceil]`
   - renormalize by mean
2. If explicit lambdas absent:
   - prevalence-based: `lam_t ∝ (1/p_t)^lambda_power`, normalized by mean

### 8.2 Positive class weights (`pos_weight`)

`pos_weight_t = neg_t / max(pos_t, 1)` with clipping:
- either global scalar clip
- or per-task clips `posw_clip_t0..t3`

### 8.3 Focal gamma per task

`gamma = [gamma_t0, gamma_t1, gamma_t2, gamma_t3]`

### 8.4 Oversampling and batch balancing

#### Weighted sampler (`make_weighted_sampler`)

Rarity severity per task:
- If `rare_prev_thr` set:
  - binary rarity: `severity_t = 1[p_t < rare_prev_thr]`
- Else:
  - smooth deficiency toward target prevalence:
  - `severity_t = clip((rare_target_prev - p_t)/rare_target_prev, 0, 1)`

Per-sample rarity:
- `rarity_i = max_t(y_it * severity_t)`

Sample weight:
- `w_i = clip(1 + rare_oversample_mult * rarity_i, 1, sample_weight_cap)`

#### Balanced batch sampler (`MultitaskBalancedBatchSampler`)

- Enforces positive quota per batch:
  - target positives: `round(batch_size * batch_pos_fraction)`
  - clamp with `min_pos_per_batch`, positivity/negativity availability
- Positive draws are rarity-weighted and unique within a batch whenever the dataset
  has enough distinct rows to support that.
- Negatives are drawn uniformly from all-negative samples.
- If a tiny dataset cannot supply enough unique rows, replacement is used only as
  a final fallback to complete the batch.

---

## 9. Metrics and Optimization Target

### 9.1 Validation metrics

Prediction post-processing:
- logits sanitized: `nan->0`, `+inf->50`, `-inf->-50`
- probabilities: `sigmoid(logits)`

Per-task AP (`ap_per_task`):
- weighted by `w_cls` only for tasks `(0,1)`
- if a task has zero positives, AP set to `0.0`

Per-task ROC-AUC (`roc_auc_per_task`):
- weighted by `w_cls` only for tasks `(0,1)`
- if undefined (single-class target), fallback `0.5`

### 9.2 Fold score during HPO

Fixed objective mode: `macro_plus_min`

Definitions:
- `macro_ap = mean(AP_t0..AP_t3)`
- `min_ap = min(AP_t0..AP_t3)`
- `score = (1 - min_w) * macro_ap + min_w * min_ap`

Interpretation:
- optimizes average quality while penalizing neglect of weakest task.

Trial value:
- mean of fold scores across configured CV folds.

### 9.3 Logged outputs per fold/trial

Fold detail includes:
- trained epochs
- best epoch
- macro PR-AUC/AP
- per-task PR-AUC/AP
- macro ROC-AUC
- per-task ROC-AUC
- final fold score and objective settings

---

## 10. Hyperparameter Search Space (Current)

Source: `training/search_space.py`

Task index mapping in code:
- `t0 -> Transmittance_340`
- `t1 -> Transmittance_450`
- `t2 -> Fluorescence_340_450`
- `t3 -> Fluorescence_more_than_480`

Observed prevalence prior used to tighten ranges (documented in code comments):
- T340: `~0.056`
- T450: `~0.015`
- F340450: `~0.167`
- Fgt480: `~0.0024`

### 10.1 Architecture and regularization

- `mol_hidden`: `{128, 256}`
- `mol_layers`: `[2, 5]`
- `mol_dropout`: `[0.10, 0.25]`
- `inst_hidden`: `{128, 256}`
- `inst_layers`: `[3, 5]`
- `inst_dropout`: `[0.05, 0.15]`
- `proj_dim`: `{256, 512}`
- `attn_heads`: `{8, 16, 32}`
- `attn_dropout`: `[0.05, 0.2]`
- `mixer_hidden`: `{128, 256}`
- `mixer_layers`: `[3, 5]`
- `mixer_dropout`: `[0.05, 0.2]`
- `activation`: `{GELU, SiLU, Mish, ReLU, LeakyReLU}`
- `mol_embedder_name`: `{mlp_v3_2d}`
- `inst_embedder_name`: `{mlp_v3_3d}`
- `aggregator_name`: `{task_attention_pool}`
- `predictor_name`: `{mlp_v3}`

Head-specific knobs:
- `head_num_layers`: `{2, 3, 4, 6}`
- `head_dropout`: `[0.0, 0.2]`
- `head_stochastic_depth`: `[0.0, 0.1]`
- `head_fc2_gain_non_last`: `{1e-3, 3e-3, 1e-2}`

### 10.2 Optimization/runtime

- `lr`: `[8e-5, 8e-4]` (log)
- `weight_decay`: `[3e-6, 3e-4]` (log)
- `batch_size`: `{128, 256, 512}`
- `accumulate_grad_batches`: `{8, 16}`

### 10.3 Imbalance/task weights

Pos-weight clips:
- `posw_clip_t0`: `[12, 28]` (log)
- `posw_clip_t1`: `[35, 90]` (log)
- `posw_clip_t2`: `[3, 10]` (log)
- `posw_clip_t3`: `[90, 220]` (log)

Focal gamma:
- `gamma_t0`: `[0.5, 2.0]`
- `gamma_t1`: `[1.0, 3.0]`
- `gamma_t2`: `[0.0, 1.5]`
- `gamma_t3`: `[1.5, 4.0]`

Sampling:
- `rare_oversample_mult`: `[2.0, 10.0]`
- `rare_target_prev`: `[0.06, 0.12]`
- `sample_weight_cap`: `[6.0, 9.0]`

Lambda weights:
- `lam_t0`: `[0.6, 1.6]` (log)
- `lam_t1`: `[1.0, 2.4]` (log)
- `lam_t2`: `[0.25, 0.9]` (log)
- `lam_t3`: `[1.8, 3.5]` (log)
- `lam_floor`: `[0.35, 0.85]`
- `lam_ceil`: `[1.30, 2.20]`

Aux weights:
- `lambda_aux_abs`: `[0.05, 0.5]`
- `lambda_aux_fluo`: `[0.05, 0.5]`
- `lambda_aux_bitmask`: `[0.02, 0.08]`
- `reg_loss_type`: `{mse}`

HPO objective control:
- `min_w`: `[0.1, 0.6]`

### 10.4 Effective dimensional bounds

Given search space:
- 2D encoder hidden width <= 256
- 3D encoder hidden width <= 256
- projection dim <= 512
- mixer input dim = `K * proj_dim`, with `K in {1,2,3}` depending on enabled modalities
  - 2D-only: `K=1`, max mixer input `512`
  - 3D-only (geom+qm active): `K=2`, max mixer input `1024`
  - full multimodal: `K=3`, max mixer input `1536`
- mixer hidden width <= 256
- head input dim = mixer hidden <= 256

This keeps encoder/head widths compact while allowing modality-dependent fusion width.

---

## 11. Trainer, Pruning, Reproducibility, and Resource Policy

### 11.1 Lightning trainer settings

`LightningTrainerFactory`:
- Early stopping on `val_macro_ap`, mode `max`, patience from config
- Optional checkpoint callback on same metric
- Optional Optuna pruning callback
- deterministic=True
- gradient_clip_val=0.0
- logger/progress/model-summary disabled for lean runtime

### 11.2 Pruning policy

Optuna study defaults (`StudyConfig`):
- `pruner_kind = "percentile"`
- `pruner_warmup_steps = 8`
- `pruner_startup_trials = 10`
- `pruner_percentile = 25.0`

Interpretation:
- Warmup steps are validation-report steps ignored before prune checks.
- Startup trials run unpruned before pruner activates.
- Percentile pruner at 25th percentile is intentionally less aggressive than median pruning.

Alternative:
- `pruner_kind="median"` switches to `MedianPruner` with same startup/warmup counts.

### 11.3 Checkpoint and disk behavior

CV folds (`MILFoldTrainer`):
- `save_checkpoint=False` to reduce disk usage.
- temporary fold ckpt directories are removed after fold run.

Final run (`MILFinalTrainer`):
- saves exactly one best checkpoint (`save_top_k=1`) in `final_best_train_vs_leaderboard/`.

---

## 12. Pipeline Orchestration

Two orchestration paths are implemented in the same CLI entrypoint (`entrypoints/hpo_pipeline.py`):
- default MIL orchestrator: `MILPipelineOrchestrator`
- family-suite dispatch (`--run_family_suite`): `entrypoints/family_suite.py::run_family_suite`

### 12.0 Dispatch rules

- If `--run_family_suite` is set:
  - parser still validates base data/runtime args
  - execution is dispatched to family-suite runner
  - default MIL orchestrator path is skipped
- Else:
  - default MIL pipeline runs (HPO + final) as described below.

### 12.1 Environment setup

`PipelineEnvironmentFactory.prepare(...)`:
- sets seeds (`set_all_seeds`)
- sets torch matmul precision hint (`high`) best-effort
- writes `run_meta.json`
- resolves workers:
  - if `num_workers >= 0`, use it directly
  - else infer from `SLURM_CPUS_PER_TASK`/`os.cpu_count()` and cap to `[0..23]`
- `pin_memory` only if CLI flag true and CUDA available

### 12.2 HPO stage

If `--run_hpo`:
1. Build `MILCVData` from `--use_splits` rows.
2. Build `CVRunConfig`.
3. Create study:
   - study name fixed: `multimodal_mil_aux_gpu`
   - storage: `sqlite:///<study_dir>/multimodal_mil_aux_gpu.sqlite3`
   - `load_if_exists=True`
4. Run `study.optimize(...)`.
5. Save artifacts:
   - `multimodal_mil_aux_gpu_trials.csv`
   - `multimodal_mil_aux_gpu_best_params.json`
   - `multimodal_mil_aux_gpu_best_fold_metrics.json`

If not `--run_hpo`:
- load params from `--best_params_json` or default `<study_dir>/multimodal_mil_aux_gpu_best_params.json`.

### 12.3 Final stage (always runs)

Always executed after HPO/param load:
1. Build final instance index using IDs from `train U leaderboard_split`.
2. Train model on `train` only.
3. Validate on `leaderboard_split`.
4. Write final artifacts in `final_best_train_vs_leaderboard/`.

No condition on `--export_leaderboard_attn`; this flag is compatibility-only.

---

## 13. Final Evaluation and Export Artifacts

Directory: `<study_dir>/final_best_train_vs_leaderboard/`

### 13.1 Core metrics

- `leaderboard_eval.json`:
  - macro PR-AUC (`macro_ap`, `macro_pr_auc`)
  - macro ROC-AUC (`macro_auc`)
  - per-task PR-AUC and ROC-AUC
  - `best_epoch`, `best_ckpt_path`
- `leaderboard_auc_per_task.csv`:
  - `task`, `auc`

### 13.2 Attention + predictions export

`export_leaderboard_attention(...)` writes one row per conformer:
- `ID`
- `conf_id`
- 4 prediction columns:
  - `pred_Transmittance_340`
  - `pred_Transmittance_450`
  - `pred_Fluorescence_340_450`
  - `pred_Fluorescence_more_than_480`
- 4 attention columns:
  - `attn_Transmittance_340`
  - `attn_Transmittance_450`
  - `attn_Fluorescence_340_450`
  - `attn_Fluorescence_more_than_480`

Attention handling details:
- takes `attn [B,4,N]`
- masks invalid positions
- renormalizes per task over valid conformers
- if sum invalid/non-positive, falls back to uniform over valid conformers

Output format:
- If extension is parquet and parquet engine missing, automatic fallback to CSV with warning.
- Default final output path: `<study_dir>/leaderboard_attn.csv` unless `--attn_out` provided.

---

## 14. Chem-ACE Integration (Optional)

Integration entry: `training/explainability_runtime.py::prepare_chem_ace_bundle`
Alignment checklist: `docs/ACE_TCAV_ALIGNMENT.md`

Enabled by:
- `--run_chem_ace`
- or implicitly when `--run_lambda_vol` is set

Hard requirements:
- RDKit must be importable
- labels table must contain `--curated_smiles_col` (default `curated_SMILES`)

### 14.1 Chem-ACE pipeline configuration in final run

Runtime sets:
- `embedding.layer_name = "feature_hybrid_multimodal"`
- `embedding.strategy = "hybrid_local_context"`
- `patch_generation.pharm3d.enabled = True` in this integrated path
- database default URI:
  - `sqlite:///<chem_ace_output_dir>/chem_ace.sqlite3`

### 14.2 Molecule preparation

- Uses `Chem.MolFromSmiles(curated_SMILES)` then `Chem.AddHs`.
- Discovery scope IDs = `train` IDs only (anti-leakage), optionally truncated by `chem_ace_max_ids`.
- Inference scope IDs (typically leaderboard) are processed separately:
  - patch embeddings are mapped to nearest frozen train concept centroids
  - no concept re-clustering or centroid updates on inference scope
  - optional distance gate: `chem_ace_infer_max_distance` (`<=0` disables gate)
- Conformers per molecule truncated by `chem_ace_max_confs_per_id` if >0.

### 14.3 Patch generation

Available generators in framework:
- local subgraph
- BRICS
- Murcko (+ optional framework)
- Pharm3D

Integrated final pipeline enables Pharm3D when conformers are available from the provided SDF.

Patch IDs/hashes are deterministic SHA1 signatures over:
- patch type
- sorted atom indices
- SMARTS/fragment representation
- metadata

### 14.4 Patch embedding in integrated final pipeline

Legacy long-concat feature projection is removed from final runtime.

Current integrated runtime uses modality-separated hybrid embedding spaces:

- `2d`
- `3d_geom`
- `3d_qm`

Patch routing:

- non-conformer patch (`conf_id=None`) -> `2d`
- conformer patch without valid QM slice -> `3d_geom`
- conformer patch with QM support -> `3d_qm`

Per-modality embedding equation:

- `z_local = MLP_local(local_features)`
- `z_ctx = MLP_ctx(context_features)`
- `z = LayerNorm(z_local + alpha * z_ctx)`
- `z = l2_normalize(z)`

Default dimensions:

- `chem_ace_embed_dim_2d = 64`
- `chem_ace_embed_dim_3d_geom = 64`
- `chem_ace_embed_dim_3d_qm = 64`
- `chem_ace_context_dim = 16`
- `chem_ace_context_alpha = 0.2`
- `chem_ace_qm_gating = True`

Stored with metadata via embedding cache and DB when embedding persistence is enabled.

Semantic source split:
- Concept discovery embeddings use scaled model-input tables.
- Semantic tagger geometry/QM summaries use raw tables only if both `--feat3d_raw` and `--feat3d_qm_raw` are provided.
- Runtime emits `explainability.chem_ace.semantics_instances source=raw|scaled`.

### 14.5 Concept discovery

Clustering algorithms configured (default):
- kmeans

Defaults (`ConceptDiscoveryConfig`):
- `kmeans_k=0` (`<=1` enables automatic `k=sqrt(n_embeddings)`)
- `min_support=8`
- `min_coherence=0.0`
- `dedup_centroid_similarity_threshold=0.98`

Coherence definition:
- `coherence = 1 / (1 + mean_distance_to_centroid)`

Membership score:
- `membership_score = 1 / (1 + distance_to_centroid)`

Medoid:
- nearest point to centroid in Euclidean distance.

Concept IDs:
- SHA1 over `(layer_name, algorithm, sorted patch IDs)`.

### 14.6 Semantic tagging and naming

Taggers produce evidence-backed tags with confidence:
- charge tags
- conjugation/aromaticity tags
- geometry tags from conformer coordinates (planarity RMSD + rotatable-bond proxy)
- geometry descriptor-family tags from 3D descriptor vectors using `geom_cols` + `inst_geom_dim`
- pharmacophore tags
- SMARTS functional-group tags from `default_functional_group_rules.json`
- SMARTS-RX reactivity tags from `smartsrx.json`
- QM descriptor-family tags derived from per-conformer QM vectors and QM column names
- optional Open Babel descriptor tags (`logP`, `TPSA`, `MR`) when `openbabel.pybel` is available
- cross-modal consistency tags (SMARTS-RX + QM + geometry + pharmacophore agreement)

Key thresholds (`SemanticTaggingConfig` defaults):
- `charge_threshold_formal = 1`
- `aromatic_fraction_threshold = 0.35`
- `conjugation_size_threshold = 6`
- `planarity_rmsd_threshold = 0.25`
- `geom_min_vectors_for_tagging = 8`
- `geom_z_threshold = 0.50`
- `geom_strong_z_threshold = 1.00`
- `qm_min_vectors_for_tagging = 8`
- `qm_z_threshold = 0.50`
- `qm_strong_z_threshold = 1.00`
- `use_smarts_rx = True`
- `smarts_rx_rules_path = None` (uses bundled SMARTS-RX registry)

Functional rules:
- source file: `explainability/chem_ace/rules/default_functional_group_rules.json`
- per-rule controls: `min_patch_rate`, `confidence`, `provenance`
- invalid SMARTS are skipped with warning; pipeline continues
- runtime augmentation:
  - Chem-ACE builds additional functional rules from dataset `curated_SMILES` using RDKit `Chem.Fragments.fr_*` counters
  - generated + merged file path:
    - `<chem_ace_output_dir>/rules_autogen/default_functional_group_rules.dataset.json`
  - fragment prevalence stats:
    - `<chem_ace_output_dir>/rules_autogen/functional_group_fragment_stats.json`
  - merged rules are injected through `SemanticTaggingConfig.functional_rules_path` and used for semantic tagging in the same run

SMARTS-RX rules:
- source file: `explainability/chem_ace/rules/smartsrx.json` (legacy fallback: `default_smarts_rx_rules.json`)
- each rule: `(tag, smarts, role, min_patch_rate, confidence, provenance)`
- emitted tags include rule tags (e.g., `rx_michael_acceptor`) and role tags (e.g., `rx_role_electrophile`)
- intended for reactivity-aware concept naming and downstream filtering
- supported input schemas:
  - canonical `rules` schema (explicit `tag`)
  - generated `data` schema (`category/subcategory/specific_type/smarts`) with auto-derived `tag` and `role`

Naming:
- rule-based first (JSON registry)
- fallback descriptor-based label (`"<top descriptors> motif"`)

Rule sources:
- naming rules: `explainability/chem_ace/rules/default_naming_rules.json`
- functional rules: `explainability/chem_ace/rules/default_functional_group_rules.json`
- SMARTS-RX rules: `explainability/chem_ace/rules/smartsrx.json`
- RDKit-fragment generator CLI:
  - `python -m explainability.chem_ace.rules.generate_fragment_rules_from_labels --labels_csv ... --smiles_col curated_SMILES`
- explicit paths can be set via `SemanticTaggingConfig.naming_rules_path`, `SemanticTaggingConfig.functional_rules_path`, and `SemanticTaggingConfig.smarts_rx_rules_path`

QM family mapping:
- descriptor names are normalized and matched by token to families:
  - HOMO, LUMO, gap, dipole, polarizability, hardness, softness
  - electrophilicity, nucleophilicity, charge-transfer, ESP, Fukui, NBO, Mulliken/NPA
  - chemical potential (`mu_eV`), ionization potential (`vip_eV`), electron affinity (`vea_eV`)
  - bond-order (`bo_*`), conjugated bond-order (`bo_conj_*`)
  - atomic-charge distribution (`q_*`) and charge-separation distance (`q_abs_r_*`, `d_pos_neg`)
  - quadrupole (`quad_norm_au`, `quad_trace_au`)
- family stats are computed as concept-level `abs_z_mean`, `z_mean`, `z_std`
- tags are emitted from thresholded family stats (e.g., `large HOMO-LUMO gap`, `high dipole moment`, `electrophile-like electronic profile`)

Photophysics proxy formalism (heuristic):
- no explicit excited-state observables are used in current pipeline; tags are proxy-level and evidence-backed by family stats
- transmittance proxy:
  - higher gap/hardness with lower charge-transfer and dipole activity
- fluorescence proxy:
  - stronger conjugated bond-order + geometry planarity with elevated dipole or charge-transfer signatures
- additional proxy outputs:
  - `red-shifted absorption proxy`
  - `blue-shifted transparency proxy`

3D geometry family mapping:
- geometry descriptor names are normalized and token-matched to families:
  - distance, angle, dihedral, planarity, shape, size, inertia, surface_volume, ring_strain, hbond_geometry
  - global_3d_fingerprint (RDF / MORSE / WHIM / GETAWAY / 3D autocorrelation token families)
- family stats are computed as concept-level `abs_z_mean`, `z_mean`, `z_std`
- tags are emitted from thresholded family stats (e.g., `torsionally active geometry`, `shape-anisotropic geometry`, `ring-strained geometry`)
- evidence also reports coverage diagnostics:
  - `family_coverage`, `n_family_matched_features`, `n_unmatched_features`, `top_unmatched_features`

Cross-modal semantic tags:
- `electrophilic reaction-center motif`: SMARTS-RX electrophile role + elevated QM electrophilicity
- `nucleophilic reaction-center motif`: SMARTS-RX nucleophile role + elevated QM nucleophilicity
- `planar conjugated electronic motif`: aromatic signal + geometry planarity + QM gap signal
- `polar donor-acceptor electronic motif`: pharmacophore HBD/HBA + elevated QM dipole

### 14.7 Chem-ACE persistence schema

Core tables (`explainability/chem_ace/db/models.py`):
- `runs`, `tasks`, `molecules`, `conformers`
- `patches`, `patch_embeddings`
- `concept_sets` (immutable/versioned)
- `concepts`, `concept_memberships`, `concept_tags`
- `cavs`, `tcav_epoch`, `mil_concept_epoch`

Concept-set snapshots are immutable and versioned per run.

---

## 15. Lambda-Vol Integration (Optional)

Integration path:
- `training/explainability_runtime.py::build_lambda_vol_callback`
- callback: `LambdaVolLightningCallback`
- monitor core: `LambdaVolMonitor`

Enabled by:
- `--run_lambda_vol`

Automatically enforces:
- `run_lambda_vol -> run_chem_ace = True`

### 15.1 Pressure state definition

For task `x`, concept `y`, epoch `t`:
- tracked inputs:
  - `TCAV[x,y,t]`
  - `attention_support[x,y,t]`
  - `prevalence[x,y,t]`
  - task-level entropy/witness/train/val/loss/calibration

TCAV evaluation protocol in current runtime:
- CAV is trained on monitor-train split.
- Directional-derivative scoring is evaluated on monitor-holdout split when feasible.
- Holdout behavior is controlled by:
  - `lambda_vol_tcav_holdout_fraction` (default `0.2`; `<=0` disables holdout)
  - `lambda_vol_tcav_holdout_min_samples` (default `16`)
- Repeat-level significance is computed with:
  - raw binomial p-values
  - Bonferroni-corrected p-values
  - configurable `lambda_vol_tcav_significance_alpha` (default `0.05`)
  - configurable `lambda_vol_tcav_bonferroni_m` (default `0` -> auto concept count)

Smoothing and drift:
- `tcav_smoothed_t = beta * tcav_smoothed_{t-1} + (1-beta)*tcav_t` with `beta = tcav_ema_beta`
- `delta_tcav_t = tcav_t - tcav_{t-1}`

Composite pressure:
- `rho_t = alpha * tcav_smoothed_t + (1-alpha) * attention_support_t`

Drift:
- `drift_t = clip(rho_t - rho_{t-1}, -drift_clip, +drift_clip)`

Default tracker config:
- `alpha=0.6`
- `tcav_ema_beta=0.8`
- `drift_clip=5.0`

### 15.2 Regime inference (`q(t)`)

Rule-based labels:
- `warmup`
- `fitting`
- `stable_generalization`
- `overfit_onset`
- `refit`

Default thresholds:
- `warmup_epochs=3`
- `val_slope_small=1e-3`
- `overfit_gap_threshold=0.03`
- `entropy_drop_threshold=0.05`
- `concentration_rise_threshold=0.05`

Uses recent slope heuristics over aggregated per-task history (window up to 4).

### 15.3 Dynamics model (discrete)

For matrix `rho` (tasks x concepts):
- `regime_core = A_q * rho` where `A_q = regime_a[regime_label]`
- `trend_loop = trend_coeff * prev_drift`
- `revert_loop = -revert_coeff * (rho - running_mean_rho)`
- `context_term = context_scale * tanh(mean(context_covariates))` broadcasted
- `feedback = task_coupling @ rho + rho @ concept_coupling` (optional)
- `dissipation = lambda_damping * rho + phi_state_damping * rho^2`
- `predicted_next = rho + regime_core + trend_loop + revert_loop + context_term + feedback - dissipation`

Default dynamics config:
- `lambda_damping=0.08`
- `phi_state_damping=0.0`
- `trend_coeff=0.25`
- `revert_coeff=0.20`
- `context_scale=0.15`
- `use_cross_task_coupling=True`
- `use_cross_concept_coupling=True`
- regime gains:
  - warmup `0.15`
  - fitting `0.08`
  - stable_generalization `0.02`
  - overfit_onset `0.18`
  - refit `0.06`

### 15.4 Alerts and collapse detection

Detector computes:
- concentration metrics per task:
  - normalized entropy
  - gini
  - top-k mass
- runaway score per task/concept:
  - `runaway = max(drift, 0) / (abs(dissipation)+1e-6)`
- trend-vs-dissipation matrix:
  - `drift - dissipation`

Default alert thresholds:
- `runaway_threshold=1.0`
- `concentration_top_k=5`
- `concentration_entropy_drop_alert=0.10`
- `concentration_topk_mass_alert=0.75`
- `blocked_concept_positive_drift=0.05`

### 15.5 Recommendation policy

Alert-driven recommendation engine (default enabled, auto-action disabled):
- Runaway pressure:
  - concept-balanced batching
  - increase attention-entropy regularization
- Concept collapse:
  - hard-negative mining
  - increase dropout/damping
- Blocked concept drift:
  - activate blocked-concept penalty
  - oversample counterexamples

Default policy:
- `enabled=True`
- `auto_action=False`

### 15.6 Lambda-Vol artifacts

Exporter writes under `<lambda_vol_output_dir>/<run_id>/`:
- `concept_pressure_tensors.npz`
- `concept_pressure_long.csv`
- optional `concept_pressure_long.parquet`
- `task_metrics_long.csv`
- `concept_xyz.csv`
- `pressure_lattice_long.csv`
- plotly HTMLs:
  - concept manifold per task
  - pressure lattice
  - heatmaps
  - trend-vs-dissipation
  - optional coupling graph
- `alerts.json`, `alerts.md`
- `recommendations.json`
- `trend_vs_dissipation.csv`
- `diagnostics_summary.md`
- optional VTK (`concept_pressure.vtp`)
- `metadata.json`
- `tcav_significance/tcav_significance_epoch_XXXX.csv`

### 15.7 Lambda-Vol persistence schema

Tables (`explainability/lambda_vol/db/models.py`):
- `lv_runs`
- `lv_pressure_epoch`
- `lv_task_epoch`
- `lv_alerts`
- `lv_recommendations`

Query APIs include:
- planar/conjugated rising pressure
- high TCAV low prevalence
- top trend loops
- recommendations history

---

## 16. Integrated Explainability in Final Training

When enabled in final run:
1. Chem-ACE bundle prepared in two phases:
   - fit/discover on train-only IDs (anti-leakage)
   - infer memberships on leaderboard IDs using frozen train centroids (no reclustering)
2. Optional concept-guided RL controller prepared:
   - target concepts are selected per task from concepts frequent in positive train samples
   - train dataset enables molecule/conformer metadata for attention-to-concept alignment
   - model loss gets a guidance bonus term: `L_total = L_base - s_t * alignment_t`
   - `s_t` is policy action (guidance scale) sampled/updated by REINFORCE callback
3. Optional Lambda-Vol callback attached to Lightning trainer.
4. On each validation epoch end:
   - monitor loader sampled from leaderboard set
   - per-epoch frames collected (TCAV + attention + task metrics)
   - TCAV uses holdout-eval protocol and writes repeat-level significance report
   - monitor step updates rho/regime/dynamics/alerts/recommendations and DB
5. On fit end:
   - Lambda-Vol exports finalized artifacts
   - Concept-RL policy history is exported when enabled
6. Final JSON summary:
   - `final_best_train_vs_leaderboard/explainability_artifacts.json`

Payload includes paths for Chem-ACE, Lambda-Vol, and Concept-RL artifacts (when enabled).

---

## 17. CLI Surface (Current Behavior)

Main parser: `entrypoints/hpo_pipeline.py`

Key controls:
- Family suite:
  - `--run_family_suite`
  - `--model_families catboost_st mt_2d mt_2d3d mt_3d`
  - `--calibration_method {platt,isotonic,temperature}`
  - `--best_params_dir <dir>`
- HPO:
  - `--run_hpo`
  - `--best_params_json`
  - `--trials` (and compatibility alias `--trials_mil`)
- Splits/folds:
  - `--use_splits ...` (HPO dataset filter)
  - `--folds ...` (optional explicit fold list)
  - `--leaderboard_split` (final validation split name)
- Runtime:
  - `--max_epochs`, `--patience`, `--seed`
  - `--nn_accelerator`, `--nn_devices`, `--precision`
  - `--num_workers`, `--pin_memory`
- Explainability:
  - optional raw semantic tables:
    - `--feat3d_raw`
    - `--feat3d_qm_raw`
  - Chem-ACE flags and limits
  - Lambda-Vol flags and monitoring limits
  - Concept-RL flags:
    - `--run_concept_rl`
    - `--run_concept_rl_ablation`
    - `--concept_rl_top_k_per_task`
    - `--concept_rl_min_pos_coverage`
    - `--concept_rl_init_scale`
    - `--concept_rl_max_scale`
    - `--concept_rl_policy_lr`
    - `--concept_rl_policy_sigma`
    - `--concept_rl_reward_alignment_w`
    - `--concept_rl_baseline_momentum`

Automatic dependency normalization:
- `--run_lambda_vol` implies `--run_chem_ace`
- `--run_concept_rl` implies `--run_chem_ace`
- `--run_concept_rl_ablation` implies `--run_chem_ace`

Compatibility flags still accepted:
- `--do_mil` (MIL-only pipeline)
- `--export_leaderboard_attn` (deprecated compatibility; final eval/export always runs)

---

## 18. Reproducibility and Seed Usage

Global setup:
- `set_all_seeds(seed)` sets numpy, torch, cuda, and Lightning worker seed behavior.

Notable deterministic offsets:
- CV fold run seed: `seed + 5000*fold_id + trial.number`
- CV train dataset seed: `seed + fold_id`
- CV val dataset seed: `seed + 999 + fold_id`
- CV balanced sampler seed: `seed + 1000*fold_id + trial.number`
- Final balanced sampler seed: `seed + 4242`
- Lambda-Vol monitor loader seed: `seed + 707`
- Final attention export dataset seed: `seed + 123`

---

## 19. Guardrails and Failure Modes

Explicit checks/fail-fast behavior:
- Missing required columns in inputs -> `ValueError`
- Missing IDs during 2D alignment -> `ValueError`
- Empty leaderboard split in final stage -> `ValueError`
- Empty patch set in Chem-ACE -> `RuntimeError`
- Missing RDKit when Chem-ACE requested -> `RuntimeError`
- Missing layer activation hook capture -> `RuntimeError`
- Shape mismatch in tracker/dynamics -> `ValueError`
- Unknown registry names -> `ValueError`

Graceful degradation:
- HDBSCAN unavailable -> skip hdbscan clustering with warning
- Plotly unavailable -> write fallback text for HTML
- Parquet engine unavailable in attention export -> fallback CSV
- Optional VTK export only when PyVista available

---

## 20. Known Approximations and Design Choices

1. Fluorescence auxiliary duplication
- Two base regression targets are duplicated to 4 outputs.
- This is an intentional shape-alignment approximation.

2. Task-weighted metrics
- Sample weights applied only to tasks `(0,1)` in AP/AUC calculations.
- Tasks `(2,3)` are unweighted in metric computation.

3. Concept pressure composition
- `rho` is a convex blend of smoothed TCAV and attention support.
- Chosen for stability and interpretability, not physical realism.

4. Dynamics linearity
- Lambda-Vol dynamics are linear-plus-damping heuristics with bounded context projection.
- Intended for monitoring/control signals, not mechanistic simulation.

5. Chem-ACE integrated embedding strategy in final pipeline
- Uses modality-separated hybrid local+context spaces (`2d`, `3d_geom`, `3d_qm`) for discovery.
- This remains a deterministic/scalable approximation versus strict ACE discovery directly in model bottleneck activations.

6. Objective choice
- `macro_plus_min` explicitly trades global gain vs weakest-task protection.
- `min_w` controls that trade-off.

---

## 21. Extension Points (No Spaghetti Path)

### 21.1 Add new model components

- New 2D embedder:
  - register via `register_2d_embedder(name, builder)`
- New 3D embedder:
  - register via `register_3d_embedder(name, builder)`
- New aggregator:
  - register via `register_aggregator(name, builder)`
- New predictor family:
  - register via `register_predictor(name, builder)`

### 21.2 Add new Chem-ACE patching/tagging behavior

- New patch generator implementing `PatchGenerator`
- Include in `CompositePatchGenerator`
- Add semantic tag rules or custom naming registry JSON

### 21.3 Add new Lambda-Vol logic

- Custom regime classifier via `RegimeClassifier` protocol
- Custom action hooks via `InterventionActionHook`
- Custom provider backends via provider protocols

---

## 22. Artifact Index (What to Expect After a Full Run)

In `--study_dir`:
- `run_meta.json`
- HPO artifacts (if `--run_hpo`):
  - `multimodal_mil_aux_gpu.sqlite3`
  - `multimodal_mil_aux_gpu_trials.csv`
  - `multimodal_mil_aux_gpu_best_params.json`
  - `multimodal_mil_aux_gpu_best_fold_metrics.json`
- final directory:
  - `final_best_train_vs_leaderboard/leaderboard_eval.json`
  - `final_best_train_vs_leaderboard/leaderboard_auc_per_task.csv`
  - `leaderboard_attn.csv` (or custom `--attn_out`)
  - optional `final_best_train_vs_leaderboard/explainability_artifacts.json`

If `--run_family_suite` is enabled:
- per-family HPO sqlite/trials artifacts:
  - `catboost_st.sqlite3`, `catboost_st_trials.csv`
  - `mt_2d_mil.sqlite3`, `mt_2d_mil_trials.csv`
  - `mt_2d3d_mil.sqlite3`, `mt_2d3d_mil_trials.csv`
  - `mt_3d_mil.sqlite3`, `mt_3d_mil_trials.csv`
- best params directory:
  - `best_params/catboost_st.json`
  - `best_params/mt_2d.json`
  - `best_params/mt_2d3d.json`
  - `best_params/mt_3d.json`
- per-family leaderboard metrics/predictions:
  - `leaderboard_metrics_<family>.csv`
  - `leaderboard_preds_<family>.csv`
  - `leaderboard_metrics_<family>_calibrated.csv`
  - `leaderboard_preds_<family>_calibrated.csv`
- calibration metadata:
  - `calibration/<family>_calib.json`
- blended outputs:
  - `leaderboard_metrics_blend.csv`
  - `leaderboard_preds_blend.csv`
  - `blend_weights.json`

If Chem-ACE enabled:
- Chem-ACE output dir with DB, cache, and pipeline summary.

If Lambda-Vol enabled:
- Lambda-Vol output dir with tensor exports, long tables, HTML visualizations, alerts, recommendations, and metadata.

---

## 23. Full Default Constant Reference

This section lists defaults exactly as defined in typed configs and CLI parser, so no default constant is implicit.

### 23.1 `BackboneConfig` defaults

- `mol_hidden = 1024`
- `mol_layers = 2`
- `mol_dropout = 0.10`
- `inst_hidden = 256`
- `inst_layers = 3`
- `inst_dropout = 0.05`
- `proj_dim = 512`
- `attn_heads = 8`
- `attn_dropout = 0.05`
- `mixer_hidden = 512`
- `mixer_layers = 3`
- `mixer_dropout = 0.05`
- `activation = \"GELU\"`
- `mol_embedder_name = \"mlp_v3_2d\"`
- `inst_embedder_name = \"mlp_v3_3d\"`
- `aggregator_name = \"task_attention_pool\"`
- `predictor_name = \"mlp_v3\"`

### 23.2 `HeadConfig` defaults

- `num_layers = 2`
- `dropout = 0.1`
- `stochastic_depth = 0.1`
- `fc2_gain_non_last = 1e-2`

### 23.3 `OptimizationConfig` defaults

- `lr = 8e-5`
- `weight_decay = 3e-6`

### 23.4 `RuntimeConfig` defaults

- `batch_size = 128`
- `accumulate_grad_batches = 8`

### 23.5 `SamplerConfig` defaults

- `rare_oversample_mult = 0.0`
- `rare_target_prev = 0.10`
- `rare_prev_thr = None`
- `sample_weight_cap = 10.0`
- `use_balanced_batch_sampler = True`
- `batch_pos_fraction = 0.35`
- `min_pos_per_batch = 1`

### 23.6 `LossWeightingConfig` defaults

- `lam_t0 = None`
- `lam_t1 = None`
- `lam_t2 = None`
- `lam_t3 = None`
- `lam_floor = 0.25`
- `lam_ceil = 3.5`
- `lambda_power = 1.0`
- `posw_clip_t0 = None`
- `posw_clip_t1 = None`
- `posw_clip_t2 = None`
- `posw_clip_t3 = None`
- `pos_weight_clip = 50.0`
- `gamma_t0 = 0.0`
- `gamma_t1 = 0.0`
- `gamma_t2 = 0.0`
- `gamma_t3 = 0.0`
- `lambda_aux_abs = 0.05`
- `lambda_aux_fluo = 0.05`
- `lambda_aux_bitmask = 0.05`
- `bitmask_group_top_k = 6`
- `bitmask_group_weight_alpha = 0.5`
- `bitmask_group_weight_cap = 5.0`
- `reg_loss_type = \"mse\"`

### 23.7 `ObjectiveConfig` defaults

- `mode = \"macro_plus_min\"`
- `min_w = 0.30`

### 23.8 Study/pruner defaults (`StudyConfig`)

- `direction = \"maximize\"`
- `pruner_kind = \"percentile\"`
- `pruner_warmup_steps = 8`
- `pruner_startup_trials = 10`
- `pruner_percentile = 25.0`

### 23.9 Final explainability defaults (`FinalExplainabilityConfig`)

- `run_chem_ace = False`
- `run_lambda_vol = False`
- `curated_smiles_col = \"curated_SMILES\"`
- `chem_ace_output_dir = None`
- `chem_ace_db_uri = None`
- `chem_ace_max_ids = 0` (`0` means use all IDs in scope)
- `chem_ace_max_confs_per_id = 0` (`<=0` means use all conformers)
- `chem_ace_embed_dim_2d = 64`
- `chem_ace_embed_dim_3d_geom = 64`
- `chem_ace_embed_dim_3d_qm = 64`
- `chem_ace_context_dim = 16`
- `chem_ace_context_alpha = 0.2`
- `chem_ace_qm_gating = True`
- `chem_ace_max_2d_dim = 0` (deprecated compatibility flag, ignored by hybrid runtime)
- `chem_ace_max_3dqm_dim = 0` (deprecated compatibility flag, ignored by hybrid runtime)
- `chem_ace_persist_patch_embeddings = False` (avoid writing millions of per-patch embedding files by default)
- `chem_ace_top_concepts = 0` (`<=0` means keep all discovered concepts)
- `lambda_vol_output_dir = None`
- `lambda_vol_db_uri = None`
- `lambda_vol_layer_name = \"mixer_post_norm\"`
- `lambda_vol_top_concepts = 0` (`<=0` means evaluate all concepts from Chem-ACE)
- `lambda_vol_monitor_max_samples = 512`
- `lambda_vol_tcav_repeats = 2`
- `lambda_vol_random_counterexamples = 96`
- `lambda_vol_min_concept_samples = 8`
- `lambda_vol_tcav_holdout_fraction = 0.2`
- `lambda_vol_tcav_holdout_min_samples = 16`
- `lambda_vol_tcav_significance_alpha = 0.05`
- `lambda_vol_tcav_bonferroni_m = 0` (`<=0` auto-uses number of concepts in scope)

### 23.10 CLI parser defaults (`entrypoints/hpo_pipeline.py`)

Data/columns:
- `--id_col ID`
- `--conf_col conf_id`
- `--split_col split`
- `--fold_col cv_fold`
- `--feat3d_raw None`
- `--feat3d_qm_raw None`
- `--use_splits train`
- `--folds None`

Runtime/HPO:
- `--max_epochs 150`
- `--patience 20`
- `--trials 50`
- `--trials_mil None` (compatibility alias to `--trials`)
- `--run_family_suite False`
- `--model_families catboost_st mt_2d mt_2d3d mt_3d`
- `--calibration_method platt`
- `--best_params_dir None` (defaults to `<study_dir>/best_params` in family-suite mode)
- `--seed 0`
- `--nn_accelerator gpu`
- `--nn_devices 1`
- `--precision 16-mixed`
- `--num_workers -1` (auto-resolve)
- `--cpu_workers -1` (auto-resolve CPU pool budget)
- `--pin_memory False`

Final export:
- `--leaderboard_split leaderboard`
- `--attn_out None` (defaults to `<study_dir>/leaderboard_attn.csv`)
- `--export_leaderboard_attn False` (compatibility only)
- `--do_mil False` (compatibility only; pipeline remains MIL-only)

Explainability:
- `--run_chem_ace False`
- `--run_lambda_vol False`
- `--run_concept_rl False`
- `--run_concept_rl_ablation False`
- `--curated_smiles_col curated_SMILES`
- `--chem_ace_output_dir None`
- `--chem_ace_db_uri None`
- `--chem_ace_max_ids 0` (`0` means use all IDs in scope)
- `--chem_ace_max_confs_per_id 0` (`<=0` means use all conformers)
- `--chem_ace_embed_dim_2d 64`
- `--chem_ace_embed_dim_3d_geom 64`
- `--chem_ace_embed_dim_3d_qm 64`
- `--chem_ace_context_dim 16`
- `--chem_ace_context_alpha 0.2`
- `--chem_ace_qm_gating` (`--no-chem_ace_qm_gating` to disable)
- `--chem_ace_max_2d_dim 0` (deprecated compatibility flag, ignored by hybrid runtime)
- `--chem_ace_max_3dqm_dim 0` (deprecated compatibility flag, ignored by hybrid runtime)
- `--chem_ace_persist_patch_embeddings false` (enable only if you explicitly need patch-embedding files/rows)
- `--chem_ace_top_concepts 0` (`<=0` means keep all discovered concepts)
- `--lambda_vol_output_dir None`
- `--lambda_vol_db_uri None`
- `--lambda_vol_layer_name mixer_post_norm`
- `--lambda_vol_top_concepts 0` (`<=0` means evaluate all concepts from Chem-ACE)
- `--lambda_vol_monitor_max_samples 512`
- `--lambda_vol_tcav_repeats 2`
- `--lambda_vol_random_counterexamples 96`
- `--lambda_vol_min_concept_samples 8`
- `--lambda_vol_tcav_holdout_fraction 0.2`
- `--lambda_vol_tcav_holdout_min_samples 16`
- `--lambda_vol_tcav_significance_alpha 0.05`
- `--lambda_vol_tcav_bonferroni_m 0` (`<=0` auto-uses number of concepts in scope)
- `--concept_rl_top_k_per_task 8` (`<=0` means all passing concepts, uncapped)

---

## 24. Canonical Runtime Commands

From repository root (`../` relative to this file):

Run HPO + final:
```bash
python ../opt_net_fast.py \
  --labels <labels.csv> \
  --feat2d_scaled <scaled_2d.csv> \
  --feat3d_scaled <scaled_3d.csv> \
  --feat3d_qm_scaled <scaled_3d_quantum.csv> \
  --feat3d_raw <raw_3d.csv> \
  --feat3d_qm_raw <raw_3d_quantum.csv> \
  --study_dir <out_dir> \
  --use_splits train \
  --run_hpo \
  --trials 50
```

Skip HPO, reuse best params JSON:
```bash
python ../opt_net_fast.py \
  --labels <labels.csv> \
  --feat2d_scaled <scaled_2d.csv> \
  --feat3d_scaled <scaled_3d.csv> \
  --feat3d_qm_scaled <scaled_3d_quantum.csv> \
  --feat3d_raw <raw_3d.csv> \
  --feat3d_qm_raw <raw_3d_quantum.csv> \
  --study_dir <out_dir> \
  --best_params_json <multimodal_mil_aux_gpu_best_params.json>
```

Run 4-family suite (HPO + refit + calibration + blend):
```bash
python ../opt_net_fast.py \
  --labels <labels.csv> \
  --feat2d_scaled <scaled_2d.csv> \
  --feat3d_scaled <scaled_3d.csv> \
  --feat3d_qm_scaled <scaled_3d_quantum.csv> \
  --study_dir <out_dir> \
  --use_splits train \
  --leaderboard_split leaderboard \
  --run_family_suite \
  --run_hpo \
  --trials 50
```

Run 4-family suite without HPO (reuse per-family JSONs):
```bash
python ../opt_net_fast.py \
  --labels <labels.csv> \
  --feat2d_scaled <scaled_2d.csv> \
  --feat3d_scaled <scaled_3d.csv> \
  --feat3d_qm_scaled <scaled_3d_quantum.csv> \
  --study_dir <out_dir> \
  --run_family_suite \
  --best_params_dir <out_dir>/best_params
```

`--feat3d_raw` and `--feat3d_qm_raw` are optional and should be supplied together.

Enable explainability in final run:
```bash
python ../opt_net_fast.py ... --run_hpo --run_lambda_vol
```

(`--run_lambda_vol` auto-enables Chem-ACE.)

## 25. Multi-Family Optimization + Calibration + Blending Suite

Implementation:
- runner: `entrypoints/family_suite.py`
- trigger: `--run_family_suite`
- supported families (`FAMILY_CHOICES`):
  - `catboost_st`
  - `mt_2d`
  - `mt_2d3d`
  - `mt_3d`

### 25.1 Family definitions

- `catboost_st`:
  - single-task CatBoost classifier per endpoint (`4` models total)
  - input: 2D features only
  - independent per-task training/prediction
- `mt_2d`:
  - uses `MILTaskAttnMixerWithAux` with only 2D modality active
  - instance bags are synthetic one-instance dummy bags to keep dataset/trainer contracts unchanged
- `mt_2d3d`:
  - full current MIL model path
  - input: 2D + 3D geometry + 3D QM
- `mt_3d`:
  - uses `MILTaskAttnMixerWithAux` with 3D geometry + 3D QM active and `mol_dim=0`
  - input: 3D only

### 25.2 HPO protocol (train CV)

Common CV contract:
- folds from `--fold_col` inside rows selected by `--use_splits`
- trial objective is CV mean macro PR-AUC:
  - per fold: `macro_pr_auc = mean(AP_task0..AP_task3)`
  - trial value: mean of fold macro PR-AUC

MIL-family HPO details:
- search space: shared `training/search_space.py`
- family objective enforces pure macro PR-AUC (`min_w` overridden to `0.0` in family-suite CV)
- training engine: same Lightning fold trainer stack (`MILFoldTrainer`)
- pruning: Optuna percentile pruner with configured warmup (`--pruner_warmup_steps`)

CatBoost-family HPO details:
- search space is CatBoost-specific in `family_suite.py`
- one parameter set is optimized jointly across 4 tasks using macro PR-AUC over tasks/folds
- model fit remains per-task

Best-params artifacts:
- directory: `<study_dir>/best_params/` (or `--best_params_dir`)
- file per family: `<family>.json`
- payload includes:
  - `family`
  - `best_params`
  - `best_value_macro_ap_cv`

### 25.3 Final refit + leaderboard evaluation

For each selected family:
1. Rebuild family-specific train/leaderboard tensors from same source tables.
2. Refit model(s) on train split with best params.
3. Predict probabilities on leaderboard split.
4. Save metrics:
   - per-task PR-AUC
   - per-task ROC-AUC
   - per-task NLL
   - per-task Brier
   - macro row over tasks

Outputs per family:
- `leaderboard_preds_<family>.csv`
- `leaderboard_metrics_<family>.csv`
- for MIL families also `/<study_dir>/<family>/leaderboard_eval.json`

### 25.4 Post-hoc calibration stage (leaderboard-fitted)

Calibration is applied independently per `(family, task)` on leaderboard predictions:
- supported methods:
  - `platt` (default)
  - `isotonic`
  - `temperature`
- selected by `--calibration_method`

Important contract:
- calibrators are explicitly fit on leaderboard for post-hoc evaluation/ensembling quality.
- this is evaluation-time calibration; not a leakage-safe deployment protocol.

Saved calibration artifacts:
- `calibration/<family>_calib.json` with per-task calibrator parameters
- `leaderboard_preds_<family>_calibrated.csv`
- `leaderboard_metrics_<family>_calibrated.csv`

### 25.5 Blending/ensemble stage

Input to blend stage:
- calibrated probabilities from all selected families
- common leaderboard ID intersection across families

Blend model:
- per task logistic stacking:
  - features: calibrated family probabilities for that task
  - target: task label on leaderboard
  - learned coefficients and intercept stored per task
- single-class fallback uses uniform averaging

Blend outputs:
- `leaderboard_preds_blend.csv`
- `leaderboard_metrics_blend.csv`
- `blend_weights.json`:
  - family list
  - per-task coefficients
  - per-task intercept
  - fallback metadata where applicable

### 25.6 Unified artifact contract (requested outputs)

The family-suite run provides the requested artifact classes:
- `best_params/<family>.json`
- `leaderboard_metrics_<family>.csv`
- `calibration/<family>_calib.json`
- `leaderboard_metrics_blend.csv`
- `blend_weights.json`

Additional useful artifacts produced by implementation:
- `leaderboard_preds_<family>.csv`
- `leaderboard_preds_<family>_calibrated.csv`
- `leaderboard_preds_blend.csv`
 
## Ricci Geometry Layer (Lambda-Vol Integration)

Detailed Ricci reference:
- `RICCI_FLOW_README.md`

The explainability pack now includes a discrete graph-Ricci module integrated into Lambda-Vol epoch monitoring:

- Build per-task concept graph from sparse concept co-activation (not global prevalence/correlation mix).
- Node score per concept is `w_tcav * tcav_ema + w_attention * attn_ema` (2D concepts force `w_attn=0`).
- Keep top-k concepts per task/epoch before edge construction.
- Same-modality edges use Jaccard/co-occurrence.
- Cross-modality (2D<->3D) edges use weighted overlap in the same molecules.
- Compute Forman-Ricci curvature on concept edges.
- Run Ricci-flow-style edge reweighting (iterative length update; similarity is inverse length).
- Export per-edge/per-task geometry artifacts:
  - `ricci_edges_long.csv`
  - `ricci_task_summary.csv`
  - `ricci_flow_tensors.npz`
- Emit curvature-driven alerts:
  - `ricci_negative_curvature_surge`
  - `ricci_bridge_concentration`
  - `ricci_extreme_negative_bridge`
- Feed flowed concept graph into Λ-Vol concept coupling when enabled (`use_flow_as_concept_coupling`).

This geometry layer is an additional control signal and does not replace TCAV/attention/prevalence. It is designed to flag concept bottlenecks and shortcut-like bridge structure during training.

### Prediction + Text Explanation Link

Final prediction export is linked with Chem-ACE semantic concepts and optional Ricci bridge diagnostics:

- Base table: `leaderboard_attn.csv` (or requested path) with `pred_*`, `pred_label_*`, `attn_*`.
- Enriched table: `*_explained.csv` with:
  - `top_concepts_<task>`
  - `top_concept_labels_<task>`
  - `prediction_explanation_<task>`
  - `prediction_explanation`

Explanation text is built from:
- concept activation at molecule/conformer level (`concept_mol_map`, `concept_conf_map`),
- semantic naming/tags (`label_auto`, concept tags),
- task attention weight at that row,
- optional Ricci bridge score from latest epoch (`ricci_edges_long.csv`).
