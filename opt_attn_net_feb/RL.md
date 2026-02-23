# Reinforcement Learning Loop (Concept-RL)

This document describes the reinforcement learning loop implemented in this project for concept-guided MIL training.

Important scope note:

- This is the `Concept-RL` loop controlling concept-guidance scale in the MIL model.
- It is separate from Lambda-Vol recommendation policy (which is alert/recommendation logic, not policy-gradient training).

Primary implementation files:

- `callbacks/concept_rl.py`
- `models/multimodal_mil/model.py`
- `training/explainability_runtime.py`
- `training/execution.py`
- `entrypoints/hpo_pipeline.py`

## 1) Goal

The RL loop learns how strongly to bias training toward concept-aligned attention patterns.

In each epoch, the controller chooses a scalar action `a_t` (guidance scale). That action changes training loss:

- `L_total = L_base - a_t * alignment_t`

where:

- `L_base` is the multitask supervised loss.
- `alignment_t` is how much attention mass lands on target concepts for positive samples.

If stronger concept guidance helps validation quality and concept alignment, the controller increases expected action. If not, it decreases it.

## 2) Theory and policy-gradient update

The controller uses a REINFORCE-style Gaussian policy over one continuous action:

- policy: `a_t ~ N(mu_t, sigma^2)`, clipped to `[0, max_scale]`
- learnable parameter: `mu_t`
- fixed exploration: `sigma`

Reward at epoch end:

- `r_t = reward_model_t + w_align * reward_alignment_t`
- in code defaults:
  - `reward_model_t = val_macro_ap`
  - `reward_alignment_t = train_concept_alignment`
  - `w_align = concept_rl_reward_alignment_w` (default `0.25`)

Baseline and advantage:

- baseline is EMA: `b_t <- m * b_{t-1} + (1-m) * r_t`
- advantage: `A_t = r_t - b_t_before_update`

Policy-gradient step on `mu`:

- `grad_log_pi = (a_t - mu_t) / sigma^2`
- `mu_{t+1} = clip(mu_t + lr * A_t * grad_log_pi, 0, max_scale)`

This is implemented in `ConceptRLControllerCallback.on_validation_epoch_end`.

## 3) End-to-end loop logic in training

### Step A: build target concepts from Chem-ACE

Before final training, Chem-ACE concepts are prepared.

Then target concepts are selected per task from positive train IDs:

- top `k` concepts by positive-coverage
- only concepts with coverage >= `min_pos_coverage`

Implemented in:

- `training/explainability_runtime.py::build_positive_concept_targets`

### Step B: enable concept guidance in model

Model receives:

- per-task target concept IDs
- concept-to-(mol,conf) map
- concept-to-molecule map
- initial and max guidance scale

Implemented in:

- `models/multimodal_mil/model.py::configure_rl_concept_guidance`

### Step C: train-time alignment signal

When RL is enabled and metadata is available in batch:

1. model runs with `return_attn=True`
2. for each positive task label in batch:
   - normalize task attention over valid conformers
   - if matching `(mol_id, conf_id)` exists in target concept set, score is attention mass on those conformers
   - else if molecule belongs to target concept at molecule level, score is `1.0`
   - else score is `0.0`
3. average scores -> `train_concept_alignment`
4. apply `concept_bonus = rl_guidance_scale * train_concept_alignment`
5. optimize `L_total = L_base - concept_bonus`

Implemented in:

- `models/multimodal_mil/model.py::_concept_alignment_score`
- `models/multimodal_mil/model.py::training_step`

### Step D: policy action and update each epoch

At train epoch start:

- sample action `a_t`, set model guidance scale

At validation epoch end:

- read `val_macro_ap` and `train_concept_alignment`
- compute reward, advantage
- update policy mean `mu`
- log policy stats

Implemented in:

- `callbacks/concept_rl.py`

### Step E: persist RL trace

After fit:

- write `concept_rl_policy_history.json`
- include config + per-epoch records

Also exported into:

- `final_best_train_vs_leaderboard/explainability_artifacts.json`

## 4) Runtime activation and CLI controls

Enable RL:

- `--run_concept_rl`

This auto-enables Chem-ACE preparation in pipeline.

Main knobs:

- `--concept_rl_top_k_per_task`
- `--concept_rl_min_pos_coverage`
- `--concept_rl_init_scale`
- `--concept_rl_max_scale`
- `--concept_rl_policy_lr`
- `--concept_rl_policy_sigma`
- `--concept_rl_reward_alignment_w`
- `--concept_rl_baseline_momentum`

Current callback reward keys are fixed in final trainer wiring:

- `reward_key = "val_macro_ap"`
- `alignment_key = "train_concept_alignment"`

## 5) What this RL loop is optimizing in practice

The loop does not directly optimize endpoint logits by RL.

It optimizes one control parameter (guidance scale) that changes the supervised objective shape. So it is best interpreted as:

- policy-gradient control over a training hyperparameter,
- with reward tied to validation quality + concept alignment.

This design is stable and low-risk compared with full RL over model actions.

## 6) Failure modes and diagnostics

Common failure modes:

- no target concepts selected -> RL requested but disabled
- sparse concept matches -> weak/noisy alignment signal
- too large `sigma` -> unstable guidance oscillation
- too high `learning_rate` -> policy mean saturation at bounds
- reward scale mismatch (`val_macro_ap` vs alignment) -> one term dominates

Useful logs/metrics:

- `train_concept_alignment`
- `train_concept_bonus`
- `train_rl_guidance_scale`
- `rl_action_scale`
- `rl_policy_mean`
- `rl_reward_total`
- `rl_advantage`

## 7) Potential improvements

### 7.1 Better reward design

- normalize reward components (running z-score) before combining
- include task-floor signal (for example min AP) to avoid weak-task neglect
- include calibration or robustness penalty terms

### 7.2 Stronger policy model

- move from scalar global action to vector action per task: `a_t in R^{num_tasks}`
- actor-critic instead of vanilla REINFORCE to reduce variance
- learn state-conditioned policy from regime/context features (Lambda-Vol signals)

### 7.3 Better exploration and constraints

- anneal `sigma` over epochs
- trust-region or clipped updates on `mu`
- explicit monotonicity/safety constraints to avoid sudden guidance spikes

### 7.4 Better alignment signal

- weight concept matches by concept confidence/support
- soft matching using concept similarity instead of binary membership
- add negative-concept penalties (blocked concepts) directly in alignment term

### 7.5 Better credit assignment

- delayed reward smoothing (multi-epoch return)
- evaluate action on held-out mini-validation windows
- off-policy replay of `(action, reward)` history for robust updates

### 7.6 Better data and target set selection

- dynamic target concept refresh every N epochs
- separate target sets for rare endpoints
- include conformer-level prevalence weighting in target coverage

## 8) Minimal pseudocode

```text
prepare Chem-ACE concepts
build target_concepts_by_task from positive train IDs
configure model with target concept maps

initialize policy mean mu
for epoch t:
  sample action a_t ~ Normal(mu, sigma), clip [0, max_scale]
  set model.rl_guidance_scale = a_t

  train one epoch with:
    L_total = L_base - a_t * train_concept_alignment

  validate -> get val_macro_ap
  read train_concept_alignment
  reward r_t = val_macro_ap + w_align * train_concept_alignment

  advantage A_t = r_t - baseline
  baseline <- EMA(baseline, r_t)

  grad_log_pi = (a_t - mu) / sigma^2
  mu <- clip(mu + lr * A_t * grad_log_pi, 0, max_scale)

save policy history
```

## 9) Suggested next upgrades (practical order)

1. Add reward normalization and separate weights for `val_macro_ap` and `min_ap`.
2. Move to per-task guidance scales (4 actions) with independent means.
3. Add actor-critic baseline network using Lambda-Vol context features.
4. Add guardrails: update clipping, sigma schedule, and saturation alarms.
