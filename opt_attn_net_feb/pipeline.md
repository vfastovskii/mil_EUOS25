# End-to-End Pipeline Runbook (All Explainability + RL Ablation)

This document describes a full run of the MIL pipeline with:

- Chem-ACE (concept discovery + semantic tagging),
- Lambda-Vol (concept-pressure dynamics + TCAV/CAV monitoring),
- Ricci-flow diagnostics,
- Concept-RL ablation (`no_rl` vs `with_rl`) with automatic comparison.

It is aligned with current code paths in:

- `/Users/vfastovskii/Desktop/mil_explainability_2026/opt_attn_net_feb/entrypoints/hpo_pipeline.py`
- `/Users/vfastovskii/Desktop/mil_explainability_2026/opt_attn_net_feb/training/execution.py`
- `/Users/vfastovskii/Desktop/mil_explainability_2026/opt_attn_net_feb/training/explainability_runtime.py`
- `/Users/vfastovskii/Desktop/mil_explainability_2026/opt_attn_net_feb/explainability/lambda_vol/exporters.py`

## 1) Objective of this run

Primary training objective:

- Optimize and evaluate MIL multitask model quality with PR-AUC focus.
- In HPO mode, objective is `macro_plus_min` (mean PR-AUC + weakest-task PR-AUC weighting).

Explainability objective:

- Build leakage-safe concept system from train scope only.
- Monitor concept reliance over epochs (TCAV + attention/prevalence signals).
- Detect instability/collapse and generate intervention recommendations.

RL ablation objective:

- Run final training twice with same seed/params:
1. `no_rl`: concept RL disabled
2. `with_rl`: concept RL enabled
- Compare `macro_pr_auc`, `min_task_pr_auc`, leaderboard metrics, and RL policy stability.

## 2) Inputs required

Required files:

- labels CSV (`--labels`) with columns including:
  - ID column (default `ID`)
  - split column (default `split`)
  - fold column for CV (default `cv_fold`)
  - curated SMILES column (default `curated_SMILES`)
- 2D features CSV (`--feat2d_scaled`)
- 3D geom features CSV (`--feat3d_scaled`)
- 3D QM features CSV (`--feat3d_qm_scaled`)

Optional semantic-only raw instance tables:

- raw 3D geometry CSV (`--feat3d_raw`)
- raw 3D QM CSV (`--feat3d_qm_raw`)

Raw tables are used only for Chem-ACE semantic descriptor/tag summaries.
They do not replace scaled model inputs used by training/HPO/TCAV.
Both raw tables must be provided together.

Optional but strongly recommended for 3D patching:

- precomputed conformer SDF (`--chem_ace_conformer_sdf`)
- conf id property name (`--chem_ace_sdf_conf_id_prop`)
  - if property missing, pipeline falls back to SDF record `_Name`

## 3) Recommended commands

### A) Skip HPO, use existing best params, run full explainability + RL ablation

```bash
python /Users/vfastovskii/Desktop/mil_explainability_2026/opt_attn_net_feb/opt_net_fast.py \
  --labels /path/master_table_labels_final_modelling_ready_1401_with_cv_split.csv \
  --feat2d_scaled /path/scaled_2d.csv \
  --feat3d_scaled /path/scaled_3d.csv \
  --feat3d_qm_scaled /path/scaled_3d_quantum.csv \
  --feat3d_raw /path/raw_3d.csv \
  --feat3d_qm_raw /path/raw_3d_quantum.csv \
  --study_dir /path/study_run \
  --use_splits train \
  --leaderboard_split leaderboard \
  --best_params_json /path/best_params_279756.json \
  --run_chem_ace \
  --run_lambda_vol \
  --lambda_vol_run_ricci \
  --run_concept_rl_ablation \
  --curated_smiles_col curated_SMILES \
  --chem_ace_conformer_sdf /path/confs.sdf \
  --chem_ace_sdf_conf_id_prop _Name \
  --max_epochs 150 \
  --patience 10 \
  --seed 0 \
  --nn_accelerator gpu \
  --nn_devices 1 \
  --precision 16-mixed \
  --num_workers 23 \
  --cpu_workers 18
```

### B) Run HPO first, then final ablation

Use same command, replace `--best_params_json ...` with:

```bash
--run_hpo --trials 50
```

## 4) What happens step by step

### Stage 0: CLI normalization and run metadata

- Parses args.
- If any of `--run_lambda_vol`, `--run_concept_rl`, `--run_concept_rl_ablation` is set, it auto-enables `run_chem_ace`.
- Writes `run_meta.json` in `study_dir`.

### Stage 1: HPO data build

- Loads labels and filters to `--use_splits` (default `train`).
- Aligns 2D features by ID.
- Loads and merges 3D + QM instance features.
- Builds bag indices and fold structure for CV.

### Stage 2: Params source

- If `--run_hpo`: runs Optuna CV and stores best params JSON.
- Else: loads params from `--best_params_json` or `<study_dir>/multimodal_mil_aux_gpu_best_params.json`.

### Stage 3: Final phase (ablation-aware orchestration)

If `--run_concept_rl_ablation` is **off**:

- One final run executes at `<study_dir>/final_best_train_vs_leaderboard`.

If `--run_concept_rl_ablation` is **on**:

- Two final runs execute:
1. `<study_dir>/ablation/no_rl`
2. `<study_dir>/ablation/with_rl`

Both runs reuse identical best params and seed; only `run_concept_rl` differs.

### Stage 4: Final run internals (per run)

1. Train/leaderboard split prep:
- train = `split == "train"`
- val = `split == --leaderboard_split`

2. Build model/dataloaders and class imbalance controls.

3. Explainability prep:

- Chem-ACE bundle preparation runs first when enabled.
- Anti-leakage logic:
  - concept discovery is train-only.
  - leaderboard gets inference-only memberships via nearest frozen centroids.
- Semantic-source logic:
  - if both raw tables are provided, Chem-ACE semantic tagger uses raw per-conformer vectors;
  - otherwise it falls back to scaled vectors.
  - source is logged as `explainability.chem_ace.semantics_instances source=raw|scaled`.

4. Optional callbacks during training:

- Lambda-Vol callback: computes TCAV/pressure each validation epoch.
- Concept-RL callback (only `with_rl`):
  - samples action at train epoch start,
  - updates policy on validation epoch end from reward.

5. Best checkpoint load and leaderboard evaluation.

6. Export attention/predictions and explainability payload.

## 5) Chem-ACE details in this run

Train-scope discovery flow:

1. `generate_patches` (2D + 3D/QM-aware patch records)
2. `embed_patches`
3. `discover_concepts`
4. `tag_concepts`
5. `calibrate_semantics` (if enabled)

Infer-scope flow (leaderboard IDs):

1. `generate_patches` (infer scope)
2. `embed_patches`
3. `infer_memberships` to frozen train centroids (no reclustering)
4. persist inferred memberships

Exports include:

- concept DB (`chem_ace.sqlite3` by default)
- `a_priori_tags.csv`, `a_priori_vs_concepts.csv`
- infer-scope variants of above
- calibrated tag outputs when enabled

## 6) Lambda-Vol details in this run

During training (epoch end), Lambda-Vol computes and logs:

- TCAV matrix per task x concept
- attention support / prevalence context
- concept pressure dynamics and alerts

Typical repeated logs:

- `explainability.lambda_vol.tcav.compute.start`
- `explainability.lambda_vol.tcav.compute.done`

This repetition is expected per validation epoch.

Final artifacts in `lambda_vol/`:

- `concept_pressure_tensors.npz`
- `concept_pressure_long.csv`
- `pressure_lattice_long.csv`
- `pressure_lattice.html`
- `alerts.json`
- `recommendations.json`
- concept manifold HTML files
- if Ricci enabled:
  - `ricci_edges_long.csv`
  - `ricci_task_summary.csv`
  - `ricci_flow_tensors.npz`
  - `concept_coupling_3d.html`

## 7) Concept-RL details in this run

Only active in `with_rl` ablation branch.

Policy mechanics:

- action sampled each train epoch start (`rl guidance scale`)
- update each validation epoch end with reward:
  - `reward_total = val_macro_ap + w * train_concept_alignment`
- target-concept cap:
  - `concept_rl_top_k_per_task > 0`: capped per task
  - `concept_rl_top_k_per_task <= 0`: uncapped all-passing concepts

Policy history is exported to:

- `<run>/final_best_train_vs_leaderboard/concept_rl_policy_history.json`

## 8) Final outputs (single run vs ablation)

### Per final run

Inside `<run>/final_best_train_vs_leaderboard/`:

- `leaderboard_eval.json`
- `leaderboard_auc_per_task.csv`
- attention + prediction table (`leaderboard_attn.csv` or custom `--attn_out`)
- explained prediction table (`*_explained.csv`) if Chem-ACE enabled
- `explainability_artifacts.json`
- optional `concept_rl_policy_history.json` (RL-active runs)

### Additional ablation summary outputs

At top `study_dir`:

- `final_concept_rl_ablation_comparison.csv`
- `final_concept_rl_ablation_comparison.json`
- `final_concept_rl_ablation_comparison.md`

Comparison includes:

- `macro_pr_auc` delta (`with_rl - no_rl`)
- `min_task_pr_auc` delta
- winner by macro and min-task metrics
- RL policy stability stats:
  - mean absolute step
  - sign-flip rate
  - tail std
  - `policy_converged`
  - `policy_likely_oscillating`

## 9) How to read logs quickly

All major stages use structured logs:

- `[START] <step>`
- `[PROGRESS] <step> ...`
- `[DONE] <step> elapsed_s=...`
- `[FAIL] <step> elapsed_s=... error=...`

For crash localization:

1. Find last `[START]` without matching `[DONE]`.
2. Use the paired `[FAIL]` line for exact failing stage.

## 10) Practical tuning knobs for large data

If runtime/memory/disk pressure is high:

- reduce patch growth:
  - `--chem_ace_local_radii 1`
  - `--chem_ace_target_total_patches` lower value
  - `--chem_ace_patch_cap_per_mol` fixed cap
- keep embedding persistence off (default):
  - `--no-chem_ace_persist_patch_embeddings`
- limit concept scope:
  - `--chem_ace_max_ids`
  - `--chem_ace_max_confs_per_id`
- control CPU pressure:
  - `--cpu_workers`
  - `--num_workers`

## 11) Expected success criteria for this run

You should have:

1. two complete final runs (`no_rl`, `with_rl`) with identical params/seed baseline control,
2. full explainability artifacts (Chem-ACE + Lambda-Vol + Ricci where enabled),
3. an ablation report that directly answers whether RL improved:
   - macro PR-AUC,
   - weakest-task PR-AUC,
   - and whether RL policy looked stable/convergent.
