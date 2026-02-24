# Usage

## Run optimization + final training

```bash
python /Users/vfastovskii/Desktop/mil_explainability_2026/opt_attn_net_feb/opt_net_fast.py ... --run_hpo --trials 50
```

CPU parallelization controls (recommended on GPU+CPU node):

- `--num_workers -1` : dataloader workers auto from node CPUs.
- `--cpu_workers -1` : CPU worker budget for CPU-bound stages (Chem-ACE patching/tagging/feature embedding + torch CPU thread pools).

## Skip optimization and use an existing params JSON

```bash
python /Users/vfastovskii/Desktop/mil_explainability_2026/opt_attn_net_feb/opt_net_fast.py ... --best_params_json /path/to/multimodal_mil_aux_gpu_best_params.json
```

## Skip optimization and use default params JSON in `study_dir`

```bash
python /Users/vfastovskii/Desktop/mil_explainability_2026/opt_attn_net_feb/opt_net_fast.py ...
```

Expected file in this mode:

- `<study_dir>/multimodal_mil_aux_gpu_best_params.json`

## Pipeline step logs (crash localization)

The pipeline now prints structured progress lines for every major stage:

- `[YYYY-mm-dd HH:MM:SS] [START] <step> ...`
- `[YYYY-mm-dd HH:MM:SS] [DONE] <step> elapsed_s=...`
- `[YYYY-mm-dd HH:MM:SS] [FAIL] <step> elapsed_s=... error=...`

If a crash happens, check the last printed `START/INFO` step and the corresponding `FAIL` line to pinpoint where it failed.

## Run Chem-ACE demo (concept discovery + TCAV + SQLite)

```bash
python -m opt_attn_net_feb.entrypoints.chem_ace_demo --output_dir /tmp/chem_ace_demo --with_3d
```

Outputs:

- `/tmp/chem_ace_demo/chem_ace.sqlite3`
- `/tmp/chem_ace_demo/artifacts/*`
- `/tmp/chem_ace_demo/chem_ace_demo_summary.json`

## Enable Chem-ACE in the real MIL pipeline (optimized final training)

```bash
python -m entrypoints.hpo_pipeline ... \
  --run_hpo \
  --run_chem_ace \
  --curated_smiles_col curated_SMILES \
  --chem_ace_conformer_sdf /path/to/precomputed_conformers.sdf \
  --chem_ace_sdf_conf_id_prop conf_id
```

Useful controls:

- `--chem_ace_max_ids 0` (default: auto use all IDs in scope)
- `--chem_ace_max_confs_per_id 0` (default: use all conformers)
- `--chem_ace_local_radii 1` (default)
- `--chem_ace_patch_cap_per_mol 0` (default: dynamic auto-cap)
- `--chem_ace_target_total_patches 1200000` (used when auto-cap is enabled)
- `--chem_ace_top_concepts 64`
- `--chem_ace_output_dir /path/to/chem_ace_out`
- `--chem_ace_conformer_sdf /path/to/precomputed_conformers.sdf`
- `--chem_ace_sdf_conf_id_prop conf_id` (falls back to SDF record name if missing)
- `--run_activity_calibration` / `--no-run_activity_calibration` (default: enabled)
- `--activity_calibration_keep_threshold 0.55`
- `--activity_calibration_min_concept_support 12`
- `--activity_calibration_min_tag_support 24`
- `--activity_calibration_prior_strength 32.0`
- `--activity_calibration_min_w 0.40`
- `--activity_calibration_task_weight 0.70`
- `--activity_calibration_bitmask_weight 0.30`
- `--activity_calibration_bitmask_min_count 20`
- `--activity_calibration_ratio_cap 8.0`
- `--cpu_workers -1`

Anti-leakage behavior:

- Chem-ACE concept discovery in final pipeline is train-only.
- Leaderboard explainability is inference-only: leaderboard patches are assigned to frozen train centroids (no reclustering).
- Use `--chem_ace_infer_max_distance` to drop far leaderboard assignments (`<=0` disables gating).
- Activity calibration is train-only and leakage-safe: it uses only train IDs and train labels/bitmasks.

Behavior for conformers:

- 2D patches are generated once per molecule.
- 3D Pharm3D patches are generated only for conformers found in the SDF.
- If a `conf_id` from the features is missing in SDF, it is skipped (no conformer generation fallback).

Functional-group rules behavior:

- Chem-ACE auto-augments functional-group rules from all unique `curated_SMILES` in the provided labels table using RDKit `Chem.Fragments.fr_*`.
- Merged runtime rules are written to:
  - `<chem_ace_output_dir>/rules_autogen/default_functional_group_rules.dataset.json`
  - `<chem_ace_output_dir>/rules_autogen/functional_group_fragment_stats.json`

Manual rule generation:

```bash
python -m explainability.chem_ace.rules.generate_fragment_rules_from_labels \
  --labels_csv /path/to/master_table_labels_final_modelling_ready_1401_with_cv_split.csv \
  --smiles_col curated_SMILES \
  --output_json /path/to/default_functional_group_rules.json \
  --stats_json /path/to/functional_group_fragment_stats.json
```

Patch-budget behavior:

- Local-subgraph default is `radii=(1,)`.
- If `--chem_ace_patch_cap_per_mol <= 0`, cap is derived dynamically from:
  `ceil(chem_ace_target_total_patches / n_molecules_in_scope)`, clipped to `[16, 256]`.
- If `--chem_ace_patch_cap_per_mol > 0`, that fixed cap is used.
- Capping uses deterministic per-molecule sampling (stable across reruns with same inputs).

## Enable Lambda-Vol monitoring during final training (requires Chem-ACE concepts)

```bash
python -m entrypoints.hpo_pipeline ... \
  --run_hpo \
  --run_lambda_vol
```

`--run_lambda_vol` automatically enables Chem-ACE concept preparation.

Detailed Ricci-flow documentation:

- `/Users/vfastovskii/Desktop/mil_explainability_2026/opt_attn_net_feb/RICCI_FLOW_README.md`

Useful controls:

- `--lambda_vol_layer_name mixer_post_norm`
- `--lambda_vol_top_concepts 24`
- `--lambda_vol_monitor_max_samples 512`
- `--lambda_vol_tcav_repeats 2`
- `--lambda_vol_output_dir /path/to/lambda_vol_out`
- `--lambda_vol_run_ricci` / `--no-lambda_vol_run_ricci`
- `--lambda_vol_ricci_edge_keep_quantile 0.75`
- `--lambda_vol_ricci_flow_steps 8`
- `--lambda_vol_ricci_flow_step_size 0.12`
- `--lambda_vol_ricci_use_flow_as_coupling` / `--no-lambda_vol_ricci_use_flow_as_coupling`

## Enable concept-guided RL control during final training

```bash
python -m entrypoints.hpo_pipeline ... \
  --run_hpo \
  --run_concept_rl
```

`--run_concept_rl` automatically enables Chem-ACE concept preparation.

Useful controls:

- `--concept_rl_top_k_per_task 8`
- `--concept_rl_min_pos_coverage 0.02`
- `--concept_rl_init_scale 0.02`
- `--concept_rl_max_scale 0.20`
- `--concept_rl_policy_lr 0.05`
- `--concept_rl_policy_sigma 0.02`
- `--concept_rl_reward_alignment_w 0.25`
- `--concept_rl_baseline_momentum 0.90`

Final outputs include:

- `<study_dir>/final_best_train_vs_leaderboard/explainability_artifacts.json`
- Chem-ACE DB/artifacts in `chem_ace_output_dir` (or `<study_dir>/chem_ace`)
- Chem-ACE molecule-level semantic baseline exports:
  - `a_priori_tags.csv`
  - `a_priori_vs_concepts.csv`
  - `a_priori_tags_infer_scope.csv`
  - `a_priori_vs_concepts_infer_scope.csv`
- Chem-ACE activity-calibrated semantic exports (when enabled):
  - `concept_tags_calibrated.csv`
  - `concept_tags_calibration_summary.json`
- Lambda-Vol tensor/log/html artifacts in `lambda_vol_output_dir` (or `<study_dir>/lambda_vol`)
- Concept-RL policy history JSON (`concept_rl_policy_history.json`) inside final run directory when enabled
- Ricci artifacts (`ricci_edges_long.csv`, `ricci_task_summary.csv`, `ricci_flow_tensors.npz`) inside Lambda-Vol output
- Prediction explanations CSV (`*_explained.csv`) with per-task text explanations when Chem-ACE is enabled
