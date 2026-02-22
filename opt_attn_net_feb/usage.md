# Usage

## Run optimization + final training

```bash
python /Users/vfastovskii/Desktop/mil_explainability_2026/opt_attn_net_feb/opt_net_fast.py ... --run_hpo --trials 50
```

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
  --curated_smiles_col curated_SMILES
```

Useful controls:

- `--chem_ace_max_ids 10000`
- `--chem_ace_max_confs_per_id 4`
- `--chem_ace_top_concepts 64`
- `--chem_ace_output_dir /path/to/chem_ace_out`

## Enable Lambda-Vol monitoring during final training (requires Chem-ACE concepts)

```bash
python -m entrypoints.hpo_pipeline ... \
  --run_hpo \
  --run_lambda_vol
```

`--run_lambda_vol` automatically enables Chem-ACE concept preparation.

Useful controls:

- `--lambda_vol_layer_name mixer_post_norm`
- `--lambda_vol_top_concepts 24`
- `--lambda_vol_monitor_max_samples 512`
- `--lambda_vol_tcav_repeats 2`
- `--lambda_vol_output_dir /path/to/lambda_vol_out`

Final outputs include:

- `<study_dir>/final_best_train_vs_leaderboard/explainability_artifacts.json`
- Chem-ACE DB/artifacts in `chem_ace_output_dir` (or `<study_dir>/chem_ace`)
- Lambda-Vol tensor/log/html artifacts in `lambda_vol_output_dir` (or `<study_dir>/lambda_vol`)
