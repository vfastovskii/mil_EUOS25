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
