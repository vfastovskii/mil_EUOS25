# Lambda-Vol Concept-Pressure Dynamics

This module adds a training-time monitoring/control layer for concept reliance dynamics in multitask molecular models.

For full Ricci geometry details (formulas, thresholds, artifact schema, interpretation), see:

- `/Users/vfastovskii/Desktop/mil_explainability_2026/opt_attn_net_feb/RICCI_FLOW_README.md`

## What it implements

- Per-epoch concept pressure tracking:
  - `TCAV[x,y,t]`, `delta_TCAV`, `attention_support`, `prevalence`, `rho[x,y,t]`
  - task metrics: attention entropy, witness rate, train/val metrics, loss, calibration error
  - attention semantics are strict:
    - `attention_support` is computed only from 3D conformer attention (`3d_geom`, `3d_qm`)
    - `2d` concepts contribute via `TCAV` + `prevalence`, not synthetic attention mass
- TCAV protocol hardening:
  - CAV fit on monitor-train split
  - directional-derivative evaluation on holdout split when feasible
  - repeat-level raw and Bonferroni-corrected significance logging
- Regime inference `q(t)`:
  - rule-based regimes: `warmup`, `fitting`, `stable_generalization`, `overfit_onset`, `refit`
  - classifier interface stub for future learned regime labeling
- Lambda-Vol-inspired dynamics decomposition:
  - `regime_core`, `feedback`, `trend_loop`, `revert_loop`, `dissipation`, `context_term`
- Alerting:
  - runaway pressure
  - concentration/collapse (`top-k mass`, entropy drop)
  - blocked concept positive drift
- Ricci geometry diagnostics + flow (new):
  - per-task concept graph construction from sparse co-activation in top concepts
  - node score per concept: `w_tcav * tcav_ema + w_attention * attn_ema` (2D uses `w_attn=0`)
  - same-modality edges from Jaccard/co-occurrence
  - cross-modality edges from weighted co-activation in the same molecules
  - Forman-Ricci edge curvature per epoch
  - Ricci-flow-style edge reweighting
  - bridge/bottleneck alerts from negative-curvature structure
  - optional coupling of flowed concept graph into Λ-Vol dynamics
- Intervention recommendations:
  - concept-balanced batching
  - hard-negative mining
  - attention-entropy regularization increase
  - blocked-concept penalty activation
  - oversample counterexamples
- Persistence:
  - SQLite + SQLAlchemy tables (`lv_*`) with query API
- Artifacts:
  - pressure tensors (`.npz`)
  - long-form logs (`.csv`, optional parquet)
  - Plotly 3D manifold/lattice/coupling HTML (if Plotly available)
  - alert/recommendation reports
  - optional VTK point-cloud export (if PyVista available)

## Package layout

- `/Users/vfastovskii/Desktop/mil_explainability_2026/opt_attn_net_feb/explainability/lambda_vol/config.py`
- `/Users/vfastovskii/Desktop/mil_explainability_2026/opt_attn_net_feb/explainability/lambda_vol/types.py`
- `/Users/vfastovskii/Desktop/mil_explainability_2026/opt_attn_net_feb/explainability/lambda_vol/providers.py`
- `/Users/vfastovskii/Desktop/mil_explainability_2026/opt_attn_net_feb/explainability/lambda_vol/tracker.py`
- `/Users/vfastovskii/Desktop/mil_explainability_2026/opt_attn_net_feb/explainability/lambda_vol/regime.py`
- `/Users/vfastovskii/Desktop/mil_explainability_2026/opt_attn_net_feb/explainability/lambda_vol/dynamics.py`
- `/Users/vfastovskii/Desktop/mil_explainability_2026/opt_attn_net_feb/explainability/lambda_vol/detectors.py`
- `/Users/vfastovskii/Desktop/mil_explainability_2026/opt_attn_net_feb/explainability/lambda_vol/policy.py`
- `/Users/vfastovskii/Desktop/mil_explainability_2026/opt_attn_net_feb/explainability/lambda_vol/exporters.py`
- `/Users/vfastovskii/Desktop/mil_explainability_2026/opt_attn_net_feb/explainability/lambda_vol/monitor.py`
- `/Users/vfastovskii/Desktop/mil_explainability_2026/opt_attn_net_feb/explainability/lambda_vol/integrations/pytorch.py`
- `/Users/vfastovskii/Desktop/mil_explainability_2026/opt_attn_net_feb/explainability/lambda_vol/integrations/lightning.py`
- `/Users/vfastovskii/Desktop/mil_explainability_2026/opt_attn_net_feb/explainability/lambda_vol/db/*`
- `/Users/vfastovskii/Desktop/mil_explainability_2026/opt_attn_net_feb/entrypoints/lambda_vol_demo.py`

## Quickstart demo

From the project root (`/Users/vfastovskii/Desktop/mil_explainability_2026/opt_attn_net_feb`):

```bash
python -m entrypoints.lambda_vol_demo \
  --output_dir /tmp/lambda_vol_demo \
  --epochs 16
```

This produces:

- `/tmp/lambda_vol_demo/<run_id>/concept_pressure_tensors.npz`
- `/tmp/lambda_vol_demo/<run_id>/concept_pressure_long.csv`
- `/tmp/lambda_vol_demo/<run_id>/task_metrics_long.csv`
- `/tmp/lambda_vol_demo/<run_id>/attention_focus_top.csv`
- `/tmp/lambda_vol_demo/<run_id>/metadata.json`
- `/tmp/lambda_vol_demo/<run_id>/alerts.json`
- `/tmp/lambda_vol_demo/<run_id>/recommendations.json`
- `/tmp/lambda_vol_demo/<run_id>/pressure_lattice.html` (if Plotly installed)
- `/tmp/lambda_vol_demo/<run_id>/concept_manifold_<task>.html` (if Plotly installed)
- `/tmp/lambda_vol_demo/lambda_vol_demo_summary.json`

Ricci-specific files are produced only when `--lambda_vol_run_ricci` is enabled:

- `/tmp/lambda_vol_demo/<run_id>/ricci_edges_long.csv`
- `/tmp/lambda_vol_demo/<run_id>/ricci_task_summary.csv`
- `/tmp/lambda_vol_demo/<run_id>/ricci_flow_tensors.npz`

In MIL final runs, Lambda-Vol additionally writes per-epoch TCAV significance tables:

- `<lambda_vol_output_dir>/tcav_significance/tcav_significance_epoch_XXXX.csv`

## Integrating into training

### Plain PyTorch loop

Use `/Users/vfastovskii/Desktop/mil_explainability_2026/opt_attn_net_feb/explainability/lambda_vol/integrations/pytorch.py`.

```python
adapter = LambdaVolPyTorchAdapter(monitor=monitor)
adapter.on_epoch_end(
    epoch=epoch,
    tcav_df=tcav_df,
    concept_attention_df=concept_attention_df,
    task_attention_df=task_attention_df,
    task_metrics_df=task_metrics_df,
    context_covariates=context,
)
```

### PyTorch Lightning

Use `/Users/vfastovskii/Desktop/mil_explainability_2026/opt_attn_net_feb/explainability/lambda_vol/integrations/lightning.py`.

- Implement `LightningFrameProvider.collect_epoch_frames(...)` to return DataFrames.
- Attach `LambdaVolLightningCallback` to `Trainer(callbacks=[...])`.

## TCAV controls (pipeline)

When running `hpo_pipeline.py` / `opt_net_fast.py`, TCAV monitor controls include:

- `--lambda_vol_layer_name` (default `mixer_post_norm`)
- `--lambda_vol_top_concepts` (default `0` -> all concepts)
- `--lambda_vol_monitor_max_samples` (default `512`)
- `--lambda_vol_tcav_repeats` (default `2`)
- `--lambda_vol_random_counterexamples` (default `96`)
- `--lambda_vol_min_concept_samples` (default `8`)
- `--lambda_vol_tcav_holdout_fraction` (default `0.2`, `<=0` disables holdout)
- `--lambda_vol_tcav_holdout_min_samples` (default `16`)
- `--lambda_vol_tcav_significance_alpha` (default `0.05`)
- `--lambda_vol_tcav_bonferroni_m` (default `0`, auto-uses current concept count)

Significance export schema:

- one row per `(epoch, task_id, concept_id, repeat_idx)`
- includes:
  - `sign_rate`, `mean_directional_derivative`
  - `p_value_raw`, `p_value_bonferroni`
  - `significant_raw`, `significant_bonferroni`
  - `n_eval_samples`, `holdout_used`

Attention concentration export:

- `attention_focus_top.csv` includes per `(epoch, task)` top attention concepts with:
  - `attention_support`
  - `attention_share` (normalized within task/epoch)
  - `attention_rank`
  - `modality`, `label_auto`

## Ricci controls (pipeline)

When running `hpo_pipeline.py` / `opt_net_fast.py`, Ricci monitoring is available via:

- `--lambda_vol_run_ricci` / `--no-lambda_vol_run_ricci`
- `--lambda_vol_ricci_edge_keep_quantile`
- `--lambda_vol_ricci_min_edge_weight`
- `--lambda_vol_ricci_top_k_per_node`
- `--lambda_vol_ricci_flow_steps`
- `--lambda_vol_ricci_flow_step_size`
- `--lambda_vol_ricci_use_flow_as_coupling` / `--no-lambda_vol_ricci_use_flow_as_coupling`
- `--lambda_vol_ricci_coupling_strength`

Ricci defaults and exports:
- `--lambda_vol_run_ricci` defaults to `False` (recommended baseline for speed/clarity)
- when disabled, Ricci artifacts are not produced and no Ricci coupling is injected
- enable Ricci only for dedicated geometry diagnostics runs

When enabled, exported as:

- `ricci_edges_long.csv`
- `ricci_task_summary.csv`
- `ricci_flow_tensors.npz`

### Ricci implementation summary

- Concept node score is built from:
  - `tcav_ema` for all concepts
  - `attn_ema` for 3D concepts (`3d_geom`, `3d_qm`)
- For each task/epoch, a top-k concept subset is selected before edge construction.
- Concept graph edges are built from sample-level concept co-activation:
  - same-modality: Jaccard/co-occurrence
  - cross-modality: weighted overlap in the same molecules
- Edges are sparsified by quantile/min-threshold + top-k per node.
- Curvature uses Forman-Ricci style discrete edge curvature.
- Flow iteratively updates edge lengths/weights (`flow_steps`, `flow_step_size`).
- Ricci can run on interval (`update_interval_epochs`) instead of every epoch.
- Optional: mean flowed similarity is injected as concept coupling in dynamics.
- Detector emits Ricci alerts:
  - `ricci_negative_curvature_surge`
  - `ricci_bridge_concentration`
  - `ricci_extreme_negative_bridge`

## Query API examples

`LambdaVolQueryService` wraps repository queries:

- planar+conjugated concepts with rising pressure (joins `concept_tags` if present)
- high-TCAV low-prevalence concepts
- top trend loops vs dissipation
- intervention recommendations in recent epochs

## Assumptions and stubs

- Generic model integration is adapter-based; no model class is hardcoded.
- TCAV/attention providers are expected to produce per-epoch DataFrames.
- If optional dependencies are missing:
  - no Plotly => HTML plots skipped
  - no pyarrow => parquet skipped
  - no PyVista => VTK export skipped
- If a Chem-ACE concept DB is reused, `concept_tags` can enrich Lambda-Vol queries automatically.

## TODO (advanced)

- Learned regime classifier implementation
- CAV-direction penalties wired to actual optimizer updates
- richer MIL attention witness diagnostics
- UMAP trajectory embeddings over epochs
- direct coupling estimation from gradients/causal probes
