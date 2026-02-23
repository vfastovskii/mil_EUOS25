# Lambda-Vol Concept-Pressure Dynamics

This module adds a training-time monitoring/control layer for concept reliance dynamics in multitask molecular models.

For full Ricci geometry details (formulas, thresholds, artifact schema, interpretation), see:

- `/Users/vfastovskii/Desktop/mil_explainability_2026/opt_attn_net_feb/RICCI_FLOW_README.md`

## What it implements

- Per-epoch concept pressure tracking:
  - `TCAV[x,y,t]`, `delta_TCAV`, `attention_support`, `prevalence`, `rho[x,y,t]`
  - task metrics: attention entropy, witness rate, train/val metrics, loss, calibration error
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
  - per-task concept graph construction from `rho/attention/prevalence + TCAV-history corr`
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
- `/tmp/lambda_vol_demo/<run_id>/metadata.json`
- `/tmp/lambda_vol_demo/<run_id>/alerts.json`
- `/tmp/lambda_vol_demo/<run_id>/recommendations.json`
- `/tmp/lambda_vol_demo/<run_id>/ricci_edges_long.csv`
- `/tmp/lambda_vol_demo/<run_id>/ricci_task_summary.csv`
- `/tmp/lambda_vol_demo/<run_id>/ricci_flow_tensors.npz`
- `/tmp/lambda_vol_demo/<run_id>/pressure_lattice.html` (if Plotly installed)
- `/tmp/lambda_vol_demo/<run_id>/concept_manifold_<task>.html` (if Plotly installed)
- `/tmp/lambda_vol_demo/lambda_vol_demo_summary.json`

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

By default, Ricci is enabled for Lambda-Vol runs and exported as:

- `ricci_edges_long.csv`
- `ricci_task_summary.csv`
- `ricci_flow_tensors.npz`

### Ricci implementation summary

- Concept graph edges are built from weighted combination of:
  - pressure `rho`
  - attention support
  - prevalence
  - absolute TCAV-history correlation
- Edges are sparsified by quantile/min-threshold + top-k per node.
- Curvature uses Forman-Ricci style discrete edge curvature.
- Flow iteratively updates edge lengths/weights (`flow_steps`, `flow_step_size`).
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
