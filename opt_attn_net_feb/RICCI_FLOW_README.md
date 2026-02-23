# Ricci Flow in Lambda-Vol (Detailed Reference)

This document explains the discrete Ricci-curvature and Ricci-flow layer integrated into Lambda-Vol monitoring.

Implementation files:

- `explainability/lambda_vol/ricci.py`
- `explainability/lambda_vol/config.py`
- `explainability/lambda_vol/monitor.py`
- `explainability/lambda_vol/detectors.py`
- `explainability/lambda_vol/exporters.py`

## 1) Purpose

Ricci functionality is used as a geometry signal over the concept graph:

- identify bottleneck-like concept bridges (negative curvature edges)
- quantify geometric instability/collapse risk per task
- optionally feed flowed concept similarity back into Lambda-Vol dynamics as concept coupling

It is a monitoring/control signal, not a standalone oracle.

## 2) Inputs per epoch

For each epoch and each task, Ricci analyzer consumes:

- `rho[t, c]` (concept pressure)
- `attention_support[t, c]`
- `prevalence[t, c]`
- optional `tcav_history_by_task[task_id]` (matrix of historical TCAV over epochs)

Shapes are validated against `(n_tasks, n_concepts)`.

## 3) Concept graph construction

For each task, a concept-concept similarity matrix is built:

1. Nonnegative channels:
   - `x_rho = max(rho, 0)`
   - `x_attn = max(attention_support, 0)`
   - `x_prev = max(prevalence, 0)`
2. Pairwise similarity components:
   - `sim_rho(i,j) = sqrt(x_rho[i] * x_rho[j])`
   - `sim_attn(i,j) = sqrt(x_attn[i] * x_attn[j])`
   - `sim_prev(i,j) = sqrt(x_prev[i] * x_prev[j])`
   - `sim_tcav_corr(i,j) = abs(corr(TCAV_i_history, TCAV_j_history))` (if history available)
3. Weighted sum:
   - `sim = w_rho*sim_rho + w_attention*sim_attn + w_prevalence*sim_prev + w_tcav_corr*sim_tcav_corr`
4. Normalize by global max (if positive).
5. Sparsify:
   - keep edges above `max(min_edge_weight, quantile(edge_keep_quantile))`
   - additionally keep top-`k` per node (`top_k_per_node`)
   - symmetrize keep-mask and zero diagonal

Default weights/config:

- `w_rho=0.40`
- `w_attention=0.30`
- `w_prevalence=0.15`
- `w_tcav_corr=0.15`
- `edge_keep_quantile=0.75`
- `min_edge_weight=0.05`
- `top_k_per_node=4`

## 4) Curvature definition (Forman-Ricci style)

For each kept edge `(i,j)` with weight `w_ij > 0`:

- `kappa_ij = 2 - sum_{k!=i,j, w_ik>0} 1/sqrt(w_ij*w_ik) - sum_{k!=i,j, w_jk>0} 1/sqrt(w_ij*w_jk)`

Interpretation in this pipeline:

- more negative curvature -> more bridge/bottleneck-like edge
- positive/near-zero curvature -> denser/redundant local structure

## 5) Ricci-flow-style reweighting

If flow is enabled, similarity weights are updated for `flow_steps` iterations:

1. Convert edge weight to length:
   - `L_ij = 1 / (w_ij + eps)`
2. Update by curvature:
   - `factor = clip(1 - flow_step_size * kappa_ij, 0.05, 20.0)`
   - `L'_ij = L_ij * factor`
3. Convert back:
   - `w'_ij = 1 / (L'_ij + eps)`
4. Renormalize matrix by max and zero diagonal.

Defaults:

- `flow_enabled=True`
- `flow_steps=8`
- `flow_step_size=0.12`
- `flow_eps=1e-4`

## 6) Coupling back into Lambda-Vol dynamics

If enabled, mean flowed similarity across tasks is converted to concept coupling matrix:

- `S = coupling_strength * mean_flowed_similarity`
- diagonal forced to zero
- assigned to dynamics model as cross-concept coupling

Controls:

- `use_flow_as_concept_coupling=True`
- `coupling_strength=0.05`

This affects the `feedback` term in Lambda-Vol dynamics decomposition.

## 7) Ricci-derived summaries and alerts

Per task and epoch, summary stats include:

- `n_edges`
- `mean_curvature`, `std_curvature`, `min_curvature`, `max_curvature`
- `negative_edge_fraction` (`kappa < negative_curvature_threshold`)
- `strong_negative_edge_fraction` (`kappa < strong_negative_curvature_threshold`)
- top-most negative edge endpoints

Default curvature thresholds:

- `negative_curvature_threshold=-0.15`
- `strong_negative_curvature_threshold=-0.35`

Detector alert thresholds:

- `ricci_negative_edge_fraction_alert=0.40`
- `ricci_strong_negative_fraction_alert=0.20`
- `ricci_min_curvature_alert=-0.45`

Alert codes:

- `ricci_negative_curvature_surge`
- `ricci_bridge_concentration`
- `ricci_extreme_negative_bridge`

## 8) Exported Ricci artifacts

When Ricci is active, exporter writes:

- `ricci_edges_long.csv`
- `ricci_task_summary.csv`
- `ricci_flow_tensors.npz`

`ricci_edges_long.csv` columns:

- `epoch`, `task_id`
- `concept_src`, `concept_dst`
- `weight_raw` (pre-flow edge weight)
- `curvature`
- `weight_flow` (post-flow edge weight)

`ricci_task_summary.csv` columns:

- `epoch`, `task_id`
- `n_nodes`, `n_edges`
- `mean_curvature`, `std_curvature`, `min_curvature`, `max_curvature`
- `negative_edge_fraction`, `strong_negative_edge_fraction`
- `top_negative_src`, `top_negative_dst`, `top_negative_curvature`

`ricci_flow_tensors.npz` arrays:

- `ricci_weight_flow` shape `[T, C, C, E]`
- `ricci_curvature` shape `[T, C, C, E]`
- `ricci_weight_raw` shape `[T, C, C, E]`
- plus `task_ids`, `concept_ids`, `epochs`

## 9) CLI controls

From `entrypoints/hpo_pipeline.py`:

- `--lambda_vol_run_ricci` / `--no-lambda_vol_run_ricci`
- `--lambda_vol_ricci_edge_keep_quantile`
- `--lambda_vol_ricci_min_edge_weight`
- `--lambda_vol_ricci_top_k_per_node`
- `--lambda_vol_ricci_flow_steps`
- `--lambda_vol_ricci_flow_step_size`
- `--lambda_vol_ricci_use_flow_as_coupling` / `--no-lambda_vol_ricci_use_flow_as_coupling`
- `--lambda_vol_ricci_coupling_strength`

## 10) Practical interpretation guidance

Use Ricci together with TCAV/attention/prevalence/performance.

Recommended reading pattern per task:

1. Check `negative_edge_fraction` and `min_curvature` trends.
2. Inspect `top_negative_src -> top_negative_dst` edges.
3. Correlate with pressure concentration (`topk_mass`, entropy drop) and alerts.
4. If both geometry and pressure drift are adverse, apply balancing recommendations.

Do not interpret curvature alone as definitive causality.
