# Chem-ACE (Automatic Concept Discovery + Semantic Tagging)

This document describes the Chem-ACE subsystem added to this project for CAV/TCAV-ready molecular explainability.

## Scope

Implemented modules cover:

- Patch generation (`local_subgraph`, `BRICS`, `Murcko`, optional `Pharm3D`).
- Patch embedding with model-layer hooks (`masked_input`, `node_pooling`) and deterministic cache.
- Concept discovery (k-means, hierarchical, optional HDBSCAN), coherence/support filtering, centroid dedup, medoid selection.
- Semantic tagging (charge, conjugation/aromaticity, geometry, pharmacophore) with confidence/evidence.
- Human-readable naming (rule-based registry + descriptor fallback).
- CAV/TCAV (linear separator, repeated random counterexample sets, sign-rate + mean directional derivative).
- SQLite persistence with SQLAlchemy (runs, patches, embeddings, concept sets, concepts, memberships, tags, CAVs, TCAV epoch, MIL concept epoch).
- Query layer for trend/collapse analytics.
- Demo CLI to run end-to-end.

## Package Layout

- `opt_attn_net_feb/explainability/chem_ace/config.py`
- `opt_attn_net_feb/explainability/chem_ace/types.py`
- `opt_attn_net_feb/explainability/chem_ace/optional_deps.py`
- `opt_attn_net_feb/explainability/chem_ace/patches/*`
- `opt_attn_net_feb/explainability/chem_ace/embedding/*`
- `opt_attn_net_feb/explainability/chem_ace/concepts/clustering.py`
- `opt_attn_net_feb/explainability/chem_ace/concepts/pipeline.py`
- `opt_attn_net_feb/explainability/chem_ace/cav/tcav.py`
- `opt_attn_net_feb/explainability/chem_ace/semantics/*`
- `opt_attn_net_feb/explainability/chem_ace/db/*`
- `opt_attn_net_feb/explainability/chem_ace/analytics/queries.py`
- `opt_attn_net_feb/entrypoints/chem_ace_demo.py`

## Core Interfaces

### Patch generators

Each generator follows `PatchGenerator.generate(mol_id, mol, conf_id)` and returns `PatchRecord` with:

- `patch_id`, `mol_id`, `conf_id`, `patch_type`
- `atom_indices`, `patch_hash`
- optional `smarts`, `fragment_repr`, `feature_metadata`

### Embedding adapters

- `PatchInputBuilder` protocol:
  - `build_masked_input(patch)`
  - `build_full_input(patch) -> (model_input, atom_indices)`
- Layer capture uses `LayerActivationHook(model, layer_name)`.
- Cache maps `(patch_id, layer_name, strategy)` to deterministic embedding URI.

### Model/TCAV adapters

- `ModelTaskAdapter.forward(model_input)`
- `ModelTaskAdapter.get_task_scalar(model_output, task_id)`

Used for directional derivatives and task-explicit TCAV.

## Concept Discovery Details

`discover_concepts(...)` does:

1. cluster embeddings (`kmeans`, `hierarchical`, optional `hdbscan`)
2. compute support + coherence
3. filter by `min_support` and `min_coherence`
4. deduplicate by centroid cosine similarity threshold
5. choose medoid patch per concept

Output is immutable snapshot payload `DiscoveredConceptSet` and persisted `concept_set` DB rows with versioning (`run_id`, `version`).

## Semantic Tagging Details

Implemented tag groups:

- Charge: anionic, cationic, zwitterionic-like, neutral polar/nonpolar
- Conjugation/aromaticity: aromatic pi-system, extended conjugation, isolated unsaturation, heteroaromatic, aliphatic
- Geometry (3D when available): planar, non-planar, twisted, rigid, flexible
- Pharmacophore: HBD, HBA, aromatic centroid, cationic center, anionic center, HBD/HBA pair

Each tag stores:

- `confidence`
- `evidence_json` (descriptor summary)
- `provenance` (tagger rule source)

Naming is two-stage:

1. rule-based labels (`rules/default_naming_rules.json`)
2. fallback descriptor phrase

`label_auto` is persisted in `concepts.label_auto`.

## CAV / TCAV Details

`run_tcav_from_arrays(...)` supports repeated random counterexample sets:

- train linear separator (`logreg` or `svm`) -> CAV
- compute TCAV sign-rate and mean directional derivative
- store per-repeat `CAVRecord` and `TCAVRecord`
- produce summary mean/std

For model-based gradients use `collect_gradients_for_layer(...)`.

## Database Schema

Main tables:

- `runs`, `tasks`, `molecules`, `conformers`
- `patches`, `patch_embeddings`
- `concept_sets` (immutable/versioned)
- `concepts`, `concept_memberships`, `concept_tags`
- `cavs`, `tcav_epoch`
- `mil_concept_epoch` (optional MIL attention/witness/prevalence metrics)

## Query API Examples

`ConceptQueryService` supports:

- `planar_conjugated_rising_tcav(task_id, last_n_epochs)`
- `top_attention_support(task_id, epoch, limit)`
- `high_tcav_low_prevalence(task_id, epoch, tcav_thr, prevalence_thr)`
- `concept_collapse_indicator(task_id, epoch, top_k)`

## Demo Run

From parent directory of package (so `opt_attn_net_feb` is importable):

```bash
python -m opt_attn_net_feb.entrypoints.chem_ace_demo --output_dir /tmp/chem_ace_demo --with_3d
```

Outputs:

- SQLite DB in `output_dir/chem_ace.sqlite3`
- cached embeddings in `output_dir/chem_ace_cache/...`
- vector artifacts in `output_dir/artifacts/...`
- summary report `output_dir/chem_ace_demo_summary.json`

## Assumptions and Stubs

- RDKit/HDBSCAN are optional; missing deps degrade gracefully where possible.
- Model-specific data plumbing is adapter-driven (`PatchInputBuilder`, `ModelTaskAdapter`).
- Current demo uses toy node-feature model for end-to-end validation.
- MIL-specific metrics are schema + repository ready; integration with live MIL loop is a follow-up hook.

## TODO (Advanced Extensions)

- 3D pharmacophore patch expansion with richer geometric grouping.
- MIL attention-aware concept scoring directly from attention maps.
- Diffusion-guidance integration for concept-conditioned generation.
- Active learning / bandit controller hooks for concept balancing.

