# Chem-ACE (Automatic Concept Discovery + Semantic Tagging)

This document is the code-aligned technical reference for the current Chem-ACE implementation in this repository.

It explains:

- how patches are created from 2D/3D/3D-QM context
- how those patches are embedded and clustered into concepts
- how semantic tags are assigned and calibrated using activity/bitmask supervision
- how anti-leakage train/infer split behavior is enforced
- how outputs are consumed downstream (Lambda-Vol, Concept-RL, prediction text explanations)
- what each major runtime log stage means

Primary implementation entrypoints:

- `training/explainability_runtime.py`
- `explainability/chem_ace/concepts/pipeline.py`
- `explainability/chem_ace/semantics/taggers.py`
- `explainability/chem_ace/semantics/calibration.py`

## 1) Runtime Scope and Trigger Conditions

Chem-ACE runs in final pipeline when `--run_chem_ace` is enabled.

Chem-ACE is automatically enabled if either of these is requested:

- `--run_lambda_vol`
- `--run_concept_rl`

This is normalized in `entrypoints/hpo_pipeline.py`.

## 2) Anti-Leakage Contract (Current Behavior)

Chem-ACE now follows strict two-phase behavior:

1. `discover_train` phase:
- generate patches only from train IDs
- embed only train patches
- discover concepts only on train embeddings
- tag concepts only from train memberships

2. `infer_scope` phase:
- generate/embed patches for infer IDs (leaderboard scope)
- assign infer patches to frozen train centroids
- optional distance gate via `--chem_ace_infer_max_distance`
- no concept re-clustering and no concept-definition updates

This is enforced in `prepare_chem_ace_bundle(...)`.

## 3) Data Modalities and Feature Basis

Chem-ACE uses all three feature modalities already prepared for MIL training:

- 2D molecular vector (`x2d` per molecule ID)
- 3D geometry + QM vector (`xinst` per `(mol_id, conf_id)`)
- curated molecular graph from `curated_SMILES`

For semantic interpretation, task labels are from:

- `Transmittance_340`
- `Transmittance_450`
- `Fluorescence_340_450`
- `Fluorescence_more_than_480`

Bitmask IDs are built from the 4 binary tasks in this order (LSB-first):

- bit 0: `Transmittance_340`
- bit 1: `Transmittance_450`
- bit 2: `Fluorescence_340_450`
- bit 3: `Fluorescence_more_than_480`

## 4) Patch Object and Identity

Patch model is `PatchRecord` in `explainability/chem_ace/types.py`.

Key fields:

- `patch_id`
- `mol_id`
- `conf_id` (`None` for 2D-only patches)
- `patch_type`
- `atom_indices`
- `patch_hash`
- optional `smarts`, `fragment_repr`, `feature_metadata`

Identity rules:

1. `patch_hash` depends on patch content (not molecule ID).
2. `patch_id` is derived from `(mol_id, conf_id, patch_hash)`.

Consequence:

- same motif across conformers can share `patch_hash`
- but still gets distinct `patch_id` when `conf_id` differs

## 5) Patch Generators and 2D/3D Split

Patch generators:

- `local_subgraph`
- `brics`
- `murcko`
- `murcko_framework`
- `pharm3d`

Generator split in `ChemACEPipeline`:

- 2D generators: run once per molecule with `conf_id=None`
- 3D generators (`requires_conformer=True`): run per conformer

Current default for local subgraph radii:

- `(1,)`

This was tightened to reduce patch explosion.

## 6) Conformer Ingestion (SDF-Backed, No On-The-Fly Generation)

Current final runtime does not compute conformers.

Conformers are loaded from provided SDF (`--chem_ace_conformer_sdf`) and matched by:

1. configured property (`--chem_ace_sdf_conf_id_prop`, default `conf_id`)
2. fallback `_Name`

If a requested conformer is missing in SDF:

- it is skipped for 3D patching
- no fallback conformer generation is performed

Per-molecule conformers are merged only when atom counts are compatible.

## 7) Patch Volume Controls and Dynamic Auto-Cap

Main controls:

- `--chem_ace_max_ids`
- `--chem_ace_max_confs_per_id`
- `--chem_ace_local_radii`
- `--chem_ace_patch_cap_per_mol`
- `--chem_ace_target_total_patches`

If `chem_ace_patch_cap_per_mol <= 0`, auto-cap is used:

- `raw_cap = ceil(target_total_patches / n_molecules)`
- clipped to `[16, 256]`

If molecule exceeds cap:

- deterministic SHA1-based sampling keeps a stable subset

## 8) Patch Embedding in Final Runtime (Feature Projection Path)

Final integration uses feature fusion embedding in `_build_feature_patch_embeddings(...)`.

Per patch vector = concat of:

1. `v2d`:
- molecule-level 2D vector
- truncate/pad to effective `max_2d_dim`
- if `--chem_ace_max_2d_dim <= 0`, full raw 2D dim is used dynamically

2. `v3dqm`:
- conformer vector from `(mol_id, conf_id)` if available
- else molecule mean over conformers
- else zeros
- truncate/pad to effective `max_3dqm_dim`
- if `--chem_ace_max_3dqm_dim <= 0`, full raw merged 3D+QM dim is used dynamically

3. patch descriptor block:
- 10 structural descriptors:
  - atom count
  - aromatic fraction
  - hetero fraction
  - formal charge sum
  - conjugated bond fraction
  - ring bond fraction
  - mean atomic number
  - std atomic number
  - mean degree
  - std degree
- plus 5-way patch-type one-hot

Descriptor scaling:

- `RobustScaler` fitted on discover-train descriptor rows (`max_fit_samples=200000`)
- transform is reused for infer-scope patches

## 9) Embedding Persistence Policy (Disk Safety)

Default:

- `--no-chem_ace_persist_patch_embeddings` (effective default)

Meaning:

- patch embeddings stay in memory for concept discovery
- no per-patch `.npy/.json` cache writes
- no patch-embedding DB upsert

When explicitly enabled with `--chem_ace_persist_patch_embeddings`:

- per-patch embedding cache files are written
- embedding DB rows are persisted

This default is intentional to avoid disk exhaustion on very large patch counts.

## 10) Concept Discovery (Current Defaults)

Discovery config defaults in `ConceptDiscoveryConfig`:

- `algorithms=("kmeans",)`
- `kmeans_k=0` (auto)
- `kmeans_minibatch_over=200000`
- `kmeans_minibatch_size=4096`

KMeans `k` resolution:

- if `kmeans_k > 1`: use configured `k` (bounded to sample count)
- else: `k = floor(sqrt(n_embeddings))` (bounded)

Optional algorithms still exist but are guarded for memory:

- hierarchical:
  - `hierarchical_max_samples=25000`
  - `hierarchical_max_pairwise_gb=8.0`
- hdbscan:
  - `hdbscan_max_samples=300000`

Pipeline logs the effective concept-discovery configuration before clustering.

## 11) Semantic Tagging Pipeline

Tagging entry:

- `explainability.chem_ace.tag_concepts` in runtime

Tagging orchestration:

- concept membership index build
- per-concept semantic tagging (parallelized by CPU workers)
- DB persistence of labels and tags

Per-concept semantic extraction includes:

- formal charge/Gasteiger stats
- aromaticity and conjugation descriptors
- planarity/rotatable-bond geometry from conformers
- pharmacophore counts
- SMARTS functional matches
- SMARTS-RX matches and role rates
- geometry family summary from 3D descriptor vectors
- QM family summary from QM descriptor vectors
- optional OpenBabel summary (`logP`, `TPSA`, `MR`)

Cross-modal tags combine structural + geometric + QM signals.

Naming:

- rule-based naming first
- fallback descriptor-driven naming

## 12) Detailed Tagging Logs (Latest Update)

Tagging now has explicit substep logs:

From `concepts/pipeline.py`:

- `explainability.chem_ace.tag_concepts.membership_index`
- `explainability.chem_ace.tag_concepts.compute` (`START/PROGRESS/DONE`)
- `explainability.chem_ace.tag_concepts.compute_one` (`FAIL` with concept id)
- `explainability.chem_ace.tag_concepts.persist` (`START/PROGRESS/DONE`)

From `semantics/taggers.py`:

- OpenBabel backend status:
  - `explainability.chem_ace.tag_concepts.openbabel` with `enabled` and reason/backend
- per concept:
  - `explainability.chem_ace.tag_concepts.tag_one` (`START/DONE`)
  - `explainability.chem_ace.tag_concepts.compute_descriptors` (`START/DONE`)
  - `explainability.chem_ace.tag_concepts.compute_descriptors.loop` (`PROGRESS` for large concepts)
  - `explainability.chem_ace.tag_concepts.compute_descriptors.summary`
  - `explainability.chem_ace.tag_concepts.openbabel_summary` (`START/DONE` when OpenBabel is enabled)

This is the authoritative answer to what happens at `[START] explainability.chem_ace.tag_concepts`.

## 13) Activity-Aware Semantic Calibration (Latest Update)

A new supervised calibration layer is integrated:

- module: `explainability/chem_ace/semantics/calibration.py`
- class: `ActivityAwareSemanticCalibrator`

When it runs:

- after base concept tags are generated on `discover_train`
- using train IDs only and train task labels only
- before concept metadata is finalized for downstream outputs

### 13.1 What it calibrates

It calibrates confidence for each `(concept, tag)` pair.

It does not redefine concepts and does not use infer/leaderboard labels.

### 13.2 Signals used

For each concept and each tag:

- task enrichment profile (4 tasks)
- bitmask enrichment profile (up to 16 masks; optional exclusion of mask `0`)
- support counts with shrinkage/saturation

### 13.3 Core scoring logic (implemented)

For a selected sample mask `M` (concept-membership or tag-membership over train IDs):

1. Task channel:
- `task_prev = mean(y_train, axis=0)`
- `task_post = (hits + task_prev * prior_strength) / (|M| + prior_strength)`
- `task_ratio = task_post / task_prev`
- ratio normalized to `[0,1]` via clipping to `[1, ratio_cap]`
- `task_score = (1 - min_w) * mean(norm) + min_w * min(norm)`

2. Bitmask channel:
- same smoothed ratio logic over selected bitmask IDs
- candidate masks filtered by `bitmask_min_count`
- optional exclusion of all-negative mask id `0`
- best normalized bitmask ratio is used as bitmask score

3. Channel fusion:
- `fused = weighted_mean(task_score, bitmask_score; task_weight, bitmask_weight)`

4. Support scaling:
- concept/tag supports are mapped by sqrt saturation to `[0,1]`
- combined as average support scale

5. Final confidence:
- `total_score = (0.6 * concept_fused + 0.4 * tag_fused) * support_scale`
- `calibrated_conf = clip((1 - mix_base) * base_conf + mix_base * total_score, min_confidence, max_confidence)`

6. Keep rule:
- keep tag if `calibrated_conf >= keep_threshold`
- if none kept and `fallback_top1_if_empty=true`, keep top-1 tag

### 13.4 Persistence behavior

Base tags are still persisted by `tag_and_store_concepts`.

Calibrated tags are additionally persisted with provenance suffix:

- `|activity_calibrated`

`concept_metadata` used downstream is built from calibrated result set.

### 13.5 Calibration outputs

Written in Chem-ACE output dir when calibration produces rows:

- `concept_tags_calibrated.csv`
- `concept_tags_calibration_summary.json`

Also included in:

- `chem_ace_pipeline_summary.json`
- final `explainability_artifacts.json`

## 14) A Priori Molecule-Level Semantic Baseline

Chem-ACE exports structure-only baseline tags (independent of concept clustering):

- `a_priori_tags.csv`
- `a_priori_vs_concepts.csv`
- `a_priori_tags_infer_scope.csv`
- `a_priori_vs_concepts_infer_scope.csv`

These use:

- functional SMARTS rules
- RDKit fragment rules
- SMARTS-RX rules

No concept discovery required for baseline tag generation itself.

## 15) Functional Rule Auto-Augmentation from Dataset SMILES

During runtime, functional rules are auto-augmented from `curated_SMILES` via RDKit `Chem.Fragments.fr_*` functions.

Generated artifacts:

- `rules_autogen/default_functional_group_rules.dataset.json`
- `rules_autogen/functional_group_fragment_stats.json`

Merged rule file is injected into semantic config for the run.

## 16) OpenBabel Usage Semantics

OpenBabel usage is optional.

If enabled and available:

- backend availability is logged at tagger initialization
- per-concept OpenBabel summary step is logged

If missing:

- warning log is emitted
- pipeline continues with RDKit-only semantics

## 17) Main Runtime Stages and What They Mean

For `discover_train` phase:

1. `explainability.chem_ace.generate_patches`
2. `explainability.chem_ace.embed_patches`
3. `explainability.chem_ace.discover_concepts`
4. `explainability.chem_ace.tag_concepts`
5. `explainability.chem_ace.calibrate_semantics` (if enabled)

For `infer_scope` phase:

1. `explainability.chem_ace.generate_patches`
2. `explainability.chem_ace.embed_patches`
3. `explainability.chem_ace.infer_memberships`
4. `explainability.chem_ace.persist_inferred_memberships` (if any)

If progress appears to stall after patch generation reaches 100%, typical next heavy sections are:

- patch/embedding persistence (if enabled)
- clustering
- per-concept semantic tagging descriptor loops

## 18) CLI Controls (Current)

Core enable flags:

- `--run_chem_ace`
- `--run_lambda_vol`
- `--run_concept_rl`

Patch/conformer controls:

- `--curated_smiles_col`
- `--chem_ace_conformer_sdf`
- `--chem_ace_sdf_conf_id_prop`
- `--chem_ace_max_ids`
- `--chem_ace_max_confs_per_id`
- `--chem_ace_local_radii`
- `--chem_ace_patch_cap_per_mol`
- `--chem_ace_target_total_patches`

Embedding/storage controls:

- `--chem_ace_max_2d_dim`
- `--chem_ace_max_3dqm_dim`
- `--chem_ace_persist_patch_embeddings` / `--no-chem_ace_persist_patch_embeddings`
- `--cpu_workers`

Concept/infer controls:

- `--chem_ace_top_concepts`
- `--chem_ace_infer_max_distance`

Activity calibration controls:

- `--run_activity_calibration` / `--no-run_activity_calibration`
- `--activity_calibration_min_concept_support`
- `--activity_calibration_min_tag_support`
- `--activity_calibration_prior_strength`
- `--activity_calibration_min_w`
- `--activity_calibration_task_weight`
- `--activity_calibration_bitmask_weight`
- `--activity_calibration_bitmask_min_count`
- `--activity_calibration_bitmask_exclude_zero` / `--no-activity_calibration_bitmask_exclude_zero`
- `--activity_calibration_mix_base`
- `--activity_calibration_keep_threshold`
- `--activity_calibration_min_confidence`
- `--activity_calibration_max_confidence`
- `--activity_calibration_ratio_cap`
- `--activity_calibration_fallback_top1_if_empty` / `--no-activity_calibration_fallback_top1_if_empty`

## 19) Output Artifacts (Current)

Primary Chem-ACE outputs in `chem_ace_output_dir` (default: `<study_dir>/chem_ace`):

- `chem_ace.sqlite3`
- `chem_ace_pipeline_summary.json`
- rule augmentation artifacts under `rules_autogen/`
- a priori semantic CSVs
- calibrated semantic CSV/JSON (if enabled and non-empty)

Potential embedding cache directory (only if embedding persistence enabled):

- `chem_ace_cache/`

Final run summary output (`final_best_train_vs_leaderboard/explainability_artifacts.json`) includes pointers for:

- Chem-ACE core artifacts
- a priori views
- activity-calibration artifacts
- Lambda-Vol artifacts
- Concept-RL policy history
- prediction explanation CSV

## 20) Downstream Consumption

Chem-ACE concept maps feed:

- Lambda-Vol concept-pressure tracking
- Ricci diagnostics over concept relations
- Concept-RL target concept selection
- text explanations exported with per-task predictions/attention

## 21) Known Limits and Practical Guidance

- Patch counts can still be very large on huge datasets with many conformers.
- Keep local radii conservative (`1` default) and use dynamic/fixed patch caps.
- Leave embedding persistence disabled unless explicitly required.
- For large runs, keep concept discovery on scalable settings (`kmeans` default).
- Semantic tagging is CPU-heavy by design due to chemistry operations.
- OpenBabel is optional; missing backend does not break runtime.

## 22) Code Map

Core:

- `explainability/chem_ace/config.py`
- `explainability/chem_ace/types.py`
- `explainability/chem_ace/optional_deps.py`

Patches:

- `explainability/chem_ace/patches/base.py`
- `explainability/chem_ace/patches/local_subgraph.py`
- `explainability/chem_ace/patches/brics.py`
- `explainability/chem_ace/patches/murcko.py`
- `explainability/chem_ace/patches/pharm3d.py`

Concept discovery and pipeline:

- `explainability/chem_ace/concepts/clustering.py`
- `explainability/chem_ace/concepts/pipeline.py`

Semantics:

- `explainability/chem_ace/semantics/taggers.py`
- `explainability/chem_ace/semantics/calibration.py`
- `explainability/chem_ace/semantics/naming.py`

DB and analytics:

- `explainability/chem_ace/db/models.py`
- `explainability/chem_ace/db/repository.py`
- `explainability/chem_ace/analytics/queries.py`

Runtime integration:

- `training/explainability_runtime.py`
- `entrypoints/hpo_pipeline.py`
- `training/execution.py`

