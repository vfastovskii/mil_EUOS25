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

Paper-alignment checklist:

- `docs/ACE_TCAV_ALIGNMENT.md`

## 1) Runtime Scope and Trigger Conditions

Chem-ACE runs in final pipeline when `--run_chem_ace` is enabled.

Chem-ACE is automatically enabled if either of these is requested:

- `--run_lambda_vol`
- `--run_concept_rl`
- `--run_concept_rl_ablation`

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

Feature-source contract:

- model/HPO/Chem-ACE embedding-clustering path uses scaled model inputs (`--feat2d_scaled`, `--feat3d_scaled`, `--feat3d_qm_scaled`)
- semantic descriptor/tag summary can optionally use raw 3D/QM tables (`--feat3d_raw` + `--feat3d_qm_raw`)
- raw semantic mode is enabled only when both raw tables are provided; otherwise semantic summaries use scaled vectors

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

## 6.1) Optional Conformer Pharmacophore Signatures (pmapper)

To analyze why specific conformers receive high attention, Chem-ACE can compute conformer-level 3D pharmacophore signatures from the same SDF conformers used for 3D patching.

Controls:

- `--chem_ace_use_pmapper_signatures` / `--no-chem_ace_use_pmapper_signatures` (default enabled)
- `--chem_ace_pmapper_tol` (default `0`)
- `--chem_ace_pmapper_tol_alt` (default `5`)

Behavior:

- signatures are computed by `conf_id` from SDF entries
- computation is optional and dependency-safe (skips with logged reason if `pmapper` is unavailable)
- no concept leakage is introduced (signatures are conformer metadata, not label-derived)

Artifacts:

- `chem_ace/conformer_pmapper_signatures.csv`
- `chem_ace/conformer_pmapper_signatures_summary.json`

Attention export integration:

- leaderboard attention rows include `pmapper_sig_md5` and `pmapper_sig_md5_alt` (when available)
- for each task, signature-group diagnostics are added:
  - `<sig_col>_mass_<task>`: total attention mass for this signature within molecule
  - `<sig_col>_rank_<task>`: rank of signature by attention mass within molecule
  - `<sig_col>_top_<task>`: 1 when this signature is top-ranked and has non-zero mass

Prediction explanation integration:

- per-task explanation text adds a pharmacophore-signature note when signature mass/rank indicates dominant conformer-level motif reuse.

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

## 8) Patch Embedding in Final Runtime (Hybrid Local+Context by Modality)

Legacy long-vector discovery embedding (`concat(full_2d, full_3dqm, patch_desc)`) is removed from the final runtime path.

Current runtime uses `_build_hybrid_patch_embeddings(...)` with three separate modality spaces:

- `2d`
- `3d_geom`
- `3d_qm`

Each patch is routed into exactly one modality:

- `conf_id is None` -> `2d`
- conformer patch with missing/invalid QM slice -> `3d_geom`
- conformer patch with QM support -> `3d_qm`

Per modality, embedding logic is:

- `z_local = MLP_local(local_features)`
- `z_ctx = MLP_ctx(context_features)`
- `z = LayerNorm(z_local + alpha * z_ctx)`
- `z = l2_normalize(z)`

Default dimensions:

- `--chem_ace_embed_dim_2d 64`
- `--chem_ace_embed_dim_3d_geom 64`
- `--chem_ace_embed_dim_3d_qm 64`
- `--chem_ace_context_dim 16`
- `--chem_ace_context_alpha 0.2`
- `--chem_ace_qm_gating` enabled

Local feature blocks:

- `2d local`: patch subgraph sketch + patch structural descriptors
- `3d_geom local`: patch geometry summary from conformer coordinates + patch structural descriptors
- `3d_qm local`: QM signature features (optionally geometry-gated) + compact geometry signal + patch structural descriptors

Context feature blocks:

- `2d context`: molecule-level 2D vector
- `3d_geom context`: conformer geometry slice from instance vector
- `3d_qm context`: conformer QM slice from instance vector

Descriptor scaling:

- `RobustScaler` is fitted on discover-train patch descriptor rows (`max_fit_samples=200000`)
- scaler is reused for infer-scope descriptor transform
- scaler affects descriptor block only, not semantic tag post-processing

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

## 9.1) Strict Mixer-Space Re-Embedding and Rerank (New)

Chem-ACE now has a second, strict post-discovery pass in trained model space.

Goal:

- keep full-scale fast discovery for coverage
- then re-score concept relevance in actual trained MIL representation space
- improve final explanation ranking fidelity without changing concept definitions

Activation layer:

- default: `mixer_post_norm`
- configurable by `--chem_ace_strict_rerank_layer_name`

Subset used in strict pass:

1. concept medoids (from train-discovered/frozen concepts)
2. top-attention leaderboard conformer rows per task

Strict pass does **not** re-cluster concepts and does **not** update concept IDs.

### 9.1.1) How strict scores are computed

For each selected leaderboard row `(ID, conf_id, task)`:

1. run a single-conformer forward through trained MIL model
2. capture task embedding from `mixer_post_norm`
3. find row-active concepts using existing Chem-ACE membership maps:
   - conf-level membership union molecule-level membership
4. for each active concept, get medoid embedding in same layer space
5. compute cosine similarity:
   - `strict_cosine = cos(z_row_task, z_medoid_task)`

Result: strict score table with rows:

- `ID`, `conf_id`, `task`, `task_idx`, `concept_id`, `strict_cosine`, `attn`

### 9.1.2) How strict scores affect explanations

`export_prediction_text_explanations(...)` now supports strict score input.

For each candidate concept in ranking:

- load row-level strict score if available
- fallback to global `(task, concept)` strict mean
- apply gain:
  - `strict_gain = clip(1 + strict_weight * strict_score, 0.25, 2.50)`
- multiply base concept rank score by `strict_gain`

This changes concept order in explanations while preserving:

- anti-leakage contract
- concept IDs and semantic tags
- TCAV/Ricci computations

### 9.1.3) Strict-pass controls

- `--chem_ace_strict_rerank` / `--no-chem_ace_strict_rerank` (default enabled)
- `--chem_ace_strict_rerank_layer_name` (default `mixer_post_norm`)
- `--chem_ace_strict_rerank_top_rows_per_task` (default `256`)
- `--chem_ace_strict_rerank_batch_size` (default `256`)
- `--chem_ace_strict_rerank_weight` (default `0.35`)

### 9.1.4) Strict-pass artifacts

Written in final run directory (`final_best_train_vs_leaderboard/`):

- `chem_ace_strict_mixer_scores.csv`
- `chem_ace_strict_mixer_concepts.csv`
- `chem_ace_strict_mixer_summary.json`

Also referenced in:

- `final_best_train_vs_leaderboard/explainability_artifacts.json`

## 10) Concept Discovery (Current Defaults)

Discovery config defaults in `ConceptDiscoveryConfig`:

- `algorithms=("kmeans",)`
- `kmeans_k=0` (auto)
- `kmeans_auto_max_k=128` (hard ceiling in auto mode)
- `kmeans_minibatch_over=200000`
- `kmeans_minibatch_size=4096`

KMeans `k` resolution:

- if `kmeans_k > 1`: use configured `k` (bounded to sample count)
- else: `k = floor(sqrt(n_embeddings))`, then clipped by `kmeans_auto_max_k`

Optional algorithms still exist but are guarded for memory:

- hierarchical:
  - `hierarchical_max_samples=25000`
  - `hierarchical_max_pairwise_gb=8.0`
- hdbscan:
  - `hdbscan_max_samples=300000`

Pipeline logs the effective concept-discovery configuration before clustering.

Modality separation:

- discovery runs independently for `2d`, `3d_geom`, `3d_qm`
- concept IDs are namespaced by modality:
  - `2d:<sha1>`
  - `3d_geom:<sha1>`
  - `3d_qm:<sha1>`
- merged concept view is created only after per-modality clustering and deduplication
- frozen-centroid inference is also executed per modality

Practical implication on large runs:

- at `>1M` patch embeddings, auto mode will no longer explode to ~1000 clusters by default
- effective `k` is now bounded by `kmeans_auto_max_k` unless you explicitly raise it

Additional artifacts written by runtime:

- `concept_catalog.csv` (explicit modality per concept)
- `memberships_train_inferred.csv`
- `memberships_infer_scope.csv`

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

Semantic vector source:

- if raw tables are provided as a pair (`--feat3d_raw`, `--feat3d_qm_raw`), semantic geom/QM summaries are computed from raw vectors
- otherwise semantic geom/QM summaries are computed from scaled vectors
- runtime logs this explicitly via:
  - `explainability.chem_ace.semantics_instances source=raw`
  - or `source=scaled`

TCAV note:

- TCAV in Lambda-Vol is computed from model activations, not directly from raw descriptor values
- using raw tables affects semantic label evidence, not the model activation geometry used for TCAV
- Lambda-Vol TCAV now uses holdout evaluation by default:
  - CAV fit on monitor-train split
  - directional derivatives evaluated on monitor-holdout split when feasible
- Per-epoch repeat-level significance tables are exported under:
  - `<lambda_vol_output_dir>/tcav_significance/tcav_significance_epoch_XXXX.csv`
- Significance columns include raw and Bonferroni-corrected p-values/flags.

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
- `--run_concept_rl_ablation`

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

- `--chem_ace_embed_dim_2d`
- `--chem_ace_embed_dim_3d_geom`
- `--chem_ace_embed_dim_3d_qm`
- `--chem_ace_context_dim`
- `--chem_ace_context_alpha`
- `--chem_ace_qm_gating` / `--no-chem_ace_qm_gating`
- `--chem_ace_max_2d_dim` (deprecated compatibility flag, ignored by hybrid runtime)
- `--chem_ace_max_3dqm_dim` (deprecated compatibility flag, ignored by hybrid runtime)
- `--feat3d_raw` (optional, semantic summaries only)
- `--feat3d_qm_raw` (optional, semantic summaries only)
- `--chem_ace_persist_patch_embeddings` / `--no-chem_ace_persist_patch_embeddings`
- `--cpu_workers`
- `--chem_ace_strict_rerank` / `--no-chem_ace_strict_rerank`
- `--chem_ace_strict_rerank_layer_name`
- `--chem_ace_strict_rerank_top_rows_per_task`
- `--chem_ace_strict_rerank_batch_size`
- `--chem_ace_strict_rerank_weight`

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

Advanced geometry/topology controls (new):

- `--chem_ace_use_advanced_geom_topology` / `--no-chem_ace_use_advanced_geom_topology`
- `--chem_ace_advanced_geom_topology_max_patches`
- `--chem_ace_advanced_geom_topology_min_atoms`
- `--chem_ace_advanced_geom_topology_max_torsion_paths`
- `--chem_ace_advanced_geom_use_convex_hull` / `--no-chem_ace_advanced_geom_use_convex_hull`
- `--chem_ace_advanced_geom_use_persistent_homology` / `--no-chem_ace_advanced_geom_use_persistent_homology`
- `--chem_ace_advanced_geom_persistence_max_atoms`

Optional ORCA descriptor-table controls (new):

- `--chem_ace_use_orca_descriptors` / `--no-chem_ace_use_orca_descriptors`
- `--chem_ace_orca_descriptors_path`
- `--chem_ace_orca_conf_id_col`
- `--chem_ace_orca_mol_id_col`
- `--chem_ace_orca_descriptor_cols`
- `--chem_ace_orca_min_vectors_for_tagging`
- `--chem_ace_orca_z_threshold`

Optional dependency notes for new controls:

- persistent homology requires `ripser`
- convex-hull cavity/surface proxies require `scipy`
- ORCA table ingestion uses standard `pandas` readers (`csv`/`json`/`parquet`)

## 19) Output Artifacts (Current)

Primary Chem-ACE outputs in `chem_ace_output_dir` (default: `<study_dir>/chem_ace`):

- `chem_ace.sqlite3`
- `chem_ace_pipeline_summary.json`
- rule augmentation artifacts under `rules_autogen/`
- a priori semantic CSVs
- calibrated semantic CSV/JSON (if enabled and non-empty)

`chem_ace_pipeline_summary.json` includes:

- `semantics_instance_source` (`raw` or `scaled`)

Potential embedding cache directory (only if embedding persistence enabled):

- `chem_ace_cache/`

Strict rerank outputs in final run directory:

- `final_best_train_vs_leaderboard/chem_ace_strict_mixer_scores.csv`
- `final_best_train_vs_leaderboard/chem_ace_strict_mixer_concepts.csv`
- `final_best_train_vs_leaderboard/chem_ace_strict_mixer_summary.json`

Final run summary output (`final_best_train_vs_leaderboard/explainability_artifacts.json`) includes pointers for:

- Chem-ACE core artifacts
- a priori views
- activity-calibration artifacts
- Lambda-Vol TCAV significance per-epoch CSVs (when Lambda-Vol is enabled)

Advanced semantic evidence now also includes:

- `advanced_geometry_topology` summary block per concept in semantic evidence
- `orca_summary` block per concept when ORCA descriptors are enabled and loaded
- new provenance families in concept tags:
  - `advanced_geometry_topology_tagger`
  - `orca_descriptor_tagger`
  - `cross_modal_orca_tagger`
- Lambda-Vol artifacts
- Concept-RL policy history
- prediction explanation CSV

## 20) Downstream Consumption

Chem-ACE concept maps feed:

- Lambda-Vol concept-pressure tracking
- Ricci diagnostics over concept relations
- Concept-RL target concept selection
  - targets are filtered to injectable concepts only (must have conformer support in train scope)
  - this keeps RL aligned with attention-net controllable channels
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
