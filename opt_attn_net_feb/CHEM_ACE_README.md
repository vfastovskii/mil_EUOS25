# Chem-ACE (Automatic Concept Discovery + Semantic Tagging)

This document is the detailed technical reference for Chem-ACE in this repository:

- what a "patch" is
- why patches are computed
- how patch generation differs for 2D vs 3D
- how patches are embedded, clustered, tagged, and consumed downstream
- what exactly happens for conformer-aware 3D patching

Code references are provided so behavior can be verified directly.

## 1) Why Chem-ACE exists in this project

Chem-ACE converts raw molecules into reusable concept units, then links those concepts to model behavior.
The goal is not only to produce importance numbers, but to produce:

- cluster-level concepts
- semantic concept labels
- concept-to-task influence signals (TCAV/CAV)
- concept prevalence/support signals for MIL and Lambda-Vol monitoring

In practice, Chem-ACE runs as a stage in the final pipeline when `--run_chem_ace` is enabled (and is automatically enabled if Lambda-Vol or Concept-RL is requested).

Primary orchestration:

- `training/explainability_runtime.py`
- `explainability/chem_ace/concepts/pipeline.py`

## 2) Patch definition (core object)

A patch is a deterministic molecular part candidate.
Data model:

- `PatchRecord` in `explainability/chem_ace/types.py`

Fields:

- `patch_id`: unique, deterministic, includes molecule + conformer context
- `mol_id`: molecule identifier
- `conf_id`: conformer identifier, `None` for 2D-only patches
- `patch_type`: generator type (`local_subgraph`, `brics`, `murcko`, `murcko_framework`, `pharm3d`)
- `atom_indices`: atom IDs in the source molecule for this patch
- `patch_hash`: deterministic content hash of patch payload
- optional fields: `smarts`, `fragment_repr`, `feature_metadata`

### Identity rules

Implemented in `explainability/chem_ace/patches/base.py`:

1. `patch_hash` hashes patch content only:
   - type
   - atom indices
   - SMARTS/fragment
   - metadata
2. `patch_id` hashes `(mol_id, conf_id, patch_hash)`.

Consequence:

- same motif at different conformers has same `patch_hash` but different `patch_id`
- this is required for conformer-aware downstream analysis

## 3) Patch generators implemented

Patch generators implement `PatchGenerator.generate(mol_id, mol, conf_id)`.

### 3.1 Local subgraph (`local_subgraph`)

File:

- `explainability/chem_ace/patches/local_subgraph.py`

Logic:

1. iterate every atom as center
2. for each configured radius, extract atom environment
3. include center atom (default `include_center_atom=True`)
4. export best-effort `fragment_repr` and `smarts`
5. store metadata:
   - `center_atom`
   - `radius`

Default radius configuration in current code:

- `radii=(1,)`

This was intentionally tightened to reduce patch explosion.

### 3.2 BRICS (`brics`)

File:

- `explainability/chem_ace/patches/brics.py`

Logic:

1. break BRICS bonds
2. enumerate fragments
3. preserve original atom indices via `_orig_idx`
4. export fragment SMARTS/SMILES
5. emit patch per unique BRICS fragment

### 3.3 Murcko scaffold (`murcko`, `murcko_framework`)

File:

- `explainability/chem_ace/patches/murcko.py`

Logic:

1. compute Murcko scaffold
2. map scaffold back to source atom indices
3. emit scaffold patch
4. optionally emit generic framework patch (`murcko_framework`)

### 3.4 3D pharmacophore (`pharm3d`)

File:

- `explainability/chem_ace/patches/pharm3d.py`

Logic:

1. requires conformers (`requires_conformer=True`)
2. build RDKit feature factory (`BaseFeatures.fdef`)
3. compute ChemicalFeatures for target conformer
4. per feature, create patch using feature atom IDs
5. store metadata:
   - `family`
   - `feature_type`
   - `position` (`x`, `y`, `z`)

## 4) 2D vs 3D generation flow (critical behavior)

Pipeline file:

- `explainability/chem_ace/concepts/pipeline.py`

Generators are split by `requires_conformer`:

- `patch_generators_2d`: run once per molecule with `conf_id=None`
- `patch_generators_3d`: run per conformer with real `conf_id`

For each molecule:

1. run all 2D generators exactly once
2. if conformers are available, run all 3D generators for each conformer
3. deduplicate by `patch_id`
4. apply deterministic per-molecule patch cap

This means:

- 2D patches are not multiplied by conformer count
- only truly 3D patch types scale with conformer count

## 5) 3D conformer ingestion and matching

Runtime file:

- `training/explainability_runtime.py`

### 5.1 Source of conformers

Conformers are loaded from precomputed SDF when `--chem_ace_conformer_sdf` is passed.
Chem-ACE does not generate conformers in this final runtime path.

### 5.2 How conformer IDs are resolved

SDF lookup key priority:

1. property `--chem_ace_sdf_conf_id_prop` (default `conf_id`)
2. fallback to `_Name` property

Functions:

- `_load_sdf_conformers_by_conf_id(...)`

### 5.3 Per-molecule conformer merge

For each molecule ID, candidate `conf_id`s come from instance data.
Then:

1. locate corresponding entries in SDF map
2. keep only conformers with compatible atom count
3. merge conformers into one RDKit molecule
4. store map `_chemace_conf_id_map: {external_conf_id -> internal_conf_index}`

Function:

- `_merge_sdf_conformers_for_molecule(...)`

Skipped categories are counted and logged:

- missing in SDF
- incompatible atom counts
- missing conformer IDs
- duplicates in SDF index

### 5.4 How `pharm3d` picks conformer index

`Pharm3DPatchGenerator._resolve_conf_idx(...)`:

1. try `int(conf_id)` as direct RDKit conformer index
2. else try `_chemace_conf_id_map`
3. else if exactly one conformer exists, use index `0`
4. else skip patch

This guarantees no silent wrong conformer assignment.

### 5.5 Exact scope used for `n_molecules` in logs

The `n_molecules` printed at:

- `explainability.chem_ace.generate_patches n_molecules=...`

is **not** "all rows in labels".

In the final pipeline it is built from:

1. IDs in split `train`
2. after dropping IDs with no valid conformer bags
3. after applying `--chem_ace_max_ids` (if > 0)
4. after deduplication to unique IDs

This is an intentional anti-leakage policy with two phases:

- concept discovery uses train-only IDs
- leaderboard IDs are **not** used to define/update concept clusters
- leaderboard explainability uses inference-only assignment:
  - generate leaderboard patches + embeddings
  - assign each leaderboard patch to nearest frozen train centroid
  - optional distance gate via `--chem_ace_infer_max_distance`
  - no reclustering on leaderboard

Additional scope logs now emitted:

- `final.prepare_chem_ace_bundle.scope n_train_ids=... n_leaderboard_ids=... n_scope_ids=... scope_splits=train anti_leakage=enabled`
- `explainability.chem_ace.scope_ids n_discover_ids_input=... n_discover_ids_selected=... n_infer_ids_input=... n_infer_ids_selected=... chem_ace_max_ids=...`
- `explainability.chem_ace.generate_patches phase=discover_train|infer_scope ...`
- `explainability.chem_ace.embed_patches phase=discover_train|infer_scope ...`
- `explainability.chem_ace.infer_memberships ...`

## 6) Patch volume control and scaling

Config sources:

- `explainability/chem_ace/config.py`
- `training/explainability_runtime.py`

Key controls:

- `--chem_ace_max_ids` (default `0` = all IDs in scope)
- `--chem_ace_max_confs_per_id` (default `0` = all conformers per molecule)
- `--chem_ace_local_radii` (default `1`)
- `--chem_ace_patch_cap_per_mol` (default `0`, means auto)
- `--chem_ace_target_total_patches` (default `1200000`)
- `--chem_ace_infer_max_distance` (default `-1.0`, disabled; if `>0`, drops far leaderboard assignments)

### Auto-cap math

If `patch_cap_per_mol <= 0`, pipeline computes:

- `raw_cap = ceil(target_total_patches / n_molecules)`
- then clips to `[16, 256]`

So final cap is:

- `cap = min(256, max(16, raw_cap))`

### Deterministic sampling under cap

If a molecule produces more than `cap` patches:

1. rank patches by deterministic SHA1 key based on `(mol_id, patch_id, constant-salt)`
2. keep top `cap`

This ensures reproducible subset selection across runs with same inputs.

## 7) What happens after patches are generated

### 7.1 Persistence

Patches are persisted in DB:

- `patches` table
- parent `molecules` / `conformers` rows

Large writes use chunked upsert for SQLite in:

- `explainability/chem_ace/db/repository.py`

### 7.2 Patch embedding

There are two general Chem-ACE embedding interfaces:

- `masked_input` (hook activation from masked forward)
- `node_pooling` (pool node activations on patch atoms)

Files:

- `explainability/chem_ace/embedding/base.py`
- `explainability/chem_ace/embedding/strategies.py`

### Current final MIL pipeline embedding path

In this repository's final runtime integration, patch vectors are currently built via feature fusion (`feature_projection`) in:

- `_build_feature_patch_embeddings(...)` in `training/explainability_runtime.py`

Per patch vector is:

1. 2D molecular vector (truncate/pad to resolved 2D dim)
   - `chem_ace_max_2d_dim > 0`: use that cap
   - `chem_ace_max_2d_dim <= 0`: auto-use full 2D raw dimension
2. 3D+QM vector:
   - use conformer-specific `(mol_id, conf_id)` when available
   - else use molecule-level mean across conformers
   - else zero vector
3. patch descriptors:
   - 10 structural stats (size, aromaticity, hetero fraction, charge, conjugation, ring, atomic number/degree stats)
   - 5-way patch-type one-hot

Embedding record is cached and persisted (`patch_embeddings` table plus vector artifact URI).

### 7.3 Concept discovery

File:

- `explainability/chem_ace/concepts/clustering.py`

Algorithms:

- k-means (required)
- hierarchical (required)
- HDBSCAN (optional if installed)

Flow:

1. cluster embeddings per algorithm
   - k-means switches to MiniBatchKMeans for large `N` (`kmeans_minibatch_over`)
   - hierarchical is guarded by `hierarchical_max_samples` and `hierarchical_max_pairwise_gb` to avoid O(N^2) memory blowups
   - HDBSCAN is guarded by `hdbscan_max_samples`
   - if one algorithm fails/skips, others still run
2. compute support and coherence
3. filter by `min_support` and `min_coherence`
4. compute centroid + medoid patch
5. deduplicate near-duplicate concepts by centroid cosine similarity
6. persist immutable concept-set snapshot + concept + membership rows

### 7.4 Semantic tagging and naming

File:

- `explainability/chem_ace/semantics/taggers.py`
- `explainability/chem_ace/rules/default_naming_rules.json`
- `explainability/chem_ace/rules/default_functional_group_rules.json`
- `explainability/chem_ace/rules/smartsrx.json`

Computed descriptor families:

- charge/formal + Gasteiger
- aromaticity/conjugation
- geometry from conformer coordinates (if conformers available and resolvable)
- geometry from 3D descriptor vectors (column-name token families from `geom_cols`)
- pharmacophore counts
- SMARTS functional-group coverage (carboxylate, amide, sulfonamide, phosphates, amines, heteroaromatics, halogen motifs, boron motifs, ring classes, etc.)
- SMARTS-RX reactivity-function coverage (electrophile/nucleophile/acid-base/leaving-group/redox/coordination/cycloaddition motifs)
- QM descriptor semantics from per-conformer QM columns (token-mapped families such as HOMO/LUMO/gap, dipole, polarizability, hardness/softness, electrophilicity/nucleophilicity, charge-transfer/Fukui/ESP proxies)
- optional Open Babel descriptor summary (`logP`, `TPSA`, `MR`) when `openbabel.pybel` is available

3D geometry tags are derived from two sources:

1. coordinate-derived:
   - planarity RMSD
   - rotatable-bond proxy for rigid/flexible labels
2. descriptor-derived (from `inst_geom_cols` + `inst_geom_dim`, per patch):
   - family summaries for distance / angle / dihedral / planarity / shape / size / inertia / surface-volume / ring-strain / hbond-geometry
   - global 3D descriptor blocks (RDF / MORSE / WHIM / GETAWAY / 3D autocorrelation tokens)
   - tags gated by:
     - `geom_min_vectors_for_tagging`
     - `geom_z_threshold`
     - `geom_strong_z_threshold`

Per-concept descriptor evidence now includes family-coverage diagnostics:

- `geom_summary.family_coverage`
- `geom_summary.n_family_matched_features`
- `geom_summary.n_unmatched_features`
- `geom_summary.top_unmatched_features`

Cross-modal tags are also emitted when signals agree across modalities:

- SMARTS-RX electrophile + QM electrophilicity -> `electrophilic reaction-center motif`
- SMARTS-RX nucleophile + QM nucleophilicity -> `nucleophilic reaction-center motif`
- aromatic + planar geometry + QM gap signal -> `planar conjugated electronic motif`
- HBD/HBA + QM dipole signal -> `polar donor-acceptor electronic motif`

Tag outputs include:

- `tag`
- `confidence`
- `provenance`
- `evidence_json`

Concept receives `label_auto` from:

1. rule-based registry
2. fallback descriptor-based description

Rule loading behavior:

1. naming rules: explicit `naming_rules_path` if set, else bundled `default_naming_rules.json`
2. functional rules: explicit `functional_rules_path` if set, else bundled `default_functional_group_rules.json`
3. SMARTS-RX rules: explicit `smarts_rx_rules_path` if set, else bundled `smartsrx.json` (legacy fallback: `default_smarts_rx_rules.json`)
4. invalid SMARTS are skipped with warning; pipeline continues

SMARTS-RX file formats supported:

1. canonical: `{ "rules": [ {tag, smarts, role, min_patch_rate, confidence, provenance}, ... ] }`
2. SMARTS-RX generated registry: `{ "data": [ {category, subcategory, specific_type, smarts}, ... ] }`
   - loader auto-derives:
     - `tag = rx_<specific_type|subcategory|category>` (normalized)
     - `role = <category>` (normalized)
     - defaults for confidence/threshold/provenance when missing

Molecule-level baseline exports (a priori, structure-only):

1. `a_priori_tags.csv`:
   - one row per molecule in Chem-ACE scope
   - tags from functional SMARTS / RDKit fragment rules / SMARTS-RX only
2. `a_priori_vs_concepts.csv`:
   - same rows, plus concept-level post-discovery annotations
3. `a_priori_tags_infer_scope.csv`:
   - infer/test subset (leaderboard scope in final run)
4. `a_priori_vs_concepts_infer_scope.csv`:
   - infer/test subset with concept-level annotations

RDKit Fragments augmentation:

1. during final Chem-ACE runtime, rules are auto-augmented from the labels table `curated_SMILES` column using all available `rdkit.Chem.Fragments.fr_*` functions
2. generated rules are merged with bundled defaults and written to:
   - `<chem_ace_output_dir>/rules_autogen/default_functional_group_rules.dataset.json`
   - `<chem_ace_output_dir>/rules_autogen/functional_group_fragment_stats.json`
3. this merged path is injected into `SemanticTaggingConfig.functional_rules_path` for concept tagging
4. tags from these rules use `provenance = "rdkit_fragment_tagger"`

Manual regeneration command:

```bash
python -m explainability.chem_ace.rules.generate_fragment_rules_from_labels \
  --labels_csv /path/to/master_table_labels_final_modelling_ready_1401_with_cv_split.csv \
  --smiles_col curated_SMILES \
  --output_json /path/to/default_functional_group_rules.json \
  --stats_json /path/to/functional_group_fragment_stats.json
```

QM semantic interpretation logic:

1. concept-level QM vectors are pulled from conformer-specific `(mol_id, conf_id)` rows
2. if conformer vector is missing, molecule-level mean vector is used as fallback
3. vector is split into geometry + QM by `inst_geom_dim` and `inst_qm_dim`
4. geometry and QM feature names are passed explicitly (`geom_cols`, `qm_cols`) and normalized
5. descriptor families are assigned by normalized column-name token matching
6. tags are emitted only when enough QM vectors are available (`qm_min_vectors_for_tagging`)
7. thresholds:
   - moderate: `qm_z_threshold`
   - strong: `qm_strong_z_threshold`

Descriptor basis from `docs/quantum_descriptors_list.pdf` is now explicitly covered in family token maps:

- frontier/conceptual DFT:
  - `homo_eV`, `lumo_eV`, `gap_eV`, `mu_eV`, `eta_eV`, `softness_1_per_eV`, `chi_eV`, `omega_eV`
- electrostatics:
  - `dipole_D`, `quad_norm_au`, `quad_trace_au`
- bond-order and conjugation:
  - `bo_sum`, `bo_max`, `bo_mean_bonds`, `bo_conj_sum`, `bo_conj_mean`
- atomic-charge distribution:
  - `q_min`, `q_max`, `q_mean`, `q_std`, `q_abs_sum`, `q_range`, `q_pos_top3_mean`, `q_neg_top3_mean`
- charge-separation geometry:
  - `q_abs_r_mean`, `q_abs_r2_rms`, `d_pos_neg`
- redox and local reactivity:
  - `vip_eV`, `vea_eV`, `fplus_max`, `fplus_sum_pos`, `fplus_top3_mean`, `fminus_max`, `fminus_sum_pos`, `fminus_top3_mean`

Additional QM tags now include:

- ionization/electron-affinity profile tags
- bond-order rigidification + conjugated bond-order network
- polarized atomic-charge landscape + long-range charge-separation profile
- Fukui+ / Fukui- hotspot profile tags
- anisotropic quadrupole field

Photophysics proxy tags (heuristic, not TD-DFT):

- `transmittance-favored photophysics proxy`
- `fluorescence-favored photophysics proxy`
- `red-shifted absorption proxy`
- `blue-shifted transparency proxy`

These are emitted by combining QM families with geometry/conjugation cues and are intentionally marked as proxy semantics.

SMARTS-RX semantics:

1. each SMARTS-RX rule defines `tag`, `smarts`, `role`, `min_patch_rate`, `confidence`
2. tags are emitted as rule-level tags (e.g., `rx_michael_acceptor`) plus role tags (`rx_role_electrophile`)
3. naming rules can combine structural tags + SMARTS-RX tags (for example: `rx_michael_acceptor` + `rx_role_electrophile`)
4. this gives concept labels that explicitly encode predicted reactivity class, not only structure

### 7.5 Downstream usage in Lambda-Vol and Concept-RL

After concept discovery, runtime builds:

- `concept_mol_map`: concept -> molecule IDs
- `concept_conf_map`: concept -> (`mol_id`, `conf_id`) pairs

These are used by Lambda-Vol monitoring to compute per-task/per-concept:

- prevalence
- attention support
- TCAV

They are also used to build positive concept target sets for Concept-RL.

Ricci geometry details for Lambda-Vol are documented in:

- `RICCI_FLOW_README.md`

## 8) Dedicated 3D patch lifecycle (end-to-end)

For one `(mol_id, conf_id)`:

1. molecule is built from `curated_SMILES` and Hs added
2. conformer from SDF is matched by configured property or `_Name`
3. conformer is merged into molecule; mapping is recorded
4. `pharm3d` extracts conformer-specific feature patches
5. each 3D patch keeps:
   - `conf_id`
   - atom indices
   - pharmacophore metadata with 3D coordinates
6. embedding stage pulls conformer-specific 3D/QM vector when present
7. patch participates in clustering with all other patch types
8. if selected into concept:
   - contributes to concept support
   - contributes to geometry/pharmacophore tags
   - contributes to conformer-level concept presence in Lambda-Vol metrics

If conformer is missing or incompatible:

- no 3D patch is generated for that `conf_id`
- 2D patching still proceeds for the molecule

## 9) Runtime logging and expected long steps

Chem-ACE emits structured progress events for:

- patch generation
- patch persistence
- embedding compute
- embedding persistence

Patch generation progress now includes explicit 2D/3D visibility:

- `patches` total accumulated patches
- `patches_2d` accumulated patches with `conf_id=None`
- `patches_3d` accumulated conformer-aware patches (`conf_id!=None`)
- `mols_with_conf_done` processed molecules that had at least one conformer candidate
- `mols_with_3d_patches_done` processed molecules that produced at least one 3D patch

End-of-stage summary log:

- `explainability.chem_ace.generate_patches.summary ... patches_total=... patches_2d=... patches_3d=...`

Post-generation ready log:

- `explainability.chem_ace.patches_ready n_patches=... n_patches_2d=... n_patches_3d=...`

Typical pattern:

1. `generate_patches` reaches `100%`
2. then DB persistence starts
3. then embedding compute starts
4. then concept discovery/tagging

So "no new patch progress after 100%" can be normal if it is in persistence/embedding stages.

## 10) CLI knobs relevant to patching

From `entrypoints/hpo_pipeline.py`:

- `--run_chem_ace`
- `--curated_smiles_col`
- `--chem_ace_conformer_sdf`
- `--chem_ace_sdf_conf_id_prop`
- `--chem_ace_max_ids`
- `--chem_ace_max_confs_per_id` (`<=0` means use all conformers per molecule)
- `--chem_ace_local_radii`
- `--chem_ace_patch_cap_per_mol`
- `--chem_ace_target_total_patches`
- `--chem_ace_max_2d_dim` (`<=0` means auto-use full 2D raw dimension)
- `--chem_ace_max_3dqm_dim` (`<=0` means auto-use full merged 3D+QM raw dimension)
- `--chem_ace_top_concepts`
- `--cpu_workers`

Example final-only run with pre-optimized params:

```bash
python opt_net_fast.py \
  --do_mil \
  --best_params_json /path/to/best_params.json \
  --run_chem_ace \
  --chem_ace_conformer_sdf /path/to/conformers.sdf \
  --chem_ace_sdf_conf_id_prop _Name \
  --curated_smiles_col curated_SMILES \
  --chem_ace_local_radii 1 \
  --chem_ace_patch_cap_per_mol 0 \
  --chem_ace_target_total_patches 1200000 \
  --cpu_workers 18
```

## 11) Outputs and artifacts

Main outputs (default under run `study_dir`):

- Chem-ACE DB: `chem_ace/chem_ace.sqlite3`
- cached embeddings: `chem_ace/chem_ace_cache/...`
- vector artifacts: `chem_ace/artifacts/...`
- run summary: `chem_ace/chem_ace_pipeline_summary.json`

Persisted DB entities include:

- runs, tasks
- molecules, conformers
- patches, patch_embeddings
- concept_sets (immutable snapshots), concepts, memberships
- concept_tags
- CAV/TCAV rows when TCAV is computed
- optional MIL concept epoch metrics

## 12) Package layout

- `explainability/chem_ace/config.py`
- `explainability/chem_ace/types.py`
- `explainability/chem_ace/optional_deps.py`
- `explainability/chem_ace/patches/*`
- `explainability/chem_ace/embedding/*`
- `explainability/chem_ace/concepts/clustering.py`
- `explainability/chem_ace/concepts/pipeline.py`
- `explainability/chem_ace/cav/tcav.py`
- `explainability/chem_ace/rules/smartsrx.json`
- `explainability/chem_ace/semantics/*`
- `explainability/chem_ace/db/*`
- `explainability/chem_ace/analytics/queries.py`
- `entrypoints/chem_ace_demo.py`

## 13) External descriptor ecosystem for larger semantic coverage

Current implementation is RDKit-first, then optional Open Babel.

Recommended expansion path:

1. RDKit functional hierarchy:
   - import hierarchy names/SMARTS from RDKit functional-group hierarchy file
   - auto-generate additional SMARTS rules into a versioned JSON registry
2. SMARTS-RX rule expansion:
   - curate rule blocks by reaction family (SNAr, SN2, acylation, Michael addition, click, metal coordination, redox)
   - keep each rule explicit: `(tag, smarts, role, min_patch_rate, confidence)` and version the JSON registry
3. Open Babel descriptors:
   - use `obabel -L descriptors` to enumerate plugins available in your environment
   - map selected descriptors to semantic tags (lipophilicity, polarity, refractivity, etc.)
4. QM parser toolchains:
   - if raw quantum outputs are available, parse with cclib and map parsed attributes to schema-level descriptor families
5. high-dimensional descriptor toolkits:
   - Mordred / PaDEL-style descriptors can be used for extra concept annotation channels, then compressed to stable semantic families

Design constraint:

- keep semantic tags interpretable and sparse; large descriptor sets should feed family-level tags, not raw-feature labels.

## 14) Assumptions and known limitations

- RDKit is required for full Chem-ACE behavior.
- HDBSCAN is optional; pipeline degrades gracefully if missing.
- Final MIL integration currently uses feature-fusion patch embedding (`feature_projection`) rather than hook-based embedding strategies.
- Patch volume can still be large; use radius/cap/target knobs aggressively on big datasets.

## 15) TODO (advanced extensions)

- richer 3D pharmacophore grouping and spatial motifs
- direct MIL attention-region extraction as native patch type
- tighter coupling between concept clusters and intervention policies
- active-learning hooks for concept-balanced data acquisition
