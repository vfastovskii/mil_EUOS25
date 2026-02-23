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

## 6) Patch volume control and scaling

Config sources:

- `explainability/chem_ace/config.py`
- `training/explainability_runtime.py`

Key controls:

- `--chem_ace_max_ids` (default `0` = all IDs in scope)
- `--chem_ace_local_radii` (default `1`)
- `--chem_ace_patch_cap_per_mol` (default `0`, means auto)
- `--chem_ace_target_total_patches` (default `1200000`)

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

1. 2D molecular vector (truncate/pad to `chem_ace_max_2d_dim`)
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
2. compute support and coherence
3. filter by `min_support` and `min_coherence`
4. compute centroid + medoid patch
5. deduplicate near-duplicate concepts by centroid cosine similarity
6. persist immutable concept-set snapshot + concept + membership rows

### 7.4 Semantic tagging and naming

File:

- `explainability/chem_ace/semantics/taggers.py`

Computed descriptor families:

- charge/formal + Gasteiger
- aromaticity/conjugation
- geometry (if conformers available and resolvable)
- pharmacophore counts

3D geometry tags are derived from conformer coordinates:

- planarity RMSD
- rotatable-bond proxy for rigid/flexible labels

Tag outputs include:

- `tag`
- `confidence`
- `provenance`
- `evidence_json`

Concept receives `label_auto` from:

1. rule-based registry
2. fallback descriptor-based description

### 7.5 Downstream usage in Lambda-Vol and Concept-RL

After concept discovery, runtime builds:

- `concept_mol_map`: concept -> molecule IDs
- `concept_conf_map`: concept -> (`mol_id`, `conf_id`) pairs

These are used by Lambda-Vol monitoring to compute per-task/per-concept:

- prevalence
- attention support
- TCAV

They are also used to build positive concept target sets for Concept-RL.

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
- `--chem_ace_max_2d_dim`
- `--chem_ace_max_3dqm_dim`
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
- `explainability/chem_ace/semantics/*`
- `explainability/chem_ace/db/*`
- `explainability/chem_ace/analytics/queries.py`
- `entrypoints/chem_ace_demo.py`

## 13) Assumptions and known limitations

- RDKit is required for full Chem-ACE behavior.
- HDBSCAN is optional; pipeline degrades gracefully if missing.
- Final MIL integration currently uses feature-fusion patch embedding (`feature_projection`) rather than hook-based embedding strategies.
- Patch volume can still be large; use radius/cap/target knobs aggressively on big datasets.

## 14) TODO (advanced extensions)

- richer 3D pharmacophore grouping and spatial motifs
- direct MIL attention-region extraction as native patch type
- tighter coupling between concept clusters and intervention policies
- active-learning hooks for concept-balanced data acquisition
