# ACE/TCAV Alignment Checklist

This file tracks how the current Chem-ACE implementation aligns with ACE/TCAV principles and where guardrails are enforced in code.

## 1) ACE-Style Requirements Mapping

### 1.1 Patch-centric concept candidates

Implemented.

- Patch generation is explicit (`local_subgraph`, `brics`, `murcko`, `pharm3d`).
- Train-only discovery scope is enforced before any leaderboard inference.
- Conformer-aware 3D patching runs only when conformers are available.

Code:

- `explainability/chem_ace/concepts/pipeline.py`
- `training/explainability_runtime.py`

### 1.2 Concept discovery in activation-relevant space

Partially aligned (engineering approximation in current runtime).

- Removed runtime use of `concat(full_2d, full_or_mean_3dqm, patch_desc)`.
- Added modality-separated hybrid spaces:
  - `2d`
  - `3d_geom`
  - `3d_qm`
- Each patch is assigned to exactly one modality space.
- Current discovery vectors are hybrid local+context feature embeddings, not direct bottleneck activations from model forward hooks.
- Strict activation-space patch embedders exist (`masked_input`, `node_pooling`) but are not the default runtime path in final pipeline.

Code:

- `training/explainability_runtime.py`
  - `_build_hybrid_patch_embeddings(...)`
  - `_discover_and_store_concepts_by_modality(...)`

### 1.3 Concept clustering / support / filtering

Implemented.

- Discovery runs per modality, then merges concepts with modality-qualified IDs.
- Scalable k-means defaults remain active for large sample counts.

Code:

- `explainability/chem_ace/concepts/clustering.py`
- `training/explainability_runtime.py`

### 1.4 Leakage-safe concept transfer to test scope

Implemented.

- Concepts are discovered on train scope only.
- Leaderboard scope uses nearest-centroid inference to frozen train concepts.
- No leaderboard-driven concept redefinition.

Code:

- `training/explainability_runtime.py`

## 2) TCAV Requirements Mapping

### 2.1 CAV training with random counterexamples

Implemented.

- Repeated random counterexample sets are used.
- Configurable repeats and counterexample counts.

Code:

- `explainability/chem_ace/cav/tcav.py`

### 2.2 Directional derivative scoring

Implemented.

- TCAV sign-rate and mean directional derivative are computed.
- Metrics are stored per `(task, concept, epoch, seed)`.

Code:

- `explainability/chem_ace/cav/tcav.py`
- `explainability/chem_ace/db/repository.py`

### 2.3 Significance protocol vs original TCAV

Partially aligned.

- Current implementation uses repeated random counterexample draws and reports per-repeat binomial p-values on sign-rate.
- Summary p-value is computed via pooled binomial test across repeats.
- It still does not implement the paper-style random-concept baseline hypothesis test (e.g., comparing against a bank of random concepts with statistical test over repeats).

Code:

- `explainability/chem_ace/cav/tcav.py`

### 2.4 Fused-layer TCAV for multimodal model

Implemented.

- TCAV monitoring is intentionally computed at `mixer_post_norm`.
- Concepts remain modality-specific while evaluation layer is shared/fused.

Code:

- `training/explainability_runtime.py`

## 3) Refactor Decisions (Current Contract)

- Old discovery embedding mode is removed from final runtime path.
- New embedding mode is `hybrid_local_context` by modality.
- Concept IDs are modality-qualified (`2d:...`, `3d_geom:...`, `3d_qm:...`).
- Exports include modality-aware concept fields.

## 4) Open Items

- Add optional strict ACE runtime mode that discovers concepts directly from model activations (`masked_input` / `node_pooling`) instead of hybrid engineered vectors.
- Add optional TCAV random-concept significance track to match original TCAV statistical protocol more closely.
