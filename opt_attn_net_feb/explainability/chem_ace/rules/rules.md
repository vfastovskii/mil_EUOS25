# Rules System Specification (Chem-ACE)

This document is the canonical, implementation-level summary of all rule logic currently used in Chem-ACE semantic tagging and concept naming.

## 1. Scope

This document covers:

1. Functional-group rules (SMARTS + RDKit fragment-counter based rules)
2. SMARTS-RX reactivity rules
3. Naming rules for concept labels
4. Runtime dataset-driven rule augmentation from `curated_SMILES`
5. Exact formulas, thresholds, and fallback behavior

This document does not cover clustering, CAV/TCAV math, or Lambda-Vol dynamics except where they directly consume rule outputs.

## 2. Rule Artifacts and Current Sizes

Primary files in repository:

1. `explainability/chem_ace/rules/default_functional_group_rules.json`
2. `explainability/chem_ace/rules/smartsrx.json`
3. `explainability/chem_ace/rules/default_naming_rules.json`
4. `explainability/chem_ace/rules/rdkit_fragment_rules.py`
5. `explainability/chem_ace/rules/generate_fragment_rules_from_labels.py`

Current bundled rule counts:

1. Functional-group rules: 52
2. SMARTS-RX rules: 44
3. Naming rules: 76

## 3. Functional-Group Rules

### 3.1 Data model

Implemented in `FunctionalGroupRule` (`explainability/chem_ace/semantics/taggers.py`):

1. `tag: str` (required)
2. `smarts: str = ""` (optional)
3. `rdkit_fragment: str = ""` (optional)
4. `min_patch_rate: float = 0.03`
5. `confidence: float = 0.8`
6. `provenance: str = "functional_smarts_tagger"`

Validation/load rule:

1. `tag` must be non-empty
2. At least one of `smarts` or `rdkit_fragment` must be non-empty
3. Otherwise row is skipped

### 3.2 Source JSON schema

Bundled file currently contains SMARTS-only rows (keys observed in file):

1. `tag`
2. `smarts`
3. `min_patch_rate`
4. `confidence`

Runtime-generated rows can additionally contain:

1. `rdkit_fragment`
2. `provenance`
3. `source_prevalence`
4. `source_hit_molecules`
5. `source_total_count`

### 3.3 Runtime loading and compilation

In `SemanticTagger.__init__`:

1. Load functional rules from `SemanticTaggingConfig.functional_rules_path` if set; otherwise default file.
2. Compile SMARTS patterns for rules with non-empty `smarts`.
3. Resolve RDKit fragment functions for rules with non-empty `rdkit_fragment` by `getattr(rdkit.Chem.Fragments, name)`.

Failure behavior:

1. Missing file: warning, empty functional rule set.
2. Invalid SMARTS row: warning, row skipped.
3. Unknown RDKit fragment function name: warning, rule remains but fragment callable is unavailable.
4. RDKit unavailable: SMARTS compilation and RDKit-fragment backend are disabled.

### 3.4 Descriptor computation behavior

During concept descriptor extraction (`_compute_descriptors`):

1. For each valid patch, a molecule-level cache is used for SMARTS matches (`functional_cache`).
2. SMARTS match contributes only if matched atoms overlap patch atom set.
3. For RDKit fragment rules, molecule-level fragment counters are computed once per molecule (`functional_fragment_cache`).
4. If fragment count for a rule is `> 0`, each patch from that molecule receives:
   1. `functional_patch_presence[tag] += 1`
   2. `functional_match_hits[tag] += max(1, round(fragment_count))`

Important nuance:

1. SMARTS logic is atom-overlap patch-aware.
2. RDKit fragment-counter logic is molecule-level (not atom-localized), then propagated to patches from that molecule.

Derived descriptor fields:

1. `functional_group_patch_rate[tag] = patch_presence[tag] / n_valid_patches`
2. `functional_group_match_hits[tag] = integer hit count`

### 3.5 Tag emission rule

In `_functional_group_tags`:

1. A rule emits a tag only if `functional_group_patch_rate[tag] >= min_patch_rate`.
2. Tag confidence is:
   `max(rule.confidence, min(0.99, 0.55 + 0.9 * patch_rate))`
3. Provenance is copied from rule `provenance`.

## 4. SMARTS-RX Rules

### 4.1 Data model

Implemented in `SmartsRxRule` (`explainability/chem_ace/semantics/taggers.py`):

1. `tag: str` (required)
2. `smarts: str` (required)
3. `role: str = ""` (optional)
4. `min_patch_rate: float = 0.02`
5. `confidence: float = 0.82`
6. `provenance: str = "smarts_rx_tagger"`

### 4.2 Source JSON schema

Supported SMARTS-RX input schemas:

1. Canonical schema:
   - top-level: `{ "rules": [...] }`
   - keys per row:
     1. `tag`
     2. `smarts`
     3. `role`
     4. `min_patch_rate`
     5. `confidence`

2. Generated SMARTS-RX schema:
   - top-level: `{ "data": [...] }`
   - keys per row:
     1. `category`
     2. `subcategory`
     3. `specific_type`
     4. `smarts`

When `data` schema is used, loader auto-derives:

1. `tag = rx_<specific_type|subcategory|category>` (normalized)
2. `role = <category>` (normalized)
3. `min_patch_rate = 0.02` if missing
4. `confidence = 0.82` if missing
5. `provenance = "smarts_rx_tagger"` if missing

### 4.3 Runtime behavior

Pattern compilation:

1. Load from `SemanticTaggingConfig.smarts_rx_rules_path` if set; else default bundled file.
2. Default bundled file is `smartsrx.json` (legacy fallback: `default_smarts_rx_rules.json`).
3. Compile SMARTS patterns.
4. Invalid SMARTS rows are skipped with warning.
5. Duplicate derived tags are auto-disambiguated with suffixes (`_2`, `_3`, ...).

Descriptor outputs:

1. `smarts_rx_patch_rate`
2. `smarts_rx_match_hits`
3. `smarts_rx_role_rate`

Tag emission:

1. Rule tag emits if `smarts_rx_patch_rate[tag] >= min_patch_rate`.
2. Rule tag confidence:
   `max(rule.confidence, min(0.99, 0.60 + 0.9 * rate))`
3. Role tag emits if `role_rate >= 0.02`.
4. Role tag name:
   1. if role already starts with `rx_role_`, keep it
   2. else prefix `rx_role_`
5. Role tag confidence:
   `min(0.99, 0.55 + 0.8 * role_rate)`
6. Role tag provenance: `smarts_rx_role_tagger`

## 5. Naming Rules

### 5.1 Data model and source

`NamingRule` fields (`explainability/chem_ace/semantics/naming.py`):

1. `label`
2. `required_tags`
3. `forbidden_tags` (optional)

Bundled file:

1. `explainability/chem_ace/rules/default_naming_rules.json`

### 5.2 Selection logic

`choose_label(tags, registry, descriptor_rank)`:

1. Iterate rules in file order.
2. First rule where all `required_tags` are present and all `forbidden_tags` are absent wins.
3. If no rule matches, fallback to descriptor-based label:
   1. take first up to 3 items from `descriptor_rank`
   2. output `"{x1}, {x2}, {x3} motif"` (shorter if fewer descriptors)
4. If descriptor rank is empty, label is `unlabeled molecular motif`.

Determinism:

1. Rule order in JSON is semantically significant.
2. Earlier rules have higher precedence.

## 6. Tagger-Level Heuristic Rules (Code-Defined)

These are not JSON-based but are part of the active rule system:

1. Charge tags (`anionic`, `cationic`, `zwitterionic-like`, `neutral polar`, `neutral nonpolar`)
2. Conjugation tags (`aromatic pi-system`, `extended conjugation`, `isolated unsaturation`, `aliphatic`, `heteroaromatic`)
3. Geometry tags (`planar`, `non-planar`, `twisted`, `rigid`, `flexible`)
4. Pharmacophore tags (`HBD`, `HBA`, `aromatic centroid`, `cationic center`, `anionic center`, `HBD/HBA pair`)
5. QM tags (frontier orbital, gap, dipole, polarizability, hardness/softness, electrophile/nucleophile-like profiles, charge-transfer, Fukui, ESP)
6. Optional Open Babel tags (`lipophilic`, `high polar surface area`, `high refractivity`)

Config thresholds from `SemanticTaggingConfig`:

1. `charge_threshold_formal = 1`
2. `aromatic_fraction_threshold = 0.35`
3. `conjugation_size_threshold = 6`
4. `planarity_rmsd_threshold = 0.25`
5. `qm_min_vectors_for_tagging = 8`
6. `qm_z_threshold = 0.50`
7. `qm_strong_z_threshold = 1.00`
8. `use_smarts_rx = True`
9. `use_openbabel_descriptors = True`

Post-processing:

1. All tags are deduplicated by tag string.
2. If duplicates exist, highest-confidence instance is retained.

## 7. Dataset-Driven RDKit Fragment Rule Augmentation

### 7.1 Runtime integration point

Integrated in:

1. `training/explainability_runtime.py::_prepare_dataset_functional_rules`
2. Called inside `prepare_chem_ace_bundle`

Data source:

1. Entire labels dataframe column `curated_SMILES` (or configured `curated_smiles_col`)
2. Pipeline operation:
   1. drop null
   2. strip whitespace
   3. drop empty
   4. deduplicate by SMILES string

Runtime generation thresholds (current hardcoded values):

1. `min_count = 10`
2. `min_prevalence = 0.0002`

Runtime outputs:

1. `<chem_ace_output_dir>/rules_autogen/default_functional_group_rules.dataset.json`
2. `<chem_ace_output_dir>/rules_autogen/functional_group_fragment_stats.json`

Runtime metadata stored in merged JSON under `rdkit_fragment_generation`:

1. `n_smiles_seen`
2. `n_smiles_valid`
3. `n_fragment_functions`
4. `n_generated_rules`
5. `min_count`
6. `min_prevalence`
7. `smiles_col`
8. `n_unique_smiles`
9. `source = "prepare_chem_ace_bundle"`

### 7.2 Merge semantics

`merge_rules(base_rules, generated_rules)`:

1. Merge key is `tag`.
2. If generated tag exists in base, generated row overwrites base row.
3. If generated tag is new, append.
4. Base ordering is preserved except for appended new tags.

### 7.3 Injection into semantic tagging

The generated merged path is wired into:

1. `ChemACEConfig.semantics = SemanticTaggingConfig(functional_rules_path=<generated path>)`

Effect:

1. Current run tags concepts using dataset-augmented functional rules.

## 8. RDKit Fragment Rule Generator Details

Core module:

1. `explainability/chem_ace/rules/rdkit_fragment_rules.py`

CLI wrapper:

1. `explainability/chem_ace/rules/generate_fragment_rules_from_labels.py`

### 8.1 RDKit fragment enumeration

1. Enumerate callables in `rdkit.Chem.Fragments` where name starts with `fr_`.
2. Each function is evaluated per valid molecule from SMILES.

### 8.2 Rule generation formulas

For fragment function `fr_xxx`:

1. tag:
   `fg_rdkit_{slug(xxx)}`
2. prevalence:
   `hit_molecules / n_valid_smiles`
3. include only if:
   1. `hit_molecules >= min_count`
   2. `prevalence >= min_prevalence`
4. generated confidence:
   `min(0.95, 0.65 + 0.45 * min(1.0, prevalence / 0.2))`
5. generated `min_patch_rate`:
   `clamp(0.5 * prevalence, 0.01, 0.20)`
6. provenance:
   `rdkit_fragment_tagger`

### 8.3 CLI defaults

Manual script defaults:

1. `--smiles_col curated_SMILES`
2. `--min_count 25`
3. `--min_prevalence 0.001`
4. `--max_molecules 0` (no cap)
5. `--output_json` defaults to bundled functional rules file path

### 8.4 RDKit dependency behavior

1. If RDKit is missing, generator raises clear `RuntimeError`.
2. In tagger runtime, missing RDKit disables SMARTS and fragment backends gracefully (empty/disabled behavior).

## 9. Rule Priority and Precedence Summary

End-to-end precedence order:

1. Configured explicit rule paths (if provided) override bundled defaults.
2. In final pipeline run, dataset-autogen functional rules override bundled functional rules by tag.
3. Functional and SMARTS-RX tag thresholds gate which tags are emitted.
4. Deduplication keeps highest confidence per tag.
5. Naming registry rule order decides first-match label.
6. Fallback descriptor naming applies only if no naming rule matches.

## 10. Logging and Debugging Signals

Relevant runtime events:

1. `explainability.chem_ace.functional_rules.no_smiles`
2. `explainability.chem_ace.functional_rules.generation_failed`
3. `explainability.chem_ace.functional_rules.base_load_failed`
4. `explainability.chem_ace.functional_rules.generated`

Useful inspection files after run:

1. `rules_autogen/default_functional_group_rules.dataset.json`
2. `rules_autogen/functional_group_fragment_stats.json`
3. Concept-level tag evidence JSON in DB/exported artifacts

## 11. Practical Maintenance Workflow

When updating rule coverage:

1. Edit bundled rule files in `explainability/chem_ace/rules/`.
2. Keep tag names stable when possible to preserve downstream naming compatibility.
3. Add new naming rules near top if they should have stronger precedence.
4. Run generator script on latest labels SMILES to refresh RDKit fragment rules.
5. Validate with a short Chem-ACE run and inspect:
   1. generated rule counts
   2. high-frequency emitted tags
   3. concept labels for expected chemistry semantics

Recommended sanity checks:

1. Ensure no empty `tag`.
2. Ensure each functional row has at least one of `smarts` or `rdkit_fragment`.
3. Check `min_patch_rate` is in sensible range for prevalence.
4. Confirm critical naming labels are not shadowed by broader earlier rules.

## 12. Current Known Limitations

1. RDKit fragment-counter rules are molecule-level counters, not atom-localized SMARTS matches.
2. Therefore fragment-based tags can be less patch-specific than SMARTS overlap-based tags.
3. Runtime autogen thresholds are currently hardcoded in `prepare_chem_ace_bundle` (`10`, `0.0002`).
4. Bundled functional JSON currently ships SMARTS-only rows; RDKit-fragment rows are generated at runtime/manual generation.

## 13. Quick Commands

Manual generation from labels:

```bash
python -m explainability.chem_ace.rules.generate_fragment_rules_from_labels \
  --labels_csv /path/to/master_table_labels_final_modelling_ready_1401_with_cv_split.csv \
  --smiles_col curated_SMILES \
  --output_json /path/to/default_functional_group_rules.json \
  --stats_json /path/to/functional_group_fragment_stats.json
```

Syntax check:

```bash
python -m py_compile \
  explainability/chem_ace/rules/rdkit_fragment_rules.py \
  explainability/chem_ace/rules/generate_fragment_rules_from_labels.py \
  explainability/chem_ace/semantics/taggers.py \
  training/explainability_runtime.py
```
