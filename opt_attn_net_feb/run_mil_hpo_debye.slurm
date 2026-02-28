#!/bin/bash
#SBATCH --job-name=milExpl
#SBATCH --export=ALL,MODULESHOME='',MODULEPATH=''   # Required on Debye
#SBATCH --partition=qGPU
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -eo pipefail
ulimit -n 4096 || true

# -------------------- conda --------------------
source /sw/target/generic/app/anaconda/anaconda3-2024.10/etc/profile.d/conda.sh
conda activate vf_conda_env

# -------------------- C++ runtime compatibility --------------------
# Ensure torch uses libstdc++ from the conda env (fixes GLIBCXX_3.4.26 errors).
unset LD_PRELOAD || true
if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi
echo "[INFO] CONDA_PREFIX=${CONDA_PREFIX:-<unset>}"
echo "[INFO] LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}"

if [[ -f "${CONDA_PREFIX}/lib/libstdc++.so.6" ]]; then
  if strings "${CONDA_PREFIX}/lib/libstdc++.so.6" | grep -q "GLIBCXX_3.4.26"; then
    echo "[INFO] libstdc++ check: GLIBCXX_3.4.26 found in conda env."
  else
    echo "[ERROR] ${CONDA_PREFIX}/lib/libstdc++.so.6 does not provide GLIBCXX_3.4.26"
    echo "[HINT] Run: conda install -n vf_conda_env -c conda-forge 'libstdcxx-ng>=12' 'libgcc-ng>=12'"
    exit 3
  fi
else
  echo "[WARN] ${CONDA_PREFIX}/lib/libstdc++.so.6 not found; continuing to python preflight."
fi

python3 - <<'PY'
import sys
try:
    import torch
    print(f"[INFO] torch import OK version={torch.__version__} cuda_available={torch.cuda.is_available()}")
except Exception as exc:
    print(f"[ERROR] torch import failed: {exc!r}")
    print("[HINT] Try: conda install -n vf_conda_env -c conda-forge 'libstdcxx-ng>=12' 'libgcc-ng>=12'")
    sys.exit(3)
PY

# -------------------- python hygiene --------------------
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# -------------------- threading hygiene --------------------
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MKL_DYNAMIC=FALSE

# CUDA is assigned by Slurm; keep this for logging only
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# -------------------- paths --------------------
# Repository root with wrapper entrypoint + package dir.
REPO_DIR="${REPO_DIR:-/scratch/vfastovskii/opt_attn_net_feb}"
SCRIPT="${SCRIPT:-${REPO_DIR}/opt_net_fast.py}"
PKGDIR="${PKGDIR:-${REPO_DIR}/opt_attn_net_feb}"

# Data location (defaults to REPO_DIR layout used in this project).
DATA_DIR="${DATA_DIR:-${REPO_DIR}}"
LABELS="${LABELS:-${DATA_DIR}/master_table_labels_final_modelling_ready_1401_with_cv_split.csv}"
FEAT2D_SCALED="${FEAT2D_SCALED:-${DATA_DIR}/scaled_2d.csv}"
FEAT3D_SCALED="${FEAT3D_SCALED:-${DATA_DIR}/scaled_3d.csv}"
FEAT3D_QM="${FEAT3D_QM:-${DATA_DIR}/scaled_3d_quantum.csv}"
# Optional raw 3D tables for Chem-ACE semantic tagging (physical units).
FEAT3D_RAW="${FEAT3D_RAW:-}"
FEAT3D_QM_RAW="${FEAT3D_QM_RAW:-}"

# Best params JSON used when RUN_HPO=0.
BEST_PARAMS_JSON="${BEST_PARAMS_JSON:-${REPO_DIR}/best_params_279756.json}"

# Precomputed conformer SDF for 3D patching (one conformer per entry).
CONF_SDF="${CONF_SDF:-${REPO_DIR}/full_data_set_confs.sdf}"
# Your SDF stores conformer ID in _Name.
CONF_ID_PROP="${CONF_ID_PROP:-_Name}"

# -------------------- mode / knobs --------------------
# RUN_HPO=1: CV optimization + final fit/eval.
# RUN_HPO=0: skip optimization and use BEST_PARAMS_JSON.
RUN_HPO="${RUN_HPO:-0}"
TRIALS="${TRIALS:-35}"

MAX_EPOCHS="${MAX_EPOCHS:-150}"
PATIENCE="${PATIENCE:-10}"
SEED="${SEED:-0}"
PRECISION="${PRECISION:-16-mixed}"
NUM_WORKERS="${NUM_WORKERS:-23}"
CPU_WORKERS="${CPU_WORKERS:-23}"
PIN_MEMORY="${PIN_MEMORY:-0}"

LEADERBOARD_SPLIT="${LEADERBOARD_SPLIT:-leaderboard}"
CURATED_SMILES_COL="${CURATED_SMILES_COL:-curated_SMILES}"

# Explainability toggles
RUN_CHEM_ACE="${RUN_CHEM_ACE:-1}"
RUN_LAMBDA_VOL="${RUN_LAMBDA_VOL:-1}"
RUN_CONCEPT_RL="${RUN_CONCEPT_RL:-1}"
RUN_CONCEPT_RL_ABLATION="${RUN_CONCEPT_RL_ABLATION:-1}"
LAMBDA_VOL_RUN_RICCI="${LAMBDA_VOL_RUN_RICCI:-1}"
LAMBDA_VOL_RICCI_USE_FLOW_AS_COUPLING="${LAMBDA_VOL_RICCI_USE_FLOW_AS_COUPLING:-1}"

# Chem-ACE controls
CHEMACE_OUTPUT_DIR="${CHEMACE_OUTPUT_DIR:-}"
CHEMACE_DB_URI="${CHEMACE_DB_URI:-}"
CHEMACE_MAX_IDS="${CHEMACE_MAX_IDS:-0}"
CHEMACE_MAX_CONFS_PER_ID="${CHEMACE_MAX_CONFS_PER_ID:-0}"
CHEMACE_LOCAL_RADII="${CHEMACE_LOCAL_RADII:-1}"
CHEMACE_PATCH_CAP_PER_MOL="${CHEMACE_PATCH_CAP_PER_MOL:-0}"
CHEMACE_TARGET_TOTAL_PATCHES="${CHEMACE_TARGET_TOTAL_PATCHES:-1200000}"
# Hybrid embedding controls
CHEMACE_EMBED_DIM_2D="${CHEMACE_EMBED_DIM_2D:-64}"
CHEMACE_EMBED_DIM_3D_GEOM="${CHEMACE_EMBED_DIM_3D_GEOM:-64}"
CHEMACE_EMBED_DIM_3D_QM="${CHEMACE_EMBED_DIM_3D_QM:-64}"
CHEMACE_CONTEXT_DIM="${CHEMACE_CONTEXT_DIM:-16}"
CHEMACE_CONTEXT_ALPHA="${CHEMACE_CONTEXT_ALPHA:-0.20}"
CHEMACE_QM_GATING="${CHEMACE_QM_GATING:-1}"
# Deprecated compatibility knobs (ignored by hybrid runtime).
CHEMACE_MAX_2D_DIM="${CHEMACE_MAX_2D_DIM:-0}"
CHEMACE_MAX_3DQM_DIM="${CHEMACE_MAX_3DQM_DIM:-0}"
# <=0 means all discovered Chem-ACE concepts.
CHEMACE_TOP_CONCEPTS="${CHEMACE_TOP_CONCEPTS:-0}"
CHEMACE_INFER_MAX_DISTANCE="${CHEMACE_INFER_MAX_DISTANCE:--1.0}"
CHEMACE_PERSIST_PATCH_EMBEDDINGS="${CHEMACE_PERSIST_PATCH_EMBEDDINGS:-0}"

# Lambda-Vol controls
LAMBDA_VOL_OUTPUT_DIR="${LAMBDA_VOL_OUTPUT_DIR:-}"
LAMBDA_VOL_DB_URI="${LAMBDA_VOL_DB_URI:-}"
LAMBDA_VOL_LAYER_NAME="${LAMBDA_VOL_LAYER_NAME:-mixer_post_norm}"
# <=0 means all Chem-ACE concepts are evaluated by Lambda-Vol.
LAMBDA_VOL_TOP_CONCEPTS="${LAMBDA_VOL_TOP_CONCEPTS:-0}"
LAMBDA_VOL_MONITOR_MAX_SAMPLES="${LAMBDA_VOL_MONITOR_MAX_SAMPLES:-512}"
LAMBDA_VOL_TCAV_REPEATS="${LAMBDA_VOL_TCAV_REPEATS:-2}"
LAMBDA_VOL_RANDOM_COUNTEREXAMPLES="${LAMBDA_VOL_RANDOM_COUNTEREXAMPLES:-96}"
LAMBDA_VOL_MIN_CONCEPT_SAMPLES="${LAMBDA_VOL_MIN_CONCEPT_SAMPLES:-8}"
LAMBDA_VOL_RICCI_EDGE_KEEP_QUANTILE="${LAMBDA_VOL_RICCI_EDGE_KEEP_QUANTILE:-0.75}"
LAMBDA_VOL_RICCI_MIN_EDGE_WEIGHT="${LAMBDA_VOL_RICCI_MIN_EDGE_WEIGHT:-0.05}"
LAMBDA_VOL_RICCI_TOP_K_PER_NODE="${LAMBDA_VOL_RICCI_TOP_K_PER_NODE:-4}"
LAMBDA_VOL_RICCI_FLOW_STEPS="${LAMBDA_VOL_RICCI_FLOW_STEPS:-8}"
LAMBDA_VOL_RICCI_FLOW_STEP_SIZE="${LAMBDA_VOL_RICCI_FLOW_STEP_SIZE:-0.12}"
LAMBDA_VOL_RICCI_COUPLING_STRENGTH="${LAMBDA_VOL_RICCI_COUPLING_STRENGTH:-0.05}"

# Concept-RL controls
CONCEPT_RL_TOP_K_PER_TASK="${CONCEPT_RL_TOP_K_PER_TASK:-0}"   # <=0: all passing concepts
CONCEPT_RL_MIN_POS_COVERAGE="${CONCEPT_RL_MIN_POS_COVERAGE:-0.02}"
CONCEPT_RL_MIN_POS_HITS="${CONCEPT_RL_MIN_POS_HITS:-8}"
CONCEPT_RL_MIN_LIFT="${CONCEPT_RL_MIN_LIFT:-1.05}"
CONCEPT_RL_OVERLAP_WEIGHT="${CONCEPT_RL_OVERLAP_WEIGHT:-0.35}"
CONCEPT_RL_GLOBAL_WEIGHT="${CONCEPT_RL_GLOBAL_WEIGHT:-0.20}"
CONCEPT_RL_LIFT_WEIGHT="${CONCEPT_RL_LIFT_WEIGHT:-0.25}"
CONCEPT_RL_GENERAL_TOP_K="${CONCEPT_RL_GENERAL_TOP_K:-4}"
CONCEPT_RL_MIN_MULTI_ACTIVE_COUNT="${CONCEPT_RL_MIN_MULTI_ACTIVE_COUNT:-32}"
CONCEPT_RL_REQUIRE_CONF_SUPPORT="${CONCEPT_RL_REQUIRE_CONF_SUPPORT:-1}"
CONCEPT_RL_MIN_CONF_POS_COVERAGE="${CONCEPT_RL_MIN_CONF_POS_COVERAGE:-0.005}"
CONCEPT_RL_INIT_SCALE="${CONCEPT_RL_INIT_SCALE:-0.02}"
CONCEPT_RL_MAX_SCALE="${CONCEPT_RL_MAX_SCALE:-0.20}"
CONCEPT_RL_POLICY_LR="${CONCEPT_RL_POLICY_LR:-0.05}"
CONCEPT_RL_POLICY_SIGMA="${CONCEPT_RL_POLICY_SIGMA:-0.02}"
CONCEPT_RL_REWARD_ALIGNMENT_W="${CONCEPT_RL_REWARD_ALIGNMENT_W:-0.25}"
CONCEPT_RL_REWARD_MIN_AP_W="${CONCEPT_RL_REWARD_MIN_AP_W:-0.15}"
CONCEPT_RL_BASELINE_MOMENTUM="${CONCEPT_RL_BASELINE_MOMENTUM:-0.90}"
CONCEPT_RL_NEGATIVE_PENALTY="${CONCEPT_RL_NEGATIVE_PENALTY:-0.15}"

# Lambda-Vol and Concept-RL depend on Chem-ACE concepts.
if [[ "$RUN_LAMBDA_VOL" == "1" || "$RUN_CONCEPT_RL" == "1" || "$RUN_CONCEPT_RL_ABLATION" == "1" ]]; then
  RUN_CHEM_ACE=1
fi

# -------------------- node-local scratch --------------------
WORKDIR="${SLURM_TMPDIR:-/tmp}/${SLURM_JOB_ID}"
STUDY_DIR="${WORKDIR}/study_${SLURM_JOB_ID}"
OUTDIR="${OUTDIR:-${REPO_DIR}/study_${SLURM_JOB_ID}}"

mkdir -p "$WORKDIR" "$STUDY_DIR" "$OUTDIR"

echo "[INFO] JobID: ${SLURM_JOB_ID}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Workdir: $WORKDIR"
echo "[INFO] Study dir (node-local): $STUDY_DIR"
echo "[INFO] Output dir: $OUTDIR"
echo "[INFO] CPUs=${SLURM_CPUS_PER_TASK} | CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] RUN_HPO=$RUN_HPO | RUN_CHEM_ACE=$RUN_CHEM_ACE | RUN_LAMBDA_VOL=$RUN_LAMBDA_VOL | RUN_CONCEPT_RL=$RUN_CONCEPT_RL | RUN_CONCEPT_RL_ABLATION=$RUN_CONCEPT_RL_ABLATION"
echo "[INFO] Conformer SDF: $CONF_SDF | conf_id source: $CONF_ID_PROP"

command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

# fail fast
[[ -f "$SCRIPT" ]] || { echo "[ERROR] Missing script: $SCRIPT"; exit 2; }
[[ -d "$PKGDIR" ]] || { echo "[ERROR] Missing package dir: $PKGDIR"; exit 2; }
[[ -f "$LABELS" ]] || { echo "[ERROR] Missing labels: $LABELS"; exit 2; }
[[ -f "$FEAT2D_SCALED" ]] || { echo "[ERROR] Missing 2D features: $FEAT2D_SCALED"; exit 2; }
[[ -f "$FEAT3D_SCALED" ]] || { echo "[ERROR] Missing 3D features: $FEAT3D_SCALED"; exit 2; }
[[ -f "$FEAT3D_QM" ]] || { echo "[ERROR] Missing 3D QM features: $FEAT3D_QM"; exit 2; }
if [[ -n "$FEAT3D_RAW" || -n "$FEAT3D_QM_RAW" ]]; then
  [[ -n "$FEAT3D_RAW" && -n "$FEAT3D_QM_RAW" ]] || {
    echo "[ERROR] Provide both FEAT3D_RAW and FEAT3D_QM_RAW or neither."; exit 2;
  }
  [[ -f "$FEAT3D_RAW" ]] || { echo "[ERROR] Missing raw 3D features: $FEAT3D_RAW"; exit 2; }
  [[ -f "$FEAT3D_QM_RAW" ]] || { echo "[ERROR] Missing raw 3D QM features: $FEAT3D_QM_RAW"; exit 2; }
fi
if [[ "$RUN_HPO" != "1" ]]; then
  [[ -f "$BEST_PARAMS_JSON" ]] || { echo "[ERROR] Missing best params JSON: $BEST_PARAMS_JSON"; exit 2; }
fi
if [[ "$RUN_CHEM_ACE" == "1" ]]; then
  [[ -f "$CONF_SDF" ]] || { echo "[ERROR] Missing conformer SDF: $CONF_SDF"; exit 2; }
fi

echo "[INFO] Copy code + inputs to node-local scratch..."
rsync -av \
  --exclude="__pycache__" \
  --exclude="*.pyc" \
  --exclude="*.pyo" \
  --exclude=".git" \
  "$PKGDIR" "$WORKDIR/"

cp -v "$SCRIPT" "$WORKDIR/"
cp -v "$LABELS" "$WORKDIR/"
cp -v "$FEAT2D_SCALED" "$WORKDIR/"
cp -v "$FEAT3D_SCALED" "$WORKDIR/"
cp -v "$FEAT3D_QM" "$WORKDIR/"
if [[ -n "$FEAT3D_RAW" && -n "$FEAT3D_QM_RAW" ]]; then
  cp -v "$FEAT3D_RAW" "$WORKDIR/"
  cp -v "$FEAT3D_QM_RAW" "$WORKDIR/"
fi
if [[ "$RUN_HPO" != "1" ]]; then
  cp -v "$BEST_PARAMS_JSON" "$WORKDIR/"
fi
if [[ "$RUN_CHEM_ACE" == "1" ]]; then
  cp -v "$CONF_SDF" "$WORKDIR/"
fi

# Make copied package importable.
export PYTHONPATH="$WORKDIR:${PYTHONPATH:-}"
python3 -c "import opt_attn_net_feb; print('[INFO] opt_attn_net_feb import OK from', opt_attn_net_feb.__file__)"

# Parse radii into argv-friendly array (supports "1 2" and "1,2").
CHEMACE_LOCAL_RADII_CLEAN="${CHEMACE_LOCAL_RADII//,/ }"
read -r -a CHEMACE_RADII_ARR <<< "$CHEMACE_LOCAL_RADII_CLEAN"
if [[ ${#CHEMACE_RADII_ARR[@]} -eq 0 ]]; then
  CHEMACE_RADII_ARR=(1)
fi

echo "[INFO] Running pipeline..."
CMD=(
  python3 "$WORKDIR/opt_net_fast.py"
  --labels "$WORKDIR/$(basename "$LABELS")"
  --feat2d_scaled "$WORKDIR/$(basename "$FEAT2D_SCALED")"
  --feat3d_scaled "$WORKDIR/$(basename "$FEAT3D_SCALED")"
  --feat3d_qm_scaled "$WORKDIR/$(basename "$FEAT3D_QM")"
  --study_dir "$STUDY_DIR"
  --use_splits train
  --do_mil
  --export_leaderboard_attn
  --leaderboard_split "$LEADERBOARD_SPLIT"
  --attn_out "$STUDY_DIR/leaderboard_attn.csv"
  --max_epochs "$MAX_EPOCHS"
  --patience "$PATIENCE"
  --seed "$SEED"
  --nn_accelerator gpu
  --nn_devices 1
  --precision "$PRECISION"
  --num_workers "$NUM_WORKERS"
  --cpu_workers "$CPU_WORKERS"
)

if [[ -n "$FEAT3D_RAW" && -n "$FEAT3D_QM_RAW" ]]; then
  CMD+=(
    --feat3d_raw "$WORKDIR/$(basename "$FEAT3D_RAW")"
    --feat3d_qm_raw "$WORKDIR/$(basename "$FEAT3D_QM_RAW")"
  )
fi

if [[ "$PIN_MEMORY" == "1" ]]; then
  CMD+=(--pin_memory)
fi

if [[ "$RUN_HPO" == "1" ]]; then
  CMD+=(--run_hpo --trials "$TRIALS")
else
  CMD+=(--best_params_json "$WORKDIR/$(basename "$BEST_PARAMS_JSON")")
fi

if [[ "$RUN_CHEM_ACE" == "1" ]]; then
  CMD+=(
    --run_chem_ace
    --curated_smiles_col "$CURATED_SMILES_COL"
    --chem_ace_conformer_sdf "$WORKDIR/$(basename "$CONF_SDF")"
    --chem_ace_sdf_conf_id_prop "$CONF_ID_PROP"
    --chem_ace_max_ids "$CHEMACE_MAX_IDS"
    --chem_ace_max_confs_per_id "$CHEMACE_MAX_CONFS_PER_ID"
    --chem_ace_local_radii "${CHEMACE_RADII_ARR[@]}"
    --chem_ace_patch_cap_per_mol "$CHEMACE_PATCH_CAP_PER_MOL"
    --chem_ace_target_total_patches "$CHEMACE_TARGET_TOTAL_PATCHES"
    --chem_ace_embed_dim_2d "$CHEMACE_EMBED_DIM_2D"
    --chem_ace_embed_dim_3d_geom "$CHEMACE_EMBED_DIM_3D_GEOM"
    --chem_ace_embed_dim_3d_qm "$CHEMACE_EMBED_DIM_3D_QM"
    --chem_ace_context_dim "$CHEMACE_CONTEXT_DIM"
    --chem_ace_context_alpha "$CHEMACE_CONTEXT_ALPHA"
    --chem_ace_max_2d_dim "$CHEMACE_MAX_2D_DIM"
    --chem_ace_max_3dqm_dim "$CHEMACE_MAX_3DQM_DIM"
    --chem_ace_top_concepts "$CHEMACE_TOP_CONCEPTS"
    --chem_ace_infer_max_distance "$CHEMACE_INFER_MAX_DISTANCE"
  )
  if [[ "$CHEMACE_QM_GATING" == "1" ]]; then
    CMD+=(--chem_ace_qm_gating)
  else
    CMD+=(--no-chem_ace_qm_gating)
  fi
  if [[ "$CHEMACE_PERSIST_PATCH_EMBEDDINGS" == "1" ]]; then
    CMD+=(--chem_ace_persist_patch_embeddings)
  else
    CMD+=(--no-chem_ace_persist_patch_embeddings)
  fi
  if [[ -n "$CHEMACE_OUTPUT_DIR" ]]; then
    CMD+=(--chem_ace_output_dir "$CHEMACE_OUTPUT_DIR")
  fi
  if [[ -n "$CHEMACE_DB_URI" ]]; then
    CMD+=(--chem_ace_db_uri "$CHEMACE_DB_URI")
  fi
fi

if [[ "$RUN_LAMBDA_VOL" == "1" ]]; then
  CMD+=(
    --run_lambda_vol
    --lambda_vol_layer_name "$LAMBDA_VOL_LAYER_NAME"
    --lambda_vol_top_concepts "$LAMBDA_VOL_TOP_CONCEPTS"
    --lambda_vol_monitor_max_samples "$LAMBDA_VOL_MONITOR_MAX_SAMPLES"
    --lambda_vol_tcav_repeats "$LAMBDA_VOL_TCAV_REPEATS"
    --lambda_vol_random_counterexamples "$LAMBDA_VOL_RANDOM_COUNTEREXAMPLES"
    --lambda_vol_min_concept_samples "$LAMBDA_VOL_MIN_CONCEPT_SAMPLES"
    --lambda_vol_ricci_edge_keep_quantile "$LAMBDA_VOL_RICCI_EDGE_KEEP_QUANTILE"
    --lambda_vol_ricci_min_edge_weight "$LAMBDA_VOL_RICCI_MIN_EDGE_WEIGHT"
    --lambda_vol_ricci_top_k_per_node "$LAMBDA_VOL_RICCI_TOP_K_PER_NODE"
    --lambda_vol_ricci_flow_steps "$LAMBDA_VOL_RICCI_FLOW_STEPS"
    --lambda_vol_ricci_flow_step_size "$LAMBDA_VOL_RICCI_FLOW_STEP_SIZE"
    --lambda_vol_ricci_coupling_strength "$LAMBDA_VOL_RICCI_COUPLING_STRENGTH"
  )
  if [[ "$LAMBDA_VOL_RUN_RICCI" == "1" ]]; then
    CMD+=(--lambda_vol_run_ricci)
  else
    CMD+=(--no-lambda_vol_run_ricci)
  fi
  if [[ "$LAMBDA_VOL_RICCI_USE_FLOW_AS_COUPLING" == "1" ]]; then
    CMD+=(--lambda_vol_ricci_use_flow_as_coupling)
  else
    CMD+=(--no-lambda_vol_ricci_use_flow_as_coupling)
  fi
  if [[ -n "$LAMBDA_VOL_OUTPUT_DIR" ]]; then
    CMD+=(--lambda_vol_output_dir "$LAMBDA_VOL_OUTPUT_DIR")
  fi
  if [[ -n "$LAMBDA_VOL_DB_URI" ]]; then
    CMD+=(--lambda_vol_db_uri "$LAMBDA_VOL_DB_URI")
  fi
fi

if [[ "$RUN_CONCEPT_RL" == "1" || "$RUN_CONCEPT_RL_ABLATION" == "1" ]]; then
  CMD+=(
    --concept_rl_top_k_per_task "$CONCEPT_RL_TOP_K_PER_TASK"
    --concept_rl_min_pos_coverage "$CONCEPT_RL_MIN_POS_COVERAGE"
    --concept_rl_min_pos_hits "$CONCEPT_RL_MIN_POS_HITS"
    --concept_rl_min_lift "$CONCEPT_RL_MIN_LIFT"
    --concept_rl_overlap_weight "$CONCEPT_RL_OVERLAP_WEIGHT"
    --concept_rl_global_weight "$CONCEPT_RL_GLOBAL_WEIGHT"
    --concept_rl_lift_weight "$CONCEPT_RL_LIFT_WEIGHT"
    --concept_rl_general_top_k "$CONCEPT_RL_GENERAL_TOP_K"
    --concept_rl_min_multi_active_count "$CONCEPT_RL_MIN_MULTI_ACTIVE_COUNT"
    --concept_rl_min_conf_pos_coverage "$CONCEPT_RL_MIN_CONF_POS_COVERAGE"
    --concept_rl_init_scale "$CONCEPT_RL_INIT_SCALE"
    --concept_rl_max_scale "$CONCEPT_RL_MAX_SCALE"
    --concept_rl_policy_lr "$CONCEPT_RL_POLICY_LR"
    --concept_rl_policy_sigma "$CONCEPT_RL_POLICY_SIGMA"
    --concept_rl_reward_alignment_w "$CONCEPT_RL_REWARD_ALIGNMENT_W"
    --concept_rl_reward_min_ap_w "$CONCEPT_RL_REWARD_MIN_AP_W"
    --concept_rl_baseline_momentum "$CONCEPT_RL_BASELINE_MOMENTUM"
    --concept_rl_negative_penalty "$CONCEPT_RL_NEGATIVE_PENALTY"
  )
  if [[ "$CONCEPT_RL_REQUIRE_CONF_SUPPORT" == "1" ]]; then
    CMD+=(--concept_rl_require_conf_support)
  else
    CMD+=(--no-concept_rl_require_conf_support)
  fi
fi

if [[ "$RUN_CONCEPT_RL" == "1" ]]; then
  CMD+=(--run_concept_rl)
fi
if [[ "$RUN_CONCEPT_RL_ABLATION" == "1" ]]; then
  CMD+=(--run_concept_rl_ablation)
fi

printf '[INFO] CMD: %q ' "${CMD[@]}"; echo
"${CMD[@]}"

# -------------------- copy back --------------------
echo "[INFO] Copy results back..."
rsync -av "$STUDY_DIR/" "$OUTDIR/"

echo "[INFO] Done. Results in: $OUTDIR"
