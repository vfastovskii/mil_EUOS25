#!/bin/bash
#SBATCH --job-name=Blend
#SBATCH --export=ALL,MODULESHOME='',MODULEPATH=''   # Required on Debye
#SBATCH --partition=qGPUr
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -eo pipefail
ulimit -n 4096 || true

# -------------------- conda --------------------
source /sw/target/generic/app/anaconda/anaconda3-2024.10/etc/profile.d/conda.sh
conda activate vf_conda_env

# -------------------- C++ runtime compatibility --------------------
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

# -------------------- python/thread hygiene --------------------
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MKL_DYNAMIC=FALSE
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# -------------------- paths --------------------
# Keep the same layout convention as run_mil_hpo_debye.slurm
REPO_DIR="${REPO_DIR:-/scratch/vfastovskii/opt_attn_net_feb}"
SCRIPT="${SCRIPT:-${REPO_DIR}/opt_net_fast.py}"
PKGDIR="${PKGDIR:-${REPO_DIR}/blend}"

DATA_DIR="${DATA_DIR:-${REPO_DIR}}"
LABELS="${LABELS:-${DATA_DIR}/master_table_labels_final_modelling_ready_1401_with_cv_split.csv}"
FEAT2D_SCALED="${FEAT2D_SCALED:-${DATA_DIR}/scaled_2d.csv}"
FEAT2D_RAW="${FEAT2D_RAW:-${DATA_DIR}/raw_2d.csv}"
FEAT3D_SCALED="${FEAT3D_SCALED:-${DATA_DIR}/scaled_3d.csv}"
FEAT3D_QM="${FEAT3D_QM:-${DATA_DIR}/scaled_3d_quantum.csv}"

# -------------------- mode / knobs --------------------
# Default: run HPO + final leaderboard evaluation.
RUN_HPO="${RUN_HPO:-1}"
# Optional: HPO only, no final leaderboard stage.
HPO_ONLY="${HPO_ONLY:-0}"
TRIALS="${TRIALS:-50}"
PRUNER_WARMUP_STEPS="${PRUNER_WARMUP_STEPS:-90}"

MAX_EPOCHS="${MAX_EPOCHS:-150}"
PATIENCE="${PATIENCE:-15}"
SEED="${SEED:-0}"
PRECISION="${PRECISION:-16-mixed}"
NUM_WORKERS="${NUM_WORKERS:-16}"
CPU_WORKERS="${CPU_WORKERS:-64}"
PIN_MEMORY="${PIN_MEMORY:-0}"

LEADERBOARD_SPLIT="${LEADERBOARD_SPLIT:-leaderboard}"

# Family-suite controls
MODEL_FAMILIES="${MODEL_FAMILIES:-catboost_st mt_2d mt_2d3d mt_3d}"
CALIBRATION_METHOD="${CALIBRATION_METHOD:-platt}"  # platt|isotonic|temperature
CATBOOST_HPO_PARALLEL_TASKS="${CATBOOST_HPO_PARALLEL_TASKS:-4}"
BEST_PARAMS_DIR="${BEST_PARAMS_DIR:-}"

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
echo "[INFO] RUN_HPO=$RUN_HPO | HPO_ONLY=$HPO_ONLY | PRUNER_WARMUP_STEPS=$PRUNER_WARMUP_STEPS"
echo "[INFO] MODEL_FAMILIES=$MODEL_FAMILIES | CALIBRATION_METHOD=$CALIBRATION_METHOD | CATBOOST_HPO_PARALLEL_TASKS=$CATBOOST_HPO_PARALLEL_TASKS"
echo "[INFO] 2D sources: scaled=$FEAT2D_SCALED | catboost_raw=$FEAT2D_RAW"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

# fail fast
[[ -f "$SCRIPT" ]] || { echo "[ERROR] Missing script: $SCRIPT"; exit 2; }
[[ -d "$PKGDIR" ]] || { echo "[ERROR] Missing package dir: $PKGDIR"; exit 2; }
[[ -f "$LABELS" ]] || { echo "[ERROR] Missing labels: $LABELS"; exit 2; }
[[ -f "$FEAT2D_SCALED" ]] || { echo "[ERROR] Missing 2D features: $FEAT2D_SCALED"; exit 2; }
[[ -f "$FEAT2D_RAW" ]] || { echo "[ERROR] Missing raw 2D features for CatBoost: $FEAT2D_RAW"; exit 2; }
[[ -f "$FEAT3D_SCALED" ]] || { echo "[ERROR] Missing 3D features: $FEAT3D_SCALED"; exit 2; }
[[ -f "$FEAT3D_QM" ]] || { echo "[ERROR] Missing 3D QM features: $FEAT3D_QM"; exit 2; }
if [[ "$RUN_HPO" != "1" ]]; then
  [[ -n "$BEST_PARAMS_DIR" ]] || { echo "[ERROR] RUN_HPO=0 requires BEST_PARAMS_DIR"; exit 2; }
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
cp -v "$FEAT2D_RAW" "$WORKDIR/"
cp -v "$FEAT3D_SCALED" "$WORKDIR/"
cp -v "$FEAT3D_QM" "$WORKDIR/"

export PYTHONPATH="$WORKDIR:${PYTHONPATH:-}"
python3 -c "import opt_attn_net_feb; print('[INFO] opt_attn_net_feb import OK from', opt_attn_net_feb.__file__)"

read -r -a MODEL_FAMILIES_ARR <<< "$MODEL_FAMILIES"

echo "[INFO] Running family-suite pipeline WITHOUT explainability..."
CMD=(
  python3 "$WORKDIR/opt_net_fast.py"
  --labels "$WORKDIR/$(basename "$LABELS")"
  --feat2d_scaled "$WORKDIR/$(basename "$FEAT2D_SCALED")"
  --feat2d_raw "$WORKDIR/$(basename "$FEAT2D_RAW")"
  --feat3d_scaled "$WORKDIR/$(basename "$FEAT3D_SCALED")"
  --feat3d_qm_scaled "$WORKDIR/$(basename "$FEAT3D_QM")"
  --study_dir "$STUDY_DIR"
  --use_splits train
  --leaderboard_split "$LEADERBOARD_SPLIT"
  --run_family_suite
  --model_families "${MODEL_FAMILIES_ARR[@]}"
  --calibration_method "$CALIBRATION_METHOD"
  --catboost_hpo_parallel_tasks "$CATBOOST_HPO_PARALLEL_TASKS"
  --max_epochs "$MAX_EPOCHS"
  --patience "$PATIENCE"
  --seed "$SEED"
  --nn_accelerator gpu
  --nn_devices 1
  --precision "$PRECISION"
  --num_workers "$NUM_WORKERS"
  --cpu_workers "$CPU_WORKERS"
)

if [[ "$PIN_MEMORY" == "1" ]]; then
  CMD+=(--pin_memory)
fi

if [[ "$RUN_HPO" == "1" ]]; then
  CMD+=(--run_hpo --trials "$TRIALS" --pruner_warmup_steps "$PRUNER_WARMUP_STEPS")
  if [[ "$HPO_ONLY" == "1" ]]; then
    CMD+=(--hpo_only)
  fi
else
  CMD+=(--best_params_dir "$BEST_PARAMS_DIR")
fi

printf '[INFO] CMD: %q ' "${CMD[@]}"; echo
"${CMD[@]}"

echo "[INFO] Copy results back..."
rsync -av "$STUDY_DIR/" "$OUTDIR/"
echo "[INFO] Done. Results in: $OUTDIR"
