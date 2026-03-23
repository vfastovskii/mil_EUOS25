#!/bin/bash
#SBATCH --job-name=2d3d40t
#SBATCH --export=ALL,MODULESHOME='',MODULEPATH=''
#SBATCH --partition=qGPUr
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -eo pipefail
ulimit -n 4096 || true

FAMILY_NAME="mt_2d3d"

# -------------------- conda --------------------
source /sw/target/generic/app/anaconda/anaconda3-2024.10/etc/profile.d/conda.sh
conda activate vf_conda_env

# -------------------- C++ runtime compatibility --------------------
unset LD_PRELOAD || true
if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

python3 - <<'PY'
import sys
try:
    import torch
    print(f"[INFO] torch import OK version={torch.__version__} cuda_available={torch.cuda.is_available()}")
except Exception as exc:
    print(f"[ERROR] torch import failed: {exc!r}")
    sys.exit(3)
PY

# -------------------- python/thread hygiene --------------------
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128,garbage_collection_threshold:0.8}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MKL_DYNAMIC=FALSE
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# -------------------- paths --------------------
REPO_DIR="${REPO_DIR:-/scratch/vfastovskii/opt_attn_net_feb/blend_preopt}"
SCRIPT="${SCRIPT:-${REPO_DIR}/opt_net_fast.py}"
PKGDIR="${PKGDIR:-${REPO_DIR}/opt_attn_net_feb}"

DATA_DIR="${DATA_DIR:-${REPO_DIR}}"
LABELS="${LABELS:-${DATA_DIR}/master_table_labels_final_modelling_ready_1401_with_cv_split.csv}"
FEAT2D_SCALED="${FEAT2D_SCALED:-${DATA_DIR}/scaled_2d.csv}"
FEAT2D_RAW="${FEAT2D_RAW:-${DATA_DIR}/raw_2d.csv}"
FEAT3D_SCALED="${FEAT3D_SCALED:-${DATA_DIR}/scaled_3d.csv}"
FEAT3D_QM="${FEAT3D_QM:-${DATA_DIR}/scaled_3d_quantum.csv}"

# -------------------- mode / knobs --------------------
RUN_HPO="${RUN_HPO:-1}"
HPO_ONLY="${HPO_ONLY:-0}"
TRIALS="${TRIALS:-40}"
PRUNER_WARMUP_STEPS="${PRUNER_WARMUP_STEPS:-90}"

MAX_EPOCHS="${MAX_EPOCHS:-150}"
PATIENCE="${PATIENCE:-15}"
SEED="${SEED:-0}"
PRECISION="${PRECISION:-16-mixed}"
NUM_WORKERS="${NUM_WORKERS:-14}"
CPU_WORKERS="${CPU_WORKERS:-14}"
PIN_MEMORY="${PIN_MEMORY:-0}"

LEADERBOARD_SPLIT="${LEADERBOARD_SPLIT:-leaderboard}"
CALIBRATION_METHOD="${CALIBRATION_METHOD:-platt}"
SKIP_FAMILY_EXPLAINABILITY="${SKIP_FAMILY_EXPLAINABILITY:-1}"
SKIP_FAMILY_CALIBRATION="${SKIP_FAMILY_CALIBRATION:-1}"
SKIP_FAMILY_BLENDING="${SKIP_FAMILY_BLENDING:-1}"

BEST_PARAMS_DIR="${BEST_PARAMS_DIR:-}"
MT_2D3D_PARAMS_JSON="${MT_2D3D_PARAMS_JSON:-}"
MT_2D3D_FINAL_EPOCHS_JSON="${MT_2D3D_FINAL_EPOCHS_JSON:-}"

# -------------------- node-local scratch --------------------
WORKDIR="/home-local/${USER}/tmp/${SLURM_JOB_ID}"
STUDY_DIR="${WORKDIR}/study_${SLURM_JOB_ID}"
OUTDIR="${OUTDIR:-${REPO_DIR}/study_${SLURM_JOB_ID}}"
mkdir -p "$WORKDIR" "$STUDY_DIR" "$OUTDIR"

export FAMILY_SUITE_BEST_PARAMS_MIRROR_DIR="${FAMILY_SUITE_BEST_PARAMS_MIRROR_DIR:-${OUTDIR}/best_params}"
mkdir -p "$FAMILY_SUITE_BEST_PARAMS_MIRROR_DIR"

copy_back_partial() {
  echo "[INFO] EXIT trap: syncing partial results back..."
  rsync -av "$STUDY_DIR/" "$OUTDIR/" || true
}
trap copy_back_partial EXIT

echo "[INFO] JobID: ${SLURM_JOB_ID}"
echo "[INFO] Host: $(hostname)"
echo "[INFO] Workdir: $WORKDIR"
echo "[INFO] Study dir (node-local): $STUDY_DIR"
echo "[INFO] Output dir: $OUTDIR"
echo "[INFO] Best params mirror dir: $FAMILY_SUITE_BEST_PARAMS_MIRROR_DIR"
echo "[INFO] CPUs=${SLURM_CPUS_PER_TASK} | CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# -------------------- fail fast --------------------
[[ -f "$SCRIPT" ]] || { echo "[ERROR] Missing script: $SCRIPT"; exit 2; }
[[ -d "$PKGDIR" ]] || { echo "[ERROR] Missing package dir: $PKGDIR"; exit 2; }

for _f in "$LABELS" "$FEAT2D_SCALED" "$FEAT2D_RAW" "$FEAT3D_SCALED" "$FEAT3D_QM"; do
  [[ -f "$_f" ]] || { echo "[ERROR] Missing file: $_f"; exit 2; }
done

if [[ "$RUN_HPO" != "1" ]]; then
  [[ -n "$BEST_PARAMS_DIR" ]] || { echo "[ERROR] RUN_HPO=0 requires BEST_PARAMS_DIR"; exit 2; }
fi

if [[ -n "$MT_2D3D_PARAMS_JSON" ]]; then
  [[ -f "$MT_2D3D_PARAMS_JSON" ]] || { echo "[ERROR] Missing MT_2D3D_PARAMS_JSON: $MT_2D3D_PARAMS_JSON"; exit 2; }
fi

if [[ -n "$MT_2D3D_FINAL_EPOCHS_JSON" ]]; then
  [[ -f "$MT_2D3D_FINAL_EPOCHS_JSON" ]] || { echo "[ERROR] Missing MT_2D3D_FINAL_EPOCHS_JSON: $MT_2D3D_FINAL_EPOCHS_JSON"; exit 2; }
fi

# -------------------- copy code + inputs to node-local scratch --------------------
echo "[INFO] Copy code + inputs to node-local scratch..."

# same logic pattern as the working Blend13 job:
# copy the package tree into WORKDIR, then copy the entry script separately
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

if [[ -n "$MT_2D3D_PARAMS_JSON" ]]; then
  cp -v "$MT_2D3D_PARAMS_JSON" "$WORKDIR/"
fi

if [[ -n "$MT_2D3D_FINAL_EPOCHS_JSON" ]]; then
  cp -v "$MT_2D3D_FINAL_EPOCHS_JSON" "$WORKDIR/"
fi

export PYTHONPATH="$WORKDIR:${PYTHONPATH:-}"

# same validation style, but stronger:
python3 - <<'PY'
import importlib.util
import sys

mods = [
    "opt_attn_net_feb",
    "opt_attn_net_feb.entrypoints",
    "opt_attn_net_feb.entrypoints.hpo_pipeline",
]

for m in mods:
    spec = importlib.util.find_spec(m)
    print(f"[INFO] find_spec({m}) = {spec}")
    if spec is None:
        print(f"[ERROR] Missing import target: {m}")
        sys.exit(4)

import opt_attn_net_feb
print("[INFO] opt_attn_net_feb import OK from", opt_attn_net_feb.__file__)
PY

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
  --model_families "$FAMILY_NAME"
  --calibration_method "$CALIBRATION_METHOD"
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

if [[ "$SKIP_FAMILY_EXPLAINABILITY" == "1" ]]; then
  CMD+=(--skip_family_explainability)
fi

if [[ "$SKIP_FAMILY_CALIBRATION" == "1" ]]; then
  CMD+=(--skip_family_calibration)
fi

if [[ "$SKIP_FAMILY_BLENDING" == "1" ]]; then
  CMD+=(--skip_family_blending)
fi

if [[ "$RUN_HPO" == "1" ]]; then
  CMD+=(--run_hpo --trials "$TRIALS" --pruner_warmup_steps "$PRUNER_WARMUP_STEPS")
  if [[ "$HPO_ONLY" == "1" ]]; then
    CMD+=(--hpo_only)
  fi
else
  CMD+=(--best_params_dir "$BEST_PARAMS_DIR")
fi

if [[ -n "$MT_2D3D_PARAMS_JSON" ]]; then
  CMD+=(--mt_2d3d_params_json "$WORKDIR/$(basename "$MT_2D3D_PARAMS_JSON")")
fi

if [[ -n "$MT_2D3D_FINAL_EPOCHS_JSON" ]]; then
  CMD+=(--mt_2d3d_final_epochs_json "$WORKDIR/$(basename "$MT_2D3D_FINAL_EPOCHS_JSON")")
fi

printf '[INFO] CMD: %q ' "${CMD[@]}"
echo
"${CMD[@]}"

echo "[INFO] Copy results back..."
rsync -av "$STUDY_DIR/" "$OUTDIR/"
echo "[INFO] Done. Results in: $OUTDIR"