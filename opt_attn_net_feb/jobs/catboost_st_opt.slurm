#!/bin/bash
#SBATCH --job-name=catboost_st_opt
#SBATCH --export=ALL,MODULESHOME='',MODULEPATH=''
#SBATCH --partition=qGPU
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -eo pipefail
ulimit -n 4096 || true

FAMILY_NAME="catboost_st"

source /sw/target/generic/app/anaconda/anaconda3-2024.10/etc/profile.d/conda.sh
conda activate vf_conda_env

unset LD_PRELOAD || true
if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

echo "[INFO] CONDA_PREFIX=${CONDA_PREFIX:-<unset>}"
echo "[INFO] LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}"

if [[ -f "${CONDA_PREFIX}/lib/libstdc++.so.6" ]]; then
  if ! strings "${CONDA_PREFIX}/lib/libstdc++.so.6" | grep -q "GLIBCXX_3.4.26"; then
    echo "[ERROR] ${CONDA_PREFIX}/lib/libstdc++.so.6 does not provide GLIBCXX_3.4.26"
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
    sys.exit(3)
PY

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

REPO_DIR="${REPO_DIR:-/scratch/vfastovskii/opt_attn_net_feb/blend_preopt}"
SCRIPT="${SCRIPT:-${REPO_DIR}/opt_net_fast.py}"
PKGDIR="${PKGDIR:-${REPO_DIR}/opt_attn_net_feb}"
DATA_DIR="${DATA_DIR:-${REPO_DIR}}"
LABELS="${LABELS:-${DATA_DIR}/master_table_labels_final_modelling_ready_1401_with_cv_split.csv}"
FEAT2D_SCALED="${FEAT2D_SCALED:-${DATA_DIR}/scaled_2d.csv}"
FEAT2D_RAW="${FEAT2D_RAW:-${DATA_DIR}/raw_2d.csv}"
FEAT3D_SCALED="${FEAT3D_SCALED:-${DATA_DIR}/scaled_3d.csv}"
FEAT3D_QM="${FEAT3D_QM:-${DATA_DIR}/scaled_3d_quantum.csv}"

RUN_HPO="${RUN_HPO:-1}"
HPO_ONLY="${HPO_ONLY:-0}"
TRIALS="${TRIALS:-50}"
PRUNER_WARMUP_STEPS="${PRUNER_WARMUP_STEPS:-90}"
MAX_EPOCHS="${MAX_EPOCHS:-150}"
PATIENCE="${PATIENCE:-15}"
SEED="${SEED:-0}"
PRECISION="${PRECISION:-16-mixed}"
NUM_WORKERS="${NUM_WORKERS:-6}"
CPU_WORKERS="${CPU_WORKERS:-8}"
PIN_MEMORY="${PIN_MEMORY:-0}"
LEADERBOARD_SPLIT="${LEADERBOARD_SPLIT:-leaderboard}"
CALIBRATION_METHOD="${CALIBRATION_METHOD:-platt}"
SKIP_FAMILY_EXPLAINABILITY="${SKIP_FAMILY_EXPLAINABILITY:-1}"
SKIP_FAMILY_CALIBRATION="${SKIP_FAMILY_CALIBRATION:-1}"
SKIP_FAMILY_BLENDING="${SKIP_FAMILY_BLENDING:-1}"
CATBOOST_HPO_PARALLEL_TASKS="${CATBOOST_HPO_PARALLEL_TASKS:-4}"
BEST_PARAMS_DIR="${BEST_PARAMS_DIR:-}"
CATBOOST_TASK_PARAMS_JSONS="${CATBOOST_TASK_PARAMS_JSONS:-}"
CATBOOST_FINAL_ITERATIONS_JSONS="${CATBOOST_FINAL_ITERATIONS_JSONS:-}"

WORKDIR="${SLURM_TMPDIR:-/tmp}/${SLURM_JOB_ID}"
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

echo "[INFO] FAMILY_NAME=$FAMILY_NAME"
echo "[INFO] RUN_HPO=$RUN_HPO | HPO_ONLY=$HPO_ONLY | TRIALS=$TRIALS"
echo "[INFO] SKIP_FAMILY_EXPLAINABILITY=$SKIP_FAMILY_EXPLAINABILITY | SKIP_FAMILY_CALIBRATION=$SKIP_FAMILY_CALIBRATION | SKIP_FAMILY_BLENDING=$SKIP_FAMILY_BLENDING"

[[ -f "$SCRIPT" ]] || { echo "[ERROR] Missing script: $SCRIPT"; exit 2; }
[[ -d "$PKGDIR" ]] || { echo "[ERROR] Missing package dir: $PKGDIR"; exit 2; }
[[ -f "$LABELS" ]] || { echo "[ERROR] Missing labels: $LABELS"; exit 2; }
[[ -f "$FEAT2D_SCALED" ]] || { echo "[ERROR] Missing 2D features: $FEAT2D_SCALED"; exit 2; }
[[ -f "$FEAT2D_RAW" ]] || { echo "[ERROR] Missing raw 2D features: $FEAT2D_RAW"; exit 2; }
[[ -f "$FEAT3D_SCALED" ]] || { echo "[ERROR] Missing 3D features: $FEAT3D_SCALED"; exit 2; }
[[ -f "$FEAT3D_QM" ]] || { echo "[ERROR] Missing 3D QM features: $FEAT3D_QM"; exit 2; }
if [[ "$RUN_HPO" != "1" ]]; then
  [[ -n "$BEST_PARAMS_DIR" ]] || { echo "[ERROR] RUN_HPO=0 requires BEST_PARAMS_DIR"; exit 2; }
fi
if [[ -n "$CATBOOST_TASK_PARAMS_JSONS" ]]; then
  read -r -a CATBOOST_TASK_JSON_ARR <<< "$CATBOOST_TASK_PARAMS_JSONS"
  for _p in "${CATBOOST_TASK_JSON_ARR[@]}"; do
    [[ -f "$_p" ]] || { echo "[ERROR] Missing CATBOOST task params JSON: $_p"; exit 2; }
  done
fi
if [[ -n "$CATBOOST_FINAL_ITERATIONS_JSONS" ]]; then
  read -r -a CATBOOST_ITER_JSON_ARR <<< "$CATBOOST_FINAL_ITERATIONS_JSONS"
  for _p in "${CATBOOST_ITER_JSON_ARR[@]}"; do
    [[ -f "$_p" ]] || { echo "[ERROR] Missing CATBOOST final iterations JSON: $_p"; exit 2; }
  done
fi

rsync -av --exclude="__pycache__" --exclude="*.pyc" --exclude="*.pyo" --exclude=".git" "$PKGDIR" "$WORKDIR/"
cp -v "$SCRIPT" "$WORKDIR/"
cp -v "$LABELS" "$WORKDIR/"
cp -v "$FEAT2D_SCALED" "$WORKDIR/"
cp -v "$FEAT2D_RAW" "$WORKDIR/"
cp -v "$FEAT3D_SCALED" "$WORKDIR/"
cp -v "$FEAT3D_QM" "$WORKDIR/"
if [[ -n "$CATBOOST_TASK_PARAMS_JSONS" ]]; then
  for _p in "${CATBOOST_TASK_JSON_ARR[@]}"; do cp -v "$_p" "$WORKDIR/"; done
fi
if [[ -n "$CATBOOST_FINAL_ITERATIONS_JSONS" ]]; then
  for _p in "${CATBOOST_ITER_JSON_ARR[@]}"; do cp -v "$_p" "$WORKDIR/"; done
fi

export PYTHONPATH="$WORKDIR:${PYTHONPATH:-}"

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

if [[ "$PIN_MEMORY" == "1" ]]; then CMD+=(--pin_memory); fi
if [[ "$SKIP_FAMILY_EXPLAINABILITY" == "1" ]]; then CMD+=(--skip_family_explainability); fi
if [[ "$SKIP_FAMILY_CALIBRATION" == "1" ]]; then CMD+=(--skip_family_calibration); fi
if [[ "$SKIP_FAMILY_BLENDING" == "1" ]]; then CMD+=(--skip_family_blending); fi
if [[ "$RUN_HPO" == "1" ]]; then
  CMD+=(--run_hpo --trials "$TRIALS" --pruner_warmup_steps "$PRUNER_WARMUP_STEPS")
  if [[ "$HPO_ONLY" == "1" ]]; then CMD+=(--hpo_only); fi
else
  CMD+=(--best_params_dir "$BEST_PARAMS_DIR")
fi
if [[ -n "$CATBOOST_TASK_PARAMS_JSONS" ]]; then
  _task_json_args=()
  for _p in "${CATBOOST_TASK_JSON_ARR[@]}"; do _task_json_args+=("$WORKDIR/$(basename "$_p")"); done
  CMD+=(--catboost_task_params_jsons "${_task_json_args[@]}")
fi
if [[ -n "$CATBOOST_FINAL_ITERATIONS_JSONS" ]]; then
  _iter_json_args=()
  for _p in "${CATBOOST_ITER_JSON_ARR[@]}"; do _iter_json_args+=("$WORKDIR/$(basename "$_p")"); done
  CMD+=(--catboost_final_iterations_jsons "${_iter_json_args[@]}")
fi

printf '[INFO] CMD: %q ' "${CMD[@]}"; echo
"${CMD[@]}"

echo "[INFO] Copy results back..."
rsync -av "$STUDY_DIR/" "$OUTDIR/"
echo "[INFO] Done. Results in: $OUTDIR"
