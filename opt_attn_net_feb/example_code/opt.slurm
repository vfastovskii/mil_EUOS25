#!/bin/bash
#SBATCH --job-name=mil_hpo
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

# Reduce chance of "too many open files" on some clusters.
ulimit -n 4096 || true

# -------------------- environment --------------------
# Adapt this section to your cluster.
# source /path/to/miniconda3/etc/profile.d/conda.sh
# conda activate your_env

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MKL_DYNAMIC=FALSE
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# -------------------- paths --------------------
# Project checkout (contains opt_net_fast.py + package code).
REPO_PARENT="/path/to/mil_explainability_2026"
PROJECT_DIR="${REPO_PARENT}/opt_attn_net_feb"

DATA_DIR="/path/to/data"
LABELS="${DATA_DIR}/master_table_labels_final_modelling_ready_1401_with_cv_split.csv"
FEAT2D_SCALED="${DATA_DIR}/scaled_2d.csv"
FEAT3D_SCALED="${DATA_DIR}/scaled_3d.csv"
FEAT3D_QM="${DATA_DIR}/scaled_3d_quantum.csv"

OUTDIR="/path/to/experiments/study_${SLURM_JOB_ID}"

# -------------------- node-local scratch --------------------
WORKDIR="${SLURM_TMPDIR:-/tmp}/${SLURM_JOB_ID}"
STUDY_TMP="${WORKDIR}/study"
mkdir -p "$WORKDIR" "$STUDY_TMP" "$OUTDIR"

echo "[INFO] Workdir: $WORKDIR"
echo "[INFO] Output dir: $OUTDIR"

echo "[INFO] Copy inputs to node-local scratch..."
cp -v "$LABELS"        "$WORKDIR/"
cp -v "$FEAT2D_SCALED" "$WORKDIR/"
cp -v "$FEAT3D_SCALED" "$WORKDIR/"
cp -v "$FEAT3D_QM"     "$WORKDIR/"

# -------------------- run --------------------
echo "[INFO] Running MIL HPO via project entrypoint (opt_net_fast.py)..."
echo "[INFO] CPUs=${SLURM_CPUS_PER_TASK} | GPU=${CUDA_VISIBLE_DEVICES}"

cd "$PROJECT_DIR"
python3 "$PROJECT_DIR/opt_net_fast.py" \
  --labels "$WORKDIR/$(basename "$LABELS")" \
  --feat2d_scaled "$WORKDIR/$(basename "$FEAT2D_SCALED")" \
  --feat3d_scaled "$WORKDIR/$(basename "$FEAT3D_SCALED")" \
  --feat3d_qm_scaled "$WORKDIR/$(basename "$FEAT3D_QM")" \
  --study_dir "$STUDY_TMP" \
  --use_splits train \
  --trials 50 \
  --max_epochs 110 \
  --patience 10 \
  --seed 0 \
  --nn_accelerator gpu \
  --nn_devices 1 \
  --precision 16-mixed \
  --num_workers 23

# -------------------- copy back --------------------
echo "[INFO] Copy results back..."
cp -rv "$STUDY_TMP"/* "$OUTDIR"/

echo "[INFO] Done. Results in: $OUTDIR"
