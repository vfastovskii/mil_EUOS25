#!/bin/bash
#SBATCH --job-name=milOPT3
#SBATCH --export=ALL,MODULESHOME='',MODULEPATH=''   # Required on Debye

#SBATCH --partition=qGPU
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -eo pipefail
ulimit -n 4096 || true

# -------------------- conda --------------------
source /sw/target/generic/app/anaconda/anaconda3-2024.10/etc/profile.d/conda.sh
conda activate vf_conda_env

# -------------------- threading hygiene --------------------
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MKL_DYNAMIC=FALSE

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# -------------------- paths --------------------
EXP_DIR="/scratch/vfastovskii/opt_attn_net_feb"

SCRIPT="${EXP_DIR}/opt_net_fast.py"
PKGDIR="${EXP_DIR}/opt_attn_net_feb"   # <-- correct: package folder inside repo

LABELS="${EXP_DIR}/master_table_labels_final_modelling_ready_1401_with_cv_split.csv"
FEAT2D_SCALED="${EXP_DIR}/scaled_2d.csv"
FEAT3D_SCALED="${EXP_DIR}/scaled_3d.csv"
FEAT3D_QM="${EXP_DIR}/scaled_3d_quantum.csv"

# -------------------- node-local scratch --------------------
WORKDIR="${SLURM_TMPDIR:-/tmp}/${SLURM_JOB_ID}"
STUDY_DIR="${WORKDIR}/study_${SLURM_JOB_ID}"
OUTDIR="${EXP_DIR}/study_${SLURM_JOB_ID}"

mkdir -p "$WORKDIR" "$STUDY_DIR" "$OUTDIR"

echo "[INFO] Workdir: $WORKDIR"
echo "[INFO] Study dir (node-local): $STUDY_DIR"
echo "[INFO] Output dir: $OUTDIR"

# fail fast if layout is not as expected
test -f "$SCRIPT" || { echo "[ERROR] Missing $SCRIPT"; exit 2; }
test -d "$PKGDIR" || { echo "[ERROR] Missing package dir $PKGDIR"; exit 2; }

echo "[INFO] Copy code + inputs to node-local scratch..."
cp -v  "$SCRIPT" "$WORKDIR/"
cp -rv "$PKGDIR" "$WORKDIR/"          # copies to $WORKDIR/opt_attn_net_feb
cp -v  "$LABELS" "$WORKDIR/"
cp -v  "$FEAT2D_SCALED" "$WORKDIR/"
cp -v  "$FEAT3D_SCALED" "$WORKDIR/"
cp -v  "$FEAT3D_QM" "$WORKDIR/"

# make python see the copied package (repo root == $WORKDIR)
export PYTHONPATH="$WORKDIR:${PYTHONPATH:-}"

# sanity check
python3 -c "import opt_attn_net_feb; print('[INFO] opt_attn_net_feb import OK from', opt_attn_net_feb.__file__)"

# -------------------- run --------------------
echo "[INFO] Running MIL HPO only..."
echo "[INFO] CPUs=${SLURM_CPUS_PER_TASK} | GPU=${CUDA_VISIBLE_DEVICES}"

python3 "$WORKDIR/opt_net_fast.py" \
  --labels "$WORKDIR/$(basename "$LABELS")" \
  --feat2d_scaled "$WORKDIR/$(basename "$FEAT2D_SCALED")" \
  --feat3d_scaled "$WORKDIR/$(basename "$FEAT3D_SCALED")" \
  --feat3d_qm_scaled "$WORKDIR/$(basename "$FEAT3D_QM")" \
  --study_dir "$STUDY_DIR" \
  --use_splits train \
  --do_mil \
  --run_hpo \
  --trials_mil 50 \
  --max_epochs 150 \
  --patience 10 \
  --seed 0 \
  --nn_accelerator gpu \
  --nn_devices 1 \
  --precision 16-mixed \
  --num_workers 23

# -------------------- copy back --------------------
echo "[INFO] Copy results back..."
cp -rv "$STUDY_DIR"/* "$OUTDIR"/

echo "[INFO] Done. Results in: $OUTDIR"
