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
EXP_DIR="/scratch/vfastovskii/opt_attn_net_feb"

SCRIPT="${EXP_DIR}/opt_net_fast.py"
PKGDIR="${EXP_DIR}/opt_attn_net_feb"
BEST_PARAMS_JSON="${EXP_DIR}/best_params_279756.json"

LABELS="${EXP_DIR}/master_table_labels_final_modelling_ready_1401_with_cv_split.csv"
FEAT2D_SCALED="${EXP_DIR}/scaled_2d.csv"
FEAT3D_SCALED="${EXP_DIR}/scaled_3d.csv"
FEAT3D_QM="${EXP_DIR}/scaled_3d_quantum.csv"

# Conformer SDF input (override at submit time via --export=ALL,CONF_SDF=/path/file.sdf)
CONF_SDF="${CONF_SDF:-${EXP_DIR}/test_mols_batch.sdf}"
# Property name in SDF storing conf identifier.
# Your SDF uses record name as conformer identifier.
CONF_ID_PROP="${CONF_ID_PROP:-_Name}"
# Chem-ACE patch-budget controls (override via --export=ALL,...)
CHEMACE_MAX_IDS="${CHEMACE_MAX_IDS:-0}"
CHEMACE_LOCAL_RADII="${CHEMACE_LOCAL_RADII:-1}"
CHEMACE_PATCH_CAP_PER_MOL="${CHEMACE_PATCH_CAP_PER_MOL:-0}"
CHEMACE_TARGET_TOTAL_PATCHES="${CHEMACE_TARGET_TOTAL_PATCHES:-1200000}"

# -------------------- node-local scratch --------------------
WORKDIR="${SLURM_TMPDIR:-/tmp}/${SLURM_JOB_ID}"
STUDY_DIR="${WORKDIR}/study_${SLURM_JOB_ID}"
OUTDIR="${EXP_DIR}/study_${SLURM_JOB_ID}"

mkdir -p "$WORKDIR" "$STUDY_DIR" "$OUTDIR"

echo "[INFO] JobID: ${SLURM_JOB_ID}"
echo "[INFO] Host:  $(hostname)"
echo "[INFO] Workdir: $WORKDIR"
echo "[INFO] Study dir (node-local): $STUDY_DIR"
echo "[INFO] Output dir: $OUTDIR"
echo "[INFO] CPUs=${SLURM_CPUS_PER_TASK} | CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] Conformer SDF: $CONF_SDF"
echo "[INFO] SDF conf-id prop: $CONF_ID_PROP"
echo "[INFO] Chem-ACE max IDs: $CHEMACE_MAX_IDS"
echo "[INFO] Chem-ACE local radii: $CHEMACE_LOCAL_RADII"
echo "[INFO] Chem-ACE patch cap per mol: $CHEMACE_PATCH_CAP_PER_MOL"
echo "[INFO] Chem-ACE target total patches: $CHEMACE_TARGET_TOTAL_PATCHES"

# optional: GPU info
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

# fail fast if layout is not as expected
test -f "$SCRIPT" || { echo "[ERROR] Missing $SCRIPT"; exit 2; }
test -d "$PKGDIR" || { echo "[ERROR] Missing package dir $PKGDIR"; exit 2; }
test -f "$BEST_PARAMS_JSON" || { echo "[ERROR] Missing best params file $BEST_PARAMS_JSON"; exit 2; }
test -f "$CONF_SDF" || { echo "[ERROR] Missing conformer SDF $CONF_SDF"; exit 2; }

echo "[INFO] Copy code (excluding cache) + inputs to node-local scratch..."

rsync -av \
  --exclude="__pycache__" \
  --exclude="*.pyc" \
  --exclude="*.pyo" \
  --exclude=".git" \
  "$PKGDIR" "$WORKDIR/"

cp -v "$SCRIPT" "$WORKDIR/"
cp -v "$BEST_PARAMS_JSON" "$WORKDIR/"
cp -v "$LABELS" "$WORKDIR/"
cp -v "$FEAT2D_SCALED" "$WORKDIR/"
cp -v "$FEAT3D_SCALED" "$WORKDIR/"
cp -v "$FEAT3D_QM" "$WORKDIR/"
cp -v "$CONF_SDF" "$WORKDIR/"

# make python see the copied package (repo root == $WORKDIR)
export PYTHONPATH="$WORKDIR:${PYTHONPATH:-}"

# sanity check
python3 -c "import opt_attn_net_feb; print('[INFO] opt_attn_net_feb import OK from', opt_attn_net_feb.__file__)"

# -------------------- run --------------------
echo "[INFO] Running final MIL training + all explainability from pre-optimized params..."

python3 "$WORKDIR/opt_net_fast.py" \
  --labels "$WORKDIR/$(basename "$LABELS")" \
  --feat2d_scaled "$WORKDIR/$(basename "$FEAT2D_SCALED")" \
  --feat3d_scaled "$WORKDIR/$(basename "$FEAT3D_SCALED")" \
  --feat3d_qm_scaled "$WORKDIR/$(basename "$FEAT3D_QM")" \
  --study_dir "$STUDY_DIR" \
  --use_splits train \
  --do_mil \
  --best_params_json "$WORKDIR/$(basename "$BEST_PARAMS_JSON")" \
  --run_chem_ace \
  --run_lambda_vol \
  --lambda_vol_run_ricci \
  --run_concept_rl \
  --no-chem_ace_persist_patch_embeddings \
  --curated_smiles_col curated_SMILES \
  --chem_ace_conformer_sdf "$WORKDIR/$(basename "$CONF_SDF")" \
  --chem_ace_sdf_conf_id_prop "$CONF_ID_PROP" \
  --chem_ace_max_ids "$CHEMACE_MAX_IDS" \
  --chem_ace_local_radii $CHEMACE_LOCAL_RADII \
  --chem_ace_patch_cap_per_mol "$CHEMACE_PATCH_CAP_PER_MOL" \
  --chem_ace_target_total_patches "$CHEMACE_TARGET_TOTAL_PATCHES" \
  --leaderboard_split leaderboard \
  --attn_out "$STUDY_DIR/leaderboard_attn.csv" \
  --max_epochs 150 \
  --patience 10 \
  --seed 0 \
  --nn_accelerator gpu \
  --nn_devices 1 \
  --precision 16-mixed \
  --num_workers 23 \
  --cpu_workers 18

# -------------------- copy back --------------------
echo "[INFO] Copy results back..."
rsync -av "$STUDY_DIR/" "$OUTDIR/"

echo "[INFO] Done. Results in: $OUTDIR"
