# Run on Slurm

This guide shows how to run the MIL HPO/training pipeline as a Slurm job
using the project entrypoint script:

- `opt_net_fast.py`
- Canonical maintained Slurm wrapper in this repo: `run_mil_hpo_debye.slurm`

## 1) Expected inputs

The pipeline requires these CSV files:

- labels: `--labels`
- 2D features: `--feat2d_scaled`
- 3D geometry features: `--feat3d_scaled`
- 3D QM features: `--feat3d_qm_scaled`

Optional semantic-only raw tables (recommended for physically meaningful Chem-ACE tags):

- raw 3D geometry: `--feat3d_raw`
- raw 3D QM: `--feat3d_qm_raw`

If raw tables are provided, both must be provided.

Output folder is set by `--study_dir`.

## 2) Example `sbatch` script

Create a file such as `run_mil_hpo.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=mil_hpo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# Absolute paths (edit these)
REPO_PARENT="/path/to/mil_explainability_2026"
PROJECT_DIR="${REPO_PARENT}/opt_attn_net_feb"
DATA_DIR="/path/to/data"
OUT_DIR="/path/to/experiments/mil_hpo_${SLURM_JOB_ID}"

mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${OUT_DIR}"

cd "${PROJECT_DIR}"

# If needed, activate your environment
# module load cuda/12.1
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate your_env

python "${PROJECT_DIR}/opt_net_fast.py" \
  --labels "${DATA_DIR}/labels.csv" \
  --feat2d_scaled "${DATA_DIR}/feat2d_scaled.csv" \
  --feat3d_scaled "${DATA_DIR}/feat3d_scaled.csv" \
  --feat3d_qm_scaled "${DATA_DIR}/feat3d_qm_scaled.csv" \
  --feat3d_raw "${DATA_DIR}/feat3d_raw.csv" \
  --feat3d_qm_raw "${DATA_DIR}/feat3d_qm_raw.csv" \
  --study_dir "${OUT_DIR}" \
  --id_col ID \
  --conf_col conf_id \
  --split_col split \
  --fold_col cv_fold \
  --use_splits train \
  --max_epochs 150 \
  --patience 20 \
  --trials 50 \
  --seed 0 \
  --nn_accelerator gpu \
  --nn_devices 1 \
  --precision 16-mixed \
  --num_workers -1 \
  --cpu_workers -1 \
  --pin_memory \
  --leaderboard_split leaderboard \
  --attn_out "${OUT_DIR}/leaderboard_attn.csv"
```

## 3) Submit and monitor

Submit:

```bash
sbatch run_mil_hpo.slurm
```

Check queue:

```bash
squeue -u "$USER"
```

Inspect logs:

```bash
tail -f /path/to/mil_explainability_2026/opt_attn_net_feb/logs/mil_hpo_<jobid>.out
```

## 4) Notes

- On Debye, prefer using the repository script:
  - `sbatch /Users/vfastovskii/Desktop/mil_explainability_2026/opt_attn_net_feb/run_mil_hpo_debye.slurm`
  - it already handles node-local copy, explainability toggles, RL ablation, and optional raw semantic tables.
- Keep `--num_workers -1` to auto-tune workers from `SLURM_CPUS_PER_TASK`.
- Keep `--cpu_workers -1` to auto-allocate CPU worker budget for CPU-bound stages (Chem-ACE patch generation/tagging/feature embedding and torch CPU thread pools).
- If your cluster uses different GPU resource syntax, adapt `#SBATCH --gres=gpu:1`.
- Final stage always runs after HPO:
  - train on `split == train`
  - validate on `split == leaderboard_split`
- Attention export path can be overridden with `--attn_out ...`.
- If your conformer SDF stores conformer id in SDF record name, pass:
  - `--chem_ace_sdf_conf_id_prop _Name`
