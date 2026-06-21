#!/bin/bash
#SBATCH -p p4500
#SBATCH --job-name=dlbs_sweep_best
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

nvidia-smi

set -euo pipefail
PYTHON_BIN=$(which python3)

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

# ---- (Optional) Performance/Determinismus ----
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export WANDB_API_KEY="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"

# ---- Train starten ----
CONFIG="configs/sweep_best_imb.yaml"

echo "=== RUN ==="
echo "Config: ${CONFIG}"
echo

# CUDA_CHANNEL="${CUDA_CHANNEL:-cu128}"
$PYTHON_BIN -m venv .venv
source .venv/bin/activate


# Seed 0 ist bereits aus dem Sweep vorhanden.
python dlbs/train.py --cfg "${CONFIG}" --test-split test \
  --seed 1 \
  --name yolo-seg-sweep-imb-s1

echo
echo "Finished at: $(date)"
