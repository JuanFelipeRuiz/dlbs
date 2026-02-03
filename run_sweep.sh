#!/bin/bash
#SBATCH -p performance
#SBATCH -w calc-g-008
#SBATCH --job-name=dlbs_wandb
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-4

set -euo pipefail

if [[ $# -ne 1 ]]; then
  exit 1
fi

RAW_SWEEP_ID="$1"

nvidia-smi

# Ensure working directory
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

# Environment
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# Optional but recommended on clusters
export WANDB_DIR="${SLURM_TMPDIR:-$PWD}/wandb"
mkdir -p "$WANDB_DIR"

# Activate venv (assumes it already exists)
source .venv/bin/activate

# One run per array task
srun wandb agent --count 1 "dlbs-jf/runs_segment/${RAW_SWEEP_ID}"

echo
echo "Finished at: $(date)"
