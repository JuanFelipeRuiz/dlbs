#!/bin/bash
#SBATCH -p performance
#SBATCH --job-name=dlbs
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=2:00:00
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
CONFIG="configs/baseline_no_aug.yaml"

echo "=== RUN ==="
echo "Config: ${CONFIG}"
echo

# CUDA_CHANNEL="${CUDA_CHANNEL:-cu128}"
$PYTHON_BIN -m venv .venv
source .venv/bin/activate


python infer.py --model /mnt/nas05/clusterdata01/home2/juan.ruizlopez/Repos/dlbs/runs/segment/yolo-seg-baseline-no-aug24/weights/best.pt\
                --image  /mnt/nas05/data01/slg-q1/dlbs/yolo_dataset/images/val/munster_000167_000019_leftImg8bit.png \

echo
echo "Finished at: $(date)"
