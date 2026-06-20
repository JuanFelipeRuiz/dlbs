#!/bin/bash
#SBATCH -p p4500
#SBATCH --job-name=test-seeds
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err


set -euo pipefail
PYTHON_BIN=$(which python3)

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export WANDB_API_KEY="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"

$PYTHON_BIN -m venv .venv
source .venv/bin/activate

DATA="/mnt/nas05/data01/slg-q1/dlbs/yolo_dataset/data.yaml"
OBJECTS_CSV="data/objects.csv"
IMGSZ=960
RUNS_DIR="runs/segment"

# runs to test
RUN_NAMES=(
  "yolo-seg-img_size-s0"
  "yolo-seg-img_size-s1"
  "yolo-seg-img_size-s42"
)

for NAME in "${RUN_NAMES[@]}"; do
  echo "=== TEST ${NAME} ==="
  python -m dlbs.test_model \
    --model "${RUNS_DIR}/${NAME}/weights/best.pt" \
    --data "${DATA}" \
    --split test \
    --imgsz "${IMGSZ}" \
    --stratify \
    --objects-csv "${OBJECTS_CSV}" \
    --name "${NAME}-test" \
    --wandb-project runs-segment \
    --wandb-entity dlbs-jf
  echo
done

echo "Finished at: $(date)"
