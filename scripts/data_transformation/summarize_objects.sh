#!/bin/bash
#SBATCH -p performance
#SBATCH --job-name=dlbs_objects
#SBATCH --cpus-per-task=5
#SBATCH --mem=30G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
PYTHON_BIN=$(which python3)

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

echo "=== RUN ==="
echo "Starting to create an overview of the objects from a yolo dataset"
echo


$PYTHON_BIN -m venv .venv
source .venv/bin/activate


python -m dlbs.summarizers.object_summarizer --dataset_yaml_path  /mnt/nas05/data01/slg-q1/dlbs/yolo_dataset/data.yaml  --out data/objects.csv

echo
echo "Finished at: $(date)"
