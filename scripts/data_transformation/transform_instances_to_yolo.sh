#!/bin/bash
#SBATCH -p performance
#SBATCH --job-name=dlbs_inst2yolo
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Batch-convert Cityscapes *_gtFine_instanceIds.png -> YOLO instance-segmentation
# .txt labels for ALL images (the "Weg B" path explained in
# notebooks/0_explore_data_structure.ipynb). The notebook only demonstrates the
# transformation on a single example; this script does the full dataset.

set -euo pipefail
PYTHON_BIN=$(which python3)

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

# Paths 
DATA_ROOT="${DATA_ROOT:-/mnt/nas05/data01/slg-q1/dlbs/gtFine_trainvaltest/gtFine}"
OUT_DIR="${OUT_DIR:-data/yolo_annotations_instances}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"

mkdir -p "${OUT_DIR}"

echo "=== RUN ==="
echo "Converting instanceIds -> YOLO"
echo "DATA_ROOT: ${DATA_ROOT}"
echo "OUT_DIR:   ${OUT_DIR}"
echo "WORKERS:   ${WORKERS}"
echo

$PYTHON_BIN -m venv .venv
source .venv/bin/activate


for SPLIT in train val; do
    echo "--- split: ${SPLIT} ---"
    python -m dlbs.transform_data.instancepng_to_yolo \
        "${DATA_ROOT}/${SPLIT}" \
        -o "${OUT_DIR}" \
        -w "${WORKERS}"
done

echo
echo "Finished at: $(date)"
