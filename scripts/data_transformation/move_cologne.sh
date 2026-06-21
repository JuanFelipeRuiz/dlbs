#!/bin/bash
#SBATCH -p performance
#SBATCH --job-name=move_cologne
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=0:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

# Usage:
#   ./move_cologne.sh [dataset_root]
#
# Example:
#   ./move_cologne.sh yolo_dataset
#
# This moves matching image/label pairs from:
#   <dataset_root>/images/train + <dataset_root>/labels/train
# to:
#   <dataset_root>/images/test  + <dataset_root>/labels/test

city_prefix="cologne"
dataset_root="${1:-yolo_dataset}"

images_train_dir="$dataset_root/images/train"
images_test_dir="$dataset_root/images/test"
labels_train_dir="$dataset_root/labels/train"
labels_test_dir="$dataset_root/labels/test"

for d in "$images_train_dir" "$labels_train_dir"; do
	if [[ ! -d "$d" ]]; then
		echo "Error: Directory not found: $d"
		exit 1
	fi
done

mkdir -p "$images_test_dir" "$labels_test_dir"

shopt -s nullglob

moved_count=0
skipped_count=0

for label_path in "$labels_train_dir"/"$city_prefix"_*_leftImg8bit.txt; do
	base_name="$(basename "$label_path" .txt)"
	image_path="$images_train_dir/$base_name.png"

	if [[ -f "$image_path" ]]; then
		mv "$label_path" "$labels_test_dir/"
		mv "$image_path" "$images_test_dir/"
		moved_count=$((moved_count + 1))
	else
		echo "Skip (image missing): $base_name.png"
		skipped_count=$((skipped_count + 1))
	fi
done

echo ""
echo "Moved pairs: $moved_count"
echo "Skipped (missing image): $skipped_count"
