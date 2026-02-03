"""
Prepare a YOLO dataset folder structure from a DataFrame.

Creates:
yolo/
  images/{train,val,test}/
  labels/{train,val,test}/
and copies images + .txt labels into the matching split.
"""

import logging
import shutil
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def copy_to_yolo_format(
    df,
    output_dir="yolo",
    annotation_col="instance_based_yolo",
    image_col="image_path",
    split_col="split",
    base_dir=None,
):
    """
    Copy YOLO annotations and images into YOLO-friendly folders.

    Args:
        df: DataFrame with columns [annotation_col, image_col, split_col]
        output_dir: Target root directory (created if missing)
        annotation_col: Column with .txt label paths
        image_col: Column with image paths
        split_col: Column with split values: train/val/test
        base_dir: Resolve relative paths against this directory (default: cwd)

    Returns:
        dict with per-split counts of copied images and labels
    """
    output_dir = Path(output_dir)
    base_dir = Path(base_dir) if base_dir is not None else Path.cwd()

    _make_structure(output_dir)

    stats = {
        "train": {"images": 0, "labels": 0},
        "val": {"images": 0, "labels": 0},
        "test": {"images": 0, "labels": 0},
    }

    logger.debug("Processing %d rows", len(df))

    for idx, row in df.iterrows():
        ann_raw = row.get(annotation_col, "")
        img_raw = row.get(image_col, "")
        split = str(row.get(split_col, "train")).lower()

        ok, split = _validate_row(idx, ann_raw, img_raw, split)
        if not ok:
            continue

        ann_path = _resolve_path(ann_raw, base_dir)
        img_path = _resolve_path(img_raw, base_dir)

        if not ann_path or not ann_path.is_file():
            logger.debug("Row %s: annotation missing: %s", idx, ann_path)
            continue
        if not img_path or not img_path.is_file():
            logger.debug("Row %s: image missing: %s", idx, img_path)
            continue

        dest_img, dest_label = _dest_paths(output_dir, split, img_path)
        try:
            _copy_pair(img_path, dest_img, ann_path, dest_label)
            stats[split]["images"] += 1
            stats[split]["labels"] += 1
        except Exception as e:
            logger.warning("Row %s: copy failed: %s", idx, e)
            continue

    _log_summary(output_dir, stats)
    return stats


def _make_structure(root):
    """
    Create images/ and labels/ directories for train/val/test.
    """
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)


def _validate_row(idx, ann_raw, img_raw, split):
    """
    Validate basic fields and normalize the split value.
    """
    if pd.isna(ann_raw) or not str(ann_raw).strip():
        logger.debug("Row %s: no annotation path", idx)
        return False, split
    if pd.isna(img_raw) or not str(img_raw).strip():
        logger.debug("Row %s: no image path", idx)
        return False, split
    if split not in ("train", "val", "test"):
        logger.debug("Row %s: invalid split '%s' -> 'train'", idx, split)
        split = "train"
    return True, split


def _normalize_path_str(p):
    """
    Normalize path separators and strip leading './' or '../' when present.
    """
    s = str(p).replace("\\", "/")
    if s.startswith("./"):
        return s[2:]
    if s.startswith("../"):
        return s[3:]
    return s


def _resolve_path(raw, base_dir):
    """
    Resolve a raw string path to an absolute Path under base_dir when relative.
    """
    raw = str(raw)
    # absolute path
    if Path(raw).is_absolute():
        return Path(raw).resolve()

    # handle ../ and ./ against base_dir
    cleaned = _normalize_path_str(raw)
    return (base_dir / cleaned).resolve()


def _dest_paths(root, split, img_path):
    """
    Return destination file paths for image and label within the YOLO tree.
    """
    img_dst = root / "images" / split / img_path.name
    label_name = img_path.stem + ".txt"
    label_dst = root / "labels" / split / label_name
    return img_dst, label_dst


def _copy_pair(src_img, dst_img, src_label, dst_label):
    """
    Copy image and label to their destinations, overwriting if present.
    """
    shutil.copy2(src_img, dst_img)
    shutil.copy2(src_label, dst_label)


def _log_summary(root, stats):
    """
    Log a readable summary of copied files.
    """
    total_images = sum(s["images"] for s in stats.values())
    total_labels = sum(s["labels"] for s in stats.values())
    logger.info("Copied %d images and %d labels to %s/", total_images, total_labels, root)
    logger.info("  Train: %d images, %d labels", stats["train"]["images"], stats["train"]["labels"])
    logger.info("  Val:   %d images, %d labels", stats["val"]["images"], stats["val"]["labels"])
    if stats["test"]["images"] > 0 or stats["test"]["labels"] > 0:
        logger.info("  Test:  %d images, %d labels", stats["test"]["images"], stats["test"]["labels"])


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    repo_root = Path.cwd()
    csv_path = repo_root / "data" / "overview_df.csv"

    if not csv_path.exists():
        logger.error("CSV file not found: %s", csv_path)
        sys.exit(1)

    logger.info("Reading %s", csv_path)
    overview_df = pd.read_csv(csv_path)

    # exclude the Zurich for the test split
    if "city" in overview_df.columns:
        overview_df.loc[overview_df["city"].astype(str).str.lower() == "zurich", "split"] = "test"

    # sanity check on the first label path 
    if "instance_based_yolo" in overview_df.columns and len(overview_df) > 0:
        first_path = str(overview_df.iloc[0]["instance_based_yolo"])
        if first_path.startswith("/"):
            first_path = first_path[1:]
        if first_path and not (repo_root / _normalize_path_str(first_path)).exists():
            logger.error("Path %s does not exist (relative to %s)", first_path, repo_root)
            sys.exit(1)
    else:
        logger.error("Column 'instance_based_yolo' not found in DataFrame")
        sys.exit(1)

    logger.info("Starting YOLO dataset preparation...")
    stats = copy_to_yolo_format(
        overview_df,
        output_dir="yolo_dataset",
        base_dir=repo_root,
    )
