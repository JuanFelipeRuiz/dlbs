"""
Prepare YOLO dataset structure from DataFrame.

Creates YOLO-friendly folder structure and copies/links images and annotations.
"""

import logging
import shutil
from pathlib import Path
from typing import Optional, Union
import pandas as pd

logger = logging.getLogger(__name__)


def copy_to_yolo_format(
    df: pd.DataFrame,
    output_dir: Union[str, Path] = "yolo",
    annotation_col: str = "instance_based_yolo",
    image_col: str = "image_path",
    split_col: str = "split",
    base_dir: Optional[Union[str, Path]] = None,
) -> dict:
    """
    Copy or link YOLO annotations and images to YOLO-friendly folder structure.
    
    Creates structure:
    yolo/
      images/
        train/
        val/
        test/
      labels/
        train/
        val/
        test/
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: annotation_col, image_col, split_col
    output_dir : str or Path
        Output directory name (default: "yolo")
    annotation_col : str
        Column name with YOLO annotation paths (default: "instance_based_yolo")
    image_col : str
        Column name with image paths (default: "image_path")
    split_col : str
        Column name with split information (default: "split")
    base_dir : str or Path, optional
        Base directory for resolving relative paths. If None, uses current working directory.
    
    Returns:
    --------
    dict: Statistics about copied/linked files
    """
    output_dir = Path(output_dir)
    
    # Use provided base_dir or current working directory
    if base_dir is None:
        base_dir = Path.cwd()
    else:
        base_dir = Path(base_dir)
    
    logger.debug(f"Base directory for path resolution: {base_dir}")
    logger.debug(f"Output directory: {output_dir}")
    logger.debug(f"Annotation column: {annotation_col}")
    logger.debug(f"Image column: {image_col}")
    logger.debug(f"Split column: {split_col}")
    
    # Create directory structure
    logger.debug("Creating directory structure...")
    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directories for split: {split}")
    
    stats = {
        "train": {"images": 0, "labels": 0},
        "val": {"images": 0, "labels": 0},
        "test": {"images": 0, "labels": 0}
    }
    
    logger.debug(f"Using copy mode")
    logger.debug(f"Processing {len(df)} rows from DataFrame")
    
    for idx, row in df.iterrows():
        # Get paths
        ann_path_str = row.get(annotation_col, "")
        img_path_str = row.get(image_col, "")
        split = str(row.get(split_col, "train")).lower()
        
        # Validate
        if pd.isna(ann_path_str) or not ann_path_str or str(ann_path_str) == "":
            logger.debug(f"Row {idx}: Skipping - no annotation path")
            continue
        if pd.isna(img_path_str) or not img_path_str or str(img_path_str) == "":
            logger.debug(f"Row {idx}: Skipping - no image path")
            continue
        if split not in ["train", "val", "test"]:
            logger.debug(f"Row {idx}: Invalid split '{split}', using 'train'")
            split = "train"
        
        # Resolve paths: handle both absolute and relative paths
        # Normalize path separators (handle both / and \)
        ann_path_str = str(ann_path_str).replace("\\", "/")
        img_path_str = str(img_path_str).replace("\\", "/")
        
        # If path starts with ../, remove it and treat as relative to base_dir
        # (since paths in CSV are relative to repo root but may have ../ prefix)
        if ann_path_str.startswith("../"):
            # Remove ../ prefix and resolve relative to base_dir
            ann_path_str_clean = ann_path_str[3:]  # Remove "../"
            ann_path = (base_dir / ann_path_str_clean).resolve()
        elif ann_path_str.startswith("./"):
            # Remove ./ prefix
            ann_path_str_clean = ann_path_str[2:]  # Remove "./"
            ann_path = (base_dir / ann_path_str_clean).resolve()
        elif Path(ann_path_str).is_absolute():
            # Absolute path
            ann_path = Path(ann_path_str).resolve()
        else:
            # Relative path without ../
            ann_path = (base_dir / ann_path_str).resolve()
        
        if img_path_str.startswith("../"):
            img_path_str_clean = img_path_str[3:]  # Remove "../"
            img_path = (base_dir / img_path_str_clean).resolve()
        elif img_path_str.startswith("./"):
            img_path_str_clean = img_path_str[2:]  # Remove "./"
            img_path = (base_dir / img_path_str_clean).resolve()
        elif Path(img_path_str).is_absolute():
            img_path = Path(img_path_str).resolve()
        else:
            img_path = (base_dir / img_path_str).resolve()
        
        logger.debug(f"Row {idx}: Resolved paths - ann: {ann_path}, img: {img_path}")
        
        # Check if files exist
        if not ann_path.exists() or not ann_path.is_file():
            logger.debug(f"Row {idx}: Annotation file not found: {ann_path}")
            continue
        if not img_path.exists() or not img_path.is_file():
            logger.debug(f"Row {idx}: Image file not found: {img_path}")
            continue
        
        # Get filenames (must match: image.png -> image.txt)
        img_filename = img_path.name
        label_filename = img_path.stem + ".txt"
        
        # Destination paths
        dest_img = output_dir / "images" / split / img_filename
        dest_label = output_dir / "labels" / split / label_filename
        
        logger.debug(f"Row {idx}: Processing {img_filename} (split: {split})")
        
        # Copy files
        try:
            logger.debug(f"Row {idx}: Copying {img_path.name} -> {dest_img}")
            shutil.copy2(img_path, dest_img)
            logger.debug(f"Row {idx}: Copying {ann_path.name} -> {dest_label}")
            shutil.copy2(ann_path, dest_label)
            stats[split]["images"] += 1
            stats[split]["labels"] += 1
            if (stats[split]["images"] + stats[split]["labels"]) % 100 == 0:
                logger.debug(f"Progress: {stats[split]['images']} images, {stats[split]['labels']} labels processed for {split}")
        except Exception as e:
            logger.warning(f"Row {idx}: Failed to copy files - {e}")
            continue
    
    # Print summary
    total_images = sum(s["images"] for s in stats.values())
    total_labels = sum(s["labels"] for s in stats.values())
    logger.info(f"Copied {total_images} images and {total_labels} labels to {output_dir}/")
    logger.info(f"  Train: {stats['train']['images']} images, {stats['train']['labels']} labels")
    logger.info(f"  Val: {stats['val']['images']} images, {stats['val']['labels']} labels")
    if stats['test']['images'] > 0:
        logger.info(f"  Test: {stats['test']['images']} images, {stats['test']['labels']} labels")
    
    return stats


if __name__ == "__main__":
    import sys
    
    from pathlib import Path
    
    # Find the data directory
    # File is at: dlbs/transform_data/prepare_yolo_dataset.py
    # Repo root is: dlbs/
    # Assume script is executed from repo root
    repo_root = Path.cwd()
    csv_path = repo_root / "data" / "overview_df.csv"
    
    logger.debug(f"Repo root: {repo_root}")
    logger.debug(f"CSV path: {csv_path}")
    
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        sys.exit(1)
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Read overview DataFrame
    logger.info(f"Reading {csv_path}")
    overview_df = pd.read_csv(csv_path)
    logger.debug(f"Loaded DataFrame with {len(overview_df)} rows")
    logger.debug(f"DataFrame columns: {list(overview_df.columns)}")

    overview_df.loc[overview_df["city"] == "zurich", "split"] = "test"
    
    # Check required columns
    required_cols = ["instance_based_yolo", "image_path", "split"]
    missing_cols = [col for col in required_cols if col not in overview_df.columns]
    if missing_cols:
        logger.warning(f"Missing columns in DataFrame: {missing_cols}")
    else:
        logger.debug("All required columns found")
    
    # Check for non-null values
    logger.debug(f"Rows with instance_based_yolo: {overview_df['instance_based_yolo'].notna().sum()}")
    logger.debug(f"Rows with image_path: {overview_df['image_path'].notna().sum()}")
    logger.debug(f"Split distribution: {overview_df['split'].value_counts().to_dict()}")
    
    # Run with default parameters
    logger.info("Starting YOLO dataset preparation...")
    stats = copy_to_yolo_format(
        overview_df, 
        output_dir="yolo_dataset", 
        base_dir=repo_root  # Use repo root as base for resolving relative paths
    )
    
    print("\n✅ Dataset preparation complete!")
    print(f"Output directory: yolo_dataset/")

