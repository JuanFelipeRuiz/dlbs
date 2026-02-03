"""
Plot statistics for YOLO dataset: instances per class and area per instance.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import yaml
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


def calculate_polygon_area(xs, ys):
    """Calculate polygon area using shoelace formula."""
    if len(xs) < 3 or len(ys) < 3:
        return 0.0
    xs = np.array(xs)
    ys = np.array(ys)
    return 0.5 * np.abs(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1)))


def read_yolo_annotation(ann_path, img_w, img_h):
    """Read YOLO segmentation annotation and return instances with areas."""
    ann_path = Path(ann_path)
    if not ann_path.exists() or not ann_path.is_file():
        return []
    
    instances = []
    with open(ann_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            cls_id = int(parts[0])
            coords = list(map(float, parts[1:]))
            
            # Convert normalized coordinates to pixel coordinates
            xs = [x * img_w for x in coords[0::2]]
            ys = [y * img_h for y in coords[1::2]]
            
            # Calculate area
            area = calculate_polygon_area(xs, ys)
            
            instances.append({
                "class_id": cls_id,
                "area": area,
                "xs": xs,
                "ys": ys
            })
    
    return instances


def analyze_yolo_dataset(
    dataset_dir,
    data_yaml_path=None,
    exclude_splits=None,
):
    """
    Analyze YOLO dataset and collect statistics per split.
    
    Parameters:
    -----------
    dataset_dir : str or Path
        Path to YOLO dataset directory (should contain images/ and labels/ subdirectories)
    data_yaml_path : str or Path, optional
        Path to data.yaml file. If None, looks for data.yaml in dataset_dir
    exclude_splits : list, optional
        List of splits to exclude (default: ["test"])
    
    Returns:
    --------
    dict: Statistics with class names, instance counts per split, and areas per split
    """
    if exclude_splits is None:
        exclude_splits = ["test"]
    
    dataset_dir = Path(dataset_dir)
    
    # Load class names from YAML
    if data_yaml_path is None:
        data_yaml_path = dataset_dir / "data.yaml"
    else:
        data_yaml_path = Path(data_yaml_path)
    
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found at {data_yaml_path}")
    
    with open(data_yaml_path, "r") as f:
        yaml_data = yaml.safe_load(f)
    
    class_names = yaml_data.get("names", {})
    if isinstance(class_names, list):
        class_names = {i: name for i, name in enumerate(class_names)}
    
    # Collect statistics per split
    instance_counts_per_split = defaultdict(lambda: defaultdict(int))
    areas_per_class_split = defaultdict(lambda: defaultdict(list))
    
    # Process each split
    labels_dir = dataset_dir / "labels"
    images_dir = dataset_dir / "images"
    
    for split_dir in labels_dir.iterdir():
        if not split_dir.is_dir():
            continue
        
        split_name = split_dir.name
        if split_name in exclude_splits:
            logger.info(f"Skipping split: {split_name}")
            continue
        
        logger.info(f"Processing split: {split_name}")
        
        # Process all annotation files in this split
        for ann_file in split_dir.glob("*.txt"):
            # Find corresponding image
            img_file = images_dir / split_name / (ann_file.stem + ".png")
            if not img_file.exists():
                # Try other extensions
                for ext in [".jpg", ".jpeg", ".PNG", ".JPG"]:
                    img_file = images_dir / split_name / (ann_file.stem + ext)
                    if img_file.exists():
                        break
                else:
                    logger.warning(f"Image not found for {ann_file}")
                    continue
            
            # Get image dimensions
            try:
                from PIL import Image
                img = Image.open(img_file)
                img_w, img_h = img.size
            except Exception as e:
                logger.warning(f"Could not open image {img_file}: {e}")
                continue
            
            # Read annotations
            instances = read_yolo_annotation(ann_file, img_w, img_h)
            
            for inst in instances:
                cls_id = inst["class_id"]
                instance_counts_per_split[split_name][cls_id] += 1
                areas_per_class_split[split_name][cls_id].append(inst["area"])
    
    # Convert to numpy arrays
    areas_per_class_split = {
        split: {k: np.array(v) for k, v in class_dict.items()}
        for split, class_dict in areas_per_class_split.items()
    }
    
    return {
        "class_names": class_names,
        "instance_counts_per_split": dict(instance_counts_per_split),
        "areas_per_class_split": dict(areas_per_class_split)
    }


def plot_class_statistics(
    dataset_dir,
    data_yaml_path=None,
    exclude_splits=None,
    figsize=(18, 6),
):
    """
    Plot instances per class and area distribution per class, with split distinction.
    
    Parameters:
    -----------
    dataset_dir : str or Path
        Path to YOLO dataset directory
    data_yaml_path : str or Path, optional
        Path to data.yaml file
    exclude_splits : list, optional
        List of splits to exclude (default: ["test"])
    figsize : tuple
        Figure size
    
    Returns:
    --------
    matplotlib.figure.Figure
    """
    # Analyze dataset
    stats = analyze_yolo_dataset(dataset_dir, data_yaml_path, exclude_splits)
    
    class_names = stats["class_names"]
    instance_counts_per_split = stats["instance_counts_per_split"]
    areas_per_class_split = stats["areas_per_class_split"]
    
    # Get all splits (excluding test)
    splits = sorted([s for s in instance_counts_per_split.keys() if s not in exclude_splits])
    split_colors = {"train": "#2E86AB", "val": "#A23B72", "test": "#F18F01"}
    default_colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A994E"]
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))
    
    # Plot 1: Number of instances per class (grouped by split)
    class_ids = sorted(class_names.keys())
    class_labels = [f"{class_names[cls_id]}\n(cls {cls_id})" for cls_id in class_ids]
    
    x = np.arange(len(class_ids))
    width = 0.35  # Width of bars
    
    # Create grouped bars
    for i, split in enumerate(splits):
        counts = [instance_counts_per_split[split].get(cls_id, 0) for cls_id in class_ids]
        color = split_colors.get(split, default_colors[i % len(default_colors)])
        offset = (i - len(splits)/2 + 0.5) * width / len(splits)
        bars = ax1.bar(x + offset, counts, width/len(splits), label=split, 
                       color=color, alpha=0.7)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            if count > 0:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{count}',
                        ha='center', va='bottom', fontsize=7)
    
    ax1.set_xlabel("Class", fontsize=12)
    ax1.set_ylabel("Number of Instances", fontsize=12)
    ax1.set_title("Instances per Class (by Split)", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_labels, rotation=45, ha='right', fontsize=9)
    ax1.legend(title="Split", fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Area distribution per class (boxplot with split distinction)
    # Prepare data for boxplot - group by class, then by split
    boxplot_data = []
    boxplot_positions = []
    boxplot_labels = []
    boxplot_colors_list = []
    
    pos = 1
    for cls_id in class_ids:
        class_has_data = False
        for split in splits:
            if cls_id in areas_per_class_split.get(split, {}) and len(areas_per_class_split[split][cls_id]) > 0:
                boxplot_data.append(areas_per_class_split[split][cls_id])
                boxplot_positions.append(pos)
                color = split_colors.get(split, default_colors[splits.index(split) % len(default_colors)])
                boxplot_colors_list.append(color)
                if not class_has_data:
                    boxplot_labels.append(f"{class_names[cls_id]}\n(cls {cls_id})")
                    class_has_data = True
                else:
                    boxplot_labels.append("")
                pos += 0.3
        if class_has_data:
            pos += 0.7  # Space between classes
    
    if boxplot_data:
        bp = ax2.boxplot(boxplot_data, positions=boxplot_positions, 
                        patch_artist=True, widths=0.25)
        
        # Color the boxes by split
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(boxplot_colors_list[i])
            patch.set_alpha(0.7)
        
        # Set x-axis labels only for first occurrence of each class
        unique_positions = []
        unique_labels = []
        for pos, label in zip(boxplot_positions, boxplot_labels):
            if label:  # Only add non-empty labels
                unique_positions.append(pos)
                unique_labels.append(label)
        
        ax2.set_xticks(unique_positions)
        ax2.set_xticklabels(unique_labels, rotation=45, ha='right', fontsize=9)
        
        # Create custom legend for splits
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=split_colors.get(split, default_colors[i % len(default_colors)]), 
                                alpha=0.7, label=split) 
                          for i, split in enumerate(splits)]
        ax2.legend(handles=legend_elements, title="Split", fontsize=9, loc='upper right')
        
        ax2.set_xlabel("Class", fontsize=12)
        ax2.set_ylabel("Area (pixels²)", fontsize=12)
        ax2.set_title("Area Distribution per Class (by Split)", fontsize=14, fontweight="bold")
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_yscale('log')  # Use log scale for better visualization
    
    # Plot 3: Cumulative area (total pixels) per class
    # Calculate total area per class per split
    total_areas_per_split = {}
    for split in splits:
        total_areas_per_split[split] = {}
        for cls_id in class_ids:
            if cls_id in areas_per_class_split.get(split, {}):
                total_area = np.sum(areas_per_class_split[split][cls_id])
                total_areas_per_split[split][cls_id] = total_area
            else:
                total_areas_per_split[split][cls_id] = 0
    
    x = np.arange(len(class_ids))
    width = 0.35
    
    # Create grouped bars for cumulative area
    for i, split in enumerate(splits):
        total_areas = [total_areas_per_split[split].get(cls_id, 0) for cls_id in class_ids]
        color = split_colors.get(split, default_colors[i % len(default_colors)])
        offset = (i - len(splits)/2 + 0.5) * width / len(splits)
        bars = ax3.bar(x + offset, total_areas, width/len(splits), label=split, 
                       color=color, alpha=0.7)
        
        # Add value labels on bars (format as millions or thousands)
        for bar, area in zip(bars, total_areas):
            if area > 0:
                height = bar.get_height()
                if area >= 1e6:
                    label = f'{area/1e6:.1f}M'
                elif area >= 1e3:
                    label = f'{area/1e3:.1f}K'
                else:
                    label = f'{int(area)}'
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        label,
                        ha='center', va='bottom', fontsize=7)
    
    ax3.set_xlabel("Class", fontsize=12)
    ax3.set_ylabel("Total Area (pixels²)", fontsize=12)
    ax3.set_title("Cumulative Area per Class (by Split)", fontsize=14, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(class_labels, rotation=45, ha='right', fontsize=9)
    ax3.legend(title="Split", fontsize=9)
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_yscale('log')  # Use log scale for better visualization
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Default dataset directory
    dataset_dir = Path("yolo_dataset")
    
    if not dataset_dir.exists():
        logger.error(f"Dataset directory not found: {dataset_dir}")
        sys.exit(1)
    
    logger.info(f"Analyzing dataset: {dataset_dir}")
    
    # Create plots
    fig = plot_class_statistics(
        dataset_dir=dataset_dir,
        exclude_splits=["test"],  # Always exclude test
        figsize=(18, 6)
    )
    
    # Save figure
    output_path = "class_statistics.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Plot saved to: {output_path}")
    
    plt.show()

