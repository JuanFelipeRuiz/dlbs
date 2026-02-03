import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
from pathlib import Path
import json
import logging
import pandas as pd
import yaml
import math
import yaml
import math

logger = logging.getLogger(__name__)


def _read_yolo_annotations(ann_path, selected_classes, img_w, img_h):
    """Read and parse YOLO instance segmentation annotations."""
    ann_path = Path(ann_path)
    if not ann_path.exists() or not ann_path.is_file():
        logger.warning(f"Annotation file not found or invalid: {ann_path}")
        return []
    
    with open(ann_path, "r") as f:
        yolo_lines = [line.strip() for line in f if line.strip()]

    if selected_classes is not None:
        yolo_lines = [l for l in yolo_lines if int(l.split()[0]) in selected_classes]

    instances = []
    for line in yolo_lines:
        parts = line.split()
        cls = int(parts[0])
        coords = list(map(float, parts[1:]))
        xs = [x * img_w for x in coords[0::2]]
        ys = [y * img_h for y in coords[1::2]]
        instances.append({"cls": cls, "xs": xs, "ys": ys})
    return instances


def _read_polygon_annotations(ann_path, selected_classes):
    """Read and parse polygon-based annotations (e.g. Cityscapes-style JSON)."""
    ann_path = Path(ann_path)
    if not ann_path.exists() or not ann_path.is_file():
        logger.warning(f"Annotation file not found or invalid: {ann_path}")
        return []
    
    with open(ann_path, "r") as f:
        data = json.load(f)
    objects = data.get("objects", [])
    if selected_classes:
        objects = [o for o in objects if o["label"] in selected_classes]
    instances = []
    for obj in objects:
        poly = np.array(obj["polygon"])
        instances.append({"cls": obj["label"], "xs": poly[:, 0], "ys": poly[:, 1]})
    return instances


def _draw_instances(ax, instances):
    """Draw polygons or YOLO masks on an axis."""
    for inst in instances:
        xs = np.array(inst["xs"])
        ys = np.array(inst["ys"])
        color = (random.random(), random.random(), random.random())
        ax.fill(xs, ys, color=color, alpha=0.3)
        ax.plot(np.append(xs, xs[0]), np.append(ys, ys[0]), color=color, lw=2)

        ax.text(np.mean(xs), np.mean(ys), inst["cls"], color="black", fontsize=10)
        
def plot_instances_grid(
    df,
    selected_classes=None,
    show_instances=None,
    order_column=None,
    max_images=None,
    ncols=3,
    figsize=(20, 10),
    use_yolo=True,
):
    """
    Plot multiple images (and optionally their YOLO or polygon annotations) in one figure grid.

    Parameters:
    -----------
    show_instances : bool, str, or None
        - If True: Show instances using default column ("yolo_annotation_path" if use_yolo else "annotation_json")
        - If str: Use this column name for annotations (e.g., "instance_based_yolo", "yolo_annotation_path")
        - If False/None: Don't show instances
    use_yolo : bool
        Only used if show_instances=True (not a string). Determines default column name.
    """

    # Handle show_instances parameter
    if show_instances is None or show_instances is False:
        show_instances_flag = False
        ann_col = None
    elif isinstance(show_instances, str):
        # show_instances is a column name
        show_instances_flag = True
        ann_col = show_instances
    else:
        # show_instances is True
        show_instances_flag = True
        ann_col = "yolo_annotation_path" if use_yolo else "annotation_json"

    # Sort and trim dataset
    if order_column and order_column in df.columns:
        df = df.sort_values(order_column).reset_index(drop=True)
    if max_images:
        df = df.head(max_images)

    n_images = len(df)
    nrows = int(np.ceil(n_images / ncols))
    nrows_total = nrows * (2 if show_instances_flag else 1)

    fig, axes = plt.subplots(nrows_total, ncols, figsize=figsize, constrained_layout=True)
    axes = np.atleast_2d(axes)

    plot_idx = 0  # Track actual plotted images
    for i, (_, row) in enumerate(df.iterrows()):
        img_path = Path(row["image_path"])
        
        # Check if image path exists
        if not img_path.exists() or not img_path.is_file():
            logger.debug(f"Skipping {row.get('image_name', 'unknown')}: image file not found: {img_path}")
            continue
        
        # Get annotation path column name (only if show_instances is enabled)
        instances = []
        if show_instances_flag:
            if ann_col is None:
                ann_path_str = ""
            else:
                ann_path_str = row.get(ann_col, "")
            
            # Validate annotation path
            if not ann_path_str or pd.isna(ann_path_str):
                logger.debug(f"Skipping {row.get('image_name', 'unknown')}: no annotation path in column '{ann_col}'")
                continue
                
            ann_path = Path(ann_path_str)
            
            # Check if path exists and is a file
            if not ann_path.exists() or not ann_path.is_file():
                logger.debug(f"Skipping {row.get('image_name', 'unknown')}: annotation file not found: {ann_path}")
                continue

            img = np.array(Image.open(img_path))
            img_h, img_w = img.shape[:2]

            # Read annotations - determine format from file extension or column name
            if ann_col and ("yolo" in ann_col.lower() or ann_path.suffix == ".txt"):
                instances = _read_yolo_annotations(ann_path, selected_classes, img_w, img_h)
            elif ann_path.suffix == ".json":
                instances = _read_polygon_annotations(ann_path, selected_classes)
            else:
                # Fallback: use use_yolo parameter
                instances = _read_yolo_annotations(ann_path, selected_classes, img_w, img_h) if use_yolo \
                    else _read_polygon_annotations(ann_path, selected_classes)
        else:
            # Just load image without annotations
            img = np.array(Image.open(img_path))

        # Determine grid location
        col_idx = plot_idx % ncols
        row_idx = (plot_idx // ncols) * (2 if show_instances_flag else 1)

        # Original
        ax = axes[row_idx, col_idx]
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[:].set_visible(False)
        ax.set_title(row.get("image_name", Path(img_path).name))

        # Annotated below
        if show_instances_flag:
            ax2 = axes[row_idx + 1, col_idx]
            ax2.imshow(img)
            ax2.axis("off")
            _draw_instances(ax2, instances)
        
        plot_idx += 1

    # Hide unused axes (if any)
    for j in range(plot_idx, nrows_total * ncols):
        row_idx = j // ncols
        col_idx = j % ncols
        if row_idx < axes.shape[0] and col_idx < axes.shape[1]:
            axes[row_idx, col_idx].axis("off")

    return fig


def plot_instances_by_class(
    df,
    class_names_yaml=None,
    annotation_col="instance_based_yolo",
    image_col="image_path",
    max_images_per_class=5,
    figsize_per_subplot=(4, 4),
    ncols=4,
):
    """
    Plot instances grouped by class, with each class in its own subplot.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with image_path and annotation columns
    class_names_yaml : str or Path, optional
        Path to YAML file with class names (e.g., yolo_dataset/data.yaml)
        If None, will try to infer from data.yaml in yolo_dataset/
    annotation_col : str
        Column name with YOLO annotation paths
    image_col : str
        Column name with image paths
    max_images_per_class : int
        Maximum number of images to show per class
    figsize_per_subplot : tuple
        Size of each subplot (width, height)
    ncols : int
        Number of columns in the grid
    
    Returns:
    --------
    matplotlib.figure.Figure
    """
    # Load class names from YAML if provided
    if class_names_yaml is None:
        # Try default location
        default_yaml = Path("yolo_dataset/data.yaml")
        if default_yaml.exists():
            class_names_yaml = default_yaml
        else:
            raise ValueError("class_names_yaml not provided and default not found")
    
    class_names_yaml = Path(class_names_yaml)
    with open(class_names_yaml, "r") as f:
        yaml_data = yaml.safe_load(f)
    
    class_names = yaml_data.get("names", {})
    if isinstance(class_names, list):
        # Convert list to dict if needed
        class_names = {i: name for i, name in enumerate(class_names)}
    
    num_classes = len(class_names)
    nrows = math.ceil(num_classes / ncols)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per_subplot[0] * ncols, 
                                                     figsize_per_subplot[1] * nrows))
    axes = np.atleast_2d(axes).reshape(-1)
    
    # Collect instances per class
    class_instances = {cls_id: [] for cls_id in class_names.keys()}
    
    # Process all images and collect instances by class
    for idx, (_, row) in enumerate(df.iterrows()):
        img_path = Path(row.get(image_col, ""))
        ann_path_str = row.get(annotation_col, "")
        
        if not img_path.exists() or not img_path.is_file():
            continue
        if not ann_path_str or pd.isna(ann_path_str):
            continue
        
        ann_path = Path(ann_path_str)
        if not ann_path.exists() or not ann_path.is_file():
            continue
        
        img = np.array(Image.open(img_path))
        img_h, img_w = img.shape[:2]
        
        instances = _read_yolo_annotations(ann_path, selected_classes=None, img_w=img_w, img_h=img_h)
        
        # Group instances by class
        for inst in instances:
            cls_id = inst["cls"]
            if cls_id in class_instances:
                class_instances[cls_id].append({
                    "image": img,
                    "instance": inst,
                    "image_name": row.get("image_name", f"img_{idx}")
                })
    
    # Plot each class in its own subplot
    for cls_id, class_name in class_names.items():
        ax_idx = cls_id
        if ax_idx >= len(axes):
            continue
        ax = axes[ax_idx]
        
        instances_data = class_instances[cls_id]
        num_instances = len(instances_data)
        
        if num_instances == 0:
            ax.text(0.5, 0.5, f"Class {cls_id}: {class_name}\nNo instances found", 
                   ha="center", va="center", fontsize=12, transform=ax.transAxes)
            ax.axis("off")
            continue
        
        # Select random sample if too many
        if num_instances > max_images_per_class:
            instances_data = random.sample(instances_data, max_images_per_class)
        
        # Show first instance as representative
        sample = instances_data[0]
        ax.imshow(sample["image"])
        _draw_instances(ax, [sample["instance"]])
        ax.set_title(f"Class {cls_id}: {class_name}\n{num_instances} total instances", 
                     fontsize=10, fontweight="bold")
        ax.axis("off")
    
    # Hide unused subplots
    for idx in range(num_classes, len(axes)):
        axes[idx].axis("off")
    
    plt.tight_layout()
    return fig
