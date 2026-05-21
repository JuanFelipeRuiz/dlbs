"""
Visualisation helpers for YOLO11 segmentation validation results.

Builds matplotlib figures that show ground truth masks alongside model predictions
for a given validation batch.

Main entry points:

make_val_grid_pred_gt_orig:
    One combined grid with original image, GT mask, and predicted mask side by
    side for up to ``max_show`` images.

`make_per_class_grids:
    Same layout but one figure per class, useful for inspecting per-class
    prediction quality in W&B.
"""
import numpy as np
import matplotlib.pyplot as plt

from dlbs.utils.yolo_decode_validation_batch import (
    _normalize_names,
    _resize_masks_nn,
    pred_instances,
    gt_instances_from_batch,
    filter_by_class,
    available_class_ids_in_batch,
)

# set True locally to print shape diagnostics during debugging
DEBUG_PRINT_SHAPES = False


def _to_uint8_img(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.uint8:
        return x
    x = np.asarray(x)
    if x.max() <= 1.5:
        x = x * 255.0
    return np.clip(x, 0, 255).astype(np.uint8)


def chw_tensor_to_hwc_uint8(t) -> np.ndarray:
    """Torch tensor CHW (or BCHW where B=1) -> HWC uint8."""
    a = t.detach().float().cpu().numpy()
    if a.ndim == 4:
        a = a[0]
    a = np.transpose(a, (1, 2, 0))
    return _to_uint8_img(a)


def class_palette(num_classes: int):
    """Deterministic RGB palette indexed by class id."""
    cols = []
    for i in range(max(num_classes, 1)):
        h = (i * 0.61803398875) % 1.0
        r = int(255 * (0.5 + 0.5 * np.cos(2 * np.pi * (h + 0.00))))
        g = int(255 * (0.5 + 0.5 * np.cos(2 * np.pi * (h + 0.33))))
        b = int(255 * (0.5 + 0.5 * np.cos(2 * np.pi * (h + 0.66))))
        cols.append((r, g, b))
    return cols


def instance_palette(num_instances: int):
    """Deterministic RGB palette indexed by instance id."""
    cols = []
    for i in range(max(num_instances, 1)):
        h = (i * 0.754877666) % 1.0
        r = int(255 * (0.5 + 0.5 * np.cos(2 * np.pi * (h + 0.10))))
        g = int(255 * (0.5 + 0.5 * np.cos(2 * np.pi * (h + 0.45))))
        b = int(255 * (0.5 + 0.5 * np.cos(2 * np.pi * (h + 0.80))))
        cols.append((r, g, b))
    return cols


def render_black_bg_instances_by_class(hw, masks_bool, cls_ids, class_colors, alpha=0.90):
    """Render predicted instances onto a black background, coloured by class.

    Args:
        hw: Output (H, W) in pixels.
        masks_bool: (N,H,W) bool array or None.
        cls_ids: (N,) int array or None.
        class_colors: List of (R,G,B) tuples indexed by class id.
        alpha: Blend strength for each mask (0=transparent, 1=opaque).

    Returns:
        (H,W,3) uint8 image.
    """
    H, W = hw
    base = np.zeros((H, W, 3), dtype=np.uint8)
    if masks_bool is None:
        return base

    masks_bool = np.asarray(masks_bool)
    if masks_bool.ndim == 2:
        masks_bool = masks_bool[None, ...]
    if masks_bool.shape[-2:] != (H, W):
        masks_bool = _resize_masks_nn(masks_bool.astype(np.uint8), (H, W)).astype(bool)

    if cls_ids is None:
        cls_ids = np.zeros((masks_bool.shape[0],), dtype=int)
    cls_ids = np.asarray(cls_ids).reshape(-1)

    n = min(masks_bool.shape[0], cls_ids.shape[0])
    out = base.astype(np.float32)
    for i in range(n):
        m = masks_bool[i].astype(bool)
        if not m.any():
            continue
        cid = int(cls_ids[i])
        r, g, b = class_colors[cid % len(class_colors)]
        out[m, 0] = (1 - alpha) * out[m, 0] + alpha * r
        out[m, 1] = (1 - alpha) * out[m, 1] + alpha * g
        out[m, 2] = (1 - alpha) * out[m, 2] + alpha * b
    return np.clip(out, 0, 255).astype(np.uint8)


def render_black_bg_instances_unique(hw, masks_bool, alpha=0.95):
    """Render GT instances onto a black background, each with a unique colour.

    Args:
        hw: Output (H, W) in pixels.
        masks_bool: (N,H,W) bool array or None.
        alpha: Blend strength for each mask (0=transparent, 1=opaque).

    Returns:
        (H,W,3) uint8 image.
    """
    H, W = hw
    base = np.zeros((H, W, 3), dtype=np.uint8)
    if masks_bool is None:
        return base

    masks_bool = np.asarray(masks_bool)
    if masks_bool.ndim == 2:
        masks_bool = masks_bool[None, ...]
    if masks_bool.shape[-2:] != (H, W):
        masks_bool = _resize_masks_nn(masks_bool.astype(np.uint8), (H, W)).astype(bool)

    cols = instance_palette(masks_bool.shape[0])
    out = base.astype(np.float32)
    for i in range(masks_bool.shape[0]):
        m = masks_bool[i].astype(bool)
        if not m.any():
            continue
        r, g, b = cols[i]
        out[m, 0] = (1 - alpha) * out[m, 0] + alpha * r
        out[m, 1] = (1 - alpha) * out[m, 1] + alpha * g
        out[m, 2] = (1 - alpha) * out[m, 2] + alpha * b
    return np.clip(out, 0, 255).astype(np.uint8)


def make_val_grid_pred_gt_orig(batch: dict, preds, names, max_show: int = 4):
    """Build a 3×N grid figure for the first validation batch.

    Rows: predictions (coloured by class) / GT (unique colour per instance) /
    original image. Columns: up to ``max_show`` images from the batch.

    Args:
        batch: Ultralytics batch dict containing "img", "masks", etc.
        preds: List of per-image predictions.
        names: Class names as dict or list.
        max_show: Maximum number of images to include as columns.

    Returns:
        A matplotlib Figure, or None if the batch is empty.
    """
    names = _normalize_names(names)
    imgs = batch.get("img", None)
    if imgs is None or not isinstance(preds, (list, tuple)):
        return None

    n = min(int(imgs.shape[0]), max_show)
    nc = (max(names.keys()) + 1) if names else 1
    class_colors = class_palette(nc)

    fig, axes = plt.subplots(3, n, figsize=(4 * n, 10))
    if n == 1:
        axes = np.array(axes).reshape(3, 1)

    for i in range(n):
        img_uint8 = chw_tensor_to_hwc_uint8(imgs[i])
        H, W = img_uint8.shape[:2]

        pred_masks, pred_cls, _ = pred_instances(preds[i]) if i < len(preds) else (None, None, None)
        gt_masks, gt_cls = gt_instances_from_batch(batch, i, (H, W), names=names)

        if DEBUG_PRINT_SHAPES:
            pm = 0 if pred_masks is None else int(np.asarray(pred_masks).shape[0])
            gm = 0 if gt_masks is None else int(np.asarray(gt_masks).shape[0])
            print(f"[yolo_val_viz] img[{i}] HxW={H}x{W} pred_inst={pm} gt_inst={gm}")

        axes[0, i].imshow(render_black_bg_instances_by_class((H, W), pred_masks, pred_cls, class_colors))
        axes[1, i].imshow(render_black_bg_instances_unique((H, W), gt_masks))
        axes[2, i].imshow(img_uint8)
        for r in range(3):
            axes[r, i].axis("off")

    row_labels = ["Pred (by class)", "GT (per instance)", "Original"]
    for r, label in enumerate(row_labels):
        axes[r, 0].set_ylabel(label, fontsize=12)
        axes[r, 0].yaxis.label.set_visible(True)
    plt.suptitle("Validation Batch: Predictions vs Ground Truth", fontsize=14)
    plt.tight_layout()
    return fig


def make_per_class_grids(batch: dict, preds, names, max_show: int = 4, class_ids=None):
    """Build one 3×N grid figure per class that appears in the batch.

    Rows: predicted instances of that class / GT instances / original image.
    Only produces a figure for a class if at least one instance is found.

    Args:
        batch: Ultralytics batch dict containing "img", "masks", etc.
        preds: List of per-image predictions.
        names: Class names as dict or list.
        max_show: Maximum number of images to include as columns.
        class_ids: Explicit list of class ids to render. If None, all class ids
            present in the batch are used.

    Returns:
        Dict mapping class name to matplotlib Figure.
    """
    names = _normalize_names(names)
    imgs = batch.get("img", None)
    if imgs is None or not isinstance(preds, (list, tuple)):
        return {}

    if class_ids is None:
        class_ids = sorted(available_class_ids_in_batch(batch, preds, names, max_show=max_show))
    else:
        class_ids = [int(c) for c in class_ids]

    if not class_ids:
        return {}

    n = min(int(imgs.shape[0]), max_show)
    nc = (max(names.keys()) + 1) if names else 1
    class_colors = class_palette(nc)

    out = {}

    for cid in class_ids:
        cname = names.get(cid, str(cid))

        fig, axes = plt.subplots(3, n, figsize=(4 * n, 10))
        if n == 1:
            axes = np.array(axes).reshape(3, 1)

        any_found = False

        for i in range(n):
            img_uint8 = chw_tensor_to_hwc_uint8(imgs[i])
            H, W = img_uint8.shape[:2]

            pred_masks, pred_cls, _ = pred_instances(preds[i]) if i < len(preds) else (None, None, None)
            pm_f, pc_f = filter_by_class(pred_masks, pred_cls, cid)

            gt_masks, gt_cls = gt_instances_from_batch(batch, i, (H, W), names=names)
            gm_f, _ = filter_by_class(gt_masks, gt_cls, cid)

            if (pm_f is not None) or (gm_f is not None):
                any_found = True

            axes[0, i].imshow(render_black_bg_instances_unique((H, W), pm_f))
            axes[1, i].imshow(render_black_bg_instances_unique((H, W), gm_f))
            axes[2, i].imshow(img_uint8)
            for r in range(3):
                axes[r, i].axis("off")

        row_labels = [f"Pred: {cname}", f"GT: {cname}\n(per instance)", "Original"]
        for r, label in enumerate(row_labels):
            axes[r, 0].set_ylabel(label, fontsize=12)
            axes[r, 0].yaxis.label.set_visible(True)
        fig.suptitle(f"Class: {cname}", fontsize=14)
        plt.tight_layout()

        if any_found:
            out[cname] = fig
        else:
            plt.close(fig)

    return out
