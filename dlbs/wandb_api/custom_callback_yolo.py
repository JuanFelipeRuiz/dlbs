"""
Custom W&B callbacks for YOLO11 segmentation training.

After each validation batch, agrid is logged to W&B
showing ground truth masks alongside model predictions. After each epoch, overall
and per-class mask metrics (precision, recall, mAP50, dice) are logged under the
val/ and class/ namespaces.

Callbacks are registered via add_custom_callbacks(model) before model.train().

As we are not sure how we can mock a run without the full wandb and ultralytics 
training loop, these are best tested in an integration test with a real W&B run and 
cannot be run as standalone file or unit test. 
"""

import inspect
import logging
import numpy as np
import matplotlib.pyplot as plt
import wandb

from dlbs.plots.yolo_val_viz import make_val_grid_pred_gt_orig, make_per_class_grids

logger = logging.getLogger(__name__)


def _wandb_ready() -> bool:
    """Check if a W&B run is active."""
    return wandb.run is not None


def _to_arr(x, n):
    """Convert a value to a cleaned numpy array of length n.

    Args:
        x: Input value (array-like, list, tuple, or None).
        n: Expected length; returns None if input is shorter.

    Returns:
        Float array of length n with NaN/inf replaced by 0,
            or None if input is invalid or too short.
    """
    if x is None:
        return None
    if isinstance(x, (list, tuple)) and len(x) == 0:
        return None
    a = np.asarray(x, dtype=float).reshape(-1)
    if a.size < n:
        return None
    a = a[:n]
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


def _metric_container(validator):
    """Extract the segmentation metric object from a validator or trainer.

    Returns the seg or mask metric container, or None if unavailable.
    """
    m = getattr(validator, "metrics", None)
    if m is None:
        return None
    return getattr(m, "seg", None) or getattr(m, "mask", None)


def _get_per_class_seg_arrays(source):
    """Extract per-class precision, recall, and AP50 arrays from a source.
    
    Returns (classes, P, R, AP50) where each metric is a np.ndarray
        of length nc, or None if metrics are unavailable.
    """
    names = getattr(source, "names", None) or {}
    if not names:
        return None

    seg = _metric_container(source)
    if seg is None:
        return None

    classes = [names[i] for i in sorted(names)]
    nc = len(classes)

    # precision and recall are directly available on the seg container
    precision = _to_arr(getattr(seg, "p", None), nc)
    recall = _to_arr(getattr(seg, "r", None), nc)

    ap50 = getattr(seg, "ap50", None)
    ap = getattr(seg, "ap", None)

    # ap50 may be a flat array or the first column of a 2D ap matrix
    AP50 = _to_arr(ap50, nc)
    if AP50 is None and ap is not None:
        ap_arr = np.asarray(ap, dtype=float)
        if ap_arr.ndim == 2 and ap_arr.shape[0] >= nc and ap_arr.shape[1] >= 1:
            AP50 = _to_arr(ap_arr[:nc, 0], nc)

    if precision is None or recall is None or AP50 is None:
        return None

    return classes, precision, recall, AP50


def _epoch_from_validator(validator) -> int:
    """Get the current 1-based epoch number from a validator.

    Returns current epoch (1-based), or -1 if trainer is not available.
    """
    trainer = getattr(validator, "trainer", None)
    if trainer is None:
        return -1
    return int(getattr(trainer, "epoch", 0)) + 1


def _collect_per_class_metrics(source, split: str) -> dict:
    """Collect per-class segmentation metrics for a given split.

    Args:
        source: Ultralytics validator or trainer exposing names and seg metrics.
        split: Dataset split label, e.g. "train" or "val".

    Returns:
        W&B log dict with keys like class/{split}_precision/{name},
            _recall, _dice, _mAP50. Empty dict if metrics are unavailable.
    """
    arr = _get_per_class_seg_arrays(source)
    if not arr:
        return {}

    classes, precision, recall, ap50 = arr
    log = {}

    for i, name in enumerate(classes):
        name = str(name).strip()
        if name.lower() in ("background", "bg", ""):
            continue

        p = float(precision[i])
        r = float(recall[i])
        dice = 0.0 if (p + r) <= 0 else float((2.0 * p * r) / (p + r))

        log[f"class/{split}_mask_precision/{name}"] = p
        log[f"class/{split}_mask_recall/{name}"] = r
        log[f"class/{split}_mask_dice/{name}"] = dice
        log[f"class/{split}_mask_mAP50/{name}"] = float(ap50[i])

    return log


def _to_float_or_none(x):
    """Cast a value to float. Returns None if conversion fails."""
    try:
        return float(x)
    except Exception:
        return None


def _collect_overall_metrics(source, split: str) -> dict:
    """Collect aggregated segmentation metrics across all classes.

    Args:
        source: Ultralytics validator or trainer exposing seg metrics.
        split: Dataset split label, e.g. "train" or "val".

    Returns:
        W&B log dict with keys like overall/{split}_precision,
            _recall, _mAP50, _mAP50_95, _dice. Empty dict if metrics are unavailable.
    """
    seg = _metric_container(source)
    if seg is None:
        return {}

    precision = _to_float_or_none(getattr(seg, "mp", None))
    recall = _to_float_or_none(getattr(seg, "mr", None))
    map50 = _to_float_or_none(getattr(seg, "map50", None))
    map5095 = _to_float_or_none(getattr(seg, "map", None))

    log = {}
    if precision is not None:
        log[f"{split}/mask_precision"] = precision
    if recall is not None:
        log[f"{split}/mask_recall"] = recall
    if map50 is not None:
        log[f"{split}/mask_mAP50"] = map50
    if map5095 is not None:
        log[f"{split}/mask_mAP50_95"] = map5095
    if precision is not None and recall is not None:
        log[f"{split}/mask_dice"] = 0.0 if (precision + recall) <= 0 else (2.0 * precision * recall) / (precision + recall)

    return log


def on_val_batch_end(validator):
    """YOLO callback: log prediction grid for the first validation batch.

    Fires on every ``on_val_batch_end`` event but only processes part of first 
    batch_i==0. Uses frame inspection to access the batch and preds locals from the
    Ultralytics validation loop.
    """

    # max columns in the grid for the first batch visualization
    MAX_SHOW = 4  

    if not _wandb_ready():
        return

    try:
        # walk up two frames to reach the Ultralytics validation loop locals
        fr = inspect.currentframe()
        if fr is None or fr.f_back is None or fr.f_back.f_back is None:
            return

        v = fr.f_back.f_back.f_locals

        # only process the first batch 
        batch_i = v.get("batch_i", None)
        if batch_i is None or int(batch_i) != 0:
            return

        batch = v.get("batch", None)
        preds = v.get("preds", None)
        if batch is None or preds is None:
            return

        # normalise names to a dict regardless of how Ultralytics exposes it
        names = getattr(validator, "names", None)
        if isinstance(names, list):
            names = {i: n for i, n in enumerate(names)}
        elif not isinstance(names, dict):
            names = {}

        # log the combined pred/gt/orig grid for the whole batch and all classes
        fig = make_val_grid_pred_gt_orig(batch, preds, names=names, max_show=MAX_SHOW)
        if fig is None:
            return

        # collect all images into one log call to avoid step monotonicity warnings
        log_payload = {"instance_segmentation/val_first_batch_custom_grid": wandb.Image(fig)}
        plt.close(fig)

        grids = make_per_class_grids(batch, preds, names=names, max_show=MAX_SHOW)
        for cls_name, cls_fig in grids.items():
            log_payload[f"instance_segmentation/val_first_batch_class/{cls_name}"] = wandb.Image(cls_fig)
            plt.close(cls_fig)

        wandb.log(log_payload)

    except Exception as e:
        logger.warning(f"custom val grid failed: {e}")


def on_val_end(validator):
    """YOLO callback: log overall and per-class segmentation metrics after validation.

    Args:
        validator: Ultralytics validator instance passed by the callback system.
    """
    if not _wandb_ready():
        return

    log = {"epoch": _epoch_from_validator(validator)}
    log.update(_collect_overall_metrics(validator, split="val"))
    log.update(_collect_per_class_metrics(validator, split="val"))

    wandb.log(log)


def on_train_epoch_end(trainer):
    """YOLO callback: log per-class train metrics after each epoch.

    skips  if the trainer does not expose seg arrays.
    """
    if not _wandb_ready():
        return

    log = {}
    log.update(_collect_overall_metrics(trainer, split="train_mask"))
    log.update(_collect_per_class_metrics(trainer, split="train_mask"))

    if log:
        wandb.log(log)


def add_custom_callbacks(model):
    """Register all custom W&B callbacks on a YOLO model."""
    model.add_callback("on_val_batch_end", on_val_batch_end)
    model.add_callback("on_val_end", on_val_end)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
