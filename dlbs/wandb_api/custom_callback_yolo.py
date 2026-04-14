# yolocustom_wand.py
"""
Ultralytics YOLO Segmentation + W&B (modular + clean)

Logs:
- first validation batch custom grid (every epoch)
- per-class segmentation metrics (every epoch)

No debug logs.
No custom step metrics.
No validator.plots access.
Docs-style callback usage (inspect for val batch locals).
"""

import inspect
import logging
import numpy as np
import matplotlib.pyplot as plt
import wandb

from dlbs.plots.yolo_val_viz import make_val_grid_pred_gt_orig, make_per_class_grids

logger = logging.getLogger(__name__)

MAX_SHOW = 4  # max columns in the grid


def _wandb_ready() -> bool:
    return wandb.run is not None


def _to_arr(x, n):
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
    m = getattr(validator, "metrics", None)
    if m is None:
        return None
    return getattr(m, "seg", None) or getattr(m, "mask", None)


def _get_per_class_seg_arrays(source):
    names = getattr(source, "names", None) or {}
    if not names:
        return None

    seg = _metric_container(source)
    if seg is None:
        return None

    classes = [names[i] for i in sorted(names)]
    nc = len(classes)

    P = _to_arr(getattr(seg, "p", None), nc)
    R = _to_arr(getattr(seg, "r", None), nc)

    ap50 = getattr(seg, "ap50", None)
    ap = getattr(seg, "ap", None)

    AP50 = _to_arr(ap50, nc)
    if AP50 is None and ap is not None:
        ap_arr = np.asarray(ap, dtype=float)
        if ap_arr.ndim == 2 and ap_arr.shape[0] >= nc and ap_arr.shape[1] >= 1:
            AP50 = _to_arr(ap_arr[:nc, 0], nc)

    if P is None or R is None or AP50 is None:
        return None

    return classes, P, R, AP50


def _epoch_from_validator(validator) -> int:
    trainer = getattr(validator, "trainer", None)
    if trainer is None:
        return -1
    return int(getattr(trainer, "epoch", 0)) + 1


def _collect_per_class_metrics(source, split: str) -> dict:
    """Collect per-class segmentation metrics for a split (train or val)."""
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

        log[f"class/{split}_precision/{name}"] = p
        log[f"class/{split}_recall/{name}"] = r
        log[f"class/{split}_dice/{name}"] = dice
        log[f"class/{split}_mAP50/{name}"] = float(ap50[i])

    return log


def _to_float_or_none(x):
    try:
        return float(x)
    except Exception:
        return None


def _collect_overall_metrics(source, split: str) -> dict:
    """Collect overall (not per-class) segmentation metrics."""
    seg = _metric_container(source)
    if seg is None:
        return {}

    precision = _to_float_or_none(getattr(seg, "mp", None))
    recall = _to_float_or_none(getattr(seg, "mr", None))
    map50 = _to_float_or_none(getattr(seg, "map50", None))
    map5095 = _to_float_or_none(getattr(seg, "map", None))

    log = {}
    if precision is not None:
        log[f"overall/{split}_precision"] = precision
    if recall is not None:
        log[f"overall/{split}_recall"] = recall
    if map50 is not None:
        log[f"overall/{split}_mAP50"] = map50
    if map5095 is not None:
        log[f"overall/{split}_mAP50_95"] = map5095
    if precision is not None and recall is not None:
        log[f"overall/{split}_dice"] = 0.0 if (precision + recall) <= 0 else (2.0 * precision * recall) / (precision + recall)

    return log


def on_val_batch_end(validator):
    """
    First validation batch (batch_i==0): log custom 3xN grid to W&B.
    """
    if not _wandb_ready():
        return

    try:
        fr = inspect.currentframe()
        if fr is None or fr.f_back is None or fr.f_back.f_back is None:
            return

        v = fr.f_back.f_back.f_locals

        batch_i = v.get("batch_i", None)
        if batch_i is None or int(batch_i) != 0:
            return

        batch = v.get("batch", None)
        preds = v.get("preds", None)
        if batch is None or preds is None:
            return

        names = getattr(validator, "names", None)
        if isinstance(names, list):
            names = {i: n for i, n in enumerate(names)}
        elif not isinstance(names, dict):
            names = {}

        fig = make_val_grid_pred_gt_orig(batch, preds, names=names, max_show=MAX_SHOW)
        if fig is None:
            return

        wandb.log(
            {"predictions/val_first_batch_custom_grid": wandb.Image(fig)},
            step=wandb.run.step,
        )
        plt.close(fig)

        grids = make_per_class_grids(batch, preds, names=names, max_show=MAX_SHOW)
        for cls_name, cls_fig in grids.items():
            key = f"predictions/val_first_batch_class/{cls_name}"
            wandb.log({key: wandb.Image(cls_fig)}, step=wandb.run.step)
            plt.close(cls_fig)


    except Exception as e:
        logger.warning(f"custom val grid failed: {e}")


def on_val_end(validator):
    """
    After validation: log overall + per-class segmentation metrics in one call.
    """
    if not _wandb_ready():
        return

    log = {"epoch": _epoch_from_validator(validator)}
    log.update(_collect_overall_metrics(validator, split="val"))
    log.update(_collect_per_class_metrics(validator, split="val"))

    wandb.log(log, step=wandb.run.step)


def on_train_epoch_end(trainer):
    """
    After train epoch: best-effort logging of per-class train metrics.
    (Only logs if trainer exposes per-class seg arrays.)
    """
    if not _wandb_ready():
        return

    log = {}
    log.update(_collect_overall_metrics(trainer, split="train"))
    log.update(_collect_per_class_metrics(trainer, split="train"))

    if log:
        wandb.log(log, step=wandb.run.step)


def add_custom_callbacks(model):
    model.add_callback("on_val_batch_end", on_val_batch_end)
    model.add_callback("on_val_end", on_val_end)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
