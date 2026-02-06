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


def _get_per_class_seg_arrays(validator):
    names = getattr(validator, "names", None) or {}
    if not names:
        return None

    seg = _metric_container(validator)
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


def on_val_batch_end(validator):
    """
    First validation batch (batch_i==0): log custom 3xN grid to W&B.
    """
    if not _wandb_ready():
        return

    try:
        fr = inspect.currentframe()
        if fr is None or fr.f_back is None or fr.f_back.f_back is None:
            print("No frame found")
            return

        v = fr.f_back.f_back.f_locals

        batch_i = v.get("batch_i", None)
        if batch_i is None or int(batch_i) != 0:
            return

        batch = v.get("batch", None)
        preds = v.get("preds", None)
        if batch is None or preds is None:
            print("No batch or preds found")
            return

        names = getattr(validator, "names", None)
        if isinstance(names, list):
            names = {i: n for i, n in enumerate(names)}
        elif not isinstance(names, dict):
            names = {}

        fig = make_val_grid_pred_gt_orig(batch, preds, names=names, max_show=MAX_SHOW)
        if fig is None:
            print("No figure found")
            return

        wandb.log(
            {"predictions/val_first_batch_custom_grid": wandb.Image(fig)},
            step=wandb.run.step,
        )
        plt.close(fig)
        m = batch["masks"]
        print("masks shape:", tuple(m.shape), "dtype:", m.dtype)
        print("masks unique (sample):", np.unique(m.detach().cpu().numpy())[:20])
        print("cls shape:", None if batch.get("cls") is None else tuple(batch["cls"].shape))
        print("batch_idx shape:", None if batch.get("batch_idx") is None else tuple(batch["batch_idx"].shape))


        grids = make_per_class_grids(batch, preds, names=names, max_show=MAX_SHOW)

        for j, (cls_name, cls_fig) in enumerate(grids.items()):

            key = f"predictions/val_first_batch_class/{cls_name}"
            wandb.log({key: wandb.Image(cls_fig)}, step=wandb.run.step)
            plt.close(cls_fig)


    except Exception as e:
        logger.warning(f"custom val grid failed: {e}")


def on_val_end(validator):
    """
    After validation: log per-class segmentation metrics.
    """
    if not _wandb_ready():
        return

    epoch = _epoch_from_validator(validator)
    log = {"epoch": epoch}

    arr = _get_per_class_seg_arrays(validator)
    if arr:
        classes, P, R, AP50 = arr
        for i, name in enumerate(classes):
            name = str(name).strip()
            if name.lower() in ("background", "bg", ""):
                continue
            log[f"metrics/seg/precision/{name}"] = float(P[i])
            log[f"metrics/seg/recall/{name}"] = float(R[i])
            log[f"metrics/seg/mAP50/{name}"] = float(AP50[i])

    wandb.log(log, step=wandb.run.step)


def add_custom_callbacks(model):
    model.add_callback("on_val_batch_end", on_val_batch_end)
    model.add_callback("on_val_end", on_val_end)
