import logging
import inspect
import numpy as np
import wandb

logger = logging.getLogger(__name__)

DEBUG = True


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


def _get_per_class_seg_arrays_from_validator(validator):
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


# -------------------------
# CALLBACKS
# -------------------------

def on_val_batch_end(validator):
    """
    Docs-style: access locals from the validator loop frame and call validator plotting.
    We only plot the FIRST validation batch (batch_i == 0).
    """
    if not _wandb_ready():
        return

    try:
        # exactly like Ultralytics docs: go two frames back to get locals
        frame = inspect.currentframe().f_back.f_back
        v = frame.f_locals

        batch_i = v.get("batch_i", None)
        if batch_i is None or int(batch_i) != 0:
            return  # only first batch

        batch = v.get("batch", None)
        preds = v.get("preds", None)
        if batch is None or preds is None:
            return

        # This will generate and save GT + Pred images in the validator's save_dir
        # and typically populate validator.plots as well.
        validator.plot_val_samples(batch, batch_i)
        validator.plot_predictions(batch, preds, batch_i)

        if DEBUG:
            # Log a tiny debug marker at CURRENT global step (no out-of-order from our side)
            wandb.log({"custom/debug/val_first_batch_plotted": 1}, step=wandb.run.step)

    except Exception as e:
        logger.exception(f"on_val_batch_end failed: {e}")


def on_val_end(validator):
    """
    After validation: metrics are ready here (docs say use on_val_end for validation logic).
    Log per-class seg metrics at the CURRENT global step to avoid step warnings.
    """
    if not _wandb_ready():
        return

    epoch = _epoch_from_validator(validator)

    log = {
        "custom/epoch": epoch,
    }

    arr = _get_per_class_seg_arrays_from_validator(validator)
    if arr:
        classes, P, R, AP50 = arr
        for i, name in enumerate(classes):
            name = str(name).strip()
            if name.lower() in ("background", "bg", ""):
                continue
            log[f"custom/metrics/seg/precision/{name}"] = float(P[i])
            log[f"custom/metrics/seg/recall/{name}"] = float(R[i])
            log[f"custom/metrics/seg/mAP50/{name}"] = float(AP50[i])

    if DEBUG:
        plots = getattr(validator, "plots", None) or {}
        log["custom/debug/plots_keys"] = ", ".join(sorted(list(plots.keys()))[:80])
        log["custom/debug/has_seg_or_mask"] = int(_metric_container(validator) is not None)

    # IMPORTANT: log at current global step to avoid "out of order" from our callback
    wandb.log(log, step=wandb.run.step)


def add_custom_callbacks(model):
    model.add_callback("on_val_batch_end", on_val_batch_end)
    model.add_callback("on_val_end", on_val_end)
