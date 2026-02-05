"""
Minimal YOLO (Ultralytics) + W&B callbacks (SEG) with clean epoch-based logging.

Each validation epoch:
- logs per-class seg metrics as scalars:
  metrics/seg/precision/<class>
  metrics/seg/recall/<class>
  metrics/seg/mAP50/<class>
- logs first validation batch visualization (GT + Pred masks as separate images):
  predictions/val_first_batch_gt
  predictions/val_first_batch_pred

Final epoch only:
- logs per-class bar plot (P/R/mAP50):
  plots/final_per_class_metrics

Notes:
- Assumes wandb is installed.
- Waits for wandb.run to exist (Ultralytics initializes W&B internally).
- Uses W&B define_metric so "epoch" is the x-axis for our logged keys.
- Logs exactly once per epoch (single wandb.log call, commit=True default).
- Does NOT log best model artifacts (Ultralytics already logs best.pt/last.pt).
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
import wandb

logger = logging.getLogger(__name__)


# -------------------------
# small trainer state
# -------------------------

def _state(trainer):
    if not hasattr(trainer, "_custom_wb"):
        trainer._custom_wb = {
            "config_pushed": False,
            "metrics_defined": False,
        }
    return trainer._custom_wb


def _wandb_ready() -> bool:
    # Ultralytics does wandb.init() internally; callbacks may run before that.
    return wandb.run is not None


def _ensure_wandb_config(trainer):
    if not _wandb_ready():
        return
    st = _state(trainer)
    if st["config_pushed"]:
        return

    args = getattr(trainer, "args", None)
    if args is None:
        st["config_pushed"] = True
        return

    try:
        wandb.config.update(vars(args), allow_val_change=True)
    except Exception as e:
        logger.warning(f"wandb.config.update failed: {e}")
    finally:
        st["config_pushed"] = True


def _define_epoch_metrics_once(trainer):
    if not _wandb_ready():
        return
    st = _state(trainer)
    if st["metrics_defined"]:
        return

    try:
        wandb.define_metric("epoch")
        wandb.define_metric("metrics/seg/*", step_metric="epoch")
        wandb.define_metric("predictions/*", step_metric="epoch")
        wandb.define_metric("plots/*", step_metric="epoch")
    except Exception as e:
        # Not fatal; logging still works, but x-axis might be default W&B step.
        logger.warning(f"wandb.define_metric failed: {e}")
    finally:
        st["metrics_defined"] = True


def _is_final_epoch(trainer) -> bool:
    total = int(getattr(trainer, "epochs", 0) or 0)
    ep = int(getattr(trainer, "epoch", 0)) + 1
    return total > 0 and ep == total


# -------------------------
# robust array conversion (handles [] in Ultralytics 8.3.x)
# -------------------------

def _to_arr(x, n):
    """
    Convert metric output to (n,) float array, or None.
    Ultralytics metrics may return [] when not available.
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


# -------------------------
# read per-class metrics from validator
# -------------------------

def _get_per_class_seg_arrays(trainer):
    """
    Returns (classes, P, R, AP50) or None.
    Tries metrics.seg first, then metrics.mask (Ultralytics version-dependent).
    """
    v = getattr(trainer, "validator", None)
    if v is None:
        return None

    names = getattr(v, "names", None) or {}
    if not names:
        return None

    m = getattr(v, "metrics", None)
    if m is None:
        return None

    seg = getattr(m, "seg", None) or getattr(m, "mask", None)
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
        # common: ap is (nc, 10) or similar; column 0 corresponds to AP50
        if ap_arr.ndim == 2 and ap_arr.shape[0] >= nc and ap_arr.shape[1] >= 1:
            AP50 = _to_arr(ap_arr[:nc, 0], nc)
        # sometimes ap is already (nc,)
        elif ap_arr.ndim == 1 and ap_arr.size >= nc:
            AP50 = _to_arr(ap_arr[:nc], nc)

    if P is None or R is None or AP50 is None:
        return None

    return classes, P, R, AP50


# -------------------------
# plotting
# -------------------------

def _plot_per_class_bar(trainer):
    arr = _get_per_class_seg_arrays(trainer)
    if not arr:
        return None

    classes, P, R, AP50 = arr
    keep = [i for i, n in enumerate(classes) if str(n).strip().lower() not in ("background", "bg")]
    if not keep:
        return None

    classes = [classes[i] for i in keep]
    P, R, AP50 = P[keep], R[keep], AP50[keep]

    if max(P.max(), R.max(), AP50.max()) <= 0:
        return None

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(classes))
    w = 0.25

    ax.bar(x - w, P, w, label="Precision", alpha=0.85)
    ax.bar(x,     R, w, label="Recall",    alpha=0.85)
    ax.bar(x + w, AP50, w, label="mAP50",  alpha=0.85)

    ax.set_title("Per-Class Segmentation Metrics")
    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    return fig


# -------------------------
# val batch plots (GT vs Pred)
# -------------------------

def _get_first_val_batch_gt_pred(trainer):
    """
    Returns (gt, pred) plots for the first validation batch if available.
    In Ultralytics 8.3.x these are usually file paths.
    """
    v = getattr(trainer, "validator", None)
    if v is None:
        return None, None

    plots = getattr(v, "plots", None) or {}
    if not plots:
        return None, None

    # Preferred explicit keys
    gt = plots.get("val_batch0_labels") or plots.get("val_batch_labels")
    pred = plots.get("val_batch0_pred") or plots.get("val_batch_pred")

    # permissive fallbacks
    if pred is None:
        pred = plots.get("val_batch0") or plots.get("val_batch")

    return gt, pred


# -------------------------
# callbacks
# -------------------------

def on_pretrain_routine_start(trainer):
    st = _state(trainer)
    st["config_pushed"] = False
    st["metrics_defined"] = False


def on_fit_epoch_end(trainer):
    # If Ultralytics hasn't initialized W&B yet, we can't log.
    if not _wandb_ready():
        return

    _ensure_wandb_config(trainer)
    _define_epoch_metrics_once(trainer)

    epoch = int(getattr(trainer, "epoch", 0)) + 1
    log = {"epoch": epoch}

    # per-class seg scalars (validation-derived)
    try:
        arr = _get_per_class_seg_arrays(trainer)
        if arr:
            classes, P, R, AP50 = arr
            for i, name in enumerate(classes):
                name = str(name).strip()
                if name.lower() in ("background", "bg", ""):
                    continue
                log[f"metrics/seg/precision/{name}"] = float(P[i])
                log[f"metrics/seg/recall/{name}"] = float(R[i])
                log[f"metrics/seg/mAP50/{name}"] = float(AP50[i])
    except Exception as e:
        logger.warning(f"Per-class seg metric logging failed at epoch {epoch}: {e}")

    # first val batch visualization (every epoch): GT + Pred
    try:
        gt, pred = _get_first_val_batch_gt_pred(trainer)
        if gt is not None:
            log["predictions/val_first_batch_gt"] = wandb.Image(gt, caption=f"GT epoch {epoch}")
        if pred is not None:
            log["predictions/val_first_batch_pred"] = wandb.Image(pred, caption=f"Pred epoch {epoch}")
    except Exception as e:
        logger.warning(f"Val batch image logging failed at epoch {epoch}: {e}")

    # final epoch plot
    if _is_final_epoch(trainer):
        try:
            fig = _plot_per_class_bar(trainer)
            if fig is not None:
                log["plots/final_per_class_metrics"] = wandb.Image(fig, caption="Final per-class metrics")
                plt.close(fig)
        except Exception as e:
            logger.warning(f"Final per-class plot failed: {e}")

    # Single log call per epoch (commit=True default)
    wandb.log(log)


def add_custom_callbacks(model):
    callbacks = {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_fit_epoch_end": on_fit_epoch_end,
    }
    for event, cb in callbacks.items():
        model.add_callback(event, cb)
