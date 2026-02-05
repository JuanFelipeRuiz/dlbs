"""
Custom YOLO Training steps with Weights & Biases Integration

Each validation epoch logs additional metrics to W&B:
- logs per-class seg metrics as scalars: metrics/seg/precision/<class>, metrics/seg/recall/<class>, metrics/seg/mAP50/<class>
- logs first validation batch visualization (GT vs Pred masks): media/val_first_batch

Final epoch only:
- logs per-class bar plot (P/R/mAP50): plots/final_per_class_metrics

Notes:
- Assumes W&B is installed and initialized by Ultralytics.
- Does NOT log best model artifact (Ultralytics already handles best.pt/last.pt artifacts).
- Uses W&B define_metric so "epoch" is the x-axis for everything we log.
- Logs exactly once per epoch (single wandb.log call).
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


def _ensure_wandb_config(trainer):
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
    """
    Make 'epoch' the step metric so charts are always aligned by epoch,
    regardless of Ultralytics internal step usage.
    """
    st = _state(trainer)
    if st["metrics_defined"]:
        return

    try:
        wandb.define_metric("epoch")
        wandb.define_metric("metrics/seg/*", step_metric="epoch")
        wandb.define_metric("predictions/*", step_metric="epoch")
        wandb.define_metric("plots/*", step_metric="epoch")
    except Exception as e:
        # Not fatal; logging still works, you may just get default step behavior.
        logger.warning(f"wandb.define_metric failed: {e}")
    finally:
        st["metrics_defined"] = True


def _is_final_epoch(trainer) -> bool:
    total = int(getattr(trainer, "epochs", 0) or 0)
    ep = int(getattr(trainer, "epoch", 0)) + 1
    return total > 0 and ep == total


def _nan0(x, n):
    x = np.asarray(x, dtype=float).reshape(-1)[:n]
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


# -------------------------
# read per-class metrics from validator
# -------------------------

def _get_per_class_seg_arrays(trainer):
    """
    Returns (classes, P, R, AP50) or None.
    Reads from trainer.validator.metrics.seg.{p,r,ap50} (fallback seg.ap[:,0]).
    """
    v = getattr(trainer, "validator", None)
    if v is None:
        return None

    names = getattr(v, "names", None) or {}
    if not names:
        return None

    m = getattr(v, "metrics", None)
    seg = getattr(m, "seg", None) if m is not None else None
    if seg is None:
        return None

    p = getattr(seg, "p", None)
    r = getattr(seg, "r", None)
    ap50 = getattr(seg, "ap50", None)
    ap = getattr(seg, "ap", None)

    if p is None and r is None and ap50 is None and ap is None:
        return None

    classes = [names[i] for i in sorted(names)]
    nc = len(classes)

    P = None if p is None else _nan0(p, nc)
    R = None if r is None else _nan0(r, nc)

    if ap50 is not None:
        AP50 = _nan0(ap50, nc)
    elif ap is not None:
        ap = np.asarray(ap, dtype=float)
        AP50 = _nan0(ap[:nc, 0], nc) if ap.ndim == 2 and ap.shape[1] >= 1 else None
    else:
        AP50 = None

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
    keep = [i for i, n in enumerate(classes) if str(n).lower() not in ("background", "bg")]
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


def _get_first_val_batch_plot(trainer):
    """
    Returns the validator plot image for the first val batch if available.
    Key names can differ by Ultralytics version.
    """
    v = getattr(trainer, "validator", None)
    if v is None:
        return None
    plots = getattr(v, "plots", None) or {}

    img = plots.get("val_batch")
    if img is None:
        for k in ("val_batch0", "val_batch1", "val_batch_pred", "val_batch_labels"):
            if k in plots:
                img = plots[k]
                break
    return img


# -------------------------
# callbacks
# -------------------------

def on_pretrain_routine_start(trainer):
    st = _state(trainer)
    st["config_pushed"] = False
    st["metrics_defined"] = False

    _ensure_wandb_config(trainer)
    _define_epoch_metrics_once(trainer)


def on_fit_epoch_end(trainer):
    _ensure_wandb_config(trainer)
    _define_epoch_metrics_once(trainer)

    epoch = int(getattr(trainer, "epoch", 0)) + 1
    log = {"epoch": epoch}

    # per-class seg scalars
    arr = _get_per_class_seg_arrays(trainer)
    if arr:
        classes, P, R, AP50 = arr
        for i, name in enumerate(classes):
            if str(name).lower() in ("background", "bg"):
                continue
            log[f"metrics/seg/precision/{name}"] = float(P[i])
            log[f"metrics/seg/recall/{name}"] = float(R[i])
            log[f"metrics/seg/mAP50/{name}"] = float(AP50[i])

    # first val batch visualization (every epoch)
    img = _get_first_val_batch_plot(trainer)
    if img is not None:
        log["media/val_first_batch"] = wandb.Image(img)

    # final epoch: per-class bar plot
    if _is_final_epoch(trainer):
        fig = _plot_per_class_bar(trainer)
        if fig is not None:
            log["media/final_per_class_metrics"] = wandb.Image(fig)
            plt.close(fig)

    # single log call per epoch
    wandb.log(log)


def add_custom_callbacks(model):
    callbacks = {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_fit_epoch_end": on_fit_epoch_end,
    }
    for event, cb in callbacks.items():
        model.add_callback(event, cb)
