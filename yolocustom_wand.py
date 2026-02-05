"""
YOLO (Ultralytics 8.3.x) + W&B callbacks (SEG) with DEBUG mode.

What it does:
- At BEGIN of each TRAIN epoch:
  - logs a "dummy" (fictive) plot so you can verify per-epoch media logging works:
    plots/debug_dummy_epoch_start

- At END of each FIT epoch (i.e., after validation):
  - logs per-class seg metrics:
    metrics/seg/precision/<class>
    metrics/seg/recall/<class>
    metrics/seg/mAP50/<class>
  - logs first val batch GT + Pred plots (if available):
    predictions/val_first_batch_gt
    predictions/val_first_batch_pred

DEBUG mode additionally logs (to W&B + optionally prints):
- debug/plots_keys
- debug/has_validator, debug/has_metrics, debug/has_seg, debug/has_mask
- debug/p_info, debug/r_info, debug/ap50_info, debug/ap_info
- debug/gt_key_present, debug/pred_key_present

Notes:
- Assumes wandb is installed.
- Waits for wandb.run to exist (Ultralytics initializes W&B internally).
- Uses wandb.define_metric so "epoch" is the x-axis for our logged keys.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
import wandb

logger = logging.getLogger(__name__)

DEBUG = True          # set False to silence debug logs
DEBUG_PRINT = False   # set True if you also want stdout prints


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
        wandb.define_metric("debug/*", step_metric="epoch")
    except Exception as e:
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
# per-class seg metrics from validator
# -------------------------

def _get_per_class_seg_arrays(trainer):
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
        if ap_arr.ndim == 2 and ap_arr.shape[0] >= nc and ap_arr.shape[1] >= 1:
            AP50 = _to_arr(ap_arr[:nc, 0], nc)
        elif ap_arr.ndim == 1 and ap_arr.size >= nc:
            AP50 = _to_arr(ap_arr[:nc], nc)

    if P is None or R is None or AP50 is None:
        return None

    return classes, P, R, AP50


# -------------------------
# val batch GT vs Pred (8.3.x common keys)
# -------------------------

def _get_first_val_batch_gt_pred(trainer):
    v = getattr(trainer, "validator", None)
    if v is None:
        return None, None, {}, None

    plots = getattr(v, "plots", None) or {}
    if not plots:
        return None, None, plots, None

    # Prefer explicit first-batch keys
    gt = plots.get("val_batch0_labels") or plots.get("val_batch_labels")
    pred = plots.get("val_batch0_pred") or plots.get("val_batch_pred")

    # permissive fallbacks
    if pred is None:
        pred = plots.get("val_batch0") or plots.get("val_batch")

    # which "seg container" exists?
    m = getattr(v, "metrics", None)
    seg = getattr(m, "seg", None) if m else None
    mask = getattr(m, "mask", None) if m else None
    seg_container = seg or mask

    return gt, pred, plots, seg_container


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


def _make_dummy_plot(epoch: int):
    """Small 'fictive' plot to verify per-epoch media logging works."""
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.plot([0, 1], [0, 1], marker="o")
    ax.set_title(f"DEBUG dummy plot @ epoch {epoch}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def _info(x):
    """Describe a metric object for debug logs."""
    if x is None:
        return "None"
    if isinstance(x, (list, tuple)):
        return f"{type(x).__name__}(len={len(x)})"
    try:
        a = np.asarray(x)
        return f"ndarray(shape={a.shape}, dtype={a.dtype})"
    except Exception:
        return f"type={type(x).__name__}"


# -------------------------
# callbacks
# -------------------------

def on_pretrain_routine_start(trainer):
    st = _state(trainer)
    st["config_pushed"] = False
    st["metrics_defined"] = False


def on_train_epoch_start(trainer):
    """
    Called at the beginning of each train epoch.
    Logs a dummy plot so you can confirm per-epoch logging works even before val.
    """
    if not _wandb_ready():
        return

    _ensure_wandb_config(trainer)
    _define_epoch_metrics_once(trainer)

    epoch = int(getattr(trainer, "epoch", 0)) + 1
    log = {"epoch": epoch}

    try:
        fig = _make_dummy_plot(epoch)
        log["plots/debug_dummy_epoch_start"] = wandb.Image(fig, caption=f"epoch {epoch} start")
        plt.close(fig)
    except Exception as e:
        logger.warning(f"Dummy plot logging failed at epoch {epoch}: {e}")

    if DEBUG:
        log["debug/epoch_start_seen"] = 1

    wandb.log(log)


def on_fit_epoch_end(trainer):
    """
    Called after validation each epoch.
    Logs per-class seg metrics + GT/PRED first val batch visuals.
    Also logs debug signals to pinpoint what is missing.
    """
    if not _wandb_ready():
        return

    _ensure_wandb_config(trainer)
    _define_epoch_metrics_once(trainer)

    epoch = int(getattr(trainer, "epoch", 0)) + 1
    log = {"epoch": epoch}

    try:
        # --- per-class seg scalars
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

        # --- first val batch visuals (GT + Pred)
        gt, pred, plots, seg_container = _get_first_val_batch_gt_pred(trainer)
        if gt is not None:
            log["predictions/val_first_batch_gt"] = wandb.Image(gt, caption=f"GT epoch {epoch}")
        if pred is not None:
            log["predictions/val_first_batch_pred"] = wandb.Image(pred, caption=f"Pred epoch {epoch}")

        # --- final epoch plot
        if _is_final_epoch(trainer):
            fig = _plot_per_class_bar(trainer)
            if fig is not None:
                log["plots/final_per_class_metrics"] = wandb.Image(fig, caption="Final per-class metrics")
                plt.close(fig)

        # --- DEBUG signals
        if DEBUG:
            v = getattr(trainer, "validator", None)
            m = getattr(v, "metrics", None) if v else None
            seg = getattr(m, "seg", None) if m else None
            mask = getattr(m, "mask", None) if m else None

            log["debug/has_validator"] = int(v is not None)
            log["debug/has_metrics"] = int(m is not None)
            log["debug/has_seg"] = int(seg is not None)
            log["debug/has_mask"] = int(mask is not None)

            keys = sorted(list((plots or {}).keys()))
            log["debug/plots_keys"] = ", ".join(keys[:60])  # keep it short

            # show presence of key candidates
            log["debug/gt_key_present"] = int(any(k in (plots or {}) for k in ("val_batch0_labels", "val_batch_labels")))
            log["debug/pred_key_present"] = int(any(k in (plots or {}) for k in ("val_batch0_pred", "val_batch_pred", "val_batch0", "val_batch")))

            # metric arrays info
            target = seg_container
            if target is not None:
                log["debug/p_info"] = _info(getattr(target, "p", None))
                log["debug/r_info"] = _info(getattr(target, "r", None))
                log["debug/ap50_info"] = _info(getattr(target, "ap50", None))
                log["debug/ap_info"] = _info(getattr(target, "ap", None))
            else:
                log["debug/p_info"] = "no(seg|mask)"
                log["debug/r_info"] = "no(seg|mask)"
                log["debug/ap50_info"] = "no(seg|mask)"
                log["debug/ap_info"] = "no(seg|mask)"

            if DEBUG_PRINT:
                print(f"[DEBUG] epoch={epoch} keys={keys[:20]}")
                print(f"[DEBUG] seg={type(seg)} mask={type(mask)}")
                print(f"[DEBUG] p={log['debug/p_info']} r={log['debug/r_info']} ap50={log['debug/ap50_info']} ap={log['debug/ap_info']}")

    except Exception as e:
        logger.exception(f"Custom callback crashed at epoch {epoch}: {e}")
        # Still log something so you see the epoch reached fit-end
        if DEBUG:
            log["debug/callback_crashed"] = 1
            log["debug/callback_error"] = str(e)[:500]

    wandb.log(log)


def add_custom_callbacks(model):
    callbacks = {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_train_epoch_start": on_train_epoch_start,  # dummy plot here
        "on_fit_epoch_end": on_fit_epoch_end,          # val-dependent stuff here
    }
    for event, cb in callbacks.items():
        model.add_callback(event, cb)