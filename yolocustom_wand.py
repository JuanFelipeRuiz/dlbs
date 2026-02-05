"""
YOLO (Ultralytics 8.3.x) + W&B callbacks (SEG) – val-dependent logging moved to on_val_end.

WHY:
In Ultralytics 8.3.x, validator.plots and validator.metrics.(seg|mask) are often finalized
only at `on_val_end(validator)`. If you read them in `on_fit_epoch_end(trainer)`, they can be empty.

What it does:

1) BEGIN of each TRAIN epoch (on_train_epoch_start):
   - logs a dummy "fictive" plot to confirm per-epoch media logging works:
     plots/debug_dummy_epoch_start

2) END of VALIDATION (on_val_end):
   - logs per-class seg metrics:
     metrics/seg/precision/<class>
     metrics/seg/recall/<class>
     metrics/seg/mAP50/<class>
   - logs first val-batch visuals (GT + Pred) if available:
     predictions/val_first_batch_gt
     predictions/val_first_batch_pred
   - logs debug signals to pinpoint what exists / what's missing:
     debug/*

3) FINAL epoch only (inside on_val_end, once metrics exist):
   - logs per-class bar plot:
     plots/final_per_class_metrics

Notes:
- Assumes wandb is installed. (No wandb-is-None checks)
- Still waits for `wandb.run` to exist (Ultralytics initializes W&B internally).
- Uses wandb.define_metric so "epoch" is the x-axis for our logged keys.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
import wandb

logger = logging.getLogger(__name__)

DEBUG = True          # set False to silence debug logs to W&B
DEBUG_PRINT = False   # set True to also print debug to stdout


# -------------------------
# helpers
# -------------------------

def _wandb_ready() -> bool:
    return wandb.run is not None


def _state(obj):
    """Attach minimal state to either trainer or validator (anything with __dict__)."""
    if not hasattr(obj, "_custom_wb"):
        obj._custom_wb = {"config_pushed": False, "metrics_defined": False}
    return obj._custom_wb


def _ensure_wandb_config(trainer_or_validator):
    if not _wandb_ready():
        return

    # trainer args live on trainer; validator often has .trainer
    trainer = getattr(trainer_or_validator, "trainer", None) or trainer_or_validator
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


def _define_epoch_metrics_once(trainer_or_validator):
    if not _wandb_ready():
        return

    trainer = getattr(trainer_or_validator, "trainer", None) or trainer_or_validator
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


def _epoch_from(trainer_or_validator) -> int:
    # validator has .trainer, which has .epoch (0-based)
    trainer = getattr(trainer_or_validator, "trainer", None) or trainer_or_validator
    return int(getattr(trainer, "epoch", 0)) + 1


def _is_final_epoch(trainer_or_validator) -> bool:
    trainer = getattr(trainer_or_validator, "trainer", None) or trainer_or_validator
    total = int(getattr(trainer, "epochs", 0) or 0)
    ep = int(getattr(trainer, "epoch", 0)) + 1
    return total > 0 and ep == total


def _to_arr(x, n):
    """Ultralytics metrics may return [] when not available."""
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
    """
    Returns (classes, P, R, AP50) or None.
    Works on validator directly (preferred for on_val_end).
    """
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
        elif ap_arr.ndim == 1 and ap_arr.size >= nc:
            AP50 = _to_arr(ap_arr[:nc], nc)

    if P is None or R is None or AP50 is None:
        return None
    return classes, P, R, AP50


def _get_first_val_batch_gt_pred(validator):
    """
    Returns (gt, pred, plots_dict).
    In Ultralytics 8.3.x plots are commonly file paths.
    """
    plots = getattr(validator, "plots", None) or {}

    gt = plots.get("val_batch0_labels") or plots.get("val_batch_labels")
    pred = plots.get("val_batch0_pred") or plots.get("val_batch_pred")

    # permissive fallbacks
    if pred is None:
        pred = plots.get("val_batch0") or plots.get("val_batch")

    return gt, pred, plots


def _plot_per_class_bar(classes, P, R, AP50):
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
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.plot([0, 1], [0, 1], marker="o")
    ax.set_title(f"DEBUG dummy plot @ epoch {epoch}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def _info(x):
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
    Begin of train epoch: log a dummy plot to verify per-epoch media logging works.
    """
    if not _wandb_ready():
        return

    _ensure_wandb_config(trainer)
    _define_epoch_metrics_once(trainer)

    epoch = _epoch_from(trainer)
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


def on_val_end(validator):
    """
    End of validation: THIS is where Ultralytics 8.3.x usually has plots+metrics ready.
    We log val-dependent stuff here.
    """
    if not _wandb_ready():
        return

    _ensure_wandb_config(validator)
    _define_epoch_metrics_once(validator)

    epoch = _epoch_from(validator)
    log = {"epoch": epoch}

    try:
        # --- per-class seg scalars (validation-derived)
        arr = _get_per_class_seg_arrays_from_validator(validator)
        if arr:
            classes, P, R, AP50 = arr
            for i, name in enumerate(classes):
                name = str(name).strip()
                if name.lower() in ("background", "bg", ""):
                    continue
                log[f"metrics/seg/precision/{name}"] = float(P[i])
                log[f"metrics/seg/recall/{name}"] = float(R[i])
                log[f"metrics/seg/mAP50/{name}"] = float(AP50[i])

            # final epoch: per-class bar plot (now we *know* arrays exist)
            if _is_final_epoch(validator):
                fig = _plot_per_class_bar(classes, P, R, AP50)
                if fig is not None:
                    log["plots/final_per_class_metrics"] = wandb.Image(fig, caption="Final per-class metrics")
                    plt.close(fig)

        # --- val batch visuals (GT + Pred)
        gt, pred, plots = _get_first_val_batch_gt_pred(validator)
        if gt is not None:
            log["predictions/val_first_batch_gt"] = wandb.Image(gt, caption=f"GT epoch {epoch}")
        if pred is not None:
            log["predictions/val_first_batch_pred"] = wandb.Image(pred, caption=f"Pred epoch {epoch}")

        # --- DEBUG signals
        if DEBUG:
            m = getattr(validator, "metrics", None)
            seg = getattr(m, "seg", None) if m else None
            mask = getattr(m, "mask", None) if m else None
            container = seg or mask

            log["debug/has_validator"] = 1
            log["debug/has_metrics"] = int(m is not None)
            log["debug/has_seg"] = int(seg is not None)
            log["debug/has_mask"] = int(mask is not None)

            keys = sorted(list((plots or {}).keys()))
            log["debug/plots_keys"] = ", ".join(keys[:80])

            log["debug/gt_key_present"] = int(any(k in (plots or {}) for k in ("val_batch0_labels", "val_batch_labels")))
            log["debug/pred_key_present"] = int(any(k in (plots or {}) for k in ("val_batch0_pred", "val_batch_pred", "val_batch0", "val_batch")))

            if container is not None:
                log["debug/p_info"] = _info(getattr(container, "p", None))
                log["debug/r_info"] = _info(getattr(container, "r", None))
                log["debug/ap50_info"] = _info(getattr(container, "ap50", None))
                log["debug/ap_info"] = _info(getattr(container, "ap", None))
            else:
                log["debug/p_info"] = "no(seg|mask)"
                log["debug/r_info"] = "no(seg|mask)"
                log["debug/ap50_info"] = "no(seg|mask)"
                log["debug/ap_info"] = "no(seg|mask)"

            if DEBUG_PRINT:
                print(f"[DEBUG] on_val_end epoch={epoch}")
                print(f"[DEBUG] plots keys (head): {keys[:25]}")
                print(f"[DEBUG] seg={type(seg)} mask={type(mask)}")
                print(f"[DEBUG] p={log['debug/p_info']} r={log['debug/r_info']} ap50={log['debug/ap50_info']} ap={log['debug/ap_info']}")

    except Exception as e:
        logger.exception(f"Custom on_val_end crashed at epoch {epoch}: {e}")
        if DEBUG:
            log["debug/callback_crashed"] = 1
            log["debug/callback_error"] = str(e)[:500]

    wandb.log(log)


def add_custom_callbacks(model):
    callbacks = {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_train_epoch_start": on_train_epoch_start,  # dummy plot here
        "on_val_end": on_val_end,                      # VAL-dependent logging here
    }
    for event, cb in callbacks.items():
        model.add_callback(event, cb)
