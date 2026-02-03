"""
Custom YOLO Training with Weights & Biases Integration

Logs ONLY per-class segmentation metrics (validator.metrics.seg):
- metrics/seg/precision/<class>
- metrics/seg/recall/<class>
- metrics/seg/mAP50/<class>

And logs YOUR plots (per-class bar plot + loss components plot + optional samples)
ONLY at the final epoch to avoid empty media/tables.

Usage:
    from yolocustom_wand import add_custom_callbacks
    from ultralytics import YOLO

    model = YOLO("yolo11n-seg.pt")
    add_custom_callbacks(model)
    model.train(data="dataset_yolo/data.yaml", epochs=50)
"""

import logging
from collections import defaultdict
from typing import Dict, Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

import wandb
import seaborn as sns


_metrics_history = defaultdict(list)
_wandb_config_pushed = False

# Store per-class history over epochs for W&B line charts
# metric -> class_name -> list of (epoch, value)
_per_class_history = {
    "precision": defaultdict(list),
    "recall": defaultdict(list),
    "map50": defaultdict(list),
}


def _ensure_wandb_config(trainer):
    """
    Push config once, after wandb.run exists, to avoid init races.
    """
    global _wandb_config_pushed
    if _wandb_config_pushed:
        return
    if wandb is None or wandb.run is None:
        return
    if not hasattr(trainer, "args"):
        _wandb_config_pushed = True
        return
    try:
        wandb.config.update(vars(trainer.args), allow_val_change=True)
        _wandb_config_pushed = True
    except Exception as e:
        logger.warning(f"Could not update wandb config: {e}")


def _is_final_epoch(trainer):
    """
    True if current epoch is the final one (based on trainer.epochs).
    """
    total_epochs = int(getattr(trainer, "epochs", 0) or 0)
    step = int(getattr(trainer, "epoch", 0)) + 1
    return total_epochs > 0 and step == total_epochs


def get_per_class_seg_arrays(trainer):
    """
    Fetch per-class precision, recall, and AP50 arrays from validator.metrics.seg.

    Returns:
        (classes, P, R, AP50) or None if unavailable or not ready.
    """
    v = getattr(trainer, "validator", None)
    if v is None:
        return None

    names_dict = getattr(v, "names", None) or {}
    if not names_dict:
        return None

    m = getattr(v, "metrics", None)
    if m is None:
        return None

    seg = getattr(m, "seg", None)
    if seg is None:
        return None

    p = getattr(seg, "p", None)
    r = getattr(seg, "r", None)
    ap50 = getattr(seg, "ap50", None)
    ap = getattr(seg, "ap", None)  # fallback

    # If Ultralytics hasn't populated arrays yet, skip (do NOT log fake zeros)
    if p is None and r is None and ap50 is None and ap is None:
        return None

    classes = [names_dict[i] for i in sorted(names_dict.keys())]
    nc = len(classes)

    def to_arr(x):
        if x is None:
            return None
        x = np.asarray(x, dtype=float)
        if x.ndim == 0:
            return None
        x = x[:nc]
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    P = to_arr(p)
    R = to_arr(r)

    if ap50 is not None:
        AP50 = to_arr(ap50)
    elif ap is not None:
        ap = np.asarray(ap, dtype=float)
        if ap.ndim == 2 and ap.shape[0] >= nc and ap.shape[1] >= 1:
            AP50 = np.nan_to_num(ap[:nc, 0], nan=0.0, posinf=0.0, neginf=0.0)
        else:
            AP50 = None
    else:
        AP50 = None

    if P is None or R is None or AP50 is None:
        return None

    return classes, P, R, AP50


def create_per_class_metrics_plot(trainer):
    """
    Combined bar plot for per-class Precision, Recall, AP50 (seg only).
    """
    arr = get_per_class_seg_arrays(trainer)
    if not arr:
        return None

    classes, P, R, AP50 = arr

    keep = [i for i, n in enumerate(classes) if str(n).lower() not in ("background", "bg")]
    if not keep:
        return None

    classes = [classes[i] for i in keep]
    P = P[keep]
    R = R[keep]
    AP50 = AP50[keep]

    if (np.max(P) <= 0.0) and (np.max(R) <= 0.0) and (np.max(AP50) <= 0.0):
        return None

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(classes))
    w = 0.25

    ax.bar(x - w, P, w, label="Precision", alpha=0.8)
    ax.bar(x, R, w, label="Recall", alpha=0.8)
    ax.bar(x + w, AP50, w, label="mAP50", alpha=0.8)

    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Per-Class Segmentation Metrics", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim([0, 1.1])
    plt.tight_layout()
    return fig


def create_loss_components_plot():
    """
    Plot training loss components over epochs from _metrics_history.
    """
    loss_keys = [k for k in _metrics_history.keys() if "loss" in k.lower()]
    if not loss_keys:
        return None
    if not any(len(_metrics_history[k]) > 0 for k in loss_keys):
        return None

    fig, ax = plt.subplots(figsize=(12, 6))
    for key in sorted(loss_keys):
        values = _metrics_history[key]
        if not values:
            continue
        epochs = range(1, len(values) + 1)
        ax.plot(epochs, values, label=key.replace("train/", ""), marker="o", markersize=3)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Loss Components Over Time", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def create_prediction_samples(trainer):
    """
    Return a list of wandb.Image examples if validator provides plots['val_batch'].
    Logged only at final epoch.
    """
    if wandb is None:
        return []

    validator = getattr(trainer, "validator", None)
    if validator is None:
        return []

    plots = getattr(validator, "plots", None)
    if not plots or "val_batch" not in plots:
        return []

    val_batch_plot = plots.get("val_batch")
    if val_batch_plot is None:
        return []

    try:
        return [wandb.Image(val_batch_plot, caption="Validation Batch Predictions")]
    except Exception as e:
        logger.warning(f"Could not create prediction sample: {e}")
        return []


def _append_per_class_history(step, classes, P, R, AP50):
    """
    Save per-class values per epoch for later W&B line charts.
    """
    for i, name in enumerate(classes):
        if str(name).lower() in ("background", "bg"):
            continue
        _per_class_history["precision"][name].append((step, float(P[i])))
        _per_class_history["recall"][name].append((step, float(R[i])))
        _per_class_history["map50"][name].append((step, float(AP50[i])))


def _log_wandb_line_charts_from_history(step):
    """
    Build one W&B Table with epoch + one column per class for each metric,
    then create line charts where hue=class (series per class).
    """
    if wandb is None or wandb.run is None:
        return

    def build_series_table(metric_name):
        series = _per_class_history[metric_name]
        if not series:
            return None, None

        class_names = sorted(series.keys())
        if not class_names:
            return None, None

        # collect epochs from the first class (all should have same epochs, but be robust)
        all_epochs = sorted(set(e for cls in class_names for (e, _) in series[cls]))
        if not all_epochs:
            return None, None

        columns = ["epoch"] + class_names
        table = wandb.Table(columns=columns)

        # for each epoch, fill row values per class (fallback nan -> 0)
        for ep in all_epochs:
            row = [ep]
            for cls in class_names:
                d = dict(series[cls])
                v = d.get(ep, np.nan)
                if np.isnan(v):
                    v = 0.0
                row.append(float(v))
            table.add_data(*row)

        return table, class_names

    for metric_name, title, y_label in [
        ("precision", "Seg Precision per Class over Epochs", "Precision"),
        ("recall", "Seg Recall per Class over Epochs", "Recall"),
        ("map50", "Seg mAP50 per Class over Epochs", "mAP50"),
    ]:
        table, class_names = build_series_table(metric_name)
        if table is None:
            continue

        try:
            # Create a multi-series line plot: each class is one series (hue)
            chart = wandb.plot.line_series(
                table,
                x="epoch",
                y=class_names,
                title=title,
            )
            wandb.log({f"plots/seg_{metric_name}_over_epochs": chart}, step=step)
        except Exception as e:
            logger.warning(f"Could not create W&B line_series for {metric_name}: {e}")


def on_pretrain_routine_start(trainer):
    """
    Reset state at the very beginning of training.
    """
    global _wandb_config_pushed
    _metrics_history.clear()
    _wandb_config_pushed = False

    for k in _per_class_history.keys():
        _per_class_history[k].clear()

    if wandb is None:
        logger.warning("wandb not installed. Custom wandb logging will be disabled.")
        return

    _ensure_wandb_config(trainer)
    logger.info("Custom wandb callbacks initialized")


def on_train_epoch_end(trainer):
    """
    Keep a history of train loss components for plotting later.
    """
    if hasattr(trainer, "tloss"):
        try:
            loss_dict = trainer.label_loss_items(trainer.tloss, prefix="train/")
            for k, v in loss_dict.items():
                _metrics_history[k].append(float(v))
        except Exception as e:
            logger.debug(f"on_train_epoch_end: could not record loss items: {e}")


def on_fit_epoch_end(trainer):
    """
    Log ONLY per-class seg metrics as scalars each epoch (when available),
    and log plots only at final epoch.
    """
    if wandb is None or wandb.run is None:
        return

    _ensure_wandb_config(trainer)
    step = int(trainer.epoch) + 1

    # Per-class seg arrays are VALIDATION metrics
    arr = get_per_class_seg_arrays(trainer)
    if arr:
        classes, P, R, AP50 = arr

        # store history for final metric-over-time charts
        _append_per_class_history(step, classes, P, R, AP50)

        # (optional) also log per-class scalars each epoch
        # If you want ZERO scalar spam, comment this out.
        metrics_to_log = {}
        for i, name in enumerate(classes):
            if str(name).lower() in ("background", "bg"):
                continue
            metrics_to_log[f"metrics/seg/precision/{name}"] = float(P[i])
            metrics_to_log[f"metrics/seg/recall/{name}"] = float(R[i])
            metrics_to_log[f"metrics/seg/mAP50/{name}"] = float(AP50[i])
        if metrics_to_log:
            wandb.log(metrics_to_log, step=step)

    # YOUR plots only at final epoch
    if _is_final_epoch(trainer):
        try:
            per_class_fig = create_per_class_metrics_plot(trainer)
            if per_class_fig is not None:
                wandb.log({"plots/final_per_class_metrics": wandb.Image(per_class_fig)}, step=step)
                plt.close(per_class_fig)
        except Exception as e:
            logger.warning(f"Could not create per-class metrics plot: {e}")

        try:
            loss_fig = create_loss_components_plot()
            if loss_fig is not None:
                wandb.log({"plots/final_loss_components": wandb.Image(loss_fig)}, step=step)
                plt.close(loss_fig)
        except Exception as e:
            logger.warning(f"Could not create loss components plot: {e}")

        try:
            samples = create_prediction_samples(trainer)
            if samples:
                wandb.log({"predictions/final_val_samples": samples}, step=step)
        except Exception as e:
            logger.warning(f"Could not create prediction samples: {e}")

        # NEW: grouped per metric, hue=class
        try:
            _log_wandb_line_charts_from_history(step=step)
        except Exception as e:
            logger.warning(f"Could not log metric-over-epochs charts: {e}")

    logger.debug(f"Logged custom per-class seg metrics for epoch(step)={step}")


def on_val_end(validator):
    """
    Hook left minimal to avoid duplicates.
    """
    pass


def on_train_end(trainer):
    """
    Final logging at the end of training run (artifact etc.).
    """
    if wandb is None or wandb.run is None:
        return

    _ensure_wandb_config(trainer)

    try:
        if hasattr(trainer, "best") and trainer.best:
            wandb.log_artifact(str(trainer.best), name="best_model", type="model")
    except Exception as e:
        logger.warning(f"Could not log best model artifact: {e}")

    logger.info("Custom wandb logging completed")


def add_custom_callbacks(model):
    """
    Attach all custom callbacks to a YOLO model.
    """
    callbacks = {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_train_epoch_end": on_train_epoch_end,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_val_end": on_val_end,
        "on_train_end": on_train_end,
    }
    for event, cb in callbacks.items():
        try:
            model.add_callback(event, cb)
            logger.debug(f"Added callback {cb.__name__} to {event}")
        except Exception as e:
            logger.warning(f"Could not add callback {cb.__name__} to {event}: {e}")
    logger.info("Custom callbacks added to model")


if __name__ == "__main__":
    import argparse
    import yaml
    from ultralytics import YOLO

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Train YOLO with custom wandb callbacks")
    parser.add_argument("--cfg", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--model", type=str, default="yolo11n-seg.pt", help="Model path")
    args = parser.parse_args()

    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)

    model = YOLO(args.model)
    add_custom_callbacks(model)
    model.train(**cfg)

    logger.info("Training completed with custom wandb logging")