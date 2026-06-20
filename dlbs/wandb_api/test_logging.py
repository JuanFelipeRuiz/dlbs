import wandb

from dlbs.wandb_api.custom_callback_yolo import (
    _collect_overall_metrics,
    _collect_per_class_metrics,
)


def start_wandb_run(cfg: dict, model_path: str, project, entity):
    """Start or capture the W&B run so test metrics can resume the same run later."""
    run = wandb.run
    started_here = False

    if run is None:
        run = wandb.init(
            project=project,
            entity=entity,
            name=cfg.get("name"),
            config=cfg | {"model": model_path},
            reinit=False,
        )
        started_here = True

    return {
        "id": run.id,
        "project": project or run.project,
        "entity": entity or run.entity,
    }, started_here


def _ensure_wandb_run(run_meta: dict):
    if wandb.run is not None:
        return False

    wandb.init(
        id=run_meta["id"],
        project=run_meta.get("project"),
        entity=run_meta.get("entity"),
        resume="allow",
        reinit=True,
    )
    return True


def _to_float(value):
    if value is None:
        return None
    return float(value)


def _collect_metric_family(metrics, family: str, prefix: str) -> dict:
    container = getattr(metrics, family, None)
    if container is None:
        return {}

    names = {
        "mp": "precision",
        "mr": "recall",
        "map50": "mAP50",
        "map": "mAP50_95",
    }
    log = {}
    for attr, suffix in names.items():
        value = _to_float(getattr(container, attr, None))
        if value is not None:
            log[f"{prefix}/{family}_{suffix}"] = value
    return log


def collect_test_metrics(metrics, prefix: str) -> dict:
    """Collect test metrics in the same namespace style as the custom callbacks."""
    log = {}
    log.update(_collect_metric_family(metrics, "box", prefix))
    log.update(_collect_overall_metrics(metrics, split=prefix))
    log.update(_collect_per_class_metrics(metrics, split=prefix))
    return log


def log_test_metrics(metrics, prefix: str, run_meta: dict, finish_after_log: bool, extra: dict | None = None):
    """Log test metrics to the current or resumed W&B run.

    ``extra`` holds optional pre-computed metrics (e.g. stratified city/size buckets)
    that are merged into the same log call so the run is finished only once.
    """
    log = collect_test_metrics(metrics, prefix=prefix)
    if extra:
        log.update(extra)
    if not log:
        raise ValueError("No test metrics found to log to W&B.")

    resumed_run = _ensure_wandb_run(run_meta)
    wandb.log(log)

    for key, value in log.items():
        wandb.run.summary[key] = value

    print(f"OK: Logged {len(log)} test metrics to W&B run.")

    if resumed_run or finish_after_log:
        wandb.finish()
