"""
usage:
    python -m dlbs.test_model --model runs/segment/yolo-seg-baseline/weights/best.pt \
                              --data dataset.yaml
    python -m dlbs.test_model --model best.pt --data dataset.yaml --split test --name my-test
"""

import argparse

from dlbs.evaluation import run_test_validation
from dlbs.wandb_api.test_logging import log_test_metrics, start_wandb_run


def test_model(
    model_path: str,
    data_yaml: str = "dataset.yaml",
    split: str = "test",
    prefix: str | None = None,
    name: str | None = None,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    imgsz=None,
    batch=None,
    device=None,
    workers=None,
    plots: bool = True,
):
    """Validate a trained model on a split and log the metrics to W&B."""
    prefix = prefix or split

    # start_wandb_run reads `name` from cfg and stores the rest as run config.
    cfg = {"data": data_yaml, "split": split, "name": name}
    run_meta, started_here = start_wandb_run(
        cfg=cfg,
        model_path=str(model_path),
        project=wandb_project,
        entity=wandb_entity,
    )

    metrics = run_test_validation(
        model_path,
        data_yaml,
        split=split,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        plots=plots,
        name=name,
    )

    log_test_metrics(
        metrics,
        prefix=prefix,
        run_meta=run_meta,
        finish_after_log=started_here,
    )
    return metrics


def argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to weights (e.g. best.pt)")
    p.add_argument("--data", default="dataset.yaml", help="dataset.yaml")
    p.add_argument("--split", default="test", help="Dataset split to validate")
    p.add_argument("--prefix", type=str, help="W&B metric prefix (defaults to --split)")
    p.add_argument("--name", type=str, help="Run name (W&B run + val output dir)")
    p.add_argument("--wandb-project", type=str, help="W&B project")
    p.add_argument("--wandb-entity", type=str, help="W&B entity")
    p.add_argument("--imgsz", type=int)
    p.add_argument("--batch", type=int)
    p.add_argument("--device", type=str)
    p.add_argument("--workers", type=int)
    p.add_argument("--no-plots", action="store_true", help="Disable validation plots")
    return p


if __name__ == "__main__":
    args = argparser().parse_args()
    test_model(
        model_path=args.model,
        data_yaml=args.data,
        split=args.split,
        prefix=args.prefix,
        name=args.name,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        plots=not args.no_plots,
    )
