"""
usage:
    python train.py --cfg configs/baseline_no_aug.yaml
    python train.py --cfg configs/aug_default.yaml --epochs 100
    python train.py --cfg configs/aug_strong.yaml --set lr0=0.005 --set cos_lr=true
    python train.py --cfg configs/baseline_no_aug.yaml --seed 123

"""

import argparse
import os
from pathlib import Path
from ultralytics import YOLO, settings

from dlbs.utils.metrics_standard import run_test_validation
from dlbs.utils.train_helpers import (
    as_bool,
    cli_overrides,
    load_cfg,
    save_resolved_cfg,
    set_seed,
    use_mask_only_fitness,
)
from dlbs.wandb_api.custom_callback_yolo import add_custom_callbacks
from dlbs.wandb_api.test_logging import log_test_metrics, start_wandb_run


def train(cfg_path: str, overrides: dict):
    cfg = load_cfg(cfg_path)

    # Apply CLI overrides before extracting runtime-only keys.
    cfg.update({k: v for k, v in overrides.items() if v is not None})

    # Set the seed before model setup if CLI or config provides one.
    seed = cfg.get("seed", None)
    if seed is not None:
        set_seed(seed)
        cfg["seed"] = seed

    # W&B is required for this training entry point.
    cfg.pop("wandb", None)
    settings.update({"wandb": True})

    # Optionally set WANDB_PROJECT/ENTITY via YAML without coupling to Ultralytics.
    wandb_project = cfg.pop("wandb_project", None)
    wandb_entity = cfg.pop("wandb_entity", None)
    if wandb_project:
        os.environ["WANDB_PROJECT"] = str(wandb_project)
    if wandb_entity:
        os.environ["WANDB_ENTITY"] = str(wandb_entity)

    test_after_train = as_bool(cfg.pop("test_after_train", True))
    test_split = str(cfg.pop("test_split", "test"))
    test_prefix = str(cfg.pop("test_prefix", test_split))

    model_path = cfg.pop("model")
    wandb_run_meta, wandb_started_here = start_wandb_run(
        cfg=cfg,
        model_path=model_path,
        project=wandb_project,
        entity=wandb_entity,
    )

    model = YOLO(model_path)

    add_custom_callbacks(model)
    print("Custom wandb callbacks added")

    # Early stopping and best.pt selection should track mask quality only (boxes are
    # irrelevant for this instance-segmentation task). Logging is unaffected.
    use_mask_only_fitness()

    # Train
    results = model.train(**cfg)

    # Save resolved config in run dir
    save_dir = Path(model.trainer.save_dir)

    if test_after_train:
        print(
            "Running post-training validation "
            f"on data='{cfg['data']}', split='{test_split}', prefix='{test_prefix}'."
        )
        test_metrics = run_test_validation(
            save_dir / "weights" / "best.pt",
            cfg["data"],
            split=test_split,
            imgsz=cfg.get("imgsz"),
            batch=cfg.get("batch"),
            device=cfg.get("device"),
            workers=cfg.get("workers"),
            plots=cfg.get("plots", True),
            project=str(save_dir.parent),
            name=f"{save_dir.name}-{test_prefix}",
        )
        log_test_metrics(
            test_metrics,
            prefix=test_prefix,
            run_meta=wandb_run_meta,
            finish_after_log=wandb_started_here,
        )

    resolved_cfg = cfg | {
        "model": model_path,
        "wandb": True,
        "wandb_project": wandb_project,
        "wandb_entity": wandb_entity,
        "test_after_train": test_after_train,
        "test_split": test_split,
        "test_prefix": test_prefix,
    }
    save_resolved_cfg(save_dir, resolved_cfg, cfg_path)
    print(f"OK: Training finished. Run dir: {save_dir}")
    return results


def argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", required=True, help="Path to configs/*.yaml")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch", type=int)
    p.add_argument("--imgsz", type=int)
    p.add_argument("--device", type=str)
    p.add_argument("--name", type=str)
    p.add_argument("--seed", type=int, help="Random seed for reproducibility (overrides config)")
    p.add_argument("--no-test-after-train", action="store_true", help="Skip post-training validation on the test split")
    p.add_argument("--test-split", type=str, help="Dataset split to validate after training")
    p.add_argument("--test-prefix", type=str, help="W&B prefix for the post-training test")
    p.add_argument("--set", action="append", default=[], help="Arbitrary overrides: key=value (repeatable)")
    return p


if __name__ == "__main__":
    p = argparser()
    args = p.parse_args()

    train(args.cfg, cli_overrides(args))
