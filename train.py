'''
usage:
    python train.py --cfg configs/baseline_no_aug.yaml
    python train.py --cfg configs/aug_default.yaml --epochs 100
    python train.py --cfg configs/aug_strong.yaml --set lr0=0.005 --set cos_lr=true
    python train.py --cfg configs/baseline_no_aug.yaml --seed 123

'''

import argparse
import os
import random
from pathlib import Path
import yaml
import torch
import numpy as np
from ultralytics import YOLO, settings


def create_dataset_structure(root="dataset"):
    """Optional: erstellt dataset/images|labels train/val Struktur."""
    dirs = [
        f"{root}/images/train",
        f"{root}/images/val",
        f"{root}/labels/train",
        f"{root}/labels/val",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print(f"✓ Dataset-Struktur erstellt unter: {root}/")


def _parse_scalar(v: str):
    """Parse CLI --set key=value robust (int/float/bool/null/str)."""
    s = v.strip()
    if s.lower() in {"null", "none"}:
        return None
    if s.lower() in {"true", "false"}:
        return s.lower() == "true"
    try:
        if "." in s or "e" in s.lower():
            return float(s)
        return int(s)
    except ValueError:
        return s


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Config YAML muss ein dict (key: value) sein.")
    return cfg


def save_resolved_cfg(save_dir: Path, cfg: dict, cfg_path: str):
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / "resolved_config.yaml"
    cfg_to_save = dict(cfg)
    cfg_to_save["_source_cfg"] = str(Path(cfg_path).resolve())
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_to_save, f, sort_keys=False, allow_unicode=True)
    print(f"✓ Resolved config gespeichert: {out}")


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Set deterministic mode for CUDA operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Set Seed: {seed}")


def train(cfg_path: str, cli_overrides: dict):
    cfg = load_cfg(cfg_path)

    # Seed setzen (vor allem anderen, falls in CLI oder Config gesetzt)
    seed = cli_overrides.get("seed", cfg.get("seed", None))
    if seed is not None:
        set_seed(seed)
        # Seed auch in cfg setzen für Ultralytics
        cfg["seed"] = seed

    # W&B togglen (Ultralytics) - speichere den Wert vor dem pop
    wandb_enabled = bool(cfg.pop("wandb", True))
    settings.update({"wandb": wandb_enabled})

    # Optional: WANDB_PROJECT/ENTITY via YAML setzen, ohne hart an Ultralytics gebunden zu sein
    wandb_project = cfg.pop("wandb_project", None)
    wandb_entity = cfg.pop("wandb_entity", None)
    if wandb_project:
        os.environ["WANDB_PROJECT"] = str(wandb_project)
    if wandb_entity:
        os.environ["WANDB_ENTITY"] = str(wandb_entity)

    # CLI overrides anwenden (nur wenn gesetzt)
    cfg.update({k: v for k, v in cli_overrides.items() if v is not None})

    # Device fallback
    dev = cfg.get("device", None)
    if dev in [0, "0", "cuda", "cuda:0"] and not torch.cuda.is_available():
        print("CUDA nicht verfügbar → fallback auf CPU.")
        cfg["device"] = "cpu"

    model_path = cfg.pop("model")
    model = YOLO(model_path)

    # Add custom wandb callbacks if wandb is enabled
    if wandb_enabled:
        try:
            from yolocustom_wand import add_custom_callbacks
            add_custom_callbacks(model)
            print("Custom wandb callbacks added")
        except ImportError as e:
            print(f"Could not import custom wandb callbacks: {e}")
        except Exception as e:
            print(f"Could not add custom wandb callbacks: {e}")

    # Train
    results = model.train(**cfg)

    # Save resolved config in run dir
    save_dir = None
    try:
        save_dir = Path(model.trainer.save_dir)
    except Exception:
        # not fatal
        save_dir = Path(cfg.get("project", "runs")) / str(cfg.get("name", "exp"))

    save_resolved_cfg(save_dir, cfg | {"model": model_path}, cfg_path)
    print(f"✓ Training fertig. Run dir: {save_dir}")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", required=True, help="Pfad zu configs/*.yaml")
    p.add_argument("--init-dataset", action="store_true", help="Dataset-Ordnerstruktur erstellen")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch", type=int)
    p.add_argument("--imgsz", type=int)
    p.add_argument("--device", type=str)
    p.add_argument("--name", type=str)
    p.add_argument("--seed", type=int, help="Random seed for reproducibility (overrides config)")
    p.add_argument("--set", action="append", default=[], help="Beliebige Overrides: key=value (mehrfach)")

    args = p.parse_args()

    if args.init_dataset:
        create_dataset_structure()

    overrides = {
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "name": args.name,
        "seed": args.seed,
    }

    # zusätzliche --set key=value Paare
    for item in args.set:
        if "=" not in item:
            raise ValueError(f"--set erwartet key=value, bekommen: {item}")
        k, v = item.split("=", 1)
        overrides[k.strip()] = _parse_scalar(v)

    train(args.cfg, overrides)
