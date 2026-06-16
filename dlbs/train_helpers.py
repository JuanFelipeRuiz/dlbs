import random
from pathlib import Path

import numpy as np
import torch
import yaml


def parse_scalar(value: str):
    """Parse CLI --set key=value robustly as int/float/bool/null/str."""
    s = value.strip()
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


def as_bool(value) -> bool:
    """Parse bool-like config values."""
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Config YAML must be a dict (key: value).")
    return cfg


def save_resolved_cfg(save_dir: Path, cfg: dict, cfg_path: str):
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / "resolved_config.yaml"
    cfg_to_save = dict(cfg)
    cfg_to_save["_source_cfg"] = str(Path(cfg_path).resolve())
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_to_save, f, sort_keys=False, allow_unicode=True)
    print(f"OK: Resolved config saved: {out}")


def cli_overrides(args) -> dict:
    overrides = {
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "name": args.name,
        "seed": args.seed,
        "test_after_train": False if args.no_test_after_train else None,
        "test_split": args.test_split,
        "test_prefix": args.test_prefix,
    }

    for item in args.set:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got: {item}")
        key, value = item.split("=", 1)
        overrides[key.strip()] = parse_scalar(value)

    return overrides


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Set Seed: {seed}")
