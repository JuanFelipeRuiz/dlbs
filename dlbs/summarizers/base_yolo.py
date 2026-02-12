"""
Base class for YOLO-dataset summarizers.

Handles YAML loading, split iteration, city extraction, and image path
resolution so that concrete summarizers only need to implement:

- ``_iter_files``    – which files to iterate over per split
- ``_process_file``  – how to turn one file into a (partial) DataFrame
- ``_post_process``  – any DataFrame-wide transformations after concat
"""

import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")
SPLIT_KEYS = ("train", "val", "test")


def load_dataset_yaml(yaml_path: Path) -> dict:
    """Parse a YOLO data.yaml into ``{names, splits, nc}``."""
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {yaml_path}")

    data = yaml.safe_load(yaml_path.read_text()) or {}

    names = data.get("names", {})
    if isinstance(names, list):
        names = {i: v for i, v in enumerate(names)}
    elif isinstance(names, dict):
        names = {int(k): v for k, v in names.items()}

    yaml_dir = yaml_path.parent
    splits = {}
    for key in SPLIT_KEYS:
        val = data.get(key)
        if val:
            p = Path(val)
            splits[key] = p if p.is_absolute() else yaml_dir / p

    return {"names": names, "splits": splits, "nc": data.get("nc", len(names))}


class YoloDatasetBase(ABC):
    """Skeleton that walks every split of a YOLO dataset.

    Parameters
    ----------
    dataset_yaml_path : str | Path
        Path to the YOLO ``data.yaml`` file.
    """

    def __init__(self, dataset_yaml_path: str | Path, workers: int = 1):
        self.yaml_path = Path(dataset_yaml_path)
        self.yaml_dir = self.yaml_path.parent
        self.workers = max(1, workers)

        cfg = load_dataset_yaml(self.yaml_path)
        self.class_names: dict[int, str] = cfg["names"]
        self.splits: dict[str, Path] = cfg["splits"]
        self.nc: int = cfg["nc"]

    def run(self) -> pd.DataFrame:
        """Iterate over every split, concat results, and post-process."""
        parts = [self._process_split(name, img_dir) for name, img_dir in self.splits.items()]
        parts = [p for p in parts if p is not None and len(p)]

        if not parts:
            logger.warning("No records found in any split")
            return pd.DataFrame()

        df = pd.concat(parts, ignore_index=True)
        df = self._post_process(df)
        logger.info(f"Total rows: {len(df)} across {df['split'].nunique()} splits")
        return df

    def _process_split(self, split_name: str, images_dir: Path) -> Optional[pd.DataFrame]:
        """Collect DataFrames from every file in one split (parallel when workers > 1)."""
        files = list(self._iter_files(images_dir))
        if not files:
            logger.warning(f"No files for split '{split_name}'")
            return None

        logger.info(f"Split '{split_name}': processing {len(files)} files (workers={self.workers})")

        def _work(fp: Path):
            return self._process_file(fp, images_dir, split_name)

        if self.workers > 1:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                file_dfs = list(pool.map(_work, files))
        else:
            file_dfs = [_work(f) for f in files]

        file_dfs = [d for d in file_dfs if d is not None and len(d)]
        return pd.concat(file_dfs, ignore_index=True) if file_dfs else None

    @abstractmethod
    def _iter_files(self, split_dir: Path):
        """Yield the file paths to process for a given split directory."""

    @abstractmethod
    def _process_file(self, file_path: Path, images_dir: Path, split: str) -> Optional[pd.DataFrame]:
        """Turn one file into a (partial) DataFrame (or None)."""

    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optional hook for DataFrame-wide transformations after concat."""
        return df

    @staticmethod
    def extract_city(filename: str) -> str:
        """Extract city name from Cityscapes-style stem ``aachen_000000_000019_leftImg8bit``."""
        parts = filename.split("_")
        for i, part in enumerate(parts):
            if part.isdigit():
                return "_".join(parts[:i])
        return parts[0] if parts else ""

    def find_image_relpath(self, images_dir: Path, stem: str) -> str:
        """Try common extensions and return a path string relative to the YAML dir."""
        for ext in IMAGE_EXTENSIONS:
            candidate = images_dir / (stem + ext)
            if candidate.exists():
                try:
                    return str(candidate.relative_to(self.yaml_dir))
                except ValueError:
                    return str(candidate)
        return f"{images_dir.name}/{stem}.*"

    def find_image_path(self, images_dir: Path, stem: str) -> Optional[Path]:
        """Return the absolute Path to the image, or None."""
        for ext in IMAGE_EXTENSIONS:
            candidate = images_dir / (stem + ext)
            if candidate.exists():
                return candidate
        return None

