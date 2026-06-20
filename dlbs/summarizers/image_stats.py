"""
Summarizes image meta information and statistics. It assumes that the images are
stored in a yolo compatible dataset format and the naming convetions of the cityscapes dataset.

Computed per image:
- image_relpath, image_filename, split, city
- width, height
- hash            (MD5 of raw pixel bytes – exact-duplicate detection)
- brightness      (mean V channel, HSV)
- saturation      (mean S channel, HSV)
- contrast        (std  V channel, HSV)
- edge_density    (fraction of Canny edge pixels)

Usage:
    python -m dlbs.helpers.summarize_files --dataset_yaml_path dataset_yolo/data.yaml
    python -m dlbs.helpers.summarize_files --dataset_yaml_path dataset_yolo/data.yaml --out files.csv
    python -m dlbs.helpers.summarize_files --dataset_yaml_path dataset_yolo/data.yaml --workers 8
"""

import argparse
import logging
from hashlib import md5
from pathlib import Path
from typing import Optional

import cv2
import pandas as pd

from dlbs.summarizers.base_yolo import IMAGE_EXTENSIONS, YoloDatasetBase

logger = logging.getLogger(__name__)


def compute_image_statistics(image_path: str | Path) -> dict:
    """Read one image and return a dict of pixel-level statistics."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    h, w = img.shape[:2]
    pixel_hash = md5(img.tobytes()).hexdigest()

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s_norm = hsv[..., 1] / 255.0
    v_norm = hsv[..., 2] / 255.0

    brightness = float(v_norm.mean())
    saturation = float(s_norm.mean())
    contrast = float(v_norm.std())

    edges = cv2.Canny(img, 100, 200)
    edge_density = float((edges > 0).mean())

    return {
        "width": w,
        "height": h,
        "hash": pixel_hash,
        "brightness": brightness,
        "saturation": saturation,
        "contrast": contrast,
        "edge_density": edge_density,
    }


class YoloFileSummarizer(YoloDatasetBase):
    """Walks a YOLO dataset and computes per-image statistics.

    Parameters
    ----------
    dataset_yaml_path : str | Path
        Path to the YOLO ``data.yaml`` file.
    workers : int
        Number of threads for parallel image reading.  1 = sequential.
    """

    def __init__(self, dataset_yaml_path: str | Path, workers: int = 4):
        super().__init__(dataset_yaml_path, workers=workers)

    def _iter_files(self, split_dir: Path):
        """Yield image files from the images directory for this split."""
        if not split_dir.exists():
            logger.warning(f"Images dir not found: {split_dir}")
            return []
        files = []
        for ext in IMAGE_EXTENSIONS:
            files.extend(split_dir.glob(f"*{ext}"))
            files.extend(split_dir.glob(f"*{ext.upper()}"))
        return sorted(set(files))

    def _process_file(self, file_path: Path, images_dir: Path, split: str) -> Optional[pd.DataFrame]:
        """Compute stats for a single image and return a one-row DataFrame."""
        stem = file_path.stem
        try:
            stats = compute_image_statistics(file_path)
        except Exception as e:
            logger.warning(f"Skipping {file_path.name}: {e}")
            return None

        stats["image_filename"] = stem
        stats["image_relpath"] = self.find_image_relpath(images_dir, stem)
        stats["split"] = split
        stats["city"] = self.extract_city(stem)
        return pd.DataFrame([stats])


def summarize_files(dataset_yaml_path: str | Path, workers: int = 4) -> pd.DataFrame:
    """Shortcut: create a :class:`YoloFileSummarizer` and call ``run()``."""
    return YoloFileSummarizer(dataset_yaml_path, workers=workers).run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Summarize image files from a YOLO dataset.")
    parser.add_argument("--dataset_yaml_path", type=str, required=True, help="Path to the YOLO data.yaml")
    parser.add_argument("--out", type=str, default=None, help="Output CSV path (default: prints summary)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel threads for image reading")
    args = parser.parse_args()

    df = summarize_files(args.dataset_yaml_path, workers=args.workers)

    if args.out:
        df.to_csv(args.out, index=False)
        logger.info(f"Saved {len(df)} rows to {args.out}")
    else:
        print(f"\n{'='*60}")
        print(f"Images: {len(df)}")
        print(f"Splits: {dict(df['split'].value_counts()) if len(df) else 'none'}")
        print(f"Cities: {sorted(df['city'].unique()) if len(df) else 'none'}")
        if len(df):
            print(f"\nStats (mean):")
            for col in ("brightness", "saturation", "contrast", "edge_density"):
                if col in df.columns:
                    print(f"  {col:15s}: {df[col].mean():.4f}")
            dupes = df["hash"].duplicated().sum()
            print(f"\nDuplicate hashes: {dupes}")
        print(f"{'='*60}\n")
        print(df.head(10).to_string(index=False))
