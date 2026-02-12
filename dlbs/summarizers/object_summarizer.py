""" 
Collects the diffrent objects from a yolo dataset format and summarizes them in a table. 
It collects following information:

- image_relpath
- image_filename
- split
- city
- object_id
- class_id (+ optional class_name)
- bbox_xc, bbox_yc, bbox_w, bbox_h (norm)
- bbox_area_rel, bbox_aspect_ratio
- seg_area_rel, seg_to_bbox_ratio
- n_polygon_points

Usage:
    python -m dlbs.helpers.summarize_objects --dataset_yaml_path dataset_yolo/data.yaml
    python -m dlbs.helpers.summarize_objects --dataset_yaml_path dataset_yolo/data.yaml --out objects.csv
"""

import argparse
import logging
from pandas.core.series import Series
from numpy import ndarray
from numpy._typing._shape import _Shape
from numpy._typing._array_like import NDArray
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from dlbs.summarizers.base_yolo import YoloDatasetBase

logger = logging.getLogger(__name__)


def shoelace_area(xs: np.ndarray, ys: np.ndarray) -> float:
    """Polygon area via shoelace formula. *xs/ys* are 1-D arrays."""
    if len(xs) < 3:
        return 0.0
    return 0.5 * float(np.abs(xs @ np.roll(ys, -1) - ys @ np.roll(xs, -1)))


def bbox_from_polygon(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, float, float]:
    """Return (xc, yc, w, h) bounding box from polygon vertices."""
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    return (
        (x_min + x_max) / 2.0,
        (y_min + y_max) / 2.0,
        x_max - x_min,
        y_max - y_min,
    )


class YoloObjectSummarizer(YoloDatasetBase):
    """Parses every label in a YOLO dataset and builds a per-object DataFrame.

    Parameters:
    ----------
    dataset_yaml_path: Path to the YOLO ``data.yaml`` file.
    """

    def _process_file(self, file_path: Path, images_dir: Path, split: str) -> Optional[pd.DataFrame]:
        """Read one label .txt and return a DataFrame with one row per object."""
        text = file_path.read_text()
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if not lines:
            return None

        records = self._parse_lines(lines)
        if records is None or len(records) == 0:
            return None

        stem = file_path.stem
        records["image_filename"] = stem
        records["image_relpath"] = self.find_image_relpath(images_dir, stem)
        records["split"] = split
        records["city"] = self.extract_city(stem)
        return records

    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add class names and derived geometry columns."""
        df = self._add_class_names(df)
        df = self._compute_derived_columns(df)
        return df

    def _iter_files(self, split_dir: Path):
        """Yield label .txt files from the labels/ directory for this split."""
        labels_dir = self._labels_dir_for(split_dir)
        if not labels_dir.exists():
            logger.warning(f"Labels dir not found: {labels_dir}")
            return []
        return sorted(labels_dir.glob("*.txt"))

    def _parse_lines(self, lines: list[str]) -> Optional[pd.DataFrame]:
        """Parse label lines into a DataFrame. ``class_id x1 y1 x2 y2 … xn yn``"""
        class_ids, all_xs, all_ys, obj_ids = [], [], [], []

        for obj_id, line in enumerate(lines):
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cid = int(parts[0])
                coords = np.array(parts[1:], dtype=np.float64)
            except ValueError:
                continue
            if len(coords) % 2 != 0 or len(coords) < 4:
                continue

            class_ids.append(cid)
            all_xs.append(coords[0::2])
            all_ys.append(coords[1::2])
            obj_ids.append(obj_id)

        if not class_ids:
            return None

        bbox_xc, bbox_yc, bbox_w, bbox_h = zip(*(bbox_from_polygon(x, y) for x, y in zip(all_xs, all_ys)))
        seg_areas = [shoelace_area(x, y) for x, y in zip(all_xs, all_ys)]
        n_points = [len(x) for x in all_xs]

        return pd.DataFrame({
            "object_id": np.array(obj_ids, dtype=np.int32),
            "class_id": np.array(class_ids, dtype=np.int32),
            "bbox_xc": np.array(bbox_xc, dtype=np.float64),
            "bbox_yc": np.array(bbox_yc, dtype=np.float64),
            "bbox_w": np.array(bbox_w, dtype=np.float64),
            "bbox_h": np.array(bbox_h, dtype=np.float64),
            "seg_area_rel": np.array(seg_areas, dtype=np.float64),
            "n_polygon_points": np.array(n_points, dtype=np.int32),
        })

    @staticmethod
    def _compute_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Add bbox_area_rel, bbox_aspect_ratio, seg_to_bbox_ratio."""
        df["bbox_area_rel"] = df["bbox_w"] * df["bbox_h"]
        df["bbox_aspect_ratio"] = np.where(df["bbox_h"] > 0, df["bbox_w"] / df["bbox_h"], 0.0)
        df["seg_to_bbox_ratio"] = np.where(df["bbox_area_rel"] > 0, df["seg_area_rel"] / df["bbox_area_rel"], 0.0)
        return df

    def _add_class_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map class_id to class_name using the YAML names dict."""
        df["class_name"] = df["class_id"].map(self.class_names).fillna(
            df["class_id"].apply(lambda c: f"class_{c}")
        )
        return df

    def _labels_dir_for(images_dir: Path) -> Path:
        """Replace images/train with labels/train"""
        return images_dir.parent.parent / "labels" / images_dir.name


def summarize_objects(dataset_yaml_path: str | Path) -> pd.DataFrame:
    """Shortcut to create a Class and execute it."""
    return YoloObjectSummarizer(dataset_yaml_path).run()

def args_objects():
    parser = argparse.ArgumentParser(description="Summarize objects from a YOLO dataset into a table.")
    parser.add_argument("--dataset_yaml_path", type=str, required=True, help="Path to the YOLO data.yaml")
    parser.add_argument("--out", type=str, default=None, help="Output CSV path (default: prints summary)")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    args = args_objects()

    df = summarize_objects(args.dataset_yaml_path)

    if args.out:
        df.to_csv(args.out, index=False)
        logger.info(f"Saved {len(df)} rows to {args.out}")
    else:
        print(f"\n{'='*60}")
        print(f"Objects: {len(df)}")
        print(f"Splits:  {dict[Any, Series | Any | ndarray[_Shape, Any] | NDArray](df['split'].value_counts()) if len(df) else 'none'}")
        print(f"Classes: {dict(df['class_name'].value_counts()) if len(df) else 'none'}")
        print(f"Cities:  {sorted(df['city'].unique()) if len(df) else 'none'}")
        print(f"{'='*60}\n")
        print(df.head(20).to_string(index=False))
