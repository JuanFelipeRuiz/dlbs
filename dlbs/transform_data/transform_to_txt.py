"""
Transform Cityscapes-style polygon JSON annotations to YOLO instance segmentation format.

Cityscapes JSON:
{
    "imgWidth": 2048,
    "imgHeight": 1024,
    "objects": [
        {"label": "car", "polygon": [[100,100], [200,100], [200,200], [100,200]]}
    ]
}

YOLO line: class_id x1 y1 x2 y2 x3 y3 ...   (normalized 0..1)
"""

import json
import argparse
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YOLOConverterBase:
    """
    Shared I/O, batching, and worker logic for converters.
    Subclasses must implement convert_single(input_path, output_path=None).
    """

    def __init__(self, **init_kwargs):
        """Store kwargs so worker processes can re-instantiate the converter."""
        self._init_kwargs = dict(init_kwargs)

    def prepare_paths(self, input_dir=None, df=None, pattern="*.json", col=None):
        """
        Return a list of paths from a directory or a DataFrame column.
        """
        if df is not None and col:
            return [Path(p) for p in df[col] if isinstance(p, (str, Path))]
        if input_dir:
            return list(Path(input_dir).rglob(pattern))
        raise ValueError("Either input_dir or df must be provided.")

    def _build_output_path(self, output_path, input_path):
        """
        Build the final output .txt path (supports directory or file path).
        """
        output_path = Path(output_path) if output_path is not None else None
        if output_path is None:
            return None
        if output_path.suffix.lower() != ".txt":
            name = self._build_filename(input_path)
            output_path = output_path / name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    def _build_filename(self, input_path):
        """
        Default filename mapping. Subclasses may override.
        """
        return Path(input_path).stem + ".txt"

    def _atomic_write_lines(self, path, lines):
        """
        Atomically write lines to file (write temp then replace).
        """
        tmp = Path(str(path) + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
        tmp.replace(path)

    def _new_instance(self):
        """
        Create a new instance of the subclass with the same init kwargs.
        """
        return self.__class__(**getattr(self, "_init_kwargs", {}))

    def worker(self, input_path, output_dir):
        """
        Worker for multiprocessing. Recreates converter and runs convert_single.
        """
        conv = self._new_instance()
        return conv.convert_single(input_path, output_dir)

    def convert(self, input_dir=None, df=None, output_dir=None, pattern="*.json", col=None, n_workers=4):
        """
        Convert many files by calling convert_single on each path.
        """
        paths = self.prepare_paths(input_dir=input_dir, df=df, pattern=pattern, col=col)
        logger.info("Converting %d files...", len(paths))

        if n_workers and n_workers > 1:
            results = []
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                futures = {ex.submit(self.worker, str(p), output_dir): p for p in paths}
                for fut in as_completed(futures):
                    p = futures[fut]
                    try:
                        results.append(fut.result())
                    except Exception as e:
                        logger.error("Error processing %s: %s", p, e)
                        results.append(None)
            return results

        results = []
        for p in paths:
            try:
                results.append(self.convert_single(p, output_dir))
            except Exception as e:
                logger.error("Error processing %s: %s", p, e)
                results.append(None)
        return results

    def convert_single(self, input_path, output_path=None):
        """
        Abstract method — implement in subclass. Return list of lines or output path.
        """
        raise NotImplementedError("Subclass must implement convert_single()")


class PolygonToYOLO(YOLOConverterBase):
    """
    Convert Cityscapes-style polygon JSON to YOLO instance segmentation lines.
    """

    def __init__(self, class_mapping=None, skip_classes=None, label_aliases=None):
        """
        Initialize converter.

        class_mapping: dict label->id (if None, it will be built from data)
        skip_classes: set of labels to ignore
        label_aliases: dict to normalize labels (e.g., {"bike": "bicycle"})
        """
        super().__init__(class_mapping=class_mapping, skip_classes=skip_classes, label_aliases=label_aliases)
        self.class_mapping = dict(class_mapping) if class_mapping else {}
        self.skip_classes = set(skip_classes) if skip_classes else set()
        self.label_aliases = dict(label_aliases) if label_aliases else {}

    def _read_json(self, path):
        """
        Read a JSON file safely.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error("Failed to read JSON %s: %s", path, e)
            return None

    def _normalize_label(self, label):
        """
        Normalize a raw label through aliases.
        """
        if not label:
            return label
        return self.label_aliases.get(label, label)

    def _collect_labels(self, json_paths):
        """
        Scan all JSONs to collect unique, normalized labels.
        """
        labels = set()
        for jp in json_paths:
            data = self._read_json(jp)
            if not data:
                continue
            for obj in data.get("objects", []):
                label = self._normalize_label(obj.get("label", "").strip())
                if label:
                    labels.add(label)
        return sorted(labels)

    def get_label_paths(self, json_paths):
        """
        Build class_mapping by scanning paths, unless already provided.
        """
        if self.class_mapping:
            logger.info("Using provided class_mapping (skipping auto build)")
            return self.class_mapping
        labels = [l for l in self._collect_labels(json_paths) if l not in self.skip_classes]
        self.class_mapping = {label: i for i, label in enumerate(labels)}
        logger.info("Class mapping: %s", self.class_mapping)
        return self.class_mapping

    def _build_filename(self, json_path):
        """
        Map Cityscapes polygons filename to YOLO txt filename.
        """
        return Path(json_path).stem.replace("gtFine_polygons", "leftImg8bit") + ".txt"

    def _polygon_to_norm_coords(self, polygon, img_w, img_h):
        """
        Convert [[x,y], ...] to a flat list of normalized coordinates clamped to [0,1].
        """
        out = []
        for pt in polygon:
            if not isinstance(pt, (list, tuple)) or len(pt) != 2:
                continue
            x, y = pt
            try:
                nx = max(0.0, min(1.0, float(x) / float(img_w)))
                ny = max(0.0, min(1.0, float(y) / float(img_h)))
                out.extend([nx, ny])
            except Exception:
                continue
        return out

    def convert_single(self, json_path, output_path=None):
        """
        Convert one JSON to YOLO lines or write to a .txt if output_path is provided.
        """
        data = self._read_json(json_path)
        if data is None:
            return None

        img_w = data.get("imgWidth")
        img_h = data.get("imgHeight")
        if not img_w or not img_h:
            logger.error("Missing imgWidth/imgHeight in %s", json_path)
            return None

        if not self.class_mapping:
            self.get_label_paths([Path(json_path)])

        lines = []
        for obj in data.get("objects", []):
            label = self._normalize_label(obj.get("label", "").strip())
            poly = obj.get("polygon", [])
            if not label or label in self.skip_classes:
                continue
            if not isinstance(poly, list) or len(poly) < 3:
                continue
            cid = self.class_mapping.get(label)
            if cid is None:
                logger.warning("Label '%s' not in mapping; skipping in %s", label, json_path)
                continue
            coords = self._polygon_to_norm_coords(poly, img_w, img_h)
            if len(coords) < 6:
                continue
            lines.append(f"{cid} " + " ".join(f"{v:.6f}" for v in coords))

        if output_path is None:
            return lines

        out = self._build_output_path(output_path, json_path)
        self._atomic_write_lines(out, lines)
        return str(out)


def transformer_args():
    """
    Parse command-line arguments for the transformer script.
    """
    parser = argparse.ArgumentParser(description="Convert Cityscapes polygon JSON to YOLO instance segmentation format")
    parser.add_argument("input", type=str, help="Input directory containing JSON files")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output directory (default: alongside inputs)")
    parser.add_argument("-w", "--workers", type=int, default=4, help="Parallel workers (set 1 to disable)")
    parser.add_argument("--skip-classes", nargs="+", default=None, help="Classes to skip")
    parser.add_argument("--aliases", type=str, default=None, help="JSON file of label aliases")
    parser.add_argument("--save-mapping", type=str, default=None, help="Path to save class mapping (JSON)")
    return parser.parse_args()


def main():
    """
    CLI entry point for polygon JSON → YOLO.
    """
    args = transformer_args()

    label_aliases = {}
    if args.aliases:
        try:
            with open(args.aliases, "r", encoding="utf-8") as f:
                label_aliases = json.load(f)
        except Exception as e:
            logger.warning("Failed to read aliases from %s: %s", args.aliases, e)

    skip = set(args.skip_classes) if args.skip_classes else None
    converter = PolygonToYOLO(skip_classes=skip, label_aliases=label_aliases)

    outputs = converter.convert(input_dir=args.input, output_dir=args.output, pattern="*_polygons.json", n_workers=args.workers)

    if args.save_mapping:
        try:
            with open(args.save_mapping, "w", encoding="utf-8") as f:
                json.dump(converter.class_mapping, f, indent=2, ensure_ascii=False)
            logger.info("Saved class mapping to %s", args.save_mapping)
        except Exception as e:
            logger.error("Failed to save mapping: %s", e)

    logger.info("Converted %d files", len(outputs))
    return 0


if __name__ == "__main__":
    main()
