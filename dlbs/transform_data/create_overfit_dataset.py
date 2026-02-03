"""
Create a small overfit dataset for testing overfitting behavior.

Usage to create a small overfit dataset of exactly one image:
    python create_overfit_dataset.py 

Use help to see all options and their descriptions.

    python create_overfit_dataset.py --help
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class OverfitDatasetPreparer:
    """
    Builds a tiny overfit dataset and produces a compact-ID data_overfit.yaml.

    Workflow:
      1. Copy N images and labels from source to target (train/val).
      2. Read original class names from source data.yaml.
      3. Collect used class IDs from the new target dataset.
      4. Build a compact mapping {old_id -> new_id} for the new target dataset.
      5. Rewrite all label files with the new IDs.
      6. Write a new data_overfit.yaml file with updated class mapping.
    """

    def __init__(self, source_dir: Path, target_dir: Path, source_yaml: Path, split: str = "train", num_images: int = 1):
        """Initialize the dataset preparer with paths and parameters."""
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.source_yaml = Path(source_yaml)
        self.split = split
        self.num_images = int(num_images)

        self.labels_root = self.target_dir / "labels"
        self.images_root = self.target_dir / "images"

        self.class_names: dict[int, str] = {}
        self.used_ids: set[int] = set()
        self.id_map: dict[int, int] = {}

    def run(self) -> None:
        """Run the complete overfit dataset preparation pipeline."""
        self._validate_inputs()
        self._copy_subset()
        self._read_class_names_from_yaml()
        self._collect_used_ids()
        self._build_id_map()
        self._remap_all_labels()
        self._write_overfit_yaml()

    def _validate_inputs(self) -> None:
        """Validate source directory and arguments."""
        if not self.source_dir.exists():
            raise FileNotFoundError(f"Source directory does not exist: {self.source_dir}")
        if self.num_images < 1:
            raise ValueError("--num_images must be >= 1")

    def _copy_subset(self) -> None:
        """Copy a limited number of images and labels to the target dataset."""
        src_images = self.source_dir / "images" / self.split
        src_labels = self.source_dir / "labels" / self.split
        dst_images = self.images_root / self.split
        dst_labels = self.labels_root / self.split

        for d in (dst_images, dst_labels):
            d.mkdir(parents=True, exist_ok=True)
            for p in d.glob("*"):
                if p.is_file():
                    p.unlink()

        exts = [".jpg", ".jpeg", ".png", ".bmp"]
        image_files = []
        for ext in exts:
            image_files.extend(src_images.glob(f"*{ext}"))
            image_files.extend(src_images.glob(f"*{ext.upper()}"))

        if not image_files:
            raise FileNotFoundError(f"No images found in {src_images}")

        image_files = sorted(image_files)[: self.num_images]

        copied_pairs = 0
        for img in image_files:
            shutil.copy2(img, dst_images / img.name)
            label_src = src_labels / (img.stem + ".txt")
            if label_src.exists():
                shutil.copy2(label_src, dst_labels / label_src.name)
                copied_pairs += 1
            else:
                logger.warning(f"Label not found for {img.name}: {label_src}")

        logger.info(f"Copied {len(image_files)} images, {copied_pairs} labels to {dst_images.parent}")

        if self.split == "train":
            self._mirror_to_val(image_files, src_labels)

    def _mirror_to_val(self, image_files, src_labels: Path) -> None:
        """Duplicate the training subset into the validation split for overfitting tests."""
        val_images = self.images_root / "val"
        val_labels = self.labels_root / "val"
        for d in (val_images, val_labels):
            d.mkdir(parents=True, exist_ok=True)
            for p in d.glob("*"):
                if p.is_file():
                    p.unlink()

        val_copied = 0
        for img in image_files:
            shutil.copy2(img, val_images / img.name)
            label_src = src_labels / (img.stem + ".txt")
            if label_src.exists():
                shutil.copy2(label_src, val_labels / label_src.name)
                val_copied += 1

        logger.info(f"Mirrored {len(image_files)} images, {val_copied} labels to validation split")

    def _read_class_names_from_yaml(self) -> None:
        """Read class names from the original data.yaml file."""
        if not self.source_yaml.exists():
            raise FileNotFoundError(f"Source data.yaml does not exist: {self.source_yaml}")

        data = yaml.safe_load(self.source_yaml.read_text()) or {}
        names = data.get("names", {})
        if isinstance(names, dict):
            self.class_names = {int(k): v for k, v in names.items()}
        elif isinstance(names, list):
            self.class_names = {i: v for i, v in enumerate(names)}
        else:
            self.class_names = {}

        logger.info(f"Loaded {len(self.class_names)} class names from {self.source_yaml}")

    def _collect_used_ids(self) -> None:
        """Scan label files and record which class IDs are actually used."""
        used = set()
        for sp in ("train", "val"):
            labels_dir = self.labels_root / sp
            if not labels_dir.exists():
                continue
            for label_file in labels_dir.glob("*.txt"):
                try:
                    for line in label_file.read_text().splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        try:
                            cid = int(parts[0])
                            used.add(cid)
                        except ValueError:
                            continue
                except Exception as e:
                    logger.warning(f"Error reading {label_file}: {e}")
        self.used_ids = used
        logger.info(f"Found {len(self.used_ids)} used class IDs: {sorted(self.used_ids)}")

    def _build_id_map(self) -> None:
        """Create a compact mapping from old to new class IDs."""
        self.id_map = {old: new for new, old in enumerate(sorted(self.used_ids))}
        logger.info(f"Old to New ID map: {self.id_map}")

    def _remap_all_labels(self) -> None:
        """Apply the class ID remapping to all label files."""
        if not self.id_map:
            logger.warning("No ID map to apply (no labels remapped).")
            return

        for sp in ("train", "val"):
            labels_dir = self.labels_root / sp
            if not labels_dir.exists():
                continue
            for label_file in labels_dir.glob("*.txt"):
                self._remap_label_file(label_file)

    def _remap_label_file(self, label_file: Path) -> None:
        """Rewrite a single label file, replacing old class IDs with new ones."""
        out_lines = []
        for line in label_file.read_text().splitlines():
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            try:
                old_id = int(parts[0])
                if old_id in self.id_map:
                    parts[0] = str(self.id_map[old_id])
                out_lines.append(" ".join(parts))
            except ValueError:
                out_lines.append(s)
        label_file.write_text("\n".join(out_lines) + "\n")

    def _write_overfit_yaml(self) -> None:
        """Write the data_overfit.yaml file with updated class information."""
        target_yaml = self.target_dir / "data_overfit.yaml"

        if not self.used_ids:
            data = {"train": "images/train", "val": "images/val", "nc": 0, "names": {}}
            yaml.safe_dump(data, open(target_yaml, "w"), sort_keys=False)
            logger.warning(f"No classes used. Wrote empty {target_yaml}")
            return

        new_names = {new: self.class_names.get(old, f"class_{old}") for old, new in self.id_map.items()}

        data = {
            "train": "images/train",
            "val": "images/val",
            "nc": len(new_names),
            "names": dict(sorted(new_names.items())),
        }
        with open(target_yaml, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)

        logger.info(f"Dataset YAML: {target_yaml}")


def _parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Create a small overfit dataset")
    p.add_argument("--source_dir", type=Path, default=Path("dataset_yolo"), help="Source dataset dir")
    p.add_argument("--target_dir", type=Path, default=Path("dataset_yolo_overfit"), help="Target overfit dir")
    p.add_argument("--num_images", type=int, default=1, help="Number of images to copy")
    p.add_argument("--split", type=str, default="train", choices=["train", "val"], help="Split to copy from")
    p.add_argument("--source_yaml", type=Path, default=None, help="Path to source data.yaml (defaults to source_dir/data.yaml)")
    return p.parse_args()


def main():
    """Entry point for command-line execution."""
    logger.info(sys.argv)
    args = _parse_args()
    source_yaml = args.source_yaml or (args.source_dir / "data.yaml")

    preparer = OverfitDatasetPreparer(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        source_yaml=source_yaml,
        split=args.split,
        num_images=args.num_images,
    )
    preparer.run()

    logger.info(f"Images:  {args.target_dir / 'images' / args.split}")
    logger.info(f"Labels:  {args.target_dir / 'labels' / args.split}")


if __name__ == "__main__":
    main()
