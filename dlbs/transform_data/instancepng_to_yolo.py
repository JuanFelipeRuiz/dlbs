import logging
import math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image
from scipy import ndimage
import cv2
import matplotlib.pyplot as plt

from dlbs.transform_data.transform_to_txt import YOLOConverterBase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InstanceImageToYOLO(YOLOConverterBase):
    """
    Convert Cityscapes-style instanceId rasters to YOLO segmentation labels.
    Each connected component (or connected group) becomes one polygon instance.
    """

    def __init__(self, cityscape_class_map=None, own_class_map=None, visualize=False, split_groups=True, min_pixels=10):
        """
        Initialize converter.

        cityscape_class_map: mapping from Cityscapes class IDs to names
        own_class_map: optional YOLO id -> name mapping (only for metadata/logging)
        visualize: show debug plots of connected groups
        split_groups: split each instance into connected components
        min_pixels: minimum pixels per group to consider
        """
        super().__init__(cityscape_class_map=cityscape_class_map, own_class_map=own_class_map,
                         visualize=visualize, split_groups=split_groups, min_pixels=min_pixels)

        if cityscape_class_map is None:
            cityscape_class_map = {
                24: "person",
                25: "rider",
                26: "car",
                27: "truck",
                28: "bus",
                31: "train",
                32: "motorcycle",
                33: "bicycle",
            }
        self.cityscape_class_map = cityscape_class_map
        self.own_class_map = own_class_map
        self.visualize = visualize
        self.split_groups = split_groups
        self.min_pixels = int(min_pixels)

        ids = sorted(self.cityscape_class_map.keys())
        self.id_mapping = {cid: i for i, cid in enumerate(ids)}
        self.yolo_to_cityscapes = {i: cid for cid, i in self.id_mapping.items()}
        self.class_map = own_class_map if own_class_map is not None else {
            yid: self.cityscape_class_map[cid] for yid, cid in self.yolo_to_cityscapes.items()
        }

    def _build_filename(self, img_path):
        """
        Map instanceIds filename to YOLO txt filename.
        """
        return Path(img_path).stem.replace("gtFine_instanceIds", "leftImg8bit") + ".txt"

    def _load_image(self, img_path):
        """
        Load instanceIds raster as a numpy array.
        """
        return np.array(Image.open(img_path))

    def _get_hw(self, arr):
        """
        Return height and width of an image array.
        """
        h, w = arr.shape[:2]
        return h, w

    def _unique_ids(self, instance_img):
        """
        Return sorted unique instance IDs present in the image.
        """
        return np.unique(instance_img)

    def _instances_for_class(self, uniq, cityscapes_id):
        """
        Return all instanceIds corresponding to a Cityscapes class.
        """
        base = cityscapes_id * 1000
        return sorted([i for i in uniq if base <= i < base + 1000], key=lambda x: x % 1000)

    def _split_into_connected_groups(self, mask, iid):
        """
        Split a binary mask into connected components.
        """
        labeled, n = ndimage.label(mask)
        out = []
        for g in range(1, n + 1):
            grp = labeled == g
            if int(grp.sum()) > self.min_pixels:
                out.append((iid, g, grp))
        return out

    def _build_groups(self, instance_img, instance_ids):
        """
        Build connected-component groups for each instance id.
        """
        groups = []
        for iid in instance_ids:
            mask = (instance_img == iid).astype(bool)
            if self.split_groups:
                groups.extend(self._split_into_connected_groups(mask, iid))
            else:
                if int(mask.sum()) > self.min_pixels:
                    groups.append((iid, 1, mask))
        return groups

    def _largest_contour(self, mask):
        """
        Return the largest external contour for a mask, or None.
        """
        cnts, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        return max(cnts, key=lambda c: len(c))

    def _contour_to_polygon(self, contour, h, w):
        """
        Convert an OpenCV contour to a normalized polygon [x1,y1,x2,y2,...].
        """
        poly = []
        for pt in contour:
            x, y = pt[0]
            poly.append(x / w)
            poly.append(y / h)
        return poly

    def _groups_to_yolo(self, groups, cityscapes_id, h, w):
        """
        Convert group masks into YOLO polygon lines for one class.
        """
        yid = self.id_mapping[cityscapes_id]
        lines = []
        for _, _, mask in groups:
            contour = self._largest_contour(mask)
            if contour is None:
                continue
            poly = self._contour_to_polygon(contour, h, w)
            if len(poly) >= 6:
                lines.append(f"{yid} " + " ".join(f"{p:.6f}" for p in poly))
        return lines

    def _visualize_groups(self, groups, cityscapes_id):
        """
        Show a grid of masks for visual debugging.
        """
        cols = 4
        rows = max(1, math.ceil(len(groups) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = np.array(axes).reshape(-1)

        class_name = self.cityscape_class_map.get(cityscapes_id, str(cityscapes_id))
        yolo_id = self.id_mapping.get(cityscapes_id, cityscapes_id)

        for ax, (_, gid, mask) in zip(axes, groups):
            ax.imshow(mask, cmap="gray")
            ax.set_title(f"Group {gid} (YOLO: {yolo_id})", fontsize=10)
            ax.axis("off")

        for ax in axes[len(groups):]:
            ax.axis("off")

        plt.suptitle(f"Class {cityscapes_id}: {class_name} → YOLO ID {yolo_id}", fontsize=12)
        plt.tight_layout()
        plt.show()

    def convert_single(self, input_path, output_path=None):
        """
        Convert one instanceIds image to YOLO lines or write a .txt if output_path is provided.
        """
        instance_img = self._load_image(input_path)
        h, w = self._get_hw(instance_img)
        uniq = self._unique_ids(instance_img)
        lines = []

        for cid in sorted(self.cityscape_class_map.keys()):
            inst_ids = self._instances_for_class(uniq, cid)
            groups = self._build_groups(instance_img, inst_ids)
            if self.visualize:
                self._visualize_groups(groups, cid)
            lines.extend(self._groups_to_yolo(groups, cid, h, w))

        if output_path is None:
            return lines

        out = self._build_output_path(output_path, input_path)
        self._atomic_write_lines(out, lines)
        return str(out)


def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Convert Cityscapes instanceIds to YOLO segmentation labels")
    p.add_argument("input", type=str, help="Directory with *instanceIds.png")
    p.add_argument("-o", "--output", type=str, required=True, help="Directory to write YOLO .txt files")
    p.add_argument("-w", "--workers", type=int, default=4, help="Parallel workers (set 1 to disable)")
    p.add_argument("--no-split", action="store_true", help="Do not split instances into connected components")
    p.add_argument("--min-pixels", type=int, default=10, help="Minimum pixels per group")
    p.add_argument("--visualize", action="store_true", help="Show debug plots of groups")
    return p.parse_args()


def main():
    args = _parse_args()
    conv = InstanceImageToYOLO(
        visualize=args.visualize,
        split_groups=not args.no_split,
        min_pixels=args.min_pixels,
    )
    outputs = conv.convert(input_dir=args.input, output_dir=args.output, pattern="*instanceIds.png", n_workers=args.workers)
    ok = sum(1 for r in outputs if r)
    logger.info("Wrote %d/%d label files to %s", ok, len(outputs), args.output)


if __name__ == "__main__":
    main()
