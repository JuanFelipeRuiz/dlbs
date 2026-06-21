"""Stratified test metrics for YOLO11 segmentation.

Precision / recall / dice per bucket (overall, per city, per instance-size quartile),
computed only when test_model runs with --stratify. Matching mirrors the standard val
metrics (IoU >= 0.5, conf >= 0.001); dice is the harmonic mean of precision and recall.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from dlbs.summarizers.base_yolo import YoloDatasetBase, load_dataset_yaml

logger = logging.getLogger(__name__)

# Cityscapes images are a fixed size; relative areas map to pixels with this.
IMG_W, IMG_H = 2048, 1024

# Mask-area quartile edges in PIXELS, Source: notebooks/2_explore_instacnes_and_labels.ipynb (cell 20). Frozen on purpose
SIZE_QUARTILE_EDGES_PX = (0.5, 238.0, 871.5, 3670.5, 916554.0)
SIZE_LABELS = ("Q1", "Q2", "Q3", "Q4")

# Match criterion, kept identical to the existing val metrics.
DEFAULT_IOU = 0.5
DEFAULT_CONF = 0.001
DEFAULT_NMS_IOU = 0.7  # Ultralytics val() NMS default

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def size_bucket(seg_area_rel):
    """Map a relative mask area to its fixed quartile label (Q1..Q4)."""
    area_px = float(seg_area_rel) * IMG_W * IMG_H
    edges = SIZE_QUARTILE_EDGES_PX
    for i in range(1, len(edges) - 1):
        if area_px <= edges[i]:
            return SIZE_LABELS[i - 1]
    return SIZE_LABELS[-1]


def mask_iou(a, b):
    """IoU of two boolean masks of equal shape."""
    inter = int(np.logical_and(a, b).sum())
    if inter == 0:
        return 0.0
    union = int(np.logical_or(a, b).sum())
    return inter / union if union else 0.0


def match_predictions(pred_classes, pred_confs, pred_masks, gt_classes, gt_masks, iou_thr=DEFAULT_IOU):
    """Greedy confidence-ordered matching per class.

    A prediction is a TP if it overlaps an unmatched GT of the same class with
    IoU >= iou_thr (highest wins), else a FP; unmatched GTs are FN. Returns
    (tp_pairs, fp_pred_idx, fn_gt_idx).
    """
    order = sorted(range(len(pred_confs)), key=lambda k: -float(pred_confs[k]))
    gt_taken = [False] * len(gt_classes)
    tp_pairs, fp_idx = [], []

    for pi in order:
        best_iou, best_gi = iou_thr, -1
        for gi in range(len(gt_classes)):
            if gt_taken[gi] or gt_classes[gi] != pred_classes[pi]:
                continue
            iou = mask_iou(pred_masks[pi], gt_masks[gi])
            if iou >= best_iou:
                best_iou, best_gi = iou, gi
        if best_gi >= 0:
            gt_taken[best_gi] = True
            tp_pairs.append((pi, best_gi))
        else:
            fp_idx.append(pi)

    fn_idx = [gi for gi, taken in enumerate(gt_taken) if not taken]
    return tp_pairs, fp_idx, fn_idx


def precision_recall_dice(tp, fp, fn):
    """Precision, recall and dice (harmonic mean of P/R) from raw counts."""
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    dice = 0.0 if (p + r) <= 0 else (2.0 * p * r) / (p + r)
    return p, r, dice


def _bump(counts, key, field):
    counts.setdefault(key, {"tp": 0, "fp": 0, "fn": 0})[field] += 1


def accumulate(counts, *, tp_pairs, fp_idx, fn_idx, pred_size_labels, gt_size_labels, city):
    """Tally one image's TP/FP/FN into the overall, city and size buckets.

    TP/FN take the size bucket from the GT instance, FP from the prediction.
    """
    def keys_for(size_val):
        return [("overall",), ("city", city), ("size", size_val)]

    for _, gi in tp_pairs:
        for k in keys_for(gt_size_labels[gi]):
            _bump(counts, k, "tp")
    for pi in fp_idx:
        for k in keys_for(pred_size_labels[pi]):
            _bump(counts, k, "fp")
    for gi in fn_idx:
        for k in keys_for(gt_size_labels[gi]):
            _bump(counts, k, "fn")
    return counts


def _key_to_scope(key):
    """('overall',) -> 'overall'; ('city', 'zurich') -> 'city/zurich'; etc."""
    if key[0] == "overall":
        return "overall"
    return f"{key[0]}/{key[1]}"


def summarise(counts, prefix="test"):
    """Flatten bucket counts into a W&B log dict of precision/recall/dice + support."""
    log = {}
    for key, c in sorted(counts.items()):
        p, r, dice = precision_recall_dice(c["tp"], c["fp"], c["fn"])
        base = f"strata_{prefix}/{_key_to_scope(key)}"
        log[f"{base}/mask_precision"] = p
        log[f"{base}/mask_recall"] = r
        log[f"{base}/mask_dice"] = dice
        log[f"{base}/support_gt"] = c["tp"] + c["fn"]  # GT instances in bucket
        log[f"{base}/n_pred"] = c["tp"] + c["fp"]      # predictions in bucket
    return log


def _rasterize_polygon(coords_norm, h, w):
    """Rasterize a normalized YOLO polygon to a boolean (h, w) mask."""
    import cv2

    xs = coords_norm[0::2] * w
    ys = coords_norm[1::2] * h
    poly = np.stack([xs, ys], axis=1).round().astype(np.int32)
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(m, [poly], 1)
    return m.astype(bool)


def _load_gt(label_path, h, w, objects_for_image=None):
    """Parse a YOLO seg label file into (class_ids, masks, size_labels).

    Size uses objects.csv seg_area_rel when given, else the rasterized mask area.
    object_id is the 0-based line index, matching object_summarizer.
    """
    class_ids, masks, size_labels = [], [], []
    label_path = Path(label_path)
    if not label_path.exists():
        return class_ids, masks, size_labels

    lines = [ln.strip() for ln in label_path.read_text().splitlines() if ln.strip()]
    for obj_id, line in enumerate(lines):
        parts = line.split()
        if len(parts) < 7:  # class id + at least 3 points
            continue
        try:
            cid = int(parts[0])
            coords = np.asarray(parts[1:], dtype=np.float64)
        except ValueError:
            continue
        if len(coords) % 2 or len(coords) < 6:
            continue

        mask = _rasterize_polygon(coords, h, w)

        seg_area_rel = None
        if objects_for_image is not None and obj_id in objects_for_image.index:
            seg_area_rel = float(objects_for_image.loc[obj_id, "seg_area_rel"])
        if seg_area_rel is None:
            seg_area_rel = mask.sum() / float(h * w)

        class_ids.append(cid)
        masks.append(mask)
        size_labels.append(size_bucket(seg_area_rel))

    return class_ids, masks, size_labels


def _predict(model, image_path, conf, nms_iou, device, imgsz):
    """Run YOLO predict on one image -> (h, w, classes, confs, masks)."""
    kwargs = dict(
        source=str(image_path),
        conf=conf,
        iou=nms_iou,
        retina_masks=True,  # masks at original resolution -> align with the GT raster
        device=device,
        verbose=False,
    )
    if imgsz is not None:
        kwargs["imgsz"] = imgsz  # match the resolution used for the standard val metrics
    res = model.predict(**kwargs)
    r = res[0]
    h, w = int(r.orig_shape[0]), int(r.orig_shape[1])
    if r.masks is None or len(r.masks) == 0:
        return h, w, [], [], []
    masks = list(r.masks.data.cpu().numpy().astype(bool))
    classes = r.boxes.cls.cpu().numpy().astype(int).tolist()
    confs = r.boxes.conf.cpu().numpy().astype(float).tolist()
    return h, w, classes, confs, masks


def _load_objects_index(objects_csv, split):
    """{image_filename: DataFrame indexed by object_id} for a split, or {}.

    Missing file or missing columns degrade to {} (objects_csv is optional).
    """
    if objects_csv is None or not Path(objects_csv).exists():
        return {}
    odf = pd.read_csv(objects_csv)
    required = {"split", "image_filename", "object_id", "seg_area_rel"}
    missing = required - set(odf.columns)
    if missing:
        logger.warning(
            "objects_csv %s missing columns %s; falling back to rasterized mask areas",
            objects_csv, sorted(missing),
        )
        return {}
    odf = odf[odf["split"] == split]
    return {str(fn): g.set_index("object_id") for fn, g in odf.groupby("image_filename")}


def compute_stratified_metrics(model_path, data_yaml, *, split="test", objects_csv=None, device=None, prefix=None, imgsz=None):
    """Stratified precision/recall/dice over a split, as a flat W&B log dict.

    Matches predictions to GT per image (IoU >= 0.5, conf >= 0.001) and tallies
    TP/FP/FN into the overall, per-city and per-size buckets in one pass. Pass the
    same ``imgsz`` as the standard val run so the numbers stay comparable.
    """
    prefix = prefix or split

    from ultralytics import YOLO

    cfg = load_dataset_yaml(data_yaml)
    images_dir = cfg["splits"].get(split)
    if images_dir is None:
        raise FileNotFoundError(f"Split '{split}' not found in {data_yaml}")
    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Split images dir not found: {images_dir}")
    labels_dir = images_dir.parent.parent / "labels" / images_dir.name

    obj_groups = _load_objects_index(objects_csv, split)
    model = YOLO(str(model_path))

    image_paths = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    logger.info(
        "stratified eval: %d images, imgsz=%s, IoU=%.2f, conf=%.4f",
        len(image_paths), imgsz, DEFAULT_IOU, DEFAULT_CONF,
    )

    counts = {}
    for n, img_path in enumerate(image_paths, 1):
        stem = img_path.stem
        city = YoloDatasetBase.extract_city(stem)
        h, w, p_cls, p_conf, p_masks = _predict(model, img_path, DEFAULT_CONF, DEFAULT_NMS_IOU, device, imgsz)
        g_cls, g_masks, g_size = _load_gt(labels_dir / f"{stem}.txt", h, w, obj_groups.get(stem))
        p_size = [size_bucket(m.sum() / float(h * w)) for m in p_masks]
        tp, fp, fn = match_predictions(p_cls, p_conf, p_masks, g_cls, g_masks, DEFAULT_IOU)
        accumulate(
            counts,
            tp_pairs=tp,
            fp_idx=fp,
            fn_idx=fn,
            pred_size_labels=p_size,
            gt_size_labels=g_size,
            city=city,
        )
        if n % 50 == 0:
            logger.info("stratified eval: %d/%d images", n, len(image_paths))

    log = summarise(counts, prefix=prefix)
    logger.info("stratified eval: %d buckets, size edges (px)=%s", len(counts), list(SIZE_QUARTILE_EDGES_PX))
    return log
