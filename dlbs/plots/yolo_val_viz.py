# yolo_val_viz.py
import numpy as np
import matplotlib.pyplot as plt


# Set True to print shapes once per image (safe + minimal).
# You can also set this via env var in your own code if you want.
DEBUG_PRINT_SHAPES = True


def _to_uint8_img(x: np.ndarray) -> np.ndarray:
    """Accepts HWC float [0..1] or [0..255] or uint8; returns uint8 HWC."""
    if x.dtype == np.uint8:
        return x
    x = np.asarray(x)
    if x.max() <= 1.5:
        x = x * 255.0
    x = np.clip(x, 0, 255).astype(np.uint8)
    return x


def chw_tensor_to_hwc_uint8(t) -> np.ndarray:
    """Torch tensor CHW (or BCHW where B=1) -> HWC uint8. (No torch import needed.)"""
    a = t.detach().float().cpu().numpy()
    if a.ndim == 4:
        a = a[0]
    a = np.transpose(a, (1, 2, 0))
    return _to_uint8_img(a)


def class_palette(num_classes: int):
    """Deterministic palette per class id (no extra deps)."""
    cols = []
    for i in range(max(num_classes, 1)):
        h = (i * 0.61803398875) % 1.0  # golden ratio spacing
        r = int(255 * (0.5 + 0.5 * np.cos(2 * np.pi * (h + 0.00))))
        g = int(255 * (0.5 + 0.5 * np.cos(2 * np.pi * (h + 0.33))))
        b = int(255 * (0.5 + 0.5 * np.cos(2 * np.pi * (h + 0.66))))
        cols.append((r, g, b))
    return cols


def pred_instances(pred_item):
    """
    Extract predicted instances from an Ultralytics Results-like object.

    Returns:
      masks_bool: (N,H,W) bool or None
      cls_ids:    (N,) int or None
      confs:      (N,) float or None
    """
    masks = getattr(pred_item, "masks", None)
    boxes = getattr(pred_item, "boxes", None)

    masks_bool = None
    if masks is not None and getattr(masks, "data", None) is not None:
        m = masks.data.detach().float().cpu().numpy()  # (N,h,w) (often lower-res)
        masks_bool = m > 0.5

    cls_ids = None
    confs = None
    if boxes is not None:
        if getattr(boxes, "cls", None) is not None:
            cls_ids = boxes.cls.detach().cpu().numpy().astype(int).reshape(-1)
        if getattr(boxes, "conf", None) is not None:
            confs = boxes.conf.detach().cpu().numpy().astype(float).reshape(-1)

    # Align lengths if possible
    if masks_bool is not None and cls_ids is not None:
        n = min(masks_bool.shape[0], cls_ids.shape[0])
        masks_bool = masks_bool[:n]
        cls_ids = cls_ids[:n]
        if confs is not None:
            confs = confs[:n]

    return masks_bool, cls_ids, confs


def gt_instances_from_batch(batch: dict, img_index: int, hw: tuple[int, int]):
    """
    Best-effort GT instance extraction across Ultralytics versions.

    Typical ultralytics:
      batch["masks"]     : (N,H,W)
      batch["batch_idx"] : (N,)
      batch["cls"]       : (N,1) or (N,)

    Returns:
      masks_bool: (N,H,W) bool or None
      cls_ids:    (N,) int or None
    """
    H, W = hw
    masks_all = batch.get("masks", None)
    bidx = batch.get("batch_idx", None)
    cls_all = batch.get("cls", None)

    if masks_all is None or bidx is None:
        return None, None

    try:
        ma = masks_all.detach().float().cpu().numpy()  # (N,H,W) or (B,H,W)
        b = bidx.detach().cpu().numpy().astype(int).reshape(-1)

        # Case A: semantic-ish (B,H,W) where B==batch size (rare)
        if ma.ndim == 3 and "img" in batch and ma.shape[0] == int(batch["img"].shape[0]) and ma.shape[-2:] == (H, W):
            m = ma[img_index] > 0.5
            return (m[None, ...], None) if m.shape == (H, W) else (None, None)

        # Case B: instance masks (N,H,W) mapped by batch_idx (most common)
        if ma.ndim != 3:
            return None, None
        if ma.shape[0] != b.shape[0]:
            return None, None

        sel = (b == img_index)
        masks = ma[sel] > 0.5
        if masks.size == 0:
            return None, None

        cls_ids = None
        if cls_all is not None:
            c = cls_all.detach().cpu().numpy().reshape(-1)
            if c.shape[0] == b.shape[0]:
                cls_ids = c[sel].astype(int)

        return masks, cls_ids

    except Exception:
        return None, None


def _resize_mask_nn(mask: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbor resize for a single 2D mask without extra deps."""
    out_h, out_w = out_hw
    in_h, in_w = mask.shape[-2:]
    if (in_h, in_w) == (out_h, out_w):
        return mask
    y = (np.linspace(0, in_h - 1, out_h)).astype(np.int64)
    x = (np.linspace(0, in_w - 1, out_w)).astype(np.int64)
    return mask[y][:, x]


def _resize_masks_nn(masks: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Resize (N,H,W) masks to (N,out_h,out_w) with nearest neighbor."""
    if masks is None:
        return None
    masks = np.asarray(masks)
    if masks.ndim == 2:
        masks = masks[None, ...]
    out = np.zeros((masks.shape[0], out_hw[0], out_hw[1]), dtype=masks.dtype)
    for i in range(masks.shape[0]):
        out[i] = _resize_mask_nn(masks[i], out_hw)
    return out


def overlay_instances_by_class(base_uint8: np.ndarray, masks_bool, cls_ids, class_colors, alpha: float) -> np.ndarray:
    """
    Overlay instance masks onto base image, colored by class id.
    Resizes masks to match base image shape if needed.
    """
    out = base_uint8.astype(np.float32).copy()
    h, w = out.shape[:2]

    if masks_bool is None:
        return base_uint8

    masks_bool = np.asarray(masks_bool)
    if masks_bool.size == 0:
        return base_uint8
    if masks_bool.ndim == 2:
        masks_bool = masks_bool[None, ...]

    # Resize instead of skipping when shapes differ (common in YOLO seg: low-res masks)
    if masks_bool.shape[-2:] != (h, w):
        masks_bool = _resize_masks_nn(masks_bool.astype(np.uint8), (h, w)).astype(bool)

    if cls_ids is None:
        cls_ids = np.zeros((masks_bool.shape[0],), dtype=int)

    cls_ids = np.asarray(cls_ids).reshape(-1)
    n = min(masks_bool.shape[0], cls_ids.shape[0])
    masks_bool = masks_bool[:n]
    cls_ids = cls_ids[:n]

    for i in range(n):
        m = masks_bool[i].astype(bool)
        if not m.any():
            continue
        cid = int(cls_ids[i])
        r, g, b = class_colors[cid % len(class_colors)]
        out[m, 0] = (1 - alpha) * out[m, 0] + alpha * r
        out[m, 1] = (1 - alpha) * out[m, 1] + alpha * g
        out[m, 2] = (1 - alpha) * out[m, 2] + alpha * b

    return np.clip(out, 0, 255).astype(np.uint8)


def make_custom_val_grid(batch: dict, preds, names: dict, max_show: int = 4, show_text: bool = True):
    """
    Build a 3xN grid (N <= max_show).

    Row 1: predicted instance masks on black background (colored by class)
    Row 2: original image
    Row 3: original with GT + Pred overlay (both colored by class)

    No boxes, only instance masks (no background regions).
    """
    imgs = batch.get("img", None)
    if imgs is None or not isinstance(preds, (list, tuple)):
        return None

    B = int(imgs.shape[0])
    n = min(B, max_show)

    fig, axes = plt.subplots(3, n, figsize=(4 * n, 10))
    if n == 1:
        axes = np.array(axes).reshape(3, 1)

    # determine #classes from names dict if possible
    try:
        nc = int(max(names.keys())) + 1 if isinstance(names, dict) and len(names) else 1
    except Exception:
        nc = 1

    class_colors = class_palette(num_classes=nc)

    for i in range(n):
        img_uint8 = chw_tensor_to_hwc_uint8(imgs[i])
        H, W = img_uint8.shape[:2]

        pred_masks, pred_cls, pred_conf = pred_instances(preds[i]) if i < len(preds) else (None, None, None)
        gt_masks, gt_cls = gt_instances_from_batch(batch, i, (H, W))

        if DEBUG_PRINT_SHAPES:
            pm_shape = None if pred_masks is None else tuple(pred_masks.shape)
            gm_shape = None if gt_masks is None else tuple(gt_masks.shape)
            print(
                f"[yolo_val_viz] img[{i}] HxW={H}x{W} | "
                f"pred_masks={pm_shape} pred_cls={'None' if pred_cls is None else pred_cls.shape} | "
                f"gt_masks={gm_shape} gt_cls={'None' if gt_cls is None else gt_cls.shape}"
            )

        black = np.zeros_like(img_uint8)

        row1 = overlay_instances_by_class(black, pred_masks, pred_cls, class_colors, alpha=0.85)
        row2 = img_uint8
        row3 = img_uint8.copy()
        row3 = overlay_instances_by_class(row3, gt_masks, gt_cls, class_colors, alpha=0.35)      # GT first
        row3 = overlay_instances_by_class(row3, pred_masks, pred_cls, class_colors, alpha=0.35)  # Pred on top

        axes[0, i].imshow(row1)
        axes[1, i].imshow(row2)
        axes[2, i].imshow(row3)

        for r in range(3):
            axes[r, i].axis("off")

        # Optional: show top-k predicted labels (class name + conf)
        if show_text and pred_cls is not None and pred_conf is not None and len(pred_cls) > 0:
            k = min(5, len(pred_cls))
            lines = []
            for j in range(k):
                cid = int(pred_cls[j])
                cname = names.get(cid, str(cid)) if isinstance(names, dict) else str(cid)
                lines.append(f"{cname} {float(pred_conf[j]):.2f}")
            axes[2, i].text(
                6, 18, "\n".join(lines),
                color="white", fontsize=9,
                bbox=dict(facecolor="black", alpha=0.45, pad=3),
            )

    axes[0, 0].set_ylabel("Pred (by class)\nblack bg", fontsize=12)
    axes[1, 0].set_ylabel("Original", fontsize=12)
    axes[2, 0].set_ylabel("GT + Pred\n(by class)", fontsize=12)

    plt.tight_layout()
    return fig
