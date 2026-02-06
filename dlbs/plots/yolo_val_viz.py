# yolo_val_viz.py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

DEBUG_PRINT_SHAPES = True


def _to_uint8_img(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.uint8:
        return x
    x = np.asarray(x)
    if x.max() <= 1.5:
        x = x * 255.0
    return np.clip(x, 0, 255).astype(np.uint8)


def chw_tensor_to_hwc_uint8(t) -> np.ndarray:
    a = t.detach().float().cpu().numpy()
    if a.ndim == 4:
        a = a[0]
    a = np.transpose(a, (1, 2, 0))
    return _to_uint8_img(a)


def class_palette(num_classes: int):
    cols = []
    for i in range(max(num_classes, 1)):
        h = (i * 0.61803398875) % 1.0
        r = int(255 * (0.5 + 0.5 * np.cos(2 * np.pi * (h + 0.00))))
        g = int(255 * (0.5 + 0.5 * np.cos(2 * np.pi * (h + 0.33))))
        b = int(255 * (0.5 + 0.5 * np.cos(2 * np.pi * (h + 0.66))))
        cols.append((r, g, b))
    return cols


def pred_instances(pred_item):
    """
    Returns:
      masks_bool (N,h,w) or None
      cls_ids    (N,) or None
      confs      (N,) or None
    """
    masks_bool = None
    cls_ids = None
    confs = None

    masks = getattr(pred_item, "masks", None)
    boxes = getattr(pred_item, "boxes", None)

    if masks is not None:
        data = getattr(masks, "data", None)
        if data is not None:
            m = data.detach().float().cpu().numpy()
            if m.ndim == 3 and m.shape[0] > 0:
                masks_bool = m > 0.5

    if boxes is not None:
        if getattr(boxes, "cls", None) is not None:
            cls_ids = boxes.cls.detach().cpu().numpy().astype(int).reshape(-1)
        if getattr(boxes, "conf", None) is not None:
            confs = boxes.conf.detach().cpu().numpy().astype(float).reshape(-1)

    if masks_bool is not None and cls_ids is not None:
        n = min(masks_bool.shape[0], cls_ids.shape[0])
        masks_bool = masks_bool[:n]
        cls_ids = cls_ids[:n]
        if confs is not None:
            confs = confs[:n]

    return masks_bool, cls_ids, confs


def _resize_mask_nn(mask: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    out_h, out_w = out_hw
    in_h, in_w = mask.shape[-2:]
    if (in_h, in_w) == (out_h, out_w):
        return mask
    y = (np.linspace(0, in_h - 1, out_h)).astype(np.int64)
    x = (np.linspace(0, in_w - 1, out_w)).astype(np.int64)
    return mask[y][:, x]


def _resize_masks_nn(masks: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
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
    out = base_uint8.astype(np.float32).copy()
    h, w = out.shape[:2]

    if masks_bool is None:
        return base_uint8

    masks_bool = np.asarray(masks_bool)
    if masks_bool.size == 0:
        return base_uint8
    if masks_bool.ndim == 2:
        masks_bool = masks_bool[None, ...]

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


def _segments_to_masks(segments, hw: tuple[int, int]) -> np.ndarray | None:
    """
    segments: list of polygons (each polygon Nx2), in pixel coords OR normalized [0..1].
    Returns (N,H,W) uint8 masks or None
    """
    H, W = hw
    if segments is None:
        return None

    masks = []
    for poly in segments:
        if poly is None:
            continue
        poly = np.asarray(poly, dtype=float)
        if poly.ndim != 2 or poly.shape[1] != 2 or poly.shape[0] < 3:
            continue

        # if normalized coordinates, scale up
        if poly.max() <= 1.5:
            poly[:, 0] *= W
            poly[:, 1] *= H

        im = Image.new("L", (W, H), 0)
        dr = ImageDraw.Draw(im)
        dr.polygon([tuple(p) for p in poly], outline=1, fill=1)
        masks.append(np.array(im, dtype=np.uint8))

    if not masks:
        return None
    return np.stack(masks, axis=0)


def gt_instances_from_batch(batch: dict, img_index: int, hw: tuple[int, int]):
    """
    Tries:
      A) batch["masks"] + batch["batch_idx"] (+ batch["cls"])
      B) batch["segments"] + batch["batch_idx"] (+ batch["cls"])  -> rasterize polygons
    """
    H, W = hw

    bidx = batch.get("batch_idx", None)
    cls_all = batch.get("cls", None)

    # Need batch_idx to select instances per image
    if bidx is None:
        return None, None
    b = bidx.detach().cpu().numpy().astype(int).reshape(-1)

    # cls ids per instance if present
    cls_ids_all = None
    if cls_all is not None:
        try:
            c = cls_all.detach().cpu().numpy().reshape(-1)
            if c.shape[0] == b.shape[0]:
                cls_ids_all = c.astype(int)
        except Exception:
            cls_ids_all = None

    sel = (b == img_index)

    # A) masks already available
    masks_all = batch.get("masks", None)
    if masks_all is not None:
        try:
            ma = masks_all.detach().float().cpu().numpy()
            if ma.ndim == 3 and ma.shape[0] == b.shape[0]:
                m = (ma[sel] > 0.5)
                if m.size == 0:
                    return None, None
                cls_sel = cls_ids_all[sel] if cls_ids_all is not None else None
                return m, cls_sel
        except Exception:
            pass

    # B) segments polygons fallback
    segments_all = batch.get("segments", None)
    if segments_all is None:
        return None, None

    try:
        # segments_all is usually a list with length N_instances
        if isinstance(segments_all, (list, tuple)) and len(segments_all) == b.shape[0]:
            seg_sel = [segments_all[i] for i in np.where(sel)[0].tolist()]
            m = _segments_to_masks(seg_sel, (H, W))
            if m is None or m.size == 0:
                return None, None
            cls_sel = cls_ids_all[sel] if cls_ids_all is not None else None
            return m.astype(bool), cls_sel
    except Exception:
        return None, None

    return None, None


def make_custom_val_grid(batch: dict, preds, names: dict, max_show: int = 4, show_text: bool = True):
    imgs = batch.get("img", None)
    if imgs is None or not isinstance(preds, (list, tuple)):
        return None

    if DEBUG_PRINT_SHAPES:
        try:
            print(f"[yolo_val_viz] batch keys: {sorted(list(batch.keys()))}")
            if len(preds) > 0:
                print(f"[yolo_val_viz] preds[0] type: {type(preds[0]).__name__}, has masks={hasattr(preds[0], 'masks')}, has boxes={hasattr(preds[0], 'boxes')}")
        except Exception:
            pass

    B = int(imgs.shape[0])
    n = min(B, max_show)

    fig, axes = plt.subplots(3, n, figsize=(4 * n, 10))
    if n == 1:
        axes = np.array(axes).reshape(3, 1)

    # determine #classes
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
            pm = 0 if pred_masks is None else int(pred_masks.shape[0])
            gm = 0 if gt_masks is None else int(gt_masks.shape[0])
            print(f"[yolo_val_viz] img[{i}] HxW={H}x{W} | pred_inst={pm} gt_inst={gm} | pred_masks_shape={None if pred_masks is None else pred_masks.shape}")

        black = np.zeros_like(img_uint8)

        row1 = overlay_instances_by_class(black, pred_masks, pred_cls, class_colors, alpha=0.85)
        row2 = img_uint8
        row3 = img_uint8.copy()
        row3 = overlay_instances_by_class(row3, gt_masks, gt_cls, class_colors, alpha=0.35)
        row3 = overlay_instances_by_class(row3, pred_masks, pred_cls, class_colors, alpha=0.35)

        axes[0, i].imshow(row1)
        axes[1, i].imshow(row2)
        axes[2, i].imshow(row3)
        for r in range(3):
            axes[r, i].axis("off")

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
