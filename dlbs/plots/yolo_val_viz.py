# yolo_val_viz.py
import numpy as np
import matplotlib.pyplot as plt

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


# -------------------------
# PRED extraction (dict or Results)
# -------------------------

def pred_instances(pred_item):
    """
    Supports:
      - Ultralytics Results-like object with .masks.data + .boxes.cls/.conf
      - dict-based output (your case): try common keys

    Returns:
      masks_bool: (N,h,w) bool or None
      cls_ids:    (N,) int or None
      confs:      (N,) float or None
    """
    # Case 1: dict outputs
    if isinstance(pred_item, dict):
        # try common key candidates
        masks = None
        for k in ("masks", "mask", "segments", "mask_probs"):
            if k in pred_item:
                masks = pred_item[k]
                break

        cls_ids = None
        for k in ("cls", "classes", "class_ids"):
            if k in pred_item:
                cls_ids = pred_item[k]
                break

        confs = None
        for k in ("conf", "confs", "scores"):
            if k in pred_item:
                confs = pred_item[k]
                break

        masks_bool = None
        if masks is not None:
            try:
                if hasattr(masks, "detach"):
                    masks = masks.detach().float().cpu().numpy()
                else:
                    masks = np.asarray(masks)
                # expected (N,h,w)
                if masks.ndim == 3 and masks.shape[0] > 0:
                    masks_bool = masks > 0.5
            except Exception:
                masks_bool = None

        if cls_ids is not None:
            try:
                if hasattr(cls_ids, "detach"):
                    cls_ids = cls_ids.detach().cpu().numpy()
                cls_ids = np.asarray(cls_ids).astype(int).reshape(-1)
            except Exception:
                cls_ids = None

        if confs is not None:
            try:
                if hasattr(confs, "detach"):
                    confs = confs.detach().cpu().numpy()
                confs = np.asarray(confs).astype(float).reshape(-1)
            except Exception:
                confs = None

        # align
        if masks_bool is not None and cls_ids is not None:
            n = min(masks_bool.shape[0], cls_ids.shape[0])
            masks_bool = masks_bool[:n]
            cls_ids = cls_ids[:n]
            if confs is not None:
                confs = confs[:n]

        return masks_bool, cls_ids, confs

    # Case 2: Results-like objects
    masks = getattr(pred_item, "masks", None)
    boxes = getattr(pred_item, "boxes", None)

    masks_bool = None
    if masks is not None and getattr(masks, "data", None) is not None:
        m = masks.data.detach().float().cpu().numpy()
        if m.ndim == 3 and m.shape[0] > 0:
            masks_bool = m > 0.5

    cls_ids = None
    confs = None
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


# -------------------------
# GT extraction (batch masks)
# -------------------------

def gt_instances_from_batch(batch: dict, img_index: int, hw: tuple[int, int]):
    """
    Supports:
      - batch["masks"] (B,H,W) -> treat as single mask instance for that image
      - batch["masks"] (N,H,W) + batch["batch_idx"] -> instance masks

    Returns:
      masks_bool: (N,H,W) bool or None
      cls_ids:    (N,) int or None
    """
    H, W = hw
    masks_all = batch.get("masks", None)
    cls_all = batch.get("cls", None)
    bidx = batch.get("batch_idx", None)

    if masks_all is None:
        return None, None

    try:
        ma = masks_all.detach().float().cpu().numpy()
    except Exception:
        return None, None

    # If GT masks are per-image: (B,H,W)
    if ma.ndim == 3 and "img" in batch and ma.shape[0] == int(batch["img"].shape[0]):
        m = ma[img_index]
        # resize if needed
        if m.shape != (H, W):
            m = _resize_mask_nn(m, (H, W))
        m = (m > 0.5)
        # no per-instance cls here
        return m[None, ...], None

    # If GT masks are per-instance: (N,H,W) with batch_idx mapping
    if ma.ndim == 3 and bidx is not None:
        try:
            b = bidx.detach().cpu().numpy().astype(int).reshape(-1)
        except Exception:
            return None, None

        if ma.shape[0] != b.shape[0]:
            return None, None

        sel = (b == img_index)
        masks = ma[sel]
        if masks.size == 0:
            return None, None

        if masks.shape[-2:] != (H, W):
            masks = _resize_masks_nn(masks, (H, W))

        masks_bool = masks > 0.5

        cls_ids = None
        if cls_all is not None:
            try:
                c = cls_all.detach().cpu().numpy().reshape(-1)
                if c.shape[0] == b.shape[0]:
                    cls_ids = c[sel].astype(int)
            except Exception:
                cls_ids = None

        return masks_bool, cls_ids

    return None, None


# -------------------------
# GRID
# -------------------------

def make_custom_val_grid(batch: dict, preds, names: dict, max_show: int = 4, show_text: bool = True):
    imgs = batch.get("img", None)
    if imgs is None or not isinstance(preds, (list, tuple)):
        return None

    if DEBUG_PRINT_SHAPES:
        try:
            print(f"[yolo_val_viz] batch keys: {sorted(list(batch.keys()))}")
            if len(preds) > 0:
                print(f"[yolo_val_viz] preds[0] type: {type(preds[0]).__name__}")
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
            print(
                f"[yolo_val_viz] img[{i}] HxW={H}x{W} | "
                f"pred_inst={pm} gt_inst={gm} | "
                f"pred_masks_shape={None if pred_masks is None else pred_masks.shape} | "
                f"gt_masks_shape={None if gt_masks is None else gt_masks.shape}"
            )

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
