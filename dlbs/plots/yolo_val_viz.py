# yolo_val_viz.py
import numpy as np
import matplotlib.pyplot as plt

DEBUG_PRINT_SHAPES = False  # set True for local debugging


# -------------------------
# basic utils
# -------------------------

def _to_uint8_img(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.uint8:
        return x
    x = np.asarray(x)
    if x.max() <= 1.5:
        x = x * 255.0
    return np.clip(x, 0, 255).astype(np.uint8)


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
        h = (i * 0.61803398875) % 1.0
        r = int(255 * (0.5 + 0.5 * np.cos(2 * np.pi * (h + 0.00))))
        g = int(255 * (0.5 + 0.5 * np.cos(2 * np.pi * (h + 0.33))))
        b = int(255 * (0.5 + 0.5 * np.cos(2 * np.pi * (h + 0.66))))
        cols.append((r, g, b))
    return cols


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
    """Resize (N,H,W) -> (N,out_h,out_w) NN."""
    if masks is None:
        return None
    masks = np.asarray(masks)
    if masks.ndim == 2:
        masks = masks[None, ...]
    out = np.zeros((masks.shape[0], out_hw[0], out_hw[1]), dtype=masks.dtype)
    for i in range(masks.shape[0]):
        out[i] = _resize_mask_nn(masks[i], out_hw)
    return out


def _normalize_names(names):
    """names can be dict or list; always return dict[int,str]."""
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {i: str(n) for i, n in enumerate(names)}
    return {}


# -------------------------
# instance extraction (pred + gt)
# -------------------------

def pred_instances(pred_item):
    """
    Supports:
      - dict-based output (your Ultralytics version)
      - Results-like object with .masks.data + .boxes.cls/.conf

    Returns:
      masks_bool: (N,h,w) bool or None
      cls_ids:    (N,) int or None
      confs:      (N,) float or None
    """
    # Case 1: dict outputs
    if isinstance(pred_item, dict):
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


def gt_instances_from_batch(batch: dict, img_index: int, hw: tuple[int, int]):
    """
    Supports:
      - batch["masks"] (B,H,W)  -> semantic-ish per image => return single instance
      - batch["masks"] (N,H,W) + batch["batch_idx"] -> instance masks per image

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

    # Case A: per-image masks (B,H,W)
    if ma.ndim == 3 and "img" in batch and ma.shape[0] == int(batch["img"].shape[0]):
        m = ma[img_index]
        if m.shape != (H, W):
            m = _resize_mask_nn(m, (H, W))
        m = (m > 0.5)
        return m[None, ...], None

    # Case B: per-instance masks (N,H,W) with batch_idx mapping
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
# mask filtering + rendering
# -------------------------

def filter_by_class(masks_bool, cls_ids, class_id: int):
    """Keep only instances with cls==class_id."""
    if masks_bool is None:
        return None, None
    masks_bool = np.asarray(masks_bool)
    if masks_bool.ndim == 2:
        masks_bool = masks_bool[None, ...]

    if cls_ids is None:
        # if no classes, can't filter -> return none
        return None, None

    cls_ids = np.asarray(cls_ids).reshape(-1)
    n = min(masks_bool.shape[0], cls_ids.shape[0])
    masks_bool = masks_bool[:n]
    cls_ids = cls_ids[:n]

    sel = (cls_ids == int(class_id))
    if not sel.any():
        return None, None
    return masks_bool[sel], cls_ids[sel]


def render_black_bg_instances(hw: tuple[int, int], masks_bool, cls_ids, class_colors, alpha: float = 0.85):
    """Return HWC uint8 image: black background + instances (colored by class)."""
    H, W = hw
    base = np.zeros((H, W, 3), dtype=np.uint8)

    if masks_bool is None:
        return base

    masks_bool = np.asarray(masks_bool)
    if masks_bool.ndim == 2:
        masks_bool = masks_bool[None, ...]

    # resize if needed
    if masks_bool.shape[-2:] != (H, W):
        masks_bool = _resize_masks_nn(masks_bool.astype(np.uint8), (H, W)).astype(bool)

    if cls_ids is None:
        cls_ids = np.zeros((masks_bool.shape[0],), dtype=int)
    cls_ids = np.asarray(cls_ids).reshape(-1)

    n = min(masks_bool.shape[0], cls_ids.shape[0])
    masks_bool = masks_bool[:n]
    cls_ids = cls_ids[:n]

    out = base.astype(np.float32)
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
# main reusable grid builders
# -------------------------

def make_val_grid_pred_gt_orig(batch: dict, preds, names, max_show: int = 4):
    """
    Grid (3xN):
      Row 1: black bg + Predictions
      Row 2: black bg + GT
      Row 3: Original image
    """
    names = _normalize_names(names)

    imgs = batch.get("img", None)
    if imgs is None or not isinstance(preds, (list, tuple)):
        return None

    B = int(imgs.shape[0])
    n = min(B, max_show)

    # determine #classes
    nc = (max(names.keys()) + 1) if names else 1
    class_colors = class_palette(nc)

    fig, axes = plt.subplots(3, n, figsize=(4 * n, 10))
    if n == 1:
        axes = np.array(axes).reshape(3, 1)

    for i in range(n):
        img_uint8 = chw_tensor_to_hwc_uint8(imgs[i])
        H, W = img_uint8.shape[:2]

        pred_masks, pred_cls, _pred_conf = pred_instances(preds[i]) if i < len(preds) else (None, None, None)
        gt_masks, gt_cls = gt_instances_from_batch(batch, i, (H, W))

        if DEBUG_PRINT_SHAPES:
            pm = 0 if pred_masks is None else int(np.asarray(pred_masks).shape[0])
            gm = 0 if gt_masks is None else int(np.asarray(gt_masks).shape[0])
            print(f"[yolo_val_viz] img[{i}] HxW={H}x{W} pred_inst={pm} gt_inst={gm}")

        row1 = render_black_bg_instances((H, W), pred_masks, pred_cls, class_colors, alpha=0.90)
        row2 = render_black_bg_instances((H, W), gt_masks, gt_cls, class_colors, alpha=0.90)
        row3 = img_uint8

        axes[0, i].imshow(row1)
        axes[1, i].imshow(row2)
        axes[2, i].imshow(row3)

        for r in range(3):
            axes[r, i].axis("off")

    axes[0, 0].set_ylabel("Pred (black bg)", fontsize=12)
    axes[1, 0].set_ylabel("GT (black bg)", fontsize=12)
    axes[2, 0].set_ylabel("Original", fontsize=12)

    plt.tight_layout()
    return fig


def available_class_ids_in_batch(batch: dict, preds, names, max_show: int = 4):
    """
    Returns a set of class_ids that are present in either:
      - predictions (in the first max_show images)
      - GT (if GT cls_ids are available)
    """
    names = _normalize_names(names)
    imgs = batch.get("img", None)
    if imgs is None or not isinstance(preds, (list, tuple)):
        return set()

    B = int(imgs.shape[0])
    n = min(B, max_show)

    present = set()

    for i in range(n):
        img_uint8 = chw_tensor_to_hwc_uint8(imgs[i])
        H, W = img_uint8.shape[:2]

        pred_masks, pred_cls, _ = pred_instances(preds[i]) if i < len(preds) else (None, None, None)
        gt_masks, gt_cls = gt_instances_from_batch(batch, i, (H, W))

        if pred_masks is not None and pred_cls is not None:
            present.update(int(x) for x in np.asarray(pred_cls).reshape(-1).tolist())

        if gt_masks is not None and gt_cls is not None:
            present.update(int(x) for x in np.asarray(gt_cls).reshape(-1).tolist())

    # keep only ids that exist in names mapping if provided
    if names:
        present = {cid for cid in present if cid in names}

    return present


def make_per_class_grids(batch: dict, preds, names, max_show: int = 4, class_ids=None):
    """
    Creates per-class grids ONLY for classes that are present.
    Returns dict: {class_name: fig}

    Layout (3xN) per class:
      Row 1: black bg + Pred instances of that class only
      Row 2: black bg + GT instances of that class only
      Row 3: Original
    """
    names = _normalize_names(names)
    imgs = batch.get("img", None)
    if imgs is None or not isinstance(preds, (list, tuple)):
        return {}

    if class_ids is None:
        class_ids = sorted(list(available_class_ids_in_batch(batch, preds, names, max_show=max_show)))
    else:
        class_ids = [int(c) for c in class_ids]

    if not class_ids:
        return {}

    B = int(imgs.shape[0])
    n = min(B, max_show)

    nc = (max(names.keys()) + 1) if names else 1
    class_colors = class_palette(nc)

    out = {}

    for cid in class_ids:
        cname = names.get(cid, str(cid))

        fig, axes = plt.subplots(3, n, figsize=(4 * n, 10))
        if n == 1:
            axes = np.array(axes).reshape(3, 1)

        any_found = False

        for i in range(n):
            img_uint8 = chw_tensor_to_hwc_uint8(imgs[i])
            H, W = img_uint8.shape[:2]

            pred_masks, pred_cls, _ = pred_instances(preds[i]) if i < len(preds) else (None, None, None)
            gt_masks, gt_cls = gt_instances_from_batch(batch, i, (H, W))

            pred_m_f, pred_c_f = filter_by_class(pred_masks, pred_cls, cid)
            gt_m_f, gt_c_f = filter_by_class(gt_masks, gt_cls, cid)

            if pred_m_f is not None or gt_m_f is not None:
                any_found = True

            row1 = render_black_bg_instances((H, W), pred_m_f, pred_c_f, class_colors, alpha=0.95)
            row2 = render_black_bg_instances((H, W), gt_m_f, gt_c_f, class_colors, alpha=0.95)
            row3 = img_uint8

            axes[0, i].imshow(row1)
            axes[1, i].imshow(row2)
            axes[2, i].imshow(row3)
            for r in range(3):
                axes[r, i].axis("off")

        axes[0, 0].set_ylabel(f"Pred: {cname}", fontsize=12)
        axes[1, 0].set_ylabel(f"GT: {cname}", fontsize=12)
        axes[2, 0].set_ylabel("Original", fontsize=12)
        fig.suptitle(f"Class: {cname}", fontsize=14)
        plt.tight_layout()

        # only return grids that actually contain something
        if any_found:
            out[cname] = fig
        else:
            plt.close(fig)

    return out
