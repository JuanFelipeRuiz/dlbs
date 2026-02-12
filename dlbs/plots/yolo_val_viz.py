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
    """Torch tensor CHW (or BCHW where B=1) -> HWC uint8."""
    a = t.detach().float().cpu().numpy()
    if a.ndim == 4:
        a = a[0]
    a = np.transpose(a, (1, 2, 0))
    return _to_uint8_img(a)


def class_palette(num_classes: int):
    """Deterministic palette per class id."""
    cols = []
    for i in range(max(num_classes, 1)):
        h = (i * 0.61803398875) % 1.0
        r = int(255 * (0.5 + 0.5 * np.cos(2 * np.pi * (h + 0.00))))
        g = int(255 * (0.5 + 0.5 * np.cos(2 * np.pi * (h + 0.33))))
        b = int(255 * (0.5 + 0.5 * np.cos(2 * np.pi * (h + 0.66))))
        cols.append((r, g, b))
    return cols


def instance_palette(num_instances: int):
    """Deterministic palette per instance id."""
    cols = []
    for i in range(max(num_instances, 1)):
        h = (i * 0.754877666) % 1.0
        r = int(255 * (0.5 + 0.5 * np.cos(2 * np.pi * (h + 0.10))))
        g = int(255 * (0.5 + 0.5 * np.cos(2 * np.pi * (h + 0.45))))
        b = int(255 * (0.5 + 0.5 * np.cos(2 * np.pi * (h + 0.80))))
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


def _torch_to_np(x):
    try:
        return x.detach().cpu().numpy()
    except Exception:
        return None


# -------------------------
# prediction extraction
# -------------------------

def pred_instances(pred_item):
    """
    Supports dict outputs (your Ultralytics version) and Results-like objects.
    Returns:
      masks_bool: (N,h,w) bool or None
      cls_ids:    (N,) int or None
      confs:      (N,) float or None
    """
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
        if confs is not None:
            keep = confs >= 0.5
            if masks_bool is not None:
                masks_bool = masks_bool[keep]
            if cls_ids is not None:
                cls_ids = cls_ids[keep]
            confs = confs[keep]

        return masks_bool, cls_ids, confs

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
# GT decoding (instance segmentation)
# -------------------------

def gt_instances_from_batch(batch: dict, img_index: int, out_hw: tuple[int, int], names=None):
    """
    Returns GT as instance masks + class ids:
      masks_bool: (N,H,W) bool or None
      cls_ids:    (N,) int or None

    Handles:
      A) per-instance masks: (N,h,w) with batch_idx (N,) and cls (N,)
      B) overlap mask per image: (B,h,w) with pixel=local_instance_id (1..k); cls/batch_idx provides class per instance
      C) semantic mask: (B,h,w) with pixel=class_id (fallback)
    """
    names = _normalize_names(names)
    nc = (max(names.keys()) + 1) if names else None

    masks_all = batch.get("masks", None)
    bidx = batch.get("batch_idx", None)
    cls_all = batch.get("cls", None)

    if masks_all is None:
        return None, None

    ma = _torch_to_np(masks_all)
    if ma is None:
        return None, None

    H, W = out_hw

    # -----------------
    # A) per-instance masks: (N,h,w) + batch_idx (N,)
    # -----------------
    if ma.ndim == 3 and bidx is not None:
        b = _torch_to_np(bidx)
        c = _torch_to_np(cls_all) if cls_all is not None else None
        if b is not None:
            b = b.astype(int).reshape(-1)
            if ma.shape[0] == b.shape[0]:
                sel = (b == int(img_index))
                masks = ma[sel]
                if masks.size == 0:
                    return None, None
                if masks.shape[-2:] != (H, W):
                    masks = _resize_masks_nn(masks, (H, W))
                masks_bool = masks > 0.5

                cls_ids = None
                if c is not None:
                    c = c.reshape(-1)
                    if c.shape[0] == b.shape[0]:
                        cls_ids = c[sel].astype(int)

                return masks_bool, cls_ids

    # -----------------
    # B/C) per-image mask: (B,h,w)
    # -----------------
    if ma.ndim == 3:
        if img_index >= ma.shape[0]:
            return None, None

        m = ma[img_index]
        if m.shape != (H, W):
            m = _resize_mask_nn(m, (H, W))

        m_int = np.rint(m).astype(np.int64)
        u = np.unique(m_int)

        # If it looks like semantic class mask (pixel<=nc-1), use fallback C
        if nc is not None and u.size and u.max() <= (nc - 1):
            masks = []
            cls_ids = []
            for cid in u.tolist():
                if cid == 0:
                    continue
                mm = (m_int == cid)
                if mm.any():
                    masks.append(mm)
                    cls_ids.append(int(cid))
            if not masks:
                return None, None
            return np.stack(masks, axis=0).astype(bool), np.asarray(cls_ids, dtype=int)

        # Otherwise treat as overlap instance-id mask (your case)
        if bidx is None or cls_all is None:
            # no class mapping available -> still return instance masks
            masks = []
            for inst_id in u.tolist():
                if inst_id == 0:
                    continue
                mm = (m_int == inst_id)
                if mm.any():
                    masks.append(mm)
            if not masks:
                return None, None
            return np.stack(masks, axis=0).astype(bool), None

        b = _torch_to_np(bidx)
        c = _torch_to_np(cls_all)
        if b is None or c is None:
            return None, None
        b = b.astype(int).reshape(-1)
        c = c.reshape(-1).astype(int)

        sel = (b == int(img_index))
        cls_img = c[sel]
        k = int(cls_img.shape[0])

        # Ultralytics overlap-mask convention: local instance ids are 1..k in order
        masks = []
        cls_ids = []
        for local_id in range(1, k + 1):
            mm = (m_int == local_id)
            if not mm.any():
                continue
            masks.append(mm)
            cls_ids.append(int(cls_img[local_id - 1]))

        if not masks:
            return None, None

        masks_bool = np.stack(masks, axis=0).astype(bool)
        cls_ids = np.asarray(cls_ids, dtype=int)

        if DEBUG_PRINT_SHAPES:
            print(f"[gt_decode] img={img_index} overlap: k={k} mask_max={int(u.max())} returned={masks_bool.shape[0]}")

        return masks_bool, cls_ids

    return None, None


# -------------------------
# filtering + rendering
# -------------------------

def filter_by_class(masks_bool, cls_ids, class_id: int):
    if masks_bool is None or cls_ids is None:
        return None, None
    masks_bool = np.asarray(masks_bool)
    if masks_bool.ndim == 2:
        masks_bool = masks_bool[None, ...]
    cls_ids = np.asarray(cls_ids).reshape(-1)

    n = min(masks_bool.shape[0], cls_ids.shape[0])
    masks_bool = masks_bool[:n]
    cls_ids = cls_ids[:n]

    sel = (cls_ids == int(class_id))
    if not sel.any():
        return None, None
    return masks_bool[sel], cls_ids[sel]


def render_black_bg_instances_by_class(hw, masks_bool, cls_ids, class_colors, alpha=0.90):
    """Pred: color by class."""
    H, W = hw
    base = np.zeros((H, W, 3), dtype=np.uint8)
    if masks_bool is None:
        return base

    masks_bool = np.asarray(masks_bool)
    if masks_bool.ndim == 2:
        masks_bool = masks_bool[None, ...]
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


def render_black_bg_instances_unique(hw, masks_bool, alpha=0.95):
    """GT: each instance different color."""
    H, W = hw
    base = np.zeros((H, W, 3), dtype=np.uint8)
    if masks_bool is None:
        return base

    masks_bool = np.asarray(masks_bool)
    if masks_bool.ndim == 2:
        masks_bool = masks_bool[None, ...]
    if masks_bool.shape[-2:] != (H, W):
        masks_bool = _resize_masks_nn(masks_bool.astype(np.uint8), (H, W)).astype(bool)

    cols = instance_palette(masks_bool.shape[0])
    out = base.astype(np.float32)

    for i in range(masks_bool.shape[0]):
        m = masks_bool[i].astype(bool)
        if not m.any():
            continue
        r, g, b = cols[i]
        out[m, 0] = (1 - alpha) * out[m, 0] + alpha * r
        out[m, 1] = (1 - alpha) * out[m, 1] + alpha * g
        out[m, 2] = (1 - alpha) * out[m, 2] + alpha * b

    return np.clip(out, 0, 255).astype(np.uint8)


# -------------------------
# grids
# -------------------------

def available_class_ids_in_batch(batch: dict, preds, names, max_show: int = 4):
    names = _normalize_names(names)
    imgs = batch.get("img", None)
    if imgs is None or not isinstance(preds, (list, tuple)):
        return set()

    B = int(imgs.shape[0])
    n = min(B, max_show)

    present = set()

    # preds
    for i in range(n):
        pm, pc, _ = pred_instances(preds[i]) if i < len(preds) else (None, None, None)
        if pm is not None and pc is not None:
            present.update(int(x) for x in np.asarray(pc).reshape(-1).tolist())

    # GT
    for i in range(n):
        img_uint8 = chw_tensor_to_hwc_uint8(imgs[i])
        H, W = img_uint8.shape[:2]
        gm, gc = gt_instances_from_batch(batch, i, (H, W), names=names)
        if gc is not None:
            present.update(int(x) for x in np.asarray(gc).reshape(-1).tolist())

    if names:
        present = {cid for cid in present if cid in names}

    return present


def make_val_grid_pred_gt_orig(batch: dict, preds, names, max_show: int = 4):
    """
    Grid (3xN):
      Row 1: black bg + Predictions (colored by class)
      Row 2: black bg + GT (each instance unique color)
      Row 3: Original image
    """
    names = _normalize_names(names)
    imgs = batch.get("img", None)
    if imgs is None or not isinstance(preds, (list, tuple)):
        return None

    B = int(imgs.shape[0])
    n = min(B, max_show)

    nc = (max(names.keys()) + 1) if names else 1
    class_colors = class_palette(nc)

    fig, axes = plt.subplots(3, n, figsize=(4 * n, 10))
    if n == 1:
        axes = np.array(axes).reshape(3, 1)

    for i in range(n):
        img_uint8 = chw_tensor_to_hwc_uint8(imgs[i])
        H, W = img_uint8.shape[:2]

        pred_masks, pred_cls, _ = pred_instances(preds[i]) if i < len(preds) else (None, None, None)
        gt_masks, gt_cls = gt_instances_from_batch(batch, i, (H, W), names=names)

        if DEBUG_PRINT_SHAPES:
            pm = 0 if pred_masks is None else int(np.asarray(pred_masks).shape[0])
            gm = 0 if gt_masks is None else int(np.asarray(gt_masks).shape[0])
            print(f"[yolo_val_viz] img[{i}] HxW={H}x{W} pred_inst={pm} gt_inst={gm}")

        row1 = render_black_bg_instances_by_class((H, W), pred_masks, pred_cls, class_colors, alpha=0.90)
        row2 = render_black_bg_instances_unique((H, W), gt_masks, alpha=0.95)
        row3 = img_uint8

        axes[0, i].imshow(row1)
        axes[1, i].imshow(row2)
        axes[2, i].imshow(row3)
        for r in range(3):
            axes[r, i].axis("off")

    axes[0, 0].set_ylabel("Pred (by class)", fontsize=12)
    axes[1, 0].set_ylabel("GT (per instance)", fontsize=12)
    axes[2, 0].set_ylabel("Original", fontsize=12)

    plt.tight_layout()
    return fig


def make_per_class_grids(batch: dict, preds, names, max_show: int = 4, class_ids=None):
    """
    Creates per-class grids ONLY for classes present.
    Returns dict: {class_name: fig}

    Layout (3xN) per class:
      Row 1: black bg + Pred instances of that class (colored by class)
      Row 2: black bg + GT instances of that class (unique colors per instance)
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
            pm_f, pc_f = filter_by_class(pred_masks, pred_cls, cid)

            gt_masks, gt_cls = gt_instances_from_batch(batch, i, (H, W), names=names)
            gm_f, _ = filter_by_class(gt_masks, gt_cls, cid)

            if (pm_f is not None) or (gm_f is not None):
                any_found = True

            row1 = render_black_bg_instances_unique((H, W), pm_f, alpha=0.95) 
            row2 = render_black_bg_instances_unique((H, W), gm_f, alpha=0.95)
            row3 = img_uint8

            axes[0, i].imshow(row1)
            axes[1, i].imshow(row2)
            axes[2, i].imshow(row3)
            for r in range(3):
                axes[r, i].axis("off")

        axes[0, 0].set_ylabel(f"Pred: {cname}", fontsize=12)
        axes[1, 0].set_ylabel(f"GT: {cname}\n(per instance)", fontsize=12)
        axes[2, 0].set_ylabel("Original", fontsize=12)
        fig.suptitle(f"Class: {cname}", fontsize=14)
        plt.tight_layout()

        if any_found:
            out[cname] = fig
        else:
            plt.close(fig)

    return out
