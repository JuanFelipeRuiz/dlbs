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
    """Deterministic palette per instance id (different from class colors)."""
    cols = []
    for i in range(max(num_instances, 1)):
        h = (i * 0.754877666) % 1.0  # another irrational spacing
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


def gt_semantic_mask_image(batch: dict, img_index: int):
    """
    Your GT is semantic:
      batch["masks"] has shape (B,h,w) with integer-ish pixel values = class_id.
    Returns a 2D array (h,w) of class ids, or None.
    """
    masks_all = batch.get("masks", None)
    if masks_all is None:
        return None
    try:
        ma = masks_all.detach().cpu().numpy()
    except Exception:
        return None
    if ma.ndim != 3 or img_index >= ma.shape[0]:
        return None
    return ma[img_index]


# -------------------------
# GT instance splitting (connected components)
# -------------------------

def connected_components_bool(mask_bool: np.ndarray, max_instances: int = 200):
    """
    Split a boolean mask into connected components (4-neighborhood).
    Returns list of bool masks, each one component.
    Pure numpy flood-fill (no scipy/opencv). Suitable for small masks.
    """
    m = np.asarray(mask_bool).astype(bool)
    if m.ndim != 2:
        return []
    h, w = m.shape
    if not m.any():
        return []

    visited = np.zeros((h, w), dtype=bool)
    comps = []

    # iterate pixels; flood-fill when we find a new True pixel
    for y in range(h):
        row = m[y]
        for x in np.where(row & (~visited[y]))[0]:
            if visited[y, x] or not m[y, x]:
                continue
            if len(comps) >= max_instances:
                return comps

            stack = [(y, x)]
            visited[y, x] = True
            comp = np.zeros((h, w), dtype=bool)
            comp[y, x] = True

            while stack:
                cy, cx = stack.pop()
                # 4-neighbors
                if cy > 0:
                    ny, nx = cy - 1, cx
                    if m[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        comp[ny, nx] = True
                        stack.append((ny, nx))
                if cy + 1 < h:
                    ny, nx = cy + 1, cx
                    if m[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        comp[ny, nx] = True
                        stack.append((ny, nx))
                if cx > 0:
                    ny, nx = cy, cx - 1
                    if m[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        comp[ny, nx] = True
                        stack.append((ny, nx))
                if cx + 1 < w:
                    ny, nx = cy, cx + 1
                    if m[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        comp[ny, nx] = True
                        stack.append((ny, nx))

            comps.append(comp)

    return comps


def gt_instances_semantic_for_class(batch: dict, img_index: int, class_id: int, out_hw: tuple[int, int] | None = None):
    """
    From semantic GT mask:
      - select pixels == class_id
      - split into instances (connected components)
    Returns list of instance masks (bool 2D). Optionally resized to out_hw.
    """
    m = gt_semantic_mask_image(batch, img_index)
    if m is None:
        return []

    # operate on GT native resolution (small, fast)
    m2 = np.asarray(m).astype(np.int64)
    sel = (m2 == int(class_id))
    if not sel.any():
        return []

    comps = connected_components_bool(sel, max_instances=200)

    if out_hw is not None and len(comps) > 0:
        comps = [_resize_mask_nn(c.astype(np.uint8), out_hw).astype(bool) for c in comps]

    return comps


def gt_instances_semantic_all_classes(batch: dict, img_index: int, class_ids, out_hw: tuple[int, int] | None = None):
    """
    Instances for all provided class_ids. Returns list of instance masks (bool) for rendering.
    """
    all_comps = []
    for cid in class_ids:
        comps = gt_instances_semantic_for_class(batch, img_index, cid, out_hw=out_hw)
        all_comps.extend(comps)
    return all_comps


# -------------------------
# mask filtering + rendering
# -------------------------

def filter_by_class(masks_bool, cls_ids, class_id: int):
    """Keep only predicted instances with cls==class_id."""
    if masks_bool is None:
        return None, None
    masks_bool = np.asarray(masks_bool)
    if masks_bool.ndim == 2:
        masks_bool = masks_bool[None, ...]

    if cls_ids is None:
        return None, None

    cls_ids = np.asarray(cls_ids).reshape(-1)
    n = min(masks_bool.shape[0], cls_ids.shape[0])
    masks_bool = masks_bool[:n]
    cls_ids = cls_ids[:n]

    sel = (cls_ids == int(class_id))
    if not sel.any():
        return None, None
    return masks_bool[sel], cls_ids[sel]


def render_black_bg_instances_by_class(hw: tuple[int, int], masks_bool, cls_ids, class_colors, alpha: float = 0.90):
    """Pred rendering: black background + instances (colored by class)."""
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


def render_black_bg_instances_unique(hw: tuple[int, int], instance_masks_2d_list, alpha: float = 0.95):
    """
    GT rendering: black background + each instance gets its own color.
    instance_masks_2d_list: list of 2D bool masks (H,W)
    """
    H, W = hw
    base = np.zeros((H, W, 3), dtype=np.uint8)
    if not instance_masks_2d_list:
        return base

    cols = instance_palette(len(instance_masks_2d_list))
    out = base.astype(np.float32)

    for i, m in enumerate(instance_masks_2d_list):
        m = np.asarray(m).astype(bool)
        if m.shape != (H, W):
            m = _resize_mask_nn(m.astype(np.uint8), (H, W)).astype(bool)
        if not m.any():
            continue
        r, g, b = cols[i]
        out[m, 0] = (1 - alpha) * out[m, 0] + alpha * r
        out[m, 1] = (1 - alpha) * out[m, 1] + alpha * g
        out[m, 2] = (1 - alpha) * out[m, 2] + alpha * b

    return np.clip(out, 0, 255).astype(np.uint8)


# -------------------------
# main reusable grid builders
# -------------------------

def available_class_ids_in_batch(batch: dict, preds, names, max_show: int = 4):
    """
    Returns class_ids present in:
      - predictions (first max_show images)
      - semantic GT mask pixels (first max_show images)
    """
    names = _normalize_names(names)
    imgs = batch.get("img", None)
    if imgs is None or not isinstance(preds, (list, tuple)):
        return set()

    B = int(imgs.shape[0])
    n = min(B, max_show)

    present = set()

    # from preds
    for i in range(n):
        pred_masks, pred_cls, _ = pred_instances(preds[i]) if i < len(preds) else (None, None, None)
        if pred_masks is not None and pred_cls is not None:
            present.update(int(x) for x in np.asarray(pred_cls).reshape(-1).tolist())

    # from semantic GT pixels
    for i in range(n):
        m = gt_semantic_mask_image(batch, i)
        if m is None:
            continue
        u = np.unique(np.asarray(m).astype(np.int64))
        for cid in u.tolist():
            if cid == 0:
                continue
            present.add(int(cid))

    if names:
        present = {cid for cid in present if cid in names}

    return present


def make_val_grid_pred_gt_orig(batch: dict, preds, names, max_show: int = 4):
    """
    Grid (3xN):
      Row 1: black bg + Predictions (colored by class)
      Row 2: black bg + GT (semantic -> connected components, colored per instance)
      Row 3: Original image
    """
    names = _normalize_names(names)
    imgs = batch.get("img", None)
    if imgs is None or not isinstance(preds, (list, tuple)):
        return None

    B = int(imgs.shape[0])
    n = min(B, max_show)

    # pred colors
    nc = (max(names.keys()) + 1) if names else 1
    class_colors = class_palette(nc)

    # gt classes present in batch (for row2)
    gt_class_ids = sorted(list(available_class_ids_in_batch(batch, preds, names, max_show=max_show)))

    fig, axes = plt.subplots(3, n, figsize=(4 * n, 10))
    if n == 1:
        axes = np.array(axes).reshape(3, 1)

    for i in range(n):
        img_uint8 = chw_tensor_to_hwc_uint8(imgs[i])
        H, W = img_uint8.shape[:2]

        pred_masks, pred_cls, _pred_conf = pred_instances(preds[i]) if i < len(preds) else (None, None, None)

        # GT instances: semantic mask -> per class -> components
        gt_instances = gt_instances_semantic_all_classes(batch, i, gt_class_ids, out_hw=(H, W))

        if DEBUG_PRINT_SHAPES:
            pm = 0 if pred_masks is None else int(np.asarray(pred_masks).shape[0])
            gm = len(gt_instances)
            print(f"[yolo_val_viz] img[{i}] HxW={H}x{W} pred_inst={pm} gt_inst={gm}")

        row1 = render_black_bg_instances_by_class((H, W), pred_masks, pred_cls, class_colors, alpha=0.90)
        row2 = render_black_bg_instances_unique((H, W), gt_instances, alpha=0.95)
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
    Creates per-class grids ONLY for classes that are present.
    Returns dict: {class_name: fig}

    Layout (3xN) per class:
      Row 1: black bg + Pred instances of that class only (colored by class)
      Row 2: black bg + GT instances of that class (semantic -> components, colored per instance)
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
            pred_m_f, pred_c_f = filter_by_class(pred_masks, pred_cls, cid)

            # GT instances from semantic mask for this class only (colored per instance)
            gt_instances = gt_instances_semantic_for_class(batch, i, cid, out_hw=(H, W))

            if (pred_m_f is not None) or (len(gt_instances) > 0):
                any_found = True

            row1 = render_black_bg_instances_by_class((H, W), pred_m_f, pred_c_f, class_colors, alpha=0.95)
            row2 = render_black_bg_instances_unique((H, W), gt_instances, alpha=0.95)
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
