"""
Decoding helpers for YOLO11 segmentation validation batches.

Converts raw Ultralytics batch dicts and prediction outputs into numpy arrays
(boolean instance masks, class ids, confidences) that can be used for
visualisation or metric computation.

"""
import numpy as np


def _torch_to_np(x):
    try:
        return x.detach().cpu().numpy()
    except Exception:
        return None


def _normalize_names(names):
    """Normalise class names to dict[int, str] regardless of input type."""
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {i: str(n) for i, n in enumerate(names)}
    return {}


def _resize_mask_nn(mask: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbor resize for a single 2D mask without extra deps."""
    out_h, out_w = out_hw
    in_h, in_w = mask.shape[-2:]
    if (in_h, in_w) == (out_h, out_w):
        return mask
    y = np.linspace(0, in_h - 1, out_h).astype(np.int64)
    x = np.linspace(0, in_w - 1, out_w).astype(np.int64)
    return mask[y][:, x]


def _resize_masks_nn(masks: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Resize (N,H,W) -> (N,out_h,out_w) using nearest-neighbor interpolation."""
    if masks is None:
        return None
    masks = np.asarray(masks)
    if masks.ndim == 2:
        masks = masks[None, ...]
    out = np.zeros((masks.shape[0], out_hw[0], out_hw[1]), dtype=masks.dtype)
    for i in range(masks.shape[0]):
        out[i] = _resize_mask_nn(masks[i], out_hw)
    return out


def pred_instances(pred_item: dict):
    """Extract predicted masks, class ids, and confidences from one prediction.

    Expects the dict format produced by Ultralytics 8.3.x with keys
    'masks', 'cls', 'conf'.

    Args:
        pred_item: Prediction dict with keys 'bboxes', 'conf', 'cls', 'masks'.

    Returns:
        Tuple (masks_bool, cls_ids, confs) where masks_bool is (N,h,w) bool
        or None, cls_ids is (N,) int or None, confs is (N,) float or None.
        Predictions with confidence below 0.5 are filtered out.
    """
    masks = pred_item.get("masks", None)
    cls_ids = pred_item.get("cls", None)
    confs = pred_item.get("conf", None)

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


def gt_instances_from_batch(batch: dict, img_index: int, out_hw: tuple[int, int], names=None):
    """Extract ground truth instance masks and class ids for one image.

    Handles three mask formats Ultralytics may produce:

    A) Per-instance masks (N,h,w) with batch_idx (N,) and cls (N,).
    B) Overlap mask (B,h,w) where pixel value is a local instance id (1..k);
       class mapping comes from cls/batch_idx.
    C) Semantic mask (B,h,w) where pixel value is the class id (fallback).

    Args:
        batch: Ultralytics batch dict containing "masks", "batch_idx", "cls".
        img_index: Index of the image within the batch.
        out_hw: Target (H, W) to resize masks to.
        names: Class names as dict or list.

    Returns:
        Tuple (masks_bool, cls_ids) where masks_bool is (N,H,W) bool or None
        and cls_ids is (N,) int or None.
    """
    print("Entering gt_instances_from_batch with img_index:", img_index)  # Debug print
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

    # Case A: per-instance masks (N,h,w) with batch_idx
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

    # Cases B and C: per-image mask (B,h,w)
    if ma.ndim == 3:
        if img_index >= ma.shape[0]:
            return None, None

        m = ma[img_index]
        if m.shape != (H, W):
            m = _resize_mask_nn(m, (H, W))

        m_int = np.rint(m).astype(np.int64)
        u = np.unique(m_int)

        # Case C: semantic mask where pixel value == class id
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

        # Case B: overlap instance-id mask; no class mapping available
        if bidx is None or cls_all is None:

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


        return masks_bool, cls_ids

    return None, None


def filter_by_class(masks_bool, cls_ids, class_id: int):
    """Keep only masks belonging to a given class id.

    Args:
        masks_bool: (N,H,W) bool array or None.
        cls_ids: (N,) int array or None.
        class_id: Class id to filter for.

    Returns:
        Tuple (masks_bool, cls_ids) filtered to the requested class, or
        (None, None) if no matching instances are found.
    """
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


def available_class_ids_in_batch(batch: dict, preds, names, max_show: int = 4):
    """Collect class ids that appear in predictions or GT for a batch.

    Args:
        batch: Ultralytics batch dict.
        preds: List of per-image predictions.
        names: Class names as dict or list.
        max_show: Maximum number of images to inspect.

    Returns:
        Set of integer class ids present in either predictions or GT, filtered
        to only ids that appear in names.
    """
    names = _normalize_names(names)
    imgs = batch.get("img", None)
    if imgs is None or not isinstance(preds, (list, tuple)):
        return set()

    n = min(int(imgs.shape[0]), max_show)
    present = set()

    for i in range(n):
        pm, pc, _ = pred_instances(preds[i]) if i < len(preds) else (None, None, None)
        if pm is not None and pc is not None:
            present.update(int(x) for x in np.asarray(pc).reshape(-1).tolist())

    for i in range(n):
        # read image shape directly from the CHW tensor without converting to uint8
        img = imgs[i]
        H, W = int(img.shape[-2]), int(img.shape[-1])
        gm, gc = gt_instances_from_batch(batch, i, (H, W), names=names)
        if gc is not None:
            present.update(int(x) for x in np.asarray(gc).reshape(-1).tolist())

    if names:
        present = {cid for cid in present if cid in names}

    return present
