# yolo_val_viz.py
import numpy as np
import matplotlib.pyplot as plt


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


def palette(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    cols = rng.integers(low=50, high=255, size=(max(n, 1), 3), dtype=np.uint8)
    return [tuple(c.tolist()) for c in cols]


def overlay_instances(base_uint8: np.ndarray, masks: np.ndarray, color_rgbs: list, alpha: float) -> np.ndarray:
    """
    base_uint8: HWC uint8
    masks: (N,H,W) boolean/0-1
    color_rgbs: list of (r,g,b) uint8
    """
    out = base_uint8.astype(np.float32).copy()
    h, w = out.shape[:2]

    if masks is None:
        return base_uint8

    masks = np.asarray(masks)
    if masks.size == 0:
        return base_uint8
    if masks.ndim == 2:
        masks = masks[None, ...]

    if masks.shape[-2:] != (h, w):
        return base_uint8  # mismatch -> skip

    for i in range(masks.shape[0]):
        m = masks[i].astype(bool)
        if not m.any():
            continue
        r, g, b = color_rgbs[i % len(color_rgbs)]
        out[m, 0] = (1 - alpha) * out[m, 0] + alpha * r
        out[m, 1] = (1 - alpha) * out[m, 1] + alpha * g
        out[m, 2] = (1 - alpha) * out[m, 2] + alpha * b

    return np.clip(out, 0, 255).astype(np.uint8)


def pred_instance_masks(pred_item):
    """
    Extract predicted instance masks from Ultralytics Results-like object.
    Returns (N,H,W) bool or None.
    """
    masks = getattr(pred_item, "masks", None)
    if masks is None:
        return None
    data = getattr(masks, "data", None)
    if data is None:
        return None
    m = data.detach().float().cpu().numpy()  # (N,H,W)
    return m > 0.5


def gt_instance_masks_from_batch(batch: dict, img_index: int, hw: tuple[int, int]):
    """
    Best-effort GT extraction. Ultralytics batch formats differ by version.
    Returns (N,H,W) bool or None.
    """
    H, W = hw
    masks_all = batch.get("masks", None)
    if masks_all is None:
        return None

    try:
        ma = masks_all.detach().float().cpu().numpy()

        # Case A: (B,H,W) semantic-ish
        if ma.ndim == 3 and "img" in batch and ma.shape[0] == int(batch["img"].shape[0]):
            m = ma[img_index] > 0.5
            return m[None, ...] if m.shape == (H, W) else None

        # Case B: (N,H,W) all instances, mapped by batch_idx
        bidx = batch.get("batch_idx", None)
        if bidx is None:
            return None
        bidx = bidx.detach().cpu().numpy().astype(int).reshape(-1)

        if ma.ndim == 3 and ma.shape[-2:] == (H, W) and ma.shape[0] == bidx.shape[0]:
            sel = (bidx == img_index)
            m = ma[sel] > 0.5
            return m if m.size else None

    except Exception:
        return None

    return None


def make_custom_val_grid(batch: dict, preds, max_show: int = 4):
    """
    Build a 3xN matplotlib figure (N <= max_show).

    Row 1: predicted instance masks on black background (no boxes)
    Row 2: original image
    Row 3: original with GT (green) + Pred overlay (colored), instances only (no background)

    preds: list/tuple of Ultralytics Results for each image in batch
    """
    imgs = batch.get("img", None)
    if imgs is None:
        return None

    if not isinstance(preds, (list, tuple)):
        return None

    B = int(imgs.shape[0])
    n = min(B, max_show)

    fig, axes = plt.subplots(3, n, figsize=(4 * n, 10))
    if n == 1:
        axes = np.array(axes).reshape(3, 1)

    pred_colors = palette(50, seed=1)
    gt_colors = [(0, 255, 0)] * 50

    for i in range(n):
        img_uint8 = chw_tensor_to_hwc_uint8(imgs[i])
        H, W = img_uint8.shape[:2]

        pred_masks = pred_instance_masks(preds[i]) if i < len(preds) else None
        gt_masks = gt_instance_masks_from_batch(batch, i, (H, W))

        black = np.zeros_like(img_uint8)
        row1 = overlay_instances(black, pred_masks, pred_colors, alpha=0.75)
        row2 = img_uint8
        row3 = img_uint8.copy()
        row3 = overlay_instances(row3, gt_masks, gt_colors, alpha=0.35)
        row3 = overlay_instances(row3, pred_masks, pred_colors, alpha=0.35)

        axes[0, i].imshow(row1)
        axes[1, i].imshow(row2)
        axes[2, i].imshow(row3)
        for r in range(3):
            axes[r, i].axis("off")

    axes[0, 0].set_ylabel("Pred (instances)\nblack bg", fontsize=12)
    axes[1, 0].set_ylabel("Original", fontsize=12)
    axes[2, 0].set_ylabel("GT (green) + Pred", fontsize=12)

    plt.tight_layout()
    return fig
