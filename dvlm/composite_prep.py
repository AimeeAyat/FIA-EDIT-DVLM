"""
Composite construction utilities for reference-guided composition.

Provides:
  - Poisson / seamless blending for natural boundary integration
  - Optional bounding-box expansion for partially-visible objects
"""

import numpy as np
import torch
import torch.nn.functional as F


# ── Seamless / Poisson composite ─────────────────────────────────────────────

def _to_cv_uint8(tensor_chw: torch.Tensor) -> np.ndarray:
    """Convert a [C,H,W] float32 tensor in [0,1] to uint8 HWC numpy."""
    arr = tensor_chw.permute(1, 2, 0).cpu().numpy()
    return (arr.clip(0, 1) * 255).astype(np.uint8)


def _from_cv_uint8(arr_hwc: np.ndarray) -> torch.Tensor:
    """Convert uint8 HWC numpy back to [C,H,W] float32 tensor in [0,1]."""
    return torch.from_numpy(arr_hwc.astype(np.float32) / 255.0).permute(2, 0, 1)


def seamless_composite(
    bg_tensor: torch.Tensor,       # [1,C,H,W] float32
    ref_cropped: torch.Tensor,     # [1,C,rh,rw] float32 — already resized ref crop
    mask_cropped: torch.Tensor,    # [1,1,rh,rw] float32 — binary mask of ref object
    y1: int, y2: int, x1: int, x2: int,
) -> torch.Tensor:
    """
    Blend the reference crop into the background using Poisson seamless cloning
    (OpenCV).  Falls back to a soft-edge alpha blend if seamlessClone fails.

    Returns a new [1,C,H,W] tensor (same shape as bg_tensor).
    """
    try:
        import cv2
    except ImportError:
        return _soft_blend_composite(bg_tensor, ref_cropped, mask_cropped, y1, y2, x1, x2)

    bg  = _to_cv_uint8(bg_tensor[0])               # HWC uint8
    ref = _to_cv_uint8(ref_cropped[0])              # rh×rw uint8
    msk = (mask_cropped[0, 0].cpu().numpy() * 255).astype(np.uint8)

    # Resize ref and mask to the bounding-box size
    bh, bw = y2 - y1, x2 - x1
    if ref.shape[:2] != (bh, bw):
        ref = cv2.resize(ref, (bw, bh), interpolation=cv2.INTER_LINEAR)
        msk = cv2.resize(msk, (bw, bh), interpolation=cv2.INTER_NEAREST)

    # Poisson clone centre position (must be inside both images)
    cx = x1 + bw // 2
    cy = y1 + bh // 2

    # Pad bg if the bounding box is too close to the border (seamlessClone fails
    # when centre is within ~3 px of the edge)
    pad = 4
    if cx < pad or cy < pad or cx >= bg.shape[1] - pad or cy >= bg.shape[0] - pad:
        return _soft_blend_composite(bg_tensor, ref_cropped, mask_cropped, y1, y2, x1, x2)

    try:
        result_bgr = cv2.seamlessClone(
            ref[:, :, ::-1],   # RGB → BGR
            bg[:, :, ::-1],
            msk,
            (cx, cy),
            cv2.NORMAL_CLONE,
        )
        result_rgb = result_bgr[:, :, ::-1]
        result_t = _from_cv_uint8(result_rgb).unsqueeze(0)
        return result_t
    except cv2.error:
        return _soft_blend_composite(bg_tensor, ref_cropped, mask_cropped, y1, y2, x1, x2)


def _soft_blend_composite(
    bg_tensor: torch.Tensor,
    ref_cropped: torch.Tensor,
    mask_cropped: torch.Tensor,
    y1: int, y2: int, x1: int, x2: int,
    edge_blur_px: int = 4,
) -> torch.Tensor:
    """
    Hard paste with a soft Gaussian edge to reduce boundary artefacts.
    """
    bh, bw = y2 - y1, x2 - x1
    ref_r = F.interpolate(ref_cropped, size=(bh, bw), mode="bilinear", align_corners=False)
    msk_r = F.interpolate(mask_cropped, size=(bh, bw), mode="bilinear", align_corners=False)

    # Blur the mask slightly for soft edges
    if edge_blur_px > 0:
        k = edge_blur_px * 2 + 1
        pad_amt = edge_blur_px
        weight = torch.ones(1, 1, k, k, dtype=msk_r.dtype, device=msk_r.device) / (k * k)
        msk_r = F.pad(msk_r, [pad_amt] * 4, mode="reflect")
        msk_r = F.conv2d(msk_r, weight)
    msk_r = msk_r.clamp(0, 1)

    composite = bg_tensor.clone()
    composite[:, :, y1:y2, x1:x2] = (
        composite[:, :, y1:y2, x1:x2] * (1 - msk_r)
        + ref_r * msk_r
    )
    return composite


# ── Bounding-box expansion for partial objects ────────────────────────────────

def expand_bbox_for_partial(
    x1: int, y1: int, x2: int, y2: int,
    img_h: int, img_w: int,
    expand_bottom_frac: float = 0.18,
    expand_top_frac: float = 0.0,
    expand_sides_frac: float = 0.0,
) -> tuple:
    """
    Directional bounding-box expansion for partially visible objects.

    Why directional?
    ─ Objects are almost always cut off at the BOTTOM (legs, feet, base).
    ─ Expanding left/right or top would regenerate surrounding background
      (grass, sky, other objects) unnecessarily.
    ─ By default only the bottom is expanded; sides and top are left alone.

    RF-Inversion ensures the expanded background strip around the object
    stays close to the original pixels even inside the enlarged bbox.

    Args:
        expand_bottom_frac:  fraction of bbox height to add BELOW y2 (default 0.18)
        expand_top_frac:     fraction to add ABOVE y1   (default 0 — rarely needed)
        expand_sides_frac:   fraction to add LEFT/RIGHT (default 0 — avoids background)

    Returns (x1, y1, x2, y2) clamped to image bounds.
    """
    bh = y2 - y1
    bw = x2 - x1
    return (
        max(0,    x1 - int(bw * expand_sides_frac)),
        max(0,    y1 - int(bh * expand_top_frac)),
        min(img_w, x2 + int(bw * expand_sides_frac)),
        min(img_h, y2 + int(bh * expand_bottom_frac)),
    )


