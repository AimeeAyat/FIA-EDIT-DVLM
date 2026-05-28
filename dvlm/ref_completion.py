"""
dvlm/ref_completion.py — Reference object pre-completion for partial/truncated objects.

Problem:
    TF-ICON reference images often show objects that are partially cut off at the
    image boundary (e.g. a sheep whose legs extend below the frame). The provided
    segmentation mask follows the visible pixels only, so the pasted composite has
    a floating torso. Reference token injection improves body identity, but it cannot
    invent correct legs if the reference itself shows none.

Solution:
    Before composition, detect whether the mask touches the image boundary.
    If so, extend the canvas (primarily downward) and fill the new region with either:

    Fast path  (~50–100 ms, default):
        OpenCV Navier-Stokes inpainting — a PDE-based classical method that
        propagates local texture from the visible edges into the blank zone.
        No ML model, no downloads.

    Quality path  (~2–4 s, opt-in via --complete_partial_flux):
        Re-use the already-loaded FLUX transformer to run a short inpainting
        pass (12 steps) on the extended canvas. Produces a photorealistic
        completion at the cost of extra inference time.

Both paths return an enlarged reference PIL image + an expanded binary mask PIL image,
ready to be passed back into the main composition pipeline.
"""

import numpy as np
from PIL import Image

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


# ── Partial detection ─────────────────────────────────────────────────────────

def is_partial(mask_pil: Image.Image, edge_px: int = 4) -> bool:
    """
    Return True if any foreground pixel (>128) falls within `edge_px` pixels of
    any image boundary, indicating the object is cropped by the image edge.
    """
    arr = np.array(mask_pil.convert("L"))
    h, w = arr.shape
    ep = max(1, edge_px)
    return bool(
        arr[:ep,  :].max() > 128 or    # top strip
        arr[-ep:, :].max() > 128 or    # bottom strip
        arr[:, :ep ].max() > 128 or    # left strip
        arr[:, -ep:].max() > 128       # right strip
    )


# ── Mask dilation helpers ─────────────────────────────────────────────────────

def _dilate_mask_downward(mask_arr: np.ndarray, extend_h: int) -> np.ndarray:
    """
    Extend `mask_arr` downward by `extend_h` pixels by continuing the lowest
    foreground pixel in each column.  The new region is filled for any column
    that has at least one foreground pixel in its bottom `extend_h` rows.
    """
    h, w = mask_arr.shape
    new_arr = np.zeros((h + extend_h, w), dtype=np.uint8)
    new_arr[:h] = mask_arr
    for x in range(w):
        # Find the lowest foreground pixel in this column
        col = mask_arr[:, x]
        fg_rows = np.where(col > 128)[0]
        if len(fg_rows) == 0:
            continue
        lowest = fg_rows[-1]
        # Only extend downward if the object reaches the bottom half of the image
        if lowest >= h * 0.35:
            new_arr[lowest : h + extend_h, x] = 255
    return new_arr


# ── Fast path: OpenCV classical inpainting ────────────────────────────────────

def complete_reference_fast(
    ref_pil: Image.Image,
    mask_pil: Image.Image,
    extend_frac: float = 0.28,
) -> tuple:
    """
    Extend the reference canvas downward and fill the new region using OpenCV
    Navier-Stokes inpainting.  No ML model required.

    Args:
        ref_pil:       Reference image (RGB PIL).
        mask_pil:      Binary segmentation mask (L or RGB PIL, white = object).
        extend_frac:   Fraction of the original image height to extend downward.

    Returns:
        (completed_ref_pil, expanded_mask_pil) — both larger than the inputs.
    """
    ref_pil  = ref_pil.convert("RGB")
    mask_pil = mask_pil.convert("L")

    w, h   = ref_pil.size
    ext_h  = max(4, int(h * extend_frac))
    new_h  = h + ext_h

    # ── extend the reference canvas ───────────────────────────────────────────
    ref_arr = np.array(ref_pil)                   # H × W × 3
    extended = np.zeros((new_h, w, 3), dtype=np.uint8)
    extended[:h] = ref_arr

    # ── build the inpainting mask for the extension zone ─────────────────────
    mask_arr = np.array(mask_pil)                 # H × W, uint8
    extended_mask = _dilate_mask_downward(mask_arr, ext_h)  # (H+ext_h) × W

    # Inpaint ONLY the extension region that belongs to the object
    inpaint_region = np.zeros((new_h, w), dtype=np.uint8)
    inpaint_region[h:] = extended_mask[h:]        # only the new rows

    if _CV2_AVAILABLE and inpaint_region.max() > 0:
        extended_bgr = cv2.cvtColor(extended, cv2.COLOR_RGB2BGR)
        filled_bgr   = cv2.inpaint(extended_bgr, inpaint_region, 5, cv2.INPAINT_NS)
        filled_rgb   = cv2.cvtColor(filled_bgr, cv2.COLOR_BGR2RGB)
    else:
        # Fallback: copy the last row of each column into the extension
        filled_rgb = extended.copy()
        for x in range(w):
            if inpaint_region[h, x] > 0:
                filled_rgb[h:, x] = ref_arr[-1, x]

    completed_ref_pil   = Image.fromarray(filled_rgb)
    expanded_mask_pil   = Image.fromarray(extended_mask)
    return completed_ref_pil, expanded_mask_pil


# ── Quality path: FLUX inpainting (already-loaded model) ─────────────────────

def complete_reference_flux(
    pipe,
    ref_pil: Image.Image,
    mask_pil: Image.Image,
    prompt: str = "complete the object, natural pose, full body",
    extend_frac: float = 0.28,
    steps: int = 12,
    guidance_scale: float = 3.5,
    seed: int = 42,
) -> tuple:
    """
    Use the already-loaded FLUX transformer to inpaint the reference extension zone.
    Significantly higher quality than the fast path at the cost of ~2–4 s.

    Re-uses `pipe.transformer`, `pipe.vae`, `pipe.text_encoder_*` etc.
    Wraps them in a `FluxInpaintPipeline` without downloading new weights.

    Args:
        pipe:          The loaded EnhancedFluxCompositionPipeline instance.
        ref_pil:       Reference image (RGB PIL).
        mask_pil:      Binary segmentation mask (L PIL, white = object).
        prompt:        Text prompt guiding the FLUX completion.
        extend_frac:   Canvas extension fraction.
        steps:         Number of FLUX denoising steps (12 = fast schedule).
        guidance_scale: FLUX guidance embedding value.
        seed:          Generator seed for reproducibility.

    Returns:
        (completed_ref_pil, expanded_mask_pil)
    """
    import torch
    from diffusers import FluxInpaintPipeline

    # First produce the fast-path canvas so we have the pixel-filled starting point
    fast_ref, expanded_mask = complete_reference_fast(ref_pil, mask_pil, extend_frac)

    # Build a lightweight FluxInpaintPipeline from already-loaded components
    inpaint_pipe = FluxInpaintPipeline(
        scheduler=pipe.scheduler,
        vae=pipe.vae,
        text_encoder=pipe.text_encoder,
        text_encoder_2=pipe.text_encoder_2,
        tokenizer=pipe.tokenizer,
        tokenizer_2=pipe.tokenizer_2,
        transformer=pipe.transformer,
    )
    # Keep on same device — no extra .to() needed
    inpaint_pipe.enable_model_cpu_offload = lambda: None   # already managed

    device = next(pipe.transformer.parameters()).device
    dtype  = next(pipe.transformer.parameters()).dtype

    # The inpainting mask: 1 = region to generate (extension zone)
    w, h_orig = ref_pil.size
    ext_h = max(4, int(h_orig * extend_frac))

    inpaint_mask_arr = np.zeros((h_orig + ext_h, w), dtype=np.uint8)
    inpaint_mask_arr[h_orig:] = np.array(expanded_mask.convert("L"))[h_orig:]
    inpaint_mask_pil = Image.fromarray(inpaint_mask_arr)

    # Resize to 512×512 (or nearest 64-divisible size) for FLUX
    tgt_w = 512
    tgt_h = int(round((h_orig + ext_h) / tgt_w * 512 / 64) * 64)
    tgt_h = max(64, tgt_h)

    fast_ref_rs   = fast_ref.resize((tgt_w, tgt_h), Image.LANCZOS)
    mask_rs       = inpaint_mask_pil.resize((tgt_w, tgt_h), Image.NEAREST)

    with torch.no_grad():
        result = inpaint_pipe(
            prompt=prompt,
            image=fast_ref_rs,
            mask_image=mask_rs,
            height=tgt_h,
            width=tgt_w,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=torch.Generator(device=device).manual_seed(seed),
            strength=0.99,
        )

    completed_rs = result.images[0]
    # Resize back to original dimensions
    completed_ref_pil = completed_rs.resize((w, h_orig + ext_h), Image.LANCZOS)
    return completed_ref_pil, expanded_mask
