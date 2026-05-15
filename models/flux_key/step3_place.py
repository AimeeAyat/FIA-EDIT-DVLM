"""
Step 3 — Direct placement + harmonization (the "best" composite result).

What this does:
  1. rembg: remove reference background (no grass bleed)
  2. PIL composite: paste clean reference at EXACT mask bbox position in source
     (pixel-accurate, ref_aligned.png is already sized to bbox dimensions)
  3. Encode composite → z_comp
  4. Add small noise (t_start=0.35): keeps 65% reference signal
  5. ~10-step denoising with text guidance + background latent masking
     (background tokens restored to source at every step — zero overhead)

No hooks. No V injection. Pure FLUX text + scene context harmonization.
Reference appearance comes directly from the composited latent.

This produced the result: reference dog visible on chair, background clean,
some cat at mask edges (expected with t_start=0.35 short denoising).
"""

import os
import numpy as np
import torch
from PIL import Image, ImageFilter
from einops import rearrange

from flux.sampling import get_schedule, prepare
from flux.util import load_ae, load_flow_model
from models.flux_key.step2_features import load_mask_from_json, mask_to_token_indices
from models.flux_key.step1_extract import mask_bbox as _mask_bbox


def _remove_bg(img: Image.Image) -> Image.Image:
    try:
        from rembg import remove
        return remove(img)
    except ImportError:
        print("  [INFO] pip install rembg for background removal")
        rgba = img.convert("RGBA"); rgba.putalpha(255); return rgba


@torch.no_grad()
def run_edit_place(source_path, ref_aligned_path, json_path, key,
                   prompt_edit,
                   t_start: float  = 0.35,
                   num_steps: int  = 28,
                   guidance: float = 3.5,
                   use_rembg: bool = True,
                   offload: bool   = False,
                   device: str     = "cuda",
                   out_dir: str    = "test_output/flux_key_place/",
                   ae=None, model=None, t5=None, clip_enc=None):

    os.makedirs(out_dir, exist_ok=True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    def _on(m):
        if offload: m.to(device)
    def _off(m):
        if offload: m.cpu(); torch.cuda.empty_cache()

    # ── mask + bbox ───────────────────────────────────────────────────────────
    mask_np          = load_mask_from_json(json_path, key)
    bbox             = [int(x) for x in _mask_bbox(mask_np)]
    tx0, ty0, tx1, ty1 = bbox
    bw, bh           = max(1, tx1 - tx0), max(1, ty1 - ty0)

    # ── reference: remove background + soft edge ──────────────────────────────
    ref_pil = Image.open(ref_aligned_path).convert("RGB").resize((bw, bh), Image.LANCZOS)
    if use_rembg:
        print("  Removing reference background...")
        ref_rgba = _remove_bg(ref_pil)
        r, g, b, a = ref_rgba.split()
        a_soft   = a.filter(ImageFilter.GaussianBlur(radius=1))
        ref_rgba = Image.merge("RGBA", (r, g, b, a_soft))
    else:
        ref_rgba = ref_pil.convert("RGBA"); ref_rgba.putalpha(255)

    # ── PIL composite ─────────────────────────────────────────────────────────
    src_pil   = Image.open(source_path).convert("RGB").resize((512, 512))
    composite = src_pil.copy().convert("RGBA")
    composite.paste(ref_rgba, (tx0, ty0), ref_rgba)
    composite = composite.convert("RGB")
    print(f"  Composited reference at bbox [{tx0},{ty0}]→[{tx1},{ty1}]")

    # ── encode source + composite ─────────────────────────────────────────────
    def _enc(pil):
        arr = np.array(pil).astype(np.float32) / 127.5 - 1.0
        return ae.encode(
            torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        ).to(torch.bfloat16)

    _on(ae.encoder)
    z_src  = _enc(src_pil)
    z_comp = _enc(composite)
    _off(ae.encoder)

    B, C, H_l, W_l = z_src.shape
    z_src_tok  = rearrange(z_src,  'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=2, p2=2)
    z_comp_tok = rearrange(z_comp, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=2, p2=2)
    N_tok = z_src_tok.shape[1]

    mask_indices = mask_to_token_indices(mask_np).to(device)
    bg_mask      = torch.ones(N_tok, dtype=torch.bool, device=device)
    bg_mask[mask_indices] = False
    bg_idx = bg_mask.nonzero(as_tuple=True)[0]
    print(f"  Tokens — fg: {len(mask_indices)}  bg: {len(bg_idx)}  t_start: {t_start}")

    # ── text embeddings ───────────────────────────────────────────────────────
    def _to_dev(d):
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in d.items()}

    _on(t5); _on(clip_enc)
    inp_edit = _to_dev(prepare(t5, clip_enc, z_src, prompt=prompt_edit))
    _off(t5); _off(clip_enc)

    g_edit = torch.full((B,), guidance, device=device, dtype=z_src_tok.dtype)

    # ── schedule: start from t_start (~10 steps for t_start=0.35) ────────────
    all_ts    = get_schedule(num_steps, N_tok, shift=True)
    start_idx = next((i for i, t in enumerate(all_ts) if t <= t_start), 0)
    timesteps = all_ts[start_idx:]
    print(f"  Denoising {len(timesteps)-1} steps: "
          f"t={timesteps[0]:.3f} → {timesteps[-1]:.3f}")

    # pre-compute source noise schedule with fixed ε (for bg masking)
    eps_bg   = torch.randn_like(z_src_tok)
    z_src_at = [(1.0 - t) * z_src_tok + t * eps_bg for t in timesteps]

    # ── init: noised composite (65% reference signal preserved at t=0.35) ─────
    eps_init = torch.randn_like(z_comp_tok)
    z = (1.0 - timesteps[0]) * z_comp_tok + timesteps[0] * eps_init

    # ── denoising loop — no hooks, just text + bg masking ────────────────────
    _on(model)
    for step_idx, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((B,), t_curr, device=device, dtype=z.dtype)

        pred = model(img=z, img_ids=inp_edit["img_ids"],
                     txt=inp_edit["txt"], txt_ids=inp_edit["txt_ids"],
                     timesteps=t_vec, y=inp_edit["vec"], guidance=g_edit)

        if pred.isnan().any():
            pred = pred.nan_to_num(0.0)

        z = z + (t_prev - t_curr) * pred

        # restore background to source at correct noise level — free, no forward pass
        z[:, bg_idx, :] = z_src_at[step_idx + 1][:, bg_idx, :]

    _off(model)

    # ── decode ────────────────────────────────────────────────────────────────
    z_out = rearrange(z, 'b (h w) (c p1 p2) -> b c (h p1) (w p2)',
                      h=H_l//2, w=W_l//2, p1=2, p2=2, c=C)
    _on(ae.decoder)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        decoded = ae.decode(z_out)
    _off(ae.decoder)

    decoded = decoded.nan_to_num(0.0).clamp(-1, 1).cpu()
    out_arr = rearrange(decoded[0], 'c h w -> h w c')
    out_img = Image.fromarray((127.5 * (out_arr + 1.0)).byte().numpy())

    out_img.save(os.path.join(out_dir, f"{key}_edited.png"))
    cmp = Image.new("RGB", (1024, 512))
    cmp.paste(src_pil, (0, 0)); cmp.paste(out_img, (512, 0))
    cmp.save(os.path.join(out_dir, f"{key}_comparison.png"))
    print(f"  Saved: {out_dir}/{key}_comparison.png")
    return out_img
