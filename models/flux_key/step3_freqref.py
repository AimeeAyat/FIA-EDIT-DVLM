"""
Step 3 v3b — Frequency-domain reference manipulation + composite.

Inspired by FSI-Edit (NeurIPS 2025) frequency residual fusion.

Core idea:
  In attention, V encodes BOTH structure (low-frequency) AND appearance (high-frequency).
  Simple V replacement → pose locked to reference (wrong).
  Simple V blend     → weak appearance transfer.

  Frequency solution:
    V_fused = IFFT(
        low_freq(FFT(V_current))    ← correct pose / spatial structure
      + high_freq(FFT(V_reference)) ← breed texture / fine appearance
    )

  This is the correct separation: current denoising drives the pose,
  reference drives the appearance. No magnitude mismatch because both
  are at the same noise level (reference noised to current timestep t).

Pipeline:
  1. rembg → PIL composite at mask bbox  (reference placed, background clean)
  2. Encode composite → z_comp            (reference content in latent)
  3. Noise to t_start=0.5                 (partial noise, reference still visible)
  4. Denoising loop (background latent masking at every step):
       - All steps: text + background masking
       - t < t_inject: extract V_ref at same t, freq_blend into foreground V
  5. Decode

freq_blend vs plain V-blend:
  plain: V = alpha*V_cur + (1-alpha)*V_ref  → linear mix, mixes pose with appearance
  freq:  V = IFFT(low(V_cur) + high(V_ref)) → clean separation, no pose bleed
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


# ── Background removal ────────────────────────────────────────────────────────

def _remove_bg(img: Image.Image) -> Image.Image:
    try:
        from rembg import remove
        return remove(img)
    except ImportError:
        rgba = img.convert("RGBA"); rgba.putalpha(255); return rgba


# ── Frequency-domain V blending (FSI-Edit inspired) ──────────────────────────

def freq_blend_V(v_cur: torch.Tensor,
                 v_ref: torch.Tensor,
                 gamma: float = 0.5) -> torch.Tensor:
    """
    1-D rfft along token sequence (dim=1).
    v_cur : [B, L, D]  current denoising V (carries pose / global structure)
    v_ref : [B, L, D]  reference V at same noise level (carries appearance)
    gamma : fraction of low-freq bins taken from current (rest from reference)

    Result: current pose (low-freq) + reference appearance (high-freq).

    Magnitude normalisation: reference high-freq scaled to match current
    magnitude before mixing — prevents attention instability.
    """
    # normalise ref to match current magnitude per token
    c_norm = v_cur.float().norm(dim=-1, keepdim=True).clamp(min=1e-6)
    r_norm = v_ref.float().norm(dim=-1, keepdim=True).clamp(min=1e-6)
    v_ref_scaled = (v_ref.float() * (c_norm / r_norm))

    C = torch.fft.rfft(v_cur.float(),         dim=1)   # [B, F, D]
    R = torch.fft.rfft(v_ref_scaled,           dim=1)

    n_freq = C.shape[1]
    split  = max(1, int(gamma * n_freq))       # bins 0..split-1 from current
    mixed  = R.clone()
    mixed[:, :split, :] = C[:, :split, :]     # low-freq  ← current (pose)
    # high-freq (split..) stays from R         # high-freq ← reference (appearance)

    out = torch.fft.irfft(mixed, n=v_cur.shape[1], dim=1)
    return out.to(v_cur.dtype)


# ── Reference V extraction with freq-blend hook ───────────────────────────────

def _extract_ref_V(model, z_ref_t, inp, t_vec, g_1, fg_idx, n_double, device):
    """Collect reference V at foreground token positions (GPU, no CPU copy)."""
    ref_V   = {}
    handles = []

    def _make(idx):
        def h(module, inp_, out):
            HD = out.shape[2] // 3
            V  = out[:, :, 2*HD:]
            fg = fg_idx[fg_idx < out.shape[1]]
            if len(fg) > 0:
                ref_V[idx] = V[:, fg, :].detach()
        return h

    for i in range(n_double):
        handles.append(
            model.double_blocks[i].img_attn.qkv.register_forward_hook(_make(i)))
    with torch.no_grad():
        _ = model(img=z_ref_t, img_ids=inp["img_ids"], txt=inp["txt"],
                  txt_ids=inp["txt_ids"], timesteps=t_vec,
                  y=inp["vec"], guidance=g_1)
    for h in handles: h.remove()
    return ref_V


# ── Frequency-blend V injection hook ─────────────────────────────────────────

class _FreqInjectV:
    """
    Replaces foreground V with freq_blend_V(V_current, V_reference).
    Low-freq of current V: preserves pose/structure.
    High-freq of reference V: injects appearance/breed details.
    """

    def __init__(self, ref_V, fg_idx, n_double, gamma=0.5, device="cuda"):
        self.ref_V    = ref_V
        self.fg_idx   = fg_idx.to(device)
        self.n_double = n_double
        self.gamma    = gamma
        self.device   = device
        self._handles = []

    def _hook(self, idx):
        def h(module, inp_, out):
            B, N, three_HD = out.shape
            HD = three_HD // 3
            Q, K, V = out[:, :, :HD], out[:, :, HD:2*HD], out[:, :, 2*HD:]
            fg = self.fg_idx[self.fg_idx < N]
            if not len(fg): return out
            v_ref = self.ref_V.get(idx)
            if v_ref is None: return out

            V_new = V.clone()
            V_new[:, fg, :] = freq_blend_V(
                V[:, fg, :],    # current V for foreground tokens
                v_ref,          # reference V (already fg-indexed)
                gamma=self.gamma
            )
            return torch.cat([Q, K, V_new], dim=-1)
        return h

    def attach(self, model):
        for i in range(self.n_double):
            self._handles.append(
                model.double_blocks[i].img_attn.qkv.register_forward_hook(self._hook(i)))

    def detach(self):
        for h in self._handles: h.remove()
        self._handles.clear()


# ── Main ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_edit_freqref(source_path, ref_aligned_path, json_path, key,
                     prompt_edit,
                     t_start: float    = 0.5,
                     t_inject: float   = 0.7,
                     gamma: float      = 0.5,
                     inject_every: int = 2,
                     num_steps: int    = 28,
                     guidance: float   = 4.0,
                     use_rembg: bool   = True,
                     offload: bool     = False,
                     low_vram: bool    = False,
                     device: str       = "cuda",
                     out_dir: str      = "test_output/flux_key_freqref/",
                     ae=None, model=None, t5=None, clip_enc=None):
    """
    Frequency-domain reference blending.

    gamma : low-freq split fraction.
            0.3 = take 30% low-freq from current (mostly reference appearance).
            0.6 = take 60% low-freq from current (more structure from current).
    t_inject : start frequency blending early so breed appears during generation.
    t_start  : 0.5 — enough noise to blend boundaries while keeping reference shape.
    """
    os.makedirs(out_dir, exist_ok=True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    def _on(m):
        if (offload or low_vram) and m is not None: m.to(device)

    def _off(m):
        if (offload or low_vram) and m is not None:
            m.cpu(); torch.cuda.empty_cache()

    # ── mask + bbox ───────────────────────────────────────────────────────────
    mask_np = load_mask_from_json(json_path, key)
    bbox    = [int(x) for x in _mask_bbox(mask_np)]
    tx0, ty0, tx1, ty1 = bbox
    bw, bh  = max(1, tx1 - tx0), max(1, ty1 - ty0)

    # ── reference: rembg + composite ─────────────────────────────────────────
    ref_orig = Image.open(ref_aligned_path).convert("RGB").resize((bw, bh), Image.LANCZOS)
    if use_rembg:
        print("  Removing reference background...")
        ref_rgba = _remove_bg(ref_orig)
        r, g_, b, a = ref_rgba.split()
        ref_rgba = Image.merge("RGBA", (r, g_, b, a.filter(ImageFilter.GaussianBlur(1))))
    else:
        ref_rgba = ref_orig.convert("RGBA"); ref_rgba.putalpha(255)

    src_pil   = Image.open(source_path).convert("RGB").resize((512, 512))
    composite = src_pil.copy().convert("RGBA")
    composite.paste(ref_rgba, (tx0, ty0), ref_rgba)
    composite = composite.convert("RGB")
    print(f"  Composited at bbox [{tx0},{ty0},{tx1},{ty1}]")

    # reference full-frame for V extraction
    ref_on_white = Image.new("RGB", (512, 512), (255, 255, 255))
    if use_rembg:
        a512  = ref_rgba.split()[3].resize((512, 512))
        rgb512 = ref_orig.resize((512, 512), Image.LANCZOS)
        ref_rgba512 = rgb512.convert("RGBA")
        ref_rgba512.putalpha(a512)
        ref_on_white.paste(ref_rgba512, (0, 0), ref_rgba512)
    else:
        ref_on_white = ref_orig.resize((512, 512), Image.LANCZOS)

    # ── encode ────────────────────────────────────────────────────────────────
    def _enc(pil):
        arr = np.array(pil).astype(np.float32) / 127.5 - 1.0
        return ae.encode(
            torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        ).to(torch.bfloat16)

    _on(ae.encoder)
    z_src  = _enc(src_pil)
    z_comp = _enc(composite)
    z_ref  = _enc(ref_on_white)
    _off(ae.encoder)

    B, C, H_l, W_l = z_src.shape
    z_src_tok  = rearrange(z_src,  'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=2, p2=2)
    z_comp_tok = rearrange(z_comp, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=2, p2=2)
    z_ref_tok  = rearrange(z_ref,  'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=2, p2=2)
    N_tok = z_src_tok.shape[1]

    mask_indices = mask_to_token_indices(mask_np).to(device)
    bg_mask      = torch.ones(N_tok, dtype=torch.bool, device=device)
    bg_mask[mask_indices] = False
    bg_idx = bg_mask.nonzero(as_tuple=True)[0]
    print(f"  Foreground: {len(mask_indices)}/{N_tok}  "
          f"t_start={t_start}  t_inject={t_inject}  gamma={gamma}")

    # ── text embeddings ───────────────────────────────────────────────────────
    def _to_dev(d):
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in d.items()}

    if low_vram:
        inp_edit = _to_dev(prepare(t5, clip_enc, z_src.cpu(), prompt=prompt_edit))
    else:
        _on(t5); _on(clip_enc)
        inp_edit = _to_dev(prepare(t5, clip_enc, z_src, prompt=prompt_edit))
        _off(t5); _off(clip_enc)

    g_edit = torch.full((B,), guidance, device=device, dtype=z_src_tok.dtype)
    g_1    = torch.ones (B,            device=device, dtype=z_src_tok.dtype)
    n_double = len(model.double_blocks)

    # ── schedule ──────────────────────────────────────────────────────────────
    all_ts    = get_schedule(num_steps, N_tok, shift=True)
    start_idx = next((i for i, t in enumerate(all_ts) if t <= t_start), 0)
    timesteps = all_ts[start_idx:]
    print(f"  Steps: {len(timesteps)-1}  "
          f"t={timesteps[0]:.3f}→{timesteps[-1]:.3f}")

    eps_bg   = torch.randn_like(z_src_tok)
    z_src_at = [(1.0 - t) * z_src_tok + t * eps_bg for t in timesteps]

    # init: noised composite (reference placed, partially preserved)
    eps_init = torch.randn_like(z_comp_tok)
    z = (1.0 - timesteps[0]) * z_comp_tok + timesteps[0] * eps_init

    # ── denoising loop ────────────────────────────────────────────────────────
    _on(model)
    cached_ref_V  = {}
    inj_count = 0

    for step_idx, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((B,), t_curr, device=device, dtype=z.dtype)
        hooks = None

        if t_curr < t_inject:
            if inj_count % inject_every == 0:
                eps_ref = torch.randn_like(z_ref_tok)
                z_ref_t = (1.0 - t_curr) * z_ref_tok + t_curr * eps_ref
                cached_ref_V = _extract_ref_V(model, z_ref_t, inp_edit,
                                              t_vec, g_1, mask_indices,
                                              n_double, device)
            inj_count += 1
            if cached_ref_V:
                hooks = _FreqInjectV(cached_ref_V, mask_indices, n_double,
                                     gamma=gamma, device=device)
                hooks.attach(model)

        pred = model(img=z, img_ids=inp_edit["img_ids"],
                     txt=inp_edit["txt"], txt_ids=inp_edit["txt_ids"],
                     timesteps=t_vec, y=inp_edit["vec"], guidance=g_edit)

        if hooks is not None: hooks.detach()
        if pred.isnan().any(): pred = pred.nan_to_num(0.0)

        z = z + (t_prev - t_curr) * pred
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
    cmp = Image.new("RGB", (512 * 2, 512))
    cmp.paste(src_pil, (0, 0)); cmp.paste(out_img, (512, 0))
    cmp.save(os.path.join(out_dir, f"{key}_comparison.png"))
    print(f"  Saved: {out_dir}/{key}_comparison.png")
    return out_img
