"""
Step 3 v3 — Text-guided generation + late reference appearance injection.

Design rationale (based on failure analysis of v1/v2/v3-composite):

  v3-composite failure: pasting reference pixels directly forces the WRONG POSE
    (standing dog on grass → sitting cat position mismatch).
    10 denoising steps from t=0.35 cannot change a standing dog to a sitting dog.

  Correct approach:
    1. Start from NOISED SOURCE at t_start (default 0.75 for object replacement).
       The noisy source retains ~25% of spatial context (rough position, size, pose
       direction) WITHOUT locking the appearance or exact pose.
    2. Text drives structure: "a dog sitting on a wooden chair" → correct pose.
    3. Background latent masking restores chair/floor at every step (single tensor
       op, zero overhead, no extra forward pass).
    4. Reference V injection in LATE steps only (t < t_inject, default 0.35):
       after the pose is established, inject German Shepherd fur/breed appearance
       into the late double blocks. One extra forward pass per late step.

  This separates structure (text + noisy source context) from appearance (reference),
  which is the correct decomposition for object-category replacement.

Timing on RTX 5090:
  - ~17 steps × 1 pass  = 17 passes (t ≥ t_inject)
  - ~11 steps × 2 passes = 22 passes (t < t_inject, late injection active)
  - Total: ~39 passes ≈ 1.5-2 min  (vs v2: 84 passes ≈ 30 min)

For attribute edits (texture/colour, same pose):
  Use t_start=0.4, t_inject=0.6 — structure mostly preserved, appearance transferred.

For object replacement (cat→dog, etc.):
  Use t_start=0.75, t_inject=0.35 (defaults) — pose regenerated from text.
"""

import os
import numpy as np
import torch
from PIL import Image
from einops import rearrange

from flux.sampling import get_schedule, prepare
from flux.util import load_ae, load_flow_model
from models.flux_key.step2_features import load_mask_from_json, mask_to_token_indices


# ── Optional reference background removal ────────────────────────────────────

def _remove_bg(img: Image.Image) -> Image.Image:
    """rembg background removal; falls back to full RGBA if not installed."""
    try:
        from rembg import remove
        return remove(img)
    except ImportError:
        rgba = img.convert("RGBA")
        rgba.putalpha(255)
        return rgba


# ── Late-step reference V extraction (stays on GPU) ──────────────────────────

def _extract_ref_V(model, z_ref_t, inp, t_vec, g_1, fg_idx, n_double, device):
    """
    One forward pass on noised reference; collect V tensors from late double
    blocks (n_double//2 onward) for the foreground tokens.
    All tensors kept on GPU — no CPU round-trip.
    """
    ref_V  = {}
    handles = []

    def _make(idx):
        def h(module, inp_, out):
            B, N, three_HD = out.shape
            HD = three_HD // 3
            V  = out[:, :, 2*HD:]
            fg = fg_idx[fg_idx < N]
            if len(fg) > 0:
                ref_V[idx] = V[:, fg, :].detach()   # GPU, no copy
        return h

    for i in range(n_double):
        handles.append(
            model.double_blocks[i].img_attn.qkv.register_forward_hook(_make(i)))

    with torch.no_grad():
        _ = model(img=z_ref_t,
                  img_ids=inp["img_ids"], txt=inp["txt"],
                  txt_ids=inp["txt_ids"], timesteps=t_vec,
                  y=inp["vec"], guidance=g_1)

    for h in handles:
        h.remove()
    return ref_V


# ── Late-step V injection hook ────────────────────────────────────────────────

class _InjectV:
    """
    Injects reference V into late double blocks for foreground tokens.
    Blends: (1-alpha)*V_ref + alpha*V_target  (lower alpha = more reference).
    Normalises reference V to target magnitude before blending.
    """

    def __init__(self, ref_V, fg_idx, n_double, alpha=0.35, device="cuda"):
        self.ref_V   = ref_V
        self.fg_idx  = fg_idx.to(device)
        self.n_double = n_double
        self.alpha   = alpha
        self.device  = device
        self._handles = []

    def _hook(self, idx):
        def h(module, inp_, out):
            B, N, three_HD = out.shape
            HD = three_HD // 3
            Q = out[:, :, :HD]
            K = out[:, :, HD:2*HD]
            V = out[:, :, 2*HD:]

            fg = self.fg_idx[self.fg_idx < N]
            if len(fg) == 0:
                return out

            v_ref = self.ref_V.get(idx)
            if v_ref is None:
                return out

            # magnitude-normalise reference V to match target scale
            t_n = V[:, fg, :].float().norm(dim=-1, keepdim=True).clamp(min=1e-6)
            r_n = v_ref.float().norm(dim=-1, keepdim=True).clamp(min=1e-6)
            v_ref_n = (v_ref.float() * (t_n / r_n)).to(V.dtype)

            V_new = V.clone()
            V_new[:, fg, :] = self.alpha * V[:, fg, :] + (1 - self.alpha) * v_ref_n
            return torch.cat([Q, K, V_new], dim=-1)
        return h

    def attach(self, model):
        # inject into ALL double blocks — breed/appearance features form at all depths
        for i in range(self.n_double):
            self._handles.append(
                model.double_blocks[i].img_attn.qkv.register_forward_hook(
                    self._hook(i)))

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ── Main v3 function ──────────────────────────────────────────────────────────

@torch.no_grad()
def run_edit_v3(source_path, ref_aligned_path, json_path, key,
                prompt_edit,
                t_start: float      = 0.999,
                t_inject: float     = 0.65,
                inject_alpha: float = 0.15,
                inject_every: int   = 2,
                num_steps: int      = 28,
                guidance: float     = 5.0,
                use_rembg: bool     = True,
                offload: bool       = False,
                device: str         = "cuda",
                out_dir: str        = "test_output/flux_key_v3/",
                ae=None, model=None, t5=None, clip_enc=None):
    """
    Text-guided generation + reference appearance injection.

    offload=True: moves T5/CLIP/AE to CPU when not in use so the 12B FLUX
    transformer fits in VRAM without throttling.  Peak usage per phase:
      encode  : AE.encoder  ~0.5 GB
      prepare : T5+CLIP     ~11.5 GB
      denoise : FLUX        ~24.5 GB   ← bottleneck, runs at full clock
      decode  : AE.decoder  ~0.5 GB

    inject_every: run reference extraction every N injection steps (default 2)
    to halve the extra forward passes without measurable quality loss.
    """
    os.makedirs(out_dir, exist_ok=True)

    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    # ── helper: move model on/off GPU ─────────────────────────────────────────
    def _on(m):
        if offload:
            m.to(device)

    def _off(m):
        if offload:
            m.cpu()
            torch.cuda.empty_cache()

    # ── Phase 1: encode images  (only AE encoder needed) ─────────────────────
    src_pil = Image.open(source_path).convert("RGB").resize((512, 512))
    src_arr = np.array(src_pil).astype(np.float32) / 127.5 - 1.0
    src_t   = torch.from_numpy(src_arr).permute(2, 0, 1).unsqueeze(0)

    ref_pil_raw = Image.open(ref_aligned_path).convert("RGB")
    if use_rembg:
        print("  Removing reference background...")
        ref_rgba     = _remove_bg(ref_pil_raw)
        ref_on_white = Image.new("RGB", ref_rgba.size, (255, 255, 255))
        ref_on_white.paste(ref_rgba, mask=ref_rgba.split()[3])
        ref_pil = ref_on_white.resize((512, 512), Image.LANCZOS)
    else:
        ref_pil = ref_pil_raw.resize((512, 512), Image.LANCZOS)

    ref_arr = np.array(ref_pil).astype(np.float32) / 127.5 - 1.0
    ref_t_  = torch.from_numpy(ref_arr).permute(2, 0, 1).unsqueeze(0)

    _on(ae.encoder)
    z_src = ae.encode(src_t.to(device)).to(torch.bfloat16)
    z_ref = ae.encode(ref_t_.to(device)).to(torch.bfloat16)
    _off(ae.encoder)

    B, C, H_l, W_l = z_src.shape
    z_src_tok = rearrange(z_src, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=2, p2=2)
    z_ref_tok = rearrange(z_ref, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=2, p2=2)
    N_tok = z_src_tok.shape[1]

    # ── mask ──────────────────────────────────────────────────────────────────
    mask_np      = load_mask_from_json(json_path, key)
    mask_indices = mask_to_token_indices(mask_np).to(device)
    bg_mask      = torch.ones(N_tok, dtype=torch.bool, device=device)
    bg_mask[mask_indices] = False
    bg_idx = bg_mask.nonzero(as_tuple=True)[0]
    print(f"  Foreground tokens: {len(mask_indices)} / {N_tok}  "
          f"t_start={t_start}  t_inject={t_inject}")

    # ── Phase 2: text embeddings  (T5 + CLIP, then offloaded) ────────────────
    def _to_dev(d):
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in d.items()}

    _on(t5); _on(clip_enc)
    inp_edit = _to_dev(prepare(t5, clip_enc, z_src, prompt=prompt_edit))
    _off(t5); _off(clip_enc)

    g_edit   = torch.full((B,), guidance, device=device, dtype=z_src_tok.dtype)
    g_1      = torch.ones (B,            device=device, dtype=z_src_tok.dtype)

    # ── Phase 3: denoising  (FLUX on GPU for entire loop) ────────────────────
    all_ts    = get_schedule(num_steps, N_tok, shift=True)
    start_idx = next((i for i, t in enumerate(all_ts) if t <= t_start), 0)
    timesteps = all_ts[start_idx:]
    n_double  = len(model.double_blocks)

    n_inj  = sum(1 for t in timesteps[:-1] if t < t_inject)
    n_extr = (n_inj + inject_every - 1) // inject_every   # ref passes (every N)
    print(f"  Steps: {len(timesteps)-1}  "
          f"injection steps: {n_inj}  "
          f"ref extractions: {n_extr}  "
          f"(every {inject_every} inj steps)")

    eps_bg   = torch.randn_like(z_src_tok)
    z_src_at = [(1.0 - t) * z_src_tok + t * eps_bg for t in timesteps]

    z = z_src_at[0].clone()
    z[:, mask_indices, :] = torch.randn_like(z[:, mask_indices, :])

    _on(model)
    cached_ref_V = {}
    inj_step_count = 0

    for step_idx, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((B,), t_curr, device=device, dtype=z.dtype)
        hooks = None

        if t_curr < t_inject:
            # re-extract reference V every inject_every injection steps
            if inj_step_count % inject_every == 0:
                eps_ref = torch.randn_like(z_ref_tok)
                z_ref_t = (1.0 - t_curr) * z_ref_tok + t_curr * eps_ref
                cached_ref_V = _extract_ref_V(model, z_ref_t, inp_edit,
                                              t_vec, g_1, mask_indices,
                                              n_double, device)
            inj_step_count += 1
            if cached_ref_V:
                hooks = _InjectV(cached_ref_V, mask_indices, n_double,
                                 alpha=inject_alpha, device=device)
                hooks.attach(model)

        pred = model(img=z, img_ids=inp_edit["img_ids"],
                     txt=inp_edit["txt"], txt_ids=inp_edit["txt_ids"],
                     timesteps=t_vec, y=inp_edit["vec"], guidance=g_edit)

        if hooks is not None:
            hooks.detach()

        if pred.isnan().any():
            pred = pred.nan_to_num(0.0)
            print(f"  [WARN] NaN at t={t_curr:.3f}")

        z = z + (t_prev - t_curr) * pred
        z[:, bg_idx, :] = z_src_at[step_idx + 1][:, bg_idx, :]

    _off(model)

    # ── Phase 4: decode  (only AE decoder needed) ─────────────────────────────
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
    cmp.paste(src_pil, (0,   0))
    cmp.paste(out_img, (512, 0))
    cmp.save(os.path.join(out_dir, f"{key}_comparison.png"))
    print(f"  Saved: {out_dir}/{key}_comparison.png")
    return out_img
