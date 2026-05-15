"""
Step 3 v2 — FlowEdit + noise-consistent reference injection.

Fixes the three fundamental failures of v1 (SSI-based):

  1. NO partial noising:  FlowEdit dual-branch keeps all tokens at the
     same timestep t at every step → eliminates sparkle artifacts.

  2. Noise-consistent features: source and reference are BOTH noised to
     the current timestep t using the SAME ε before their K,V are
     extracted → no clean/noisy magnitude mismatch, no NaN.

  3. V preserved for foreground: per QK-Edit (ICCV 2025), replacing V
     in FLUX's joint attention suppresses editability.  We inject K only
     for foreground; V stays from the target branch.

Algorithm (per timestep t_i):
  ε  ~ N(0,1)                              (shared noise)
  z_src_t = (1-t_i)·z_src + t_i·ε         source at t_i
  z_ref_t = (1-t_i)·z_ref + t_i·ε         reference at SAME t_i
  z_tar_t = z_FE + z_src_t - z_src         FlowEdit target alignment

  v_src, K_bg, V_bg  ← forward(z_src_t)   extract bg features + src velocity
  K_ref              ← forward(z_ref_t)   extract fg reference K
  v_tar              ← forward(z_tar_t, inject K_bg,V_bg,K_ref)

  v_Δ = v_tar - v_src                      delta cancels shared noise
  z_FE += (t_{i-1} - t_i) · v_Δ
"""

import os
import numpy as np
import torch
from PIL import Image
from einops import rearrange

from flux.sampling import get_schedule, prepare
from flux.util import load_ae, load_flow_model
from models.flux_key.step2_features import load_mask_from_json, mask_to_token_indices


# ── Read-only hooks (extract K or K+V from a forward pass) ───────────────────

class _ReadHooks:
    """
    Attaches read-only hooks to extract background K,V  (capture_bg=True)
    or foreground K  (capture_bg=False) from a forward pass.
    Does NOT modify model outputs.
    """

    def __init__(self, fg_idx, device, capture_bg=True):
        self.fg_idx    = fg_idx.to(device)
        self.device    = device
        self.capture_bg = capture_bg
        self.data      = {}
        self._handles  = []

    # ── double-block hook ────────────────────────────────────────────────────
    def _dbl(self, idx):
        TXT_LEN = 512  # not used in double blocks, but keep for reference
        def hook(module, inp, out):
            B, N, three_HD = out.shape
            HD = three_HD // 3
            K = out[:, :, HD:2*HD]
            V = out[:, :, 2*HD:]

            fg = self.fg_idx[self.fg_idx < N]
            bg_mask = torch.ones(N, dtype=torch.bool, device=self.device)
            bg_mask[fg] = False
            bg = bg_mask.nonzero(as_tuple=True)[0]

            if self.capture_bg:
                self.data[f"d{idx}_K"]  = K[:, bg, :].detach().cpu()
                self.data[f"d{idx}_V"]  = V[:, bg, :].detach().cpu()
                self.data[f"d{idx}_bg"] = bg.cpu()
            else:
                if len(fg) > 0:
                    self.data[f"d{idx}_K"]  = K[:, fg, :].detach().cpu()
                    self.data[f"d{idx}_fg"] = fg.cpu()
        return hook

    # ── single-block hook ────────────────────────────────────────────────────
    def _sgl(self, idx):
        TXT_LEN = 512
        HD      = 3072
        def hook(module, inp, out):
            B, L, _ = out.shape
            if L <= TXT_LEN:
                return
            K_all = out[:, :, HD:2*HD]
            V_all = out[:, :, 2*HD:3*HD]
            N_img = L - TXT_LEN

            fg_img  = self.fg_idx[self.fg_idx < N_img]
            fg_full = fg_img + TXT_LEN
            bg_img  = torch.ones(N_img, dtype=torch.bool, device=self.device)
            bg_img[fg_img] = False
            bg_full = bg_img.nonzero(as_tuple=True)[0] + TXT_LEN

            if self.capture_bg:
                self.data[f"s{idx}_K"]  = K_all[:, bg_full, :].detach().cpu()
                self.data[f"s{idx}_V"]  = V_all[:, bg_full, :].detach().cpu()
                self.data[f"s{idx}_bg"] = (bg_full - TXT_LEN).cpu()
            else:
                if len(fg_full) > 0:
                    self.data[f"s{idx}_K"]  = K_all[:, fg_full, :].detach().cpu()
                    self.data[f"s{idx}_fg"] = (fg_img).cpu()
        return hook

    def attach(self, model):
        for i, blk in enumerate(model.double_blocks):
            self._handles.append(
                blk.img_attn.qkv.register_forward_hook(self._dbl(i)))
        for i, blk in enumerate(model.single_blocks):
            self._handles.append(
                blk.linear1.register_forward_hook(self._sgl(i)))

    def detach(self):
        for h in self._handles: h.remove()
        self._handles.clear()


# ── Write hooks (inject background K,V and reference K into target pass) ─────

class _InjectHooks:
    """
    Background tokens  → K,V replaced from source branch (noise-consistent).
    Foreground tokens  → K blended: α·K_target + (1-α)·K_ref  (V unchanged).
    """

    def __init__(self, bg_data, ref_data, fg_idx,
                 n_double, n_single,
                 alpha_max=0.3, alpha_min=0.05,
                 device="cuda"):
        self.bg        = bg_data
        self.ref       = ref_data
        self.fg_idx    = fg_idx.to(device)
        self.n_double  = n_double
        self.n_single  = n_single
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        self.device    = device
        self._handles  = []

    def _alpha(self, ell):
        L    = self.n_double + self.n_single
        base = self.alpha_max - (ell / max(1, L - 1)) * (self.alpha_max - self.alpha_min)
        return float(base)

    @staticmethod
    def _norm_match(src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Scale src to match tgt's per-token L2 norm."""
        t_n = tgt.float().norm(dim=-1, keepdim=True).clamp(min=1e-6)
        s_n = src.float().norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return (src.float() * (t_n / s_n)).to(tgt.dtype)

    # ── double-block hook ────────────────────────────────────────────────────
    def _dbl(self, idx):
        def hook(module, inp, out):
            B, N, three_HD = out.shape
            HD  = three_HD // 3
            Q   = out[:, :, :HD]
            K   = out[:, :, HD:2*HD]
            V   = out[:, :, 2*HD:]

            fg = self.fg_idx[self.fg_idx < N]
            bg_mask = torch.ones(N, dtype=torch.bool, device=self.device)
            bg_mask[fg] = False
            bg = bg_mask.nonzero(as_tuple=True)[0]

            K_new = K.clone()
            V_new = V.clone()

            # ── background: hard replace from source (same t) ────────────
            k_bg = self.bg.get(f"d{idx}_K")
            v_bg = self.bg.get(f"d{idx}_V")
            if k_bg is not None and len(bg) > 0:
                K_new[:, bg, :] = k_bg.to(self.device).nan_to_num(0.0)
                V_new[:, bg, :] = v_bg.to(self.device).nan_to_num(0.0)

            # ── foreground: blend K only (V from target preserved) ────────
            k_ref = self.ref.get(f"d{idx}_K")
            if k_ref is not None and len(fg) > 0:
                k_ref = self._norm_match(
                    k_ref.to(self.device).nan_to_num(0.0), K[:, fg, :])
                a = self._alpha(idx)
                K_new[:, fg, :] = a * K[:, fg, :] + (1 - a) * k_ref

            return torch.cat([Q, K_new, V_new], dim=-1)
        return hook

    # ── single-block hook ────────────────────────────────────────────────────
    def _sgl(self, idx):
        TXT_LEN = 512
        HD      = 3072
        def hook(module, inp, out):
            B, L, D = out.shape
            if L <= TXT_LEN:
                return out
            Q_all  = out[:, :, :HD]
            K_all  = out[:, :, HD:2*HD]
            V_all  = out[:, :, 2*HD:3*HD]
            rest   = out[:, :, 3*HD:]
            N_img  = L - TXT_LEN

            fg_img  = self.fg_idx[self.fg_idx < N_img]
            fg_full = fg_img + TXT_LEN
            bg_img  = torch.ones(N_img, dtype=torch.bool, device=self.device)
            bg_img[fg_img] = False
            bg_full = bg_img.nonzero(as_tuple=True)[0] + TXT_LEN

            K_new = K_all.clone()
            V_new = V_all.clone()

            # background
            k_bg = self.bg.get(f"s{idx}_K")
            v_bg = self.bg.get(f"s{idx}_V")
            if k_bg is not None and len(bg_full) > 0:
                K_new[:, bg_full, :] = k_bg.to(self.device).nan_to_num(0.0)
                V_new[:, bg_full, :] = v_bg.to(self.device).nan_to_num(0.0)

            # foreground (K only)
            k_ref = self.ref.get(f"s{idx}_K")
            if k_ref is not None and len(fg_full) > 0:
                k_ref = self._norm_match(
                    k_ref.to(self.device).nan_to_num(0.0), K_all[:, fg_full, :])
                ell = self.n_double + idx
                a   = self._alpha(ell)
                K_new[:, fg_full, :] = a * K_all[:, fg_full, :] + (1 - a) * k_ref

            return torch.cat([Q_all, K_new, V_new, rest], dim=-1)
        return hook

    def attach(self, model):
        for i, blk in enumerate(model.double_blocks):
            self._handles.append(
                blk.img_attn.qkv.register_forward_hook(self._dbl(i)))
        for i, blk in enumerate(model.single_blocks):
            self._handles.append(
                blk.linear1.register_forward_hook(self._sgl(i)))
        print(f"  Attached {len(self._handles)} v2 injection hooks")

    def detach(self):
        for h in self._handles: h.remove()
        self._handles.clear()


# ── Main editing function ─────────────────────────────────────────────────────

@torch.no_grad()
def run_edit_v2(source_path, ref_aligned_path, json_path, key,
                prompt_edit, prompt_src="",
                num_steps=28, guidance=3.5,
                alpha_max=0.3, alpha_min=0.05,
                device="cuda", out_dir="test_output/flux_key_v2/",
                ae=None, model=None, t5=None, clip_enc=None):
    """
    FlowEdit + noise-consistent reference injection.

    Parameters
    ----------
    source_path     : path to source image I_A
    ref_aligned_path: path to aligned reference crop (output of Step 1)
    json_path       : PIE-Bench mapping_file.json
    key             : PIE-Bench sample key string
    prompt_edit     : editing prompt (e.g. "a dog sitting on a wooden chair")
    prompt_src      : source description (default: "" = unconditional)
    alpha_max       : text weight at early blocks (lower → more reference)
    alpha_min       : text weight at late blocks
    """
    os.makedirs(out_dir, exist_ok=True)

    # ── models ────────────────────────────────────────────────────────────────
    from flux.util import load_t5 as _lt5, load_clip as _lclip
    if ae is None:
        ae = load_ae("flux-dev", device)
    if model is None:
        model = load_flow_model("flux-dev", device=device)
        model.eval()
    if t5 is None:
        t5 = _lt5(device, max_length=512)
    if clip_enc is None:
        clip_enc = _lclip(device)

    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    # ── encode images ─────────────────────────────────────────────────────────
    def _load(path):
        img = Image.open(path).convert("RGB").resize((512, 512))
        arr = np.array(img).astype(np.float32) / 127.5 - 1.0
        return img, torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    src_pil, src_t = _load(source_path)
    _,        ref_t = _load(ref_aligned_path)

    z_src = ae.encode(src_t.to(device)).to(torch.bfloat16)
    z_ref = ae.encode(ref_t.to(device)).to(torch.bfloat16)

    B, C, H_l, W_l = z_src.shape
    # pack to token sequence [B, N, C*p1*p2]
    z_src_tok = rearrange(z_src, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=2, p2=2)
    z_ref_tok = rearrange(z_ref, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=2, p2=2)
    N_tok = z_src_tok.shape[1]

    # ── mask ──────────────────────────────────────────────────────────────────
    mask_np      = load_mask_from_json(json_path, key)
    mask_indices = mask_to_token_indices(mask_np).to(device)
    print(f"Mask tokens: {len(mask_indices)} / {N_tok}")

    # ── text embeddings (computed once) ───────────────────────────────────────
    def _to_dev(d):
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in d.items()}

    inp_edit = _to_dev(prepare(t5, clip_enc, z_src, prompt=prompt_edit))
    inp_src  = _to_dev(prepare(t5, clip_enc, z_src, prompt=prompt_src))

    g_edit = torch.full((B,), guidance, device=device, dtype=z_src_tok.dtype)
    g_1    = torch.ones (B,             device=device, dtype=z_src_tok.dtype)

    # ── denoising schedule ────────────────────────────────────────────────────
    timesteps  = get_schedule(num_steps, N_tok, shift=True)
    n_double   = len(model.double_blocks)
    n_single   = len(model.single_blocks)

    # ── FlowEdit loop ─────────────────────────────────────────────────────────
    z_FE = z_src_tok.clone()
    print(f"FlowEdit v2: {len(timesteps)-1} steps, guidance={guidance}")
    print(f"  alpha_max={alpha_max}  alpha_min={alpha_min}")

    for step_idx, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        # shared noise — ALL branches noised with the SAME ε
        eps = torch.randn_like(z_src_tok)

        z_src_t = (1.0 - t_curr) * z_src_tok + t_curr * eps   # source at t
        z_ref_t = (1.0 - t_curr) * z_ref_tok + t_curr * eps   # reference at t
        z_tar_t = z_FE + z_src_t - z_src_tok                   # FlowEdit target

        t_vec = torch.full((B,), t_curr, device=device, dtype=z_src_tok.dtype)

        # ── Pass 1: source → get background K,V + source velocity ─────────
        bg_hooks = _ReadHooks(mask_indices, device, capture_bg=True)
        bg_hooks.attach(model)
        v_src = model(img=z_src_t, img_ids=inp_src["img_ids"],
                      txt=inp_src["txt"], txt_ids=inp_src["txt_ids"],
                      timesteps=t_vec, y=inp_src["vec"], guidance=g_1)
        bg_hooks.detach()
        bg_data = bg_hooks.data

        # ── Pass 2: reference → get foreground K ──────────────────────────
        ref_hooks = _ReadHooks(mask_indices, device, capture_bg=False)
        ref_hooks.attach(model)
        _ = model(img=z_ref_t, img_ids=inp_edit["img_ids"],
                  txt=inp_edit["txt"], txt_ids=inp_edit["txt_ids"],
                  timesteps=t_vec, y=inp_edit["vec"], guidance=g_1)
        ref_hooks.detach()
        ref_data = ref_hooks.data

        # ── Pass 3: target → inject K_bg,V_bg,K_ref; get edit velocity ────
        inj = _InjectHooks(bg_data, ref_data, mask_indices,
                           n_double, n_single,
                           alpha_max=alpha_max, alpha_min=alpha_min,
                           device=device)
        inj.attach(model)
        v_tar = model(img=z_tar_t, img_ids=inp_edit["img_ids"],
                      txt=inp_edit["txt"], txt_ids=inp_edit["txt_ids"],
                      timesteps=t_vec, y=inp_edit["vec"], guidance=g_edit)
        inj.detach()

        # ── NaN guard ─────────────────────────────────────────────────────
        nan_found = v_tar.isnan().any() or v_src.isnan().any()
        if nan_found:
            v_tar = v_tar.nan_to_num(0.0)
            v_src = v_src.nan_to_num(0.0)
            print(f"  [WARN] NaN clamped at t={t_curr:.3f}")

        # ── delta-velocity update ──────────────────────────────────────────
        z_FE = z_FE + (t_prev - t_curr) * (v_tar - v_src)

        if step_idx % 7 == 0 or step_idx == len(timesteps) - 2:
            print(f"  step {step_idx+1:02d}/{len(timesteps)-1}  t={t_curr:.3f}→{t_prev:.3f}")

    # ── decode ────────────────────────────────────────────────────────────────
    z_out = rearrange(z_FE, 'b (h w) (c p1 p2) -> b c (h p1) (w p2)',
                      h=H_l//2, w=W_l//2, p1=2, p2=2, c=C)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        decoded = ae.decode(z_out)
    decoded  = decoded.nan_to_num(0.0).clamp(-1, 1).cpu()
    out_arr  = rearrange(decoded[0], 'c h w -> h w c')
    out_img  = Image.fromarray((127.5 * (out_arr + 1.0)).byte().numpy())

    out_img.save(os.path.join(out_dir, f"{key}_edited.png"))
    cmp = Image.new("RGB", (512 * 2, 512))
    cmp.paste(src_pil, (0, 0))
    cmp.paste(out_img, (512, 0))
    cmp.save(os.path.join(out_dir, f"{key}_comparison.png"))
    print(f"Saved: {out_dir}/{key}_comparison.png")
    return out_img
