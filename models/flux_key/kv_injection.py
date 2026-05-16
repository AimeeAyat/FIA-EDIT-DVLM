"""
KV injection hooks for Flux & Key.

Provides two functions that temporarily patch FLUX transformer blocks:
  1. freq_lock_background_kv  — replaces background K,V with frequency-mixed source/target
  2. inject_reference_kv      — replaces foreground K,V with text/reference interpolation

Both functions use register_forward_hook (which CAN return modified outputs in PyTorch)
and are designed to be applied to a model, used, then removed cleanly.

No modifications are made to original FLUX model files.
"""

import torch
import torch.nn.functional as F
from einops import rearrange


# ── Frequency mixing (background) ────────────────────────────────────────────

def freq_mix_1d(src: torch.Tensor, tgt: torch.Tensor, gamma: float = 0.5) -> torch.Tensor:
    """
    1-D rfft along the token-sequence dimension (dim=1).
    src : [B, L, D]  clean source features (t=0), magnitude-normalised to match tgt
    tgt : [B, L, D]  noisy target features (timestep t)
    gamma : fraction of low-freq bins taken from target; high-freq bins from source
    """
    src_f = src.float()
    tgt_f = tgt.float()
    # normalise src per-token to match tgt magnitude — prevents phase-amplitude spikes
    tgt_norm = tgt_f.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    src_norm = src_f.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    src_f = src_f * (tgt_norm / src_norm)

    S = torch.fft.rfft(src_f, dim=1)
    T = torch.fft.rfft(tgt_f, dim=1)
    split = max(1, int(gamma * S.shape[1]))
    mixed = T.clone()
    mixed[:, split:, :] = S[:, split:, :]
    out = torch.fft.irfft(mixed, n=tgt.shape[1], dim=1)
    return out.to(tgt.dtype)


def freq_mix_2d(src_full: torch.Tensor, tgt_full: torch.Tensor,
                bg_idx: torch.Tensor, gamma: float = 0.5,
                grid_h: int = 32, grid_w: int = 32) -> torch.Tensor:
    """
    2-D rfft on the full spatial grid, then extract background tokens.
    Preserves isotropic spatial frequencies (edges, textures) better than 1-D.

    src_full : [B, N, D]  source features for ALL N=1024 tokens
    tgt_full : [B, N, D]  current denoising features for ALL N=1024 tokens
    bg_idx   : [n_bg]     indices of background tokens in [0, N)
    Returns  : [B, n_bg, D]  mixed features for background tokens only
    """
    import math
    B, N, D = src_full.shape
    src_f = src_full.float()
    tgt_f = tgt_full.float()
    # normalise src per-token to match tgt magnitude
    tgt_norm = tgt_f.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    src_norm = src_f.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    src_f = src_f * (tgt_norm / src_norm)
    # rearrange to spatial grid [B, D, H, W]
    s = rearrange(src_f, 'b (h w) d -> b d h w', h=grid_h, w=grid_w)
    t = rearrange(tgt_f, 'b (h w) d -> b d h w', h=grid_h, w=grid_w)
    S = torch.fft.rfft2(s)   # [B, D, H, W//2+1]
    T = torch.fft.rfft2(t)
    H_f, W_f = S.shape[-2], S.shape[-1]
    # radial frequency mask
    u = torch.arange(H_f, device=src_full.device).float()
    v = torch.arange(W_f, device=src_full.device).float()
    uu, vv = torch.meshgrid(u, v, indexing='ij')
    max_r = math.sqrt((H_f - 1) ** 2 + (W_f - 1) ** 2) + 1e-6
    r = torch.sqrt(uu ** 2 + vv ** 2) / max_r   # [H_f, W_f], range [0,1]
    high_mask = (r > gamma).unsqueeze(0).unsqueeze(0)  # [1,1,H_f,W_f]
    mixed = T.clone()
    mixed[high_mask.expand_as(mixed)] = S[high_mask.expand_as(S)]
    result = torch.fft.irfft2(mixed, s=(grid_h, grid_w))   # [B, D, H, W]
    result_seq = rearrange(result, 'b d h w -> b (h w) d').to(tgt_full.dtype)
    return result_seq[:, bg_idx, :]


# keep old name as alias (used by existing code paths)
def freq_mix(src: torch.Tensor, tgt: torch.Tensor, gamma: float = 0.5) -> torch.Tensor:
    return freq_mix_1d(src, tgt, gamma)


# ── Hook builder ──────────────────────────────────────────────────────────────

class KVInjectionHooks:
    """
    Attaches forward hooks to every img QKV linear layer in the FLUX model.

    For each block at forward time:
      - Background tokens: K_bg = freq_mix(K_src, K_tgt), V_bg = freq_mix(V_src, V_tgt)
      - Foreground tokens: K_fg = alpha*K_text + (1-alpha)*K_ref,  same for V

    The hooks return modified QKV tensors, so the rest of the attention
    computation (RoPE, softmax) operates on the injected features automatically.
    """

    def __init__(self,
                 feat_src: dict,
                 feat_ref: dict,
                 mask_indices: torch.Tensor,
                 n_double: int,
                 n_single: int,
                 alpha_max: float = 0.8,
                 alpha_min: float = 0.2,
                 gamma: float = 0.5,
                 freq_2d: bool = False,
                 device: str = "cuda"):
        self.feat_src = feat_src
        self.feat_ref = feat_ref
        self.fg_idx   = mask_indices.to(device)
        self.bg_mask  = None   # built on first call based on actual seq len
        self.n_double = n_double
        self.n_single = n_single
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        self.gamma = gamma
        self.freq_2d = freq_2d
        self.device = device
        self._handles: list = []
        self._t: float = 0.0   # current timestep, updated each step
        # Per-tag device cache to avoid repeated CPU->GPU transfers every hook call.
        self._src_cache: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        self._ref_cache: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        # Keep only a few tensors resident on GPU at a time.
        self._cache_limit = 8

    # def _alpha(self, ell: int, L_total: int) -> float:
    #     """Layer-wise + time-aware schedule.
    #     base: early blocks → alpha_max (text-dominant), late blocks → alpha_min (ref-dominant)
    #     time: at high t, push alpha toward alpha_max to avoid clean/noisy mismatch.
    #     Net effect: reference injection is strongest at late blocks AND late timesteps (low t).
    #     """
    #     base = self.alpha_max - (ell / max(1, L_total - 1)) * (self.alpha_max - self.alpha_min)
    #     return float(base + (1.0 - base) * self._t)

    def _alpha(self, ell: int, L_total: int) -> float:
        """Layer-wise + time-aware schedule.
        base: early blocks → alpha_max (text-dominant), late blocks → alpha_min (ref-dominant)
        time: at high t, push alpha toward alpha_max to avoid clean/noisy mismatch.
        Net effect: reference injection is strongest at late blocks AND late timesteps (low t).
        """
        base = self.alpha_max - (ell / max(1, L_total - 1)) * (self.alpha_max - self.alpha_min)
        return float(base + (1.0 - base) * self._t)

    def _gamma(self) -> float:
        """Time-aware background lock strength.
        At t=1 (pure noise): gamma=1.0 → take all from target, no source injection.
        At t=0 (clean):      gamma=self.gamma (configured value, e.g. 0.5).
        Prevents magnitude mismatch between clean source and noisy target at high t.
        """
        return self.gamma + (1.0 - self.gamma) * self._t

    def _get_src(self, tag: str) -> torch.Tensor | None:
        t = self._src_cache.pop(tag, None)
        if t is not None:
            self._src_cache[tag] = t
            return t
        src = self.feat_src.get(tag)
        if src is None:
            return None
        t = src.to(self.device, dtype=torch.bfloat16).nan_to_num(0.0)
        if len(self._src_cache) >= self._cache_limit:
            _, old = self._src_cache.popitem(last=False)
            del old
        self._src_cache[tag] = t
        return t

    def _get_ref(self, tag: str) -> torch.Tensor | None:
        t = self._ref_cache.pop(tag, None)
        if t is not None:
            self._ref_cache[tag] = t
            return t
        ref = self.feat_ref.get(tag)
        if ref is None:
            return None
        t = ref.to(self.device, dtype=torch.bfloat16).nan_to_num(0.0)
        if len(self._ref_cache) >= self._cache_limit:
            _, old = self._ref_cache.popitem(last=False)
            del old
        self._ref_cache[tag] = t
        return t

    def _make_double_img_hook(self, block_idx: int):
        """Hook on DoubleStreamBlock img_attn.qkv — img tokens only, shape [B, N_img, 3*HD]."""
        def hook(module, inp, output):
            B, N, three_HD = output.shape
            HD = three_HD // 3
            Q   = output[:, :, :HD]
            K   = output[:, :, HD:2*HD]
            V   = output[:, :, 2*HD:]

            # Build bg/fg masks on first call
            bg_mask = torch.ones(N, dtype=torch.bool, device=self.device)
            fg = self.fg_idx[self.fg_idx < N]
            bg_mask[fg] = False
            bg = bg_mask.nonzero(as_tuple=True)[0]

            tag_k = f"double_{block_idx}_img_K"
            tag_v = f"double_{block_idx}_img_V"

            K_new = K.clone()
            V_new = V.clone()

            # ── background: frequency lock ────────────────────────────────
            if len(bg) > 0:
                k_src_full = self._get_src(tag_k)
                v_src_full = self._get_src(tag_v)
            else:
                k_src_full = v_src_full = None
            if k_src_full is not None and v_src_full is not None and len(bg) > 0:
                gamma = self._gamma()
                if self.freq_2d and k_src_full.shape[1] == N:
                    K_new[:, bg, :] = freq_mix_2d(k_src_full, K, bg, gamma)
                    V_new[:, bg, :] = freq_mix_2d(v_src_full, V, bg, gamma)
                else:
                    k_src_bg = k_src_full[:, bg, :] if k_src_full.shape[1] > max(bg).item() else k_src_full
                    v_src_bg = v_src_full[:, bg, :] if v_src_full.shape[1] > max(bg).item() else v_src_full
                    K_new[:, bg, :] = freq_mix_1d(k_src_bg, K[:, bg, :], gamma)
                    V_new[:, bg, :] = freq_mix_1d(v_src_bg, V[:, bg, :], gamma)

            # ── foreground: reference interpolation ───────────────────────
            if len(fg) > 0:
                k_ref = self._get_ref(tag_k)
                v_ref = self._get_ref(tag_v)
            else:
                k_ref = v_ref = None
            if k_ref is not None and v_ref is not None and len(fg) > 0:
                # ref was extracted on I_B which may have same N
                k_ref_fg = k_ref[:, fg, :] if k_ref.shape[1] > max(fg).item() else k_ref[:, :len(fg), :]
                v_ref_fg = v_ref[:, fg, :] if v_ref.shape[1] > max(fg).item() else v_ref[:, :len(fg), :]

                # normalise ref to match current magnitude before blending
                curr_k_norm = K[:, fg, :].float().norm(dim=-1, keepdim=True).clamp(min=1e-6)
                ref_k_norm  = k_ref_fg.float().norm(dim=-1, keepdim=True).clamp(min=1e-6)
                k_ref_fg = (k_ref_fg.float() * (curr_k_norm / ref_k_norm)).to(K.dtype)

                curr_v_norm = V[:, fg, :].float().norm(dim=-1, keepdim=True).clamp(min=1e-6)
                ref_v_norm  = v_ref_fg.float().norm(dim=-1, keepdim=True).clamp(min=1e-6)
                v_ref_fg = (v_ref_fg.float() * (curr_v_norm / ref_v_norm)).to(V.dtype)

                alpha = self._alpha(block_idx, self.n_double + self.n_single)
                K_new[:, fg, :] = alpha * K[:, fg, :] + (1 - alpha) * k_ref_fg
                V_new[:, fg, :] = alpha * V[:, fg, :] + (1 - alpha) * v_ref_fg

            return torch.cat([Q, K_new, V_new], dim=-1)
        return hook

    def _make_single_hook(self, block_idx: int):
        """
        Hook on SingleStreamBlock linear1 — concatenated [txt, img] tokens.
        Shape: [B, N_txt + N_img, 3*HD + MLP_dim]
        Only modify the img portion (tokens after txt_len=512).
        """
        TXT_LEN = 512
        HD = 3072   # FLUX hidden size

        def hook(module, inp, output):
            B, L, D_out = output.shape
            if L <= TXT_LEN:
                return output   # safety: skip if no img tokens

            # QKV slice: first 3*HD channels
            Q_all = output[:, :, :HD]
            K_all = output[:, :, HD:2*HD]
            V_all = output[:, :, 2*HD:3*HD]
            rest  = output[:, :, 3*HD:]

            # img token indices in full sequence
            img_offset = TXT_LEN
            N_img = L - TXT_LEN
            fg_in_img = self.fg_idx[self.fg_idx < N_img]
            fg_full = fg_in_img + img_offset

            bg_img = torch.ones(N_img, dtype=torch.bool, device=self.device)
            bg_img[fg_in_img] = False
            bg_full = bg_img.nonzero(as_tuple=True)[0] + img_offset

            tag_k = f"single_{block_idx}_K"
            tag_v = f"single_{block_idx}_V"

            K_new = K_all.clone()
            V_new = V_all.clone()

            # background frequency lock
            if len(bg_full) > 0:
                k_src_full = self._get_src(tag_k)
                v_src_full = self._get_src(tag_v)
            else:
                k_src_full = v_src_full = None
            if k_src_full is not None and v_src_full is not None and len(bg_full) > 0:
                bg_img_idx = bg_full - img_offset
                gamma = self._gamma()
                if self.freq_2d and k_src_full.shape[1] == N_img:
                    K_new[:, bg_full, :] = freq_mix_2d(k_src_full, K_all[:, img_offset:, :], bg_img_idx, gamma)
                    V_new[:, bg_full, :] = freq_mix_2d(v_src_full, V_all[:, img_offset:, :], bg_img_idx, gamma)
                else:
                    k_src_bg = k_src_full[:, bg_img_idx, :] if k_src_full.shape[1] > max(bg_img_idx).item() else k_src_full
                    v_src_bg = v_src_full[:, bg_img_idx, :] if v_src_full.shape[1] > max(bg_img_idx).item() else v_src_full
                    K_new[:, bg_full, :] = freq_mix_1d(k_src_bg, K_all[:, bg_full, :], gamma)
                    V_new[:, bg_full, :] = freq_mix_1d(v_src_bg, V_all[:, bg_full, :], gamma)

            # foreground reference injection
            if len(fg_full) > 0:
                k_ref = self._get_ref(tag_k)
                v_ref = self._get_ref(tag_v)
            else:
                k_ref = v_ref = None
            if k_ref is not None and v_ref is not None and len(fg_full) > 0:
                fg_img_idx = fg_full - img_offset
                k_ref_fg = k_ref[:, fg_img_idx, :] if k_ref.shape[1] > max(fg_img_idx).item() else k_ref[:, :len(fg_img_idx), :]
                v_ref_fg = v_ref[:, fg_img_idx, :] if v_ref.shape[1] > max(fg_img_idx).item() else v_ref[:, :len(fg_img_idx), :]

                curr_k_norm = K_all[:, fg_full, :].float().norm(dim=-1, keepdim=True).clamp(min=1e-6)
                ref_k_norm  = k_ref_fg.float().norm(dim=-1, keepdim=True).clamp(min=1e-6)
                k_ref_fg = (k_ref_fg.float() * (curr_k_norm / ref_k_norm)).to(K_all.dtype)

                curr_v_norm = V_all[:, fg_full, :].float().norm(dim=-1, keepdim=True).clamp(min=1e-6)
                ref_v_norm  = v_ref_fg.float().norm(dim=-1, keepdim=True).clamp(min=1e-6)
                v_ref_fg = (v_ref_fg.float() * (curr_v_norm / ref_v_norm)).to(V_all.dtype)

                ell = self.n_double + block_idx
                alpha = self._alpha(ell, self.n_double + self.n_single)
                K_new[:, fg_full, :] = alpha * K_all[:, fg_full, :] + (1 - alpha) * k_ref_fg
                V_new[:, fg_full, :] = alpha * V_all[:, fg_full, :] + (1 - alpha) * v_ref_fg

            return torch.cat([Q_all, K_new, V_new, rest], dim=-1)
        return hook

    def attach(self, model):
        for i, block in enumerate(model.double_blocks):
            h = block.img_attn.qkv.register_forward_hook(
                self._make_double_img_hook(i))
            self._handles.append(h)

        for i, block in enumerate(model.single_blocks):
            h = block.linear1.register_forward_hook(
                self._make_single_hook(i))
            self._handles.append(h)

        print(f"  Attached {len(self._handles)} KV injection hooks")

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        # Release cached device tensors aggressively in low-VRAM workflows.
        self._src_cache.clear()
        self._ref_cache.clear()
