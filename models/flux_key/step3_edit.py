"""
Step 3: SSI + KV-injection denoising (background freq-lock + reference foreground).

Borrows from original pipeline:
  - flux.sampling.get_schedule / unpack  (unchanged)
  - flux.util.load_ae / load_flow_model  (unchanged)
  - models.flux_key.step2_features       (step2 utilities)
  - models.flux_key.kv_injection         (our contribution)

Run standalone:
    python -m models.flux_key.step3_edit \
        --features test_output/flux_key/000000000002/features.pt \
        --source   data/pie_bench/annotation_images/0_random_140/000000000002.jpg \
        --key      000000000002 \
        --tau      0.75 \
        --prompt   "a dog sitting on a wooden chair" \
        --out_dir  test_output/flux_key/000000000002/
"""

import argparse
import os
import json
import numpy as np
import torch
from PIL import Image
from einops import rearrange

from flux.sampling import get_schedule                  # original, unchanged
from flux.util import load_ae, load_flow_model          # original, unchanged
from models.flux_key.step2_features import load_mask_from_json, mask_to_token_indices
from models.flux_key.kv_injection import KVInjectionHooks   # our contribution


# ── SSI ───────────────────────────────────────────────────────────────────────

def apply_ssi(z_A, mask_indices, tau, device):
    """Eq. ssi: noise foreground tokens only (rectified flow convention)."""
    z = z_A.clone().to(device)
    eps = torch.randn_like(z[:, mask_indices, :])
    z[:, mask_indices, :] = (1.0 - tau) * z[:, mask_indices, :] + tau * eps
    return z


# ── Denoising loop with KV injection ─────────────────────────────────────────

@torch.no_grad()
def denoise_flux_key(model, inp_A, z_init, feat_src, feat_ref,
                     mask_indices, timesteps, guidance,
                     alpha_max, alpha_min, gamma_bg=0.5, freq_2d=False, device="cuda"):
    img     = z_init.clone()
    img_ids = inp_A["img_ids"]
    txt     = inp_A["txt"]
    txt_ids = inp_A["txt_ids"]
    vec     = inp_A["vec"]

    guidance_vec = torch.full((img.shape[0],), guidance,
                               device=device, dtype=img.dtype)

    hooks = KVInjectionHooks(
        feat_src=feat_src, feat_ref=feat_ref,
        mask_indices=mask_indices,
        n_double=len(model.double_blocks),
        n_single=len(model.single_blocks),
        alpha_max=alpha_max, alpha_min=alpha_min,
        gamma=gamma_bg, freq_2d=freq_2d, device=device,
    )
    hooks.attach(model)

    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        hooks._t = float(t_curr)
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=device)
        pred = model(
            img=img, img_ids=img_ids,
            txt=txt, txt_ids=txt_ids,
            timesteps=t_vec, y=vec,
            guidance=guidance_vec,
        )
        if pred.isnan().any():
            print(f"  [WARN] NaN in prediction at t={t_curr:.3f} — clamping to continue")
            pred = pred.nan_to_num(0.0)
        img = img + (t_prev - t_curr) * pred
        del pred, t_vec                  # ← add this
        torch.cuda.empty_cache()         # ← add this
    hooks.detach()
    return img


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_edit(source_path, features_path, json_path, key,
             tau, alpha_max, alpha_min, prompt,
             num_steps, guidance, device, out_dir, fg_inject=True, freq_2d=False,
             ae=None, model=None, low_vram: bool = False):

    os.makedirs(out_dir, exist_ok=True)

    # Load features to CPU; hooks move individual tensors to GPU on demand
    saved        = torch.load(features_path, map_location="cpu", weights_only=False)
    feat_src     = saved["feat_src"]
    feat_ref     = saved["feat_ref"]
    mask_indices = saved["mask_indices"].to(device)
    inp_A        = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                   for k, v in saved["inp_A"].items()}

    if ae is None:
        ae = load_ae("flux-dev", device)
    if model is None:
        model = load_flow_model("flux-dev", device=device)
        model.eval()

    def _on(m):
        if low_vram and m is not None:
            m.to(device)

    def _off(m):
        if low_vram and m is not None:
            m.cpu()
            torch.cuda.empty_cache()

    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    src_img = Image.open(source_path).convert("RGB").resize((512, 512))
    src_arr = np.array(src_img).astype(np.float32) / 127.5 - 1.0
    src_t   = torch.from_numpy(src_arr).permute(2, 0, 1).unsqueeze(0)
    enc_dev = next(ae.parameters()).device
    src_in = src_t.to(enc_dev)
    z_A = ae.encode(src_in).to(torch.bfloat16)
    if low_vram:
        z_A = z_A.to(device)  # move latent only for FLUX denoising

    B, C, H_l, W_l = z_A.shape
    z_A_tok = rearrange(z_A, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=2, p2=2)

    z_ssi = apply_ssi(z_A_tok, mask_indices, tau, device)
    print(f"SSI applied: tau={tau}, foreground tokens={len(mask_indices)}")

    all_ts    = get_schedule(num_steps, z_A_tok.shape[1], shift=True)
    start_idx = next((i for i, ts in enumerate(all_ts) if ts <= tau), 0)
    timesteps = all_ts[start_idx:]
    print(f"Denoising from t={timesteps[0]:.3f} to {timesteps[-1]:.3f} ({len(timesteps)-1} steps)")

    # Disable foreground injection to test background-lock-only mode
    _alpha_max = alpha_max if fg_inject else 0.0
    _alpha_min = alpha_min if fg_inject else 0.0
    print(f"fg_inject={fg_inject}  alpha_max={_alpha_max}  alpha_min={_alpha_min}")

    _on(model)
    z_final = denoise_flux_key(
        model, inp_A, z_ssi, feat_src, feat_ref,
        mask_indices, timesteps, guidance,
        _alpha_max, _alpha_min, freq_2d=freq_2d, device=device,
    )
    _off(model)

    z_out   = rearrange(z_final, 'b (h w) (c p1 p2) -> b c (h p1) (w p2)',
                        h=H_l//2, w=W_l//2, p1=2, p2=2, c=C)
    ae_dev = next(ae.parameters()).device
    z_dec = z_out.to(ae_dev)
    decoded = ae.decode(z_dec.to(ae.dtype if hasattr(ae, 'dtype') else torch.float32))
    decoded = decoded.cpu()
    decoded = decoded.nan_to_num(0.0).clamp(-1, 1).cpu()
    nan_pct = (z_final.isnan().float().mean() * 100).item()
    if nan_pct > 0:
        print(f"  [WARN] {nan_pct:.1f}% NaN in final latent — output may be degraded")
    out_arr = rearrange(decoded[0], 'c h w -> h w c')
    out_img = Image.fromarray((127.5 * (out_arr + 1.0)).byte().numpy())

    out_img.save(os.path.join(out_dir, f"{key}_edited.png"))
    print(f"Saved: {out_dir}/{key}_edited.png")

    cmp = Image.new("RGB", (512 * 2, 512))
    cmp.paste(src_img, (0, 0))
    cmp.paste(out_img, (512, 0))
    cmp.save(os.path.join(out_dir, f"{key}_comparison.png"))
    print(f"Saved comparison: {out_dir}/{key}_comparison.png")
    return out_img


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features",  required=True)
    p.add_argument("--source",    required=True)
    p.add_argument("--json_path", default="data/pie_bench/mapping_file.json")
    p.add_argument("--key",       required=True)
    p.add_argument("--tau",       type=float, default=0.7)
    p.add_argument("--alpha_max", type=float, default=0.8)
    p.add_argument("--alpha_min", type=float, default=0.2)
    p.add_argument("--prompt",    default="")
    p.add_argument("--num_steps", type=int,   default=28)
    p.add_argument("--guidance",   type=float, default=3.5)
    p.add_argument("--out_dir",    default="test_output/flux_key/")
    p.add_argument("--device",     default="cuda")
    p.add_argument("--fg_inject",  action="store_true", default=True,
                   help="Enable foreground reference KV injection (default: on; pass --no-fg_inject to disable)")
    p.add_argument("--no-fg_inject", dest="fg_inject", action="store_false")
    p.add_argument("--freq_2d",    action="store_true", default=False,
                   help="Use 2-D spatial FFT for background frequency lock (default: 1-D)")
    return p.parse_args()


def main():
    args = parse_args()
    run_edit(source_path=args.source, features_path=args.features,
             json_path=args.json_path, key=args.key,
             tau=args.tau, alpha_max=args.alpha_max, alpha_min=args.alpha_min,
             prompt=args.prompt, num_steps=args.num_steps, guidance=args.guidance,
             device=args.device, out_dir=args.out_dir,
             fg_inject=args.fg_inject, freq_2d=args.freq_2d)


if __name__ == "__main__":
    main()
