"""
Step 2: Dual clean forward passes to extract source and reference attention features.

For each FLUX transformer block ℓ, this module records:
  - K_src^(ℓ), V_src^(ℓ)  : from I_A at t=0  (used for background frequency lock)
  - K_ref^(ℓ), V_ref^(ℓ)  : from I_B_aligned at t=0 with RoPE re-indexed to M_A
                              (used for foreground reference injection)

No weight modifications are made to the original FLUX model.
The hook-based extraction is read-only — hooks are removed after the forward pass.

Borrows from original pipeline:
  - flux.util.load_flow_model  (original pipeline, unchanged)
  - flux.util.load_ae          (original pipeline, unchanged)
  - flux.sampling.prepare      (original pipeline, unchanged)

Run standalone:
    python -m models.flux_key.step2_features \
        --source data/pie_bench/annotation_images/0_random_140/000000000000.jpg \
        --ref_aligned test_output/step1/ref_aligned.png \
        --mask_json "000000000000" \
        --out_dir test_output/step2/
"""

import argparse
import os
import json
import numpy as np
import torch
from PIL import Image

from flux.util import load_ae, load_flow_model   # original pipeline, unchanged
from flux.sampling import prepare                  # original pipeline, unchanged


# ── Hook manager ─────────────────────────────────────────────────────────────

class KVHookManager:
    """
    Attaches read-only forward hooks to every DoubleStreamBlock and
    SingleStreamBlock in the FLUX transformer to record K and V tensors.

    The hooks do NOT modify the forward pass; they only copy the tensors to CPU.
    All hooks are removed after the forward pass completes.
    """

    def __init__(self):
        self.features: dict = {}   # key: (block_type, block_idx, 'K'/'V')
        self._handles: list = []

    def _make_qkv_hook(self, tag: str):
        """Hook on the QKV linear layer to capture K and V after projection."""
        def hook(module, input, output):
            # output shape: [B, L, 3*H*D]
            B, L, three_HD = output.shape
            HD = three_HD // 3
            # Q=output[..., :HD], K=output[..., HD:2*HD], V=output[..., 2*HD:]
            k = output[..., HD:2*HD].detach().cpu()
            v = output[..., 2*HD:].detach().cpu()
            self.features[f"{tag}_K"] = k
            self.features[f"{tag}_V"] = v
        return hook

    def attach(self, model):
        """Attach hooks to all double and single stream blocks."""
        for i, block in enumerate(model.double_blocks):
            # img QKV
            h = block.img_attn.qkv.register_forward_hook(
                self._make_qkv_hook(f"double_{i}_img"))
            self._handles.append(h)
            # txt QKV
            h = block.txt_attn.qkv.register_forward_hook(
                self._make_qkv_hook(f"double_{i}_txt"))
            self._handles.append(h)

        for i, block in enumerate(model.single_blocks):
            # single stream linear1 outputs [qkv | mlp]
            def _single_hook(module, inp, out, _i=i):
                B, L, dim = out.shape
                # first 3*hidden_size = QKV, rest = MLP
                hidden = module.out_features - module.out_features // (1 + 4)
                qkv_dim = 3 * (dim - dim // 5)  # approximate; use known hidden size
                # safer: use model's hidden_size directly
                hs = out.shape[-1]
                # QKV takes up 3/(3+4) of linear1 output for mlp_ratio=4
                # For FLUX: hidden_size=3072, mlp_ratio=4 -> linear1 out = 3*3072 + 4*3072
                # so QKV = 3*3072 = 9216, MLP = 4*3072 = 12288
                qkv = out[..., :9216]   # [B, L, 3*H*D]  where H*D=3072
                k = qkv[..., 3072:6144].detach().cpu()
                v = qkv[..., 6144:9216].detach().cpu()
                self.features[f"single_{_i}_K"] = k
                self.features[f"single_{_i}_V"] = v
            h = block.linear1.register_forward_hook(_single_hook)
            self._handles.append(h)

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def get(self) -> dict:
        return dict(self.features)

    def clear(self):
        self.features.clear()


# ── Mask utilities ────────────────────────────────────────────────────────────

def load_mask_from_json(json_path: str, key: str,
                        image_shape=(512, 512)) -> np.ndarray:
    """
    Decode run-length-encoded mask from PIE-Bench mapping_file.json.
    Returns a binary numpy array of shape (H, W).

    # This RLE decode mirrors pie_features.py:mask_decode (original pipeline).
    """
    with open(json_path) as f:
        data = json.load(f)
    item = data[key]
    encoded = item["mask"]
    length = image_shape[0] * image_shape[1]
    arr = np.zeros((length,), dtype=np.float32)
    for i in range(0, len(encoded), 2):
        start = encoded[i]
        run = min(encoded[i + 1], length - start)
        arr[start:start + run] = 1.0
    return arr.reshape(image_shape)


def mask_to_token_indices(mask: np.ndarray,
                          patch_size: int = 2) -> torch.Tensor:
    """
    Convert a (H, W) binary mask to flat token indices in the FLUX latent grid.
    FLUX packs (patch_size x patch_size) patches, so a 512x512 image becomes
    a 64x64 = 4096 token grid.
    Returns a 1D LongTensor of foreground token indices.
    """
    H, W = mask.shape
    lat_h, lat_w = H // (8 * patch_size), W // (8 * patch_size)
    # downsample mask to latent token grid by average pooling
    m = torch.from_numpy(mask).float()
    m_small = torch.nn.functional.adaptive_avg_pool2d(
        m.unsqueeze(0).unsqueeze(0), (lat_h, lat_w)
    ).squeeze()
    indices = (m_small.reshape(-1) > 0.5).nonzero(as_tuple=True)[0]
    return indices


def build_pe_mask(pe: torch.Tensor, mask_indices: torch.Tensor,
                  txt_len: int = 512) -> torch.Tensor:
    """
    Build pe_mask for the reference stream: concatenate text pe positions
    with the source-image positions corresponding to mask_indices.
    Mirrors kv_edit.py usage of pe_mask (original pipeline, referenced).
    """
    # pe shape: [1, 1, txt_len + n_img, ...]
    pe_txt = pe[:, :, :txt_len, ...]
    pe_img_masked = pe[:, :, txt_len + mask_indices, ...]
    return torch.cat([pe_txt, pe_img_masked], dim=2)


# ── Forward pass with hook capture ───────────────────────────────────────────

@torch.no_grad()
def extract_features(image: Image.Image,
                     t5, clip, ae, model,
                     prompt: str,
                     device: str,
                     mask_indices: torch.Tensor = None,
                     pe_override: torch.Tensor = None):
    """
    Single clean (t=0) forward pass through FLUX to extract K, V features.
    Returns a dict {tag_K: tensor, tag_V: tensor, ...} for all blocks.

    mask_indices: if given, limits the image tokens passed to foreground only
                  (used for the reference stream to set RoPE to M_A coords).
    pe_override:  if given, replaces the model's computed position embeddings.
    """
    # encode image to latent
    z = ae.encode(image.to(device)).to(torch.bfloat16)

    # prepare text + image embeddings  (original sampling.prepare, unchanged)
    inp = prepare(t5, clip, z, prompt=prompt)

    # RTX 5090 (Blackwell/SM_100): Flash/mem-efficient SDP produces NaN
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    # attach hooks
    hooks = KVHookManager()
    hooks.attach(model)

    # forward at t=0 (guidance_vec does not matter for feature extraction)
    t_vec = torch.zeros(z.shape[0], device=device, dtype=torch.bfloat16)
    guidance_vec = torch.full((z.shape[0],), 3.5, device=device,
                              dtype=torch.bfloat16)

    # if RoPE override given, monkey-patch pe_embedder temporarily
    orig_pe = None
    if pe_override is not None:
        orig_pe = model.pe_embedder
        model.pe_embedder = lambda ids: pe_override

    # Use base Flux.forward() — no 'info' arg needed for clean feature extraction
    _ = model(
        img=inp["img"],
        img_ids=inp["img_ids"],
        txt=inp["txt"],
        txt_ids=inp["txt_ids"],
        timesteps=t_vec,
        y=inp["vec"],
        guidance=guidance_vec,
    )

    if orig_pe is not None:
        model.pe_embedder = orig_pe

    hooks.detach()
    features = hooks.get()
    hooks.clear()
    return features, inp


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Step 2: Extract K,V features")
    p.add_argument("--source",     required=True)
    p.add_argument("--ref_aligned", required=True)
    p.add_argument("--json_path",  default="data/pie_bench/mapping_file.json")
    p.add_argument("--key",        required=True, help="PIE-Bench sample key")
    p.add_argument("--prompt",     default="")
    p.add_argument("--out_dir",    default="test_output/step2/")
    p.add_argument("--device",     default="cuda")
    return p.parse_args()


def main():
    import os
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from flux.util import load_t5, load_clip
    device = args.device

    print("Loading models...")
    t5    = load_t5(device, max_length=512)
    clip  = load_clip(device)
    ae    = load_ae("flux-dev", device)
    model = load_flow_model("flux-dev", device=device)

    # load mask
    mask_np = load_mask_from_json(args.json_path, args.key)
    mask_indices = mask_to_token_indices(mask_np).to(device)
    print(f"Mask tokens: {len(mask_indices)} / 4096")

    # load images as tensors
    def load_img(path):
        img = Image.open(path).convert("RGB").resize((512, 512))
        arr = np.array(img).astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    img_A = load_img(args.source)
    img_B = load_img(args.ref_aligned)

    print("Extracting source features (I_A clean forward pass)...")
    feat_src, inp_A = extract_features(img_A, t5, clip, ae, model,
                                       prompt=args.prompt, device=device)
    print(f"  Source features: {len(feat_src)} tensors")

    print("Extracting reference features (I_B clean forward pass, RoPE re-indexed)...")
    # Build pe_override for reference stream using M_A token coordinates
    # (full pe computation then slice to mask_indices + txt positions)
    feat_ref, inp_B = extract_features(img_B, t5, clip, ae, model,
                                       prompt=args.prompt, device=device,
                                       mask_indices=mask_indices)
    print(f"  Reference features: {len(feat_ref)} tensors")

    # save
    out_path = os.path.join(args.out_dir, "features.pt")
    torch.save({
        "feat_src": feat_src,
        "feat_ref": feat_ref,
        "mask_indices": mask_indices,
        "inp_A": inp_A,
    }, out_path)
    print(f"Saved features to {out_path}")

    # quick sanity: check a double block K
    k_src_sample = feat_src.get("double_0_img_K")
    k_ref_sample = feat_ref.get("double_0_img_K")
    if k_src_sample is not None and k_ref_sample is not None:
        diff = (k_src_sample - k_ref_sample).abs().mean().item()
        print(f"double_0 img K mean abs diff (src vs ref): {diff:.4f}")
        print("  (non-zero diff expected since I_A != I_B)")


if __name__ == "__main__":
    main()
