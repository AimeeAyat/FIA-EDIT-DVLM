"""
dvlm/ref_inject.py — Reference K,V injection for FLUX joint transformer blocks.

Optimization: reference K,V are projected and normalized ONCE during extraction
(not at every denoising step). Per-step cost is a single torch.cat — no linear
projection, no norm.

No MyCodes/ files are modified.

Usage:
    ref_kv = extract_ref_kv(pipe, ref_image_pil, prompt_embeds,
                             pooled_prompt_embeds, text_ids,
                             num_blocks, device, dtype)
    install_ref_inject(pipe, ref_kv, num_blocks, ref_inject_steps)
    pipe.gen(...)
    remove_ref_inject(pipe)
"""

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF


# ── Attention Processor ────────────────────────────────────────────────────────

class RefInjectAttnProcessor:
    """
    Replaces MyFluxAttnProcessor2_0 on targeted joint transformer blocks.

    For steps 0 .. ref_inject_steps-1:
        Appends pre-computed reference K,V to the [text + image] K,V sequence
        after RoPE.  Per-step cost: two torch.cat calls — no projections.

    From step ref_inject_steps onward or for single-stream blocks:
        Falls back to the base processor unchanged.
    """

    def __init__(self, base_processor, ref_key, ref_value, ref_inject_steps, layer_idx):
        self.base             = base_processor
        # ref_key / ref_value: [1, heads, seq_ref, head_dim] — pre-projected & normed
        self.ref_key          = ref_key
        self.ref_value        = ref_value
        self.ref_inject_steps = ref_inject_steps
        self.layer_idx        = layer_idx

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        image_rotary_emb=None,
        cache_dic=None,
        current=None,
        fresh_indices=None,
        use_attn_map=False,
        use_cache=False,
    ):
        step = (current or {}).get("step", 0)

        # Single-stream blocks or after injection window → base processor
        if step >= self.ref_inject_steps or encoder_hidden_states is None:
            return self.base(
                attn, hidden_states, encoder_hidden_states,
                attention_mask, image_rotary_emb,
                cache_dic, current, fresh_indices, use_attn_map, use_cache,
            )

        batch_size = hidden_states.shape[0]
        inner_dim  = attn.to_k.out_features
        head_dim   = inner_dim // attn.heads

        def _proj(linear, x):
            return linear(x).view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # ── Current image Q, K, V ─────────────────────────────────────────────
        query = _proj(attn.to_q, hidden_states)
        key   = _proj(attn.to_k, hidden_states)
        value = _proj(attn.to_v, hidden_states)

        if attn.norm_q is not None: query = attn.norm_q(query)
        if attn.norm_k is not None: key   = attn.norm_k(key)

        # ── Text encoder Q, K, V ──────────────────────────────────────────────
        enc_q = _proj(attn.add_q_proj, encoder_hidden_states)
        enc_k = _proj(attn.add_k_proj, encoder_hidden_states)
        enc_v = _proj(attn.add_v_proj, encoder_hidden_states)

        if attn.norm_added_q is not None: enc_q = attn.norm_added_q(enc_q)
        if attn.norm_added_k is not None: enc_k = attn.norm_added_k(enc_k)

        # ── Concat [text | img] ───────────────────────────────────────────────
        query = torch.cat([enc_q, query], dim=2)
        key   = torch.cat([enc_k, key],   dim=2)
        value = torch.cat([enc_v, value], dim=2)

        # ── RoPE on [text + img] tokens ───────────────────────────────────────
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            query = apply_rotary_emb(query, image_rotary_emb)
            key   = apply_rotary_emb(key,   image_rotary_emb)

        # ── Append pre-computed ref K,V — O(1) cost: just cat ─────────────────
        dev, dt = hidden_states.device, hidden_states.dtype
        ref_k = self.ref_key.to(device=dev, dtype=dt)
        ref_v = self.ref_value.to(device=dev, dtype=dt)
        if ref_k.shape[0] != batch_size:
            ref_k = ref_k.expand(batch_size, -1, -1, -1)
            ref_v = ref_v.expand(batch_size, -1, -1, -1)

        key   = torch.cat([key,   ref_k], dim=2)
        value = torch.cat([value, ref_v], dim=2)

        # ── Attention ─────────────────────────────────────────────────────────
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False
        )
        hidden_states = (
            hidden_states.transpose(1, 2)
            .reshape(batch_size, -1, attn.heads * head_dim)
            .to(query.dtype)
        )

        # ── Split [text | img] ────────────────────────────────────────────────
        enc_len = encoder_hidden_states.shape[1]
        enc_out = hidden_states[:, :enc_len]
        img_out = hidden_states[:, enc_len:]

        img_out = attn.to_out[0](img_out)
        img_out = attn.to_out[1](img_out)
        enc_out = attn.to_add_out(enc_out)

        return img_out, enc_out, None


# ── Extraction ─────────────────────────────────────────────────────────────────

def extract_ref_kv(
    pipe,
    ref_image_pil,
    prompt_embeds,
    pooled_prompt_embeds,
    text_ids,
    num_blocks: int,
    device,
    dtype,
):
    """
    Run one forward pass with the reference image to capture hidden_states
    at the to_k input of the first num_blocks joint transformer blocks, then
    immediately project + normalize them into K,V tensors.

    Pre-projecting here (once) instead of inside the processor (every step)
    removes 2 × num_blocks × ref_inject_steps linear ops from the hot path.

    Returns:
        dict[int, tuple[Tensor, Tensor]]:
            {block_idx: (ref_key, ref_value)}
            Each tensor: [1, heads, seq_ref, head_dim], already normed.
    """
    n_blocks = min(num_blocks, len(pipe.transformer.transformer_blocks))
    if n_blocks == 0:
        return {}

    # ── VAE encode ref image ──────────────────────────────────────────────────
    ref_pil    = ref_image_pil.convert("RGB").resize((512, 512), Image.LANCZOS)
    ref_tensor = TF.to_tensor(ref_pil).unsqueeze(0).to(device=device, dtype=dtype)
    ref_tensor = ref_tensor * 2.0 - 1.0   # [0,1] → [-1,1]

    with torch.no_grad():
        ref_lat = pipe.vae.encode(ref_tensor).latent_dist.sample()
    ref_lat = (ref_lat - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor

    # ── Pack into FLUX token format ───────────────────────────────────────────
    B, C, H, W  = ref_lat.shape
    ref_packed  = pipe._pack_latents(ref_lat, B, C, H, W)
    ref_img_ids = _make_image_ids(H // 2, W // 2, device, dtype)

    # ── Capture raw hidden_states at to_k input for first n_blocks ───────────
    raw_hs_store: dict = {}
    hooks = []
    for i in range(n_blocks):
        h = pipe.transformer.transformer_blocks[i].attn.to_k.register_forward_hook(
            _capture_hook(raw_hs_store, i)
        )
        hooks.append(h)

    minimal_jak = {
        "use_attn_map": False,
        "use_cache":    False,
        "cache_dic":    {"cache_type": "none", "cache": []},
        "current":      {"step": 0, "layer": 0},
    }
    with torch.no_grad():
        pipe.transformer(
            hidden_states=ref_packed,
            timestep=torch.tensor([0.5], device=device, dtype=dtype),
            guidance=torch.tensor([3.5], device=device, dtype=dtype),
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=ref_img_ids,
            joint_attention_kwargs=minimal_jak,
            return_dict=False,
        )

    for h in hooks:
        h.remove()

    # ── Pre-project captured hidden_states into K,V (one-time cost) ──────────
    ref_kv: dict = {}
    for i, raw_hs in raw_hs_store.items():
        block     = pipe.transformer.transformer_blocks[i]
        attn      = block.attn
        inner_dim = attn.to_k.out_features
        head_dim  = inner_dim // attn.heads
        bs        = raw_hs.shape[0]

        with torch.no_grad():
            ref_k = attn.to_k(raw_hs).view(bs, -1, attn.heads, head_dim).transpose(1, 2)
            ref_v = attn.to_v(raw_hs).view(bs, -1, attn.heads, head_dim).transpose(1, 2)
            if attn.norm_k is not None:
                ref_k = attn.norm_k(ref_k)

        ref_kv[i] = (ref_k, ref_v)

    return ref_kv


def _capture_hook(storage: dict, idx: int):
    def _fn(module, inp, out):
        storage[idx] = inp[0].detach().clone()
    return _fn


def _make_image_ids(h: int, w: int, device, dtype):
    ids = torch.zeros(h, w, 3)
    ids[..., 1] = torch.arange(h)[:, None]
    ids[..., 2] = torch.arange(w)[None, :]
    return ids.reshape(h * w, 3).to(device=device, dtype=dtype)


# ── Install / Remove ───────────────────────────────────────────────────────────

def install_ref_inject(
    pipe,
    ref_kv_per_block: dict,
    num_blocks: int,
    ref_inject_steps: int,
):
    """
    Swap the attention processor on the first num_blocks joint transformer blocks
    with RefInjectAttnProcessor instances carrying pre-computed K,V.
    Idempotent: no-op if ref_kv_per_block is empty or ref_inject_steps == 0.
    """
    if not ref_kv_per_block or ref_inject_steps <= 0:
        return

    saved = {}
    n = min(num_blocks, len(pipe.transformer.transformer_blocks))
    for i in range(n):
        if i not in ref_kv_per_block:
            continue
        block    = pipe.transformer.transformer_blocks[i]
        ref_k, ref_v = ref_kv_per_block[i]
        saved[i] = block.attn.processor
        block.attn.processor = RefInjectAttnProcessor(
            base_processor=block.attn.processor,
            ref_key=ref_k,
            ref_value=ref_v,
            ref_inject_steps=ref_inject_steps,
            layer_idx=i,
        )

    pipe.transformer._ref_inject_saved_processors = saved


def remove_ref_inject(pipe):
    """Restore original processors. Safe to call even if install was never called."""
    saved = getattr(pipe.transformer, "_ref_inject_saved_processors", None)
    if saved is None:
        return
    for i, proc in saved.items():
        pipe.transformer.transformer_blocks[i].attn.processor = proc
    del pipe.transformer._ref_inject_saved_processors
