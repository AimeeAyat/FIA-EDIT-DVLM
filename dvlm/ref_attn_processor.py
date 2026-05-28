"""
Reference-aware attention processor for FLUX double-stream blocks.

Injects reference image K,V tokens so the main image can attend to reference
appearance features during composition. This is the core improvement over the
original EEdit pipeline, which only uses the reference as a pixel paste.

Key design decisions:
  - K,V only injection (not Q) → main→ref attention without O(n²) Q growth
  - No RoPE for reference tokens → position-free global content context
  - Only in first N double-stream blocks → high-level composition, not detail refinement
  - Only for the first M denoising steps → matches the RF-Inversion correction window
  - x_embedder + LayerNorm normalisation of ref tokens before projection
"""

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention


class RefAwareFluxAttnProcessor2_0:
    """
    Drop-in replacement for MyFluxAttnProcessor2_0.
    Falls back to standard behaviour when ref_tokens_embedded is absent from
    joint_attention_kwargs, so it can be permanently installed on the model.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Requires PyTorch 2.0+")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask=None,
        image_rotary_emb=None,
        # EEdit caching kwargs
        cache_dic=None,
        current=None,
        fresh_indices=None,
        use_attn_map: bool = False,
        use_cache: bool = False,
        # Reference injection kwargs (new)
        ref_tokens_embedded=None,   # [B, ref_seq, model_dim] — pre-projected ref latents
        ref_inject_blocks: int = 8,
        ref_inject_steps: int = 14,
        **kwargs,
    ) -> torch.FloatTensor:

        # Determine whether to inject reference tokens this call
        layer = current.get("layer", 9999) if current is not None else 9999
        step  = current.get("step",  9999) if current is not None else 9999
        is_double_stream = encoder_hidden_states is not None
        should_inject = (
            ref_tokens_embedded is not None
            and is_double_stream
            and layer < ref_inject_blocks
            and step < ref_inject_steps
        )

        batch_size = hidden_states.shape[0]

        # ── image token projections ──────────────────────────────────────────
        query = attn.to_q(hidden_states)
        key   = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim  = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key   = key  .view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if use_cache and cache_dic is not None and cache_dic.get("cache_type") == "kv-norm":
            cache_dic["cache"][-1][layer]["attn_v_norm"] = torch.norm(value, dim=-1, p=2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # ── text token projections (double-stream only) ──────────────────────
        if is_double_stream:
            enc_q = attn.add_q_proj(encoder_hidden_states)
            enc_k = attn.add_k_proj(encoder_hidden_states)
            enc_v = attn.add_v_proj(encoder_hidden_states)

            enc_q = enc_q.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            enc_k = enc_k.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            enc_v = enc_v.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if use_cache and cache_dic is not None and cache_dic.get("cache_type") == "kv-norm":
                cache_dic["cache"][-1][layer]["cross_attn_v_norm"] = torch.norm(enc_v, dim=-1, p=2)

            if attn.norm_added_q is not None:
                enc_q = attn.norm_added_q(enc_q)
            if attn.norm_added_k is not None:
                enc_k = attn.norm_added_k(enc_k)

            query = torch.cat([enc_q, query], dim=2)
            key   = torch.cat([enc_k, key],   dim=2)
            value = torch.cat([enc_v, value],  dim=2)

        # ── reference token K,V injection ───────────────────────────────────
        if should_inject:
            # ref_tokens_embedded is [B, ref_seq, model_dim], after x_embedder.
            # Apply a simple LayerNorm (no time-adaptive scale) so the projection
            # operates on a normalised distribution.
            ref_norm = F.layer_norm(
                ref_tokens_embedded,
                [ref_tokens_embedded.shape[-1]],
            ).to(hidden_states.dtype)

            ref_k = attn.to_k(ref_norm)
            ref_v = attn.to_v(ref_norm)

            ref_k = ref_k.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ref_v = ref_v.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_k is not None:
                ref_k = attn.norm_k(ref_k)

            # Append reference K,V AFTER the existing text+image tokens.
            # Reference tokens receive NO rotary embedding (global content context).
            key   = torch.cat([key,   ref_k], dim=2)
            value = torch.cat([value, ref_v], dim=2)

        # ── rotary embeddings (applied to text + image tokens only) ─────────
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            rope_len = image_rotary_emb[0].shape[0]   # seq len covered by RoPE

            q_rope = query[:, :, :rope_len]
            k_rope = key  [:, :, :rope_len]

            q_rope = apply_rotary_emb(q_rope, image_rotary_emb)
            k_rope = apply_rotary_emb(k_rope, image_rotary_emb)

            # Preserve any reference tokens beyond the rope coverage
            query = torch.cat([q_rope, query[:, :, rope_len:]], dim=2)
            key   = torch.cat([k_rope, key  [:, :, rope_len:]], dim=2)

        # ── attention ────────────────────────────────────────────────────────
        if use_attn_map:
            q = query * (head_dim ** -0.5)
            attn_weights = torch.matmul(q, key.transpose(-2, -1))
            attn_map     = attn_weights.softmax(dim=-1)
            hidden_states = torch.matmul(attn_map, value)
        else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, dropout_p=0.0, is_causal=False
            )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # ── output projections ───────────────────────────────────────────────
        if is_double_stream:
            enc_len = encoder_hidden_states.shape[1]
            encoder_hidden_states_out = hidden_states[:, :enc_len]
            hidden_states             = hidden_states[:, enc_len:]

            encoder_hidden_states_out = attn.to_add_out(encoder_hidden_states_out)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if is_double_stream:
            return hidden_states, encoder_hidden_states_out
        return hidden_states


def install_ref_attn_processors(transformer, num_double_blocks: int = 8):
    """
    Replace the attention processor on the first `num_double_blocks`
    double-stream transformer blocks.  Falls back gracefully when no
    reference tokens are provided (so the model can still be used
    without reference injection).
    """
    proc = RefAwareFluxAttnProcessor2_0()
    for i, block in enumerate(transformer.transformer_blocks):
        if i < num_double_blocks:
            block.attn.set_processor(proc)
    # Single-stream blocks keep their original processor unchanged
