"""
dvlm/pipeline_patches.py — Tail-CFG patch for FluxCompositionPipeline.

Optimization (FIA-Edit style): neg + pos are batched into a SINGLE transformer
forward pass instead of two sequential ones, halving the tail-CFG overhead.

Cache is disabled for the batched call to avoid batch-size mismatches with the
existing cache_dic (which was built for batch=1). The cache savings over 4-6
tail steps are negligible.

No MyCodes/ files are modified.
"""

import torch


def install_cfg_tail_patch(
    pipe,
    num_inference_steps: int,
    cfg_tail_steps: int,
    cfg_scale: float,
    neg_prompt_embeds: torch.Tensor,
    neg_pooled_prompt_embeds: torch.Tensor,
):
    """
    Monkey-patch pipe.transformer.forward so that for the last cfg_tail_steps
    denoising steps, neg and pos embeddings are processed in one batched forward
    pass and blended: out = neg + cfg_scale * (pos - neg).

    Before start_cfg_step: unchanged single forward pass.
    Idempotent: calling twice is a no-op.
    """
    if cfg_tail_steps <= 0:
        return
    if hasattr(pipe.transformer, "_cfg_tail_original_forward"):
        return

    start_cfg_step    = num_inference_steps - cfg_tail_steps
    _original_forward = pipe.transformer.forward

    def _patched_forward(
        hidden_states,
        timestep=None,
        guidance=None,
        pooled_projections=None,
        encoder_hidden_states=None,
        txt_ids=None,
        img_ids=None,
        joint_attention_kwargs=None,
        return_dict=False,
        **kwargs,
    ):
        step = (joint_attention_kwargs or {}).get("current", {}).get("step", 0)

        # Early steps: unchanged
        if step < start_cfg_step:
            return _original_forward(
                hidden_states,
                timestep=timestep,
                guidance=guidance,
                pooled_projections=pooled_projections,
                encoder_hidden_states=encoder_hidden_states,
                txt_ids=txt_ids,
                img_ids=img_ids,
                joint_attention_kwargs=joint_attention_kwargs,
                return_dict=return_dict,
                **kwargs,
            )

        dt = hidden_states.dtype

        # ── Batch neg + pos into one forward pass (FIA-Edit style) ────────────
        # Cache would break with batch=2 → disable for this call only
        cfg_jak = {**(joint_attention_kwargs or {}), "use_cache": False}

        batch_hs     = torch.cat([hidden_states, hidden_states], dim=0)
        batch_pooled = torch.cat([neg_pooled_prompt_embeds.to(dt),
                                  pooled_projections], dim=0)
        batch_enc    = torch.cat([neg_prompt_embeds.to(dt),
                                  encoder_hidden_states], dim=0)

        # timestep / guidance are 1-D; expand to batch=2
        batch_t = timestep.expand(2)  if timestep is not None else timestep
        batch_g = guidance.expand(2)  if guidance  is not None else guidance
        # txt_ids / img_ids have no batch dim in FLUX — pass as-is

        batch_out = _original_forward(
            batch_hs,
            timestep=batch_t,
            guidance=batch_g,
            pooled_projections=batch_pooled,
            encoder_hidden_states=batch_enc,
            txt_ids=txt_ids,
            img_ids=img_ids,
            joint_attention_kwargs=cfg_jak,
            return_dict=False,
            **kwargs,
        )[0]                                            # [2, seq, dim]

        neg_out, pos_out = batch_out.chunk(2, dim=0)   # each [1, seq, dim]
        blended = neg_out + cfg_scale * (pos_out - neg_out)
        return (blended,) if not return_dict else blended

    pipe.transformer.forward = _patched_forward
    pipe.transformer._cfg_tail_original_forward = _original_forward


def remove_cfg_tail_patch(pipe):
    """Restore the original forward. Safe to call even if never patched."""
    original = getattr(pipe.transformer, "_cfg_tail_original_forward", None)
    if original is not None:
        pipe.transformer.forward = original
        del pipe.transformer._cfg_tail_original_forward
