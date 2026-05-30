"""
dvlm/pipeline_patches.py — Tail-CFG patch for FluxCompositionPipeline.

The transformer is called during BOTH inversion (controlled_forward_ode) and
the denoising loop.  We must NOT apply CFG during inversion.

Detection strategy:
  - During inversion, current['step'] is incremented internally by the
    PREDEFINE transformer (it uses this to decide which layers to recompute).
    By the end of inversion it is well above 0.
  - The denoising loop explicitly resets current['step'] = 0 at its first
    iteration (i=0), then 1, 2, ... for subsequent steps.
  - We detect the inversion→denoising transition by observing current['step']
    drop from a value > 0 back to 0.  From that point on we use normal
    step-based gating.

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
    Monkey-patch pipe.transformer.forward so that CFG is applied only on the
    last cfg_tail_steps denoising steps (not during inversion).

    Neg+pos are batched into one forward pass (FIA-Edit style) to halve cost.
    Cache is disabled for the batched call to avoid batch-size mismatches.
    Idempotent.
    """
    if cfg_tail_steps <= 0:
        return
    if hasattr(pipe.transformer, "_cfg_tail_original_forward"):
        return

    start_cfg_step    = num_inference_steps - cfg_tail_steps
    _original_forward = pipe.transformer.forward

    # ── Phase tracking ────────────────────────────────────────────────────────
    # _prev_step: the last value of current['step'] we saw.
    # _in_denoising: True once we detect the loop reset (high → 0).
    _prev_step     = [-1]
    _in_denoising  = [False]

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

        # Detect inversion → denoising transition:
        # The denoising loop resets current['step'] = 0 before each step.
        # After inversion it was > 0, so the drop to 0 marks denoising start.
        if not _in_denoising[0] and step == 0 and _prev_step[0] > 0:
            _in_denoising[0] = True
        _prev_step[0] = step

        # Single pass: inversion OR early denoising steps
        if not _in_denoising[0] or step < start_cfg_step:
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

        # ── Tail window: batch neg + pos (FIA-Edit style) ─────────────────────
        cfg_jak = {**(joint_attention_kwargs or {}), "use_cache": False}
        dt = hidden_states.dtype

        batch_hs     = torch.cat([hidden_states, hidden_states], dim=0)
        batch_pooled = torch.cat([neg_pooled_prompt_embeds.to(dt),
                                  pooled_projections], dim=0)
        batch_enc    = torch.cat([neg_prompt_embeds.to(dt),
                                  encoder_hidden_states], dim=0)
        batch_t      = timestep.expand(2) if timestep is not None else timestep
        batch_g      = guidance.expand(2)  if guidance  is not None else guidance

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
        )[0]

        neg_out, pos_out = batch_out.chunk(2, dim=0)
        blended = neg_out + cfg_scale * (pos_out - neg_out)
        return (blended,) if not return_dict else blended

    pipe.transformer.forward = _patched_forward
    pipe.transformer._cfg_tail_original_forward = _original_forward


def remove_cfg_tail_patch(pipe):
    """Restore original forward. Safe to call even if never patched."""
    original = getattr(pipe.transformer, "_cfg_tail_original_forward", None)
    if original is not None:
        pipe.transformer.forward = original
        del pipe.transformer._cfg_tail_original_forward
