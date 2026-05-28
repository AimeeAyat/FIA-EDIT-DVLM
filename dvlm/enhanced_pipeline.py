"""
EnhancedFluxCompositionPipeline — drop-in replacement for FluxCompositionPipeline.

Changes vs the original EEdit composition pipeline:

1. REFERENCE TOKEN INJECTION (biggest quality win)
   The original pipeline pastes the reference as pixels and relies on RF-Inversion
   to preserve it.  For cross-domain (Real→Cartoon/Painting/Sketch) the diffusion
   prior overwhelms the pasted reference → identity loss.

   This pipeline pre-projects the reference latents through x_embedder and injects
   them as additional K,V tokens in the first `ref_inject_blocks` double-stream
   attention blocks for the first `ref_inject_steps` denoising steps.

   Main image tokens can now *directly attend* to reference appearance features
   at each attention layer, preventing the domain prior from fully overriding them.

2. DOMAIN-ADAPTIVE ETA
   For cross-domain tasks the RF-Inversion correction (eta) HURTS inside the
   bounding box: it pulls the generation toward the photorealistic pasted composite,
   conflicting with the scene's style.  We use the per-domain config to lower eta
   for cross-domain groups (RC: 0.5, RP: 0.6, RS: 0.4, RR: 1.0).

3. BETTER COMPOSITE INITIALISATION
   Optional seamless/Poisson blending at the reference boundary instead of a hard
   pixel paste. Reduces the domain-boundary shock that RF-Inversion amplifies.

4. BOUNDING-BOX EXPANSION FOR PARTIAL OBJECTS
   When a reference object has hidden parts (legs, feet) that fall outside its
   mask, the bbox is expanded downward so the model has room to complete them.
   Positive prompts include "complete figure, full body visible" guidance and
   negative prompts include "truncated, cut off, partial figure" to steer the
   model toward generating the full object.

5. NEGATIVE PROMPTS (optional tail-CFG)
   The original pipeline accepts neg_prompt but never uses it.  When
   `use_tail_cfg=True` (config flag), the last `cfg_tail_steps` denoising steps
   run standard CFG with the negative prompt.  This doubles compute only for those
   steps (typically 4–6 steps out of 28).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput

from MyCodes.MyFluxCompositionPipeline import (
    FluxCompositionPipeline,
    calculate_shift,
    retrieve_timesteps,
)
from dvlm.ref_attn_processor import install_ref_attn_processors
from dvlm.composite_prep import (
    seamless_composite,
    _soft_blend_composite,
    expand_bbox_for_partial,
    domain_preprocess_ref,
    color_harmonize_ref,
)
from dvlm.prompt_utils import augment_prompt, get_negative_prompt, detect_domain


class EnhancedFluxCompositionPipeline(FluxCompositionPipeline):
    """
    Inherits all model-loading and utility methods from FluxCompositionPipeline.
    Only `gen()` is overridden.
    """

    def _install_ref_processors(self, num_blocks: int):
        install_ref_attn_processors(self.transformer, num_double_blocks=num_blocks)

    @torch.no_grad()
    def gen(
        self,
        prompt: Union[str, List[str]] = None,
        neg_prompt: Optional[Union[str, List[str]]] = None,
        main_image=None,
        ref_image=None,
        ref_segment=None,
        x1: Optional[int] = None,
        y1: Optional[int] = None,
        x2: Optional[int] = None,
        y2: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        use_rf_inversion: bool = True,
        eta: float = 1.0,
        gamma: float = 1.0,
        start_timestep: int = 0,
        stop_timestep: int = 4,
        blend_ratio: float = 0,
        skip_T: int = 1,
        # Reference injection
        ref_inject_blocks: int = 8,
        ref_inject_steps: int = 14,
        # Better composite
        use_seamless_blend: bool = True,
        expand_bottom_frac: float = 0.18,
        expand_top_frac: float = 0.0,
        expand_sides_frac: float = 0.0,
        domain: str = "RR",
        preprocess_ref: bool = True,
        color_harmonize: bool = True,
        # Optional tail-CFG with negative prompt
        use_tail_cfg: bool = False,
        cfg_scale: float = 3.5,
        cfg_tail_steps: int = 5,
        # Standard pipeline args
        padding_mask_crop: Optional[int] = None,
        strength: float = 1.0,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        num_images_per_prompt: Optional[int] = 1,
        generator=None,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end=None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        # ── ref token injection: install processors once ─────────────────────
        self._install_ref_processors(ref_inject_blocks)

        # ── optionally preprocess reference for domain adaptation ─────────────
        if preprocess_ref and domain != "RR":
            from PIL import Image as _PIL_Image
            if not isinstance(ref_image, _PIL_Image.Image):
                ref_image = ref_image if hasattr(ref_image, "convert") else _PIL_Image.fromarray(ref_image)
            ref_image = domain_preprocess_ref(ref_image, domain)

        masked_image_latents = None
        main_latents = None
        ref_latents  = None

        if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
            ref_height = abs(y2 - y1)
            ref_width  = abs(x2 - x1)
        else:
            ref_height = ref_width = None

        orig_ref_height = ref_image.height if hasattr(ref_image, "height") else ref_image.shape[-2]
        orig_ref_width  = ref_image.width  if hasattr(ref_image, "width")  else ref_image.shape[-1]

        height = height or self.default_sample_size * self.vae_scale_factor
        width  = width  or self.default_sample_size * self.vae_scale_factor

        # ── default negative prompt ───────────────────────────────────────────
        if neg_prompt is None:
            neg_prompt = get_negative_prompt(domain)

        # ── input checks (parent handles edge cases) ──────────────────────────
        self.check_inputs(
            prompt=prompt,
            main_image=main_image,
            ref_segment=ref_segment,
            ref_image=ref_image,
            x1=x1, y1=y1, x2=x2, y2=y2,
            strength=strength,
            height=height,
            width=width,
            output_type=output_type,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            padding_mask_crop=padding_mask_crop,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # ── image preprocessing ───────────────────────────────────────────────
        crops_coords = None
        resize_mode  = "default"
        if padding_mask_crop is not None:
            crops_coords = self.mask_processor.get_crop_region(
                ref_segment, width, height, pad=padding_mask_crop
            )
            resize_mode = "fill"

        init_image = self.image_processor.preprocess(
            main_image, height=height, width=width,
            crops_coords=crops_coords, resize_mode=resize_mode,
        ).to(torch.float32)

        ref_image_t = self.image_processor.preprocess(
            ref_image, height=orig_ref_height, width=orig_ref_width,
            crops_coords=crops_coords, resize_mode=resize_mode,
        ).to(torch.float32)

        # ── call parameters ───────────────────────────────────────────────────
        batch_size = 1 if isinstance(prompt, str) else (
            len(prompt) if isinstance(prompt, list) else prompt_embeds.shape[0]
        )
        device     = self._execution_device
        lora_scale = (
            self.joint_attention_kwargs.get("scale", None)
            if self.joint_attention_kwargs else None
        )

        (prompt_embeds, pooled_prompt_embeds, text_ids) = self.encode_prompt(
            prompt=prompt, prompt_2="",
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
        (null_prompt_embeds, null_pooled_prompt_embeds, null_text_ids) = self.encode_prompt(
            prompt="", prompt_2="",
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        # Negative prompt embeddings (for optional tail-CFG)
        if use_tail_cfg:
            (neg_prompt_embeds, neg_pooled_prompt_embeds, _) = self.encode_prompt(
                prompt=neg_prompt, prompt_2="",
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )

        # ── timesteps ────────────────────────────────────────────────────────
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = (int(height) // self.vae_scale_factor) * (int(width) // self.vae_scale_factor)
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas, mu=mu
        )
        timesteps, num_inference_steps, sigmas = self.get_timesteps(num_inference_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # ── latent preparation ────────────────────────────────────────────────
        num_channels_latents = self.transformer.config.in_channels // 4

        main_latents, main_noise, main_image_latents, main_latent_image_ids = self.prepare_latents(
            image=init_image, timestep=latent_timestep,
            batch_size=batch_size * num_images_per_prompt,
            num_channels_latents=num_channels_latents,
            height=height, width=width,
            dtype=prompt_embeds.dtype, device=device, generator=generator,
            latents=main_latents,
        )
        orig_main_latents = main_image_latents.clone()

        guidance = None
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(main_latents.shape[0])

        ref_latents, ref_noise, ref_image_latents, ref_latent_image_ids = self.prepare_latents(
            image=ref_image_t, timestep=latent_timestep,
            batch_size=batch_size * num_images_per_prompt,
            num_channels_latents=num_channels_latents,
            height=orig_ref_height, width=orig_ref_width,
            dtype=prompt_embeds.dtype, device=device, generator=generator,
            latents=ref_latents,
        )
        orig_ref_latents = ref_image_latents.clone()

        # ── reference mask processing ─────────────────────────────────────────
        ref_mask_condition = self.mask_processor.preprocess(
            image=ref_segment,
            height=orig_ref_height, width=orig_ref_width,
            resize_mode=resize_mode, crops_coords=crops_coords,
        )
        ref_masked_image = ref_image_t * (ref_mask_condition > 0.5)

        resized_ref_masked_image = F.interpolate(
            ref_masked_image, size=(ref_height, ref_width), mode="bilinear"
        )
        resized_ref_mask_condition = F.interpolate(
            ref_mask_condition, size=(ref_height, ref_width), mode="bilinear"
        )

        # ── Step 0b: Reinhard colour harmonisation ────────────────────────────
        if color_harmonize and x1 is not None:
            resized_ref_masked_image = color_harmonize_ref(
                resized_ref_masked_image,
                resized_ref_mask_condition,
                init_image,
                x1, y1, x2, y2,
            )

        # ── optionally expand bounding box downward for partial objects ──────
        # Sides are NOT expanded by default to avoid regenerating surrounding
        # background (grass, sky, adjacent objects).
        if (expand_bottom_frac > 0.0 or expand_top_frac > 0.0 or expand_sides_frac > 0.0) \
                and x1 is not None:
            x1e, y1e, x2e, y2e = expand_bbox_for_partial(
                x1, y1, x2, y2, height, width,
                expand_bottom_frac=expand_bottom_frac,
                expand_top_frac=expand_top_frac,
                expand_sides_frac=expand_sides_frac,
            )
        else:
            x1e, y1e, x2e, y2e = x1, y1, x2, y2

        # ── build initial composite ───────────────────────────────────────────
        if use_seamless_blend:
            composite = seamless_composite(
                init_image,
                resized_ref_masked_image.unsqueeze(0) if resized_ref_masked_image.dim() == 3 else resized_ref_masked_image,
                resized_ref_mask_condition.unsqueeze(0) if resized_ref_mask_condition.dim() == 3 else resized_ref_mask_condition,
                y1e, y2e, x1e, x2e,
            )
        else:
            composite = _soft_blend_composite(
                init_image,
                resized_ref_masked_image,
                resized_ref_mask_condition,
                y1e, y2e, x1e, x2e,
            )

        composite_latents_var = None
        composite_latents_var, composite_noise, composite_image_latents, composite_latent_image_ids = \
            self.prepare_latents(
                image=composite, timestep=latent_timestep,
                batch_size=batch_size * num_images_per_prompt,
                num_channels_latents=num_channels_latents,
                height=height, width=width,
                dtype=prompt_embeds.dtype, device=device, generator=generator,
                latents=composite_latents_var,
            )

        # ── RF-Inversion of composite ─────────────────────────────────────────
        if use_rf_inversion:
            composite_inv_latents = self.controlled_forward_ode(
                image_latents=composite_image_latents,
                latent_image_ids=composite_latent_image_ids,
                sigmas=sigmas,
                gamma=gamma,
                null_prompt_embeds=null_prompt_embeds,
                null_pooled_prompt_embeds=null_pooled_prompt_embeds,
                null_text_ids=null_text_ids,
                timesteps=timesteps,
                guidance=guidance,
                height=height, width=width,
                joint_attention_kwargs=joint_attention_kwargs,
                skip_T=skip_T,
            )
        else:
            _h = 2 * (int(height) // self.vae_scale_factor)
            _w = 2 * (int(width)  // self.vae_scale_factor)
            _shp = (batch_size, num_channels_latents, _h, _w)
            composite_inv_latents = randn_tensor(
                _shp, generator=generator, device=device, dtype=prompt_embeds.dtype
            )
            composite_inv_latents = self._pack_latents(
                composite_inv_latents, batch_size, num_channels_latents, _h, _w
            )

        # ── bounding-box mask (packed latent space) ───────────────────────────
        _traj = self._unpack_latents(main_latents.clone(), height, width, self.vae_scale_factor)
        bbox_mask_spatial = torch.zeros_like(_traj)
        bbox_mask_spatial[
            :, :,
            2 * y1e // self.vae_scale_factor : 2 * y2e // self.vae_scale_factor,
            2 * x1e // self.vae_scale_factor : 2 * x2e // self.vae_scale_factor,
        ] = 1
        bounding_box_mask = self._pack_latents(
            bbox_mask_spatial, batch_size, num_channels_latents,
            2 * (int(height) // self.vae_scale_factor),
            2 * (int(width)  // self.vae_scale_factor),
        )

        gen_latents = composite_inv_latents
        y_0 = composite_image_latents.clone()

        # ── Augment joint_attention_kwargs with ref injection params ──────────
        if joint_attention_kwargs is None:
            joint_attention_kwargs = {}
        if ref_inject_blocks > 0:
            # Only add ref keys when injection is active — MyFluxAttnProcessor2_0
            # has no **kwargs and will crash if it receives unknown keyword args.
            ref_tokens_embedded = self.transformer.x_embedder(
                orig_ref_latents.to(device, dtype=prompt_embeds.dtype)
            )
            joint_attention_kwargs["ref_tokens_embedded"] = ref_tokens_embedded
            joint_attention_kwargs["ref_inject_blocks"]    = ref_inject_blocks
            joint_attention_kwargs["ref_inject_steps"]     = ref_inject_steps
        self._joint_attention_kwargs = joint_attention_kwargs

        # ── denoising loop ────────────────────────────────────────────────────
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        current = joint_attention_kwargs.get("current", {})

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                current["step"] = i
                joint_attention_kwargs["current"] = current
                t_i = 1 - t / 1000

                if self.interrupt:
                    continue

                timestep = t.expand(gen_latents.shape[0]).to(gen_latents.dtype)

                # Single forward pass (positive prompt)
                noise_pred = self.transformer(
                    hidden_states=gen_latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=main_latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # Optional tail-CFG with negative prompt
                if use_tail_cfg and i >= (num_inference_steps - cfg_tail_steps):
                    noise_pred_neg = self.transformer(
                        hidden_states=gen_latents,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=neg_pooled_prompt_embeds,
                        encoder_hidden_states=neg_prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=main_latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_pred_neg + cfg_scale * (noise_pred - noise_pred_neg)

                # RF-Inversion velocity correction
                if use_rf_inversion:
                    v_t_cond = (y_0 - gen_latents) / (1 - t_i)
                    eta_t = eta if start_timestep <= i < stop_timestep else 0.0
                    v_hat_t = -noise_pred + eta_t * (v_t_cond + noise_pred)
                else:
                    v_hat_t = -noise_pred

                gen_latents = self.scheduler.step(-v_hat_t, t, gen_latents, return_dict=False)[0]

                # Background replacement
                init_latents_proper = orig_main_latents
                if i < len(timesteps) - 1:
                    noise_timestep = timesteps[i + 1]
                    sigmas_sched = self.scheduler.sigmas.to(
                        device=orig_main_latents.device, dtype=orig_main_latents.dtype
                    )
                    sigma = sigmas_sched[i + 1].flatten()
                    init_latents_proper = sigma * main_noise + (1.0 - sigma) * orig_main_latents

                gen_latents = (
                    (1 - bounding_box_mask) * init_latents_proper
                    + bounding_box_mask * gen_latents
                )

                if callback_on_step_end is not None:
                    cb_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs
                                 if k in locals()}
                    cb_out = callback_on_step_end(self, i, t, cb_kwargs)
                    gen_latents = cb_out.pop("latents", gen_latents)
                    prompt_embeds = cb_out.pop("prompt_embeds", prompt_embeds)

                if (
                    i == len(timesteps) - 1
                    or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0)
                ):
                    progress_bar.update()

        # ── decode ────────────────────────────────────────────────────────────
        if output_type == "latent":
            image = gen_latents
        else:
            gen_latents = self._unpack_latents(gen_latents, height, width, self.vae_scale_factor)
            gen_latents = (gen_latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(gen_latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)
        return FluxPipelineOutput(images=image)
