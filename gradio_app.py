"""
EEdit Gradio App — Reference-Guided Image Composition
Uses SAM for foreground mask extraction and the original EEdit / FLUX.1 pipeline.

Launch:
    python gradio_app.py [--weights ./weights] [--port 7860] [--share]
"""

import argparse
import os
import sys
import time
import types

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

# ── Ensure repo root is importable ───────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Compatibility shim: newer transformers dropped FLAX_WEIGHTS_NAME but older
# diffusers still imports it. Inject the missing name before diffusers loads.
import transformers.utils as _tu
if not hasattr(_tu, "FLAX_WEIGHTS_NAME"):
    _tu.FLAX_WEIGHTS_NAME = "flax_model.msgpack"

# Patch LayerNorm to fp32 for Blackwell (sm_120) GPUs where bfloat16 underflows
def _ln_fp32_forward(self, x):
    w = self.weight.float() if self.weight is not None else None
    b = self.bias.float() if self.bias is not None else None
    return F.layer_norm(x.float(), self.normalized_shape, w, b, self.eps).to(x.dtype)

nn.LayerNorm.forward = _ln_fp32_forward

import json as _json
from pathlib import Path as _Path

from cache_functions import (cache_init, edit_region_parser, convert_to_cache_index,
                              predefine_cache_fresh_indices)
from MyCodes import MyFluxForward
from MyCodes.MyFluxCompositionPipeline import FluxCompositionPipeline
from MyCodes.myutils import seed_everything
from transformers import SamModel, SamProcessor, T5EncoderModel
from dvlm.prompt_utils import augment_prompt, get_negative_prompt
from dvlm.pipeline_patches import install_cfg_tail_patch, remove_cfg_tail_patch
from dvlm.ref_inject import extract_ref_kv, install_ref_inject, remove_ref_inject

# ── Load domain-specific generation configs ───────────────────────────────────
_DVLM_CFG_DIR = _Path(__file__).parent / "dvlm" / "domain_configs"
DOMAIN_PARAM: dict = {}
for _dk in ["RC", "RP", "RR", "RS"]:
    _f = _DVLM_CFG_DIR / f"{_dk}_config.json"
    if _f.exists():
        DOMAIN_PARAM[_dk] = _json.loads(_f.read_text())["params"][0]

DOMAIN_CHOICES = [
    ("Choose style…",        ""),
    ("Real → Cartoon  (RC)", "RC"),
    ("Real → Painting (RP)", "RP"),
    ("Real → Sketch   (RS)", "RS"),
    ("Real → Real     (RR)", "RR"),
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Global singletons (loaded once on demand) ─────────────────────────────────
_pipe: FluxCompositionPipeline | None = None
_inpaint_pipe = None   # built lazily from _pipe components — zero reload cost
_sam_model: SamModel | None = None
_sam_processor: SamProcessor | None = None


# ╔══════════════════════════════════════════════════════════╗
# ║  Model loading                                           ║
# ╚══════════════════════════════════════════════════════════╝

def _resolve_transformer_config(weights_dir: str) -> str:
    import shutil
    dir_path = os.path.join(weights_dir, "transformer")
    if os.path.isfile(os.path.join(dir_path, "config.json")):
        return dir_path
    fallback = os.path.join(weights_dir, "transformer_config_dir")
    os.makedirs(fallback, exist_ok=True)
    dst = os.path.join(fallback, "config.json")
    if not os.path.exists(dst):
        shutil.copy2(os.path.join(weights_dir, "transformer_config.json"), dst)
    return fallback


def _load_eedit_pipeline(weights_dir: str, dtype=torch.bfloat16) -> FluxCompositionPipeline:
    from MyCodes.FluxTransformer2DModel import FluxTransformer2DModel

    transformer = FluxTransformer2DModel.from_single_file(
        pretrained_model_link_or_path_or_dict=os.path.join(weights_dir, "flux1-dev.safetensors"),
        config=_resolve_transformer_config(weights_dir),
        torch_dtype=dtype,
        local_files_only=True,
    )
    text_encoder_2 = T5EncoderModel.from_pretrained(
        weights_dir, subfolder="text_encoder_2", torch_dtype=dtype
    )
    pipe = FluxCompositionPipeline.from_pretrained(
        weights_dir, transformer=None, text_encoder_2=None, torch_dtype=dtype
    )
    pipe.transformer = transformer
    pipe.text_encoder_2 = text_encoder_2
    pipe.transformer.forward = types.MethodType(MyFluxForward.forward, pipe.transformer)
    pipe.enable_model_cpu_offload()
    return pipe


def load_models(weights_dir: str):
    """Load SAM + EEdit pipeline at startup. Returns final status string."""
    global _pipe, _sam_model, _sam_processor

    weights_dir = (weights_dir or "").strip()
    if not weights_dir or not os.path.isdir(weights_dir):
        return f"❌  Weights directory not found: '{weights_dir}'"

    print("Loading SAM (facebook/sam-vit-base)…")
    try:
        _sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        _sam_model = SamModel.from_pretrained("facebook/sam-vit-base").eval().to("cpu")
    except Exception as exc:
        return f"❌  SAM load failed: {exc}"

    print("Loading EEdit / FLUX.1 pipeline…")
    try:
        _pipe = _load_eedit_pipeline(weights_dir)
    except Exception as exc:
        return f"❌  EEdit load failed: {exc}"

    return "✅  All models loaded — ready!"


# ╔══════════════════════════════════════════════════════════╗
# ║  SAM mask extraction                                     ║
# ╚══════════════════════════════════════════════════════════╝

def _bbox_from_layer(editor_val, orig_size=None):
    """Return (x1,y1,x2,y2) in original image coordinates, or None.

    Gradio's ImageEditor layer is rendered at canvas/display resolution, which
    is often much smaller than the original image. orig_size=(w,h) is used to
    scale the bbox back to original image space.
    """
    if editor_val is None:
        return None
    layers = editor_val.get("layers") or []
    if not layers or layers[0] is None:
        return None
    arr = np.asarray(layers[0])
    if arr.ndim == 3 and arr.shape[2] == 4:
        painted = arr[:, :, 3] > 10
    elif arr.ndim == 3:
        painted = arr.any(axis=2)
    else:
        return None
    if not painted.any():
        return None
    rows = np.where(painted.any(axis=1))[0]
    cols = np.where(painted.any(axis=0))[0]
    x1, y1, x2, y2 = int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1])
    # Scale from layer (canvas) coordinates to original image coordinates
    if orig_size is not None:
        orig_w, orig_h = orig_size
        layer_h, layer_w = arr.shape[:2]
        if layer_w != orig_w or layer_h != orig_h:
            x1 = int(x1 * orig_w / layer_w)
            y1 = int(y1 * orig_h / layer_h)
            x2 = int(x2 * orig_w / layer_w)
            y2 = int(y2 * orig_h / layer_h)
    return x1, y1, x2, y2


def _run_sam(ref_pil: Image.Image, point_xy: tuple[int, int]):
    """Segment the foreground at `point_xy` in `ref_pil` using SAM.

    Returns (mask_L_pil, mask_rgb_pil, overlay_pil).
    """
    px, py = point_xy
    inputs = _sam_processor(
        images=ref_pil,
        input_points=[[[px, py]]],
        input_labels=[[1]],
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = _sam_model(**inputs)

    masks = _sam_processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )
    scores = outputs.iou_scores[0, 0]
    best = int(scores.argmax())
    mask_np = masks[0][0][best].numpy().astype(np.uint8) * 255  # H×W uint8

    mask_pil = Image.fromarray(mask_np, mode="L")

    # Grayscale RGB display for the mask panel
    mask_rgb = np.stack([mask_np] * 3, axis=2)
    mask_display = Image.fromarray(mask_rgb)

    # Overlay: dim background, keep foreground bright
    ref_arr = np.array(ref_pil)
    overlay = ref_arr.copy()
    bg = mask_np < 128
    overlay[bg] = (overlay[bg] * 0.3).astype(np.uint8)
    # Green tint on mask region outline
    overlay_pil = Image.fromarray(overlay)

    return mask_pil, mask_display, overlay_pil


def extract_mask_draw(ref_editor_val):
    """
    Extract mask from the reference ImageEditor.
    If the user drew a rectangle → use it as SAM bbox prompt.
    If no drawing → fall back to the image centre point.
    """
    if _sam_model is None:
        return None, None, None, "⚠️  Load models first"
    if ref_editor_val is None:
        return None, None, None, "⚠️  Upload a reference image first"

    bg = ref_editor_val.get("background")
    if bg is None:
        return None, None, None, "⚠️  Upload a reference image first"
    ref_pil = Image.fromarray(np.asarray(bg)).convert("RGB")

    bbox = _bbox_from_layer(ref_editor_val, orig_size=ref_pil.size)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        inputs = _sam_processor(
            images=ref_pil, input_boxes=[[[x1, y1, x2, y2]]], return_tensors="pt"
        )
        inputs = {k: v.to("cpu") for k, v in inputs.items()}
        with torch.no_grad():
            out = _sam_model(**inputs)
        prompt_desc = f"bbox ({x1},{y1})→({x2},{y2})"
    else:
        cx, cy = ref_pil.width // 2, ref_pil.height // 2
        mask_pil, mask_display, overlay_pil = _run_sam(ref_pil, (cx, cy))
        return mask_pil, mask_display, overlay_pil, f"✅  Auto-mask via centre ({cx},{cy}) — draw a rectangle for better results"

    masks = _sam_processor.image_processor.post_process_masks(
        out.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )
    scores = out.iou_scores[0, 0]
    best   = int(scores.argmax())
    mask_np = masks[0][0][best].numpy().astype(np.uint8) * 255

    mask_pil     = Image.fromarray(mask_np, mode="L")
    mask_display = Image.fromarray(np.stack([mask_np]*3, axis=2))
    ref_arr      = np.array(ref_pil)
    overlay      = ref_arr.copy()
    overlay[mask_np < 128] = (overlay[mask_np < 128] * 0.3).astype(np.uint8)
    overlay_pil  = Image.fromarray(overlay)
    return mask_pil, mask_display, overlay_pil, f"✅  Mask via {prompt_desc}"


def extract_mask_center(ref_np):
    """Kept for backward compatibility — uses centre point."""
    if _sam_model is None:
        return None, None, None, "⚠️  Load models first"
    if ref_np is None:
        return None, None, None, "Upload a reference image first"
    ref_pil = Image.fromarray(ref_np).convert("RGB")
    cx, cy = ref_pil.width // 2, ref_pil.height // 2
    mask_pil, mask_display, overlay_pil = _run_sam(ref_pil, (cx, cy))
    return mask_pil, mask_display, overlay_pil, f"✅  Auto-mask via centre point ({cx}, {cy})"


# ╔══════════════════════════════════════════════════════════╗
# ║  Reference modification (pose / style via prompt)        ║
# ╚══════════════════════════════════════════════════════════╝

def _get_inpaint_pipe():
    """Build FluxInpaintPipeline from already-loaded _pipe components — zero reload."""
    global _inpaint_pipe
    if _inpaint_pipe is not None:
        return _inpaint_pipe
    from diffusers import FluxInpaintPipeline
    _inpaint_pipe = FluxInpaintPipeline(
        scheduler=_pipe.scheduler,
        vae=_pipe.vae,
        text_encoder=_pipe.text_encoder,
        text_encoder_2=_pipe.text_encoder_2,
        tokenizer=_pipe.tokenizer,
        tokenizer_2=_pipe.tokenizer_2,
        transformer=_pipe.transformer,
    )
    return _inpaint_pipe


def modify_reference(ref_editor_val, mask_pil, mod_prompt, strength, seed):
    """
    Edit the reference object's pose / style using EEdit's inpaint pipeline
    with RF-Inversion and cache acceleration.

    The mask from Step ① defines WHICH region to edit.
    The modification prompt describes the desired result.
    """
    if _pipe is None:
        return None, None, "⚠️  Load models first"
    bg = ref_editor_val.get("background") if ref_editor_val else None
    if bg is None:
        return None, None, "⚠️  Upload a reference image first"
    if mask_pil is None:
        return None, None, "⚠️  Extract mask first (Step ①)"
    if not mod_prompt or not mod_prompt.strip():
        return None, None, "⚠️  Enter a modification prompt"

    ref_pil  = Image.fromarray(np.asarray(bg)).convert("RGB")
    mask_pil = mask_pil.convert("L")

    # Resize to nearest 64-divisible size FLUX requires
    w, h   = ref_pil.size
    tgt_w  = (min(w, 512) // 64) * 64 or 64
    tgt_h  = (min(h, 512) // 64) * 64 or 64
    ref_rs = ref_pil.resize((tgt_w, tgt_h), Image.LANCZOS)
    msk_rs = mask_pil.resize((tgt_w, tgt_h), Image.NEAREST)

    inpaint_pipe = _get_inpaint_pipe()

    # EEdit cache kwargs — same acceleration used in composition
    from cache_functions import cache_init, predefine_cache_fresh_indices
    model_kwargs = {
        "fresh_ratio": 0.1, "cache_type": "ours_predefine",
        "ratio_scheduler": "constant", "force_fresh": "global",
        "fresh_threshold": 3, "soft_fresh_weight": 0.25,
        "tailing_step": 1, "edit_base": 2,
        "hw": (tgt_h // 16, tgt_w // 16),
    }
    num_steps = 20
    cache_dic, current = cache_init(model_kwargs, num_steps, None)
    predefine_cache_fresh_indices(cache_dic, current)
    joint_attn_kwargs = {
        "use_attn_map": False, "cache_dic": cache_dic,
        "use_cache": True, "current": current,
    }

    try:
        with torch.no_grad():
            result = inpaint_pipe(
                prompt=mod_prompt.strip(),
                image=ref_rs,
                mask_image=msk_rs,
                height=tgt_h,
                width=tgt_w,
                num_inference_steps=num_steps,
                guidance_scale=7.0,
                strength=float(strength),
                generator=torch.Generator(device=DEVICE).manual_seed(int(seed)),
                joint_attention_kwargs=joint_attn_kwargs,
            )
        modified_pil = result.images[0].resize((w, h), Image.LANCZOS)
        # Overlay modified result with mask for display
        overlay = np.array(modified_pil.copy())
        mask_np = np.array(mask_pil.resize((w, h), Image.NEAREST)) > 128
        overlay[~mask_np] = (overlay[~mask_np] * 0.4).astype(np.uint8)
        overlay_pil = Image.fromarray(overlay)
        return modified_pil, overlay_pil, f"✅  Modified — prompt: '{mod_prompt[:60]}'"
    except Exception as exc:
        return None, None, f"❌  {exc}"


# ╔══════════════════════════════════════════════════════════╗
# ║  Placement helpers                                       ║
# ╚══════════════════════════════════════════════════════════╝

def load_source_to_editor(src_np):
    """Load the (resized) source image as the ImageEditor background."""
    if src_np is None:
        return gr.update()
    arr = np.array(Image.fromarray(src_np).convert("RGB").resize((512, 512)))
    return {"background": arr, "layers": [], "composite": arr}


def bbox_from_drawing(editor_val) -> tuple[int, int, int, int]:
    """Compute bounding box from the painted region in the ImageEditor layer."""
    if editor_val is None:
        return 50, 50, 250, 350
    layers = editor_val.get("layers") or []
    if not layers:
        return 50, 50, 250, 350
    arr = layers[0]
    if arr is None:
        return 50, 50, 250, 350
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[2] == 4:
        painted = arr[:, :, 3] > 10
    elif arr.ndim == 3:
        painted = arr.any(axis=2)
    else:
        return 50, 50, 250, 350
    if not painted.any():
        return 50, 50, 250, 350
    rows = np.where(painted.any(axis=1))[0]
    cols = np.where(painted.any(axis=0))[0]
    return int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1])


def placement_preview(source_np, ref_np, mask_pil, x1, y1, x2, y2):
    """Composite preview: paste masked reference into source at the given bbox."""
    if source_np is None or ref_np is None or mask_pil is None:
        return None
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)

    src = Image.fromarray(source_np).convert("RGB").resize((512, 512))
    ref = Image.fromarray(ref_np).convert("RGB").resize((bw, bh), Image.LANCZOS)
    msk = mask_pil.convert("L").resize((bw, bh), Image.LANCZOS)

    comp = src.copy()
    comp.paste(ref, (x1, y1), msk)

    draw = ImageDraw.Draw(comp)
    draw.rectangle([x1, y1, x2, y2], outline="#FF4444", width=3)
    return comp


# ╔══════════════════════════════════════════════════════════╗
# ║  EEdit generation                                        ║
# ╚══════════════════════════════════════════════════════════╝

def generate(source_np, ref_np, mask_pil, target_prompt: str, domain: str,
             x1, y1, x2, y2, seed: int = 42):
    if _pipe is None:
        return None, "Load models first"
    if source_np is None or ref_np is None:
        return None, "Provide source and reference images"
    if mask_pil is None:
        return None, "Extract the reference mask first (Step 1)"

    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    if x1 >= x2 or y1 >= y2:
        return None, f"Invalid bbox ({x1},{y1})->({x2},{y2}) — ensure x1<x2, y1<y2"
    if not target_prompt.strip():
        return None, "Enter a target prompt"

    param = DOMAIN_PARAM.get(domain, DOMAIN_PARAM.get("RR", {}))
    if not param:
        return None, f"No domain config found for '{domain}'"

    seed_everything(seed)
    final_prompt = augment_prompt(target_prompt.strip(), domain)
    height = width = 512

    main_image  = Image.fromarray(source_np).convert("RGB").resize((512, 512))
    ref_image   = Image.fromarray(ref_np).convert("RGB")
    ref_segment = mask_pil.convert("L")

    # RS domain: convert reference to grayscale to match the sketch scene
    if domain == "RS":
        ref_image = ref_image.convert("L").convert("RGB")

    num_steps  = param["num_inference_steps"]
    cache_type = param.get("cache_type", "ours_predefine")

    model_kwargs = {
        "fresh_ratio":       param["fresh_ratio"],
        "cache_type":        cache_type,
        "ratio_scheduler":   "constant",
        "force_fresh":       "global",
        "fresh_threshold":   param["fresh_threshold"],
        "soft_fresh_weight": param["soft_fresh_weight"],
        "tailing_step":      param["tailing_step"],
        "edit_base":         2,
        "hw":                (height // 16, width // 16),
    }

    cascade_num = param.get("cascade_num", 5)
    edit_idx = (
        None if cascade_num == 0
        else edit_region_parser(x1, y1, x2, y2, cascade_num=cascade_num,
                                height=height, width=width)
    )
    cache_dic, current = cache_init(model_kwargs, num_steps, edit_idx)
    current["edit_idx_merged"] = convert_to_cache_index(
        edit_idx, edit_base=2, bonus_ratio=0.8, height=height, width=width
    ).to(DEVICE)
    if cache_type == "ours_predefine":
        predefine_cache_fresh_indices(cache_dic, current)

    joint_attention_kwargs = {
        "use_attn_map": False,
        "cache_dic":    cache_dic,
        "use_cache":    param.get("use_cache", True),
        "current":      current,
    }

    # With enable_model_cpu_offload the transformer parameters sit on CPU;
    # always use DEVICE (cuda) so VAE / transformer forward passes land on GPU.
    device = torch.device(DEVICE)
    dtype  = torch.bfloat16

    # ── Tail CFG patch ────────────────────────────────────────────────────────
    cfg_tail_steps = param.get("cfg_tail_steps", 0)
    cfg_scale_val  = param.get("cfg_scale", 3.5)
    if cfg_tail_steps > 0:
        neg_embeds, neg_pooled, _ = _pipe.encode_prompt(
            prompt=get_negative_prompt(domain), prompt_2=None,
            device=device, num_images_per_prompt=1, max_sequence_length=512,
        )
        install_cfg_tail_patch(_pipe, num_steps, cfg_tail_steps,
                               cfg_scale_val, neg_embeds, neg_pooled)

    # ── Ref injection ─────────────────────────────────────────────────────────
    ref_inject_blocks = param.get("ref_inject_blocks", 0)
    ref_inject_steps  = param.get("ref_inject_steps", 0)
    if ref_inject_blocks > 0 and ref_inject_steps > 0:
        pos_embeds, pos_pooled, text_ids = _pipe.encode_prompt(
            prompt=final_prompt, prompt_2=None,
            device=device, num_images_per_prompt=1, max_sequence_length=512,
        )
        ref_kv = extract_ref_kv(_pipe, ref_image, pos_embeds, pos_pooled,
                                 text_ids, ref_inject_blocks, device, dtype)
        install_ref_inject(_pipe, ref_kv, ref_inject_blocks, ref_inject_steps)

    t0 = time.time()
    neg_prompt_for_gen = None if cfg_tail_steps > 0 else get_negative_prompt(domain)

    try:
        res = _pipe.gen(
            prompt=final_prompt,
            neg_prompt=neg_prompt_for_gen,
            main_image=main_image,
            ref_image=ref_image,
            ref_segment=ref_segment,
            height=height, width=width,
            x1=x1, y1=y1, x2=x2, y2=y2,
            num_inference_steps=num_steps,
            guidance_scale=cfg_scale_val,
            joint_attention_kwargs=joint_attention_kwargs,
            use_rf_inversion=param.get("use_rf_inversion", True),
            eta=param["eta"],
            gamma=param["gamma"],
            start_timestep=param.get("start_timestep", 0),
            stop_timestep=param.get("stop_timestep", 13),
            blend_ratio=param.get("blend_ratio", 0.0),
            generator=torch.Generator(device=DEVICE).manual_seed(seed),
            skip_T=param.get("inv_skip", 2),
        )
    finally:
        remove_cfg_tail_patch(_pipe)
        remove_ref_inject(_pipe)

    elapsed = time.time() - t0
    return res.images[0], (f"Done in {elapsed:.1f}s  |  domain={domain}  "
                           f"|  ref_inject={ref_inject_blocks}b/{ref_inject_steps}s  "
                           f"|  cfg_tail={cfg_tail_steps}")


# ╔══════════════════════════════════════════════════════════╗
# ║  Gradio UI                                               ║
# ╚══════════════════════════════════════════════════════════╝

def build_app(weights_dir: str = "./weights") -> gr.Blocks:
    _weights_dir = weights_dir
    with gr.Blocks(title="EEdit — Reference-Guided Composition") as demo:

        gr.Markdown(
            "## EEdit — Reference-Guided Image Composition\n"
            "Paint over the **object** in the reference image → extract its mask.  \n"
            "Paint the **placement area** on the background → click Generate."
        )

        status_box = gr.Textbox(
            label="Status", interactive=False, value="Loading models…"
        )

        gr.Markdown("---")

        # ── Prompt section (mirrors smoke app) ───────────────────────────────
        with gr.Row():
            domain_dd = gr.Dropdown(
                label="Domain (scene style)",
                choices=[c[0] for c in DOMAIN_CHOICES],
                value=DOMAIN_CHOICES[0][0],
                scale=1,
            )
            target_prompt = gr.Textbox(
                label="Base prompt",
                placeholder="e.g. 'a sheep in the forest'",
                scale=2,
            )
        aug_prompt_box = gr.Textbox(
            label="Augmented prompt (what FLUX sees)",
            value="", interactive=False, lines=3,
        )

        gr.Markdown("---")

        # ── Shared state ─────────────────────────────────────────────────────
        mask_state        = gr.State(None)
        modified_ref_state = gr.State(None)

        # ── Two-column panel (mirrors smoke app) ─────────────────────────────
        with gr.Row():

            # ── LEFT: Reference + SAM ────────────────────────────────────────
            with gr.Column():
                gr.Markdown("### ① Reference — paint around the object")
                gr.Markdown("*Use the brush to mark the object, or enter exact pixel coordinates below.*")
                ref_editor = gr.ImageEditor(
                    label="Upload reference image, then paint over the object",
                    type="numpy",
                    brush=gr.Brush(default_size=20, colors=["#FF4444"]),
                    height=300,
                )
                with gr.Row():
                    extract_draw_btn = gr.Button("Extract Mask", variant="primary")
                    auto_btn         = gr.Button("Auto: centre point", variant="secondary")
                mask_status = gr.Textbox(label="", interactive=False, lines=1, max_lines=1)
                with gr.Row():
                    mask_display = gr.Image(label="Mask (B&W)",          type="pil", interactive=False, height=180)
                    mask_overlay = gr.Image(label="Reference + Overlay", type="pil", interactive=False, height=180)

                with gr.Accordion("Optional: modify reference pose/style before compositing", open=False):
                    mod_prompt = gr.Textbox(
                        label="Modification prompt",
                        placeholder="e.g. 'sheep jumping in the air'",
                        lines=2,
                    )
                    with gr.Row():
                        mod_strength = gr.Slider(0.5, 1.0, value=0.80, step=0.05, label="Edit strength")
                        mod_seed     = gr.Number(value=42, precision=0, label="Seed")
                    with gr.Row():
                        mod_btn   = gr.Button("Modify Reference",     variant="primary")
                        regen_btn = gr.Button("Regenerate (+1 seed)", variant="secondary")
                    mod_status = gr.Textbox(label="", interactive=False, lines=1, max_lines=1)
                    with gr.Row():
                        mod_out     = gr.Image(label="Modified reference", height=180, interactive=False)
                        mod_overlay = gr.Image(label="Modified + mask",    height=180, interactive=False)

            # ── RIGHT: Background + Placement ────────────────────────────────
            with gr.Column():
                gr.Markdown("### ② Background — paint the placement region")
                gr.Markdown("*Paint where the object should appear, or enter exact coordinates below.*")
                bg_editor = gr.ImageEditor(
                    label="Upload background image, then paint the placement area",
                    type="numpy",
                    brush=gr.Brush(default_size=20, colors=["#4488FF"]),
                    height=300,
                )
                with gr.Accordion("Or enter exact pixel coordinates", open=False):
                    with gr.Row():
                        x1_in = gr.Number(label="x1 (left)",   value=0, precision=0, minimum=0, maximum=512)
                        y1_in = gr.Number(label="y1 (top)",    value=0, precision=0, minimum=0, maximum=512)
                        x2_in = gr.Number(label="x2 (right)",  value=0, precision=0, minimum=0, maximum=512)
                        y2_in = gr.Number(label="y2 (bottom)", value=0, precision=0, minimum=0, maximum=512)
                    gr.Markdown("*Enter all four to override the painted region.*")

                comp_btn    = gr.Button("Preview Composite", variant="secondary")
                comp_status = gr.Textbox(label="", interactive=False, lines=1, max_lines=1)
                comp_out    = gr.Image(label="Composite preview (red box = placement)", interactive=False, height=300)

        gr.Markdown("---")

        # ── Seed (only manual override needed — all other params from domain config) ──
        with gr.Accordion("Advanced", open=False):
            seed_in = gr.Number(label="Seed", value=42, precision=0)

        # ── Generate + Result ────────────────────────────────────────────────
        generate_btn = gr.Button("Generate Image", variant="primary", interactive=False, size="lg")

        with gr.Row():
            result_img = gr.Image(label="Generated Image", interactive=False, height=512)
        gen_status = gr.Textbox(label="", interactive=False, lines=1, max_lines=1)

        # ── Event wiring ─────────────────────────────────────────────────────

        def _domain_key(label):
            for lbl, key in DOMAIN_CHOICES:
                if lbl == label:
                    return key
            return "RR"

        # Load models on startup
        def _startup():
            msg = load_models(_weights_dir)
            return msg, gr.update(interactive="✅" in msg)

        demo.load(_startup, outputs=[status_box, generate_btn])

        # Prompt preview
        def _preview_aug_prompt(base, domain_label):
            key = _domain_key(domain_label)
            if not key:
                return ""
            return augment_prompt(base.strip(), key) if base.strip() else augment_prompt("(your prompt here)", key)

        for trigger in [target_prompt, domain_dd]:
            trigger.change(_preview_aug_prompt, inputs=[target_prompt, domain_dd], outputs=[aug_prompt_box])

        # SAM: draw → mask
        extract_draw_btn.click(
            extract_mask_draw,
            inputs=[ref_editor],
            outputs=[mask_state, mask_display, mask_overlay, mask_status],
        )

        # SAM: auto centre
        def _auto_from_editor(ed):
            bg = ed.get("background") if ed else None
            if bg is None:
                return None, None, None, "Upload a reference image first"
            return extract_mask_center(np.asarray(bg))

        auto_btn.click(
            _auto_from_editor,
            inputs=[ref_editor],
            outputs=[mask_state, mask_display, mask_overlay, mask_status],
        )

        # Modify reference
        def _modify(ref_ed, mask, prompt, strength, seed):
            out, overlay, status = modify_reference(ref_ed, mask, prompt, strength, seed)
            return out, overlay, status, out

        def _regen(ref_ed, mask, prompt, strength, seed):
            return _modify(ref_ed, mask, prompt, strength, int(seed) + 1)

        mod_btn.click(
            _modify,
            inputs=[ref_editor, mask_state, mod_prompt, mod_strength, mod_seed],
            outputs=[mod_out, mod_overlay, mod_status, modified_ref_state],
        )
        regen_btn.click(
            _regen,
            inputs=[ref_editor, mask_state, mod_prompt, mod_strength, mod_seed],
            outputs=[mod_out, mod_overlay, mod_status, modified_ref_state],
        )

        def _get_bg_np(bg_ed):
            """Extract background numpy array from ImageEditor value."""
            if bg_ed is None:
                return None
            bg = bg_ed.get("background")
            return np.asarray(bg) if bg is not None else None

        def _get_bbox(bg_ed, x1, y1, x2, y2):
            """Use painted region if no manual coords given, else use coords.
            Scales layer (canvas) coordinates to 512×512 space to match placement_preview."""
            _x1, _y1, _x2, _y2 = int(x1), int(y1), int(x2), int(y2)
            if _x2 > _x1 and _y2 > _y1:
                return _x1, _y1, _x2, _y2
            bbox = _bbox_from_layer(bg_ed, orig_size=(512, 512))
            if bbox:
                return bbox
            return _x1, _y1, _x2, _y2

        # Composite preview
        def _composite_preview(bg_ed, ref_ed, modified_ref, mask, x1, y1, x2, y2):
            src_np = _get_bg_np(bg_ed)
            if src_np is None:
                return None, "Upload a background image first"
            ref_np = np.array(modified_ref) if modified_ref is not None else None
            if ref_np is None:
                bg = ref_ed.get("background") if ref_ed else None
                ref_np = np.asarray(bg) if bg is not None else None
            if ref_np is None:
                return None, "Upload a reference image first"
            bx1, by1, bx2, by2 = _get_bbox(bg_ed, x1, y1, x2, y2)
            comp = placement_preview(src_np, ref_np, mask, bx1, by1, bx2, by2)
            return comp, f"Composite ready — placement ({bx1},{by1})->({bx2},{by2})"

        comp_btn.click(
            _composite_preview,
            inputs=[bg_editor, ref_editor, modified_ref_state, mask_state,
                    x1_in, y1_in, x2_in, y2_in],
            outputs=[comp_out, comp_status],
        )

        # Generation
        def _generate_wrap(bg_ed, ref_ed, modified_ref, mask, prompt, domain_label,
                           x1, y1, x2, y2, seed):
            src_np = _get_bg_np(bg_ed)
            ref_np = np.array(modified_ref) if modified_ref is not None else None
            if ref_np is None:
                bg = ref_ed.get("background") if ref_ed else None
                ref_np = np.asarray(bg) if bg is not None else None
            bx1, by1, bx2, by2 = _get_bbox(bg_ed, x1, y1, x2, y2)
            return generate(src_np, ref_np, mask, prompt, _domain_key(domain_label),
                            bx1, by1, bx2, by2, int(seed))

        generate_btn.click(
            _generate_wrap,
            inputs=[
                bg_editor, ref_editor, modified_ref_state, mask_state,
                target_prompt, domain_dd,
                x1_in, y1_in, x2_in, y2_in,
                seed_in,
            ],
            outputs=[result_img, gen_status],
        )

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="EEdit Gradio App")
    p.add_argument("--weights", type=str, default="./weights",
                   help="Default weights directory shown in the UI")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true",
                   help="Create a public Gradio share link")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_app(weights_dir=args.weights).launch(server_port=args.port, share=args.share)
