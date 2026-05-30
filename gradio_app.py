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

# Patch LayerNorm to fp32 for Blackwell (sm_120) GPUs where bfloat16 underflows
def _ln_fp32_forward(self, x):
    w = self.weight.float() if self.weight is not None else None
    b = self.bias.float() if self.bias is not None else None
    return F.layer_norm(x.float(), self.normalized_shape, w, b, self.eps).to(x.dtype)

nn.LayerNorm.forward = _ln_fp32_forward

from cache_functions import cache_init, edit_region_parser, convert_to_cache_index
from MyCodes import MyFluxForward
from MyCodes.MyFluxCompositionPipeline import FluxCompositionPipeline
from MyCodes.myutils import seed_everything
from transformers import SamModel, SamProcessor, T5EncoderModel
from dvlm.prompt_utils import augment_prompt, get_negative_prompt

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
    """Generator: yields (status_text, generate_btn_update) — streams progress to UI."""
    global _pipe, _sam_model, _sam_processor

    weights_dir = (weights_dir or "").strip()
    if not weights_dir or not os.path.isdir(weights_dir):
        yield "❌  Invalid weights directory — check the path", gr.update(interactive=False)
        return

    yield "⏳  Loading SAM (facebook/sam-vit-base)…", gr.update(interactive=False)
    try:
        _sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        _sam_model = SamModel.from_pretrained("facebook/sam-vit-base").eval().to("cpu")
    except Exception as exc:
        yield f"❌  SAM load failed: {exc}", gr.update(interactive=False)
        return

    yield "⏳  Loading EEdit / FLUX.1 pipeline (may take several minutes)…", gr.update(interactive=False)
    try:
        _pipe = _load_eedit_pipeline(weights_dir)
    except Exception as exc:
        yield f"❌  EEdit load failed: {exc}", gr.update(interactive=False)
        return

    yield "✅  All models loaded — ready!", gr.update(interactive=True)


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

def generate(
    source_np,
    ref_np,
    mask_pil,
    target_prompt: str,
    domain: str,
    x1, y1, x2, y2,
    num_steps: int,
    guidance_scale: float,
    eta: float,
    gamma: float,
    blend_ratio: float,
    use_cache: bool,
    cascade_num: int,
):
    if _pipe is None:
        return None, "⚠️  Load models first"
    if source_np is None or ref_np is None:
        return None, "❌  Provide source and reference images"
    if mask_pil is None:
        return None, "❌  Extract the reference mask first (Step ①)"

    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    if x1 >= x2 or y1 >= y2:
        return None, f"❌  Invalid bbox ({x1},{y1})→({x2},{y2}) — ensure x1<x2 and y1<y2"
    if not target_prompt.strip():
        return None, "❌  Enter a target prompt"

    seed_everything(42)

    # Augment prompt with domain-specific style/integration language
    final_prompt = augment_prompt(target_prompt.strip(), domain)

    main_image   = Image.fromarray(source_np).convert("RGB").resize((512, 512))
    ref_image    = Image.fromarray(ref_np).convert("RGB")
    ref_segment  = mask_pil.convert("L")
    height, width = 512, 512

    # Build cache / model kwargs identical to composition_gen.py
    model_kwargs = {
        "fresh_ratio":      0.1,
        "cache_type":       "ours_cache",
        "ratio_scheduler":  "constant",
        "force_fresh":      "global",
        "fresh_threshold":  3,
        "soft_fresh_weight": 0.25,
        "tailing_step":     1,
        "edit_base":        2,
        "hw":               (height // 16, width // 16),
    }

    edit_idx = (
        None
        if int(cascade_num) == 0
        else edit_region_parser(x1, y1, x2, y2, cascade_num=int(cascade_num), height=height, width=width)
    )

    cache_dic, current = cache_init(model_kwargs, int(num_steps), edit_idx)
    current["edit_idx_merged"] = convert_to_cache_index(
        edit_idx, edit_base=2, bonus_ratio=0.8, height=height, width=width
    ).to(DEVICE)

    joint_attention_kwargs = {
        "use_attn_map": False,
        "cache_dic":    cache_dic,
        "use_cache":    bool(use_cache),
        "current":      current,
    }

    t0 = time.time()
    generator = torch.Generator(device=DEVICE).manual_seed(42)

    res = _pipe.gen(
        prompt=final_prompt,
        neg_prompt=get_negative_prompt(domain),
        main_image=main_image,
        ref_image=ref_image,
        ref_segment=ref_segment,
        height=height,
        width=width,
        x1=x1, y1=y1, x2=x2, y2=y2,
        num_inference_steps=int(num_steps),
        guidance_scale=float(guidance_scale),
        joint_attention_kwargs=joint_attention_kwargs,
        use_rf_inversion=True,
        eta=float(eta),
        gamma=float(gamma),
        start_timestep=0,
        stop_timestep=10,
        blend_ratio=float(blend_ratio),
        generator=generator,
        skip_T=2,
    )

    elapsed = time.time() - t0
    return res.images[0], f"✅  Generated in {elapsed:.1f}s  |  domain={domain}  |  prompt: {final_prompt[:80]}…"


# ╔══════════════════════════════════════════════════════════╗
# ║  Gradio UI                                               ║
# ╚══════════════════════════════════════════════════════════╝

def build_app() -> gr.Blocks:
    with gr.Blocks(title="EEdit — Reference-Guided Composition") as demo:

        gr.Markdown(
            "# EEdit — Reference-Guided Image Composition\n"
            "Paste a foreground object from a **reference image** into a "
            "**source/background image** using FLUX.1 + RF-Inversion."
        )

        # ── Model loading row ────────────────────────────────────────────────
        with gr.Row():
            weights_box = gr.Textbox(
                label="Weights Directory",
                value="./weights",
                placeholder="/path/to/weights",
                scale=3,
            )
            load_btn = gr.Button("Load Models", variant="primary", scale=1)
        status_box = gr.Textbox(
            label="Status", interactive=False, value="Models not loaded yet"
        )

        gr.Markdown("---")

        # ── Two-column layout ────────────────────────────────────────────────
        with gr.Row(equal_height=False):

            # ── Left: inputs & params ────────────────────────────────────────
            with gr.Column(scale=1, min_width=340):
                gr.Markdown("### Inputs")

                source_img = gr.Image(
                    label="Source / Background Image  (resized to 512×512 internally)",
                    type="numpy",
                    height=210,
                )
                ref_editor = gr.ImageEditor(
                    label="Reference Image — draw a rectangle around the object to extract",
                    type="numpy",
                    brush=gr.Brush(default_size=8, colors=["#FF4444"]),
                    height=210,
                )

                domain_dd = gr.Dropdown(
                    label="Domain (scene style)",
                    choices=[c[0] for c in DOMAIN_CHOICES],
                    value=DOMAIN_CHOICES[0][0],
                )
                target_prompt = gr.Textbox(
                    label="Base Prompt",
                    placeholder="e.g. 'a sheep in the forest'",
                    lines=2,
                )
                aug_prompt_box = gr.Textbox(
                    label="Augmented Prompt (auto-generated — what FLUX sees)",
                    value="",
                    lines=3,
                    interactive=False,
                )

                with gr.Accordion("Advanced Parameters", open=False):
                    num_steps   = gr.Slider(8,  50,  value=28,  step=1,    label="Inference Steps")
                    guidance    = gr.Slider(1,  15,  value=7.0, step=0.5,  label="Guidance Scale")
                    eta         = gr.Slider(0,  1,   value=0.6, step=0.05, label="Eta  (RF-Inversion)")
                    gamma       = gr.Slider(0,  2,   value=0.6, step=0.05, label="Gamma")
                    blend_ratio = gr.Slider(0,  1,   value=0.0, step=0.05, label="Blend Ratio")
                    use_cache   = gr.Checkbox(value=True, label="Cache Acceleration")
                    cascade_num = gr.Slider(0,  10,  value=5,   step=1,    label="Cascade Num")

                generate_btn = gr.Button(
                    "Generate Image", variant="primary", interactive=False, size="lg"
                )

            # ── Right: step panels ───────────────────────────────────────────
            with gr.Column(scale=1, min_width=440):
                with gr.Tabs():

                    # ── Step 1: Mask ─────────────────────────────────────────
                    with gr.Tab("① Extract Mask"):
                        gr.Markdown(
                            "**Draw** a rectangle on the reference image (left panel), "
                            "then click **Extract Mask**.  \n"
                            "No drawing? Click **Auto: use centre** as a fallback."
                        )
                        extract_draw_btn = gr.Button("Extract Mask from Drawing", variant="primary")
                        auto_btn = gr.Button("Auto: use centre point", variant="secondary")
                        mask_status = gr.Textbox(
                            label="", interactive=False, lines=1, max_lines=1
                        )
                        with gr.Row():
                            mask_display = gr.Image(
                                label="Extracted Mask",
                                type="pil", interactive=False, height=200,
                            )
                            mask_overlay = gr.Image(
                                label="Reference + Mask Overlay",
                                type="pil", interactive=False, height=200,
                            )
                        mask_state = gr.State(None)

                    # ── Step 1b: Modify Reference ────────────────────────────
                    with gr.Tab("① b Modify Reference"):
                        gr.Markdown(
                            "**Optionally** change the reference object's pose or style "
                            "before compositing.  \n"
                            "Uses EEdit's FLUX inpaint pipeline with cache acceleration.  \n"
                            "Skip this tab if you want to use the original reference as-is."
                        )
                        mod_prompt = gr.Textbox(
                            label="Modification prompt",
                            placeholder="e.g. 'sheep jumping in the air' or 'sheep facing left'",
                            lines=2,
                        )
                        with gr.Row():
                            mod_strength = gr.Slider(0.5, 1.0, value=0.80, step=0.05,
                                                     label="Edit strength  (higher = more change)")
                            mod_seed = gr.Number(value=42, precision=0, label="Seed")
                        with gr.Row():
                            mod_btn   = gr.Button("Modify Reference",     variant="primary")
                            regen_btn = gr.Button("Regenerate (+1 seed)", variant="secondary")
                        mod_status = gr.Textbox(label="", interactive=False, lines=1, max_lines=1)
                        with gr.Row():
                            mod_out     = gr.Image(label="Modified reference", height=220, interactive=False)
                            mod_overlay = gr.Image(label="Modified + mask",    height=220, interactive=False)
                        modified_ref_state = gr.State(None)

                    # ── Step 2: Placement ────────────────────────────────────
                    with gr.Tab("② Set Placement"):
                        gr.Markdown(
                            "1. Click **Load Source →** to bring your background into the editor.  \n"
                            "2. **Paint / draw** over the region where the object should appear.  \n"
                            "3. Click **Extract Bbox** — or type coordinates directly below."
                        )
                        load_editor_btn = gr.Button("Load Source →", variant="secondary")
                        placement_editor = gr.ImageEditor(
                            label="Draw placement region on source image",
                            type="numpy",
                            brush=gr.Brush(default_size=14, colors=["#FF4444"]),
                            height=300,
                        )
                        extract_bbox_btn = gr.Button(
                            "Extract Bbox from Drawing", variant="secondary"
                        )
                        with gr.Row():
                            x1_in = gr.Number(label="x1", value=100, precision=0, minimum=0, maximum=512)
                            y1_in = gr.Number(label="y1", value=100, precision=0, minimum=0, maximum=512)
                            x2_in = gr.Number(label="x2", value=300, precision=0, minimum=0, maximum=512)
                            y2_in = gr.Number(label="y2", value=400, precision=0, minimum=0, maximum=512)

                    # ── Step 3: Preview ──────────────────────────────────────
                    with gr.Tab("③ Composite Preview"):
                        gr.Markdown(
                            "Preview the masked reference pasted at your chosen location. "
                            "Adjust bbox coordinates in Step ② and refresh."
                        )
                        preview_btn = gr.Button("Refresh Preview", variant="secondary")
                        comp_preview = gr.Image(
                            label="Composite Preview (red box = bbox)",
                            interactive=False, height=340,
                        )

                    # ── Step 4: Result ───────────────────────────────────────
                    with gr.Tab("④ Result"):
                        result_img = gr.Image(
                            label="Generated Image", interactive=False, height=340
                        )
                        gen_status = gr.Textbox(
                            label="", interactive=False, lines=1, max_lines=1
                        )

        # ── Event wiring ─────────────────────────────────────────────────────

        load_btn.click(
            load_models,
            inputs=[weights_box],
            outputs=[status_box, generate_btn],
        )

        # SAM: draw rectangle on reference → extract mask
        extract_draw_btn.click(
            extract_mask_draw,
            inputs=[ref_editor],
            outputs=[mask_state, mask_display, mask_overlay, mask_status],
        )

        # SAM: auto centre fallback
        def _auto_from_editor(ed):
            bg = ed.get("background") if ed else None
            if bg is None:
                return None, None, None, "⚠️  Upload a reference image first"
            ref_np = np.asarray(bg)
            return extract_mask_center(ref_np)

        auto_btn.click(
            _auto_from_editor,
            inputs=[ref_editor],
            outputs=[mask_state, mask_display, mask_overlay, mask_status],
        )

        # ── Modify reference wiring ───────────────────────────────────────────
        def _modify(ref_ed, mask, prompt, strength, seed):
            out, overlay, status = modify_reference(ref_ed, mask, prompt, strength, seed)
            return out, overlay, status, out   # last = modified_ref_state

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

        # Placement editor
        load_editor_btn.click(
            load_source_to_editor,
            inputs=[source_img],
            outputs=[placement_editor],
        )
        extract_bbox_btn.click(
            bbox_from_drawing,
            inputs=[placement_editor],
            outputs=[x1_in, y1_in, x2_in, y2_in],
        )

        # Live augmented-prompt preview
        def _domain_key(label):
            for lbl, key in DOMAIN_CHOICES:
                if lbl == label:
                    return key
            return "RR"

        def _preview_aug_prompt(base, domain_label):
            key = _domain_key(domain_label)
            if not key:
                return ""
            return augment_prompt(base.strip(), key) if base.strip() else augment_prompt("(your prompt here)", key)

        for trigger in [target_prompt, domain_dd]:
            trigger.change(
                _preview_aug_prompt,
                inputs=[target_prompt, domain_dd],
                outputs=[aug_prompt_box],
            )

        # Composite preview — uses modified ref if available, else original
        def _placement_preview_wrap(src_np, ref_ed, modified_ref, mask, x1, y1, x2, y2):
            if modified_ref is not None:
                ref_np = np.array(modified_ref)
            else:
                bg = ref_ed.get("background") if ref_ed else None
                ref_np = np.asarray(bg) if bg is not None else None
            return placement_preview(src_np, ref_np, mask, x1, y1, x2, y2)

        preview_btn.click(
            _placement_preview_wrap,
            inputs=[source_img, ref_editor, modified_ref_state, mask_state,
                    x1_in, y1_in, x2_in, y2_in],
            outputs=[comp_preview],
        )

        # Generation — uses modified ref if available, else original
        def _generate_wrap(src_np, ref_ed, modified_ref, mask, prompt, domain_label,
                           x1, y1, x2, y2,
                           steps, guidance, eta, gamma, blend, cache, cascade):
            if modified_ref is not None:
                ref_np = np.array(modified_ref)
            else:
                bg = ref_ed.get("background") if ref_ed else None
                ref_np = np.asarray(bg) if bg is not None else None
            domain_key = _domain_key(domain_label)
            return generate(src_np, ref_np, mask, prompt, domain_key,
                            x1, y1, x2, y2,
                            steps, guidance, eta, gamma, blend, cache, cascade)

        generate_btn.click(
            _generate_wrap,
            inputs=[
                source_img, ref_editor, modified_ref_state, mask_state,
                target_prompt, domain_dd,
                x1_in, y1_in, x2_in, y2_in,
                num_steps, guidance, eta, gamma, blend_ratio,
                use_cache, cascade_num,
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
    app = build_app()
    # Pre-fill the weights box default from --weights arg
    app.launch(server_port=args.port, share=args.share)
