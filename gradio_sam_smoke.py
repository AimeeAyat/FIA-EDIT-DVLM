"""
SAM smoke-test — no FLUX, no EEdit, no dataset required.

Tests:
  LEFT  — draw a rectangle on the reference image  → SAM extracts the mask
           (or type a text label; without a drawing SAM uses the image centre)
  RIGHT — draw a rectangle on the background image → composite preview

Launch:
    python gradio_sam_smoke.py              # local  http://localhost:7860
    python gradio_sam_smoke.py --share      # public link (needed on Colab)
"""

import argparse
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import gradio as gr
from PIL import Image, ImageDraw

from dvlm.prompt_utils import augment_prompt, get_negative_prompt

DOMAIN_CHOICES = [
    ("Real → Cartoon  (RC)", "RC"),
    ("Real → Painting (RP)", "RP"),
    ("Real → Sketch   (RS)", "RS"),
    ("Real → Real     (RR)", "RR"),
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_sam_model     = None
_sam_processor = None


def _load_sam():
    global _sam_model, _sam_processor
    if _sam_model is not None:
        return
    from transformers import SamModel, SamProcessor
    print("Loading SAM (facebook/sam-vit-base) …")
    _sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    _sam_model     = SamModel.from_pretrained("facebook/sam-vit-base").eval().to(DEVICE)
    print("SAM loaded on", DEVICE)


# ── Extract bbox from a painted ImageEditor layer ─────────────────────────────

def _bbox_from_layer(editor_val):
    """Return (x1,y1,x2,y2) bounding box of whatever the user painted, or None."""
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
    return int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1])


def _get_bg_image(editor_val):
    """Pull the background (uploaded) image from an ImageEditor value."""
    if editor_val is None:
        return None
    bg = editor_val.get("background")
    if bg is None:
        return None
    return Image.fromarray(np.asarray(bg)).convert("RGB")


def _green_overlay(img_pil, mask_np, alpha=0.45):
    img_rgba = img_pil.convert("RGBA")
    ov = Image.new("RGBA", img_rgba.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(ov)
    ys, xs = np.where(mask_np)
    for y, x in zip(ys.tolist(), xs.tolist()):
        draw.point((x, y), fill=(0, 255, 0, int(alpha * 255)))
    return Image.alpha_composite(img_rgba, ov).convert("RGB")


# ── SAM extraction ────────────────────────────────────────────────────────────

def run_sam(ref_editor, ref_text):
    """
    Draw a rectangle on the reference image → SAM uses it as bbox prompt.
    If no drawing, falls back to the image centre point.
    Text label is shown in the status but does not change SAM behaviour
    (full text-guided segmentation needs Grounded SAM).
    """
    img_pil = _get_bg_image(ref_editor)
    if img_pil is None:
        return None, None, None, None, "⚠️  Upload a reference image first (use the left panel)."

    _load_sam()

    bbox = _bbox_from_layer(ref_editor)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        prompt_desc = f"bbox ({x1},{y1})→({x2},{y2})"
        inputs = _sam_processor(
            images=img_pil, input_boxes=[[[x1, y1, x2, y2]]], return_tensors="pt"
        ).to(DEVICE)
    else:
        cx, cy = img_pil.width // 2, img_pil.height // 2
        prompt_desc = f"centre point ({cx},{cy}) — draw a rectangle for better results"
        inputs = _sam_processor(
            images=img_pil,
            input_points=[[[cx, cy]]],
            input_labels=[[1]],
            return_tensors="pt",
        ).to(DEVICE)

    with torch.no_grad():
        out = _sam_model(**inputs)

    masks = _sam_processor.image_processor.post_process_masks(
        out.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )
    scores  = out.iou_scores[0, 0].cpu().tolist()
    best    = int(np.argmax(scores))
    mask_np = masks[0][0, best].numpy().astype(bool)

    mask_pil    = Image.fromarray((mask_np * 255).astype(np.uint8), "L")
    overlay_pil = _green_overlay(img_pil, mask_np)

    label   = f' — label: "{ref_text}"' if ref_text and ref_text.strip() else ""
    status  = (f"✅ SAM done{label} | prompt: {prompt_desc} | "
               f"coverage {mask_np.sum()/mask_np.size*100:.1f}% | "
               f"IoU {[f'{s:.2f}' for s in scores]} (best={best})")
    return img_pil, mask_pil, overlay_pil, mask_pil, status


# ── Composite preview ─────────────────────────────────────────────────────────

def composite_preview(bg_editor, ref_editor, mask_state):
    """
    Draw a rectangle on the background image → that's where the reference is placed.
    """
    bg_pil  = _get_bg_image(bg_editor)
    ref_pil = _get_bg_image(ref_editor)

    if bg_pil is None:
        return None, "⚠️  Upload a background image first (use the right panel)."
    if ref_pil is None:
        return None, "⚠️  Upload a reference image first (use the left panel)."

    bbox = _bbox_from_layer(bg_editor)
    if bbox is None:
        return None, "⚠️  Draw a rectangle on the background to set the placement region."

    x1, y1, x2, y2 = bbox
    bw, bh = max(1, x2-x1), max(1, y2-y1)

    ref_r = ref_pil.resize((bw, bh), Image.LANCZOS)
    msk   = mask_state.resize((bw, bh), Image.LANCZOS) if mask_state is not None \
            else Image.new("L", (bw, bh), 255)

    comp  = bg_pil.copy()
    comp.paste(ref_r, (x1, y1), msk)
    ImageDraw.Draw(comp).rectangle([x1, y1, x2, y2], outline="#FF4444", width=3)

    note   = "(mask applied)" if mask_state is not None else "(no mask — run Extract Mask first)"
    status = f"✅ Composite ready — placement ({x1},{y1})→({x2},{y2}) {note}"
    return comp, status


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(title="SAM Smoke Test") as demo:
        gr.Markdown(
            "## SAM Smoke Test — Draw to select, no coordinates needed\n"
            "**Left:** paint over the object in the reference image → Extract Mask.  \n"
            "**Right:** paint the area on the background where the object should go → Preview Composite."
        )

        mask_state = gr.State(None)

        # ── Prompt section ────────────────────────────────────────────────────
        with gr.Row():
            domain_dd = gr.Dropdown(
                label="Domain (scene style)",
                choices=[c[0] for c in DOMAIN_CHOICES],
                value=DOMAIN_CHOICES[0][0],
                scale=1,
            )
            base_prompt = gr.Textbox(
                label="Base prompt",
                placeholder="e.g. 'a sheep in the forest'",
                scale=2,
            )
        aug_pos = gr.Textbox(label="Augmented positive prompt (what FLUX sees)", interactive=False, lines=3)
        aug_neg = gr.Textbox(label="Negative prompt",                            interactive=False, lines=2)

        def _domain_key(label):
            for lbl, key in DOMAIN_CHOICES:
                if lbl == label:
                    return key
            return "RR"

        def _update_prompts(base, domain_label):
            key = _domain_key(domain_label)
            pos = augment_prompt(base.strip(), key) if base.strip() else augment_prompt("(your prompt here)", key)
            neg = get_negative_prompt(key)
            return pos, neg

        for trigger in [base_prompt, domain_dd]:
            trigger.change(_update_prompts, inputs=[base_prompt, domain_dd], outputs=[aug_pos, aug_neg])

        with gr.Row():

            # ── LEFT: Reference + SAM ─────────────────────────────────────────
            with gr.Column():
                gr.Markdown("### ① Reference — draw around the object")
                ref_editor = gr.ImageEditor(
                    label="Upload reference, then draw a rectangle around the object",
                    type="numpy",
                    brush=gr.Brush(default_size=8, colors=["#FF4444"]),
                    height=320,
                )
                ref_text = gr.Textbox(
                    label="Object label (optional — e.g. 'sheep')",
                    placeholder="Describe what you're extracting…",
                    lines=1,
                )
                sam_btn    = gr.Button("Extract Mask", variant="primary")
                sam_status = gr.Textbox(label="Status", interactive=False, lines=2)

                with gr.Row():
                    out_orig    = gr.Image(label="Reference",          height=180)
                    out_mask    = gr.Image(label="Mask (B&W)",          height=180)
                    out_overlay = gr.Image(label="Overlay (green=FG)", height=180)

            # ── RIGHT: Background + Placement ────────────────────────────────
            with gr.Column():
                gr.Markdown("### ② Background — draw the placement region")
                bg_editor = gr.ImageEditor(
                    label="Upload background, then draw where the object should appear",
                    type="numpy",
                    brush=gr.Brush(default_size=8, colors=["#4488FF"]),
                    height=320,
                )
                comp_btn    = gr.Button("Preview Composite", variant="primary")
                comp_status = gr.Textbox(label="Status", interactive=False, lines=2)
                comp_out    = gr.Image(label="Composite (red box = placement)", height=320)

        # ── Event wiring ──────────────────────────────────────────────────────

        def _run_sam_store(ref_ed, text):
            orig, mask, overlay, mask_for_state, status = run_sam(ref_ed, text)
            return orig, mask, overlay, mask_for_state, status

        sam_btn.click(
            _run_sam_store,
            inputs=[ref_editor, ref_text],
            outputs=[out_orig, out_mask, out_overlay, mask_state, sam_status],
        )

        comp_btn.click(
            composite_preview,
            inputs=[bg_editor, ref_editor, mask_state],
            outputs=[comp_out, comp_status],
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port",  type=int, default=7860)
    args = parser.parse_args()
    build_ui().launch(share=args.share, server_port=args.port)
