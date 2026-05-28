"""
SAM smoke-test — no FLUX, no EEdit, no dataset required.

Tests:
  1. Upload reference image → draw bbox → SAM extracts foreground mask
  2. Upload background image → set placement bbox → preview composite

Launch:
    python gradio_sam_smoke.py              # local  http://localhost:7860
    python gradio_sam_smoke.py --share      # public link (needed on Colab)
"""

import argparse
import numpy as np
import torch
import gradio as gr
from PIL import Image, ImageDraw

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── SAM singleton ─────────────────────────────────────────────────────────────
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _draw_rect(img_pil, x1, y1, x2, y2, colour="red", width=3):
    out = img_pil.copy().convert("RGB")
    ImageDraw.Draw(out).rectangle([x1, y1, x2, y2], outline=colour, width=width)
    return out


def _green_overlay(img_pil, mask_np, alpha=0.45):
    img_rgba = img_pil.convert("RGBA")
    ov = Image.new("RGBA", img_rgba.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(ov)
    ys, xs = np.where(mask_np)
    for y, x in zip(ys.tolist(), xs.tolist()):
        draw.point((x, y), fill=(0, 255, 0, int(alpha * 255)))
    return Image.alpha_composite(img_rgba, ov).convert("RGB")


# ── Auto-fill bbox on upload ──────────────────────────────────────────────────

def _default_bbox(img):
    if img is None:
        return 50, 50, 200, 200
    h, w = img.shape[:2]
    return int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.9)


def _default_placement(img):
    if img is None:
        return 50, 50, 250, 400
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    bw, bh = w // 4, h // 3
    return cx - bw // 2, cy - bh // 2, cx + bw // 2, cy + bh // 2


# ── SAM extraction ────────────────────────────────────────────────────────────

def run_sam(ref_img, x1, y1, x2, y2):
    if ref_img is None:
        return None, None, None, "⚠️  Upload a reference image first."

    img_pil = Image.fromarray(ref_img).convert("RGB")
    w, h    = img_pil.size
    x1, y1, x2, y2 = max(0,int(x1)), max(0,int(y1)), min(w,int(x2)), min(h,int(y2))

    if x2 <= x1 or y2 <= y1:
        return None, None, None, "⚠️  Invalid box — x2 must be > x1 and y2 must be > y1."

    _load_sam()
    preview = _draw_rect(img_pil, x1, y1, x2, y2)

    inputs = _sam_processor(
        images=img_pil,
        input_boxes=[[[x1, y1, x2, y2]]],
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

    mask_pil    = Image.fromarray((mask_np * 255).astype(np.uint8), mode="L")
    overlay_pil = _green_overlay(img_pil, mask_np)

    status = (f"✅ SAM done — coverage {mask_np.sum()/mask_np.size*100:.1f}%  |  "
              f"IoU scores: {[f'{s:.2f}' for s in scores]}  (best={best})")
    return preview, mask_pil, overlay_pil, status


# ── Composite preview ─────────────────────────────────────────────────────────

def composite_preview(bg_img, ref_img, mask_state, px1, py1, px2, py2):
    if bg_img is None:
        return None, "⚠️  Upload a background image first."
    if ref_img is None:
        return None, "⚠️  Upload a reference image first."

    px1, py1 = max(0,int(px1)), max(0,int(py1))
    px2, py2 = int(px2), int(py2)
    bw, bh   = max(1, px2-px1), max(1, py2-py1)

    bg  = Image.fromarray(bg_img).convert("RGB")
    ref = Image.fromarray(ref_img).convert("RGB").resize((bw, bh), Image.LANCZOS)

    if mask_state is not None:
        msk = mask_state.resize((bw, bh), Image.LANCZOS)
    else:
        msk = Image.new("L", (bw, bh), 255)   # no mask yet → paste full ref

    comp = bg.copy()
    comp.paste(ref, (px1, py1), msk)
    ImageDraw.Draw(comp).rectangle([px1, py1, px2, py2], outline="#FF4444", width=3)

    status = (f"✅ Preview ready — placement box ({px1},{py1})→({px2},{py2})  "
              + ("(mask applied)" if mask_state is not None else "(no mask yet — run SAM first)"))
    return comp, status


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(title="SAM Smoke Test") as demo:
        gr.Markdown(
            "## SAM Smoke Test — Foreground Selection + Placement Preview\n"
            "**Left:** upload reference, draw SAM box → extract mask.  \n"
            "**Right:** upload background, set placement box → preview composite."
        )

        mask_state = gr.State(None)   # holds the extracted PIL mask

        with gr.Row():

            # ── LEFT: Reference + SAM ────────────────────────────────────────
            with gr.Column():
                gr.Markdown("### ① Reference Image — extract foreground mask")
                ref_img = gr.Image(label="Reference image", type="numpy")

                gr.Markdown("**SAM bounding box** around the object to segment")
                with gr.Row():
                    rx1 = gr.Number(label="x1", value=50,  precision=0)
                    ry1 = gr.Number(label="y1", value=50,  precision=0)
                with gr.Row():
                    rx2 = gr.Number(label="x2", value=200, precision=0)
                    ry2 = gr.Number(label="y2", value=200, precision=0)

                sam_btn    = gr.Button("Extract Mask", variant="primary")
                sam_status = gr.Textbox(label="SAM status", interactive=False)

                with gr.Row():
                    out_preview = gr.Image(label="Box preview",        height=200)
                    out_mask    = gr.Image(label="Mask (B&W)",          height=200)
                    out_overlay = gr.Image(label="Overlay (green=FG)", height=200)

            # ── RIGHT: Background + Placement ────────────────────────────────
            with gr.Column():
                gr.Markdown("### ② Background Image — set placement region")
                bg_img = gr.Image(label="Background / source image", type="numpy")

                gr.Markdown("**Placement bounding box** — where the object will be pasted")
                with gr.Row():
                    px1 = gr.Number(label="x1", value=50,  precision=0)
                    py1 = gr.Number(label="y1", value=50,  precision=0)
                with gr.Row():
                    px2 = gr.Number(label="x2", value=250, precision=0)
                    py2 = gr.Number(label="y2", value=400, precision=0)

                comp_btn    = gr.Button("Preview Composite", variant="primary")
                comp_status = gr.Textbox(label="Composite status", interactive=False)
                comp_out    = gr.Image(label="Composite preview (red box = placement)", height=340)

        # ── Event wiring ──────────────────────────────────────────────────────

        # Auto-fill SAM bbox when reference is uploaded
        ref_img.upload(_default_bbox, inputs=ref_img, outputs=[rx1, ry1, rx2, ry2])

        # Auto-fill placement bbox when background is uploaded
        bg_img.upload(_default_placement, inputs=bg_img, outputs=[px1, py1, px2, py2])

        # SAM extraction — also stores mask in state
        def _run_and_store(ref, x1, y1, x2, y2):
            preview, mask, overlay, status = run_sam(ref, x1, y1, x2, y2)
            return preview, mask, overlay, status, mask   # last = mask_state

        sam_btn.click(
            _run_and_store,
            inputs=[ref_img, rx1, ry1, rx2, ry2],
            outputs=[out_preview, out_mask, out_overlay, sam_status, mask_state],
        )

        # Composite preview
        comp_btn.click(
            composite_preview,
            inputs=[bg_img, ref_img, mask_state, px1, py1, px2, py2],
            outputs=[comp_out, comp_status],
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true",
                        help="Public Gradio link (required on Colab)")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    build_ui().launch(share=args.share, server_port=args.port)
