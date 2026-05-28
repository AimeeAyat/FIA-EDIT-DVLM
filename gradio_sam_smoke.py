"""
SAM smoke-test — no FLUX, no EEdit, no dataset required.

Tests:
  1. SAM loads and produces a mask from a bounding-box prompt
  2. Rectangle drawing UI works (user can click-drag or type coordinates)

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
        return "✅ SAM already loaded"
    from transformers import SamModel, SamProcessor
    print("Loading SAM (facebook/sam-vit-base) …")
    _sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    _sam_model     = SamModel.from_pretrained("facebook/sam-vit-base").eval().to(DEVICE)
    return "✅ SAM loaded on " + DEVICE


# ── Helpers ───────────────────────────────────────────────────────────────────

def _draw_rect(img_pil, x1, y1, x2, y2, colour="red", width=3):
    out = img_pil.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    draw.rectangle([x1, y1, x2, y2], outline=colour, width=width)
    return out


def _overlay_mask(img_pil, mask_np, colour=(0, 255, 0), alpha=0.45):
    """Green semi-transparent overlay on the foreground mask."""
    img_rgba = img_pil.convert("RGBA")
    overlay  = Image.new("RGBA", img_rgba.size, (0, 0, 0, 0))
    draw     = ImageDraw.Draw(overlay)
    for y in range(mask_np.shape[0]):
        for x in range(mask_np.shape[1]):
            if mask_np[y, x]:
                draw.point((x, y), fill=colour + (int(alpha * 255),))
    return Image.alpha_composite(img_rgba, overlay).convert("RGB")


# ── Core function ──────────────────────────────────────────────────────────────

def run_sam(image, x1, y1, x2, y2):
    """
    1. Draw the rectangle on the image (preview).
    2. Run SAM with the bbox prompt.
    3. Return: preview, mask (B&W), overlay.
    """
    if image is None:
        return None, None, None, "⚠️  Upload an image first."

    img_pil = Image.fromarray(image).convert("RGB")
    w, h    = img_pil.size

    # Clamp coordinates to image bounds
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))

    if x2 <= x1 or y2 <= y1:
        return None, None, None, "⚠️  Invalid box — x2 must be > x1 and y2 must be > y1."

    preview = _draw_rect(img_pil, x1, y1, x2, y2)

    # Load SAM if needed
    msg = _load_sam()

    # SAM bbox prompt: [[x1, y1, x2, y2]]
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

    # Pick the highest-scored mask
    scores = out.iou_scores[0, 0].cpu().tolist()
    best   = int(np.argmax(scores))
    mask_np = masks[0][0, best].numpy().astype(bool)

    mask_pil    = Image.fromarray((mask_np * 255).astype(np.uint8), mode="L")
    overlay_pil = _overlay_mask(img_pil, mask_np)

    status = (f"✅ SAM done — mask coverage: "
              f"{mask_np.sum() / mask_np.size * 100:.1f}%  |  "
              f"IoU scores: {[f'{s:.2f}' for s in scores]}  (best={best})")
    return preview, mask_pil, overlay_pil, status


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(title="SAM Smoke Test") as demo:
        gr.Markdown("## SAM Foreground Mask — Smoke Test\n"
                    "Upload an image, set the bounding box around the object, "
                    "then click **Extract Mask**.")

        with gr.Row():
            with gr.Column():
                inp_image = gr.Image(label="Input image", type="numpy")

                gr.Markdown("**Bounding box** — pixel coordinates (x=left→right, y=top→bottom)")
                with gr.Row():
                    x1 = gr.Number(label="x1 (left)",   value=50,  precision=0)
                    y1 = gr.Number(label="y1 (top)",    value=50,  precision=0)
                with gr.Row():
                    x2 = gr.Number(label="x2 (right)",  value=200, precision=0)
                    y2 = gr.Number(label="y2 (bottom)", value=200, precision=0)

                btn = gr.Button("Extract Mask", variant="primary")
                status = gr.Textbox(label="Status", interactive=False)

            with gr.Column():
                out_preview = gr.Image(label="Preview (box drawn)")
                out_mask    = gr.Image(label="Mask (B&W)")
                out_overlay = gr.Image(label="Overlay (green = foreground)")

        # Auto-fill default box when image is uploaded
        def on_upload(img):
            if img is None:
                return 50, 50, 200, 200
            h, w = img.shape[:2]
            return int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.9)

        inp_image.upload(on_upload, inputs=inp_image, outputs=[x1, y1, x2, y2])

        btn.click(
            fn=run_sam,
            inputs=[inp_image, x1, y1, x2, y2],
            outputs=[out_preview, out_mask, out_overlay, status],
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true",
                        help="Create public Gradio link (required on Colab)")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    build_ui().launch(share=args.share, server_port=args.port)
