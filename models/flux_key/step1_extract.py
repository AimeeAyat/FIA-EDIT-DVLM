"""
Step 1: Reference object extraction.

Given a reference image I_B and a text query describing the object of interest,
this module:
  1. Runs Grounding DINO (via transformers) to detect the object bounding box.
  2. Crops and alpha-masks the object region.
  3. Resizes the crop to match the spatial extent of the edit mask M_A in I_A.

No weights from the original KV-Edit or FIA-Edit pipelines are used here.
Run standalone for testing:
    python -m models.flux_key.step1_extract \
        --ref_image path/to/ref.jpg \
        --text "a dog" \
        --source_mask path/to/mask.png \
        --out_dir test_output/step1/
"""

import argparse
import os
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


GDINO_MODEL = "IDEA-Research/grounding-dino-base"


def load_gdino(device="cuda"):
    print("Loading Grounding DINO...")
    processor = AutoProcessor.from_pretrained(GDINO_MODEL)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_MODEL).to(device)
    model.eval()
    return processor, model


def detect_object(image: Image.Image, text: str, processor, model,
                  box_threshold: float = 0.35,
                  text_threshold: float = 0.25,
                  device: str = "cuda"):
    """
    Run Grounding DINO on image, return best bounding box [x0, y0, x1, y1]
    in pixel coordinates, or None if nothing detected above threshold.
    """
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],  # (H, W)
    )[0]

    if len(results["boxes"]) == 0:
        return None, None

    # pick highest-confidence box
    best_idx = results["scores"].argmax().item()
    box = results["boxes"][best_idx].cpu().numpy().astype(int)  # [x0,y0,x1,y1]
    score = results["scores"][best_idx].item()
    return box, score


def extract_and_align(ref_image: Image.Image,
                      text: str,
                      target_bbox,          # [x0, y0, x1, y1] of M_A in source image
                      processor, model,
                      device: str = "cuda",
                      box_threshold: float = 0.35,
                      src_mask_bbox=None):  # fallback crop region if detection fails
    """
    Full Step 1:
      - Detect object in ref_image with text query.
      - Crop bounding box region.
      - Resize crop to match target_bbox dimensions.
      - Return: aligned_crop (PIL Image, RGB), detection_box, score.
    If detection fails, falls back to src_mask_bbox crop (if provided).
    """
    box, score = detect_object(ref_image, text, processor, model,
                               box_threshold=box_threshold, device=device)

    if box is None:
        if src_mask_bbox is not None:
            print(f"  [WARN] No detection for '{text}', using source mask bbox as fallback.")
            box = src_mask_bbox
            score = 0.0
        else:
            print(f"  [WARN] No detection for '{text}', using full image as fallback.")
            w_img, h_img = ref_image.size
            box = [0, 0, w_img, h_img]
            score = 0.0

    x0, y0, x1, y1 = box
    # clamp to image bounds
    w_img, h_img = ref_image.size
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w_img, x1), min(h_img, y1)

    crop = ref_image.crop((x0, y0, x1, y1))

    # target size from M_A bounding box
    tx0, ty0, tx1, ty1 = target_bbox
    target_w = max(1, tx1 - tx0)
    target_h = max(1, ty1 - ty0)

    aligned = crop.resize((target_w, target_h), Image.LANCZOS)
    return aligned, box, score


def mask_bbox(mask_array: np.ndarray):
    """Return tight [x0, y0, x1, y1] bounding box around non-zero region."""
    rows = np.any(mask_array > 0, axis=1)
    cols = np.any(mask_array > 0, axis=0)
    if not rows.any():
        h, w = mask_array.shape[:2]
        return [0, 0, w, h]
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    return [int(x0), int(y0), int(x1) + 1, int(y1) + 1]


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Step 1: Reference object extraction")
    p.add_argument("--ref_image",     required=True, help="Path to reference image I_B")
    p.add_argument("--text",          required=True, help="Text query for object (e.g. 'a dog')")
    p.add_argument("--source_mask",   required=True, help="Path to edit mask M_A (PNG, 0/255)")
    p.add_argument("--out_dir",       default="test_output/step1/")
    p.add_argument("--box_threshold", type=float, default=0.35)
    p.add_argument("--device",        default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ref_image = Image.open(args.ref_image).convert("RGB")
    mask = np.array(Image.open(args.source_mask).convert("L"))
    target_bbox = mask_bbox(mask)
    print(f"Source mask bbox: {target_bbox}")

    processor, model = load_gdino(args.device)

    aligned, det_box, score = extract_and_align(
        ref_image, args.text, target_bbox, processor, model,
        device=args.device, box_threshold=args.box_threshold,
        src_mask_bbox=target_bbox   # use mask bbox as fallback if detection fails
    )

    print(f"Detected '{args.text}' at {det_box} (score={score:.3f})")
    print(f"Aligned crop size: {aligned.size}")

    # save
    ref_out = os.path.join(args.out_dir, "ref_detected.jpg")
    aligned_out = os.path.join(args.out_dir, "ref_aligned.png")

    # draw box on ref for visual inspection
    from PIL import ImageDraw
    ref_vis = ref_image.copy()
    draw = ImageDraw.Draw(ref_vis)
    x0, y0, x1, y1 = det_box
    draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
    ref_vis.save(ref_out)
    aligned.save(aligned_out)

    print(f"Saved: {ref_out}")
    print(f"Saved: {aligned_out}")


if __name__ == "__main__":
    main()
