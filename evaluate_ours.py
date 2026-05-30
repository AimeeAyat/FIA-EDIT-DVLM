#!/usr/bin/env python3
"""
evaluate_ours.py — Compute composition metrics for our generated results.

Reads reference data from composition.zip and generated images from
full_{group}_ours.zip files, then writes ours_composition_metrics.json
in the same format as EEDit_baseline/composition_metrics.json.

Usage:
    python evaluate_ours.py
"""

import sys, io, json, zipfile, os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT / "img_metrics"))

from calculate_ssim import calculate_ssim
from calculate_psnr import calculate_psnr
from calculate_lpips import calculate_lpips
from calculate_mse import calculate_mse

SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

RESULT_DIR = REPO_ROOT / "result-generation"
COMP_ZIP = RESULT_DIR / "composition.zip"
GROUPS = ["Real-Cartoon", "Real-Painting", "Real-Real", "Real-Sketch"]


def load_tensor(data: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(data)).convert("RGB").resize((SIZE, SIZE))
    return T.ToTensor()(img)  # [3, H, W] in [0, 1]


def load_mask_tensor(data: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(data)).convert("L").resize((SIZE, SIZE))
    return T.ToTensor()(img).float()  # [1, H, W] in [0, 1]


def clip_score(img_pil: Image.Image, prompt: str, model, processor) -> float:
    inputs = processor(text=[prompt], images=img_pil, return_tensors="pt", padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
        img_emb = out.image_embeds
        txt_emb = out.text_embeds
        img_emb = img_emb / img_emb.norm(p=2, dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(p=2, dim=-1, keepdim=True)
        return ((img_emb * txt_emb).sum(dim=-1) * 100).item()


def main():
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE).eval()
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    comp_results = {}

    with zipfile.ZipFile(COMP_ZIP) as ref_zip:
        for group in GROUPS:
            result_zip_path = RESULT_DIR / f"full_{group}_ours.zip"
            if not result_zip_path.exists():
                print(f"Skipping {group}: {result_zip_path} not found")
                continue

            with zipfile.ZipFile(result_zip_path) as res_zip:
                gen_files = sorted(
                    f for f in res_zip.namelist()
                    if f.endswith(".png") and "/generated/" in f
                )
                if not gen_files:
                    print(f"No generated PNGs found in {result_zip_path.name}")
                    continue

                ref_prefix = f"composition/{group}/"
                sample_dirs = sorted(set(
                    f.split("/")[2]
                    for f in ref_zip.namelist()
                    if f.startswith(ref_prefix) and f.count("/") >= 3
                ))

                if len(gen_files) != len(sample_dirs):
                    print(
                        f"  Warning: {group} has {len(gen_files)} generated files "
                        f"but {len(sample_dirs)} reference dirs — using min"
                    )

                n_samples = min(len(gen_files), len(sample_dirs))
                group_results = []

                for i in tqdm(range(n_samples), desc=group):
                    gen_path = gen_files[i]
                    sample_dir = sample_dirs[i]
                    prompt = sample_dir  # full dir name, e.g. "0000 a cartoon animation..."

                    sample_prefix = f"{ref_prefix}{sample_dir}/"
                    sample_entries = [f for f in ref_zip.namelist() if f.startswith(sample_prefix)]

                    bg_files = [
                        f for f in sample_entries
                        if os.path.basename(f).lower().startswith("bg")
                        and not os.path.basename(f) == ""
                        and os.path.splitext(f)[1].lower() in (".png", ".jpg", ".jpeg")
                    ]
                    mask_files = [
                        f for f in sample_entries
                        if "mask_bg_fg" in os.path.basename(f)
                    ]

                    if not bg_files or not mask_files:
                        print(f"  Missing bg or mask for {sample_dir}")
                        continue

                    orig = load_tensor(ref_zip.read(bg_files[0]))
                    comp_mask = load_mask_tensor(ref_zip.read(mask_files[0]))
                    edited = load_tensor(res_zip.read(gen_path))

                    # Background region: where the foreground was NOT placed
                    bg_mask = (comp_mask < 0.5).float()
                    orig_bg = orig * bg_mask
                    edit_bg = edited * bg_mask

                    # CLIP score on full generated image
                    img_pil = Image.open(io.BytesIO(res_zip.read(gen_path))).convert("RGB").resize((SIZE, SIZE))
                    clip_s = clip_score(img_pil, prompt, clip_model, clip_proc)

                    group_results.append({
                        "file": f"{i + 1:03d}.png",
                        "prompt": prompt,
                        "mse": calculate_mse(orig_bg, edit_bg)["value"],
                        "psnr": calculate_psnr(orig_bg, edit_bg)["value"],
                        "ssim": calculate_ssim(orig_bg, edit_bg)["value"],
                        "lpips": calculate_lpips(orig_bg, edit_bg, DEVICE)["value"][0],
                        "clip": clip_s,
                    })

                if group_results:
                    n = len(group_results)
                    avg = {
                        k: sum(r[k] for r in group_results) / n
                        for k in ("mse", "psnr", "ssim", "lpips", "clip")
                    }
                    comp_results[group] = {"individual": group_results, "average": avg, "count": n}
                    print(
                        f"{group} ({n}): "
                        f"MSE={avg['mse']:.4f}  PSNR={avg['psnr']:.2f}  "
                        f"SSIM={avg['ssim']:.4f}  LPIPS={avg['lpips']:.4f}  "
                        f"CLIP={avg['clip']:.2f}"
                    )

    all_items = [r for g in comp_results.values() for r in g["individual"]]
    overall_avg = {}
    if all_items:
        n = len(all_items)
        overall_avg = {
            k: sum(r[k] for r in all_items) / n
            for k in ("mse", "psnr", "ssim", "lpips", "clip")
        }
        print(
            f"\nOverall ({n}):  "
            f"MSE={overall_avg['mse']:.4f}  PSNR={overall_avg['psnr']:.2f}  "
            f"SSIM={overall_avg['ssim']:.4f}  LPIPS={overall_avg['lpips']:.4f}  "
            f"CLIP={overall_avg['clip']:.2f}"
        )

    out_path = REPO_ROOT / "ours_composition_metrics.json"
    with open(out_path, "w") as f:
        json.dump({"per_group": comp_results, "overall_average": overall_avg}, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
