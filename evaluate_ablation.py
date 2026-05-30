#!/usr/bin/env python3
"""
evaluate_ablation.py
One grid per domain:  5 rows (conditions) × 4 columns (BG | Ref | Gen1 | Gen2)

BG and Ref are from sample 1 of that domain (shared input shown per row).
Gen1/Gen2 are the two generated outputs for that condition.

Outputs (in EEDit_baseline/):
  ablation_metrics.json
  ablation_grid_{domain}.png

Usage:
    python evaluate_ablation.py
"""

import sys, io, json, zipfile, os
from pathlib import Path
from collections import defaultdict

import torch
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

REPO_ROOT  = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT / "img_metrics"))
from calculate_ssim import calculate_ssim
from calculate_psnr import calculate_psnr
from calculate_lpips import calculate_lpips
from calculate_mse  import calculate_mse

SIZE      = 512
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
N_SAMPLES = 2

RESULT_DIR   = REPO_ROOT / "result-generation"
ABLATION_ZIP = RESULT_DIR / "EEdit_ablation.zip"
COMP_ZIP     = RESULT_DIR / "composition.zip"
OUT_DIR      = REPO_ROOT / "EEDit_baseline"

GROUPS = ["Real-Cartoon", "Real-Painting", "Real-Real", "Real-Sketch"]

CONDITION_ORDER = ["no_all", "no_kv", "no_tail_cfg", "no_prompt_aug", "ours"]
ROW_LABEL = {
    "no_all":        "No All\n(ablation)",
    "no_kv":         "No KV\nInject",
    "no_tail_cfg":   "No Tail\nCFG",
    "no_prompt_aug": "No Prompt\nAugment",
    "ours":          "Ours\n(full)",
}
ROW_LBL_COL = {c: (50, 50, 50) for c in CONDITION_ORDER}
ROW_LBL_COL["ours"] = (30, 60, 30)
ROW_LBL_FG  = {c: (210, 210, 210) for c in CONDITION_ORDER}
ROW_LBL_FG["ours"]  = (110, 240, 110)

# 4 fixed column headers
COL_HEADERS = ["Background", "Reference\n(Foreground)", "Generated\nSample 1", "Generated\nSample 2"]


# ── Metric helpers ─────────────────────────────────────────────────────────────

def load_t(data: bytes) -> torch.Tensor:
    return T.ToTensor()(Image.open(io.BytesIO(data)).convert("RGB").resize((SIZE, SIZE)))

def load_mask_t(data: bytes) -> torch.Tensor:
    return T.ToTensor()(Image.open(io.BytesIO(data)).convert("L").resize((SIZE, SIZE))).float()

def clip_score(pil, prompt, model, proc) -> float:
    inp = proc(text=[prompt], images=pil, return_tensors="pt", padding=True)
    inp = {k: v.to(DEVICE) for k, v in inp.items()}
    with torch.no_grad():
        o  = model(**inp)
        ie = o.image_embeds; ie = ie / ie.norm(p=2, dim=-1, keepdim=True)
        te = o.text_embeds;  te = te / te.norm(p=2, dim=-1, keepdim=True)
        return ((ie * te).sum(-1) * 100).item()

def compute_metrics(gen_b, bg_b, mask_b, prompt, model, proc):
    orig = load_t(bg_b); mask = load_mask_t(mask_b); edited = load_t(gen_b)
    bm   = (mask < 0.5).float()
    pil  = Image.open(io.BytesIO(gen_b)).convert("RGB").resize((SIZE, SIZE))
    return {
        "mse":   calculate_mse(orig * bm, edited * bm)["value"],
        "psnr":  calculate_psnr(orig * bm, edited * bm)["value"],
        "ssim":  calculate_ssim(orig * bm, edited * bm)["value"],
        "lpips": calculate_lpips(orig * bm, edited * bm, DEVICE)["value"][0],
        "clip":  clip_score(pil, prompt, model, proc),
    }


# ── Grid builder  (5 rows × 4 cols) ──────────────────────────────────────────

THUMB    = 220      # image thumbnail size
LABEL_W  = 115      # left label column width
COL_HDR_H = 42     # column header height (top)
ROW_HDR_H = 36     # domain-name header height (very top)
PAD      = 4
N_COLS   = 4        # BG | Ref | Gen1 | Gen2
BG_COL   = (22, 22, 22)
HDR_COL  = (40, 40, 40)
COL_HDR_BG = (33, 33, 50)

def _font(sz):
    for name in ["arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"]:
        try:
            return ImageFont.truetype(name, sz)
        except Exception:
            pass
    return ImageFont.load_default()

F_SM = _font(11)
F_LG = _font(13)


def build_grid(row_data: dict, bg_pil: Image.Image, ref_pil: Image.Image,
               domain: str) -> Image.Image:
    """
    row_data: {cond: [gen1_PIL, gen2_PIL]}
    bg_pil, ref_pil: shared input images for this domain (from sample 1).
    Grid: 5 rows × 4 cols = [BG | Ref | Gen1 | Gen2]
    """
    n_rows = len(CONDITION_ORDER)
    w = LABEL_W + N_COLS * (THUMB + PAD) + PAD
    h = ROW_HDR_H + COL_HDR_H + n_rows * (THUMB + PAD) + PAD

    canvas = Image.new("RGB", (w, h), BG_COL)
    draw   = ImageDraw.Draw(canvas)

    # ── Domain header (very top) ──────────────────────────────────────────────
    draw.rectangle([0, 0, w, ROW_HDR_H], fill=HDR_COL)
    draw.text((6, 9), domain, font=F_LG, fill=(255, 200, 80))

    # ── Column headers ────────────────────────────────────────────────────────
    y_ch = ROW_HDR_H
    draw.rectangle([0, y_ch, w, y_ch + COL_HDR_H], fill=COL_HDR_BG)
    for c, hdr in enumerate(COL_HEADERS):
        x0   = LABEL_W + PAD + c * (THUMB + PAD)
        cx   = x0 + THUMB // 2
        lines = hdr.split("\n")
        lh   = 14
        ty   = y_ch + (COL_HDR_H - len(lines) * lh) // 2
        for line in lines:
            # rough centre
            tw = len(line) * 6
            draw.text((cx - tw // 2, ty), line, font=F_SM, fill=(200, 200, 220))
            ty += lh

    # ── Condition rows ────────────────────────────────────────────────────────
    for r, cond in enumerate(CONDITION_ORDER):
        y0      = ROW_HDR_H + COL_HDR_H + PAD + r * (THUMB + PAD)
        is_ours = cond == "ours"
        lbl_bg  = ROW_LBL_COL[cond]
        lbl_fg  = ROW_LBL_FG[cond]

        # Label cell
        draw.rectangle([0, y0, LABEL_W - PAD, y0 + THUMB], fill=lbl_bg)
        lines  = ROW_LABEL.get(cond, cond).split("\n")
        line_h = 16
        ty     = y0 + (THUMB - len(lines) * line_h) // 2
        for line in lines:
            draw.text((6, ty), line, font=F_SM, fill=lbl_fg)
            ty += line_h

        # 4 image columns
        cell_imgs = [bg_pil, ref_pil] + (row_data.get(cond) or [None, None])
        for c, img in enumerate(cell_imgs[:N_COLS]):
            x0c = LABEL_W + PAD + c * (THUMB + PAD)
            if img is not None:
                thumb = img.resize((THUMB, THUMB), Image.LANCZOS)
                canvas.paste(thumb, (x0c, y0))
                if is_ours and c >= 2:   # green border only on generated cols
                    draw.rectangle([x0c, y0, x0c + THUMB - 1, y0 + THUMB - 1],
                                   outline=(70, 200, 70), width=3)
            else:
                draw.rectangle([x0c, y0, x0c + THUMB, y0 + THUMB], fill=(55, 55, 55))
                draw.text((x0c + 8, y0 + THUMB // 2 - 6),
                          "missing", font=F_SM, fill=(140, 70, 70))

    return canvas


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}")

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE).eval()
    clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    # ── Parse ablation zip ────────────────────────────────────────────────────
    ablation: dict = defaultdict(lambda: defaultdict(dict))
    with zipfile.ZipFile(ABLATION_ZIP) as az:
        for entry in az.namelist():
            parts = entry.split("/")
            if len(parts) < 4 or not parts[-1].endswith(".png"):
                continue
            cond_key = "_".join(parts[1].split("_")[:-2])
            ablation[cond_key][parts[2]][parts[3]] = az.read(entry)
    print(f"Ablation conditions: {sorted(ablation.keys())}")

    # ── Load "ours" first N images per group ──────────────────────────────────
    for group in GROUPS:
        zp = RESULT_DIR / f"full_{group}_ours.zip"
        if not zp.exists():
            print(f"  WARNING: {zp.name} not found")
            ablation["ours"][group] = {}
            continue
        with zipfile.ZipFile(zp) as oz:
            gen = sorted(f for f in oz.namelist()
                         if f.endswith(".png") and "/generated/" in f)[:N_SAMPLES]
            ablation["ours"][group] = {Path(f).name: oz.read(f) for f in gen}

    # ── Load reference data from composition.zip ──────────────────────────────
    ref_data: dict   = {}   # {group: [(prompt, bg_b, mask_b), ...]}
    input_imgs: dict = {}   # {group: (bg_PIL, ref_PIL)}  — from sample 1

    with zipfile.ZipFile(COMP_ZIP) as rz:
        all_names = rz.namelist()
        for group in GROUPS:
            prefix = f"composition/{group}/"
            sample_dirs = sorted(set(
                f.split("/")[2] for f in all_names
                if f.startswith(prefix) and f.count("/") >= 3
            ))
            entries = []
            bg_pil = ref_pil = None

            for idx, sd in enumerate(sample_dirs[:N_SAMPLES]):
                sp    = f"{prefix}{sd}/"
                files = [f for f in all_names if f.startswith(sp) and not f.endswith("/")]

                bg_f   = [f for f in files
                          if os.path.basename(f).lower().startswith("bg")
                          and os.path.splitext(f)[1].lower() in (".png", ".jpg", ".jpeg")]
                fg_f   = [f for f in files
                          if os.path.basename(f).lower().startswith("fg")
                          and "_mask" not in os.path.basename(f).lower()
                          and os.path.splitext(f)[1].lower() in (".jpg", ".jpeg", ".png")]
                mask_f = [f for f in files if "mask_bg_fg" in os.path.basename(f)]

                if not bg_f or not mask_f:
                    continue

                bg_b   = rz.read(bg_f[0])
                mask_b = rz.read(mask_f[0])
                entries.append((sd, bg_b, mask_b))

                if idx == 0:   # use sample 1 as representative inputs for the grid
                    bg_pil  = Image.open(io.BytesIO(bg_b)).convert("RGB")
                    ref_pil = (Image.open(io.BytesIO(rz.read(fg_f[0]))).convert("RGB")
                               if fg_f else Image.new("RGB", (SIZE, SIZE), (80, 80, 80)))

            ref_data[group]   = entries
            input_imgs[group] = (bg_pil, ref_pil)

    # ── Compute metrics ───────────────────────────────────────────────────────
    all_results: dict = {}

    for cond in CONDITION_ORDER:
        all_results[cond] = {}
        for group in GROUPS:
            img_dict    = ablation.get(cond, {}).get(group, {})
            sorted_imgs = sorted(img_dict.keys())[:N_SAMPLES]
            ref_entries = ref_data.get(group, [])
            n = min(len(sorted_imgs), len(ref_entries))

            group_results = []
            for i in tqdm(range(n), desc=f"{cond}/{group}", leave=False):
                sd, bg_b, mask_b = ref_entries[i]
                m = compute_metrics(img_dict[sorted_imgs[i]], bg_b, mask_b,
                                    sd, clip_model, clip_proc)
                m["file"] = sorted_imgs[i]; m["prompt"] = sd
                group_results.append(m)

            n_r = len(group_results)
            avg = ({k: sum(r[k] for r in group_results) / n_r
                    for k in ("mse", "psnr", "ssim", "lpips", "clip")} if n_r else {})
            all_results[cond][group] = {
                "individual": group_results, "average": avg, "count": n_r
            }

    for cond in CONDITION_ORDER:
        items = [r for g in all_results[cond].values() for r in g["individual"]]
        if items:
            n = len(items)
            all_results[cond]["_overall"] = {
                k: sum(r[k] for r in items) / n
                for k in ("mse", "psnr", "ssim", "lpips", "clip")
            }

    out_json = OUT_DIR / "ablation_metrics.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nMetrics saved -> {out_json}")

    print(f"\n{'Condition':<18} {'MSE':>8} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'CLIP':>8}")
    print("-" * 62)
    for cond in CONDITION_ORDER:
        ov = all_results[cond].get("_overall", {})
        if ov:
            print(f"{cond:<18} {ov['mse']:>8.4f} {ov['psnr']:>8.2f} "
                  f"{ov['ssim']:>8.4f} {ov['lpips']:>8.4f} {ov['clip']:>8.2f}")

    # ── Build one grid per domain (5 rows × 4 cols) ───────────────────────────
    for group in GROUPS:
        bg_pil, ref_pil = input_imgs.get(group, (None, None))

        row_data: dict = {}
        for cond in CONDITION_ORDER:
            img_dict    = ablation.get(cond, {}).get(group, {})
            sorted_keys = sorted(img_dict.keys())[:N_SAMPLES]
            row_data[cond] = [
                Image.open(io.BytesIO(img_dict[k])).convert("RGB")
                for k in sorted_keys
            ]

        grid     = build_grid(row_data, bg_pil, ref_pil, group)
        out_path = OUT_DIR / f"ablation_grid_{group.replace('-', '_')}.png"
        grid.save(out_path)
        print(f"Grid saved -> {out_path}")


if __name__ == "__main__":
    main()
