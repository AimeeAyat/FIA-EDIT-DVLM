#!/usr/bin/env python3
"""
Generate side-by-side comparison plots from result-generation/ zips.
Layout per sample: Reference (cp_bg_fg) | Mask | Paper | Ours
Outputs saved to comparisons/<category>/<NNN>.png
"""

import io
import re
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = Path("result-generation")
COMP_ZIP  = BASE / "composition.zip"
OUT_DIR   = Path("comparisons")

# category label → (papers zip, ours zip, comp subfolder, comp_index_offset)
#   offset: composition_entry_index = image_number + offset
#   Real-Cartoon/Painting/Sketch: comp 0 = img 001  → offset = -1
#   Real-Real:                    comp 1 = img 001  → offset =  0
CATEGORIES = {
    "Real-Cartoon":  ("Real-Cartoon-Papers.zip",  "full_Real-Cartoon_ours.zip",  "Real-Cartoon",  -1),
    "Real-Painting": ("Real-Painting-Papers.zip",  "full_Real-Painting_ours.zip", "Real-Painting", -1),
    "Real-Real":     ("Real-Real-papers.zip",       "full_Real-Real_ours.zip",     "Real-Real",      0),
    "Real-Sketch":   ("Real-Sketch-papers.zip",     "full_Real-Sketch_ours.zip",   "Real-Sketch",   -1),
}

COL_LABELS = ["Source", "Reference", "Mask", "EEdit", "Ours"]
IMG_SIZE   = 3      # inches per panel
DPI        = 150


# ── Helpers ───────────────────────────────────────────────────────────────────

def open_img(zf: zipfile.ZipFile, name: str) -> np.ndarray:
    with zf.open(name) as f:
        return np.array(Image.open(io.BytesIO(f.read())).convert("RGB"))


def img_index(path: str) -> int:
    stem = Path(path).stem
    digits = re.sub(r"\D", "", stem)
    return int(digits) if digits else -1


def build_index(zf: zipfile.ZipFile, filter_fn) -> dict[int, str]:
    """Map image number → zip path for all PNGs passing filter_fn."""
    result = {}
    for name in zf.namelist():
        if name.endswith(".png") and filter_fn(name):
            idx = img_index(name)
            if idx >= 0:
                result[idx] = name
    return result


def build_comp_index(zf: zipfile.ZipFile, cat: str) -> dict[int, dict[str, str]]:
    """Map composition entry index → {filename: zip_path} for a category."""
    result: dict[int, dict[str, str]] = {}
    prefix = f"composition/{cat}/"
    for name in zf.namelist():
        if not name.startswith(prefix):
            continue
        rest = name[len(prefix):]
        parts = rest.split("/")
        if len(parts) < 2 or not parts[1]:
            continue
        entry, fname = parts[0], parts[1]
        m = re.match(r"^(\d+)", entry)
        if m:
            entry_idx = int(m.group(1))
            result.setdefault(entry_idx, {})[fname] = name
    return result


def save_comparison(source: np.ndarray, ref: np.ndarray, mask: np.ndarray,
                    EEdit: np.ndarray, ours: np.ndarray,
                    out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(IMG_SIZE * 5, IMG_SIZE + 0.7))
    fig.suptitle(title, fontsize=10, y=1.01)
    for ax, img, label in zip(axes, [source, ref, mask, EEdit, ours], COL_LABELS):
        ax.imshow(img)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.axis("off")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def pick_comp_images(files: dict[str, str], zf_comp: zipfile.ZipFile):
    """Return (source_array, reference_array, mask_array) from a composition entry."""
    # source: background image bg*.png/jpg
    src_name  = next((k for k in files
                      if re.match(r"bg\w*\.(jpg|png)", k, re.I)), None)
    # foreground image: fg*.jpg/png but NOT fg*_mask.*
    ref_name  = next((k for k in files
                      if re.match(r"fg\w+\.(jpg|png)", k, re.I)
                      and "mask" not in k.lower()), None)
    mask_name = next((k for k in files if k == "mask_bg_fg.jpg"), None)

    placeholder = np.full((256, 256, 3), 180, dtype=np.uint8)
    src  = open_img(zf_comp, files[src_name])  if src_name  else placeholder
    ref  = open_img(zf_comp, files[ref_name])  if ref_name  else placeholder
    mask = open_img(zf_comp, files[mask_name]) if mask_name else placeholder
    return src, ref, mask


# ── Main ──────────────────────────────────────────────────────────────────────

def process_category(cat: str, papers_zip: str, ours_zip: str,
                     comp_cat: str, offset: int,
                     zf_comp: zipfile.ZipFile) -> int:
    print(f"\n{'='*60}")
    print(f"  {cat}")
    print(f"{'='*60}")

    comp_idx_map = build_comp_index(zf_comp, comp_cat)

    with (zipfile.ZipFile(BASE / papers_zip) as zf_paper,
          zipfile.ZipFile(BASE / ours_zip)   as zf_ours):

        paper_map = build_index(zf_paper, lambda n: n.endswith(".png"))
        ours_map  = build_index(zf_ours,  lambda n: n.endswith(".png"))

        common = sorted(set(paper_map) & set(ours_map))
        print(f"  Matched samples: {len(common)}  (paper={len(paper_map)}, ours={len(ours_map)})")

        saved = 0
        for img_num in common:
            comp_entry_idx = img_num + offset

            if comp_entry_idx not in comp_idx_map:
                print(f"  [skip] no composition entry for img {img_num:03d} "
                      f"(comp idx {comp_entry_idx})")
                continue

            src_arr, ref_arr, mask_arr = pick_comp_images(comp_idx_map[comp_entry_idx], zf_comp)
            paper_arr = open_img(zf_paper, paper_map[img_num])
            ours_arr  = open_img(zf_ours,  ours_map[img_num])

            out_path = OUT_DIR / cat / f"{img_num:03d}.png"
            save_comparison(src_arr, ref_arr, mask_arr, paper_arr, ours_arr,
                            out_path, f"{cat}  —  sample {img_num:03d}")
            saved += 1

            if saved % 20 == 0 or img_num == common[-1]:
                print(f"  [{saved}/{len(common)}] saved {out_path.name}")

    return saved


def main() -> None:
    if not COMP_ZIP.exists():
        raise FileNotFoundError(f"composition.zip not found at {COMP_ZIP}")

    with zipfile.ZipFile(COMP_ZIP) as zf_comp:
        total = 0
        for cat, (papers_zip, ours_zip, comp_cat, offset) in CATEGORIES.items():
            pz = BASE / papers_zip
            oz = BASE / ours_zip
            if not pz.exists():
                print(f"[skip] missing {pz}")
                continue
            if not oz.exists():
                print(f"[skip] missing {oz}")
                continue
            total += process_category(cat, papers_zip, ours_zip,
                                      comp_cat, offset, zf_comp)

    print(f"\nDone. {total} comparison images saved under '{OUT_DIR}/'")


if __name__ == "__main__":
    main()
