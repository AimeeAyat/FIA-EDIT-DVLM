"""
5-image end-to-end test for Flux & Key.

Tests all three steps sequentially on 5 PIE-Bench samples.
Saves outputs to test_output/flux_key/ for visual inspection.

Usage:
    python run_flux_key_test.py

For each sample you need a reference image in test_refs/.
Place reference images as: test_refs/<key>_ref.jpg
e.g.  test_refs/000000000002_ref.jpg  (a reference dog image for the cat->dog edit)

If a reference image is not found, the source image itself is used as a fallback
(degenerate test: verifies pipeline runs without checking reference injection quality).
"""

import os, json, argparse
import numpy as np
import torch
from PIL import Image

# ── 5 test samples from PIE-Bench ────────────────────────────────────────────
# key | editing_type | original_prompt | editing_prompt | what to use as reference
TEST_CASES = [
    {
        "key":       "000000000002",
        "text_ref":  "a dog",                        # Grounding DINO query on ref image
        "prompt":    "a dog sitting on a wooden chair",  # text for FLUX text branch
        "tau":       0.75,
        "note":      "cat → dog  (change object)",
    },
    {
        "key":       "000000000001",
        "text_ref":  "a square cake",
        "prompt":    "a square cake with orange frosting on a wooden plate",
        "tau":       0.5,
        "note":      "round cake → square cake  (change attribute)",
    },
    {
        "key":       "000000000003",
        "text_ref":  "a dog",
        "prompt":    "blue light, a black and white dog is playing with a flower",
        "tau":       0.75,
        "note":      "cat → dog  (change object)",
    },
    {
        "key":       "000000000004",
        "text_ref":  "a silver cat sculpture",
        "prompt":    "a silver cat sculpture sitting next to a mirror",
        "tau":       0.6,
        "note":      "cat → silver cat sculpture  (change attribute)",
    },
    {
        "key":       "000000000000",
        "text_ref":  "a rusty bicycle",
        "prompt":    "a slanted rusty mountain bicycle on the road",
        "tau":       0.4,
        "note":      "bicycle → rusty bicycle  (change attribute)",
    },
]

DEVICE     = "cuda"
JSON_PATH  = "data/pie_bench/mapping_file.json"
IMG_DIR    = "data/pie_bench/annotation_images"
REFS_DIR   = "test_refs"
OUT_BASE   = "test_output/flux_key"

# ── helpers ───────────────────────────────────────────────────────────────────

def prepare_test_dirs():
    os.makedirs(REFS_DIR, exist_ok=True)
    os.makedirs(OUT_BASE, exist_ok=True)
    print(f"\nPlace reference images in: {os.path.abspath(REFS_DIR)}/")
    print("Expected filenames:")
    for tc in TEST_CASES:
        print(f"  {tc['key']}_ref.jpg  ({tc['note']})")
    print()


def get_source_path(key: str) -> str:
    with open(JSON_PATH) as f:
        data = json.load(f)
    return os.path.join(IMG_DIR, data[key]["image_path"])


def get_mask_png(key: str) -> str:
    """Convert RLE mask to PNG and cache it. Returns path to saved PNG."""
    from models.flux_key.step2_features import load_mask_from_json
    out_path = os.path.join(OUT_BASE, f"{key}_mask.png")
    if not os.path.exists(out_path):
        mask = load_mask_from_json(JSON_PATH, key)
        Image.fromarray((mask * 255).astype(np.uint8)).save(out_path)
    return out_path


def check_ref(key: str) -> str:
    """Return reference image path, or None if not found."""
    ref_path = os.path.join(REFS_DIR, f"{key}_ref.jpg")
    if os.path.exists(ref_path):
        return ref_path
    ref_path_png = os.path.join(REFS_DIR, f"{key}_ref.png")
    if os.path.exists(ref_path_png):
        return ref_path_png
    return None


# ── Step 1 ────────────────────────────────────────────────────────────────────

def run_step1(tc: dict, out_dir: str, processor, model_gdino):
    from models.flux_key.step1_extract import extract_and_align, mask_bbox

    ref_path = check_ref(tc["key"])
    if ref_path is None:
        print(f"  [WARN] No reference image for {tc['key']}, using source as fallback")
        src_path = get_source_path(tc["key"])
        ref_image = Image.open(src_path).convert("RGB")
    else:
        ref_image = Image.open(ref_path).convert("RGB")

    mask_png = get_mask_png(tc["key"])
    mask_arr = np.array(Image.open(mask_png).convert("L"))
    target_bbox = mask_bbox(mask_arr)

    aligned, det_box, score = extract_and_align(
        ref_image, tc["text_ref"], target_bbox,
        processor, model_gdino, device=DEVICE,
        src_mask_bbox=target_bbox   # fallback: use mask bbox if DINO fails
    )

    aligned_path = os.path.join(out_dir, "ref_aligned.png")
    aligned.save(aligned_path)
    print(f"  Step1 OK  bbox={det_box}  score={score:.3f}  "
          f"aligned_size={aligned.size}")
    return aligned_path


# ── Step 2 ────────────────────────────────────────────────────────────────────

def run_step2(tc: dict, ref_aligned_path: str, out_dir: str,
              t5, clip_enc, ae, model_flux):
    from models.flux_key.step2_features import (
        load_mask_from_json, mask_to_token_indices, extract_features
    )
    from flux.sampling import prepare

    mask_np = load_mask_from_json(JSON_PATH, tc["key"])
    mask_indices = mask_to_token_indices(mask_np).to(DEVICE)

    def load_img(path):
        img = Image.open(path).convert("RGB").resize((512, 512))
        arr = np.array(img).astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    img_A = load_img(get_source_path(tc["key"]))
    img_B = load_img(ref_aligned_path)

    print(f"  Extracting source features...")
    feat_src, inp_A = extract_features(img_A, t5, clip_enc, ae, model_flux,
                                       prompt="", device=DEVICE)
    print(f"  Extracting reference features...")
    feat_ref, _     = extract_features(img_B, t5, clip_enc, ae, model_flux,
                                       prompt=tc["prompt"], device=DEVICE,
                                       mask_indices=mask_indices)

    feat_path = os.path.join(out_dir, "features.pt")
    torch.save({
        "feat_src":     feat_src,
        "feat_ref":     feat_ref,
        "mask_indices": mask_indices,
        "inp_A":        inp_A,
    }, feat_path)
    print(f"  Step2 OK  mask_tokens={len(mask_indices)}")
    return feat_path


# ── Step 3 ────────────────────────────────────────────────────────────────────

def run_step3(tc: dict, feat_path: str, out_dir: str, ae, model_flux,
              fg_inject: bool = True, freq_2d: bool = False):
    from models.flux_key.step3_edit import run_edit
    run_edit(
        source_path=get_source_path(tc["key"]),
        features_path=feat_path,
        json_path=JSON_PATH,
        key=tc["key"],
        tau=tc["tau"],
        alpha_max=0.5,
        alpha_min=0.1,
        prompt=tc["prompt"],
        num_steps=28,
        guidance=3.5,
        device=DEVICE,
        out_dir=out_dir,
        fg_inject=fg_inject,
        freq_2d=freq_2d,
        ae=ae,
        model=model_flux,
    )
    print(f"  Step3(v1) OK  output -> {out_dir}/{tc['key']}_edited.png")


def run_step3_v2(tc: dict, ref_aligned_path: str, out_dir: str,
                 ae, model_flux, t5, clip_enc):
    """FlowEdit + noise-consistent reference injection (v2)."""
    from models.flux_key.step3_flowedit import run_edit_v2
    out_dir_v2 = os.path.join(OUT_BASE + "_v2", tc["key"])
    os.makedirs(out_dir_v2, exist_ok=True)
    run_edit_v2(
        source_path=get_source_path(tc["key"]),
        ref_aligned_path=ref_aligned_path,
        json_path=JSON_PATH,
        key=tc["key"],
        prompt_edit=tc["prompt"],
        prompt_src="",
        num_steps=28,
        guidance=3.5,
        alpha_max=0.3,
        alpha_min=0.05,
        device=DEVICE,
        out_dir=out_dir_v2,
        ae=ae,
        model=model_flux,
        t5=t5,
        clip_enc=clip_enc,
    )
    print(f"  Step3(v2) OK  output -> {out_dir_v2}/{tc['key']}_edited.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    parser = argparse.ArgumentParser()
    parser.add_argument("--fg_inject", action="store_true", default=False,
                        help="Enable foreground reference KV injection (default: off)")
    parser.add_argument("--step3_only", action="store_true", default=False,
                        help="Skip step1/2 (reuse saved features.pt)")
    parser.add_argument("--freq_2d", action="store_true", default=False,
                        help="Use 2-D spatial FFT for background lock (default: 1-D)")
    parser.add_argument("--key", default=None,
                        help="Run only this PIE-Bench key (e.g. 000000000002)")
    parser.add_argument("--tau", type=float, default=None,
                        help="Override tau for all test cases")
    parser.add_argument("--v2", action="store_true", default=False,
                        help="Use v2 FlowEdit pipeline (noise-consistent reference injection)")
    args = parser.parse_args()

    prepare_test_dirs()
    fft_mode = "2d" if args.freq_2d else "1d"
    mode = ("bg+fg" if args.fg_inject else "bg_only") + f"_fft{fft_mode}"
    print(f"Mode: {mode}  (fg_inject={args.fg_inject})")

    if not args.step3_only:
        print("Loading Grounding DINO...")
        from models.flux_key.step1_extract import load_gdino
        gdino_proc, gdino_model = load_gdino(DEVICE)
    else:
        gdino_proc = gdino_model = None

    from flux.util import load_t5, load_clip, load_ae, load_flow_model
    # v2 always needs T5+CLIP (used in FlowEdit per-step forward passes)
    if args.step3_only and not args.v2:
        print("Loading AE + FLUX transformer (step3-only v1, skipping T5/CLIP)...")
        t5 = clip_enc = None
    else:
        print("Loading FLUX models (T5, CLIP, AE, transformer)...")
        t5       = load_t5(DEVICE, max_length=512)
        clip_enc = load_clip(DEVICE)
    ae    = load_ae("flux-dev", DEVICE)
    model = load_flow_model("flux-dev", device=DEVICE)
    model.eval()

    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    cases = [tc for tc in TEST_CASES if args.key is None or tc["key"] == args.key]
    if args.key and not cases:
        print(f"Key {args.key} not found in TEST_CASES")
        return

    results = []
    for i, tc in enumerate(cases):
        tc = dict(tc)  # copy so we can override tau
        if args.tau is not None:
            tc["tau"] = args.tau
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(cases)}] {tc['key']}  |  {tc['note']}  tau={tc['tau']}")
        out_dir = os.path.join(OUT_BASE, tc["key"])
        os.makedirs(out_dir, exist_ok=True)

        try:
            # ── Step 1: get aligned reference (always needed) ──────────────
            if args.step3_only and os.path.exists(os.path.join(out_dir, "ref_aligned.png")):
                ref_aligned = os.path.join(out_dir, "ref_aligned.png")
            else:
                ref_aligned = run_step1(tc, out_dir, gdino_proc, gdino_model)

            if args.v2:
                # v2: FlowEdit — no features.pt needed
                run_step3_v2(tc, ref_aligned, out_dir, ae, model, t5, clip_enc)
            else:
                # v1: SSI + KV injection
                if args.step3_only:
                    feat_path = os.path.join(out_dir, "features.pt")
                    if not os.path.exists(feat_path):
                        raise FileNotFoundError(f"features.pt not found: {feat_path}")
                else:
                    feat_path = run_step2(tc, ref_aligned, out_dir,
                                          t5, clip_enc, ae, model)
                run_step3(tc, feat_path, out_dir, ae, model,
                          fg_inject=args.fg_inject, freq_2d=args.freq_2d)
            status = "OK"
        except Exception as e:
            import traceback
            traceback.print_exc()
            status = f"FAILED: {e}"

        results.append({"key": tc["key"], "note": tc["note"], "status": status})
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"SUMMARY  [{mode}]")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['key']}  {r['note']:<35}  {r['status']}")
    print(f"\nOutputs saved to: {os.path.abspath(OUT_BASE)}/")
    print("Open <key>_comparison.png for side-by-side source vs. edited.")


if __name__ == "__main__":
    main()
