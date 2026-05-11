"""
Compare our KV-Edit results against paper Table 1.
Run after evaluation completes:
    python compare_results.py --result_dir output/step_28_skip_4_1.5_5.5
"""
import argparse
import os
import pandas as pd

# Paper Table 1 values (already scaled as shown in paper headers)
PAPER = {
    "VAE*":       {"HPS":24.93, "AS":6.37, "PSNR":37.65, "LPIPS":7.93,  "MSE":3.86,  "CLIP":19.69},
    "P2P":        {"HPS":25.40, "AS":6.27, "PSNR":17.86, "LPIPS":208.43,"MSE":219.22,"CLIP":22.24},
    "MasaCtrl":   {"HPS":23.46, "AS":5.91, "PSNR":22.20, "LPIPS":105.74,"MSE":86.15, "CLIP":20.83},
    "RF Inv.":    {"HPS":27.99, "AS":6.74, "PSNR":20.20, "LPIPS":179.73,"MSE":139.85,"CLIP":21.71},
    "RF Edit":    {"HPS":27.60, "AS":6.56, "PSNR":24.44, "LPIPS":113.20,"MSE":56.26, "CLIP":22.08},
    "BrushEdit":  {"HPS":25.81, "AS":6.17, "PSNR":32.16, "LPIPS":17.22, "MSE":8.46,  "CLIP":22.44},
    "FLUX Fill":  {"HPS":25.76, "AS":6.31, "PSNR":32.53, "LPIPS":25.59, "MSE":8.55,  "CLIP":22.40},
    "Ours":       {"HPS":27.21, "AS":6.49, "PSNR":35.87, "LPIPS":9.92,  "MSE":4.69,  "CLIP":22.39},
    "Ours+NS+RI": {"HPS":28.05, "AS":6.40, "PSNR":33.30, "LPIPS":14.80, "MSE":7.45,  "CLIP":23.62},
}

# Scaling to match paper's display format
SCALES = {
    "HPS V2.1":              ("HPS",  100),
    "Aesthetic Score":       ("AS",   1),
    "psnr_unedit_part":      ("PSNR", 1),
    "lpips_unedit_part":     ("LPIPS",1000),
    "mse_unedit_part":       ("MSE",  10000),
    "clip_similarity_target_image": ("CLIP", 1),
}

LOWER_IS_BETTER = {"LPIPS", "MSE", "structure_distance"}


def load_overall(result_dir):
    path = os.path.join(result_dir, "metrics_overall.csv")
    df = pd.read_csv(path, header=None, names=["metric", "value"])
    return dict(zip(df["metric"], df["value"].astype(float)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", required=True)
    args = parser.parse_args()

    overall = load_overall(args.result_dir)

    # Build our row with paper scaling
    ours = {}
    for csv_col, (paper_col, scale) in SCALES.items():
        if csv_col in overall:
            ours[paper_col] = overall[csv_col] * scale
        else:
            ours[paper_col] = float("nan")

    # Print comparison table
    cols = ["HPS", "AS", "PSNR", "LPIPS", "MSE", "CLIP"]
    header = f"{'Method':<16}" + "".join(f"{c:>10}" for c in cols)
    sep = "-" * len(header)

    print("\n" + sep)
    print("  PIE-Bench Comparison  (paper scaling: HPS×100, LPIPS×1000, MSE×10000)")
    print(sep)
    print(header)
    print(sep)

    for method, vals in PAPER.items():
        row = f"{method:<16}"
        for c in cols:
            v = vals.get(c, float("nan"))
            row += f"{v:>10.2f}"
        print(row)

    print(sep)
    row = f"{'Ours (this run)':<16}"
    for c in cols:
        v = ours.get(c, float("nan"))
        row += f"{v:>10.2f}"
    print(row)
    print(sep)

    # Delta vs paper "Ours"
    print("\nDelta vs paper 'Ours' row (positive = we do better):")
    for c in cols:
        ours_val = ours.get(c, float("nan"))
        paper_val = PAPER["Ours"].get(c, float("nan"))
        delta = (paper_val - ours_val) if c in LOWER_IS_BETTER else (ours_val - paper_val)
        sign = "+" if delta >= 0 else ""
        direction = "^ we better" if delta >= 0 else "v paper better"
        print(f"  {c:<8}: {sign}{delta:+.3f}  ({direction})")

    # Also print raw overall values
    print("\nRaw overall values (unscaled):")
    for k, v in overall.items():
        print(f"  {k}: {v:.6f}")


if __name__ == "__main__":
    main()
