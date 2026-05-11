"""
Verification script for Table 3 (Ablation study).
Only the full model (FIJ=✓, FRI=freq) can be verified from our CSV.
The other ablation variants are not in evaluation_result_mean.csv.
"""
import pandas as pd

# Table 3, last row: FIJ=checkmark, FRI=freq (full model)
PAPER_FULL_MODEL = {
    "Distance_1e3": 10.34,
    "PSNR":         27.32,
    "LPIPS_1e3":    55.02,
    "MSE_1e4":      28.66,
    "SSIM_1e2":     89.21,
    "CLIP_Whole":   25.89,
    "CLIP_Edited":  22.82,
}

# Table 3, other ablation rows (hardcoded from paper — not in CSV)
ABLATION_VARIANTS = {
    "FIJ=✗ FRI=✗ (baseline)": {
        "Distance_1e3": 23.62, "PSNR": 23.21, "LPIPS_1e3": 93.81,
        "MSE_1e4": 69.95, "SSIM_1e2": 85.09, "CLIP_Whole": 26.78, "CLIP_Edited": 23.73,
    },
    "FIJ=✓ FRI=✗": {
        "Distance_1e3": 14.89, "PSNR": 25.59, "LPIPS_1e3": 70.18,
        "MSE_1e4": 41.74, "SSIM_1e2": 87.51, "CLIP_Whole": 26.30, "CLIP_Edited": 23.12,
    },
    "FIJ=✓ FRI=add": {
        "Distance_1e3": 16.50, "PSNR": 25.93, "LPIPS_1e3": 85.44,
        "MSE_1e4": 38.72, "SSIM_1e2": 86.51, "CLIP_Whole": 26.05, "CLIP_Edited": 22.68,
    },
}

TOLERANCE = 0.05


def load_overall(csv_path: str) -> dict:
    df = pd.read_csv(csv_path, index_col="Category")
    row = df.loc["Overall"]
    m = "FIA-Edit_SD35"
    return {
        "Distance_1e3": row[f"{m}|structure_distance"] * 1e3,
        "PSNR":         row[f"{m}|psnr_unedit_part"],
        "LPIPS_1e3":    row[f"{m}|lpips_unedit_part"] * 1e3,
        "MSE_1e4":      row[f"{m}|mse_unedit_part"] * 1e4,
        "SSIM_1e2":     row[f"{m}|ssim_unedit_part"] * 1e2,
        "CLIP_Whole":   row[f"{m}|clip_similarity_target_image"],
        "CLIP_Edited":  row[f"{m}|clip_similarity_target_image_edit_part"],
    }


def verify_full_model(computed: dict) -> None:
    header = f"{'Metric':<18} {'Computed':>10} {'Paper':>10} {'Diff':>8} {'OK?':>6}"
    print("=" * len(header))
    print("Table 3 -- Full model (FIJ=checkmark, FRI=freq) Verification")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    all_ok = True
    for key in PAPER_FULL_MODEL:
        comp = computed[key]
        pap = PAPER_FULL_MODEL[key]
        diff = abs(comp - pap)
        ok = "OK" if diff <= TOLERANCE else "FAIL"
        if ok == "FAIL":
            all_ok = False
        print(f"{key:<18} {comp:>10.2f} {pap:>10.2f} {diff:>8.4f} {ok:>6}")
    print("-" * len(header))
    print("Overall:", "ALL MATCH" if all_ok else "SOME MISMATCHES")
    print()


def print_ablation_summary() -> None:
    print("=" * 60)
    print("Table 3 — Ablation variants (paper values, not in CSV)")
    print("=" * 60)
    for variant, vals in ABLATION_VARIANTS.items():
        print(f"\n  {variant}")
        for k, v in vals.items():
            print(f"    {k:<18} {v:.2f}")
    print()


if __name__ == "__main__":
    computed = load_overall("evaluation_result_mean.csv")
    verify_full_model(computed)
    print_ablation_summary()
