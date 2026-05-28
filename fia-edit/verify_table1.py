"""
Verification script for Table 1 (PIE-Bench quantitative comparison).
Loads evaluation_result_mean.csv and checks FIA-Edit ("Ours") numbers
against the paper-reported values.
"""
import pandas as pd

PAPER_VALUES = {
    "Distance_1e3":  10.34,
    "PSNR":          27.32,
    "LPIPS_1e3":     55.02,
    "MSE_1e4":       28.66,
    "SSIM_1e2":      89.21,
    "CLIP_Whole":    25.89,
    "CLIP_Edited":   22.82,
}

TOLERANCE = 0.05  # acceptable absolute difference after scaling


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


def verify(computed: dict, paper: dict) -> None:
    header = f"{'Metric':<18} {'Computed':>10} {'Paper':>10} {'Diff':>8} {'OK?':>6}"
    print("=" * len(header))
    print("Table 1 — FIA-Edit (Ours) Verification")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    all_ok = True
    for key in paper:
        comp = computed[key]
        pap = paper[key]
        diff = abs(comp - pap)
        ok = "OK" if diff <= TOLERANCE else "FAIL"
        if ok == "FAIL":
            all_ok = False
        print(f"{key:<18} {comp:>10.2f} {pap:>10.2f} {diff:>8.4f} {ok:>6}")
    print("-" * len(header))
    print("Overall:", "ALL MATCH" if all_ok else "SOME MISMATCHES")
    print()


if __name__ == "__main__":
    computed = load_overall("evaluation_result_mean.csv")
    verify(computed, PAPER_VALUES)
