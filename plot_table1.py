"""
Plot Table 1: Quantitative comparison on PIE-Bench.
FIA-Edit numbers are loaded from evaluation_result_mean.csv.
All baseline numbers are hardcoded from the paper.
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
BASELINES = {
    "P2P":        {"Distance_1e3": 11.65, "PSNR": 27.22, "LPIPS_1e3": 54.55, "MSE_1e4": 32.86, "SSIM_1e2": 84.76, "CLIP_Whole": 25.02, "CLIP_Edited": 22.10},
    "PnP":        {"Distance_1e3": 24.29, "PSNR": 22.46, "LPIPS_1e3": 106.06,"MSE_1e4": 80.45, "SSIM_1e2": 79.68, "CLIP_Whole": 25.41, "CLIP_Edited": 22.62},
    "MasaCtrl":   {"Distance_1e3": 24.70, "PSNR": 22.64, "LPIPS_1e3": 87.94, "MSE_1e4": 81.09, "SSIM_1e2": 81.33, "CLIP_Whole": 24.38, "CLIP_Edited": 21.35},
    "FlexiEdit":  {"Distance_1e3": 22.13, "PSNR": 25.74, "LPIPS_1e3": 80.45, "MSE_1e4": 58.45, "SSIM_1e2": 82.62, "CLIP_Whole": 25.15, "CLIP_Edited": 22.87},
    "FreeDiff":   {"Distance_1e3": 18.70, "PSNR": 24.73, "LPIPS_1e3": 89.76, "MSE_1e4": 55.32, "SSIM_1e2": 81.68, "CLIP_Whole": 25.03, "CLIP_Edited": 22.12},
    "RF-Inv":     {"Distance_1e3": 48.76, "PSNR": 19.51, "LPIPS_1e3": 195.85,"MSE_1e4": 155.74,"SSIM_1e2": 68.95, "CLIP_Whole": 25.11, "CLIP_Edited": 22.50},
    "StableFlow": {"Distance_1e3": 19.24, "PSNR": 23.04, "LPIPS_1e3": 76.94, "MSE_1e4": 84.85, "SSIM_1e2": 87.22, "CLIP_Whole": 24.30, "CLIP_Edited": 21.28},
    "RF-Edit":    {"Distance_1e3": 27.70, "PSNR": 23.22, "LPIPS_1e3": 131.18,"MSE_1e4": 75.00, "SSIM_1e2": 81.44, "CLIP_Whole": 25.22, "CLIP_Edited": 22.40},
    "DCEdit":     {"Distance_1e3": 22.36, "PSNR": 25.41, "LPIPS_1e3": 94.17, "MSE_1e4": 48.09, "SSIM_1e2": 85.60, "CLIP_Whole": 25.47, "CLIP_Edited": 22.71},
    "FTEdit":     {"Distance_1e3": 18.17, "PSNR": 26.62, "LPIPS_1e3": 80.55, "MSE_1e4": 40.24, "SSIM_1e2": 91.50, "CLIP_Whole": 25.74, "CLIP_Edited": 22.27},
    "FlowEdit":   {"Distance_1e3": 23.62, "PSNR": 23.21, "LPIPS_1e3": 93.81, "MSE_1e4": 69.95, "SSIM_1e2": 85.09, "CLIP_Whole": 26.78, "CLIP_Edited": 23.73},
    "DNAEdit":    {"Distance_1e3": 14.19, "PSNR": 26.66, "LPIPS_1e3": 74.57, "MSE_1e4": 32.76, "SSIM_1e2": 88.63, "CLIP_Whole": 25.63, "CLIP_Edited": 22.71},
}

LOWER_IS_BETTER = {"Distance_1e3", "LPIPS_1e3", "MSE_1e4"}

METRIC_LABELS = {
    "Distance_1e3":  "Structure Distance\n(×10³ ↓)",
    "PSNR":          "PSNR (↑)",
    "LPIPS_1e3":     "LPIPS\n(×10³ ↓)",
    "MSE_1e4":       "MSE\n(×10⁴ ↓)",
    "SSIM_1e2":      "SSIM\n(×10² ↑)",
    "CLIP_Whole":    "CLIP Similarity\nWhole (↑)",
    "CLIP_Edited":   "CLIP Similarity\nEdited (↑)",
}


def load_fia_edit(csv_path: str) -> dict:
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


def plot_table1(fia_edit: dict, save_path: str = "plots/table1_comparison.png") -> None:
    methods = list(BASELINES.keys()) + ["FIA-Edit\n(Ours)"]
    metrics = list(METRIC_LABELS.keys())
    n_methods = len(methods)
    n_metrics = len(metrics)

    all_data = {m: v.copy() for m, v in BASELINES.items()}
    all_data["FIA-Edit\n(Ours)"] = fia_edit

    fig, axes = plt.subplots(1, n_metrics, figsize=(22, 6), constrained_layout=True)
    fig.suptitle("Table 1: Quantitative Comparison on PIE-Bench", fontsize=14, fontweight="bold")

    colors = plt.cm.tab20(np.linspace(0, 1, n_methods))
    ours_color = "#d62728"  # red for our method

    for ax, metric in zip(axes, metrics):
        values = [all_data[m][metric] for m in methods]
        bar_colors = [ours_color if "Ours" in m else "#4878d0" for m in methods]
        bars = ax.bar(range(n_methods), values, color=bar_colors, edgecolor="white", linewidth=0.5)

        # Mark best value
        best_idx = np.argmin(values) if metric in LOWER_IS_BETTER else np.argmax(values)
        bars[best_idx].set_edgecolor("gold")
        bars[best_idx].set_linewidth(2.5)

        ax.set_title(METRIC_LABELS[metric], fontsize=9, pad=4)
        ax.set_xticks(range(n_methods))
        ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=7)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)

    blue_patch = mpatches.Patch(color="#4878d0", label="Baselines")
    red_patch = mpatches.Patch(color=ours_color, label="FIA-Edit (Ours)")
    gold_patch = mpatches.Patch(facecolor="white", edgecolor="gold", linewidth=2, label="Best value")
    fig.legend(handles=[blue_patch, red_patch, gold_patch], loc="lower center",
               bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=9)

    import os; os.makedirs("plots", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    fia_edit = load_fia_edit("evaluation_result_mean.csv")
    print("FIA-Edit computed values:")
    for k, v in fia_edit.items():
        print(f"  {k:<18} {v:.2f}")
    print()
    plot_table1(fia_edit)
