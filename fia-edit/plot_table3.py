"""
Plot Table 3: Ablation study.
Baseline (FIJ=✗ FRI=✗) and intermediate rows are hardcoded from the paper.
Full model (FIJ=✓ FRI=freq) is loaded from evaluation_result_mean.csv.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

PAPER_ABLATION = {
    "Baseline\n(FIJ=✗ FRI=✗)":   {"Distance_1e3": 23.62, "PSNR": 23.21, "LPIPS_1e3": 93.81, "MSE_1e4": 69.95, "SSIM_1e2": 85.09, "CLIP_Whole": 26.78, "CLIP_Edited": 23.73},
    "+FIJ\n(FRI=✗)":              {"Distance_1e3": 14.89, "PSNR": 25.59, "LPIPS_1e3": 70.18, "MSE_1e4": 41.74, "SSIM_1e2": 87.51, "CLIP_Whole": 26.30, "CLIP_Edited": 23.12},
    "+FIJ +FRI\n(add fusion)":    {"Distance_1e3": 16.50, "PSNR": 25.93, "LPIPS_1e3": 85.44, "MSE_1e4": 38.72, "SSIM_1e2": 86.51, "CLIP_Whole": 26.05, "CLIP_Edited": 22.68},
    "Full Model\n(FIJ=✓ FRI=freq)": None,  # loaded from CSV
}

LOWER_IS_BETTER = {"Distance_1e3", "LPIPS_1e3", "MSE_1e4"}

METRIC_LABELS = {
    "Distance_1e3": "Structure\nDistance×10³ ↓",
    "PSNR":         "PSNR ↑",
    "LPIPS_1e3":    "LPIPS×10³ ↓",
    "MSE_1e4":      "MSE×10⁴ ↓",
    "SSIM_1e2":     "SSIM×10² ↑",
    "CLIP_Whole":   "CLIP Whole ↑",
    "CLIP_Edited":  "CLIP Edited ↑",
}


def load_fia_full(csv_path: str) -> dict:
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


def plot_table3(csv_path: str = "evaluation_result_mean.csv",
                save_path: str = "plots/table3_ablation.png") -> None:
    full_model = load_fia_full(csv_path)
    PAPER_ABLATION["Full Model\n(FIJ=✓ FRI=freq)"] = full_model

    variants = list(PAPER_ABLATION.keys())
    metrics = list(METRIC_LABELS.keys())
    n_variants = len(variants)
    n_metrics = len(metrics)

    colors = ["#aec7e8", "#6baed6", "#2171b5", "#d62728"]  # light→dark→red for ours

    fig, axes = plt.subplots(1, n_metrics, figsize=(20, 5), constrained_layout=True)
    fig.suptitle("Table 3: Ablation Study on Key Components", fontsize=13, fontweight="bold")

    for ax, metric in zip(axes, metrics):
        values = [PAPER_ABLATION[v][metric] for v in variants]
        bars = ax.bar(range(n_variants), values, color=colors, edgecolor="white", linewidth=0.5)

        # Mark best
        best_idx = np.argmin(values) if metric in LOWER_IS_BETTER else np.argmax(values)
        bars[best_idx].set_edgecolor("gold")
        bars[best_idx].set_linewidth(2.5)

        ax.set_title(METRIC_LABELS[metric], fontsize=8, pad=4)
        ax.set_xticks(range(n_variants))
        ax.set_xticklabels(variants, rotation=45, ha="right", fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.005,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=6.5)

    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=c, label=v.replace("\n", " ")) for c, v in zip(colors, variants)]
    gold_p = mpatches.Patch(facecolor="white", edgecolor="gold", linewidth=2, label="Best per metric")
    fig.legend(handles=patches + [gold_p], loc="lower center", bbox_to_anchor=(0.5, -0.04),
               ncol=5, fontsize=8)

    import os; os.makedirs("plots", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    plot_table3()
