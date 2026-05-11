"""
Plot Table 4: Bleeding classification performance.
All values are hardcoded from the paper.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

DATA = {
    "ConvNeXt-T": {"AUC": 81.54, "PR-AUC": 38.66, "Precision": 50.90, "Recall": 29.49, "F1": 37.35, "Accuracy": 91.93},
    "Aug":        {"AUC": 82.10, "PR-AUC": 38.81, "Precision": 49.18, "Recall": 30.84, "F1": 37.91, "Accuracy": 91.76},
    "PnP":        {"AUC": 81.98, "PR-AUC": 38.53, "Precision": 53.03, "Recall": 26.57, "F1": 35.40, "Accuracy": 92.09},
    "MasaCtrl":   {"AUC": 84.22, "PR-AUC": 38.82, "Precision": 51.92, "Recall": 27.17, "F1": 35.67, "Accuracy": 92.01},
    "FlexiEdit":  {"AUC": 82.05, "PR-AUC": 37.97, "Precision": 52.62, "Recall": 26.03, "F1": 34.83, "Accuracy": 92.05},
    "FreeDiff":   {"AUC": 82.25, "PR-AUC": 38.60, "Precision": 53.68, "Recall": 25.65, "F1": 34.71, "Accuracy": 92.13},
    "FlowEdit":   {"AUC": 83.83, "PR-AUC": 40.34, "Precision": 50.88, "Recall": 31.44, "F1": 38.86, "Accuracy": 91.93},
    "FIA-Edit\n(Ours)": {"AUC": 85.05, "PR-AUC": 43.81, "Precision": 54.01, "Recall": 32.90, "F1": 40.89, "Accuracy": 92.24},
}

METRIC_LABELS = {
    "AUC":       "AUC (%) ↑",
    "PR-AUC":    "PR-AUC (%) ↑",
    "Precision": "Precision (%) ↑",
    "Recall":    "Recall (%) ↑",
    "F1":        "F1-score (%) ↑",
    "Accuracy":  "Accuracy (%) ↑",
}


def plot_table4(save_path: str = "plots/table4_classification.png") -> None:
    methods = list(DATA.keys())
    metrics = list(METRIC_LABELS.keys())
    n_methods = len(methods)
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=(20, 5), constrained_layout=True)
    fig.suptitle("Table 4: Bleeding Classification Performance", fontsize=13, fontweight="bold")

    for ax, metric in zip(axes, metrics):
        values = [DATA[m][metric] for m in methods]
        bar_colors = ["#d62728" if "Ours" in m else "#4878d0" for m in methods]
        bars = ax.bar(range(n_methods), values, color=bar_colors, edgecolor="white", linewidth=0.5)

        # Mark best
        best_idx = np.argmax(values)
        bars[best_idx].set_edgecolor("gold")
        bars[best_idx].set_linewidth(2.5)

        ax.set_title(METRIC_LABELS[metric], fontsize=9, pad=4)
        ax.set_xticks(range(n_methods))
        ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=7.5)
        ax.tick_params(axis="y", labelsize=8)

        # Zoom y-axis to differences
        margin = (max(values) - min(values)) * 0.5
        ax.set_ylim(min(values) - margin, max(values) + margin * 1.5)

        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + margin * 0.05,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=6.5)

    blue_patch = mpatches.Patch(color="#4878d0", label="Baselines")
    red_patch  = mpatches.Patch(color="#d62728", label="FIA-Edit (Ours)")
    gold_patch = mpatches.Patch(facecolor="white", edgecolor="gold", linewidth=2, label="Best per metric")
    fig.legend(handles=[blue_patch, red_patch, gold_patch], loc="lower center",
               bbox_to_anchor=(0.5, -0.03), ncol=3, fontsize=9)

    import os; os.makedirs("plots", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close(fig)


    # --- Radar chart ---
    categories = list(METRIC_LABELS.keys())
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig2, ax2 = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True), constrained_layout=True)
    ax2.set_title("Table 4: Classification Radar Chart", fontsize=12, fontweight="bold", pad=20)

    all_vals = {m: [DATA[m][c] for c in categories] for m in methods}
    # Normalise per metric to [0, 1] for radar
    for ci, metric in enumerate(categories):
        col_vals = [all_vals[m][ci] for m in methods]
        mn, mx = min(col_vals), max(col_vals)
        for m in methods:
            all_vals[m][ci] = (all_vals[m][ci] - mn) / (mx - mn + 1e-9)

    palette = plt.cm.tab10(np.linspace(0, 1, n_methods))
    for i, m in enumerate(methods):
        vals = all_vals[m] + [all_vals[m][0]]
        lw = 2.5 if "Ours" in m else 1.0
        ax2.plot(angles, vals, color=palette[i], linewidth=lw, label=m.replace("\n", " "))
        ax2.fill(angles, vals, color=palette[i], alpha=0.05 if "Ours" not in m else 0.15)

    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories, fontsize=9)
    ax2.set_yticklabels([])
    ax2.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)

    radar_path = "plots/table4_radar.png"
    fig2.savefig(radar_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {radar_path}")
    plt.close(fig2)


if __name__ == "__main__":
    plot_table4()
