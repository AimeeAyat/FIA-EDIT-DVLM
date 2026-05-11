import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse

warnings.filterwarnings("ignore")

METRICS = [
    "structure_distance",
    "psnr_unedit_part",
    "lpips_unedit_part",
    "mse_unedit_part",
    "ssim_unedit_part",
    "clip_similarity_source_image",
    "clip_similarity_target_image",
    "clip_similarity_target_image_edit_part",
    "HPS V2.1",
    "Aesthetic Score",
]

METRIC_LABELS = [
    "Struct\nDist↓",
    "PSNR\n(bg)↑",
    "LPIPS\n(bg)↓",
    "MSE\n(bg)↓",
    "SSIM\n(bg)↑",
    "CLIP\nSrc↑",
    "CLIP\nTgt↑",
    "CLIP\nTgt\n(edit)↑",
    "HPS\nV2.1↑",
    "Aesthetic\nScore↑",
]

LOWER_IS_BETTER = {"structure_distance", "lpips_unedit_part", "mse_unedit_part"}

CLASS_NAMES = {
    '0': 'Random',
    '1': 'Change Object',
    '2': 'Add Object',
    '3': 'Delete Object',
    '4': 'Change Content',
    '5': 'Change Pose',
    '6': 'Change Color',
    '7': 'Change Material',
    '8': 'Change Background',
    '9': 'Change Style',
}

CLASS_COLORS = plt.cm.tab10(np.linspace(0, 1, 10))


def load_data(result_dir):
    group_df = pd.read_csv(os.path.join(result_dir, "metrics_group.csv"))
    group_df["category"] = group_df["category"].astype(str)

    overall_df = pd.read_csv(os.path.join(result_dir, "metrics_overall.csv"),
                             header=None, index_col=0).T
    overall_df.columns = overall_df.columns.astype(str)
    overall_df.insert(0, "category", "Overall")
    overall_df = overall_df.reset_index(drop=True)

    mean_df = pd.concat([overall_df, group_df], ignore_index=True)

    per_img_df = pd.read_csv(os.path.join(result_dir, "evaluation_result.csv"))
    per_img_df = per_img_df.apply(pd.to_numeric, errors="coerce")

    return mean_df, per_img_df


def _available(df, metrics):
    return [m for m in metrics if m in df.columns]


def plot_overall_bar(ax, mean_df):
    overall = mean_df[mean_df["category"] == "Overall"].iloc[0]
    metrics = _available(mean_df, METRICS)
    labels = [METRIC_LABELS[METRICS.index(m)] for m in metrics]
    vals = [float(overall[m]) for m in metrics]
    colors = ["#d9534f" if m in LOWER_IS_BETTER else "#4C72B0" for m in metrics]

    bars = ax.bar(range(len(metrics)), vals, color=colors, edgecolor="white", linewidth=0.8, width=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.01,
                f"{v:.4f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title("Overall Metric Scores  (KV-Edit, PIE-Bench)", fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel("Score", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#4C72B0", label="Higher is better"),
                       Patch(color="#d9534f", label="Lower is better")],
              fontsize=8, loc="upper right")


def plot_class_heatmap(ax, mean_df):
    metrics = _available(mean_df, METRICS)
    classes = [c for c in mean_df["category"].tolist() if c != "Overall"]
    data = np.array([[float(mean_df[mean_df["category"] == c].iloc[0][m])
                      for m in metrics] for c in classes], dtype=float)

    norm = np.zeros_like(data)
    for j, m in enumerate(metrics):
        col_vals = data[:, j]
        mn, mx = np.nanmin(col_vals), np.nanmax(col_vals)
        if mx == mn:
            norm[:, j] = 0.5
        elif m in LOWER_IS_BETTER:
            norm[:, j] = 1 - (col_vals - mn) / (mx - mn)
        else:
            norm[:, j] = (col_vals - mn) / (mx - mn)

    im = ax.imshow(norm, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(metrics)):
            v = data[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7,
                    color="black" if 0.3 < norm[i, j] < 0.8 else "white")

    labels = [METRIC_LABELS[METRICS.index(m)] for m in metrics]
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels([CLASS_NAMES.get(c, c) for c in classes], fontsize=8)
    ax.set_title("Per-Class Heatmap  (green = better, normalised per metric)",
                 fontsize=11, fontweight="bold", pad=8)
    plt.colorbar(im, ax=ax, shrink=0.8, label="Normalised score")


def plot_boxplots(ax, per_img_df):
    metrics = _available(per_img_df, METRICS)
    data = [per_img_df[m].dropna().values for m in metrics]
    labels = [METRIC_LABELS[METRICS.index(m)] for m in metrics]
    colors = ["#f0a0a0" if m in LOWER_IS_BETTER else "#a0c4f0" for m in metrics]

    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=1.5),
                    whiskerprops=dict(linewidth=1), capprops=dict(linewidth=1),
                    flierprops=dict(marker=".", markersize=2, alpha=0.4))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    ax.set_xticks(range(1, len(metrics) + 1))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title("Per-Image Score Distributions", fontsize=11, fontweight="bold", pad=8)
    ax.set_ylabel("Score", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")


def plot_radar(ax, mean_df):
    metrics = _available(mean_df, METRICS)
    classes = [c for c in mean_df["category"].tolist() if c != "Overall"]

    all_vals = {m: [float(mean_df[mean_df["category"] == c].iloc[0][m]) for c in classes]
                for m in metrics}
    g_min = {m: min(all_vals[m]) for m in metrics}
    g_max = {m: max(all_vals[m]) for m in metrics}

    def norm(v, m):
        mn, mx = g_min[m], g_max[m]
        if mx == mn:
            return 0.5
        n = (v - mn) / (mx - mn)
        return 1 - n if m in LOWER_IS_BETTER else n

    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    for i, c in enumerate(classes):
        row = mean_df[mean_df["category"] == c].iloc[0]
        vals = [norm(float(row[m]), m) for m in metrics] + [norm(float(row[metrics[0]]), metrics[0])]
        ax.plot(angles, vals, linewidth=1.2, color=CLASS_COLORS[i],
                label=CLASS_NAMES.get(c, c))
        ax.fill(angles, vals, alpha=0.05, color=CLASS_COLORS[i])

    overall = mean_df[mean_df["category"] == "Overall"].iloc[0]
    ov = [norm(float(overall[m]), m) for m in metrics] + [norm(float(overall[metrics[0]]), metrics[0])]
    ax.plot(angles, ov, linewidth=2.5, color="black", linestyle="--", label="Overall")

    labels = [METRIC_LABELS[METRICS.index(m)] for m in metrics]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=6, color="grey")
    ax.set_title("Normalised Radar by Edit Class", fontsize=11, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.55, 1.15), fontsize=7.5, framealpha=0.8)
    ax.grid(color="grey", linestyle="--", linewidth=0.5, alpha=0.5)


def plot_per_class_bars(axes, mean_df):
    """One subplot per metric showing per-class bar chart."""
    metrics = _available(mean_df, METRICS)
    classes = [c for c in mean_df["category"].tolist() if c != "Overall"]
    class_labels = [CLASS_NAMES.get(c, c) for c in classes]
    x = np.arange(len(classes))

    for ax, m in zip(axes, metrics):
        vals = [float(mean_df[mean_df["category"] == c].iloc[0][m]) for c in classes]
        overall = float(mean_df[mean_df["category"] == "Overall"].iloc[0][m])
        color = "#d9534f" if m in LOWER_IS_BETTER else "#4C72B0"
        bars = ax.bar(x, vals, color=color, alpha=0.8, edgecolor="white", linewidth=0.6)
        ax.axhline(overall, color="black", linestyle="--", linewidth=1, label=f"Overall: {overall:.3f}")
        ax.set_xticks(x)
        ax.set_xticklabels(class_labels, rotation=30, ha="right", fontsize=7)
        label = METRIC_LABELS[METRICS.index(m)]
        ax.set_title(label.replace("\n", " "), fontsize=9, fontweight="bold")
        ax.legend(fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3, linestyle="--")


def save_fig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.result_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    mean_df, per_img_df = load_data(args.result_dir)

    # Combined figure
    fig = plt.figure(figsize=(22, 18))
    fig.patch.set_facecolor("#f8f8f8")
    fig.suptitle("KV-Edit — PIE-Bench Evaluation Results",
                 fontsize=15, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.35,
                           left=0.07, right=0.97, top=0.94, bottom=0.05)
    plot_overall_bar(fig.add_subplot(gs[0, :]), mean_df)
    plot_class_heatmap(fig.add_subplot(gs[1, :]), mean_df)
    plot_boxplots(fig.add_subplot(gs[2, 0]), per_img_df)
    plot_radar(fig.add_subplot(gs[2, 1], polar=True), mean_df)
    save_fig(fig, os.path.join(out_dir, "combined.png"))

    # Per-class bars — one subplot per metric
    metrics_avail = _available(mean_df, METRICS)
    n = len(metrics_avail)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4 * nrows))
    fig.patch.set_facecolor("#f8f8f8")
    fig.suptitle("Per-Class Metrics — KV-Edit", fontsize=13, fontweight="bold")
    axes_flat = axes.flatten() if n > 1 else [axes]
    plot_per_class_bars(axes_flat[:n], mean_df)
    for ax in axes_flat[n:]:
        ax.set_visible(False)
    plt.tight_layout()
    save_fig(fig, os.path.join(out_dir, "5_per_class_bars.png"))

    # Individual figures
    for fn, name, polar, fs in [
        (plot_overall_bar,   "1_overall_bar.png",    False, (14, 5)),
        (plot_class_heatmap, "2_class_heatmap.png",  False, (16, 7)),
        (plot_boxplots,      "3_distributions.png",  False, (14, 5)),
        (plot_radar,         "4_radar.png",           True,  (10, 8)),
    ]:
        fig, ax = plt.subplots(figsize=fs, subplot_kw={"projection": "polar"} if polar else {})
        fig.patch.set_facecolor("#f8f8f8")
        fn(ax, per_img_df if fn == plot_boxplots else mean_df)
        save_fig(fig, os.path.join(out_dir, name))


if __name__ == "__main__":
    main()
