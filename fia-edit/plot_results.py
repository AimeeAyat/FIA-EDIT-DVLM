import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings("ignore")

MEAN_CSV   = "evaluation_result_mean.csv"
PER_IMG_CSV = "evaluation_result.csv"
OUTPUT_DIR  = "plots"
OUTPUT_FILE = f"{OUTPUT_DIR}/evaluation_plots.png"

METHOD = "FIA-Edit_SD35"

METRICS = [
    "structure_distance",
    "psnr_unedit_part",
    "lpips_unedit_part",
    "mse_unedit_part",
    "ssim_unedit_part",
    "clip_similarity_source_image",
    "clip_similarity_target_image",
    "clip_similarity_target_image_edit_part",
]

METRIC_LABELS = [
    "Structure\nDistance↓",
    "PSNR\n(bg)↑",
    "LPIPS\n(bg)↓",
    "MSE\n(bg)↓",
    "SSIM\n(bg)↑",
    "CLIP\nSource↑",
    "CLIP\nTarget↑",
    "CLIP\nTarget\n(edit)↑",
]

# lower-is-better metrics get flipped for normalized radar/heatmap display
LOWER_IS_BETTER = {"structure_distance", "lpips_unedit_part", "mse_unedit_part"}

CLASS_NAMES = {
    "Overall":  "Overall",
    "Class_0":  "Random",
    "Class_1":  "Change Object",
    "Class_2":  "Add Object",
    "Class_3":  "Delete Object",
    "Class_4":  "Change Attribute",
    "Class_5":  "Change Background",
    "Class_6":  "Change Style",
    "Class_7":  "Change Texture",
    "Class_8":  "Change Color",
    "Class_9":  "Change Action",
}

PALETTE = "#4C72B0"
CLASS_COLORS = plt.cm.tab10(np.linspace(0, 1, 10))

# ── helpers ──────────────────────────────────────────────────────────────────

def col(metric):
    return f"{METHOD}|{metric}"

def load_mean():
    df = pd.read_csv(MEAN_CSV)
    df = df.rename(columns={"Category": "category"})
    return df

def load_per_image():
    df = pd.read_csv(PER_IMG_CSV)
    df = df.rename(columns={"file_id": "file_id"})
    return df

def normalize_row(values, metrics):
    """Min-max normalise; flip lower-is-better so 1 = best."""
    normed = {}
    for m, v in zip(metrics, values):
        if m in LOWER_IS_BETTER:
            normed[m] = 1 - v  # will be rescaled per-metric later
        else:
            normed[m] = v
    return normed

# ── plots ─────────────────────────────────────────────────────────────────────

def plot_overall_bar(ax, df_mean):
    overall = df_mean[df_mean["category"] == "Overall"].iloc[0]
    vals = [overall[col(m)] for m in METRICS]

    colors = [("#d9534f" if m in LOWER_IS_BETTER else "#4C72B0") for m in METRICS]
    bars = ax.bar(range(len(METRICS)), vals, color=colors, edgecolor="white", linewidth=0.8, width=0.6)

    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(range(len(METRICS)))
    ax.set_xticklabels(METRIC_LABELS, fontsize=8)
    ax.set_title("Overall Metric Scores", fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel("Score", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#4C72B0", label="Higher is better"),
                        Patch(color="#d9534f", label="Lower is better")],
              fontsize=8, loc="upper right")


def plot_class_heatmap(ax, df_mean):
    classes = [c for c in df_mean["category"] if c != "Overall"]
    data = []
    for cls in classes:
        row = df_mean[df_mean["category"] == cls].iloc[0]
        data.append([row[col(m)] for m in METRICS])

    data = np.array(data, dtype=float)

    # normalise each column to [0, 1] with direction awareness
    norm_data = np.zeros_like(data)
    for j, m in enumerate(METRICS):
        col_vals = data[:, j]
        mn, mx = col_vals.min(), col_vals.max()
        if mx == mn:
            norm_data[:, j] = 0.5
        elif m in LOWER_IS_BETTER:
            norm_data[:, j] = 1 - (col_vals - mn) / (mx - mn)
        else:
            norm_data[:, j] = (col_vals - mn) / (mx - mn)

    im = ax.imshow(norm_data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    # annotate with raw values
    for i in range(len(classes)):
        for j in range(len(METRICS)):
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center",
                    fontsize=7, color="black" if 0.3 < norm_data[i, j] < 0.8 else "white")

    y_labels = [CLASS_NAMES.get(c, c) for c in classes]
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xticks(range(len(METRICS)))
    ax.set_xticklabels(METRIC_LABELS, fontsize=7.5)
    ax.set_title("Per-Class Performance Heatmap\n(green = better, normalised per metric)", fontsize=11, fontweight="bold", pad=8)
    plt.colorbar(im, ax=ax, shrink=0.8, label="Normalised score")


def plot_boxplots(ax, df_img):
    data_by_metric = []
    for m in METRICS:
        vals = pd.to_numeric(df_img[col(m)], errors="coerce").dropna().values
        data_by_metric.append(vals)

    bp = ax.boxplot(data_by_metric, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=1.5),
                    whiskerprops=dict(linewidth=1),
                    capprops=dict(linewidth=1),
                    flierprops=dict(marker=".", markersize=2, alpha=0.4))

    colors = [("#f0a0a0" if m in LOWER_IS_BETTER else "#a0c4f0") for m in METRICS]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    ax.set_xticks(range(1, len(METRICS) + 1))
    ax.set_xticklabels(METRIC_LABELS, fontsize=8)
    ax.set_title("Per-Image Score Distributions", fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel("Score", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")


def plot_radar(ax, df_mean):
    classes = [c for c in df_mean["category"] if c != "Overall"]

    # normalise globally so radar arms are comparable
    all_vals = {m: [] for m in METRICS}
    for cls in classes:
        row = df_mean[df_mean["category"] == cls].iloc[0]
        for m in METRICS:
            all_vals[m].append(row[col(m)])

    global_min = {m: min(all_vals[m]) for m in METRICS}
    global_max = {m: max(all_vals[m]) for m in METRICS}

    def norm(val, m):
        mn, mx = global_min[m], global_max[m]
        if mx == mn:
            return 0.5
        n = (val - mn) / (mx - mn)
        return 1 - n if m in LOWER_IS_BETTER else n

    N = len(METRICS)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    for i, cls in enumerate(classes):
        row = df_mean[df_mean["category"] == cls].iloc[0]
        values = [norm(row[col(m)], m) for m in METRICS] + [norm(row[col(METRICS[0])], METRICS[0])]
        ax.plot(angles, values, linewidth=1.2, color=CLASS_COLORS[i],
                label=CLASS_NAMES.get(cls, cls))
        ax.fill(angles, values, alpha=0.05, color=CLASS_COLORS[i])

    # overall
    overall = df_mean[df_mean["category"] == "Overall"].iloc[0]
    ov_vals = [norm(overall[col(m)], m) for m in METRICS] + [norm(overall[col(METRICS[0])], METRICS[0])]
    ax.plot(angles, ov_vals, linewidth=2.5, color="black", linestyle="--", label="Overall")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(METRIC_LABELS, fontsize=7.5)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=6, color="grey")
    ax.set_title("Normalised Radar by Edit Class", fontsize=11, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.45, 1.15), fontsize=7.5, framealpha=0.8)
    ax.grid(color="grey", linestyle="--", linewidth=0.5, alpha=0.5)


def plot_class_bar(ax, df_mean):
    """Side-by-side bar for CLIP target score across classes."""
    classes = [c for c in df_mean["category"] if c != "Overall"]
    metric = "clip_similarity_target_image"
    vals = [df_mean[df_mean["category"] == c].iloc[0][col(metric)] for c in classes]
    labels = [CLASS_NAMES.get(c, c) for c in classes]

    bars = ax.barh(range(len(classes)), vals, color=CLASS_COLORS[:len(classes)],
                   edgecolor="white", linewidth=0.6)
    for bar, v in zip(bars, vals):
        ax.text(v + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{v:.2f}", va="center", fontsize=8)

    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("CLIP Similarity (Target)", fontsize=9)
    ax.set_title("Target CLIP Score by Edit Class", fontsize=11, fontweight="bold", pad=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    overall_val = df_mean[df_mean["category"] == "Overall"].iloc[0][col(metric)]
    ax.axvline(overall_val, color="black", linestyle="--", linewidth=1.2, label=f"Overall: {overall_val:.2f}")
    ax.legend(fontsize=8)


# ── main ──────────────────────────────────────────────────────────────────────

def save_individual(plot_fn, filename, df_mean=None, df_img=None, polar=False, figsize=(12, 6)):
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"} if polar else {})
    fig.patch.set_facecolor("#f8f8f8")
    if df_img is not None:
        plot_fn(ax, df_img)
    else:
        plot_fn(ax, df_mean)
    path = f"{OUTPUT_DIR}/{filename}"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {path}")


def main():
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_mean = load_mean()
    df_img  = load_per_image()

    # ── combined figure ───────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 18))
    fig.patch.set_facecolor("#f8f8f8")
    fig.suptitle(f"FIA-Edit Evaluation Results on PIE-Bench  |  Method: {METHOD}",
                 fontsize=15, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.35,
                           left=0.07, right=0.97, top=0.94, bottom=0.05)

    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :])
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[2, 1], polar=True)

    plot_overall_bar(ax1, df_mean)
    plot_class_heatmap(ax2, df_mean)
    plot_boxplots(ax3, df_img)
    plot_radar(ax4, df_mean)

    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved combined → {OUTPUT_FILE}")
    plt.close(fig)

    # ── individual figures ────────────────────────────────────────────────────
    print("Saving individual plots...")
    save_individual(plot_overall_bar,   "1_overall_bar.png",    df_mean=df_mean, figsize=(14, 5))
    save_individual(plot_class_heatmap, "2_class_heatmap.png",  df_mean=df_mean, figsize=(16, 7))
    save_individual(plot_boxplots,      "3_score_distributions.png", df_img=df_img, figsize=(14, 5))
    save_individual(plot_radar,         "4_radar_by_class.png", df_mean=df_mean, polar=True, figsize=(10, 8))
    save_individual(plot_class_bar,     "5_clip_target_by_class.png", df_mean=df_mean, figsize=(10, 6))

    plt.show()


if __name__ == "__main__":
    main()
