"""
Plot Table 2: Memory and runtime comparison.
All values are hardcoded from the paper.
RF-Inv, StableFlow, RF-Edit ran on A100 80GB GPU (italic in paper);
all others on RTX 4090.
"""
import matplotlib.pyplot as plt
import numpy as np

# GPU memory (GB) and inference time (seconds) per method
DATA = {
    "P2P":        {"gpu": 10.95, "time": 34.84, "a100": False},
    "PnP":        {"gpu":  8.99, "time": 18.09, "a100": False},
    "MasaCtrl":   {"gpu": 11.42, "time": 21.71, "a100": False},
    "FlexiEdit":  {"gpu": 18.73, "time": 38.97, "a100": False},
    "FreeDiff":   {"gpu":  6.08, "time": 17.41, "a100": False},
    "RF-Inv":     {"gpu": 69.22, "time": 76.74, "a100": True},
    "StableFlow": {"gpu": 35.39, "time": 26.07, "a100": True},
    "RF-Edit":    {"gpu": 32.91, "time": 34.51, "a100": True},
    "FlowEdit":   {"gpu": 17.93, "time":  3.49, "a100": False},
    "FIA-Edit\n(Ours)": {"gpu": 17.93, "time":  6.30, "a100": False},
}


def plot_table2(save_path: str = "plots/table2_efficiency.png") -> None:
    fig, (ax_gpu, ax_time) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    fig.suptitle("Table 2: Memory and Runtime Comparison", fontsize=13, fontweight="bold")

    methods = list(DATA.keys())
    gpus  = [DATA[m]["gpu"]  for m in methods]
    times = [DATA[m]["time"] for m in methods]
    a100  = [DATA[m]["a100"] for m in methods]

    def _color(i):
        if "Ours" in methods[i]:
            return "#d62728"
        if a100[i]:
            return "#ff7f0e"
        return "#4878d0"

    bar_colors = [_color(i) for i in range(len(methods))]
    x = np.arange(len(methods))

    # GPU bar chart
    bars_gpu = ax_gpu.bar(x, gpus, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax_gpu.set_xticks(x)
    ax_gpu.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
    ax_gpu.set_ylabel("GPU Memory (GB)", fontsize=10)
    ax_gpu.set_title("GPU Memory (↓ better)", fontsize=10)
    ax_gpu.grid(axis="y", alpha=0.3)
    ax_gpu.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars_gpu, gpus):
        ax_gpu.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=7)

    # Time bar chart
    bars_time = ax_time.bar(x, times, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax_time.set_xticks(x)
    ax_time.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
    ax_time.set_ylabel("Inference Time (s)", fontsize=10)
    ax_time.set_title("Inference Time (↓ better)", fontsize=10)
    ax_time.grid(axis="y", alpha=0.3)
    ax_time.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars_time, times):
        ax_time.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=7)

    # Legend
    import matplotlib.patches as mpatches
    patches = [
        mpatches.Patch(color="#4878d0", label="RTX 4090"),
        mpatches.Patch(color="#ff7f0e", label="A100 80GB"),
        mpatches.Patch(color="#d62728", label="FIA-Edit (Ours)"),
    ]
    fig.legend(handles=patches, loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=3, fontsize=9)

    import os; os.makedirs("plots", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close(fig)


    # Scatter: GPU vs Time
    fig2, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for i, m in enumerate(methods):
        color = _color(i)
        ax.scatter(DATA[m]["time"], DATA[m]["gpu"], color=color, s=80, zorder=3)
        ax.annotate(m.replace("\n", " "), (DATA[m]["time"], DATA[m]["gpu"]),
                    textcoords="offset points", xytext=(5, 3), fontsize=7)
    ax.set_xlabel("Inference Time (s)", fontsize=10)
    ax.set_ylabel("GPU Memory (GB)", fontsize=10)
    ax.set_title("Table 2: Efficiency Trade-off", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig2.legend(handles=patches, loc="upper right", fontsize=8)
    scatter_path = "plots/table2_scatter.png"
    fig2.savefig(scatter_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {scatter_path}")
    plt.close(fig2)


if __name__ == "__main__":
    plot_table2()
