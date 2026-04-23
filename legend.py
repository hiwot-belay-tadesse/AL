
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

distinct_colors = [
    "#377eb8",  # blue
    "#e41a1c",  # red
    "#4daf4a",  # green
    "#ff7f00",  # orange
    "#984ea3",  # purple
    "#ffff33",  # yellow
    "#a65628",  # brown
    "#f781bf",  # pink
    "#999999",  # gray
]


def plot_horizontal_legend(colors, labels=None, out_path="legend_horizontal.png"):
    if labels is None:
        labels = colors
    handles = [Patch(facecolor=c, edgecolor="black") for c in colors]
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(colors)), 1.6))
    ax.axis("off")
    ax.legend(
        handles,
        labels,
        ncol=len(labels),
        loc="center",
        frameon=False,
        handlelength=3.0,
        columnspacing=1.2,
        fontsize=14,
    )
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    # plt.show()
    plt.close(fig)


if __name__ == "__main__":
    plot_horizontal_legend(
        distinct_colors,
        labels=["ID10", "ID12", "ID20", "ID21", "ID27"],
    )
