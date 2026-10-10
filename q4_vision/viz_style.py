"""Shared plotting style and small helpers for the Question 4 scripts."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e4e3df"
# categorical slots, fixed order (blue, orange, aqua) - at most 3 series per chart
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "axes.titlecolor": TEXT,
    "xtick.color": TEXT2, "ytick.color": TEXT2, "text.color": TEXT,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "legend.frameon": False,
})

RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)


def save(fig, name):
    path = os.path.join(RESULTS, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def log_metrics(section, metrics):
    """Append numbers to results/metrics.json so the write-up can quote them."""
    path = os.path.join(RESULTS, "metrics.json")
    allm = json.load(open(path)) if os.path.exists(path) else {}
    allm[section] = metrics
    json.dump(allm, open(path, "w"), indent=2)


def image_axes(ax, title):
    ax.set_title(title, fontsize=10)
    ax.axis("off")
