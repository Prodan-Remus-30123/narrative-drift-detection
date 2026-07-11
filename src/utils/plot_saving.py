from pathlib import Path
import re
import matplotlib.pyplot as plt


# Standard single-panel line/bar chart size, shared by the plot_*
# modules that don't have a size-driving reason to differ (e.g. a
# heatmap sized by entity count, or a multi-panel dashboard).
DEFAULT_FIGSIZE = (12, 6)


def safe_filename(value):
    value = str(value)
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = value.strip("_")
    return value or "plot"


def save_plot(output_dir, plot_name, source=None, dpi=200):
    output_dir = Path(output_dir)

    if source:
        output_dir = output_dir / safe_filename(source)

    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / f"{safe_filename(plot_name)}.png"

    plt.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight"
    )

    plt.close()

    return path