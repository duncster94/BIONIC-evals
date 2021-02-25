from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ..state import State

PALETTE = sns.color_palette("BuPu")
EDGE_WIDTH = 1.25
EDGE_COLOUR = "#666666"
CAPSIZE = 0.05


def plot_coannotation():
    results: pd.DataFrame = State.coannotation_evaluations
    results = results.melt(ignore_index=False)
    results["Dataset"] = results.index
    results.columns = ["Standard", "Average Precision", "Dataset"]

    out_path = State.result_path / Path(f"{State.config_name}_coannotation.png")
    plot_bars(
        results, "Standard", "Average Precision", "Dataset", "Co-annotation Evaluation", out_path
    )


def plot_module_detection():
    results: pd.DataFrame = State.module_detection_evaluations
    out_path = State.result_path / Path(f"{State.config_name}_module_detection.png")
    plot_bars(
        results,
        "Standard",
        "Module Match Score (AMI)",
        "Dataset",
        "Module Detection Evaluation",
        out_path,
    )


def plot_bars(df: pd.DataFrame, x: str, y: str, hue: str, title: str, out_path: Path):
    plt.clf()
    plt.figure(figsize=(6, 4))

    ax = sns.barplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        palette=PALETTE,
        linewidth=EDGE_WIDTH,
        edgecolor=EDGE_COLOUR,
        capsize=CAPSIZE,
    )
    sns.despine(offset={"left": 10})
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1))
    plt.title(title)
    plt.tight_layout()

    plt.savefig(out_path)
