from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import PrecisionRecallDisplay

from ..state import State

PALETTE = sns.color_palette("BuPu")
EDGE_WIDTH = 1.25
EDGE_COLOUR = "#666666"
CAPSIZE = 0.05


def plot_coannotation(results: pd.DataFrame):

    # plot AVP
    out_path = State.result_path / Path(f"{State.config_name}_coannotation_avp.png")
    plot_bars(
        results, "Standard", "Average Precision", "Dataset", "Co-annotation Evaluation", out_path
    )

    # plot maximal F1
    out_path = State.result_path / Path(f"{State.config_name}_coannotation_maxf1.png")
    plot_bars(results, "Standard", "Maximal F1", "Dataset", "Co-annotation Evaluation", out_path)


def plot_module_detection():
    results: pd.DataFrame = State.module_detection_evaluations

    # plot AMI
    out_path = State.result_path / Path(f"{State.config_name}_module_detection_AMI.png")
    plot_bars(
        results,
        "Standard",
        "Module Match Score (AMI)",
        "Dataset",
        "Module Detection Evaluation",
        out_path,
    )


def plot_function_prediction():
    results: pd.DataFrame = State.function_prediction_evaluations

    # plot micro f1
    out_path = State.result_path / Path(f"{State.config_name}_function_prediction_micro_f1.png")
    plot_bars(
        results, "Standard", "Micro F1", "Dataset", "Gene Function Prediction Evaluation", out_path,
    )

    # plot macro f1
    out_path = State.result_path / Path(f"{State.config_name}_function_prediction_macro_f1.png")
    plot_bars(
        results, "Standard", "Macro F1", "Dataset", "Gene Function Prediction Evaluation", out_path,
    )

    # plot accuracy
    out_path = State.result_path / Path(f"{State.config_name}_function_prediction_accuracy.png")
    plot_bars(
        results, "Standard", "Accuracy", "Dataset", "Gene Function Prediction Evaluation", out_path,
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
