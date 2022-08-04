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
    # reformat the results table and extract Precision and Recall values
    results = results.melt(ignore_index=False)
    results["Dataset"] = results.index
    results["Average Precision"] = [x[0] for x in results['value']]
    results["Maximum F1"] = [x[3] for x in results['value']]
    results["Standard"] = results["variable"]

    # plot PR cruve
    for row in results.itertuples():
        # invoke sklearn.metrics.PrecisionRecallDisplay
        dataset, standard, precision, recall = row.Dataset, row.Standard, row.value[1], row.value[2]

        # linear scale
        disp = PrecisionRecallDisplay(precision=precision, recall=recall)
        disp.plot()
        path = State.result_path / Path(f"{State.config_name}—coannotation—{dataset}—{standard}.png")
        path2 = State.result_path / Path(f"{State.config_name}—coannotation—{dataset}—{standard}.svg")
        plt.title(f"PR——{dataset}——{standard}")
        plt.tight_layout()
        plt.grid(axis="y")
        plt.savefig(path)
        plt.savefig(path2, format="svg")
        plt.clf()

        # log scale
        disp.plot()
        disp.ax_.set_xscale('log')
        path = State.result_path / Path(f"{State.config_name}—coannotation—{dataset}—{standard}—log.png")
        path2 = State.result_path / Path(f"{State.config_name}—coannotation—{dataset}—{standard}—log.svg")
        plt.title(f"PR—{dataset}—{standard}—log")
        plt.tight_layout()
        plt.grid(axis="y")
        plt.savefig(path)
        plt.savefig(path2, format="svg")
        plt.clf()
        plt.close(disp.figure_)

    # plot metrics
    plt.figure()
    out_path = State.result_path / Path(f"{State.config_name}_coannotation.png")
    out_path2 = State.result_path / Path(f"{State.config_name}_coannotation.svg")
    plot_bars(
        results, "Standard", "Average Precision", "Dataset", "Co-annotation Evaluation", out_path, out_path2
    )


def plot_module_detection():
    results: pd.DataFrame = State.module_detection_evaluations

    # plot AMI
    out_path = State.result_path / Path(f"{State.config_name}_module_detection_AMI.png")
    out_path2 = State.result_path / Path(f"{State.config_name}_module_detection_AMI.svg")
    plot_bars(
        results,
        "Standard",
        "Module Match Score (AMI)",
        "Dataset",
        "Module Detection Evaluation",
        out_path,
        out_path2
    )

    # plot RI
    out_path = State.result_path / Path(f"{State.config_name}_module_detection_RI.png")
    out_path2 = State.result_path / Path(f"{State.config_name}_module_detection_RI.svg")
    plot_bars(
        results,
        "Standard",
        "Adjusted Rand Index(RI)",
        "Dataset",
        "Module Detection Evaluation",
        out_path,
        out_path2
    )


def plot_function_prediction():
    results: pd.DataFrame = State.function_prediction_evaluations

    # plot micro f1
    out_path = State.result_path / Path(f"{State.config_name}_function_prediction_micro_f1.png")
    out_path2 = State.result_path / Path(f"{State.config_name}_function_prediction_micro_f1.svg")
    plot_bars(
        results,
        "Standard",
        "Micro F1",
        "Dataset",
        "Gene Function Prediction Evaluation",
        out_path,
        out_path2
    )

    # plot macro f1
    out_path = State.result_path / Path(f"{State.config_name}_function_prediction_macro_f1.png")
    out_path2 = State.result_path / Path(f"{State.config_name}_function_prediction_macro_f1.svg")
    plot_bars(
        results,
        "Standard",
        "Macro F1",
        "Dataset",
        "Gene Function Prediction Evaluation",
        out_path,
        out_path2
    )

    # plot accuracy
    out_path = State.result_path / Path(f"{State.config_name}_function_prediction_accuracy.png")
    out_path2 = State.result_path / Path(f"{State.config_name}_function_prediction_accuracy.svg")
    plot_bars(
        results,
        "Standard",
        "Accuracy",
        "Dataset",
        "Gene Function Prediction Evaluation",
        out_path,
        out_path2
    )


def plot_bars(df: pd.DataFrame, x: str, y: str, hue: str, title: str, out_path: Path, out_path2: Path):
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
    plt.savefig(out_path2, format="svg")
