import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, Union
from pathlib import Path
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from ..state import State
from ..utils.file_utils import consolidate_features, consolidate_networks
from ..plotting.plotting import plot_coannotation


def coannotation_eval():
    """Runs the co-annotation evaluation by comparing pairs of genes in the feature and
    network datasets with known gene pairs given by the standards.
    """

    # keep track of evaluations from all standards
    all_evaluations = {}

    # perform evaluation for each standard
    for config_standard in State.config_standards:
        if config_standard["task"] != "coannotation":
            continue

        # import standard
        standard = import_coannotation_standard(config_standard)

        # reduce standard to only include genes common to standard and datasets
        shared_genes = np.intersect1d(list(standard.nodes()), State.consolidated_genes)
        standard = standard.subgraph(shared_genes)
        standard = nx.to_pandas_edgelist(standard)

        # reduce datasets to only include these common genes as well
        features = consolidate_features(State.features, shared_genes)
        networks = consolidate_networks(State.networks, shared_genes)

        # evaluate feature sets and networks
        evaluations = {}
        for name, feat in features.items():
            evaluations[name] = evaluate_features(feat, standard)

        for name, net in networks.items():
            evaluations[name] = evaluate_network(net, standard)

        # add the evaluations of this standard to the coannotation results table
        all_evaluations[config_standard["name"]] = evaluations

    # reorganize `all_evaluations` to use in DataFrame
    avp_plot_data, max_f1_plot_data, dataset_plot_data, standard_plot_data = [], [], [], []
    for standard, standard_evals in all_evaluations.items():

        for dataset, dataset_evals in standard_evals.items():
            avp = dataset_evals["Average Precision"]
            max_f1 = dataset_evals["Maximal F1"]

            avp_plot_data.append(avp)
            max_f1_plot_data.append(max_f1)
            dataset_plot_data.append(dataset)
            standard_plot_data.append(standard)

    all_evaluations = pd.DataFrame(
        [standard_plot_data, dataset_plot_data, avp_plot_data, max_f1_plot_data]
    ).T
    all_evaluations.columns = ["Standard", "Dataset", "Average Precision", "Maximal F1"]

    all_evaluations = pd.DataFrame(all_evaluations)
    State.coannotation_evaluations = all_evaluations

    # plotting
    if State.plot:
        plot_coannotation(all_evaluations)

    # output results
    all_evaluations.to_csv(
        State.result_path / Path(f"{State.config_name}_coannotation.tsv"), sep="\t", index=False
    )


def import_coannotation_standard(standard: Dict[str, Union[str, Path]]) -> nx.Graph:
    standard = nx.read_weighted_edgelist(standard["path"], delimiter=standard["delimiter"])
    return standard


def evaluate_features(features: pd.DataFrame, standard: pd.DataFrame) -> float:
    sim = cosine_similarity(features.values)
    features = pd.DataFrame(sim, index=features.index, columns=features.index).fillna(0)
    avp = compute_average_precision(features, standard)
    max_f1 = compute_max_f1(features, standard)
    return {"Average Precision": avp, "Maximal F1": max_f1}


def evaluate_network(network: nx.Graph, standard: pd.DataFrame) -> float:
    # map network to DataFrame for pairwise lookup
    network = nx.to_pandas_adjacency(network)
    avp = compute_average_precision(network, standard)
    max_f1 = compute_max_f1(network, standard)
    return {"Average Precision": avp, "Maximal F1": max_f1}


def compute_average_precision(dataset: pd.DataFrame, standard: pd.DataFrame) -> float:
    y_true = standard.iloc[:, 2].values
    # NOTE: `lookup` is depreciated for no good reason, I suspect it will be undepreciated
    # soon, see here: https://github.com/pandas-dev/pandas/issues/39171
    y_score = dataset.lookup(standard.iloc[:, 0], standard.iloc[:, 1])
    avp = average_precision_score(y_true, y_score)
    return avp


def compute_max_f1(dataset: pd.DataFrame, standard: pd.DataFrame) -> float:
    y_true = standard.iloc[:, 2].values
    y_score = dataset.lookup(standard.iloc[:, 0], standard.iloc[:, 1])
    precision, recall, _ = precision_recall_curve(y_true, y_score)

    # compute f1 scores
    denom = recall + precision
    f1_scores = np.divide(
        2 * recall * precision, denom, out=np.zeros_like(denom), where=(denom != 0)
    )
    max_f1 = np.max(f1_scores)
    return max_f1
