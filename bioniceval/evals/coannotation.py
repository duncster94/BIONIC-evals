import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, Union
from pathlib import Path
from sklearn.metrics import average_precision_score
from sklearn.metrics import (precision_recall_curve, PrecisionRecallDisplay)
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

    # plotting
    all_evaluations = pd.DataFrame(all_evaluations)
    if State.plot:
        plot_coannotation(all_evaluations)

    # reformat the results and save it to State.coannotation_evaluations
    all_evaluations = all_evaluations.melt(ignore_index=False)
    all_evaluations["Dataset"] = all_evaluations.index
    all_evaluations["Average Precision"] = [x[0] for x in all_evaluations['value']]
    all_evaluations["Maximum F1"] = [x[3] for x in all_evaluations['value']]
    all_evaluations["Standard"] = all_evaluations["variable"]
    all_evaluations.reset_index(inplace=True)
    all_evaluations = all_evaluations.iloc[:, 3:]
    State.coannotation_evaluations = all_evaluations

    # output results
    all_evaluations.to_csv(
        State.result_path / Path(f"{State.config_name}_coannotation.tsv"), sep="\t"
    )


def import_coannotation_standard(standard: Dict[str, Union[str, Path]]) -> nx.Graph:
    standard = nx.read_weighted_edgelist(standard["path"], delimiter=standard["delimiter"])
    return standard


def evaluate_features(features: pd.DataFrame, standard: pd.DataFrame) -> float:
    sim = cosine_similarity(features.values)
    features = pd.DataFrame(sim, index=features.index, columns=features.index).fillna(0)
    avp = compute_average_precision(features, standard)
    prc = compute_max_f1_core(features, standard)
    return [avp] + prc


def evaluate_network(network: nx.Graph, standard: pd.DataFrame) -> float:
    # map network to DataFrame for pairwise lookup
    network = nx.to_pandas_adjacency(network)
    avp = compute_average_precision(network, standard)
    prc = compute_max_f1_core(network, standard)
    return [avp] + prc


def compute_average_precision(dataset: pd.DataFrame, standard: pd.DataFrame) -> float:
    y_true = standard.iloc[:, 2].values
    # NOTE: `lookup` is depreciated for no good reason, I suspect it will be undepreciated
    # soon, see here: https://github.com/pandas-dev/pandas/issues/39171
    y_score = dataset.lookup(standard.iloc[:, 0], standard.iloc[:, 1])
    avp = average_precision_score(y_true, y_score)
    return avp


def compute_max_f1_core(dataset: pd.DataFrame, standard: pd.DataFrame) -> float:
    y_true = standard.iloc[:, 2].values
    # NOTE: `lookup` is depreciated for no good reason, I suspect it will be undepreciated
    # soon, see here: https://github.com/pandas-dev/pandas/issues/39171
    probas_pred = dataset.lookup(standard.iloc[:, 0], standard.iloc[:, 1])
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html
    precision, recall, thresholds = precision_recall_curve(y_true, probas_pred)
    denom = recall + precision
    f1_scores = np.divide(2 * recall * precision, denom, out=np.zeros_like(denom), where=(denom != 0))
    max_f1 = np.max(f1_scores)
    return [precision, recall, max_f1]
