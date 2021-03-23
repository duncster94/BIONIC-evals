import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, Union
from pathlib import Path
from sklearn.metrics import average_precision_score
from sklearn.metrics.pairwise import cosine_similarity

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
            avp_score = evaluate_features(feat, standard)
            evaluations[name] = avp_score

        for name, net in networks.items():
            avp_score = evaluate_network(net, standard)
            evaluations[name] = avp_score

        all_evaluations[config_standard["name"]] = evaluations

    all_evaluations = pd.DataFrame(all_evaluations)
    State.coannotation_evaluations = all_evaluations

    # save results
    all_evaluations.to_csv(
        State.result_path / Path(f"{State.config_name}_coannotation.tsv"), sep="\t"
    )

    if State.plot:
        plot_coannotation()


def import_coannotation_standard(standard: Dict[str, Union[str, Path]]) -> nx.Graph:
    standard = nx.read_weighted_edgelist(standard["path"], delimiter=standard["delimiter"])
    return standard


def evaluate_features(features: pd.DataFrame, standard: pd.DataFrame) -> float:
    sim = cosine_similarity(features.values)
    features = pd.DataFrame(sim, index=features.index, columns=features.index).fillna(0)
    avp = compute_average_precision(features, standard)
    return avp


def evaluate_network(network: nx.Graph, standard: pd.DataFrame) -> float:
    # map network to DataFrame for pairwise lookup
    network = nx.to_pandas_adjacency(network)
    avp = compute_average_precision(network, standard)
    return avp


def compute_average_precision(dataset: pd.DataFrame, standard: pd.DataFrame) -> float:
    y_true = standard.iloc[:, 2].values
    # NOTE: `lookup` is depreciated for no good reason, I suspect it will be undepreciated
    # soon, see here: https://github.com/pandas-dev/pandas/issues/39171
    y_score = dataset.lookup(standard.iloc[:, 0], standard.iloc[:, 1])
    avp = average_precision_score(y_true, y_score)
    return avp
