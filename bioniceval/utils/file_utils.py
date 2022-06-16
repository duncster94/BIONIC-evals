from functools import reduce
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from typing import Optional, Dict, List
from pathlib import Path
from ..state import State


def import_datasets(consolidation: str = "union"):
    """Imports datasets and consolidates them to have the same number of genes,
    as given by the `consolidation` strategy.
    """

    features = {
        item["name"]: pd.read_csv(item["path"], delimiter=item["delimiter"], index_col=0)
        for item in State.config_features
    }

    networks = {
        item["name"]: nx.read_weighted_edgelist(item["path"], delimiter=item["delimiter"])
        for item in State.config_networks
    }
    consolidate_datasets(features, networks)


def import_standard():
    pass


def consolidate_datasets(
        features: Optional[Dict[str, pd.DataFrame]] = [],
        networks: Optional[Dict[str, nx.Graph]] = []
):
    consolidation = State.consolidation

    # compute consolidated genes
    feature_genes = [list(feat.index) for feat in features.values()]
    network_genes = [list(net.nodes()) for net in networks.values()]
    if consolidation == "union":
        consolidated_genes = reduce(np.union1d, feature_genes + network_genes)
    elif consolidation == "intersection":
        consolidated_genes = reduce(np.intersect1d, feature_genes + network_genes)
    elif (not State.baseline == []) and consolidation == "baseline":
        # if a baseline file is specified and the config consolidation mode is baseline
        consolidated_genes = State.baseline
    else:
        raise ValueError(f"Consolidation strategy '{consolidation}' is not supported.")

    features = consolidate_features(features, consolidated_genes)
    networks = consolidate_networks(networks, consolidated_genes)

    # test if features and networks are not empty
    if features and networks:
        assert list(list(features.values())[0].index) == list(list(networks.values())[0].nodes())

    State.features = features
    State.networks = networks
    State.consolidated_genes = consolidated_genes


def consolidate_features(
        features: Dict[str, pd.DataFrame], genes: List[str]
) -> Dict[str, pd.DataFrame]:
    """Consolidates features by ensuring they share the same genes in the same order."""

    consolidated = {}
    for name, feat in features.items():
        consolidated[name] = feat.reindex(genes).fillna(0)
    return consolidated


def consolidate_networks(networks: Dict[str, nx.Graph], genes: List[str]) -> Dict[str, nx.Graph]:
    """Consolidates networks by ensuring they share the same genes in the same order."""

    consolidated = {}
    for name, net in networks.items():
        new_net = nx.Graph()
        new_net.add_nodes_from(genes)
        new_net.add_edges_from(net.subgraph(genes).edges(data=True))
        consolidated[name] = new_net
    return consolidated


def generate_results(combined_table_path):
    """Generate Combined Results Table for all evaluation tasks in the config."""
    coannotation_pivot = State.coannotation_evaluations.pivot_table(
        ["Average Precision", "Maximum F1"], ["Dataset"],
        "Standard") if "coannotation_evaluations" in dir(State) else pd.DataFrame()
    coannotation = pd.concat([coannotation_pivot], keys=["Coannotation"], axis=1)

    module_detection_pivot = State.module_detection_evaluations.pivot_table(
        ["Module Match Score (AMI)", "Adjusted Rand Index(RI)"], ["Dataset"],
        "Standard") if "module_detection_evaluations" in dir(State) else pd.DataFrame()
    module_detection = pd.concat([module_detection_pivot], keys=["Module Detection"], axis=1)

    function_prediction_pivot = State.function_prediction_evaluations.pivot_table(
        ["Accuracy", "Macro F1", "Micro F1"],
        ["Dataset"], "Standard") if "function_prediction_evaluations" in dir(State) else pd.DataFrame()
    function_prediction = pd.concat([function_prediction_pivot], keys=["Function Prediction"], axis=1)

    Combined = pd.concat([coannotation, module_detection, function_prediction], axis=1)
    Combined.to_csv(combined_table_path / Path(f"{State.config_name}_all_results.tsv"), sep="\t")
