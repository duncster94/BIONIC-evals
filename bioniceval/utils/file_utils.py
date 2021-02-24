from functools import reduce
import numpy as np
import pandas as pd
import networkx as nx
from typing import Optional, Dict, List

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
    networks: Optional[Dict[str, nx.Graph]] = [],
):
    consolidation = State.consolidation

    # compute consolidated genes
    feature_genes = [list(feat.index) for feat in features.values()]
    network_genes = [list(net.nodes()) for net in networks.values()]
    if consolidation == "union":
        consolidated_genes = reduce(np.union1d, feature_genes + network_genes)
    elif consolidation == "intersection":
        consolidated_genes = reduce(np.intersect1d, feature_genes + network_genes)
    else:
        raise ValueError(f"Consolidation strategy '{consolidation}' is not supported.")

    features = consolidate_features(features, consolidated_genes)
    networks = consolidate_networks(networks, consolidated_genes)
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
        new_net.add_edges_from(net.edges())
        new_net = new_net.subgraph(genes)
        consolidated[name] = new_net
    return consolidated