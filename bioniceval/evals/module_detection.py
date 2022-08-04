import json
from multiprocessing import Pool
import os
import random
from functools import reduce, partial
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster, maxdists
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from ..state import State
from ..utils.file_utils import consolidate_features, consolidate_networks
from ..plotting.plotting import plot_module_detection


def module_detection_eval():

    # keep track of evaluations from all standards
    evaluations = []

    for config_standard in State.config_standards:
        if config_standard["task"] != "module_detection":
            continue

        standard_name = config_standard["name"]

        # import standard
        standard = import_module_detection_standard(config_standard)

        # reduce standard to only include genes common to standard and datasets
        standard = reduce_standard(standard)

        # Reduce datasets to only include these common genes as well. These datasets
        # will be reduced again after gene sampling, but this saves time in the computation.
        shared_genes = reduce(np.union1d, list(standard.values()))
        features = consolidate_features(State.features, shared_genes)
        networks = consolidate_networks(State.networks, shared_genes)

        # invert `standard` dictionary to make sampling more efficient
        inverted_standard = invert_standard(standard)

        # evaluate features and networks using multiprocessing
        with Pool() as p:
            args = [standard, inverted_standard, features, networks, config_standard, standard_name]
            _evaluate_partial = partial(_evaluate, args)
            results = p.map(_evaluate_partial, range(config_standard["samples"]))

        # take the results and store into the results table
        for res in results:
            evaluations.extend(res)

    evaluations = pd.DataFrame(
        evaluations, columns=["Standard", "Dataset", "Module Match Score (AMI)"]
    )
    State.module_detection_evaluations = evaluations

    # output results
    evaluations.to_csv(
        State.result_path / Path(f"{State.config_name}_module_detection.tsv"), sep="\t", index=False
    )

    if State.plot:
        plot_module_detection()


def import_module_detection_standard(standard: Dict[str, Any]) -> Dict[str, List[str]]:
    with Path(standard["path"]).open("r") as f:
        standard = json.load(f)
        return standard


def reduce_standard(standard: Dict[str, Any]) -> Dict[str, List[str]]:
    new_standard = {}
    consolidated_gene_set = set(State.consolidated_genes)
    for module_name, members in standard.items():
        new_members = [gene for gene in members if gene in consolidated_gene_set]
        if len(new_members) > 1:  # don't include empty or single element modules
            new_standard[module_name] = new_members
    return new_standard


def invert_standard(standard: Dict[str, List[str]]) -> Dict[str, List[str]]:
    inverted_standard = defaultdict(list)
    for module, gene_list in standard.items():
        for gene in gene_list:
            inverted_standard[gene].append(module)
    return inverted_standard


def sample_standard(
    standard: Dict[str, List[str]], inverted_standard: Dict[str, List[str]]
) -> Tuple[List[str], List[int]]:
    """Subsamples the modules in the standard to ensure the resulting module set has
    no overlapping modules (this allows clustering metrics like AMI to be used).
    """

    shared_genes = list(inverted_standard.keys())
    shuffled_genes = np.random.choice(shared_genes, size=len(shared_genes), replace=False)

    # track newly sampled standard and sampled genes
    sampled_standard = {}
    sampled_genes = set()

    for label, gene in enumerate(shuffled_genes):

        # if `gene` has already been sampled, skip it
        if gene in sampled_genes:
            continue

        sampled_module = random.sample(inverted_standard[gene], 1)[0]
        sampled_module_genes = standard[sampled_module]

        # check for overlaps
        in_sampled_genes = [
            True if gene_ in sampled_genes else False for gene_ in sampled_module_genes
        ]

        # if any overlaps exist (i.e. `gene` exists in another module), ignore current
        # `sampled_module`
        if any(in_sampled_genes):
            continue

        # record genes in sampled module and assign these genes to a module label
        for gene_ in sampled_module_genes:
            sampled_genes.add(gene_)
            sampled_standard[gene_] = label

    sampled_genes = list(sampled_genes)
    standard_labels = [sampled_standard[gene_] for gene_ in sampled_genes]
    return sampled_genes, standard_labels


def evaluate_features(features: pd.DataFrame, labels: List[int], config: Dict[str, Any]) -> float:
    # record best AMI score
    best_score = 0

    # iterate over parameter combinations and identify best scoring module set and record score
    for method in config["methods"]:
        for metric in config["metrics"]:
            features_ = pdist(features.values, metric=metric)

            # set NaN values (due to pairwise distance of zero vectors) to 0
            np.nan_to_num(features_, copy=False)

            link = linkage(features_, method=method)
            for t in np.linspace(0, np.max(maxdists(link)), num=config["thresholds"]):
                cluster_labels = fcluster(link, t)
                score = adjusted_mutual_info_score(labels, cluster_labels)
                if score > best_score:
                    best_score = score

    return best_score


def evaluate_network(network: nx.Graph, labels: List[int], config: Dict[str, Any]) -> float:
    # record best AMI score
    best_score = 0

    # create adjacency matrix from network and take reciprocal (distances instead of similarities)
    adjacency = nx.to_pandas_adjacency(network).fillna(0)
    adjacency = adjacency.max().max() - adjacency
    np.fill_diagonal(adjacency.values, 0)

    # iterate over parameter combinations and identify best scoring module set and record score
    for method in config["methods"]:
        for metric in config["metrics"]:
            adjacency_ = pdist(adjacency.values, metric=metric)

            # set NaN values (due to pairwise distance of zero vectors) to 0
            np.nan_to_num(adjacency_, copy=False)

            link = linkage(adjacency_, method=method)
            for t in np.linspace(0, np.max(maxdists(link)), num=config["thresholds"]):
                cluster_labels = fcluster(link, t)
                score = adjusted_mutual_info_score(labels, cluster_labels)
                if score > best_score:
                    best_score = score

    return best_score


def _evaluate(args, _):
    """Wrapper function for running evaluations in in a single process.
    """

    evaluations = []
    standard, inverted_standard, features, networks, config_standard, standard_name = args

    # create sampled standard
    sampled_genes, labels = sample_standard(standard, inverted_standard)

    # reduce features and networks
    features_ = consolidate_features(features, sampled_genes)
    networks_ = consolidate_networks(networks, sampled_genes)

    for name, feat in features_.items():
        ami_score = evaluate_features(feat, labels, config_standard)
        evaluations.append([standard_name, name, ami_score])

    for name, network in networks_.items():
        ami_score = evaluate_network(network, labels, config_standard)
        evaluations.append([standard_name, name, ami_score])

    return evaluations
