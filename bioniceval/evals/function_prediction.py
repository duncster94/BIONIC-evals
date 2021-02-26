from pathlib import Path
from typing import Dict, Union, List
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.exceptions import ConvergenceWarning
from scipy.sparse import csr_matrix

from ..state import State
from ..utils.file_utils import consolidate_features, consolidate_networks
from ..plotting.plotting import plot_function_prediction


def function_prediction_eval():
    """Performs gene function prediction on the feature and networks datasets."""

    # keep track of evaluations from standards
    evaluations = []

    # perform evaluation for each standard
    for config_standard in State.config_standards:
        if config_standard["task"] != "function_prediction":
            continue

        # import standard
        standard = import_function_prediction_standard(config_standard)

        # reduce standard to only include genes common to standard and datasets
        shared_genes = np.intersect1d(list(standard.index), State.consolidated_genes)
        standard = standard.reindex(shared_genes)

        # reduce datasets to only include these common genes as well
        features = consolidate_features(State.features, shared_genes)
        networks = consolidate_networks(State.networks, shared_genes)

        # evaluate feature sets and networks
        for name, feat in features.items():
            scores = evaluate_features(feat, standard, config_standard)
            for score in scores:
                evaluations.append([config_standard["name"], name, score])

        for name, net in networks.items():
            scores = evaluate_network(net, standard, config_standard)
            for score in scores:
                evaluations.append([config_standard["name"], name, score])

    evaluations = pd.DataFrame(
        evaluations, columns=["Standard", "Dataset", "Function Prediction Score (Micro F1)"]
    )
    State.function_prediction_evaluations = evaluations

    # save results
    evaluations.to_csv(
        State.result_path / Path(f"{State.config_name}_function_prediction.tsv"), sep="\t"
    )

    if State.plot:
        plot_function_prediction()


def import_function_prediction_standard(standard: Dict[str, Union[str, Path]]) -> pd.DataFrame:

    standard = pd.read_csv(standard["path"], sep=standard["delimiter"], header=None)

    # construct dictionary from standard
    standard_dict: Dict[str, List[str]] = defaultdict(list)
    for _, gene, label in standard.itertuples():
        standard_dict[gene].append(label)

    # map class labels to multi-hot encodings
    mlb = MultiLabelBinarizer()
    multi_hot_encoding = mlb.fit_transform(list(standard_dict.values()))
    standard = pd.DataFrame(multi_hot_encoding, index=list(standard_dict.keys()))
    return standard


def evaluate_network(network: nx.Graph, standard: pd.DataFrame, config: dict) -> List[float]:

    # scale network features and then reduce using PCA
    adjacency = nx.to_pandas_adjacency(network)
    scaler = StandardScaler()
    pca = PCA(n_components=min(128, adjacency.shape[1]))
    adjacency = pd.DataFrame(
        scaler.fit_transform(pca.fit_transform(adjacency.values)), index=adjacency.index
    )

    # evaluate network
    scores = core_eval(adjacency, standard, config)
    return scores


def evaluate_features(features: pd.DataFrame, standard: pd.DataFrame, config: dict) -> List[float]:

    # scale features and then reduce using PCA
    scaler = StandardScaler()
    pca = PCA(n_components=min(128, features.shape[1]))
    features = pd.DataFrame(
        scaler.fit_transform(pca.fit_transform(features.values)), index=features.index
    )

    # evaluate features
    scores = core_eval(features, standard, config)
    return scores


def core_eval(dataset: pd.DataFrame, standard: pd.DataFrame, config: dict) -> List[float]:
    """Performs N trials of SVM-based K-fold cross validation."""

    # make parameter space from config
    gamma_config = config["gamma"]
    reg_config = config["regularization"]  # regularization config
    gamma = np.linspace(gamma_config["minimum"], gamma_config["maximum"], gamma_config["samples"])
    regularization = np.linspace(
        reg_config["minimum"], reg_config["maximum"], reg_config["samples"]
    )

    # track scores
    scores = []

    for trial in range(config["trials"]):

        # Get train-test split. Random state ensures all datasets get the
        # same splits so comparison is fair.
        X_train, X_test, y_train, y_test = train_test_split(
            dataset.values,
            standard.values,
            test_size=config["test_size"],
            random_state=4242 * trial,
        )

        # https://stackoverflow.com/questions/12632992/gridsearch-for-an-estimator-inside-a-onevsrestclassifier
        estimator = OneVsRestClassifier(SVC(cache_size=300))

        parameters = {"estimator__gamma": gamma, "estimator__C": regularization}

        model = GridSearchCV(
            estimator,
            param_grid=parameters,
            cv=min(X_train.shape[0], config["folds"]),
            n_jobs=-1,
            verbose=1,
            scoring="f1_micro",
            refit="f1_micro",
        )

        y_train = csr_matrix(y_train)
        model.fit(X_train, y_train)

        y_pred = csr_matrix(model.predict(X_test))
        y_test = csr_matrix(y_test)
        # acc = f1_score(y_test, y_pred, average="samples")
        # macro_f1 = f1_score(y_test, y_pred, average="macro")
        micro_f1 = f1_score(y_test, y_pred, average="micro")

        scores.append(micro_f1)

    return scores
