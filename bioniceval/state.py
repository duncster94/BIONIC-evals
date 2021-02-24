import pandas as pd
import networkx as nx
from pathlib import Path
from typing import List, Union, Dict


class State:
    """Keeps track of the evaluation state."""

    config_path: Path
    config_name: Path
    config_features: List[Dict[str, Union[str, Path]]]
    config_networks: List[Dict[str, Union[str, Path]]]
    config_standards: List[Dict[str, Union[str, Path]]]
    result_path: Path = Path("bioniceval/results")
    consolidation: str
    plot: bool = True

    features: Dict[str, pd.DataFrame]  # actual feature sets evaluated by the library
    networks: Dict[str, nx.Graph]  # actual networks evaluated by the library
    consolidated_genes: List[
        str
    ]  # shared genes if `consolidation` == "intersection", all genes otherwise

    coannotation_evaluations: pd.DataFrame
