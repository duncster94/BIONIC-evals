import json
from typing import List
from pathlib import Path
from ..state import State


def process_config(exclude_tasks: List[str], exclude_standards: List[str], baseline_path: Path):
    with State.config_path.open() as f:
        config = json.load(f)
        if not baseline_path == Path(''):
            # Read each row of the baseline genes list and save them to a list if a baseline file is specified
            State.baseline = [line.strip() for line in baseline_path.open()]
        for key, value in config.items():
            if key == "standards":
                # filter out excluded tasks and standards
                value = [
                    standard
                    for standard in value
                    if standard["task"] not in exclude_tasks
                       and standard["name"] not in exclude_standards
                ]

            if key == "features":
                State.config_features = value
            if key == "networks":
                State.config_networks = value
            if key == "standards":
                State.config_standards = value
            if key == "consolidation":
                State.consolidation = value
            if key == "plot":
                State.plot = value
            if key == "result_path":
                State.result_path = value
