import json

from ..state import State


def process_config(exclude_tasks: List[str], exclude_standards: List[str]):
    with State.config_path.open() as f:
        config = json.load(f)
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
