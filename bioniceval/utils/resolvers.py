from pathlib import Path
from typing import List

from ..state import State


def resolve_config_path(path: Path):
    if path == Path(path.name):
        path = Path("bioniceval/config") / path
    name = path.stem
    State.config_path = path
    State.config_name = name


def resolve_tasks() -> List[str]:
    tasks = list(set([standard["task"] for standard in State.config_standards]))
    return tasks