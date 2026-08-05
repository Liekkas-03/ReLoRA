from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_task_arg(tasks_arg: str | None, default_tasks: Iterable[str]) -> list[str]:
    if not tasks_arg:
        return list(default_tasks)
    tasks = [item.strip() for item in tasks_arg.split(",")]
    return [item for item in tasks if item]


def save_json(path: str | Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def safe_mkdir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
