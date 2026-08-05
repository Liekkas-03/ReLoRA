from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np

from .task_specs import TaskSpec


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def decode_label_id(text: str, task: TaskSpec) -> int:
    norm = normalize_text(text)
    labels = [normalize_text(label) for label in task.label_names]
    for idx, label in enumerate(labels):
        if norm == label:
            return idx
    for idx, label in enumerate(labels):
        if label and label in norm:
            return idx
    return -1


def accuracy_from_texts(pred_texts: Sequence[str], gold_ids: Sequence[int], task: TaskSpec) -> dict[str, float]:
    pred_ids = [decode_label_id(text, task) for text in pred_texts]
    gold = np.asarray(gold_ids, dtype=np.int64)
    pred = np.asarray(pred_ids, dtype=np.int64)
    valid = pred >= 0
    accuracy = float((pred == gold).mean()) if len(gold) else 0.0
    parse_rate = float(valid.mean()) if len(valid) else 0.0
    return {
        "accuracy": accuracy,
        "parse_rate": parse_rate,
        "num_samples": int(len(gold)),
    }
