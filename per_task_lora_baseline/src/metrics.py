from __future__ import annotations

import string
from collections.abc import Sequence

try:
    from rouge_score import rouge_scorer
except ImportError:  # pragma: no cover - only used when optional dependency is absent.
    rouge_scorer = None


ANSWER_PREFIX = "Answer:"


def normalize_answer(text: str) -> str:
    """Lower text, remove punctuation, and fix whitespace, matching O-LoRA."""

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    return " ".join(remove_punc(text.lower()).split())


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def rouge_score(prediction: str, ground_truth: str, rouge_type: str) -> float:
    if rouge_scorer is None:
        return 0.0
    scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)
    return scorer.score(prediction=prediction, target=ground_truth)[rouge_type].fmeasure


def strip_answer_prefix(text: str) -> str:
    if ANSWER_PREFIX in text:
        return text.split(ANSWER_PREFIX)[-1].strip()
    return ""


def compute_olora_metrics(predictions: Sequence[str], references: Sequence[str]) -> dict[str, float]:
    if len(predictions) != len(references):
        raise ValueError(
            f"# of predictions {len(predictions)} does not match # of references {len(references)}."
        )
    if not references:
        return {"exact_match": 0.0, "rouge1": 0.0, "rougeL": 0.0}

    exact_match = 0.0
    rouge1 = 0.0
    rouge_l = 0.0
    for prediction, reference in zip(predictions, references):
        exact_match += float(exact_match_score(prediction, reference))
        rouge1 += rouge_score(prediction, reference, "rouge1")
        rouge_l += rouge_score(prediction, reference, "rougeL")

    total = len(references)
    return {
        "exact_match": round(100.0 * exact_match / total, 4),
        "rouge1": round(100.0 * rouge1 / total, 4),
        "rougeL": round(100.0 * rouge_l / total, 4),
    }
