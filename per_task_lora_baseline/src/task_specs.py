from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TaskSpec:
    name: str
    hf_path: str
    hf_name: str | None
    label_column: str
    train_split: str
    eval_split: str
    label_names: list[str]
    prompt_template: str

    def label_text(self, label_id: int) -> str:
        return self.label_names[int(label_id)]

    def build_prompt(self, example: dict[str, Any]) -> str:
        return self.prompt_template.format(**example)


TASK_SPECS: dict[str, TaskSpec] = {
    "ag_news": TaskSpec(
        name="ag_news",
        hf_path="ag_news",
        hf_name=None,
        label_column="label",
        train_split="train",
        eval_split="test",
        label_names=["world", "sports", "business", "science and technology"],
        prompt_template=(
            "Classify the news topic.\n"
            "Choices: world; sports; business; science and technology.\n"
            "Text: {text}\n"
            "Answer:"
        ),
    ),
    "amazon": TaskSpec(
        name="amazon",
        hf_path="amazon_polarity",
        hf_name=None,
        label_column="label",
        train_split="train",
        eval_split="test",
        label_names=["negative", "positive"],
        prompt_template=(
            "Classify the sentiment of this Amazon product review.\n"
            "Choices: negative; positive.\n"
            "Title: {title}\n"
            "Review: {content}\n"
            "Answer:"
        ),
    ),
    "yelp": TaskSpec(
        name="yelp",
        hf_path="yelp_polarity",
        hf_name=None,
        label_column="label",
        train_split="train",
        eval_split="test",
        label_names=["negative", "positive"],
        prompt_template=(
            "Classify the sentiment of this Yelp restaurant review.\n"
            "Choices: negative; positive.\n"
            "Review: {text}\n"
            "Answer:"
        ),
    ),
    "dbpedia": TaskSpec(
        name="dbpedia",
        hf_path="dbpedia_14",
        hf_name=None,
        label_column="label",
        train_split="train",
        eval_split="test",
        label_names=[
            "company",
            "educational institution",
            "artist",
            "athlete",
            "office holder",
            "means of transportation",
            "building",
            "natural place",
            "village",
            "animal",
            "plant",
            "album",
            "film",
            "written work"
        ],
        prompt_template=(
            "Classify the topic of the DBpedia text.\n"
            "Choices: company; educational institution; artist; athlete; "
            "office holder; means of transportation; building; natural place; "
            "village; animal; plant; album; film; written work.\n"
            "Title: {title}\n"
            "Text: {content}\n"
            "Answer:"
        ),
    ),
    "yahoo": TaskSpec(
        name="yahoo",
        hf_path="yahoo_answers_topics",
        hf_name=None,
        label_column="topic",
        train_split="train",
        eval_split="test",
        label_names=[
            "society and culture",
            "science and mathematics",
            "health",
            "education and reference",
            "computers and internet",
            "sports",
            "business and finance",
            "entertainment and music",
            "family and relationships",
            "politics and government"
        ],
        prompt_template=(
            "Classify the Yahoo Answers topic.\n"
            "Choices: society and culture; science and mathematics; health; "
            "education and reference; computers and internet; sports; "
            "business and finance; entertainment and music; family and relationships; "
            "politics and government.\n"
            "Question title: {question_title}\n"
            "Question content: {question_content}\n"
            "Best answer: {best_answer}\n"
            "Answer:"
        ),
    ),
}


def get_task_spec(task_name: str) -> TaskSpec:
    try:
        return TASK_SPECS[task_name]
    except KeyError as exc:
        valid = ", ".join(sorted(TASK_SPECS))
        raise ValueError(f"Unknown task '{task_name}'. Valid tasks: {valid}") from exc
