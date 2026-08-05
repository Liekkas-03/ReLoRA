from __future__ import annotations

from dataclasses import dataclass


SC_INSTRUCTION = "What is the sentiment of the following paragraph? Choose one from the option.\n"
TC_INSTRUCTION = "What is the topic of the following paragraph? Choose one from the option.\n"


@dataclass(frozen=True)
class TaskSpec:
    name: str
    task_type: str
    dataset_name: str

    @property
    def instruction(self) -> str:
        if self.task_type == "SC":
            return SC_INSTRUCTION
        if self.task_type == "TC":
            return TC_INSTRUCTION
        raise ValueError(f"Unsupported task type: {self.task_type}")


TASK_SPECS: dict[str, TaskSpec] = {
    "agnews": TaskSpec(name="agnews", task_type="TC", dataset_name="agnews"),
    "amazon": TaskSpec(name="amazon", task_type="SC", dataset_name="amazon"),
    "yelp": TaskSpec(name="yelp", task_type="SC", dataset_name="yelp"),
    "dbpedia": TaskSpec(name="dbpedia", task_type="TC", dataset_name="dbpedia"),
    "yahoo": TaskSpec(name="yahoo", task_type="TC", dataset_name="yahoo"),
}

TASK_ALIASES = {
    "ag_news": "agnews",
    "agnews": "agnews",
}


def normalize_task_name(task_name: str) -> str:
    return TASK_ALIASES.get(task_name, task_name)


def get_task_spec(task_name: str) -> TaskSpec:
    task_name = normalize_task_name(task_name)
    try:
        return TASK_SPECS[task_name]
    except KeyError as exc:
        valid = ", ".join(sorted(TASK_SPECS))
        raise ValueError(f"Unknown task '{task_name}'. Valid tasks: {valid}") from exc
