from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from .config import DataConfig
from .task_specs import TaskSpec


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_task_dir(task: TaskSpec, data_cfg: DataConfig) -> Path:
    task_dir = Path(data_cfg.benchmark_data_dir) / task.task_type / task.dataset_name
    if not task_dir.exists():
        raise FileNotFoundError(
            f"O-LoRA benchmark task directory not found: {task_dir}\n"
            "Prepare it on AutoDL with scripts/prepare_olora_benchmark.py, "
            "or set data.benchmark_data_dir in the config."
        )
    return task_dir


def load_task_labels(task: TaskSpec, data_cfg: DataConfig) -> list[str]:
    labels_path = get_task_dir(task, data_cfg) / "labels.json"
    if not labels_path.exists():
        raise FileNotFoundError(f"labels.json not found: {labels_path}")
    labels = _read_json(labels_path)
    if not isinstance(labels, list):
        raise ValueError(f"labels.json must contain a list: {labels_path}")
    return [str(label) for label in labels]


def _sample_instances(
    records: list[dict[str, Any]],
    max_samples: int | None,
    shuffle: bool,
    seed: int,
) -> list[dict[str, Any]]:
    records = list(records)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(records)
    if max_samples is None or max_samples >= len(records):
        return records
    return records[:max_samples]


def _build_olora_instruction(task: TaskSpec, labels: list[str]) -> str:
    labels_str = ", ".join(labels)
    return task.instruction + "Option: " + labels_str + " \n" + "{0}" + "\nAnswer:"


def _build_olora_examples(
    task: TaskSpec,
    records: list[dict[str, Any]],
    labels: list[str],
    subset: str,
) -> list[dict[str, Any]]:
    instruction = _build_olora_instruction(task, labels)
    examples: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        if "sentence" not in record or "label" not in record:
            raise ValueError("O-LoRA CL_Benchmark records must contain 'sentence' and 'label'.")
        label = str(record["label"])
        examples.append(
            {
                "Task": task.task_type,
                "Dataset": task.dataset_name,
                "Samples": [],
                "subset": subset,
                "Instance": {
                    "id": str(idx),
                    "sentence": str(record["sentence"]),
                    "label": label,
                    "ground_truth": label,
                    "instruction": instruction,
                },
            }
        )
    return examples


def load_task_split(task: TaskSpec, data_cfg: DataConfig, split: str, max_samples: int | None) -> Dataset:
    task_dir = get_task_dir(task, data_cfg)
    split_path = task_dir / f"{split}.json"
    if not split_path.exists():
        raise FileNotFoundError(f"{split}.json not found: {split_path}")
    records = _read_json(split_path)
    if not isinstance(records, list):
        raise ValueError(f"{split}.json must contain a list: {split_path}")
    labels = load_task_labels(task, data_cfg)
    records = _sample_instances(records, max_samples, data_cfg.shuffle, data_cfg.data_seed)
    examples = _build_olora_examples(task, records, labels, subset=split)
    return Dataset.from_list(examples)


def get_train_eval_raw(task: TaskSpec, data_cfg: DataConfig) -> tuple[Dataset, Dataset, list[str]]:
    train_max = data_cfg.train_samples_per_task
    if train_max is None:
        train_max = data_cfg.max_train_samples
    train = load_task_split(task, data_cfg, "train", train_max)
    eval_ds = load_task_split(task, data_cfg, "test", data_cfg.max_eval_samples)
    labels = load_task_labels(task, data_cfg)
    return train, eval_ds, labels


def build_instance_instruction(example: dict[str, Any], data_cfg: DataConfig) -> str:
    instance = example["Instance"]
    instruction = instance["instruction"]
    content = instance["sentence"]
    prefix = ""
    if data_cfg.add_task_name:
        prefix += "Task:" + example["Task"] + "\n"
    if data_cfg.add_dataset_name:
        prefix += "Dataset:" + example["Dataset"] + "\n"
    if prefix:
        instruction = prefix + instruction
    return instruction.format(content)


def build_model_prompt(tokenizer: PreTrainedTokenizerBase, instruction: str) -> str:
    bos = tokenizer.bos_token or ""
    return bos + instruction


def _tokenize_no_specials(tokenizer: PreTrainedTokenizerBase, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


class OLoRADecoderCollator:
    """Qwen-compatible collator mirroring O-LoRA's decoder prompt/label layout."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        data_cfg: DataConfig,
        train: bool,
        label_pad_token_id: int = -100,
    ):
        self.tokenizer = tokenizer
        self.data_cfg = data_cfg
        self.train = train
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        self.tokenizer.padding_side = "left"
        input_ids_list: list[list[int]] = []
        labels_list: list[list[int]] = []
        metadata: list[dict[str, Any]] = []
        limit_input_len = self.data_cfg.max_seq_length

        eos = self.tokenizer.eos_token or ""
        for example in batch:
            instruction = build_instance_instruction(example, self.data_cfg)
            task_input = build_model_prompt(self.tokenizer, instruction)
            input_ids = _tokenize_no_specials(self.tokenizer, task_input)
            label_text = example["Instance"]["label"]

            if self.train:
                label = label_text + eos
                label_ids = _tokenize_no_specials(self.tokenizer, label)
                full_ids = (input_ids + label_ids)[:limit_input_len]
                visible_label_len = max(0, len(full_ids) - min(len(input_ids), len(full_ids)))
                labels = [self.label_pad_token_id] * len(full_ids)
                if visible_label_len:
                    labels[-visible_label_len:] = full_ids[-visible_label_len:]
                input_ids_list.append(full_ids)
                labels_list.append(labels)
            else:
                input_ids_list.append(input_ids[:limit_input_len])

            metadata.append(
                {
                    "Task": example["Task"],
                    "Dataset": example["Dataset"],
                    "Instance": example["Instance"],
                    "instruction": instruction,
                }
            )

        features = [{"input_ids": ids, "attention_mask": [1] * len(ids)} for ids in input_ids_list]
        padded = self.tokenizer.pad(features, padding=True, return_tensors="pt")

        if self.train:
            max_len = padded["input_ids"].shape[1]
            padded_labels = []
            for labels in labels_list:
                pad_len = max_len - len(labels)
                padded_labels.append([self.label_pad_token_id] * pad_len + labels)

            import torch

            padded["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        else:
            padded["metadata"] = metadata
        return padded
