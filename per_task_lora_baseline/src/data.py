from __future__ import annotations

from typing import Any

from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase

from .config import DataConfig
from .task_specs import TaskSpec


def load_raw_task_dataset(task: TaskSpec, data_cfg: DataConfig, cache_dir: str | None = None):
    kwargs: dict[str, Any] = {
        "path": task.hf_path,
        "cache_dir": cache_dir,
        "trust_remote_code": data_cfg.trust_remote_code,
    }
    if task.hf_name is not None:
        kwargs["name"] = task.hf_name
    return load_dataset(**kwargs)


def _select_max_samples(dataset: Dataset, max_samples: int | None) -> Dataset:
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    return dataset.select(range(max_samples))


def get_train_eval_raw(task: TaskSpec, data_cfg: DataConfig, cache_dir: str | None = None):
    raw = load_raw_task_dataset(task, data_cfg, cache_dir=cache_dir)
    train = _select_max_samples(raw[task.train_split], data_cfg.max_train_samples)
    eval_ds = _select_max_samples(raw[task.eval_split], data_cfg.max_eval_samples)
    return train, eval_ds


def build_chat_prompt(tokenizer: PreTrainedTokenizerBase, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt + "\n"


def preprocess_causal_lm_dataset(
    dataset: Dataset,
    task: TaskSpec,
    tokenizer: PreTrainedTokenizerBase,
    data_cfg: DataConfig,
) -> Dataset:
    label_column = task.label_column
    eos = tokenizer.eos_token or ""

    def convert_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
        size = len(batch[label_column])
        all_input_ids = []
        all_attention_mask = []
        all_labels = []
        gold_ids = []
        for idx in range(size):
            example = {key: values[idx] for key, values in batch.items()}
            label_id = int(example[label_column])
            prompt = build_chat_prompt(tokenizer, task.build_prompt(example))
            answer = task.label_text(label_id) + eos

            prompt_ids = tokenizer(
                prompt,
                add_special_tokens=False,
                max_length=data_cfg.max_source_length,
                truncation=True,
            )["input_ids"]
            answer_ids = tokenizer(
                answer,
                add_special_tokens=False,
                max_length=data_cfg.max_target_length,
                truncation=True,
            )["input_ids"]
            input_ids = prompt_ids + answer_ids
            labels = [-100] * len(prompt_ids) + answer_ids

            all_input_ids.append(input_ids)
            all_attention_mask.append([1] * len(input_ids))
            all_labels.append(labels)
            gold_ids.append(label_id)

        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_mask,
            "labels": all_labels,
            "gold_label": gold_ids,
        }

    remove_columns = list(dataset.column_names)
    return dataset.map(
        convert_batch,
        batched=True,
        remove_columns=remove_columns,
        num_proc=data_cfg.num_proc,
        desc=f"Tokenizing {task.name}",
    )


def preprocess_generation_dataset(
    dataset: Dataset,
    task: TaskSpec,
    tokenizer: PreTrainedTokenizerBase,
    data_cfg: DataConfig,
) -> Dataset:
    label_column = task.label_column

    def convert_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
        size = len(batch[label_column])
        all_input_ids = []
        all_attention_mask = []
        gold_ids = []
        for idx in range(size):
            example = {key: values[idx] for key, values in batch.items()}
            label_id = int(example[label_column])
            prompt = build_chat_prompt(tokenizer, task.build_prompt(example))
            encoded = tokenizer(
                prompt,
                add_special_tokens=False,
                max_length=data_cfg.max_source_length,
                truncation=True,
            )
            all_input_ids.append(encoded["input_ids"])
            all_attention_mask.append(encoded["attention_mask"])
            gold_ids.append(label_id)
        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_mask,
            "gold_label": gold_ids,
        }

    remove_columns = list(dataset.column_names)
    return dataset.map(
        convert_batch,
        batched=True,
        remove_columns=remove_columns,
        num_proc=data_cfg.num_proc,
        desc=f"Preparing generation inputs for {task.name}",
    )


class CausalLMCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        features = [dict(feature) for feature in features]
        labels = [feature.pop("labels") for feature in features]
        for feature in features:
            feature.pop("gold_label", None)
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")

        max_len = batch["input_ids"].shape[1]
        padded_labels = []
        for label in labels:
            pad_len = max_len - len(label)
            if self.tokenizer.padding_side == "left":
                padded = [-100] * pad_len + label
            else:
                padded = label + [-100] * pad_len
            padded_labels.append(padded)
        import torch

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


class GenerationCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        features = [dict(feature) for feature in features]
        gold = [feature.pop("gold_label") for feature in features]
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        import torch

        batch["gold_label"] = torch.tensor(gold, dtype=torch.long)
        return batch
