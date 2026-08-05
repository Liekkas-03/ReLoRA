from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DataConfig:
    max_source_length: int = 512
    max_target_length: int = 16
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    num_proc: int = 1
    trust_remote_code: bool = False


@dataclass
class LoraConfigValues:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


@dataclass
class TrainConfig:
    num_train_epochs: float = 3
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.03
    logging_steps: int = 20
    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"
    predict_with_generate: bool = True
    generation_max_new_tokens: int = 16
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = False
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    overwrite_output_dir: bool = False
    save_total_limit: int = 2
    report_to: str = "none"


@dataclass
class ExperimentConfig:
    run_name: str
    base_model_name_or_path: str
    output_root: str
    dataset_cache_dir: str | None = None
    model_cache_dir: str | None = None
    seed: int = 42
    tasks: list[str] = field(default_factory=list)
    data: DataConfig = field(default_factory=DataConfig)
    lora: LoraConfigValues = field(default_factory=LoraConfigValues)
    training: TrainConfig = field(default_factory=TrainConfig)

    @property
    def run_output_dir(self) -> Path:
        return Path(self.output_root) / self.run_name


def _pick(data: dict[str, Any], key: str, cls: type) -> Any:
    value = data.get(key, {})
    if isinstance(value, cls):
        return value
    return cls(**value)


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    raw["data"] = _pick(raw, "data", DataConfig)
    raw["lora"] = _pick(raw, "lora", LoraConfigValues)
    raw["training"] = _pick(raw, "training", TrainConfig)
    return ExperimentConfig(**raw)
