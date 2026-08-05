from __future__ import annotations

from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ExperimentConfig


def load_tokenizer(cfg: ExperimentConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model_name_or_path,
        cache_dir=cfg.model_cache_dir,
        trust_remote_code=cfg.data.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_causal_lm(cfg: ExperimentConfig):
    train_cfg = cfg.training
    kwargs: dict[str, Any] = {
        "cache_dir": cfg.model_cache_dir,
        "trust_remote_code": cfg.data.trust_remote_code,
    }
    if train_cfg.load_in_8bit:
        kwargs["load_in_8bit"] = True
        kwargs["device_map"] = "auto"
    elif train_cfg.load_in_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = torch.float16 if train_cfg.fp16 else torch.float32

    model = AutoModelForCausalLM.from_pretrained(cfg.base_model_name_or_path, **kwargs)
    model.config.pad_token_id = model.config.pad_token_id or model.config.eos_token_id
    if train_cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.config.use_cache = False
    if train_cfg.load_in_8bit or train_cfg.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    return model


def attach_new_lora(model, cfg: ExperimentConfig):
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model
