from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase

from .config import DataConfig

if TYPE_CHECKING:
    from datasets import Dataset


def get_llm_query(sequence_hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    """PCLR-style masked mean pooling followed by L2 normalization."""
    output = sequence_hidden_states
    if attention_mask is not None:
        mask = attention_mask.to(device=output.device, dtype=output.dtype)
        masked_hidden_states = output * mask.unsqueeze(-1)
        valid_sums = masked_hidden_states.sum(dim=1)
        valid_lengths = mask.sum(dim=1).unsqueeze(-1)
        valid_lengths = valid_lengths.clamp(min=1.0)
        query = valid_sums / valid_lengths
    else:
        query = torch.mean(output, dim=1)
    return F.normalize(query, p=2, dim=-1)


def _iter_active_rank_pool_modules(model):
    for module in model.modules():
        if hasattr(module, "active_rank_mask") and hasattr(module, "set_active_ranks"):
            yield module


def _disable_rank_pool_adapters(model) -> list[tuple[object, torch.Tensor]]:
    snapshots = []
    for module in _iter_active_rank_pool_modules(model):
        snapshots.append((module, module.active_rank_mask.detach().clone()))
        module.set_active_ranks([])
    return snapshots


def _restore_rank_pool_adapters(snapshots: list[tuple[object, torch.Tensor]]) -> None:
    for module, active_rank_mask in snapshots:
        module.active_rank_mask.copy_(active_rank_mask)


def _forward_backbone_last_hidden(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    backbone = getattr(model, "model", None)
    if backbone is not None:
        outputs = backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        return outputs[0]

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    if not getattr(outputs, "hidden_states", None):
        raise RuntimeError("Model did not return hidden_states for task query extraction.")
    return outputs.hidden_states[-1]


@torch.no_grad()
def compute_task_query(
    model,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Dataset,
    data_cfg: DataConfig,
    num_samples: int,
    batch_size: int = 8,
) -> torch.Tensor:
    from .data import build_instance_instruction, build_model_prompt

    model_was_training = model.training
    model.eval()
    samples = [dataset[idx] for idx in range(min(num_samples, len(dataset)))]
    if not samples:
        raise ValueError("Cannot compute task query from an empty dataset.")

    queries: list[torch.Tensor] = []
    device = next(model.parameters()).device
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    rank_pool_snapshots = _disable_rank_pool_adapters(model)

    try:
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            prompts = []
            for example in batch:
                instruction = build_instance_instruction(example, data_cfg)
                prompts.append(build_model_prompt(tokenizer, instruction))
            tokenized = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=data_cfg.max_seq_length,
                return_tensors="pt",
                add_special_tokens=False,
            )
            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)
            last_hidden_state = _forward_backbone_last_hidden(model, input_ids, attention_mask)
            queries.append(get_llm_query(last_hidden_state, attention_mask).detach().cpu())
    finally:
        _restore_rank_pool_adapters(rank_pool_snapshots)
        tokenizer.padding_side = old_padding_side
        if model_was_training:
            model.train()

    query = torch.cat(queries, dim=0).mean(dim=0)
    return F.normalize(query.float(), p=2, dim=0)
