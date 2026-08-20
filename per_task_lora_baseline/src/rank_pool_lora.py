from __future__ import annotations

import math
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from .utils import safe_mkdir


class RankPoolKeyModule(nn.Module):
    def __init__(self, group_keys: torch.Tensor):
        super().__init__()
        if group_keys.ndim != 2:
            raise ValueError(f"group_keys must be 2D, got shape {tuple(group_keys.shape)}.")
        self.group_keys = nn.Parameter(F.normalize(group_keys.detach().clone().float(), p=2, dim=-1))
        self.register_buffer(
            "trainable_group_mask",
            torch.zeros(group_keys.shape[0], dtype=torch.float32),
            persistent=False,
        )
        self.group_keys.register_hook(self._mask_key_grad)

    def _mask_key_grad(self, grad: torch.Tensor) -> torch.Tensor:
        mask = self.trainable_group_mask.to(device=grad.device, dtype=grad.dtype).view(-1, 1)
        return grad * mask

    def set_trainable_groups(self, group_ids: list[int]) -> None:
        mask = torch.zeros(
            self.trainable_group_mask.shape[0],
            device=self.trainable_group_mask.device,
            dtype=self.trainable_group_mask.dtype,
        )
        if group_ids:
            mask[torch.tensor(group_ids, device=mask.device, dtype=torch.long)] = 1.0
        self.trainable_group_mask.copy_(mask)

    def set_group_keys(self, group_keys: torch.Tensor) -> None:
        if tuple(group_keys.shape) != tuple(self.group_keys.shape):
            raise ValueError(f"group_keys shape {tuple(group_keys.shape)} != {tuple(self.group_keys.shape)}.")
        normalized = F.normalize(group_keys.detach().float(), p=2, dim=-1)
        self.group_keys.data.copy_(normalized.to(device=self.group_keys.device, dtype=self.group_keys.dtype))

    def normalized_keys(self) -> torch.Tensor:
        return F.normalize(self.group_keys.float(), p=2, dim=-1)

    def alignment_loss(self, task_query: torch.Tensor, selected_groups: list[int]) -> torch.Tensor:
        if not selected_groups:
            return self.group_keys.sum() * 0.0
        query = F.normalize(task_query.to(device=self.group_keys.device, dtype=torch.float32), p=2, dim=0)
        selected = torch.tensor(selected_groups, device=self.group_keys.device, dtype=torch.long)
        keys = self.normalized_keys().index_select(0, selected)
        scores = torch.einsum("kd,d->k", keys, query)
        return torch.mean(1.0 - scores)


class FixedRankPoolLinear(nn.Module):
    def __init__(
        self,
        base_linear: nn.Linear,
        global_rank: int,
        group_rank: int,
        lora_alpha: int,
        lora_dropout: float,
    ):
        super().__init__()
        if global_rank % group_rank != 0:
            raise ValueError("global_rank must be divisible by group_rank.")
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.global_rank = global_rank
        self.group_rank = group_rank
        self.num_groups = global_rank // group_rank
        self.scaling = float(lora_alpha) / float(global_rank)
        self.weight = nn.Parameter(base_linear.weight.detach().clone(), requires_grad=False)
        if base_linear.bias is not None:
            self.bias = nn.Parameter(base_linear.bias.detach().clone(), requires_grad=False)
        else:
            self.bias = None

        self.lora_A = nn.Parameter(
            torch.empty(
                global_rank,
                self.in_features,
                device=base_linear.weight.device,
                dtype=base_linear.weight.dtype,
            )
        )
        self.lora_B = nn.Parameter(
            torch.zeros(
                self.out_features,
                global_rank,
                device=base_linear.weight.device,
                dtype=base_linear.weight.dtype,
            )
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        self.register_buffer("active_rank_mask", torch.zeros(global_rank, dtype=torch.float32), persistent=False)
        self.register_buffer("trainable_rank_mask", torch.zeros(global_rank, dtype=torch.float32), persistent=False)
        self.lora_A.register_hook(self._mask_lora_a_grad)
        self.lora_B.register_hook(self._mask_lora_b_grad)

    def _mask_lora_a_grad(self, grad: torch.Tensor) -> torch.Tensor:
        mask = self.trainable_rank_mask.to(device=grad.device, dtype=grad.dtype).view(-1, 1)
        return grad * mask

    def _mask_lora_b_grad(self, grad: torch.Tensor) -> torch.Tensor:
        mask = self.trainable_rank_mask.to(device=grad.device, dtype=grad.dtype).view(1, -1)
        return grad * mask

    def set_active_ranks(self, ranks: list[int]) -> None:
        mask = torch.zeros(self.global_rank, device=self.active_rank_mask.device, dtype=self.active_rank_mask.dtype)
        if ranks:
            mask[torch.tensor(ranks, device=mask.device, dtype=torch.long)] = 1.0
        self.active_rank_mask.copy_(mask)

    def set_trainable_ranks(self, ranks: list[int]) -> None:
        mask = torch.zeros(
            self.global_rank,
            device=self.trainable_rank_mask.device,
            dtype=self.trainable_rank_mask.dtype,
        )
        if ranks:
            mask[torch.tensor(ranks, device=mask.device, dtype=torch.long)] = 1.0
        self.trainable_rank_mask.copy_(mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = F.linear(x, self.weight, self.bias)
        active = self.active_rank_mask.to(device=x.device, dtype=x.dtype)
        if torch.count_nonzero(active).item() == 0:
            return result
        dropped = self.dropout(x)
        hidden = F.linear(dropped, self.lora_A)
        hidden = hidden * active
        return result + F.linear(hidden, self.lora_B) * self.scaling


def _get_parent_module(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def replace_with_rank_pool_lora(
    model: nn.Module,
    target_modules: list[str],
    global_rank: int,
    group_rank: int,
    lora_alpha: int,
    lora_dropout: float,
) -> list[str]:
    replaced: list[str] = []
    module_names = [name for name, _ in model.named_modules()]
    for module_name in module_names:
        module = model.get_submodule(module_name) if module_name else model
        if not isinstance(module, nn.Linear):
            continue
        if not any(module_name.endswith(target_name) for target_name in target_modules):
            continue
        parent, child_name = _get_parent_module(model, module_name)
        new_module = FixedRankPoolLinear(
            module,
            global_rank=global_rank,
            group_rank=group_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        setattr(parent, child_name, new_module)
        replaced.append(module_name)
    if not replaced:
        raise ValueError(f"No target modules found for {target_modules}.")
    return replaced


def iter_rank_pool_layers(model: nn.Module) -> Iterator[tuple[str, FixedRankPoolLinear]]:
    for name, module in model.named_modules():
        if isinstance(module, FixedRankPoolLinear):
            yield name, module


def set_active_groups(model: nn.Module, group_ids: list[int], group_rank: int) -> None:
    ranks = []
    for group_id in group_ids:
        ranks.extend(range(group_id * group_rank, (group_id + 1) * group_rank))
    for _, module in iter_rank_pool_layers(model):
        module.set_active_ranks(ranks)


def set_trainable_groups(model: nn.Module, group_ids: list[int], group_rank: int) -> None:
    ranks = []
    for group_id in group_ids:
        ranks.extend(range(group_id * group_rank, (group_id + 1) * group_rank))
    for _, module in iter_rank_pool_layers(model):
        module.set_trainable_ranks(ranks)


def freeze_non_lora_parameters(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = "lora_A" in name or "lora_B" in name or "rank_pool_key_module" in name


def attach_rank_pool_key_module(model: nn.Module, group_keys: torch.Tensor) -> RankPoolKeyModule:
    key_module = RankPoolKeyModule(group_keys)
    setattr(model, "rank_pool_key_module", key_module)
    return key_module


def get_rank_pool_key_module(model: nn.Module) -> RankPoolKeyModule:
    key_module = getattr(model, "rank_pool_key_module", None)
    if key_module is None:
        raise AttributeError("Model does not have rank_pool_key_module.")
    return key_module


def set_trainable_key_groups(model: nn.Module, group_ids: list[int]) -> None:
    get_rank_pool_key_module(model).set_trainable_groups(group_ids)


def count_rank_pool_parameters(model: nn.Module) -> int:
    total = 0
    for _, module in iter_rank_pool_layers(model):
        total += module.lora_A.numel() + module.lora_B.numel()
    return total


def count_effective_trainable_rank_pool_parameters(model: nn.Module) -> int:
    total = 0
    for _, module in iter_rank_pool_layers(model):
        mask = module.trainable_rank_mask.detach().cpu()
        trainable_rank = int(torch.count_nonzero(mask).item())
        total += trainable_rank * (module.in_features + module.out_features)
    return total


def rank_pool_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, module in iter_rank_pool_layers(model):
        state[f"{name}.lora_A"] = module.lora_A.detach().cpu()
        state[f"{name}.lora_B"] = module.lora_B.detach().cpu()
    return state


def clone_rank_pool_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in rank_pool_state_dict(model).items()}


def load_rank_pool_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    missing = []
    for name, module in iter_rank_pool_layers(model):
        key_a = f"{name}.lora_A"
        key_b = f"{name}.lora_B"
        if key_a not in state or key_b not in state:
            missing.append(name)
            continue
        module.lora_A.data.copy_(state[key_a].to(device=module.lora_A.device, dtype=module.lora_A.dtype))
        module.lora_B.data.copy_(state[key_b].to(device=module.lora_B.device, dtype=module.lora_B.dtype))
    if missing:
        raise KeyError(f"Missing rank-pool LoRA weights for modules: {missing[:5]}")


def reset_rank_pool_groups(model: nn.Module, group_ids: list[int], group_rank: int) -> None:
    ranks: list[int] = []
    for group_id in group_ids:
        ranks.extend(range(group_id * group_rank, (group_id + 1) * group_rank))
    if not ranks:
        return
    rank_index = torch.tensor(ranks, dtype=torch.long)
    for _, module in iter_rank_pool_layers(model):
        device_index = rank_index.to(module.lora_A.device)
        with torch.no_grad():
            new_a = torch.empty(
                len(ranks),
                module.in_features,
                device=module.lora_A.device,
                dtype=module.lora_A.dtype,
            )
            nn.init.kaiming_uniform_(new_a, a=math.sqrt(5))
            module.lora_A.data.index_copy_(
                0,
                device_index,
                new_a,
            )
            module.lora_B.data[:, device_index] = 0.0


def save_rank_pool_weights(model: nn.Module, path: str | Path, extra_tensors: dict[str, torch.Tensor] | None = None) -> None:
    path = Path(path)
    safe_mkdir(path.parent)
    state = rank_pool_state_dict(model)
    if extra_tensors:
        state.update({key: value.detach().cpu() for key, value in extra_tensors.items()})
    save_file(state, str(path), metadata={"format": "pt"})


def load_rank_pool_weights(model: nn.Module, path: str | Path) -> dict[str, torch.Tensor]:
    state = load_file(str(path))
    lora_state = {key: value for key, value in state.items() if key.endswith(".lora_A") or key.endswith(".lora_B")}
    load_rank_pool_state_dict(model, lora_state)
    return state
