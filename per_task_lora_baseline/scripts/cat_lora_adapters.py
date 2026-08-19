from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.utils import safe_mkdir, save_json


def read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: dict[str, Any]) -> None:
    safe_mkdir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_target_modules(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return sorted(str(item) for item in value)
    raise ValueError(f"Unsupported target_modules value: {value!r}")


def is_lora_a_key(key: str) -> bool:
    return "lora_A" in key.split(".") and key.endswith("weight")


def is_lora_b_key(key: str) -> bool:
    return "lora_B" in key.split(".") and key.endswith("weight")


def check_adapter_dir(path: Path) -> None:
    missing = []
    for filename in ["adapter_config.json", "adapter_model.safetensors"]:
        if not (path / filename).exists():
            missing.append(filename)
    if missing:
        raise FileNotFoundError(f"{path} is missing: {', '.join(missing)}")


def load_adapters(adapter_paths: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, torch.Tensor]]]:
    configs = []
    states = []
    for path in adapter_paths:
        check_adapter_dir(path)
        configs.append(read_json(path / "adapter_config.json"))
        states.append(load_file(str(path / "adapter_model.safetensors")))
    return configs, states


def check_configs(configs: list[dict[str, Any]], adapter_paths: list[Path]) -> tuple[int, int]:
    first = configs[0]
    r = int(first["r"])
    alpha = int(first["lora_alpha"])
    target_modules = normalize_target_modules(first["target_modules"])
    peft_type = first.get("peft_type")
    task_type = first.get("task_type")
    base_model_name = first.get("base_model_name_or_path")

    if first.get("rank_pattern"):
        raise ValueError("rank_pattern is not empty; fixed-rank concatenation is not supported.")
    if first.get("alpha_pattern"):
        raise ValueError("alpha_pattern is not empty; fixed-alpha concatenation is not supported.")
    if first.get("use_rslora"):
        raise ValueError("use_rslora=True changes scaling; this script expects alpha/r scaling.")
    if first.get("use_dora"):
        raise ValueError("use_dora=True is not supported by rank concatenation.")

    for path, cfg in zip(adapter_paths[1:], configs[1:]):
        if int(cfg["r"]) != r:
            raise ValueError(f"{path} has r={cfg['r']}, expected r={r}.")
        if int(cfg["lora_alpha"]) != alpha:
            raise ValueError(f"{path} has lora_alpha={cfg['lora_alpha']}, expected {alpha}.")
        if normalize_target_modules(cfg["target_modules"]) != target_modules:
            raise ValueError(f"{path} has different target_modules.")
        if cfg.get("peft_type") != peft_type:
            raise ValueError(f"{path} has peft_type={cfg.get('peft_type')}, expected {peft_type}.")
        if cfg.get("task_type") != task_type:
            raise ValueError(f"{path} has task_type={cfg.get('task_type')}, expected {task_type}.")
        if cfg.get("rank_pattern"):
            raise ValueError(f"{path} has non-empty rank_pattern.")
        if cfg.get("alpha_pattern"):
            raise ValueError(f"{path} has non-empty alpha_pattern.")
        if cfg.get("use_rslora"):
            raise ValueError(f"{path} has use_rslora=True.")
        if cfg.get("use_dora"):
            raise ValueError(f"{path} has use_dora=True.")
        other_base = cfg.get("base_model_name_or_path")
        if other_base != base_model_name:
            print(
                "[warn] adapter base_model_name_or_path differs: "
                f"{path} has {other_base!r}, first adapter has {base_model_name!r}"
            )
    return r, alpha


def check_state_keys(states: list[dict[str, torch.Tensor]], adapter_paths: list[Path]) -> list[str]:
    first_keys = set(states[0])
    for path, state in zip(adapter_paths[1:], states[1:]):
        if set(state) != first_keys:
            missing = sorted(first_keys - set(state))
            extra = sorted(set(state) - first_keys)
            raise ValueError(
                f"{path} has different LoRA keys. "
                f"Missing first keys: {missing[:5]}; extra keys: {extra[:5]}"
            )

    lora_keys = sorted(key for key in first_keys if is_lora_a_key(key) or is_lora_b_key(key))
    if not lora_keys:
        raise ValueError("No lora_A/lora_B weights found in adapter_model.safetensors.")

    for key in lora_keys:
        shape = tuple(states[0][key].shape)
        for path, state in zip(adapter_paths[1:], states[1:]):
            if tuple(state[key].shape) != shape:
                raise ValueError(f"{path} key {key} has shape {tuple(state[key].shape)}, expected {shape}.")
    return sorted(first_keys)


def concat_lora_states(states: list[dict[str, torch.Tensor]], all_keys: list[str]) -> dict[str, torch.Tensor]:
    merged: dict[str, torch.Tensor] = {}
    for key in all_keys:
        tensors = [state[key].detach().cpu() for state in states]
        if is_lora_a_key(key):
            merged[key] = torch.cat(tensors, dim=0)
        elif is_lora_b_key(key):
            merged[key] = torch.cat(tensors, dim=1)
        else:
            first = tensors[0]
            if not all(torch.equal(first, tensor) for tensor in tensors[1:]):
                raise ValueError(f"Non-LoRA tensor differs across adapters: {key}")
            merged[key] = first
    return merged


def build_merged_config(configs: list[dict[str, Any]], new_r: int, new_alpha: int) -> dict[str, Any]:
    merged_cfg = dict(configs[0])
    merged_cfg["r"] = new_r
    merged_cfg["lora_alpha"] = new_alpha
    merged_cfg["inference_mode"] = True
    merged_cfg["rank_pattern"] = {}
    merged_cfg["alpha_pattern"] = {}
    return merged_cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Experiment config, used for defaults and merge metadata.")
    parser.add_argument(
        "--adapter_paths",
        nargs="+",
        required=True,
        help="Adapter directories to concatenate in rank dimension.",
    )
    parser.add_argument(
        "--adapter_names",
        nargs="+",
        default=None,
        help="Optional names matching --adapter_paths, saved only in merge_info.json.",
    )
    parser.add_argument(
        "--output_adapter_path",
        default="outputs/merged_lora/cat_rank40_mixed_len/adapter",
        help="Directory for the merged rank-concatenated adapter.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace output_adapter_path if it exists.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    adapter_paths = [Path(path) for path in args.adapter_paths]
    if len(adapter_paths) < 2:
        raise ValueError("Need at least two adapters to concatenate.")
    if args.adapter_names is not None and len(args.adapter_names) != len(adapter_paths):
        raise ValueError("--adapter_names must have the same length as --adapter_paths.")

    output_adapter_path = Path(args.output_adapter_path)
    if output_adapter_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_adapter_path} already exists. Use --overwrite to replace it.")
        shutil.rmtree(output_adapter_path)

    configs, states = load_adapters(adapter_paths)
    r, alpha = check_configs(configs, adapter_paths)
    all_keys = check_state_keys(states, adapter_paths)
    merged_state = concat_lora_states(states, all_keys)

    num_adapters = len(adapter_paths)
    new_r = r * num_adapters
    new_alpha = alpha * num_adapters
    merged_cfg = build_merged_config(configs, new_r, new_alpha)

    safe_mkdir(output_adapter_path)
    save_file(
        merged_state,
        str(output_adapter_path / "adapter_model.safetensors"),
        metadata={"format": "pt"},
    )
    write_json(output_adapter_path / "adapter_config.json", merged_cfg)
    save_json(
        output_adapter_path / "merge_info.json",
        {
            "method": "rank_concat",
            "formula": "A=cat(A_i, dim=0), B=cat(B_i, dim=1), alpha_new/r_new=alpha_old/r_old",
            "adapter_paths": [str(path) for path in adapter_paths],
            "adapter_names": args.adapter_names,
            "old_r": r,
            "old_lora_alpha": alpha,
            "old_scaling": alpha / r,
            "num_adapters": num_adapters,
            "new_r": new_r,
            "new_lora_alpha": new_alpha,
            "new_scaling": new_alpha / new_r,
            "config": args.config,
            "base_model_name_or_path": cfg.base_model_name_or_path,
        },
    )
    print(f"[done] saved rank-concat LoRA adapter to {output_adapter_path}")
    print(f"old r={r}, old alpha={alpha}, old alpha/r={alpha / r}")
    print(f"new r={new_r}, new alpha={new_alpha}, new alpha/r={new_alpha / new_r}")


if __name__ == "__main__":
    main()
