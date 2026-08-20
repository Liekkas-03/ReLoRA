from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data import get_train_eval_raw
from src.eval_generation import evaluate_generation_model
from src.modeling import load_base_causal_lm, load_tokenizer
from src.rank_pool import RankPoolState
from src.rank_pool_lora import (
    freeze_non_lora_parameters,
    load_rank_pool_weights,
    replace_with_rank_pool_lora,
    set_active_groups,
    set_trainable_groups,
)
from src.task_query import compute_task_query
from src.task_specs import get_task_spec
from src.utils import safe_mkdir, save_json, set_global_seed, split_task_arg


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_rank_pool(cfg, checkpoint: Path, state_path: Path, tasks: list[str], output_root: Path) -> None:
    tokenizer = load_tokenizer(cfg)
    model = load_base_causal_lm(cfg)
    replace_with_rank_pool_lora(
        model,
        target_modules=cfg.rank_pool.target_modules,
        global_rank=cfg.rank_pool.global_rank,
        group_rank=cfg.rank_pool.group_rank,
        lora_alpha=cfg.rank_pool.lora_alpha,
        lora_dropout=cfg.rank_pool.lora_dropout,
    )
    freeze_non_lora_parameters(model)
    model.config.pad_token_id = tokenizer.pad_token_id

    checkpoint_state = load_rank_pool_weights(model, checkpoint)
    state_payload = _load_json(state_path)
    rank_pool = RankPoolState.from_dict(state_payload, checkpoint_state["group_keys"])

    if not (cfg.training.load_in_8bit or cfg.training.load_in_4bit):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    safe_mkdir(output_root)
    all_metrics: dict[str, dict] = {}
    all_routes: dict[str, dict] = {}
    for task_name in tasks:
        task = get_task_spec(task_name)
        train_raw, _, _ = get_train_eval_raw(task, cfg.data)
        query = compute_task_query(
            model=model,
            tokenizer=tokenizer,
            dataset=train_raw,
            data_cfg=cfg.data,
            num_samples=cfg.rank_pool.query_samples,
        )
        route = rank_pool.route(query, cfg.rank_pool.top_k_groups)
        set_active_groups(model, route.selected_groups, cfg.rank_pool.group_rank)
        set_trainable_groups(model, [], cfg.rank_pool.group_rank)
        metrics = evaluate_generation_model(
            cfg=cfg,
            task_name=task.name,
            model=model,
            tokenizer=tokenizer,
            task_output_dir=output_root / task.name,
        )
        all_metrics[task.name] = metrics
        all_routes[task.name] = route.to_dict()
        print(f"{task.name}: route={route.selected_groups}, metrics={metrics}")

    save_json(output_root / "rank_pool_eval_metrics.json", all_metrics)
    save_json(output_root / "rank_pool_eval_routes.json", all_routes)
    print(f"Saved metrics to {output_root / 'rank_pool_eval_metrics.json'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    parser.add_argument("--checkpoint", default=None, help="Path to rank_pool_model.safetensors.")
    parser.add_argument("--state", default=None, help="Path to rank_pool_state.json.")
    parser.add_argument("--tasks", default=None, help="Comma-separated task names. Defaults to config tasks.")
    parser.add_argument("--output_root", default=None, help="Directory for prediction files and metrics.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg.seed)
    tasks = split_task_arg(args.tasks, cfg.tasks)
    checkpoint = Path(args.checkpoint) if args.checkpoint else cfg.run_output_dir / "final_rank_pool_model.safetensors"
    state_path = Path(args.state) if args.state else cfg.run_output_dir / "final_rank_pool_state.json"
    output_root = Path(args.output_root) if args.output_root else cfg.run_output_dir / "eval_final"
    evaluate_rank_pool(cfg, checkpoint, state_path, tasks, output_root)


if __name__ == "__main__":
    main()
