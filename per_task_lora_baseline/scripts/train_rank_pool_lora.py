from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import Trainer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train_per_task_lora import build_training_args
from src.config import load_config
from src.data import OLoRADecoderCollator, get_train_eval_raw
from src.eval_generation import evaluate_generation_model
from src.modeling import load_base_causal_lm, load_tokenizer
from src.rank_pool import RankPoolState
from src.rank_pool_lora import (
    attach_rank_pool_key_module,
    count_effective_trainable_rank_pool_parameters,
    count_rank_pool_parameters,
    freeze_non_lora_parameters,
    get_rank_pool_key_module,
    replace_with_rank_pool_lora,
    save_rank_pool_weights,
    set_active_groups,
    set_trainable_key_groups,
    set_trainable_groups,
)
from src.rank_recycling import recycle_one_group
from src.task_query import compute_task_query
from src.task_specs import get_task_spec
from src.utils import safe_mkdir, save_json, set_global_seed, split_task_arg


def _step_dir(run_dir: Path, step_index: int, task_name: str) -> Path:
    return run_dir / f"step_{step_index:02d}_{task_name}"


def _save_task_queries(path: Path, task_queries: dict[str, torch.Tensor]) -> None:
    if not task_queries:
        return
    safe_mkdir(path.parent)
    save_file({name: value.detach().cpu() for name, value in task_queries.items()}, str(path))


class RankPoolTrainer(Trainer):
    def __init__(
        self,
        *args,
        key_module,
        task_query: torch.Tensor,
        selected_groups: list[int],
        key_loss_weight: float,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.key_module = key_module
        self.task_query = task_query.detach().cpu()
        self.selected_groups = selected_groups
        self.key_loss_weight = key_loss_weight
        self.last_key_loss: float | None = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        if isinstance(outputs, dict):
            loss = outputs["loss"]
        else:
            loss = outputs.loss
        if self.key_loss_weight and self.key_loss_weight > 0:
            key_loss = self.key_module.alignment_loss(self.task_query, self.selected_groups)
            loss = loss + self.key_loss_weight * key_loss
            self.last_key_loss = float(key_loss.detach().cpu())
        return (loss, outputs) if return_outputs else loss


def _build_model_and_pool(cfg):
    tokenizer = load_tokenizer(cfg)
    model = load_base_causal_lm(cfg)
    replaced_modules = replace_with_rank_pool_lora(
        model,
        target_modules=cfg.rank_pool.target_modules,
        global_rank=cfg.rank_pool.global_rank,
        group_rank=cfg.rank_pool.group_rank,
        lora_alpha=cfg.rank_pool.lora_alpha,
        lora_dropout=cfg.rank_pool.lora_dropout,
    )
    freeze_non_lora_parameters(model)
    model.config.pad_token_id = tokenizer.pad_token_id

    hidden_size = model.get_input_embeddings().embedding_dim
    rank_pool = RankPoolState(
        global_rank=cfg.rank_pool.global_rank,
        group_rank=cfg.rank_pool.group_rank,
        hidden_size=hidden_size,
    )
    key_module = attach_rank_pool_key_module(model, rank_pool.group_keys)
    freeze_non_lora_parameters(model)
    if not (cfg.training.load_in_8bit or cfg.training.load_in_4bit):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    key_module.set_group_keys(rank_pool.group_keys)
    return tokenizer, model, rank_pool, replaced_modules


def train_rank_pool(cfg, tasks: list[str]) -> None:
    run_dir = cfg.run_output_dir
    if run_dir.exists() and cfg.training.overwrite_output_dir:
        shutil.rmtree(run_dir)
    safe_mkdir(run_dir)

    tokenizer, model, rank_pool, replaced_modules = _build_model_and_pool(cfg)
    total_rank_pool_params = count_rank_pool_parameters(model)
    save_json(
        run_dir / "rank_pool_setup.json",
        {
            "base_model_name_or_path": cfg.base_model_name_or_path,
            "global_rank": cfg.rank_pool.global_rank,
            "group_rank": cfg.rank_pool.group_rank,
            "num_groups": cfg.rank_pool.num_groups,
            "top_k_groups": cfg.rank_pool.top_k_groups,
            "lora_alpha": cfg.rank_pool.lora_alpha,
            "lora_dropout": cfg.rank_pool.lora_dropout,
            "target_modules": cfg.rank_pool.target_modules,
            "replaced_module_count": len(replaced_modules),
            "rank_pool_trainable_parameter_budget": total_rank_pool_params,
            "rank_key_parameter_count": int(rank_pool.group_keys.numel()),
            "task_query": "frozen base-model last_hidden_state with PCLR-style masked mean pooling",
            "rank_key": "PCLR-style randomly initialized learnable group_keys updated by query-key alignment loss",
            "key_loss_weight": cfg.rank_pool.key_loss_weight,
            "rank_selection": "PCLR-style cosine top-k over all rank group keys during training; top-k over occupied keys during inference.",
            "recycling": {
                "enabled": cfg.recycling.enabled,
                "importance_weight": cfg.recycling.importance_weight,
                "redundancy_weight": cfg.recycling.redundancy_weight,
                "usage_weight": cfg.recycling.usage_weight,
                "replay_samples_per_task": cfg.recycling.replay_samples_per_task,
                "eval_samples_per_task": cfg.recycling.eval_samples_per_task,
                "recovery_steps": cfg.recycling.recovery_steps,
                "distill_temperature": cfg.recycling.distill_temperature,
                "replay_loss_weight": cfg.recycling.replay_loss_weight,
                "epsilon": cfg.recycling.epsilon,
                "max_trials": cfg.recycling.max_trials,
                "recovery_learning_rate": cfg.recycling.recovery_learning_rate,
            },
            "note": "Fixed-rank LoRA pool. Each task activates top-k rank groups; total rank never grows.",
        },
    )

    task_queries: dict[str, torch.Tensor] = {}
    train_raw_by_task: dict[str, object] = {}
    eval_raw_by_task: dict[str, object] = {}
    performance_matrix: dict[str, dict] = {}
    task_order: list[str] = []

    for step_index, task_name in enumerate(tasks, start=1):
        task = get_task_spec(task_name)
        step_dir = _step_dir(run_dir, step_index, task.name)
        if step_dir.exists() and cfg.training.overwrite_output_dir:
            shutil.rmtree(step_dir)
        safe_mkdir(step_dir)

        print(f"\n===== Rank-pool LoRA step {step_index}: {task.name} =====")
        print(f"Base model: {cfg.base_model_name_or_path}")
        print(
            "Pool: "
            f"global_rank={cfg.rank_pool.global_rank}, "
            f"group_rank={cfg.rank_pool.group_rank}, "
            f"groups={cfg.rank_pool.num_groups}, "
            f"top_k={cfg.rank_pool.top_k_groups}"
        )

        train_raw, eval_raw, labels = get_train_eval_raw(task, cfg.data)
        train_raw_by_task[task.name] = train_raw
        eval_raw_by_task[task.name] = eval_raw
        query = compute_task_query(
            model=model,
            tokenizer=tokenizer,
            dataset=train_raw,
            data_cfg=cfg.data,
            num_samples=cfg.rank_pool.query_samples,
        )
        task_queries[task.name] = query.detach().cpu()

        selection = rank_pool.select_for_task(
            task_name=task.name,
            task_query=query,
            top_k_groups=cfg.rank_pool.top_k_groups,
        )
        recycling_results = []
        while rank_pool.missing_groups(selection.selected_groups, cfg.rank_pool.top_k_groups) > 0:
            recycle_slot = len(selection.recycled_groups) + 1
            print(f"Rank pool has no free group for {task.name}; recycling slot {recycle_slot}.")
            recycle_result = recycle_one_group(
                cfg=cfg,
                model=model,
                tokenizer=tokenizer,
                rank_pool=rank_pool,
                seen_tasks=task_order,
                task_queries=task_queries,
                train_raw_by_task=train_raw_by_task,
                eval_raw_by_task=eval_raw_by_task,
                exclude_groups=set(selection.selected_groups),
                output_dir=step_dir / "recycling" / f"slot_{recycle_slot:02d}",
            )
            recycling_results.append(recycle_result.to_dict())
            save_json(step_dir / "recycling" / f"slot_{recycle_slot:02d}" / "recycling_result.json", recycle_result.to_dict())
            if not recycle_result.success or recycle_result.recycled_group is None:
                save_json(step_dir / "recycling" / "failed_recycling_results.json", {"attempts": recycling_results})
                raise RuntimeError(f"Rank recycling failed for task {task.name}.")
            recycled_group = recycle_result.recycled_group
            selection.selected_groups.append(recycled_group)
            selection.recycled_groups.append(recycled_group)

        rank_pool.initialize_free_group_keys(query, selection.free_groups + selection.recycled_groups)
        if selection.recycled_groups:
            rank_pool.assign_groups_to_task(task.name, selection.recycled_groups, rank_pool.score(query))
        get_rank_pool_key_module(model).set_group_keys(rank_pool.group_keys)
        set_active_groups(model, selection.selected_groups, cfg.rank_pool.group_rank)
        set_trainable_groups(model, selection.selected_groups, cfg.rank_pool.group_rank)
        set_trainable_key_groups(model, selection.selected_groups)
        effective_trainable = count_effective_trainable_rank_pool_parameters(model)
        selected_ranks = rank_pool.ranks_for_groups(selection.selected_groups)

        print(f"Selected groups for {task.name}: {selection.selected_groups}")
        print(f"Selected rank ids: {selected_ranks}")
        print(f"Effective trainable LoRA parameters this step: {effective_trainable}")

        save_json(
            step_dir / "selection.json",
            {
                **selection.to_dict(),
                "selected_rank_ids": selected_ranks,
                "labels": labels,
                "effective_trainable_rank_pool_parameters": effective_trainable,
                "free_group_key_init": "task_query",
                "recycled_group_key_init": "task_query",
                "key_loss_weight": cfg.rank_pool.key_loss_weight,
                "recycling": recycling_results,
            },
        )

        data_collator = OLoRADecoderCollator(tokenizer=tokenizer, data_cfg=cfg.data, train=True)
        trainer = RankPoolTrainer(
            model=model,
            args=build_training_args(cfg, step_dir),
            train_dataset=train_raw,
            data_collator=data_collator,
            key_module=get_rank_pool_key_module(model),
            task_query=query,
            selected_groups=selection.selected_groups,
            key_loss_weight=cfg.rank_pool.key_loss_weight,
        )
        train_result = trainer.train()
        train_metrics = dict(train_result.metrics)
        if trainer.last_key_loss is not None:
            train_metrics["last_key_alignment_loss"] = trainer.last_key_loss
        save_json(step_dir / "train_metrics.json", train_metrics)

        if count_rank_pool_parameters(model) != total_rank_pool_params:
            raise RuntimeError("Rank-pool parameter budget changed during training.")
        rank_pool.update_selected_keys(
            selected_groups=selection.selected_groups,
            learned_group_keys=get_rank_pool_key_module(model).normalized_keys(),
        )
        get_rank_pool_key_module(model).set_group_keys(rank_pool.group_keys)
        set_trainable_key_groups(model, [])

        if task.name not in task_order:
            task_order.append(task.name)

        save_rank_pool_weights(
            model,
            step_dir / "rank_pool_model.safetensors",
            extra_tensors={"group_keys": rank_pool.group_keys},
        )
        save_json(step_dir / "rank_pool_state.json", rank_pool.to_dict())
        _save_task_queries(step_dir / "task_queries.safetensors", task_queries)

        step_metrics: dict[str, dict] = {}
        step_routes: dict[str, dict] = {}
        for eval_task_name in task_order:
            route = rank_pool.route(task_queries[eval_task_name], cfg.rank_pool.top_k_groups)
            set_active_groups(model, route.selected_groups, cfg.rank_pool.group_rank)
            set_trainable_groups(model, [], cfg.rank_pool.group_rank)
            step_routes[eval_task_name] = route.to_dict()
            metrics = evaluate_generation_model(
                cfg=cfg,
                task_name=eval_task_name,
                model=model,
                tokenizer=tokenizer,
                task_output_dir=step_dir / "eval" / eval_task_name,
            )
            step_metrics[eval_task_name] = metrics
            print(f"after {task.name} -> {eval_task_name}: {metrics}")

        performance_matrix[f"after_{step_index:02d}_{task.name}"] = step_metrics
        save_json(step_dir / "eval_routes.json", step_routes)
        save_json(step_dir / "seen_task_metrics.json", step_metrics)
        save_json(run_dir / "performance_matrix.json", performance_matrix)
        save_json(run_dir / "rank_pool_state.json", rank_pool.to_dict())
        _save_task_queries(run_dir / "task_queries.safetensors", task_queries)

        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_rank_pool_weights(
        model,
        run_dir / "final_rank_pool_model.safetensors",
        extra_tensors={"group_keys": rank_pool.group_keys},
    )
    save_json(run_dir / "final_rank_pool_state.json", rank_pool.to_dict())
    save_json(
        run_dir / "run_summary.json",
        {
            "tasks": task_order,
            "final_checkpoint": str(run_dir / "final_rank_pool_model.safetensors"),
            "final_state": str(run_dir / "final_rank_pool_state.json"),
            "performance_matrix": str(run_dir / "performance_matrix.json"),
        },
    )
    print(f"\n[done] rank-pool run saved to {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    parser.add_argument("--tasks", default=None, help="Comma-separated task names. Defaults to config tasks.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg.seed)
    tasks = split_task_arg(args.tasks, cfg.tasks)
    print(f"Run output: {cfg.run_output_dir}")
    print(f"Tasks: {', '.join(tasks)}")
    train_rank_pool(cfg, tasks)


if __name__ == "__main__":
    main()
