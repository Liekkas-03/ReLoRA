from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .metrics import strip_answer_prefix
from .rank_pool import RankPoolState
from .rank_pool_lora import (
    clone_rank_pool_state_dict,
    iter_rank_pool_layers,
    load_rank_pool_state_dict,
    reset_rank_pool_groups,
    save_rank_pool_weights,
    set_active_groups,
    set_trainable_groups,
)
from .task_specs import get_task_spec
from .utils import safe_mkdir


@dataclass
class RecyclingCandidateScore:
    group_id: int
    importance: float
    redundancy: float
    usage: float
    importance_norm: float
    redundancy_norm: float
    usage_norm: float
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RecyclingAttempt:
    group_id: int
    success: bool
    mean_accuracy_before: float
    mean_accuracy_after: float
    accuracy_drop: float
    before_by_task: dict[str, float]
    after_by_task: dict[str, float]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RecyclingResult:
    success: bool
    recycled_group: int | None
    candidate_scores: list[RecyclingCandidateScore] = field(default_factory=list)
    attempts: list[RecyclingAttempt] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "recycled_group": self.recycled_group,
            "candidate_scores": [item.to_dict() for item in self.candidate_scores],
            "attempts": [item.to_dict() for item in self.attempts],
        }


def _normalize_map(values: dict[int, float]) -> dict[int, float]:
    if not values:
        return {}
    min_value = min(values.values())
    max_value = max(values.values())
    if abs(max_value - min_value) < 1e-12:
        return {key: 0.0 for key in values}
    return {key: (value - min_value) / (max_value - min_value) for key, value in values.items()}


def _take_examples(dataset, max_samples: int) -> list[dict[str, Any]]:
    return [dataset[idx] for idx in range(min(max_samples, len(dataset)))]


def _group_inner_products(model, group_rank: int, group_ids: list[int]) -> dict[tuple[int, int], float]:
    products = {(left, right): 0.0 for left in group_ids for right in group_ids}
    for _, module in iter_rank_pool_layers(model):
        for left in group_ids:
            left_ranks = slice(left * group_rank, (left + 1) * group_rank)
            left_a = module.lora_A[left_ranks, :].float()
            left_b = module.lora_B[:, left_ranks].float()
            for right in group_ids:
                right_ranks = slice(right * group_rank, (right + 1) * group_rank)
                right_a = module.lora_A[right_ranks, :].float()
                right_b = module.lora_B[:, right_ranks].float()
                a_cross = left_a @ right_a.T
                b_cross = left_b.T @ right_b
                products[(left, right)] += float(torch.sum(a_cross * b_cross).detach().cpu())
    return products


def compute_group_redundancy(model, group_rank: int, group_ids: list[int]) -> dict[int, float]:
    if len(group_ids) <= 1:
        return {group_id: 0.0 for group_id in group_ids}
    products = _group_inner_products(model, group_rank, group_ids)
    redundancy: dict[int, float] = {}
    for left in group_ids:
        left_norm = max(products[(left, left)], 0.0) ** 0.5
        similarities = []
        for right in group_ids:
            if left == right:
                continue
            right_norm = max(products[(right, right)], 0.0) ** 0.5
            denom = max(left_norm * right_norm, 1e-12)
            similarities.append(products[(left, right)] / denom)
        redundancy[left] = max(similarities) if similarities else 0.0
    return redundancy


def compute_group_usage(rank_pool: RankPoolState, group_ids: list[int]) -> dict[int, float]:
    return {
        group_id: float(rank_pool.groups[group_id].usage_count + len(rank_pool.groups[group_id].history))
        for group_id in group_ids
    }


@torch.no_grad()
def evaluate_label_accuracy(
    cfg,
    model,
    tokenizer,
    task_name: str,
    examples: list[dict[str, Any]],
    active_groups: list[int],
) -> float:
    if not examples:
        return 0.0
    from .data import OLoRADecoderCollator

    model.eval()
    tokenizer.padding_side = "left"
    device = next(model.parameters()).device
    set_active_groups(model, active_groups, cfg.rank_pool.group_rank)
    collator = OLoRADecoderCollator(tokenizer=tokenizer, data_cfg=cfg.data, train=False)
    loader = DataLoader(
        examples,
        batch_size=cfg.training.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    correct = 0
    total = 0
    for batch in loader:
        metadata = batch.pop("metadata")
        batch = {key: value.to(device) for key, value in batch.items() if torch.is_tensor(value)}
        generated = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            max_new_tokens=cfg.training.generation_max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )
        decoded = tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        for text, meta in zip(decoded, metadata):
            prediction = strip_answer_prefix(text)
            correct += int(prediction == meta["Instance"]["label"])
            total += 1
    return 100.0 * correct / max(total, 1)


def evaluate_seen_task_accuracies(
    cfg,
    model,
    tokenizer,
    rank_pool: RankPoolState,
    seen_tasks: list[str],
    task_queries: dict[str, torch.Tensor],
    eval_examples_by_task: dict[str, list[dict[str, Any]]],
    excluded_groups: set[int] | None = None,
) -> dict[str, float]:
    excluded_groups = excluded_groups or set()
    accuracies: dict[str, float] = {}
    for task_name in seen_tasks:
        query = task_queries[task_name]
        if excluded_groups:
            route = rank_pool.route_excluding(query, cfg.rank_pool.top_k_groups, excluded_groups)
        else:
            route = rank_pool.route(query, cfg.rank_pool.top_k_groups)
        accuracies[task_name] = evaluate_label_accuracy(
            cfg=cfg,
            model=model,
            tokenizer=tokenizer,
            task_name=task_name,
            examples=eval_examples_by_task[task_name],
            active_groups=route.selected_groups,
        )
    return accuracies


def compute_group_importance(
    cfg,
    model,
    tokenizer,
    rank_pool: RankPoolState,
    group_ids: list[int],
    seen_tasks: list[str],
    task_queries: dict[str, torch.Tensor],
    eval_examples_by_task: dict[str, list[dict[str, Any]]],
) -> dict[int, float]:
    importance: dict[int, float] = {}
    base_by_task = evaluate_seen_task_accuracies(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        rank_pool=rank_pool,
        seen_tasks=seen_tasks,
        task_queries=task_queries,
        eval_examples_by_task=eval_examples_by_task,
        excluded_groups=None,
    )
    for group_id in group_ids:
        masked_by_task = evaluate_seen_task_accuracies(
            cfg=cfg,
            model=model,
            tokenizer=tokenizer,
            rank_pool=rank_pool,
            seen_tasks=seen_tasks,
            task_queries=task_queries,
            eval_examples_by_task=eval_examples_by_task,
            excluded_groups={group_id},
        )
        drops = []
        for task_name in seen_tasks:
            drops.append(max(0.0, base_by_task[task_name] - masked_by_task[task_name]))
        importance[group_id] = sum(drops) / max(len(drops), 1)
    return importance


def score_recycling_candidates(
    cfg,
    rank_pool: RankPoolState,
    model,
    candidate_groups: list[int],
    importance: dict[int, float],
) -> list[RecyclingCandidateScore]:
    redundancy = compute_group_redundancy(model, cfg.rank_pool.group_rank, candidate_groups)
    usage = compute_group_usage(rank_pool, candidate_groups)
    importance_norm = _normalize_map(importance)
    redundancy_norm = _normalize_map(redundancy)
    usage_norm = _normalize_map(usage)

    scored: list[RecyclingCandidateScore] = []
    for group_id in candidate_groups:
        score = (
            cfg.recycling.importance_weight * importance_norm[group_id]
            - cfg.recycling.redundancy_weight * redundancy_norm[group_id]
            + cfg.recycling.usage_weight * usage_norm[group_id]
        )
        scored.append(
            RecyclingCandidateScore(
                group_id=group_id,
                importance=importance[group_id],
                redundancy=redundancy[group_id],
                usage=usage[group_id],
                importance_norm=importance_norm[group_id],
                redundancy_norm=redundancy_norm[group_id],
                usage_norm=usage_norm[group_id],
                score=score,
            )
        )
    return sorted(scored, key=lambda item: item.score)


def _distill_kl_on_label_positions(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    shifted_labels = labels[:, 1:]
    mask = shifted_labels.ne(-100)
    if int(mask.sum().item()) == 0:
        return student_logits.sum() * 0.0
    student = student_logits[:, :-1, :][mask] / temperature
    teacher = teacher_logits[:, :-1, :][mask] / temperature
    student_log_probs = F.log_softmax(student, dim=-1)
    teacher_probs = F.softmax(teacher, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature**2)


def _next_replay_batch(loaders: dict[str, DataLoader], iters: dict[str, Any], task_name: str):
    try:
        return next(iters[task_name])
    except StopIteration:
        iters[task_name] = iter(loaders[task_name])
        return next(iters[task_name])


def run_knowledge_recovery(
    cfg,
    model,
    tokenizer,
    rank_pool: RankPoolState,
    candidate_group: int,
    teacher_state: dict[str, torch.Tensor],
    seen_tasks: list[str],
    task_queries: dict[str, torch.Tensor],
    replay_examples_by_task: dict[str, list[dict[str, Any]]],
    eval_examples_by_task: dict[str, list[dict[str, Any]]],
) -> RecyclingAttempt:
    from .data import OLoRADecoderCollator

    load_rank_pool_state_dict(model, teacher_state)
    before_by_task = evaluate_seen_task_accuracies(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        rank_pool=rank_pool,
        seen_tasks=seen_tasks,
        task_queries=task_queries,
        eval_examples_by_task=eval_examples_by_task,
    )
    before_mean = sum(before_by_task.values()) / max(len(before_by_task), 1)

    load_rank_pool_state_dict(model, teacher_state)
    set_active_groups(model, [], cfg.rank_pool.group_rank)
    set_trainable_groups(model, [], cfg.rank_pool.group_rank)
    model.train()

    collator = OLoRADecoderCollator(tokenizer=tokenizer, data_cfg=cfg.data, train=True)
    loaders = {
        task_name: DataLoader(
            replay_examples_by_task[task_name],
            batch_size=cfg.training.per_device_train_batch_size,
            shuffle=True,
            collate_fn=collator,
        )
        for task_name in seen_tasks
        if replay_examples_by_task[task_name]
    }
    if not loaders:
        return RecyclingAttempt(
            group_id=candidate_group,
            success=False,
            mean_accuracy_before=before_mean,
            mean_accuracy_after=before_mean,
            accuracy_drop=0.0,
            before_by_task=before_by_task,
            after_by_task=before_by_task,
            reason="no replay examples available",
        )

    device = next(model.parameters()).device
    lr = cfg.recycling.recovery_learning_rate or cfg.training.learning_rate
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=lr,
        weight_decay=cfg.training.weight_decay,
    )
    iters = {task_name: iter(loader) for task_name, loader in loaders.items()}
    train_tasks = list(loaders)

    for step in range(cfg.recycling.recovery_steps):
        task_name = train_tasks[step % len(train_tasks)]
        batch = _next_replay_batch(loaders, iters, task_name)
        batch = {key: value.to(device) for key, value in batch.items() if torch.is_tensor(value)}
        query = task_queries[task_name]
        teacher_route = rank_pool.route(query, cfg.rank_pool.top_k_groups)
        student_route = rank_pool.route_excluding(query, cfg.rank_pool.top_k_groups, {candidate_group})

        student_state = clone_rank_pool_state_dict(model)
        load_rank_pool_state_dict(model, teacher_state)
        set_active_groups(model, teacher_route.selected_groups, cfg.rank_pool.group_rank)
        with torch.no_grad():
            teacher_outputs = model(**batch)
            teacher_logits = teacher_outputs.logits.detach()

        load_rank_pool_state_dict(model, student_state)
        set_active_groups(model, student_route.selected_groups, cfg.rank_pool.group_rank)
        set_trainable_groups(model, student_route.selected_groups, cfg.rank_pool.group_rank)
        optimizer.zero_grad(set_to_none=True)
        student_outputs = model(**batch)
        distill_loss = _distill_kl_on_label_positions(
            student_outputs.logits,
            teacher_logits,
            batch["labels"],
            cfg.recycling.distill_temperature,
        )
        replay_loss = student_outputs.loss
        loss = distill_loss + cfg.recycling.replay_loss_weight * replay_loss
        loss.backward()
        optimizer.step()

    set_trainable_groups(model, [], cfg.rank_pool.group_rank)
    after_by_task = evaluate_seen_task_accuracies(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        rank_pool=rank_pool,
        seen_tasks=seen_tasks,
        task_queries=task_queries,
        eval_examples_by_task=eval_examples_by_task,
        excluded_groups={candidate_group},
    )
    after_mean = sum(after_by_task.values()) / max(len(after_by_task), 1)
    drop = before_mean - after_mean
    success = drop < cfg.recycling.epsilon
    return RecyclingAttempt(
        group_id=candidate_group,
        success=success,
        mean_accuracy_before=before_mean,
        mean_accuracy_after=after_mean,
        accuracy_drop=drop,
        before_by_task=before_by_task,
        after_by_task=after_by_task,
        reason="accepted" if success else "accuracy drop exceeds epsilon",
    )


def recycle_one_group(
    cfg,
    model,
    tokenizer,
    rank_pool: RankPoolState,
    seen_tasks: list[str],
    task_queries: dict[str, torch.Tensor],
    train_raw_by_task: dict[str, Any],
    eval_raw_by_task: dict[str, Any],
    exclude_groups: set[int],
    output_dir: Path,
) -> RecyclingResult:
    safe_mkdir(output_dir)
    occupied = [group_id for group_id in rank_pool.occupied_group_ids() if group_id not in exclude_groups]
    if not occupied:
        raise RuntimeError("Rank recycling requested, but no occupied group is available for recycling.")
    if not cfg.recycling.enabled:
        raise RuntimeError("Rank pool is full and recycling is disabled.")
    if not seen_tasks:
        raise RuntimeError("Rank recycling requested before any seen task is available.")

    replay_examples = {
        task_name: _take_examples(train_raw_by_task[task_name], cfg.recycling.replay_samples_per_task)
        for task_name in seen_tasks
    }
    eval_examples = {
        task_name: _take_examples(eval_raw_by_task[task_name], cfg.recycling.eval_samples_per_task)
        for task_name in seen_tasks
    }

    teacher_state = clone_rank_pool_state_dict(model)
    if cfg.recycling.save_teacher_before_recycling:
        save_rank_pool_weights(
            model,
            output_dir / "teacher_before_recycling.safetensors",
            extra_tensors={"group_keys": rank_pool.group_keys},
        )

    importance = compute_group_importance(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        rank_pool=rank_pool,
        group_ids=occupied,
        seen_tasks=seen_tasks,
        task_queries=task_queries,
        eval_examples_by_task=eval_examples,
    )
    candidate_scores = score_recycling_candidates(
        cfg=cfg,
        rank_pool=rank_pool,
        model=model,
        candidate_groups=occupied,
        importance=importance,
    )

    attempts: list[RecyclingAttempt] = []
    for candidate in candidate_scores[: cfg.recycling.max_trials]:
        load_rank_pool_state_dict(model, teacher_state)
        attempt = run_knowledge_recovery(
            cfg=cfg,
            model=model,
            tokenizer=tokenizer,
            rank_pool=rank_pool,
            candidate_group=candidate.group_id,
            teacher_state=teacher_state,
            seen_tasks=seen_tasks,
            task_queries=task_queries,
            replay_examples_by_task=replay_examples,
            eval_examples_by_task=eval_examples,
        )
        attempts.append(attempt)
        if attempt.success:
            rank_pool.release_group(candidate.group_id, reason="rank recycling accepted")
            reset_rank_pool_groups(model, [candidate.group_id], cfg.rank_pool.group_rank)
            return RecyclingResult(
                success=True,
                recycled_group=candidate.group_id,
                candidate_scores=candidate_scores,
                attempts=attempts,
            )
        load_rank_pool_state_dict(model, teacher_state)

    return RecyclingResult(
        success=False,
        recycled_group=None,
        candidate_scores=candidate_scores,
        attempts=attempts,
    )
