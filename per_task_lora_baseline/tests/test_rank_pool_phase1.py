from __future__ import annotations

import torch
import torch.nn as nn

from src.rank_pool import RankPoolState
from src.rank_recycling import compute_group_redundancy, compute_group_usage
from src.rank_pool_lora import (
    FixedRankPoolLinear,
    RankPoolKeyModule,
    count_rank_pool_parameters,
    reset_rank_pool_groups,
    set_active_groups,
    set_trainable_groups,
)
from src.task_query import get_llm_query


def test_get_llm_query_uses_masked_mean_and_normalization():
    hidden = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [100.0, 100.0]],
            [[2.0, 0.0], [0.0, 2.0], [2.0, 2.0]],
        ]
    )
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

    query = get_llm_query(hidden, mask)

    expected = torch.tensor([[0.5, 0.5], [4.0 / 3.0, 4.0 / 3.0]])
    expected = torch.nn.functional.normalize(expected, p=2, dim=-1)
    assert torch.allclose(query, expected)


def test_rank_pool_linear_masks_inactive_rank_gradients():
    base = nn.Linear(5, 3, bias=False)
    wrapped = nn.Sequential(
        FixedRankPoolLinear(base, global_rank=8, group_rank=4, lora_alpha=16, lora_dropout=0.0)
    )
    with torch.no_grad():
        wrapped[0].lora_B.normal_()

    set_active_groups(wrapped, [0], group_rank=4)
    set_trainable_groups(wrapped, [0], group_rank=4)
    out = wrapped(torch.randn(2, 5)).sum()
    out.backward()

    assert wrapped[0].lora_A.grad[:4].abs().sum() > 0
    assert wrapped[0].lora_B.grad[:, :4].abs().sum() > 0
    assert torch.allclose(wrapped[0].lora_A.grad[4:], torch.zeros_like(wrapped[0].lora_A.grad[4:]))
    assert torch.allclose(wrapped[0].lora_B.grad[:, 4:], torch.zeros_like(wrapped[0].lora_B.grad[:, 4:]))
    assert count_rank_pool_parameters(wrapped) == 8 * (5 + 3)


def test_rank_pool_key_module_masks_unselected_group_gradients():
    keys = torch.eye(4)
    key_module = RankPoolKeyModule(keys)
    key_module.set_trainable_groups([1])

    loss = key_module.alignment_loss(torch.tensor([1.0, 0.0, 0.0, 0.0]), selected_groups=[1])
    loss.backward()

    assert key_module.group_keys.grad[1].abs().sum() > 0
    assert torch.allclose(key_module.group_keys.grad[0], torch.zeros_like(key_module.group_keys.grad[0]))
    assert torch.allclose(key_module.group_keys.grad[2:], torch.zeros_like(key_module.group_keys.grad[2:]))


def test_rank_pool_selects_topk_groups_by_query_key_similarity():
    state = RankPoolState(global_rank=8, group_rank=2, hidden_size=4)
    state.set_group_keys(torch.eye(4))
    state.assign_groups_to_task("old_task", [1])

    first = state.select_for_task(
        task_name="new_task",
        task_query=torch.tensor([0.0, 1.0, 0.0, 0.0]),
        top_k_groups=2,
    )
    assert first.selected_groups == [1, 0]
    assert first.reused_groups == [1]
    assert first.free_groups == [0]


def test_full_rank_pool_still_uses_topk_key_matching():
    state = RankPoolState(global_rank=4, group_rank=2, hidden_size=2)
    state.set_group_keys(torch.eye(2))
    state.assign_groups_to_task("task_a", [0])
    state.assign_groups_to_task("task_b", [1])

    selection = state.select_for_task(
        task_name="task_c",
        task_query=torch.tensor([0.0, 1.0]),
        top_k_groups=2,
    )

    assert selection.selected_groups == [1, 0]
    assert selection.reused_groups == [1, 0]
    assert selection.free_groups == []
    assert selection.recycled_groups == []
    assert state.missing_groups(selection.selected_groups, top_k_groups=2) == 0


def test_rank_pool_state_updates_selected_keys_from_learned_keys():
    state = RankPoolState(global_rank=8, group_rank=2, hidden_size=4)
    task_query = torch.tensor([0.0, 1.0, 0.0, 0.0])
    state.initialize_free_group_keys(task_query, [2])
    assert torch.allclose(state.group_keys[2], task_query)

    learned = torch.randn(4, 4)
    state.update_selected_keys(selected_groups=[2], learned_group_keys=learned)
    expected = torch.nn.functional.normalize(learned, p=2, dim=-1)[2]
    assert torch.allclose(state.group_keys[2], expected)
    assert state.groups[2].key_updates == 1


def test_recycling_usage_and_release_state():
    state = RankPoolState(global_rank=4, group_rank=2, hidden_size=2)
    state.assign_groups_to_task("task_a", [0])
    state.assign_groups_to_task("task_b", [0, 1])

    usage = compute_group_usage(state, [0, 1])
    assert usage[0] > usage[1]

    state.release_group(0, reason="unit-test")
    assert state.groups[0].state == "free"
    assert state.groups[0].usage_count == 0
    assert state.groups[0].recycle_count == 1
    assert state.groups[0].recycled_from[0]["reason"] == "unit-test"


def test_group_redundancy_and_reset_rank_pool_group():
    base = nn.Linear(3, 2, bias=False)
    model = nn.Sequential(FixedRankPoolLinear(base, global_rank=4, group_rank=2, lora_alpha=8, lora_dropout=0.0))
    layer = model[0]
    with torch.no_grad():
        layer.lora_A[:2] = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        layer.lora_A[2:] = layer.lora_A[:2]
        layer.lora_B[:, :2] = torch.eye(2)
        layer.lora_B[:, 2:] = torch.eye(2)

    redundancy = compute_group_redundancy(model, group_rank=2, group_ids=[0, 1])
    assert redundancy[0] > 0.99
    assert redundancy[1] > 0.99

    reset_rank_pool_groups(model, [1], group_rank=2)
    assert torch.allclose(layer.lora_B[:, 2:], torch.zeros_like(layer.lora_B[:, 2:]))
