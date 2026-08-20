from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class RankGroup:
    group_id: int
    rank_start: int
    rank_end: int
    state: str = "free"
    owner: str | None = None
    history: list[str] = field(default_factory=list)
    usage_count: int = 0
    last_score: float | None = None
    key_updates: int = 0
    recycle_count: int = 0
    recycled_from: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ranks(self) -> list[int]:
        return list(range(self.rank_start, self.rank_end))


@dataclass
class RankPoolSelection:
    task_name: str
    selected_groups: list[int]
    reused_groups: list[int]
    free_groups: list[int]
    recycled_groups: list[int]
    scores: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RankPoolState:
    def __init__(
        self,
        global_rank: int,
        group_rank: int,
        hidden_size: int,
        group_keys: torch.Tensor | None = None,
        groups: list[RankGroup] | None = None,
    ):
        if global_rank % group_rank != 0:
            raise ValueError("global_rank must be divisible by group_rank.")
        self.global_rank = global_rank
        self.group_rank = group_rank
        self.num_groups = global_rank // group_rank
        self.hidden_size = hidden_size
        if group_keys is None:
            group_keys = F.normalize(torch.randn(self.num_groups, hidden_size), p=2, dim=-1)
        self.group_keys = group_keys.detach().cpu()
        if groups is None:
            groups = [
                RankGroup(
                    group_id=idx,
                    rank_start=idx * group_rank,
                    rank_end=(idx + 1) * group_rank,
                )
                for idx in range(self.num_groups)
            ]
        self.groups = groups

    @classmethod
    def from_dict(cls, payload: dict[str, Any], group_keys: torch.Tensor) -> "RankPoolState":
        groups = []
        for item in payload["groups"]:
            item = dict(item)
            item.setdefault("recycle_count", 0)
            item.setdefault("recycled_from", [])
            groups.append(RankGroup(**item))
        return cls(
            global_rank=int(payload["global_rank"]),
            group_rank=int(payload["group_rank"]),
            hidden_size=int(payload["hidden_size"]),
            group_keys=group_keys,
            groups=groups,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_rank": self.global_rank,
            "group_rank": self.group_rank,
            "num_groups": self.num_groups,
            "hidden_size": self.hidden_size,
            "groups": [asdict(group) for group in self.groups],
        }

    def occupied_group_ids(self) -> list[int]:
        return [group.group_id for group in self.groups if group.state == "occupied"]

    def free_group_ids(self) -> list[int]:
        return [group.group_id for group in self.groups if group.state == "free"]

    def ranks_for_groups(self, group_ids: list[int]) -> list[int]:
        ranks: list[int] = []
        for group_id in group_ids:
            ranks.extend(self.groups[group_id].ranks)
        return ranks

    def set_group_keys(self, group_keys: torch.Tensor) -> None:
        if tuple(group_keys.shape) != (self.num_groups, self.hidden_size):
            raise ValueError(
                f"group_keys shape {tuple(group_keys.shape)} != {(self.num_groups, self.hidden_size)}."
            )
        self.group_keys = F.normalize(group_keys.detach().cpu().float(), p=2, dim=-1)

    def score(self, task_query: torch.Tensor) -> torch.Tensor:
        query = task_query.detach().cpu().float()
        if query.ndim != 1:
            raise ValueError(f"task_query must be 1D, got shape {tuple(query.shape)}.")
        if query.numel() != self.hidden_size:
            raise ValueError(f"task_query hidden size {query.numel()} != {self.hidden_size}.")
        query = F.normalize(query, p=2, dim=0)
        keys = F.normalize(self.group_keys.float(), p=2, dim=-1)
        return torch.mv(keys, query)

    def topk_group_ids(
        self,
        task_query: torch.Tensor,
        top_k_groups: int,
        candidate_groups: list[int] | None = None,
    ) -> tuple[list[int], torch.Tensor]:
        scores = self.score(task_query)
        candidates = candidate_groups if candidate_groups is not None else list(range(self.num_groups))
        ranked = sorted(candidates, key=lambda group_id: (-float(scores[group_id]), group_id))
        return ranked[:top_k_groups], scores

    def route(self, task_query: torch.Tensor, top_k_groups: int) -> RankPoolSelection:
        selected, scores = self.topk_group_ids(task_query, top_k_groups, self.occupied_group_ids())
        return RankPoolSelection(
            task_name="__route__",
            selected_groups=selected,
            reused_groups=selected,
            free_groups=[],
            recycled_groups=[],
            scores={str(idx): round(float(scores[idx]), 6) for idx in range(self.num_groups)},
        )

    def route_excluding(
        self,
        task_query: torch.Tensor,
        top_k_groups: int,
        excluded_groups: set[int] | list[int],
    ) -> RankPoolSelection:
        excluded = set(excluded_groups)
        occupied = [group_id for group_id in self.occupied_group_ids() if group_id not in excluded]
        selected, scores = self.topk_group_ids(task_query, top_k_groups, occupied)
        return RankPoolSelection(
            task_name="__route__",
            selected_groups=selected,
            reused_groups=selected,
            free_groups=[],
            recycled_groups=[],
            scores={str(idx): round(float(scores[idx]), 6) for idx in range(self.num_groups)},
        )

    def select_for_task(
        self,
        task_name: str,
        task_query: torch.Tensor,
        top_k_groups: int,
    ) -> RankPoolSelection:
        occupied = set(self.occupied_group_ids())
        free = set(self.free_group_ids())
        selected, scores = self.topk_group_ids(task_query, top_k_groups)
        reused = [group_id for group_id in selected if group_id in occupied]
        new_free = [group_id for group_id in selected if group_id in free]

        self.assign_groups_to_task(task_name, selected, scores)

        return RankPoolSelection(
            task_name=task_name,
            selected_groups=selected,
            reused_groups=reused,
            free_groups=new_free,
            recycled_groups=[],
            scores={str(idx): round(float(scores[idx]), 6) for idx in range(self.num_groups)},
        )

    def missing_groups(self, selected_groups: list[int], top_k_groups: int) -> int:
        return max(0, top_k_groups - len(selected_groups))

    def assign_groups_to_task(
        self,
        task_name: str,
        group_ids: list[int],
        scores: torch.Tensor | None = None,
    ) -> None:
        for group_id in group_ids:
            group = self.groups[group_id]
            group.state = "occupied"
            group.owner = task_name if group.owner is None else group.owner
            if task_name not in group.history:
                group.history.append(task_name)
            group.usage_count += 1
            if scores is not None:
                group.last_score = float(scores[group_id])

    def release_group(self, group_id: int, reason: str) -> None:
        group = self.groups[group_id]
        group.recycled_from.append(
            {
                "owner": group.owner,
                "history": list(group.history),
                "usage_count": group.usage_count,
                "last_score": group.last_score,
                "reason": reason,
            }
        )
        group.state = "free"
        group.owner = None
        group.history = []
        group.usage_count = 0
        group.last_score = None
        group.key_updates = 0
        group.recycle_count += 1

    def initialize_free_group_keys(self, task_query: torch.Tensor, group_ids: list[int]) -> None:
        if not group_ids:
            return
        normalized_query = F.normalize(task_query.detach().cpu().float(), p=2, dim=0)
        for group_id in group_ids:
            self.group_keys[group_id] = normalized_query

    def update_selected_keys(
        self,
        selected_groups: list[int],
        learned_group_keys: torch.Tensor,
    ) -> None:
        learned = F.normalize(learned_group_keys.detach().cpu().float(), p=2, dim=-1)
        if tuple(learned.shape) != (self.num_groups, self.hidden_size):
            raise ValueError(
                f"learned_group_keys shape {tuple(learned.shape)} != {(self.num_groups, self.hidden_size)}."
            )
        for group_id in selected_groups:
            self.group_keys[group_id] = learned[group_id]
            self.groups[group_id].key_updates += 1
