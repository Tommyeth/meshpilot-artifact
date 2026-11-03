"""Offline intent baseline and fault injection used by synthetic replay."""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any

from .models import AgentDecision, CandidatePlan, TaskIntent
from .planner import candidate_objective, profile_weights


class ResourceAgent(ABC):
    name = "abstract"

    @abstractmethod
    def select(self, task: TaskIntent, candidates: list[CandidatePlan]) -> AgentDecision:
        raise NotImplementedError


def infer_profile_from_text(text: str) -> str:
    lowered = text.lower()
    scores = {"balanced": 0, "latency": 0, "cost": 0, "resilience": 0}
    keywords = {
        "latency": ("latency", "fast", "deadline", "interactive", "实时", "时延", "最快", "截止"),
        "cost": ("cost", "cheap", "budget", "save", "价格", "成本", "便宜", "预算"),
        "resilience": ("reliable", "resilience", "failure", "stable", "可靠", "容错", "稳定", "故障"),
    }
    for profile, words in keywords.items():
        scores[profile] = sum(1 for word in words if word in lowered)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "balanced"


class KeywordIntentAgent(ResourceAgent):
    """Transparent non-LLM intent baseline and offline pipeline backend."""

    name = "keyword"

    def select(self, task: TaskIntent, candidates: list[CandidatePlan]) -> AgentDecision:
        if not candidates:
            return AgentDecision(
                action="abstain",
                rationale="No feasible candidates.",
                confidence=1.0,
                backend=self.name,
                api_status="not_called",
            )
        profile = infer_profile_from_text(task.operator_intent)
        selected = min(candidates, key=lambda candidate: candidate_objective(candidate, profile_weights(profile)))
        return AgentDecision(
            action="select",
            plan_id=selected.plan_id,
            rationale=f"Keyword intent compiler selected the lowest {profile}-profile objective.",
            confidence=0.70,
            inferred_profile=profile,
            backend=self.name,
            api_status="completed",
        )


class FaultInjectingAgent(ResourceAgent):
    """Wraps an agent to test whether the safety layer catches bad actions."""

    name = "fault_injecting"

    def __init__(self, delegate: ResourceAgent, invalid_rate: float = 0.15, seed: int = 7) -> None:
        self.delegate = delegate
        self.invalid_rate = invalid_rate
        self.rng = random.Random(seed)

    def set_context(self, **context: Any) -> None:
        setter = getattr(self.delegate, "set_context", None)
        if setter is not None:
            setter(**context)

    @property
    def last_replayed_latency_ms(self) -> float | None:
        return getattr(self.delegate, "last_replayed_latency_ms", None)

    def select(self, task: TaskIntent, candidates: list[CandidatePlan]) -> AgentDecision:
        decision = self.delegate.select(task, candidates)
        if decision.action == "select" and self.rng.random() < self.invalid_rate:
            # Keep every provenance field of the real decision; only the action
            # payload is replaced by the injected unknown identifier.
            return decision.model_copy(
                update={
                    "action": "select",
                    "plan_id": f"hallucinated:{task.task_id}",
                    "rationale": "Injected invalid action for verifier evaluation.",
                    "confidence": 0.99,
                }
            )
        return decision
