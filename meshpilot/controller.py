from __future__ import annotations

import time

from .agents import ResourceAgent
from .models import AgentDecision, CandidatePlan, ClusterSnapshot, DecisionRecord, TaskIntent
from .planner import CandidatePlanner, select_baseline
from .verifier import PlanVerifier


class ClosedLoopController:
    def __init__(self, planner: CandidatePlanner, verifier: PlanVerifier) -> None:
        self.planner = planner
        self.verifier = verifier

    def schedule(
        self,
        task: TaskIntent,
        snapshot: ClusterSnapshot,
        agent: ResourceAgent,
        policy_name: str,
        allow_fallback: bool = True,
    ) -> DecisionRecord:
        planning_started = time.perf_counter_ns()
        candidates = self.planner.generate(task, snapshot)
        planning_ms = (time.perf_counter_ns() - planning_started) / 1e6
        agent_started = time.perf_counter_ns()
        decision = agent.select(task, candidates)
        agent_ms = (time.perf_counter_ns() - agent_started) / 1e6
        return self.schedule_with_decision(
            task=task,
            snapshot=snapshot,
            decision=decision,
            policy_name=policy_name,
            allow_fallback=allow_fallback,
            candidates=candidates,
            planning_latency_ms=planning_ms,
            agent_latency_ms=agent_ms,
        )

    def schedule_with_decision(
        self,
        task: TaskIntent,
        snapshot: ClusterSnapshot,
        decision: AgentDecision,
        policy_name: str,
        allow_fallback: bool,
        candidates: list[CandidatePlan] | None = None,
        planning_latency_ms: float = 0.0,
        agent_latency_ms: float = 0.0,
    ) -> DecisionRecord:
        if candidates is None:
            started = time.perf_counter_ns()
            candidates = self.planner.generate(task, snapshot)
            planning_latency_ms = (time.perf_counter_ns() - started) / 1e6
        verification = self.verifier.verify(task, snapshot, candidates, decision)
        selected = None
        used_fallback = False
        if verification.accepted:
            selected = next(candidate for candidate in candidates if candidate.plan_id == verification.selected_plan_id)
        elif allow_fallback:
            selected = select_baseline("joint_fixed", candidates)
            used_fallback = selected is not None
        return DecisionRecord(
            task_id=task.task_id,
            policy=policy_name,
            selected_plan=selected,
            agent_decision=decision,
            verification=verification,
            used_fallback=used_fallback,
            planning_latency_ms=planning_latency_ms,
            agent_latency_ms=agent_latency_ms,
        )


def unverified_decision_record(
    task: TaskIntent,
    snapshot: ClusterSnapshot,
    decision: AgentDecision,
    candidates: list[CandidatePlan],
    policy_name: str,
    planning_latency_ms: float,
    agent_latency_ms: float,
) -> DecisionRecord:
    selected = None
    if decision.action == "select":
        selected = next((candidate for candidate in candidates if candidate.plan_id == decision.plan_id), None)
    return DecisionRecord(
        task_id=task.task_id,
        policy=policy_name,
        selected_plan=selected,
        agent_decision=decision,
        used_fallback=False,
        planning_latency_ms=planning_latency_ms,
        agent_latency_ms=agent_latency_ms,
    )
