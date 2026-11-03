from __future__ import annotations

import csv
import json
import math
import platform
import random
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Iterable, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .agents import FaultInjectingAgent, KeywordIntentAgent, ResourceAgent
from .controller import ClosedLoopController, unverified_decision_record
from .intent_benchmark import INTENT_PHRASES, phrase_for
from .models import (
    CandidatePlan,
    ClusterSnapshot,
    DecisionRecord,
    ExecutionOutcome,
    LinkState,
    NodeState,
    TaskIntent,
)
from .planner import (
    COST_USD_TO_MS,
    FRAGMENTATION_TO_MS,
    OBJECTIVE_SCALE_VERSION,
    RISK_TO_MS,
    CandidatePlanner,
    PlannerConfig,
    candidate_objective,
    profile_weights,
    select_baseline,
    select_oracle,
)
from .verifier import PlanVerifier


SCENARIOS = ("stable", "bandwidth_drop", "node_failure", "gpu_pressure", "memory_pressure")
STATIC_POLICIES = ("round_robin", "least_loaded", "network_greedy", "dominant_resource", "joint_fixed")


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")  # a misspelled key must not run silently

    artifact_schema_version: int = Field(default=2, ge=2, le=2)
    objective_scale_version: str = OBJECTIVE_SCALE_VERSION
    seeds: list[int] = Field(default_factory=lambda: [7, 11, 19, 23, 31])
    nodes: int = Field(default=12, ge=4)
    tasks_per_seed: int = Field(default=24, ge=1)
    scenarios: list[str] = Field(default_factory=lambda: list(SCENARIOS))
    top_k: int = Field(default=16, ge=2)
    control_interval_ms: float = Field(default=1_000.0, ge=0)
    injected_invalid_rate: float = Field(default=0.15, ge=0, le=1)
    deadline_scale: float = Field(default=1.0, ge=0.25, le=4.0)
    agent_backend: str = "keyword"
    model: str | None = None
    llm_timeout_s: float = Field(default=30.0, gt=0)
    # Source-request settings retained for configuration validation.
    reasoning_max_tokens: int = Field(default=0, ge=0)
    max_output_tokens: int = Field(default=512, ge=64)
    llm_temperature: float | None = 0.0
    llm_max_retries: int = Field(default=5, ge=0)
    llm_max_consecutive_errors: int = Field(default=3, ge=1)
    llm_schema_variant: Literal["strict", "portable"] = "strict"
    reasoning_effort: Literal["off", "minimal", "low", "medium", "high", "provider_default"] = "off"
    llm_require_parameters: bool = True
    prompt_variant: Literal["base", "objective", "scores", "weights"] = "base"
    decision_cache: str | None = None

    @model_validator(mode="after")
    def reasoning_budget_fits_output_limit(self) -> "ExperimentConfig":
        if self.reasoning_max_tokens > 0 and self.max_output_tokens < self.reasoning_max_tokens + 256:
            raise ValueError("max_output_tokens must exceed reasoning_max_tokens by at least 256")
        if self.reasoning_max_tokens > 0 and self.llm_timeout_s < 60:
            raise ValueError("llm_timeout_s must be at least 60 s when a reasoning budget is set")
        if self.reasoning_max_tokens > 0 and self.reasoning_effort != "off":
            raise ValueError("set either reasoning_max_tokens or reasoning_effort, not both")
        return self


def build_agent(config: ExperimentConfig, seed: int) -> ResourceAgent:
    """Build the offline replay backend; this artifact makes no provider calls."""
    if config.agent_backend != "keyword":
        raise ValueError("the offline artifact supports only the keyword backend")
    if config.decision_cache:
        raise ValueError("decision caches are not used by the offline artifact")
    base: ResourceAgent = KeywordIntentAgent()
    return FaultInjectingAgent(base, invalid_rate=config.injected_invalid_rate, seed=seed)


def generate_snapshot(seed: int, nodes: int) -> ClusterSnapshot:
    rng = random.Random(seed)
    node_states: list[NodeState] = []
    links: list[LinkState] = []
    for index in range(nodes):
        has_gpu = index % 3 == 0
        cpu_capacity = float(rng.choice([4, 8, 16, 32]))
        ram_capacity = float(rng.choice([16, 32, 64, 128]))
        gpu_capacity = float(rng.choice([1, 2])) if has_gpu else 0.0
        vram_capacity = gpu_capacity * float(rng.choice([12, 24, 40])) if has_gpu else 0.0
        cpu_available = cpu_capacity * rng.uniform(0.55, 0.95)
        ram_available = ram_capacity * rng.uniform(0.55, 0.95)
        gpu_available = gpu_capacity if has_gpu and rng.random() > 0.15 else max(0.0, gpu_capacity - 1.0)
        vram_available = vram_capacity * rng.uniform(0.55, 0.95) if has_gpu else 0.0
        node_states.append(
            NodeState(
                node_id=f"node-{index:02d}",
                region=f"region-{index % 4}",
                provider=f"provider-{index % 3}",
                cpu_capacity=cpu_capacity,
                cpu_available=cpu_available,
                ram_capacity_gib=ram_capacity,
                ram_available_gib=ram_available,
                gpu_capacity=gpu_capacity,
                gpu_available=gpu_available,
                vram_capacity_gib=vram_capacity,
                vram_available_gib=min(vram_available, vram_capacity),
                compute_speed=rng.uniform(0.55, 2.4) * (1.35 if has_gpu else 1.0),
                active_tasks=rng.randint(0, 3),
                price_per_cpu_hour=rng.uniform(0.01, 0.09),
                price_per_gpu_hour=rng.uniform(0.20, 1.80) if has_gpu else 0.0,
                price_per_gb_egress=rng.uniform(0.0, 0.12),
                failure_probability=rng.uniform(0.002, 0.06),
                trust_level=rng.choice([1, 1, 2, 2, 3]),
            )
        )
        links.append(
            LinkState(
                client_id="client-0",
                node_id=f"node-{index:02d}",
                bandwidth_mbps=10 ** rng.uniform(math.log10(40), math.log10(2_000)),
                rtt_ms=rng.uniform(4, 180),
                loss_rate=rng.uniform(0.0, 0.035),
                jitter_ms=rng.uniform(0.2, 15),
                telemetry_age_s=rng.uniform(0, 45),
                reservable_fraction=rng.uniform(0.55, 0.90),
            )
        )
    return ClusterSnapshot(snapshot_id=f"snapshot-{seed}-base", version=1, nodes=node_states, links=links)


def generate_tasks(seed: int, count: int, deadline_scale: float = 1.0) -> list[TaskIntent]:
    rng = random.Random(seed + 10_000)
    tasks: list[TaskIntent] = []
    profiles = list(INTENT_PHRASES)
    kinds = ("cpu", "gpu", "memory", "network")
    for index in range(count):
        profile = profiles[index % len(profiles)]
        profile_cycle = index // len(profiles)
        intent_phrase_id, operator_intent = phrase_for(profile, seed, profile_cycle)
        kind = kinds[(index // len(profiles)) % len(kinds)]
        common = {
            "task_id": f"task-{seed}-{index:03d}",
            "operator_intent": operator_intent,
            "evaluation_profile": profile,
            "deadline_is_hard": False,
            "max_loss_rate": 0.15,
            "minimum_trust": 1,
            "metadata": {"kind": kind, "intent_phrase_id": intent_phrase_id},
        }
        if kind == "cpu":
            task = TaskIntent(
                **common,
                cpu_cores=rng.choice([1.0, 2.0, 4.0]),
                ram_gib=rng.choice([2.0, 4.0, 8.0]),
                estimated_compute_ms=rng.uniform(700, 4_000),
                input_mib=rng.uniform(1, 16),
                output_mib=rng.uniform(0.1, 2),
                deadline_ms=rng.uniform(3_500, 10_000) * deadline_scale,
            )
        elif kind == "gpu":
            task = TaskIntent(
                **common,
                cpu_cores=2.0,
                ram_gib=rng.choice([8.0, 12.0, 16.0]),
                gpu_units=1.0,
                vram_gib=rng.choice([6.0, 8.0, 10.0]),
                estimated_compute_ms=rng.uniform(600, 2_500),
                input_mib=rng.uniform(16, 96),
                output_mib=rng.uniform(1, 8),
                deadline_ms=rng.uniform(2_500, 7_000) * deadline_scale,
            )
        elif kind == "memory":
            task = TaskIntent(
                **common,
                cpu_cores=2.0,
                ram_gib=rng.choice([12.0, 20.0, 28.0]),
                estimated_compute_ms=rng.uniform(900, 4_000),
                input_mib=rng.uniform(8, 64),
                output_mib=rng.uniform(1, 4),
                deadline_ms=rng.uniform(4_000, 10_000) * deadline_scale,
            )
        else:
            task = TaskIntent(
                **common,
                cpu_cores=1.0,
                ram_gib=4.0,
                estimated_compute_ms=rng.uniform(300, 1_200),
                input_mib=rng.uniform(256, 1_024),
                output_mib=rng.uniform(8, 64),
                min_bandwidth_mbps=rng.choice([10.0, 20.0, 30.0]),
                deadline_ms=rng.uniform(8_000, 20_000) * deadline_scale,
            )
        tasks.append(task)
    return tasks


def perturb_snapshot(base: ClusterSnapshot, scenario: str, seed: int) -> ClusterSnapshot:
    rng = random.Random(seed + sum(ord(char) for char in scenario))
    snapshot = base.model_copy(deep=True)
    snapshot.snapshot_id = f"{base.snapshot_id}-{scenario}"
    snapshot.version = base.version + 1
    if scenario == "stable":
        return snapshot
    if scenario == "bandwidth_drop":
        affected = rng.sample(snapshot.links, max(1, len(snapshot.links) // 3))
        for link in affected:
            link.bandwidth_mbps = max(1.0, link.bandwidth_mbps * rng.uniform(0.05, 0.20))
            link.loss_rate = min(0.35, link.loss_rate + rng.uniform(0.03, 0.12))
            link.telemetry_age_s = 0.0
        return snapshot
    if scenario == "node_failure":
        affected = rng.sample(snapshot.nodes, max(1, len(snapshot.nodes) // 5))
        for node in affected:
            node.online = False
        return snapshot
    if scenario == "gpu_pressure":
        gpu_nodes = [node for node in snapshot.nodes if node.gpu_capacity > 0]
        for node in rng.sample(gpu_nodes, max(1, len(gpu_nodes) // 2)):
            node.gpu_available = 0.0
            node.vram_available_gib = 0.0
            node.active_tasks += 2
        return snapshot
    if scenario == "memory_pressure":
        affected = rng.sample(snapshot.nodes, max(1, len(snapshot.nodes) // 3))
        for node in affected:
            node.ram_available_gib = min(node.ram_available_gib, rng.uniform(0.5, 4.0))
            node.active_tasks += 1
        return snapshot
    raise ValueError(f"unknown scenario: {scenario}")


def baseline_record(policy: str, task: TaskIntent, candidates: list[CandidatePlan], sequence: int) -> DecisionRecord:
    started = time.perf_counter_ns()
    selected = select_baseline(policy, candidates, sequence=sequence)
    latency_ms = (time.perf_counter_ns() - started) / 1e6
    return DecisionRecord(
        task_id=task.task_id,
        policy=f"{policy}_static",
        selected_plan=selected,
        planning_latency_ms=latency_ms,
    )


def reactive_record(policy: str, task: TaskIntent, candidates: list[CandidatePlan]) -> DecisionRecord:
    started = time.perf_counter_ns()
    selected = select_baseline("joint_fixed", candidates) if policy == "joint_reactive" else select_oracle(candidates, task)
    latency_ms = (time.perf_counter_ns() - started) / 1e6
    return DecisionRecord(task_id=task.task_id, policy=policy, selected_plan=selected, planning_latency_ms=latency_ms)


def execute_record(
    record: DecisionRecord,
    task: TaskIntent,
    actual: ClusterSnapshot,
    planner: CandidatePlanner,
    recovery: bool,
    control_interval_ms: float,
) -> ExecutionOutcome:
    plan = record.selected_plan
    if plan is None:
        invalid_agent_action = bool(record.agent_decision and record.agent_decision.action == "select")
        return ExecutionOutcome(
            task_id=task.task_id,
            policy=record.policy,
            success=False,
            safety_violation=invalid_agent_action,
            slo_violation=True,
            recovered=False,
            used_fallback=record.used_fallback,
            decision_latency_ms=record.planning_latency_ms + record.agent_latency_ms,
            agent_latency_ms=record.agent_latency_ms,
            reason="no executable plan",
        )
    node = actual.node(plan.node_id)
    if node is None:
        return ExecutionOutcome(
            task_id=task.task_id,
            policy=record.policy,
            selected_node_id=plan.node_id,
            success=False,
            safety_violation=True,
            slo_violation=True,
            used_fallback=record.used_fallback,
            reason="selected node disappeared",
        )
    actual_plan = planner.evaluate(task, actual, node)
    if not actual_plan.feasible:
        return ExecutionOutcome(
            task_id=task.task_id,
            policy=record.policy,
            selected_node_id=plan.node_id,
            success=False,
            safety_violation=True,
            slo_violation=True,
            used_fallback=record.used_fallback,
            decision_latency_ms=record.planning_latency_ms + record.agent_latency_ms,
            agent_latency_ms=record.agent_latency_ms,
            reason=actual_plan.infeasible_reason,
        )
    slo_violation = bool(task.deadline_ms and actual_plan.predicted_completion_ms > task.deadline_ms)
    utility = candidate_objective(actual_plan, profile_weights(task.evaluation_profile))
    decision_latency = record.planning_latency_ms + record.agent_latency_ms
    return ExecutionOutcome(
        task_id=task.task_id,
        policy=record.policy,
        selected_node_id=plan.node_id,
        success=True,
        slo_violation=slo_violation,
        actual_completion_ms=actual_plan.predicted_completion_ms,
        actual_cost_usd=actual_plan.predicted_cost_usd,
        utility_score=utility,
        recovered=recovery,
        recovery_time_ms=(control_interval_ms + decision_latency) if recovery else None,
        used_fallback=record.used_fallback,
        decision_latency_ms=decision_latency,
        agent_latency_ms=record.agent_latency_ms,
    )


def reference_requires_recovery(
    task: TaskIntent,
    reference: CandidatePlan | None,
    actual: ClusterSnapshot,
    planner: CandidatePlanner,
    degradation_threshold: float = 0.10,
) -> bool:
    """Return whether the disruption materially invalidated/degraded a frozen plan.

    Recovery metrics are only eligible when the pre-disruption fixed-joint plan
    became infeasible, crossed a task deadline, or slowed by at least the frozen
    threshold. This avoids labeling every reactive decision as a recovery.
    """

    if reference is None:
        return True
    node = actual.node(reference.node_id)
    if node is None:
        return True
    fresh = planner.evaluate(task, actual, node)
    if not fresh.feasible:
        return True
    if task.deadline_ms is not None:
        crossed_deadline = reference.predicted_completion_ms <= task.deadline_ms < fresh.predicted_completion_ms
        if crossed_deadline:
            return True
    return fresh.predicted_completion_ms > reference.predicted_completion_ms * (1.0 + degradation_threshold)


def run_experiment(config: ExperimentConfig) -> tuple[list[dict], dict]:
    started_at = datetime.now(timezone.utc).isoformat()
    rows: list[dict] = []
    for seed in config.seeds:
        base = generate_snapshot(seed, config.nodes)
        tasks = generate_tasks(seed, config.tasks_per_seed, config.deadline_scale)
        planner = CandidatePlanner(PlannerConfig(top_k=config.top_k))
        verifier = PlanVerifier(planner)
        controller = ClosedLoopController(planner, verifier)
        agent = build_agent(config, seed)

        for scenario in config.scenarios:
            actual = perturb_snapshot(base, scenario, seed)
            context_setter = getattr(agent, "set_context", None)
            if context_setter is not None:
                context_setter(seed=seed, scenario=scenario)
            for sequence, task in enumerate(tasks):
                initial_candidates = planner.generate(task, base)
                actual_candidates = planner.generate(task, actual)
                reference = select_baseline("joint_fixed", initial_candidates)
                recovery_required = scenario != "stable" and reference_requires_recovery(
                    task, reference, actual, planner
                )
                records = [baseline_record(policy, task, initial_candidates, sequence) for policy in STATIC_POLICIES]
                records.append(reactive_record("joint_reactive", task, actual_candidates))
                records.append(reactive_record("oracle_reactive", task, actual_candidates))

                planning_started = time.perf_counter_ns()
                agent_candidates = planner.generate(task, actual)
                planning_ms = (time.perf_counter_ns() - planning_started) / 1e6
                task_json = json.dumps(task.model_dump(mode="json"), separators=(",", ":"), sort_keys=True)
                initial_candidates_json = json.dumps(
                    [candidate.model_dump(mode="json") for candidate in initial_candidates],
                    separators=(",", ":"),
                    sort_keys=True,
                )
                actual_candidates_json = json.dumps(
                    [candidate.model_dump(mode="json") for candidate in agent_candidates],
                    separators=(",", ":"),
                    sort_keys=True,
                )
                snapshot_bundle_json = (
                    json.dumps(
                        {
                            "base": base.model_dump(mode="json"),
                            "actual": actual.model_dump(mode="json"),
                        },
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    if sequence == 0
                    else None
                )
                agent_started = time.perf_counter_ns()
                decision = agent.select(task, agent_candidates)
                agent_ms = (time.perf_counter_ns() - agent_started) / 1e6
                replayed_ms = getattr(agent, "last_replayed_latency_ms", None)
                if replayed_ms is not None:
                    agent_ms = float(replayed_ms)  # a cache replay keeps the original call latency
                records.append(
                    unverified_decision_record(
                        task,
                        actual,
                        decision,
                        agent_candidates,
                        policy_name=f"{config.agent_backend}_agent_unverified",
                        planning_latency_ms=planning_ms,
                        agent_latency_ms=agent_ms,
                    )
                )
                records.append(
                    controller.schedule_with_decision(
                        task,
                        actual,
                        decision,
                        policy_name=f"{config.agent_backend}_agent_verified",
                        allow_fallback=True,
                        candidates=agent_candidates,
                        planning_latency_ms=planning_ms,
                        agent_latency_ms=agent_ms,
                    )
                )

                evaluated: list[tuple[DecisionRecord, ExecutionOutcome]] = []
                for record in records:
                    reactive = not record.policy.endswith("_static")
                    outcome = execute_record(
                        record,
                        task,
                        actual,
                        planner,
                        recovery=reactive and recovery_required,
                        control_interval_ms=config.control_interval_ms,
                    )
                    evaluated.append((record, outcome))

                oracle_outcome = next(outcome for record, outcome in evaluated if record.policy == "oracle_reactive")
                for record, outcome in evaluated:
                    is_evidence_row = record.policy == f"{config.agent_backend}_agent_verified"
                    if outcome.utility_score is not None and oracle_outcome.utility_score not in (None, 0):
                        normalized_regret = (outcome.utility_score - oracle_outcome.utility_score) / oracle_outcome.utility_score
                    else:
                        normalized_regret = None
                    row = {
                        "seed": seed,
                        "scenario": scenario,
                        "task_id": task.task_id,
                        "task_kind": task.metadata.get("kind"),
                        "intent_phrase_id": task.metadata.get("intent_phrase_id"),
                        "intent_profile": task.evaluation_profile,
                        "recovery_required": recovery_required,
                        "artifact_schema_version": config.artifact_schema_version,
                        "task_json": task_json if is_evidence_row else None,
                        "initial_candidates_json": initial_candidates_json if is_evidence_row else None,
                        "actual_candidates_json": actual_candidates_json if is_evidence_row else None,
                        "snapshot_bundle_json": snapshot_bundle_json if is_evidence_row else None,
                        **outcome.model_dump(),
                        "agent_inferred_profile": (
                            record.agent_decision.inferred_profile
                            if record.agent_decision and record.agent_decision.api_status != "error"
                            else None
                        ),
                        "agent_action_valid": (
                            record.verification.accepted if record.verification is not None else None
                        ),
                        "agent_backend": record.agent_decision.backend if record.agent_decision else None,
                        "agent_requested_model_id": (
                            record.agent_decision.requested_model_id if record.agent_decision else None
                        ),
                        "agent_model_id": record.agent_decision.model_id if record.agent_decision else None,
                        "agent_api_status": record.agent_decision.api_status if record.agent_decision else None,
                        "agent_response_id": record.agent_decision.response_id if record.agent_decision else None,
                        "agent_input_tokens": record.agent_decision.input_tokens if record.agent_decision else None,
                        "agent_output_tokens": record.agent_decision.output_tokens if record.agent_decision else None,
                        "agent_error_type": record.agent_decision.error_type if record.agent_decision else None,
                        "agent_error_message": record.agent_decision.error_message if record.agent_decision else None,
                        "agent_decision_json": (
                            record.agent_decision.model_dump_json() if record.agent_decision else None
                        ),
                        "agent_action": record.agent_decision.action if record.agent_decision else None,
                        "agent_plan_id": record.agent_decision.plan_id if record.agent_decision else None,
                        "agent_confidence": record.agent_decision.confidence if record.agent_decision else None,
                        "agent_rationale": record.agent_decision.rationale if record.agent_decision else None,
                        "agent_prompt_version": record.agent_decision.prompt_version if record.agent_decision else None,
                        "agent_prompt_sha256": record.agent_decision.prompt_sha256 if record.agent_decision else None,
                        "verification_reasons_json": (
                            json.dumps(record.verification.reasons, separators=(",", ":"))
                            if record.verification is not None
                            else None
                        ),
                        "verification_selected_plan_id": (
                            record.verification.selected_plan_id if record.verification is not None else None
                        ),
                        "oracle_node_agreement": (
                            outcome.selected_node_id == oracle_outcome.selected_node_id
                            if outcome.selected_node_id is not None and oracle_outcome.selected_node_id is not None
                            else False
                        ),
                        "normalized_regret_to_oracle": normalized_regret,
                    }
                    rows.append(row)
    summary = summarize(rows, config)
    summary["run_metadata"] = {
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "artifact_schema_version": config.artifact_schema_version,
        "objective_scale": {
            "version": OBJECTIVE_SCALE_VERSION,
            "cost_usd_to_ms": COST_USD_TO_MS,
            "risk_to_ms": RISK_TO_MS,
            "fragmentation_to_ms": FRAGMENTATION_TO_MS,
        },
    }
    return rows, summary


def percentile(values: Iterable[float], q: float) -> float | None:
    materialized = list(values)
    return float(np.quantile(np.asarray(materialized, dtype=np.float64), q)) if materialized else None


def bootstrap_mean_ci(values: Iterable[float], seed: int = 2026, resamples: int = 2_000) -> list[float] | None:
    materialized = np.asarray(list(values), dtype=np.float64)
    if materialized.size == 0:
        return None
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, materialized.size, size=(resamples, materialized.size))
    means = materialized[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def summarize(rows: list[dict], config: ExperimentConfig) -> dict:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["policy"]), str(row["scenario"]))].append(row)
    summaries: list[dict] = []
    for (policy, scenario), group in sorted(grouped.items()):
        completions = [float(row["actual_completion_ms"]) for row in group if row["actual_completion_ms"] is not None]
        utilities = [float(row["utility_score"]) for row in group if row["utility_score"] is not None]
        decision_latencies = [float(row["decision_latency_ms"]) for row in group]
        regrets = [float(row["normalized_regret_to_oracle"]) for row in group if row["normalized_regret_to_oracle"] is not None]
        inferred = [row for row in group if row["agent_inferred_profile"] is not None]
        agent_rows = [row for row in group if row["agent_backend"] is not None]
        input_tokens = [int(row["agent_input_tokens"]) for row in agent_rows if row["agent_input_tokens"] is not None]
        output_tokens = [int(row["agent_output_tokens"]) for row in agent_rows if row["agent_output_tokens"] is not None]
        decisions_json: list[dict] = []
        for row in agent_rows:
            raw = row.get("agent_decision_json")
            if raw:
                try:
                    decisions_json.append(json.loads(raw))
                except (TypeError, ValueError):
                    continue
        reasoning_tokens = [
            int(item["reasoning_tokens"]) for item in decisions_json if item.get("reasoning_tokens") is not None
        ]
        retry_counts = [int(item["retry_count"]) for item in decisions_json if item.get("retry_count") is not None]
        recovery_eligible = [row for row in group if row["recovery_required"]]
        success_values = [1.0 if row["success"] else 0.0 for row in group]
        slo_values = [1.0 if row["slo_violation"] else 0.0 for row in group]
        safety_values = [1.0 if row["safety_violation"] else 0.0 for row in group]
        summaries.append(
            {
                "policy": policy,
                "scenario": scenario,
                "requests": len(group),
                "success_rate": fmean(success_values),
                "success_rate_bootstrap_95ci": bootstrap_mean_ci(success_values),
                "slo_violation_rate": fmean(slo_values),
                "slo_violation_rate_bootstrap_95ci": bootstrap_mean_ci(slo_values),
                "safety_violation_rate": fmean(safety_values),
                "safety_violation_rate_bootstrap_95ci": bootstrap_mean_ci(safety_values),
                "fallback_rate": fmean(1.0 if row["used_fallback"] else 0.0 for row in group),
                "recovery_eligible_requests": len(recovery_eligible),
                "recovery_success_rate": (
                    fmean(1.0 if row["recovered"] else 0.0 for row in recovery_eligible)
                    if recovery_eligible
                    else None
                ),
                "mean_completion_ms": fmean(completions) if completions else None,
                "p95_completion_ms": percentile(completions, 0.95),
                "mean_utility": fmean(utilities) if utilities else None,
                "p95_decision_latency_ms": percentile(decision_latencies, 0.95),
                "oracle_node_agreement_rate": fmean(1.0 if row["oracle_node_agreement"] else 0.0 for row in group),
                "mean_normalized_regret_to_oracle": fmean(regrets) if regrets else None,
                "mean_normalized_regret_bootstrap_95ci": bootstrap_mean_ci(regrets),
                "intent_profile_accuracy": (
                    fmean(1.0 if row["agent_inferred_profile"] == row["intent_profile"] else 0.0 for row in inferred)
                    if inferred
                    else None
                ),
                "intent_profile_coverage": len(inferred) / len(group) if agent_rows else None,
                "intent_profile_end_to_end_accuracy": (
                    fmean(
                        1.0 if row["agent_inferred_profile"] == row["intent_profile"] else 0.0
                        for row in group
                    )
                    if agent_rows
                    else None
                ),
                "agent_api_error_rate": (
                    fmean(1.0 if row["agent_api_status"] == "error" else 0.0 for row in agent_rows)
                    if agent_rows
                    else None
                ),
                "mean_agent_input_tokens": fmean(input_tokens) if input_tokens else None,
                "mean_agent_output_tokens": fmean(output_tokens) if output_tokens else None,
                "mean_agent_reasoning_tokens": fmean(reasoning_tokens) if reasoning_tokens else None,
                "agent_retry_rate": (
                    fmean(1.0 if count > 0 else 0.0 for count in retry_counts) if retry_counts else None
                ),
            }
        )
    return {
        "config": config.model_dump(),
        "disclosure": (
            "Keyword-agent runs validate the pipeline and guardrail mechanism only; they are not LLM results. "
            "Model decisions used by the artifact come from the frozen campaign records; "
            "this offline replay makes no model calls."
        ),
        "summaries": summaries,
    }


def write_results(rows: list[dict], summary: dict, output_dir: str | Path) -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows_path = output / "requests.csv"
    summary_path = output / "summary.json"
    with rows_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return rows_path, summary_path
