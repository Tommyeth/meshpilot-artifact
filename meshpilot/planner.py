from __future__ import annotations

import math
from dataclasses import dataclass

from .models import (
    PROFILE_WEIGHTS,
    CandidatePlan,
    ClusterSnapshot,
    LinkState,
    NodeState,
    ObjectiveWeights,
    TaskIntent,
)


# Fixed millisecond-equivalent scales for the configured objective.
# These affect placement ranking; hard feasibility is checked separately.
OBJECTIVE_SCALE_VERSION = "meshpilot-ms-equivalent-v2"
COST_USD_TO_MS = 5_000_000.0
RISK_TO_MS = 20_000.0
FRAGMENTATION_TO_MS = 1_000.0


@dataclass(slots=True)
class PlannerConfig:
    max_telemetry_age_s: float = 120.0
    queue_penalty_per_task: float = 0.15
    loss_transfer_cap: float = 5.0
    default_reservation_mbps: float = 50.0
    top_k: int = 16


def profile_weights(profile: str) -> ObjectiveWeights:
    return PROFILE_WEIGHTS.get(profile, PROFILE_WEIGHTS["balanced"])


def candidate_objective(candidate: CandidatePlan, weights: ObjectiveWeights) -> float:
    """Dimensioned scalar used only for comparable placement ranking.

    Cost, risk, and fragmentation are converted into millisecond-equivalent
    penalties. The conversion constants are fixed before a campaign and must be
    swept in sensitivity experiments.
    """

    return (
        weights.latency * candidate.predicted_completion_ms
        + weights.cost * candidate.predicted_cost_usd * COST_USD_TO_MS
        + weights.risk * candidate.risk_score * RISK_TO_MS
        + weights.fragmentation * candidate.dominant_share_after * FRAGMENTATION_TO_MS
    )


class CandidatePlanner:
    def __init__(self, config: PlannerConfig | None = None) -> None:
        self.config = config or PlannerConfig()

    def generate(self, task: TaskIntent, snapshot: ClusterSnapshot, include_infeasible: bool = False) -> list[CandidatePlan]:
        plans = [self.evaluate(task, snapshot, node) for node in snapshot.nodes]
        if not include_infeasible:
            plans = [plan for plan in plans if plan.feasible]
        # The joint objective is used only to bound LLM context. It does not
        # prune by compute alone, which would hide network-optimal placements.
        plans.sort(key=lambda plan: (not plan.feasible, candidate_objective(plan, profile_weights("balanced")), plan.node_id))
        return plans[: self.config.top_k]

    def evaluate(self, task: TaskIntent, snapshot: ClusterSnapshot, node: NodeState) -> CandidatePlan:
        reason = self._hard_infeasibility(task, snapshot, node)
        link = snapshot.link(task.client_id, node.node_id)
        if link is None:
            reason = reason or "missing client-to-node link telemetry"

        compute_ms = task.estimated_compute_ms / max(node.compute_speed, 1e-9)
        queue_ms = compute_ms * self.config.queue_penalty_per_task * node.active_tasks
        reserved_mbps = self._reservation(task, link, compute_ms + queue_ms) if link else 0.0
        network_ms = self._network_time_ms(task, link, reserved_mbps) if link else float("inf")
        completion_ms = compute_ms + queue_ms + network_ms
        cost = self._cost(task, node, completion_ms)
        dominant_share = self._dominant_share_after(task, node)
        risk = self._risk(node, link)

        if task.deadline_is_hard and task.deadline_ms and completion_ms > task.deadline_ms:
            reason = reason or "predicted completion exceeds hard deadline"
        if task.max_cost_usd is not None and cost > task.max_cost_usd:
            reason = reason or "predicted cost exceeds hard budget"

        return CandidatePlan(
            plan_id=f"{task.task_id}:{snapshot.version}:{node.node_id}",
            task_id=task.task_id,
            node_id=node.node_id,
            snapshot_version=snapshot.version,
            allocated_cpu_cores=task.cpu_cores,
            allocated_ram_gib=task.ram_gib,
            allocated_gpu_units=task.gpu_units,
            allocated_vram_gib=task.vram_gib,
            reserved_bandwidth_mbps=reserved_mbps,
            predicted_compute_ms=compute_ms,
            predicted_network_ms=network_ms,
            predicted_queue_ms=queue_ms,
            predicted_completion_ms=completion_ms,
            predicted_cost_usd=cost,
            dominant_share_after=dominant_share,
            risk_score=risk,
            feasible=reason is None,
            infeasible_reason=reason,
        )

    def _hard_infeasibility(self, task: TaskIntent, snapshot: ClusterSnapshot, node: NodeState) -> str | None:
        if not node.online:
            return "node offline"
        if node.trust_level < task.minimum_trust:
            return "node trust below requirement"
        if node.cpu_available + 1e-9 < task.cpu_cores:
            return "insufficient CPU"
        if node.ram_available_gib + 1e-9 < task.ram_gib:
            return "insufficient RAM"
        if node.gpu_available + 1e-9 < task.gpu_units:
            return "insufficient GPU"
        if node.vram_available_gib + 1e-9 < task.vram_gib:
            return "insufficient VRAM"
        link = snapshot.link(task.client_id, node.node_id)
        if link is None:
            return "missing link"
        if link.loss_rate > task.max_loss_rate:
            return "loss exceeds requirement"
        if link.telemetry_age_s > self.config.max_telemetry_age_s:
            return "stale link telemetry"
        if link.reservable_mbps + 1e-9 < task.min_bandwidth_mbps:
            return "insufficient reservable bandwidth"
        return None

    def _reservation(self, task: TaskIntent, link: LinkState, non_network_ms: float) -> float:
        desired = max(task.min_bandwidth_mbps, min(self.config.default_reservation_mbps, link.reservable_mbps))
        if task.deadline_ms is not None:
            remaining_s = (task.deadline_ms - non_network_ms - link.rtt_ms) / 1_000.0
            if remaining_s > 0:
                transfer_megabits = (task.input_mib + task.output_mib) * 8.388608
                desired = max(desired, transfer_megabits / remaining_s)
        return min(desired, link.reservable_mbps)

    def _network_time_ms(self, task: TaskIntent, link: LinkState, reservation_mbps: float) -> float:
        transfer_megabits = (task.input_mib + task.output_mib) * 8.388608
        payload_ms = transfer_megabits / max(reservation_mbps, 1e-9) * 1_000.0
        loss_multiplier = min(1.0 / max(1.0 - link.loss_rate, 0.01), self.config.loss_transfer_cap)
        return link.rtt_ms + 2.0 * link.jitter_ms + payload_ms * loss_multiplier

    @staticmethod
    def _cost(task: TaskIntent, node: NodeState, completion_ms: float) -> float:
        hours = completion_ms / 3_600_000.0
        compute = hours * (
            task.cpu_cores * node.price_per_cpu_hour + task.gpu_units * node.price_per_gpu_hour
        )
        egress_gb = task.output_mib / 1024.0
        return compute + egress_gb * node.price_per_gb_egress

    @staticmethod
    def _dominant_share_after(task: TaskIntent, node: NodeState) -> float:
        allocated_cpu = node.cpu_capacity - node.cpu_available + task.cpu_cores
        allocated_ram = node.ram_capacity_gib - node.ram_available_gib + task.ram_gib
        shares = [allocated_cpu / node.cpu_capacity, allocated_ram / node.ram_capacity_gib]
        if node.gpu_capacity > 0:
            shares.append((node.gpu_capacity - node.gpu_available + task.gpu_units) / node.gpu_capacity)
        if node.vram_capacity_gib > 0:
            shares.append((node.vram_capacity_gib - node.vram_available_gib + task.vram_gib) / node.vram_capacity_gib)
        return max(shares)

    def _risk(self, node: NodeState, link: LinkState | None) -> float:
        if link is None:
            return 1.0
        freshness = min(link.telemetry_age_s / max(self.config.max_telemetry_age_s, 1e-9), 1.0)
        network = min(1.0, 3.0 * link.loss_rate + link.jitter_ms / 100.0)
        return min(1.0, 0.55 * node.failure_probability + 0.25 * freshness + 0.20 * network)


def select_baseline(policy: str, candidates: list[CandidatePlan], sequence: int = 0) -> CandidatePlan | None:
    if not candidates:
        return None
    if policy == "round_robin":
        return sorted(candidates, key=lambda candidate: candidate.node_id)[sequence % len(candidates)]
    if policy == "least_loaded":
        return min(candidates, key=lambda candidate: (candidate.dominant_share_after, candidate.predicted_completion_ms))
    if policy == "network_greedy":
        return min(candidates, key=lambda candidate: (candidate.predicted_network_ms, candidate.predicted_completion_ms))
    if policy == "dominant_resource":
        # Placement proxy inspired by DRF/Tetris. It is deliberately not named
        # DRF because true DRF allocates across users rather than choosing nodes.
        return min(candidates, key=lambda candidate: (candidate.dominant_share_after, candidate.predicted_cost_usd))
    if policy == "joint_fixed":
        weights = profile_weights("balanced")
        return min(candidates, key=lambda candidate: candidate_objective(candidate, weights))
    raise ValueError(f"unknown baseline policy: {policy}")


def select_oracle(candidates: list[CandidatePlan], task: TaskIntent) -> CandidatePlan | None:
    if not candidates:
        return None
    weights = profile_weights(task.evaluation_profile)
    return min(candidates, key=lambda candidate: candidate_objective(candidate, weights))


def finite_candidate(candidate: CandidatePlan) -> bool:
    return all(
        math.isfinite(value)
        for value in (
            candidate.predicted_compute_ms,
            candidate.predicted_network_ms,
            candidate.predicted_completion_ms,
            candidate.predicted_cost_usd,
            candidate.risk_score,
        )
    )
