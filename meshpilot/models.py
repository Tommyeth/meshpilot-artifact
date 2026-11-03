from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


IntentProfile = Literal["balanced", "latency", "cost", "resilience"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ObjectiveWeights(BaseModel):
    latency: float = Field(default=1.0, ge=0)
    cost: float = Field(default=0.20, ge=0)
    risk: float = Field(default=0.50, ge=0)
    fragmentation: float = Field(default=0.10, ge=0)


PROFILE_WEIGHTS: dict[str, ObjectiveWeights] = {
    "balanced": ObjectiveWeights(latency=1.0, cost=0.20, risk=0.50, fragmentation=0.10),
    "latency": ObjectiveWeights(latency=1.8, cost=0.05, risk=0.35, fragmentation=0.05),
    "cost": ObjectiveWeights(latency=0.35, cost=2.0, risk=0.30, fragmentation=0.10),
    "resilience": ObjectiveWeights(latency=0.70, cost=0.10, risk=2.0, fragmentation=0.10),
}


class TaskIntent(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid4()))
    client_id: str = "client-0"
    operator_intent: str = "Balance latency, cost, and reliability."
    cpu_cores: float = Field(default=1.0, gt=0)
    ram_gib: float = Field(default=1.0, gt=0)
    gpu_units: float = Field(default=0.0, ge=0)
    vram_gib: float = Field(default=0.0, ge=0)
    input_mib: float = Field(default=1.0, ge=0)
    output_mib: float = Field(default=0.1, ge=0)
    estimated_compute_ms: float = Field(default=1000.0, gt=0)
    deadline_ms: float | None = Field(default=None, gt=0)
    deadline_is_hard: bool = False
    max_cost_usd: float | None = Field(default=None, gt=0)
    min_bandwidth_mbps: float = Field(default=0.0, ge=0)
    max_loss_rate: float = Field(default=0.20, ge=0, le=1)
    minimum_trust: int = Field(default=1, ge=0, le=3)
    evaluation_profile: IntentProfile = "balanced"
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)

    @model_validator(mode="after")
    def gpu_requires_vram(self) -> "TaskIntent":
        if self.gpu_units == 0 and self.vram_gib > 0:
            raise ValueError("vram_gib requires gpu_units > 0")
        return self


class NodeState(BaseModel):
    node_id: str
    region: str = "unknown"
    provider: str = "unknown"
    online: bool = True
    cpu_capacity: float = Field(gt=0)
    cpu_available: float = Field(ge=0)
    ram_capacity_gib: float = Field(gt=0)
    ram_available_gib: float = Field(ge=0)
    gpu_capacity: float = Field(default=0.0, ge=0)
    gpu_available: float = Field(default=0.0, ge=0)
    vram_capacity_gib: float = Field(default=0.0, ge=0)
    vram_available_gib: float = Field(default=0.0, ge=0)
    compute_speed: float = Field(default=1.0, gt=0)
    active_tasks: int = Field(default=0, ge=0)
    price_per_cpu_hour: float = Field(default=0.02, ge=0)
    price_per_gpu_hour: float = Field(default=0.0, ge=0)
    price_per_gb_egress: float = Field(default=0.0, ge=0)
    failure_probability: float = Field(default=0.01, ge=0, le=1)
    trust_level: int = Field(default=1, ge=0, le=3)

    @model_validator(mode="after")
    def availability_not_above_capacity(self) -> "NodeState":
        pairs = (
            (self.cpu_available, self.cpu_capacity, "cpu"),
            (self.ram_available_gib, self.ram_capacity_gib, "ram"),
            (self.gpu_available, self.gpu_capacity, "gpu"),
            (self.vram_available_gib, self.vram_capacity_gib, "vram"),
        )
        for available, capacity, name in pairs:
            if available > capacity + 1e-9:
                raise ValueError(f"{name} availability exceeds capacity")
        return self


class LinkState(BaseModel):
    client_id: str
    node_id: str
    bandwidth_mbps: float = Field(gt=0)
    rtt_ms: float = Field(ge=0)
    loss_rate: float = Field(default=0.0, ge=0, le=1)
    jitter_ms: float = Field(default=0.0, ge=0)
    telemetry_age_s: float = Field(default=0.0, ge=0)
    reservable_fraction: float = Field(default=0.80, gt=0, le=1)

    @property
    def reservable_mbps(self) -> float:
        return self.bandwidth_mbps * self.reservable_fraction


class ClusterSnapshot(BaseModel):
    snapshot_id: str = Field(default_factory=lambda: str(uuid4()))
    version: int = Field(default=1, ge=1)
    observed_at: datetime = Field(default_factory=utc_now)
    nodes: list[NodeState]
    links: list[LinkState]

    def node(self, node_id: str) -> NodeState | None:
        return next((node for node in self.nodes if node.node_id == node_id), None)

    def link(self, client_id: str, node_id: str) -> LinkState | None:
        return next((link for link in self.links if link.client_id == client_id and link.node_id == node_id), None)


class CandidatePlan(BaseModel):
    plan_id: str
    task_id: str
    node_id: str
    snapshot_version: int
    allocated_cpu_cores: float
    allocated_ram_gib: float
    allocated_gpu_units: float
    allocated_vram_gib: float
    reserved_bandwidth_mbps: float
    predicted_compute_ms: float
    predicted_network_ms: float
    predicted_queue_ms: float
    predicted_completion_ms: float
    predicted_cost_usd: float
    dominant_share_after: float
    risk_score: float
    feasible: bool = True
    infeasible_reason: str | None = None


class AgentDecision(BaseModel):
    action: Literal["select", "abstain"]
    plan_id: str | None = None
    rationale: str = Field(max_length=500)
    confidence: float = Field(ge=0, le=1)
    inferred_profile: IntentProfile = "balanced"
    backend: str | None = None
    requested_model_id: str | None = None
    model_id: str | None = None
    api_status: str | None = None
    response_id: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    error_type: str | None = None
    error_message: str | None = Field(default=None, max_length=500)
    prompt_version: str | None = None
    prompt_sha256: str | None = None
    # Optional provenance used by the OpenRouter backend; serialized inside
    # agent_decision_json and never read by the evaluation logic.
    reasoning_tokens: int | None = Field(default=None, ge=0)
    reasoning_max_tokens: int | None = Field(default=None, ge=0)
    reasoning_effort: str | None = None
    provider: str | None = None
    retry_count: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)
    # Continuous objective weights returned by the "weights" prompt variant.
    objective_weights: dict[str, float] | None = None

    @model_validator(mode="after")
    def selected_plan_is_present(self) -> "AgentDecision":
        if self.action == "select" and not self.plan_id:
            raise ValueError("select action requires plan_id")
        return self


class VerificationResult(BaseModel):
    accepted: bool
    reasons: list[str] = Field(default_factory=list)
    selected_plan_id: str | None = None


class DecisionRecord(BaseModel):
    task_id: str
    policy: str
    selected_plan: CandidatePlan | None = None
    agent_decision: AgentDecision | None = None
    verification: VerificationResult | None = None
    used_fallback: bool = False
    planning_latency_ms: float = Field(default=0.0, ge=0)
    agent_latency_ms: float = Field(default=0.0, ge=0)


class ExecutionOutcome(BaseModel):
    task_id: str
    policy: str
    selected_node_id: str | None = None
    success: bool
    safety_violation: bool = False
    slo_violation: bool = False
    actual_completion_ms: float | None = None
    actual_cost_usd: float | None = None
    utility_score: float | None = None
    recovered: bool = False
    recovery_time_ms: float | None = None
    used_fallback: bool = False
    decision_latency_ms: float = 0.0
    agent_latency_ms: float = 0.0
    reason: str | None = None
