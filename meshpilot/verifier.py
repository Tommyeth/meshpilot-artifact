from __future__ import annotations

from dataclasses import dataclass

from .models import AgentDecision, CandidatePlan, ClusterSnapshot, TaskIntent, VerificationResult
from .planner import CandidatePlanner, finite_candidate


@dataclass(slots=True)
class VerifierConfig:
    reject_stale_snapshot: bool = True
    max_telemetry_age_s: float = 120.0


class PlanVerifier:
    """Validates a plan identifier against fresh deterministic state.

    The LLM never supplies resource values or commands. It can only select an
    opaque candidate ID or abstain. All numerical fields are recomputed here.
    """

    def __init__(self, planner: CandidatePlanner, config: VerifierConfig | None = None) -> None:
        self.planner = planner
        self.config = config or VerifierConfig(max_telemetry_age_s=planner.config.max_telemetry_age_s)

    def verify(
        self,
        task: TaskIntent,
        snapshot: ClusterSnapshot,
        candidates: list[CandidatePlan],
        decision: AgentDecision,
    ) -> VerificationResult:
        reasons: list[str] = []
        if decision.action != "select":
            reasons.append("agent abstained")
            return VerificationResult(accepted=False, reasons=reasons)

        selected = next((candidate for candidate in candidates if candidate.plan_id == decision.plan_id), None)
        if selected is None:
            return VerificationResult(accepted=False, reasons=["unknown or hallucinated plan_id"])
        if selected.task_id != task.task_id:
            reasons.append("candidate belongs to another task")
        if self.config.reject_stale_snapshot and selected.snapshot_version != snapshot.version:
            reasons.append("candidate snapshot version is stale")
        if not finite_candidate(selected):
            reasons.append("candidate contains non-finite metrics")

        node = snapshot.node(selected.node_id)
        if node is None:
            reasons.append("selected node no longer exists")
        else:
            fresh = self.planner.evaluate(task, snapshot, node)
            if not fresh.feasible:
                reasons.append(fresh.infeasible_reason or "fresh candidate is infeasible")
            if fresh.plan_id != selected.plan_id:
                reasons.append("candidate was not generated from current snapshot")

        return VerificationResult(
            accepted=not reasons,
            reasons=reasons,
            selected_plan_id=selected.plan_id if not reasons else None,
        )
