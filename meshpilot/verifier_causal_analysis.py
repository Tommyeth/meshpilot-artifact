from __future__ import annotations

import argparse
import csv
import hashlib
import json
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .models import AgentDecision, CandidatePlan, ClusterSnapshot, LinkState, NodeState, TaskIntent
from .reservation import ReservationLedger
from .planner import CandidatePlanner, finite_candidate, select_baseline
from .artifact_io import read_rows
from .verifier import PlanVerifier


@dataclass(slots=True)
class ProposalCase:
    seed: str
    scenario: str
    task: TaskIntent
    snapshot: ClusterSnapshot
    proposal: CandidatePlan


@dataclass(frozen=True, slots=True)
class FaultSpec:
    name: str
    expected_reason: str
    prepare: Callable[[TaskIntent, ClusterSnapshot, CandidatePlan], tuple[TaskIntent, ClusterSnapshot, CandidatePlan]]
    requires_gpu: bool = False
    # None means that either a fresh fallback or abstention is a valid safe
    # response; False is reserved for the deliberately infeasible surface.
    expect_fallback: bool | None = None


def _decision(plan_id: str) -> AgentDecision:
    return AgentDecision(
        action="select",
        plan_id=plan_id,
        rationale="frozen causal verifier trial",
        confidence=1.0,
        inferred_profile="balanced",
    )


def _snapshot_index(rows: list[dict[str, str]]) -> dict[tuple[str, str], ClusterSnapshot]:
    output: dict[tuple[str, str], ClusterSnapshot] = {}
    for row in rows:
        raw = row.get("snapshot_bundle_json") or ""
        if not raw:
            continue
        bundle = json.loads(raw)
        output[(row["seed"], row["scenario"])] = ClusterSnapshot.model_validate(bundle["actual"])
    if not output:
        raise ValueError("no execution snapshots found")
    return output


def _proposal_cases(rows: list[dict[str, str]], compiled_policy: str) -> list[ProposalCase]:
    snapshots = _snapshot_index(rows)
    output: list[ProposalCase] = []
    for row in rows:
        if row.get("policy") != compiled_policy:
            continue
        task = TaskIntent.model_validate_json(row["task_json"])
        candidates = [
            CandidatePlan.model_validate(candidate)
            for candidate in json.loads(row["actual_candidates_json"])
        ]
        plan_id = row.get("compiled_plan_id") or row.get("verification_selected_plan_id")
        proposal = next((candidate for candidate in candidates if candidate.plan_id == plan_id), None)
        if proposal is None:
            raise ValueError(f"compiled plan {plan_id!r} is absent for {task.task_id}")
        key = (row["seed"], row["scenario"])
        if key not in snapshots:
            raise ValueError(f"snapshot is absent for {key}")
        output.append(ProposalCase(row["seed"], row["scenario"], task, snapshots[key], proposal))
    if not output:
        raise ValueError(f"compiled policy {compiled_policy!r} is absent")
    return output


def _balanced_sample(cases: list[ProposalCase], limit: int) -> list[ProposalCase]:
    groups: dict[tuple[str, str], list[ProposalCase]] = defaultdict(list)
    for case in cases:
        groups[(case.seed, case.scenario)].append(case)
    for group in groups.values():
        group.sort(key=lambda case: case.task.task_id)
    selected: list[ProposalCase] = []
    keys = sorted(groups, key=lambda key: (int(key[0]), key[1]))
    index = 0
    while len(selected) < min(limit, len(cases)):
        added = False
        for key in keys:
            if index < len(groups[key]):
                selected.append(groups[key][index])
                added = True
                if len(selected) == min(limit, len(cases)):
                    break
        if not added:
            break
        index += 1
    return selected


def _node_and_link(task: TaskIntent, snapshot: ClusterSnapshot, proposal: CandidatePlan) -> tuple[NodeState, LinkState]:
    node = snapshot.node(proposal.node_id)
    link = snapshot.link(task.client_id, proposal.node_id)
    if node is None or link is None:
        raise ValueError("proposal references missing node or link")
    return node, link


def _state_fault(
    task: TaskIntent,
    snapshot: ClusterSnapshot,
    proposal: CandidatePlan,
    mutation: Callable[[TaskIntent, ClusterSnapshot, CandidatePlan], None],
) -> tuple[TaskIntent, ClusterSnapshot, CandidatePlan]:
    fresh = snapshot.model_copy(deep=True)
    fresh.version += 1
    mutation(task, fresh, proposal)
    return task, fresh, proposal


def _control(
    task: TaskIntent, snapshot: ClusterSnapshot, proposal: CandidatePlan
) -> tuple[TaskIntent, ClusterSnapshot, CandidatePlan]:
    return task, snapshot.model_copy(deep=True), proposal


def _stale_version(task: TaskIntent, snapshot: ClusterSnapshot, proposal: CandidatePlan):
    return _state_fault(task, snapshot, proposal, lambda *_: None)


def _offline(task: TaskIntent, snapshot: ClusterSnapshot, proposal: CandidatePlan):
    def mutate(task: TaskIntent, fresh: ClusterSnapshot, proposal: CandidatePlan) -> None:
        node, _ = _node_and_link(task, fresh, proposal)
        node.online = False

    return _state_fault(task, snapshot, proposal, mutate)


def _resource_depletion(field: str, amount: Callable[[TaskIntent], float]):
    def prepare(task: TaskIntent, snapshot: ClusterSnapshot, proposal: CandidatePlan):
        def mutate(task: TaskIntent, fresh: ClusterSnapshot, proposal: CandidatePlan) -> None:
            node, _ = _node_and_link(task, fresh, proposal)
            setattr(node, field, max(0.0, amount(task) - 1e-6))

        return _state_fault(task, snapshot, proposal, mutate)

    return prepare


def _bandwidth_degradation(task: TaskIntent, snapshot: ClusterSnapshot, proposal: CandidatePlan):
    planner = CandidatePlanner()
    node, link = _node_and_link(task, snapshot, proposal)
    minimum = min(link.reservable_mbps * 0.75, max(1.0, proposal.reserved_bandwidth_mbps * 0.5))
    prepared_task = task.model_copy(update={"min_bandwidth_mbps": minimum})
    prepared = planner.evaluate(prepared_task, snapshot, node)
    if not prepared.feasible:
        raise ValueError("failed to prepare feasible bandwidth proposal")

    def mutate(task: TaskIntent, fresh: ClusterSnapshot, proposal: CandidatePlan) -> None:
        _, fresh_link = _node_and_link(task, fresh, proposal)
        fresh_link.bandwidth_mbps = minimum * 0.5 / fresh_link.reservable_fraction

    return _state_fault(prepared_task, snapshot, prepared, mutate)


def _loss_degradation(task: TaskIntent, snapshot: ClusterSnapshot, proposal: CandidatePlan):
    planner = CandidatePlanner()
    node, link = _node_and_link(task, snapshot, proposal)
    maximum = min(0.90, max(link.loss_rate + 0.01, 0.05))
    prepared_task = task.model_copy(update={"max_loss_rate": maximum})
    prepared = planner.evaluate(prepared_task, snapshot, node)

    def mutate(task: TaskIntent, fresh: ClusterSnapshot, proposal: CandidatePlan) -> None:
        _, fresh_link = _node_and_link(task, fresh, proposal)
        fresh_link.loss_rate = min(1.0, maximum + 0.10)

    return _state_fault(prepared_task, snapshot, prepared, mutate)


def _telemetry_staleness(task: TaskIntent, snapshot: ClusterSnapshot, proposal: CandidatePlan):
    def mutate(task: TaskIntent, fresh: ClusterSnapshot, proposal: CandidatePlan) -> None:
        _, link = _node_and_link(task, fresh, proposal)
        link.telemetry_age_s = 121.0

    return _state_fault(task, snapshot, proposal, mutate)


def _trust_downgrade(task: TaskIntent, snapshot: ClusterSnapshot, proposal: CandidatePlan):
    planner = CandidatePlanner()
    node, _ = _node_and_link(task, snapshot, proposal)
    if node.trust_level <= 0:
        raise ValueError("trust downgrade requires positive initial trust")
    prepared_task = task.model_copy(update={"minimum_trust": node.trust_level})
    prepared = planner.evaluate(prepared_task, snapshot, node)

    def mutate(task: TaskIntent, fresh: ClusterSnapshot, proposal: CandidatePlan) -> None:
        fresh_node, _ = _node_and_link(task, fresh, proposal)
        fresh_node.trust_level = task.minimum_trust - 1

    return _state_fault(prepared_task, snapshot, prepared, mutate)


def _deadline_overrun(task: TaskIntent, snapshot: ClusterSnapshot, proposal: CandidatePlan):
    planner = CandidatePlanner()
    node, _ = _node_and_link(task, snapshot, proposal)
    prepared_task = task.model_copy(
        update={"deadline_ms": max(proposal.predicted_completion_ms * 2.0, 1.0), "deadline_is_hard": True}
    )
    prepared = planner.evaluate(prepared_task, snapshot, node)
    if not prepared.feasible:
        raise ValueError("failed to prepare feasible deadline proposal")

    def mutate(task: TaskIntent, fresh: ClusterSnapshot, proposal: CandidatePlan) -> None:
        fresh_node, _ = _node_and_link(task, fresh, proposal)
        fresh_node.compute_speed *= 0.01

    return _state_fault(prepared_task, snapshot, prepared, mutate)


def _budget_overrun(task: TaskIntent, snapshot: ClusterSnapshot, proposal: CandidatePlan):
    planner = CandidatePlanner()
    node, _ = _node_and_link(task, snapshot, proposal)
    prepared_task = task.model_copy(update={"max_cost_usd": max(proposal.predicted_cost_usd * 2.0, 1e-9)})
    prepared = planner.evaluate(prepared_task, snapshot, node)
    if not prepared.feasible:
        raise ValueError("failed to prepare feasible budget proposal")

    def mutate(task: TaskIntent, fresh: ClusterSnapshot, proposal: CandidatePlan) -> None:
        fresh_node, _ = _node_and_link(task, fresh, proposal)
        fresh_node.price_per_cpu_hour = max(1.0, fresh_node.price_per_cpu_hour * 10_000)
        fresh_node.price_per_gpu_hour = max(1.0, fresh_node.price_per_gpu_hour * 10_000)
        fresh_node.price_per_gb_egress = max(1.0, fresh_node.price_per_gb_egress * 10_000)

    return _state_fault(prepared_task, snapshot, prepared, mutate)


def _removed_node(task: TaskIntent, snapshot: ClusterSnapshot, proposal: CandidatePlan):
    def mutate(task: TaskIntent, fresh: ClusterSnapshot, proposal: CandidatePlan) -> None:
        fresh.nodes = [node for node in fresh.nodes if node.node_id != proposal.node_id]
        fresh.links = [link for link in fresh.links if link.node_id != proposal.node_id]

    return _state_fault(task, snapshot, proposal, mutate)


def _no_feasible_fallback(task: TaskIntent, snapshot: ClusterSnapshot, proposal: CandidatePlan):
    def mutate(task: TaskIntent, fresh: ClusterSnapshot, proposal: CandidatePlan) -> None:
        for node in fresh.nodes:
            node.online = False

    return _state_fault(task, snapshot, proposal, mutate)


def _nonfinite_candidate(task: TaskIntent, snapshot: ClusterSnapshot, proposal: CandidatePlan):
    invalid = proposal.model_copy(update={"predicted_completion_ms": float("nan")})
    return task, snapshot.model_copy(deep=True), invalid


def _task_mismatch(task: TaskIntent, snapshot: ClusterSnapshot, proposal: CandidatePlan):
    other = task.model_copy(update={"task_id": f"{task.task_id}-other"})
    return other, snapshot.model_copy(deep=True), proposal


FAULTS = (
    FaultSpec("control", "", _control),
    FaultSpec("stale_version", "stale", _stale_version),
    FaultSpec("offline", "offline", _offline),
    FaultSpec("cpu_depletion", "CPU", _resource_depletion("cpu_available", lambda task: task.cpu_cores)),
    FaultSpec("ram_depletion", "RAM", _resource_depletion("ram_available_gib", lambda task: task.ram_gib)),
    FaultSpec(
        "gpu_depletion",
        "GPU",
        _resource_depletion("gpu_available", lambda task: task.gpu_units),
        requires_gpu=True,
    ),
    FaultSpec(
        "vram_depletion",
        "VRAM",
        _resource_depletion("vram_available_gib", lambda task: task.vram_gib),
        requires_gpu=True,
    ),
    FaultSpec("bandwidth_degradation", "bandwidth", _bandwidth_degradation),
    FaultSpec("loss_degradation", "loss", _loss_degradation),
    FaultSpec("stale_telemetry", "stale link telemetry", _telemetry_staleness),
    FaultSpec("trust_downgrade", "trust", _trust_downgrade),
    FaultSpec("deadline_overrun", "deadline", _deadline_overrun),
    FaultSpec("budget_overrun", "budget", _budget_overrun),
    FaultSpec("removed_node", "no longer exists", _removed_node),
    FaultSpec("no_feasible_fallback", "offline", _no_feasible_fallback, expect_fallback=False),
    FaultSpec("nonfinite_candidate", "non-finite", _nonfinite_candidate),
    FaultSpec("task_mismatch", "another task", _task_mismatch),
)


def _run_fault(case: ProposalCase, spec: FaultSpec) -> dict[str, object]:
    planner = CandidatePlanner()
    verifier = PlanVerifier(planner)
    task, fresh, proposal = spec.prepare(
        case.task.model_copy(deep=True),
        case.snapshot.model_copy(deep=True),
        case.proposal.model_copy(deep=True),
    )
    verification = verifier.verify(task, fresh, [proposal], _decision(proposal.plan_id))
    node = fresh.node(proposal.node_id)
    current = planner.evaluate(task, fresh, node) if node is not None else None
    unverified_invalid_attempt = (
        proposal.task_id != task.task_id
        or proposal.snapshot_version != fresh.version
        or not finite_candidate(proposal)
        or current is None
        or not current.feasible
    )
    fallback = select_baseline("joint_fixed", planner.generate(task, fresh))
    is_control = spec.name == "control"
    return {
        "fault": spec.name,
        "seed": case.seed,
        "scenario": case.scenario,
        "task_id": case.task.task_id,
        "plan_id": proposal.plan_id,
        "verifier_accepted": verification.accepted,
        "verification_reasons_json": json.dumps(verification.reasons),
        "specific_predicate_detected": is_control
        or any(spec.expected_reason.lower() in reason.lower() for reason in verification.reasons),
        "unverified_would_attempt_invalid": unverified_invalid_attempt,
        "fallback_available": fallback is not None,
        "safe_resolution": "fallback" if fallback is not None else "abstain",
        "expected_fallback_available": spec.expect_fallback,
    }


def _concurrency_trial(index: int) -> dict[str, object]:
    snapshot = ClusterSnapshot(
        snapshot_id=f"concurrency-{index}",
        nodes=[
            NodeState(
                node_id="n",
                cpu_capacity=4,
                cpu_available=4,
                ram_capacity_gib=8,
                ram_available_gib=8,
            )
        ],
        links=[
            LinkState(
                client_id="client-0",
                node_id="n",
                bandwidth_mbps=100,
                rtt_ms=1,
                reservable_fraction=0.8,
            )
        ],
    )
    planner = CandidatePlanner()
    tasks = [
        TaskIntent(task_id=f"concurrency-{index}-{suffix}", cpu_cores=3, ram_gib=1, min_bandwidth_mbps=60)
        for suffix in ("a", "b")
    ]
    plans = [planner.generate(task, snapshot)[0] for task in tasks]
    ledger = ReservationLedger(snapshot)
    start = threading.Barrier(3)
    attempted = threading.Barrier(3)
    release = threading.Event()

    def reserve(plan: CandidatePlan) -> tuple[bool, str]:
        start.wait()
        reservation = None
        try:
            reservation = ledger.reserve(plan)
            outcome = (True, "accepted")
        except Exception as exc:
            outcome = (False, str(exc))
        finally:
            attempted.wait()
            release.wait()
            if reservation is not None:
                ledger.release(reservation)
        return outcome

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(reserve, plan) for plan in plans]
        start.wait()
        attempted.wait()
        release.set()
        outcomes = [future.result() for future in futures]
    accepted = sum(success for success, _ in outcomes)
    rejected = len(outcomes) - accepted
    naive_cpu = sum(plan.allocated_cpu_cores for plan in plans)
    naive_bandwidth = sum(plan.reserved_bandwidth_mbps for plan in plans)
    return {
        "trial": index,
        "independent_precheck_accepts": 2,
        "naive_cpu_overcommit": naive_cpu > snapshot.nodes[0].cpu_available,
        "naive_bandwidth_overcommit": naive_bandwidth > snapshot.links[0].reservable_mbps,
        "atomic_ledger_accepts": accepted,
        "atomic_ledger_rejects": rejected,
        "atomic_overcommit": accepted > 1,
        "outcomes_json": json.dumps([reason for _, reason in outcomes]),
    }


def evaluate_causal_matrix(
    rows: list[dict[str, str]],
    *,
    sample_size: int = 100,
    concurrency_trials: int = 100,
    compiled_policy: str = "vertex_intent_compiled",
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    proposals = _proposal_cases(rows, compiled_policy)
    general = _balanced_sample(proposals, sample_size)
    gpu = _balanced_sample([case for case in proposals if case.task.gpu_units > 0], sample_size)
    trials: list[dict[str, object]] = []
    for spec in FAULTS:
        selected = gpu if spec.requires_gpu else general
        for case in selected:
            try:
                trials.append(_run_fault(case, spec))
            except ValueError:
                if spec.name != "trust_downgrade":
                    raise
    concurrency = [_concurrency_trial(index) for index in range(concurrency_trials)]

    surface_summaries: list[dict[str, object]] = []
    for spec in FAULTS:
        subset = [row for row in trials if row["fault"] == spec.name]
        surface_summaries.append(
            {
                "fault": spec.name,
                "trials": len(subset),
                "verifier_accepts": sum(bool(row["verifier_accepted"]) for row in subset),
                "verifier_rejections": sum(not bool(row["verifier_accepted"]) for row in subset),
                "specific_predicate_detections": sum(bool(row["specific_predicate_detected"]) for row in subset),
                "unverified_invalid_attempts": sum(bool(row["unverified_would_attempt_invalid"]) for row in subset),
                "fallbacks_available": sum(bool(row["fallback_available"]) for row in subset),
                "expected_fallback_available": spec.expect_fallback,
            }
        )

    control = next(item for item in surface_summaries if item["fault"] == "control")
    faults = [item for item in surface_summaries if item["fault"] != "control"]
    failures: list[str] = []
    if control["verifier_accepts"] != control["trials"]:
        failures.append("unchanged-state controls were rejected")
    for item in faults:
        if item["trials"] == 0:
            failures.append(f"{item['fault']}: no trials")
        if item["verifier_rejections"] != item["trials"]:
            failures.append(f"{item['fault']}: verifier accepted a fault")
        if item["specific_predicate_detections"] != item["trials"]:
            failures.append(f"{item['fault']}: expected predicate reason was absent")
        if item["unverified_invalid_attempts"] != item["trials"]:
            failures.append(f"{item['fault']}: verifier-off counterfactual was not invalid")
        if item["expected_fallback_available"] is True and item["fallbacks_available"] != item["trials"]:
            failures.append(f"{item['fault']}: expected fallback was absent")
        if item["expected_fallback_available"] is False and item["fallbacks_available"] != 0:
            failures.append(f"{item['fault']}: unexpected fallback was available")
    if any(row["atomic_ledger_accepts"] != 1 or row["atomic_ledger_rejects"] != 1 for row in concurrency):
        failures.append("atomic ledger did not admit exactly one contender")
    if any(bool(row["atomic_overcommit"]) for row in concurrency):
        failures.append("atomic ledger overcommitted")

    summary = {
        "schema_version": 1,
        "protocol": "verifier-on-off-predicate-matrix-v1",
        "analysis_status": "controlled_deterministic_fault_injection",
        "source_compiled_policy": compiled_policy,
        "general_proposals": len(general),
        "gpu_proposals": len(gpu),
        "predicate_trials": len(trials),
        "fault_trials": sum(int(item["trials"]) for item in faults),
        "control_trials": int(control["trials"]),
        "fault_verifier_rejections": sum(int(item["verifier_rejections"]) for item in faults),
        "fault_unverified_invalid_attempts": sum(int(item["unverified_invalid_attempts"]) for item in faults),
        "fault_specific_predicate_detections": sum(int(item["specific_predicate_detections"]) for item in faults),
        "fault_fallback_resolutions": sum(int(item["fallbacks_available"]) for item in faults),
        "fault_abstention_resolutions": sum(
            int(item["trials"]) - int(item["fallbacks_available"]) for item in faults
        ),
        "no_fallback_abstentions": next(
            int(item["trials"]) - int(item["fallbacks_available"])
            for item in faults
            if item["fault"] == "no_feasible_fallback"
        ),
        "concurrency_trials": len(concurrency),
        "naive_precheck_overcommit_exposures": sum(
            bool(row["naive_cpu_overcommit"] or row["naive_bandwidth_overcommit"]) for row in concurrency
        ),
        "atomic_ledger_overcommits": sum(bool(row["atomic_overcommit"]) for row in concurrency),
        "scope": (
            "The verifier-off arm is a safe instrumented counterfactual, not dispatch of unsafe work. "
            "Atomic admission is established only for one controller process; distributed commit, "
            "ambiguous remote completion, and Byzantine telemetry remain outside scope."
        ),
        "surfaces": surface_summaries,
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
    }
    return trials, concurrency, summary


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_causal_artifact(
    requests: Path,
    output: Path,
    *,
    sample_size: int = 100,
    concurrency_trials: int = 100,
    compiled_policy: str = "vertex_intent_compiled",
) -> dict[str, object]:
    rows = read_rows(requests)
    trials, concurrency, summary = evaluate_causal_matrix(
        rows,
        sample_size=sample_size,
        concurrency_trials=concurrency_trials,
        compiled_policy=compiled_policy,
    )
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(trials, output / "predicate_trials.csv")
    _write_csv(concurrency, output / "concurrency_trials.csv")
    summary = {
        **summary,
        "source_requests": str(requests),
        "source_requests_sha256": hashlib.sha256(requests.read_bytes()).hexdigest(),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    validation = {
        "schema_version": 1,
        "status": summary["status"],
        "predicate_rows": len(trials),
        "concurrency_rows": len(concurrency),
        "failures": summary["failures"],
    }
    (output / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    return validation


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run causal verifier-on/off predicate and concurrency matrix")
    parser.add_argument("--requests", default=Path("data/decisions.csv.gz"), type=Path)
    parser.add_argument("--output", default=Path("reproduced/verifier"), type=Path)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--concurrency-trials", type=int, default=100)
    parser.add_argument("--compiled-policy", default="vertex_intent_compiled")
    args = parser.parse_args(argv)
    validation = build_causal_artifact(
        args.requests,
        args.output,
        sample_size=args.sample_size,
        concurrency_trials=args.concurrency_trials,
        compiled_policy=args.compiled_policy,
    )
    print(json.dumps(validation, indent=2))
    if validation["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
