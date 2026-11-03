"""Scale replay: reuse recorded LLM intent profiles on larger synthetic clusters.

The primary live campaign recorded one LLM-inferred intent profile per decision
(seed, scenario, task) on a 12-node synthetic cluster. Because the task generator
is independent of the cluster size, the same 600 tasks can be placed on larger
clusters without any new model call: the recorded profile is replayed through
the deterministic planner ("compiled policy" semantics, see
``meshpilot.compiled_policy``) against a fresh, complete feasible set regenerated
from the execution snapshot of a scaled synthetic cluster.

For every cluster size N the module

1. runs the frozen primary configuration with ``nodes=N`` and the offline
   keyword backend (no API) to materialise the scaled snapshots and tasks,
2. asserts that every ``task_json`` is byte-identical to the primary artifact,
3. regenerates the complete feasible set from the execution snapshot
   S^{k'} (``snapshot_bundle_json["actual"]``) with ``top_k=N`` and computes the
   compiled selection, the declared-objective reference, normalised regret on
   the full set and on the frozen top-16 truncation, and a fresh balanced
   selector as comparison, and
4. times ``CandidatePlanner.generate`` (full node set) and
   ``PlanVerifier.verify`` (compiled selection) per decision.

Zero LLM calls are made; everything is deterministic given the frozen seeds
except the wall-clock overhead measurements.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import resource
import statistics
import sys
import time
import tracemalloc
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

from .artifact_io import read_rows
from .experiment import (
    ExperimentConfig,
    bootstrap_mean_ci,
    generate_snapshot,
    generate_tasks,
    percentile,
    perturb_snapshot,
    run_experiment,
    write_results,
)
from .models import PROFILE_WEIGHTS, AgentDecision, CandidatePlan, ClusterSnapshot, TaskIntent
from .planner import CandidatePlanner, PlannerConfig, candidate_objective, profile_weights
from .verifier import PlanVerifier


SCHEMA_VERSION = 1
DEFAULT_PRIMARY_CONFIG = Path("configs/primary.json")
PRIMARY_SOURCE_POLICY = "vertex_agent_verified"
DECISION_POLICY = "oracle_reactive"
REPLAY_BACKEND = "keyword"
FROZEN_TOP_K = 16
DEFAULT_NODES = (12, 50, 100, 200, 500, 1000)
FALLBACK_PROFILE = "balanced"

# Snapshot bundles for 1000-node clusters exceed the default csv field limit.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

DecisionKey = tuple[str, str, str]

ROW_FIELDS = (
    "nodes",
    "seed",
    "scenario",
    "task_id",
    "task_kind",
    "intent_phrase_id",
    "declared_profile",
    "recorded_profile",
    "recorded_fallback",
    "profile_match",
    "feasible_set_size_full",
    "feasible_set_size_top16",
    "truncated_by_top16",
    "compiled_node_full",
    "reference_node_full",
    "compiled_objective_full",
    "reference_objective_full",
    "compiled_regret_full",
    "reference_agreement_full",
    "compiled_slo_violation_full",
    "compiled_node_top16",
    "reference_node_top16",
    "compiled_regret_top16",
    "reference_agreement_top16",
    "fresh_balanced_node_full",
    "fresh_balanced_regret_full",
    "fresh_balanced_agreement_full",
    "sim_oracle_node",
    "sim_oracle_matches_reference_top16",
    "sim_top16_plan_ids_match",
    "verifier_accepted",
    "planner_ms",
    "verifier_ms",
)


@dataclass(slots=True)
class RecordedProfile:
    """One recorded LLM decision from the primary artifact."""

    profile: str
    used_fallback: bool
    task_json: str
    declared_profile: str


def decision_key(row: Mapping[str, object]) -> DecisionKey:
    return (str(row["seed"]), str(row["scenario"]), str(row["task_id"]))


def read_source_rows(primary: Path, source_policy: str = PRIMARY_SOURCE_POLICY) -> list[dict[str, str]]:
    rows = [row for row in read_rows(primary) if row.get("policy") == source_policy]
    if not rows:
        raise ValueError(f"no rows with policy {source_policy!r} in {primary}")
    return rows


def build_profile_map(source_rows: Iterable[Mapping[str, str]]) -> dict[DecisionKey, RecordedProfile]:
    """Map (seed, scenario, task_id) -> recorded profile; empty profile -> balanced with fallback flag."""

    recorded: dict[DecisionKey, RecordedProfile] = {}
    for row in source_rows:
        key = decision_key(row)
        if key in recorded:
            raise ValueError(f"duplicate decision key {key} in primary artifact")
        raw = (row.get("agent_inferred_profile") or "").strip()
        if raw and raw not in PROFILE_WEIGHTS:
            raise ValueError(f"unknown recorded profile {raw!r} for decision {key}")
        task_json = row.get("task_json") or ""
        if not task_json:
            raise ValueError(f"decision {key} has no task_json in the primary artifact")
        recorded[key] = RecordedProfile(
            profile=raw or FALLBACK_PROFILE,
            used_fallback=not raw,
            task_json=task_json,
            declared_profile=str(row.get("intent_profile") or ""),
        )
    return recorded


def load_recorded_profiles(
    primary: Path, source_policy: str = PRIMARY_SOURCE_POLICY
) -> dict[DecisionKey, RecordedProfile]:
    return build_profile_map(read_source_rows(primary, source_policy))


def build_scale_config(
    primary_config: Path,
    nodes: int,
    overrides: Mapping[str, object] | None = None,
) -> ExperimentConfig:
    """Frozen primary configuration with ``nodes=N`` and the offline keyword backend."""

    config = ExperimentConfig.model_validate_json(primary_config.read_text(encoding="utf-8"))
    update: dict[str, object] = {"nodes": nodes, "agent_backend": REPLAY_BACKEND, "model": None}
    if overrides:
        update.update(overrides)
    return config.model_copy(update=update)


def verify_task_json(
    rows: Iterable[Mapping[str, object]],
    recorded: Mapping[DecisionKey, RecordedProfile],
    evidence_policy: str,
) -> dict[str, object]:
    evidence = [row for row in rows if row["policy"] == evidence_policy]
    matched = 0
    missing: list[DecisionKey] = []
    mismatched: list[DecisionKey] = []
    for row in evidence:
        key = decision_key(row)
        source = recorded.get(key)
        if source is None:
            missing.append(key)
        elif str(row["task_json"]) == source.task_json:
            matched += 1
        else:
            mismatched.append(key)
    replayed = {decision_key(row) for row in evidence}
    return {
        "decisions": len(evidence),
        "matched": matched,
        "match_rate": matched / len(evidence) if evidence else 0.0,
        "missing_in_primary": [list(key) for key in missing[:10]],
        "mismatched": [list(key) for key in mismatched[:10]],
        "primary_decisions_not_replayed": len(set(recorded) - replayed),
    }


def snapshot_fingerprint(snapshot: ClusterSnapshot) -> str:
    return json.dumps(snapshot.model_dump(mode="json", exclude={"observed_at"}), separators=(",", ":"), sort_keys=True)


def task_fingerprint(task: TaskIntent) -> str:
    """Same serialisation experiment.run_experiment uses for ``task_json``."""

    return json.dumps(task.model_dump(mode="json"), separators=(",", ":"), sort_keys=True)


def execution_snapshots(
    rows: Iterable[Mapping[str, object]], evidence_policy: str
) -> dict[tuple[str, str], ClusterSnapshot]:
    """Execution snapshot S^{k'} (``bundle["actual"]``) per (seed, scenario) from the evidence rows."""

    snapshots: dict[tuple[str, str], ClusterSnapshot] = {}
    for row in rows:
        if row["policy"] != evidence_policy or not row.get("snapshot_bundle_json"):
            continue
        bundle = json.loads(str(row["snapshot_bundle_json"]))
        snapshots[(str(row["seed"]), str(row["scenario"]))] = ClusterSnapshot.model_validate(bundle["actual"])
    return snapshots


def _argmin(candidates: list[CandidatePlan], profile: str) -> CandidatePlan | None:
    weights = profile_weights(profile)
    return min(candidates, key=lambda candidate: candidate_objective(candidate, weights)) if candidates else None


def normalized_regret(selected: CandidatePlan | None, reference: CandidatePlan | None, profile: str) -> float | None:
    """(J_w(selected) - J_w(reference)) / J_w(reference) under the declared profile w."""

    if selected is None or reference is None:
        return None
    weights = profile_weights(profile)
    reference_objective = candidate_objective(reference, weights)
    if reference_objective <= 0:
        return None
    return (candidate_objective(selected, weights) - reference_objective) / reference_objective


def replay_decision(
    task: TaskIntent,
    snapshot: ClusterSnapshot,
    recorded: RecordedProfile,
    planner: CandidatePlanner,
    verifier: PlanVerifier,
    frozen_top_k: int = FROZEN_TOP_K,
) -> tuple[dict[str, object], list[CandidatePlan]]:
    """Replay one recorded decision on one execution snapshot; returns (metrics, full feasible set).

    ``planner`` must be configured with ``top_k >= len(snapshot.nodes)`` so the
    generated set is the complete feasible set. The generate call is timed and
    its output is the set used for selection.
    """

    started = time.perf_counter()
    full = planner.generate(task, snapshot)
    planner_ms = (time.perf_counter() - started) * 1e3
    top16 = full[:frozen_top_k]
    declared = task.evaluation_profile

    compiled_full = _argmin(full, recorded.profile)
    reference_full = _argmin(full, declared)
    compiled_top16 = _argmin(top16, recorded.profile)
    reference_top16 = _argmin(top16, declared)
    balanced_full = _argmin(full, FALLBACK_PROFILE)

    verifier_ms: float | None = None
    verifier_accepted: bool | None = None
    if compiled_full is not None:
        decision = AgentDecision(
            action="select",
            plan_id=compiled_full.plan_id,
            rationale="scale replay of the recorded intent profile",
            confidence=1.0,
            inferred_profile=recorded.profile,  # type: ignore[arg-type]
        )
        started = time.perf_counter()
        verification = verifier.verify(task, snapshot, full, decision)
        verifier_ms = (time.perf_counter() - started) * 1e3
        verifier_accepted = verification.accepted

    metrics: dict[str, object] = {
        "declared_profile": declared,
        "recorded_profile": recorded.profile,
        "recorded_fallback": recorded.used_fallback,
        "profile_match": recorded.profile == declared,
        "feasible_set_size_full": len(full),
        "feasible_set_size_top16": len(top16),
        "truncated_by_top16": len(full) > len(top16),
        "compiled_node_full": compiled_full.node_id if compiled_full else None,
        "reference_node_full": reference_full.node_id if reference_full else None,
        "compiled_objective_full": (
            candidate_objective(compiled_full, profile_weights(declared)) if compiled_full else None
        ),
        "reference_objective_full": (
            candidate_objective(reference_full, profile_weights(declared)) if reference_full else None
        ),
        "compiled_regret_full": normalized_regret(compiled_full, reference_full, declared),
        "reference_agreement_full": bool(
            compiled_full and reference_full and compiled_full.node_id == reference_full.node_id
        ),
        "compiled_slo_violation_full": bool(
            compiled_full and task.deadline_ms and compiled_full.predicted_completion_ms > task.deadline_ms
        ),
        "compiled_node_top16": compiled_top16.node_id if compiled_top16 else None,
        "reference_node_top16": reference_top16.node_id if reference_top16 else None,
        "compiled_regret_top16": normalized_regret(compiled_top16, reference_top16, declared),
        "reference_agreement_top16": bool(
            compiled_top16 and reference_top16 and compiled_top16.node_id == reference_top16.node_id
        ),
        "fresh_balanced_node_full": balanced_full.node_id if balanced_full else None,
        "fresh_balanced_regret_full": normalized_regret(balanced_full, reference_full, declared),
        "fresh_balanced_agreement_full": bool(
            balanced_full and reference_full and balanced_full.node_id == reference_full.node_id
        ),
        "verifier_accepted": verifier_accepted,
        "planner_ms": planner_ms,
        "verifier_ms": verifier_ms,
    }
    return metrics, full


def _timing_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p99": None, "max": None, "samples": 0}
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p99": percentile(values, 0.99),
        "max": max(values),
        "samples": len(values),
    }


def _peak_rss_mib() -> float:
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS and kibibytes on Linux.
    return maxrss / (1024.0 * 1024.0) if sys.platform == "darwin" else maxrss / 1024.0


def _mean_or_none(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def summarize_nodes(rows: list[dict[str, object]]) -> dict[str, object]:
    regret_full = [float(row["compiled_regret_full"]) for row in rows if row["compiled_regret_full"] is not None]
    regret_top16 = [float(row["compiled_regret_top16"]) for row in rows if row["compiled_regret_top16"] is not None]
    regret_balanced = [
        float(row["fresh_balanced_regret_full"]) for row in rows if row["fresh_balanced_regret_full"] is not None
    ]
    planner = [float(row["planner_ms"]) for row in rows if row["planner_ms"] is not None]
    verifier = [float(row["verifier_ms"]) for row in rows if row["verifier_ms"] is not None]
    sizes = [int(row["feasible_set_size_full"]) for row in rows]
    return {
        "decisions": len(rows),
        "mean_feasible_set_size": _mean_or_none([float(size) for size in sizes]),
        "median_feasible_set_size": statistics.median(sizes) if sizes else None,
        "min_feasible_set_size": min(sizes) if sizes else None,
        "max_feasible_set_size": max(sizes) if sizes else None,
        "empty_feasible_sets": sum(1 for size in sizes if size == 0),
        "truncated_decisions": sum(1 for row in rows if row["truncated_by_top16"]),
        "compiled_mean_regret_full": _mean_or_none(regret_full),
        "compiled_mean_regret_full_bootstrap_95ci": bootstrap_mean_ci(regret_full),
        "compiled_mean_regret_top16": _mean_or_none(regret_top16),
        "compiled_mean_regret_top16_bootstrap_95ci": bootstrap_mean_ci(regret_top16),
        "fresh_balanced_mean_regret_full": _mean_or_none(regret_balanced),
        "compiled_nonzero_regret_decisions_full": sum(1 for value in regret_full if value > 0),
        "compiled_max_regret_full": max(regret_full) if regret_full else None,
        "reference_agreement_full": _mean_or_none([1.0 if row["reference_agreement_full"] else 0.0 for row in rows]),
        "reference_agreement_top16": _mean_or_none(
            [1.0 if row["reference_agreement_top16"] else 0.0 for row in rows]
        ),
        "fresh_balanced_agreement_full": _mean_or_none(
            [1.0 if row["fresh_balanced_agreement_full"] else 0.0 for row in rows]
        ),
        "compiled_slo_violation_rate_full": _mean_or_none(
            [1.0 if row["compiled_slo_violation_full"] else 0.0 for row in rows]
        ),
        "profile_match_rate": _mean_or_none([1.0 if row["profile_match"] else 0.0 for row in rows]),
        "fallback_count": sum(1 for row in rows if row["recorded_fallback"]),
        "verifier_accept_rate": _mean_or_none(
            [1.0 if row["verifier_accepted"] else 0.0 for row in rows if row["verifier_accepted"] is not None]
        ),
        "sim_top16_match_rate": _mean_or_none([1.0 if row["sim_top16_plan_ids_match"] else 0.0 for row in rows]),
        "sim_oracle_matches_reference_top16_rate": _mean_or_none(
            [1.0 if row["sim_oracle_matches_reference_top16"] else 0.0 for row in rows]
        ),
        "planner_ms": _timing_summary(planner),
        "verifier_ms": _timing_summary(verifier),
    }


def replay_nodes(
    nodes: int,
    recorded: Mapping[DecisionKey, RecordedProfile],
    primary_config: Path,
    output: Path,
    *,
    config_overrides: Mapping[str, object] | None = None,
    decision_policy: str = DECISION_POLICY,
    frozen_top_k: int = FROZEN_TOP_K,
    log: bool = True,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Run the scaled simulation for one cluster size and replay every recorded decision on it."""

    config = build_scale_config(primary_config, nodes, config_overrides)
    evidence_policy = f"{config.agent_backend}_agent_verified"
    if log:
        print(f"[scale-replay] nodes={nodes}: running simulation ...", file=sys.stderr, flush=True)
    started = time.perf_counter()
    sim_rows, sim_summary = run_experiment(config)
    simulation_s = time.perf_counter() - started
    requests_path, summary_path = write_results(sim_rows, sim_summary, output / f"nodes-{nodes}")

    task_check = verify_task_json(sim_rows, recorded, evidence_policy)
    if task_check["match_rate"] != 1.0:
        raise RuntimeError(
            f"task_json mismatch at nodes={nodes}: {task_check['matched']}/{task_check['decisions']} matched; "
            f"missing in primary: {task_check['missing_in_primary']}; mismatched: {task_check['mismatched']}. "
            "Tasks must be independent of the node count; check seeds/tasks_per_seed/deadline_scale."
        )

    # The execution snapshot S^{k'} is regenerated deterministically in-process
    # (base -> perturbed) for every (seed, scenario). The simulation only stores
    # snapshot_bundle_json on the first task of each (seed, scenario); those
    # bundles are used as a cross-check of the regeneration, not as the source.
    snapshots: dict[tuple[str, str], ClusterSnapshot] = {}
    for seed in config.seeds:
        base = generate_snapshot(seed, nodes)
        for scenario in config.scenarios:
            snapshots[(str(seed), scenario)] = perturb_snapshot(base, scenario, seed)
    bundles = execution_snapshots(sim_rows, evidence_policy)
    bundle_matches = sum(
        1
        for key, bundled in bundles.items()
        if key in snapshots and snapshot_fingerprint(bundled) == snapshot_fingerprint(snapshots[key])
    )
    if bundles and bundle_matches != len(bundles):
        raise RuntimeError(
            f"nodes={nodes}: regenerated execution snapshots disagree with the simulation bundles "
            f"({bundle_matches}/{len(bundles)} match)"
        )

    # Tasks are regenerated from the frozen task generator (independent of N) and
    # must serialise byte-identically to the primary artifact's task_json.
    tasks: dict[tuple[str, str], TaskIntent] = {}
    for seed in config.seeds:
        for task in generate_tasks(seed, config.tasks_per_seed, config.deadline_scale):
            tasks[(str(seed), task.task_id)] = task

    evidence_by_key = {decision_key(row): row for row in sim_rows if row["policy"] == evidence_policy}
    decision_rows = [row for row in sim_rows if row["policy"] == decision_policy]
    if not decision_rows:
        raise RuntimeError(f"decision policy {decision_policy!r} absent from the nodes={nodes} simulation")

    planner = CandidatePlanner(PlannerConfig(top_k=nodes))
    verifier = PlanVerifier(planner)

    # Warm-up (and traced peak memory) on the first decision, outside the timing loop.
    first = decision_rows[0]
    first_task = tasks[(str(first["seed"]), str(first["task_id"]))]
    first_snapshot = snapshots[(str(first["seed"]), str(first["scenario"]))]
    replay_decision(first_task, first_snapshot, recorded[decision_key(first)], planner, verifier, frozen_top_k)
    tracemalloc.start()
    planner.generate(first_task, first_snapshot)
    _, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    if log:
        print(f"[scale-replay] nodes={nodes}: replaying {len(decision_rows)} decisions ...", file=sys.stderr, flush=True)
    rows: list[dict[str, object]] = []
    for decision in decision_rows:
        key = decision_key(decision)
        evidence = evidence_by_key[key]
        task = tasks[(key[0], key[2])]
        if task_fingerprint(task) != recorded[key].task_json:
            raise RuntimeError(f"nodes={nodes}: regenerated task {key} differs from the primary task_json")
        snapshot = snapshots[(key[0], key[1])]
        metrics, full = replay_decision(task, snapshot, recorded[key], planner, verifier, frozen_top_k)
        recorded_top16_ids = [candidate["plan_id"] for candidate in json.loads(str(evidence["actual_candidates_json"]))]
        fresh_top16_ids = [candidate.plan_id for candidate in full[:frozen_top_k]]
        row: dict[str, object] = {
            "nodes": nodes,
            "seed": key[0],
            "scenario": key[1],
            "task_id": key[2],
            "task_kind": evidence.get("task_kind"),
            "intent_phrase_id": evidence.get("intent_phrase_id"),
            **metrics,
            "sim_oracle_node": decision.get("selected_node_id"),
            "sim_oracle_matches_reference_top16": decision.get("selected_node_id") == metrics["reference_node_top16"],
            "sim_top16_plan_ids_match": recorded_top16_ids == fresh_top16_ids,
        }
        rows.append(row)

    summary: dict[str, object] = {
        "nodes": nodes,
        "task_json_match_rate": task_check["match_rate"],
        "task_json_check": task_check,
        "execution_snapshots": len(snapshots),
        "simulation_bundles": len(bundles),
        "bundle_regeneration_match_rate": bundle_matches / len(bundles) if bundles else None,
        **summarize_nodes(rows),
        "planner_peak_traced_kib": traced_peak / 1024.0,
        "process_peak_rss_mib": _peak_rss_mib(),
        "simulation": {
            "requests_csv": str(requests_path),
            "summary_json": str(summary_path),
            "rows": len(sim_rows),
            "elapsed_s": simulation_s,
            "config": config.model_dump(),
        },
    }
    return rows, summary


def primary_reference_stats(source_rows: list[dict[str, str]], frozen_top_k: int = FROZEN_TOP_K) -> dict[str, object]:
    """Cross-check the primary artifact itself: compiled regret on recorded candidates and full-set regeneration."""

    bundles: dict[tuple[str, str], ClusterSnapshot] = {}
    for row in source_rows:
        if row.get("snapshot_bundle_json"):
            bundle = json.loads(row["snapshot_bundle_json"])
            bundles[(row["seed"], row["scenario"])] = ClusterSnapshot.model_validate(bundle["actual"])
    regrets: list[float] = []
    full_regrets: list[float] = []
    regeneration_matches = 0
    full_equals_top16 = 0
    regenerated_decisions = 0
    sizes: list[int] = []
    for row in source_rows:
        task = TaskIntent.model_validate(json.loads(row["task_json"]))
        candidates = [CandidatePlan.model_validate(item) for item in json.loads(row["actual_candidates_json"])]
        profile = (row.get("agent_inferred_profile") or "").strip() or FALLBACK_PROFILE
        regret = normalized_regret(_argmin(candidates, profile), _argmin(candidates, task.evaluation_profile), task.evaluation_profile)
        if regret is not None:
            regrets.append(regret)
        sizes.append(len(candidates))
        snapshot = bundles.get((row["seed"], row["scenario"]))
        if snapshot is None:
            continue
        regenerated_decisions += 1
        planner = CandidatePlanner(PlannerConfig(top_k=len(snapshot.nodes)))
        full = planner.generate(task, snapshot)
        if [candidate.plan_id for candidate in full[:frozen_top_k]] == [candidate.plan_id for candidate in candidates]:
            regeneration_matches += 1
        if len(full) <= frozen_top_k and [candidate.plan_id for candidate in full] == [
            candidate.plan_id for candidate in candidates
        ]:
            full_equals_top16 += 1
        full_regret = normalized_regret(_argmin(full, profile), _argmin(full, task.evaluation_profile), task.evaluation_profile)
        if full_regret is not None:
            full_regrets.append(full_regret)
    return {
        "decisions": len(source_rows),
        "nodes": len(next(iter(bundles.values())).nodes) if bundles else None,
        "execution_snapshots": len(bundles),
        "fallback_count": sum(1 for row in source_rows if not (row.get("agent_inferred_profile") or "").strip()),
        "compiled_mean_regret_recorded_candidates": _mean_or_none(regrets),
        "compiled_mean_regret_full_regenerated": _mean_or_none(full_regrets),
        "mean_recorded_candidate_set_size": _mean_or_none([float(size) for size in sizes]),
        "max_recorded_candidate_set_size": max(sizes) if sizes else None,
        "regenerated_decisions": regenerated_decisions,
        "recorded_candidates_match_regenerated_top16_rate": (
            regeneration_matches / regenerated_decisions if regenerated_decisions else None
        ),
        "recorded_candidates_equal_full_feasible_set_rate": (
            full_equals_top16 / regenerated_decisions if regenerated_decisions else None
        ),
    }


def _fmt_pct(value: object) -> str:
    return "n/a" if value is None else f"{float(value) * 100:.4f}%"


def _fmt_rate(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def _fmt_timing(timing: Mapping[str, object]) -> str:
    if timing.get("mean") is None:
        return "n/a"
    return f"{float(timing['mean']):.2f} / {float(timing['median']):.2f} / {float(timing['p99']):.2f}"


def markdown_table(per_nodes: list[dict[str, object]]) -> str:
    header = (
        "| N | decisions | task_json match | mean \\|F\\| | truncated>16 | compiled regret (full) | "
        "compiled regret (top-16) | fresh balanced regret (full) | ref. agreement (full) | "
        "planner ms mean/med/p99 | verifier ms mean/med/p99 | fallbacks |"
    )
    separator = "|" + "---|" * 12
    lines = [header, separator]
    for item in per_nodes:
        lines.append(
            "| {nodes} | {decisions} | {match} | {size:.1f} | {truncated} | {full} | {top16} | {balanced} | "
            "{agreement} | {planner} | {verifier} | {fallbacks} |".format(
                nodes=item["nodes"],
                decisions=item["decisions"],
                match=f"{float(item['task_json_match_rate']) * 100:.1f}%",
                size=float(item["mean_feasible_set_size"] or 0.0),
                truncated=item["truncated_decisions"],
                full=_fmt_pct(item["compiled_mean_regret_full"]),
                top16=_fmt_pct(item["compiled_mean_regret_top16"]),
                balanced=_fmt_pct(item["fresh_balanced_mean_regret_full"]),
                agreement=_fmt_rate(item["reference_agreement_full"]),
                planner=_fmt_timing(item["planner_ms"]),
                verifier=_fmt_timing(item["verifier_ms"]),
                fallbacks=item["fallback_count"],
            )
        )
    return "\n".join(lines)


def write_rows(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ROW_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in ROW_FIELDS})


def run_scale_replay(
    primary: Path,
    output: Path,
    nodes: Iterable[int],
    *,
    primary_config: Path = DEFAULT_PRIMARY_CONFIG,
    source_policy: str = PRIMARY_SOURCE_POLICY,
    decision_policy: str = DECISION_POLICY,
    config_overrides: Mapping[str, object] | None = None,
    frozen_top_k: int = FROZEN_TOP_K,
    log: bool = True,
) -> dict[str, object]:
    started_at = datetime.now(timezone.utc).isoformat()
    node_sweep = sorted({int(value) for value in nodes})
    source_rows = read_source_rows(primary, source_policy)
    recorded = build_profile_map(source_rows)
    if log:
        print(
            f"[scale-replay] loaded {len(recorded)} recorded profiles from {primary} "
            f"({sum(1 for item in recorded.values() if item.used_fallback)} fallbacks)",
            file=sys.stderr,
            flush=True,
        )
    primary_stats = primary_reference_stats(source_rows, frozen_top_k)

    all_rows: list[dict[str, object]] = []
    per_nodes: list[dict[str, object]] = []
    for count in node_sweep:
        rows, summary = replay_nodes(
            count,
            recorded,
            primary_config,
            output,
            config_overrides=config_overrides,
            decision_policy=decision_policy,
            frozen_top_k=frozen_top_k,
            log=log,
        )
        all_rows.extend(rows)
        per_nodes.append(summary)
        if log:
            print(
                f"[scale-replay] nodes={count}: compiled regret full={_fmt_pct(summary['compiled_mean_regret_full'])} "
                f"top16={_fmt_pct(summary['compiled_mean_regret_top16'])} planner={_fmt_timing(summary['planner_ms'])} ms",
                file=sys.stderr,
                flush=True,
            )

    primary_regret = primary_stats["compiled_mean_regret_recorded_candidates"]
    baseline = next((item for item in per_nodes if item["nodes"] == primary_stats["nodes"]), None)
    reproduction = None
    if baseline is not None and primary_regret is not None and baseline["compiled_mean_regret_full"] is not None:
        same_decisions = int(baseline["decisions"]) == int(primary_stats["decisions"])
        difference = abs(float(baseline["compiled_mean_regret_full"]) - float(primary_regret))
        reproduction = {
            "nodes": baseline["nodes"],
            "same_decision_set": same_decisions,
            "primary_compiled_mean_regret": primary_regret,
            "replay_compiled_mean_regret_full": baseline["compiled_mean_regret_full"],
            "absolute_difference": difference,
            "reproduced_within_1e-9": same_decisions and difference < 1e-9,
        }

    summary = {
        "schema_version": SCHEMA_VERSION,
        "primary_requests": str(primary),
        "primary_config": str(primary_config),
        "source_policy": source_policy,
        "decision_policy": decision_policy,
        "replay_backend": REPLAY_BACKEND,
        "frozen_top_k": frozen_top_k,
        "node_sweep": node_sweep,
        "api_calls_added": 0,
        "recorded_decisions": len(recorded),
        "config_overrides": dict(config_overrides) if config_overrides else {},
        "primary_reference": primary_stats,
        "primary_reproduction": reproduction,
        "per_nodes": per_nodes,
        "interpretation": (
            "Recorded LLM-inferred intent profiles from the primary 12-node campaign are replayed with zero new model "
            "calls on larger synthetic clusters generated from the same frozen seeds. Tasks are identical across N "
            "(task_json byte-equal). Regret is measured against the declared-profile argmin over the complete fresh "
            "feasible set (full) and over the frozen top-16 balanced-order truncation the LLM would have seen (top-16). "
            "Overhead numbers are single-process wall-clock measurements of CandidatePlanner.generate over all N nodes "
            "and PlanVerifier.verify of the compiled selection."
        ),
        "run_metadata": {
            "started_at": started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    write_rows(all_rows, output / "rows.csv")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    table = markdown_table(per_nodes)
    (output / "summary.md").write_text(table + "\n", encoding="utf-8")
    summary["markdown_table"] = table
    return summary


def parse_nodes(value: str) -> list[int]:
    nodes = [int(item) for item in value.split(",") if item.strip()]
    if not nodes:
        raise argparse.ArgumentTypeError("--nodes needs at least one integer")
    return nodes


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Replay recorded LLM intent profiles on larger synthetic clusters")
    parser.add_argument("--primary", default=Path("data/decisions.csv.gz"), type=Path, help="frozen decision records (.csv or .csv.gz)")
    parser.add_argument("--output", default=Path("reproduced/scaling"), type=Path)
    parser.add_argument("--nodes", default=",".join(str(value) for value in DEFAULT_NODES), type=parse_nodes)
    parser.add_argument("--primary-config", default=DEFAULT_PRIMARY_CONFIG, type=Path)
    parser.add_argument("--source-policy", default=PRIMARY_SOURCE_POLICY)
    parser.add_argument("--decision-policy", default=DECISION_POLICY)
    args = parser.parse_args(argv)
    summary = run_scale_replay(
        args.primary,
        args.output,
        args.nodes,
        primary_config=args.primary_config,
        source_policy=args.source_policy,
        decision_policy=args.decision_policy,
    )
    print(summary["markdown_table"])
    print(
        json.dumps(
            {
                "rows": str(args.output / "rows.csv"),
                "summary": str(args.output / "summary.json"),
                "primary_reproduction": summary["primary_reproduction"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
