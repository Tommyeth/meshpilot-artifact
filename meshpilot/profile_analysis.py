"""Post-hoc same-state fixed-balanced control and failure audit for the poster.

No model or hardware call is made. Selection reuses the original validated model
classes and planner objective on each recorded ``actual_candidates_json`` list;
it does not regenerate telemetry or use the precomputed ``joint_fixed`` rows.
Candidate order is retained, including the existing first-minimum tie rule.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .models import CandidatePlan, PROFILE_WEIGHTS, TaskIntent
from .paired_analysis import _cluster_percentile_interval, analyze_non_injected_pairs
from .planner import candidate_objective, profile_weights
from .artifact_io import PAIR_KEY, read_rows

TOLERANCE = 1e-12
DIRECT_POLICY = "vertex_agent_verified"
COMPILED_POLICY = "vertex_intent_compiled"


def _sign(value: float) -> str:
    return "win" if value < -TOLERANCE else "loss" if value > TOLERANCE else "tie"


def _compare(rows: list[dict[str, object]], target: str, baseline: str) -> dict[str, object]:
    by_seed: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_seed[str(row["seed"])].append(float(row[f"{target}_regret"]) - float(row[f"{baseline}_regret"]))
    differences = [value for values in by_seed.values() for value in values]
    low, high = _cluster_percentile_interval(by_seed, seed=20260829, draws=5000)
    return {
        "pairs": len(rows),
        "mean_target_regret": statistics.fmean(float(row[f"{target}_regret"]) for row in rows),
        "mean_baseline_regret": statistics.fmean(float(row[f"{baseline}_regret"]) for row in rows),
        "mean_target_minus_baseline": statistics.fmean(differences),
        "target_minus_baseline_cluster_ci95": [low, high],
        "target_wins": sum(_sign(value) == "win" for value in differences),
        "ties": sum(_sign(value) == "tie" for value in differences),
        "target_losses": sum(_sign(value) == "loss" for value in differences),
    }


def analyze_profiles(rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    paired, paired_summary = analyze_non_injected_pairs(rows)
    known_keys = {tuple(str(row[field]) for field in PAIR_KEY) for row in paired}
    indexed: dict[tuple[str, str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        if row.get("policy") in {DIRECT_POLICY, COMPILED_POLICY}:
            key = tuple(row[field] for field in PAIR_KEY)
            if row["policy"] in indexed[key]:
                raise ValueError(f"duplicate {row['policy']} row for {key}")
            indexed[key][row["policy"]] = row

    audited: list[dict[str, object]] = []
    max_recorded_regret_error = 0.0
    for key, policies in sorted(indexed.items()):
        direct, compiled = policies[DIRECT_POLICY], policies[COMPILED_POLICY]
        task = TaskIntent.model_validate(json.loads(direct["task_json"]))
        candidate_json = direct["actual_candidates_json"]
        if json.loads(candidate_json) != json.loads(compiled["actual_candidates_json"]):
            raise ValueError(f"direct and compiled candidate sets differ for {key}")
        candidates = [CandidatePlan.model_validate(item) for item in json.loads(candidate_json)]
        if not candidates or not all(candidate.feasible for candidate in candidates):
            raise ValueError(f"expected a nonempty feasible candidate set for {key}")
        by_plan = {candidate.plan_id: candidate for candidate in candidates}
        if len(by_plan) != len(candidates):
            raise ValueError(f"duplicate candidate IDs for {key}")
        declared = task.evaluation_profile
        inferred = direct.get("agent_inferred_profile") or "balanced"
        if declared not in PROFILE_WEIGHTS or inferred not in PROFILE_WEIGHTS:
            raise ValueError(f"unknown profile for {key}")
        choices = {
            profile: min(candidates, key=lambda candidate: candidate_objective(candidate, profile_weights(profile)))
            for profile in PROFILE_WEIGHTS
        }
        oracle = choices[declared]
        reference = candidate_objective(oracle, profile_weights(declared))
        if reference <= 0:
            raise ValueError(f"nonpositive declared-profile oracle for {key}")

        def regret(candidate: CandidatePlan) -> float:
            return (candidate_objective(candidate, profile_weights(declared)) - reference) / reference

        compiled_choice, balanced_choice = choices[inferred], choices["balanced"]
        if compiled_choice.plan_id != compiled["compiled_plan_id"]:
            raise ValueError(f"compiled selection does not reproduce for {key}")
        if compiled_choice.node_id != compiled["selected_node_id"]:
            raise ValueError(f"compiled node does not reproduce for {key}")
        compiled_regret = regret(compiled_choice)
        error = abs(compiled_regret - float(compiled["normalized_regret_to_oracle"]))
        max_recorded_regret_error = max(max_recorded_regret_error, error)
        if error > TOLERANCE:
            raise ValueError(f"compiled regret does not reproduce for {key}: {error}")
        is_known = key in known_keys
        direct_choice = by_plan.get(direct["agent_plan_id"]) if is_known else None
        if is_known and (direct_choice is None or direct_choice.node_id != direct["selected_node_id"]):
            raise ValueError(f"known direct ID does not reproduce for {key}")
        direct_regret = regret(direct_choice) if direct_choice else None
        if direct_regret is not None:
            error = abs(direct_regret - float(direct["normalized_regret_to_oracle"]))
            max_recorded_regret_error = max(max_recorded_regret_error, error)
            if error > TOLERANCE:
                raise ValueError(f"direct regret does not reproduce for {key}: {error}")
        profile_regrets = {profile: regret(choice) for profile, choice in choices.items()}
        profile_nodes = {profile: choice.node_id for profile, choice in choices.items()}
        row: dict[str, object] = {
            **dict(zip(PAIR_KEY, key)),
            "known_id_pair": is_known,
            "intent_phrase_id": direct["intent_phrase_id"],
            "operator_intent": task.operator_intent,
            "declared_profile": declared,
            "inferred_profile": direct.get("agent_inferred_profile") or "",
            "effective_compiled_profile": inferred,
            "profile_error": bool(direct.get("agent_inferred_profile")) and inferred != declared,
            "feasible_candidates": len(candidates),
            "actual_candidates_sha256": hashlib.sha256(candidate_json.encode()).hexdigest(),
            "candidate_source": f"{DIRECT_POLICY}.actual_candidates_json",
            "selection_rule": "existing candidate_objective/profile_weights; first minimum in frozen candidate order",
            "direct_provenance": "recorded agent_plan_id, confirmed present in fresh feasible candidates" if is_known else "excluded: direct model action not a valid known ID; execution may be fallback",
            "compiled_provenance": "deterministic argmin for recorded inferred profile on same fresh candidates; checked against recorded compiled_plan_id",
            "balanced_provenance": "post-hoc deterministic argmin for fixed balanced profile on same fresh candidates; no model call",
            "oracle_node_id": oracle.node_id,
            "oracle_plan_id": oracle.plan_id,
            "direct_model_plan_id": direct["agent_plan_id"],
            "recorded_executed_node_id": direct["selected_node_id"],
            "direct_node_id": direct_choice.node_id if direct_choice else None,
            "direct_plan_id": direct_choice.plan_id if direct_choice else None,
            "direct_regret": direct_regret,
            "compiled_node_id": compiled_choice.node_id,
            "compiled_plan_id": compiled_choice.plan_id,
            "compiled_regret": compiled_regret,
            "balanced_node_id": balanced_choice.node_id,
            "balanced_plan_id": balanced_choice.plan_id,
            "balanced_regret": regret(balanced_choice),
            "compiled_vs_direct": _sign(compiled_regret - direct_regret) if direct_regret is not None else "excluded",
            "compiled_vs_balanced": _sign(compiled_regret - regret(balanced_choice)),
            "all_profile_choices_same_node": len(set(profile_nodes.values())) == 1,
            "all_profiles_choose_oracle": all(value <= TOLERANCE for value in profile_regrets.values()),
            "profile_choice_nodes_json": json.dumps(profile_nodes, sort_keys=True),
            "profile_choice_regrets_json": json.dumps(profile_regrets, sort_keys=True),
        }
        audited.append(row)

    known = [row for row in audited if row["known_id_pair"]]
    errors = [row for row in audited if row["profile_error"]]
    ties = [row for row in known if row["compiled_vs_direct"] == "tie"]
    losses = [row for row in known if row["compiled_vs_direct"] == "loss"]
    # Select the first key among profile errors that worsen direct regret.
    example = next((row for row in losses if row["profile_error"]), None)
    summary: dict[str, object] = {
        "schema_version": 1,
        "analysis_status": "exploratory_post_hoc",
        "source_decisions": len(audited),
        "known_id_pairs": len(known),
        "excluded_pairs": len(audited) - len(known),
        "candidate_source": "frozen direct-row actual_candidates_json, identical to compiled-row candidates",
        "algorithm_reuse": "meshpilot.models.CandidatePlan/TaskIntent and meshpilot.planner.candidate_objective/profile_weights",
        "tie_breaking": "first minimum in recorded candidate-list order, as in compiled_policy.py",
        "baseline": "fresh fixed-balanced; precomputed joint_fixed rows are not used",
        "api_calls_added": 0,
        "estimand": "pooled decision-level mean normalized regret under each request's declared profile",
        "bootstrap_statistic": paired_summary["bootstrap_statistic"],
        "bootstrap_seed": 20260829,
        "bootstrap_draws": 5000,
        "bootstrap_cluster_order": paired_summary["bootstrap_cluster_order"],
        "bootstrap_percentile_rule": paired_summary["bootstrap_percentile_rule"],
        "comparison_tolerance": TOLERANCE,
        "mean_direct_regret": statistics.fmean(float(row["direct_regret"]) for row in known),
        "mean_compiled_regret": statistics.fmean(float(row["compiled_regret"]) for row in known),
        "mean_balanced_regret": statistics.fmean(float(row["balanced_regret"]) for row in known),
        "mean_fresh_balanced_regret": statistics.fmean(float(row["balanced_regret"]) for row in known),
        "compiled_vs_direct": _compare(known, "compiled", "direct"),
        "compiled_vs_balanced": _compare(known, "compiled", "balanced"),
        "balanced_vs_direct": _compare(known, "balanced", "direct"),
        "per_declared_profile": {
            profile: {
                "compiled_vs_balanced": _compare([row for row in known if row["declared_profile"] == profile], "compiled", "balanced"),
                "compiled_vs_direct": _compare([row for row in known if row["declared_profile"] == profile], "compiled", "direct"),
            }
            for profile in PROFILE_WEIGHTS
            if any(row["declared_profile"] == profile for row in known)
        },
        "profile_errors": {
            "all_decisions": len(errors),
            "known_id_pairs": sum(bool(row["known_id_pair"]) for row in errors),
            "excluded_pairs": sum(not row["known_id_pair"] for row in errors),
            "compiled_zero_regret": sum(float(row["compiled_regret"]) <= TOLERANCE for row in errors),
            "compiled_positive_regret": sum(float(row["compiled_regret"]) > TOLERANCE for row in errors),
            "known_compiled_wins": sum(row["compiled_vs_direct"] == "win" for row in errors),
            "known_ties": sum(row["compiled_vs_direct"] == "tie" for row in errors),
            "known_compiled_losses": sum(row["compiled_vs_direct"] == "loss" for row in errors),
            "mean_compiled_regret_all_errors": statistics.fmean(float(row["compiled_regret"]) for row in errors) if errors else 0.0,
        },
        "compiled_losses_vs_direct": {
            "count": len(losses),
            "with_profile_error": sum(bool(row["profile_error"]) for row in losses),
            "direct_zero_regret": sum(float(row["direct_regret"]) <= TOLERANCE for row in losses),
        },
        "compiled_direct_ties": {
            "count": len(ties),
            "same_node": sum(row["direct_node_id"] == row["compiled_node_id"] for row in ties),
            "both_zero_regret": sum(float(row["compiled_regret"]) <= TOLERANCE and float(row["direct_regret"]) <= TOLERANCE for row in ties),
            "both_positive_regret": sum(float(row["compiled_regret"]) > TOLERANCE and float(row["direct_regret"]) > TOLERANCE for row in ties),
            "with_profile_error": sum(bool(row["profile_error"]) for row in ties),
            "all_profile_choices_same_node": sum(bool(row["all_profile_choices_same_node"]) for row in ties),
            "all_profiles_choose_oracle": sum(bool(row["all_profiles_choose_oracle"]) for row in ties),
            "fixed_balanced_same_node": sum(row["balanced_node_id"] == row["compiled_node_id"] for row in ties),
            "fixed_balanced_positive_regret": sum(float(row["balanced_regret"]) > TOLERANCE for row in ties),
        },
        "max_recorded_regret_reconstruction_error": max_recorded_regret_error,
        "example_selection_rule": "first lexicographic (seed, scenario, task_id) known-ID compiled loss with a profile error",
        "failure_example": example,
        "interpretation": "Post-hoc replay of frozen synthetic candidate metrics under the author-declared objective, not new LLM calls or evidence of realized operator utility. The paired seed-cluster interval uses only five clusters and remains descriptive.",
    }
    return audited, summary


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else list(PAIR_KEY))
        writer.writeheader()
        writer.writerows(rows)


def build_profile_artifact(requests: Path, output: Path) -> dict[str, object]:
    audited, summary = analyze_profiles(read_rows(requests))
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(audited, output / "all_decisions.csv")
    _write_csv([row for row in audited if row["known_id_pair"]], output / "paired_rows.csv")
    _write_csv([row for row in audited if row["profile_error"]], output / "profile_errors.csv")
    _write_csv([row for row in audited if row["compiled_vs_direct"] == "loss"], output / "compiled_losses.csv")
    (output / "failure_examples.json").write_text(
        json.dumps({
            "selection_rule": summary["example_selection_rule"],
            "recommended_example": summary["failure_example"],
            "all_known_id_compiled_losses": [row for row in audited if row["compiled_vs_direct"] == "loss"],
            "all_profile_errors": [row for row in audited if row["profile_error"]],
        }, indent=2) + "\n", encoding="utf-8"
    )
    summary.update(source_requests=str(requests), source_requests_sha256=hashlib.sha256(requests.read_bytes()).hexdigest())
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    validation = {
        "status": "PASS",
        "all_decisions": len(audited),
        "known_id_pairs": summary["known_id_pairs"],
        "same_fresh_candidates_checked": len(audited),
        "compiled_selections_reproduced": len(audited),
        "known_direct_selections_reproduced": summary["known_id_pairs"],
        "max_recorded_regret_reconstruction_error": summary["max_recorded_regret_reconstruction_error"],
        "api_calls_added": 0,
    }
    (output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", default=Path("data/decisions.csv.gz"), type=Path)
    parser.add_argument("--output", default=Path("reproduced/profiles"), type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(build_profile_artifact(args.requests, args.output), indent=2))


if __name__ == "__main__":
    main()
