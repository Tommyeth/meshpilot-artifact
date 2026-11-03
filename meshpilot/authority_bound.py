"""Empirical regret bound for profile-level versus plan-ID-level LLM authority.

For every recorded decision the fresh feasible candidate set, the author
reference profile, and the planner objective are frozen in the artifact, so
the worst regret that ANY profile error can cause is computable exactly:

    B_i = max_{w' != w_i} [J_{w_i}(argmin_p J_{w'}(p)) - J_{w_i}(p_i^*)] / J_{w_i}(p_i^*)

A compiled policy that only lets the model choose a profile therefore has
per-decision regret <= B_i no matter what the model says. A policy that lets
the model choose a plan ID can reach

    M_i = max_{p in P_i} [J_{w_i}(p) - J_{w_i}(p_i^*)] / J_{w_i}(p_i^*)

on a valid ID and has no objective value at all on an invalid ID. The script
reports both bounds and the regrets actually observed in the frozen campaign.
No model call is made.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Iterable

from meshpilot.models import CandidatePlan, TaskIntent
from meshpilot.paired_analysis import _is_injected_unknown_id, _is_model_unknown_id, _is_valid_direct_action
from meshpilot.planner import PROFILE_WEIGHTS, candidate_objective, profile_weights
from meshpilot.artifact_io import read_rows

PROFILES = tuple(PROFILE_WEIGHTS)


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _regret(candidate: CandidatePlan, reference: CandidatePlan, weights) -> float:
    base = candidate_objective(reference, weights)
    return (candidate_objective(candidate, weights) - base) / base


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": _percentile(values, 0.95),
        "max": max(values),
        "count": len(values),
    }


def compute_bounds(
    rows: list[dict[str, str]],
    *,
    evidence_policy: str = "vertex_agent_verified",
    compiled_policy: str = "vertex_intent_compiled",
) -> tuple[list[dict[str, object]], dict[str, object]]:
    by_key: dict[tuple[str, str, str], dict[str, dict[str, str]]] = {}
    for row in rows:
        key = (row["seed"], row["scenario"], row["task_id"])
        by_key.setdefault(key, {})[row["policy"]] = row

    per_decision: list[dict[str, object]] = []
    for key, policies in sorted(by_key.items()):
        source = policies.get(evidence_policy)
        if source is None:
            continue
        task = TaskIntent.model_validate(json.loads(source["task_json"]))
        candidates = [CandidatePlan.model_validate(c) for c in json.loads(source["actual_candidates_json"])]
        if not candidates:
            continue
        ref_profile = task.evaluation_profile
        ref_weights = profile_weights(ref_profile)
        reference = min(candidates, key=lambda c: candidate_objective(c, ref_weights))
        if candidate_objective(reference, ref_weights) <= 0:
            continue

        profile_regret: dict[str, float] = {}
        for profile in PROFILES:
            chosen = min(candidates, key=lambda c: candidate_objective(c, profile_weights(profile)))
            profile_regret[profile] = _regret(chosen, reference, ref_weights)
        wrong = {p: r for p, r in profile_regret.items() if p != ref_profile}
        profile_bound = max(wrong.values())
        worst_profile = max(wrong, key=wrong.get)
        id_bound = max(_regret(c, reference, ref_weights) for c in candidates)

        # Known-ID direct regret follows the non-injected pairing: the
        # guarded-direct row executed the model's own valid plan, so its regret
        # is the direct-ID regret. Injected unknown IDs, model-emitted unknown
        # IDs, abstentions, and API errors are excluded (fallback regret is not
        # a direct-ID regret).
        injected = _is_injected_unknown_id(source)
        model_unknown = _is_model_unknown_id(source)
        direct_valid = (not injected) and (not model_unknown) and _is_valid_direct_action(source)
        direct_regret = (
            float(source["normalized_regret_to_oracle"])
            if direct_valid and source.get("normalized_regret_to_oracle") not in (None, "")
            else None
        )
        compiled = policies.get(compiled_policy)
        compiled_regret = (
            float(compiled["normalized_regret_to_oracle"])
            if compiled and compiled.get("normalized_regret_to_oracle") not in (None, "")
            else None
        )
        inferred = source.get("agent_inferred_profile") or ""

        per_decision.append(
            {
                "seed": key[0],
                "scenario": key[1],
                "task_id": key[2],
                "intent_phrase_id": source.get("intent_phrase_id"),
                "reference_profile": ref_profile,
                "inferred_profile": inferred,
                "profile_match": inferred == ref_profile,
                "feasible_candidates": len(candidates),
                "profile_bound_regret": profile_bound,
                "worst_wrong_profile": worst_profile,
                **{f"regret_if_{p}": profile_regret[p] for p in PROFILES},
                "id_bound_regret": id_bound,
                "direct_id_valid": direct_valid,
                "direct_id_injected": injected,
                "direct_id_model_unknown": model_unknown,
                "direct_id_regret": direct_regret,
                "compiled_regret": compiled_regret,
                "direct_exceeds_profile_bound": (
                    direct_regret is not None and direct_regret > profile_bound + 1e-12
                ),
            }
        )

    profile_bounds = [float(r["profile_bound_regret"]) for r in per_decision]
    id_bounds = [float(r["id_bound_regret"]) for r in per_decision]
    direct_rows = [r for r in per_decision if r["direct_id_regret"] is not None]
    direct_regrets = [float(r["direct_id_regret"]) for r in direct_rows]
    compiled_regrets = [float(r["compiled_regret"]) for r in per_decision if r["compiled_regret"] is not None]
    # A missing inferred profile means abstention/API error (fallback to
    # balanced), which is a fallback, not a profile error.
    errors = [r for r in per_decision if r["inferred_profile"] and not r["profile_match"]]
    fallback_rows = [r for r in per_decision if not r["inferred_profile"]]

    summary = {
        "schema_version": 1,
        "decisions": len(per_decision),
        "profiles": list(PROFILES),
        "profile_bound_regret": _stats(profile_bounds),
        "id_bound_regret": _stats(id_bounds),
        "decisions_with_zero_profile_bound": sum(1 for p in profile_bounds if p <= 0),
        "id_bound_over_profile_bound_ratio_median": statistics.median(
            [i / p for i, p in zip(id_bounds, profile_bounds) if p > 0]
        ),
        "decisions_where_id_bound_exceeds_profile_bound": sum(
            1 for i, p in zip(id_bounds, profile_bounds) if i > p + 1e-12
        ),
        "observed_compiled_regret": _stats(compiled_regrets) if compiled_regrets else None,
        "observed_compiled_regret_violates_profile_bound": sum(
            1
            for r in per_decision
            if r["compiled_regret"] is not None
            and float(r["compiled_regret"]) > float(r["profile_bound_regret"]) + 1e-9
        ),
        "observed_direct_id_regret_known_id": _stats(direct_regrets) if direct_regrets else None,
        "direct_known_id_decisions": len(direct_rows),
        "direct_invalid_id_decisions": len(per_decision) - len(direct_rows),
        "direct_injected_id_decisions": sum(1 for r in per_decision if r["direct_id_injected"]),
        "direct_model_unknown_id_decisions": sum(1 for r in per_decision if r["direct_id_model_unknown"]),
        "direct_abstain_or_error_decisions": sum(
            1
            for r in per_decision
            if not r["direct_id_injected"] and not r["direct_id_model_unknown"] and not r["direct_id_valid"]
        ),
        "direct_known_id_exceeding_profile_bound": sum(
            1 for r in direct_rows if r["direct_exceeds_profile_bound"]
        ),
        "direct_known_id_exceeding_profile_bound_rate": (
            statistics.fmean(1.0 if r["direct_exceeds_profile_bound"] else 0.0 for r in direct_rows)
            if direct_rows
            else None
        ),
        "profile_errors": len(errors),
        "profile_fallback_rows": len(fallback_rows),
        "profile_error_conditional_regret_mean": (
            statistics.fmean(float(r["compiled_regret"]) for r in errors if r["compiled_regret"] is not None)
            if errors
            else 0.0
        ),
        "worst_wrong_profile_histogram": {
            p: sum(1 for r in per_decision if r["worst_wrong_profile"] == p) for p in PROFILES
        },
        "interpretation": (
            "profile_bound_regret is the largest regret any profile error can cause on each "
            "frozen decision; a profile-only authority contract can never exceed it. "
            "id_bound_regret is the largest regret a valid but poorly chosen plan ID can cause; "
            "an invalid ID has no objective value and is outside both bounds. Observed regrets "
            "come from the frozen primary campaign; no model call was made."
        ),
    }
    return per_decision, summary


def write_artifact(
    requests: Path,
    output: Path,
    *,
    evidence_policy: str = "vertex_agent_verified",
    compiled_policy: str = "vertex_intent_compiled",
) -> dict[str, object]:
    rows = read_rows(requests)
    per_decision, summary = compute_bounds(rows, evidence_policy=evidence_policy, compiled_policy=compiled_policy)
    if not per_decision:
        raise ValueError(f"no decisions found for evidence policy {evidence_policy!r}")
    output.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in per_decision for k in r))
    with (output / "per_decision_bounds.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_decision)
    summary["source_requests"] = str(requests)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Empirical regret bounds for profile vs plan-ID authority")
    parser.add_argument("--requests", default=Path("data/decisions.csv.gz"), type=Path)
    parser.add_argument("--output", default=Path("reproduced/authority-bound"), type=Path)
    parser.add_argument("--evidence-policy", default="vertex_agent_verified")
    parser.add_argument("--compiled-policy", default="vertex_intent_compiled")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            write_artifact(
                args.requests,
                args.output,
                evidence_policy=args.evidence_policy,
                compiled_policy=args.compiled_policy,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
