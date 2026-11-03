from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Iterable

from meshpilot.models import CandidatePlan, TaskIntent
from meshpilot.planner import candidate_objective, profile_weights
from meshpilot.artifact_io import read_rows


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def compile_intent_policy(
    rows: list[dict[str, str]],
    *,
    source_policy: str,
    policy_name: str,
) -> list[dict[str, object]]:
    source_rows = [row for row in rows if row["policy"] == source_policy]
    if not source_rows:
        raise ValueError(f"source policy {source_policy!r} not found")

    compiled: list[dict[str, object]] = []
    for source in source_rows:
        task = TaskIntent.model_validate(json.loads(source["task_json"]))
        candidates = [
            CandidatePlan.model_validate(candidate)
            for candidate in json.loads(source["actual_candidates_json"])
        ]
        inferred_profile = source.get("agent_inferred_profile") or "balanced"
        used_fallback = not bool(source.get("agent_inferred_profile"))
        selected = (
            min(candidates, key=lambda candidate: candidate_objective(candidate, profile_weights(inferred_profile)))
            if candidates
            else None
        )
        oracle = (
            min(
                candidates,
                key=lambda candidate: candidate_objective(candidate, profile_weights(task.evaluation_profile)),
            )
            if candidates
            else None
        )
        output: dict[str, object] = dict(source)
        output.update(
            {
                "policy": policy_name,
                "selected_node_id": selected.node_id if selected else None,
                "success": selected is not None,
                "safety_violation": False,
                "slo_violation": bool(
                    selected and task.deadline_ms and selected.predicted_completion_ms > task.deadline_ms
                ),
                "actual_completion_ms": selected.predicted_completion_ms if selected else None,
                "actual_cost_usd": selected.predicted_cost_usd if selected else None,
                "utility_score": (
                    candidate_objective(selected, profile_weights(task.evaluation_profile)) if selected else None
                ),
                "recovered": bool(selected and _truthy(source["recovery_required"])),
                "used_fallback": used_fallback,
                "reason": "deterministic candidate selection from the Agent-inferred intent profile",
                "agent_action_valid": selected is not None,
                "verification_reasons_json": "[]" if selected else '["no feasible candidate"]',
                "verification_selected_plan_id": selected.plan_id if selected else None,
                "oracle_node_agreement": bool(selected and oracle and selected.node_id == oracle.node_id),
                "normalized_regret_to_oracle": (
                    (
                        candidate_objective(selected, profile_weights(task.evaluation_profile))
                        - candidate_objective(oracle, profile_weights(task.evaluation_profile))
                    )
                    / candidate_objective(oracle, profile_weights(task.evaluation_profile))
                    if selected and oracle and candidate_objective(oracle, profile_weights(task.evaluation_profile)) > 0
                    else None
                ),
                "compiled_profile": inferred_profile,
                "compiled_plan_id": selected.plan_id if selected else None,
            }
        )
        compiled.append(output)
    return compiled


def _write_union_csv(rows: list[dict[str, object]], path: Path) -> None:
    fields = list(dict.fromkeys(field for row in rows for field in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def build_compiled_artifact(
    requests: Path,
    output: Path,
    *,
    source_policy: str,
    policy_name: str,
) -> dict[str, object]:
    rows = read_rows(requests)
    compiled = compile_intent_policy(rows, source_policy=source_policy, policy_name=policy_name)
    output.mkdir(parents=True, exist_ok=True)
    _write_union_csv(compiled, output / "compiled_policy.csv")
    _write_union_csv([*rows, *compiled], output / "augmented_requests.csv")
    regrets = [float(row["normalized_regret_to_oracle"]) for row in compiled]
    report = {
        "schema_version": 1,
        "source_requests": str(requests),
        "source_policy": source_policy,
        "compiled_policy": policy_name,
        "compiled_rows": len(compiled),
        "api_calls_added": 0,
        "success_rate": statistics.fmean(1.0 if row["success"] else 0.0 for row in compiled),
        "safety_violation_rate": statistics.fmean(
            1.0 if row["safety_violation"] else 0.0 for row in compiled
        ),
        "mean_normalized_regret_to_oracle": statistics.fmean(regrets),
        "missing_profile_fallbacks": sum(bool(row["used_fallback"]) for row in compiled),
        "interpretation": (
            "This ablation reuses recorded LLM-inferred profiles and fresh feasible candidate sets. "
            "The deterministic planner, not the LLM, performs numerical selection."
        ),
    }
    (output / "compiled_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compile recorded LLM intent into deterministic plan selection")
    parser.add_argument("--requests", default=Path("data/decisions.csv.gz"), type=Path)
    parser.add_argument("--output", default=Path("reproduced/compiled"), type=Path)
    parser.add_argument("--source-policy", default="vertex_agent_verified")
    parser.add_argument("--policy-name", default="vertex_intent_compiled")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            build_compiled_artifact(
                args.requests,
                args.output,
                source_policy=args.source_policy,
                policy_name=args.policy_name,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
