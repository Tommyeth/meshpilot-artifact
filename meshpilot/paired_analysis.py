from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .artifact_io import PAIR_KEY, read_rows


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _is_injected_unknown_id(row: dict[str, str]) -> bool:
    """Only the deterministic wrapper produces the ``hallucinated:`` prefix."""

    return str(row.get("agent_plan_id") or "").startswith("hallucinated:")


def _is_model_unknown_id(row: dict[str, str]) -> bool:
    """An unknown ID the model emitted on its own (never injected)."""

    reasons = str(row.get("verification_reasons_json") or "").lower()
    return (not _is_injected_unknown_id(row)) and "unknown or hallucinated plan_id" in reasons


def _is_valid_direct_action(row: dict[str, str]) -> bool:
    """The verified row executed the model's own plan, not a fallback."""

    if "agent_action" not in row and "agent_action_valid" not in row:
        return True  # rows without action provenance cannot be classified further
    return (
        str(row.get("agent_action") or "") == "select"
        and _truthy(row.get("agent_action_valid", ""))
        and str(row.get("agent_api_status") or "") != "error"
    )


def _cluster_percentile_interval(
    differences_by_seed: dict[str, list[float]],
    *,
    seed: int,
    draws: int,
) -> tuple[float, float]:
    # Resample whole seed clusters, then pool their decision-level totals.
    # Averaging sampled seed means instead estimates a different quantity when
    # the valid-ID filtering leaves unequal numbers of decisions per seed.
    cluster_totals = {
        cluster: (sum(values), len(values))
        for cluster, values in differences_by_seed.items()
        if values
    }
    clusters = sorted(cluster_totals)
    if not clusters:
        raise ValueError("cluster interval requires at least one nonempty seed")
    if draws <= 0:
        raise ValueError("cluster interval requires at least one bootstrap draw")
    rng = random.Random(seed)
    estimates = []
    for _ in range(draws):
        sampled = [cluster_totals[rng.choice(clusters)] for _ in clusters]
        estimates.append(sum(total for total, _ in sampled) / sum(count for _, count in sampled))
    estimates.sort()
    low = estimates[int(0.025 * (len(estimates) - 1))]
    high = estimates[int(0.975 * (len(estimates) - 1))]
    return low, high


def analyze_non_injected_pairs(
    rows: list[dict[str, str]],
    *,
    direct_policy: str = "vertex_agent_verified",
    compiled_policy: str = "vertex_intent_compiled",
    bootstrap_seed: int = 20260829,
    bootstrap_draws: int = 5_000,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    indexed: dict[str, dict[tuple[str, str, str], dict[str, str]]] = defaultdict(dict)
    for row in rows:
        policy = row.get("policy", "")
        if policy not in {direct_policy, compiled_policy}:
            continue
        key = tuple(str(row[field]) for field in PAIR_KEY)
        if key in indexed[policy]:
            raise ValueError(f"duplicate {policy} row for {key}")
        indexed[policy][key] = row

    if not indexed[direct_policy] or not indexed[compiled_policy]:
        raise ValueError("direct and compiled policies must both be present")
    if set(indexed[direct_policy]) != set(indexed[compiled_policy]):
        raise ValueError("direct and compiled policy keys do not match")

    paired_rows: list[dict[str, object]] = []
    injected_pairs = 0
    model_unknown_pairs = 0
    invalid_action_pairs = 0
    by_seed: dict[str, list[float]] = defaultdict(list)
    for key in sorted(indexed[direct_policy]):
        direct = indexed[direct_policy][key]
        compiled = indexed[compiled_policy][key]
        if _is_injected_unknown_id(direct):
            injected_pairs += 1
            continue
        if _is_model_unknown_id(direct):
            model_unknown_pairs += 1
            continue
        if not _is_valid_direct_action(direct):
            # Abstention, API error, or a stale/infeasible model ID: the verified
            # row executed the fallback, so its regret is not a direct-ID regret.
            invalid_action_pairs += 1
            continue
        direct_regret = float(direct["normalized_regret_to_oracle"])
        compiled_regret = float(compiled["normalized_regret_to_oracle"])
        difference = compiled_regret - direct_regret
        seed, scenario, task_id = key
        by_seed[seed].append(difference)
        paired_rows.append(
            {
                "seed": seed,
                "scenario": scenario,
                "task_id": task_id,
                "intent_phrase_id": direct.get("intent_phrase_id", ""),
                "intent_profile": direct.get("intent_profile", ""),
                "direct_regret": direct_regret,
                "compiled_regret": compiled_regret,
                "compiled_minus_direct": difference,
            }
        )

    if not paired_rows:
        raise ValueError("no known-ID pairs remain after filtering")
    low, high = _cluster_percentile_interval(
        by_seed,
        seed=bootstrap_seed,
        draws=bootstrap_draws,
    )
    direct_values = [float(row["direct_regret"]) for row in paired_rows]
    compiled_values = [float(row["compiled_regret"]) for row in paired_rows]
    differences = [float(row["compiled_minus_direct"]) for row in paired_rows]
    per_seed = [
        {
            "seed": seed,
            "pairs": len(by_seed[seed]),
            "mean_compiled_minus_direct": statistics.fmean(by_seed[seed]),
        }
        for seed in sorted(by_seed, key=int)
    ]
    summary = {
        "schema_version": 2,
        "protocol": "known-id-paired-effect-attribution-v2",
        "analysis_status": "exploratory_post_hoc",
        "direct_policy": direct_policy,
        "compiled_policy": compiled_policy,
        "total_paired_keys": len(indexed[direct_policy]),
        "injected_unknown_id_pairs_excluded": injected_pairs,
        "model_unknown_id_pairs_excluded": model_unknown_pairs,
        "abstain_or_error_pairs_excluded": invalid_action_pairs,
        "known_id_pairs": len(paired_rows),
        "seed_clusters": len(by_seed),
        "mean_direct_regret": statistics.fmean(direct_values),
        "mean_compiled_regret": statistics.fmean(compiled_values),
        "mean_compiled_minus_direct": statistics.fmean(differences),
        "compiled_minus_direct_cluster_ci95": [low, high],
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_draws": bootstrap_draws,
        "estimand": "pooled decision-level mean of compiled regret minus direct regret",
        "bootstrap_statistic": "sum of differences / sum of decision counts in resampled seed clusters",
        "bootstrap_resampling": "sample seed_clusters whole clusters with replacement per draw",
        "bootstrap_cluster_order": sorted(by_seed),
        "bootstrap_percentile_rule": "sorted estimates at floor(q * (draws - 1)), q = 0.025 and 0.975",
        "target_wins": sum(value < 0 for value in differences),
        "ties": sum(value == 0 for value in differences),
        "target_losses": sum(value > 0 for value in differences),
        "per_seed": per_seed,
        "interpretation": (
            "This post-hoc paired analysis excludes deterministically injected unknown IDs. "
            "It isolates direct-ID versus profile-compiled selection under the manuscript's "
            "declared objective; it is not external evidence of operator utility or realized optimality."
        ),
    }
    return paired_rows, summary


def build_non_injected_artifact(
    requests: Path,
    output: Path,
    *,
    direct_policy: str = "vertex_agent_verified",
    compiled_policy: str = "vertex_intent_compiled",
) -> dict[str, object]:
    rows = read_rows(requests)
    paired_rows, summary = analyze_non_injected_pairs(
        rows, direct_policy=direct_policy, compiled_policy=compiled_policy
    )
    output.mkdir(parents=True, exist_ok=True)
    fields = list(paired_rows[0])
    with (output / "paired_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(paired_rows)
    summary = {
        **summary,
        "source_requests": str(requests),
        "source_requests_sha256": hashlib.sha256(requests.read_bytes()).hexdigest(),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    validation = {
        "schema_version": 1,
        "status": "PASS"
        if summary["known_id_pairs"]
        + summary["injected_unknown_id_pairs_excluded"]
        + summary["model_unknown_id_pairs_excluded"]
        + summary["abstain_or_error_pairs_excluded"]
        == summary["total_paired_keys"]
        else "FAIL",
        "known_id_pairs": summary["known_id_pairs"],
        "injected_unknown_id_pairs_excluded": summary["injected_unknown_id_pairs_excluded"],
        "model_unknown_id_pairs_excluded": summary["model_unknown_id_pairs_excluded"],
        "abstain_or_error_pairs_excluded": summary["abstain_or_error_pairs_excluded"],
        "seed_clusters": summary["seed_clusters"],
        "row_file_rows": len(paired_rows),
    }
    (output / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    return validation


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Analyze known-ID direct versus compiled policy pairs")
    parser.add_argument("--requests", default=Path("data/decisions.csv.gz"), type=Path)
    parser.add_argument("--output", default=Path("reproduced/paired"), type=Path)
    parser.add_argument("--direct-policy", default="vertex_agent_verified")
    parser.add_argument("--compiled-policy", default="vertex_intent_compiled")
    args = parser.parse_args(argv)
    validation = build_non_injected_artifact(
        args.requests,
        args.output,
        direct_policy=args.direct_policy,
        compiled_policy=args.compiled_policy,
    )
    print(json.dumps(validation, indent=2))
    if validation["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
