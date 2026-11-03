"""Small regression checks for the poster's paired statistic and fresh control.

Run with: python -m unittest discover -s tests -p 'test_analysis.py'
"""
import copy
import unittest
from pathlib import Path
from unittest.mock import patch

from meshpilot.paired_analysis import _cluster_percentile_interval
from meshpilot.profile_analysis import analyze_profiles
from meshpilot.artifact_io import read_rows


class ClusterStatisticTests(unittest.TestCase):
    def test_unequal_clusters_weight_each_resampled_decision_equally(self):
        # One success among one decision, and zero successes among three:
        # drawing each seed once must estimate 1/4, not the seed-average 1/2.
        clusters = {"short": [1.0], "long": [0.0, 0.0, 0.0]}
        with patch("meshpilot.paired_analysis.random.Random") as rng:
            rng.return_value.choice.side_effect = ["short", "long"]
            self.assertEqual(
                _cluster_percentile_interval(clusters, seed=5, draws=1),
                (0.25, 0.25),
            )

    def test_cluster_is_duplicated_with_its_observations(self):
        # With three seed draws, duplicating the three-decision seed gives
        # seven observations, and the only success has weight 1/7.
        clusters = {"short": [1.0], "long": [0.0, 0.0, 0.0], "unused": [8.0]}
        with patch("meshpilot.paired_analysis.random.Random") as rng:
            rng.return_value.choice.side_effect = ["short", "long", "long"]
            low, high = _cluster_percentile_interval(clusters, seed=5, draws=1)
        self.assertAlmostEqual(low, 1 / 7)
        self.assertEqual(low, high)

    def test_empty_clusters_and_zero_draws_are_rejected(self):
        with self.assertRaises(ValueError):
            _cluster_percentile_interval({"empty": []}, seed=5, draws=1)
        with self.assertRaises(ValueError):
            _cluster_percentile_interval({"a": [0.0]}, seed=5, draws=0)


class FrozenPosterRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.requests = Path(__file__).resolve().parents[1] / "data/decisions.csv.gz"
        cls.source = read_rows(cls.requests)
        cls.audited, cls.summary = analyze_profiles(cls.source)

    def test_same_state_control_and_paired_statistic(self):
        summary = self.summary
        self.assertEqual(summary["known_id_pairs"], 500)
        self.assertEqual(summary["max_recorded_regret_reconstruction_error"], 0.0)
        self.assertAlmostEqual(summary["mean_fresh_balanced_regret"], 0.04680521866812509)
        ci = summary["compiled_vs_direct"]["target_minus_baseline_cluster_ci95"]
        self.assertAlmostEqual(ci[0], -0.13255682355673096)
        self.assertAlmostEqual(ci[1], -0.021083529872797734)
        comparison = summary["compiled_vs_balanced"]
        self.assertEqual([comparison[k] for k in ("target_wins", "ties", "target_losses")], [124, 374, 2])

    def test_historical_baseline_rows_do_not_affect_fresh_control(self):
        # Deliberately nonsensical old baseline contents must be ignored: this
        # analysis depends on actual candidates of the matched direct rows.
        stale = {"policy": "joint_fixed", "seed": "unused", "scenario": "unused", "task_id": "unused", "actual_candidates_json": "not JSON", "normalized_regret_to_oracle": "not a number"}
        rows, summary = analyze_profiles([*self.source, stale])
        self.assertEqual(rows, self.audited)
        self.assertEqual(summary, self.summary)

    def test_corrupted_compiled_candidate_provenance_is_rejected(self):
        source = copy.deepcopy(self.source)
        compiled = next(row for row in source if row["policy"] == "vertex_intent_compiled")
        compiled["actual_candidates_json"] = "[]"
        with self.assertRaisesRegex(ValueError, "candidate sets differ"):
            analyze_profiles(source)

    def test_error_and_tie_counts_are_not_confused(self):
        summary = self.summary
        errors = summary["profile_errors"]
        self.assertEqual([errors[k] for k in ("all_decisions", "known_id_pairs", "excluded_pairs")], [10, 8, 2])
        self.assertEqual([errors[k] for k in ("known_compiled_wins", "known_ties", "known_compiled_losses")], [2, 1, 5])
        ties = summary["compiled_direct_ties"]
        self.assertEqual([ties[k] for k in ("count", "same_node", "both_zero_regret", "both_positive_regret")], [397, 397, 396, 1])
        self.assertEqual(summary["compiled_losses_vs_direct"]["with_profile_error"], 5)
        example = summary["failure_example"]
        self.assertEqual((example["seed"], example["scenario"], example["task_id"]), ("11", "gpu_pressure", "task-11-007"))
        self.assertEqual(example["operator_intent"], "Keep spend under control while breaking trade-offs toward lower operational risk.")
        self.assertEqual((example["declared_profile"], example["inferred_profile"]), ("resilience", "balanced"))
        self.assertEqual((example["direct_node_id"], example["compiled_node_id"], example["balanced_node_id"]), ("node-09", "node-00", "node-00"))
        self.assertAlmostEqual(example["compiled_regret"], 0.032691441055262144)


if __name__ == "__main__":
    unittest.main()
