"""Protect clean-gallery roles, search denominators and operating-point ties."""

import unittest

import numpy as np

from evaluation.fixed_gallery import (
    add_references,
    assign_roles,
    metric_inputs,
    select_point,
    sweep_thresholds,
    threshold_metrics,
    validate_roles,
)


class SearchEvaluationTests(unittest.TestCase):
    def test_invalid_image_stays_query_and_reserved_reference_is_not_query(self):
        images = {
            f"{name}{i}": dict(image_id=f"{name}{i}", identity=name)
            for name, count in [("A", 3), ("B", 2), ("C", 1)]
            for i in range(count)
        }
        valid = set(images) - {"A2"}
        roles = assign_roles(images, valid, 42)
        self.assertEqual(roles, assign_roles(images, valid, 42))
        validate_roles(roles, images, valid)
        self.assertEqual(len(roles["primary_references"]), 2)
        self.assertEqual(len(roles["extra_references"]), 1)
        self.assertIn("A2", roles["query_ids"])
        self.assertIn("C0", roles["query_ids"])
        self.assertFalse(set(roles["extra_references"]) & set(roles["query_ids"]))
        roles["query_ids"].append(roles["primary_references"][0])
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_roles(roles, images, valid)

    def test_extra_reference_max_cannot_reduce_identity_scores(self):
        a = np.array([[0.6, 0.2], [0.5, 0.7]])
        b, extra = add_references(a, np.array([[0.4], [0.9]]), [0])
        np.testing.assert_array_equal(b, [[0.6, 0.2], [0.9, 0.7]])
        np.testing.assert_array_equal(extra, [[False, False], [True, False]])
        np.testing.assert_array_equal(a, [[0.6, 0.2], [0.5, 0.7]])

    def test_correct_score_below_wrong_candidate_is_identification_failure(self):
        rows = [
            dict(kind="mated", status="scoreable", top_score=0.6, rank1_correct=False),
            dict(kind="mated", status="scoreable", top_score=0.4, rank1_correct=True),
            dict(
                kind="mated",
                status="target_not_detected",
                top_score=None,
                rank1_correct=False,
            ),
            dict(
                kind="non_mated", status="scoreable", top_score=0.4, rank1_correct=False
            ),
        ]
        inputs = metric_inputs(rows)
        point = threshold_metrics(inputs, 0.4)
        self.assertEqual(point["known_correct_accepted"], 1)
        self.assertEqual(point["known_wrong_accepted"], 1)
        self.assertEqual(point["known_rejected"], 0)
        self.assertEqual(point["rank1_fnir"], 0.5)
        self.assertEqual(point["fpir"], 1)
        self.assertEqual(point["known_extraction_failures"], 1)
        self.assertAlmostEqual(point["all_input_known_correct_acceptance"], 1 / 3)

    def test_threshold_selection_respects_whole_tie_group_and_budget(self):
        inputs = dict(
            correct=np.array([0.4, 0.5, 0.7]),
            wrong=np.array([0.6]),
            unknown=np.array([0.2, 0.4, 0.4]),
            known_scoreable=4,
            unknown_scoreable=3,
            known_attempted=4,
            unknown_attempted=3,
        )
        sweep = sweep_thresholds(inputs)
        chosen = select_point(sweep, 1 / 3)
        self.assertEqual(chosen["allowed_unknown_false_accepts"], 1)
        self.assertEqual(chosen["unknown_false_accepted"], 0)
        self.assertGreater(chosen["threshold"], 0.4)
        self.assertLess(chosen["threshold"], 0.5)
        self.assertEqual(chosen["known_correct_accepted"], 2)
        feasible = [r for r in sweep if r["unknown_false_accepted"] <= 1]
        self.assertEqual(chosen["rank1_fnir"], min(r["rank1_fnir"] for r in feasible))

    def test_empty_query_group_does_not_report_zero_error(self):
        with self.assertRaises(ValueError):
            metric_inputs([dict(kind="mated", status="target_not_detected")])


if __name__ == "__main__":
    unittest.main()
