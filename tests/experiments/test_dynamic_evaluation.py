"""Check causal reference updates, idempotency and non-cancelling count errors."""

import unittest

from evaluation.dynamic import (
    DynamicGallery,
    Policy,
    Scores,
    aggregate,
    evaluate_stream,
    membership_errors,
)


class DynamicEvaluationTests(unittest.TestCase):
    def test_count_selection_does_not_apply_historical_merge_budget(self):
        images = {k: dict(identity=k) for k in ["a", "b", "c"]}
        row, _ = evaluate_stream(
            images, self.scores(), Policy(0.4, None), list(images), 42
        )
        better = dict(
            row,
            policy="low_cost",
            membership_cost=0.1,
            first_arrival_false_merge_rate=0.02,
        )
        worse = dict(
            row,
            policy="higher_cost",
            membership_cost=0.2,
            first_arrival_false_merge_rate=0,
        )
        self.assertEqual(aggregate([better, worse], "count")[1]["policy"], "low_cost")
        self.assertEqual(
            aggregate([better, worse], "first-arrival-budget")[1]["policy"],
            "higher_cost",
        )

    def scores(self):
        return Scores(
            ["a", "b", "c"],
            [
                [1, 0.45, 0.30],
                [0.45, 1, 0.90],
                [0.30, 0.90, 1],
            ],
        )

    def test_enroll_without_self_match_and_update_only_after_search(self):
        engine = DynamicGallery(self.scores(), Policy(0.4, 0))
        a = engine.process("e1", "a", 0)
        self.assertIsNone(a["top_score"])
        self.assertTrue(a["new_group"])
        b = engine.process("e2", "b", 1)
        self.assertEqual(b["top_score"], 0.45)
        self.assertEqual(b["winning_reference"], "a")
        self.assertTrue(b["reference_added"])
        c = engine.process("e3", "c", 2)
        self.assertEqual(c["winning_reference"], "b")
        self.assertEqual(c["assigned_group"], 0)

    def test_strict_update_keeps_accepted_event_countable(self):
        engine = DynamicGallery(self.scores(), Policy(0.4, 0.1))
        engine.process("e1", "a", 0)
        b = engine.process("e2", "b", 1)
        self.assertEqual(b["decision"], "accept")
        self.assertEqual(b["predicted_count"], 2)
        self.assertFalse(b["reference_added"])
        c = engine.process("e3", "c", 2)
        self.assertTrue(c["new_group"])
        self.assertEqual(c["predicted_count"], 1)

    def test_acceptance_and_update_are_inclusive(self):
        scores = Scores(["a", "b", "c"], [[1, 0.4, 0.5], [0.4, 1, 0.1], [0.5, 0.1, 1]])
        engine = DynamicGallery(scores, Policy(0.4, 0.1))
        engine.process("e1", "a", 0)
        self.assertEqual(engine.process("e2", "b", 1)["decision"], "accept")
        self.assertTrue(engine.process("e3", "c", 2)["reference_added"])

    def test_duplicate_is_noop_and_same_photo_new_event_counts(self):
        engine = DynamicGallery(self.scores(), Policy(0.4, 0), 10)
        first = engine.process("e1", "a", 0)
        second = engine.process("e2", "a", 1)
        self.assertEqual(second["predicted_count"], 2)
        references = len(engine.references)
        self.assertEqual(engine.process("e1", "a", 0), dict(first, is_retry=True))
        self.assertEqual(len(engine.references), references)
        for image, seconds in [("b", 0), ("a", 2)]:
            with self.assertRaises(ValueError):
                engine.process("e1", image, seconds)

    def test_window_lower_boundary_excluded_references_retained(self):
        engine = DynamicGallery(self.scores(), Policy(0.4, None), 10)
        engine.process("e1", "a", 0)
        engine.process("e2", "b", 1)
        result = engine.process("e3", "a", 10)
        self.assertEqual(result["predicted_event_ids"], ("e2", "e3"))
        self.assertEqual(result["winning_reference"], "a")
        self.assertEqual(result["references"], 1)

    def test_unresolved_does_not_enroll_but_remains_in_true_history(self):
        images = {k: dict(identity="A") for k in ["bad", "a"]}
        summary, rows = evaluate_stream(
            images, self.scores(), Policy(0.4, None), ["bad", "a"], 42
        )
        self.assertEqual(rows[0]["decision"], "unresolved")
        self.assertEqual(rows[0]["groups"], 0)
        self.assertEqual(rows[1]["expected_count"], 2)
        self.assertEqual(rows[1]["predicted_count"], 1)
        self.assertEqual(rows[1]["missed_events"], 1)
        self.assertEqual(summary["unresolved"], 1)

    def test_equal_counts_do_not_cancel_membership_errors(self):
        self.assertEqual(membership_errors(["e1", "wrong"], ["e1", "missed"]), (1, 1))

    def test_contamination_carries_into_later_decisions(self):
        images = {
            k: dict(identity=identity)
            for k, identity in [("a", "A"), ("b", "B"), ("c", "B")]
        }
        baseline, base_rows = evaluate_stream(
            images, self.scores(), Policy(0.4, None), ["a", "b", "c"], 42
        )
        adaptive, adaptive_rows = evaluate_stream(
            images, self.scores(), Policy(0.4, 0), ["a", "b", "c"], 42
        )
        self.assertEqual(adaptive["wrong_appended_references"], 2)
        self.assertEqual(baseline["wrong_appended_references"], 0)
        self.assertEqual(adaptive_rows[2]["winning_reference"], "b")
        self.assertEqual(base_rows[2]["decision"], "enroll")
        self.assertEqual(adaptive_rows[2]["extra_events"], 1)

    def test_true_labels_never_change_matching(self):
        first = {k: dict(identity=k) for k in ["a", "b", "c"]}
        second = {k: dict(identity="same") for k in first}
        _, rows_a = evaluate_stream(
            first, self.scores(), Policy(0.4, 0), list(first), 42
        )
        _, rows_b = evaluate_stream(
            second, self.scores(), Policy(0.4, 0), list(first), 42
        )
        for a, b in zip(rows_a, rows_b):
            for key in [
                "assigned_group",
                "reference_added",
                "top_score",
                "predicted_count",
            ]:
                self.assertEqual(a[key], b[key])

    def test_repeated_runs_reset_all_state(self):
        images = {k: dict(identity=k) for k in ["a", "b", "c"]}
        args = (images, self.scores(), Policy(0.4, 0), list(images), 42)
        self.assertEqual(evaluate_stream(*args), evaluate_stream(*args))

    def test_identity_tie_uses_group_order_not_reference_order(self):
        # Group0 adds c after group1 enrolls b. q ties b and c, not a.
        scores = Scores(
            ["a", "b", "c", "q"],
            [
                [1, 0.1, 0.8, 0.1],
                [0.1, 1, 0.1, 0.7],
                [0.8, 0.1, 1, 0.7],
                [0.1, 0.7, 0.7, 1],
            ],
        )
        engine = DynamicGallery(scores, Policy(0.4, 0))
        for i, k in enumerate(["a", "b", "c"]):
            engine.process(str(i), k, i)
        result = engine.process("3", "q", 3)
        self.assertEqual(result["assigned_group"], 0)
        self.assertEqual(result["winning_reference"], "c")

    def test_presence_is_before_enrollment_and_rates_have_distinct_denominators(self):
        images = {k: dict(identity="A") for k in ["a", "b", "c"]}
        summary, rows = evaluate_stream(
            images, self.scores(), Policy(0.4, None), ["a", "b", "c"], 42
        )
        self.assertFalse(rows[0]["gallery_identity_present_before"])
        self.assertTrue(rows[1]["gallery_identity_present_before"])
        self.assertEqual(summary["gallery_absent_queries"], 1)
        self.assertEqual(summary["gallery_present_queries"], 2)
        self.assertEqual(summary["dynamic_fpir"], 0)
        self.assertEqual(summary["dynamic_rank1_fnir"], 0.5)
        self.assertEqual(summary["present_rejections"], 1)
        self.assertEqual(summary["present_wrong_accepts"], 0)

    def test_seen_person_in_wrong_group_remains_absent_despite_own_reference(self):
        images = {k: dict(identity=i) for k, i in [("a", "A"), ("b", "B"), ("c", "B")]}
        summary, rows = evaluate_stream(
            images, self.scores(), Policy(0.4, 0), ["a", "b", "c"], 42
        )
        self.assertTrue(rows[2]["returning"])
        self.assertFalse(rows[2]["gallery_identity_present_before"])
        self.assertTrue(rows[2]["dynamic_false_positive"])
        self.assertEqual(summary["gallery_absent_queries"], 3)
        self.assertEqual(summary["dynamic_fpir"], 2 / 3)
        self.assertEqual(summary["first_arrival_false_merge_rate"], 0.5)
        self.assertIsNone(summary["dynamic_rank1_fnir"])

    def test_corrective_enrollment_after_false_merge_is_not_false_negative(self):
        images = {k: dict(identity=i) for k, i in [("a", "A"), ("b", "B"), ("c", "B")]}
        _, rows = evaluate_stream(
            images, self.scores(), Policy(0.4, None), ["a", "b", "c"], 42
        )
        self.assertTrue(rows[2]["returning_new_group"])
        self.assertFalse(rows[2]["gallery_identity_present_before"])
        self.assertFalse(rows[2]["dynamic_false_negative"])
        self.assertFalse(rows[2]["dynamic_false_positive"])

    def test_wrong_rank1_counts_as_fnir_even_when_correct_identity_above_threshold(
        self,
    ):
        scores = Scores(["a", "b", "q"], [[1, 0.1, 0.5], [0.1, 1, 0.8], [0.5, 0.8, 1]])
        images = {k: dict(identity=i) for k, i in [("a", "A"), ("b", "B"), ("q", "A")]}
        summary, rows = evaluate_stream(
            images, scores, Policy(0.4, 0), ["a", "b", "q"], 42
        )
        self.assertTrue(rows[2]["present_wrong_accept"])
        self.assertFalse(rows[2]["dynamic_false_positive"])
        self.assertEqual(summary["dynamic_rank1_fnir"], 1)
        self.assertEqual(summary["present_wrong_accepts"], 1)
        self.assertEqual(summary["present_rejections"], 0)

    def test_matching_same_person_fragment_can_succeed_with_wrong_count(self):
        scores = Scores(["a", "b", "q"], [[1, 0.1, 0.5], [0.1, 1, 0.8], [0.5, 0.8, 1]])
        images = {k: dict(identity="A") for k in ["a", "b", "q"]}
        _, rows = evaluate_stream(images, scores, Policy(0.4, 0), ["a", "b", "q"], 42)
        self.assertTrue(rows[1]["dynamic_false_negative"])
        self.assertFalse(rows[2]["dynamic_false_negative"])
        self.assertEqual(rows[2]["missed_events"], 1)
        self.assertFalse(rows[2]["count_exact"])

    def test_unresolved_inputs_do_not_enter_matching_denominators(self):
        images = {k: dict(identity="A") for k in ["bad", "a", "bad2"]}
        summary, rows = evaluate_stream(
            images, self.scores(), Policy(0.4, None), ["bad", "a", "bad2"], 42
        )
        self.assertEqual(summary["gallery_absent_queries"], 1)
        self.assertEqual(summary["gallery_present_queries"], 0)
        self.assertIsNone(summary["dynamic_rank1_fnir"])
        self.assertTrue(rows[2]["gallery_identity_present_before"])
        self.assertFalse(rows[2]["dynamic_false_negative"])


if __name__ == "__main__":
    unittest.main()
