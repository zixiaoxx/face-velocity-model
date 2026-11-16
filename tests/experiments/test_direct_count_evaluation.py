"""Check direct event retrieval semantics and the optimized threshold sweep."""

import unittest

import numpy as np

from evaluation.direct_count import DirectEventCounter, replay, sweep_order
from evaluation.dynamic import Scores


class DirectCountTests(unittest.TestCase):
    def scores(self):
        return Scores(["a", "b", "c"], [[1, 0.4, 0.2], [0.4, 1, 0.9], [0.2, 0.9, 1]])

    def test_inclusive_and_current_once(self):
        counter = DirectEventCounter(self.scores(), 0.4)
        self.assertEqual(counter.process("e1", "a", 0)["predicted_event_ids"], ("e1",))
        self.assertEqual(
            counter.process("e2", "b", 1)["predicted_event_ids"], ("e1", "e2")
        )

    def test_every_usable_event_retained_without_matching(self):
        counter = DirectEventCounter(self.scores(), 0.8)
        counter.process("e1", "a", 0)
        self.assertEqual(counter.process("e2", "b", 1)["predicted_count"], 1)
        self.assertEqual(
            counter.process("e3", "c", 2)["predicted_event_ids"], ("e2", "e3")
        )

    def test_repeated_photo_distinct_event_and_redelivery(self):
        counter = DirectEventCounter(self.scores(), 0.8)
        original = counter.process("e1", "a", 0)
        self.assertEqual(counter.process("e2", "a", 1)["predicted_count"], 2)
        self.assertEqual(counter.process("e1", "a", 0), dict(original, is_retry=True))
        self.assertEqual(len(counter.events), 2)
        with self.assertRaises(ValueError):
            counter.process("e1", "b", 0)

    def test_expiry_failure_and_chronology(self):
        counter = DirectEventCounter(self.scores(), 0.4, 10)
        counter.process("e1", "a", 0)
        counter.process("e2", "b", 1)
        self.assertIsNone(counter.process("bad", "missing", 2)["predicted_count"])
        self.assertEqual(
            counter.process("e3", "a", 10)["predicted_event_ids"], ("e2", "e3")
        )
        with self.assertRaises(ValueError):
            counter.process("e4", "a", 9)

    def test_labels_do_not_affect_predictions(self):
        first = {k: dict(identity=k) for k in ["a", "b", "c"]}
        second = {k: dict(identity="A") for k in first}
        a = replay(first, self.scores(), list(first), 42, 0.4)
        b = replay(second, self.scores(), list(first), 42, 0.4)
        self.assertEqual(
            [r["predicted_event_ids"] for r in a], [r["predicted_event_ids"] for r in b]
        )

    def test_failed_history_is_missed(self):
        images = {k: dict(identity="A") for k in ["bad", "a"]}
        rows = replay(images, self.scores(), list(images), 42, 0.4)
        self.assertFalse(rows[0]["scoreable"])
        self.assertEqual(rows[1]["missed_events"], 1)
        self.assertEqual(rows[1]["expected_count"], 2)

    def test_sweep_matches_explicit_replay_including_cancellation(self):
        rng = np.random.default_rng(812)
        ids = [f"i{i}" for i in range(25)]
        images = {k: dict(identity=i % 4) for i, k in enumerate(ids)}
        valid = ids[:-2]
        vectors = rng.normal(size=(len(valid), 5))
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        scores = Scores(valid, np.clip(vectors @ vectors.T, -1, 1))
        thresholds = np.array([-1, -0.4, 0, 0.4, 0.8, 1])
        for seed in [42, 43, 44]:
            order = list(np.random.default_rng(seed).permutation(ids))
            swept = sweep_order(
                images, scores, order, thresholds, interval=2, window=17
            )
            for result in swept:
                rows = replay(images, scores, order, seed, result["threshold"], 2, 17)
                usable = [r for r in rows if r["scoreable"]]
                returning = [r for r in usable if r["returning"]]
                for key, field in [
                    ("extra", "extra_events"),
                    ("missed", "missed_events"),
                    ("absolute", "absolute_count_error"),
                    ("exact", "count_exact"),
                    ("membership_exact", "membership_exact"),
                ]:
                    self.assertEqual(result[key], sum(r[field] for r in usable))
                self.assertEqual(
                    result["returning_extra"], sum(r["extra_events"] for r in returning)
                )
                self.assertEqual(
                    result["returning_missed"],
                    sum(r["missed_events"] for r in returning),
                )
                self.assertEqual(
                    result["returning_exact"], sum(r["count_exact"] for r in returning)
                )


if __name__ == "__main__":
    unittest.main()
