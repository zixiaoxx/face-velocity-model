"""Protect split leakage, pair definitions and inclusive threshold semantics."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.pairwise import (
    choose_final_identities,
    error_counts,
    make_pairs,
    read_manifest,
)


class MatchingEvaluationTests(unittest.TestCase):
    def test_equal_threshold_is_accepted(self):
        result = error_counts(
            np.array([0.2, 0.4, 0.4, 0.8]), np.array([0.1, 0.4, 0.5]), 0.4
        )
        self.assertEqual(result["false_matches"], 2)
        self.assertEqual(result["false_nonmatches"], 1)
        self.assertEqual(result["fnmr"], 0.25)
        self.assertAlmostEqual(result["fmr"], 2 / 3)

    def test_no_scoreable_pairs_does_not_report_zero_error(self):
        with self.assertRaises(ValueError):
            error_counts(np.array([]), np.array([0.2]), 0.4)

    def test_pairs_are_distinct_labeled_and_reproducible(self):
        images = [
            dict(identity=name, image_id=f"{name}_{i}")
            for name in ["A", "B", "C"]
            for i in range(3)
        ]
        pairs = make_pairs(images, 42, impostor_count=10, genuine_cap=2)
        self.assertEqual(pairs, make_pairs(images, 42, 10, 2))
        self.assertEqual(sum(p["kind"] == "genuine" for p in pairs), 6)
        seen = set()
        for pair in pairs:
            left, right = pair["left"], pair["right"]
            self.assertNotEqual(left, right)
            key = tuple(sorted((left, right)))
            self.assertNotIn(key, seen)
            seen.add(key)
            self.assertEqual(left[0] == right[0], pair["kind"] == "genuine")

    def test_exposed_identities_cannot_enter_final(self):
        groups = {f"id{i:02}": list(range(2 + i % 5)) for i in range(40)}
        exposed = {"id00", "id01", "id02", "id03"}
        final, _ = choose_final_identities(
            groups, exposed, 12, np.random.default_rng(3)
        )
        self.assertEqual(len(final), 12)
        self.assertFalse(final & exposed)

    def test_final_manifest_is_refused_before_inference(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "manifest.json"
            path.write_text(json.dumps(dict(partition="final_test")))
            with self.assertRaisesRegex(ValueError, "final-test"):
                read_manifest(path)

    def test_duplicate_bytes_are_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "manifest.json"
            images = [
                dict(image_id=str(i), identity="A", sha256="same") for i in range(2)
            ]
            path.write_text(json.dumps(dict(partition="development", images=images)))
            with self.assertRaisesRegex(ValueError, "Duplicate selected"):
                read_manifest(path)


if __name__ == "__main__":
    unittest.main()
