"""Causal matching policy checks and parity with the preserved research engine."""

import unittest

import numpy as np

from evaluation.dynamic import DynamicGallery, Scores
from evaluation.dynamic import Policy as ResearchPolicy
from face_velocity.pipeline import Policy, associate, normalize, select_target


class PipelineTests(unittest.TestCase):
    def test_center_target_rejects_ambiguity(self):
        face = dict(bbox=[0, 0, 20, 20], status="detected")
        status, _ = select_target(dict(image_shape=[20, 20, 3], faces=[face, face]))
        self.assertEqual(status, "ambiguous_target")

    def test_finite_vector_required(self):
        for vector in [[0, 0], [float("nan"), 1], [[1, 0]]]:
            with self.assertRaises(ValueError):
                normalize(vector)

    def test_unresolved_has_no_assignment(self):
        self.assertEqual(associate(None, [], 1)["decision"], "unresolved")

    def test_acceptance_and_reference_admission_are_separate(self):
        reference = [dict(identity=1, sequence=0, event_id="a", embedding=[1, 0])]
        query = [0.4, np.sqrt(1 - 0.4**2)]
        decision = associate(query, reference, 2)
        self.assertEqual(decision["assigned_identity"], 1)
        self.assertFalse(decision["reference_added"])
        self.assertEqual(associate([0, 1], reference, 2)["assigned_identity"], 2)

    def test_inclusive_boundaries_and_tie_order(self):
        refs = [
            dict(identity=i, sequence=sequence, event_id=str(i), embedding=[1, 0])
            for i, sequence in [(2, 0), (1, 1)]
        ]
        result = associate([1, 0], refs, 3, Policy(1, 1))
        self.assertEqual(result["assigned_identity"], 1)
        self.assertTrue(result["reference_added"])

    def test_replay_matches_research_decisions(self):
        generator = np.random.default_rng(51)
        vectors = np.stack([normalize(v) for v in generator.normal(size=(100, 12))])
        ids = [str(i) for i in range(len(vectors))]
        oracle = DynamicGallery(
            Scores(ids, vectors @ vectors.T), ResearchPolicy(0.385, 0.05)
        )
        refs, groups = [], 0
        for index, (key, vector) in enumerate(zip(ids, vectors)):
            expected = oracle.process(key, key, index)
            actual = associate(vector, refs, groups + 1)
            if actual["decision"] == "enroll":
                groups += 1
            self.assertEqual(
                actual["assigned_identity"] - 1, expected["assigned_group"]
            )
            self.assertEqual(actual["reference_added"], expected["reference_added"])
            self.assertEqual(actual["winning_reference"], expected["winning_reference"])
            if actual["reference_added"]:
                refs.append(
                    dict(
                        identity=actual["assigned_identity"],
                        sequence=index,
                        event_id=key,
                        embedding=vector.tolist(),
                    )
                )


if __name__ == "__main__":
    unittest.main()
