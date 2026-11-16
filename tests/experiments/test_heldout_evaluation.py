"""Protect explicit final access, split integrity and protocol immutability."""

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.heldout import validate_split, verify_protocol
from evaluation.pairwise import digest, read_manifest


class HeldoutEvaluationTests(unittest.TestCase):
    def manifests(self):
        development = {"images": [{"identity": "A", "image_id": "a", "sha256": "1"}]}
        final = {
            "partition": "final_test",
            "previously_exposed_identities": [],
            "images": [{"identity": "B", "image_id": "b", "sha256": "2"}],
        }
        return development, final

    def test_final_access_requires_explicit_partition(self):
        _, final = self.manifests()
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "manifest.json"
            p.write_text(json.dumps(final))
            with self.assertRaises(ValueError):
                read_manifest(p)
            self.assertEqual(read_manifest(p, expected_partition="final_test"), final)

    def test_split_rejects_identity_image_and_byte_overlap(self):
        for key in ["identity", "image_id", "sha256"]:
            development, final = self.manifests()
            validate_split(development, final)
            final["images"][0][key] = development["images"][0][key]
            with self.assertRaises(ValueError):
                validate_split(development, final)

    def test_prior_illustration_identity_is_refused(self):
        development, final = self.manifests()
        final["previously_exposed_identities"] = ["B"]
        with self.assertRaises(ValueError):
            validate_split(development, final)

    def test_modified_protocol_is_refused_before_input_access(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            p = out / "protocol.json"
            p.write_text('{"acceptance": 0.385}')
            (out / "protocol.sha256").write_text(digest(p))
            p.write_text('{"acceptance": 0.4}')
            with self.assertRaisesRegex(ValueError, "Frozen protocol changed"):
                verify_protocol(out)


if __name__ == "__main__":
    unittest.main()
