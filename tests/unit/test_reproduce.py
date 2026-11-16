"""Protect saved evidence and require prerequisites before a reproduction run."""

import tempfile
import unittest
from pathlib import Path

from scripts.reproduce import check_outputs


class ReproductionSafetyTests(unittest.TestCase):
    def test_existing_results_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = root / "artifacts/casia_dynamic_heldout/summary.json"
            result.parent.mkdir(parents=True)
            result.write_text("original evidence")
            with self.assertRaises(FileExistsError):
                check_outputs("main", root)
            self.assertEqual(result.read_text(), "original evidence")

    def test_appendix_requires_main_results(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                check_outputs("appendix", Path(directory))

    def test_empty_checkout_allows_main_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            check_outputs("main", Path(directory))
