"""sigma_ground.field.datasets: where the JPL kernel is, by environment variable first, and None (not a made-up path) when it is not installed."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class TestDatasets(unittest.TestCase):
    def test_environment_variable_wins_and_the_kernel_is_found_under_jpl(self):
        from sigma_ground.field import datasets
        with tempfile.TemporaryDirectory() as t:
            (Path(t) / "jpl").mkdir()
            (Path(t) / "jpl" / "de440s.bsp").write_bytes(b"x")
            with mock.patch.dict(os.environ, {"SIGMA_DATASETS": t}):
                self.assertEqual(datasets.datasets_root(), Path(t))
                self.assertEqual(datasets.de440_kernel(), Path(t) / "jpl" / "de440s.bsp")

    def test_a_root_without_the_kernel_gives_none_not_a_guessed_path(self):                   # BUG if a tool were handed a path that does not exist
        from sigma_ground.field import datasets
        with tempfile.TemporaryDirectory() as t:
            (Path(t) / "jpl").mkdir()
            with mock.patch.dict(os.environ, {"SIGMA_DATASETS": t}):
                self.assertIsNone(datasets.de440_kernel())

    def test_a_missing_environment_path_falls_through_to_the_defaults(self):                  # BY CONSTRUCTION: a typo in the variable must not hide a good default
        from sigma_ground.field import datasets
        with mock.patch.dict(os.environ, {"SIGMA_DATASETS": "/no/such/dir"}):
            r = datasets.datasets_root()
            self.assertTrue(r is None or r.is_dir())
            self.assertNotEqual(r, Path("/no/such/dir"))


if __name__ == "__main__":
    unittest.main()
