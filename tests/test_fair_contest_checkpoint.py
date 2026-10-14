"""scripts/fair_jpl_contest.py: a job that resumes from its per-lead checkpoint reproduces the uninterrupted run bit for bit.

Ground truth: the uninterrupted run itself (same code, same inputs). A coarse step keeps it to seconds; only the equality of the continuation matters here.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

TRUTH = Path("D:/datasets/jpl/de440s.bsp")


@unittest.skipUnless(TRUTH.exists(), "DE440s kernel not available")
class TestCheckpointResume(unittest.TestCase):
    def test_resumed_job_equals_straight_job(self):                             # BUG: the checkpoint must carry the whole state
        import fair_jpl_contest as fc
        de = json.loads(fc.FIXTURE.read_text())
        key = "j2012"
        snap = de["snapshots"][key]
        jd = snap["epoch"]["jd_tdb"]

        def job(leads, ckpt):
            return (("DE440(control)", None, key, jd, 2.0, leads, None, snap, None, None, False, 2.2382e-5, "iau", ckpt))

        with tempfile.TemporaryDirectory() as d:
            ck = str(Path(d) / "j.ckpt.json")
            straight = fc.run_one(job([1, 2, 3], None))
            fc.run_one(job([1, 2], ck))                                          # leaves a checkpoint after lead 2
            self.assertTrue(Path(ck).exists())
            resumed = fc.run_one(job([1, 2, 3], ck))
        for body, per_lead in straight["bodies"].items():
            for lead, cell in per_lead.items():
                self.assertEqual(cell["eO"], resumed["bodies"][body][lead]["eO"], (body, lead))
        self.assertEqual(sorted(straight["bodies"]["Moon"]), sorted(resumed["bodies"]["Moon"]))


if __name__ == "__main__":
    unittest.main()
