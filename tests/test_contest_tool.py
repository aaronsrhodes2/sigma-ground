"""prediction_contest (C26): the lookup over the fair contest's report, and what it says when there is none.

Ground truth here is a small report written by the test (the format scripts/fair_contest_report.py writes), so the tool's words are checked against known numbers; the 2026-10-12 contest re-read
(Moon 3.399 km dynamics gap at +5 yr) is the number the report script reproduced from the stored jsonl.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from sigma_ground.field.interface import contest


def _report(path: Path) -> None:
    doc = {"metadata": {"what": "test", "runs": 10, "truth": "DE442", "predictor": "restated_math", "vintages": ["DE405", "DE421"], "targets": [2010], "issue_years": [1998.7, 2008.1]},
           "table": {"Moon": {"5": {"eO_med": 1.2, "eV_med": 0.4, "dyn_med": 1.5, "e0_med": 0.3, "ours_wins": 1, "n": 10, "p_sign": 0.021, "eT_med": 0.002}},
                     "Jupiter": {"5": {"eO_med": 50.1, "eV_med": 50.0, "dyn_med": 2.1, "e0_med": 49.0, "ours_wins": 5, "n": 10, "p_sign": 1.0, "eT_med": 9.5}}},
           "verdict_M_B": {}, "blankets": {}}
    path.write_text(json.dumps(doc), encoding="utf-8")


class TestContestLookup(unittest.TestCase):
    def test_words_and_numbers(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.json"
            _report(p)
            contest._table.cache_clear()
            m = contest.prediction_contest("the moon", 5, path=p)
            self.assertEqual(m["ours_km"], 1.2)
            self.assertEqual(m["ours_closer"], "1 of 10")
            self.assertIn("further from reality than JPL's prediction at the time", m["summary"])
            self.assertIn("0.002 km", m["summary"])
            j = contest.prediction_contest("jupiter", 5, path=p)
            self.assertIn("indistinguishable", j["summary"])
            self.assertEqual(j["reality"], "DE442")
            with self.assertRaises(contest.ContestError):
                contest.prediction_contest("io", 5, path=p)                    # not a contest body
            with self.assertRaises(contest.ContestError):
                contest.prediction_contest("moon", 20, path=p)                 # no such lead in this report

    def test_missing_report_is_a_plain_decline(self):
        with self.assertRaises(contest.ContestError) as cm:
            contest.prediction_contest("mars", 5, path=Path("D:/nowhere/FAIR_CONTEST_v2.json"))
        self.assertIn("has not been run", str(cm.exception))

    def test_tool_wrapper_declines_without_a_table(self):
        from sigma_ground.mcp.tools import celestial_tools as ct
        if contest._FILE.is_file():
            self.skipTest("a real report is present; the decline path is not reachable")
        r = ct.prediction_contest("mars", 5)
        self.assertIsNone(r.value)
        self.assertIn("has not been run", r.notes)


if __name__ == "__main__":
    unittest.main()
