"""The sky_scoreboard tool (C36) answers from the committed fixtures: the summary, a body, a miss; body_skill reaches it for a body the whole-sky pass has scored."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
FIX = ROOT / "sigma_ground/field/interface/fixtures/mpc_live"


class TestSkyScoreboard(unittest.TestCase):
    def test_summary_reads_the_report(self):
        from sigma_ground.mcp.tools import astronomy as t
        r = t.sky_scoreboard()
        rep = json.loads((FIX / "sky_report.json").read_text(encoding="utf-8"))
        self.assertEqual(r.value["n_obs"], rep["n_obs"]); self.assertIn("Whole sky as of", r.notes); self.assertIn("never refreshed", r.notes)
        self.assertEqual(r.provenance_tag, "DERIVED")

    def test_a_scored_body_and_a_miss(self):
        from sigma_ground.mcp.tools import astronomy as t
        lat = json.loads((FIX / "sky_scoreboard_latest.json").read_text(encoding="utf-8"))
        row = lat["rows"][0]
        r = t.sky_scoreboard(str(row["number"] if row.get("number") else row["designation"]))
        self.assertEqual(r.value["packed"], row["packed"]); self.assertIn("observations scored", r.notes)
        miss = t.sky_scoreboard("no such body 1234567")
        self.assertIsNone(miss.value); self.assertIn("not among the bodies scored", miss.notes)

    def test_a_roster_body_with_nothing_new_says_so(self):
        from sigma_ground.mcp.tools import astronomy as t
        rep = json.loads((FIX / "sky_report.json").read_text(encoding="utf-8"))
        number, w = next(iter(rep["roster_waiting"].items()))
        r = t.sky_scoreboard(number)
        self.assertEqual(r.value["last_obs"], w["last_obs"]); self.assertIn("nothing has been observed since", r.notes)

    def test_manifest_and_keywords_carry_it(self):
        from sigma_ground.mcp import manifest, tool_keywords
        names = {e["name"] for e in manifest.TOOLS} if hasattr(manifest, "TOOLS") else set(json.dumps(manifest.__dict__, default=str).split('"name": "')[1:]) and {x.split('"')[0] for x in json.dumps(manifest.__dict__, default=str).split('"name": "')[1:]}
        self.assertIn("sky_scoreboard", names)
        self.assertTrue(any("sky_scoreboard" in str(k) for k in vars(tool_keywords).values() if isinstance(k, dict)))


if __name__ == "__main__":
    unittest.main()
