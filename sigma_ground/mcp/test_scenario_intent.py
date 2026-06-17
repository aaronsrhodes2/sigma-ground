"""Tests for the Mentat-side scenario-intent seam (Q4 manifest enrichment).

Covers coarse intent detection, the provisional per-subsystem emphasis, manifest
assembly, the forward-compatible spine-emphasis read, and that the manifest +
'why' actually travel in the ToolResult that simulate() returns.
"""
import unittest

from sigma_ground.mcp.tools import scenario_intent as si


class TestDetectIntent(unittest.TestCase):
    CASES = [
        ("tip the chair over and see if it falls", "tip"),
        ("drop a 5 cm copper ball from 10 meters", "drop"),
        ("will the wine glass shatter if it hits the floor", "shatter"),
        ("fill the cup with water", "fill"),
        ("does the copper ball float or sink", "float"),
        ("what does the chair look like, render it", "still"),
        ("the quick brown fox jumps", None),
    ]

    def test_detect(self):
        for text, expected in self.CASES:
            self.assertEqual(si.detect_intent(text), expected, msg=text)


class TestEmphasis(unittest.TestCase):
    def test_tip_emphasizes_com_and_friction(self):
        e = si.emphasis_for("tip")
        self.assertIn("center_of_mass", e["physics"])
        self.assertIn("friction_static", e["physics"])
        self.assertEqual(e["render"], "low")

    def test_still_emphasizes_optics_not_dynamics(self):
        e = si.emphasis_for("still")
        self.assertEqual(e["render"], "high")
        self.assertEqual(e["physics"], [])

    def test_same_object_different_emphasis(self):
        """The whole point: same object, intent drives the accuracy budget."""
        self.assertNotEqual(si.emphasis_for("tip"), si.emphasis_for("still"))

    def test_unknown_intent_default(self):
        self.assertEqual(si.emphasis_for(None)["render"], "medium")


class TestManifest(unittest.TestCase):
    def test_build(self):
        m = si.build_manifest("tip the chair over", object_handle="chair")
        self.assertEqual(m["object"], "chair")
        self.assertEqual(m["intent"], "tip")
        self.assertEqual(m["intent_text"], "tip the chair over")
        self.assertIn("center_of_mass", m["emphasis"]["physics"])
        self.assertEqual(m["emphasis_source"], "mentat_provisional")

    def test_why_sentence(self):
        why = si.why_sentence(si.build_manifest("tip the chair over", "chair"))
        self.assertIn("tip", why)
        self.assertIn("center_of_mass", why)
        self.assertIn("chair", why)

    def test_read_emphasis_prefers_spine(self):
        class FakeSpec:
            emphasis = {"physics": ["from_spine"], "render": "x"}
        self.assertEqual(si.read_emphasis(FakeSpec()),
                         {"physics": ["from_spine"], "render": "x"})
        self.assertIsNone(si.read_emphasis(object(), None))


class TestSimulateCarriesManifest(unittest.TestCase):
    """The manifest + 'why' must travel in the ToolResult (decline OR success)."""

    def test_manifest_and_why_in_result(self):
        from sigma_ground.mcp.tools.simulation import simulate
        out = simulate("tip the wooden chair over", use_llm=False,
                       object_handle="chair").to_dict()
        man = out["inputs"]["scenario_manifest"]
        self.assertEqual(man["intent"], "tip")
        self.assertEqual(man["object"], "chair")
        self.assertIn("because the intent is 'tip'", out["notes"])


if __name__ == "__main__":
    unittest.main()
