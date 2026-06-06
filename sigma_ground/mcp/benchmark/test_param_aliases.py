"""Regression guard for the param-alias normalizer.

The 2026-06-05 switchboard regression (85.3% -> 52.7%) was caused by Qwen's
`velocity` reaching `velocity_m_s` tools un-renamed (the alias table maps
velocity -> speed_m_s, which those tools don't accept) -> pydantic validation
error -> 8x tool-call loop -> no answer. The fix added a general prefix-fallback
to normalize_kwargs. These tests lock that in.
"""
import unittest

from sigma_ground.mcp.benchmark.param_aliases import normalize_kwargs


class TestNormalizeKwargs(unittest.TestCase):
    def _assert_renames(self, kwargs, real, expected):
        out, _ = normalize_kwargs(dict(kwargs), set(real))
        self.assertEqual(out, expected)
        # every emitted key must be a real param (no validation errors)
        for k in out:
            self.assertIn(k, real, f"{k} not in tool signature")

    def test_velocity_to_velocity_m_s(self):
        """The exact regression: velocity -> velocity_m_s (kinetic_energy/momentum)."""
        self._assert_renames({"mass_kg": 70, "velocity": 5},
                             {"mass_kg", "velocity_m_s"},
                             {"mass_kg": 70, "velocity_m_s": 5})

    def test_velocity_to_speed_m_s_still_works(self):
        """Explicit alias still wins for speed_m_s tools."""
        self._assert_renames({"velocity": 5}, {"speed_m_s"}, {"speed_m_s": 5})

    def test_bare_names_prefix_fallback(self):
        self._assert_renames({"mass": 70, "velocity": 5},
                             {"mass_kg", "velocity_m_s"},
                             {"mass_kg": 70, "velocity_m_s": 5})

    def test_gravity_and_height(self):
        self._assert_renames({"gravity": 1.625, "height": 10},
                             {"g_m_s2", "height_m"},
                             {"g_m_s2": 1.625, "height_m": 10})

    def test_already_canonical_noop(self):
        self._assert_renames({"mass_kg": 1.0, "velocity_m_s": 2.0},
                             {"mass_kg", "velocity_m_s"},
                             {"mass_kg": 1.0, "velocity_m_s": 2.0})

    def test_ambiguous_prefix_not_renamed(self):
        """`n` must NOT rename when two real params share the n_ prefix."""
        out, _ = normalize_kwargs({"n": 1.0}, {"n_from", "n_to"})
        self.assertEqual(out, {"n": 1.0})   # left for pydantic / explicit alias

    def test_unknown_passthrough(self):
        out, _ = normalize_kwargs({"banana": 1}, {"mass_kg"})
        self.assertEqual(out, {"banana": 1})


if __name__ == "__main__":
    unittest.main()
