"""MCP spectra tools (mcp/tools/relativity_spectra.py).

zeeman_effect_analysis passed j straight to zeeman_splitting_count, which was
int(2j + 1) of anything: j = 1.3 reported 3 sublevels, j = -2 reported -3.
"""
import json

from sigma_ground.mcp.tools import relativity_spectra as RS


def test_zeeman_sublevels_for_physical_j():
    for j, n in ((0, 1), (0.5, 2), (1, 3), (1.5, 4), (2.5, 6)):
        r = RS.zeeman_effect_analysis(j)
        assert r["value"] == {"zeeman_sublevels": n}, j


def test_zeeman_declines_non_physical_j_with_strict_json():
    for j in (1.3, -2.0, float("nan"), float("inf")):
        r = RS.zeeman_effect_analysis(j)
        json.dumps(r, allow_nan=False)             # nan/inf aren't JSON
        assert r["value"] is None, j
        assert "half-integer" in r["notes"]
