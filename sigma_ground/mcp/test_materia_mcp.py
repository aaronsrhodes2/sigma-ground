"""Tests for the Materia↔MCP exposure layer (mcp/tools/simulation.py).

Verifies the seam: Mentat wraps Materia's results in a ToolResult with a
top-level `value` (so the switchboard extractor works), carries the worked
chain in inputs, and preserves the honest decline (value None) for scenarios
Materia can't model. Uses use_llm=False so no ollama dependency.
"""
import os
import sys

_CANON = r"D:\Aaron\development\sigma-ground"
if os.path.isdir(_CANON) and _CANON not in sys.path:
    sys.path.insert(0, _CANON)

from sigma_ground.mcp.tools.simulation import (
    simulate, run_simulation, list_simulation_scenarios)


def test_run_simulation_structured():
    tr = run_simulation("terminal_velocity_drop",
                        {"material_key": "copper", "radius_m": 0.05,
                         "drop_altitude_m": 10000}).to_dict()
    assert tr["value"] is not None
    assert 100 < tr["value"] < 220          # impact speed ≈ 160 m/s
    assert tr["units"] == "m/s"
    assert tr["provenance_tag"] == "DERIVED"
    assert "chain" in tr["inputs"] and len(tr["inputs"]["chain"]) == 1


def test_simulate_nl_deterministic():
    tr = simulate("how fast does a 5 cm copper ball hit the ground from 10 km?",
                  use_llm=False).to_dict()
    assert tr["value"] is not None and 100 < tr["value"] < 220
    assert tr["units"] == "m/s"
    assert "materia" in tr["source"]


def test_simulate_declines_non_physics():
    """A non-scenario returns a clarification with value None — never a fake."""
    tr = simulate("what is the capital of France?", use_llm=False).to_dict()
    assert tr["value"] is None
    assert tr["provenance_tag"] == "SPECULATIVE-PENDING"
    assert tr["notes"]


def test_list_scenarios_discovery():
    tr = list_simulation_scenarios().to_dict()
    verbs = tr["value"]
    for v in ("terminal_velocity_drop", "drag_heating_drop",
              "high_altitude_descent", "supersonic_projectile", "vertical_launch"):
        assert v in verbs, v
    cat = tr["inputs"]["catalog"]
    assert "slots" in cat["terminal_velocity_drop"]
    assert "outputs" in cat["terminal_velocity_drop"]


def test_unknown_verb_declines_gracefully():
    tr = run_simulation("not_a_verb", {}).to_dict()
    assert tr["value"] is None
    assert "Unknown simulation verb" in tr["notes"]
