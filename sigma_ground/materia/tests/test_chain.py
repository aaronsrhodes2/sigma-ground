"""Chain tests — decomposition with data flow between steps.

These pin Phase 1: a multi-step Spec where a later step's input is BOUND to an
earlier step's output. The launch→descend→heat chain throws a ball up, then
feeds the apex altitude into the descent and heating verbs.
"""
import os
import sys

_CANON = r"D:\Aaron\development\sigma-ground"
if os.path.isdir(_CANON) and _CANON not in sys.path:
    sys.path.insert(0, _CANON)

import pytest

from sigma_ground.materia import (SimulationSpec, SpecStep, run_spec,
                                  terminal_velocity_drop)


def _launch_descend_heat():
    """Throw a steel ball up; the apex feeds the descent AND the heating step."""
    return SimulationSpec(
        "Throw a steel ball up at 400 m/s — how fast and how hot when it lands?",
        steps=[
            SpecStep("vertical_launch",
                     {"material_key": "steel_mild", "radius_m": 0.05,
                      "launch_speed_m_s": 400.0}),
            SpecStep("terminal_velocity_drop",
                     {"material_key": "steel_mild", "radius_m": 0.05},
                     bindings={"drop_altitude_m": (0, "apex_altitude_m")}),
            SpecStep("drag_heating_drop",
                     {"material_key": "steel_mild", "radius_m": 0.05},
                     bindings={"drop_altitude_m": (0, "apex_altitude_m")}),
        ])


def test_chain_threads_apex_forward():
    """Step 0's apex must flow into steps 1 and 2 as their drop altitude."""
    res = run_spec(_launch_descend_heat())
    apex = res[0].outputs["apex_altitude_m"]
    assert apex > 100
    assert abs(res[1].inputs["drop_altitude_m"] - apex) < 1e-9
    assert abs(res[2].inputs["drop_altitude_m"] - apex) < 1e-9


def test_chain_every_step_self_validates():
    """A chain is trustworthy only if every sub-step passed its own self-check."""
    res = run_spec(_launch_descend_heat())
    assert all(r.validation.get("passed", True) for r in res), \
        [(r.name, r.validation.get("note")) for r in res]


def test_binding_to_missing_output_raises():
    """A binding to a nonexistent output fails loudly — never silently defaults."""
    bad = SimulationSpec("bad chain", steps=[
        SpecStep("vertical_launch", {"launch_speed_m_s": 100.0}),
        SpecStep("terminal_velocity_drop",
                 {"material_key": "iron", "radius_m": 0.05},
                 bindings={"drop_altitude_m": (0, "no_such_field")}),
    ])
    with pytest.raises(KeyError):
        run_spec(bad)


def test_outputs_present_for_chaining():
    """Scenarios expose named outputs so they can be chained from."""
    r = terminal_velocity_drop("copper", 0.05, 10_000.0)
    assert "impact_speed_m_s" in r.outputs and r.outputs["impact_speed_m_s"] > 0


def test_launch_chain_routes_and_auto_binds():
    """'Throw up … how fast/hot when it lands' plans the 3-step chain, auto-bound."""
    from sigma_ground.materia import translate
    spec = translate("Throw a steel ball straight up at 400 m/s — how fast and "
                     "how hot is it when it lands?", use_qwen=False)
    assert [s.verb for s in spec.steps] == ["vertical_launch",
                                            "terminal_velocity_drop",
                                            "drag_heating_drop"]
    assert abs(spec.steps[0].params["launch_speed_m_s"] - 400.0) < 1e-9
    assert spec.steps[1].bindings["drop_altitude_m"] == (0, "apex_altitude_m")
    assert spec.steps[2].bindings["drop_altitude_m"] == (0, "apex_altitude_m")


def test_speed_unit_not_read_as_altitude():
    """'400 m/s' is a speed, not 400 m of altitude."""
    from sigma_ground.materia.translator import _extract_lengths
    assert _extract_lengths("thrown up at 400 m/s")["altitude_found"] is False
