"""Tests for the plan comparison in tools/eval_router_val.py with v4 labels: quantities compare as quantities, old parameter names align, nothing else loosens."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))
_spec = importlib.util.spec_from_file_location("eval_router_val", REPO / "tools" / "eval_router_val.py")
V = importlib.util.module_from_spec(_spec)
sys.modules["eval_router_val"] = V
_spec.loader.exec_module(V)

pytest.importorskip("pint")


def plan(tool, **args):
    return {"tool": tool, "tool_args": args}


@pytest.mark.parametrize("pred,gold", [
    ("5.09 km", "5.09 km"), ("5.09km", "5.09 km"), (5090.0, "5.09 km"), ("5090 m", "5.09 km"), ("5.09 km", 5090.0), ("5.09 kilometers", "5.09 km"),
])
def test_equal_quantities_are_equal_however_they_are_written(pred, gold):
    assert V.compare(plan("gravitational_redshift", mass_kg=1.0, radius_m=pred), plan("gravitational_redshift", mass_kg=1.0, radius_m=gold)) == (True, True, "")


@pytest.mark.parametrize("pred,gold", [("5.09 m", "5.09 km"), ("5.09 kg", "5.09 km"), (509.0, "5.09 km"), ("five km", "5.09 km"), ("5.09 km", "5.1 km"),
                                       ("5090", "5.09 km"), ("5090.0", 5090.0), (5090.0, "5090.0")])           # a bare numeric string is not a quantity: still wrong, as the older tests demand
def test_different_quantities_are_still_different(pred, gold):
    same_tools, exact, why = V.compare(plan("gravitational_redshift", mass_kg=1.0, radius_m=pred), plan("gravitational_redshift", mass_kg=1.0, radius_m=gold))
    assert same_tools and not exact and "radius_m" in why


def test_a_number_that_is_not_unit_bearing_is_still_compared_exactly_as_before():
    assert V.compare(plan("elastic_analysis", material="steel", strain=0.001), plan("elastic_analysis", material="steel", strain=0.001))[1]
    assert not V.compare(plan("elastic_analysis", material="steel", strain=0.002), plan("elastic_analysis", material="steel", strain=0.001))[1]
    assert not V.compare(plan("elastic_analysis", material="steel", strain="0.001 m"), plan("elastic_analysis", material="steel", strain=0.001))[1]      # strain has no unit


def test_a_reference_to_an_earlier_result_must_be_identical():
    steps = lambda ref: {"steps": [{"tool": "solar_system_body", "tool_args": {"name": "Mars"}}, {"tool": "gravitational_redshift", "tool_args": {"mass_kg": ref, "radius_m": "5 km"}}]}
    assert V.compare(steps("$1.mass_kg"), steps("$1.mass_kg"))[1]
    assert not V.compare(steps("$1.radius_m"), steps("$1.mass_kg"))[1]


def test_old_parameter_names_in_a_prediction_align_with_the_golds_canonical_ones():
    gold = plan("elastic_analysis", material="steel", strain=0.001)
    assert V.compare(plan("elastic_analysis", material_key="steel", strain=0.001), gold) == (True, True, "")
    assert V.compare(plan("thermal_contact_analysis", material_1="copper", material_2="aluminum"), plan("thermal_contact_analysis", material_a="copper", material_b="aluminum"))[1]
    assert not V.compare(plan("elastic_analysis", material_key="copper", strain=0.001), gold)[1]            # aligned names, still a different value
    assert not V.compare(plan("elastic_analysis", bogus="steel", strain=0.001), gold)[1]                     # an unknown name is not aligned to anything


def test_a_clarification_plan_still_compares_by_the_variable_asked_for():
    ask = lambda v, text: plan("request_clarification", variable=v, question=text)
    assert V.compare(ask("mass_kg", "How heavy is it?"), ask("mass_kg", "What mass?"))[1]
    assert not V.compare(ask("radius_m", "How heavy is it?"), ask("mass_kg", "What mass?"))[1]


def test_the_smoke_test_reads_the_mass_whether_it_is_a_number_or_a_quantity():
    assert V.mass_in_kg(3.977e30) == 3.977e30 and V.mass_in_kg(5) == 5.0
    assert V.mass_in_kg("3.977e30 kg") == pytest.approx(3.977e30)
    assert V.mass_in_kg("2 solar masses") == pytest.approx(2 * 1.98847e30, rel=1e-3)          # what the model wrote after 50 steps of v4 training
    for bad in (None, True, "two solar masses", "5 m", [1], {"a": 1}):
        assert V.mass_in_kg(bad) is None
