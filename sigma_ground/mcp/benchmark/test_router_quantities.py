"""Tests for v4 labels: the router quotes a quantity as the question wrote it, parameters under canonical names (2026-10-15)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from sigma_ground.mcp import contract as C
from sigma_ground.mcp.benchmark import router_plan as RP
from sigma_ground.mcp.benchmark import router_quantities as Q

pytest.importorskip("pint")
REPO = Path(__file__).resolve().parents[3]


# ── finding the quantity in the question ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("question,value,unit,expected", [
    ("What is the redshift for mass of 1.05e29 kg and radius of 5.09 km?", 5090.0, "m", "5.09 km"),
    ("What is the redshift for mass of 1.05e29 kg and radius of 5.09 km?", 1.05e29, "kg", "1.05e29 kg"),
    ("A car at 60 miles per hour hits a wall", 26.8224, "m/s", "60 miles per hour"),
    ("A rod 1,500 m long", 1500.0, "m", "1500 m"),                                                      # thousands comma removed, the unit as written
    ("Water at 27 °C cools", 300.15, "K", "27 °C"),                                                    # an offset unit
    ("temperature of -40 degC", 233.15, "K", "-40 degC"),
    ("a ball of mass 145 g thrown at 90 mph", 0.145, "kg", "145 g"),
    ("a ball of mass 145 g thrown at 90 mph", 40.2336, "m/s", "90 mph"),
    ("radius is 6.41km", 6410.0, "m", "6.41 km"),                                                       # no space in the question
    ("a wavelength of 550 nm.", 5.5e-7, "m", "550 nm"),                                                 # trailing full stop
    ("energy of 2.5 eV?", 2.5 * 1.602176634e-19, "J", "2.5 eV"),
    ("5 kg and 7 kg", 7.0, "kg", "7 kg"),                                                               # the right one of two
])
def test_the_quantity_whose_value_equals_the_label_is_found(question, value, unit, expected):
    assert Q.find_quantity(question, value, unit) == expected


def test_a_two_word_unit_is_found_and_the_server_reads_it():
    q = Q.find_quantity("Find the Schwarzschild radius for mass of 7.35 solar masses.", 7.35 * C._registry().Quantity(1, "solar_mass").to("kg").magnitude, "kg")
    assert q == "7.35 solar masses"
    assert C.to_si(q, "mass_kg", C.SPECS["mass_kg"])[0] == pytest.approx(7.35 * C._registry().Quantity(1, "solar_mass").to("kg").magnitude)


def test_a_bare_number_or_a_value_the_question_does_not_state_is_left_alone():
    assert Q.find_quantity("a mass of 5", 5.0, "kg") is None
    assert Q.find_quantity("a mass of 5 kg", 6.0, "kg") is None                                         # derived, not stated
    assert Q.find_quantity("no numbers here", 1.0, "m") is None


# ── relabelling a plan ─────────────────────────────────────────────────────────────────────────────

SIGS = {
    "gravitational_redshift": [{"name": "mass_kg", "ann": "float", "required": True}, {"name": "radius_m", "ann": "float", "required": True}],
    "elastic_analysis": [{"name": "material_key", "ann": "str", "required": False}, {"name": "strain", "ann": "float", "required": False}],
    "kinetic_energy": [{"name": "mass_kg", "ann": "float", "required": True}, {"name": "velocity_m_s", "ann": "float", "required": True}],
    "solar_system_body": [{"name": "name", "ann": "str", "required": True}],
}


def test_a_single_call_is_quoted_and_its_other_arguments_untouched():
    plan = {"tool": "gravitational_redshift", "tool_args": {"mass_kg": 1.05e29, "radius_m": 5090.0}}
    new, n = Q.relabel_plan("Redshift for mass of 1.05e29 kg and radius of 5.09 km?", plan, SIGS)
    assert new == {"tool": "gravitational_redshift", "tool_args": {"mass_kg": "1.05e29 kg", "radius_m": "5.09 km"}} and n == 2
    assert plan["tool_args"]["radius_m"] == 5090.0                                                      # the input is not mutated


def test_parameters_get_their_canonical_names_and_non_quantities_stay():
    new, n = Q.relabel_plan("elastic response of steel at 0.001 strain", {"tool": "elastic_analysis", "tool_args": {"material_key": "steel", "strain": 0.001}}, SIGS)
    assert new["tool_args"] == {"material": "steel", "strain": 0.001} and n == 0                       # strain has no unit; the name is canonical


def test_references_and_clarifications_and_declines_are_not_touched():
    steps = {"steps": [{"tool": "solar_system_body", "tool_args": {"name": "Mars"}},
                       {"tool": "kinetic_energy", "tool_args": {"mass_kg": "$1.mass_kg", "velocity_m_s": 25.0}}]}
    new, n = Q.relabel_plan("kinetic energy of Mars at 25 m/s", steps, SIGS)
    assert new["steps"][1]["tool_args"] == {"mass_kg": "$1.mass_kg", "velocity_m_s": "25 m/s"} and n == 1
    clar = {"tool": "request_clarification", "tool_args": {"variable": "mass_kg", "question": "How heavy?"}}
    assert Q.relabel_plan("how fast?", clar, SIGS) == (clar, 0)
    assert Q.relabel_plan("tell me a joke", {"decline": "not physics"}, SIGS) == ({"decline": "not physics"}, 0)


def test_an_unknown_tool_is_left_alone():
    plan = {"tool": "no_such_tool", "tool_args": {"mass_kg": 5.0}}
    assert Q.relabel_plan("5 kg", plan, SIGS) == (plan, 0)


# ── the v4 prompt and the builder ──────────────────────────────────────────────────────────────────

def test_v4_replaces_the_conversion_instruction_and_nothing_else():
    v3, v4 = RP.system_prompt("v3"), RP.system_prompt("v4")
    assert "Convert every input" in v3 and "Convert every input" not in v4
    assert "exactly as the question gives it" in v4 and "the tool converts" in v4
    assert v4.replace(RP._V4_UNIT_SENTENCE, RP._V3_UNIT_SENTENCE) == v3                                 # the only difference
    assert "v4" in RP.MULTISTEP and RP.system_prompt("v3") == RP.ROUTER_SYSTEM_PROMPT_V3               # v3 is unchanged


@pytest.fixture(scope="module")
def builder():
    sys.path.insert(0, str(REPO / "tools"))
    spec = importlib.util.spec_from_file_location("build_router_v2_for_test", REPO / "tools" / "build_router_v2.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_router_v2_for_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_the_builder_writes_v4_records_with_quoted_quantities(builder):
    old = builder.PLAN
    try:
        builder.PLAN = "v4"
        rec = builder.record("Calculate the gravitational redshift for mass of 4.42e27 kg and radius of 640 km.",
                             {"tool": "gravitational_redshift", "tool_args": {"mass_kg": 4.42e27, "radius_m": 640000.0}}, {"group": "g"})
        assert rec["messages"][0]["content"] == RP.system_prompt("v4")
        assert json.loads(rec["messages"][2]["content"]) == {"tool": "gravitational_redshift", "tool_args": {"mass_kg": "4.42e27 kg", "radius_m": "640 km"}}
        assert rec["meta"]["quoted"] == 2
        builder.validate(rec)
        builder.PLAN = "v3"
        v3 = builder.record("Calculate the gravitational redshift for mass of 4.42e27 kg and radius of 640 km.",
                            {"tool": "gravitational_redshift", "tool_args": {"mass_kg": 4.42e27, "radius_m": 640000.0}}, {"group": "g"})
        assert json.loads(v3["messages"][2]["content"])["tool_args"]["radius_m"] == 640000.0 and "quoted" not in v3["meta"]       # v3 builds are unchanged
    finally:
        builder.PLAN = old


def test_the_builder_refuses_an_old_parameter_name_in_a_v4_label(builder):
    old = builder.PLAN
    try:
        builder.PLAN = "v4"
        rec = {"messages": [{"role": "system", "content": RP.system_prompt("v4")}, {"role": "user", "content": "elastic response of steel"},
                            {"role": "assistant", "content": json.dumps({"tool": "elastic_analysis", "tool_args": {"material_key": "steel"}})}], "meta": {}}
        with pytest.raises(AssertionError, match="canonical parameter names"):
            builder.validate(rec)
    finally:
        builder.PLAN = old


# ── the contract learns two-word units ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("written,unit", [("solar masses", "solar_mass"), ("Solar Mass", "solar_mass"), ("light years", "light_year"),
                                          ("earth radii", "earth_radius"), ("astronomical units", "astronomical_unit"), ("nautical miles", "nautical_mile"),
                                          ("km/h", "km/h"), ("°C", "degC")])
def test_two_word_units_are_joined_for_pint(written, unit):
    assert C.normalize_unit_text(written) == unit
