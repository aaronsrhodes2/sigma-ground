"""Tests for the tool contract (mcp/contract.py + field/bounds.py): quantities, bounds, documentation, uniform errors."""
from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from sigma_ground.field.bounds import Bounds, Regime, classify, describe
from sigma_ground.mcp import contract as C

REPO = Path(__file__).resolve().parents[2]


# ── bounds (GOLDEN_RULES Rule 2) ──────────────────────────────────────────────────────────────────

def test_bounds_classification():
    b = Bounds(hard_lo=0.0, hard_hi=1.0, wall_lo=0.1, wall_hi=0.9)
    assert classify(0.5, b)[0] is Regime.SAFE
    assert classify(0.12, b)[0] is Regime.EDGE and classify(0.89, b)[0] is Regime.EDGE
    assert classify(0.05, b)[0] is Regime.WALL and classify(0.95, b)[0] is Regime.WALL
    assert classify(-0.1, b)[0] is Regime.BEYOND and classify(1.2, b)[0] is Regime.BEYOND
    assert classify(float("nan"), b)[0] is Regime.BEYOND


def test_open_and_closed_limits():
    assert classify(0.0, Bounds(hard_lo=0.0, lo_open=True))[0] is Regime.BEYOND          # a mass must be > 0
    assert classify(0.0, Bounds(hard_lo=0.0))[0] is Regime.SAFE                          # a temperature may be 0 K
    assert describe(Bounds(hard_lo=0.0, lo_open=True)) == "> 0" and describe(Bounds(hard_lo=0.0, hard_hi=1.0)) == "in [0, 1]"


# ── the spec table ────────────────────────────────────────────────────────────────────────────────

def test_every_unit_in_the_spec_table_is_a_unit_pint_knows():
    from sigma_ground.mcp.tools.units import _get_registry
    reg = _get_registry()
    for name, spec in C.SPECS.items():
        if spec.unit:
            reg.Unit(spec.unit)                          # raises on an unknown unit


def test_every_spec_example_is_accepted_by_its_own_spec():
    for name, spec in C.SPECS.items():
        if spec.unit and spec.example:
            C.to_si(spec.one_example, name, spec)        # the documented example must itself parse and convert


@pytest.mark.parametrize("text,name,expected", [
    ("54 km/h", "velocity_m_s", 15.0), ("150 cm", "height_m", 1.5), ("500 g", "mass_kg", 0.5), ("2 min", "time_s", 120.0),
    ("2.4 GHz", "frequency_hz", 2.4e9), ("1 keV", "energy_ev", 1000.0), ("1 atm", "pressure_pa", 101325.0),
    ("27 degC", "temperature_k", 300.15), ("80 degF", "temperature_k", 299.8166666), ("300 K", "temperature_k", 300.0),
    ("550 nm", "wavelength_m", 5.5e-7), ("10 uF", "capacitance_f", 1e-5), ("25 mL", "acid_volume_mL", 25.0), ("0.1 M", "concentration_mol_l", 0.1),
    ("1 mph", "speed_m_s", 0.44704), ("1 au", "distance_m", 1.495978707e11), ("0.5 c", "v_m_s", 149896229.0), ("1 solar_mass", "central_mass_kg", 1.98841e30),
    ("5", "mass_kg", 5.0), ("  7.5e2 ", "time_s", 750.0), ("-3", "t_celsius", -3.0), ("90 degree", "angle_deg", 90.0), ("1 rad", "angle_deg", 57.29577951),
])
def test_quantity_strings_convert_to_the_parameters_unit(text, name, expected):
    value, _note = C.to_si(text, name, C.SPECS[name])
    assert value == pytest.approx(expected, rel=1e-6)


def test_a_conversion_is_reported_but_a_same_unit_string_is_not():
    assert C.to_si("150 cm", "height_m", C.SPECS["height_m"])[1] == "150 cm = 1.5 m"
    assert C.to_si("2 m", "height_m", C.SPECS["height_m"])[1] is None


@pytest.mark.parametrize("text,name,code", [
    ("54 kg", "velocity_m_s", "wrong_dimension"), ("3 m", "mass_kg", "wrong_dimension"), ("abc", "mass_kg", "bad_quantity"),
    ("5 widgets", "mass_kg", "bad_quantity"), ("3 m", "emissivity", "bad_quantity"), ("", "mass_kg", "bad_quantity"),
])
def test_unusable_quantities_are_refused_with_a_code_and_what_was_expected(text, name, code):
    with pytest.raises(C.ContractError) as e:
        C.to_si(text, name, C.SPECS[name])
    assert e.value.code == code and e.value.param == name
    payload = e.value.payload()
    assert payload["message"] and "expected" in payload


# ── a tool wrapped by the contract ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def census():
    pytest.importorskip("mcp")
    spec = importlib.util.spec_from_file_location("mentat_contract_census", REPO / "tools" / "mentat_contract_census.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mentat_contract_census"] = mod
    spec.loader.exec_module(mod)
    return mod, mod.build_server()


def _call(census, name, args):
    mod, server = census
    r = mod.call(server, name, args)
    assert r["status"] == "ok", r
    return r["result"]


def test_a_quantity_string_gives_the_same_answer_as_the_number(census):
    by_number = _call(census, "kinetic_energy", {"mass_kg": 0.145, "velocity_m_s": 40.2336})
    by_string = _call(census, "kinetic_energy", {"mass_kg": "145 g", "velocity_m_s": "90 mph"})
    assert by_string["value"] == pytest.approx(by_number["value"], rel=1e-12)
    assert by_string["converted"] == {"mass_kg": "145 g = 0.145 kg", "velocity_m_s": "90 mph = 40.2336 m/s"}
    assert "converted" not in by_number


def test_relativistic_speed_as_a_fraction_of_c(census):
    r = _call(census, "lorentz_factor", {"velocity_m_s": "0.9 c"})
    assert r["value"] == pytest.approx(2.294157, rel=1e-6)


def test_a_missing_required_argument_is_a_structured_error_with_an_example(census):
    r = _call(census, "schwarzschild_radius", {})
    assert r["value"] is None and r["error"]["code"] == "missing_argument" and r["error"]["param"] == "mass_kg"
    assert "mass_kg" in json.dumps(r["error"]["example"])


def test_a_wrong_type_wrong_dimension_and_impossible_value_are_structured_errors(census):
    assert _call(census, "schwarzschild_radius", {"mass_kg": "abc"})["error"]["code"] == "bad_quantity"
    assert _call(census, "schwarzschild_radius", {"mass_kg": "54 km/h"})["error"]["code"] == "wrong_dimension"
    r = _call(census, "blackbody_peak_wavelength", {"temperature_k": "-300 degC"})
    assert r["value"] is None and r["error"]["code"] == "out_of_range" and r["error"]["param"] == "temperature_k"
    assert _call(census, "blackbody_peak_wavelength", {"temperature_k": -5})["error"]["code"] == "out_of_range"


def test_a_tool_that_declines_carries_an_error_code(census):
    r = _call(census, "density", {"material": "unobtainium"})
    assert r["value"] is None and r["error"]["code"] == "unknown_key" and "iron" in r["error"]["message"]
    assert _call(census, "density", {"material": "iron"})["error"] is None


def test_an_unknown_argument_is_reported_not_silently_dropped(census):
    r = _call(census, "schwarzschild_radius", {"mass_kg": 1.989e30, "radius": 3})
    assert r["value"] == pytest.approx(2954.13, rel=1e-4) and r["ignored_arguments"] == ["radius"]
    assert "ignored unknown argument 'radius'" in r["notes"]
    r = _call(census, "schwarzschild_radius", {"mass_kg": 1.989e30, "mass": 3})
    assert "did you mean 'mass_kg'" in r["notes"]


def test_an_unknown_tool_is_a_structured_error(census):
    r = _call(census, "no_such_tool", {})
    assert r["value"] is None and r["error"]["code"] == "invalid_arguments"


def test_the_schema_documents_every_parameter_and_accepts_quantity_strings(census):
    mod, server = census
    tools = {t.name: t for t in asyncio.run(server.list_tools())}
    t = tools["kinetic_energy"]
    props = t.inputSchema["properties"]
    assert "string" in json.dumps(props["velocity_m_s"]) and "number" in json.dumps(props["velocity_m_s"])
    assert "m/s" in props["velocity_m_s"]["description"] and "e.g." in props["velocity_m_s"]["description"]
    assert "Args:" in t.description and "Example: kinetic_energy(" in t.description and "Units:" in t.description
    assert t.inputSchema["required"] == ["mass_kg", "velocity_m_s"]


def test_the_contract_is_transparent_for_a_plain_call(census):
    """Numbers in, numbers out: the same value as the function called directly."""
    from sigma_ground.mcp.tools import gr
    direct = gr.schwarzschild_radius(1.989e30).to_dict()
    via = _call(census, "schwarzschild_radius", {"mass_kg": 1.989e30})
    for k in ("value", "units", "source", "formula"):
        assert via[k] == direct[k]
    assert via["error"] is None and via["assumed"] is None


def test_errors_are_uniform_over_the_real_stdio_protocol_too():
    """The protocol handler is bound when the server is built; install() must rebind it, or only in-process callers get the contract."""
    pytest.importorskip("mcp")
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def run():
        params = StdioServerParameters(command=sys.executable, args=["-m", "sigma_ground.mcp.server"])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                missing = await session.call_tool("schwarzschild_radius", {})
                quantity = await session.call_tool("kinetic_energy", {"mass_kg": "145 g", "velocity_m_s": "90 mph"})
                return missing, quantity

    missing, quantity = asyncio.run(run())
    assert not missing.isError, "a bad call must come back as a normal result with an `error`, not a protocol error"
    err = json.loads(missing.content[0].text)
    assert err["value"] is None and err["error"]["code"] == "missing_argument"
    assert not quantity.isError
    assert json.loads(quantity.content[0].text)["converted"]["mass_kg"] == "145 g = 0.145 kg"
