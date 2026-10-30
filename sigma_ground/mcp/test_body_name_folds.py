"""escape_velocity and the GR mass tools take a body's name (mcp/tools/body_lookup.py; misc/FOLDED_CHAINS.md): one call where the router planned
solar_system_body > escape_velocity, or typed the mass in from memory."""
from __future__ import annotations

import math

import pytest

from sigma_ground.field.constants import C
from sigma_ground.field.interface import body_data
from sigma_ground.mcp.contract import ContractError
from sigma_ground.mcp.tools import gr, kinematics
from sigma_ground.mcp.tools.body_lookup import with_body


def _gm_r(name):
    d = body_data.describe(body_data.resolve(name)[0])
    return d["gm_km3_s2"] * 1e9, d["radius_km"] * 1e3


@pytest.mark.parametrize("name", ["earth", "mars", "sun", "jupiter", "moon", "io", "ceres"])
def test_escape_velocity_from_a_named_body_is_sqrt_2gm_over_r_with_the_catalogue_values(name):
    gm, r = _gm_r(name)
    res = with_body(kinematics.escape_velocity_classical, name, None, needs_radius=True)
    assert res.value == pytest.approx(math.sqrt(2 * gm / r), rel=1e-12)
    assert res.provenance_tag == "DERIVED" and res.units == "m/s"
    assert body_data.describe(body_data.resolve(name)[0])["sources"]["gm_km3_s2"] in res.source
    assert res.inputs["radius_m"] == pytest.approx(r)


def test_altitude_and_an_explicit_radius():
    gm, r = _gm_r("earth")
    up = with_body(kinematics.escape_velocity_classical, "earth", None, altitude_km=400, needs_radius=True)
    assert up.value == pytest.approx(math.sqrt(2 * gm / (r + 4e5)), rel=1e-12) and up.inputs["altitude_km"] == 400
    at = with_body(kinematics.escape_velocity_classical, "earth", None, radius_m=4.2164e7, needs_radius=True)
    assert at.value == pytest.approx(math.sqrt(2 * gm / 4.2164e7), rel=1e-12)


def test_a_mass_typed_from_memory_loses_to_the_catalogue_and_says_so():
    gm, _ = _gm_r("jupiter")
    res = with_body(gr.schwarzschild_radius, "jupiter", 1.9e27)
    assert res.value == pytest.approx(2 * gm / C**2, rel=1e-12)
    assert "was not used" in res.notes


def test_without_a_body_the_tools_are_unchanged():
    assert with_body(gr.schwarzschild_radius, None, 2e30).value == gr.schwarzschild_radius(2e30).value
    assert with_body(kinematics.escape_velocity_classical, None, 6e24, radius_m=6.4e6, needs_radius=True).value == \
        kinematics.escape_velocity_classical(6e24, 6.4e6).value
    with pytest.raises(ContractError) as e:
        with_body(gr.schwarzschild_radius, None, None)
    assert e.value.code == "missing_argument" and e.value.param == "mass_kg"
    with pytest.raises(ContractError) as e:                                                                  # no radius, no body: refused, not guessed
        with_body(kinematics.escape_velocity_classical, None, 6e24, needs_radius=True)
    assert e.value.param == "radius_m"


def test_an_unknown_body_is_refused_with_close_names():
    res = with_body(gr.schwarzschild_radius, "Marz", None)
    assert res.value is None and "not in the body catalogue" in res.notes


def test_through_the_server(tmp_path):
    pytest.importorskip("mcp")
    import importlib.util
    import sys
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("mentat_contract_census", Path(__file__).resolve().parents[2] / "tools" / "mentat_contract_census.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mentat_contract_census"] = mod
    spec.loader.exec_module(mod)
    srv = mod.build_server()
    gm, r = _gm_r("mars")
    out = mod.call(srv, "escape_velocity", {"body_name": "Red Planet"})["result"]
    assert out["value"] == pytest.approx(math.sqrt(2 * gm / r), rel=1e-12)
    for tool in ("schwarzschild_radius", "isco_radius", "photon_sphere_radius", "hawking_temperature", "hawking_evaporation_time"):
        assert mod.call(srv, tool, {"body_name": "sun"})["result"]["value"] > 0, tool
        assert mod.call(srv, tool, {"mass_kg": 2e30})["result"]["value"] > 0, tool
    for tool in ("gravitational_time_dilation", "gravitational_redshift", "escape_velocity"):
        assert mod.call(srv, tool, {"body_name": "earth", "altitude_km": 20200})["result"]["value"] > 0, tool
