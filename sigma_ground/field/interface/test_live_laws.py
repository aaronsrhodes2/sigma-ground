"""The temperature laws the live cards (block at T) stand on, fixed 2026-10-31:

- metals: Wiedemann-Franz with the resistivity AT T (it used the 300 K one, so kappa grew in proportion to T);
- liquid water: measured (NIST WebBook, IAPWS-95) instead of an uncalibrated two-state H-bond model;
- ice: measured (Engineering ToolBox) instead of the derived phonon values (21.7 W/m/K, ten times too high);
- block(material, T): measured value at 293.15 K times the law's change with temperature.
"""
import pytest

from sigma_ground.field.interface import electronics, liquid_water, thermal
from sigma_ground.field.interface.block import T_REF_K, block


@pytest.mark.parametrize("metal", ["copper", "aluminum", "tungsten", "gold", "silver"])
def test_metal_conductivity_falls_gently_as_it_heats(metal):
    """Above the Debye temperature rho ~ T, so kappa_el = L0 T / rho(T) is roughly flat, falling slightly: never ~ T."""
    k300, k1000 = thermal.thermal_conductivity(metal, 300.0), thermal.thermal_conductivity(metal, 1000.0)
    assert k1000 < k300
    assert k1000 > 0.7 * k300


def test_electronic_conductivity_is_wiedemann_franz_with_rho_at_T():
    for T in (300.0, 700.0, 1300.0):
        assert thermal.electronic_thermal_conductivity("copper", T) == pytest.approx(
            2.44e-8 * T / electronics.resistivity("copper", T), rel=2e-3)


def test_water_heat_capacity_is_measured():
    for T in (275.0, 298.15, 350.0):
        c = liquid_water.water_heat_capacity(T)
        assert 75.0 < c < 76.3                               # J/(mol K): measured water is ~75.3-76, nearly flat
    assert liquid_water.water_specific_heat_j_kg_K(298.15) == pytest.approx(4181.7, rel=2e-3)


def test_water_and_ice_cards_are_measured():
    w, i = block("water", 293.15)["properties"], block("ice", 263.15)["properties"]
    assert w["thermal_conductivity"]["value"] == pytest.approx(0.598, rel=0.01)
    assert w["specific_heat"]["value"] == pytest.approx(4184, rel=0.005)
    assert i["thermal_conductivity"]["value"] == pytest.approx(2.30, rel=0.01)
    assert i["specific_heat"]["value"] == pytest.approx(2000, rel=0.01)
    assert "NIST" in w["thermal_conductivity"]["source"] and "ice" in i["thermal_conductivity"]["source"]


def test_frozen_water_and_melted_ice_take_the_other_phase():
    assert block("water", 263.15)["properties"]["thermal_conductivity"]["value"] == pytest.approx(2.30, rel=0.01)
    assert block("ice", 293.15)["properties"]["thermal_conductivity"]["value"] == pytest.approx(0.598, rel=0.01)
    d = block("water", 293.15)["properties"]
    assert d["density_solid"]["value"] < d["density_liquid"]["value"]            # ice floats


@pytest.mark.parametrize("prop", ["thermal_conductivity", "specific_heat", "density"])
def test_block_at_the_reference_temperature_is_the_measured_value(prop):
    assert block("iron", T_REF_K)["properties"][prop]["value"] == block("iron")["properties"][prop]["value"]


def test_block_follows_temperature():
    cold, hot = block("iron", 293.15)["properties"], block("iron", 1200.0)["properties"]
    assert hot["thermal_conductivity"]["value"] < cold["thermal_conductivity"]["value"]
    assert hot["specific_heat"]["value"] > cold["specific_heat"]["value"]
    assert hot["density"]["value"] < cold["density"]["value"]                     # it expanded
    assert hot["emissivity"]["value"] > cold["emissivity"]["value"]
    assert "at 1200 K" in hot["thermal_conductivity"]["source"]


def test_a_melted_metal_uses_its_solid_laws_at_the_melting_point():
    Tm = block("lead")["properties"]["melting_point"]["value"]
    at_melt = block("lead", Tm)["properties"]["thermal_conductivity"]["value"]
    assert block("lead", 1200.0)["properties"]["thermal_conductivity"]["value"] == pytest.approx(at_melt, rel=1e-6)
