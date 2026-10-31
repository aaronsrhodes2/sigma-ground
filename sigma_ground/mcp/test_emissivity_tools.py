"""The emissivity and thermal_glow tools: Mentat answers "how well does X radiate" and "what colour does X glow"."""
import pytest

from sigma_ground.field.interface import thermal
from sigma_ground.mcp.tools import materials as m


@pytest.mark.parametrize("name", ["alumina", "aluminium oxide", "aluminum oxide", "Al2O3"])
def test_alumina_by_any_name(name):
    r = m.emissivity(name)
    assert r.value == pytest.approx(0.80)
    assert "Engineering ToolBox" in r.source


def test_metal_total_emissivity_rises_with_temperature():
    cold, hot = m.emissivity("iron", 300.0).value, m.emissivity("iron", 1500.0).value
    assert hot > cold > 0
    assert hot == pytest.approx(thermal.emissivity("iron", 1500.0))
    assert "Hagen-Rubens" in m.emissivity("iron", 1500.0).source


def test_spectral_emissivity_of_a_measured_metal():
    r = m.emissivity("tungsten", 2000.0, wavelength_m=550e-9)
    assert 0 < r.value < 1
    assert "n+ik" in r.source


def test_spectral_without_nk_falls_back_to_total_and_says_so():
    r = m.emissivity("rubber", 300.0, wavelength_m=550e-9)
    assert r.value == pytest.approx(0.91)
    assert "no measured n+ik" in r.notes


def test_no_basis_is_none_never_a_default():
    r = m.emissivity("kevlar")
    assert r.value is None
    assert "no emissivity basis" in r.notes


def test_glow_red_hot_iron():
    g = m.thermal_glow("iron", 1300.0).value
    assert g["visible"]
    r, gr, b = g["rgb"]
    assert r == 1.0 and gr < 0.5 and b < gr          # red, not white
    assert g["radiated_W_m2"] == pytest.approx(thermal.thermal_emission_power("iron", 1300.0))


def test_glow_below_draper_point_is_invisible():
    assert not m.thermal_glow("iron", 500.0).value["visible"]


def test_glow_without_total_emissivity_names_the_gap():
    g = m.thermal_glow("kevlar", 400.0).value
    assert g["radiated_W_m2"] is None
    assert any("radiated power unknown" in x for x in g["gaps"])
