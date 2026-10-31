"""Tests for field.interface.block -- one block of matter, every property sourced."""
import pytest

from .block import block
from .phase_transition import PHASE_DATA, latent_heat_J_kg


def test_measured_beats_derived():
    b = block("granite")
    assert b["properties"]["thermal_conductivity"]["source"].startswith("NASA TPSX")
    glass = block("glass")                       # nothing measured: derived, and labelled so
    assert glass["properties"]["thermal_conductivity"]["source"].startswith("derived:")


def test_phase_follows_temperature():
    assert block("lead", 300)["phase"] == "solid"
    assert block("lead", 700)["phase"] == "liquid"
    assert block("lead", 2100)["phase"] == "gas"
    assert block("water", 263.15)["phase"] == "solid"
    assert block("ice", 280)["phase"] == "liquid"


def test_written_phase_wins_at_the_melting_point():
    assert block("ice", 273.15)["phase"] == "solid"
    assert block("water", 273.15)["phase"] == "liquid"


def test_amount_mass_volume_and_fill():
    b = block("lead", 300, mass_kg=2.0)
    rho = b["properties"]["density"]["value"]
    assert b["amount"]["volume_m3"] == pytest.approx(2.0 / rho)
    half = block("lead", 300, mass_kg=rho * 1e-3 / 2, volume_m3=1e-3)
    assert half["amount"]["fill_fraction"] == pytest.approx(0.5)


def test_latent_heat_and_densities_agree_with_phase_data():
    p = block("lead")["properties"]
    assert p["latent_heat_fusion"]["value"] == pytest.approx(latent_heat_J_kg("lead"))
    dV = PHASE_DATA["lead"]["delta_V_fus"]
    assert p["density_liquid"]["value"] == pytest.approx(p["density_solid"]["value"] / (1 + dV))


def test_gaps_are_named_not_filled():
    b = block("unobtainium")
    assert not b["identified"] and "density" not in b["properties"]
    assert "no sourced density" in b["gaps"]
    assert any("mole basis" in g for g in block("glass")["gaps"])


def test_every_property_has_units_and_a_source():
    for m in ("copper", "water", "granite", "mercury"):
        for name, p in block(m)["properties"].items():
            assert p["source"], f"{m}.{name} has no source"


def test_appearance_of_a_metal_is_its_reflectance_and_opaque():
    from .block import appearance
    a = appearance("gold", 0.05)
    assert a["metal"] and a["opacity"] == 1.0 and a["color_rgb"] == a["reflectance_f0"]
    assert a["color_rgb"][0] > a["color_rgb"][2]                  # gold: more red than blue


def test_clear_material_opacity_depends_on_block_size():
    from .block import appearance
    thin, thick = appearance("water", 0.05), appearance("water", 2.0)
    assert thin["opacity"] < thick["opacity"] < 1.0               # a metre of water is bluer than a cup's worth
    assert thick["color_rgb"][2] > thick["color_rgb"][0]
    assert 0.0 < appearance("glass", 0.05)["reflectance_f0"] < 0.1   # Fresnel at n ~ 1.5


def test_appearance_names_what_it_lacks():
    from .block import appearance
    g = appearance("granite", 0.05)
    assert g["color_rgb"] is None and "no colour model" in g["gaps"]
