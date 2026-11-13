"""Tests for the osmium atom, lattice and electron sea (osmium_atom.py)."""
import math

import pytest

from sigma_ground.field.interface import element as el
from sigma_ground.field.interface import osmium_atom as oa


def test_the_shells_hold_all_76_electrons_in_the_aufbau_groups():
    g = oa.shell_groups()
    assert sum(s["electrons"] for s in g) == 76
    assert [s["label"] for s in g][-3:] == ["5sp", "5d", "6sp"] or g[-1]["label"] == "6sp"
    assert g[-1]["electrons"] == 2 and g[-2]["electrons"] == 6                     # 6s2 and 5d6


def test_the_outermost_group_agrees_with_the_librarys_own_slater_calculation():
    outer = oa.shell_groups()[-1]
    assert outer["zeff"] == pytest.approx(el.slater_zeff(76))
    assert outer["radius_m"] == pytest.approx(el.slater_radius_m(76))


def test_the_shells_get_larger_and_less_tightly_bound_outwards():
    g = oa.shell_groups()
    radii = [s["radius_m"] for s in g]
    binding = [s["binding_ev"] for s in g]
    assert radii == sorted(radii) and binding == sorted(binding, reverse=True)
    assert g[0]["radius_m"] < 1e-12 and g[-1]["radius_m"] > 1e-10                    # from hundredths of an angstrom to several
    assert all(s["binding_ev"] > 2.5 for s in g[:-1])                                # every bound group needs more than a visible photon's energy


def test_the_nucleus_is_a_few_femtometres_a_hundred_thousandth_of_the_atom():
    assert oa.nucleus_radius_m() == pytest.approx(7.0e-15, rel=0.05)
    assert oa.shell_groups()[-1]["radius_m"] / oa.nucleus_radius_m() > 1e4


def test_the_crystal_is_hcp_with_two_free_electrons_an_atom():
    lat = oa.lattice()
    assert lat["structure"] == "hcp" and lat["free_electrons_per_atom"] == 2
    assert 2e-10 < lat["a_m"] < 3e-10
    assert lat["layer_spacing_m"] == pytest.approx(lat["a_m"] * math.sqrt(2.0 / 3.0))


def test_the_seas_density_is_two_electrons_an_atom_of_the_measured_density():
    per_atom = oa.sea()["n_e_m3"] / (22590.0 / el.atomic_mass_kg(76))
    assert per_atom == pytest.approx(2.0, rel=0.05)


def test_the_plasma_frequency_is_in_the_ultraviolet_so_visible_light_is_turned_back():
    s = oa.sea()
    assert 10 < s["plasma_energy_ev"] < 20 and 5e-9 < s["skin_depth_m"] < 5e-8      # ~14 eV, a skin of ~14 nm
    assert s["skin_depth_m"] == pytest.approx(2.998e8 / s["plasma_angular_hz"], rel=1e-3)
    green = oa.light(500e-9)
    assert green["reflected"] and green["omega_over_plasma"] < 0.3
    assert not oa.light(50e-9)["reflected"]                                          # hard ultraviolet gets in


def test_a_visible_wave_is_a_uniform_field_at_atomic_scale_and_flips_every_femtosecond():
    green = oa.light(500e-9)
    assert green["period_s"] == pytest.approx(1.67e-15, rel=0.01)
    assert green["wavelength_in_blocks"] == pytest.approx(1e4)                      # 10,000 half-angstrom blocks to one wavelength
    assert green["energy_ev"] == pytest.approx(2.48, rel=0.01)


def test_the_whole_lattice_shares_the_photons_momentum():
    n = oa.atoms_in_grain(20e-6)
    assert n == pytest.approx(2.4e15, rel=0.05)
    assert oa.recoil_per_atom(500e-9, 20e-6) == pytest.approx(1.325e-27 / n, rel=1e-6)
