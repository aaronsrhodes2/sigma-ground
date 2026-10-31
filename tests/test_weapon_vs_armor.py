"""Weapon vs. Armor (games/weapon_vs_armor): it loads, every preset solves, and the laser is an energy balance."""
from pathlib import Path

import pytest

from games.weapon_vs_armor.physics import laser
from games.weapon_vs_armor.physics.solver import solve
from games.weapon_vs_armor.presets import ARMOR_PRESETS, WEAPON_PRESETS, get_armor, get_weapon
from sigma_ground.field.interface.block import block


def test_every_weapon_solves_against_every_armour():
    for w in WEAPON_PRESETS:
        for a in ARMOR_PRESETS:
            r = solve(get_weapon(w), get_armor(a), "atmosphere_earth", 100)
            assert r.layers and r.total_layers == sum(1 for L in get_armor(a) if not L.get("gap"))


def test_no_speculative_constant_in_the_game():
    # SIGMA_CONV (the SSBM conversion threshold) moved to ssbm-theory: importing it made the whole game fail to load
    game = Path(__file__).resolve().parents[1] / "games" / "weapon_vs_armor"
    for f in game.rglob("*.py"):
        assert not any("import" in l and "SIGMA_CONV" in l for l in f.read_text(encoding="utf-8").splitlines()), f


def test_a_laser_crater_sits_at_its_front_temperature_not_millions_of_kelvin():
    for w in ("hel_100kw", "hel_megawatt", "gamma_laser"):
        for a in ("medieval_plate", "nij_iv", "tank_composite"):
            for L in solve(get_weapon(w), get_armor(a), "atmosphere_earth", 100).layers:
                if L.material != "gap" and L.damage_depth_m > 0:
                    p = block(L.material)["properties"]
                    assert L.temperature_k <= p.get("boiling_point", p["melting_point"])["value"] + 1


def test_metals_reflect_most_of_an_infrared_laser():
    al, _ = laser.absorptivity("aluminum", 1064)
    fe, how = laser.absorptivity("steel_mild", 1064)
    assert al < 0.2 < fe < 0.6 and "iron" in how
    assert laser.absorptivity("kevlar", 1064) == (1.0, "no measured reflectance: all absorbed (gap)")


def test_burn_through_depth_is_energy_over_spot_area_times_melt_enthalpy():
    H, T, how = laser.ablation_front("steel_mild", 3.0)
    assert "melted" in how and T == block("steel_mild")["properties"]["melting_point"]["value"]
    w = {"type": "laser", "power_w": 1e5, "wavelength_nm": 1064, "pulse_duration_s": 3.0, "spot_radius_m": 0.008}
    r = laser.interact(w, "steel_mild", 1.0)
    A, _ = laser.absorptivity("steel_mild", 1064)
    import math
    assert r["damage_depth_m"] == pytest.approx(1e5 * 3.0 * A / (math.pi * 0.008 ** 2 * H), rel=1e-6)
