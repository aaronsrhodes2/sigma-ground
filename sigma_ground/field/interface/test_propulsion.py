"""
Tests for propulsion.py -- Tsiolkovsky rocket equation, Isp/exhaust-
velocity convention, mass-flow rate, propellant-mass inversion, the
curated engine catalog, and the two AppliedForce adapters.

Plain pytest functions (not unittest.TestCase), matching
sigma_ground/dynamics/test_continuum_eos.py / tests/test_applied_force.py
style within this repo's own test suite.
"""
import math

import pytest

from sigma_ground.dynamics.mechanisms.applied import AppliedForce
from sigma_ground.field.interface.propulsion import (
    STANDARD_GRAVITY_M_S2,
    PROPULSION_SYSTEMS,
    PropulsionSystem,
    exhaust_velocity_from_isp,
    impulse_burn_applied_force,
    mass_flow_rate,
    propellant_mass_for_delta_v,
    sustained_burn_applied_force,
    tsiolkovsky_delta_v,
)


# ═══════════════════════════════════════════════════════════════════════════
# § 1. exhaust_velocity_from_isp
# ═══════════════════════════════════════════════════════════════════════════

def test_exhaust_velocity_from_isp_hand_check():
    # v_e = Isp * g0; a hand-checkable round number
    assert exhaust_velocity_from_isp(100.0) == pytest.approx(980.665)
    assert exhaust_velocity_from_isp(300.0) == pytest.approx(2941.995)


def test_exhaust_velocity_from_isp_default_g0_is_standard_gravity():
    assert STANDARD_GRAVITY_M_S2 == pytest.approx(9.80665)
    assert exhaust_velocity_from_isp(1.0) == pytest.approx(9.80665)


def test_exhaust_velocity_from_isp_custom_g0():
    assert exhaust_velocity_from_isp(300.0, g0=10.0) == pytest.approx(3000.0)


def test_exhaust_velocity_from_isp_rejects_nonpositive():
    with pytest.raises(ValueError):
        exhaust_velocity_from_isp(0.0)
    with pytest.raises(ValueError):
        exhaust_velocity_from_isp(-50.0)
    with pytest.raises(ValueError):
        exhaust_velocity_from_isp(300.0, g0=0.0)


def test_chemical_isp_gives_exhaust_velocity_in_the_expected_range():
    # Real chemical engines: vacuum Isp ~ 250-460s -> v_e roughly 3000-4500 m/s
    rs25 = PROPULSION_SYSTEMS["rs25"]
    ve = exhaust_velocity_from_isp(rs25.isp_s)
    assert 3000.0 < ve < 4500.0
    assert ve == pytest.approx(rs25.exhaust_velocity_m_s)


def test_ion_isp_gives_exhaust_velocity_an_order_of_magnitude_higher():
    chemical_ve = exhaust_velocity_from_isp(PROPULSION_SYSTEMS["rs25"].isp_s)
    ion_ve = exhaust_velocity_from_isp(PROPULSION_SYSTEMS["nstar"].isp_s)
    # ~4400 m/s (chemical) vs ~30400 m/s (ion) -- comfortably an order of
    # magnitude apart, not asserting an exact ratio
    assert ion_ve / chemical_ve > 5.0


# ═══════════════════════════════════════════════════════════════════════════
# § 2. tsiolkovsky_delta_v
# ═══════════════════════════════════════════════════════════════════════════

def test_tsiolkovsky_delta_v_hand_check():
    # v_e = 1000 m/s, mass ratio e (2.71828...) -> delta_v = v_e * ln(e) = v_e
    m0 = 1000.0
    v_e = 1000.0
    m1 = m0 / math.e
    dv = tsiolkovsky_delta_v(v_e, m0, m1)
    assert dv == pytest.approx(v_e, rel=1e-9)


def test_tsiolkovsky_delta_v_zero_when_no_mass_is_lost():
    dv = tsiolkovsky_delta_v(3000.0, 500.0, 500.0)
    assert dv == pytest.approx(0.0, abs=1e-9)


def test_tsiolkovsky_delta_v_scales_with_exhaust_velocity():
    dv1 = tsiolkovsky_delta_v(2000.0, 1000.0, 400.0)
    dv2 = tsiolkovsky_delta_v(4000.0, 1000.0, 400.0)
    assert dv2 == pytest.approx(2.0 * dv1)


def test_tsiolkovsky_delta_v_rejects_bad_inputs():
    with pytest.raises(ValueError):
        tsiolkovsky_delta_v(0.0, 1000.0, 400.0)
    with pytest.raises(ValueError):
        tsiolkovsky_delta_v(3000.0, -1.0, 400.0)
    with pytest.raises(ValueError):
        tsiolkovsky_delta_v(3000.0, 1000.0, 0.0)
    with pytest.raises(ValueError):
        # final mass exceeding initial mass -- a rocket gaining mass
        tsiolkovsky_delta_v(3000.0, 400.0, 1000.0)


# ═══════════════════════════════════════════════════════════════════════════
# § 3. mass_flow_rate
# ═══════════════════════════════════════════════════════════════════════════

def test_mass_flow_rate_hand_check():
    # F = v_e * mdot -> mdot = F / v_e
    assert mass_flow_rate(10000.0, 2000.0) == pytest.approx(5.0)


def test_mass_flow_rate_rejects_nonpositive():
    with pytest.raises(ValueError):
        mass_flow_rate(0.0, 2000.0)
    with pytest.raises(ValueError):
        mass_flow_rate(1000.0, 0.0)
    with pytest.raises(ValueError):
        mass_flow_rate(-5.0, 2000.0)


# ═══════════════════════════════════════════════════════════════════════════
# § 4. propellant_mass_for_delta_v -- the inverse Tsiolkovsky relation
# ═══════════════════════════════════════════════════════════════════════════

def test_propellant_mass_for_delta_v_round_trips_against_tsiolkovsky():
    v_e = 3000.0
    m0 = 5000.0
    m1 = 3200.0
    dv = tsiolkovsky_delta_v(v_e, m0, m1)
    propellant = propellant_mass_for_delta_v(dv, v_e, m0)
    assert propellant == pytest.approx(m0 - m1, rel=1e-9)


def test_propellant_mass_for_delta_v_zero_delta_v_needs_no_propellant():
    assert propellant_mass_for_delta_v(0.0, 3000.0, 1000.0) == pytest.approx(0.0)


def test_propellant_mass_for_delta_v_cannot_exceed_initial_mass():
    # an enormous delta-v demand asymptotes to consuming the whole rocket,
    # never more
    propellant = propellant_mass_for_delta_v(1.0e9, 3000.0, 1000.0)
    assert propellant <= 1000.0
    assert propellant == pytest.approx(1000.0, rel=1e-6)


def test_propellant_mass_for_delta_v_rejects_bad_inputs():
    with pytest.raises(ValueError):
        propellant_mass_for_delta_v(-1.0, 3000.0, 1000.0)
    with pytest.raises(ValueError):
        propellant_mass_for_delta_v(100.0, 0.0, 1000.0)
    with pytest.raises(ValueError):
        propellant_mass_for_delta_v(100.0, 3000.0, 0.0)


# ═══════════════════════════════════════════════════════════════════════════
# § 5. The curated propulsion-system catalog
# ═══════════════════════════════════════════════════════════════════════════

REQUIRED_CATEGORIES = {
    "chemical_bipropellant", "chemical_solid", "electric_ion", "nuclear_thermal",
}


def test_catalog_covers_the_required_minimum_categories():
    have = {sys_.category for sys_ in PROPULSION_SYSTEMS.values()}
    assert REQUIRED_CATEGORIES <= have


def test_catalog_entries_are_all_facts_with_sources():
    # every entry is a specific tested/flown engine -- none invented
    for key, sys_ in PROPULSION_SYSTEMS.items():
        assert sys_.provenance == "fact", key
        assert sys_.source, f"{key} missing a source citation"
        assert sys_.isp_s > 0, key


def test_catalog_exhaust_velocity_matches_module_function():
    for sys_ in PROPULSION_SYSTEMS.values():
        assert sys_.exhaust_velocity_m_s == pytest.approx(
            exhaust_velocity_from_isp(sys_.isp_s))


def test_catalog_nuclear_thermal_beats_chemical_but_trails_electric():
    chemical = exhaust_velocity_from_isp(PROPULSION_SYSTEMS["rs25"].isp_s)
    nuclear = exhaust_velocity_from_isp(
        PROPULSION_SYSTEMS["nerva_pewee1"].isp_s)
    electric = exhaust_velocity_from_isp(PROPULSION_SYSTEMS["next_c"].isp_s)
    assert chemical < nuclear < electric


def test_propulsion_system_rejects_bad_provenance():
    with pytest.raises(ValueError):
        PropulsionSystem(category="chemical_bipropellant", name="x",
                         isp_s=300.0, provenance="guess", source="none")


def test_propulsion_system_rejects_nonpositive_isp():
    with pytest.raises(ValueError):
        PropulsionSystem(category="chemical_bipropellant", name="x",
                         isp_s=0.0, provenance="fact", source="none")


def test_propulsion_system_cite_includes_name_and_source():
    rs25 = PROPULSION_SYSTEMS["rs25"]
    cite = rs25.cite()
    assert "RS-25" in cite
    assert rs25.source in cite


# ═══════════════════════════════════════════════════════════════════════════
# § 6. impulse_burn_applied_force -- the closed-taxonomy adapter
# ═══════════════════════════════════════════════════════════════════════════

def test_impulse_burn_produces_j_equals_m_delta_v():
    af = impulse_burn_applied_force(
        "probe", mass_before_kg=500.0, delta_v_m_s=200.0,
        direction_unit_vector=[1.0, 0.0, 0.0],
        exhaust_velocity_m_s=3000.0)
    assert af.kind == "impulse"
    assert af.target == "probe"
    assert af.vector == pytest.approx([500.0 * 200.0, 0.0, 0.0])
    assert af.duration == 0.0


def test_impulse_burn_normalizes_a_nonunit_direction():
    af = impulse_burn_applied_force(
        "probe", mass_before_kg=100.0, delta_v_m_s=50.0,
        direction_unit_vector=[3.0, 4.0, 0.0],   # magnitude 5, not unit
        exhaust_velocity_m_s=3000.0)
    j = 100.0 * 50.0
    assert af.vector == pytest.approx([j * 0.6, j * 0.8, 0.0])


def test_impulse_burn_discloses_propellant_consumed():
    mass_before = 1000.0
    dv = 300.0
    v_e = 3000.0
    af = impulse_burn_applied_force(
        "probe", mass_before_kg=mass_before, delta_v_m_s=dv,
        direction_unit_vector=[0.0, 1.0, 0.0], exhaust_velocity_m_s=v_e)
    expected = propellant_mass_for_delta_v(dv, v_e, mass_before)
    assert af.variables["propellant_mass_consumed_kg"] == pytest.approx(
        expected, rel=1e-6)
    assert af.variables["exhaust_velocity_m_s"] == pytest.approx(v_e)
    assert af.variables["delta_v_m_s"] == pytest.approx(dv)
    # propulsion is never a free force -- some propellant is always spent
    assert af.variables["propellant_mass_consumed_kg"] > 0.0


def test_impulse_burn_rejects_nonpositive_mass():
    with pytest.raises(ValueError):
        impulse_burn_applied_force(
            "probe", mass_before_kg=0.0, delta_v_m_s=100.0,
            direction_unit_vector=[1.0, 0.0, 0.0],
            exhaust_velocity_m_s=3000.0)


def test_impulse_burn_rejects_zero_direction():
    with pytest.raises(ValueError):
        impulse_burn_applied_force(
            "probe", mass_before_kg=100.0, delta_v_m_s=100.0,
            direction_unit_vector=[0.0, 0.0, 0.0],
            exhaust_velocity_m_s=3000.0)


def test_impulse_burn_round_trips_through_to_plug_and_from_dict():
    af = impulse_burn_applied_force(
        "probe", mass_before_kg=800.0, delta_v_m_s=150.0,
        direction_unit_vector=[0.0, 0.0, 1.0],
        exhaust_velocity_m_s=3100.0)
    plug = af.to_plug()
    assert plug["kind"] == "impulse"
    assert plug["target"] == "probe"
    assert plug["adjustable"] is True
    assert plug["variables"]["propellant_mass_consumed_kg"] == pytest.approx(
        af.variables["propellant_mass_consumed_kg"])

    back = AppliedForce.from_dict(plug)
    assert back.kind == af.kind
    assert back.target == af.target
    assert back.vector == pytest.approx(af.vector)
    assert back.variables["propellant_mass_consumed_kg"] == pytest.approx(
        af.variables["propellant_mass_consumed_kg"])


# ═══════════════════════════════════════════════════════════════════════════
# § 7. sustained_burn_applied_force -- the closed-taxonomy adapter
# ═══════════════════════════════════════════════════════════════════════════

def test_sustained_burn_produces_thrust_vector_and_duration():
    af = sustained_burn_applied_force(
        "lander", thrust_n=5000.0, duration_s=12.0,
        direction_unit_vector=[0.0, 0.0, 1.0],
        exhaust_velocity_m_s=3100.0)
    assert af.kind == "force"
    assert af.target == "lander"
    assert af.vector == pytest.approx([0.0, 0.0, 5000.0])
    assert af.duration == pytest.approx(12.0)


def test_sustained_burn_discloses_propellant_consumed():
    thrust = 5000.0
    duration = 12.0
    v_e = 3100.0
    af = sustained_burn_applied_force(
        "lander", thrust_n=thrust, duration_s=duration,
        direction_unit_vector=[1.0, 0.0, 0.0], exhaust_velocity_m_s=v_e)
    expected_mdot = mass_flow_rate(thrust, v_e)
    assert af.variables["mass_flow_rate_kg_s"] == pytest.approx(expected_mdot)
    assert af.variables["propellant_mass_consumed_kg"] == pytest.approx(
        expected_mdot * duration, rel=1e-6)
    assert af.variables["propellant_mass_consumed_kg"] > 0.0


def test_sustained_burn_rejects_nonpositive_thrust_or_duration():
    with pytest.raises(ValueError):
        sustained_burn_applied_force(
            "lander", thrust_n=0.0, duration_s=10.0,
            direction_unit_vector=[1.0, 0.0, 0.0],
            exhaust_velocity_m_s=3000.0)
    with pytest.raises(ValueError):
        sustained_burn_applied_force(
            "lander", thrust_n=1000.0, duration_s=0.0,
            direction_unit_vector=[1.0, 0.0, 0.0],
            exhaust_velocity_m_s=3000.0)


def test_sustained_burn_round_trips_through_to_plug_and_from_dict():
    af = sustained_burn_applied_force(
        "lander", thrust_n=4200.0, duration_s=8.0,
        direction_unit_vector=[0.0, 1.0, 0.0],
        exhaust_velocity_m_s=exhaust_velocity_from_isp(311.0))
    plug = af.to_plug()
    assert plug["kind"] == "force"
    assert plug["variables"]["duration_s"] == pytest.approx(8.0)
    assert plug["variables"]["propellant_mass_consumed_kg"] == pytest.approx(
        af.variables["propellant_mass_consumed_kg"])

    back = AppliedForce.from_dict(plug)
    assert back.kind == "force"
    assert back.duration == pytest.approx(8.0)
    assert back.vector == pytest.approx(af.vector)
    assert back.variables["propellant_mass_consumed_kg"] == pytest.approx(
        af.variables["propellant_mass_consumed_kg"])


def test_sustained_burn_uses_a_catalog_engine_end_to_end():
    """A NEXT-C ion burn: low thrust, high Isp, sipping propellant --
    exercises the catalog and both adapters together against a real
    engine's figures."""
    next_c = PROPULSION_SYSTEMS["next_c"]
    v_e = next_c.exhaust_velocity_m_s
    thrust = 0.236   # N -- representative NEXT-C throttle point (not asserted
                      # against a source; only used to exercise the adapter)
    af = sustained_burn_applied_force(
        "probe", thrust_n=thrust, duration_s=3600.0,
        direction_unit_vector=[1.0, 0.0, 0.0], exhaust_velocity_m_s=v_e)
    # the disclosed value is rounded to 6 decimals (see to_plug()'s own
    # rounding convention), so compare with a looser tolerance than the
    # exact unrounded product
    assert af.variables["propellant_mass_consumed_kg"] == pytest.approx(
        mass_flow_rate(thrust, v_e) * 3600.0, rel=1e-4)
    # an ion engine sips propellant: an hour of thrust burns well under a
    # gram given these figures
    assert af.variables["propellant_mass_consumed_kg"] < 0.1
