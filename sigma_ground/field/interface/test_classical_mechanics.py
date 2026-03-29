"""Tests for classical_mechanics.py — energy, momentum, work, power, collisions.

Covers:
  1. Basic physics correctness (known textbook values)
  2. Energy conservation in elastic collisions
  3. Momentum conservation in elastic and inelastic collisions
  4. sigma-field scaling (mass-dependent quantities scale with e^sigma)
  5. Edge cases (zero velocity, zero height, perfectly inelastic cor=0)
  6. mechanics_report returns expected keys
"""

import math

import pytest
from pytest import approx

from sigma_ground.field.constants import SIGMA_HERE
from sigma_ground.field.scale import scale_ratio
from sigma_ground.field.interface.classical_mechanics import (
    gravitational_pe,
    kinetic_energy,
    rotational_ke,
    total_mechanical_energy,
    work_done,
    power_mechanical,
    friction_dissipation,
    momentum,
    impulse,
    velocity_from_impulse,
    elastic_collision_velocities,
    inelastic_collision_velocity,
    collision_energy_loss,
    sigma_energy_shift,
    mechanics_report,
    G_EARTH,
)


# ── 1. Basic physics correctness ──────────────────────────────────────


def test_gravitational_pe_1kg_10m():
    """U = mgh = 1 * 9.80665 * 10 = 98.0665 J."""
    assert gravitational_pe(1.0, 10.0) == approx(98.0665, rel=1e-6)


def test_gravitational_pe_custom_g():
    """Moon gravity: g = 1.625 m/s^2, 5 kg at 3 m."""
    assert gravitational_pe(5.0, 3.0, g=1.625) == approx(24.375, rel=1e-6)


def test_kinetic_energy_known():
    """KE = 0.5 * 2 * 3^2 = 9.0 J."""
    assert kinetic_energy(2.0, 3.0) == approx(9.0, rel=1e-9)


def test_kinetic_energy_10kg_at_10ms():
    """KE = 0.5 * 10 * 100 = 500 J."""
    assert kinetic_energy(10.0, 10.0) == approx(500.0, rel=1e-9)


def test_rotational_ke_known():
    """KE_rot = 0.5 * 4.0 * 5.0^2 = 50.0 J."""
    assert rotational_ke(4.0, 5.0) == approx(50.0, rel=1e-9)


def test_total_mechanical_energy_point_mass():
    """E = KE + PE for a point mass (no rotation)."""
    mass, vel, h = 2.0, 5.0, 10.0
    expected = 0.5 * mass * vel**2 + mass * G_EARTH * h
    assert total_mechanical_energy(mass, vel, h) == approx(expected, rel=1e-9)


def test_total_mechanical_energy_with_rotation():
    """E = KE + PE + KE_rot."""
    mass, vel, h = 2.0, 5.0, 10.0
    I, omega = 0.5, 10.0
    ke = 0.5 * mass * vel**2
    pe = mass * G_EARTH * h
    ke_rot = 0.5 * I * omega**2
    expected = ke + pe + ke_rot
    result = total_mechanical_energy(mass, vel, h, inertia=I,
                                     angular_velocity=omega)
    assert result == approx(expected, rel=1e-9)


def test_work_done_parallel():
    """W = Fd when force is parallel (angle=0)."""
    assert work_done(100.0, 5.0) == approx(500.0, rel=1e-9)


def test_work_done_perpendicular():
    """W = 0 when force is perpendicular (angle=pi/2)."""
    assert work_done(100.0, 5.0, angle=math.pi / 2) == approx(0.0, abs=1e-10)


def test_work_done_at_60_degrees():
    """W = Fd cos(60) = 100 * 5 * 0.5 = 250 J."""
    assert work_done(100.0, 5.0, angle=math.pi / 3) == approx(250.0, rel=1e-6)


def test_power_mechanical_known():
    """P = Fv = 50 * 4 = 200 W."""
    assert power_mechanical(50.0, 4.0) == approx(200.0, rel=1e-9)


def test_friction_dissipation_known():
    """W_f = |f| * |d| = 20 * 10 = 200 J."""
    assert friction_dissipation(20.0, 10.0) == approx(200.0, rel=1e-9)


def test_friction_dissipation_negative_inputs():
    """Friction dissipation uses absolute values."""
    assert friction_dissipation(-15.0, -8.0) == approx(120.0, rel=1e-9)


def test_momentum_known():
    """p = mv = 3 * 7 = 21 kg*m/s."""
    assert momentum(3.0, 7.0) == approx(21.0, rel=1e-9)


def test_momentum_negative_velocity():
    """Momentum with negative velocity is negative."""
    assert momentum(5.0, -3.0) == approx(-15.0, rel=1e-9)


def test_impulse_known():
    """J = F * dt = 100 * 0.05 = 5.0 N*s."""
    assert impulse(100.0, 0.05) == approx(5.0, rel=1e-9)


def test_velocity_from_impulse_from_rest():
    """v_f = 0 + J/m = 10/2 = 5 m/s."""
    assert velocity_from_impulse(10.0, 2.0) == approx(5.0, rel=1e-9)


def test_velocity_from_impulse_with_initial():
    """v_f = 3 + 10/2 = 8 m/s."""
    assert velocity_from_impulse(10.0, 2.0, v_initial=3.0) == approx(8.0, rel=1e-9)


# ── 2. Energy conservation in elastic collisions ─────────────────────


def test_elastic_collision_ke_conserved():
    """Total KE before = total KE after for elastic collision."""
    m1, v1, m2, v2 = 3.0, 5.0, 2.0, -3.0
    ke_before = 0.5 * m1 * v1**2 + 0.5 * m2 * v2**2

    v1f, v2f = elastic_collision_velocities(m1, v1, m2, v2)
    ke_after = 0.5 * m1 * v1f**2 + 0.5 * m2 * v2f**2

    assert ke_after == approx(ke_before, rel=1e-10)


def test_elastic_collision_ke_conserved_equal_masses():
    """Equal-mass elastic collision: velocities swap."""
    m1, v1, m2, v2 = 5.0, 10.0, 5.0, 0.0
    v1f, v2f = elastic_collision_velocities(m1, v1, m2, v2)

    assert v1f == approx(0.0, abs=1e-10)
    assert v2f == approx(10.0, rel=1e-10)


def test_elastic_collision_energy_loss_zero():
    """Energy loss should be zero for elastic collision (cor=1)."""
    m1, v1, m2, v2 = 4.0, 6.0, 3.0, -2.0
    loss = collision_energy_loss(m1, v1, m2, v2, cor=1.0)
    assert loss == approx(0.0, abs=1e-10)


# ── 3. Momentum conservation ────────────────────────────────────────


def test_elastic_collision_momentum_conserved():
    """Total momentum before = total momentum after (elastic)."""
    m1, v1, m2, v2 = 3.0, 5.0, 2.0, -3.0
    p_before = m1 * v1 + m2 * v2

    v1f, v2f = elastic_collision_velocities(m1, v1, m2, v2)
    p_after = m1 * v1f + m2 * v2f

    assert p_after == approx(p_before, rel=1e-10)


def test_inelastic_collision_momentum_conserved():
    """Momentum conserved for any coefficient of restitution."""
    m1, v1, m2, v2 = 4.0, 8.0, 6.0, -2.0
    p_before = m1 * v1 + m2 * v2

    for cor in [0.0, 0.3, 0.5, 0.7, 1.0]:
        v1f, v2f = inelastic_collision_velocity(m1, v1, m2, v2, cor=cor)
        p_after = m1 * v1f + m2 * v2f
        assert p_after == approx(p_before, rel=1e-10), f"Failed at cor={cor}"


def test_inelastic_cor1_matches_elastic():
    """cor=1.0 in inelastic should match elastic collision result."""
    m1, v1, m2, v2 = 3.0, 5.0, 2.0, -3.0

    v1f_e, v2f_e = elastic_collision_velocities(m1, v1, m2, v2)
    v1f_i, v2f_i = inelastic_collision_velocity(m1, v1, m2, v2, cor=1.0)

    assert v1f_i == approx(v1f_e, rel=1e-10)
    assert v2f_i == approx(v2f_e, rel=1e-10)


# ── 4. sigma-field scaling ───────────────────────────────────────────


SIGMA_TEST = 1.0  # scale_ratio(1.0) = e ~ 2.71828


def test_sigma_here_is_unity_scaling():
    """At SIGMA_HERE, scale_ratio returns 1.0 so results are standard."""
    assert scale_ratio(SIGMA_HERE) == approx(1.0, rel=1e-12)


def test_gravitational_pe_sigma_scales():
    """PE scales linearly with scale_ratio(sigma)."""
    pe_standard = gravitational_pe(1.0, 10.0, sigma=SIGMA_HERE)
    pe_shifted = gravitational_pe(1.0, 10.0, sigma=SIGMA_TEST)
    ratio = pe_shifted / pe_standard
    assert ratio == approx(scale_ratio(SIGMA_TEST), rel=1e-10)


def test_kinetic_energy_sigma_scales():
    """KE scales linearly with scale_ratio(sigma)."""
    ke_standard = kinetic_energy(2.0, 5.0, sigma=SIGMA_HERE)
    ke_shifted = kinetic_energy(2.0, 5.0, sigma=SIGMA_TEST)
    ratio = ke_shifted / ke_standard
    assert ratio == approx(scale_ratio(SIGMA_TEST), rel=1e-10)


def test_rotational_ke_sigma_scales():
    """Rotational KE scales linearly with scale_ratio(sigma)."""
    ke_std = rotational_ke(3.0, 4.0, sigma=SIGMA_HERE)
    ke_shifted = rotational_ke(3.0, 4.0, sigma=SIGMA_TEST)
    ratio = ke_shifted / ke_std
    assert ratio == approx(scale_ratio(SIGMA_TEST), rel=1e-10)


def test_momentum_sigma_scales():
    """Momentum scales linearly with scale_ratio(sigma)."""
    p_std = momentum(5.0, 3.0, sigma=SIGMA_HERE)
    p_shifted = momentum(5.0, 3.0, sigma=SIGMA_TEST)
    ratio = p_shifted / p_std
    assert ratio == approx(scale_ratio(SIGMA_TEST), rel=1e-10)


def test_sigma_energy_shift_known():
    """sigma_energy_shift(E, sigma) = E * scale_ratio(sigma)."""
    energy = 100.0
    shifted = sigma_energy_shift(energy, SIGMA_TEST)
    assert shifted == approx(energy * math.e, rel=1e-10)


def test_sigma_energy_shift_zero_sigma():
    """At sigma=0 (flat spacetime), scale_ratio=1, energy unchanged."""
    energy = 42.0
    assert sigma_energy_shift(energy, 0.0) == approx(energy, rel=1e-12)


def test_elastic_collision_sigma_independent():
    """Elastic collision velocities are independent of sigma (sigma cancels)."""
    m1, v1, m2, v2 = 3.0, 5.0, 2.0, -3.0

    v1f_std, v2f_std = elastic_collision_velocities(m1, v1, m2, v2,
                                                     sigma=SIGMA_HERE)
    v1f_shifted, v2f_shifted = elastic_collision_velocities(m1, v1, m2, v2,
                                                            sigma=SIGMA_TEST)

    assert v1f_shifted == approx(v1f_std, rel=1e-10)
    assert v2f_shifted == approx(v2f_std, rel=1e-10)


def test_collision_energy_loss_sigma_scales():
    """Energy loss in inelastic collision scales with sigma (through reduced mass)."""
    m1, v1, m2, v2, cor = 4.0, 6.0, 3.0, -2.0, 0.5
    loss_std = collision_energy_loss(m1, v1, m2, v2, cor, sigma=SIGMA_HERE)
    loss_shifted = collision_energy_loss(m1, v1, m2, v2, cor, sigma=SIGMA_TEST)
    ratio = loss_shifted / loss_std
    assert ratio == approx(scale_ratio(SIGMA_TEST), rel=1e-10)


def test_velocity_from_impulse_sigma_scales():
    """Higher sigma means heavier mass, so same impulse gives less delta-v."""
    v_std = velocity_from_impulse(10.0, 2.0, sigma=SIGMA_HERE)
    v_shifted = velocity_from_impulse(10.0, 2.0, sigma=SIGMA_TEST)
    # v = J/m; if m scales by e^sigma, then delta-v scales by e^{-sigma}
    delta_v_std = 10.0 / 2.0  # = 5.0
    delta_v_shifted = 10.0 / (2.0 * scale_ratio(SIGMA_TEST))
    assert v_shifted == approx(delta_v_shifted, rel=1e-10)


def test_negative_sigma_scaling():
    """Negative sigma gives scale_ratio < 1, reducing effective mass."""
    sigma_neg = -0.5
    s = scale_ratio(sigma_neg)
    assert s < 1.0
    ke_neg = kinetic_energy(2.0, 5.0, sigma=sigma_neg)
    ke_std = kinetic_energy(2.0, 5.0, sigma=SIGMA_HERE)
    assert ke_neg < ke_std
    assert ke_neg == approx(ke_std * s, rel=1e-10)


# ── 5. Edge cases ───────────────────────────────────────────────────


def test_kinetic_energy_zero_velocity():
    """KE = 0 when velocity is zero."""
    assert kinetic_energy(10.0, 0.0) == approx(0.0, abs=1e-15)


def test_gravitational_pe_zero_height():
    """PE = 0 at reference height."""
    assert gravitational_pe(10.0, 0.0) == approx(0.0, abs=1e-15)


def test_rotational_ke_zero_omega():
    """Rotational KE = 0 when not spinning."""
    assert rotational_ke(5.0, 0.0) == approx(0.0, abs=1e-15)


def test_momentum_zero_velocity():
    """Zero velocity gives zero momentum."""
    assert momentum(10.0, 0.0) == approx(0.0, abs=1e-15)


def test_work_done_zero_distance():
    """No displacement means no work done."""
    assert work_done(500.0, 0.0) == approx(0.0, abs=1e-15)


def test_impulse_zero_duration():
    """Zero time interval gives zero impulse."""
    assert impulse(100.0, 0.0) == approx(0.0, abs=1e-15)


def test_perfectly_inelastic_collision_stick_together():
    """cor=0: objects stick together, share same final velocity."""
    m1, v1, m2, v2 = 4.0, 6.0, 2.0, 0.0
    v1f, v2f = inelastic_collision_velocity(m1, v1, m2, v2, cor=0.0)

    # Both should have the center-of-mass velocity
    v_cm = (m1 * v1 + m2 * v2) / (m1 + m2)
    assert v1f == approx(v_cm, rel=1e-10)
    assert v2f == approx(v_cm, rel=1e-10)


def test_perfectly_inelastic_max_energy_loss():
    """cor=0 gives maximum kinetic energy loss."""
    m1, v1, m2, v2 = 4.0, 6.0, 2.0, 0.0
    loss = collision_energy_loss(m1, v1, m2, v2, cor=0.0)

    # ΔKE = 0.5 * μ * (v1 - v2)^2 for cor=0
    mu = m1 * m2 / (m1 + m2)
    expected = 0.5 * mu * (v1 - v2) ** 2
    assert loss == approx(expected, rel=1e-10)


def test_friction_dissipation_zero_distance():
    """No sliding means no energy dissipated."""
    assert friction_dissipation(50.0, 0.0) == approx(0.0, abs=1e-15)


def test_power_zero_velocity():
    """No motion means no power delivered."""
    assert power_mechanical(100.0, 0.0) == approx(0.0, abs=1e-15)


def test_total_mechanical_energy_all_zero():
    """Everything zero gives zero total energy."""
    assert total_mechanical_energy(0.0, 0.0, 0.0) == approx(0.0, abs=1e-15)


def test_elastic_collision_equal_opposite():
    """Equal masses, equal and opposite velocities: both reverse."""
    m1, v1, m2, v2 = 5.0, 4.0, 5.0, -4.0
    v1f, v2f = elastic_collision_velocities(m1, v1, m2, v2)
    assert v1f == approx(-4.0, rel=1e-10)
    assert v2f == approx(4.0, rel=1e-10)


def test_elastic_collision_heavy_vs_light_stationary():
    """Heavy mass hitting light stationary: heavy barely slows, light flies off."""
    m1, v1, m2, v2 = 100.0, 5.0, 1.0, 0.0
    v1f, v2f = elastic_collision_velocities(m1, v1, m2, v2)
    # Heavy mass barely changes
    assert v1f == approx(v1, rel=0.05)
    # Light mass gets nearly 2x heavy mass velocity
    assert v2f == approx(2.0 * v1 * m1 / (m1 + m2), rel=1e-10)


# ── 6. mechanics_report ─────────────────────────────────────────────


def test_mechanics_report_keys():
    """Report dict must contain all expected keys."""
    report = mechanics_report(2.0, 5.0, height=10.0)
    expected_keys = {
        'mass_kg', 'velocity_m_s', 'height_m', 'sigma',
        'kinetic_energy_J', 'potential_energy_J', 'rotational_ke_J',
        'total_energy_J', 'momentum_kg_m_s', 'origin',
    }
    assert set(report.keys()) == expected_keys


def test_mechanics_report_values_consistent():
    """Report values should match individual function outputs."""
    mass, vel, h = 3.0, 4.0, 5.0
    I, omega = 0.5, 6.0
    report = mechanics_report(mass, vel, h, inertia=I, angular_velocity=omega)

    assert report['kinetic_energy_J'] == approx(
        kinetic_energy(mass, vel), rel=1e-10)
    assert report['potential_energy_J'] == approx(
        gravitational_pe(mass, h), rel=1e-10)
    assert report['rotational_ke_J'] == approx(
        rotational_ke(I, omega), rel=1e-10)
    assert report['momentum_kg_m_s'] == approx(
        momentum(mass, vel), rel=1e-10)
    assert report['total_energy_J'] == approx(
        report['kinetic_energy_J'] + report['potential_energy_J']
        + report['rotational_ke_J'], rel=1e-10)


def test_mechanics_report_sigma_field():
    """Report reflects the sigma value passed in."""
    report = mechanics_report(1.0, 1.0, sigma=SIGMA_TEST)
    assert report['sigma'] == SIGMA_TEST
    assert report['mass_kg'] == approx(scale_ratio(SIGMA_TEST), rel=1e-10)


def test_mechanics_report_origin_tag():
    """Report origin string mentions first principles."""
    report = mechanics_report(1.0, 1.0)
    assert 'FIRST_PRINCIPLES' in report['origin']
    assert 'Wheeler' in report['origin']
