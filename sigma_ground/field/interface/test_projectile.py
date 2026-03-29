"""Tests for projectile.py — projectile motion, drag, inclined planes, sigma coupling.

Tests cover:
  1. Ideal projectile range, height, time of flight at canonical angles
  2. Complementary angle symmetry (30 deg = 60 deg range)
  3. Vertical launch max height = v0^2 / (2g)
  4. Trajectory boundary conditions (starts at origin, ends at ground)
  5. Drag force scales with velocity squared
  6. Terminal velocity for known objects (baseball)
  7. Drag reduces range below ideal
  8. Incline frictionless acceleration = g sin(theta)
  9. Critical angle = arctan(mu)
 10. High friction clamps acceleration to zero
 11. Sigma coupling: terminal velocity ratio = sqrt(scale_ratio)
 12. Projectile report with and without drag
"""

import math

import pytest

from .projectile import (
    projectile_range,
    projectile_max_height,
    projectile_time_of_flight,
    projectile_trajectory,
    drag_force,
    terminal_velocity,
    projectile_with_drag,
    incline_acceleration,
    incline_critical_angle,
    incline_sliding_distance,
    incline_speed_at_bottom,
    sigma_terminal_velocity_ratio,
    projectile_report,
    G_EARTH,
    RHO_AIR,
)
from ..scale import scale_ratio
from ..constants import SIGMA_HERE


# =====================================================================
# IDEAL PROJECTILE — RANGE
# =====================================================================

def test_range_45_degrees():
    """45 deg gives maximum range: R = v0^2 / g for sin(90)=1."""
    v0 = 10.0
    R = projectile_range(v0, math.radians(45))
    expected = v0 ** 2 / G_EARTH  # sin(90) = 1
    assert R == pytest.approx(expected, rel=1e-9)
    assert R == pytest.approx(10.197, rel=1e-3)


def test_range_complementary_angles():
    """Range at 30 deg equals range at 60 deg (complementary angles)."""
    v0 = 20.0
    R30 = projectile_range(v0, math.radians(30))
    R60 = projectile_range(v0, math.radians(60))
    assert R30 == pytest.approx(R60, rel=1e-9)


def test_range_zero_angle():
    """Horizontal launch on flat ground has zero range (sin(0)=0)."""
    assert projectile_range(10.0, 0.0) == pytest.approx(0.0, abs=1e-12)


def test_range_90_degrees():
    """Vertical launch has zero range (sin(180)=0)."""
    R = projectile_range(10.0, math.radians(90))
    assert R == pytest.approx(0.0, abs=1e-9)


def test_range_scales_with_v0_squared():
    """Doubling v0 quadruples the range."""
    angle = math.radians(45)
    R1 = projectile_range(10.0, angle)
    R2 = projectile_range(20.0, angle)
    assert R2 == pytest.approx(4.0 * R1, rel=1e-9)


def test_range_custom_gravity():
    """Range on the Moon (g=1.625) is much larger."""
    v0, angle = 10.0, math.radians(45)
    R_earth = projectile_range(v0, angle)
    R_moon = projectile_range(v0, angle, g=1.625)
    assert R_moon > R_earth
    assert R_moon == pytest.approx(v0 ** 2 / 1.625, rel=1e-9)


# =====================================================================
# IDEAL PROJECTILE — MAX HEIGHT
# =====================================================================

def test_max_height_90_degrees():
    """Vertical launch: H = v0^2 / (2g)."""
    v0 = 10.0
    H = projectile_max_height(v0, math.radians(90))
    expected = v0 ** 2 / (2 * G_EARTH)
    assert H == pytest.approx(expected, rel=1e-9)


def test_max_height_45_degrees():
    """Max height at 45 deg = v0^2 sin^2(45) / (2g) = v0^2 / (4g)."""
    v0 = 10.0
    H = projectile_max_height(v0, math.radians(45))
    expected = v0 ** 2 / (4.0 * G_EARTH)
    assert H == pytest.approx(expected, rel=1e-9)


def test_max_height_zero_angle():
    """Horizontal launch has zero max height."""
    assert projectile_max_height(10.0, 0.0) == pytest.approx(0.0, abs=1e-12)


def test_max_height_increases_with_angle():
    """Higher launch angle gives greater max height (up to 90 deg)."""
    v0 = 15.0
    H30 = projectile_max_height(v0, math.radians(30))
    H60 = projectile_max_height(v0, math.radians(60))
    H90 = projectile_max_height(v0, math.radians(90))
    assert H30 < H60 < H90


# =====================================================================
# IDEAL PROJECTILE — TIME OF FLIGHT
# =====================================================================

def test_time_of_flight_basic():
    """T = 2 v0 sin(theta) / g."""
    v0 = 10.0
    angle = math.radians(45)
    T = projectile_time_of_flight(v0, angle)
    expected = 2 * v0 * math.sin(angle) / G_EARTH
    assert T == pytest.approx(expected, rel=1e-9)


def test_time_of_flight_vertical():
    """Vertical launch: T = 2 v0 / g."""
    v0 = 10.0
    T = projectile_time_of_flight(v0, math.radians(90))
    assert T == pytest.approx(2 * v0 / G_EARTH, rel=1e-9)


def test_time_of_flight_zero_angle():
    """Horizontal launch: T = 0 (on flat ground)."""
    assert projectile_time_of_flight(10.0, 0.0) == pytest.approx(0.0, abs=1e-12)


# =====================================================================
# IDEAL PROJECTILE — TRAJECTORY
# =====================================================================

def test_trajectory_starts_at_origin():
    """First trajectory point is at (0, 0, 0)."""
    traj = projectile_trajectory(10.0, math.radians(45))
    x0, y0, t0 = traj[0]
    assert x0 == pytest.approx(0.0, abs=1e-12)
    assert y0 == pytest.approx(0.0, abs=1e-12)
    assert t0 == pytest.approx(0.0, abs=1e-12)


def test_trajectory_ends_at_ground():
    """Last trajectory point has y = 0 and x approx equal to range."""
    v0, angle = 10.0, math.radians(45)
    traj = projectile_trajectory(v0, angle)
    x_end, y_end, t_end = traj[-1]
    R = projectile_range(v0, angle)
    assert y_end == pytest.approx(0.0, abs=1e-6)
    assert x_end == pytest.approx(R, rel=1e-3)


def test_trajectory_length():
    """Trajectory has steps+1 points."""
    traj = projectile_trajectory(10.0, math.radians(45), steps=50)
    assert len(traj) == 51


def test_trajectory_all_y_non_negative():
    """All y values in the trajectory are non-negative."""
    traj = projectile_trajectory(10.0, math.radians(30), steps=200)
    for x, y, t in traj:
        assert y >= 0.0


def test_trajectory_peak_near_midpoint():
    """Maximum height occurs near the midpoint of the trajectory."""
    traj = projectile_trajectory(10.0, math.radians(45), steps=100)
    heights = [y for _, y, _ in traj]
    peak_idx = heights.index(max(heights))
    # Peak should be roughly in the middle half
    assert 25 <= peak_idx <= 75


# =====================================================================
# DRAG FORCE
# =====================================================================

def test_drag_force_basic():
    """F_d = 0.5 * rho * Cd * A * v^2."""
    v, Cd, A, rho = 10.0, 0.47, 0.01, 1.225
    F = drag_force(v, Cd, A, rho)
    expected = 0.5 * rho * Cd * A * v ** 2
    assert F == pytest.approx(expected, rel=1e-9)


def test_drag_force_zero_velocity():
    """No drag at zero velocity."""
    assert drag_force(0.0, 0.47, 0.01) == pytest.approx(0.0, abs=1e-12)


def test_drag_force_scales_with_v_squared():
    """Doubling velocity quadruples drag force."""
    Cd, A = 0.47, 0.01
    F1 = drag_force(10.0, Cd, A)
    F2 = drag_force(20.0, Cd, A)
    assert F2 == pytest.approx(4.0 * F1, rel=1e-9)


def test_drag_force_scales_with_area():
    """Doubling area doubles drag force."""
    v, Cd = 10.0, 0.47
    F1 = drag_force(v, Cd, 0.01)
    F2 = drag_force(v, Cd, 0.02)
    assert F2 == pytest.approx(2.0 * F1, rel=1e-9)


def test_drag_force_scales_with_density():
    """Doubling fluid density doubles drag force."""
    v, Cd, A = 10.0, 0.47, 0.01
    F1 = drag_force(v, Cd, A, rho_fluid=1.0)
    F2 = drag_force(v, Cd, A, rho_fluid=2.0)
    assert F2 == pytest.approx(2.0 * F1, rel=1e-9)


# =====================================================================
# TERMINAL VELOCITY
# =====================================================================

def test_terminal_velocity_baseball():
    """Terminal velocity of a baseball: m=0.145 kg, Cd=0.3, A=0.00426 m^2."""
    m, Cd, A = 0.145, 0.3, 0.00426
    vt = terminal_velocity(m, Cd, A, sigma=0.0)
    # v_t = sqrt(2mg / (rho*Cd*A)) at sigma=0, scale_ratio(0)=1
    expected = math.sqrt(2 * m * G_EARTH / (RHO_AIR * Cd * A))
    assert vt == pytest.approx(expected, rel=1e-6)
    # Sanity: baseball terminal velocity is roughly 30-45 m/s
    assert 25 < vt < 50


def test_terminal_velocity_increases_with_mass():
    """Heavier objects have higher terminal velocity."""
    Cd, A = 0.47, 0.01
    vt1 = terminal_velocity(1.0, Cd, A, sigma=0.0)
    vt2 = terminal_velocity(4.0, Cd, A, sigma=0.0)
    assert vt2 == pytest.approx(2.0 * vt1, rel=1e-6)  # sqrt(4) = 2


def test_terminal_velocity_decreases_with_area():
    """Larger cross section gives lower terminal velocity."""
    m, Cd = 1.0, 0.47
    vt1 = terminal_velocity(m, Cd, 0.01, sigma=0.0)
    vt2 = terminal_velocity(m, Cd, 0.04, sigma=0.0)
    assert vt2 == pytest.approx(0.5 * vt1, rel=1e-6)  # sqrt(1/4) = 0.5


# =====================================================================
# PROJECTILE WITH DRAG
# =====================================================================

def test_drag_reduces_range():
    """Range with drag is less than ideal range."""
    v0, angle = 20.0, math.radians(45)
    mass, Cd, A = 0.145, 0.3, 0.00426  # baseball
    ideal_R = projectile_range(v0, angle)
    result = projectile_with_drag(v0, angle, mass, Cd, A, sigma=0.0)
    assert result['range_m'] < ideal_R


def test_drag_reduces_max_height():
    """Max height with drag is less than ideal max height."""
    v0, angle = 20.0, math.radians(45)
    mass, Cd, A = 0.145, 0.3, 0.00426
    ideal_H = projectile_max_height(v0, angle)
    result = projectile_with_drag(v0, angle, mass, Cd, A, sigma=0.0)
    assert result['max_height_m'] < ideal_H


def test_drag_trajectory_starts_at_origin():
    """Drag trajectory starts at (0, 0, 0)."""
    result = projectile_with_drag(
        10.0, math.radians(45), 0.145, 0.3, 0.00426, sigma=0.0
    )
    x0, y0, t0 = result['trajectory'][0]
    assert x0 == pytest.approx(0.0, abs=1e-12)
    assert y0 == pytest.approx(0.0, abs=1e-12)
    assert t0 == pytest.approx(0.0, abs=1e-12)


def test_drag_trajectory_ends_at_ground():
    """Drag trajectory ends at y = 0."""
    result = projectile_with_drag(
        10.0, math.radians(45), 0.145, 0.3, 0.00426, sigma=0.0
    )
    _, y_end, _ = result['trajectory'][-1]
    assert y_end == pytest.approx(0.0, abs=1e-6)


def test_drag_result_has_required_keys():
    """Result dict contains all expected keys."""
    result = projectile_with_drag(
        10.0, math.radians(45), 0.145, 0.3, 0.00426, sigma=0.0
    )
    for key in ['trajectory', 'range_m', 'max_height_m',
                'time_of_flight_s', 'impact_velocity_m_s',
                'drag_coefficient', 'origin']:
        assert key in result


def test_drag_impact_velocity_less_than_launch():
    """Impact speed with drag is less than launch speed."""
    v0 = 20.0
    result = projectile_with_drag(
        v0, math.radians(45), 0.145, 0.3, 0.00426, sigma=0.0
    )
    assert result['impact_velocity_m_s'] < v0


def test_zero_drag_approximates_ideal():
    """With Cd ~ 0 and tiny area, drag trajectory approximates ideal."""
    v0, angle = 10.0, math.radians(45)
    # Near-zero drag: very small Cd and area
    result = projectile_with_drag(
        v0, angle, 1.0, 1e-10, 1e-10, dt=0.0001, sigma=0.0
    )
    ideal_R = projectile_range(v0, angle)
    assert result['range_m'] == pytest.approx(ideal_R, rel=0.01)


# =====================================================================
# INCLINED PLANE
# =====================================================================

def test_incline_frictionless():
    """Frictionless incline: a = g sin(theta)."""
    angle = math.radians(30)
    a = incline_acceleration(angle)
    expected = G_EARTH * math.sin(angle)
    assert a == pytest.approx(expected, rel=1e-9)


def test_incline_with_friction():
    """Incline with friction: a = g(sin(theta) - mu*cos(theta))."""
    angle = math.radians(45)
    mu = 0.2
    a = incline_acceleration(angle, mu)
    expected = G_EARTH * (math.sin(angle) - mu * math.cos(angle))
    assert a == pytest.approx(expected, rel=1e-9)


def test_incline_high_friction_clamps_to_zero():
    """High friction on shallow incline: acceleration clamped to 0."""
    angle = math.radians(10)
    mu = 1.0  # mu > tan(10 deg) ~ 0.176
    a = incline_acceleration(angle, mu)
    assert a == pytest.approx(0.0, abs=1e-12)


def test_incline_critical_angle_basic():
    """Critical angle = arctan(mu)."""
    mu = 0.5
    theta_c = incline_critical_angle(mu)
    assert theta_c == pytest.approx(math.atan(mu), rel=1e-9)


def test_incline_critical_angle_zero_friction():
    """Zero friction: critical angle is 0 (always slides)."""
    assert incline_critical_angle(0.0) == pytest.approx(0.0, abs=1e-12)


def test_incline_critical_angle_high_friction():
    """mu = 1.0: critical angle = 45 deg."""
    theta_c = incline_critical_angle(1.0)
    assert theta_c == pytest.approx(math.radians(45), rel=1e-9)


def test_incline_at_critical_angle_no_acceleration():
    """At exactly the critical angle, acceleration is zero."""
    mu = 0.5
    theta_c = incline_critical_angle(mu)
    a = incline_acceleration(theta_c, mu)
    assert a == pytest.approx(0.0, abs=1e-6)


# =====================================================================
# INCLINE — SLIDING DISTANCE
# =====================================================================

def test_sliding_distance_basic():
    """d = v0^2 / (2g(sin(theta) + mu*cos(theta)))."""
    v0 = 5.0
    angle = math.radians(30)
    mu = 0.3
    d = incline_sliding_distance(v0, angle, mu)
    decel = G_EARTH * (math.sin(angle) + mu * math.cos(angle))
    expected = v0 ** 2 / (2 * decel)
    assert d == pytest.approx(expected, rel=1e-9)


def test_sliding_distance_frictionless():
    """Frictionless: d = v0^2 / (2g*sin(theta))."""
    v0 = 5.0
    angle = math.radians(45)
    d = incline_sliding_distance(v0, angle, 0.0)
    expected = v0 ** 2 / (2 * G_EARTH * math.sin(angle))
    assert d == pytest.approx(expected, rel=1e-9)


def test_sliding_distance_increases_with_speed():
    """Faster launch goes farther up the incline."""
    angle, mu = math.radians(30), 0.2
    d1 = incline_sliding_distance(5.0, angle, mu)
    d2 = incline_sliding_distance(10.0, angle, mu)
    assert d2 == pytest.approx(4.0 * d1, rel=1e-9)


# =====================================================================
# INCLINE — SPEED AT BOTTOM
# =====================================================================

def test_speed_at_bottom_frictionless():
    """Frictionless: v = sqrt(2gh)."""
    h = 10.0
    angle = math.radians(30)
    v = incline_speed_at_bottom(h, angle, mu_friction=0.0)
    expected = math.sqrt(2 * G_EARTH * h)
    assert v == pytest.approx(expected, rel=1e-9)


def test_speed_at_bottom_with_friction():
    """With friction: v = sqrt(2gh(1 - mu/tan(theta)))."""
    h, angle, mu = 10.0, math.radians(45), 0.3
    v = incline_speed_at_bottom(h, angle, mu)
    factor = 1.0 - mu / math.tan(angle)
    expected = math.sqrt(2 * G_EARTH * h * factor)
    assert v == pytest.approx(expected, rel=1e-9)


def test_speed_at_bottom_high_friction_returns_zero():
    """If friction is too high, speed at bottom is 0 (can't slide)."""
    h, angle = 10.0, math.radians(20)
    mu = 2.0  # mu/tan(20) >> 1
    v = incline_speed_at_bottom(h, angle, mu)
    assert v == pytest.approx(0.0, abs=1e-12)


def test_speed_at_bottom_zero_angle_returns_zero():
    """Flat surface (angle=0): no speed."""
    v = incline_speed_at_bottom(10.0, 0.0)
    assert v == pytest.approx(0.0, abs=1e-12)


# =====================================================================
# SIGMA COUPLING
# =====================================================================

def test_sigma_terminal_velocity_ratio_at_zero():
    """At sigma=0, scale_ratio=1, ratio=1."""
    ratio = sigma_terminal_velocity_ratio(0.0)
    assert ratio == pytest.approx(1.0, rel=1e-9)


def test_sigma_terminal_velocity_ratio_positive():
    """At sigma > 0, ratio = sqrt(scale_ratio(sigma)) > 1."""
    sigma = 1.0
    ratio = sigma_terminal_velocity_ratio(sigma)
    expected = math.sqrt(scale_ratio(sigma))
    assert ratio == pytest.approx(expected, rel=1e-9)
    assert ratio > 1.0


def test_sigma_terminal_velocity_ratio_scales():
    """Ratio increases monotonically with sigma."""
    r1 = sigma_terminal_velocity_ratio(0.5)
    r2 = sigma_terminal_velocity_ratio(1.0)
    r3 = sigma_terminal_velocity_ratio(2.0)
    assert r1 < r2 < r3


def test_terminal_velocity_sigma_coupling():
    """Terminal velocity at sigma > 0 is higher than at sigma = 0."""
    m, Cd, A = 1.0, 0.47, 0.01
    vt_0 = terminal_velocity(m, Cd, A, sigma=0.0)
    vt_1 = terminal_velocity(m, Cd, A, sigma=1.0)
    ratio = vt_1 / vt_0
    expected_ratio = math.sqrt(scale_ratio(1.0))
    assert ratio == pytest.approx(expected_ratio, rel=1e-6)
    assert vt_1 > vt_0


# =====================================================================
# PROJECTILE REPORT
# =====================================================================

def test_report_ideal_only():
    """Report without drag contains ideal fields, no drag fields."""
    r = projectile_report(10.0, math.radians(45))
    assert 'ideal_range_m' in r
    assert 'ideal_max_height_m' in r
    assert 'ideal_time_of_flight_s' in r
    assert 'drag_range_m' not in r
    assert r['angle_deg'] == pytest.approx(45.0, rel=1e-9)
    assert r['ideal_range_m'] == pytest.approx(
        projectile_range(10.0, math.radians(45)), rel=1e-9
    )


def test_report_with_drag():
    """Report with Cd > 0 and area > 0 includes drag fields."""
    r = projectile_report(
        20.0, math.radians(45), mass=0.145, Cd=0.3, area=0.00426
    )
    assert 'drag_range_m' in r
    assert 'drag_max_height_m' in r
    assert 'drag_time_of_flight_s' in r
    assert 'range_reduction_pct' in r
    assert r['drag_range_m'] < r['ideal_range_m']
    assert r['range_reduction_pct'] > 0


def test_report_has_origin():
    """Report always has an origin string."""
    r = projectile_report(10.0, math.radians(45))
    assert 'origin' in r
    assert isinstance(r['origin'], str)
    assert 'FIRST_PRINCIPLES' in r['origin']


def test_report_sigma_field():
    """Report records the sigma value used."""
    r = projectile_report(10.0, math.radians(45), sigma=0.0)
    assert r['sigma'] == 0.0
