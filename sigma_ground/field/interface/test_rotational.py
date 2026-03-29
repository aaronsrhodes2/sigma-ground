"""Tests for rotational mechanics module.

Test categories:
  1. Moment of inertia — known I values for standard shapes
  2. Parallel axis theorem — shifted I > I_cm
  3. i_factor — known ratios, unknown shape raises ValueError
  4. Torque — perpendicular vs parallel force
  5. Angular acceleration — division by I, ValueError for I <= 0
  6. Angular momentum — L = Iw scaling
  7. Rolling kinematics — v = wr consistency
  8. Rolling dynamics — incline acceleration, speed from height
  9. Rolling friction — distance, time, zero-friction edge case
  10. Ramp-to-flat — combined scenario, dict structure
  11. Shape-based functions — Sphere instance integration
  12. sigma coupling — inertia shifts with sigma
  13. rotational_properties — Nagatha export format
"""

import math
import pytest

from sigma_ground.field.interface.rotational import (
    moment_of_inertia_sphere,
    moment_of_inertia_hollow_sphere,
    moment_of_inertia_cylinder,
    moment_of_inertia_disk,
    moment_of_inertia_rod,
    parallel_axis,
    i_factor,
    torque,
    angular_acceleration,
    angular_momentum,
    rolling_velocity,
    rolling_angular_velocity,
    rolling_acceleration_incline,
    rolling_speed_from_height,
    rolling_distance_on_flat,
    rolling_time_on_flat,
    ramp_to_flat_distance,
    shape_moment_of_inertia,
    shape_rolling_acceleration,
    shape_rolling_speed_from_height,
    sigma_inertia_shift,
    rotational_properties,
    G_EARTH,
)
from sigma_ground.field.scale import scale_ratio
from sigma_ground.shapes import Sphere


# =====================================================================
# 1. MOMENT OF INERTIA — KNOWN VALUES
# =====================================================================

def test_solid_sphere_I():
    """Solid sphere: I = (2/5)mr^2."""
    m, r = 10.0, 0.5
    I = moment_of_inertia_sphere(m, r, sigma=0)
    assert I == pytest.approx(0.4 * m * r ** 2)


def test_hollow_sphere_I():
    """Hollow sphere: I = (2/3)mr^2."""
    m, r = 6.0, 0.3
    I = moment_of_inertia_hollow_sphere(m, r, sigma=0)
    expected = (2.0 / 3.0) * m * r ** 2
    assert I == pytest.approx(expected)


def test_cylinder_I():
    """Solid cylinder: I = (1/2)mr^2."""
    m, r = 4.0, 0.2
    I = moment_of_inertia_cylinder(m, r, sigma=0)
    assert I == pytest.approx(0.5 * m * r ** 2)


def test_disk_equals_cylinder():
    """Disk and cylinder have identical I for same mass and radius."""
    m, r = 5.0, 0.4
    I_disk = moment_of_inertia_disk(m, r, sigma=0)
    I_cyl = moment_of_inertia_cylinder(m, r, sigma=0)
    assert I_disk == pytest.approx(I_cyl)


def test_rod_I():
    """Thin rod about center: I = (1/12)ml^2."""
    m, L = 3.0, 1.2
    I = moment_of_inertia_rod(m, L, sigma=0)
    assert I == pytest.approx((1.0 / 12.0) * m * L ** 2)


def test_hollow_sphere_greater_than_solid():
    """Hollow sphere I > solid sphere I (same mass, radius)."""
    m, r = 5.0, 0.3
    I_solid = moment_of_inertia_sphere(m, r, sigma=0)
    I_hollow = moment_of_inertia_hollow_sphere(m, r, sigma=0)
    assert I_hollow > I_solid


# =====================================================================
# 2. PARALLEL AXIS THEOREM
# =====================================================================

def test_parallel_axis_increases_I():
    """Parallel axis: I_new = I_cm + md^2 > I_cm."""
    m, r = 2.0, 0.1
    I_cm = moment_of_inertia_sphere(m, r, sigma=0)
    d = 0.5
    I_offset = parallel_axis(I_cm, m, d, sigma=0)
    assert I_offset > I_cm


def test_parallel_axis_zero_distance():
    """At d=0, parallel axis returns I_cm (with sigma scaling)."""
    I_cm = 0.04
    m = 2.0
    I_offset = parallel_axis(I_cm, m, 0.0, sigma=0)
    assert I_offset == pytest.approx(I_cm)


def test_parallel_axis_known_value():
    """Parallel axis: I = I_cm + md^2 for a known case."""
    I_cm = 1.0
    m, d = 5.0, 2.0
    I_offset = parallel_axis(I_cm, m, d, sigma=0)
    assert I_offset == pytest.approx(1.0 + 5.0 * 4.0)  # 21.0


# =====================================================================
# 3. i_factor — DIMENSIONLESS RATIO
# =====================================================================

def test_i_factor_solid_sphere():
    assert i_factor('solid_sphere') == pytest.approx(2.0 / 5.0)


def test_i_factor_hollow_sphere():
    assert i_factor('hollow_sphere') == pytest.approx(2.0 / 3.0)


def test_i_factor_solid_cylinder():
    assert i_factor('solid_cylinder') == pytest.approx(0.5)


def test_i_factor_disk():
    assert i_factor('disk') == pytest.approx(0.5)


def test_i_factor_thin_ring():
    assert i_factor('thin_ring') == pytest.approx(1.0)


def test_i_factor_unknown_raises():
    """Unknown shape string raises ValueError."""
    with pytest.raises(ValueError, match="Unknown shape"):
        i_factor('banana')


# =====================================================================
# 4. TORQUE
# =====================================================================

def test_torque_perpendicular():
    """Perpendicular force: tau = r * F."""
    F, r = 10.0, 0.5
    tau = torque(F, r)  # angle defaults to pi/2
    assert tau == pytest.approx(F * r)


def test_torque_parallel():
    """Parallel force (angle=0): tau = 0."""
    tau = torque(10.0, 0.5, angle=0.0)
    assert tau == pytest.approx(0.0)


def test_torque_45_degrees():
    """Torque at 45 degrees: tau = rF sin(pi/4)."""
    F, r = 20.0, 1.0
    tau = torque(F, r, angle=math.pi / 4)
    assert tau == pytest.approx(F * r * math.sin(math.pi / 4))


# =====================================================================
# 5. ANGULAR ACCELERATION
# =====================================================================

def test_angular_acceleration_basic():
    """alpha = tau / I."""
    alpha = angular_acceleration(10.0, 2.0)
    assert alpha == pytest.approx(5.0)


def test_angular_acceleration_zero_inertia_raises():
    """I = 0 raises ValueError."""
    with pytest.raises(ValueError, match="positive"):
        angular_acceleration(10.0, 0.0)


def test_angular_acceleration_negative_inertia_raises():
    """I < 0 raises ValueError."""
    with pytest.raises(ValueError, match="positive"):
        angular_acceleration(10.0, -1.0)


# =====================================================================
# 6. ANGULAR MOMENTUM
# =====================================================================

def test_angular_momentum_basic():
    """L = I * omega at sigma=0."""
    L = angular_momentum(4.0, 3.0, sigma=0)
    assert L == pytest.approx(12.0)


def test_angular_momentum_zero_omega():
    """Zero angular velocity gives zero angular momentum."""
    L = angular_momentum(4.0, 0.0, sigma=0)
    assert L == pytest.approx(0.0)


# =====================================================================
# 7. ROLLING KINEMATICS — v = wr
# =====================================================================

def test_rolling_velocity():
    """v = omega * r."""
    v = rolling_velocity(10.0, 0.5)
    assert v == pytest.approx(5.0)


def test_rolling_angular_velocity():
    """omega = v / r."""
    omega = rolling_angular_velocity(5.0, 0.5)
    assert omega == pytest.approx(10.0)


def test_rolling_roundtrip():
    """v -> omega -> v is identity."""
    v0 = 3.7
    r = 0.25
    omega = rolling_angular_velocity(v0, r)
    v_back = rolling_velocity(omega, r)
    assert v_back == pytest.approx(v0)


def test_rolling_angular_velocity_zero_radius_raises():
    """Zero radius raises ValueError."""
    with pytest.raises(ValueError, match="positive"):
        rolling_angular_velocity(5.0, 0.0)


# =====================================================================
# 8. ROLLING DOWN INCLINE
# =====================================================================

def test_solid_sphere_faster_than_hollow():
    """Solid sphere accelerates faster down incline than hollow sphere.

    Solid sphere: a = g sin(theta) / (1 + 2/5) = (5/7) g sin(theta)
    Hollow sphere: a = g sin(theta) / (1 + 2/3) = (3/5) g sin(theta)
    5/7 > 3/5, so solid sphere wins.
    """
    angle = math.pi / 6
    a_solid = rolling_acceleration_incline(angle, 'solid_sphere')
    a_hollow = rolling_acceleration_incline(angle, 'hollow_sphere')
    assert a_solid > a_hollow


def test_solid_sphere_incline_value():
    """Solid sphere: a = (5/7) g sin(theta)."""
    angle = math.pi / 4
    a = rolling_acceleration_incline(angle, 'solid_sphere')
    expected = (5.0 / 7.0) * G_EARTH * math.sin(angle)
    assert a == pytest.approx(expected)


def test_cylinder_incline_value():
    """Cylinder: a = (2/3) g sin(theta)."""
    angle = math.pi / 6
    a = rolling_acceleration_incline(angle, 'solid_cylinder')
    expected = (2.0 / 3.0) * G_EARTH * math.sin(angle)
    assert a == pytest.approx(expected)


def test_rolling_speed_solid_sphere():
    """Solid sphere: v = sqrt(10gh/7)."""
    h = 2.0
    v = rolling_speed_from_height(h, 'solid_sphere')
    expected = math.sqrt(10.0 * G_EARTH * h / 7.0)
    assert v == pytest.approx(expected)


def test_rolling_speed_solid_faster_than_hollow():
    """Solid sphere reaches higher speed than hollow from same height."""
    h = 5.0
    v_solid = rolling_speed_from_height(h, 'solid_sphere')
    v_hollow = rolling_speed_from_height(h, 'hollow_sphere')
    assert v_solid > v_hollow


def test_rolling_speed_zero_height():
    """Zero height gives zero speed."""
    v = rolling_speed_from_height(0.0, 'solid_sphere')
    assert v == pytest.approx(0.0)


def test_rolling_speed_negative_height_raises():
    """Negative height raises ValueError."""
    with pytest.raises(ValueError, match="non-negative"):
        rolling_speed_from_height(-1.0, 'solid_sphere')


# =====================================================================
# 9. ROLLING FRICTION — DISTANCE AND TIME
# =====================================================================

def test_rolling_distance_basic():
    """Known friction scenario."""
    v0 = 10.0
    mu = 0.01
    d = rolling_distance_on_flat(v0, mu)
    expected = v0 ** 2 / (2.0 * mu * G_EARTH)
    assert d == pytest.approx(expected)


def test_rolling_time_basic():
    """t = v0 / (mu * g)."""
    v0 = 10.0
    mu = 0.01
    t = rolling_time_on_flat(v0, mu)
    expected = v0 / (mu * G_EARTH)
    assert t == pytest.approx(expected)


def test_zero_friction_infinite_distance():
    """Zero friction means infinite rolling distance."""
    d = rolling_distance_on_flat(5.0, 0.0)
    assert d == float('inf')


def test_zero_friction_infinite_time():
    """Zero friction means infinite rolling time."""
    t = rolling_time_on_flat(5.0, 0.0)
    assert t == float('inf')


def test_negative_friction_infinite_distance():
    """Negative friction coefficient treated as zero friction."""
    d = rolling_distance_on_flat(5.0, -0.01)
    assert d == float('inf')


# =====================================================================
# 10. RAMP TO FLAT DISTANCE
# =====================================================================

def test_ramp_to_flat_returns_dict():
    """ramp_to_flat_distance returns a dict with required keys."""
    result = ramp_to_flat_distance(1.0, math.pi / 6, 0.01, 'solid_sphere')
    assert isinstance(result, dict)
    required_keys = [
        'ramp_height_m', 'ramp_angle_rad', 'ramp_length_m',
        'ramp_horizontal_m', 'exit_speed_m_s', 'flat_distance_m',
        'flat_time_s', 'total_horizontal_m', 'shape', 'i_factor',
        'rolling_friction', 'origin',
    ]
    for key in required_keys:
        assert key in result, f"Missing key: {key}"


def test_ramp_to_flat_exit_speed():
    """Exit speed matches rolling_speed_from_height."""
    h = 2.0
    result = ramp_to_flat_distance(h, math.pi / 4, 0.01, 'solid_sphere')
    v_expected = rolling_speed_from_height(h, 'solid_sphere')
    assert result['exit_speed_m_s'] == pytest.approx(v_expected)


def test_ramp_to_flat_total_horizontal():
    """Total horizontal = ramp horizontal + flat distance."""
    result = ramp_to_flat_distance(1.0, math.pi / 6, 0.02, 'solid_cylinder')
    total = result['ramp_horizontal_m'] + result['flat_distance_m']
    assert result['total_horizontal_m'] == pytest.approx(total)


def test_ramp_to_flat_shape_preserved():
    """Shape string is preserved in output."""
    result = ramp_to_flat_distance(1.0, math.pi / 4, 0.01, 'hollow_sphere')
    assert result['shape'] == 'hollow_sphere'


def test_ramp_to_flat_higher_ramp_longer_distance():
    """Higher ramp -> faster exit -> longer flat distance."""
    r1 = ramp_to_flat_distance(1.0, math.pi / 4, 0.01, 'solid_sphere')
    r2 = ramp_to_flat_distance(2.0, math.pi / 4, 0.01, 'solid_sphere')
    assert r2['flat_distance_m'] > r1['flat_distance_m']


# =====================================================================
# 11. SHAPE-BASED FUNCTIONS (Sphere instance)
# =====================================================================

def test_shape_moment_of_inertia_sphere():
    """Shape-based I for a sphere matches analytical I = (2/5)mr^2."""
    s = Sphere(0.1)  # 10cm radius
    m = 2.0
    I = shape_moment_of_inertia(s, m, axis='z', sigma=0)
    I_analytical = (2.0 / 5.0) * m * 0.1 ** 2
    assert I == pytest.approx(I_analytical)


def test_shape_moment_of_inertia_rejects_non_shape():
    """Passing a non-Shape object raises TypeError."""
    with pytest.raises(TypeError, match="Shape instance"):
        shape_moment_of_inertia("sphere", 2.0, axis='z', sigma=0)


def test_shape_rolling_acceleration_sphere():
    """Shape-based rolling acceleration for a sphere."""
    s = Sphere(0.1)
    angle = math.pi / 6
    a = shape_rolling_acceleration(s, angle, axis='z')
    # For sphere: c = inertia_factor / r^2 = (2/5)*r^2 / r^2 = 2/5
    expected = G_EARTH * math.sin(angle) / (1.0 + 2.0 / 5.0)
    assert a == pytest.approx(expected)


def test_shape_rolling_acceleration_rejects_non_shape():
    """Passing a non-Shape raises TypeError."""
    with pytest.raises(TypeError, match="Shape instance"):
        shape_rolling_acceleration("sphere", math.pi / 6)


def test_shape_rolling_speed_from_height_sphere():
    """Shape-based rolling speed matches analytical for sphere."""
    s = Sphere(0.1)
    h = 3.0
    v = shape_rolling_speed_from_height(s, h, axis='z')
    # c = 2/5 for sphere
    v_expected = math.sqrt(2.0 * G_EARTH * h / (1.0 + 2.0 / 5.0))
    assert v == pytest.approx(v_expected)


def test_shape_rolling_speed_rejects_non_shape():
    """Passing a non-Shape raises TypeError."""
    with pytest.raises(TypeError, match="Shape instance"):
        shape_rolling_speed_from_height("not_a_shape", 1.0)


def test_shape_rolling_speed_negative_height_raises():
    """Negative height raises ValueError even with Shape."""
    s = Sphere(0.1)
    with pytest.raises(ValueError, match="non-negative"):
        shape_rolling_speed_from_height(s, -1.0)


# =====================================================================
# 12. SIGMA COUPLING — INERTIA SHIFTS
# =====================================================================

def test_sigma_inertia_shift_at_zero():
    """At sigma=0, I is unchanged: scale_ratio(0) = 1."""
    I0 = 5.0
    I_shifted = sigma_inertia_shift(I0, sigma=0)
    assert I_shifted == pytest.approx(I0)


def test_sigma_inertia_shift_scales():
    """At sigma=1, I scales by e (Euler's number)."""
    I0 = 5.0
    I_shifted = sigma_inertia_shift(I0, sigma=1.0)
    assert I_shifted == pytest.approx(I0 * math.e)


def test_moment_of_inertia_sphere_sigma_scaling():
    """Sphere I at sigma=1 is e times I at sigma=0."""
    m, r = 3.0, 0.2
    I0 = moment_of_inertia_sphere(m, r, sigma=0)
    I1 = moment_of_inertia_sphere(m, r, sigma=1.0)
    assert I1 == pytest.approx(I0 * math.e)


def test_angular_momentum_sigma_scaling():
    """Angular momentum scales with sigma through I."""
    I0 = 2.0
    omega = 5.0
    L0 = angular_momentum(I0, omega, sigma=0)
    L1 = angular_momentum(I0, omega, sigma=1.0)
    assert L1 == pytest.approx(L0 * math.e)


def test_parallel_axis_sigma_scaling():
    """Parallel axis at sigma=1 scales both I_cm and md^2 by e."""
    I_cm = 1.0
    m, d = 2.0, 0.5
    I0 = parallel_axis(I_cm, m, d, sigma=0)
    I1 = parallel_axis(I_cm, m, d, sigma=1.0)
    assert I1 == pytest.approx(I0 * math.e)


# =====================================================================
# 13. ROTATIONAL PROPERTIES — NAGATHA EXPORT
# =====================================================================

def test_rotational_properties_returns_dict():
    """rotational_properties returns a dict with expected keys."""
    props = rotational_properties(2.0, 0.1, 'solid_sphere', 10.0, sigma=0)
    assert isinstance(props, dict)
    required_keys = [
        'mass_kg', 'radius_m', 'shape', 'i_factor',
        'moment_of_inertia_kg_m2', 'angular_velocity_rad_s',
        'angular_momentum_kg_m2_s', 'rotational_ke_J',
        'rolling_velocity_m_s', 'sigma', 'origin',
    ]
    for key in required_keys:
        assert key in props, f"Missing key: {key}"


def test_rotational_properties_I_value():
    """I in props matches hand calculation."""
    m, r = 2.0, 0.1
    props = rotational_properties(m, r, 'solid_sphere', 10.0, sigma=0)
    I_expected = 0.4 * m * r ** 2
    assert props['moment_of_inertia_kg_m2'] == pytest.approx(I_expected)


def test_rotational_properties_angular_momentum():
    """L = I * omega."""
    m, r, omega = 2.0, 0.1, 10.0
    props = rotational_properties(m, r, 'solid_sphere', omega, sigma=0)
    I = 0.4 * m * r ** 2
    assert props['angular_momentum_kg_m2_s'] == pytest.approx(I * omega)


def test_rotational_properties_rotational_ke():
    """KE_rot = (1/2) I omega^2."""
    m, r, omega = 2.0, 0.1, 10.0
    props = rotational_properties(m, r, 'solid_sphere', omega, sigma=0)
    I = 0.4 * m * r ** 2
    assert props['rotational_ke_J'] == pytest.approx(0.5 * I * omega ** 2)


def test_rotational_properties_rolling_velocity():
    """Rolling velocity = omega * r."""
    omega, r = 10.0, 0.1
    props = rotational_properties(2.0, r, 'solid_sphere', omega, sigma=0)
    assert props['rolling_velocity_m_s'] == pytest.approx(omega * r)


def test_rotational_properties_zero_omega():
    """Zero angular velocity gives zero L, KE, and rolling v."""
    props = rotational_properties(2.0, 0.1, 'solid_sphere', 0.0, sigma=0)
    assert props['angular_momentum_kg_m2_s'] == pytest.approx(0.0)
    assert props['rotational_ke_J'] == pytest.approx(0.0)
    assert props['rolling_velocity_m_s'] == pytest.approx(0.0)


def test_rotational_properties_has_origin():
    """Origin tag is a non-empty string."""
    props = rotational_properties(2.0, 0.1, 'solid_sphere', 10.0, sigma=0)
    assert isinstance(props['origin'], str)
    assert len(props['origin']) > 0
