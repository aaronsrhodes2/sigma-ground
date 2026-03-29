"""
Rotational mechanics — moment of inertia, torque, rolling dynamics.

Derivation chains:

  1. Moment of inertia for rigid bodies
     I = ∫ r² dm
     FIRST_PRINCIPLES: Euler (1750).
     Exact solutions for sphere, cylinder, disk, rod, shell.

  2. Torque and angular acceleration
     τ = Iα = r × F
     FIRST_PRINCIPLES: Newton-Euler equations.

  3. Rolling without slipping
     v = ωr (no-slip constraint)
     a = g sin θ / (1 + I/(mr²))
     FIRST_PRINCIPLES: energy conservation + constraint.

  σ-dependence:
    Moment of inertia I ∝ m (for fixed geometry).
    Mass scales as m(σ) = m₀ × scale_ratio(σ).
    Therefore I(σ) = I₀ × scale_ratio(σ).
    Angular momentum L = Iω and torque τ = Iα both scale.
    Rolling acceleration a = g sinθ / (1 + I/(mr²)) is σ-independent
    because I/(mr²) is a pure geometric ratio (mass cancels).

  Shape integration:
    Moment of inertia DERIVES from geometry. The correct way to compute I
    is: I = mass × shape.inertia_factor(axis). The raw mass+radius
    functions below are convenience wrappers; prefer shape_moment_of_inertia()
    when a Shape object is available.
"""

import math

from ..constants import SIGMA_HERE
from ..scale import scale_ratio
from ...shapes import Shape


# ── Constants ────────────────────────────────────────────────────────

G_EARTH = 9.80665   # m/s², standard gravity (MEASURED, NIST)


# ── Moment of Inertia ───────────────────────────────────────────────

def moment_of_inertia_sphere(mass, radius, sigma=SIGMA_HERE):
    """Moment of inertia of a solid sphere: I = ⅖mr².

    FIRST_PRINCIPLES: volume integral of r²dm over uniform sphere.

    Args:
        mass: mass in kg
        radius: radius in m
        sigma: σ-field value

    Returns:
        Moment of inertia in kg·m².
    """
    m = mass * scale_ratio(sigma)
    return (2.0 / 5.0) * m * radius ** 2


def moment_of_inertia_hollow_sphere(mass, radius, sigma=SIGMA_HERE):
    """Moment of inertia of a thin hollow sphere: I = ⅔mr².

    FIRST_PRINCIPLES: surface integral of r²dm over spherical shell.

    Args:
        mass: mass in kg
        radius: radius in m
        sigma: σ-field value

    Returns:
        Moment of inertia in kg·m².
    """
    m = mass * scale_ratio(sigma)
    return (2.0 / 3.0) * m * radius ** 2


def moment_of_inertia_cylinder(mass, radius, sigma=SIGMA_HERE):
    """Moment of inertia of a solid cylinder about its axis: I = ½mr².

    FIRST_PRINCIPLES: volume integral of r²dm over uniform cylinder.

    Args:
        mass: mass in kg
        radius: radius in m
        sigma: σ-field value

    Returns:
        Moment of inertia in kg·m².
    """
    m = mass * scale_ratio(sigma)
    return 0.5 * m * radius ** 2


def moment_of_inertia_disk(mass, radius, sigma=SIGMA_HERE):
    """Moment of inertia of a thin disk about its axis: I = ½mr².

    Same as solid cylinder (thickness doesn't matter for axial rotation).

    Args:
        mass: mass in kg
        radius: radius in m
        sigma: σ-field value

    Returns:
        Moment of inertia in kg·m².
    """
    return moment_of_inertia_cylinder(mass, radius, sigma)


def moment_of_inertia_rod(mass, length, sigma=SIGMA_HERE):
    """Moment of inertia of a thin rod about its center: I = ¹⁄₁₂ml².

    FIRST_PRINCIPLES: integral of x²(m/L)dx from -L/2 to L/2.

    Args:
        mass: mass in kg
        length: length in m
        sigma: σ-field value

    Returns:
        Moment of inertia in kg·m².
    """
    m = mass * scale_ratio(sigma)
    return (1.0 / 12.0) * m * length ** 2


def parallel_axis(I_cm, mass, distance, sigma=SIGMA_HERE):
    """Parallel axis theorem: I = I_cm + md².

    FIRST_PRINCIPLES: Steiner/Huygens (1673).

    Args:
        I_cm: moment of inertia about center of mass (kg·m²)
        mass: mass in kg
        distance: distance from CM to new axis in m
        sigma: σ-field value

    Returns:
        Moment of inertia about offset axis in kg·m².
    """
    m = mass * scale_ratio(sigma)
    I_cm_shifted = I_cm * scale_ratio(sigma)
    return I_cm_shifted + m * distance ** 2


def i_factor(shape='solid_sphere'):
    """Dimensionless I/(mr²) ratio for standard shapes.

    This ratio appears in rolling dynamics and is purely geometric
    (σ-independent because mass cancels).

    Args:
        shape: 'solid_sphere', 'hollow_sphere', 'solid_cylinder',
               'thin_ring', 'disk'

    Returns:
        I/(mr²) dimensionless ratio.
    """
    factors = {
        'solid_sphere': 2.0 / 5.0,      # ⅖
        'hollow_sphere': 2.0 / 3.0,      # ⅔
        'solid_cylinder': 1.0 / 2.0,     # ½
        'disk': 1.0 / 2.0,               # ½
        'thin_ring': 1.0,                # 1
    }
    if shape not in factors:
        raise ValueError(f"Unknown shape '{shape}'. "
                         f"Available: {sorted(factors.keys())}")
    return factors[shape]


# ── Torque and Angular Dynamics ──────────────────────────────────────

def torque(force, lever_arm, angle=math.pi / 2):
    """Torque τ = rF sin θ (N·m).

    FIRST_PRINCIPLES: Archimedes (3rd century BC), lever principle.

    Args:
        force: force magnitude in N
        lever_arm: distance from pivot in m
        angle: angle between r and F in radians (π/2 for perpendicular)

    Returns:
        Torque in N·m.
    """
    return force * lever_arm * math.sin(angle)


def angular_acceleration(torque_val, inertia):
    """Angular acceleration α = τ/I (rad/s²).

    FIRST_PRINCIPLES: Newton's second law for rotation.

    Args:
        torque_val: net torque in N·m
        inertia: moment of inertia in kg·m²

    Returns:
        Angular acceleration in rad/s².
    """
    if inertia <= 0:
        raise ValueError("Moment of inertia must be positive")
    return torque_val / inertia


def angular_momentum(inertia, angular_velocity, sigma=SIGMA_HERE):
    """Angular momentum L = Iω (kg·m²/s).

    FIRST_PRINCIPLES: rotational analogue of p = mv.

    Args:
        inertia: moment of inertia in kg·m²
        angular_velocity: angular speed in rad/s
        sigma: σ-field value

    Returns:
        Angular momentum in kg·m²/s.
    """
    I = inertia * scale_ratio(sigma)
    return I * angular_velocity


# ── Rolling Dynamics ─────────────────────────────────────────────────

def rolling_velocity(angular_velocity, radius):
    """Translational velocity for rolling without slipping: v = ωr.

    FIRST_PRINCIPLES: no-slip constraint at contact point.

    Args:
        angular_velocity: angular speed in rad/s
        radius: radius in m

    Returns:
        Translational velocity in m/s.
    """
    return angular_velocity * radius


def rolling_angular_velocity(velocity, radius):
    """Angular velocity for rolling without slipping: ω = v/r.

    Args:
        velocity: translational speed in m/s
        radius: radius in m

    Returns:
        Angular velocity in rad/s.
    """
    if radius <= 0:
        raise ValueError("Radius must be positive")
    return velocity / radius


def rolling_acceleration_incline(angle, shape='solid_sphere', g=G_EARTH):
    """Linear acceleration of a rolling body down an incline.

    a = g sin θ / (1 + I/(mr²))

    FIRST_PRINCIPLES: Lagrangian mechanics with rolling constraint.
    The factor I/(mr²) is purely geometric — σ-independent.

    A solid sphere (I/(mr²) = ⅖) accelerates as a = (5/7)g sinθ.
    A hollow sphere (⅔) → a = (3/5)g sinθ.
    A cylinder (½) → a = (2/3)g sinθ.

    Args:
        angle: incline angle in radians
        shape: body shape (see i_factor())
        g: gravitational acceleration (m/s²)

    Returns:
        Linear acceleration in m/s².
    """
    c = i_factor(shape)
    return g * math.sin(angle) / (1.0 + c)


def rolling_speed_from_height(height, shape='solid_sphere', g=G_EARTH):
    """Speed at bottom of incline from energy conservation.

    ½mv² + ½Iω² = mgh
    with v = ωr and I = c·mr²:
    v = √(2gh / (1 + c))

    FIRST_PRINCIPLES: energy conservation + no-slip constraint.
    σ-independent: mass cancels entirely.

    Args:
        height: vertical drop in m
        shape: body shape (see i_factor())
        g: gravitational acceleration (m/s²)

    Returns:
        Speed at bottom in m/s.
    """
    if height < 0:
        raise ValueError("Height must be non-negative")
    c = i_factor(shape)
    return math.sqrt(2.0 * g * height / (1.0 + c))


def rolling_distance_on_flat(v0, rolling_friction_coeff, g=G_EARTH):
    """Distance a rolling body travels on flat ground before stopping.

    Deceleration a = μ_roll × g
    d = v₀² / (2 × μ_roll × g)

    FIRST_PRINCIPLES: work-energy theorem with rolling friction.

    Args:
        v0: initial speed in m/s
        rolling_friction_coeff: μ_roll (dimensionless, typically 0.001–0.1)
        g: gravitational acceleration (m/s²)

    Returns:
        Distance in m before stopping.
    """
    if rolling_friction_coeff <= 0:
        return float('inf')  # no friction → rolls forever
    a = rolling_friction_coeff * g
    return v0 ** 2 / (2.0 * a)


def rolling_time_on_flat(v0, rolling_friction_coeff, g=G_EARTH):
    """Time for a rolling body to stop on flat ground.

    t = v₀ / (μ_roll × g)

    Args:
        v0: initial speed in m/s
        rolling_friction_coeff: μ_roll
        g: gravitational acceleration (m/s²)

    Returns:
        Time in seconds.
    """
    if rolling_friction_coeff <= 0:
        return float('inf')
    a = rolling_friction_coeff * g
    return v0 / a


def ramp_to_flat_distance(ramp_height, ramp_angle, rolling_friction_coeff,
                          shape='solid_sphere', g=G_EARTH):
    """Total horizontal distance: ball rolls down ramp then across flat.

    Combines rolling_speed_from_height → rolling_distance_on_flat.

    Args:
        ramp_height: vertical height of ramp in m
        ramp_angle: ramp angle in radians
        rolling_friction_coeff: μ_roll on flat surface
        shape: body shape
        g: gravitational acceleration (m/s²)

    Returns:
        Dict with ramp_length, exit_speed, flat_distance, total_horizontal.
    """
    v_bottom = rolling_speed_from_height(ramp_height, shape, g)
    ramp_length = ramp_height / math.sin(ramp_angle) if ramp_angle > 0 else 0
    ramp_horizontal = ramp_height / math.tan(ramp_angle) if ramp_angle > 0 else 0
    flat_distance = rolling_distance_on_flat(v_bottom, rolling_friction_coeff, g)
    flat_time = rolling_time_on_flat(v_bottom, rolling_friction_coeff, g)

    return {
        'ramp_height_m': ramp_height,
        'ramp_angle_rad': ramp_angle,
        'ramp_length_m': ramp_length,
        'ramp_horizontal_m': ramp_horizontal,
        'exit_speed_m_s': v_bottom,
        'exit_angular_velocity_rad_s': v_bottom / 1.0 if shape else 0,  # needs radius
        'flat_distance_m': flat_distance,
        'flat_time_s': flat_time,
        'total_horizontal_m': ramp_horizontal + flat_distance,
        'shape': shape,
        'i_factor': i_factor(shape),
        'rolling_friction': rolling_friction_coeff,
        'origin': (
            'FIRST_PRINCIPLES: energy conservation (mgh → ½mv² + ½Iω²) '
            'with no-slip constraint v=ωr, then work-energy theorem '
            'with rolling friction on flat. σ-independent: mass cancels.'
        ),
    }


# ── σ-field coupling ─────────────────────────────────────────────────

def shape_moment_of_inertia(shape, mass, axis='z', sigma=SIGMA_HERE):
    """Moment of inertia from a Shape object — the correct way.

    I = m(σ) × shape.inertia_factor(axis)

    The shape determines the geometry (how mass is distributed).
    Mass determines the scale. σ enters through mass only.

    FIRST_PRINCIPLES: I = ∫ r² dm, evaluated analytically for each
    primitive shape (Euler 1750).

    Args:
        shape: Shape instance (Sphere, Cylinder, Box, Ellipsoid, Cone, etc.)
        mass: mass in kg (at σ=0)
        axis: rotation axis ('x', 'y', or 'z')
        sigma: σ-field value

    Returns:
        Moment of inertia in kg·m².

    Raises:
        TypeError: if shape is not a Shape instance.
    """
    if not isinstance(shape, Shape):
        raise TypeError(f"Expected a Shape instance, got {type(shape).__name__}. "
                        f"Use Sphere(r), Cylinder(r,h), Box(x,y,z), etc.")
    m = mass * scale_ratio(sigma)
    return m * shape.inertia_factor(axis)


def shape_rolling_acceleration(shape, angle, axis='z', g=G_EARTH):
    """Rolling acceleration down an incline, derived from shape geometry.

    a = g sin θ / (1 + I/(mr²))

    The I/(mr²) ratio is shape.inertia_factor(axis) / bounding_radius()².
    This is purely geometric — σ-independent.

    FIRST_PRINCIPLES: Lagrangian mechanics with rolling constraint.

    Args:
        shape: Shape instance
        angle: incline angle in radians
        axis: rotation axis
        g: gravitational acceleration (m/s²)

    Returns:
        Linear acceleration in m/s².
    """
    if not isinstance(shape, Shape):
        raise TypeError(f"Expected a Shape instance, got {type(shape).__name__}")
    r = shape.bounding_radius()
    if r <= 0:
        return g * math.sin(angle)  # point mass
    c = shape.inertia_factor(axis) / (r ** 2)
    return g * math.sin(angle) / (1.0 + c)


def shape_rolling_speed_from_height(shape, height, axis='z', g=G_EARTH):
    """Speed at bottom of incline from energy conservation, using shape geometry.

    ½mv² + ½Iω² = mgh
    with v = ωr and I = c·mr²:
    v = √(2gh / (1 + c))

    where c = inertia_factor / r² is the dimensionless I/(mr²) ratio.

    Args:
        shape: Shape instance
        height: vertical drop in m
        axis: rotation axis
        g: gravitational acceleration (m/s²)

    Returns:
        Speed at bottom in m/s.
    """
    if not isinstance(shape, Shape):
        raise TypeError(f"Expected a Shape instance, got {type(shape).__name__}")
    if height < 0:
        raise ValueError("Height must be non-negative")
    r = shape.bounding_radius()
    if r <= 0:
        return math.sqrt(2.0 * g * height)  # point mass, no rotation
    c = shape.inertia_factor(axis) / (r ** 2)
    return math.sqrt(2.0 * g * height / (1.0 + c))


# ── σ-field coupling ─────────────────────────────────────────────────

def sigma_inertia_shift(inertia, sigma):
    """Moment of inertia at σ: I(σ) = I₀ × scale_ratio(σ).

    I ∝ m for fixed geometry. Mass scales with σ.

    Args:
        inertia: moment of inertia at σ=0 in kg·m²
        sigma: σ-field value

    Returns:
        Moment of inertia at σ in kg·m².
    """
    return inertia * scale_ratio(sigma)


# ── Nagatha Export ───────────────────────────────────────────────────

def rotational_properties(mass, radius, shape='solid_sphere',
                          angular_velocity=0.0, sigma=SIGMA_HERE):
    """Export rotational properties in Nagatha-compatible format.

    Args:
        mass: mass in kg
        radius: radius in m
        shape: body shape
        angular_velocity: angular speed in rad/s
        sigma: σ-field value

    Returns:
        Dict with rotational properties and origin tags.
    """
    c = i_factor(shape)
    m = mass * scale_ratio(sigma)
    I = c * m * radius ** 2
    L = I * angular_velocity
    ke = 0.5 * I * angular_velocity ** 2
    v = angular_velocity * radius

    return {
        'mass_kg': m,
        'radius_m': radius,
        'shape': shape,
        'i_factor': c,
        'moment_of_inertia_kg_m2': I,
        'angular_velocity_rad_s': angular_velocity,
        'angular_momentum_kg_m2_s': L,
        'rotational_ke_J': ke,
        'rolling_velocity_m_s': v,
        'sigma': sigma,
        'origin': (
            f'FIRST_PRINCIPLES: I = {c:.4f}·mr² ({shape}). '
            f'L = Iω. KE_rot = ½Iω². v = ωr (no-slip). '
            f'σ-shift through mass scaling.'
        ),
    }
