"""
Projectile motion and inclined plane physics.

Derivation chains:

  1. Projectile range, height, time of flight
     R = v₀² sin(2θ) / g
     H = v₀² sin²(θ) / (2g)
     T = 2v₀ sin(θ) / g
     FIRST_PRINCIPLES: Newton (1687), parabolic motion under constant g.

  2. Projectile with drag
     ma = -mg ĵ - ½ρCdAv² v̂
     FIRST_PRINCIPLES: Newton + empirical drag (Prandtl, 1904).
     Numerical integration required — no closed form.

  3. Inclined plane
     a = g(sin θ - μ cos θ)  for sliding
     FIRST_PRINCIPLES: Newton's second law with normal force constraint.

  σ-dependence:
    Projectile range R = v₀² sin(2θ)/g is σ-independent (no mass).
    Drag force ∝ ρ_fluid × v² — fluid density may shift at extreme σ.
    Incline acceleration is σ-independent (mass cancels).
"""

import math

from ..constants import SIGMA_HERE
from ..scale import scale_ratio


# ── Constants ────────────────────────────────────────────────────────

G_EARTH = 9.80665    # m/s², standard gravity (MEASURED, NIST)
RHO_AIR = 1.225      # kg/m³, sea-level air density at 15°C (MEASURED, ISA)


# ── Ideal Projectile (no drag) ───────────────────────────────────────

def projectile_range(v0, angle, g=G_EARTH):
    """Horizontal range of a projectile on flat ground.

    R = v₀² sin(2θ) / g

    FIRST_PRINCIPLES: parabolic trajectory under constant g.
    Maximum range at θ = 45°.

    Args:
        v0: launch speed in m/s
        angle: launch angle in radians (0 to π/2)
        g: gravitational acceleration (m/s²)

    Returns:
        Range in metres.
    """
    return v0 ** 2 * math.sin(2 * angle) / g


def projectile_max_height(v0, angle, g=G_EARTH):
    """Maximum height of a projectile.

    H = v₀² sin²(θ) / (2g)

    Args:
        v0: launch speed in m/s
        angle: launch angle in radians
        g: gravitational acceleration (m/s²)

    Returns:
        Maximum height in metres.
    """
    return v0 ** 2 * math.sin(angle) ** 2 / (2.0 * g)


def projectile_time_of_flight(v0, angle, g=G_EARTH):
    """Total time of flight for a projectile on flat ground.

    T = 2v₀ sin(θ) / g

    Args:
        v0: launch speed in m/s
        angle: launch angle in radians
        g: gravitational acceleration (m/s²)

    Returns:
        Time of flight in seconds.
    """
    return 2.0 * v0 * math.sin(angle) / g


def projectile_trajectory(v0, angle, steps=100, g=G_EARTH):
    """Compute trajectory points for an ideal projectile.

    Returns list of (x, y, t) tuples from launch to landing.

    Args:
        v0: launch speed in m/s
        angle: launch angle in radians
        steps: number of trajectory points
        g: gravitational acceleration (m/s²)

    Returns:
        List of (x, y, t) tuples in metres and seconds.
    """
    T = projectile_time_of_flight(v0, angle, g)
    vx = v0 * math.cos(angle)
    vy = v0 * math.sin(angle)
    trajectory = []
    for i in range(steps + 1):
        t = T * i / steps
        x = vx * t
        y = vy * t - 0.5 * g * t ** 2
        if y < 0 and i > 0:
            y = 0.0
        trajectory.append((x, max(y, 0.0), t))
    return trajectory


# ── Projectile with Drag ─────────────────────────────────────────────

def drag_force(velocity, Cd, area, rho_fluid=RHO_AIR):
    """Aerodynamic drag force F_d = ½ρCdAv².

    FIRST_PRINCIPLES: Prandtl (1904), boundary layer theory.

    Args:
        velocity: speed in m/s
        Cd: drag coefficient (dimensionless)
        area: cross-sectional area in m²
        rho_fluid: fluid density in kg/m³

    Returns:
        Drag force in N (opposing motion).
    """
    return 0.5 * rho_fluid * Cd * area * velocity ** 2


def terminal_velocity(mass, Cd, area, rho_fluid=RHO_AIR, g=G_EARTH,
                      sigma=SIGMA_HERE):
    """Terminal velocity where drag = weight.

    v_t = √(2mg / (ρCdA))

    FIRST_PRINCIPLES: force balance at steady state.

    Args:
        mass: mass in kg
        Cd: drag coefficient
        area: cross-sectional area in m²
        rho_fluid: fluid density in kg/m³
        g: gravitational acceleration (m/s²)
        sigma: σ-field value

    Returns:
        Terminal velocity in m/s.
    """
    m = mass * scale_ratio(sigma)
    return math.sqrt(2.0 * m * g / (rho_fluid * Cd * area))


def projectile_with_drag(v0, angle, mass, Cd, area,
                         rho_fluid=RHO_AIR, g=G_EARTH, dt=0.001,
                         sigma=SIGMA_HERE):
    """Projectile trajectory with quadratic drag (numerical).

    Uses Euler integration. Stops when y ≤ 0 after launch.

    FIRST_PRINCIPLES + NUMERICAL: Newton's second law with
    drag F_d = ½ρCdAv² opposing velocity.

    Args:
        v0: launch speed in m/s
        angle: launch angle in radians
        mass: mass in kg
        Cd: drag coefficient
        area: cross-sectional area in m²
        rho_fluid: fluid density in kg/m³
        g: gravitational acceleration (m/s²)
        dt: time step in seconds
        sigma: σ-field value

    Returns:
        Dict with trajectory, range, max_height, time_of_flight.
    """
    m = mass * scale_ratio(sigma)
    vx = v0 * math.cos(angle)
    vy = v0 * math.sin(angle)
    x, y, t = 0.0, 0.0, 0.0
    trajectory = [(x, y, t)]
    max_h = 0.0

    max_steps = int(1000.0 / dt)  # safety limit: 1000 seconds
    for _ in range(max_steps):
        v = math.sqrt(vx ** 2 + vy ** 2)
        if v > 0:
            fd = 0.5 * rho_fluid * Cd * area * v ** 2
            ax = -fd * vx / (m * v)
            ay = -g - fd * vy / (m * v)
        else:
            ax = 0.0
            ay = -g

        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt
        t += dt

        if y > max_h:
            max_h = y

        if y <= 0 and t > dt:
            # Interpolate landing
            y = 0.0
            trajectory.append((x, 0.0, t))
            break

        trajectory.append((x, y, t))

    return {
        'trajectory': trajectory,
        'range_m': x,
        'max_height_m': max_h,
        'time_of_flight_s': t,
        'impact_velocity_m_s': math.sqrt(vx ** 2 + vy ** 2),
        'drag_coefficient': Cd,
        'origin': (
            'FIRST_PRINCIPLES + NUMERICAL: Newton with quadratic drag. '
            'Euler integration, dt={:.4f}s. '
            'σ-shift through mass scaling.'.format(dt)
        ),
    }


# ── Inclined Plane ───────────────────────────────────────────────────

def incline_acceleration(angle, mu_friction=0.0, g=G_EARTH):
    """Acceleration of a sliding object on an inclined plane.

    a = g(sin θ - μ cos θ)

    FIRST_PRINCIPLES: Newton's second law with normal force.
    Returns 0 if friction prevents motion (μ ≥ tan θ).

    Args:
        angle: incline angle in radians
        mu_friction: kinetic friction coefficient
        g: gravitational acceleration (m/s²)

    Returns:
        Acceleration in m/s² (0 if friction holds).
    """
    a = g * (math.sin(angle) - mu_friction * math.cos(angle))
    return max(a, 0.0)


def incline_critical_angle(mu_friction):
    """Minimum angle for sliding to begin: θ_c = arctan(μ).

    FIRST_PRINCIPLES: balance of gravitational and friction forces.

    Args:
        mu_friction: static friction coefficient

    Returns:
        Critical angle in radians.
    """
    return math.atan(mu_friction)


def incline_sliding_distance(v0, angle, mu_friction, g=G_EARTH):
    """Distance traveled up an incline before stopping.

    d = v₀² / (2g(sin θ + μ cos θ))

    FIRST_PRINCIPLES: work-energy theorem.

    Args:
        v0: initial speed up the incline in m/s
        angle: incline angle in radians
        mu_friction: kinetic friction coefficient
        g: gravitational acceleration (m/s²)

    Returns:
        Distance along incline in metres.
    """
    decel = g * (math.sin(angle) + mu_friction * math.cos(angle))
    if decel <= 0:
        return float('inf')  # net downhill force, won't stop
    return v0 ** 2 / (2.0 * decel)


def incline_speed_at_bottom(height, angle, mu_friction=0.0, g=G_EARTH):
    """Speed at the bottom of an incline (sliding, no rotation).

    v = √(2gh(1 - μ/tan θ))

    FIRST_PRINCIPLES: energy conservation with friction loss.

    Args:
        height: vertical height in m
        angle: incline angle in radians
        mu_friction: kinetic friction coefficient
        g: gravitational acceleration (m/s²)

    Returns:
        Speed in m/s. Returns 0 if friction prevents sliding.
    """
    if angle <= 0:
        return 0.0
    tan_theta = math.tan(angle)
    if tan_theta <= 0:
        return 0.0
    factor = 1.0 - mu_friction / tan_theta
    if factor <= 0:
        return 0.0  # friction too high, doesn't slide
    return math.sqrt(2.0 * g * height * factor)


# ── σ-field coupling ─────────────────────────────────────────────────

def sigma_terminal_velocity_ratio(sigma):
    """Ratio of terminal velocity at σ to σ=0.

    v_t ∝ √m, so v_t(σ)/v_t(0) = √(scale_ratio(σ)).

    Heavier objects at high σ fall faster through air.

    Args:
        sigma: σ-field value

    Returns:
        v_t(σ) / v_t(0), dimensionless.
    """
    return math.sqrt(scale_ratio(sigma))


# ── Nagatha Export ───────────────────────────────────────────────────

def projectile_report(v0, angle, mass=1.0, Cd=0.0, area=0.0,
                      rho_fluid=RHO_AIR, g=G_EARTH, sigma=SIGMA_HERE):
    """Export projectile analysis in Nagatha-compatible format.

    If Cd > 0 and area > 0, includes drag calculation.

    Args:
        v0: launch speed in m/s
        angle: launch angle in radians
        mass: mass in kg
        Cd: drag coefficient (0 for ideal)
        area: cross-sectional area in m²
        rho_fluid: fluid density in kg/m³
        g: gravitational acceleration (m/s²)
        sigma: σ-field value

    Returns:
        Dict with trajectory analysis and origin tags.
    """
    result = {
        'v0_m_s': v0,
        'angle_rad': angle,
        'angle_deg': math.degrees(angle),
        'ideal_range_m': projectile_range(v0, angle, g),
        'ideal_max_height_m': projectile_max_height(v0, angle, g),
        'ideal_time_of_flight_s': projectile_time_of_flight(v0, angle, g),
        'sigma': sigma,
    }

    if Cd > 0 and area > 0:
        drag_result = projectile_with_drag(
            v0, angle, mass, Cd, area, rho_fluid, g, sigma=sigma
        )
        result['drag_range_m'] = drag_result['range_m']
        result['drag_max_height_m'] = drag_result['max_height_m']
        result['drag_time_of_flight_s'] = drag_result['time_of_flight_s']
        result['range_reduction_pct'] = (
            (1 - drag_result['range_m'] / result['ideal_range_m']) * 100
            if result['ideal_range_m'] > 0 else 0
        )

    result['origin'] = (
        'FIRST_PRINCIPLES: Newtonian parabolic trajectory. '
        'Range R = v₀²sin(2θ)/g. '
        + ('Drag correction via numerical integration. ' if Cd > 0 else '')
        + 'σ-independent for ideal case (mass cancels).'
    )
    return result
