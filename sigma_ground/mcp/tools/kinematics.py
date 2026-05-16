"""Classical-mechanics kinematics: free fall, projectile, energy, momentum.

Pure formulas — no external library dependencies beyond the standard
math module and the sigma-ground constants. Every tool returns a
ToolResult with the formula echoed so the LLM can explain the math.

Standard surface gravity g = 9.80665 m/s^2 (defined exactly, GRS-80
WGS-84 reference, ISO 80000-3). For non-Earth bodies, use the value
explicitly via lookup_constant or the astronomy tools.
"""

from __future__ import annotations

import math

from sigma_ground.mcp.provenance import ToolResult


# Standard gravitational acceleration at Earth's surface (exact, defined).
# Definition: 1 standard gravity = 9.80665 m/s^2 (BIPM, CGPM 1901).
_G_EARTH = 9.80665


def free_fall_time(height_m: float, g_m_s2: float = _G_EARTH) -> ToolResult:
    """Time to fall from rest through a vertical distance under constant g.

    Solves h = (1/2) g t^2 for t. Assumes vacuum (no air resistance);
    for dense small bodies at modest heights, the assumption is good
    to better than 1%. For featherweight objects or long falls, air
    resistance dominates and this answer is wrong.

    Parameters
    ----------
    height_m : float
        Drop height in meters. Must be positive.
    g_m_s2 : float
        Local gravitational acceleration (default: Earth's standard
        9.80665 m/s^2). Use ~1.625 for Moon, ~3.711 for Mars.

    Returns
    -------
    ToolResult with time in seconds.
    """
    if height_m <= 0:
        return ToolResult(value=None, source="invalid input",
                           notes="height_m must be positive",
                           inputs={"height_m": height_m, "g_m_s2": g_m_s2})
    if g_m_s2 <= 0:
        return ToolResult(value=None, source="invalid input",
                           notes="g_m_s2 must be positive",
                           inputs={"height_m": height_m, "g_m_s2": g_m_s2})
    t = math.sqrt(2.0 * height_m / g_m_s2)
    return ToolResult(
        value=t,
        units="s",
        source="sigma-ground (standard kinematics, vacuum approximation)",
        formula="t = sqrt(2 h / g)",
        inputs={"height_m": height_m, "g_m_s2": g_m_s2},
        notes=("Vacuum assumption. For air at sea level, terminal velocity "
                "for a 1 cm copper ball is ~30 m/s -- much higher than the "
                "impact speed at modest heights, so the assumption is good."),
    )


def free_fall_velocity(height_m: float, g_m_s2: float = _G_EARTH) -> ToolResult:
    """Velocity after falling from rest through height h.

    Uses v^2 = 2gh (vacuum). Returns impact speed.
    """
    if height_m <= 0 or g_m_s2 <= 0:
        return ToolResult(value=None, source="invalid input",
                           notes="height_m and g_m_s2 must be positive",
                           inputs={"height_m": height_m, "g_m_s2": g_m_s2})
    v = math.sqrt(2.0 * g_m_s2 * height_m)
    return ToolResult(
        value=v,
        units="m/s",
        source="sigma-ground (standard kinematics, vacuum approximation)",
        formula="v = sqrt(2 g h)",
        inputs={"height_m": height_m, "g_m_s2": g_m_s2},
    )


def projectile_range(initial_speed_m_s: float, launch_angle_deg: float,
                      g_m_s2: float = _G_EARTH) -> ToolResult:
    """Horizontal range of a projectile launched from ground level.

    Uses R = v^2 sin(2 theta) / g. Vacuum approximation.

    Parameters
    ----------
    initial_speed_m_s : float
        Muzzle speed in m/s.
    launch_angle_deg : float
        Launch angle above horizontal, in degrees. Maximum range at 45 deg.
    g_m_s2 : float
        Local gravity (default Earth standard).
    """
    if initial_speed_m_s <= 0 or g_m_s2 <= 0:
        return ToolResult(value=None, source="invalid input",
                           inputs={"initial_speed_m_s": initial_speed_m_s,
                                   "launch_angle_deg": launch_angle_deg,
                                   "g_m_s2": g_m_s2})
    theta_rad = math.radians(launch_angle_deg)
    R = (initial_speed_m_s ** 2) * math.sin(2.0 * theta_rad) / g_m_s2
    return ToolResult(
        value=R,
        units="m",
        source="sigma-ground (standard kinematics, vacuum, level ground)",
        formula="R = v^2 sin(2 theta) / g",
        inputs={"initial_speed_m_s": initial_speed_m_s,
                "launch_angle_deg": launch_angle_deg, "g_m_s2": g_m_s2},
        notes="Vacuum, level ground. Air resistance reduces real range "
               "substantially for fast/small projectiles.",
    )


def projectile_max_height(initial_speed_m_s: float, launch_angle_deg: float,
                           g_m_s2: float = _G_EARTH) -> ToolResult:
    """Maximum height of a projectile launched from ground level.

    Uses h_max = (v sin theta)^2 / (2 g). Vacuum approximation.
    """
    if initial_speed_m_s <= 0 or g_m_s2 <= 0:
        return ToolResult(value=None, source="invalid input",
                           inputs={"initial_speed_m_s": initial_speed_m_s,
                                   "launch_angle_deg": launch_angle_deg,
                                   "g_m_s2": g_m_s2})
    theta_rad = math.radians(launch_angle_deg)
    v_y = initial_speed_m_s * math.sin(theta_rad)
    h_max = v_y ** 2 / (2.0 * g_m_s2)
    return ToolResult(
        value=h_max,
        units="m",
        source="sigma-ground (standard kinematics, vacuum)",
        formula="h_max = (v sin(theta))^2 / (2 g)",
        inputs={"initial_speed_m_s": initial_speed_m_s,
                "launch_angle_deg": launch_angle_deg, "g_m_s2": g_m_s2},
    )


def projectile_flight_time(initial_speed_m_s: float, launch_angle_deg: float,
                            g_m_s2: float = _G_EARTH) -> ToolResult:
    """Total flight time for a projectile (ground-level launch and landing).

    Uses t = 2 v sin(theta) / g.
    """
    if initial_speed_m_s <= 0 or g_m_s2 <= 0:
        return ToolResult(value=None, source="invalid input",
                           inputs={"initial_speed_m_s": initial_speed_m_s,
                                   "launch_angle_deg": launch_angle_deg,
                                   "g_m_s2": g_m_s2})
    theta_rad = math.radians(launch_angle_deg)
    t = 2.0 * initial_speed_m_s * math.sin(theta_rad) / g_m_s2
    return ToolResult(
        value=t,
        units="s",
        source="sigma-ground (standard kinematics, vacuum)",
        formula="t = 2 v sin(theta) / g",
        inputs={"initial_speed_m_s": initial_speed_m_s,
                "launch_angle_deg": launch_angle_deg, "g_m_s2": g_m_s2},
    )


def kinetic_energy(mass_kg: float, velocity_m_s: float) -> ToolResult:
    """Non-relativistic kinetic energy KE = (1/2) m v^2."""
    if mass_kg < 0:
        return ToolResult(value=None, source="invalid input",
                           notes="mass_kg must be non-negative",
                           inputs={"mass_kg": mass_kg,
                                   "velocity_m_s": velocity_m_s})
    KE = 0.5 * mass_kg * velocity_m_s ** 2
    return ToolResult(
        value=KE,
        units="J",
        source="sigma-ground (Newtonian kinematics)",
        formula="KE = (1/2) m v^2",
        inputs={"mass_kg": mass_kg, "velocity_m_s": velocity_m_s},
        notes=("Non-relativistic. For v > 0.1 c (~3e7 m/s) use the "
                "relativistic form K = (gamma - 1) m c^2."),
    )


def momentum(mass_kg: float, velocity_m_s: float) -> ToolResult:
    """Non-relativistic momentum p = m v."""
    if mass_kg < 0:
        return ToolResult(value=None, source="invalid input",
                           notes="mass_kg must be non-negative",
                           inputs={"mass_kg": mass_kg,
                                   "velocity_m_s": velocity_m_s})
    p = mass_kg * velocity_m_s
    return ToolResult(
        value=p,
        units="kg m/s",
        source="sigma-ground (Newtonian kinematics)",
        formula="p = m v",
        inputs={"mass_kg": mass_kg, "velocity_m_s": velocity_m_s},
    )


def gravitational_potential_energy(mass_kg: float, height_m: float,
                                     g_m_s2: float = _G_EARTH) -> ToolResult:
    """Uniform-gravity potential energy U = m g h relative to ground."""
    if mass_kg < 0 or g_m_s2 <= 0:
        return ToolResult(value=None, source="invalid input",
                           inputs={"mass_kg": mass_kg, "height_m": height_m,
                                   "g_m_s2": g_m_s2})
    U = mass_kg * g_m_s2 * height_m
    return ToolResult(
        value=U,
        units="J",
        source="sigma-ground (uniform gravity approximation)",
        formula="U = m g h",
        inputs={"mass_kg": mass_kg, "height_m": height_m, "g_m_s2": g_m_s2},
        notes=("Valid when h is small compared to Earth's radius "
                "(6.371e6 m). For tall orbits use the inverse-square form "
                "U = -G M m / r."),
    )


def friction_stopping_distance(mass_kg: float, initial_velocity_m_s: float,
                                 friction_coefficient: float,
                                 g_m_s2: float = _G_EARTH) -> ToolResult:
    """Distance to stop under kinetic friction (no other forces, level surface).

    Uses d = v^2 / (2 mu g). Mass cancels out in the kinematic formula.
    """
    if friction_coefficient <= 0 or g_m_s2 <= 0:
        return ToolResult(value=None, source="invalid input",
                           inputs={"mass_kg": mass_kg,
                                   "initial_velocity_m_s": initial_velocity_m_s,
                                   "friction_coefficient": friction_coefficient,
                                   "g_m_s2": g_m_s2})
    d = (initial_velocity_m_s ** 2) / (2.0 * friction_coefficient * g_m_s2)
    return ToolResult(
        value=d,
        units="m",
        source="sigma-ground (Coulomb friction model)",
        formula="d = v^2 / (2 mu g)",
        inputs={"mass_kg": mass_kg,
                "initial_velocity_m_s": initial_velocity_m_s,
                "friction_coefficient": friction_coefficient,
                "g_m_s2": g_m_s2},
        notes=("Mass cancels; result depends only on v, mu, g. "
                "Coulomb friction model assumes constant mu, no slip-stick."),
    )


def circular_orbit_velocity(central_mass_kg: float, radius_m: float) -> ToolResult:
    """Circular-orbit speed v = sqrt(G M / r) for a satellite at radius r.

    Parameters
    ----------
    central_mass_kg : float
        Mass of the central body (Earth ~5.972e24, Sun ~1.989e30).
    radius_m : float
        Orbital radius from the center of the central body.
    """
    if central_mass_kg <= 0 or radius_m <= 0:
        return ToolResult(value=None, source="invalid input",
                           inputs={"central_mass_kg": central_mass_kg,
                                   "radius_m": radius_m})
    from sigma_ground.field.constants import G
    v = math.sqrt(G * central_mass_kg / radius_m)
    return ToolResult(
        value=v,
        units="m/s",
        source="sigma-ground (Keplerian circular orbit)",
        formula="v = sqrt(G M / r)",
        inputs={"central_mass_kg": central_mass_kg, "radius_m": radius_m},
        notes=("Newtonian; ignores GR (good to <1e-8 in solar system). "
                "Radius is measured from CENTER of central body, not surface."),
    )


def escape_velocity_classical(mass_kg: float, radius_m: float) -> ToolResult:
    """Escape velocity v_esc = sqrt(2 G M / r) from surface or any radius."""
    if mass_kg <= 0 or radius_m <= 0:
        return ToolResult(value=None, source="invalid input",
                           inputs={"mass_kg": mass_kg, "radius_m": radius_m})
    from sigma_ground.field.constants import G
    v = math.sqrt(2.0 * G * mass_kg / radius_m)
    return ToolResult(
        value=v,
        units="m/s",
        source="sigma-ground (classical escape velocity)",
        formula="v_esc = sqrt(2 G M / r)",
        inputs={"mass_kg": mass_kg, "radius_m": radius_m},
        notes=("Velocity needed to escape gravitational binding to infinity. "
                "For Earth's surface, ~11.2 km/s. For the Sun's surface, "
                "~617 km/s. When v_esc -> c, you're at the Schwarzschild "
                "radius."),
    )
