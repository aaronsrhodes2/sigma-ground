"""Free-surface fluid dynamics: buoyancy, wind shear stress, and capillary-
gravity (wind-ripple) waves.

Built for the wind-over-a-pond scenario (WATER_SIM_PLAN.md): a wind blowing
across water exerts a tangential surface stress that excites capillary-gravity
ripples; a body's density decides how deep it floats (or that it sinks). These
are standard analytic results (Golden Rule 6 — prefer the exact closed form);
the water property seeds (σ, ρ) come from liquid_water.py (NIST-anchored).

References:
  - Archimedes' principle (buoyancy).
  - Lamb, *Hydrodynamics* (1932) §267 — capillary-gravity dispersion.
  - Large & Pond, J. Phys. Oceanogr. 11, 324 (1981) — neutral drag C_d ≈ 1.3e-3.
"""
from __future__ import annotations

import math

# Standard gravity (defined value, 3rd CGPM 1901). Not a fundamental constant,
# so it lives here rather than in constants.py (cf. gas.py's local _GRAVITY).
STANDARD_GRAVITY = 9.80665  # m/s^2


# ── Buoyancy (Archimedes) ────────────────────────────────────────────────────

def submerged_fraction(body_density: float, fluid_density: float) -> float:
    """Fraction of a body's volume below the surface at static equilibrium.

    f = ρ_body / ρ_fluid  (Archimedes). A floating body displaces its own weight
    of fluid, so f is the density ratio. If ρ_body ≥ ρ_fluid the body sinks (no
    static floating equilibrium); this returns 1.0 (fully submerged) in that case.

    Args:
        body_density: body density (kg/m³)
        fluid_density: fluid density (kg/m³)

    Returns:
        Submerged volume fraction (0-1).
    """
    if fluid_density <= 0:
        raise ValueError(f"fluid_density={fluid_density} ≤ 0")
    f = body_density / fluid_density
    return f if f < 1.0 else 1.0


def floats(body_density: float, fluid_density: float) -> bool:
    """True iff the body floats (ρ_body < ρ_fluid)."""
    return body_density < fluid_density


# ── Wind surface stress ──────────────────────────────────────────────────────

def wind_shear_stress(wind_speed: float, air_density: float = 1.225,
                      drag_coefficient: float = 1.3e-3) -> float:
    """Tangential (skin-friction) wind stress on a water surface (Pa).

    τ = C_d · ρ_air · U²

    Bulk aerodynamic formula; C_d ≈ 1.3e-3 is the neutral 10 m drag coefficient
    for low-to-moderate wind (Large & Pond 1981). At 5 m/s, ρ_air=1.225:
    τ ≈ 0.040 Pa.

    Args:
        wind_speed: wind speed U (m/s)
        air_density: air density (kg/m³)
        drag_coefficient: dimensionless C_d

    Returns:
        Surface shear stress in Pa.
    """
    return drag_coefficient * air_density * wind_speed ** 2


def friction_velocity(stress: float, fluid_density: float) -> float:
    """Friction velocity u* = √(τ/ρ) in the water (m/s)."""
    if fluid_density <= 0:
        raise ValueError(f"fluid_density={fluid_density} ≤ 0")
    return math.sqrt(max(stress, 0.0) / fluid_density)


# ── Capillary-gravity (ripple) waves ─────────────────────────────────────────

def capillary_gravity_phase_speed(wavelength: float, surface_tension: float,
                                  density: float,
                                  g: float = STANDARD_GRAVITY) -> float:
    """Phase speed of a deep-water capillary-gravity wave (m/s).

    c = √(g/k + σk/ρ),   k = 2π/λ

    Gravity dominates at long λ (c grows with λ); surface tension dominates at
    short λ (c grows as λ shrinks). Between them sits a minimum (see
    capillary_gravity_min_phase_speed). Lamb, Hydrodynamics §267.

    Args:
        wavelength: wavelength λ (m)
        surface_tension: σ (N/m)
        density: fluid density ρ (kg/m³)
        g: gravitational acceleration (m/s²)

    Returns:
        Phase speed in m/s.
    """
    k = 2.0 * math.pi / wavelength
    return math.sqrt(g / k + surface_tension * k / density)


def capillary_gravity_angular_frequency(wavelength: float, surface_tension: float,
                                        density: float,
                                        g: float = STANDARD_GRAVITY) -> float:
    """Angular frequency of a deep-water capillary-gravity wave (rad/s).

    ω = √(g·k + σ·k³/ρ),   k = 2π/λ   (the dispersion relation).
    """
    k = 2.0 * math.pi / wavelength
    return math.sqrt(g * k + surface_tension * k ** 3 / density)


def capillary_gravity_group_speed(wavelength: float, surface_tension: float,
                                  density: float,
                                  g: float = STANDARD_GRAVITY) -> float:
    """Group speed c_g = dω/dk = (g + 3σk²/ρ) / (2ω) (m/s)."""
    k = 2.0 * math.pi / wavelength
    omega = math.sqrt(g * k + surface_tension * k ** 3 / density)
    return (g + 3.0 * surface_tension * k ** 2 / density) / (2.0 * omega)


def capillary_gravity_min_phase_speed(surface_tension: float, density: float,
                                      g: float = STANDARD_GRAVITY):
    """The slowest possible surface wave and the wavelength at which it occurs.

    Minimising c²(k)=g/k+σk/ρ gives k_min = √(ρg/σ), so:
      λ_min = 2π·√(σ/(ρg))
      c_min = √2 · (gσ/ρ)^(1/4)

    For water this is the capillary-gravity crossover: c_min ≈ 0.23 m/s at
    λ_min ≈ 1.7 cm. A wind must blow faster than c_min to raise ripples — this is
    why a breeze first roughens water at ~centimetre scales.

    Returns:
        (c_min in m/s, λ_min in m).
    """
    k_min = math.sqrt(density * g / surface_tension)
    lambda_min = 2.0 * math.pi / k_min
    c_min = math.sqrt(2.0) * (g * surface_tension / density) ** 0.25
    return c_min, lambda_min
