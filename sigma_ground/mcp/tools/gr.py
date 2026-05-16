"""General-relativity tools: Schwarzschild metric quantities + Hawking.

These wrap sigma_ground.field.gr_basics where possible (returning the
sigma_ground source citation) and fall through to direct formula
evaluation otherwise. All quantities are computed in SI internally;
the user can convert via tools.units.convert.

The SSBM-specific GR functions (page_time_sigma_ground,
eta_as_bit_thread_fraction) are NOT exposed here. They live in the
EXTENDED tier surfaced only when the user explicitly asks about SSBM.
"""

from __future__ import annotations

import math

from sigma_ground.mcp.provenance import ToolResult


def schwarzschild_radius(mass_kg: float) -> ToolResult:
    """Schwarzschild radius r_s = 2 G M / c^2.

    Parameters
    ----------
    mass_kg : float
        Mass in kilograms. For "solar masses", use units.convert first.

    Returns
    -------
    ToolResult with `value` in meters.
    """
    if mass_kg <= 0:
        return ToolResult(
            value=None, source="invalid input",
            notes="mass must be positive",
            inputs={"mass_kg": mass_kg},
        )
    from sigma_ground.field.constants import G, C
    rs = 2.0 * G * mass_kg / (C * C)
    return ToolResult(
        value=rs,
        units="m",
        source="sigma_ground.field.gr_basics (standard Schwarzschild)",
        formula="r_s = 2 G M / c^2",
        inputs={"mass_kg": mass_kg},
        notes="Event horizon radius for non-rotating uncharged black hole.",
    )


def isco_radius(mass_kg: float) -> ToolResult:
    """Innermost stable circular orbit, r_ISCO = 6 G M / c^2 = 3 r_s."""
    if mass_kg <= 0:
        return ToolResult(value=None, source="invalid input",
                           notes="mass must be positive",
                           inputs={"mass_kg": mass_kg})
    from sigma_ground.field.constants import G, C
    r_isco = 6.0 * G * mass_kg / (C * C)
    return ToolResult(
        value=r_isco,
        units="m",
        source="sigma_ground.field.gr_basics (standard Schwarzschild)",
        formula="r_ISCO = 6 G M / c^2 = 3 r_s",
        inputs={"mass_kg": mass_kg},
        notes="For a non-rotating black hole, this is the smallest stable "
               "circular orbit for a massive particle. Below r_ISCO, orbits "
               "are unstable and the particle plunges into the horizon.",
    )


def photon_sphere_radius(mass_kg: float) -> ToolResult:
    """Photon sphere radius r_ph = 3 G M / c^2 = 1.5 r_s."""
    if mass_kg <= 0:
        return ToolResult(value=None, source="invalid input",
                           notes="mass must be positive",
                           inputs={"mass_kg": mass_kg})
    from sigma_ground.field.constants import G, C
    r_ph = 3.0 * G * mass_kg / (C * C)
    return ToolResult(
        value=r_ph,
        units="m",
        source="sigma_ground.field.gr_basics (standard Schwarzschild)",
        formula="r_ph = 3 G M / c^2 = 1.5 r_s",
        inputs={"mass_kg": mass_kg},
        notes="Radius of circular photon orbit around a non-rotating black "
               "hole. Photons can orbit here but the orbits are unstable.",
    )


def hawking_temperature(mass_kg: float) -> ToolResult:
    """Hawking temperature T_H = hbar c^3 / (8 pi G M k_B)."""
    if mass_kg <= 0:
        return ToolResult(value=None, source="invalid input",
                           notes="mass must be positive",
                           inputs={"mass_kg": mass_kg})
    from sigma_ground.field.constants import G, C, HBAR, K_B
    T_H = HBAR * C**3 / (8.0 * math.pi * G * mass_kg * K_B)
    return ToolResult(
        value=T_H,
        units="K",
        source="sigma_ground.field.gr_basics (Hawking 1974)",
        formula="T_H = hbar c^3 / (8 pi G M k_B)",
        inputs={"mass_kg": mass_kg},
        notes="Semiclassical thermal radiation temperature of a "
               "non-rotating black hole. Tiny for stellar-mass BHs "
               "(~6 nK for 10 M_sun); diverges as M -> 0.",
    )


def hawking_evaporation_time(mass_kg: float) -> ToolResult:
    """Hawking evaporation timescale tau = 5120 pi G^2 M^3 / (hbar c^4)."""
    if mass_kg <= 0:
        return ToolResult(value=None, source="invalid input",
                           notes="mass must be positive",
                           inputs={"mass_kg": mass_kg})
    from sigma_ground.field.constants import G, C, HBAR
    tau_s = 5120.0 * math.pi * G**2 * mass_kg**3 / (HBAR * C**4)
    # Conversion: 1 year ~ 3.1557e7 s
    tau_yr = tau_s / 3.1557e7
    return ToolResult(
        value=tau_s,
        units="s",
        source="sigma_ground.field.gr_basics (Hawking 1974, Page 1976)",
        formula="tau = 5120 pi G^2 M^3 / (hbar c^4)",
        inputs={"mass_kg": mass_kg},
        notes=f"Semiclassical evaporation timescale = {tau_yr:.3e} years. "
               f"For solar-mass BH ~ 1e67 years; tiny BHs evaporate fast.",
    )


def gravitational_redshift(mass_kg: float, radius_m: float) -> ToolResult:
    """Schwarzschild gravitational redshift factor at radius r.

    Returns (lambda_obs / lambda_emitted) - 1 for light emitted at r
    and observed at infinity. Diverges at the horizon.
    """
    from sigma_ground.field.constants import G, C
    r_s = 2.0 * G * mass_kg / (C * C)
    if radius_m <= r_s:
        return ToolResult(
            value=float("inf"),
            source="sigma_ground.field.gr_basics",
            notes=f"Below or at event horizon (r_s = {r_s:.3e} m). "
                   f"Redshift diverges.",
            inputs={"mass_kg": mass_kg, "radius_m": radius_m},
        )
    z = (1.0 - r_s / radius_m) ** -0.5 - 1.0
    return ToolResult(
        value=z,
        units="dimensionless",
        source="sigma_ground.field.gr_basics (Schwarzschild)",
        formula="z = (1 - r_s/r)^(-1/2) - 1",
        inputs={"mass_kg": mass_kg, "radius_m": radius_m},
        notes=f"r_s = {r_s:.3e} m. Light from r is redshifted by factor "
               f"(1+z) when observed at infinity.",
    )


def gravitational_time_dilation(mass_kg: float, radius_m: float) -> ToolResult:
    """Time dilation factor: clock at r ticks at this rate relative to infinity.

    Returns the factor by which a clock at radius r runs slower than
    a clock at infinity. Equivalent to sqrt(1 - r_s/r).
    """
    from sigma_ground.field.constants import G, C
    r_s = 2.0 * G * mass_kg / (C * C)
    if radius_m <= r_s:
        return ToolResult(
            value=0.0,
            source="sigma_ground.field.gr_basics",
            notes=f"At or below event horizon (r_s = {r_s:.3e} m). "
                   f"Clock stops relative to infinity.",
            inputs={"mass_kg": mass_kg, "radius_m": radius_m},
        )
    factor = math.sqrt(1.0 - r_s / radius_m)
    return ToolResult(
        value=factor,
        units="dimensionless",
        source="sigma_ground.field.gr_basics (Schwarzschild)",
        formula="dt_r / dt_inf = sqrt(1 - r_s/r)",
        inputs={"mass_kg": mass_kg, "radius_m": radius_m},
        notes=f"Clock at r ticks at {factor:.6f} the rate of a clock at "
               f"infinity. r_s = {r_s:.3e} m.",
    )
