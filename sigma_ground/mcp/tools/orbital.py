"""Orbital mechanics & two-body gravity — body-aware, multi-step tools.

These tools exist to collapse the chains the benchmark kept failing on.
Standard tools like circular_orbit_velocity require a raw central mass in
kg and a radius from the body's center. But real questions arrive as
"how fast is the ISS going at 408 km altitude?" — a body NAME plus an
ALTITUDE. The LLM (or a pre-classifier) then had to: look up Earth's
mass, look up Earth's radius, add the altitude, then call the formula.
Three steps the library should own.

So every tool here accepts natural inputs:
  - a central-body NAME ("earth", "sun", "jupiter") OR an explicit mass
  - altitude_km above the surface OR orbital_radius_m from center OR
    semimajor_axis_au

and does the lookups + unit conversions internally. "It's just math" —
so the library does the math, and the LLM only has to name the body and
the altitude.
"""

from __future__ import annotations

import math

from sigma_ground.mcp.provenance import ToolResult

_AU_M = 1.495978707e11
_M_SUN = 1.98892e30


def _resolve_central(central_body: str | None,
                       central_mass_kg: float | None
                       ) -> tuple[float | None, float | None, str]:
    """Return (mass_kg, radius_m, label). radius_m is the body's physical
    radius (for altitude math); None if unknown. Raises nothing — returns
    (None, None, reason) on failure.
    """
    if central_mass_kg is not None:
        return float(central_mass_kg), None, f"mass={central_mass_kg:g} kg"
    if central_body:
        from sigma_ground.mcp.tools.astronomy import _SOLAR_SYSTEM_BODIES
        from sigma_ground.mcp.tools.aliases import BODY_ALIASES, resolve
        key = resolve(central_body, BODY_ALIASES)
        data = _SOLAR_SYSTEM_BODIES.get(key)
        if data:
            return data.get("mass_kg"), data.get("radius_m"), key
    return None, None, "unresolved central body"


def orbital_velocity(central_body: str | None = None,
                       central_mass_kg: float | None = None,
                       altitude_km: float | None = None,
                       orbital_radius_m: float | None = None,
                       semimajor_axis_au: float | None = None) -> ToolResult:
    """Circular orbital speed v = sqrt(G M / r), with body + altitude lookup.

    Provide the central body ONE of these ways:
      - central_body="earth" (name; mass & radius looked up internally)
      - central_mass_kg=5.972e24 (explicit)

    Provide the orbit radius ONE of these ways (checked in this order):
      - altitude_km: height ABOVE the body's surface (radius added for you)
      - orbital_radius_m: distance from the body's CENTER
      - semimajor_axis_au: for heliocentric orbits (× 1 AU)

    Examples:
      orbital_velocity("earth", altitude_km=408)        # ISS
      orbital_velocity("earth", altitude_km=35786)      # geostationary
      orbital_velocity("sun", semimajor_axis_au=5.2038) # Jupiter
      orbital_velocity("earth", orbital_radius_m=3.84e8)# Moon
    """
    from sigma_ground.field.constants import G
    mass_kg, body_radius_m, label = _resolve_central(central_body,
                                                        central_mass_kg)
    if mass_kg is None or mass_kg <= 0:
        return ToolResult(value=None, source="invalid input",
                           notes="Provide central_body (name) or central_mass_kg.",
                           inputs={"central_body": central_body})

    if orbital_radius_m is not None:
        r = float(orbital_radius_m)
        rsrc = "orbital_radius_m (from center)"
    elif altitude_km is not None:
        if body_radius_m is None:
            return ToolResult(value=None, source="missing body radius",
                               notes=("altitude_km needs the body's radius; "
                                      "pass orbital_radius_m instead, or use a "
                                      "named body with known radius."),
                               inputs={"central_body": central_body,
                                       "altitude_km": altitude_km})
        r = body_radius_m + float(altitude_km) * 1000.0
        rsrc = f"radius {body_radius_m:g} m + altitude {altitude_km} km"
    elif semimajor_axis_au is not None:
        r = float(semimajor_axis_au) * _AU_M
        rsrc = f"{semimajor_axis_au} AU"
    else:
        return ToolResult(value=None, source="invalid input",
                           notes=("Provide altitude_km, orbital_radius_m, "
                                  "or semimajor_axis_au."),
                           inputs={"central_body": central_body})

    if r <= 0:
        return ToolResult(value=None, source="invalid input",
                           inputs={"radius_m": r})
    v = math.sqrt(G * mass_kg / r)
    return ToolResult(
        value=v, units="m/s",
        source="sigma-ground (Keplerian circular orbit)",
        formula="v = sqrt(G M / r)",
        inputs={"central_body": label, "central_mass_kg": mass_kg,
                "orbital_radius_m": r},
        notes=f"Central: {label}. Radius from {rsrc}. Newtonian.",
    )


def orbital_period(semimajor_axis_au: float | None = None,
                     semimajor_axis_m: float | None = None,
                     central_body: str | None = "sun",
                     central_mass_kg: float | None = None) -> ToolResult:
    """Kepler's third law: T = 2π sqrt(a³ / (G M)).

    Defaults to the Sun as the central body, so a heliocentric orbit only
    needs its semi-major axis:
      orbital_period(semimajor_axis_au=3.0)   # asteroid at 3 AU -> ~5.2 yr

    Returns the period in SECONDS (convert to days/years as needed).
    """
    from sigma_ground.field.constants import G
    mass_kg, _, label = _resolve_central(central_body, central_mass_kg)
    if mass_kg is None or mass_kg <= 0:
        return ToolResult(value=None, source="invalid input",
                           notes="Provide central_body or central_mass_kg.",
                           inputs={"central_body": central_body})
    if semimajor_axis_m is not None:
        a = float(semimajor_axis_m)
        asrc = f"{semimajor_axis_m:g} m"
    elif semimajor_axis_au is not None:
        a = float(semimajor_axis_au) * _AU_M
        asrc = f"{semimajor_axis_au} AU"
    else:
        return ToolResult(value=None, source="invalid input",
                           notes="Provide semimajor_axis_au or semimajor_axis_m.",
                           inputs={})
    if a <= 0:
        return ToolResult(value=None, source="invalid input",
                           inputs={"a": a})
    T = 2.0 * math.pi * math.sqrt(a ** 3 / (G * mass_kg))
    return ToolResult(
        value=T, units="s",
        source="sigma-ground (Kepler's third law)",
        formula="T = 2π sqrt(a³ / (G M))",
        inputs={"central_body": label, "central_mass_kg": mass_kg,
                "semimajor_axis_m": a},
        notes=(f"Central: {label}. a from {asrc}. "
                f"T = {T/86400:.4g} days = {T/(365.25*86400):.4g} years."),
    )


def gravitational_force(mass1_kg: float, mass2_kg: float,
                          separation_m: float) -> ToolResult:
    """Newton's law of universal gravitation: F = G m1 m2 / r².

    Examples:
      gravitational_force(5.972e24, 7.342e22, 3.844e8)  # Earth-Moon
    """
    from sigma_ground.field.constants import G
    if mass1_kg <= 0 or mass2_kg <= 0 or separation_m <= 0:
        return ToolResult(value=None, source="invalid input",
                           inputs={"mass1_kg": mass1_kg, "mass2_kg": mass2_kg,
                                   "separation_m": separation_m})
    F = G * mass1_kg * mass2_kg / (separation_m ** 2)
    return ToolResult(
        value=F, units="N",
        source="sigma-ground (Newtonian gravitation)",
        formula="F = G m1 m2 / r²",
        inputs={"mass1_kg": mass1_kg, "mass2_kg": mass2_kg,
                "separation_m": separation_m},
        notes="Attractive force between two point/spherical masses.",
    )


def orbital_raise_energy(mass_kg: float,
                           central_body: str | None = "earth",
                           central_mass_kg: float | None = None,
                           from_altitude_km: float | None = 0.0,
                           to_altitude_km: float | None = None,
                           from_radius_m: float | None = None,
                           to_radius_m: float | None = None) -> ToolResult:
    """Gravitational potential energy needed to raise a mass between orbits.

    ΔU = G M m (1/r₁ − 1/r₂), with r measured from the body's center.
    Altitudes are above the surface (radius added internally for named bodies).

    Example (lift 1000 kg from Earth's surface to geostationary):
      orbital_raise_energy(1000, "earth", from_altitude_km=0,
                             to_altitude_km=35786)

    NOTE: this is the change in gravitational PE only (the energy to raise
    it quasi-statically); it is NOT the full energy to inject into a
    circular orbit (which also needs orbital kinetic energy).
    """
    from sigma_ground.field.constants import G
    mass_central, body_radius_m, label = _resolve_central(central_body,
                                                            central_mass_kg)
    if mass_central is None or mass_central <= 0:
        return ToolResult(value=None, source="invalid input",
                           notes="Provide central_body or central_mass_kg.",
                           inputs={"central_body": central_body})

    def _radius(alt_km, rad_m):
        if rad_m is not None:
            return float(rad_m)
        if alt_km is not None:
            if body_radius_m is None:
                return None
            return body_radius_m + float(alt_km) * 1000.0
        return None

    r1 = _radius(from_altitude_km, from_radius_m)
    r2 = _radius(to_altitude_km, to_radius_m)
    if r1 is None or r2 is None or r1 <= 0 or r2 <= 0:
        return ToolResult(value=None, source="invalid input",
                           notes=("Need both radii. Use altitudes with a named "
                                  "body, or pass from_radius_m/to_radius_m."),
                           inputs={"r1": r1, "r2": r2})
    dU = G * mass_central * mass_kg * (1.0 / r1 - 1.0 / r2)
    return ToolResult(
        value=dU, units="J",
        source="sigma-ground (gravitational PE difference)",
        formula="ΔU = G M m (1/r₁ − 1/r₂)",
        inputs={"central_body": label, "mass_kg": mass_kg,
                "r1_m": r1, "r2_m": r2},
        notes=(f"Central: {label}. PE change only (not orbital KE). "
                f"r1={r1:.4g} m, r2={r2:.4g} m."),
    )


def surface_gravity(mass_kg: float, radius_m: float) -> ToolResult:
    """Surface gravitational acceleration of a body: g = G M / R².

    Example (Mars): surface_gravity(6.39e23, 3.3895e6) -> 3.71 m/s².
    """
    if mass_kg <= 0 or radius_m <= 0:
        return ToolResult(value=None, source="invalid input",
                           inputs={"mass_kg": mass_kg, "radius_m": radius_m})
    from sigma_ground.field.constants import G
    g = G * mass_kg / (radius_m ** 2)
    return ToolResult(
        value=g, units="m/s^2",
        source="sigma-ground (Newtonian surface gravity)",
        formula="g = G M / R²",
        inputs={"mass_kg": mass_kg, "radius_m": radius_m},
        notes="Surface gravitational acceleration of a spherical body.",
    )
