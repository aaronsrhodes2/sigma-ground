"""Solar-system reference bodies for Materia's gravity / orbit verbs.

Materia is **tier 3**. The body mass/radius table its orbital math needs lives
otherwise only in ``mcp.tools.astronomy`` (**tier 4**) — which Materia must not
import (the layering contract: a module imports only its own tier or below).

So Materia carries its own small reference table: a handful of well-known
solar-system bodies, with orbits computed from the **tier-1** gravitational
constant ``field.constants.G``. The mass/radius values match IAU /
``mcp.tools.astronomy`` to the digit, so numbers are identical across the
stack — this is reference *data*, not a second physics engine.

A "body" has a minimum reality (no true zero): every entry is a real,
finite-mass, finite-radius object, so ``v = √(GM/r)`` never divides by zero.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..field.constants import G  # tier 1 — the single gravitational constant


@dataclass(frozen=True)
class Body:
    """A point-mass-equivalent reference body: name, mass, mean radius."""
    key: str
    label: str
    mass_kg: float
    radius_m: float

    @property
    def mu(self) -> float:
        """Standard gravitational parameter μ = G·M  (m³/s²)."""
        return G * self.mass_kg

    @property
    def surface_gravity_m_s2(self) -> float:
        """g = G·M / R²  at the body's surface."""
        return G * self.mass_kg / (self.radius_m * self.radius_m)


# Values match mcp.tools.astronomy._SOLAR_SYSTEM_BODIES to the digit so orbital
# numbers are identical whether routed through Materia or the MCP tools.
_BODIES: dict[str, Body] = {
    "sun":     Body("sun", "the Sun", 1.98892e30, 695_700_000.0),
    "mercury": Body("mercury", "Mercury", 3.3011e23, 2_439_700.0),
    "venus":   Body("venus", "Venus", 4.8675e24, 6_051_800.0),
    "earth":   Body("earth", "Earth", 5.97219e24, 6_378_100.0),
    "moon":    Body("moon", "the Moon", 7.342e22, 1_737_400.0),
    "mars":    Body("mars", "Mars", 6.4171e23, 3_389_500.0),
    "jupiter": Body("jupiter", "Jupiter", 1.89813e27, 69_911_000.0),
    "saturn":  Body("saturn", "Saturn", 5.6834e26, 58_232_000.0),
    "uranus":  Body("uranus", "Uranus", 8.681e25, 25_362_000.0),
    "neptune": Body("neptune", "Neptune", 1.02413e26, 24_622_000.0),
    "pluto":   Body("pluto", "Pluto", 1.303e22, 1_188_300.0),
}

# Common aliases the translator may hand us.
_ALIASES = {"earth's": "earth", "the sun": "sun", "the moon": "moon",
            "luna": "moon", "sol": "sun", "terra": "earth"}


def resolve_body(name: str | None) -> Body:
    """Look up a body by (case-insensitive) name; default to Earth.

    Never raises — an unknown name falls back to Earth, because a Materia sim
    always needs *a* body to move around. The label still echoes Earth so the
    answer is honest about what was assumed.
    """
    if not name:
        return _BODIES["earth"]
    key = str(name).strip().lower()
    key = _ALIASES.get(key, key)
    return _BODIES.get(key, _BODIES["earth"])


def circular_orbital_velocity(body: Body, orbital_radius_m: float) -> float:
    """v = √(G·M / r)  for a circular orbit of radius r from the body's centre.

    r is clamped to the body's own radius (you cannot orbit below the surface),
    which also guarantees r > 0 — the atomic/Planck floor for an orbit.
    """
    r = max(float(orbital_radius_m), body.radius_m)
    return math.sqrt(body.mu / r)


def orbital_period(orbital_radius_m: float, body: Body) -> float:
    """Kepler III for a circular orbit: T = 2πr / v = 2π√(r³/μ)."""
    r = max(float(orbital_radius_m), body.radius_m)
    return 2.0 * math.pi * math.sqrt(r * r * r / body.mu)


def escape_velocity(body: Body, from_radius_m: float | None = None) -> float:
    """v_esc = √(2·G·M / r);  r defaults to the body's surface radius."""
    r = body.radius_m if from_radius_m is None else max(float(from_radius_m),
                                                        body.radius_m)
    return math.sqrt(2.0 * body.mu / r)
