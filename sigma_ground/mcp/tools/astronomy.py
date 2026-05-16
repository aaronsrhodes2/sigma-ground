"""Astronomy tools: planet/star/named-body data via astropy + curated tables.

Provides quick lookup of standard solar-system body parameters and a
limited set of named bright stars. For deep astronomy (cosmological
distances, redshift catalogs, etc.), the user should use astropy
directly outside the MCP scope.
"""

from __future__ import annotations

from sigma_ground.mcp.provenance import ToolResult


# Solar-system body parameters. Sources: NASA Planetary Fact Sheets,
# IAU 2015 nominal values. All quantities in SI.
# Each entry: {mass_kg, radius_m, semimajor_axis_au_or_km, orbital_period_d,
#              source}
_SOLAR_SYSTEM_BODIES: dict[str, dict] = {
    "sun": {
        "mass_kg":          1.98892e30,
        "radius_m":         6.957e8,
        "luminosity_w":     3.828e26,
        "surface_temp_k":   5772.0,
        "source": "IAU 2015 nominal solar values",
    },
    "mercury": {
        "mass_kg":           3.3011e23,
        "radius_m":          2.4397e6,
        "semimajor_axis_au": 0.387098,
        "orbital_period_d":  87.969,
        "source": "NASA Planetary Fact Sheet 2024",
    },
    "venus": {
        "mass_kg":           4.8675e24,
        "radius_m":          6.0518e6,
        "semimajor_axis_au": 0.723332,
        "orbital_period_d":  224.701,
        "source": "NASA Planetary Fact Sheet 2024",
    },
    "earth": {
        "mass_kg":           5.97219e24,
        "radius_m":          6.3781e6,
        "semimajor_axis_au": 1.000001018,
        "orbital_period_d":  365.256,
        "surface_g_ms2":     9.80665,
        "source": "NASA Planetary Fact Sheet 2024",
    },
    "moon": {
        "mass_kg":           7.342e22,
        "radius_m":          1.7374e6,
        "semimajor_axis_km": 384399.0,
        "orbital_period_d":  27.3217,
        "surface_g_ms2":     1.625,
        "source": "NASA Moon Fact Sheet 2024",
    },
    "mars": {
        "mass_kg":           6.4171e23,
        "radius_m":          3.3895e6,
        "semimajor_axis_au": 1.523679,
        "orbital_period_d":  686.971,
        "surface_g_ms2":     3.711,
        "source": "NASA Planetary Fact Sheet 2024",
    },
    "jupiter": {
        "mass_kg":           1.89813e27,
        "radius_m":          6.9911e7,
        "semimajor_axis_au": 5.2038,
        "orbital_period_d":  4332.59,
        "source": "NASA Planetary Fact Sheet 2024",
    },
    "saturn": {
        "mass_kg":           5.6834e26,
        "radius_m":          5.8232e7,
        "semimajor_axis_au": 9.5826,
        "orbital_period_d":  10759.22,
        "source": "NASA Planetary Fact Sheet 2024",
    },
    "uranus": {
        "mass_kg":           8.6810e25,
        "radius_m":          2.5362e7,
        "semimajor_axis_au": 19.2184,
        "orbital_period_d":  30688.5,
        "source": "NASA Planetary Fact Sheet 2024",
    },
    "neptune": {
        "mass_kg":           1.02413e26,
        "radius_m":          2.4622e7,
        "semimajor_axis_au": 30.07,
        "orbital_period_d":  60182.0,
        "source": "NASA Planetary Fact Sheet 2024",
    },
    "pluto": {
        "mass_kg":           1.303e22,
        "radius_m":          1.1883e6,
        "semimajor_axis_au": 39.48,
        "orbital_period_d":  90560.0,
        "source": "NASA Planetary Fact Sheet 2024",
    },
}

# Bright named stars. Distances in parsec, absolute V magnitude.
# SIMBAD / Hipparcos / Gaia DR3.
_NAMED_STARS: dict[str, dict] = {
    "sirius_a": {
        "distance_pc":        2.667,
        "abs_v_mag":           1.43,
        "spectral_type":      "A1V",
        "mass_solar":          2.063,
        "luminosity_solar":   25.4,
        "source": "Hipparcos + Gaia DR3 (alpha Canis Majoris)",
    },
    "alpha_centauri_a": {
        "distance_pc":        1.347,
        "abs_v_mag":           4.38,
        "spectral_type":      "G2V",
        "mass_solar":          1.10,
        "luminosity_solar":   1.52,
        "source": "Gaia DR3",
    },
    "betelgeuse": {
        "distance_pc":        168.0,
        "abs_v_mag":          -5.85,
        "spectral_type":      "M2Iab",
        "mass_solar":         11.6,
        "luminosity_solar":   126000.0,
        "source": "Gaia DR3 + literature",
    },
    "rigel": {
        "distance_pc":        264.0,
        "abs_v_mag":          -7.84,
        "spectral_type":      "B8Iab",
        "mass_solar":         21.0,
        "luminosity_solar":   120000.0,
        "source": "Gaia DR3 + literature",
    },
    "vega": {
        "distance_pc":        7.68,
        "abs_v_mag":           0.58,
        "spectral_type":      "A0V",
        "mass_solar":          2.135,
        "luminosity_solar":   40.12,
        "source": "Hipparcos + Gaia DR3",
    },
    "polaris": {
        "distance_pc":        133.0,
        "abs_v_mag":          -3.6,
        "spectral_type":      "F7Ib",
        "mass_solar":          5.4,
        "luminosity_solar":   1260.0,
        "source": "Gaia DR3",
    },
    "proxima_centauri": {
        "distance_pc":        1.301,
        "abs_v_mag":          15.60,
        "spectral_type":      "M5.5Ve",
        "mass_solar":          0.122,
        "luminosity_solar":   0.0017,
        "source": "Gaia DR3 (nearest known star to the Sun)",
    },
}


def solar_system_body(body_name: str) -> ToolResult:
    """Look up canonical parameters for a solar-system body.

    Examples: 'earth', 'mars', 'jupiter', 'moon', 'sun', 'pluto'.

    Returns a dict of (parameter -> value) with the data source.
    """
    key = body_name.lower().strip().replace(" ", "_")
    if key not in _SOLAR_SYSTEM_BODIES:
        return ToolResult(
            value=None,
            source="sigma-ground (solar system body lookup)",
            notes=(f"'{body_name}' not in lookup. Available: "
                    f"{sorted(_SOLAR_SYSTEM_BODIES.keys())}"),
            inputs={"body_name": body_name},
        )
    data = dict(_SOLAR_SYSTEM_BODIES[key])
    src = data.pop("source")
    return ToolResult(
        value=data,
        source=f"sigma-ground via {src}",
        provenance_tag="VERIFIED",
        inputs={"body_name": body_name},
        notes=("All values in SI unless noted. semimajor_axis_au = AU = "
                "1.496e11 m. orbital_period_d = days."),
    )


def named_star(star_name: str) -> ToolResult:
    """Look up parameters for a named star.

    Examples: 'sirius_a', 'vega', 'betelgeuse', 'alpha_centauri_a',
    'proxima_centauri', 'polaris', 'rigel'.
    """
    key = star_name.lower().strip().replace(" ", "_")
    if key not in _NAMED_STARS:
        return ToolResult(
            value=None,
            source="sigma-ground (named star lookup)",
            notes=(f"'{star_name}' not in lookup. Available: "
                    f"{sorted(_NAMED_STARS.keys())}"),
            inputs={"star_name": star_name},
        )
    data = dict(_NAMED_STARS[key])
    src = data.pop("source")
    return ToolResult(
        value=data,
        source=f"sigma-ground via {src}",
        provenance_tag="VERIFIED",
        inputs={"star_name": star_name},
        notes=("distance_pc = parsecs (1 pc = 3.086e16 m). "
                "abs_v_mag = absolute visual magnitude. "
                "luminosity_solar = L / L_sun."),
    )


def light_travel_time(distance_m: float) -> ToolResult:
    """Time for light to traverse a given distance in vacuum: t = d / c."""
    from sigma_ground.field.constants import C
    if distance_m < 0:
        return ToolResult(value=None, source="invalid input",
                           inputs={"distance_m": distance_m})
    t = distance_m / C
    # Helpful conversions
    yrs = t / (365.25 * 86400.0)
    return ToolResult(
        value=t,
        units="s",
        source="sigma-ground (light travel time)",
        formula="t = d / c",
        inputs={"distance_m": distance_m},
        notes=(f"~{yrs:.4f} years. Sun-Earth (1 AU): 8.32 min. "
                f"Earth-Moon: 1.28 s. Alpha Centauri (4.24 ly): 4.24 yr."),
    )


def parsec_to_meters() -> ToolResult:
    """1 parsec definition: 3.0857e16 m exactly (IAU 2015)."""
    return ToolResult(
        value=3.0856775814913673e16,
        units="m",
        source="sigma-ground via IAU 2015 nominal parsec definition",
        formula="1 pc = 648000 / pi AU",
        inputs={},
        notes="IAU 2015 resolution B2: 1 pc = (648000/pi) AU exactly.",
    )


def astronomical_unit_to_meters() -> ToolResult:
    """1 AU definition: 1.495978707e11 m exactly (IAU 2012)."""
    return ToolResult(
        value=1.495978707e11,
        units="m",
        source="sigma-ground via IAU 2012 resolution B2 (exact)",
        inputs={},
        notes="Exact by IAU definition since 2012.",
    )


def list_bodies() -> ToolResult:
    """List all solar-system bodies and named stars available."""
    return ToolResult(
        value={
            "solar_system_bodies": sorted(_SOLAR_SYSTEM_BODIES.keys()),
            "named_stars":         sorted(_NAMED_STARS.keys()),
        },
        source="sigma-ground (astronomy lookup index)",
        notes=("For arbitrary cosmological queries (luminosity distances "
                "at given redshift, galaxy catalogs, etc.), use astropy "
                "directly."),
    )
