"""Astronomy tools: planet/star/named-body data via astropy + curated tables.

Provides quick lookup of standard solar-system body parameters and a
limited set of named bright stars. For deep astronomy (cosmological
distances, redshift catalogs, etc.), the user should use astropy
directly outside the MCP scope.
"""

from __future__ import annotations

from sigma_ground.field.constants import M_SUN_KG
from sigma_ground.mcp.provenance import ToolResult


# Solar-system body parameters. Sources: NASA Planetary Fact Sheets,
# IAU 2015 nominal values. All quantities in SI.
# Each entry: {mass_kg, radius_m, semimajor_axis_au_or_km, orbital_period_d,
#              source}
_SOLAR_SYSTEM_BODIES: dict[str, dict] = {
    "sun": {
        # M = GM☉/G from field.constants (IAU 2015 nominal GM☉). Was 1.98892e30,
        # the nominal GM over an obsolete G, which put GM 2.57e-4 high.
        "mass_kg":          M_SUN_KG,
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
        # 7.342e22 was the pre-2024 Fact Sheet value, 5.2e-4 below DE440's GM / G (4902.800118 km^3/s^2 / 6.6743e-11 = 7.34579e22 kg); corrected 2026-10-25, see body_constants and tests/test_body_data.py
        "mass_kg":           7.3458e22,
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

    Accepts formal name, common nickname, and mythological name.
    Examples:
      "earth" / "Terra" / "Gaia" / "our planet"
      "mars" / "Ares" / "Red Planet"
      "moon" / "Luna" / "Earth's Moon"
      "venus" / "Morning Star" / "Evening Star"
      "jupiter" / "Zeus" / "Jove"
      "sun" / "Sol" / "Helios"

    Returns a dict of (parameter -> value) with the data source.
    """
    from sigma_ground.mcp.tools.aliases import BODY_ALIASES, resolve
    key = resolve(body_name, BODY_ALIASES)
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
    notes = ("All values in SI unless noted. semimajor_axis_au = AU = "
             "1.496e11 m. orbital_period_d = days.")
    # The same field constants the N-body predictor reads (fixtures/de440_header_constants.json): DE440's own header for the Sun, Earth and Moon, the Juno and Cassini
    # papers for Jupiter and Saturn. radius_m above is the Fact Sheet's mean (or IAU nominal) radius; the J coefficients are defined on field_reference_radius_km, so use that one with them.
    from sigma_ground.field.interface import de440_constants
    dc = de440_constants.body(key)
    if dc is not None:
        for k_out, k_in in (("gm_km3_s2", "gm_km3_s2"), ("field_reference_radius_km", "reference_radius_km"), ("j2", "j2"), ("j3", "j3"), ("j4", "j4")):
            if k_in in dc:
                data[k_out] = dc[k_in]
        data["field_constants_source"] = dc["source"]
        notes += (" gm_km3_s2, j2, j3, j4 are the gravity-field constants of the N-body predictor (field_constants_source); they are defined on field_reference_radius_km,"
                  " which differs from radius_m (a mean radius).")
    return ToolResult(
        value=data,
        source=f"sigma-ground via {src}",
        provenance_tag="VERIFIED",
        inputs={"body_name": body_name},
        notes=notes,
    )


def named_star(star_name: str) -> ToolResult:
    """Look up parameters for a named star.

    Accepts common name, Bayer designation, nickname, and Greek-letter
    forms. Examples:
      "sirius" / "Sirius" / "α CMa" / "Alpha Canis Majoris" / "Dog Star"
      "alpha_centauri" / "α Cen" / "Rigil Kentaurus" / "Toliman"
      "betelgeuse" / "α Ori" / "Beetlejuice"
      "vega" / "α Lyr" / "Alpha Lyrae"
      "polaris" / "North Star" / "Pole Star" / "α UMi"
      "proxima_centauri" / "Proxima" / "nearest star"
    """
    from sigma_ground.mcp.tools.aliases import STAR_ALIASES, resolve
    key = resolve(star_name, STAR_ALIASES)
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


def body_constants(body_name: str) -> ToolResult:
    """Everything the n-body work holds about one body: GM, mass, radius, gravity field, orbit and physical data, each with its source.

    Covers the Sun, the planets (planet alone and, separately, planet plus moons), their moons (the 23 added from Horizons included), DE440's 343 asteroids by name or number and its Kuiper-belt objects.
    A GM from JPL is exact to its published digits; the mass in kg is GM / G and carries G's uncertainty (2.2e-5), so ask for the GM when an exact value matters.
    A moon with no published mass is reported as "mass unknown", never as zero.
    """
    from sigma_ground.field.interface import body_data
    entry, close = body_data.resolve(body_name)
    if entry is None:
        return ToolResult(
            value=None,
            source="sigma-ground body catalogue (DE440 header, satellite ephemerides, Horizons headers)",
            notes=(f"'{body_name}' is not in the catalogue." + (f" Close names: {', '.join(close)}." if close else "")
                   + " It holds the Sun, planets, 40+ moons, DE440's 343 asteroids (by name or number) and its Kuiper-belt objects."),
            inputs={"body_name": body_name},
        )
    d = body_data.describe(entry)
    jpl = "gm_km3_s2" in d and not str(d["sources"].get("gm_km3_s2", "")).startswith("predictor table")
    notes = ["GM (km^3/s^2) is the exact quantity; mass_kg = GM / G is derived and inherits G's relative uncertainty of 2.2e-5.", *d["notes"]]
    return ToolResult(
        value=d,
        units="GM in km^3/s^2, mass in kg, radii in km (see the field names)",
        source="sigma-ground: JPL DE440 header and satellite ephemerides (JUP365, SAT441, URA111, NEP095, PLU058, MAR097), JPL Horizons headers; per-field sources in value['sources']",
        provenance_tag="VERIFIED" if jpl else "EMPIRICAL-INPUT",
        notes=" ".join(notes),
        inputs={"body_name": body_name},
    )


def body_position(body_name: str, when: str, central_body: str = "sun") -> ToolResult:
    """Where is a body at a given time? A known value from the past, or a labelled prediction.

    Sun, planets and the Moon (Mars to Pluto as planet-system barycentres). `when` is an ISO date or date-time (UTC), "JD 2462000.5" (TDB) or "now". `central_body` is the body the position is measured from: "sun" (default), "ssb" (the solar-system barycentre) or "earth".
    Up to today the value is JPL's DE440 (labelled known; its fitting data end about 2020). After today it is our own N-body prediction from DE440's state at 2026-01-01, with JPL's extrapolation beside it,
    the difference, and the error this model measured against DE440 at that lead. A prediction is not an observation and not a claim to beat JPL.
    """
    from sigma_ground.field.interface import where_when
    try:
        d = where_when.position_at(body_name, when, central_body)
    except where_when.WhereWhenError as e:
        return ToolResult(value=None, source="sigma-ground body_position (JPL DE440 kernel; our N-body forecast table)", notes=str(e), inputs={"body_name": body_name, "when": when, "central_body": central_body})
    known = d["kind"] == "known"
    tag = "VERIFIED" if known and not d.get("beyond_fit_data", False) and d["method"].startswith("JPL") else "DERIVED"
    notes = f"{d['kind'].upper()}. {d['basis']}."
    if d["kind"] == "prediction" and d.get("expected_distance_from_de440"):
        e = d["expected_distance_from_de440"]
        notes += f" Measured by hindcast, this model lands typically {e['typical_km']:.3g} km (largest seen {e['largest_seen_km']:.3g} km) from DE440 at this lead." + (f" {e['note']}." if e.get("note") else "")
    notes += " " + " ".join(d["notes"])
    return ToolResult(value=d, units="km (position), km/s (velocity)", source=d["method"], provenance_tag=tag, notes=notes.strip(), inputs={"body_name": body_name, "when": when, "central_body": central_body})
