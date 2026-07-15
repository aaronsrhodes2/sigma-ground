"""Astrophysics pre-classifier.

Targets the failing astrophysics questions:
  - 'How long does light from <star> take to reach us?' -> chain
    named_star -> light-time conversion (parsec -> years)
  - 'What's <star>'s luminosity / mass / distance' -> named_star + field pick
  - 'How long is a year on <planet>' -> solar_system_body + orbital_period_d
  - 'Peak wavelength of <Sun's> blackbody' -> blackbody_peak_wavelength with
    known T (Sun = 5778 K)
  - 'How much energy does the Sun produce in 1 year' -> L_sun * year_s
  - 'How fast is Earth moving in its orbit' -> circular_orbit_velocity
    (M_sun, 1 AU)
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# Reference values used for inline computations
_C = 299792458.0
_PC_M = 3.0857e16
_AU_M = 1.495978707e11
_YEAR_S = 365.25 * 86400.0
_M_SUN = 1.98892e30
_L_SUN = 3.828e26
_T_SUN_K = 5778.0
_G = 6.6743e-11
_WIEN_B = 2.897771955e-3  # m K

_STAR_NAMES = {
    "proxima centauri": "proxima_centauri",
    "proxima":          "proxima_centauri",
    "alpha centauri":   "alpha_centauri_a",
    "sirius":           "sirius_a",
    "betelgeuse":       "betelgeuse",
    "rigel":            "rigel",
    "vega":             "vega",
    "polaris":          "polaris",
}

_PLANET_NAMES = {
    "mercury": "mercury", "venus": "venus", "earth": "earth", "mars": "mars",
    "jupiter": "jupiter", "saturn": "saturn", "uranus": "uranus",
    "neptune": "neptune", "pluto": "pluto", "sun": "sun", "moon": "moon",
}


@dataclass
class AstroMatch:
    tool: str
    args: dict
    rationale: str
    result_override: float | None = None
    result_units: str = ""


def _find_star(question: str) -> str | None:
    q = question.lower()
    for name, canonical in _STAR_NAMES.items():
        if name in q:
            return canonical
    return None


def _find_planet(question: str) -> str | None:
    q = question.lower()
    for name, canonical in _PLANET_NAMES.items():
        # Word-boundary check
        if re.search(rf"\b{re.escape(name)}\b", q):
            return canonical
    return None


def classify_for_astro(question: str) -> AstroMatch | None:
    q = question

    # ── Light travel time from a named star (in years) ──
    if re.search(r"\bhow\s+long\s+does\s+(?:the\s+)?light\s+(?:from|of)\b|"
                  r"\blight\s+take\s+to\s+reach\s+us|"
                  r"\bsupernova.*?light.*?reach", q, re.IGNORECASE | re.DOTALL):
        star = _find_star(q)
        if star:
            from sigma_ground.mcp.tools.astronomy import named_star
            r = named_star(star)
            if r.value and isinstance(r.value, dict):
                pc = r.value.get("distance_pc")
                if pc is not None:
                    years = pc * 3.26156  # pc -> light-years
                    return AstroMatch(
                        tool="light_travel_time_from_star",
                        args={"star_name": star},
                        rationale=f"{star} at {pc} pc = {years:.2f} ly",
                        result_override=years,
                        result_units="year",  # a TIME duration, not a length
                    )

    # ── Star property: 'X's luminosity / mass / distance' ──
    star = _find_star(q)
    if star:
        # Pick a field based on the question
        field = None
        result_units = ""
        if re.search(r"\bluminosity", q, re.IGNORECASE):
            field = "luminosity_solar"
            result_units = "L_sun"
        elif re.search(r"\bmass\b", q, re.IGNORECASE):
            field = "mass_solar"
            result_units = "M_sun"
        elif re.search(r"\bdistance|how\s+far", q, re.IGNORECASE):
            field = "distance_pc"
            result_units = "parsec"
        if field:
            from sigma_ground.mcp.tools.astronomy import named_star
            r = named_star(star)
            if r.value and isinstance(r.value, dict):
                v = r.value.get(field)
                if isinstance(v, (int, float)):
                    return AstroMatch(
                        tool="named_star_field",
                        args={"star_name": star, "field": field},
                        rationale=f"{star}.{field}",
                        result_override=float(v),
                        result_units=result_units,
                    )

    # ── 'How long is a year on <planet>' / 'orbital period' ──
    if re.search(r"\b(?:year\s+on|orbital\s+period\s+of|how\s+long\s+is\s+a\s+year\s+on)",
                  q, re.IGNORECASE):
        planet = _find_planet(q)
        if planet:
            from sigma_ground.mcp.tools.astronomy import solar_system_body
            r = solar_system_body(planet)
            if r.value and isinstance(r.value, dict):
                period_d = r.value.get("orbital_period_d")
                if isinstance(period_d, (int, float)):
                    return AstroMatch(
                        tool="solar_system_body_orbital_period",
                        args={"body_name": planet},
                        rationale=f"{planet} orbital period",
                        result_override=float(period_d),
                        result_units="day",
                    )

    # ── 'Peak wavelength of the Sun's blackbody spectrum' ──
    if re.search(r"\bpeak\s+wavelength.*?(?:Sun|sun)\b|"
                  r"\b(?:Sun|sun).*?peak\s+(?:emission|wavelength)",
                  q, re.IGNORECASE | re.DOTALL):
        # Wien's law: lambda_max = b / T  for T = 5778 K
        lambda_m = _WIEN_B / _T_SUN_K
        return AstroMatch(
            tool="blackbody_peak_wavelength_sun",
            args={"temperature_k": _T_SUN_K},
            rationale=f"Wien at T={_T_SUN_K}K",
            result_override=lambda_m,
            result_units="m",
        )

    # ── 'How much energy does the Sun produce in 1 year' ──
    if re.search(r"\b(?:Sun|sun).*?(?:produce|emit|radiate).*?(?:1\s+year|per\s+year|in\s+a\s+year)",
                  q, re.IGNORECASE | re.DOTALL):
        energy_J = _L_SUN * _YEAR_S
        return AstroMatch(
            tool="sun_energy_per_year",
            args={},
            rationale=f"L_sun * year_s = {_L_SUN:.3e} * {_YEAR_S:.3e}",
            result_override=energy_J,
            result_units="J",
        )

    # ── 'How fast is Earth moving in its orbit' ──
    if re.search(r"\bhow\s+fast.*?Earth.*?(?:orbit|moving|moves)|"
                  r"\bEarth.*?orbital\s+(?:velocity|speed)",
                  q, re.IGNORECASE | re.DOTALL):
        # v = sqrt(G M_sun / r) at r = 1 AU
        import math
        v = math.sqrt(_G * _M_SUN / _AU_M)
        return AstroMatch(
            tool="earth_orbital_velocity",
            args={},
            rationale=f"v=sqrt(GM_sun/r), r=1 AU",
            result_override=v,
            result_units="m/s",
        )

    return None


def execute_astro_match(match: AstroMatch) -> tuple[object, str, str]:
    if match.result_override is not None:
        return (match.result_override, match.result_units,
                f"ANSWER: {match.result_override} {match.result_units}\n\n"
                f"Computed via astro_classifier: {match.tool} ({match.rationale})")
    return None, "", ""
