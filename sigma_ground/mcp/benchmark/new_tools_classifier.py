"""Router for the body-aware/multi-step tools (Phase 0).

The 8 new tools are correct (audit 86/86) but, left in Qwen's flat tool
list, they caused selection noise (a lens question grabbed
orbital_velocity, an Ohm's-law question grabbed coulomb_force). The fix
per the routing thesis: hide them from the LLM's visible surface and
reach them DETERMINISTICALLY here, so Qwen never has to choose.

Each classifier extracts inputs from the question and dispatches the
right new tool — recovering the wins (Jupiter, asteroid period, ISS,
energy×time, de Broglie) without exposing the tools to the LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class NewToolMatch:
    tool: str
    args: dict
    rationale: str


def _num(s):
    try:
        return float(str(s).replace(",", ""))
    except (TypeError, ValueError):
        return None


def classify_for_new_tools(question: str) -> NewToolMatch | None:
    q = question

    # ── orbital_period: "orbital period ... at N AU" ──
    if re.search(r"\borbital\s+period\b|\bhow\s+long.*\borbit", q, re.IGNORECASE):
        m = re.search(r"\b([\-+]?[0-9]+(?:\.[0-9]+)?)\s*AU\b", q, re.IGNORECASE)
        if m:
            return NewToolMatch("orbital_period",
                                  {"semimajor_axis_au": _num(m.group(1)),
                                   "central_body": "sun"},
                                  f"Kepler III at {m.group(1)} AU")

    # ── orbital_velocity: planet "around the Sun" ──
    m = re.search(r"\b(mercury|venus|earth|mars|jupiter|saturn|uranus|neptune|pluto)"
                    r"['’]?s?\s+orbital\s+(?:velocity|speed)", q, re.IGNORECASE)
    if not m:
        m = re.search(r"\borbital\s+(?:velocity|speed)\s+of\s+"
                        r"(mercury|venus|earth|mars|jupiter|saturn|uranus|neptune)",
                        q, re.IGNORECASE)
    if m and re.search(r"\b(sun|around the sun)\b", q, re.IGNORECASE):
        planet = m.group(1).lower()
        from sigma_ground.mcp.tools.astronomy import solar_system_body
        data = solar_system_body(planet)
        if data.value and isinstance(data.value, dict):
            a = data.value.get("semimajor_axis_au")
            if a:
                return NewToolMatch("orbital_velocity",
                                      {"central_body": "sun",
                                       "semimajor_axis_au": a},
                                      f"{planet} heliocentric orbit at {a} AU")

    # ── orbital_velocity: satellite/ISS/geostationary at altitude ──
    is_orbit_q = (re.search(r"\b(satellite|ISS|space\s+station|geostationary|"
                              r"geosynchronous|orbit)", q, re.IGNORECASE)
                    and re.search(r"\bhow\s+fast|orbital\s+(?:velocity|speed)|"
                                  r"\bmoving\b", q, re.IGNORECASE))
    if is_orbit_q:
        # Moon at an explicit orbital radius (from center)
        if re.search(r"\bmoon\b", q, re.IGNORECASE):
            m_r = re.search(r"\b([0-9][0-9,]*(?:\.[0-9]+)?)\s*km\b", q, re.IGNORECASE)
            if m_r:
                return NewToolMatch("orbital_velocity",
                                      {"central_body": "earth",
                                       "orbital_radius_m": _num(m_r.group(1)) * 1000.0},
                                      f"Moon orbit radius {m_r.group(1)} km")
        # Any "N km" altitude above Earth (ISS, geostationary, etc.)
        if re.search(r"\bEarth\b", q, re.IGNORECASE):
            m_alt = re.search(r"\b([0-9][0-9,]*(?:\.[0-9]+)?)\s*(?:km|kilomet)",
                                q, re.IGNORECASE)
            if m_alt:
                return NewToolMatch("orbital_velocity",
                                      {"central_body": "earth",
                                       "altitude_km": _num(m_alt.group(1))},
                                      f"Earth orbit at {m_alt.group(1)} km altitude")

    # ── de_broglie_from_kinetic_energy: "X keV/MeV electron/proton" ──
    if re.search(r"\bde\s+Broglie\s+wavelength\b", q, re.IGNORECASE):
        m = re.search(r"\b([\-+]?[0-9]+(?:\.[0-9]+)?)\s*(eV|keV|MeV|GeV)\b"
                        r".{0,30}?(electron|proton|neutron|alpha|muon)|"
                        r"(electron|proton|neutron|alpha|muon).{0,30}?"
                        r"\b([\-+]?[0-9]+(?:\.[0-9]+)?)\s*(eV|keV|MeV|GeV)\b",
                        q, re.IGNORECASE)
        if m:
            g = m.groups()
            if g[0]:
                energy, unit, particle = g[0], g[1], g[2]
            else:
                particle, energy, unit = g[3], g[4], g[5]
            scale = {"ev": 1, "kev": 1e3, "mev": 1e6, "gev": 1e9}[unit.lower()]
            return NewToolMatch("de_broglie_from_kinetic_energy",
                                  {"kinetic_energy_eV": _num(energy) * scale,
                                   "particle": particle.lower()},
                                  f"{energy} {unit} {particle}")

    # ── energy_power_time: P·t=E, solve for missing ──
    if re.search(r"\b(kW|kilowatt|watt|W)\b", q, re.IGNORECASE) and \
       re.search(r"\b(hour|hr|minute|second|kJ|kilojoule|joule|dissipate|"
                   r"energy|how long)\b", q, re.IGNORECASE):
        # power
        mp = re.search(r"\b([\-+]?[0-9]+(?:\.[0-9]+)?)\s*(kW|kilowatt|W|watt)\b",
                         q, re.IGNORECASE)
        # time
        mt = re.search(r"\b([\-+]?[0-9]+(?:\.[0-9]+)?)\s*(hour|hr|minute|min|second|sec)\b",
                         q, re.IGNORECASE)
        # energy
        me = re.search(r"\b([\-+]?[0-9]+(?:\.[0-9]+)?)\s*(kJ|kilojoule|J|joule)\b",
                         q, re.IGNORECASE)
        def _pw(m_):
            v = _num(m_.group(1)); u = m_.group(2).lower()
            return v * 1000.0 if u in ("kw", "kilowatt") else v
        def _ts(m_):
            v = _num(m_.group(1)); u = m_.group(2).lower()
            return v * 3600 if u in ("hour", "hr") else (v * 60 if u in ("minute", "min") else v)
        def _ej(m_):
            v = _num(m_.group(1)); u = m_.group(2).lower()
            return v * 1000.0 if u in ("kj", "kilojoule") else v
        asks_time = bool(re.search(r"\bhow\s+long\b", q, re.IGNORECASE))
        asks_energy = bool(re.search(r"\bhow\s+many\s+joules|how much energy", q, re.IGNORECASE))
        if mp and mt and asks_energy:
            return NewToolMatch("energy_power_time",
                                  {"power_w": _pw(mp), "time_s": _ts(mt)},
                                  "E = P t")
        if mp and me and asks_time:
            return NewToolMatch("energy_power_time",
                                  {"power_w": _pw(mp), "energy_j": _ej(me)},
                                  "t = E / P")

    return None


def execute_new_tool_match(match: NewToolMatch) -> tuple[object, str, str]:
    from sigma_ground.mcp.tools import orbital as t_orb
    from sigma_ground.mcp.tools import atomic as t_atom
    from sigma_ground.mcp.tools import circuits as t_circ
    try:
        if match.tool == "orbital_velocity":
            r = t_orb.orbital_velocity(**match.args)
        elif match.tool == "orbital_period":
            r = t_orb.orbital_period(**match.args)
        elif match.tool == "de_broglie_from_kinetic_energy":
            r = t_atom.de_broglie_from_kinetic_energy(**match.args)
        elif match.tool == "energy_power_time":
            r = t_circ.energy_power_time(**match.args)
        else:
            return None, "", ""
    except Exception as e:
        return None, "", f"<new_tools_classifier ERROR: {e}>"
    val = r.value if hasattr(r, "value") else None
    units = r.units if hasattr(r, "units") else ""
    answer = (f"ANSWER: {val} {units}\n\n"
              f"Computed via new_tools_classifier: {match.tool} ({match.rationale})")
    return val, units, answer


# The 8 tools to HIDE from the LLM's flat tool list (reached only via the
# classifier above + the existing physics classifiers). Keeps Qwen's
# selection surface small while the library grows behind the router.
HIDDEN_FROM_LLM = {
    "orbital_velocity", "orbital_period", "gravitational_force",
    "orbital_raise_energy", "nuclear_binding_energy", "coulomb_force",
    "de_broglie_from_kinetic_energy", "energy_power_time",
}
