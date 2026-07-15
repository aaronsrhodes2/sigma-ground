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
    field: str | None = None   # pick a scalar from a dict-valued result


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
        # Any "N km" altitude above Earth (ISS, geostationary, etc.) -- the
        # word "Earth" itself is often implicit: "the ISS orbits at 408 km
        # altitude" never says Earth, but ISS/satellite/geostationary/
        # geosynchronous with no OTHER body named means Earth by default.
        mentions_earth = re.search(r"\bEarth\b", q, re.IGNORECASE)
        mentions_other_body = re.search(
            r"\b(moon|sun|mars|venus|jupiter|saturn|mercury|neptune|uranus)\b",
            q, re.IGNORECASE)
        implies_earth = re.search(
            r"\b(ISS|International\s+Space\s+Station|geostationary|"
            r"geosynchronous)\b", q, re.IGNORECASE)
        if mentions_earth or (implies_earth and not mentions_other_body):
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

    # ── nuclear_binding_energy: "(Z protons, N neutrons, mass M u)" ──
    if re.search(r"\bbinding\s+energy\b", q, re.IGNORECASE) and \
       re.search(r"\b(nucleon|nucleus|nuclei|nuclide|protons?|neutrons?)\b",
                   q, re.IGNORECASE):
        mp_ = re.search(r"\b([0-9]+)\s*protons?\b", q, re.IGNORECASE)
        mn_ = re.search(r"\b([0-9]+)\s*neutrons?\b", q, re.IGNORECASE)
        mm_ = (re.search(r"\bmass\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)\s*u\b", q, re.IGNORECASE)
                or re.search(r"\b([0-9]+\.[0-9]+)\s*u\b", q, re.IGNORECASE))
        if mp_ and mn_:
            args = {"protons": int(mp_.group(1)), "neutrons": int(mn_.group(1))}
            if mm_:
                args["measured_mass_u"] = _num(mm_.group(1))
            # "per nucleon" → return the per-nucleon field, else total BE
            field = ("binding_per_nucleon_MeV"
                       if re.search(r"\bper\s+nucleon\b", q, re.IGNORECASE)
                       else "binding_energy_MeV")
            return NewToolMatch("nuclear_binding_energy", args,
                                  "BE = [Z m_p + N m_n − M] c²", field=field)

    # ── coulomb_force: between two charges/electrons/protons ──
    if re.search(r"\b(coulomb|electrostatic)\s+force\b", q, re.IGNORECASE):
        r_m = _length_m(q)
        q1 = q2 = None
        # named particles → elementary charge
        if re.search(r"\btwo\s+electrons?\b|\belectron\s+and\s+electron\b", q, re.IGNORECASE):
            q1 = q2 = 1.602176634e-19
        elif re.search(r"\btwo\s+protons?\b", q, re.IGNORECASE):
            q1 = q2 = 1.602176634e-19
        elif re.search(r"\belectron\b", q, re.IGNORECASE) and re.search(r"\bproton\b", q, re.IGNORECASE):
            q1, q2 = 1.602176634e-19, -1.602176634e-19
        else:
            chs = re.findall(r"\b([0-9]+(?:\.[0-9]+)?(?:[eE][\-+]?[0-9]+)?)\s*(?:C|coulomb)\b", q)
            if len(chs) >= 2:
                q1, q2 = _num(chs[0]), _num(chs[1])
        if q1 is not None and q2 is not None and r_m is not None:
            return NewToolMatch("coulomb_force",
                                  {"charge1_c": q1, "charge2_c": q2, "separation_m": r_m},
                                  "F = q1 q2 / (4π ε₀ r²)")

    # ── gravitational_force: two masses (kg) + a separation (m) ──
    if re.search(r"\bgravitational\s+force\b", q, re.IGNORECASE):
        masses = re.findall(r"\b([0-9]+(?:\.[0-9]+)?(?:[eE][\-+]?[0-9]+)?)\s*kg\b", q)
        sep = (re.search(r"\b(?:separated|apart|distance|separation)\b.{0,25}?"
                          r"([0-9.]+(?:[eE][\-+]?[0-9]+)?)\s*m\b", q, re.IGNORECASE)
                or re.search(r"\b([0-9.]+(?:[eE][\-+]?[0-9]+)?)\s*m\b", q, re.IGNORECASE))
        if len(masses) >= 2 and sep:
            return NewToolMatch("gravitational_force",
                                  {"mass1_kg": _num(masses[0]), "mass2_kg": _num(masses[1]),
                                   "separation_m": _num(sep.group(1))},
                                  "F = G m1 m2 / r²")

    return None


def _length_m(q: str) -> float | None:
    """Extract a length, converting common units to meters."""
    for pat, scale in [(r"\b([0-9.]+)\s*[-\s]?\s*(?:fm|femtomet)", 1e-15),
                         (r"\b([0-9.]+)\s*[-\s]?\s*(?:pm|picomet)", 1e-12),
                         (r"\b([0-9.]+)\s*[-\s]?\s*(?:nm|nanomet)", 1e-9),
                         (r"\b([0-9.]+)\s*[-\s]?\s*(?:angstrom|Å)", 1e-10),
                         (r"\b([0-9.]+)\s*[-\s]?\s*(?:um|µm|micromet)", 1e-6),
                         (r"\b([0-9.]+)\s*[-\s]?\s*(?:mm|millimet)", 1e-3),
                         (r"\b([0-9.]+)\s*[-\s]?\s*(?:cm|centimet)", 1e-2),
                         (r"\b([0-9.]+)\s*[-\s]?\s*(?:km|kilomet)", 1e3),
                         (r"\b([0-9.]+(?:[eE][\-+]?[0-9]+)?)\s*[-\s]?\s*m(?:et(?:er|re)s?)?\b", 1.0)]:
        m = re.search(pat, q, re.IGNORECASE)
        if m:
            v = _num(m.group(1))
            if v is not None:
                return v * scale
    return None


def execute_new_tool_match(match: NewToolMatch) -> tuple[object, str, str]:
    from sigma_ground.mcp.tools import orbital as t_orb
    from sigma_ground.mcp.tools import atomic as t_atom
    from sigma_ground.mcp.tools import circuits as t_circ
    from sigma_ground.mcp.tools import nuclear as t_nuc
    try:
        if match.tool == "orbital_velocity":
            r = t_orb.orbital_velocity(**match.args)
        elif match.tool == "orbital_period":
            r = t_orb.orbital_period(**match.args)
        elif match.tool == "gravitational_force":
            r = t_orb.gravitational_force(**match.args)
        elif match.tool == "orbital_raise_energy":
            r = t_orb.orbital_raise_energy(**match.args)
        elif match.tool == "de_broglie_from_kinetic_energy":
            r = t_atom.de_broglie_from_kinetic_energy(**match.args)
        elif match.tool == "energy_power_time":
            r = t_circ.energy_power_time(**match.args)
        elif match.tool == "nuclear_binding_energy":
            r = t_nuc.nuclear_binding_energy(**match.args)
        elif match.tool == "coulomb_force":
            r = t_nuc.coulomb_force(**match.args)
        else:
            return None, "", ""
    except Exception as e:
        return None, "", f"<new_tools_classifier ERROR: {e}>"
    val = r.value if hasattr(r, "value") else None
    units = r.units if hasattr(r, "units") else ""
    if isinstance(val, dict) and match.field:
        val = val.get(match.field)
        # binding-energy fields are MeV
        if match.field and "MeV" in match.field:
            units = "MeV"
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
    # frontier black-hole-thermodynamics / holography (routed via
    # frontier_classifier, never exposed to the LLM's flat list)
    "bekenstein_hawking_entropy", "entanglements_to_pop_bubble",
    "holographic_matching_mass", "baryon_vs_disc",
    "gravitational_binding_energy", "unruh_temperature",
    "entanglement_channel",
}
