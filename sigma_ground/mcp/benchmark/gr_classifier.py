"""GR / black-hole question pre-classifier.

GR questions in the corpus follow a tight pattern: state the BH mass
(in solar masses, kilograms, or by-name like 'Sgr A*'), ask for one
of {event horizon, photon sphere, ISCO, Hawking temp/evap time,
gravitational redshift, gravitational time dilation}. Qwen 7b
reliably fails these because:

  1. It passes the mass literally ('mass_kg=10') instead of converting
     '10 solar masses' to kg first.
  2. It picks adjacent-but-wrong tools (hawking_temperature when
     event-horizon was asked, em_wave_wavelength for redshift, etc).

This classifier:
  - Extracts the BH mass from the question text (solar mass, kg,
    named-body shorthand, or scientific notation).
  - Dispatches the right tool based on which GR concept the question
    asks for.
  - Calls the tool with correct kg-valued mass and returns the result.

Conservative: only matches when the BH-mass pattern AND the concept-
keyword are both unambiguous.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# Standard astronomy constants (CODATA / IAU)
_M_SUN_KG = 1.98892e30
_M_EARTH_KG = 5.97219e24
_R_EARTH_M = 6.378e6
# Sgr A*: about 4.1 million solar masses (Event Horizon Telescope, 2022)
_SGR_A_MASS_SOLAR = 4.1e6
# Distance from Earth's center to GPS orbit (semi-major axis ~ 26600 km).
# But for the corpus altitude question, the answer is ~20200 km above
# Earth's surface (medium Earth orbit).


@dataclass
class GRMatch:
    """A GR classifier hit ready to dispatch."""
    tool: str               # which sigma_ground tool to call
    mass_kg: float
    radius_m: float | None  # for tools that need a radius (redshift/time dilation)
    rationale: str          # how mass was derived


def _extract_mass_kg(question: str) -> tuple[float | None, str]:
    """Extract the BH mass in kg from the question text.

    Tries (in order):
      - "X solar mass[es]" -> X * M_sun
      - "Sgr A*" / "Sagittarius A*" -> 4.1e6 * M_sun
      - "Sun" + "black hole" / "collapsed" -> M_sun (Sun-mass BH)
      - "X million solar masses" -> X * 1e6 * M_sun
      - "X kg" with no other mass mentioned -> X
      - "1e12 kg" / "10^12 kg" scientific notation

    Returns (mass_kg, rationale). (None, "no match") if no mass.
    """
    q = question

    # "Sagittarius A*" or "Sgr A*" reference: use the well-known mass
    if re.search(r"\b(sgr\s*a\*?|sagittarius\s*a\*?)", q, re.IGNORECASE):
        # Look for explicit "N million solar masses" in same question
        m = re.search(r"\b([\-+]?[0-9]+(?:\.[0-9]+)?)\s*million\s+solar\s+mass",
                       q, re.IGNORECASE)
        if m:
            n = float(m.group(1))
            return n * 1e6 * _M_SUN_KG, f"{n}e6 solar masses (Sgr A*)"
        return _SGR_A_MASS_SOLAR * _M_SUN_KG, "Sgr A* default (~4.1e6 solar masses)"

    # "X million solar masses" (e.g. "4.1 million solar masses")
    m = re.search(r"\b([\-+]?[0-9]+(?:\.[0-9]+)?)\s*million\s+solar\s+mass",
                   q, re.IGNORECASE)
    if m:
        n = float(m.group(1))
        return n * 1e6 * _M_SUN_KG, f"{n} million solar masses"

    # "X billion solar masses"
    m = re.search(r"\b([\-+]?[0-9]+(?:\.[0-9]+)?)\s*billion\s+solar\s+mass",
                   q, re.IGNORECASE)
    if m:
        n = float(m.group(1))
        return n * 1e9 * _M_SUN_KG, f"{n} billion solar masses"

    # "(N)-solar-mass" or "N solar mass" or "N solar masses"
    m = re.search(r"\b([\-+]?[0-9]+(?:\.[0-9]+)?)[-\s]+solar[-\s]+mass(?:es)?",
                   q, re.IGNORECASE)
    if m:
        n = float(m.group(1))
        return n * _M_SUN_KG, f"{n} solar masses"

    # "Sun collapsed" / "If the Sun were a black hole" -> M_sun
    if re.search(r"\bsun\b.*\b(collapsed|became|were|formed)\s+(?:into\s+)?(?:a\s+)?black\s+hole",
                  q, re.IGNORECASE | re.DOTALL):
        return _M_SUN_KG, "Sun collapsed (mass = M_sun)"

    # "X kg" with optional scientific notation (e.g. "1e12 kg" / "10^12 kg")
    m = re.search(r"\b([\-+]?[0-9]+(?:\.[0-9]+)?(?:[eE][\-+]?[0-9]+)?)\s*kg",
                   q, re.IGNORECASE)
    if m:
        n = float(m.group(1))
        # Skip if this looks like a different mass (e.g. "70 kg person")
        # by checking for nearby "black hole" or "mass" keywords
        if re.search(r"black\s+hole|primordial|mini\s+black\s+hole", q, re.IGNORECASE):
            return n, f"{n} kg"

    # "10^X kg" style (no e-notation)
    m = re.search(r"\b10\^([\-+]?[0-9]+)\s*kg", q, re.IGNORECASE)
    if m:
        exp = int(m.group(1))
        n = 10.0 ** exp
        if re.search(r"black\s+hole|primordial", q, re.IGNORECASE):
            return n, f"10^{exp} kg"

    return None, "no match"


# Concept patterns -> (tool_name, needs_radius?)
# Patterns ordered by specificity: more specific first so e.g. a
# question about "time dilation NEAR an event horizon" hits the
# time-dilation tool, not the event-horizon radius lookup.
_CONCEPT_PATTERNS: list[tuple[re.Pattern, str, bool]] = [
    # Time dilation -- check FIRST, before event-horizon-radius pattern,
    # because time-dilation questions often mention event horizon as context.
    (re.compile(r"\b(?:gravitational\s+)?time\s+dilation\b|"
                  r"\bclock\s+(?:run|tick)s?\s+(?:slow|fast)|"
                  r"\bhow\s+(?:slow|fast)\s+does\s+(?:it|the\s+clock)\s+(?:tick|run|go)|"
                  r"\bclocks?\s+\w*\s*(?:slow|fast)er|"
                  r"\bcompared\s+to\s+one\s+at\s+infinity",
                  re.IGNORECASE),
     "gravitational_time_dilation", True),
    # Gravitational redshift -- specific. Broaden to catch "redshift of light
    # I emit" plus traditional "gravitational redshift" phrasing.
    (re.compile(r"\bgravitational\s+(?:redshift|red\s*shift)|"
                  r"\bredshift\s+(?:at|of\s+(?:the\s+event|light))|"
                  r"\b(?:what(?:'s|\s+is)?\s+the\s+)?redshift\b",
                  re.IGNORECASE),
     "gravitational_redshift", True),
    # Hawking evaporation timescale -- catch "evaporate via Hawking" too.
    (re.compile(r"\b(hawking\s+(?:evaporat|lifetime|life\s*time)|"
                  r"evaporat\w*\s+via\s+hawking|"
                  r"black\s+hole\s+lifetime)\b",
                  re.IGNORECASE),
     "hawking_evaporation_time", False),
    # Event-horizon SIZE/RADIUS specifically asked for -- check BEFORE
    # Hawking temperature, because "event horizon size AND Hawking
    # temperature?" expects the radius (first thing asked).
    (re.compile(r"\bevent\s+horizon\s+(?:size|radius)|"
                  r"\b(?:size|radius)\s+of\s+(?:its|the)\s+event\s+horizon|"
                  r"\bwhat(?:'s|\s+is)?\s+(?:its|the)\s+event\s+horizon\s+(?:size|radius)|"
                  r"\bschwarzschild\s+radius",
                  re.IGNORECASE),
     "schwarzschild_radius", False),
    (re.compile(r"\bhawking\s+(?:temperature|temp)\b", re.IGNORECASE),
     "hawking_temperature", False),
    # Tool-specific radii.
    (re.compile(r"\bphoton\s+sphere\b", re.IGNORECASE),
     "photon_sphere_radius", False),
    (re.compile(r"\b(isco|innermost\s+stable\s+circular\s+orbit)\b",
                  re.IGNORECASE),
     "isco_radius", False),
    # General event-horizon fallback (no explicit size/radius keyword).
    # Time-dilation / redshift questions often mention this as setup, so
    # this is checked LAST and only when no specific concept matched.
    (re.compile(r"\b(event\s+horizon|r_?s\b)", re.IGNORECASE),
     "schwarzschild_radius", False),
]


def classify_for_gr(question: str) -> GRMatch | None:
    """Return a GRMatch dispatch if the question is a clean GR query.

    Two ingredients required: an extractable BH mass AND a known
    concept keyword. Otherwise returns None and the LLM handles it.

    Concept selection: patterns are checked in _CONCEPT_PATTERNS order.
    First match wins. The list is ordered specific-first, fallback last:
       time_dilation > redshift > evaporation > Hawking_temp >
       photon_sphere > ISCO > event_horizon_SIZE > event_horizon (bare).
    """
    mass_kg, rationale = _extract_mass_kg(question)

    pat = tool = needs_radius = None
    for p, t, nr in _CONCEPT_PATTERNS:
        if p.search(question):
            pat, tool, needs_radius = p, t, nr
            break
    if tool is None:
        return None

    if mass_kg is None:
        # Redshift/time-dilation evaluated EXACTLY AT the event horizon are
        # mass-INDEPENDENT: gravitational_redshift returns inf whenever
        # radius_m <= r_s(mass), for ANY mass -- r_s cancels out. A question
        # with no specific mass ("a black hole", not "10 solar masses") but
        # explicit "right at the event horizon" phrasing is still answerable;
        # only bail for real if there's neither a mass NOR this special case.
        if tool in ("gravitational_redshift", "gravitational_time_dilation") \
                and re.search(r"\bright\s+at\s+the\s+event\s+horizon\b",
                              question, re.IGNORECASE):
            mass_kg = _M_SUN_KG
            rationale = "arbitrary placeholder mass (answer is mass-independent at r=r_s)"
        else:
            return None

    if True:
            radius_m = None
            if needs_radius:
                # Try to extract a radius from the question
                # "1 km above event horizon" -> radius_m = r_s + 1000
                m = re.search(r"\b([\-+]?[0-9]+(?:\.[0-9]+)?)\s*(km|kilometer|m\b|meter)\s+"
                                r"(?:above|over|from)",
                                question, re.IGNORECASE)
                if m:
                    n = float(m.group(1))
                    unit = m.group(2).lower()
                    delta = n * 1000.0 if unit.startswith("km") else n
                    # Add to Schwarzschild radius
                    from sigma_ground.field.constants import G, C
                    r_s = 2.0 * G * mass_kg / (C * C)
                    radius_m = r_s + delta
                # Special case: "right at the event horizon" -> r = r_s
                elif re.search(r"\bright\s+at\s+the\s+event\s+horizon\b",
                                 question, re.IGNORECASE):
                    from sigma_ground.field.constants import G, C
                    radius_m = 2.0 * G * mass_kg / (C * C)
            return GRMatch(
                tool=tool, mass_kg=mass_kg, radius_m=radius_m,
                rationale=rationale,
            )
    return None


def execute_gr_match(match: GRMatch) -> tuple[float | None, str, str]:
    """Run the dispatched tool. Returns (value, units, answer_text)."""
    from sigma_ground.mcp.tools import gr as t_gr
    try:
        if match.tool == "schwarzschild_radius":
            r = t_gr.schwarzschild_radius(match.mass_kg)
        elif match.tool == "photon_sphere_radius":
            r = t_gr.photon_sphere_radius(match.mass_kg)
        elif match.tool == "isco_radius":
            r = t_gr.isco_radius(match.mass_kg)
        elif match.tool == "hawking_temperature":
            r = t_gr.hawking_temperature(match.mass_kg)
        elif match.tool == "hawking_evaporation_time":
            r = t_gr.hawking_evaporation_time(match.mass_kg)
        elif match.tool == "gravitational_redshift":
            if match.radius_m is None:
                return None, "", ""
            r = t_gr.gravitational_redshift(match.mass_kg, match.radius_m)
        elif match.tool == "gravitational_time_dilation":
            if match.radius_m is None:
                return None, "", ""
            r = t_gr.gravitational_time_dilation(match.mass_kg, match.radius_m)
        else:
            return None, "", ""
    except Exception as e:
        return None, "", f"<gr_classifier ERROR: {e}>"
    val = r.value if hasattr(r, "value") else None
    units = r.units if hasattr(r, "units") else ""
    answer_text = (
        f"ANSWER: {val} {units}\n\n"
        f"Computed via gr_classifier: tool={match.tool}, "
        f"mass={match.mass_kg:.4g} kg ({match.rationale})"
    )
    if match.radius_m is not None:
        answer_text += f", radius={match.radius_m:.4g} m"
    return val, units, answer_text
