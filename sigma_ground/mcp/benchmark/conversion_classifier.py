"""Programmatic unit-conversion classifier — short-circuit obvious conversions.

Qwen 2.5:7b often picks wrong tools for simple "convert X to Y" questions
even though `convert_units` is listed with rich keywords. Examples from
the WA-overlap audit:

  thermo_011 "Convert 100 Celsius to Kelvin"  -> Qwen called melting_point
  modern_008 "1 joule equals how many eV?"     -> Qwen called photon_energy
  math_006   "Convert 1 light year to meters" -> Qwen called light_travel_time

These are pattern-matchable. We detect them ourselves and compute the
answer directly via pint (the same registry the convert_units MCP tool
uses), short-circuiting the LLM call. Saves latency AND ensures
correctness on the simplest physics questions.

Conservative: only short-circuit when the pattern is unmistakable.
False positives cost us trust; false negatives just let Qwen take a swing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ConversionMatch:
    value: float
    from_units: str
    to_units: str
    result: float          # the converted value
    matched_pattern: str   # for debugging


# Patterns capture: (number, from-unit-phrase, to-unit-phrase).
# Each group is named; the patterns are tested in order.
_PATTERNS: list[re.Pattern] = [
    # "Convert 100 Celsius to Kelvin"
    re.compile(
        r"\bconvert\s+([\-+]?[0-9]+(?:\.[0-9]+)?(?:[eE][\-+]?[0-9]+)?)\s+"
        r"([a-zA-Z][a-zA-Z\s_./^-]*?)\s+to\s+"
        r"([a-zA-Z][a-zA-Z_./^-]*)\b",
        re.IGNORECASE),
    # "1 joule equals how many electronvolts" / "1 J = how many eV"
    re.compile(
        r"\b([\-+]?[0-9]+(?:\.[0-9]+)?(?:[eE][\-+]?[0-9]+)?)\s+"
        r"([a-zA-Z][a-zA-Z\s_./^-]*?)\s+"
        r"(?:equals?|=)\s+how\s+many\s+"
        r"([a-zA-Z][a-zA-Z_./^-]*)\b",
        re.IGNORECASE),
    # "How many meters in 1 light year"
    re.compile(
        r"\bhow\s+many\s+"
        r"([a-zA-Z][a-zA-Z_./^-]*)\s+"
        r"(?:in|are\s+in|equals?)\s+"
        r"([\-+]?[0-9]+(?:\.[0-9]+)?(?:[eE][\-+]?[0-9]+)?)\s+"
        r"([a-zA-Z][a-zA-Z\s_./^-]*?)(?:\?|$|\.)",
        re.IGNORECASE),
]


# Common informal -> canonical-pint unit map. Most pint defaults work
# but a few human phrasings need normalizing.
_UNIT_ALIAS: dict[str, str] = {
    "celsius":     "degC",
    "centigrade":  "degC",
    "c":           "degC",
    "kelvin":      "K",
    "fahrenheit":  "degF",
    "joule":       "joule",
    "joules":      "joule",
    "j":           "joule",
    "electronvolt":  "eV",
    "electronvolts": "eV",
    "ev":          "eV",
    "mev":         "MeV",
    "kev":         "keV",
    "gev":         "GeV",
    "tev":         "TeV",
    "light year":  "light_year",
    "light-year":  "light_year",
    "light years": "light_year",
    "ly":          "light_year",
    "parsec":      "parsec",
    "parsecs":     "parsec",
    "pc":          "parsec",
    "meter":       "meter",
    "meters":      "meter",
    "m":           "meter",
    "kilometer":   "kilometer",
    "kilometers":  "kilometer",
    "km":          "kilometer",
    "astronomical unit": "AU",
    "au":          "AU",
    "gram":        "gram",
    "grams":       "gram",
    "kg":          "kilogram",
    "kilogram":    "kilogram",
    "kilograms":   "kilogram",
    "pound":       "pound",
    "pounds":      "pound",
    "lb":          "pound",
    "second":      "second",
    "seconds":     "second",
    "s":           "second",
    "minute":      "minute",
    "minutes":     "minute",
    "hour":        "hour",
    "hours":       "hour",
    "h":           "hour",
    "day":         "day",
    "days":        "day",
    "year":        "year",
    "years":       "year",
    "yr":          "year",
}


def _normalize_unit(raw: str) -> str:
    """Map a free-text unit phrase to a pint-friendly canonical form."""
    s = raw.strip().lower()
    # Strip trailing punctuation / question mark / "to"
    s = re.sub(r"[?.,;:!]+$", "", s).strip()
    # Direct alias
    if s in _UNIT_ALIAS:
        return _UNIT_ALIAS[s]
    # Try without trailing 's' (rough plural strip)
    if s.endswith("s") and s[:-1] in _UNIT_ALIAS:
        return _UNIT_ALIAS[s[:-1]]
    # Pass through (pint may parse it)
    return s.replace(" ", "_")


def classify_for_conversion(question: str) -> ConversionMatch | None:
    """Return a ConversionMatch if the question is unambiguously a conversion.

    Returns None for questions that aren't a recognizable "convert X to Y"
    pattern.
    """
    try:
        import pint
    except ImportError:
        return None
    ureg = pint.UnitRegistry()

    for i, pat in enumerate(_PATTERNS):
        m = pat.search(question)
        if not m:
            continue
        groups = m.groups()
        if len(groups) != 3:
            continue
        # Pattern 0 and 1: (value, from, to). Pattern 2: (to, value, from).
        if i in (0, 1):
            try:
                value = float(groups[0])
            except ValueError:
                continue
            from_raw, to_raw = groups[1], groups[2]
        else:  # pattern 2 swaps
            try:
                value = float(groups[1])
            except ValueError:
                continue
            from_raw, to_raw = groups[2], groups[0]

        from_u = _normalize_unit(from_raw)
        to_u = _normalize_unit(to_raw)
        if not from_u or not to_u or from_u == to_u:
            continue
        try:
            q = ureg.Quantity(value, from_u)
            result = q.to(to_u).magnitude
        except Exception:
            continue
        return ConversionMatch(
            value=value, from_units=from_u, to_units=to_u,
            result=float(result), matched_pattern=pat.pattern[:60],
        )
    return None


def render_answer_text(match: ConversionMatch) -> str:
    """Format the canonical ANSWER: line for a conversion hit."""
    return (
        f"ANSWER: {match.result} {match.to_units}\n\n"
        f"{match.value} {match.from_units} = {match.result} {match.to_units} "
        f"(pint conversion via conversion-classifier)"
    )
