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


# Specialty patterns for energy-energy and mass-energy conversions
# that show up in nuclear-physics questions but don't fit the standard
# "convert X to Y" or "N X equals how many Y" templates.
_ENERGY_UNITS_REGEX = (
    r"(meV|eV|keV|MeV|GeV|TeV|joule|joules|J)"
)


# Detect: NUMBER + energy_unit ... "how many" + energy_unit (cross-sentence OK)
# e.g. "Typical fission releases about 200 MeV. How many joules is that?"
_ENERGY_CROSS_SENTENCE = re.compile(
    rf"\b([\-+]?[0-9]+(?:\.[0-9]+)?(?:[eE][\-+]?[0-9]+)?)\s*"
    rf"{_ENERGY_UNITS_REGEX}\b.*?"
    rf"how\s+many\s+{_ENERGY_UNITS_REGEX}\b",
    re.IGNORECASE | re.DOTALL,
)

# Detect: "X kg of mass [is fully] converted [to energy/in annihilation]"
# -> compute E=mc^2 in J. Covers nuc_001 ("1 kg of mass is fully converted")
# and the matter-antimatter / annihilation phrasings.
_MASS_TO_ENERGY = re.compile(
    r"\b([\-+]?[0-9]+(?:\.[0-9]+)?)\s*(kg|kilogram|kilograms|g|gram|grams)\s+"
    r"(?:of\s+(?:mass|matter)\s+)?"
    r"(?:is\s+|fully\s+|completely\s+)*"
    r"(?:converted|annihilat\w+|destroyed)"
    r"(?:.*?(?:annihilation|matter.antimatter|energy))?",
    re.IGNORECASE | re.DOTALL,
)

# Detect: "X kg ... megatons TNT" (mass -> TNT, via E=mc^2 then J -> MT_TNT)
_MASS_TO_TNT = re.compile(
    r"\b([\-+]?[0-9]+(?:\.[0-9]+)?)\s*(kg|kilogram|kilograms|g|gram|grams)\s+"
    r".*?(?:how\s+many\s+)?(megatons?|MT|kt|kilotons?)\s+(?:of\s+)?TNT",
    re.IGNORECASE | re.DOTALL,
)


def classify_for_conversion(question: str) -> ConversionMatch | None:
    """Return a ConversionMatch if the question is unambiguously a conversion.

    Returns None for questions that aren't a recognizable conversion pattern.
    """
    try:
        import pint
    except ImportError:
        return None
    ureg = pint.UnitRegistry()
    # 1 kg c^2 in joules (E=mc^2 with c = 299792458 m/s)
    _C2 = 299792458.0 ** 2
    # 1 megaton TNT = 4.184e15 J
    _MT_TNT_J = 4.184e15

    # === Standard convert/equals/inverted patterns ===
    for i, pat in enumerate(_PATTERNS):
        m = pat.search(question)
        if not m:
            continue
        groups = m.groups()
        if len(groups) != 3:
            continue
        if i in (0, 1):
            try:
                value = float(groups[0])
            except ValueError:
                continue
            from_raw, to_raw = groups[1], groups[2]
        else:
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

    # === Energy-to-energy across sentences (e.g. "fission releases 200 MeV. How many joules?") ===
    m = _ENERGY_CROSS_SENTENCE.search(question)
    if m:
        try:
            value = float(m.group(1))
            from_u = _normalize_unit(m.group(2))
            to_u = _normalize_unit(m.group(3))
            if from_u != to_u:
                q = ureg.Quantity(value, from_u)
                result = q.to(to_u).magnitude
                return ConversionMatch(
                    value=value, from_units=from_u, to_units=to_u,
                    result=float(result),
                    matched_pattern="cross_sentence_energy",
                )
        except Exception:
            pass

    # === Mass -> TNT (check FIRST -- more specific than mass-to-energy) ===
    # Skip if the question mentions fission or fusion: those only convert
    # ~0.09% (fission) or ~0.7% (D-T fusion) of mass to energy, NOT full
    # E=mc^2. Let the LLM handle those with the right binding-energy values.
    nuclear_partial_conversion = re.search(
        r"\b(fission|fusion|nuclear\s+(?:reaction|reactor)|binding\s+energy|deuter|trit)",
        question, re.IGNORECASE)
    m = _MASS_TO_TNT.search(question) if not nuclear_partial_conversion else None
    if m:
        try:
            value = float(m.group(1))
            mass_unit = _normalize_unit(m.group(2))
            tnt_unit_raw = m.group(3).lower()
            kg = ureg.Quantity(value, mass_unit).to("kilogram").magnitude
            energy_J = kg * _C2
            if "megaton" in tnt_unit_raw or tnt_unit_raw == "mt":
                tnt_value = energy_J / _MT_TNT_J
                to_u = "megaton_TNT"
            else:
                tnt_value = energy_J / (_MT_TNT_J / 1000.0)
                to_u = "kiloton_TNT"
            return ConversionMatch(
                value=value, from_units=mass_unit,
                to_units=to_u, result=float(tnt_value),
                matched_pattern="mass_to_TNT_E=mc^2",
            )
        except Exception:
            pass

    # === Mass -> Energy (E = mc^2) -- after the TNT check ===
    # Same nuclear-partial-conversion guard: don't match if the question
    # is about fission/fusion (partial mass conversion via binding energy).
    m = _MASS_TO_ENERGY.search(question) if not nuclear_partial_conversion else None
    if m:
        try:
            value = float(m.group(1))
            mass_unit = _normalize_unit(m.group(2))
            kg = ureg.Quantity(value, mass_unit).to("kilogram").magnitude
            energy_J = kg * _C2
            return ConversionMatch(
                value=value, from_units=mass_unit,
                to_units="joule", result=float(energy_J),
                matched_pattern="mass_to_energy_E=mc^2",
            )
        except Exception:
            pass

    return None


def render_answer_text(match: ConversionMatch) -> str:
    """Format the canonical ANSWER: line for a conversion hit."""
    return (
        f"ANSWER: {match.result} {match.to_units}\n\n"
        f"{match.value} {match.from_units} = {match.result} {match.to_units} "
        f"(pint conversion via conversion-classifier)"
    )
