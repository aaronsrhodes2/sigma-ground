"""Quantities as people write them in a question, read with Mentat's own unit tool.

"100 miles per hour", "100 mph", "60 km/h", "30 ft/s", "45 degrees", "15 °C":
each is found in the text, its unit words are mapped to a pint unit, and the
value is converted by the registry behind Mentat's ``convert_units`` tool
(sigma_ground/units_registry.py). Nothing here knows a conversion factor itself.
Without pint (a bare core install) nothing converts, and an input that needed
converting stays missing: it is then suggested, never guessed.
"""
from __future__ import annotations

import re

# Unit words → pint unit names. Longest phrases first when matching.
_UNIT_WORDS = {
    # speed
    "miles per hour": "mile/hour", "mile per hour": "mile/hour", "mph": "mile/hour", "mi/h": "mile/hour",
    "kilometers per hour": "kilometer/hour", "kilometres per hour": "kilometer/hour",
    "kilometer per hour": "kilometer/hour", "km/h": "kilometer/hour", "km/hr": "kilometer/hour",
    "kph": "kilometer/hour", "kmh": "kilometer/hour",
    "meters per second": "meter/second", "metres per second": "meter/second",
    "meter per second": "meter/second", "metre per second": "meter/second", "m/s": "meter/second",
    "feet per second": "foot/second", "foot per second": "foot/second", "ft/s": "foot/second",
    "ft/sec": "foot/second", "km/s": "kilometer/second", "kilometers per second": "kilometer/second",
    "knots": "knot", "knot": "knot",
    # length
    "millimeters": "millimeter", "millimetres": "millimeter", "millimeter": "millimeter", "mm": "millimeter",
    "centimeters": "centimeter", "centimetres": "centimeter", "centimeter": "centimeter", "cm": "centimeter",
    "kilometers": "kilometer", "kilometres": "kilometer", "kilometer": "kilometer", "km": "kilometer",
    "meters": "meter", "metres": "meter", "meter": "meter", "metre": "meter", "m": "meter",
    "miles": "mile", "mile": "mile", "mi": "mile",
    "feet": "foot", "foot": "foot", "ft": "foot", "inches": "inch", "inch": "inch",
    "yards": "yard", "yard": "yard", "nanometers": "nanometer", "nm": "nanometer",
    "micrometers": "micrometer", "microns": "micrometer", "µm": "micrometer", "um": "micrometer",
    # mass
    "kilograms": "kilogram", "kilogram": "kilogram", "kg": "kilogram",
    "grams": "gram", "gram": "gram", "g": "gram", "milligrams": "milligram", "mg": "milligram",
    "pounds": "pound", "pound": "pound", "lbs": "pound", "lb": "pound",
    "ounces": "ounce", "ounce": "ounce", "oz": "ounce", "tonnes": "tonne", "tonne": "tonne",
    # time
    "seconds": "second", "second": "second", "sec": "second", "s": "second",
    "minutes": "minute", "minute": "minute", "min": "minute", "hours": "hour", "hour": "hour",
    "hr": "hour", "days": "day", "day": "day", "years": "year", "year": "year",
    # angle
    "degrees": "degree", "degree": "degree", "deg": "degree", "°": "degree",
    "radians": "radian", "radian": "radian", "rad": "radian",
    # temperature (offset units: converted as a Quantity, never multiplied)
    "degrees celsius": "degC", "degrees c": "degC", "°c": "degC", "celsius": "degC",
    "degrees fahrenheit": "degF", "degrees f": "degF", "°f": "degF", "fahrenheit": "degF",
    "kelvin": "kelvin", "k": "kelvin",
    # energy, pressure, field
    "joules": "joule", "joule": "joule", "j": "joule", "kilojoules": "kilojoule", "kj": "kilojoule",
    "electron volts": "electron_volt", "electronvolts": "electron_volt", "ev": "electron_volt",
    "kev": "kiloelectron_volt", "mev": "megaelectron_volt", "gev": "gigaelectron_volt",
    "pascals": "pascal", "pascal": "pascal", "pa": "pascal", "kpa": "kilopascal", "mpa": "megapascal",
    "gpa": "gigapascal", "atmospheres": "atmosphere", "atmosphere": "atmosphere", "atm": "atmosphere",
    "bar": "bar", "psi": "psi", "tesla": "tesla", "t": "tesla",
}
# Unit words that are too short or too common to trust without a number right before them.
_NUM = r"(?<![\w.])(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
_UNIT_RE = re.compile(
    _NUM + r"\s*-?\s*(" + "|".join(re.escape(w) for w in sorted(_UNIT_WORDS, key=len, reverse=True)) + r")(?![\w/])",
    re.I)


def _registry():
    from sigma_ground.units_registry import get_registry
    return get_registry()


def scan(question: str) -> list:
    """Every "number unit" in the question: [{"value", "unit", "text", "start"}],
    unit being the pint name. Order of appearance."""
    out = []
    for m in _UNIT_RE.finditer(question):
        word = m.group(2).lower()
        out.append({"value": float(m.group(1)), "unit": _UNIT_WORDS[word], "text": m.group(0).strip(),
                    "start": m.start()})
    return out


def convert(value: float, unit: str, target: str):
    """value [unit] in [target], by Mentat's unit registry; None if it can't."""
    ureg = _registry()
    if ureg is None:
        return None
    try:
        return float(ureg.Quantity(value, unit).to(target).magnitude)
    except Exception:   # noqa: BLE001 — wrong dimension or an unknown unit
        return None


def same_dimension(unit: str, target: str) -> bool:
    ureg = _registry()
    if ureg is None or not target:
        return False
    try:
        return ureg.Quantity(1, unit).dimensionality == ureg.Quantity(1, target).dimensionality
    except Exception:   # noqa: BLE001
        return False


def find(question: str, target: str):
    """The first quantity in the question with the target's dimension, in the
    target's units, with its words: (value, text) or None."""
    for q in scan(question):
        if same_dimension(q["unit"], target):
            v = convert(q["value"], q["unit"], target)
            if v is not None:
                return v, q["text"]
    return None


def to_units(text: str, target: str):
    """A quantity string ("100 mph") in the target units, or None."""
    found = scan(text)
    if len(found) != 1:
        return None
    q = found[0]
    return convert(q["value"], q["unit"], target) if same_dimension(q["unit"], target) else None
