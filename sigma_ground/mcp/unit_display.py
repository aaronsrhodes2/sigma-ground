"""Answers in the measurement system the question used, with the SI value in parentheses: "1 yard (0.9144 m)".

The tools compute in SI and return `value` in SI, always: a machine reading the result never has to guess. When the caller wrote a quantity in other
units ("10 yards", "60 mph", "70 degF"), the result also carries `display`, the same number in the units that fit what the caller wrote, then the SI
figure in parentheses (the Captain's rule, 2026-10-15).

Which unit is shown:
  1. a unit the caller used for the same kind of quantity (a length answer to a question in yards is in yards; "5 cm" gives "5 cm (0.05 m)");
  2. otherwise, if the caller used a customary unit (US/imperial: feet, pounds, mph, degF, psi ...), the customary unit of this kind of quantity
     that gives the most natural number (the largest unit whose value is at least 1);
  3. otherwise nothing: SI-only questions get SI-only answers, and a metric question that never used the answer's kind of quantity is not converted.
"""
from __future__ import annotations

import math
from typing import Iterable

# Components (pint's own names) that mark a unit as customary: US/imperial or another non-metric system.
CUSTOMARY_NAMES = frozenset({
    "thou", "inch", "foot", "yard", "mile", "nautical_mile", "fathom", "furlong", "rod", "chain", "acre",
    "grain", "ounce", "pound", "stone", "ton", "short_ton", "long_ton", "hundredweight", "slug", "poundal",
    "fluid_ounce", "tablespoon", "teaspoon", "cup", "pint", "quart", "gallon",
    "degree_Fahrenheit", "force_pound", "pound_force", "pound_force_per_square_inch", "foot_pound", "horsepower", "british_thermal_unit",
    "mile_per_hour", "knot", "inch_Hg", "inch_H2O",
})

# For each kind of quantity (its SI base unit), the customary units to choose from, smallest to largest.
_CANDIDATES = {
    "meter": ["inch", "foot", "yard", "mile"],
    "kilogram": ["ounce", "pound", "ton"],
    "meter/second": ["foot/second", "mile_per_hour"],
    "meter/second**2": ["foot/second**2"],
    "newton": ["force_pound"],
    "pascal": ["pound_force_per_square_inch"],
    "joule": ["foot_pound", "british_thermal_unit"],
    "watt": ["horsepower"],
    "kelvin": ["degree_Fahrenheit"],
    "meter**2": ["inch**2", "foot**2", "acre", "mile**2"],
    "meter**3": ["inch**3", "cup", "gallon", "foot**3"],
    "kilogram/meter**3": ["pound/foot**3"],
}

# How a unit is written in an answer (singular, plural) or one fixed form. Anything else uses pint's short symbol.
_NAMES = {
    "inch": ("inch", "inches"), "foot": ("foot", "feet"), "yard": ("yard", "yards"), "mile": ("mile", "miles"),
    "ounce": ("ounce", "ounces"), "pound": ("pound", "pounds"), "ton": ("ton", "tons"), "stone": ("stone", "stone"),
    "gallon": ("gallon", "gallons"), "cup": ("cup", "cups"), "acre": ("acre", "acres"), "knot": ("knot", "knots"),
    "nautical_mile": ("nautical mile", "nautical miles"),
    "mile_per_hour": "mph", "foot / second": "ft/s", "foot / second ** 2": "ft/s²", "force_pound": "lbf",
    "pound_force_per_square_inch": "psi", "foot_pound": "ft·lbf", "british_thermal_unit": "BTU", "horsepower": "hp",
    "degree_Fahrenheit": "°F", "degree_Celsius": "°C", "inch ** 2": "in²", "foot ** 2": "ft²", "mile ** 2": "mi²",
    "inch ** 3": "in³", "foot ** 3": "ft³", "pound / foot ** 3": "lb/ft³",
}

_dimension_table: dict | None = None


def _registry():
    from sigma_ground.mcp.tools.units import _get_registry
    return _get_registry()


def _parse(reg, text: str):
    try:
        return reg.Unit(str(text).strip())
    except Exception:  # noqa: BLE001 — pint raises several things for a unit it does not know
        return None


def classify(reg, unit) -> str:
    """'si' (a coherent SI unit), 'metric' (metric but not coherent SI: cm, g, km/h, L, bar, degC, hours) or 'customary'."""
    if set(getattr(unit, "_units", {}).keys()) & CUSTOMARY_NAMES:
        return "customary"
    try:
        magnitude = reg.Quantity(1.0, unit).to_base_units().magnitude
    except Exception:  # noqa: BLE001
        return "metric"
    return "si" if math.isclose(magnitude, 1.0, rel_tol=1e-9) else "metric"


def _candidates(reg, si_unit) -> list:
    global _dimension_table
    if _dimension_table is None:
        _dimension_table = {}
        for base, names in _CANDIDATES.items():
            key = str(reg.Unit(base).dimensionality)
            _dimension_table[key] = [u for u in (_parse(reg, n) for n in names) if u is not None]
    return _dimension_table.get(str(si_unit.dimensionality), [])


def _best(reg, value: float, si_unit, candidates: list):
    """The largest candidate unit in which the number is still at least 1; if every number is smaller, the smallest unit."""
    q = reg.Quantity(float(value), si_unit)
    sized = [(abs(q.to(c).magnitude), c) for c in candidates]
    at_least_one = [(m, c) for m, c in sized if m >= 1]
    return min(at_least_one, key=lambda t: t[0])[1] if at_least_one else max(sized, key=lambda t: t[0])[1]


def _num(x: float) -> str:
    """Four significant figures; whole numbers from 1000 up are written out (101325, not 1.013e5); scientific only below 0.001 or from 1e9 up."""
    if x == 0:
        return "0"
    a = abs(x)
    if 1000 <= a < 1e9:
        return f"{round(x):d}"
    text = f"{x:.4g}"
    if "e" in text:
        mantissa, exponent = text.split("e")
        return f"{mantissa}×10^{int(exponent)}"
    return text


def _label(unit, shown: float) -> str:
    names = _NAMES.get(str(unit))
    if names is None:
        return format(unit, "~P") or str(unit)
    if isinstance(names, tuple):
        return names[0] if abs(shown) == 1 else names[1]
    return names


def display_answer(value, si_units: str, user_unit_texts: Iterable[str]) -> str | None:
    """'1 yard (0.9144 m)' for a result of 0.9144 m asked in yards; None when there is nothing to add (SI-only question, no sensible unit, not a number)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    texts = [t for t in user_unit_texts if t]
    if not si_units or not texts:
        return None
    reg = _registry()
    if reg is None:
        return None
    users = [u for u in (_parse(reg, t) for t in texts) if u is not None]
    classes = [classify(reg, u) for u in users]
    if not users or all(c == "si" for c in classes):
        return None
    si = _parse(reg, si_units.replace("^", "**"))
    if si is None or si.dimensionless:
        return None
    same_kind = [u for u in users if u.dimensionality == si.dimensionality]
    if same_kind:
        target = same_kind[0]
    elif "customary" in classes:
        options = _candidates(reg, si)
        target = _best(reg, value, si, options) if options else None
    else:
        target = None
    if target is None or str(target) == str(si):
        return None
    try:
        shown = reg.Quantity(float(value), si).to(target).magnitude
    except Exception:  # noqa: BLE001
        return None
    return f"{_num(shown)} {_label(target, shown)} ({_num(float(value))} {si_units})"
