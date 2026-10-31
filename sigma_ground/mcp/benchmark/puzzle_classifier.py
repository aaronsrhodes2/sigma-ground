"""Deterministic cues for the adversarial suite's puzzles and unit-count questions (2026-10-30).

Monty Hall, the birthday problem and two trains closing go to the tools in mcp/tools/puzzles.py; "how many heartbeats in 80 years at
70 per minute" and "if a nanosecond cost a dollar, what would 13.787 billion years cost" are a rate times a duration, converted with
pint (the same registry as convert_units). Each cue needs the numbers in the question; nothing is assumed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_NUM = r"(\d[\d,]*(?:\.\d+)?)"
_SCALE = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}
_PER = {"second": 1.0, "minute": 60.0, "hour": 3600.0, "day": 86400.0, "year": 365.25 * 86400.0}
_SMALL = {"nanosecond": 1e-9, "microsecond": 1e-6, "millisecond": 1e-3, "second": 1.0, "minute": 60.0, "hour": 3600.0, "day": 86400.0}


@dataclass
class PuzzleMatch:
    tool: str
    args: dict
    rationale: str
    result_override: float | None = None
    result_units: str = ""


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def classify_for_puzzle(question: str) -> PuzzleMatch | None:
    q = question
    # Monty Hall: switching wins with (n - 1) / n
    if re.search(r"\bmonty\s+hall\b", q, re.IGNORECASE):
        m = re.search(rf"{_NUM}\s+doors\b", q, re.IGNORECASE)
        n = int(_num(m.group(1))) if m else 3
        from sigma_ground.mcp.tools.puzzles import monty_hall_switch
        r = monty_hall_switch(n)
        if r.value is not None:
            return PuzzleMatch("monty_hall_switch", {"n": n}, r.notes, float(r.value), "probability")
    # birthday problem: n people, at least two share a birthday
    m = re.search(rf"\b(?:room|group|class|party)\s+of\s+{_NUM}\s+(?:people|persons|students|guests)\b|\b{_NUM}\s+(?:people|persons|students|guests)\b", q, re.IGNORECASE)
    if m and re.search(r"\bbirthday", q, re.IGNORECASE) and re.search(r"\bshare|\bsame\s+birthday|\bat\s+least\s+two", q, re.IGNORECASE):
        n = int(_num(m.group(1) or m.group(2)))
        from sigma_ground.mcp.tools.puzzles import birthday_collision
        r = birthday_collision(n)
        if r.value is not None:
            return PuzzleMatch("birthday_collision", {"n": n}, r.notes, float(r.value), "probability")
    # two bodies closing: "<d> km apart ... <v1> km/h ... <v2> km/h ... how long until they meet"
    if re.search(r"\b(?:approach|toward|towards|head-on|meet|collide)", q, re.IGNORECASE):
        md = re.search(rf"{_NUM}\s*(km|kilomet(?:er|re)s?|m|met(?:er|re)s?|miles?)\s+apart\b", q, re.IGNORECASE)
        mv = re.findall(rf"{_NUM}\s*(km/h|kph|km\s+per\s+hour|m/s|mph|miles\s+per\s+hour)", q, re.IGNORECASE)
        if md and len(mv) >= 2:
            d_m = _num(md.group(1)) * (1000.0 if md.group(2).lower().startswith("k") else 1609.344 if md.group(2).lower().startswith("mi") else 1.0)
            def to_ms(v, u):
                u = u.lower()
                return _num(v) * (1 / 3.6 if u.startswith("k") else 0.44704 if u.startswith("m") and "s" not in u else 1.0)
            v1, v2 = to_ms(*mv[0]), to_ms(*mv[1])
            from sigma_ground.mcp.tools.puzzles import closing_time
            r = closing_time(d_m, v1, v2)
            if r.value is not None:
                in_hours = "h" in mv[0][1].lower() and "/s" not in mv[0][1].lower()
                return PuzzleMatch("closing_time", {"distance_m": d_m, "velocity1_m_s": v1, "velocity2_m_s": v2}, r.notes,
                                   float(r.value) / 3600.0 if in_hours else float(r.value), "h" if in_hours else "s")
    # a rate times a duration: "how many heartbeats ... 80 years and 70 beats per minute"
    if re.search(r"\bhow\s+many\b", q, re.IGNORECASE):
        mr = re.search(rf"{_NUM}\s+(?:beats|times|events|breaths|steps)\s+per\s+(second|minute|hour|day|year)\b", q, re.IGNORECASE)
        mt = re.search(rf"{_NUM}\s*(?:(thousand|million|billion|trillion)\s+)?(years?|days?|hours?)\b", q, re.IGNORECASE)
        if mr and mt:
            rate_per_s = _num(mr.group(1)) / _PER[mr.group(2).lower()]
            duration_s = _num(mt.group(1)) * _SCALE.get((mt.group(2) or "").lower(), 1.0) * _PER[mt.group(3).lower().rstrip("s")]
            count = rate_per_s * duration_s
            unit = re.search(r"\bhow\s+many\s+(\w+)", q, re.IGNORECASE).group(1).lower()
            return PuzzleMatch("convert_units", {"value": _num(mt.group(1)), "from_units": mt.group(3), "to_units": mr.group(2)},
                               f"{mr.group(1)} per {mr.group(2)} for {mt.group(0)}: {count:.4g}", count, unit)
    # a duration counted in a small unit, with a price per unit: "if a nanosecond cost one dollar ... 13.787 billion years"
    ms = re.search(r"\b(nanosecond|microsecond|millisecond|second|minute|hour|day)s?\b", q, re.IGNORECASE)
    mt = re.search(rf"{_NUM}\s*(?:(thousand|million|billion|trillion)\s+)?(years?|days?|hours?)\b", q, re.IGNORECASE)
    if ms and mt and re.search(r"\bcost|\bdollars?\b|\bprice|\bhow\s+many\b", q, re.IGNORECASE) and ms.group(1).lower() != mt.group(3).lower().rstrip("s"):
        duration_s = _num(mt.group(1)) * _SCALE.get((mt.group(2) or "").lower(), 1.0) * _PER[mt.group(3).lower().rstrip("s")]
        count = duration_s / _SMALL[ms.group(1).lower()]
        mp = re.search(r"\bone\s+(?:us\s+)?dollar|\b\$?1\s+(?:us\s+)?dollar|\bper\s+dollar|\bdollars?\b", q, re.IGNORECASE)
        units = "USD" if mp else ms.group(1).lower() + "s"
        return PuzzleMatch("convert_units", {"value": _num(mt.group(1)), "from_units": mt.group(3), "to_units": ms.group(1)},
                           f"{mt.group(0)} = {count:.4g} {ms.group(1)}s", count, units)
    return None


def execute_puzzle_match(match: PuzzleMatch) -> tuple[object, str, str]:
    return (match.result_override, match.result_units,
            f"ANSWER: {match.result_override} {match.result_units}\n\nComputed via puzzle_classifier: {match.tool} ({match.rationale})")
