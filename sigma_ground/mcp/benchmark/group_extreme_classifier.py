"""Deterministic cue: the highest or lowest value of a property over an element group.

"What's the highest noble gas first ionization energy?" (atom_004) is an argmax over a set, not a lookup; the router picked Ne
(21.565 eV) where the answer is He (24.587 eV). The cue hands the group and property to `atomic.extreme_in_group`, which refuses
when the group's table is incomplete. misc/STATIC_INTENTS_PLAN.md group E, 2026-10-30.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_WHICH = r"(?P<which>highest|largest|greatest|biggest|lowest|smallest|least)"
_PROP = r"(?P<prop>(?:first\s+)?ioni[sz]ation\s+(?:energy|potential)|atomic\s+mass|atomic\s+weight|atomic\s+number|density)"
_GROUP = r"(?P<group>noble\s+gas(?:es)?|inert\s+gas(?:es)?|rare\s+gas(?:es)?|halogens?|alkali\s+metals?|alkaline\s+earth(?:\s+metals?)?|chalcogens?|pnictogens?)"
_PATTERNS = [
    # "highest noble gas first ionization energy", "the lowest halogen density"
    re.compile(rf"\b{_WHICH}\s+{_GROUP}\s+{_PROP}\b", re.IGNORECASE),
    # "highest first ionization energy of/among/in (the) noble gases"
    re.compile(rf"\b{_WHICH}\s+{_PROP}\s+(?:of|among|in|for|across)\s+(?:the\s+|any\s+|all\s+(?:the\s+)?)?{_GROUP}\b", re.IGNORECASE),
    # "which noble gas has the highest first ionization energy"
    re.compile(rf"\b{_GROUP}\s+(?:has|with)\s+the\s+{_WHICH}\s+{_PROP}\b", re.IGNORECASE),
]


@dataclass
class GroupExtremeMatch:
    tool: str
    args: dict
    rationale: str
    result_override: float | None = None
    result_units: str = ""


def classify_for_group_extreme(question: str) -> GroupExtremeMatch | None:
    for pat in _PATTERNS:
        m = pat.search(question)
        if not m:
            continue
        prop = re.sub(r"\s+", " ", m.group("prop").lower())
        prop = {"ionization energy": "first ionization energy", "ionisation energy": "first ionization energy",
                "first ionisation energy": "first ionization energy", "ionization potential": "first ionization energy",
                "ionisation potential": "first ionization energy", "first ionization potential": "first ionization energy",
                "first ionisation potential": "first ionization energy", "atomic weight": "atomic mass"}.get(prop, prop)
        group = re.sub(r"\s+", " ", m.group("group").lower())
        which = "lowest" if m.group("which").lower() in ("lowest", "smallest", "least") else "highest"
        from sigma_ground.mcp.tools.atomic import extreme_in_group
        r = extreme_in_group(prop, group, which)
        if r.value is None:
            return None                     # an incomplete table: the honest answer is the tool's refusal, which the router path gives
        return GroupExtremeMatch(tool="extreme_in_group", args={"property": prop, "group": group, "which": which},
                                 rationale=r.notes, result_override=float(r.value), result_units=r.units)
    return None


def execute_group_extreme_match(match: GroupExtremeMatch) -> tuple[object, str, str]:
    return (match.result_override, match.result_units,
            f"ANSWER: {match.result_override} {match.result_units}\n\nComputed via group_extreme_classifier: {match.tool} ({match.rationale})")
