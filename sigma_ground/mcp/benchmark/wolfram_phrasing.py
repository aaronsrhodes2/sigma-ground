"""Wolfram-phrasing: generalizable query rewriting to reduce Wolfram Alpha
parse failures (the "false negative" the Captain's goal names explicitly).

Wolfram's own query parser wants terse, formula-flavored input; this
project's corpus is written in conversational, narrative English ("If I
drop a copper ball from 10 meters..."). Two complementary mechanisms,
tried in priority order, before falling back to the existing regex
reformulation cascade (daily_job.py::reformulate_for_wolfram):

  1. HAND-AUTHORED OVERRIDE (wolfram_phrasing_overrides.json) -- reserved
     for genuinely idiosyncratic questions neither mechanism below can
     confidently rewrite. Small, explicit, easy to audit.

  2. TOOL-CALL SYNTHESIS (this module's main contribution) -- when
     sigma_ground already answered a question correctly, its tool call
     already encodes "which formula, which numbers" -- exactly what
     Wolfram wants. Render tool_name + args as a terse phrase and try it.
     This scales to future questions for free (no per-question authoring)
     since it rides on whatever sigma_ground itself already resolved.

NON-DERIVED (audit): unit-suffix stripping and the tool_name->words
rendering are string heuristics, not physics -- a wrong or ugly synthesized
phrase just fails to parse (same as today's regex variants), it never
produces a wrong ANSWER (Wolfram's own parser is the final arbiter).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_OVERRIDES_PATH = Path(__file__).parent / "wolfram_phrasing_overrides.json"

# Unit suffixes stripped from arg NAMES so the synthesized phrase reads as
# "value unit" (e.g. height_m=10 -> "10 m") instead of "height_m 10".
_UNIT_SUFFIXES = [
    "_m_s2", "_m_s", "_kg", "_m2", "_m3", "_deg", "_rad", "_hz", "_ev",
    "_mev", "_kev", "_j", "_w", "_pa", "_k", "_c", "_a", "_v", "_ohm",
    "_f", "_au", "_ly", "_pc", "_yr", "_s", "_m",
]


def _load_overrides() -> dict:
    try:
        d = json.loads(_OVERRIDES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {k: v for k, v in d.items() if not k.startswith("_")}


def override_phrasing(qid: str) -> str | None:
    """A hand-authored override for this question id, if one exists."""
    return _load_overrides().get(qid, {}).get("phrasing")


def _split_unit(arg_name: str) -> tuple[str, str]:
    """('height_m', ) -> ('height', 'm'). No match -> (arg_name, '')."""
    for suf in sorted(_UNIT_SUFFIXES, key=len, reverse=True):
        if arg_name.endswith(suf):
            return arg_name[: -len(suf)], suf[1:]
    return arg_name, ""


def synthesize_from_tool_call(tool_name: str, args: dict) -> str | None:
    """Render a terse Wolfram-flavored phrase from a successful tool call.

    e.g. free_fall_time(height_m=10, g_m_s2=9.81)
         -> "free fall time height 10 m gravity 9.81 m/s^2"

    Returns None if args is empty/unusable (a bare tool_name alone is
    rarely a good Wolfram query -- e.g. "escape velocity" with no body).
    """
    if not args:
        return None
    words = tool_name.replace("_", " ")
    parts = [words]
    for k, v in args.items():
        if v is None or isinstance(v, (dict, list)):
            continue
        name, unit = _split_unit(k)
        name = name.replace("_", " ")
        if unit == "s2" or k.endswith("_m_s2"):
            unit = "m/s^2"
        elif k.endswith("_m_s"):
            unit = "m/s"
        piece = f"{name} {v}" if not unit else f"{name} {v} {unit}"
        parts.append(piece)
    phrase = " ".join(parts)
    phrase = re.sub(r"\s+", " ", phrase).strip()
    return phrase or None


def phrasing_variants(question: str, qid: str,
                      sg_record: dict | None = None) -> list[str]:
    """Priority-ordered Wolfram query variants: override > tool-call
    synthesis > the existing regex reformulation cascade > verbatim.

    `sg_record` is this question's own sigma_ground run record (has
    'tool_calls' if a tool actually resolved it) -- pass None if
    unavailable (synthesis is then simply skipped, not an error).
    """
    from sigma_ground.mcp.benchmark.daily_job import reformulate_for_wolfram

    variants: list[str] = []

    override = override_phrasing(qid)
    if override:
        variants.append(override)

    if sg_record:
        tcs = sg_record.get("tool_calls") or []
        if tcs:
            synth = synthesize_from_tool_call(tcs[-1]["name"], tcs[-1].get("args") or {})
            if synth:
                variants.append(synth)

    variants.extend(reformulate_for_wolfram(question))

    seen = set()
    out = []
    for v in variants:
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out
