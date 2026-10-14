"""Reading a tool's result as an answer, for every shape a Mentat tool returns.

A census of all 232 tools (tools/tool_census.py shapes, one gold call each) found that the
router path could read only the tools whose `value` is a number, or a dict from which the
question's wording picks a field: 113 + 73 of 232. It could not read a symbolic string
(derivatives, integrals, factoring, matrix inverse and product, transforms: 14 tools), a
numeric string (limits, sums, determinants: 3), a list of numbers or expressions (roots,
eigenvalues, linear systems, gradients: 7), or a procedure with no `value` key (5). Every
non-numeric math tool was among them, so the whole math family failed in router mode
whatever the question. The cascade reads them: math_classifier._try_evaluate.

`present` is that same conversion, used by the router path, so the two cannot drift apart.
`tool_error` names what a rejected call got wrong, so a missing parameter is not reported as
"returned no value".

Both are opt-in in the harness (tweaks.py: `structured_results`) so every earlier run stays
reproducible.
"""
from __future__ import annotations

import json
import re
from typing import Any

from sigma_ground.mcp.benchmark.math_classifier import _try_evaluate


def present(result_text: str) -> tuple[Any, str] | None:
    """(value, units) for a result whose `value` is a string or a list, else None.

    Numbers stay as the harness reads them today; dicts go through the question-keyword field
    picker; this handles only what they cannot. A numeric string becomes a float, a list of
    numeric strings a list of floats, a symbolic string that reduces to a number that number,
    and anything still symbolic (`cos(x)`, a matrix) stays the string it is.
    """
    try:
        parsed = json.loads(result_text)
    except (TypeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    value = parsed.get("value")
    if isinstance(value, bool) or not isinstance(value, (str, list)):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    if isinstance(value, list) and not value:
        return None
    shown = _try_evaluate(value)
    return (shown, parsed.get("units") or "") if shown is not None else None


_MISSING = re.compile(r"^(\w+)\s*\n?\s*Field required", re.M)
_WRONG = re.compile(r"^(\w+)\s*\n\s*Input should be ([^\[\n]+)", re.M)
_EXTRA = re.compile(r"^(\w+)\s*\n\s*Unexpected keyword argument", re.M)


def tool_error(tool: str, result_text: str) -> str | None:
    """What a rejected call got wrong, in a few words, or None if this is not a rejection.

    The server answers a bad call with pydantic's text ("1 validation error for fooArguments /
    n_rare / Field required ..."); a tool that declines a valid-looking input answers
    `{"value": null, "source": "invalid input", "notes": ...}`.
    """
    text = result_text or ""
    if text.startswith("Error executing tool"):
        missing = _MISSING.findall(text)
        if missing:
            return f"{tool}: missing required parameter {', '.join(missing)}"
        wrong = _WRONG.findall(text)
        if wrong:
            return f"{tool}: wrong type for {', '.join(p for p, _ in wrong)} (expected {wrong[0][1].strip()})"
        extra = _EXTRA.findall(text)
        if extra:
            return f"{tool}: unknown parameter {', '.join(extra)}"
        return f"{tool}: rejected the call ({text.splitlines()[0][len('Error executing tool '):][:100]})"
    try:
        parsed = json.loads(text)
    except ValueError:
        return None
    if isinstance(parsed, dict) and parsed.get("value") is None and parsed.get("source") == "invalid input":
        return f"{tool}: invalid input ({str(parsed.get('notes') or '')[:100]})"
    return None
