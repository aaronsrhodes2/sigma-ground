"""v4 labels: the router quotes a quantity as the question wrote it, and the tool contract converts.

Until v3.1 the gold plan for "radius of 5.09 km" said `radius_m: 5090.0`: the router did the unit arithmetic in its head, which is the one place a small model
cannot be trusted (35% of the unit-bearing arguments in the v3.1 training data were in a unit other than the parameter's, and "7 solar masses" had to become
1.4e31 kg from a recalled constant). v4 writes `radius_m: "5.09 km"`; mcp/contract.py reads the quantity with pint and Mentat does the conversion
(Captain, 2026-10-15: "Return the answer to match the measurement system used").

`relabel_plan` rewrites a gold plan: parameters under their canonical names (mcp/param_names.py) and every unit-bearing numeric argument that the question states
with a unit as that quantity, copied from the question. An argument the question does not state with a unit (derived, looked up, a bare number) is left as it was.
"""
from __future__ import annotations

import copy
import math
import re
from typing import Any, Mapping

from sigma_ground.mcp import contract as C
from sigma_ground.mcp import param_names as PN

_NUMBER = re.compile(r"(?<![\w.^])([-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?)(?!\d)")
_TRAILING = ".,;:?!)"
_SKIP_TOOLS = {"request_clarification"}


def _candidates(question: str):
    """(number as written, number, [what follows it as unit text, longest first])"""
    for m in _NUMBER.finditer(question):
        text = m.group(1)
        try:
            number = float(text.replace(",", ""))
        except ValueError:
            continue
        words = [w.rstrip(_TRAILING) for w in re.findall(r"[^\s,;:()?!]+", question[m.end():])[:3]]
        units = [" ".join(words[:k]) for k in (3, 2, 1) if len(words) >= k and words[k - 1]]
        yield text, number, units


def find_quantity(question: str, value: float, unit: str) -> str | None:
    """The quantity in `question` (as "<number> <unit>") whose value in `unit` equals `value`, or None.
    The number is written without thousands commas ("1,500 m" -> "1500 m"); the unit is exactly as the question wrote it."""
    reg = C._registry()
    for text, number, units in _candidates(question):
        for cand in units:
            try:
                got = reg.Quantity(number, C.normalize_unit_text(cand)).to(unit).magnitude
            except Exception:  # noqa: BLE001 — pint raises several things for a phrase that is not a unit
                continue
            if math.isclose(float(got), float(value), rel_tol=1e-6, abs_tol=1e-12):
                return f"{text.replace(',', '')} {cand}"
    return None


def _is_float_annotation(ann: str) -> bool:
    return bool(ann) and "float" in ann and all(b.strip() in ("float", "None") for b in ann.split("|"))


def relabel_step(question: str, tool: str, args: Mapping[str, Any], signature: list[dict] | None) -> tuple[dict, int]:
    """(args under canonical names with quantities quoted, how many were quoted). `signature` is tool_census.signatures()[tool]; None leaves args alone."""
    if signature is None or tool in _SKIP_TOOLS:
        return dict(args), 0
    real = [p["name"] for p in signature]
    ann = {p["name"]: p.get("ann", "") for p in signature}
    renamed = PN.rename_map(tool, real)
    out: dict = {}
    quoted = 0
    for k, v in PN.canonical_args(tool, args, real).items():
        spec = C.SPECS.get(k)
        real_name = PN.inverse(renamed).get(k, k)
        if (spec is not None and spec.unit and isinstance(v, (int, float)) and not isinstance(v, bool)
                and _is_float_annotation(ann.get(real_name, ""))):
            q = find_quantity(question, float(v), spec.unit)
            if q is not None:
                v, quoted = q, quoted + 1
        out[k] = v
    return out, quoted


def relabel_plan(question: str, plan: dict, signatures: Mapping[str, list[dict]]) -> tuple[dict, int]:
    """The gold plan (v3/v4 shape: {"tool","tool_args"} or {"steps": [...]} or {"decline"}) relabelled; also the number of quantities quoted."""
    plan = copy.deepcopy(plan)
    steps = plan["steps"] if isinstance(plan.get("steps"), list) else ([plan] if "tool" in plan else [])
    total = 0
    for s in steps:
        key = "tool_args" if "tool_args" in s else "args"
        if key in s and isinstance(s[key], dict):
            s[key], n = relabel_step(question, s["tool"], s[key], signatures.get(s["tool"]))
            total += n
    return plan, total
