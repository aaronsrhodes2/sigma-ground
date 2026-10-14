"""The router's plan format (v2), shared by the dataset, the harness and the evaluator.

A router reads a question and replies with ONE JSON object:

    {"tool": T, "tool_args": {...}}                       one call
    {"steps": [{"tool": T1, "tool_args": {...}},          two or three calls; "$1" is the
               {"tool": T2, "tool_args": {"x": "$1"}}]}   headline value of the first call
    {"tool": "request_clarification", "tool_args": {...}}  a required input is missing
    {"decline": "reason"}                                  not a physical question

Why `tool_args` and not `args` (v1): Ollama sorts a JSON schema's properties
alphabetically, so a schema with `tool` and `args` makes the grammar write
`args` first -- the reverse of the training order -- and the tuned v1 router
fell from 99.7% to 73% right-tool on the same 300 validation records. `tool`
sorts before `tool_args`, and `decline` < `steps` < `tool`, so the grammar and
the training order agree and the tool-name enum can stay.

Everything here is pure: no server, no model. Both spellings (`args` /
`tool_args`) are accepted when reading; v2 is what gets written and trained.
"""
from __future__ import annotations

import json
import re
from typing import Any

MAX_STEPS = 3
REF_RE = re.compile(r"^\$([1-9])(?:\.([A-Za-z_][A-Za-z0-9_]*))?$")   # "$2" or "$1.mass_kg"

ROUTER_SYSTEM_PROMPT_V2 = (
    "You are Mentat's router. Read the user's question and reply with ONE JSON "
    "object and nothing else:\n"
    '  {"tool": "<tool name>", "tool_args": {...}} to call a library tool. Convert '
    "every input to the unit its parameter name states (mass_kg in kilograms, "
    "wavelength_m in metres).\n"
    '  {"steps": [{"tool": "<tool name>", "tool_args": {...}}, {"tool": "<tool name>", '
    '"tool_args": {"<parameter>": "$1"}}]} when the answer needs two or three tools in '
    'order. "$1" is the result of the first call, "$2" of the second.\n'
    '  {"tool": "request_clarification", "tool_args": {"variable": "<parameter>", '
    '"question": "<what to ask>"}} when a required input is missing. Never '
    "invent it.\n"
    '  {"decline": "<reason>"} when it is not a question about a physical '
    "system.\n"
    "Never compute or recall the answer yourself; the tool does that."
)


# v3 = v2 + a field reference: a lookup tool returns several fields (solar_system_body: mass_kg,
# radius_m, surface_temp_k, ...), and the plan names the one a later call needs. Before, the harness
# guessed the field from the question's wording; a trained router says it.
ROUTER_SYSTEM_PROMPT_V3 = ROUTER_SYSTEM_PROMPT_V2.replace(
    '"$1" is the result of the first call, "$2" of the second.\n',
    '"$1" is the result of the first call, "$2" of the second. When a call looks something up '
    'and returns several fields (solar_system_body, named_star), "$1.<field>" is one of them, '
    'e.g. "$1.mass_kg".\n')
assert ROUTER_SYSTEM_PROMPT_V3 != ROUTER_SYSTEM_PROMPT_V2

MULTISTEP = ("v2", "v3")        # plan versions whose model may write {"steps": [...]}


def system_prompt(plan: str) -> str:
    """The exact system prompt a router of this plan version is trained and run with."""
    if plan == "v3":
        return ROUTER_SYSTEM_PROMPT_V3
    if plan == "v2":
        return ROUTER_SYSTEM_PROMPT_V2
    from sigma_ground.mcp.benchmark.router_dataset import ROUTER_SYSTEM_PROMPT
    return ROUTER_SYSTEM_PROMPT


class PlanError(ValueError):
    """A plan that cannot be run (bad reference, unknown shape)."""


# ── reading ──────────────────────────────────────────────────────────────

def _step(d: Any) -> dict | None:
    if not isinstance(d, dict) or not isinstance(d.get("tool"), str):
        return None
    args = d.get("tool_args", d.get("args"))
    return {"tool": d["tool"], "args": args if isinstance(args, dict) else {}}


def canonical(plan: Any) -> dict | None:
    """A plan in either spelling -> {"kind": "decline", "reason"} or
    {"kind": "steps", "steps": [{"tool", "args"}, ...]}; None if it is neither."""
    if not isinstance(plan, dict):
        return None
    if "decline" in plan and "tool" not in plan and "steps" not in plan:
        return {"kind": "decline", "reason": plan["decline"]}
    if "steps" in plan:
        raw = plan["steps"]
        if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_STEPS:
            return None
        steps = [_step(s) for s in raw]
        return None if any(s is None for s in steps) else {"kind": "steps", "steps": steps}
    one = _step(plan)
    return None if one is None else {"kind": "steps", "steps": [one]}


def refs_in(args: dict) -> list[int]:
    """The step numbers an argument dict refers to ("$2" -> 2)."""
    out = []
    for v in (args or {}).values():
        if isinstance(v, str) and (m := REF_RE.match(v)):
            out.append(int(m.group(1)))
    return out


def check_refs(steps: list[dict]) -> None:
    """Every reference must point at an earlier step. Raises PlanError."""
    for i, s in enumerate(steps, start=1):
        for r in refs_in(s["args"]):
            if r >= i:
                raise PlanError(f"step {i} refers to ${r}, which has not run yet")


def resolve_args(args: dict, values: list[Any], raws: list[Any] | None = None) -> dict:
    """Replace "$n" with step n's value (values[0] is step 1's) and "$n.field" with that
    field of step n's whole result (raws[0] is step 1's; `values` when not given). Raises
    PlanError if that step produced no number."""
    out = {}
    for k, v in (args or {}).items():
        if isinstance(v, str) and (m := REF_RE.match(v)):
            n = int(m.group(1))
            if n > len(values):
                raise PlanError(f"${n} has not run yet")
            val = values[n - 1]
            if m.group(2):
                whole = (raws if raws is not None else values)[n - 1]
                if not isinstance(whole, dict) or m.group(2) not in whole:
                    raise PlanError(f"${n} has no field {m.group(2)!r}")
                val = whole[m.group(2)]
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise PlanError(f"${n} is not a number: {val!r}")
            out[k] = val
        else:
            out[k] = v
    return out


# ── writing ──────────────────────────────────────────────────────────────

# Ollama's grammar-constrained decoding (`format`) cannot write an exponent with
# a leading zero: it stops after the zero, so 2.42e-06 comes out as 2.42e-0, a
# plausible number wrong by a factor of a million. Measured on 181 exponents in
# constrained replies: e-05, e-06, e-07, e-09 were cut every time (46 of 46);
# e-10, e-15, e-26 and e+22 .. e+31 survived every time (135 of 135). Python
# writes the zero-padded form, so the v1 training data taught exactly the
# spelling the grammar cannot produce. `dumps_plan` writes 2.42e-6 instead
# (the same JSON number); the free format never cuts either spelling.
_JSON_TOKEN = re.compile(r'"(?:[^"\\]|\\.)*"|(-?\d+(?:\.\d+)?)[eE]([-+]?)0*(\d+)')


def _fix_exponents(text: str) -> str:
    def repl(m: re.Match) -> str:
        if m.group(1) is None:          # a string literal: leave it alone
            return m.group(0)
        return f"{m.group(1)}e{m.group(2)}{m.group(3)}"
    return _JSON_TOKEN.sub(repl, text)


def dumps_plan(plan: Any) -> str:
    """json.dumps for a plan, with exponents written without a leading zero."""
    return _fix_exponents(json.dumps(plan, ensure_ascii=False))


def make_plan(steps: list[dict]) -> dict:
    """[{"tool", "args"}, ...] -> the v2 plan object (one call or `steps`)."""
    if len(steps) == 1:
        return {"tool": steps[0]["tool"], "tool_args": steps[0]["args"]}
    return {"steps": [{"tool": s["tool"], "tool_args": s["args"]} for s in steps]}


def to_v2(plan: dict) -> dict:
    """Rewrite a v1 plan (`args`) in the v2 spelling; declines pass through."""
    c = canonical(plan)
    if c is None or c["kind"] == "decline":
        return plan
    return make_plan(c["steps"])


def router_format(tool_names: list[str], steps: bool = True) -> dict:
    """JSON schema for Ollama's `format`. Ollama orders properties alphabetically,
    which here is the training order. steps=False is single-call plans only."""
    names = sorted(set(tool_names))
    call = {"type": "object",
            "properties": {"tool": {"type": "string", "enum": names},
                           "tool_args": {"type": "object"}},
            "required": ["tool", "tool_args"]}
    alts = [call]
    if steps:
        alts.append({"type": "object",
                     "properties": {"steps": {"type": "array", "minItems": 2,
                                              "maxItems": MAX_STEPS, "items": call}},
                     "required": ["steps"]})
    alts.append({"type": "object", "properties": {"decline": {"type": "string"}},
                 "required": ["decline"]})
    return {"anyOf": alts}


def keys_in_training_order(schema: dict) -> bool:
    """True if every object in the schema lists its properties in alphabetical
    order, so Ollama's sorting cannot change the order the model was trained in."""
    def walk(node: Any) -> bool:
        if isinstance(node, dict):
            props = node.get("properties")
            if isinstance(props, dict) and list(props) != sorted(props):
                return False
            return all(walk(v) for v in node.values())
        if isinstance(node, list):
            return all(walk(v) for v in node)
        return True
    return walk(schema)
