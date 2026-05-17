"""Run the 150-question benchmark via sigma-ground MCP + qwen2.5:7b.

This is the long-deferred minimal_qwen_mcp_loop.py from
scripts/run_mcp_with_qwen.md, fleshed out and wired to the benchmark.

The MCP server is spawned as a stdio subprocess. For each question:
  1. Pull the tool list via session.list_tools()
  2. Send the question + tool definitions to qwen2.5:7b via Ollama
  3. If Qwen requests tool calls, dispatch them to the MCP server
  4. Feed tool results back, loop until Qwen produces a final answer
  5. Extract the numeric value from the "ANSWER:" line

Requires:
    pip install mcp httpx
    ollama serve (with qwen2.5:7b pulled)

Usage:
    python -m sigma_ground.mcp.benchmark.run_sigma_ground \
        --model qwen2.5:7b \
        --output sigma_ground/mcp/benchmark/results/sigma_ground_run.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Any


_SYSTEM_PROMPT_BASE = """\
You are a TRANSLATION LAYER between the user and the sigma-ground
physics library, not a physics expert. You are the interpreter /
switchboard, not the answer-giver.

YOUR JOB, IN ORDER:
  1. Read the user's natural-English physics question.
  2. Identify which sigma-ground tool answers it (see TOOL INDEX below).
  3. Call that tool with the correct parameters.
  4. Translate the tool's structured response into a readable answer.

YOU DO NOT HAVE PHYSICS KNOWLEDGE OF YOUR OWN.
  - You do not compute values from formulas in your head.
  - You do not recall constants from training data.
  - You do not "reason in prose" about the problem before calling a tool.
  - The sigma-ground library is the SINGLE source of physics truth here.

This is Q&A MODE: each question is a fresh, standalone problem with no
memory of any previous question. (Conversation mode -- where the
library acts as a persisted simulation playground across turns -- is a
future mode, NOT active now.)

ABSOLUTE RULES:

1. For every numeric value in your answer, you MUST either:
   (a) Call a sigma-ground MCP tool to obtain it, then report the value
       AND the `source` field from the tool's return. Phrase like:
           "value (sigma-ground via <source>)"
   (b) If no tool can supply the value, mark it explicitly:
           "[SOURCE: Fitted due to incompetence -- sigma-ground library
            lacks <X>; best estimate]"

2. NEVER state a numeric value from memory without one of the two tags.
   If you find yourself writing "we can use the formula ..." STOP and
   call a tool instead. That formula is already inside a tool; your
   only job is to find it and call it.

3. Begin your final answer with exactly "ANSWER:" on its own line,
   followed by the numeric value and units, e.g.:
       ANSWER: 1.43 s
   This is so the benchmark scorer can extract the value reliably.
   Put any brief framing after the ANSWER line.

4. If the question is conceptual (no numeric answer), write:
       ANSWER: <one-word or short phrase>

5. Use the EXACT tool and parameter names listed in the TOOL INDEX
   below. Do not invent synonyms. If the index lists
   `initial_speed_m_s`, do NOT pass `velocity`, `speed`, or `v0`. If a
   tool returns `"value": null`, your inputs were wrong -- pick a
   different tool or correct the parameter names/values before falling
   back to the "Fitted due to incompetence" tag.

6. For ANY question that has a numerical answer, you MUST call at least
   one tool before producing the ANSWER: line. No exceptions, no "let
   me calculate" preamble. Even for things you "know" (like the speed
   of light or g at sea level), call `lookup_constant`.

7. REFUSAL TRIAGE -- answer IMMEDIATELY, NO tool call, for these cases:

   (a) FALSE PREMISES. The question presupposes something physically
       false. Answer "ANSWER: false" plus a one-line reason.
         "Is the Earth flat?"                  -> ANSWER: false (Earth is an oblate spheroid)
         "Can perpetual motion work?"          -> ANSWER: false (violates second law of thermodynamics)
         "Can entanglement send info FTL?"     -> ANSWER: false (no-communication theorem)

   (b) CATEGORY ERRORS / NONSENSE. The question asks for a physical
       property of a non-physical entity. Answer "ANSWER: not a
       physical quantity" plus reason.
         "What's the kinetic energy of love?"  -> ANSWER: not a physical quantity (love is abstract)
         "What color is the number 7?"         -> ANSWER: not a physical quantity (numbers are abstract)
         "What's the mass of an idea?"         -> ANSWER: not a physical quantity (ideas have no mass; see Landauer's principle for the closest physics analog)

   (c) IMPOSSIBLE EXACT VALUES. The question asks for a value that
       provably doesn't exist or cannot be measured. Answer
       "ANSWER: undefined" plus reason.
         "Last digit of pi?"                   -> ANSWER: undefined (pi is irrational; no last digit)
         "Exact position AND momentum of an electron?" -> ANSWER: undefined (Heisenberg uncertainty principle)
         "Exact value of Avogadro's number?"   -> Special case: SINCE 2019 SI redefinition, Avogadro IS exact (6.02214076e23). Still call `lookup_constant`.

   Rule 6 ("must call a tool for numerical answers") does NOT apply to
   refusal cases -- the answer is a refusal, not a number.
"""


_VALUE_RE = re.compile(
    r"ANSWER:\s*([\-+]?[0-9]+(?:\.[0-9]+)?(?:[eE][\-+]?[0-9]+)?)"
    r"(?:\s*([^\n]*))?",
    re.IGNORECASE,
)

_TOOL_VALUE_RE = re.compile(
    r'"value"\s*:\s*([\-+]?[0-9]+(?:\.[0-9]+)?(?:[eE][\-+]?[0-9]+)?)'
)
_TOOL_UNITS_RE = re.compile(r'"units"\s*:\s*"([^"]*)"')


def _extract_value(answer: str) -> tuple[Any, str]:
    """Extract numeric value from an LLM 'ANSWER:' line."""
    m = _VALUE_RE.search(answer)
    if not m:
        m2 = re.search(r"ANSWER:\s*([^\n]+)", answer, re.IGNORECASE)
        if m2:
            return m2.group(1).strip(), ""
        return None, ""
    val_str = m.group(1)
    units = (m.group(2) or "").strip()
    try:
        return float(val_str), units
    except ValueError:
        return val_str, units


def _extract_value_from_tool_calls(tool_calls: list[dict],
                                       primary_tool_expected: str | None = None
                                       ) -> tuple[Any, str]:
    """Fallback: pull value+units from a tool call's JSON result.

    Strategy (in order):
      1. If primary_tool_expected is set, prefer the LAST call to that
         tool (it's the tool the corpus author meant should answer this).
      2. Otherwise fall back to the LAST tool call with a numeric value.

    This addresses the "right tool, wrong value reported" bug where
    Qwen called multiple tools and the LAST one was an exploratory
    dud (e.g., ohms_law called incidentally) but the correct tool
    (e.g., momentum) was called earlier with the right value.
    """
    def _try_extract(tc: dict) -> tuple[Any, str] | None:
        text = tc.get("result_text", "") or ""
        m_val = _TOOL_VALUE_RE.search(text)
        if not m_val:
            return None
        try:
            val = float(m_val.group(1))
        except ValueError:
            return None
        m_units = _TOOL_UNITS_RE.search(text)
        units = m_units.group(1) if m_units else ""
        return val, units

    # Pass 1: prefer the LAST call to the expected primary tool.
    if primary_tool_expected:
        for tc in reversed(tool_calls):
            if tc.get("name") == primary_tool_expected:
                got = _try_extract(tc)
                if got is not None:
                    return got
    # Pass 2: fall back to LAST tool call with any numeric value.
    for tc in reversed(tool_calls):
        got = _try_extract(tc)
        if got is not None:
            return got
    return None, ""


def _all_tool_values(tool_calls: list[dict]) -> list[float]:
    """Return every numeric "value" present in any tool result.

    Used by the answer-text validator: if Qwen's ANSWER: line gives a
    number that doesn't match ANY tool's result (within 1% relative),
    distrust it and prefer the tool-based fallback. This catches the
    'Qwen quoted the wrong tool's value' bug (mech_intro_009 reported
    8.0 V from Ohm's law when the momentum tool said 37500 kg m/s).
    """
    out: list[float] = []
    for tc in tool_calls:
        text = tc.get("result_text", "") or ""
        m = _TOOL_VALUE_RE.search(text)
        if not m:
            continue
        try:
            out.append(float(m.group(1)))
        except ValueError:
            continue
    return out


def _value_matches_any_tool(text_val: float, tool_values: list[float],
                              tol_rel: float = 0.01) -> bool:
    """Is text_val close to ANY of the tool values (within 1%)?"""
    for tv in tool_values:
        if tv == 0.0 and abs(text_val) < 1e-30:
            return True
        if tv == 0.0:
            continue
        if abs(text_val - tv) / abs(tv) <= tol_rel:
            return True
    return False


# Question-pattern -> domain hints. When a question matches a phrase
# in `_PATTERN_HINTS`, point Qwen at the relevant section of the index.
# Surfaced at the TOP of the system prompt so Qwen sees them before
# scanning the alphabetical tool list.
_PATTERN_HINTS: list[tuple[str, str]] = [
    ("event horizon | Schwarzschild | black hole | photon sphere | "
     "ISCO | Hawking | gravitational time dilation | gravitational redshift",
     "gr"),
    ("Hubble | expansion of universe | age of universe | critical density | "
     "MOND | a_0 | dark energy | cosmological",
     "cosmology"),
    ("E=mc^2 | mass to energy | energy to mass | matter conversion | "
     "nuclear binding | fission | fusion | TNT equivalent | megaton",
     "energy"),
    ("convert | unit conversion | how many X in Y | light year to meters | "
     "MeV to joules | eV to joules | electronvolt",
     "units"),
    ("solve equation | integrate | derivative | simplify | algebra | "
     "symbolic math | polynomial root",
     "symbolic"),
    ("Lorentz | time dilation | length contraction | special relativity | "
     "relativistic momentum | Doppler shift",
     "relativity"),
    ("ideal gas | blackbody | Stefan-Boltzmann | Wien | Carnot | "
     "entropy | thermal | temperature in K | melting point | boiling",
     "thermodynamics"),
    ("Snell | lens | refraction | diffraction | Rydberg | hydrogen line | "
     "single slit | double slit | grating",
     "optics"),
    ("Ohm | resistance | capacitor | inductor | RC | RL | RLC | "
     "voltage | current | power dissipated | wavelength of EM",
     "circuits"),
    ("ionization energy | hydrogen-like | photon energy | "
     "Bohr model | atomic transition",
     "atomic"),
    ("planet | star | Sirius | Vega | solar system body | "
     "mass of Earth | radius of Jupiter",
     "astronomy"),
    ("density | refractive index | Young's modulus | band gap | "
     "material property",
     "materials"),
    ("free fall | projectile | kinetic energy | momentum | "
     "escape velocity | orbital velocity",
     "kinematics"),
    ("speed of light | Planck constant | Avogadro | physical constant | "
     "lookup constant",
     "constants"),
]


def _build_tool_index(tools_for_ollama: list[dict]) -> str:
    """Tool inventory grouped by domain + pattern hints at top.

    Earlier flat-list version had Qwen calling `solar_system_body('sun')`
    14 times for 'event horizon of Sun-as-black-hole' because it
    couldn't find `schwarzschild_radius` in a flat alphabetical list.
    This version groups tools by domain and prepends pattern hints
    that map question phrases to the right domain section.

    The LLM also receives the full JSONSchema via the Ollama `tools`
    field; this textual index is the human-readable map.
    """
    # Pull domain + keywords from the manifest (richer than
    # tools_for_ollama, which only carries name + description + JSONSchema).
    try:
        from sigma_ground.mcp.manifest import _PRIMARY_TOOLS
        domain_by_name = {t["name"]: t.get("domain", "other")
                            for t in _PRIMARY_TOOLS}
        keywords_by_name = {t["name"]: t.get("keywords", [])
                              for t in _PRIMARY_TOOLS}
    except Exception:
        domain_by_name = {}
        keywords_by_name = {}

    # Group tools by domain.
    by_domain: dict[str, list[dict]] = {}
    for t in tools_for_ollama:
        name = t["function"]["name"]
        d = domain_by_name.get(name, "other")
        by_domain.setdefault(d, []).append(t)

    # Stable order: most-common physics topics first, then others.
    domain_order = [
        "constants", "units", "kinematics", "circuits", "optics",
        "thermodynamics", "atomic", "relativity", "gr", "cosmology",
        "energy", "astronomy", "materials", "symbolic", "other",
    ]
    seen = set()
    ordered_domains = [d for d in domain_order if d in by_domain]
    for d in by_domain:
        if d not in ordered_domains:
            ordered_domains.append(d)

    lines = [
        "=== TOOL INDEX ===",
        f"({len(tools_for_ollama)} tools, grouped by domain; "
        f"* = required parameter)",
        "",
        "PATTERN HINTS (match question phrasing to the right section):",
    ]
    for phrases, dom in _PATTERN_HINTS:
        lines.append(f"  [{dom}] {phrases}")
    lines.append("")

    for dom in ordered_domains:
        tools = by_domain[dom]
        lines.append(f"## {dom.upper()}")
        for t in tools:
            fn = t["function"]
            name = fn["name"]
            params = fn.get("parameters", {}) or {}
            props = params.get("properties", {}) or {}
            required = set(params.get("required", []))
            param_strs = []
            for pname in props.keys():
                marker = "*" if pname in required else ""
                param_strs.append(f"{pname}{marker}")
            desc = (fn.get("description") or "").split("\n")[0][:90].strip()
            lines.append(f"  {name}({', '.join(param_strs)})")
            if desc:
                lines.append(f"      {desc}")
            kws = keywords_by_name.get(name, [])
            if kws:
                # Show up to 6 keywords per tool; that's enough to catch
                # common phrasings without bloating the prompt.
                shown = " | ".join(kws[:6])
                lines.append(f"      AKA: {shown}")
        lines.append("")
    return "\n".join(lines)


def _build_real_params_map(tools_for_ollama: list[dict]) -> dict[str, set[str]]:
    """Map each tool name to the set of param names it actually accepts.

    Used by the param-alias normalizer to rename Qwen-style synonyms
    (gravity_ms2 -> g_m_s2, velocity -> speed_m_s, etc.) only when the
    canonical form matches the target tool's real signature.
    """
    out: dict[str, set[str]] = {}
    for t in tools_for_ollama:
        fn = t.get("function", {})
        name = fn.get("name", "")
        params = (fn.get("parameters", {}) or {}).get("properties", {}) or {}
        out[name] = set(params.keys())
    return out


async def _run_one_question(session, ollama_url: str, model: str,
                              tools_for_ollama: list[dict],
                              question: str,
                              system_prompt: str,
                              real_params_by_tool: dict[str, set[str]] | None = None,
                              primary_tool_expected: str | None = None) -> dict:
    """Multi-turn tool loop for a single question."""
    import httpx

    messages = [
        {"role": "system",    "content": system_prompt},
        {"role": "user",      "content": question},
    ]
    tool_calls_made: list[dict] = []
    t0 = time.time()
    timeout_s = 120.0
    max_turns = 14
    nudges_sent = 0
    max_nudges = 2

    async with httpx.AsyncClient(timeout=timeout_s) as http:
        for turn in range(max_turns):
            response = await http.post(ollama_url, json={
                "model":   model,
                "messages": messages,
                "tools":    tools_for_ollama,
                "stream":   False,
                "options":  {"temperature": 0.1},
            })
            response.raise_for_status()
            data = response.json()
            msg = data.get("message", {})
            messages.append(msg)

            tcs = msg.get("tool_calls") or []
            if not tcs:
                final = msg.get("content", "") or ""
                val, units = _extract_value(final)
                # If the model produced no tool call AND no ANSWER: line,
                # it has either reasoned in prose without calling a tool
                # OR signalled "let me calculate" but stopped. Nudge it.
                if val is None and "ANSWER:" not in final.upper() \
                       and nudges_sent < max_nudges:
                    # IMPORTANT: re-state the original question in the
                    # nudge. Without it, Qwen sometimes treats the
                    # nudge as a fresh conversation and replies "OK,
                    # please ask your question" -- forgetting the
                    # actual question entirely.
                    nudge = (
                        "STOP. You are a switchboard, not a physics "
                        "expert. You did not call a tool and did not "
                        "produce an ANSWER: line. The user's question "
                        "is still:\n\n"
                        f"    {question}\n\n"
                        "Call the appropriate tool from the TOOL INDEX "
                        "with the correct parameter names. If truly no "
                        "tool fits, produce the ANSWER: line with the "
                        "'[SOURCE: Fitted due to incompetence ...]' "
                        "tag. Respond with EITHER a tool call OR an "
                        "ANSWER: line. Do not reply with prose, do not "
                        "ask for clarification, do not acknowledge "
                        "this message."
                    )
                    messages.append({"role": "user", "content": nudge})
                    nudges_sent += 1
                    continue
                fallback_used = False
                # Distrust text values that don't match any tool result --
                # catches 'Qwen quoted the wrong tool's value' (e.g.
                # mech_intro_009 reported 8.0 V when momentum tool gave 37500).
                if isinstance(val, (int, float)) and tool_calls_made:
                    tool_values = _all_tool_values(tool_calls_made)
                    if tool_values and not _value_matches_any_tool(
                            float(val), tool_values):
                        # Text value diverges from all tool results -- bin it
                        val = None
                        units = ""
                # Fallback: if the LLM forgot the ANSWER: line but did call
                # tools, pull value/units from the appropriate tool result.
                # Prefer the expected primary tool's result if available.
                if val is None and tool_calls_made:
                    val, units = _extract_value_from_tool_calls(
                        tool_calls_made,
                        primary_tool_expected=primary_tool_expected)
                    fallback_used = val is not None
                return {
                    "answer_text":            final,
                    "extracted_value":        val,
                    "extracted_units":        units,
                    "tool_calls":             tool_calls_made,
                    "turns":                  turn + 1,
                    "elapsed_s":              time.time() - t0,
                    "extracted_via_fallback": fallback_used,
                    "nudges_sent":            nudges_sent,
                }

            # Dispatch each tool call to the MCP session
            for tc in tcs:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                # Normalize common Qwen-style param-name synonyms to the
                # tool's canonical names. Without this, e.g.
                # gravity_ms2 -> silently dropped, free_fall_time uses
                # default Earth g, Moon question gets Earth answer.
                renames: list[str] = []
                chain_log: list[str] = []
                if real_params_by_tool is not None and name in real_params_by_tool:
                    from sigma_ground.mcp.benchmark.param_aliases import (
                        normalize_kwargs, resolve_body_name_chain)
                    args, renames = normalize_kwargs(args, real_params_by_tool[name])
                    # Chain body_name -> solar_system_body / named_star -> mass/radius
                    # for tools that want mass_kg / radius_m but were given a body name.
                    args, chain_log = await resolve_body_name_chain(
                        session, args, real_params_by_tool[name])
                try:
                    result = await session.call_tool(name, args)
                    content_parts = []
                    for piece in (result.content or []):
                        text = getattr(piece, "text", None)
                        if text:
                            content_parts.append(text)
                    tool_text = "\n".join(content_parts) or "(empty)"
                except Exception as e:
                    tool_text = f"<TOOL ERROR: {e}>"

                tool_calls_made.append({
                    "name": name,
                    "args": args,
                    "result_text": tool_text[:2000],
                    "renames_applied": renames,
                    "chain_log": chain_log,
                })
                messages.append({
                    "role":    "tool",
                    "content": tool_text[:4000],
                })

            # Loop detection: if the last 3 tool calls are identical
            # (same name + same args), Qwen is stuck repeating itself.
            # This was the dominant failure mode on the first run:
            # 96/150 questions hit max_turns because the model kept
            # calling solar_system_body('earth') 14 times waiting for
            # a different answer that never came.
            if len(tool_calls_made) >= 3:
                last3 = tool_calls_made[-3:]
                if all((c["name"] == last3[0]["name"]
                          and json.dumps(c["args"], sort_keys=True, default=str)
                              == json.dumps(last3[0]["args"], sort_keys=True, default=str))
                         for c in last3):
                    loop_warning = (
                        f"STOP. You have called `{last3[0]['name']}` with "
                        f"the same arguments 3 times in a row. The tool's "
                        f"output will not change. Either:\n"
                        f"  (a) Call a DIFFERENT tool from the TOOL INDEX, OR\n"
                        f"  (b) Produce the ANSWER: line using the values "
                        f"you already have, OR\n"
                        f"  (c) Produce the ANSWER: line with the '[SOURCE: "
                        f"Fitted due to incompetence ...]' tag.\n"
                        f"The user's question is still: {question}"
                    )
                    messages.append({"role": "user", "content": loop_warning})
                    # Count as a nudge so we don't loop on the warning too
                    nudges_sent += 1

        # Hit max turns -- try fallback before giving up
        val, units = _extract_value_from_tool_calls(
            tool_calls_made, primary_tool_expected=primary_tool_expected)
        return {
            "answer_text":            "<exceeded max turns>",
            "extracted_value":        val,
            "extracted_units":        units,
            "tool_calls":             tool_calls_made,
            "turns":                  max_turns,
            "elapsed_s":              time.time() - t0,
            "extracted_via_fallback": val is not None,
            "nudges_sent":            nudges_sent,
        }


async def _amain(args) -> int:
    # Auto-load env vars from the dev-root .env (Ollama URL override, etc.)
    from sigma_ground.mcp.benchmark import load_env_from_dev_root
    load_env_from_dev_root(verbose=True)

    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError:
        print("ERROR: pip install 'mcp>=1.0'", file=sys.stderr)
        return 1
    try:
        import httpx  # noqa: F401
    except ImportError:
        print("ERROR: pip install httpx", file=sys.stderr)
        return 1

    with args.questions.open(encoding="utf-8") as f:
        questions = json.load(f)
    if args.limit:
        questions = questions[:args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, dict] = {}
    if args.resume and args.output.exists():
        with args.output.open(encoding="utf-8") as f:
            for rec in json.load(f):
                # Skip errored, different-model, or no-value records --
                # re-run them. A None extracted_value means we got no
                # useful answer, regardless of how the run terminated.
                ans = rec.get("answer_text", "") or ""
                if ans.startswith("<ERROR") or ans == "<exceeded max turns>":
                    continue
                if rec.get("model") and rec["model"] != args.model:
                    continue
                if rec.get("extracted_value") is None:
                    continue
                existing[rec["id"]] = rec
    out = list(existing.values())

    # Spawn the MCP server
    params = StdioServerParameters(command="sigma-ground-mcp")
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_resp = await session.list_tools()
            # Convert to Ollama's tool-format
            tools_for_ollama = [
                {
                    "type": "function",
                    "function": {
                        "name":        t.name,
                        "description": t.description or "",
                        "parameters":  t.inputSchema or {"type": "object",
                                                            "properties": {}},
                    },
                }
                for t in tools_resp.tools
            ]
            tool_index = _build_tool_index(tools_for_ollama)
            real_params_by_tool = _build_real_params_map(tools_for_ollama)
            sys_prompt = _SYSTEM_PROMPT_BASE + "\n\n" + tool_index
            print(f"MCP server has {len(tools_for_ollama)} tools available")
            print(f"System prompt is {len(sys_prompt)} chars "
                  f"(includes full tool index)")

            for i, q in enumerate(questions):
                if q["id"] in existing:
                    print(f"[{i+1}/{len(questions)}] {q['id']}: skipped (resume)")
                    continue
                print(f"[{i+1}/{len(questions)}] {q['id']}: {q['question'][:60]}...")
                try:
                    result = await _run_one_question(
                        session,
                        args.ollama_url + "/api/chat",
                        args.model,
                        tools_for_ollama,
                        q["question"],
                        sys_prompt,
                        real_params_by_tool=real_params_by_tool,
                        primary_tool_expected=q.get("primary_tool_expected"),
                    )
                except Exception as e:
                    print(f"  ERROR: {e}", file=sys.stderr)
                    result = {
                        "answer_text":     f"<ERROR: {e}>",
                        "extracted_value": None,
                        "extracted_units": "",
                        "tool_calls":      [],
                        "turns":           0,
                        "elapsed_s":       0.0,
                    }
                rec = {
                    "id":     q["id"],
                    "system": "sigma_ground",
                    "model":  args.model,
                    **result,
                }
                out.append(rec)
                with args.output.open("w", encoding="utf-8") as f:
                    json.dump(out, f, indent=2, default=str)

    print(f"\nWrote {args.output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen2.5:14b",
                        help="Ollama model tag. 14b is the default (better "
                              "synthesis after tool calls than 7b); 7b is "
                              "faster but more often forgets the ANSWER: line "
                              "(the fallback extractor catches that case).")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "results" / "sigma_ground_run.json")
    parser.add_argument("--questions", type=Path,
                        default=Path(__file__).parent / "questions.json",
                        help="Path to questions corpus JSON.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    args = parser.parse_args()
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    sys.exit(main())
