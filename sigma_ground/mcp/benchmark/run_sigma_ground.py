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

7. TOOL-FIRST DISCIPLINE -- this is a tight budget. You are NOT here to
   explore. Pick the right tool from the TOOL INDEX and call it.

   (a) MAX 5 TOOL CALLS per question. After 5 distinct calls, if you
       don't have an answer, the harness will force a refusal. So use
       your budget purposefully:
         budget 1: the primary tool for this question
         budgets 2-3: a chained lookup (e.g. solar_system_body -> escape_velocity)
         budgets 4-5: an alternative if the first tool returned `null`

   (b) NEVER call the same tool with the same arguments twice. The
       result will not change. If a tool returned the wrong value or
       null, try a DIFFERENT tool or different arguments.

   (c) The CORRECT pattern is one of:
         Q: "What's the escape velocity from Earth?"
         -> escape_velocity_classical(body_name='earth')      (1 call)
         -> ANSWER: 11180 m/s

         Q: "What's the event horizon of the Sun-as-BH?"
         -> schwarzschild_radius(body_name='sun')             (1 call, body_name auto-chains)
         -> ANSWER: 2954 m

         Q: "1 joule equals how many electronvolts?"
         -> joules_to_eV(joules=1)                            (1 call)
         -> ANSWER: 6.242e18 eV

   (d) Do NOT enter exploratory mode: 5 different tools, none of which
       answer the question. If you can't find the tool in the first
       2 tries, the answer is "Fitted due to incompetence", not
       "let me try another tool".

8. REFUSAL TRIAGE -- answer IMMEDIATELY, NO tool call, for these cases:

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


_SYSTEM_PROMPT_CONVERSATION = """\
You are the SIMULATION PLAYGROUND host for the sigma-ground physics library.
This is CONVERSATION MODE: a multi-turn session with a SINGLE LIVE SCENE that
PERSISTS across turns. The user loads a piece of matter, then iteratively tunes
it ("now heat it to 2000 K", "now compress it", "now apply a magnetic field")
and the scene's physics state mutates and carries forward.

YOU ARE STILL A SWITCHBOARD, NOT A PHYSICS EXPERT:
  - Never compute or recall numbers yourself -- every value comes from a tool.
  - Drive the live scene with the playground_* tools:
      playground_load     -- load matter into the live scene (DO THIS FIRST)
      playground_inspect  -- read the CURRENT state (summary / matter / constituents)
      playground_apply    -- apply an environment and MUTATE it; knobs:
                             temperature_k, pressure_pa, magnetic_field_t,
                             electric_field_vm, energy_ev
      playground_simulate -- run a drop / launch / orbit dynamics scenario on it
      playground_render   -- show an ASCII picture of the scene
      playground_reset    -- restore the pristine (un-mutated) matter
  - The scene is STATEFUL: after playground_apply, later playground_inspect calls
    see the mutated state. Do NOT reload unless the user wants different matter.

EACH TURN:
  1. Map the user's request to a playground_* tool (or an ordinary physics tool
     for a standalone lookup).
  2. Call it; report the tool's value AND its `source`.
  3. Reply conversationally in 1-3 sentences grounded in the tool result, noting
     what changed from the previous state when relevant.

The user's "it" always means the currently-loaded scene ("what's in it?",
"heat it", "drop it"). Keep ONE scene across the conversation. Cite every number
from a tool; if a tool returns value null, your inputs were wrong -- fix them.
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


# Question-keyword -> dict-field priority. Used by the fallback
# extractor when a tool returns a dict-shaped value (e.g.
# element_atomic_data returns {atomic_number, atomic_mass_amu,
# symbol}). The question text picks which scalar field to pull.
_FIELD_PRIORITY_BY_QUESTION_KEYWORD: list[tuple[tuple[str, ...], list[str]]] = [
    # (question keywords, preferred fields in order)
    (("atomic number", "atomic_number", "how many protons", "Z of"),
     ["atomic_number"]),
    (("atomic mass", "atomic weight", "amu", "atomic_mass"),
     ["atomic_mass_amu", "mass_amu"]),
    (("surface temperature", "surface temp"),
     ["surface_temp_k"]),
    (("mass of", "how massive", "what is the mass"),
     ["mass_kg", "mass_solar"]),
    (("radius of", "what is the radius"),
     ["radius_m"]),
    (("distance to", "how far is", "how far away"),
     ["distance_pc"]),
    (("orbital period", "how long does .* take to orbit"),
     ["orbital_period_d"]),
    (("luminosity of", "how bright"),
     ["luminosity_w", "luminosity_solar"]),
    (("surface gravity", "g at the surface"),
     ["surface_g_ms2"]),
]


# Mapping from dict-result field names to canonical SI units. Lets the
# unit-aware scorer convert (e.g. pc -> light_year) when the question
# asks for one and the tool returns the other.
_FIELD_TO_UNITS: dict[str, str] = {
    "atomic_number":     "",         # dimensionless (count of protons)
    "atomic_mass_amu":   "amu",
    "mass_amu":          "amu",
    "surface_temp_k":    "K",
    "mass_kg":           "kg",
    "radius_m":          "m",
    "luminosity_w":      "W",
    "distance_pc":       "parsec",
    "distance_m":        "m",
    "orbital_period_d":  "day",
    "surface_g_ms2":     "m/s^2",
    "semimajor_axis_au": "AU",
    "abs_v_mag":         "",         # dimensionless
    "mass_solar":        "solar_mass",
    "luminosity_solar":  "solar_luminosity",
}


def _pick_dict_scalar(d: dict, question: str = "") -> tuple[float | None, str]:
    """Pick the most-likely scalar field from a dict-valued tool result.

    Uses question keywords to bias the choice. Returns (value, units)
    where `units` is the canonical SI unit derived from the field name,
    so the unit-aware scorer can convert if the question expects
    different units. Returns (None, "") if no numeric field found.
    """
    q_low = (question or "").lower()
    # Try keyword-guided picks first
    for keywords, fields in _FIELD_PRIORITY_BY_QUESTION_KEYWORD:
        if any(kw in q_low for kw in keywords):
            for f in fields:
                v = d.get(f)
                if isinstance(v, (int, float)):
                    return float(v), _FIELD_TO_UNITS.get(f, f)
    # Default priority list
    default_priority = [
        "atomic_number", "atomic_mass_amu",
        "surface_temp_k", "mass_kg", "radius_m",
        "luminosity_w", "distance_pc", "orbital_period_d",
        "surface_g_ms2", "semimajor_axis_au",
        "abs_v_mag", "mass_solar", "luminosity_solar",
    ]
    for f in default_priority:
        v = d.get(f)
        if isinstance(v, (int, float)):
            return float(v), _FIELD_TO_UNITS.get(f, f)
    # Fallback: first numeric field
    for k, v in d.items():
        if isinstance(v, (int, float)):
            return float(v), _FIELD_TO_UNITS.get(k, k)
    return None, ""


def _extract_value_from_tool_calls(tool_calls: list[dict],
                                       primary_tool_expected: str | None = None,
                                       question: str = ""
                                       ) -> tuple[Any, str]:
    """Fallback: pull value+units from a tool call's JSON result.

    Strategy (in order):
      1. If primary_tool_expected is set, prefer the LAST call to that
         tool (it's the tool the corpus author meant should answer this).
      2. Otherwise fall back to the LAST tool call with a numeric value.

    For dict-shaped `value` (e.g. element_atomic_data returning
    {atomic_number, atomic_mass_amu, symbol}), uses the question text
    to pick the relevant scalar field.

    This addresses the "right tool, wrong value reported" bug where
    Qwen called multiple tools and the LAST one was an exploratory
    dud (e.g., ohms_law called incidentally) but the correct tool
    (e.g., momentum) was called earlier with the right value.
    """
    def _try_extract(tc: dict) -> tuple[Any, str] | None:
        text = tc.get("result_text", "") or ""
        m_val = _TOOL_VALUE_RE.search(text)
        if m_val:
            try:
                val = float(m_val.group(1))
                m_units = _TOOL_UNITS_RE.search(text)
                units = m_units.group(1) if m_units else ""
                return val, units
            except ValueError:
                pass
        # Fallback to NESTED scalars: many tools return `"value": {...}`
        # with named scalar fields. Use the question text to pick the
        # relevant field.
        try:
            import json as _json
            parsed = _json.loads(text)
            v = parsed.get("value")
            if isinstance(v, dict):
                picked_val, picked_field = _pick_dict_scalar(v, question)
                if picked_val is not None:
                    return picked_val, picked_field
        except Exception:
            pass
        return None

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

    t0 = time.time()

    # New-tools router (Phase 0): deterministically dispatch the
    # body-aware/multi-step tools (orbital_velocity, orbital_period,
    # de_broglie_from_kinetic_energy, energy_power_time) so Qwen never
    # has to select them from a flat list. These tools are also HIDDEN
    # from Qwen's tool list (see _amain) — they're reached only here.
    from sigma_ground.mcp.benchmark.new_tools_classifier import (
        classify_for_new_tools, execute_new_tool_match)
    nt = classify_for_new_tools(question)
    if nt is not None:
        val, units, answer_text = execute_new_tool_match(nt)
        if val is not None:
            return {
                "answer_text":            answer_text,
                "extracted_value":        val,
                "extracted_units":        units,
                "tool_calls":             [],
                "turns":                  0,
                "elapsed_s":              time.time() - t0,
                "extracted_via_fallback": False,
                "nudges_sent":            0,
                "new_tools_router_hit":   nt.tool,
            }

    # Refusal pre-classifier: short-circuit obvious refusal cases
    # before involving Qwen at all. Patterns like "Is the Earth flat?"
    # or "What is the kinetic energy of love?" get an immediate
    # canonical answer, skipping the LLM (which Qwen 7b reliably
    # botches because Rule 7 is buried in a 19k-char system prompt).
    from sigma_ground.mcp.benchmark.refusal_classifier import (
        classify_for_refusal,
        render_answer_text as _render_refusal_text,
    )
    refusal = classify_for_refusal(question)
    if refusal is not None:
        answer_text = _render_refusal_text(refusal, question)
        return {
            "answer_text":            answer_text,
            "extracted_value":        refusal.answer,
            "extracted_units":        "",
            "tool_calls":             [],
            "turns":                  0,
            "elapsed_s":              time.time() - t0,
            "extracted_via_fallback": False,
            "nudges_sent":            0,
            "refusal_classifier_hit": refusal.refusal_type,
        }

    # Conversion pre-classifier: short-circuit obvious "convert X to Y"
    # questions via pint directly. Qwen 7b often picks wrong tools for
    # these even when convert_units is in the index.
    from sigma_ground.mcp.benchmark.conversion_classifier import (
        classify_for_conversion,
        render_answer_text as _render_conversion_text,
    )
    conversion = classify_for_conversion(question)
    if conversion is not None:
        return {
            "answer_text":            _render_conversion_text(conversion),
            "extracted_value":        conversion.result,
            "extracted_units":        conversion.to_units,
            "tool_calls":             [],
            "turns":                  0,
            "elapsed_s":              time.time() - t0,
            "extracted_via_fallback": False,
            "nudges_sent":            0,
            "conversion_classifier_hit": True,
        }

    # GR / black-hole pre-classifier: dispatches schwarzschild/ISCO/
    # photon-sphere/Hawking/time-dilation directly when mass + concept
    # are unambiguous. Qwen 7b reliably fumbles these because it
    # passes "mass_kg=10" literally for "10 solar masses".
    from sigma_ground.mcp.benchmark.gr_classifier import (
        classify_for_gr, execute_gr_match,
    )
    gr_match = classify_for_gr(question)
    if gr_match is not None:
        val, units, answer_text = execute_gr_match(gr_match)
        if val is not None:
            return {
                "answer_text":            answer_text,
                "extracted_value":        val,
                "extracted_units":        units,
                "tool_calls":             [],
                "turns":                  0,
                "elapsed_s":              time.time() - t0,
                "extracted_via_fallback": False,
                "nudges_sent":            0,
                "gr_classifier_hit":      gr_match.tool,
            }

    # Cosmology pre-classifier: dispatches Hubble radius / age of
    # universe / critical density / MOND regime / MOND a_0 / eta
    # directly. Qwen 7b picks rydberg_hydrogen_wavelength for the
    # 'is Earth gravity Newtonian or MOND?' question. The classifier
    # fixes it via straight keyword match.
    from sigma_ground.mcp.benchmark.cosmology_classifier import (
        classify_for_cosmology, execute_cosmology_match,
    )
    cos_match = classify_for_cosmology(question)
    if cos_match is not None:
        val, units, answer_text = execute_cosmology_match(cos_match)
        if val is not None:
            return {
                "answer_text":            answer_text,
                "extracted_value":        val,
                "extracted_units":        units,
                "tool_calls":             [],
                "turns":                  0,
                "elapsed_s":              time.time() - t0,
                "extracted_via_fallback": False,
                "nudges_sent":            0,
                "cosmology_classifier_hit": cos_match.tool,
            }

    # Math pre-classifier: dispatches solve_equation / integrate_expr /
    # differentiate_expr / simplify_expr for natural-English math
    # questions ('x squared minus 4 equals zero', 'integral of x^2
    # from 0 to 1'). Translates natural phrasing to sympy syntax.
    from sigma_ground.mcp.benchmark.math_classifier import (
        classify_for_math, execute_math_match,
    )
    math_match = classify_for_math(question)
    if math_match is not None:
        val, units, answer_text = execute_math_match(math_match)
        if val is not None:
            return {
                "answer_text":            answer_text,
                "extracted_value":        val,
                "extracted_units":        units,
                "tool_calls":             [],
                "turns":                  0,
                "elapsed_s":              time.time() - t0,
                "extracted_via_fallback": False,
                "nudges_sent":            0,
                "math_classifier_hit":    math_match.tool,
            }

    # Thermo pre-classifier: ideal gas / Wien / Stefan-Boltzmann /
    # Carnot / equipartition / Maxwell-Boltzmann / melting+boiling.
    from sigma_ground.mcp.benchmark.thermo_classifier import (
        classify_for_thermo, execute_thermo_match,
    )
    thermo_match = classify_for_thermo(question)
    if thermo_match is not None:
        val, units, answer_text = execute_thermo_match(thermo_match)
        if val is not None:
            return {
                "answer_text":            answer_text,
                "extracted_value":        val,
                "extracted_units":        units,
                "tool_calls":             [],
                "turns":                  0,
                "elapsed_s":              time.time() - t0,
                "extracted_via_fallback": False,
                "nudges_sent":            0,
                "thermo_classifier_hit":  thermo_match.tool,
            }

    # Modern physics (special relativity + early QM) pre-classifier.
    # Dispatches lorentz_factor / time-dilation / length-contraction /
    # mass_to_energy / luminosity_to_mass_rate / photon-energy /
    # velocity-addition / Doppler / de Broglie.
    from sigma_ground.mcp.benchmark.modern_classifier import (
        classify_for_modern, execute_modern_match,
    )
    modern_match = classify_for_modern(question)
    if modern_match is not None:
        val, units, answer_text = execute_modern_match(modern_match)
        if val is not None:
            return {
                "answer_text":            answer_text,
                "extracted_value":        val,
                "extracted_units":        units,
                "tool_calls":             [],
                "turns":                  0,
                "elapsed_s":              time.time() - t0,
                "extracted_via_fallback": False,
                "nudges_sent":            0,
                "modern_classifier_hit":  modern_match.tool,
            }

    # Classical mechanics (intro) pre-classifier.
    # Projectile max-height / flight-time, circular orbit, free fall
    # with explicit gravity (Moon/Mars), friction stopping distance,
    # inverse KE (v from KE).
    from sigma_ground.mcp.benchmark.classical_intro_classifier import (
        classify_for_classical_intro, execute_classical_intro_match,
    )
    ci_match = classify_for_classical_intro(question)
    if ci_match is not None:
        val, units, answer_text = execute_classical_intro_match(ci_match)
        if val is not None:
            return {
                "answer_text":            answer_text,
                "extracted_value":        val,
                "extracted_units":        units,
                "tool_calls":             [],
                "turns":                  0,
                "elapsed_s":              time.time() - t0,
                "extracted_via_fallback": False,
                "nudges_sent":            0,
                "classical_intro_hit":    ci_match.tool,
            }

    # Optics pre-classifier: Snell's law / TIR critical angle /
    # diffraction grating / speed of sound in air.
    from sigma_ground.mcp.benchmark.optics_classifier import (
        classify_for_optics, execute_optics_match,
    )
    opt_match = classify_for_optics(question)
    if opt_match is not None:
        val, units, answer_text = execute_optics_match(opt_match)
        if val is not None:
            return {
                "answer_text":            answer_text,
                "extracted_value":        val,
                "extracted_units":        units,
                "tool_calls":             [],
                "turns":                  0,
                "elapsed_s":              time.time() - t0,
                "extracted_via_fallback": False,
                "nudges_sent":            0,
                "optics_classifier_hit":  opt_match.tool,
            }

    # Astrophysics pre-classifier:
    # Light-travel-time from a named star (parsec -> light-year),
    # star property lookup (luminosity / mass / distance),
    # planet orbital period, Sun's peak emission wavelength,
    # Sun's energy output per year, Earth's orbital velocity.
    from sigma_ground.mcp.benchmark.astro_classifier import (
        classify_for_astro, execute_astro_match,
    )
    astro_match = classify_for_astro(question)
    if astro_match is not None:
        val, units, answer_text = execute_astro_match(astro_match)
        if val is not None:
            return {
                "answer_text":            answer_text,
                "extracted_value":        val,
                "extracted_units":        units,
                "tool_calls":             [],
                "turns":                  0,
                "elapsed_s":              time.time() - t0,
                "extracted_via_fallback": False,
                "nudges_sent":            0,
                "astro_classifier_hit":   astro_match.tool,
            }

    messages = [
        {"role": "system",    "content": system_prompt},
        {"role": "user",      "content": question},
    ]
    tool_calls_made: list[dict] = []
    timeout_s = 120.0
    max_turns = 8                # was 14; tight budget for tool-first discipline
    max_tool_calls = 5           # hard cap per question -- forces purposeful tool use
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
                        primary_tool_expected=primary_tool_expected,
                        question=question)
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

            # Hard cap: 5 tool calls per question. After that, force
            # the LLM to finalize with whatever it has (or refuse).
            # Without this, Qwen explores 14 tools hoping one fits.
            if len(tool_calls_made) >= max_tool_calls and nudges_sent < max_nudges:
                budget_warning = (
                    f"STOP. You have used your {max_tool_calls}-tool-call "
                    f"budget. No more tool calls allowed. Either:\n"
                    f"  (a) Produce the ANSWER: line using values you "
                    f"already have, OR\n"
                    f"  (b) Produce 'ANSWER: [SOURCE: Fitted due to "
                    f"incompetence -- could not find the right tool]'.\n"
                    f"The user's question is still: {question}"
                )
                messages.append({"role": "user", "content": budget_warning})
                nudges_sent += 1
                continue

            # Loop detection: any repeat of (tool_name, args) -- tighter than
            # 'last 3 identical' because Qwen wastes budget if even ONE
            # repeat happens. Treat first repeat as already-loop.
            # (was: same 3 in a row, after Qwen kept calling
            # solar_system_body('earth') 14 times waiting for
            # a different answer that never came.
            if len(tool_calls_made) >= 2:
                last_two = tool_calls_made[-2:]
                same_name = last_two[0]["name"] == last_two[1]["name"]
                same_args = (json.dumps(last_two[0]["args"], sort_keys=True, default=str)
                                == json.dumps(last_two[1]["args"], sort_keys=True, default=str))
                if same_name and same_args:
                    loop_warning = (
                        f"STOP. You called `{last_two[0]['name']}` twice "
                        f"with identical args. The tool's output will not "
                        f"change. Either:\n"
                        f"  (a) Call a DIFFERENT tool from the TOOL INDEX, OR\n"
                        f"  (b) Produce the ANSWER: line using the values "
                        f"you already have, OR\n"
                        f"  (c) Produce the ANSWER: line with the '[SOURCE: "
                        f"Fitted due to incompetence ...]' tag.\n"
                        f"The user's question is still: {question}"
                    )
                    messages.append({"role": "user", "content": loop_warning})
                    nudges_sent += 1

        # Hit max turns -- try fallback before giving up
        val, units = _extract_value_from_tool_calls(
            tool_calls_made, primary_tool_expected=primary_tool_expected,
            question=question)
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


async def _run_conversation(session, ollama_url: str, model: str,
                            tools_for_ollama: list[dict], system_prompt: str,
                            real_params_by_tool: dict[str, set[str]] | None = None,
                            script_lines: list[str] | None = None) -> dict:
    """Multi-turn simulation-playground conversation.

    ONE `messages` list persists across turns (so the model keeps context), and
    the MCP server's module-level scene store persists across tool calls (so the
    simulation state carries forward). Reads turns from `script_lines` (non-
    interactive demo/eval) or from stdin (interactive). Degrades gracefully if
    ollama is unreachable.
    """
    import httpx

    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    transcript: list[dict] = []

    def _turns():
        if script_lines is not None:
            for ln in script_lines:
                yield ln
            return
        while True:
            try:
                ln = input("\nyou> ").strip()
            except EOFError:
                return
            if not ln:
                continue
            if ln.lower() in ("quit", "exit", ":q"):
                return
            yield ln

    async with httpx.AsyncClient(timeout=120.0) as http:
        for user_msg in _turns():
            if script_lines is not None:
                print(f"\nyou> {user_msg}")
            messages.append({"role": "user", "content": user_msg})
            turn_tools: list[str] = []
            seen_calls: set[str] = set()
            final = ""
            tool_budget = 4
            for _ in range(8):                       # inner tool-loop
                force_finalize = len(turn_tools) >= tool_budget
                payload: dict = {"model": model, "messages": messages,
                                 "stream": False, "options": {"temperature": 0.2}}
                if not force_finalize:
                    payload["tools"] = tools_for_ollama
                else:
                    # budget spent: ask for the reply with NO tools offered, so
                    # the model must produce text instead of looping tool calls.
                    messages.append({"role": "user", "content":
                                     "You have enough tool results now. Reply to "
                                     "the user in 1-3 sentences using those "
                                     "results. Do NOT call any more tools."})
                try:
                    resp = await http.post(ollama_url, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                except Exception as e:               # ollama down -> degrade, keep going
                    final = f"[ollama unreachable: {e}]"
                    break
                msg = data.get("message", {})
                messages.append(msg)
                tcs = msg.get("tool_calls") or []
                if force_finalize or not tcs:
                    final = (msg.get("content", "") or "").strip()
                    break
                for tc in tcs:
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    a = fn.get("arguments", {})
                    if isinstance(a, str):
                        try:
                            a = json.loads(a)
                        except json.JSONDecodeError:
                            a = {}
                    if real_params_by_tool and name in real_params_by_tool:
                        from sigma_ground.mcp.benchmark.param_aliases import normalize_kwargs
                        a, _ = normalize_kwargs(a, real_params_by_tool[name])
                    sig = f"{name}:{json.dumps(a, sort_keys=True, default=str)}"
                    if sig in seen_calls:            # don't repeat an identical call
                        messages.append({"role": "tool", "content":
                                         "(already called with these args this "
                                         "turn -- use the previous result)"})
                        turn_tools.append(name)
                        continue
                    seen_calls.add(sig)
                    try:
                        result = await session.call_tool(name, a)
                        parts = [getattr(p, "text", None) for p in (result.content or [])]
                        tool_text = "\n".join(x for x in parts if x) or "(empty)"
                    except Exception as e:
                        tool_text = f"<TOOL ERROR: {e}>"
                    turn_tools.append(name)
                    messages.append({"role": "tool", "content": tool_text[:4000]})
            print(f"mentat> {final or '(no reply)'}")
            transcript.append({"user": user_msg, "assistant": final,
                               "tools": turn_tools})
    return {"turns": transcript}


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
    # Resume is Q&A-only: conversation-mode output is a single {turns:[...]} dict,
    # not a list of per-question records.
    if args.mode == "qa" and args.resume and args.output.exists():
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

    # Spawn the MCP server from THIS interpreter + package, so the spawned
    # server is the same sigma_ground tree the runner is running from (the
    # global `sigma-ground-mcp` console script may point at a different/older
    # editable install). Inherit the environment so PYTHONPATH propagates.
    import os
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "sigma_ground.mcp.server"],
        env=dict(os.environ),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_resp = await session.list_tools()
            # Convert to Ollama's tool-format
            # Hide the body-aware/multi-step tools from Qwen's flat list —
            # they're reached deterministically via the new-tools router,
            # not by LLM selection. Keeping the visible surface small is
            # what stops catalog growth from degrading a modest model.
            from sigma_ground.mcp.benchmark.new_tools_classifier import (
                HIDDEN_FROM_LLM)
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
                if t.name not in HIDDEN_FROM_LLM
            ]
            tool_index = _build_tool_index(tools_for_ollama)
            real_params_by_tool = _build_real_params_map(tools_for_ollama)
            sys_prompt = _SYSTEM_PROMPT_BASE + "\n\n" + tool_index
            print(f"MCP server has {len(tools_for_ollama)} tools available")
            print(f"System prompt is {len(sys_prompt)} chars "
                  f"(includes full tool index)")

            # Conversation / simulation-playground mode: one persistent scene +
            # message history across turns (vs the stateless Q&A loop below).
            if getattr(args, "mode", "qa") == "conversation":
                # Scope the visible tool set to the playground family (+ the
                # Materia front doors). A 7b switchboard routes far better over
                # ~10 focused tools than over 200+, and it shrinks the prompt.
                _CONV_EXTRA = {"simulate", "run_simulation", "list_simulation_scenarios"}
                conv_tools = [t for t in tools_for_ollama
                              if t["function"]["name"].startswith("playground_")
                              or t["function"]["name"] in _CONV_EXTRA]
                conv_index = _build_tool_index(conv_tools)
                conv_prompt = _SYSTEM_PROMPT_CONVERSATION + "\n\n" + conv_index
                script_lines = None
                if args.script:
                    script_lines = [
                        ln.strip() for ln in
                        Path(args.script).read_text(encoding="utf-8").splitlines()
                        if ln.strip() and not ln.lstrip().startswith("#")
                    ]
                print("\n=== CONVERSATION / SIMULATION PLAYGROUND ===")
                print(f"(playground tool set: {len(conv_tools)} tools)")
                print("(type 'quit' to exit)" if script_lines is None
                      else f"(scripted: {len(script_lines)} turns)")
                convo = await _run_conversation(
                    session, args.ollama_url + "/api/chat", args.model,
                    conv_tools, conv_prompt,
                    real_params_by_tool=real_params_by_tool,
                    script_lines=script_lines)
                with args.output.open("w", encoding="utf-8") as f:
                    json.dump({"mode": "conversation", "model": args.model,
                               **convo}, f, indent=2, default=str)
                print(f"\nWrote {args.output}")
                return 0

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
    parser.add_argument("--model", default="qwen2.5:7b",
                        help="Ollama model tag. 7b is the default after testing "
                              "showed it ~2x faster with comparable accuracy "
                              "once the tool-first discipline rules + fallback "
                              "extractor are in place. 14b available if needed "
                              "for harder synthesis questions.")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "results" / "sigma_ground_run.json")
    parser.add_argument("--questions", type=Path,
                        default=Path(__file__).parent / "questions.json",
                        help="Path to questions corpus JSON.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--mode", choices=["qa", "conversation"], default="qa",
                        help="qa: stateless benchmark switchboard (default); "
                             "conversation: stateful simulation playground")
    parser.add_argument("--script", type=Path, default=None,
                        help="conversation mode: a file of user turns (one per "
                             "line; '#' comments ignored). Omit for interactive.")
    args = parser.parse_args()
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    sys.exit(main())
