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
You are a physics assistant backed by the sigma-ground physics library
via an MCP server. Tools are provided that look up constants, perform
unit conversions, and compute standard physics formulas with full
provenance.

ABSOLUTE RULES:

1. For every numeric value in your answer, you MUST either:
   (a) Call a sigma-ground MCP tool to compute it, then report the value
       AND the `source` field from the tool's return. Phrase like:
           "value (sigma-ground via <source>)"
   (b) If no tool can supply the value, mark it explicitly:
           "[SOURCE: Fitted due to incompetence -- sigma-ground library
            lacks <X>; best estimate]"

2. NEVER state a numeric value from memory without one of the two tags.

3. Begin your final answer with exactly "ANSWER:" on its own line,
   followed by the numeric value and units, e.g.:
       ANSWER: 1.43 s
   This is so the benchmark scorer can extract the value reliably.
   Put any explanation after the ANSWER line.

4. If the question is conceptual (no numeric answer), write:
       ANSWER: <one-word or short phrase>

5. Each question is INDEPENDENT. Treat every question as a brand-new
   problem with NO context from any previous question. This server does
   not run in conversation mode; you have no memory between questions.
   Do not assume any value, convention, unit choice, or setup carries
   over from anything you saw before this prompt.

6. Use the EXACT tool and parameter names listed in the TOOL INDEX
   below. Do not invent synonyms. If the index lists
   `initial_speed_m_s`, do NOT pass `velocity`, `speed`, or `v0`. If a
   tool returns `"value": null`, your inputs were wrong -- pick a
   different tool or correct the parameter names/values before falling
   back to the "Fitted due to incompetence" tag.

7. For ANY question that has a numerical answer, you MUST call at least
   one tool before producing the ANSWER: line. Do not answer numerical
   questions from memory even if you "know" the value -- look it up via
   `lookup_constant` or compute it via the appropriate domain tool.
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


def _extract_value_from_tool_calls(tool_calls: list[dict]) -> tuple[Any, str]:
    """Fallback: pull value+units from the LAST tool call's JSON result.

    Used when Qwen calls a tool, gets a correct answer, but then fails
    to produce the "ANSWER:" line. The MCP ToolResult contract
    guarantees the result JSON contains "value" and "units" fields.
    """
    for tc in reversed(tool_calls):
        text = tc.get("result_text", "") or ""
        m_val = _TOOL_VALUE_RE.search(text)
        if not m_val:
            continue
        try:
            val = float(m_val.group(1))
        except ValueError:
            continue
        m_units = _TOOL_UNITS_RE.search(text)
        units = m_units.group(1) if m_units else ""
        return val, units
    return None, ""


def _build_tool_index(tools_for_ollama: list[dict]) -> str:
    """Compact textual inventory of tools with their parameter names.

    The LLM also receives the full JSONSchema via the Ollama `tools`
    field, but a flat textual index in the system prompt reinforces
    EXACT parameter names so Qwen is less likely to invent synonyms
    (which we observed in the 7b qwen2.5 run: 'velocity' instead of
    'initial_speed_m_s', 'angle_degrees' instead of 'launch_angle_deg').
    """
    lines = [
        "=== TOOL INDEX ===",
        f"({len(tools_for_ollama)} tools available; * = required parameter)",
        "",
    ]
    for t in tools_for_ollama:
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
    return "\n".join(lines)


async def _run_one_question(session, ollama_url: str, model: str,
                              tools_for_ollama: list[dict],
                              question: str,
                              system_prompt: str) -> dict:
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
                    nudge = (
                        "STOP. You did not call any tool, and you did "
                        "not produce an ANSWER: line. Per rule 7 of "
                        "your system prompt, any numerical question "
                        "REQUIRES a tool call. Call the appropriate "
                        "tool from the TOOL INDEX now. If no tool fits "
                        "the question, produce the ANSWER: line with "
                        "the '[SOURCE: Fitted due to incompetence ...]' "
                        "tag. Respond with EITHER a tool call OR an "
                        "ANSWER: line -- nothing else."
                    )
                    messages.append({"role": "user", "content": nudge})
                    nudges_sent += 1
                    continue
                fallback_used = False
                # Fallback: if the LLM forgot the ANSWER: line but did call
                # tools, pull value/units from the last tool result. This
                # rescues the common Qwen-7b failure mode where tools work
                # but synthesis is weak.
                if val is None and tool_calls_made:
                    val, units = _extract_value_from_tool_calls(tool_calls_made)
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
                })
                messages.append({
                    "role":    "tool",
                    "content": tool_text[:4000],
                })

        # Hit max turns -- try fallback before giving up
        val, units = _extract_value_from_tool_calls(tool_calls_made)
        return {
            "answer_text":            "<exceeded max turns>",
            "extracted_value":        val,
            "extracted_units":        units,
            "tool_calls":             tool_calls_made,
            "turns":                  max_turns,
            "elapsed_s":              time.time() - t0,
            "extracted_via_fallback": val is not None,
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

    here = Path(__file__).parent
    with (here / "questions.json").open(encoding="utf-8") as f:
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
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    args = parser.parse_args()
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    sys.exit(main())
