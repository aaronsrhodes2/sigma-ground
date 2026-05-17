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
"""


_VALUE_RE = re.compile(
    r"ANSWER:\s*([\-+]?[0-9]+(?:\.[0-9]+)?(?:[eE][\-+]?[0-9]+)?)"
    r"(?:\s*([^\n]*))?",
    re.IGNORECASE,
)


def _extract_value(answer: str) -> tuple[Any, str]:
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
    max_turns = 10

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

            # If no tool calls, we're done
            tcs = msg.get("tool_calls") or []
            if not tcs:
                final = msg.get("content", "")
                val, units = _extract_value(final)
                return {
                    "answer_text":      final,
                    "extracted_value":  val,
                    "extracted_units":  units,
                    "tool_calls":       tool_calls_made,
                    "turns":            turn + 1,
                    "elapsed_s":        time.time() - t0,
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

        # Hit max turns
        return {
            "answer_text":     "<exceeded max turns>",
            "extracted_value": None,
            "extracted_units": "",
            "tool_calls":      tool_calls_made,
            "turns":           max_turns,
            "elapsed_s":       time.time() - t0,
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
            sys_prompt = _SYSTEM_PROMPT_BASE + (
                f"\n\nAvailable tools: {len(tools_for_ollama)}. "
                f"Call get_manifest() first if you need a tool inventory."
            )
            print(f"MCP server has {len(tools_for_ollama)} tools available")

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
    parser.add_argument("--model", default="qwen2.5:7b")
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
