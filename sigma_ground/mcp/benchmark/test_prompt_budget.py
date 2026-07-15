"""Guard against the documented 85%->53% regression: system-prompt/tool-index
bloat truncating Qwen's context. run_sigma_ground.py already fixed the known
cause (explicit num_ctx=32768) and caps the AKA line at 6 keywords/tool --
this test makes the CONSEQUENCE of breaking either of those cheap to catch
as natural-language-parity coverage keeps growing toward all 226 tools.

Word-count heuristic (~1.3 tokens/word for this kind of technical text) --
no tokenizer dependency needed; the margin below num_ctx is wide enough
that heuristic imprecision doesn't matter.
"""
from sigma_ground.mcp.benchmark.run_sigma_ground import (
    _SYSTEM_PROMPT_BASE, _build_tool_index)
from sigma_ground.mcp import manifest as M

_NUM_CTX = 32768
_SAFETY_CEILING_TOKENS = 20_000   # leaves headroom for tool JSONSchemas + turns


def _approx_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)


def test_full_tool_index_stays_under_prompt_budget():
    tools = [{"function": {"name": t["name"],
                           "description": t.get("summary", ""),
                           "parameters": {"properties": {}, "required": []}}}
             for t in M._PRIMARY_TOOLS]
    tool_index = _build_tool_index(tools)
    total = _approx_tokens(_SYSTEM_PROMPT_BASE) + _approx_tokens(tool_index)
    assert total < _SAFETY_CEILING_TOKENS, (
        f"system prompt + tool index ~{total} tokens, over the "
        f"{_SAFETY_CEILING_TOKENS} safety ceiling (num_ctx={_NUM_CTX}) -- "
        f"this is the exact failure mode that caused the 85%->53% regression"
    )


def test_aka_line_cap_is_six_per_tool():
    """The per-tool cap, not overall coverage, is what bounds the prompt --
    confirm no tool's AKA line exceeds it regardless of how much keyword
    data exists for it."""
    tools = [{"function": {"name": t["name"],
                           "description": t.get("summary", ""),
                           "parameters": {"properties": {}, "required": []}}}
             for t in M._PRIMARY_TOOLS]
    tool_index = _build_tool_index(tools)
    for line in tool_index.splitlines():
        line = line.strip()
        if line.startswith("AKA:"):
            n = line[len("AKA:"):].count("|") + 1
            assert n <= 6, f"AKA line exceeds the 6-keyword cap: {line[:120]!r}"
