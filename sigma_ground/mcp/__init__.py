"""sigma-ground MCP server + LLM-front-end physics assistant.

Optional subpackage that exposes the sigma-ground physics library
plus wrapped externals (scipy.constants, pint, sympy, astropy,
periodictable) as Model Context Protocol tools, designed for an LLM
front-end (Qwen, Claude, Gemini, etc.) to translate natural-language
physics questions into structured tool calls and synthesize cited
prose answers.

The MCP layer is positioned as **standard physics with rigorous
provenance**. The SSBM theoretical layer remains accessible but is
NOT foregrounded for routine queries -- see
misc/respa_attempt_2026-05-16.md and the MCP positioning memory.

Install:
    pip install sigma-ground[mcp]

Run server (stdio transport, the standard MCP mode):
    sigma-ground-mcp

Connect from an LLM client that speaks MCP. For local hosting with
Ollama-served Qwen 2.5 72B, see scripts/run_mcp_with_qwen.md (TODO).
"""

from sigma_ground.mcp.provenance import ToolResult

__all__ = ["ToolResult"]
