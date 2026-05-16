# Running the sigma-ground MCP server with local Qwen 2.5 72B

End-to-end recipe for talking to the physics MCP server using a locally-hosted Qwen 2.5 72B model through Ollama. The MCP server itself is LLM-agnostic — once it speaks the MCP protocol, any client (Claude Desktop, Cline, custom Python harness, etc.) can drive it.

## Hardware reality check

Qwen 2.5 72B at Q4_K_M quantization is ~47 GB on disk and needs roughly that much VRAM for fast inference. Realistic setups:

| Setup | Expected speed | Notes |
|---|---|---|
| 2× RTX 4090 (24 GB each) | 15–25 tok/s | Sweet spot for this size |
| 1× A100 80 GB | 30–40 tok/s | Smooth |
| 1× H100 80 GB | 50+ tok/s | Comfortable |
| 1× RTX 4090 + CPU offload | 3–8 tok/s | Workable but interactive feel is slow |
| Pure CPU + 64 GB RAM | 0.5–2 tok/s | Patience test |

If 72B is too heavy, **Qwen 2.5 32B** with Q4 quantization runs comfortably on a single 24 GB GPU at 20+ tok/s and still has solid function-calling discipline. Drop down to that if 72B is sluggish; the workflow is identical.

## Install

```bash
# 1. Install Ollama (https://ollama.com)
# 2. Pull the model
ollama pull qwen2.5:72b           # or qwen2.5:32b for smaller setups

# 3. Install sigma-ground with MCP extras
pip install -e ".[mcp]"           # from the sigma-ground checkout
# OR
pip install "sigma-ground[mcp]"    # once published
```

## Start the MCP server

The server speaks MCP over stdio (the standard transport). Run it manually to confirm it works:

```bash
sigma-ground-mcp                  # via the installed entry point
# OR
python -m sigma_ground.mcp.server
```

You should see no output — the server is waiting on stdin for MCP protocol messages. Ctrl+C to stop.

## Connect Qwen via a Python harness

The simplest test loop uses Ollama's chat API with tool calling, dispatching tool calls to the MCP server. Below is a minimal harness; expand into your own script.

```python
"""minimal_qwen_mcp_loop.py — talk to Qwen with sigma-ground MCP tools."""
import asyncio
import json
import requests

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:72b"     # or qwen2.5:32b

SYSTEM_PROMPT = """\
You are a physics assistant backed by the sigma-ground MCP server, a curated
physics library with rigorous provenance plus wrapped externals (scipy, pint,
sympy). Every numerical answer must report the `source` and (when present)
`provenance_tag` fields from each ToolResult.

Default to standard physics. The library has an SSBM (Scale-Shifted Baryonic
Matter) theoretical layer; invoke it ONLY when the user explicitly asks about
black hole interior structure, sigma-field dynamics, or cosmic-origin-as-BH.
Do not volunteer SSBM framing for ordinary physics queries.

Use SI internally; call convert_units() if the user asks for other units.
"""


async def run_loop():
    # Spawn the MCP server as a subprocess and connect to it.
    params = StdioServerParameters(command="sigma-ground-mcp")
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            # Pull the tool list from the server.
            tools_response = await session.list_tools()
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description or "",
                        "parameters": t.inputSchema or {},
                    },
                }
                for t in tools_response.tools
            ]

            messages = [{"role": "system", "content": SYSTEM_PROMPT}]

            print("Ready. Type a physics question, or 'quit'.")
            while True:
                user_input = input("\n>>> ")
                if user_input.strip().lower() in ("quit", "exit"):
                    break
                messages.append({"role": "user", "content": user_input})

                # Multi-turn tool loop: keep going until Qwen stops calling tools.
                while True:
                    response = requests.post(OLLAMA_URL, json={
                        "model": MODEL,
                        "messages": messages,
                        "tools": tools,
                        "stream": False,
                    }).json()
                    msg = response["message"]
                    messages.append(msg)

                    if not msg.get("tool_calls"):
                        print(msg["content"])
                        break

                    for tc in msg["tool_calls"]:
                        fn = tc["function"]
                        name = fn["name"]
                        args = fn["arguments"]
                        if isinstance(args, str):
                            args = json.loads(args)
                        result = await session.call_tool(name, args)
                        # FastMCP returns content as a list of TextContent.
                        result_text = result.content[0].text if result.content else ""
                        messages.append({
                            "role": "tool",
                            "content": result_text,
                            "tool_call_id": tc.get("id", name),
                        })


if __name__ == "__main__":
    asyncio.run(run_loop())
```

Save as `scripts/minimal_qwen_mcp_loop.py` and run:

```bash
ollama serve &                              # if not already running
python scripts/minimal_qwen_mcp_loop.py
```

## Try these questions

These exercise different tools and should produce well-cited answers:

1. **`What's the Schwarzschild radius of a 10 solar mass black hole?`** — should call `convert_units("10 solar_mass to kg")` → `schwarzschild_radius(mass_kg=1.98892e31)` → cite "sigma_ground.field.gr_basics (standard Schwarzschild)" → ~29.5 km.

2. **`How long does it take light to travel 1 parsec?`** — should call `parse_quantity("1 parsec")` → `convert_units(value, "m", "m")` → `lookup_constant("speed_of_light")` → divide → ~3.26 years.

3. **`Solve x^2 - 5x + 6 = 0`** — should call `solve_equation("x^2 - 5x + 6", "x")` → returns ["2", "3"].

4. **`What's the photon-sphere radius of Sgr A* (4.1 million solar masses)?`** — multi-step: convert, then `photon_sphere_radius`.

5. **`What's the value of eta and where does it come from?`** — `lookup_constant("ETA")` → returns 0.412164 with `provenance_tag=EMPIRICAL-INPUT` and `source=sigma_ground.field.constants`. Qwen should report "η = 0.4122, anchored at DESI 2024 Union3 HDE c² fit."

6. **`At what radius around a 1 solar mass BH does a clock tick at half the rate of one at infinity?`** — multi-step: solve `sqrt(1 - r_s/r) = 0.5` for r → r = r_s/(1 - 0.25) = (4/3) r_s.

7. **`Could our universe be inside a black hole?`** — this is a *standard-physics + theoretical* question. Qwen should mention various perspectives (CCC, holographic, etc.) and optionally offer SSBM as one available framework — but not lead with it.

## Tuning notes

- **Function-calling discipline.** Qwen 2.5 72B is generally clean, but if you see hallucinated tool names, prepend the system prompt with: "Call `get_manifest()` first to confirm available tools."
- **Latency.** If 72B is too slow, switch to `qwen2.5:32b`. The function-calling quality is nearly identical.
- **Privacy.** This is fully local — no calls to Anthropic, OpenAI, Google, etc. Your physics questions never leave your machine.
- **MCP clients.** You can swap the harness for any MCP-aware client (Claude Desktop, Continue, Cline, Aider). They all consume the same server.

## What's NOT in the current MCP build (follow-up work)

This first cut exposes ~15 tools. The library has more capabilities that need wrappers:

- **Electrodynamics tools** (Coulomb force, atomic levels, muon g-2)
- **Cosmology tools** (Hubble radius, HDE band check, MOND classifier)
- **Nuclear tools** (binding energy, decay timescales, neutrinoless beta)
- **N-body tools** (rolling shootout, body lookup, JPL prediction)
- **Astropy wrappers** (ephemeris, coordinate transforms)
- **Periodic table** (element/isotope data via `periodictable`)
- **Quantum / particle physics** (neutrino oscillation, axion bounds)

These follow the same pattern as the existing wrappers. Each is a couple of hours of focused work; the architecture is solid.
