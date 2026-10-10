"""The @server.tool() wrappers in server.py, checked as the server serves them.

Every other test calls the underlying functions (procedures.black_hole_profile,
tools.gr.*) directly, so a broken WRAPPER is invisible to them. That is how
procedure_black_hole_profile shipped with an empty body: a block of frontier
tools had been pasted between its docstring and its `return`, so the wrapper
returned None and FastMCP rejected every call ("validation error for
DictModel"), while the orphaned return sat as dead code after
unruh_temperature's.
"""
import ast
import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

SERVER = Path(__file__).with_name("server.py")


def _tool_wrappers():
    tree = ast.parse(SERVER.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and any(
                isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute)
                and d.func.attr == "tool" for d in node.decorator_list):
            yield node


def test_every_tool_wrapper_returns_and_has_no_dead_code():
    wrappers = list(_tool_wrappers())
    assert len(wrappers) > 200
    empty, dead = [], []
    for fn in wrappers:
        if not any(isinstance(s, ast.Return) and s.value is not None
                   for s in ast.walk(fn)):
            empty.append(f"{fn.name} (line {fn.lineno})")
        for i, stmt in enumerate(fn.body[:-1]):
            if isinstance(stmt, ast.Return):
                dead.append(f"{fn.name} (line {fn.body[i + 1].lineno})")
    assert not empty, f"tool wrappers that return nothing: {empty}"
    assert not dead, f"unreachable code after a return in: {dead}"


def test_black_hole_profile_through_the_server():
    pytest.importorskip("mcp.server.fastmcp")
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    from sigma_ground.mcp import procedures

    async def call():
        params = StdioServerParameters(command=sys.executable,
                                       args=["-m", "sigma_ground.mcp.server"],
                                       env=dict(os.environ))
        async with stdio_client(params) as (r, w):
            async with ClientSession(r, w) as session:
                await session.initialize()
                return await session.call_tool("procedure_black_hole_profile",
                                               {"mass_kg": 1.989e30})

    res = asyncio.run(call())
    text = res.content[0].text if res.content else ""
    assert not res.isError, text
    served = json.loads(text)
    direct = json.loads(json.dumps(procedures.black_hole_profile(1.989e30).to_dict(),
                                   default=str))
    assert served == direct
    steps = {s["label"]: s for s in served["steps"]}
    assert any("Schwarzschild" in label for label in steps), list(steps)
