"""A fresh Mentat server answers a pint-backed tool within seconds.

Regression: the first `import numpy` made from the worker thread FastMCP runs a plain `def`
tool in could block for minutes on Windows, so `convert_units` (pint -> numpy) never came
back from a new server. server.main() now imports numpy in the main thread first.
"""
import asyncio
import os

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("SKIP_SERVER_TESTS") == "1", reason="needs the MCP server")


def test_a_new_server_converts_units_without_stalling():
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    from sigma_ground.mcp.benchmark import router_dataset as D

    async def run():
        async with stdio_client(D.server_params(0)) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await asyncio.wait_for(
                    session.call_tool("convert_units", {"value": 1.0, "from_units": "eV", "to_units": "J"}), 60)
                return result.content[0].text

    text = asyncio.run(run())
    assert "1.602176634e-19" in text
