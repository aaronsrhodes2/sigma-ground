"""Follow one spoken request through the REAL game: file from the microphone bridge -> client -> server -> built -> persona
reply -> bridge voice -> client -> played by the game's sound system. A visible game window for about a minute.

    python stage/gmod/holodeck/skippy_e2e.py ["add two couches"]
"""
from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_tool  # noqa: E402
import skippy_bridge  # noqa: E402


def tailscale_ip() -> str:
    for exe in ("tailscale", "C:/Program Files/Tailscale/tailscale.exe"):
        try:
            return subprocess.run([exe, "ip", "-4"], capture_output=True, text=True, timeout=10).stdout.split()[0]
        except (OSError, IndexError, subprocess.SubprocessError):
            continue
    raise SystemExit("no Tailscale address found; set SKIPPY_BRIDGE_HOST")


def main(argv=None) -> int:
    request = (argv or sys.argv[1:] or ["add two couches"])[0]
    skippy_bridge.TOKEN = secrets.token_hex(8)
    # GMod's -allowlocalhttp only works on dedicated servers, so a game that hosts its own world reaches the bridge by its
    # Tailscale address (which the engine does not treat as local). The bridge listens on that address only, behind its token.
    host = os.environ.get("SKIPPY_BRIDGE_HOST") or tailscale_ip()
    srv = skippy_bridge.serve(host, 8777)
    try:
        res = run_tool.run_tool("skippy_e2e", {"bridge": f"http://{host}:8777", "token": skippy_bridge.TOKEN, "request": request},
                                client=True, visible=True, timeout=240)
    finally:
        srv.shutdown()
        srv.server_close()
    print(json.dumps(res, indent=1))
    ok = res.get("consumed") and res.get("built", 0) > 0 and res.get("played")
    print("PASS: spoken request built things and Skippy's voice played in the game" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
