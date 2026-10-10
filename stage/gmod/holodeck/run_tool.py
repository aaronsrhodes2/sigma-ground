"""Run one headless Lua job (stage/gmod/addon/lua/mentat/tools/NAME.lua) and return its JSON result.

    python stage/gmod/holodeck/run_tool.py inventory --args '{"roots":["models/props_c17/"]}' [--out file.json] [--client]

Uses the dedicated server by default (no window, no mouse). The job arms itself through data/mentat/tool.txt (see
mentat/tool.lua). Exits non-zero if the job did not finish or reported an error.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "experiments"))
import run as exp_run  # noqa: E402  (ctl(), DS_DATA, DATA)


def run_tool(name: str, args: dict | None = None, client: bool = False, map_name: str = "gm_flatgrass",
             timeout: int = 150, visible: bool = False) -> dict:
    server = not client
    data = exp_run.DS_DATA if server else exp_run.DATA
    if exp_run.ctl("status", server=server).returncode == 0:
        raise RuntimeError(f"the {'dedicated server' if server else 'game'} is already running; stop it first")
    if server:
        out = exp_run.ctl("setup", server=True)
        if out.returncode != 0:
            raise RuntimeError(out.stdout)
    data.mkdir(parents=True, exist_ok=True)
    result = data / f"tool_{name}.json"
    result.unlink(missing_ok=True)
    arm = json.dumps({"script": f"mentat/tools/{name}.lua", "id": time.strftime("%H%M%S"), "args": args or {}})
    res = None
    for attempt in (1, 2):
        # The first launch after the addon's Lua changed remounts the addon and can swallow the trigger (it is deleted
        # on read), so a launch that produced nothing is retried once with the trigger re-armed.
        (data / "tool.txt").write_text(arm, encoding="utf-8")
        started = exp_run.ctl("start", map_name, *(() if (server or visible) else ("--quiet",)), server=server)
        if started.returncode != 0:
            raise RuntimeError(started.stdout or started.stderr)
        t0 = time.time()
        try:
            while not result.exists() and time.time() - t0 < timeout:
                time.sleep(1)
            time.sleep(1.0)
            if result.exists():
                res = json.loads(result.read_text(encoding="utf-8"))
        finally:
            exp_run.ctl("stop", server=server)
        if res is not None:
            break
    if res is None:
        raise RuntimeError(f"tool {name} wrote no result in {timeout} s (two launches)")
    if res.get("error"):
        raise RuntimeError(f"tool {name} failed: {res['error']}")
    return res["result"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--args", default="{}")
    ap.add_argument("--out")
    ap.add_argument("--client", action="store_true")
    ap.add_argument("--visible", action="store_true", help="with --client: show the game window (screenshots need it)")
    ap.add_argument("--map", default="gm_flatgrass")
    ap.add_argument("--timeout", type=int, default=150)
    a = ap.parse_args(argv)
    res = run_tool(a.name, json.loads(a.args), a.client or a.visible, a.map, a.timeout, a.visible)
    text = json.dumps(res, indent=1)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
        print(f"wrote {a.out}")
    else:
        print(text[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
