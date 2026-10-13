"""Skippy alive, in the real game: drive scripted situations through the behaviour director and photograph the body language.
A visible game window for about three minutes (hands-off: the launcher waits for a calm GPU, takes no mouse). Needs the real Skippy bridge.

    python stage/gmod/holodeck/director_shots.py [--limit 3] [--out DIR] [--prefix dir]

Starts a LOCAL bridge on the Tailscale address (a listen server cannot reach loopback; see skippy_e2e.py) with its OWN private state in a
temporary folder (so Aaron's real line history, day budget and clip cache are untouched), the template model (no model call, no key) and the
bridge's own voice (SAPI unless SKIPPY_TTS says otherwise). Runs tools/director_shots.lua, saves the pictures as small JPEGs
DIR/<prefix>_<scene>_<n>.jpg and prints what the director decided, the budget, the clip path, the animation measurements and the frame cost.
See stage/gmod/AVATAR.md "Skippy alive".
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "experiments"))
import run as exp_run  # noqa: E402
import run_tool  # noqa: E402


def main(argv=None) -> int:
    from PIL import Image
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=8, help="new spoken lines per day the game may ask for (the budget demo)")
    ap.add_argument("--out", default=str(HERE / "data" / "shots"))
    ap.add_argument("--prefix", default="dir")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--map", default="gm_flatgrass")
    a = ap.parse_args(argv)
    os.environ.setdefault("GMOD_EXTRA_ARGS", "+maxplayers 4")
    state = Path(tempfile.mkdtemp(prefix="skippy_dir_live_"))
    os.environ["SKIPPY_QUIP_DIR"] = str(state)                 # BEFORE the quip module is first used
    os.environ["SKIPPY_QUIP_PROVIDER"] = "template"
    import skippy_bridge  # noqa: E402
    import skippy_quips  # noqa: E402
    import skippy_e2e  # noqa: E402  (tailscale_ip)
    skippy_bridge.TOKEN = secrets.token_hex(8)
    skippy_quips.set_engine(None)
    skippy_quips.ensure_gate(skippy_bridge)                        # the scripted scenes ask faster than a real afternoon: a wider (still bounded) allowance for this run
    skippy_bridge.GATE.per_minute["quip"], skippy_bridge.GATE.per_caller["quip"] = 60, 60
    skippy_bridge.GATE.rate["quip"] = skippy_bridge.Window(60, 60, skippy_bridge.GATE.clock)
    host = os.environ.get("SKIPPY_BRIDGE_HOST") or skippy_e2e.tailscale_ip()
    srv = skippy_bridge.serve(host, 8777)
    for old in exp_run.DATA.glob("shot_dir_*.jpg"):
        old.unlink()
    for old in exp_run.DATA.glob("dirsample_*.json"):
        old.unlink()
    for old in exp_run.DATA.glob("dirperf_*.json"):
        old.unlink()
    # the run writes the host's notebook (data/mentat/journal/<steamid64>.jsonl); a real one must not be polluted by a script: back it up, restore it after
    jdir = exp_run.DATA / "journal"
    jbak = state / "journal_backup"
    if jdir.exists():
        shutil.copytree(jdir, jbak)
    try:
        import subprocess
        subprocess.run([exp_run._bash(), str(HERE.parent / "install_skippy_body.sh").replace("\\", "/")], stdin=subprocess.DEVNULL)
        res = run_tool.run_tool("director_shots", {"bridge": f"http://{host}:8777", "token": skippy_bridge.TOKEN, "limit": a.limit},
                                client=True, visible=True, map_name=a.map, timeout=420)
    finally:
        srv.shutdown()
        srv.server_close()
        shutil.rmtree(jdir, ignore_errors=True)
        if jbak.exists():
            shutil.copytree(jbak, jdir)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    kept = []
    for name in res.get("shots", []):
        src = exp_run.DATA / f"shot_{name}.jpg"
        if not src.exists():
            print(f"missing picture: {name}")
            continue
        im = Image.open(src).convert("RGB")
        im = im.resize((a.width, int(im.height * a.width / im.width)))
        dest = out / f"{name.replace('dir_', a.prefix + '_', 1)}.jpg"
        im.save(dest, quality=85)
        kept.append(dest.name)
    res["pictures"] = kept
    res["bridge_state_dir"] = str(state)
    (out / f"{a.prefix}_live_result.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1)[:9000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
