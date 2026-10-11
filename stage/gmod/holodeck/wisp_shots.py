"""Photograph the failure wisp in the real game (a visible window, about one to two minutes per scenario).

    python stage/gmod/holodeck/wisp_shots.py lapse     # "add a gargoyle": the wisp appears, is approached, read, dissolves
    python stage/gmod/holodeck/wisp_shots.py colours   # amber (not found), cyan (a typo read as the closest), red (a model I may not share)
    python stage/gmod/holodeck/wisp_shots.py unread    # nobody reads it: the slow fade (holodeck_wisp_life shortened for the run)
    python stage/gmod/holodeck/wisp_shots.py fetch     # a download that breaks and one that goes quiet until the timeout: the wisp after the collapse
    python stage/gmod/holodeck/wisp_shots.py perf      # milliseconds per frame with three wisps in view / none in view / none

The bridge is the real one (real inventory, real resolver sentences); only the DOWNLOAD of the fetch scenario is replaced (made-up bytes that
break or stop, as in transporter_shots.py). Pictures: stage/gmod/holodeck/data/shots/wisp_<scenario>_*.jpg (gitignored) and a contact sheet.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "experiments"))
sys.path.insert(0, str(HERE.parents[1]))
import run as exp_run  # noqa: E402
import run_tool  # noqa: E402
import skippy_bridge  # noqa: E402
import transporter_shots as ts  # noqa: E402  (SlowFetch: the made-up download)


def main(argv=None) -> int:
    from PIL import Image
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", choices=["lapse", "colours", "unread", "fetch", "perf"])
    ap.add_argument("--out", default=str(HERE / "data" / "shots"))
    ap.add_argument("--map", default="gm_flatgrass")
    ap.add_argument("--width", type=int, default=960)
    a = ap.parse_args(argv)
    os.environ.setdefault("GMOD_EXTRA_ARGS", "+maxplayers 4")
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="wisp-assets-"))
    skippy_bridge.TOKEN = secrets.token_hex(8)
    skippy_bridge.warm = lambda: None
    skippy_bridge.MESH_CACHE_DIR = tmp
    plan = {}
    targs: dict = {"scenario": a.scenario}
    if a.scenario == "fetch":
        plan = {"fire_hydrant": {"mode": "error", "after": 120_000}, "cannon_01": {"mode": "stall", "after": 90_000}}
        targs.update(timeout=14, stall=5)
    if a.scenario == "unread":
        targs.update(life=7)
    skippy_bridge.FOUND_FETCHER = ts.SlowFetch(plan, 12_000, 0.4)
    skippy_bridge.found_index()
    host = os.environ.get("SKIPPY_BRIDGE_HOST") or ts.tailscale_ip()
    srv = skippy_bridge.serve(host, 8777)
    targs.update(bridge=f"http://{host}:8777", token=skippy_bridge.TOKEN)
    for old in exp_run.DATA.glob("shot_wsp_*.jpg"):
        old.unlink()
    (exp_run.DATA / "wisp_perf.txt").unlink(missing_ok=True)
    try:
        res = run_tool.run_tool("wisp_shots", targs, client=True, visible=True, map_name=a.map, timeout=300)
    finally:
        ts.STOP.set()
        srv.shutdown()
        srv.server_close()
    print(json.dumps(res, indent=1))
    perf = exp_run.DATA / "wisp_perf.txt"
    if perf.exists():
        print(perf.read_text(encoding="utf-8"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tiles = []
    for f in res.get("frames", []):
        src = exp_run.DATA / f"shot_wsp_{f['name']}.jpg"
        if not src.exists():
            print("missing picture:", f["name"])
            continue
        im = Image.open(src).convert("RGB")
        im = im.resize((a.width, int(im.height * a.width / im.width)))
        dest = out / f"wisp_{a.scenario}_{f['name']}.jpg"
        im.save(dest, quality=90)
        print(dest, f"state={f.get('state')} read={f.get('read', 0):.2f} dist={f.get('dist', -1):.0f}")
        tiles.append(im)
    if tiles:
        cols = 3
        w = 640
        sm = [t.resize((w, int(t.height * w / t.width))) for t in tiles]
        rows = (len(sm) + cols - 1) // cols
        sheet = Image.new("RGB", (w * cols, sm[0].height * rows), (20, 20, 20))
        for i, im in enumerate(sm):
            sheet.paste(im, ((i % cols) * w, (i // cols) * sm[0].height))
        sheet.save(out / f"wisp_{a.scenario}_sheet.jpg", quality=88)
        print(out / f"wisp_{a.scenario}_sheet.jpg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
