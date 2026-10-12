"""One REAL found-object request in the real game, end to end, with the real bridge and the REAL download (no throttling, no made-up bytes).

    python stage/gmod/holodeck/found_live.py <tag> "<request as a player says it>" [--purge CARD_ID ...] [--timeout 90]

tag is found_<source>_<name> (e.g. found_polyhaven_kettle). The game client runs for about a minute (visible window), the bridge runs in this
process with timing wrappers around the pipeline's stages (resolve, fetch, measure, pack, the whole /mesh), and the Lua tool
(addon/lua/mentat/tools/found_live.lua) photographs a time-lapse of the beam and the arrival, then close views, and measures what stands.
Pictures: stage/gmod/holodeck/data/shots/<tag>_<view>.jpg (gitignored), a contact sheet <tag>_sheet.jpg and a JSON report <tag>_report.json.
--purge removes a card's cached download first, so the run really downloads it.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "experiments"))
sys.path.insert(0, str(HERE.parents[1]))
import run as exp_run  # noqa: E402
import run_tool  # noqa: E402
import skippy_bridge as B  # noqa: E402

HU_M = 0.01905
EVENTS: list[dict] = []
T0 = [0.0]


def timed(label, fn, describe=None):
    def wrapper(*a, **k):
        t = time.monotonic()
        err = None
        try:
            r = fn(*a, **k)
            return r
        except BaseException as e:
            err = f"{type(e).__name__}: {e}"
            raise
        finally:
            ev = {"what": label, "start_s": round(t - T0[0], 2), "dur_s": round(time.monotonic() - t, 2)}
            if err:
                ev["error"] = err[:160]
            elif describe:
                try:
                    ev.update(describe(a, k, r))
                except Exception:
                    pass
            EVENTS.append(ev)
    return wrapper


def instrument():
    acquire, C, _, R = B._skippyverse()
    from skippyverse import meshpack
    acquire.acquire = timed("acquire (download)", acquire.acquire, lambda a, k, r: {"bytes": r.get("bytes"), "from_cache": r.get("from_cache"), "files": len(r.get("files") or [])})
    acquire.measure_acquired = timed("measure", acquire.measure_acquired, lambda a, k, r: {"size_m": [round(r[x], 4) for x in ("w", "d", "h")], "basis": r["basis"]})
    meshpack.pack = timed("pack (meshpack, lzma)", meshpack.pack, lambda a, k, r: {"bytes": len(r[0]), **{x: r[1][x] for x in ("tris_in", "tris_out", "verts", "bbox_m")}})
    B.mesh_payload = timed("mesh_payload (whole /mesh)", B.mesh_payload, lambda a, k, r: {"payload_bytes": r["bytes"], "b64_chars": len(r["b64"]), "tris": r["tris"]})
    orig = B.resolve_found

    def resolve(*a, **k):
        t = time.monotonic()
        r = orig(*a, **k)
        EVENTS.append({"what": "resolve_found (/resolve)", "start_s": round(t - T0[0], 2), "dur_s": round(time.monotonic() - t, 2), "status": r["status"],
                       "place": r["place"], "fetched": r["fetched"], "say": r["say"], "card": r.get("card"), "mesh": bool(r.get("mesh"))})
        return r
    B.resolve_found = resolve


def main(argv=None) -> int:
    from PIL import Image
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("request")
    ap.add_argument("--purge", nargs="*", default=[])
    ap.add_argument("--timeout", type=int, default=90)
    ap.add_argument("--slow-send", action="store_true", help="send the chunks 500 bytes per 0.4 s (the ONLY non-production knob) so the sending stage lasts long enough to photograph")
    ap.add_argument("--out", default=str(HERE / "data" / "shots"))
    ap.add_argument("--map", default="gm_flatgrass")
    ap.add_argument("--width", type=int, default=960)
    a = ap.parse_args(argv)
    os.environ.setdefault("GMOD_EXTRA_ARGS", "+maxplayers 4")
    from skippyverse import acquire as AQ
    for cid in a.purge:
        d = AQ.DEFAULT_CACHE / cid
        if d.is_dir():
            shutil.rmtree(d)
            print("purged", d)
    B.TOKEN = secrets.token_hex(8)
    B.warm = lambda: None
    B.found_index()
    instrument()
    from skippy_e2e import tailscale_ip
    host = os.environ.get("SKIPPY_BRIDGE_HOST") or tailscale_ip()
    srv = B.serve(host, 8777)
    log = Path(os.environ.get("GMOD_DIR", "D:/Steam/steamapps/common/GarrysMod")) / "garrysmod" / "console.log"
    for old in exp_run.DATA.glob(f"shot_{a.tag}_*.jpg"):
        old.unlink()
    T0[0] = time.monotonic()
    targs = {"request": a.request, "bridge": f"http://{host}:8777", "token": B.TOKEN, "tag": a.tag, "timeout": a.timeout}
    if a.slow_send:
        targs.update(chunk=500, pump=0.4)
    try:
        res = run_tool.run_tool("found_live", targs, client=True, visible=True, map_name=a.map, timeout=a.timeout + 120)
    finally:
        srv.shutdown()
        srv.server_close()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    perf, lines = [], []
    if log.exists():
        for ln in log.read_text(encoding="utf-8", errors="replace").splitlines():
            if "[fl-perf]" in ln:
                perf.append(ln.split("[fl-perf]", 1)[1].strip())
            if re.search(r"holodeck_mesh|\[holodeck\].*(mesh|found)|decode", ln, re.I) and "fl-perf" not in ln:
                lines.append(ln.strip()[:200])
    report = {"request": a.request, "result": res, "bridge_events": EVENTS, "perf": perf, "log_mesh_lines": lines[-25:]}
    (out / f"{a.tag}_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    tiles = []
    for f in res.get("frames", []):
        src = exp_run.DATA / f"shot_{a.tag}_{f['name']}.jpg"
        if not src.exists():
            print("missing picture:", f["name"])
            continue
        im = Image.open(src).convert("RGB")
        im = im.resize((a.width, int(im.height * a.width / im.width)))
        dest = out / f"{a.tag}_{f['name']}.jpg"
        im.save(dest, quality=90)
        tiles.append((f, im))
        print(dest, f"stage={f['stage']} progress={f['progress']:.3f} t={f['t']:.1f}s visual={f.get('visual')}")
    if tiles:
        cols, w = 3, 640
        sm = [im.resize((w, int(im.height * w / im.width))) for _, im in tiles]
        h = sm[0].height
        rows = (len(sm) + cols - 1) // cols
        sheet = Image.new("RGB", (w * cols, h * rows), (20, 20, 20))
        for i, im in enumerate(sm):
            sheet.paste(im, ((i % cols) * w, (i // cols) * h))
        sheet.save(out / f"{a.tag}_sheet.jpg", quality=88)
        print(out / f"{a.tag}_sheet.jpg")
    print(json.dumps({k: v for k, v in report.items() if k != "result"}, indent=1)[:6000])
    r = dict(res)
    r.pop("frames", None)
    print(json.dumps(r, indent=1)[:5000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
