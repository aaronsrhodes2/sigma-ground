"""Photograph what the holodeck makes, in the real game (a visible window for about a minute).

    python stage/gmod/holodeck/shots.py "a living room" [--tag lr] [--out DIR] [--views 4]
Stitches the views into one contact image DIR/<tag>.jpg and prints the holodeck's replies."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "experiments"))
import run as exp_run  # noqa: E402
import run_tool  # noqa: E402


def main(argv=None) -> int:
    from PIL import Image                      # only stitching the pictures needs it
    ap = argparse.ArgumentParser()
    ap.add_argument("requests", nargs="+")
    ap.add_argument("--tag", default="hd")
    ap.add_argument("--out", default=str(HERE / "data" / "shots"))
    ap.add_argument("--map", default="gm_flatgrass")
    ap.add_argument("--settle", type=float, default=6.0)
    ap.add_argument("--yaws", default="0,180")
    ap.add_argument("--top", action="store_true", help="add a top-down view")
    ap.add_argument("--pos", default="", help="x,y[,z] to stand at before asking (e.g. inside a room)")
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--start-bridge", action="store_true", help="run the Skippy bridge for this run (found objects and their real shapes need it)")
    ap.add_argument("--two-phase", action="store_true", help="photograph the placeholder boxes first, then fetch the real shapes and photograph again")
    ap.add_argument("--synthetic", action="store_true", help="with --start-bridge: serve the made-up six-colour 'chroma pillar' (synthetic_found.py) instead of the real inventory")
    ap.add_argument("--dist", type=float, default=1.0, help="camera distance as a multiple of the one that frames the whole scene (small objects want 2-3)")
    ap.add_argument("--outline", action="store_true", help="two-phase: draw the card's box faintly around the real shape")
    ap.add_argument("--mesh-wait", type=int, default=60, help="seconds to wait for the real shapes (two-phase)")
    a = ap.parse_args(argv)
    views = [{"yaw": float(y), "elev": 30, "dist": a.dist} for y in a.yaws.split(",")]
    if a.top:
        views.append({"yaw": 0, "elev": 80, "dist": 1.1 * a.dist})
    for old in exp_run.DATA.glob(f"shot_{a.tag}_*.jpg"):
        old.unlink()
    targs = {"requests": a.requests, "settle": a.settle, "views": views, "tag": a.tag, "yaw": a.yaw}
    if a.two_phase:
        targs.update(mesh_two_phase=True, mesh_wait=a.mesh_wait, outline=a.outline)
    srv = None
    if a.start_bridge:
        # a game that hosts its own world cannot reach localhost (-allowlocalhttp is for dedicated servers): the bridge listens on the
        # machine's Tailscale address only, behind a token made for this run (the same arrangement as skippy_e2e.py)
        import secrets
        import skippy_bridge
        from skippy_e2e import tailscale_ip
        skippy_bridge.TOKEN = secrets.token_hex(8)
        skippy_bridge.warm = lambda: None               # no language model is needed (or wanted: the GPU is the game's) for found objects
        if a.synthetic:
            import tempfile
            import synthetic_found
            from skippyverse import index as IX
            d = Path(tempfile.mkdtemp(prefix="skippyverse-synthetic-"))
            IX.build_index([synthetic_found.card()], d / "ix", synonym_groups=[])
            skippy_bridge.FOUND_INDEX = IX.load_index(d / "ix", shared=True)
            skippy_bridge.FOUND_FETCHER = synthetic_found.Fetch()
            skippy_bridge.FOUND_FETCH = False
            skippy_bridge.MESH_CACHE_DIR = d / "assets"
        skippy_bridge.found_index()                     # load the inventory now, so the first lookup is not the slow one
        host = os.environ.get("SKIPPY_BRIDGE_HOST") or tailscale_ip()
        srv = skippy_bridge.serve(host, 8777)
        targs.update(bridge=f"http://{host}:8777", token=skippy_bridge.TOKEN)
    if a.pos:
        c = [float(v) for v in a.pos.split(",")]
        targs["pos"] = {"x": c[0], "y": c[1], **({"z": c[2]} if len(c) > 2 else {})}
    try:
        res = run_tool.run_tool("holodeck_shots", targs,
                                client=True, visible=True, map_name=a.map, timeout=240 if not a.two_phase else 240 + a.mesh_wait)
    finally:
        if srv is not None:
            srv.shutdown()
            srv.server_close()
    for s in res["steps"]:
        print(f"  {s['text']!r}: {s['say']}")
    if res.get("mesh"):
        print("  real shapes:", json.dumps(res["mesh"]))
    if res.get("found"):
        print("  found entities:", json.dumps(res["found"]))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # one set of pictures, or (two-phase) the boxes and then the real shapes
    sets = [(a.tag, res["shots"])] if not a.two_phase else [(f"{a.tag}_box", res.get("box_shots", 0)), (f"{a.tag}_mesh", res["shots"])]
    rc = 0
    for tag, n in sets:
        tiles = [exp_run.DATA / f"shot_{tag}_{k + 1}.jpg" for k in range(n)]
        tiles = [t for t in tiles if t.exists()]
        if not tiles:
            print(f"no pictures were taken for {tag}")
            rc = 1
            continue
        imgs = [Image.open(t).convert("RGB") for t in tiles]
        w = 640
        imgs = [im.resize((w, int(im.height * w / im.width))) for im in imgs]
        cols = 2 if len(imgs) > 1 else 1
        rows = (len(imgs) + cols - 1) // cols
        sheet = Image.new("RGB", (w * cols, imgs[0].height * rows), (20, 20, 20))
        for i, im in enumerate(imgs):
            sheet.paste(im, ((i % cols) * w, (i // cols) * imgs[0].height))
        dest = out / f"{tag}.jpg"
        sheet.save(dest, quality=88)
        for k, im in enumerate(tiles):                  # and every view alone, at full size
            Image.open(im).convert("RGB").save(out / f"{tag}_{k + 1}.jpg", quality=90)
        print(f"{len(imgs)} views -> {dest}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
