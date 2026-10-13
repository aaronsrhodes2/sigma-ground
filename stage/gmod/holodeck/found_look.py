"""The look pass on found objects, in the real game, hands off (the window stays behind, no mouse).

    python stage/gmod/holodeck/found_look.py --phase before|after [--only kettle croissant ...]
Summons each model of the standard set one after another with the REAL bridge (cache or the one bounded download), photographs it from five fixed
cameras (iso, front, side, back, close), measures the frame cost, and writes DIR/look_<phase>_<tag>_<view>.jpg plus DIR/look_<phase>_report.json.
    python stage/gmod/holodeck/found_look.py --sheet   # builds DIR/found_look_before_after.jpg (before on top, after below) and a close-up sheet
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
if "--phase" in sys.argv and sys.argv[sys.argv.index("--phase") + 1] == "before":
    os.environ["SKIPPY_MESH_LEGACY"] = "1"                 # the "before" is the original flat-colour SVM1 payload, drawn the old flat way
import run as exp_run  # noqa: E402
import run_tool  # noqa: E402
import skippy_bridge as B  # noqa: E402

SET = [("kettle", "add a vintage electric kettle"), ("boombox", "add a boombox"), ("cannon", "add a cannon"),
       ("croissant", "add a croissant"), ("hydrant", "add a hydrant"), ("mailbox", "add a mailbox"),
       ("cake", "add a birthday cake"), ("gemini", "add a gemini"), ("lunarmodule", "add an apollo lunar module")]
VIEWS = ["iso", "front", "side", "back", "close", "top"]


def shoot(a) -> int:
    from PIL import Image
    os.environ.setdefault("GMOD_EXTRA_ARGS", "+maxplayers 4")
    items = [{"tag": t, "request": r} for t, r in SET if not a.only or t in a.only]
    B.TOKEN = secrets.token_hex(8)
    B.warm = lambda: None
    B.found_index()
    from skippy_e2e import tailscale_ip
    host = os.environ.get("SKIPPY_BRIDGE_HOST") or tailscale_ip()
    srv = B.serve(host, 8777)
    log = Path(os.environ.get("GMOD_DIR", "D:/Steam/steamapps/common/GarrysMod")) / "garrysmod" / "console.log"
    for old in exp_run.DATA.glob("shot_lk_*.jpg"):
        old.unlink()
    targs = {"items": items, "bridge": f"http://{host}:8777", "token": B.TOKEN, "timeout": 90, "prefix": "lk", "lit": a.phase != "before", "diag": os.environ.get("LK_DIAG") == "1"}
    try:
        res = run_tool.run_tool("found_look", targs, client=True, visible=True, map_name="gm_flatgrass", timeout=150 * len(items) + 200)
    finally:
        srv.shutdown()
        srv.server_close()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    perf = []
    if log.exists():
        perf = [ln.split("[fl-perf]", 1)[1].strip() for ln in log.read_text(encoding="utf-8", errors="replace").splitlines() if "[fl-perf]" in ln]
    payloads = {k: {x: v[x] for x in ("bytes", "tris", "verts") if x in v} for k, v in B.MESH_PACKED.items()}
    (out / f"look_{a.phase}_report.json").write_text(json.dumps({"result": res, "perf": perf, "payloads": payloads}, indent=1), encoding="utf-8")
    for it in items:
        for v in VIEWS:
            src = exp_run.DATA / f"shot_lk_{it['tag']}_{v}.jpg"
            if not src.exists():
                print("missing", it["tag"], v)
                continue
            im = Image.open(src).convert("RGB")
            im = im.resize((960, int(im.height * 960 / im.width)))
            im.save(out / f"look_{a.phase}_{it['tag']}_{v}.jpg", quality=90)
    print(json.dumps(res, indent=1)[:3000])
    print("\n".join(perf))
    return 0


def sheet(a) -> int:
    from PIL import Image, ImageDraw
    out = Path(a.out)
    tags = [t for t, _ in SET]
    w = 384

    def tile(phase, tag, view):
        p = out / f"look_{phase}_{tag}_{view}.jpg"
        if not p.exists():
            return Image.new("RGB", (w, w * 9 // 16), (30, 30, 30))
        im = Image.open(p).convert("RGB")
        return im.resize((w, int(im.height * w / im.width)))
    h = tile("before", tags[0], "iso").height
    for view, name in (("iso", "found_look_before_after.jpg"), ("close", "found_look_before_after_close.jpg")):
        cols = 3
        rows = ((len(tags) + cols - 1) // cols) * 2
        sh = Image.new("RGB", (w * cols, h * rows), (20, 20, 20))
        d = ImageDraw.Draw(sh)
        for i, t in enumerate(tags):
            r, c = (i // cols) * 2, i % cols
            sh.paste(tile("before", t, view), (c * w, r * h))
            sh.paste(tile("after", t, view), (c * w, (r + 1) * h))
            d.text((c * w + 4, r * h + 3), f"{t} BEFORE", fill=(255, 170, 0))
            d.text((c * w + 4, (r + 1) * h + 3), f"{t} AFTER", fill=(0, 255, 136))
        sh.save(out / name, quality=88)
        print(out / name)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["before", "after"])
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--sheet", action="store_true")
    ap.add_argument("--out", default=str(HERE / "data" / "shots"))
    a = ap.parse_args(argv)
    if a.sheet:
        return sheet(a)
    if not a.phase:
        ap.error("--phase or --sheet")
    return shoot(a)


if __name__ == "__main__":
    raise SystemExit(main())
