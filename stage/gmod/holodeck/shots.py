"""Photograph what the holodeck makes, in the real game (a visible window for about a minute).

    python stage/gmod/holodeck/shots.py "a living room" [--tag lr] [--out DIR] [--views 4]
Stitches the views into one contact image DIR/<tag>.jpg and prints the holodeck's replies."""
from __future__ import annotations

import argparse
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
    a = ap.parse_args(argv)
    views = [{"yaw": float(y), "elev": 30, "dist": 1.0} for y in a.yaws.split(",")]
    if a.top:
        views.append({"yaw": 0, "elev": 80, "dist": 1.1})
    for old in exp_run.DATA.glob(f"shot_{a.tag}_*.jpg"):
        old.unlink()
    targs = {"requests": a.requests, "settle": a.settle, "views": views, "tag": a.tag, "yaw": a.yaw}
    if a.pos:
        c = [float(v) for v in a.pos.split(",")]
        targs["pos"] = {"x": c[0], "y": c[1], **({"z": c[2]} if len(c) > 2 else {})}
    res = run_tool.run_tool("holodeck_shots", targs,
                            client=True, visible=True, map_name=a.map, timeout=240)
    for s in res["steps"]:
        print(f"  {s['text']!r}: {s['say']}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tiles = [exp_run.DATA / f"shot_{a.tag}_{k + 1}.jpg" for k in range(res["shots"])]
    tiles = [t for t in tiles if t.exists()]
    if not tiles:
        print("no pictures were taken")
        return 1
    imgs = [Image.open(t).convert("RGB") for t in tiles]
    w = 640
    imgs = [im.resize((w, int(im.height * w / im.width))) for im in imgs]
    cols = 2 if len(imgs) > 1 else 1
    rows = (len(imgs) + cols - 1) // cols
    sheet = Image.new("RGB", (w * cols, imgs[0].height * rows), (20, 20, 20))
    for i, im in enumerate(imgs):
        sheet.paste(im, ((i % cols) * w, (i // cols) * imgs[0].height))
    dest = out / f"{a.tag}.jpg"
    sheet.save(dest, quality=88)
    for k, im in enumerate(tiles):                      # and every view alone, at full size
        Image.open(im).convert("RGB").save(out / f"{a.tag}_{k + 1}.jpg", quality=90)
    print(f"{len(imgs)} views -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
