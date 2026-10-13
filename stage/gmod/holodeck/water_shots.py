"""Photograph the construct's water in the real game (a visible window kept behind everything, no mouse; about one to two minutes a scenario).

    python stage/gmod/holodeck/water_shots.py pool    # a pool: floaters of three densities, a sinking steel crate, a boulder's splash and rings, underwater, a swimming bot
    python stage/gmod/holodeck/water_shots.py sea     # a sea slab seen from its shore, a round pond with a dropped crate
    python stage/gmod/holodeck/water_shots.py moon    # the same under the Moon's gravity
    python stage/gmod/holodeck/water_shots.py perf    # milliseconds per frame: pool in view / out of view / no water
    python stage/gmod/holodeck/water_shots.py sheet   # rebuild water_sheet.jpg from the pictures already taken

Pictures: stage/gmod/holodeck/data/shots/water_<scenario>_*.jpg and the contact sheet water_sheet.jpg.  The launcher is the repository's own:
GMOD_MOUSE stays 0 (never --mouse), so the window never takes the mouse or the focus from whoever is at the PC.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "experiments"))
sys.path.insert(0, str(HERE.parents[1]))
import run as exp_run  # noqa: E402
import run_tool  # noqa: E402

OUT = HERE / "data" / "shots"


def sheet(width=520):
    from PIL import Image, ImageDraw
    files = []
    for sc in ("pool", "sea", "moon"):
        files += sorted(OUT.glob(f"water_{sc}_*.jpg"), key=lambda p: p.name)
    # numeric order inside a scenario (p1, p2, ... p10)
    def key(p):
        parts = p.stem.split("_")
        tag = parts[2] if len(parts) > 2 else ""
        digits = "".join(c for c in tag if c.isdigit())
        return (parts[1], int(digits) if digits else 0, p.name)
    files = sorted(files, key=key)
    if not files:
        print("no pictures to put on a sheet")
        return None
    tiles = []
    for f in files:
        im = Image.open(f).convert("RGB")
        im = im.resize((width, int(im.height * width / im.width)))
        d = ImageDraw.Draw(im)
        d.rectangle((0, 0, im.width, 16), fill=(0, 0, 0))
        d.text((4, 2), f.stem.replace("water_", ""), fill=(0, 255, 136))
        tiles.append(im)
    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    h = max(t.height for t in tiles)
    sh = Image.new("RGB", (width * cols, h * rows), (20, 20, 20))
    for i, im in enumerate(tiles):
        sh.paste(im, ((i % cols) * width, (i // cols) * h))
    dest = OUT / "water_sheet.jpg"
    sh.save(dest, quality=86)
    print(dest, sh.size)
    return dest


def main(argv=None) -> int:
    from PIL import Image
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", choices=["pool", "sea", "moon", "perf", "sheet"])
    ap.add_argument("--map", default="gm_flatgrass")
    ap.add_argument("--width", type=int, default=1100)
    a = ap.parse_args(argv)
    if a.scenario == "sheet":
        sheet()
        return 0
    if os.environ.get("GMOD_MOUSE", "0") != "0":
        print("refusing to run with GMOD_MOUSE set: the water run never takes the mouse")
        return 2
    os.environ.setdefault("GMOD_EXTRA_ARGS", "+maxplayers 4")
    for old in exp_run.DATA.glob("shot_wtr_*.jpg"):
        old.unlink()
    (exp_run.DATA / "water_perf.txt").unlink(missing_ok=True)
    res = run_tool.run_tool("water_shots", {"scenario": a.scenario}, client=True, visible=True, map_name=a.map, timeout=420)
    print(json.dumps({k: v for k, v in res.items() if k != "frames"}, indent=1))
    OUT.mkdir(parents=True, exist_ok=True)
    for f in res.get("frames", []):
        src = exp_run.DATA / f"shot_wtr_{f['name']}.jpg"
        if not src.exists():
            print("missing picture:", f["name"])
            continue
        im = Image.open(src).convert("RGB")
        im = im.resize((a.width, int(im.height * a.width / im.width)))
        dest = OUT / f"water_{a.scenario}_{f['name']}.jpg"
        im.save(dest, quality=90)
        print(dest, f.get("water"))
    perf = exp_run.DATA / "water_perf.txt"
    if perf.exists():
        print(perf.read_text(encoding="utf-8"))
    sheet()
    return 0


if __name__ == "__main__":
    sys.exit(main())
