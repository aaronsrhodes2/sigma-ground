"""Photograph Skippy's avatar (the beer can in the bicorn) in the real game: a visible window for about a minute.

    python stage/gmod/holodeck/avatar_shots.py [--views '[{"name":"front","dist":90,"yaw":0,"elev":8}]'] [--out DIR] [--prefix avatar]
Summons him with the sentence a player would say, takes the player's own view, then parks him and photographs him from each view
(yaw is relative to the way he faces). Pictures are saved as small JPEGs DIR/<prefix>_<name>.jpg. See stage/gmod/AVATAR.md.
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
import run as exp_run  # noqa: E402
import run_tool  # noqa: E402


def main(argv=None) -> int:
    from PIL import Image
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", default="")
    ap.add_argument("--out", default=str(HERE / "data" / "shots"))
    ap.add_argument("--prefix", default="avatar")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--map", default="gm_flatgrass")
    a = ap.parse_args(argv)
    os.environ.setdefault("GMOD_EXTRA_ARGS", "+maxplayers 4")
    targs = {}
    names = ["firstperson"]
    if a.views:
        targs["views"] = json.loads(a.views)
    names += [v["name"] for v in targs.get("views", [])] or ["front", "side", "far5m"]
    for old in exp_run.DATA.glob("shot_av_*.jpg"):
        old.unlink()
    res = run_tool.run_tool("avatar_shots", targs, client=True, visible=True, map_name=a.map, timeout=150)
    print(json.dumps(res))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rc = 0
    for n in names:
        src = exp_run.DATA / f"shot_av_{n}.jpg"
        if not src.exists():
            print(f"missing picture: {n}")
            rc = 1
            continue
        im = Image.open(src).convert("RGB")
        im = im.resize((a.width, int(im.height * a.width / im.width)))
        dest = out / f"{a.prefix}_{n}.jpg"
        im.save(dest, quality=85)
        print(dest)
    return rc


if __name__ == "__main__":
    sys.exit(main())
