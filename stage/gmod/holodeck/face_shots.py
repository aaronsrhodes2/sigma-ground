"""Photograph a picture head in the real game (hands off: the launcher keeps the window behind and takes no mouse).

    python stage/gmod/holodeck/face_shots.py [--out DIR]
A bot wears a SYNTHETIC picture (drawn here with Pillow, shrunk by the bridge's own pipeline, skippy_faces.process); the local player is the
camera. Pictures: DIR/face_<view>.jpg (default holodeck/data/shots/). The game's console lines starting [face-diag] are printed too.
Run only when no other game process is running (it starts and stops one). See holodeck/README.md, Faces.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sys
from io import BytesIO
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "experiments"))
import run as exp_run  # noqa: E402
import run_tool  # noqa: E402
import skippy_faces  # noqa: E402


def synthetic_face() -> bytes:
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (400, 400), (40, 110, 190))
    d = ImageDraw.Draw(im)
    d.ellipse((90, 40, 310, 380), fill=(235, 195, 160))                 # head
    d.ellipse((140, 150, 190, 200), fill=(255, 255, 255))
    d.ellipse((210, 150, 260, 200), fill=(255, 255, 255))
    d.ellipse((157, 163, 177, 187), fill=(20, 20, 20))
    d.ellipse((227, 163, 247, 187), fill=(20, 20, 20))
    d.arc((140, 220, 260, 320), 20, 160, fill=(150, 40, 40), width=8)    # smile
    d.rectangle((0, 0, 60, 60), fill=(255, 0, 0))                        # corner marks: the picture's orientation must be readable
    d.rectangle((340, 340, 400, 400), fill=(0, 200, 0))
    buf = BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "data" / "shots"))
    ap.add_argument("--map", default="gm_flatgrass")
    a = ap.parse_args(argv)
    os.environ.setdefault("GMOD_EXTRA_ARGS", "+maxplayers 4")
    os.environ["GMOD_MOUSE"] = "0"
    res = skippy_faces.process(synthetic_face(), use_detector=False)
    jpeg = res["jpeg"]
    args = {"b64": base64.b64encode(jpeg).decode(), "hash": hashlib.sha256(jpeg).hexdigest()[:24], "border": res["border"]}
    for old in exp_run.DATA.glob("shot_face_*.jpg"):
        old.unlink()
    result = run_tool.run_tool("face_shots", args, client=True, visible=True, map_name=a.map, timeout=240)
    print(json.dumps(result))
    log = Path(os.environ.get("GMOD_DIR", "/d/Steam/steamapps/common/GarrysMod")) / "garrysmod" / "console.log"
    try:
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            if "[face-diag]" in line or ("mentat_face" in line.lower() and "error" in line.lower()):
                print(line.strip()[:300])
    except OSError:
        print("(no console log found)")
    from PIL import Image
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rc = 0
    for src in sorted(exp_run.DATA.glob("shot_face_*.jpg")):
        im = Image.open(src).convert("RGB")
        im = im.resize((960, int(im.height * 960 / im.width)))
        dest = out / (re.sub(r"^shot_", "", src.name))
        im.save(dest, quality=85)
        print(dest)
    if not list(exp_run.DATA.glob("shot_face_*.jpg")):
        print("no pictures were taken")
        rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
