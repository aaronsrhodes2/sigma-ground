"""Photograph the catalog: one close three-quarter picture per model, turned so its assumed FRONT faces the camera,
then stitched into captioned sheets.

    python stage/gmod/holodeck/contact_sheet.py [--kinds couch,chair] [--out DIR]
    python stage/gmod/holodeck/contact_sheet.py --models props/cs_office/sofa.mdl,props_c17/bench01a.mdl

If the catalog's front is right you see the seat, the doors or the screen; if it is wrong you see the back. A red
cube marks the assumed front, a blue one the back. Needs the game window (a camera), so it shows for a minute or two;
it is the only holodeck step that does."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "experiments"))
import catalog_src  # noqa: E402
import run as exp_run  # noqa: E402
import run_tool  # noqa: E402

BACK_TO_FRONT_YAW = {"+x": 180.0, "-x": 0.0, "+y": 270.0, "-y": 90.0}
TILE = (426, 240)
COLS, ROWS = 3, 4


def front_for(kind: dict, variant, meas: dict, overrides: dict | None = None):
    """The catalog's front (yaw of the front direction in the model's own frame), or None for 'looks the same all round'."""
    path = variant if isinstance(variant, str) else variant["model"]
    if overrides and path in overrides:
        return overrides[path]
    f = variant.get("front") if isinstance(variant, dict) and variant.get("front") is not None else kind.get("front")
    if f == "none":
        return None
    if isinstance(f, (int, float)):
        return float(f)
    return BACK_TO_FRONT_YAW.get((meas.get("facing") or {}).get("back"))


def load_overrides() -> dict:
    p = HERE / "data" / "front_overrides.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "data" / "sheets"))
    ap.add_argument("--kinds", default="")
    ap.add_argument("--models", default="", help="comma list of model paths (without the leading models/)")
    ap.add_argument("--flip", default="", help="comma list of model paths to draw with the OPPOSITE front (to settle a doubtful one)")
    a = ap.parse_args(argv)
    measured = {m["model"]: m for m in json.loads((HERE / "data" / "measured.json").read_text(encoding="utf-8"))["models"]}
    over = load_overrides()
    want_kinds = {k.strip() for k in a.kinds.split(",") if k.strip()}
    want_models = {"models/" + k.strip() + ("" if k.strip().endswith(".mdl") else ".mdl") for k in a.models.split(",") if k.strip()}
    flips = {"models/" + k.strip() + ".mdl" for k in a.flip.split(",") if k.strip()}
    want_models |= flips
    items, labels = [], []
    for kind in catalog_src.KINDS:
        if want_kinds and kind["kind"] not in want_kinds:
            continue
        for v in kind["variants"]:
            path = v if isinstance(v, str) else v["model"]
            if want_models and path not in want_models:
                continue
            m = measured.get(path)
            if not m or m.get("missing") or m.get("no_physics"):
                continue
            fy = front_for(kind, v, m, over)
            if path in flips and fy is not None:
                fy = (fy + 180.0) % 360.0
            items.append({"model": path, "front_yaw": fy, "mins": m["mins"], "maxs": m["maxs"]})
            labels.append(f"{kind['kind']}: {path.replace('models/', '')}" + (" (FLIPPED)" if path in flips else ""))
    for old in exp_run.DATA.glob("shot_item_*.jpg"):
        old.unlink()
    res = run_tool.run_tool("contact_sheet", {"items": items}, client=True, visible=True, timeout=90 + 4 * len(items))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    per = COLS * ROWS
    sheets = []
    for s0 in range(0, len(items), per):
        sheet = Image.new("RGB", (TILE[0] * COLS, TILE[1] * ROWS), (20, 20, 20))
        d = ImageDraw.Draw(sheet)
        for j in range(s0, min(len(items), s0 + per)):
            src = exp_run.DATA / f"shot_item_{j + 1}.jpg"
            col, row = (j - s0) % COLS, (j - s0) // COLS
            if src.exists():
                tile = Image.open(src).convert("RGB").resize(TILE)
                sheet.paste(tile, (col * TILE[0], row * TILE[1]))
            d.rectangle((col * TILE[0], row * TILE[1], col * TILE[0] + TILE[0], row * TILE[1] + 14), fill=(0, 0, 0))
            d.text((col * TILE[0] + 3, row * TILE[1] + 2), f"{j + 1}. {labels[j]}", fill=(255, 255, 120))
            f = items[j]["front_yaw"]
            d.text((col * TILE[0] + TILE[0] - 70, row * TILE[1] + 2), "front=" + ("-" if f is None else f"{f:.0f}"), fill=(120, 255, 120))
        name = out / f"sheet_{s0 // per + 1}.jpg"
        sheet.save(name, quality=88)
        sheets.append(name)
    print(f"{res.get('shots')} pictures -> {len(sheets)} sheets in {out}")
    for i, name in enumerate(sheets):
        print(f"  {name.name}: items {i * per + 1} to {min(len(items), (i + 1) * per)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
