"""Photograph the catalog: one close three-quarter picture per model, turned so its assumed FRONT faces the camera,
then stitched into captioned sheets.

    python stage/gmod/holodeck/contact_sheet.py [--kinds couch,chair] [--out DIR]
    python stage/gmod/holodeck/contact_sheet.py --models props/cs_office/sofa.mdl,props_c17/bench01a.mdl
    python stage/gmod/holodeck/contact_sheet.py --flip props_c17/bench01a.mdl      # draw it with the OPPOSITE front

If the catalog's front is right you see the seat, the doors or the screen; if it is wrong you see the back. A red
cube marks the assumed front, a blue one the back. Model paths are written with or without the leading `models/` and the
trailing `.mdl`, and a path the catalog does not know is an error (not a silently empty sheet). Needs the game window (a
camera), so it shows for a minute or two; it is the only holodeck step that does."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "experiments"))
import build_catalog  # noqa: E402
import catalog_src  # noqa: E402
import run as exp_run  # noqa: E402
import run_tool  # noqa: E402

TILE = (426, 240)
COLS, ROWS = 3, 4


def front_for(kind: dict, variant, meas: dict, overrides: dict | None = None):
    """The catalog's front (yaw of the front direction in the model's own frame), or None for 'looks the same all round'.
    Decided by build_catalog.resolve_front, so a sheet always draws the front the catalog will use."""
    front, symmetric, _ = build_catalog.resolve_front(kind, variant, meas, overrides or {})
    return None if symmetric else front


def load_overrides() -> dict:
    p = HERE / "data" / "front_overrides.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def model_path(text: str) -> str:
    """'props_c17/bench01a', 'props_c17/bench01a.mdl' and 'models/props_c17/bench01a.mdl' all name the same model."""
    p = text.strip().replace("\\", "/")
    if not p.startswith("models/"):
        p = "models/" + p
    return p if p.endswith(".mdl") else p + ".mdl"


def split_paths(text: str) -> list[str]:
    return [model_path(k) for k in text.split(",") if k.strip()]


def plan_items(kinds: set[str], models: list[str], flips: list[str], measured: dict, overrides: dict) -> tuple[list, list, list[str], list[str]]:
    """(items for the game, captions, unknown, notes). `unknown` lists requested kinds and models the catalog does not have (an
    error: the sheet would not show what was asked for); `notes` lists requests that cannot be honoured (a model missing from this
    install or without physics, a flip of something that looks the same from every side)."""
    unknown: list[str] = []
    problems: list[str] = []
    known = {(v if isinstance(v, str) else v["model"]) for k in catalog_src.KINDS for v in k["variants"]}
    names = {k["kind"] for k in catalog_src.KINDS}
    unknown += [f"no kind named {k!r}" for k in sorted(kinds - names)]
    unknown += [f"no model {p!r} in the catalog" for p in models + flips if p not in known]
    want = set(models) | set(flips)
    items, labels = [], []
    for kind in catalog_src.KINDS:
        if kinds and kind["kind"] not in kinds:
            continue
        for v in kind["variants"]:
            path = v if isinstance(v, str) else v["model"]
            if want and path not in want:
                continue
            m = measured.get(path)
            if not m or m.get("missing") or m.get("no_physics"):
                if path in want:
                    problems.append(f"{path} cannot be photographed (missing from this install, or no physics)")
                continue
            fy = front_for(kind, v, m, overrides)
            flipped = path in flips
            if flipped:
                if fy is None:
                    problems.append(f"{path} looks the same from every side: nothing to flip")
                else:
                    fy = (fy + 180.0) % 360.0
            items.append({"model": path, "front_yaw": fy, "mins": m["mins"], "maxs": m["maxs"]})
            labels.append(f"{kind['kind']}: {path.replace('models/', '')}" + (" (FLIPPED)" if flipped and fy is not None else ""))
    return items, labels, unknown, problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "data" / "sheets"))
    ap.add_argument("--kinds", default="")
    ap.add_argument("--models", default="", help="comma list of model paths (with or without the leading models/ and the .mdl)")
    ap.add_argument("--flip", default="", help="comma list of model paths to draw with the OPPOSITE front (to settle a doubtful one)")
    a = ap.parse_args(argv)
    measured = {m["model"]: m for m in json.loads((HERE / "data" / "measured.json").read_text(encoding="utf-8"))["models"]}
    want_kinds = {k.strip() for k in a.kinds.split(",") if k.strip()}
    items, labels, unknown, notes = plan_items(want_kinds, split_paths(a.models), split_paths(a.flip), measured, load_overrides())
    for p in unknown + notes:
        print("contact_sheet:", p, file=sys.stderr)
    if unknown or not items:
        return 2                                               # do not start the game for a sheet that is wrong or empty
    from PIL import Image, ImageDraw                           # only the photographing needs it
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
