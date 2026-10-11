"""Merge the curated list (catalog_src.py), the engine's measurements (data/measured.json) and the by-eye front
corrections (data/front_overrides.json) into the library the game loads:

    python stage/gmod/holodeck/build_catalog.py

Writes stage/gmod/addon/lua/mentat/holodeck/catalog.lua (the game) and data/catalog.json (the Python tests). All lengths
are Hammer units (HU), the engine's own, so the layout code needs no conversion; one HU is 0.01905 m on this stage.

For every variant:
    front    yaw (deg) of the direction the thing faces, in the MODEL's own frame (0 = +X). A thing that looks the same from
             every side gets front=0 and symmetric=true.
    w, d, h  width across the front, depth along the front, height (HU), from the engine's bounding box
    cx, cy   centre of the footprint relative to the model's origin, in the model frame (HU)
    mins/maxs the engine's bounding box
    material, skins  as the engine reports them
    mass     the REAL-WORLD mass in kg from the fidelity layer (fidelity.py: a sourced reference scaled to the model's size and held inside the
             family's sourced range), or the engine's own gameplay mass where the sources do not describe the model (mass_status "unaudited")
    engine_mass, mass_source, mass_status ("audited" | "unaudited"), mass_low, mass_high (the sourced range, audited only),
    mass_confidence ("high" | "medium" | "low", audited only), makeup (what it is made of): the provenance of `mass`, from data/fidelity.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "experiments"))
import catalog_src  # noqa: E402
import fidelity  # noqa: E402
from luaout import lua  # noqa: E402

BACK_TO_FRONT_YAW = {"+x": 180.0, "-x": 0.0, "+y": 270.0, "-y": 90.0}
OUT_LUA = HERE.parent / "addon" / "lua" / "mentat" / "holodeck" / "catalog.lua"
OUT_JSON = HERE / "data" / "catalog.json"
VERSION = 1


def resolve_front(kind: dict, variant, meas: dict, overrides: dict):
    """(front yaw, symmetric, how it was decided)."""
    path = variant if isinstance(variant, str) else variant["model"]
    if isinstance(variant, dict) and variant.get("front") is not None:
        return float(variant["front"]), False, "curated per model"
    if path in overrides:
        return float(overrides[path]), False, "checked by eye"
    f = kind.get("front")
    if f == "none":
        return 0.0, True, "symmetric"
    if isinstance(f, (int, float)):
        return float(f), False, "curated per kind"
    back = (meas.get("facing") or {}).get("back")
    if back in BACK_TO_FRONT_YAW:
        return BACK_TO_FRONT_YAW[back], False, "from the collision mesh (confirmed by eye)"
    return 0.0, True, "unknown, treated as symmetric"


def build() -> dict:
    measured = {m["model"]: m for m in json.loads((HERE / "data" / "measured.json").read_text(encoding="utf-8"))["models"]}
    overrides_path = HERE / "data" / "front_overrides.json"
    overrides = json.loads(overrides_path.read_text(encoding="utf-8")) if overrides_path.exists() else {}
    fid = fidelity.load()["variants"]
    kinds, order, problems = {}, [], []
    for k in catalog_src.KINDS:
        variants = []
        for v in k["variants"]:
            path = v if isinstance(v, str) else v["model"]
            m = measured.get(path)
            if not m or m.get("missing") or m.get("no_physics"):
                problems.append(f"{k['kind']}: {path} is not available in the game ({'missing' if not m or m.get('missing') else 'no physics'})")
                continue
            f = fid.get(path)
            if f is None:
                problems.append(f"{k['kind']}: {path} has no entry in data/fidelity.json (run python stage/gmod/holodeck/fidelity.py)")
                continue
            front, symmetric, how = resolve_front(k, v, m, overrides)
            mn, mx = m["mins"], m["maxs"]
            ex, ey, ez = mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]
            if int(round(front)) % 180 == 0:                 # facing along x: depth is the x extent
                d, w = ex, ey
            else:
                d, w = ey, ex
            top = None
            t = m.get("top")
            if k.get("surface") and t and t["area"] >= 0.6 * t["footprint"]:
                # a flat that most of the footprint shares: things can stand on it. Heights are above the floor.
                top = {"z": t["z"] - mn[2], "x0": t["x0"], "x1": t["x1"], "y0": t["y0"], "y1": t["y1"]}
            variants.append({
                "model": path, "front": front % 360.0, "symmetric": symmetric, "front_source": how, "top": top,
                "w": w, "d": d, "h": ez, "mins": mn, "maxs": mx, "cx": (mn[0] + mx[0]) / 2, "cy": (mn[1] + mx[1]) / 2,
                "mass": f["mass_kg"], "engine_mass": m["mass"], "mass_source": f["mass_source"], "mass_status": f["status"],
                "mass_low": f["mass_low_kg"], "mass_high": f["mass_high_kg"], "mass_confidence": f["confidence"], "makeup": f.get("makeup"),
                "material": m.get("material") or "default", "skins": m.get("skins", 1),
            })
        if not variants:
            problems.append(f"{k['kind']}: no usable model")
            continue
        order.append(k["kind"])
        kinds[k["kind"]] = {
            "kind": k["kind"], "names": k["names"], "approx": k.get("approx", []), "category": k["category"],
            "seat": bool(k.get("seat")), "surface": bool(k.get("surface")), "wall": k.get("wall", "free"), "variants": variants,
        }
    recipes = {name: [[n, kind, role] for n, kind, role in lines] for name, lines in catalog_src.RECIPES.items()}
    for name, lines in recipes.items():
        for n, kind, role in lines:
            if kind not in kinds:
                problems.append(f"recipe '{name}' uses '{kind}', which is not in the library")
    return {"version": VERSION, "hu_m": 0.01905, "order": order, "kinds": kinds, "recipes": recipes, "problems": problems}


def main() -> int:
    cat = build()
    if cat["problems"]:
        print("PROBLEMS:")
        for p in cat["problems"]:
            print("  -", p)
        return 1
    header = ("-- GENERATED by stage/gmod/holodeck/build_catalog.py from catalog_src.py, the engine's own measurements and the\n"
              "-- by-eye front checks: DO NOT EDIT BY HAND. Lengths are Hammer units (1 HU = 0.01905 m on this stage).\n")
    body = {k: v for k, v in cat.items() if k != "problems"}
    OUT_LUA.parent.mkdir(parents=True, exist_ok=True)
    OUT_LUA.write_text(header + "return " + lua(body) + "\n", encoding="utf-8", newline="\n")
    OUT_JSON.write_text(json.dumps(body, indent=1), encoding="utf-8")
    n = sum(len(k["variants"]) for k in cat["kinds"].values())
    print(f"{len(cat['kinds'])} kinds, {n} models -> {OUT_LUA.name} and {OUT_JSON.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
