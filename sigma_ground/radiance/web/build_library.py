"""Generate a LIBRARY of *watchable* simulations — the renderable subset of
Materia's functional questions, saved as data/<slug>.json plus a data/scenes.json
index the viewer reads to build a clickable gallery.

Two families yield a watchable scene (the rest are numbers):
  • NAMED-OBJECT drops  → Deckard grounds the real shape → record_object_fall
  • SPHERE drops        → emergent metal colour → record_fall

Heights are kept SMALL and the camera framed so the object is actually visible
falling onto a dark stage (a 5 cm ball from 10 km is a real sim but sub-pixel).
Run:  python -m sigma_ground.radiance.web.build_library
"""
import json
import os
import sys

sys.path.insert(0, r"D:\Aaron\development\sigma-ground")

from sigma_ground import deckard
from sigma_ground.radiance import record_fall, record_object_fall
from sigma_ground.radiance.scene_export import _bake_material, _default_lighting

DATA = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA, exist_ok=True)

# (Deckard name, slug suffix, drop height m) — low enough to SEE the object fall.
OBJECTS = [
    ("feather",            "feather",  2.4384),
    ("cast iron skillet",  "skillet",  3.0),
    ("drinking glass",     "glass",    2.5),
    ("coffee cup",         "cup",      2.5),
]
# (material_key, slug, label) — a 15 cm ball from 4 m: colour is emergent, lands on the stage.
SPHERES = [
    ("copper", "copper", "Copper"), ("gold", "gold", "Gold"),
    ("aluminum", "aluminum", "Aluminum"), ("iron", "iron", "Iron"),
    ("lead", "lead", "Lead"), ("titanium", "titanium", "Titanium"),
    ("tungsten", "tungsten", "Tungsten"), ("steel_mild", "steel", "Steel"),
]

scenes, failed = [], []


def _frame(bundle, alt):
    """Clean, consistent framing: a dark stage to land on + a 3/4 camera that keeps
    the whole drop in view (y-up). Overrides whatever default the recorder chose."""
    scene = bundle["scene"]
    if not any(l.get("material") in ("stage_dark", "concrete") for l in scene["csg_leaves"]):
        scene["csg_leaves"].append({"op": "add", "material": "stage_dark",
            "shape": {"type": "Box", "center": [0.0, -0.06, 0.0], "x": 4.0, "y": 0.12, "z": 4.0}})
        f = _bake_material("concrete", 2400.0)
        f["color_rgb"] = [0.07, 0.07, 0.08]            # neutral dark stage, object reads against it
        scene["materials"]["stage_dark"] = f
    scene["camera"] = {"target": [0.0, alt * 0.42, 0.0], "orbit_radius": max(2.2, alt * 1.25),
                       "fov_deg": 44.0, "up": [0.0, 1.0, 0.0], "az0": 0.5, "el0": 0.18}
    scene["bbox"] = [[-2.0, 2.0], [-0.2, alt + 0.4], [-2.0, 2.0]]
    lit = _default_lighting([0.0, 1.0, 0.0])
    scene["lights"], scene["ambient"] = lit["lights"], lit["ambient"]
    return {"scene": scene, "trajectory": bundle["trajectory"], "kind": "trajectory"}


def _add(slug, title, question, verb, bundle):
    with open(os.path.join(DATA, slug + ".json"), "w", encoding="utf-8") as fh:
        json.dump(bundle, fh, indent=1)
    tr = bundle["trajectory"]
    scenes.append({"slug": slug, "title": title, "question": question, "verb": verb,
                   "kind": "trajectory", "frames": len(tr.get("frames", [])),
                   "fall_s": round(tr.get("natural_timescale_s") or tr.get("t_end_s") or 0.0, 2)})
    print(f"  + {slug:18s} {title}")


# 1) named-object drops (real Deckard shapes)
for name, key, alt in OBJECTS:
    try:
        c = deckard.identify(name)
        if not getattr(c, "identified", False):
            failed.append((name, "shape not grounded")); print(f"  ! {name}: not grounded"); continue
        out = record_object_fall(c, alt)
        _add(f"fall_{key}", f"{name.title()} — dropped from {alt:g} m",
             f"drop a {name} from {alt:g} m", "drop_object", _frame(out, alt))
    except Exception as e:
        failed.append((name, str(e))); print(f"  ! {name}: {type(e).__name__}: {e}")

# 2) sphere drops — emergent colour, landed on the stage
SPH_ALT, SPH_R = 4.0, 0.15
for mat, key, label in SPHERES:
    try:
        out = record_fall(mat, radius_m=SPH_R, start_altitude_m=SPH_ALT,
                          dt_max=0.004, frame_dt=0.02, target_watch_s=4.0)
        _add(f"fall_sphere_{key}", f"{label} ball — 15 cm, from {SPH_ALT:g} m",
             f"drop a 15 cm {key} ball from {SPH_ALT:g} m", "terminal_velocity_drop",
             _frame(out, SPH_ALT))
    except Exception as e:
        failed.append((mat, str(e))); print(f"  ! {mat}: {type(e).__name__}: {e}")

with open(os.path.join(DATA, "scenes.json"), "w", encoding="utf-8") as fh:
    json.dump(scenes, fh, indent=1)
print(f"\nLibrary: {len(scenes)} watchable scenes -> data/scenes.json"
      + (f"  ({len(failed)} skipped)" if failed else ""))
for name, err in failed:
    print(f"   skipped {name}: {err[:80]}")
