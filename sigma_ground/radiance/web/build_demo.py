"""Generate the viewer's demo data — the exact JSON the browser loads.

Two bundles into web/data/:
  cup.json  — the coffee cup (static): orbit a CSG construct on the GPU.
  drop.json — a copper sphere dropped onto a concrete floor (trajectory): play
              it with the time-rate knob. The floor is a STATIC leaf; the sphere
              is a DYNAMIC leaf moved by the per-frame pose.
The cup also ships Python ground-truth SDF samples for the in-page self-check.
"""
import json
import os
import sys

sys.path.insert(0, r"D:\Aaron\development\sigma-ground")

from sigma_ground import deckard
from sigma_ground.radiance import construct_to_scene, record_fall
from sigma_ground.radiance.scene_export import (sdf_samples, _bake_material,
                                               _default_lighting)

DATA = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA, exist_ok=True)


def _write(name, obj):
    path = os.path.join(DATA, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    print(f"  {name}: {os.path.getsize(path)//1024} KB")


# ── 1) coffee cup — static, orbit it ────────────────────────────────────
cup = deckard.identify("coffee cup")
cup_spec = construct_to_scene(cup)
for leaf in cup_spec["csg_leaves"]:
    leaf["dynamic"] = False
cup_spec["kind"] = "static"
cup_spec["sdf_samples"] = sdf_samples(cup, 4)     # ground truth for the self-check
_write("cup.json", cup_spec)

# ── 2) dropped sphere + floor — trajectory, play it ─────────────────────
out = record_fall("copper", radius_m=0.05, start_altitude_m=1.5,
                  dt_max=0.005, frame_dt=0.02, target_watch_s=8.0)
scene = out["scene"]
for leaf in scene["csg_leaves"]:                  # the sphere moves
    leaf["dynamic"] = True
scene["csg_leaves"].append({                      # a static concrete floor
    "op": "add", "material": "concrete", "dynamic": False,
    "shape": {"type": "Box", "center": [0.0, -0.05, 0.0],
              "x": 3.0, "y": 0.1, "z": 3.0}})
scene["materials"]["concrete"] = _bake_material("concrete", 2400.0)
scene["camera"] = {"target": [0.0, 0.55, 0.0], "orbit_radius": 2.6,
                   "fov_deg": 42.0, "up": [0.0, 1.0, 0.0]}
scene["bbox"] = [[-1.5, 1.5], [-0.1, 1.6], [-1.5, 1.5]]
_lit = _default_lighting([0.0, 1.0, 0.0])
scene["lights"] = _lit["lights"]
scene["ambient"] = _lit["ambient"]
scene["kind"] = "trajectory"
_write("drop.json", {"scene": scene, "trajectory": out["trajectory"],
                     "kind": "trajectory"})

print(f"\nframes: {len(out['trajectory']['frames'])}  "
      f"fall: {out['trajectory']['t_end_s']:.2f}s  "
      f"rate: {out['trajectory']['suggested_rate']:.3f} sim-s/wall-s")

# ── 3) emergent element colors — a grid of metal spheres ────────────────
from sigma_ground.field.interface.surface import MATERIALS
print("\nEmergent element colors (get_material_color — Drude/Fresnel; nobody chose them):")
GRID = [["copper", "gold", "silver", "aluminum"],
        ["iron", "nickel", "titanium", "lead"],
        ["tungsten", "platinum", "depleted_uranium", "steel_mild"]]
r, sx, sy = 0.06, 0.18, 0.18
leaves, mats = [], {}
for row, rowmats in enumerate(GRID):
    y = (1 - row) * sy
    for col, mk in enumerate(rowmats):
        leaves.append({"op": "add", "material": mk, "dynamic": False,
                       "shape": {"type": "Sphere",
                                 "center": [(col - 1.5) * sx, y, 0.0], "radius": r}})
        if mk not in mats:
            mats[mk] = _bake_material(mk, MATERIALS[mk]["density_kg_m3"])
            c = mats[mk]["color_rgb"]
            print(f"  {mk:18s} #{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}"
                  f"  emergent={mats[mk]['emergent']}")
_lm = _default_lighting([0.0, 1.0, 0.0])
_write("materials.json", {
    "name": "emergent element colors", "csg_leaves": leaves, "materials": mats,
    "physics": {"mass_kg": 0, "com_m": [0, 0, 0], "inertia_kgm2": [0, 0, 0]},
    "bbox": [[-0.34, 0.34], [-0.26, 0.26], [-r, r]],
    "camera": {"target": [0, 0, 0], "orbit_radius": 1.05, "fov_deg": 38.0, "up": [0, 1, 0]},
    "lights": _lm["lights"], "ambient": _lm["ambient"], "identified": True,
    "source": "each sphere's color = get_material_color (Drude/Fresnel) — nobody chose them",
    "kind": "static"})

print("\nDemo data written. Serve with:  python -m sigma_ground.radiance.web.serve")
