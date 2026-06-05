"""Stage-A demo — Deckard → SceneSpec JSON → render; Materia → Trajectory JSON.

Proves the Python→browser contract end to end on the CPU: identify the coffee
cup, serialize it, render it FROM the serialized SceneSpec (not the live
construct), and record a falling-sphere trajectory with an auto-suggested
playback rate. These two JSON files are exactly what the web viewer will load.
"""
import json
import os
import sys

sys.path.insert(0, r"D:\Aaron\development\sigma-ground")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from sigma_ground import deckard
from sigma_ground.dynamics.vec import Vec3
from sigma_ground.radiance import (construct_to_scene, scene_from_spec,
                                   record_fall, Camera)
from sigma_ground.radiance.render import render_to_png

OUT = r"D:\Aaron\development\sigma-ground\misc\renders"
os.makedirs(OUT, exist_ok=True)

print("=" * 66)
print("RADIANCE Stage A — the Python → browser contract")
print("=" * 66)

# ── Deckard → SceneSpec ────────────────────────────────────────────────
cup = deckard.identify("coffee cup")
print("\n" + cup.render())
spec = construct_to_scene(cup)
scene_path = os.path.join(OUT, "coffee_cup.scene.json")
with open(scene_path, "w", encoding="utf-8") as f:
    json.dump(spec, f, indent=2)
print(f"\nSceneSpec → {scene_path}")
print(f"  leaves: {[n['material'] for n in spec['csg_leaves']]}")
print("  baked colors:", {k: [round(c, 2) for c in v['color_rgb']]
                          for k, v in spec['materials'].items()})

# render the cup FROM the SceneSpec (browser will build the same SDF in GLSL)
scene = scene_from_spec(spec)
target = Vec3(*spec["camera"]["target"])
R = spec["camera"]["orbit_radius"]
cam = Camera(target + Vec3(R * 0.7, -R * 0.7, R * 0.45), target,
             up=Vec3(0, 0, 1), fov_deg=spec["camera"]["fov_deg"],
             width=220, height=200)
png = render_to_png(scene, cam, os.path.join(OUT, "radiance_coffee_cup_from_spec.png"))
print(f"  rendered from spec → {png}")

# ── Materia → Trajectory ───────────────────────────────────────────────
print("\n" + "-" * 66)
out = record_fall("copper", radius_m=0.05, start_altitude_m=10_000.0)
tr = out["trajectory"]
traj_path = os.path.join(OUT, "copper_fall.sim.json")
with open(traj_path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print(f"Trajectory → {traj_path}")
print(f"  frames: {len(tr['frames'])}   sim duration: {tr['t_end_s']:.1f} s")
print(f"  suggested playback rate: {tr['suggested_rate']:.2f} sim-s per wall-s "
      f"({'time-lapse' if tr['suggested_rate'] > 1 else 'slow-mo'})")
print(f"  altitude {tr['frames'][0]['bodies'][0]['pos'][1]:.0f} m "
      f"→ {tr['frames'][-1]['bodies'][0]['pos'][1]:.1f} m")
