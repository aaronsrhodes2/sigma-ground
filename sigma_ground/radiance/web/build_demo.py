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


# Each bundle is independent — a regression in one (e.g. a Deckard cup change)
# must NOT block the others. Wrap each, report per-bundle, keep going.
import traceback
_FAILED = []

def _bundle(name, fn):
    try:
        fn()
    except Exception as e:
        _FAILED.append((name, e))
        print(f"  !! {name} SKIPPED — {type(e).__name__}: {e}")
        traceback.print_exc()


# ── 1) coffee cup — static, orbit it ────────────────────────────────────
def _build_cup():
    cup = deckard.identify("coffee cup")
    cup_spec = construct_to_scene(cup)                 # all leaves static (no `body`)
    cup_spec["kind"] = "static"
    cup_spec["sdf_samples"] = sdf_samples(cup, 4)     # ground truth for the self-check
    _write("cup.json", cup_spec)
_bundle("cup.json", _build_cup)

# ── 2) dropped sphere + floor — trajectory, play it ─────────────────────
def _build_drop():
    out = record_fall("copper", radius_m=0.05, start_altitude_m=1.5,
                      dt_max=0.005, frame_dt=0.02, target_watch_s=8.0)
    scene = out["scene"]                              # sphere already tagged body:0
    scene["csg_leaves"].append({                      # a static concrete floor (no body)
        "op": "add", "material": "concrete",
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
    print(f"  frames: {len(out['trajectory']['frames'])}  "
          f"fall: {out['trajectory']['t_end_s']:.2f}s  "
          f"rate: {out['trajectory']['suggested_rate']:.3f} sim-s/wall-s")
_bundle("drop.json", _build_drop)

# ── 3) emergent color — metals (Drude/Fresnel) + semiconductors (band gap) ──
def _build_materials():
    from sigma_ground.field.interface.surface import MATERIALS
    from sigma_ground.radiance.scene_export import _bake_band_gap
    print("\nEmergent color — metals = Drude/Fresnel, semiconductors = band-gap absorption; nobody chose them:")
    METAL_ROWS = [["copper", "gold", "silver", "aluminum"],
                  ["iron", "nickel", "titanium", "lead"],
                  ["tungsten", "platinum", "depleted_uranium", "steel_mild"]]
    # 4th row: the gap, not the metal model, sets the hue — yellow, lime, blue-grey, white.
    BANDGAP_ROW = ["cadmium_sulfide", "gallium_phosphide", "silicon", "titanium_dioxide"]
    ROWS = METAL_ROWS + [BANDGAP_ROW]
    r, sx, sy = 0.06, 0.18, 0.18
    leaves, mats = [], {}
    for row, rowmats in enumerate(ROWS):
        y = (1.5 - row) * sy                          # 4 rows, centered on origin
        band = rowmats is BANDGAP_ROW
        for col, mk in enumerate(rowmats):
            leaves.append({"op": "add", "material": mk, "dynamic": False,
                           "shape": {"type": "Sphere",
                                     "center": [(col - 1.5) * sx, y, 0.0], "radius": r}})
            if mk not in mats:
                mats[mk] = _bake_band_gap(mk) if band else \
                    _bake_material(mk, MATERIALS[mk]["density_kg_m3"])
                c = mats[mk]["color_rgb"]
                tag = f"Eg={mats[mk]['band_gap_ev']}eV" if band else "metal"
                print(f"  {mk:18s} #{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}"
                      f"  emergent={mats[mk]['emergent']:d} {tag}")
    _lm = _default_lighting([0.0, 1.0, 0.0])
    _write("materials.json", {
        "name": "emergent color — metals + band-gap semiconductors", "csg_leaves": leaves,
        "materials": mats,
        "physics": {"mass_kg": 0, "com_m": [0, 0, 0], "inertia_kgm2": [0, 0, 0]},
        "bbox": [[-0.34, 0.34], [-0.34, 0.34], [-r, r]],
        "camera": {"target": [0, 0, 0], "orbit_radius": 1.5, "fov_deg": 40.0,
                   "up": [0, 1, 0], "az0": 0.0, "el0": 0.16},   # face-on: it's a flat 4×4 grid

        "lights": _lm["lights"], "ambient": _lm["ambient"], "identified": True,
        "source": "metals: Drude/Fresnel reflectance · bottom row: band-gap absorption — every color emergent, none chosen",
        "kind": "static"})
_bundle("materials.json", _build_materials)

# ── 4) kinematic chair tip — proves rigid ROTATION + MULTI-BODY playback ──
# HONEST LABEL: this is a *kinematic preview*. The chair's tip angle and the
# ball's bounce are SCRIPTED, not solved — it exists to prove the renderer can
# play back per-body rotation+translation and >1 independent body. The real
# tipping point and the clatter arrive when Materia's rigid-contact stage lands;
# the renderer is then ready to display whatever pose stream it produces.
def _build_tip():
    import math
    H = 0.5
    q_about_x = lambda a: [math.sin(a * H), 0.0, 0.0, math.cos(a * H)]

    # a crude chair (one rigid body) from CSG boxes, authored upright, base at y=0
    seat_y, seat_t = 0.45, 0.06
    parts = [([0.0, seat_y, 0.0], [0.46, seat_t, 0.46]),            # seat
             ([0.0, seat_y + 0.25, -0.20], [0.46, 0.50, 0.05])]     # backrest
    for lx in (-0.19, 0.19):                                        # 4 legs
        for lz in (-0.19, 0.19):
            parts.append(([lx, seat_y * 0.5, lz], [0.05, seat_y, 0.05]))

    CHAIR, BALL, FLOOR = "steel_mild", "copper", "concrete"
    leaves = [{"op": "add", "material": CHAIR, "body": 0,
               "shape": {"type": "Box", "center": c, "x": d[0], "y": d[1], "z": d[2]}}
              for c, d in parts]
    ball_r, ball_c = 0.13, [0.64, 0.13, 0.26]
    leaves.append({"op": "add", "material": BALL, "body": 1,
                   "shape": {"type": "Sphere", "center": ball_c, "radius": ball_r}})
    leaves.append({"op": "add", "material": FLOOR,                  # static (no body)
                   "shape": {"type": "Box", "center": [0.0, -0.05, 0.0],
                             "x": 4.0, "y": 0.1, "z": 4.0}})

    EDGE = [0.0, 0.0, -0.19]                    # chair tips about its back-leg floor edge
    bodies = [{"pivot": EDGE, "label": "chair (tips)"},
              {"pivot": ball_c, "label": "ball (drops)"}]

    dt, t_total = 0.02, 2.2
    g, rest, e = 9.81, ball_c[1], 0.45
    by, bvy = 1.25, 0.0
    th_max = math.radians(88.0)                 # rest just before the backrest clips the floor
    frames = []
    for i in range(int(t_total / dt) + 1):
        t = i * dt
        if t < 0.25:        theta = 0.0
        elif t < 1.15:      theta = th_max * ((t - 0.25) / 0.90) ** 2   # accelerating fall
        else:               theta = th_max                             # settled (no clatter yet)
        frames.append({"t_sim": round(t, 4), "bodies": [
            {"pos": EDGE, "quat": [round(x, 6) for x in q_about_x(-theta)]},   # −θ = tips backward
            {"pos": [ball_c[0], round(by, 5), ball_c[2]], "quat": [0.0, 0.0, 0.0, 1.0]},
        ]})
        bvy -= g * dt; by += bvy * dt           # scripted free-fall + restitution
        if by < rest: by, bvy = rest, -bvy * e

    _lt = _default_lighting([0.0, 1.0, 0.0])
    scene = {
        "name": "chair tip — kinematic preview (rigid rotation + multi-body)",
        "bodies": bodies, "csg_leaves": leaves,
        "materials": {CHAIR: _bake_material(CHAIR, 7850.0),
                      BALL: _bake_material(BALL, 8960.0),
                      FLOOR: _bake_material(FLOOR, 2400.0)},
        "physics": {"mass_kg": 0, "com_m": [0, 0, 0], "inertia_kgm2": [0, 0, 0]},
        "bbox": [[-0.6, 0.9], [-0.1, 1.3], [-1.25, 0.6]],
        "camera": {"target": [0.1, 0.32, -0.15], "orbit_radius": 3.4, "fov_deg": 42.0,
                   "up": [0.0, 1.0, 0.0], "az0": 0.9, "el0": 0.28},
        "lights": _lt["lights"], "ambient": _lt["ambient"], "identified": True,
        "source": "KINEMATIC preview — scripted tip + bounce proving per-body rotation & multi-body playback (NOT solved dynamics)",
    }
    _write("tip.json", {"scene": scene, "kind": "trajectory",
                        "trajectory": {"frames": frames, "t_end_s": round((len(frames) - 1) * dt, 4),
                                       "natural_timescale_s": t_total,
                                       "suggested_rate": max(1e-6, t_total / 6.0),
                                       "body_labels": ["chair", "ball"]}})
    print(f"  tip: {len(frames)} frames · chair 0->88° about back edge (body 0) · ball bounces (body 1)")
_bundle("tip.json", _build_tip)

if _FAILED:
    print(f"\n!! {len(_FAILED)} bundle(s) skipped: " + ", ".join(n for n, _ in _FAILED)
          + " -- others written OK.")
print("\nDemo data written. Serve with:  python -m sigma_ground.radiance.web.serve")
