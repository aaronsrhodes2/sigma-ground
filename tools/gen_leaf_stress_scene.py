"""gen_leaf_stress_scene.py -- deterministic synthetic scenes for the CSG
leaf-count shader compile-time sweep (R4.7).

Von Neumann's own leaf_stress_{1,2,4,8,16}x.json / leaf_stress2_*.json
fixtures (sigma_ground/radiance/web/data/, untracked) already validated
this scene shape empirically -- their `source` field cites a generator
(Von Neumann/assets/stress_test_leaf_budget[_v2].py) that no longer exists
anywhere in this repo. This script is a fresh, committed, documented
replacement: same loadable SceneSpec shape, but its own explicit,
deterministic layout algorithm (round-robin Sphere/Cylinder/Cone/Box,
--leaves-per-body isolates the per-body qrotInv cost from the per-leaf SDF
cost -- see buildPathShader's bodyDecls, one line per BODY not per leaf).

Every leaf is op:"add" (no CSG-boolean confound) and shares ONE material
label (no palette-size confound) -- the sweep this feeds is about LEAF
COUNT specifically, not materials or booleans.

No randomness: every position/dimension is a closed-form function of the
leaf/body index, so the same --n always produces byte-identical output.

Run:
    python tools/gen_leaf_stress_scene.py --n 4,8,11,16,22,32,44,64,88,128,176
    python tools/gen_leaf_stress_scene.py --n 44 --leaves-per-body 4
"""
import argparse
import json
import math
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_ROOT, "sigma_ground", "radiance", "web", "data")

_SHAPE_CYCLE = ("Sphere", "Cylinder", "Cone", "Box")
_MATERIAL = "leafsweep_hull"
_BODY_SPACING_M = 4.0          # centre-to-centre, along +x -- comfortably clears every shape below
_LEAF_LOCAL_SPACING_M = 0.6    # leaves within one body, stacked along +z in body-local space


def _shape_for(i):
    """A small, deterministic, non-degenerate shape for leaf index i."""
    kind = _SHAPE_CYCLE[i % len(_SHAPE_CYCLE)]
    r = 0.25 + 0.03 * (i % 5)          # 0.25..0.37 m -- varies slightly, never zero
    if kind == "Sphere":
        return {"type": "Sphere", "center": [0.0, 0.0, 0.0], "radius": r}
    if kind == "Cylinder":
        return {"type": "Cylinder", "center": [0.0, 0.0, 0.0], "radius": r, "height": 2.0 * r}
    if kind == "Cone":
        return {"type": "Cone", "center": [0.0, 0.0, 0.0], "radius": r, "height": 2.2 * r}
    return {"type": "Box", "center": [0.0, 0.0, 0.0], "x": 2.0 * r, "y": 1.6 * r, "z": 1.2 * r}


def _bounds_for(shape):
    """A conservative bounding sphere -- deriveBounds()'s own logic (viewer.js), inlined
    here so the generated scene ships explicit bounds like the reference fixtures do,
    rather than depending on the viewer to derive them."""
    c = shape["center"]
    if shape["type"] == "Sphere":
        r = shape["radius"]
    elif shape["type"] in ("Cylinder", "Cone"):
        r = math.hypot(shape["radius"], shape["height"] * 0.5)
    else:  # Box
        r = math.sqrt(shape["x"] ** 2 + shape["y"] ** 2 + shape["z"] ** 2) * 0.5
    return {"center": list(c), "radius": r}


def build_scene(n_leaves, leaves_per_body=1):
    if n_leaves < 1:
        raise ValueError("n_leaves must be >= 1")
    if leaves_per_body < 1:
        raise ValueError("leaves_per_body must be >= 1")
    n_bodies = math.ceil(n_leaves / leaves_per_body)

    bodies = []
    csg_leaves = []
    for b in range(n_bodies):
        body_x = (b - (n_bodies - 1) / 2.0) * _BODY_SPACING_M
        leaves_here = min(leaves_per_body, n_leaves - b * leaves_per_body)
        # a body's own bounding radius: worst-case leaf offset + its own bound
        bodies.append({"pivot": [0, 0, 0],
                       "label": f"probe_{b}",
                       "radius_m": 0.5 + _LEAF_LOCAL_SPACING_M * leaves_here})
        for j in range(leaves_here):
            i = b * leaves_per_body + j
            shape = _shape_for(i)
            # stack this body's own leaves along local +z so none overlap
            shape["center"] = [0.0, 0.0, j * _LEAF_LOCAL_SPACING_M]
            csg_leaves.append({
                "op": "add",
                "material": _MATERIAL,
                "shape": shape,
                "bounds": _bounds_for(shape),
                "body": b,
            })

    materials = {_MATERIAL: {
        "color_rgb": [0.72, 0.72, 0.72], "emergent": False, "metal": False,
        "mechanism": "no model yet", "note": "leaf-count sweep probe -- not a real material",
        "opacity": 1.0, "density_kg_m3": 2700.0, "density_method": "unknown material",
    }}

    half_span = (n_bodies - 1) / 2.0 * _BODY_SPACING_M + _BODY_SPACING_M
    z_span = leaves_per_body * _LEAF_LOCAL_SPACING_M + 1.0
    bbox = [[-half_span, half_span], [-2.0, 2.0], [-1.0, z_span]]
    orbit_radius = math.hypot(half_span, z_span) * 1.6

    frame_bodies = [{"pos": [(b - (n_bodies - 1) / 2.0) * _BODY_SPACING_M, 0.0, 0.0],
                      "quat": [0, 0, 0, 1]} for b in range(n_bodies)]

    scene = {
        "name": f"mpdr-leafsweep-n{n_leaves}",
        "bodies": bodies,
        "csg_leaves": csg_leaves,
        "materials": materials,
        "physics": {"mass_kg": 0, "com_m": [0, 0, 0], "inertia_kgm2": [0, 0, 0]},
        "bbox": bbox,
        "camera": {"target": [0.0, 0.0, 0.0], "orbit_radius": orbit_radius,
                   "fov_deg": 40.0, "up": [0.0, 0.0, 1.0]},
        "identified": False,
        "ambient": {"sky": [0.05, 0.05, 0.07], "ground": [0.02, 0.02, 0.02], "up": [0, 0, 1]},
        "lights": [{"dir": [-0.384, 0.432, -0.816], "color": [1.0, 0.9313, 0.8716],
                    "intensity": 1.05, "temperature_k": 5500}],
        "source": ("tools/gen_leaf_stress_scene.py -- deterministic synthetic scene for the "
                    "CSG leaf-count shader compile-time sweep (R4.7); analytic Sphere/Cylinder/"
                    "Cone/Box primitives only, op:add, one shared material -- not a real game "
                    "layout, never gallery-indexed (see build_scene_index._EVIDENCE_SCENE)"),
    }
    trajectory = {
        "frames": [{"t_sim": 0.0, "bodies": frame_bodies,
                    "energy_j": 0.0, "momentum": [0, 0, 0], "angular_momentum": [0, 0, 0]}],
        "t_end_s": 0.0, "natural_timescale_s": 0.0, "suggested_rate": 1.0,
        "body_labels": [b["label"] for b in bodies],
        "validation": {"note": f"leaf-count-sweep probe scene, n={n_leaves}, "
                                f"leaves_per_body={leaves_per_body}"},
    }
    return {"kind": "trajectory", "scene": scene, "trajectory": trajectory, "sim": None}


def _parse_n_list(s):
    return [int(tok) for tok in s.split(",") if tok.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=_parse_n_list,
                     default=[4, 8, 11, 16, 22, 32, 44, 64, 88, 128, 176],
                     help="comma-separated leaf counts to generate (default: the standard sweep ladder)")
    ap.add_argument("--leaves-per-body", type=int, default=1,
                     help="leaves per body (default 1: isolates per-leaf SDF cost from the "
                          "per-body bodyDecls qrotInv line)")
    ap.add_argument("--out-dir", default=DATA_DIR,
                     help="output directory (default: sigma_ground/radiance/web/data)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    written = []
    for n in args.n:
        scene_doc = build_scene(n, args.leaves_per_body)
        path = os.path.join(args.out_dir, f"mpdr_leafsweep_n{n}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(scene_doc, f)
        written.append((n, path, len(scene_doc["scene"]["csg_leaves"])))
        print(f"n={n:>4}  leaves={written[-1][2]:>4}  bodies={len(scene_doc['scene']['bodies']):>3}  -> {path}")

    print(f"\n{len(written)} scene(s) written to {args.out_dir}")
    print("(regenerable, not committed -- see .gitignore's R4.7 leafsweep stanza)")


if __name__ == "__main__":
    main()
