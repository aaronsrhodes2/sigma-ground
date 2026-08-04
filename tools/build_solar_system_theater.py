"""Build the solar-system voxel theater — the real DE440 ephemeris as the
base simulation theater, rendered natively voxel.

    python tools/build_solar_system_theater.py

Ground truth, not invented: body positions/velocities are the real J2026
epoch DE440 state vectors already on disk
(field/interface/fixtures/de440_state_vectors.json, SSB-centered, JPL
Horizons-sourced). Radii are cited NASA/IAU constants
(field/interface/fixtures/body_physical_params.json). Colors are disclosed
render-only Choices (representative, not measured spectral albedo).

Scope decision (disclosed, not silent): the WebGL2 raymarcher's per-object
voxel path binds one 3-D texture per Voxel leaf, and the browser's texture-
unit budget caps that at roughly 15 before other bindings (fields, PT
accumulation) are even counted (found this session auditing the renderer).
30 real voxel bodies would blow that ceiling outright — the actual fix
(one unified scene-level voxel grid, "D4" in the render-at-scale plan) is
large, multi-session work, not attempted here. Tonight's honest scope:
the Sun + 8 planets are voxelized for real (9 Voxel leaves, comfortably
inside budget); the 21 moons render as analytic Sphere leaves — real
position, real size, just not volumetrically textured. Nothing here is
faked; this is a disclosed rendering-fidelity Choice, not a physics one.
"""
import base64
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

FIX_DIR = os.path.join(os.path.dirname(__file__), "..", "sigma_ground",
                       "field", "interface", "fixtures")
DE440_PATH = os.path.join(FIX_DIR, "de440_state_vectors.json")
PARAMS_PATH = os.path.join(FIX_DIR, "body_physical_params.json")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "sigma_ground",
                        "radiance", "web", "data", "solar_system_theater.json")

EPOCH = "j2026"
VOXEL_PLANETS = {"Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter",
                 "Saturn", "Uranus", "Neptune"}
VOXEL_CELLS_PER_DIAMETER = 48   # fixed cells-per-diameter LOD policy — see
# the R1-lite/"min-max voxel resolution as we zoom" note in
# radiance/web/viewer.js: this governs DATA resolution (how fine the grid
# is), independent of and complementary to the render-time epsilon scaling
# that already answers zoom-time precision. True per-distance streaming
# (loading a coarser/finer grid as the camera approaches) is D4/R5 future
# work — every body here ships at one fixed, honest resolution.


def _voxel_sphere(radius_m, cells=VOXEL_CELLS_PER_DIAMETER):
    """A real Voxel shape (kernel.voxel.Voxel): an analytically-sampled
    signed-distance grid of a sphere, body-LOCAL (centered at 0,0,0) — the
    world placement happens per-frame via the body transform, never baked
    here. Cheap: pure numpy, one broadcasted distance field."""
    from sigma_ground.kernel.voxel import Voxel
    margin = 1.15                          # a little air around the surface
    n = cells + 2
    vs = (2.0 * radius_m * margin) / n
    lo = -(n * vs) / 2.0
    ax = lo + (np.arange(n) + 0.5) * vs
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    sdf = np.sqrt(X * X + Y * Y + Z * Z) - radius_m
    return Voxel(sdf.astype(np.float32), vs, center=(0.0, 0.0, 0.0),
                volume_m3=(4.0 / 3.0) * math.pi * radius_m ** 3,
                source="analytic sphere SDF, sampled to a body-local grid "
                       "(NOT a Deckard/PartNet mesh voxelization)")


def _shape_to_dict_sphere(radius_m):
    return {"type": "Sphere", "center": [0.0, 0.0, 0.0], "radius": radius_m}


def main():
    de440 = json.load(open(DE440_PATH, encoding="utf-8"))
    params = json.load(open(PARAMS_PATH, encoding="utf-8"))["bodies"]
    snap = de440["snapshots"][EPOCH]
    print(f"epoch: {snap['epoch']['iso']} (JD {snap['epoch']['jd_tdb']})")

    bodies_meta, csg_leaves, materials = [], [], {}
    frame_bodies = []
    voxel_count = 0

    for i, b in enumerate(snap["bodies"]):
        name = b["name"]
        p = params.get(name)
        if p is None:
            raise ValueError(f"no physical params for {name!r} — "
                             f"body_physical_params.json is incomplete")
        r_m = p["radius_m"]
        sv = b["state_vector"]
        pos_m = [sv["x_km"] * 1000.0, sv["y_km"] * 1000.0, sv["z_km"] * 1000.0]

        bodies_meta.append({"pivot": [0, 0, 0], "label": name, "radius_m": r_m})
        frame_bodies.append({"pos": pos_m, "quat": [0.0, 0.0, 0.0, 1.0]})

        mat_label = f"body_{name}"
        is_voxel = name in VOXEL_PLANETS
        if is_voxel:
            vox = _voxel_sphere(r_m)
            from sigma_ground.radiance.scene_export import _shape_to_dict
            shape_dict = _shape_to_dict(vox)
            voxel_count += 1
            bounds = {"center": [0.0, 0.0, 0.0], "radius": r_m * 1.15}
        else:
            shape_dict = _shape_to_dict_sphere(r_m)
            bounds = {"center": [0.0, 0.0, 0.0], "radius": r_m}
        csg_leaves.append({"op": "add", "material": mat_label, "body": i,
                           "shape": shape_dict, "bounds": bounds,
                           "temperature_k": p["temperature_k"]})

        rgb = p["visual_rgb"]
        mat = {"color_rgb": rgb, "emergent": False, "metal": False,
              "opacity": 1.0, "roughness": 0.6,
              "temperature_k": p["temperature_k"],
              "temp_provenance": p["temp_provenance"]}
        materials[mat_label] = mat
        print(f"  {name:10} r={r_m:12,.0f} m  pos=({pos_m[0]:.3e},"
             f"{pos_m[1]:.3e},{pos_m[2]:.3e}) m  {'VOXEL' if is_voxel else 'sphere'}")

    print(f"\n{voxel_count} voxelized bodies (texture-unit budget ~15), "
         f"{len(snap['bodies']) - voxel_count} analytic-sphere moons")

    # camera: framed on the inner system by default (Mercury..Mars comfortably
    # in view); R1-lite's camera-anchored rebase + view-scale epsilons make
    # zooming out to the outer planets precision-safe, not just possible
    au = 1.495978707e11
    scene_spec = {
        "name": f"Solar System Theater — real DE440 ephemeris @ {snap['epoch']['iso']}",
        "bodies": bodies_meta,
        "csg_leaves": csg_leaves,
        "materials": materials,
        "physics": {"mass_kg": sum(b["mass_kg"] for b in snap["bodies"]),
                    "com_m": [0, 0, 0], "inertia_kgm2": [0, 0, 0]},
        "bbox": [[-31 * au, 31 * au], [-31 * au, 31 * au], [-31 * au, 31 * au]],
        "camera": {"target": [0.0, 0.0, 0.0], "orbit_radius": 2.2 * au,
                  "fov_deg": 45.0, "up": [0.0, 0.0, 1.0],
                  "az0": 0.6, "el0": 0.5, "follow": False},
        "identified": True,
        "pt_env": 0.0,          # space: no ambient sky-dome (deep_space theater
                                # convention) — the Sun's own NEE emission is
                                # the only light source, as it physically is
        "ambient": {"sky": [0.0, 0.0, 0.0], "ground": [0.0, 0.0, 0.0],
                   "up": [0.0, 0.0, 1.0]},
        # NEE exposure Choice (render-only, not physics): incandescence()
        # already SATURATES the Reinhard tone-map at the Sun's 5778 K (its
        # pre-falloff value is ~1.7, the display ceiling) — the darkness is
        # entirely the multiplicative geometric solid-angle term
        # (2*(1-cosTm) ~ 2.2e-5 sr at 1 AU) plus ndl2/albedo/throughput
        # bringing L down to ~0.004-0.01 on a 0-1-ish HDR scale. Measured
        # empirically against this scene (not derived from first
        # principles — a disclosed render/exposure calibration, the same
        # kind of hand-tuned constant EMISSION_SCALE and the 1.7 Reinhard
        # gain already are, flagged NON-DERIVED in viewer.js): ~4e4 brings
        # a directly-sunlit point to a comfortably visible mid-tone.
        "nee_exposure": 6.0e3,
        "plugs": [],
        "choices": [
            {"description": "body render colors (visual_rgb)",
             "value": "representative, not measured spectral albedo",
             "adjustable": False},
            {"description": "planetary surface/cloud-top temperatures for "
                            "non-measured bodies",
             "value": "rough order-of-magnitude equilibrium estimate, no "
                      "latitude/season/diurnal model",
             "adjustable": False},
            {"description": "voxel fidelity scope",
             "value": f"Sun + 8 planets voxelized ({VOXEL_CELLS_PER_DIAMETER} "
                      f"cells/diameter); 21 moons render as analytic spheres — "
                      f"texture-unit budget, not a physics limitation",
             "adjustable": False},
            {"description": "NEE exposure multiplier (render-only tone-map "
                            "calibration for real AU-scale emitter distances, "
                            "not a physics quantity)",
             "value": 6.0e3, "adjustable": True},
        ],
        "source": (f"sigma_ground/field/interface/fixtures/de440_state_vectors.json "
                  f"(JPL Horizons DE440/DE441, {snap['epoch']['iso']}, SSB-centered "
                  f"ICRF/J2000 equatorial) + body_physical_params.json (NASA/IAU "
                  f"radii). Positions and masses are Facts; colors and non-"
                  f"measured temperatures are disclosed Choices; the sphere "
                  f"shape itself is a SIMPLIFIED_MODEL (real bodies are "
                  f"oblate/irregular; J2 data exists in nbody.py but isn't "
                  f"reflected in this render geometry)."),
    }
    trajectory = {
        "frames": [{"t_sim": 0.0, "bodies": frame_bodies, "energy_j": 0.0,
                   "momentum": [0.0, 0.0, 0.0],
                   "angular_momentum": [0.0, 0.0, 0.0]}],
        "t_end_s": 0.0,
        "natural_timescale_s": 0.0,
        "suggested_rate": 1.0,
        "body_labels": [b["label"] for b in bodies_meta],
        "validation": {"note": "static single-frame snapshot at the real "
                               "epoch above — not yet stepped through the "
                               "orbital Region (Phase E5); positions are "
                               "real, motion is not animated here"},
    }
    bundle = {"kind": "trajectory", "scene": scene_spec,
             "trajectory": trajectory, "sim": None}

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(bundle, f)
    size_mb = os.path.getsize(OUT_PATH) / 1e6
    print(f"\n-> {os.path.normpath(OUT_PATH)}  ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
