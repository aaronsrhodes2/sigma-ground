"""Phase E → viewer: two frozen bundles per comparison, engine on the slider.

Each Phase E comparison (run with record_frames=True) exports as a PAIR of
sibling bundles sharing the existing variant-slider contract — sliding
between "n-body (pure gravity)" and "SPH fluid (pressure + viscosity)"
replays the SAME seeded matter through the other engine's recording. No
new viewer machinery: this is exactly the mechanism the push-speed sweep
uses, with the ENGINE as the swept variable instead of a force.

Display choices, disclosed on the scene:
  - particle display radius is exaggerated (a ~10 m grain is invisible at
    a 12 km annulus scale) — render-only, masses/physics untouched;
  - sim frame is z-up, the viewer is y-up: positions are mapped
    (x, y, z) → (x, z, y) so the ring lies in the viewer's ground plane.

Materials bake through the real pipeline (scene_export._bake_material),
so ice is ice — same emergent color machinery as every other scene.
"""
from __future__ import annotations

import math

from .scene_export import _bake_material


def _swap(p):
    """Sim z-up → viewer y-up."""
    return [p[0], p[2], p[1]]


def comparison_to_bundles(report, seed_bodies, *, slug_base, title_base,
                          material_key, display_radius_m,
                          display_max_particles=32):
    """Report (record_frames=True) + the seed bodies → {slug: bundle} pair.

    seed_bodies supplies per-body labels and masses (the report's frames
    are positions only). display_radius_m: render-only particle size —
    a disclosed Choice, stated in the scene notes. The central body (if
    the seed has one, label "central") keeps its REAL radius.

    display_max_particles: the raymarcher unrolls one SDF branch per CSG
    leaf, and a 101-leaf scene hangs the shader compile (found live
    2026-07-30 — the tab wedged). Until the D4 scene-level voxel grid
    lands, the viewer shows an evenly-spaced SUBSAMPLE of at most this
    many particles (central body always included), disclosed as a
    render-only Choice. The report's diagnostics always cover ALL bodies.
    """
    samples = [s for s in report["samples"] if "nbody_pos" in s]
    if not samples:
        raise ValueError("report carries no frames — run_comparison needs "
                         "record_frames=True for viewer export")
    n_all = len(seed_bodies)
    variants = [
        {"slug": f"{slug_base}_nbody", "label": "n-body (pure gravity)"},
        {"slug": f"{slug_base}_sph",
         "label": "SPH fluid (pressure + viscosity)"},
    ]

    # evenly-spaced display subsample (render-only; simulation used all)
    if n_all > display_max_particles:
        step = n_all / display_max_particles
        shown = sorted({0} | {min(n_all - 1, int(i * step))
                              for i in range(display_max_particles)})
    else:
        shown = list(range(n_all))

    # one leaf + one rigid body per DISPLAYED particle; identity rotation
    leaves, bodies_meta = [], []
    for k, i in enumerate(shown):
        b = seed_bodies[i]
        r_disp = b.radius_m if b.label == "central" else display_radius_m
        leaves.append({"op": "add", "material": material_key, "body": k,
                       "shape": {"type": "Sphere", "center": [0, 0, 0],
                                 "radius": r_disp}})
        bodies_meta.append({"pivot": [0, 0, 0], "label": b.label or f"p{i}"})

    ext = max(max(abs(c) for c in s["nbody_pos"][j])
              for s in samples for j in shown)
    materials = {material_key: _bake_material(material_key)}

    out = {}
    for key, variant in (("nbody_pos", variants[0]),
                         ("sph_pos", variants[1])):
        frames = []
        for s in samples:
            frames.append({
                "t_sim": s["t_s"],
                "bodies": [{"pos": _swap(s[key][j]),
                            "quat": [0.0, 0.0, 0.0, 1.0]}
                           for j in shown],
                "energy_j": s["nbody" if key == "nbody_pos" else "sph"]["e_total_j"],
            })
        t_end = frames[-1]["t_sim"]
        scene = {
            "name": f"{title_base} — {variant['label']}",
            "bodies": bodies_meta,
            "csg_leaves": leaves,
            "materials": materials,
            "physics": {"mass_kg": sum(b.mass_kg for b in seed_bodies),
                        "com_m": [0, 0, 0], "inertia_kgm2": [0, 0, 0]},
            "bbox": [[-ext, ext], [-ext, ext], [-ext, ext]],
            "camera": {"target": [0.0, 0.0, 0.0],
                       "orbit_radius": ext * 2.6, "fov_deg": 45.0,
                       "up": [0.0, 1.0, 0.0], "az0": 0.5, "el0": 0.55,
                       "follow": False},
            "identified": True,
            "source": "phase_e",
            "variants": variants,
            "variant_current": variant["slug"],
            "choices": list(report.get("choices", ())) + [
                {"description": ("particle DISPLAY radius (render-only "
                                 "exaggeration; masses/physics untouched)"),
                 "value": display_radius_m, "unit": "m"},
                {"description": ("displayed particles (render-only evenly-"
                                 "spaced subsample — the simulation and all "
                                 "report diagnostics used every body; the "
                                 "raymarcher's per-leaf shader unroll caps "
                                 "the view until D4's scene grid)"),
                 "value": {"shown": len(shown), "simulated": n_all}}],
            "plugs": [],
            "notes": (f"Phase E comparison — the variant slider swaps the "
                      f"ENGINE, not a force: same seeded matter, two "
                      f"descriptions. {report.get('notes', '')}"),
        }
        traj = {
            "frames": frames,
            "t_end_s": t_end,
            "natural_timescale_s": t_end,
            "suggested_rate": max(1.0, t_end / 20.0),
            "body_labels": [m["label"] for m in bodies_meta],
            "validation": {
                "engine": variant["label"],
                "final_residuals": samples[-1]["residual"],
            },
        }
        out[variant["slug"]] = {"kind": "trajectory", "scene": scene,
                                "trajectory": traj, "sim": None,
                                "phase_e": {"label": report.get("label", "")}}
    return out


def kepler_display_radius(r_outer_m):
    """A visible-but-honest default: ~1.5% of the outer scale."""
    return max(1.0, 0.015 * r_outer_m)


__all__ = ["comparison_to_bundles", "kepler_display_radius"]
