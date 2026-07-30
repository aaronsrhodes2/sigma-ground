"""Generate the direct-SDF voxel-bridge demo scenes (Phase D2/D3).

Unlike build_voxel_gallery.py (real ShapeNet/PartNet meshes, voxelized),
these objects are compiled through Deckard's PRIMITIVE kit as usual and then
SAMPLED onto a dense voxel grid straight from their analytic SDF
(deckard.sdf_voxelize, riding Phase D1's vectorized sdf_batch) -- no mesh
ever exists. The flagship is the doughnut: its ~1mm sprinkles are exactly the
detail a fixed ~3.5mm pitch would erase, and the bridge's adaptive pitch
(sub-0.5mm here) must keep every one of the 24 individually resolved.

Static scenes (no physics drop): the point being demonstrated is the
geometry fidelity of the sampled grid, so each object is exported exactly
where it was compiled, through the UNMODIFIED construct_to_scene() path.
Writes a separate data/voxel_sdf_scenes.json index so it does NOT clobber
the curated scenes.json gallery (same convention as voxel_scenes.json).

Run:  python -m sigma_ground.radiance.web.build_voxel_sdf_demo
"""
import json
import os
import sys

sys.path.insert(0, r"D:\Aaron\development\sigma-ground")

from sigma_ground import deckard
from sigma_ground.deckard.sdf_voxelize import construct_to_voxel_fidelity
from sigma_ground.radiance.scene_export import construct_to_scene

DATA = os.path.join(os.path.dirname(__file__), "data")

# (deckard name, slug) -- all resolve offline (catalog/archetype), no LLM.
OBJECTS = [
    ("doughnut",   "voxsdf_doughnut"),
    ("coffee cup", "voxsdf_coffee_cup"),
]


def build():
    scenes = []
    for name, slug in OBJECTS:
        c = deckard.identify(name, allow_llm=False)
        wrapped = construct_to_voxel_fidelity(c)
        report = wrapped.validation["voxel_bridge"]
        scene = construct_to_scene(wrapped)
        # the voxel is a sampled approximation: the exact-match geometry
        # self-check doesn't apply (same convention as build_voxel_gallery).
        scene.pop("sdf_samples", None)
        path = os.path.join(DATA, slug + ".json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(scene, fh)
        kb = os.path.getsize(path) // 1024
        gd = report["grid_dims"]
        scenes.append({"slug": slug,
                       "title": f"{name} — direct-SDF voxel bridge "
                                f"({gd[0]}x{gd[1]}x{gd[2]} @ "
                                f"{report['pitch_m']*1000:.3f}mm)",
                       "question": f"show {name} as SDF-sampled voxels",
                       "verb": "voxel_sdf", "group": "voxel_sdf",
                       "kind": "static", "shape": "voxel",
                       "dims": list(gd),
                       "material": report.get("dominant_material")})
        print(f"  + {slug:20s} {name:12s} mass={c.mass_kg:8.4f}kg  "
              f"grid {gd[0]}x{gd[1]}x{gd[2]} @ {report['pitch_m']*1000:.4f}mm  "
              f"{report['occupied_cells']:,} occupied  {kb}KB")
    with open(os.path.join(DATA, "voxel_sdf_scenes.json"), "w", encoding="utf-8") as fh:
        json.dump(scenes, fh, indent=1)
    print(f"\nSDF-voxel demo: {len(scenes)} scenes -> data/voxel_sdf_scenes.json")


if __name__ == "__main__":
    build()
