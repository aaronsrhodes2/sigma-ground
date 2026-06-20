"""Generate the VOXEL object gallery — real ShapeNet/PartNet shapes, not primitives.

Each object is identified at voxel fidelity (its real per-part mesh, voxelized)
and dropped through Materia's drag + ground-contact, producing data/<slug>.json
that the viewer raymarches as a WebGL2 3-D texture — a real chair/mug/bowl, not a
rod-and-box. ONLY objects ShapeNet actually contains are here: there is no
hammer/anvil/skillet/feather in the dataset, so those stay on the primitive path
(we never fake a mesh). Writes a separate data/voxel_scenes.json index so it does
NOT clobber the curated scenes.json gallery.

Run:  python -m sigma_ground.radiance.web.build_voxel_gallery
"""
import json
import os
import sys

sys.path.insert(0, r"D:\Aaron\development\sigma-ground")

from sigma_ground import deckard
from sigma_ground.radiance import record_object_fall

DATA = os.path.join(os.path.dirname(__file__), "data")

# (deckard name, slug, drop height m) — names that resolve to a real PartNet
# category (verified via research.voxel_plan). Heights kept small so the object
# is visible falling onto the (black = empty) stage.
OBJECTS = [
    ("a wooden chair", "vox_chair",  2.5),
    ("coffee mug",     "vox_mug",    1.2),
    ("bowl",           "vox_bowl",   1.2),
    ("bottle",         "vox_bottle", 1.6),
    ("vase",           "vox_vase",   1.6),
    ("lamp",           "vox_lamp",   1.8),
    ("knife",          "vox_knife",  1.4),
    ("a wooden table", "vox_table",  2.5),
]


def build():
    scenes, skipped = [], []
    for name, slug, alt in OBJECTS:
        try:
            c = deckard.identify(name, fidelity="voxel")
            if c.validation.get("mode") != "voxel":
                skipped.append((name, f"no mesh (mode={c.validation.get('mode')})"))
                print(f"  ! {name}: no voxel mesh — skipped")
                continue
            bundle = record_object_fall(c, alt)
            scene = bundle["scene"]
            # the voxel is a sampled approximation: the exact-match geometry
            # self-check doesn't apply, so drop the samples (viewer shows
            # "(no samples)" rather than a spurious FAIL on the quantized grid).
            scene.pop("sdf_samples", None)
            out = {"scene": scene, "trajectory": bundle["trajectory"],
                   "kind": "trajectory"}
            with open(os.path.join(DATA, slug + ".json"), "w", encoding="utf-8") as fh:
                json.dump(out, fh)
            leaf = scene["csg_leaves"][0]["shape"]
            kb = os.path.getsize(os.path.join(DATA, slug + ".json")) // 1024
            scenes.append({"slug": slug, "title": f"{name} — dropped from {alt:g} m",
                           "question": f"drop a {name} from {alt:g} m",
                           "verb": "drop_object", "group": "voxel",
                           "kind": "trajectory", "shape": "voxel",
                           "dims": leaf["dims"], "material": scene["csg_leaves"][0]["material"]})
            print(f"  + {slug:11s} {name:16s} mass={c.mass_kg:6.1f}kg  "
                  f"voxel {leaf['dims']}  {kb}KB")
        except Exception as e:
            skipped.append((name, str(e)))
            print(f"  ! {name}: {type(e).__name__}: {e}")
    with open(os.path.join(DATA, "voxel_scenes.json"), "w", encoding="utf-8") as fh:
        json.dump(scenes, fh, indent=1)
    print(f"\nVoxel gallery: {len(scenes)} real-shape drops -> data/voxel_scenes.json"
          + (f"  ({len(skipped)} skipped)" if skipped else ""))
    for n, why in skipped:
        print(f"   skipped {n}: {why[:80]}")


if __name__ == "__main__":
    build()
