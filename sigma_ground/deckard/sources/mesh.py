"""Load a real model's per-part triangle meshes (opt-in trimesh).

The exemplar pools tell us WHICH real model (`anno_id`) a name resolves to; this
loads that model's actual PartNet per-part OBJ meshes off the local 242 GB so
Deckard can voxelize the real geometry. ToU: meshes are read locally and never
shipped — only distilled aggregates (the bbox exemplars) are committed.

trimesh is imported lazily (inside the function) so importing `deckard` stays
clean for the analytic-primitive core that doesn't need the [shapes] extra.
"""
from __future__ import annotations

import pathlib

# The locally-extracted PartNet data (per tools/distill_partnet.py).
_DATA = pathlib.Path("D:/Aaron/datasets/shapenet/PartNet/data_v0")


def data_available(anno_id=None) -> bool:
    """True if the PartNet mesh data (optionally a specific model) is on disk."""
    root = _DATA if anno_id is None else _DATA / str(anno_id) / "objs"
    return root.is_dir()


def load_parts(anno_id, *, scale_to_m: float | None = None, to_z_up: bool = True):
    """Return ``[(trimesh.Trimesh, obj_stem), ...]`` for a PartNet anno_id (or []).

    If ``scale_to_m`` is given, every part is uniformly scaled so the whole
    model's longest bbox edge equals that real-world size (from ShapeNetSem).

    PartNet/ShapeNet ship in a **Y-up** canonical frame; this project is **Z-up**
    (gravity along −z, floor at z=0). With ``to_z_up`` (default), each part is
    rotated +90° about X (Y→Z) so the loaded mesh is already in the project frame
    — the voxel Construct then stands up for both physics and the z-up renderer.
    """
    try:
        import trimesh
    except Exception:
        return []
    d = _DATA / str(anno_id) / "objs"
    if not d.is_dir():
        return []
    parts = []
    for f in sorted(d.glob("*.obj")):
        try:
            m = trimesh.load(f, force="mesh")
        except Exception:
            continue
        if len(getattr(m, "faces", [])):
            parts.append((m, f.stem))
    if not parts:
        return []
    if to_z_up:
        import math

        import trimesh
        R = trimesh.transformations.rotation_matrix(math.pi / 2.0, [1.0, 0.0, 0.0])
        for m, _ in parts:
            m.apply_transform(R)
    if scale_to_m and scale_to_m > 0:
        import trimesh
        full = trimesh.util.concatenate([m for m, _ in parts])
        longest = float((full.bounds[1] - full.bounds[0]).max())
        if longest > 0:
            s = scale_to_m / longest
            for m, _ in parts:
                m.apply_scale(s)
    return parts


__all__ = ["load_parts", "data_available"]
