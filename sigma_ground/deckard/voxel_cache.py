"""Voxel bake cache — voxelize once, identify instantly ever after.

``identify(fidelity="voxel")`` used to re-voxelize the mesh on every call
(~seconds per object). This caches the finished ``VoxelField`` — grids AND the
part tables — under ``local-cache/voxels/<anno>_<pitch>mm_<VERSION>.npz``
(gitignored, regenerable). The cache key carries a format VERSION so any
change to the field schema invalidates stale bakes instead of half-loading
them; a cache hit reconstructs a field that is ARRAY-EQUAL to a fresh bake
(gated by test).

numpy is an opt-in [shapes] dep here exactly as in voxelize.py — imported
lazily, and every cache miss (absent, wrong version, unreadable) returns None
so the caller just re-voxelizes. The cache can never make anything wrong,
only faster.
"""
from __future__ import annotations

import json
import pathlib

_VERSION = "v2parts"                     # bump when VoxelField's schema changes
_ROOT = pathlib.Path(__file__).resolve().parents[2] / "local-cache" / "voxels"


def _path(anno_id, pitch) -> pathlib.Path:
    return _ROOT / f"{anno_id}_{pitch * 1000:.3f}mm_{_VERSION}.npz"


def save_cached_field(anno_id, pitch, field) -> None:
    """Persist a VoxelField (best-effort — a failed save just means no cache)."""
    import numpy as np
    try:
        _ROOT.mkdir(parents=True, exist_ok=True)
        meta = {
            "voxel_size": field.voxel_size,
            "center": list(field.center),
            "materials": list(field.materials),
            "volume_m3": field.volume_m3,
            "mass_kg": field.mass_kg,
            "com_m": list(field.com_m),
            "inertia_kgm2": list(field.inertia_kgm2),
            "interfaces": [[list(k), v] for k, v in field.interfaces.items()],
            "free_surfaces": dict(field.free_surfaces),
            "watertight_frac": field.watertight_frac,
            "confidence": field.confidence,
            "density_by_label": dict(field.density_by_label or {}),
            "notes": field.notes,
            "reconciliation": field.reconciliation,
            "parts": field.parts,
            "part_of_label": field.part_of_label,
            "part_interfaces": ([[list(k), v] for k, v in
                                 field.part_interfaces.items()]
                                if field.part_interfaces is not None else None),
        }
        np.savez_compressed(
            _path(anno_id, pitch),
            sdf_grid=np.asarray(field.sdf_grid, dtype=np.float32),
            label_grid=np.asarray(field.label_grid, dtype=np.int32),
            meta=np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8))
    except Exception:
        pass                                          # cache is an optimization only


def load_cached_field(anno_id, pitch):
    """The cached VoxelField, or None (absent / stale version / unreadable)."""
    import numpy as np
    p = _path(anno_id, pitch)
    if not p.exists():
        return None
    try:
        from .voxelize import VoxelField
        with np.load(p) as z:
            meta = json.loads(bytes(z["meta"]).decode("utf-8"))
            sdf = z["sdf_grid"].astype(float)
            label = z["label_grid"]
        parts = meta["parts"]
        if parts is not None:                          # tuples don't survive JSON
            parts = [{**q, "labels": tuple(q["labels"]),
                      "com_m": tuple(q["com_m"]),
                      "inertia_kgm2": tuple(q["inertia_kgm2"]),
                      "flags": tuple(q["flags"])} for q in parts]
        pif = meta["part_interfaces"]
        if pif is not None:
            pif = {tuple(k): {**v, "centroid_m": tuple(v["centroid_m"]),
                              "principal_dir": tuple(v["principal_dir"])}
                   for k, v in pif}
        return VoxelField(
            sdf_grid=sdf, label_grid=label,
            voxel_size=float(meta["voxel_size"]), center=tuple(meta["center"]),
            materials=list(meta["materials"]), volume_m3=meta["volume_m3"],
            mass_kg=meta["mass_kg"], com_m=tuple(meta["com_m"]),
            inertia_kgm2=tuple(meta["inertia_kgm2"]),
            interfaces={tuple(k): v for k, v in meta["interfaces"]},
            free_surfaces=dict(meta["free_surfaces"]),
            watertight_frac=meta["watertight_frac"],
            confidence=meta["confidence"],
            density_by_label=dict(meta["density_by_label"]),
            notes=meta["notes"], reconciliation=meta["reconciliation"],
            parts=parts, part_of_label=meta["part_of_label"],
            part_interfaces=pif)
    except Exception:
        return None                                   # unreadable → re-voxelize


__all__ = ["load_cached_field", "save_cached_field"]
