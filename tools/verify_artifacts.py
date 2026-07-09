"""Verify viewer artifacts — every bundle is self-consistent and renderable.

For each ``radiance/web/data/*.json`` (or the slugs given on the command line):

  * every leaf's shape tree uses ONLY viewer-supported node types
    (``scene_export.SUPPORTED_SHAPE_TYPES`` — the single source of truth);
  * every leaf's material exists with a 3-vector color;
  * the stored ``sdf_samples`` REPLAY through ``scene_spec_to_sdf`` — the same
    rebuild the browser performs — and agree to 1e-6 (proves the artifact
    draws the geometry the physics weighed);
  * trajectories: strictly increasing ``t_sim``, unit quats, valid body
    indices, positive ``suggested_rate``, and every recorded contact
    dissipates energy.

Exit non-zero with a per-slug table on any failure.

    python tools/verify_artifacts.py [slug ...]
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

sys.path.insert(0, r"D:\Aaron\development\sigma-ground")

from sigma_ground.radiance.scene_export import (SUPPORTED_SHAPE_TYPES,   # noqa: E402
                                                scene_spec_to_sdf,
                                                decode_field_keyframe,
                                                field_trilinear)

_DATA = (pathlib.Path(__file__).resolve().parents[1]
         / "sigma_ground" / "radiance" / "web" / "data")


class _P:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


def _shape_types(node, out):
    out.add(node.get("type", "?"))
    for k in ("shape", "cut"):
        if isinstance(node.get(k), dict):
            _shape_types(node[k], out)
    return out


def verify(path: pathlib.Path) -> list:
    """All problems found in one artifact (empty = good)."""
    probs = []
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return [f"unreadable JSON: {e}"]
    if isinstance(doc, list):
        return []              # an index manifest (scenes.json / voxel_scenes.json
                               # family) — a gallery listing, not an artifact
    scene = doc.get("scene") or doc                       # statics are bare scenes
    leaves = scene.get("csg_leaves") or []
    if not leaves:
        return ["no csg_leaves"]
    mats = scene.get("materials") or {}
    for i, leaf in enumerate(leaves):
        types = _shape_types(leaf.get("shape") or {}, set())
        bad = types - SUPPORTED_SHAPE_TYPES
        if bad:
            probs.append(f"leaf {i}: unsupported shape type(s) {sorted(bad)}")
        m = mats.get(leaf.get("material"))
        if not (isinstance(m, dict) and isinstance(m.get("color_rgb"), list)
                and len(m["color_rgb"]) == 3):
            probs.append(f"leaf {i}: material {leaf.get('material')!r} missing/colorless")
    samples = scene.get("sdf_samples") or []
    if samples:
        try:
            sdf, _ = scene_spec_to_sdf(scene)
            worst = max(abs(sdf(_P(*s["p"])) - s["d"]) for s in samples)
            if worst > 1e-6:
                probs.append(f"sdf_samples replay off by {worst:.3g} (> 1e-6)")
        except Exception as e:
            probs.append(f"sdf rebuild failed: {type(e).__name__}: {e}")
    # ── per-cell fields: schema + the field_samples replay (stdlib trilinear) ──
    field_leaves = [i for i, l in enumerate(leaves)
                    if (l.get("fields") or {}).get("temperature_k")]
    for i in field_leaves:
        f = leaves[i]["fields"]["temperature_k"]
        g = f.get("grid") or {}
        dims = g.get("dims") or []
        if f.get("encoding") != "u8-xfast":
            probs.append(f"leaf {i}: field encoding {f.get('encoding')!r}")
            continue
        if (len(dims) != 3 or any(not isinstance(d, int) or d < 1 for d in dims)
                or not (g.get("voxel_size") or 0) > 0):
            probs.append(f"leaf {i}: bad field grid {g!r}")
            continue
        if not f.get("t_min", 0) < f.get("t_max", 0):
            probs.append(f"leaf {i}: field t_min/t_max not ordered")
        kts = [k.get("t_sim") for k in f.get("keyframes") or []]
        if not kts or any(b <= a for a, b in zip(kts, kts[1:])):
            probs.append(f"leaf {i}: field keyframes empty/non-increasing")
        for k in range(len(kts)):
            try:
                decode_field_keyframe(f, k)       # length == nx·ny·nz enforced
            except Exception as e:
                probs.append(f"leaf {i} keyframe {k}: {e}")
                break
    fsamples = scene.get("field_samples") or []
    if field_leaves and not fsamples:
        probs.append("leaf carries fields but scene ships no field_samples "
                     "(the not-faked check is mandatory)")
    if fsamples:
        raws = {}
        worst_t = 0.0
        for s in fsamples:
            li, k = s["leaf"], s["k"]
            f = (leaves[li].get("fields") or {}).get(s.get("quantity",
                                                           "temperature_k"))
            if not f:
                probs.append(f"field_samples points at leaf {li} with no field")
                break
            raw = raws.setdefault((li, k), decode_field_keyframe(f, k))
            worst_t = max(worst_t, abs(field_trilinear(raw, f, s["p"]) - s["T"]))
        if worst_t > 1e-3:
            probs.append(f"field_samples replay off by {worst_t:.3g} K (> 1e-3)")
    tr = doc.get("trajectory")
    if tr:
        frames = tr.get("frames") or []
        if len(frames) < 2:
            probs.append("trajectory with <2 frames")
        ts = [f.get("t_sim", 0) for f in frames]
        if any(b <= a for a, b in zip(ts, ts[1:])):
            probs.append("t_sim not strictly increasing")
        nb = len(scene.get("bodies") or [])
        for f in frames[:: max(1, len(frames) // 20)]:
            for b in f.get("bodies") or []:
                q = b.get("quat") or []
                if len(q) != 4 or abs(math.sqrt(sum(v * v for v in q)) - 1) > 1e-3:
                    probs.append("non-unit quat in frames")
                    break
                # optional per-frame body temperature (the thermal contract):
                # when present it must be a finite non-negative kelvin value
                t_k = b.get("temperature_k")
                if t_k is not None and not (
                        isinstance(t_k, (int, float))
                        and math.isfinite(t_k) and t_k >= 0.0):
                    probs.append(f"bad frame temperature_k: {t_k!r}")
                    break
            if len(f.get("bodies") or []) != nb and nb:
                probs.append(f"frame body count != scene bodies ({nb})")
                break
        if not (tr.get("suggested_rate") or 0) > 0:
            probs.append("suggested_rate missing/non-positive")
        contacts = ((tr.get("validation") or {}).get("contacts")) or []
        if any(c["E_after"] >= c["E_before"] for c in contacts):
            probs.append("a contact GAINED energy")
    return probs


def main(argv: list) -> None:
    paths = ([_DATA / f"{s}.json" for s in argv] if argv
             else sorted(_DATA.glob("*.json")))
    paths = [p for p in paths if p.name != "scenes.json"]
    bad = 0
    for p in paths:
        probs = verify(p)
        mark = "ok " if not probs else "BAD"
        print(f"  [{mark}] {p.stem}")
        for pr in probs:
            print(f"        - {pr}")
        bad += bool(probs)
    print(f"\n{len(paths) - bad}/{len(paths)} artifacts verified")
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
