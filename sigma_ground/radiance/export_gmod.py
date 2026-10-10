"""Bundle -> Garry's Mod scenario exporter (the Mentat stage, Phase 0).

Turns a Radiance trajectory bundle (``radiance/web/data/<name>.json``: a SceneSpec
plus a Trajectory of per-frame poses, see ``trajectory.py``) into ONE Lua file that
the ``stage/gmod`` addon can replay inside Garry's Mod. Nothing is simulated here
or in Lua: the bundle is frozen sim output and the stage only displays it
(Platinum "frozen output" doctrine, same as the web viewer).

Tier 3 (radiance), standard library only, reads bundles and nothing above. The
scenario format ("mentat-scenario/1") is engine-neutral on purpose, so an s&box
player can consume the same files.

Usage::

    python -m sigma_ground.radiance.export_gmod fall_skillet
    python -m sigma_ground.radiance.export_gmod fall_skillet --out some/dir --report-only

What the exporter decides (each decision is verified, not assumed)
  * UP AXIS is per bundle: 53 viewer bundles are Y-up, 32 theater bundles are Z-up.
    Taken from ``physics_env.gravity_dir`` when present, else ``camera.up``, and
    cross-checked against the motion (a mismatch is a warning, never silent).
  * PIVOT: the viewer places a body-bound leaf at ``pos + R*(local - pivot)``
    (viewer.js ``qrotInv(q, p-pos) + pivot``). Shapes are exported with their
    centres relative to the pivot so the GMod entity origin IS the pivot.
  * FLOOR: the stage floor (unbound slab) height along up, so the floor can be
    laid on the GMod map floor.
  * SHAPES: Sphere / Box / Cylinder only in Phase 0. Subtract (CSG) leaves, leaves
    fully contained in a same-type sibling, and zero-size leaves are dropped and
    REPORTED. A body with no supported shape is an error, not a silent sphere.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

FORMAT = "mentat-scenario/1"
SUPPORTED_SHAPES = ("Sphere", "Box", "Cylinder")
DEFAULT_GRAVITY = 9.80665
DATA_DIR = Path(__file__).resolve().parent / "web" / "data"
DEFAULT_OUT = Path(__file__).resolve().parents[2] / "stage" / "gmod" / "addon" / "lua" / "mentat" / "scenarios"
_EPS = 1e-9

# First hint hit wins; the label is tried before the material (a "cast iron skillet"
# built from ceramic-named layers is still iron to a player).
_SURFACE_HINTS = (
    ("metal", ("iron", "steel", "copper", "alumin", "gold", "lead", "tungsten", "titanium",
               "brass", "bronze", "nickel", "silver", "anvil", "hammer", "skillet", "metal")),
    ("ceramic", ("ceramic", "stoneware", "porcelain", "glaze", "mug", "cup", "plate", "vase", "bowl")),
    ("glass", ("glass", "bottle")),
    ("wood", ("wood", "oak", "timber", "chair", "table")),
    ("concrete", ("concrete", "brick", "stone", "rock")),
    ("rubber", ("rubber",)),
)


class ExportError(ValueError):
    """The bundle cannot be exported faithfully (never silently degraded)."""


# --------------------------------------------------------------------------- Lua emit
def _num(x) -> str:
    x = float(x)
    if not math.isfinite(x):
        raise ExportError(f"non-finite number in bundle: {x!r}")
    return repr(x)                      # shortest decimal that round-trips exactly


def _vec(v) -> str:
    return "{" + ", ".join(_num(c) for c in v) + "}"


def _str(s) -> str:
    out = ['"']
    for ch in str(s):
        o = ord(ch)
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif o < 32 or o == 127:
            out.append("\\%03d" % o)    # Lua 5.1 has no \u escapes; decimal escapes are safe
        else:
            out.append(ch)              # UTF-8 passes through untouched
    out.append('"')
    return "".join(out)


# --------------------------------------------------------------------------- bundle IO
def load_bundle(name_or_path: str | Path):
    """Return (bundle dict, label, sha256 of the file bytes, resolved path)."""
    p = Path(name_or_path)
    if not p.suffix:
        p = DATA_DIR / f"{name_or_path}.json"
    elif not p.exists() and (DATA_DIR / p.name).exists():
        p = DATA_DIR / p.name
    if not p.exists():
        raise ExportError(f"bundle not found: {p}")
    raw = p.read_bytes()
    bundle = json.loads(raw.decode("utf-8"))
    return bundle, p.stem, hashlib.sha256(raw).hexdigest(), p


# --------------------------------------------------------------------------- up axis
_AXES = {"x": 0, "y": 1, "z": 2}


def _axis_of(v):
    """'y'/'z' if v is (within tolerance) +Y or +Z, else None."""
    if v is None or len(v) != 3:
        return None
    for name in ("y", "z"):
        i = _AXES[name]
        if abs(v[i] - 1.0) < 1e-6 and all(abs(v[k]) < 1e-6 for k in range(3) if k != i):
            return name
    return None


def detect_up(bundle: dict):
    """(up axis 'y'|'z', source label, warnings). Never guesses silently."""
    warnings: list[str] = []
    scene = bundle.get("scene") or {}
    env = scene.get("physics_env") or {}
    up, source = None, None
    g = env.get("gravity_dir")
    if g:
        up = _axis_of([-float(c) for c in g])
        source = "physics_env.gravity_dir" if up else None
        if up is None:
            warnings.append(f"physics_env.gravity_dir {list(g)} is not axis-aligned +/-Y/Z; ignored")
    if up is None:
        cam_up = (scene.get("camera") or {}).get("up")
        up = _axis_of(cam_up)
        if up:
            source = "camera.up"
        elif cam_up is not None:
            warnings.append(f"camera.up {list(cam_up)} is not +Y/+Z; defaulting to 'y'")
    if up is None:
        up, source = "y", "default"

    # Cross-check against the motion: a body that mostly moves along one world axis
    # which is not 'up' deserves a human look (a projectile legitimately does this).
    frames = (bundle.get("trajectory") or {}).get("frames") or []
    if len(frames) >= 2:
        nb = len(frames[0]["bodies"])
        for b in range(nb):
            d = [abs(frames[-1]["bodies"][b]["pos"][i] - frames[0]["bodies"][b]["pos"][i]) for i in range(3)]
            if max(d) < 1e-3:
                continue
            dom = "xyz"[d.index(max(d))]
            if dom != up and dom in ("y", "z") and max(d) > 4.0 * max(x for i, x in enumerate(d) if "xyz"[i] != dom):
                warnings.append(
                    f"body {b} moves mostly along {dom} (dz {max(d):.3g} m) but up is '{up}' "
                    f"from {source}: verify the orientation before trusting this scenario")
                break
    return up, source, warnings


# --------------------------------------------------------------------------- shapes
def _bounds(sh):
    cx, cy, cz = sh["center"]
    t = sh["type"]
    if t == "Sphere":
        r = sh["radius"]
        return (cx - r, cy - r, cz - r), (cx + r, cy + r, cz + r)
    if t == "Box":
        x, y, z = sh["size"]
        return (cx - x / 2, cy - y / 2, cz - z / 2), (cx + x / 2, cy + y / 2, cz + z / 2)
    r, h = sh["radius"], sh["height"]
    return (cx - r, cy - r, cz - h / 2), (cx + r, cy + r, cz + h / 2)


def _degenerate(sh) -> bool:
    t = sh["type"]
    if t == "Sphere":
        return sh["radius"] <= _EPS
    if t == "Box":
        return min(sh["size"]) <= _EPS
    return sh["radius"] <= _EPS or sh["height"] <= _EPS


def _contained(a, b) -> bool:
    """True if shape a lies fully inside same-type shape b (exact for these three)."""
    if a["type"] != b["type"]:
        return False
    if a["type"] == "Sphere":
        d = math.dist(a["center"], b["center"])
        return d + a["radius"] <= b["radius"] + _EPS
    (alo, ahi), (blo, bhi) = _bounds(a), _bounds(b)
    if a["type"] == "Cylinder" and a["radius"] > b["radius"] + _EPS:
        return False
    return all(alo[i] >= blo[i] - _EPS and ahi[i] <= bhi[i] + _EPS for i in range(3))


def _surfaceprop(label: str, materials) -> str:
    for text in [label.lower()] + [m.lower() for m in materials]:
        for prop, hints in _SURFACE_HINTS:
            if any(h in text for h in hints):
                return prop
    return "default"


def _body_shapes(bundle, k, label, report):
    scene = bundle["scene"]
    mats = scene.get("materials") or {}
    raw = []
    for li, leaf in enumerate(scene.get("csg_leaves", [])):
        if leaf.get("body") != k:
            continue
        sh = leaf["shape"]
        st = sh.get("type")
        if leaf.get("op", "add") == "subtract":
            report["notes"].append(f"body {k} ({label}): subtract leaf {li} ({st}, material "
                                   f"{leaf.get('material')}) not modelled; the cavity is filled in")
            continue
        if st not in SUPPORTED_SHAPES:
            report["unsupported"].setdefault(st, 0)
            report["unsupported"][st] += 1
            report["notes"].append(f"body {k} ({label}): leaf {li} shape '{st}' unsupported in Phase 0; dropped")
            continue
        d = {"type": st, "center": [float(c) for c in sh["center"]],
             "material": leaf.get("material"), "_leaf": li}
        if st == "Sphere":
            d["radius"] = float(sh["radius"])
        elif st == "Cylinder":
            d["radius"], d["height"] = float(sh["radius"]), float(sh["height"])
        else:
            d["size"] = [float(sh["x"]), float(sh["y"]), float(sh["z"])]
        d["color"] = [float(c) for c in (mats.get(leaf.get("material")) or {}).get("color_rgb", [0.7, 0.7, 0.7])]
        raw.append(d)

    kept = []
    for d in raw:
        if _degenerate(d):
            report["notes"].append(f"body {k} ({label}): leaf {d['_leaf']} ({d['type']}, {d['material']}) "
                                   f"has zero size; dropped")
            d["_drop"] = True
    live = [d for d in raw if not d.get("_drop")]
    for i, d in enumerate(live):
        drop_for = None
        for j, o in enumerate(live):
            if i == j or o.get("_drop"):
                continue
            if _contained(d, o) and (not _contained(o, d) or j < i):
                drop_for = o
                break
        if drop_for is not None:
            d["_drop"] = True
            report["notes"].append(f"body {k} ({label}): leaf {d['_leaf']} ({d['type']}, {d['material']}) is "
                                   f"inside leaf {drop_for['_leaf']} ({drop_for['material']}); dropped as a "
                                   f"coincident layer")
        else:
            kept.append(d)
    for d in kept:
        d.pop("_leaf", None)
        d.pop("_drop", None)
    return kept


def _floor_level(bundle, up):
    """Height (m, along up) of the stage floor: top of the thinnest unbound Box slab."""
    ui = _AXES[up]
    best = None
    for leaf in (bundle["scene"].get("csg_leaves") or []):
        if leaf.get("body") is not None or leaf.get("op", "add") != "add":
            continue
        sh = leaf["shape"]
        if sh.get("type") != "Box":
            continue
        dims = [sh["x"], sh["y"], sh["z"]]
        if dims[ui] != min(dims):
            continue
        top = sh["center"][ui] + dims[ui] / 2
        if best is None or top < best:
            best = top
    return best



def _dedupe_frames(frames, name, report):
    """Collapse consecutive frames that share a timestamp AND an identical pose (nine
    mechanism bundles append their final frame twice). Equal time with a DIFFERENT
    pose is ambiguous, and decreasing time is corrupt: both are errors."""
    out = [frames[0]]
    for f in frames[1:]:
        prev = out[-1]
        if f["t_sim"] > prev["t_sim"]:
            out.append(f)
        elif f["t_sim"] == prev["t_sim"] and f["bodies"] == prev["bodies"]:
            report["notes"].append(f"dropped an exact-duplicate frame at t={f['t_sim']!r}")
        elif f["t_sim"] == prev["t_sim"]:
            raise ExportError(f"{name}: two different poses share t={f['t_sim']!r}; ambiguous")
        else:
            raise ExportError(f"{name}: frame times decrease ({prev['t_sim']!r} -> {f['t_sim']!r})")
    return out

# --------------------------------------------------------------------------- export
def export_scenario(bundle: dict, name: str, source_label: str = "", sha256: str = ""):
    """Return (lua_text, report). Raises ExportError rather than degrade silently."""
    if bundle.get("kind") != "trajectory":
        raise ExportError(f"{name}: kind={bundle.get('kind')!r}; only 'trajectory' bundles replay "
                          f"(static scene displays are a later phase)")
    traj = bundle.get("trajectory") or {}
    frames = traj.get("frames") or []
    if not frames or not frames[0].get("bodies"):
        raise ExportError(f"{name}: trajectory has no frames/bodies")
    scene = bundle["scene"]
    notes_pre: list[str] = []
    pre = {"notes": notes_pre}
    frames = _dedupe_frames(frames, name, pre)
    nbody = len(frames[0]["bodies"])
    labels = traj.get("body_labels") or [b.get("label", f"body{i}") for i, b in enumerate(scene.get("bodies") or [])]
    if len(labels) < nbody:
        labels = list(labels) + [f"body{i}" for i in range(len(labels), nbody)]

    report = {"name": name, "bodies": nbody, "frames": len(frames), "notes": list(notes_pre), "warnings": [],
              "unsupported": {}}
    up, up_source, warns = detect_up(bundle)
    report["warnings"].extend(warns)
    report["up"], report["up_source"] = up, up_source

    floor = _floor_level(bundle, up)
    if floor is None:
        floor = 0.0
        report["warnings"].append("no unbound floor slab found; assuming the floor is at height 0")
    report["floor_m"] = floor

    # temperature channel: all-or-nothing per bundle (thermal colouring is Phase 3)
    has_t = [any(f["bodies"][b].get("temperature_k") is not None for f in frames) for b in range(nbody)]
    if any(has_t):
        report["warnings"].append("temperature_k present; exported but not displayed until Phase 3")
        if not all(all(f["bodies"][b].get("temperature_k") is not None for f in frames)
                   for b in range(nbody) if has_t[b]):
            raise ExportError(f"{name}: temperature_k present on some frames only (partial channel unsupported)")
        if not all(has_t):
            raise ExportError(f"{name}: temperature_k on some bodies only (partial channel unsupported)")
    stride = 8 if any(has_t) else 7

    gravity = (scene.get("physics_env") or {}).get("gravity_m_s2")
    gravity_source = "physics_env.gravity_m_s2" if gravity else "default"
    gravity = float(gravity) if gravity else DEFAULT_GRAVITY

    pbodies = scene.get("bodies") or []
    sphys = scene.get("physics") or {}
    times = [float(f["t_sim"]) for f in frames]
    if any(times[i + 1] <= times[i] for i in range(len(times) - 1)):
        raise ExportError(f"{name}: frame times are not strictly increasing")

    body_blocks = []
    for k in range(nbody):
        label = str(labels[k])
        shapes = _body_shapes(bundle, k, label, report)
        if not shapes:
            raise ExportError(f"{name}: body {k} ({label}) has no supported shape "
                              f"(unsupported: {report['unsupported'] or 'none; all leaves dropped'})")
        pivot = [float(c) for c in ((pbodies[k] if k < len(pbodies) else {}).get("pivot") or [0.0, 0.0, 0.0])]
        mass = (pbodies[k] if k < len(pbodies) else {}).get("mass_kg")
        if mass is None and nbody == 1:
            mass = sphys.get("mass_kg")
        if mass is None:
            report["warnings"].append(f"body {k} ({label}): no mass in bundle; Lua will estimate from volume")
        surfaceprop = _surfaceprop(label, {s["material"] for s in shapes if s.get("material")})

        track = []
        for f in frames:
            b = f["bodies"][k]
            track.extend(float(c) for c in b["pos"])
            track.extend(float(c) for c in (b.get("quat") or [0.0, 0.0, 0.0, 1.0]))
            if stride == 8:
                track.append(float(b["temperature_k"]))
        body_blocks.append((label, mass, pivot, surfaceprop, shapes, track))

    # ---- emit
    L = []
    w = L.append
    w("-- GENERATED by sigma_ground.radiance.export_gmod -- DO NOT EDIT BY HAND.")
    w(f"-- source: {source_label or name}   sha256: {sha256 or 'n/a'}")
    w("-- Regenerate:  python -m sigma_ground.radiance.export_gmod " + name)
    w("return {")
    w(f"  format = {_str(FORMAT)},")
    w(f"  name = {_str(name)},")
    w(f"  source = {_str(source_label or name)},")
    w(f"  source_sha256 = {_str(sha256)},")
    w(f"  up = {_str(up)},")
    w(f"  up_source = {_str(up_source)},")
    w(f"  floor_m = {_num(floor)},")
    w(f"  gravity_m_s2 = {_num(gravity)},")
    w(f"  gravity_source = {_str(gravity_source)},")
    w(f"  t_end = {_num(traj.get('t_end_s', times[-1]))},")
    w(f"  natural_timescale = {_num(traj.get('natural_timescale_s', times[-1]))},")
    w(f"  suggested_rate = {_num(traj.get('suggested_rate', 1.0))},")
    w(f"  stride = {stride},")
    w("  times = {" + ", ".join(_num(t) for t in times) + "},")
    w("  bodies = {")
    for label, mass, pivot, surfaceprop, shapes, track in body_blocks:
        w("    {")
        w(f"      label = {_str(label)},")
        if mass is not None:
            w(f"      mass_kg = {_num(mass)},")
        w(f"      pivot = {_vec(pivot)},")
        w(f"      surfaceprop = {_str(surfaceprop)},")
        w("      shapes = {")
        for s in shapes:
            if s["type"] == "Sphere":
                dims = f"radius = {_num(s['radius'])}"
            elif s["type"] == "Cylinder":
                dims = f"radius = {_num(s['radius'])}, height = {_num(s['height'])}"
            else:
                dims = f"size = {_vec(s['size'])}"
            w(f"        {{ type = {_str(s['type'])}, center = {_vec(s['center'])}, {dims}, "
              f"color = {_vec(s['color'])}, material = {_str(s.get('material') or '')} }},")
        w("      },")
        w("      track = {")
        for i in range(0, len(track), stride):
            w("        " + ", ".join(_num(c) for c in track[i:i + stride]) + ",")
        w("      },")
        w("    },")
    w("  },")
    w("  notes = {")
    for n in report["notes"] + report["warnings"]:
        w(f"    {_str(n)},")
    w("  },")
    w("}")
    text = "\n".join(L) + "\n"

    report.update(stride=stride, t_end=traj.get("t_end_s", times[-1]), bytes=len(text.encode("utf-8")),
                  shapes=[[s["type"] for s in b[4]] for b in body_blocks])
    return text, report


def export_file(name_or_path, out_dir: Path | None = None, write: bool = True):
    bundle, name, sha, path = load_bundle(name_or_path)
    try:
        rel = path.relative_to(Path(__file__).resolve().parents[1])
        label = str(rel).replace("\\", "/")
    except ValueError:
        label = path.name
    text, report = export_scenario(bundle, name, label, sha)
    out_path = None
    if write:
        out_dir = Path(out_dir) if out_dir else DEFAULT_OUT
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{name}.lua"
        out_path.write_text(text, encoding="utf-8", newline="\n")
    report["out"] = str(out_path) if out_path else None
    return text, report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Export a Radiance bundle as a Mentat GMod scenario (Lua).")
    ap.add_argument("bundles", nargs="+", help="bundle name (in radiance/web/data) or path to a .json")
    ap.add_argument("--out", help=f"output directory (default: {DEFAULT_OUT})")
    ap.add_argument("--report-only", action="store_true", help="dry run: validate and report, write nothing")
    args = ap.parse_args(argv)
    rc = 0
    for b in args.bundles:
        try:
            _, rep = export_file(b, args.out, write=not args.report_only)
        except ExportError as e:
            print(json.dumps({"name": str(b), "error": str(e)}, indent=2))
            rc = 1
            continue
        print(json.dumps(rep, indent=2))
    return rc


if __name__ == "__main__":
    sys.exit(main())
