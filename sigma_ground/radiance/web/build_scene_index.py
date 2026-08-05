"""Consolidate every scene on disk into the ONE index the viewer reads.

The gallery drawer in viewer.js fetches exactly one file — data/scenes.json.
Before this tool existed, four build scripts each wrote their own index
(scenes.json, voxel_scenes.json, voxel_sdf_scenes.json) but only the first
was ever read, and scenes saved by other paths entirely (front_door's
_save_bundle, the trajectory demos, the theater probes) were indexed
nowhere. Net effect: 89 scene files on disk, 42 reachable in the UI.

This tool derives the index FROM THE DISK rather than from whatever a build
script happened to remember to append, so a scene cannot become orphaned
again: every data/<slug>.json is scanned, its own embedded metadata read
(scene.name, kind, frame/leaf/body counts), and one complete, grouped index
emitted. Hand-written titles/questions from the existing indexes are
preserved — this consolidates, it does not flatten curation.

Each entry also carries the renderer-feature flags actually present in that
file (bounds/opacity/energy), so "which sims still need re-rendering through
the current pipeline?" is answerable from the index instead of by re-reading
21 MB of JSON.

Idempotent: running it twice is a no-op (scenes.json is read back as its own
curation source).

Run:  python -m sigma_ground.radiance.web.build_scene_index
      python -m sigma_ground.radiance.web.build_scene_index --check
"""
import argparse
import glob
import json
import os

DATA = os.path.join(os.path.dirname(__file__), "data")
INDEX = os.path.join(DATA, "scenes.json")
# read for curated titles; no longer read by the viewer itself
LEGACY_INDEXES = ("scenes.json", "voxel_scenes.json", "voxel_sdf_scenes.json")
# E3's input-layer data files live in data/ beside the scenes but are NOT
# scenes — the binding table and the generated verb-registry mirror
# (input.js fetches both; neither could ever render in the gallery).
NON_SCENES = ("input_bindings.json", "intent_verbs.json")

# slug prefix → group, longest prefix wins. Groups the viewer doesn't know a
# label for fall back to the raw group name, so a new prefix degrades to
# "shows up under its own heading", never "disappears".
_PREFIX_GROUP = (
    ("voxsdf_", "voxel_sdf"), ("vox_", "voxel"), ("obj_", "object"),
    ("theater_", "theater"), ("scene_", "composition"),
    ("heating_", "thermal"), ("hot_", "thermal"),
    ("windmill", "mechanism"), ("gear_", "mechanism"), ("motor_", "mechanism"),
    ("pliers", "mechanism"), ("bell_", "mechanism"),
    ("clock", "mechanism"), ("escapement", "mechanism"),
    ("opacity_demo", "demo"),
    ("fall", "drop"), ("sim_", "drop"), ("drop", "drop"),
    ("iron_ball", "drop"), ("mach_", "drop"), ("high_altitude", "drop"),
    ("windward", "drop"),
)
_GROUP_ORDER = ["showcase", "composition", "voxel_sdf", "voxel", "mechanism",
                "thermal", "object", "drop", "gauntlet", "theater", "demo"]


def _group_for(slug: str) -> str:
    for prefix, group in _PREFIX_GROUP:
        if slug.startswith(prefix):
            return group
    return "drop"


def _title_from(name: str, slug: str) -> str:
    """A display title from the scene's own name, else a de-slugged fallback.
    Long provenance-laden names (the clock's cites its whole source) are cut
    at the first em-dash/hyphen separator — the full text is still in the
    scene file, this is only the gallery card's one-liner."""
    if not name:
        return slug.replace("_", " ")
    for sep in ("—", " - ", ";"):
        if sep in name:
            name = name.split(sep)[0]
            break
    name = name.strip()
    return (name[:70] + "…") if len(name) > 71 else name


def _scan_one(path: str) -> dict:
    slug = os.path.basename(path)[:-5]
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    sc = d.get("scene", d)
    traj = d.get("trajectory") or {}
    frames = traj.get("frames") or []
    leaves = sc.get("csg_leaves") or []
    mats = sc.get("materials") or {}

    entry = {
        "slug": slug,
        "title": _title_from(sc.get("name") or "", slug),
        "question": "",
        "verb": "",
        "group": _group_for(slug),
        "kind": "trajectory" if frames else (d.get("kind") or sc.get("kind") or "static"),
        "leaves": len(leaves),
    }
    if frames:
        entry["frames"] = len(frames)
    if sc.get("bodies"):
        entry["bodies"] = len(sc["bodies"])
    # renderer-feature flags — what this file was actually baked with
    entry["has_bounds"] = any("bounds" in l for l in leaves)
    entry["has_opacity"] = any("opacity" in m for m in mats.values()
                               if isinstance(m, dict))
    entry["has_energy"] = bool(frames) and "energy_j" in frames[0]
    return entry


def _curated() -> dict:
    """slug → hand-written {title, question, verb, group} from any existing
    index. Preserved verbatim; only blank fields get derived values."""
    out = {}
    for name in LEGACY_INDEXES:
        path = os.path.join(DATA, name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                rows = json.load(fh)
        except Exception:
            continue
        for row in rows:
            slug = row.get("slug")
            if slug:
                out.setdefault(slug, {}).update(
                    {k: v for k, v in row.items()
                     if k in ("title", "question", "verb", "group") and v})
    return out


def build() -> list:
    curated = _curated()
    entries = []
    for path in sorted(glob.glob(os.path.join(DATA, "*.json"))):
        if os.path.basename(path) in LEGACY_INDEXES + NON_SCENES:
            continue
        try:
            entry = _scan_one(path)
        except Exception as e:                       # a malformed file is
            print(f"  ! {os.path.basename(path)}: {type(e).__name__}: {e}")
            continue                                 # reported, never silently indexed
        entry.update(curated.get(entry["slug"], {}))  # curation wins
        entries.append(entry)
    order = {g: i for i, g in enumerate(_GROUP_ORDER)}
    entries.sort(key=lambda e: (order.get(e["group"], 99), e["title"].lower()))
    return entries


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="report coverage/staleness, write nothing")
    args = ap.parse_args()

    entries = build()
    stale = [e for e in entries if not (e["has_bounds"] and e["has_opacity"])]
    by_group = {}
    for e in entries:
        by_group.setdefault(e["group"], []).append(e)

    for g in sorted(by_group, key=lambda g: _GROUP_ORDER.index(g)
                    if g in _GROUP_ORDER else 99):
        print(f"  {g:12s} {len(by_group[g]):3d}")
    print(f"\n  {len(entries)} scenes indexed, "
          f"{len(entries) - len(stale)} rendered with the current pipeline, "
          f"{len(stale)} stale")

    if args.check:
        print("  (--check: nothing written)")
        return
    with open(INDEX, "w", encoding="utf-8") as fh:
        json.dump(entries, fh, indent=1, ensure_ascii=False)
    print(f"  -> {os.path.relpath(INDEX)}")


if __name__ == "__main__":
    main()
