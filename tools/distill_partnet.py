"""Distill PartNet into Deckard's composition prior — vocabulary AND geometry.

Two data tiers, two subcommands plus helpers:

  taxonomy  — the ORIGINAL public-taxonomy distill (PartNet repo's
              stats/after_merging_label_ids/, MIT): part VOCABULARY per category.
  sample    — pick <=150 models per category (deterministic stride over sorted
              anno_ids) from the locally extracted data_v0 JSONs and write a
              7-Zip listfile for extracting just those models' objs/ meshes.
  geometry  — the real shaping prior, from the locally extracted gated data
              (D:/Aaron/datasets/shapenet/PartNet/data_v0): per category and
              part label, aggregate over models —
                freq      fraction of models containing the part   (ALL models)
                count     median instances per containing model    (ALL models)
                size_frac median per-axis part/object extents      (sampled)
                z_frac    median up-axis center offset, [-0.5,0.5] (sampled)
                r_frac    median RADIAL lateral offset fraction    (sampled)
                shape     primitive class from bbox aspect + label prior
              and MERGE into inventory/data/compositions.json (PartNet entries
              replace what they cover; curated-only entries survive).
  verify    — directionality sanity gates (chair legs LOW, tabletop HIGH):
              catches an axis-convention mistake before anything ships.

PartNet/ShapeNet are Y-up; Deckard is Z-up — bboxes are axis-mapped
(x,y,z)_pn -> (x,z,y)_deckard at read time. ONLY AGGREGATES ship: medians and
frequencies over thousands of models, cited; meshes and per-model rows never
enter the repo (ShapeNet Terms of Use — see docs/DATA_SOURCES.md).

    python tools/distill_partnet.py taxonomy
    python tools/distill_partnet.py sample      # writes objs listfile
    python tools/distill_partnet.py geometry    # writes compositions.json
    python tools/distill_partnet.py verify
"""
from __future__ import annotations

import json
import math
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

_RAW = ("https://raw.githubusercontent.com/daerduoCarey/partnet_dataset/master/"
        "stats/after_merging_label_ids/{cat}-level-1.txt")
_OUT = (pathlib.Path(__file__).resolve().parents[1]
        / "sigma_ground" / "inventory" / "data" / "compositions.json")
_DATA = pathlib.Path("D:/Aaron/datasets/shapenet/PartNet/data_v0")
_LISTFILE = _DATA.parent / "objs_sample.txt"

_N_SAMPLE = 150            # geometry models per category (deterministic stride)
_MIN_FREQ = 0.15           # parts rarer than this are dropped
_MIN_GEOM = 8              # geometry stats need at least this many instances
_MAX_PARTS = 8             # keep entries readable (existing rule)
_WHOLE_OBJ = 0.85          # a "part" covering >85% of every axis IS the object

_PN_SOURCE = "PartNet (Mo et al. 2019) — aggregate medians"
_PN_LICENSE = ("ShapeNet/PartNet Terms of Use (non-commercial research); "
               "derived aggregate facts only, no raw rows")

# PartNet category -> (common object name, [aliases])
_CATEGORIES = {
    "Bag": ("bag", ["handbag"]), "Bed": ("bed", []), "Bottle": ("bottle", []),
    "Bowl": ("bowl", []), "Chair": ("chair", []), "Clock": ("clock", []),
    "Dishwasher": ("dishwasher", []), "Display": ("monitor", ["display", "screen"]),
    "Door": ("door", []), "Earphone": ("earphone", ["headphones", "earbuds"]),
    "Faucet": ("faucet", ["tap"]), "Hat": ("hat", ["cap"]), "Keyboard": ("keyboard", []),
    "Knife": ("knife", []), "Lamp": ("lamp", []), "Laptop": ("laptop", []),
    "Microwave": ("microwave", []), "Mug": ("mug", []),
    "Refrigerator": ("refrigerator", ["fridge"]), "Scissors": ("scissors", []),
    "StorageFurniture": ("cabinet", ["storage furniture", "dresser", "cupboard"]),
    "Table": ("table", ["desk"]), "TrashCan": ("trash can", ["bin", "garbage can"]),
    "Vase": ("vase", ["pot"]),
}

_NOISE = {"containing_things", "other", "others", "other_leaf", ""}
_SHAPE_HINT = {
    "handle": "cylinder", "leg": "cylinder", "stem": "cylinder",
    "shaft": "cylinder", "neck": "cylinder", "spout": "cylinder",
    "knob": "sphere", "wheel": "cylinder", "foot": "cylinder",
    "screw": "cylinder", "cord": "cylinder", "chain": "cylinder",
    "blade": "outline", "bowl": "ellipsoid",
}


def _clean(name: str, root: str = "") -> str:
    """Normalize a PartNet node label: strip grouping suffixes + category prefix."""
    name = (name or "").strip().lower()
    for suf in ("_side", "_group", "_set", "_unit"):
        if name.endswith(suf):
            name = name[: -len(suf)]
    if root and name.startswith(root + "_"):
        name = name[len(root) + 1:]
    return name.strip("_")


# ── hierarchy walking ────────────────────────────────────────────────────────
def _subtree_objs(node) -> list:
    objs = list(node.get("objs") or [])
    for ch in node.get("children") or []:
        objs += _subtree_objs(ch)
    return objs


def _node_instances(root_node, root_label: str) -> list:
    """Every named node below the category root as
    (cleaned_label, [subtree objs], is_leaf). A node IS an instance of its
    label; its geometry is its whole subtree. ``is_leaf`` distinguishes a real
    PART that happens to span the object (a bottle's body) from a subtype
    WRAPPER (regular_table) — only wrappers get whole-object-filtered."""
    out = []

    def walk(n, depth):
        nm = _clean(n.get("name") or "", root_label)
        if depth >= 1 and nm not in _NOISE:
            out.append((nm, _subtree_objs(n), not (n.get("children"))))
        for ch in n.get("children") or []:
            walk(ch, depth + 1)

    walk(root_node, 0)
    return out


def _load_hierarchy(model_dir: pathlib.Path):
    for fn in ("result_after_merging.json", "result.json"):
        p = model_dir / fn
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8", errors="replace"))
                if isinstance(data, list) and data:
                    return data[0]
            except Exception:
                return None
    return None


# ── geometry ────────────────────────────────────────────────────────────────
def _obj_bbox(path: pathlib.Path):
    """Stream a Wavefront OBJ's vertex bbox in DECKARD axes (PartNet y-up -> z-up:
    (x, y, z)_pn -> (x, z, y)_dk). Returns (min3, max3) or None."""
    lo = [math.inf] * 3
    hi = [-math.inf] * 3
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.startswith("v "):
                    continue
                t = line.split()
                if len(t) < 4:
                    continue
                try:
                    x, y, z = float(t[1]), float(t[2]), float(t[3])
                except ValueError:
                    continue
                for i, v in enumerate((x, z, y)):          # axis map here
                    lo[i] = min(lo[i], v)
                    hi[i] = max(hi[i], v)
    except OSError:
        return None
    if lo[0] is math.inf:
        return None
    return lo, hi


def _union(b1, b2):
    if b1 is None:
        return b2
    if b2 is None:
        return b1
    return ([min(a, b) for a, b in zip(b1[0], b2[0])],
            [max(a, b) for a, b in zip(b1[1], b2[1])])


def _model_part_stats(model_dir: pathlib.Path, root_label: str):
    """Per part label in ONE model: [(extents3, center3)] per instance, plus the
    object bbox — all in Deckard axes. None if hierarchy or geometry missing."""
    root = _load_hierarchy(model_dir)
    if root is None:
        return None
    objs_dir = model_dir / "objs"
    bbox_cache: dict = {}

    def bbox_of(obj_name):
        if obj_name not in bbox_cache:
            bbox_cache[obj_name] = _obj_bbox(objs_dir / f"{obj_name}.obj")
        return bbox_cache[obj_name]

    inst: dict[str, list] = {}
    leafness: dict[str, bool] = {}
    obj_bb = None
    for label, objs, is_leaf in _node_instances(root, root_label):
        bb = None
        for o in objs:
            bb = _union(bb, bbox_of(o))
        if bb is None:
            continue
        ext = [bb[1][i] - bb[0][i] for i in range(3)]
        ctr = [(bb[1][i] + bb[0][i]) / 2.0 for i in range(3)]
        inst.setdefault(label, []).append((ext, ctr))
        leafness[label] = leafness.get(label, True) and is_leaf
        obj_bb = _union(obj_bb, bb)
    if obj_bb is None:
        return None
    return inst, obj_bb, leafness


def _classify_prim(size_frac, label: str) -> str:
    """Primitive class from bbox aspect; the label prior wins where it speaks."""
    if label in _SHAPE_HINT:
        return _SHAPE_HINT[label]
    e = sorted(size_frac, reverse=True)
    if e[0] <= 0:
        return "box"
    if e[0] >= 2.5 * max(e[1], 1e-9) and e[1] <= 1.6 * max(e[2], 1e-9):
        return "cylinder"                                   # rod
    return "box"                                            # slab / blocky


def aggregate_category(model_dirs: list, root_label: str,
                       geom_dirs: set | None = None) -> dict:
    """Aggregate part stats over a category's models.

    freq/count come from EVERY model with a hierarchy; size/z/r fractions only
    from models whose objs/ are on disk (the geometry sample)."""
    n_models = 0
    n_geom = 0
    contain: dict[str, int] = {}
    counts: dict[str, list] = {}
    fracs: dict[str, dict] = {}
    leaf_label: dict[str, bool] = {}
    for md in model_dirs:
        root = _load_hierarchy(md)
        if root is None:
            continue
        n_models += 1
        labels: dict[str, int] = {}
        for label, _objs, _leaf in _node_instances(root, root_label):
            labels[label] = labels.get(label, 0) + 1
        for label, k in labels.items():
            contain[label] = contain.get(label, 0) + 1
            counts.setdefault(label, []).append(k)
        if geom_dirs is not None and md.name not in geom_dirs:
            continue
        got = _model_part_stats(md, root_label)
        if not got:
            continue
        n_geom += 1
        inst, obj_bb, leafness = got
        for lbl, is_leaf in leafness.items():
            leaf_label[lbl] = leaf_label.get(lbl, True) and is_leaf
        oext = [max(obj_bb[1][i] - obj_bb[0][i], 1e-9) for i in range(3)]
        octr = [(obj_bb[1][i] + obj_bb[0][i]) / 2.0 for i in range(3)]
        lat = (oext[0] + oext[1]) / 4.0                     # mean lateral half-extent
        for label, instances in inst.items():
            f = fracs.setdefault(label, {"size": [], "z": [], "r": []})
            for ext, ctr in instances:
                f["size"].append([ext[i] / oext[i] for i in range(3)])
                f["z"].append((ctr[2] - octr[2]) / oext[2])
                f["r"].append(math.hypot(ctr[0] - octr[0], ctr[1] - octr[1])
                              / max(lat, 1e-9))
    if not n_models:
        return {}

    parts = []
    for label, n_has in sorted(contain.items(), key=lambda kv: -kv[1]):
        freq = n_has / n_models
        if freq < _MIN_FREQ:
            continue
        entry = {"name": label,
                 "count": int(min(8, max(1, round(statistics.median(counts[label]))))),
                 "freq": round(freq, 3)}
        f = fracs.get(label)
        if f and len(f["size"]) >= _MIN_GEOM:
            sf = [round(statistics.median(s[i] for s in f["size"]), 3)
                  for i in range(3)]
            # a subtype WRAPPER spanning the whole object is not a part — but a
            # LEAF that spans it (a bottle's body) is the object's main part
            if all(v >= _WHOLE_OBJ for v in sf) and not leaf_label.get(label, False):
                continue
            entry["size_frac"] = sf
            entry["z_frac"] = round(statistics.median(f["z"]), 3)
            entry["r_frac"] = round(statistics.median(f["r"]), 3)
            entry["shape"] = _classify_prim(sf, label)
        else:
            entry["shape"] = _SHAPE_HINT.get(label, "")
        parts.append(entry)
    _strip_degenerate_geometry(parts)
    # rank: frequent-and-measured first, keep the table readable
    parts.sort(key=lambda p: (-p["freq"], p["name"]))
    return {"parts": parts[:_MAX_PARTS], "n_models": n_models, "n_geom": n_geom}


def _strip_degenerate_geometry(parts: list) -> None:
    """PartNet merge noise can hand a part the WHOLE object's bbox. A
    replicated part (count>=2) cannot EACH span 2+ axes (four table legs of
    [0.94,0.98,0.98] are annotation unions, not legs — though a stretcher
    spanning ONE axis is real); an all-axes-spanning part is credible only for
    the category's BODY (the most frequent such part). Degenerate parts keep
    their count/freq (vocabulary value) and lose the untrustworthy fractions."""
    spanning = [p for p in parts
                if "size_frac" in p and min(p["size_frac"]) >= _WHOLE_OBJ]
    body = max(spanning, key=lambda p: p["freq"], default=None)
    for p in parts:
        sf = p.get("size_frac")
        if not sf:
            continue
        big_axes = sum(1 for v in sf if v >= _WHOLE_OBJ)
        if (p["count"] >= 2 and big_axes >= 2) or \
                (min(sf) >= _WHOLE_OBJ and p is not body):
            for k in ("size_frac", "z_frac", "r_frac"):
                p.pop(k, None)
            p["shape"] = p.get("shape") or _SHAPE_HINT.get(p["name"], "")


# ── subcommands ─────────────────────────────────────────────────────────────
def _models_by_category() -> dict:
    """anno_id dirs bucketed by meta.json model_cat (sorted, deterministic)."""
    cats: dict[str, list] = {}
    for md in sorted(_DATA.iterdir() if _DATA.is_dir() else []):
        meta = md / "meta.json"
        if not meta.exists():
            continue
        try:
            cat = json.loads(meta.read_text(encoding="utf-8",
                                            errors="replace")).get("model_cat", "")
        except Exception:
            continue
        if cat in _CATEGORIES:
            cats.setdefault(cat, []).append(md)
    return cats


def _sampled(dirs: list) -> list:
    if len(dirs) <= _N_SAMPLE:
        return dirs
    step = len(dirs) / _N_SAMPLE
    return [dirs[int(i * step)] for i in range(_N_SAMPLE)]


def cmd_sample() -> None:
    cats = _models_by_category()
    lines = []
    for cat, dirs in sorted(cats.items()):
        for md in _sampled(dirs):
            lines.append(f"data_v0\\{md.name}\\objs\\*")
        print(f"  {cat:18s} {len(dirs):5d} models -> {min(len(dirs), _N_SAMPLE)} sampled")
    _LISTFILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {_LISTFILE} ({len(lines)} include patterns)\n"
          f'extract with:\n  7z x data_v0_chunk.zip -o"{_DATA.parent}" '
          f'-ir@"{_LISTFILE}" -y')


def cmd_geometry() -> None:
    try:
        existing = json.loads(_OUT.read_text(encoding="utf-8"))
    except Exception:
        existing = []
    cats = _models_by_category()
    pn_names = {nm.lower() for c in _CATEGORIES
                for nm in ([_CATEGORIES[c][0]] + _CATEGORIES[c][1])}
    kept = [e for e in existing if (e.get("object") or "").lower() not in pn_names]

    pn_entries = []
    for cat, dirs in sorted(cats.items()):
        name, aliases = _CATEGORIES[cat]
        geom_dirs = {md.name for md in _sampled(dirs) if (md / "objs").is_dir()}
        agg = aggregate_category(dirs, name.replace(" ", "_"), geom_dirs)
        if not agg or not agg["parts"]:
            print(f"  ! {name}: no usable models")
            continue
        pn_entries.append({
            "object": name, "aliases": aliases, "parts": agg["parts"],
            "n_models": agg["n_models"], "n_geom": agg["n_geom"],
            "source": f"{_PN_SOURCE} over {agg['n_models']} models "
                      f"(geometry n={agg['n_geom']})",
            "license": _PN_LICENSE,
        })
        measured = sum(1 for p in agg["parts"] if "size_frac" in p)
        print(f"  + {name:14s} {agg['n_models']:5d} models, geom {agg['n_geom']:3d}: "
              + ", ".join(f"{p['count']}x {p['name']}" for p in agg["parts"][:5])
              + f"  ({measured} measured)")

    out = pn_entries + kept
    _OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {_OUT}  ({len(pn_entries)} PartNet-measured + {len(kept)} curated)")


def _leaf_geo(model_dir: pathlib.Path, root_label: str):
    """Every TRUE leaf part (a node with objs and no children) of one model as
    (cleaned_label, bbox) in Deckard axes, plus the model bbox — the real
    assembled layout (four legs are four separate leaves at their real spots)."""
    root = _load_hierarchy(model_dir)
    if root is None:
        return None
    objs_dir = model_dir / "objs"
    cache: dict = {}

    def bbox_of(o):
        if o not in cache:
            cache[o] = _obj_bbox(objs_dir / f"{o}.obj")
        return cache[o]

    leaves = []

    def walk(node):
        objs, ch = node.get("objs"), node.get("children")
        if objs and not ch:                            # a real leaf with geometry
            bb = None
            for o in objs:
                bb = _union(bb, bbox_of(o))
            if bb is not None:
                leaves.append((_clean(node.get("name") or "", root_label), bb))
        for c in (ch or []):
            walk(c)

    walk(root)
    if not leaves:
        return None
    mbb = None
    for _, bb in leaves:
        mbb = _union(mbb, bb)
    return leaves, mbb


def _exemplar_shape(size_frac, label: str) -> str:
    """A primitive for a leaf from its bbox aspect — cylinder ONLY for a rod
    standing on its long (z) axis (placing it vertical is correct); everything
    else is a box (axis-aligned, never mis-oriented). Knobs -> sphere."""
    if label in ("knob", "ball"):
        return "sphere"
    e = size_frac
    longest = max(range(3), key=lambda i: e[i])
    srt = sorted(e, reverse=True)
    is_rod = srt[0] >= 2.2 * max(srt[1], 1e-9) and srt[1] <= 1.6 * max(srt[2], 1e-9)
    if is_rod and longest == 2:                        # vertical rod -> upright cylinder
        return "cylinder"
    return "box"


def _dedup_leaves(leaves, oext, octr):
    """Collapse leaves whose normalized bbox is near-identical (PartNet annotates
    the same surface at several hierarchy levels — seat / seat_single_surface)."""
    out, seen = [], []
    for lbl, bb in leaves:
        cf = tuple(round(((bb[1][i] + bb[0][i]) / 2 - octr[i]) / oext[i], 2)
                   for i in range(3))
        sf = tuple(round((bb[1][i] - bb[0][i]) / oext[i], 2) for i in range(3))
        key = cf + sf
        if any(max(abs(a - b) for a, b in zip(key, k)) < 0.04 for k in seen):
            continue                                   # a duplicate of a kept part
        seen.append(key)
        out.append((lbl, bb))
    return out


def _exemplar_quality(leaves, mbb, freq):
    """Rank a candidate model: deduped, spatially-distinct parts win; if the
    category has legs, demand vertical posts at distinct corners (reject sled
    bases and degenerate annotations); then prefer the typical part set."""
    oext = [max(mbb[1][i] - mbb[0][i], 1e-9) for i in range(3)]
    octr = [(mbb[1][i] + mbb[0][i]) / 2.0 for i in range(3)]
    leaves = _dedup_leaves(leaves, oext, octr)
    cen = [tuple(((bb[1][k] + bb[0][k]) / 2 - octr[k]) / oext[k] for k in range(3))
           for _, bb in leaves]
    coincident = sum(1 for i in range(len(cen)) for j in range(i + 1, len(cen))
                     if max(abs(cen[i][k] - cen[j][k]) for k in range(3)) < 0.04)
    leg_ok = True
    if "leg" in freq:
        legs = [(lbl, bb) for lbl, bb in leaves if lbl in ("leg", "foot")]
        leg_ok = 3 <= len(legs) <= 6
        for _, bb in legs:                             # a real leg is a vertical post
            e = [bb[1][k] - bb[0][k] for k in range(3)]
            if e[2] < 0.9 * max(e[0], e[1]):
                leg_ok = False
        xy = {(round((bb[1][0] + bb[0][0]) / 2, 2), round((bb[1][1] + bb[0][1]) / 2, 2))
              for _, bb in legs}
        if len(xy) < 3:
            leg_ok = False                             # legs must reach distinct corners
    # boxiness: a part fits a primitive well only if it's a ROD (one long axis) or
    # a SLAB (one thin axis) — a "fat" part (all axes comparable, e.g. a curved or
    # reclined back) fills its bounding box as an ugly block. Prefer models whose
    # parts are all rod/slab, so box-fitting reproduces the real shape.
    boxy = 0
    for _, bb in leaves:
        e = sorted([bb[1][k] - bb[0][k] for k in range(3)], reverse=True)
        if e[0] >= 2.0 * max(e[1], 1e-9) or e[2] <= 0.35 * max(e[1], 1e-9):
            boxy += 1
    boxiness = boxy / max(len(leaves), 1)
    cov = sum(freq.get(lbl, 0.05) for lbl in {l for l, _ in leaves if l not in _NOISE})
    return (leg_ok, coincident == 0, round(boxiness, 2), round(cov, 3), len(leaves)), leaves


def cmd_exemplar() -> None:
    """Per category, pick ONE representative real model and ship its actual part
    layout as primitives — Deckard learns the assembled shape of "a chair" from a
    real chair, not from a hand-written template."""
    try:
        census = {e["object"]: e for e in json.loads(_OUT.read_text(encoding="utf-8"))}
    except Exception:
        census = {}
    out_dir = _OUT.parent / "exemplars"
    out_dir.mkdir(parents=True, exist_ok=True)
    cats = _models_by_category()
    written = 0
    for cat, dirs in sorted(cats.items()):
        name, aliases = _CATEGORIES[cat]
        freq = {p["name"]: p.get("freq", 0.0)
                for p in census.get(name, {}).get("parts", [])}
        best = None                                    # (quality_key, leaves, mbb, md)
        for md in dirs:
            if not (md / "objs").is_dir():
                continue
            got = _leaf_geo(md, name.replace(" ", "_"))
            if not got or not (2 <= len(got[0]) <= 20):
                continue
            qual, dleaves = _exemplar_quality(got[0], got[1], freq)
            if 2 <= len(dleaves) <= 14 and (best is None or qual > best[0]):
                best = (qual, dleaves, got[1], md)
        if best is None:
            print(f"  ! {name}: no exemplar candidates")
            continue
        _qual, leaves, mbb, md = best
        oext = [max(mbb[1][i] - mbb[0][i], 1e-9) for i in range(3)]
        octr = [(mbb[1][i] + mbb[0][i]) / 2.0 for i in range(3)]
        parts = []
        for lbl, bb in leaves:
            if lbl in _NOISE:
                continue
            ext = [bb[1][i] - bb[0][i] for i in range(3)]
            ctr = [(bb[1][i] + bb[0][i]) / 2.0 for i in range(3)]
            sf = [round(ext[i] / oext[i], 4) for i in range(3)]
            cf = [round((ctr[i] - octr[i]) / oext[i], 4) for i in range(3)]
            parts.append({"name": lbl, "shape": _exemplar_shape(sf, lbl),
                          "center_frac": cf, "size_frac": sf})
        doc = {"category": name, "aliases": aliases, "source_anno_id": md.name,
               "n_parts": len(parts), "parts": parts,
               "_source": f"PartNet (Mo et al. 2019) — representative model {md.name} "
                          f"(real assembled part layout)",
               "_license": _PN_LICENSE}
        (out_dir / f"{name.replace(' ', '_')}.json").write_text(
            json.dumps(doc, indent=1), encoding="utf-8")
        written += 1
        print(f"  + {name:14s} <- model {md.name} ({len(parts)} parts): "
              + ", ".join(p["name"] for p in parts[:8]))
    print(f"\nwrote {written} exemplars to {out_dir}")


def cmd_verify() -> None:
    data = json.loads(_OUT.read_text(encoding="utf-8"))
    by_name = {e["object"]: e for e in data}
    bad = []

    def part(obj, name):
        return next((p for p in by_name.get(obj, {}).get("parts", [])
                     if p["name"] == name and "z_frac" in p), None)

    leg = part("chair", "leg")
    if leg and leg["z_frac"] >= -0.10:
        bad.append(f"chair leg z_frac={leg['z_frac']} (expected < -0.10: legs are LOW)")
    top = part("table", "tabletop")
    if top and top["z_frac"] <= 0.10:
        bad.append(f"tabletop z_frac={top['z_frac']} (expected > 0.10: tops are HIGH)")
    for obj in ("chair", "table", "lamp", "mug"):
        e = by_name.get(obj)
        if e:
            print(f"  {obj}: " + ", ".join(
                f"{p['count']}x {p['name']}"
                + (f" z={p['z_frac']:+.2f}" if "z_frac" in p else "")
                for p in e["parts"]))
    if bad:
        raise SystemExit("VERIFY FAILED (axis convention?):\n  " + "\n  ".join(bad))
    print("verify OK — directionality sane")


def cmd_taxonomy() -> None:
    """The original public-taxonomy distill (kept for bootstrap/fallback)."""
    from sigma_ground.deckard.sources import web      # polite, cached fetch

    def _parts_for(cat: str, obj: str):
        text = web.get_text(_RAW.format(cat=cat))
        if not text:
            return []
        skip = _NOISE | {obj.lower(), cat.lower()}
        parts, seen = [], set()
        for line in text.splitlines():
            toks = line.split()
            if len(toks) < 2:
                continue
            name = _clean(toks[1].split("/")[-1], toks[1].split("/")[0])
            if not name or name in skip or name in seen:
                continue
            seen.add(name)
            parts.append({"name": name, "shape": _SHAPE_HINT.get(name, ""),
                          "count": 1})
        return parts[:_MAX_PARTS]

    try:
        existing = json.loads(_OUT.read_text(encoding="utf-8"))
    except Exception:
        existing = []
    pn_names = {nm.lower() for c in _CATEGORIES
                for nm in ([_CATEGORIES[c][0]] + _CATEGORIES[c][1])}
    kept = [e for e in existing if (e.get("object") or "").lower() not in pn_names]
    pn_entries = []
    for cat, (name, aliases) in _CATEGORIES.items():
        parts = _parts_for(cat, name)
        if not parts:
            print("  (no taxonomy for", cat, "- skipped)")
            continue
        pn_entries.append({"object": name, "aliases": aliases, "parts": parts,
                           "source": "PartNet (Mo et al. 2019)", "license": "MIT"})
        print("  %-16s -> %s" % (name, ", ".join(p["name"] for p in parts)))
    _OUT.write_text(json.dumps(pn_entries + kept, indent=1), encoding="utf-8")
    print(f"wrote {_OUT}  ({len(pn_entries)} PartNet + {len(kept)} curated)")


def main(argv: list) -> None:
    cmd = argv[0] if argv else "geometry"
    {"taxonomy": cmd_taxonomy, "sample": cmd_sample, "geometry": cmd_geometry,
     "exemplar": cmd_exemplar, "verify": cmd_verify}[cmd]()


if __name__ == "__main__":
    main(sys.argv[1:])
