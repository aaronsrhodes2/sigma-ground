"""Turn what the repo already holds into cards (offline, no network):

  1. curated   - stage/gmod/holodeck/data/catalog.json: 46 kinds / 126 Half-Life 2 + CS:S models, engine-measured sizes, fidelity masses.
                 Valve content: redistributable=false (the footprints are facts; the meshes never leave Garry's Mod).
  2. placeholder - sigma_ground/inventory/data/shapenetsem_sizes.json names + median sizes (ShapeNet-derived: redistributable=false)
                 and object_sizes.json (scalar typical sizes), object_dims.json (published size standards).
  3. deckard   - sigma_ground/inventory/data/exemplars/*.json PartNet part layouts (ShapeNet-derived: redistributable=false).
  4. seed      - two second-hand sourced facts from the design document (car, jumbo jet), clearly marked unverified.

    python stage/skippyverse/ingest_local.py [--out stage/skippyverse/build/cards.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from skippyverse import cards as C  # noqa: E402
from skippyverse.taxonomy import guess_category  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CATALOG = REPO / "stage" / "gmod" / "holodeck" / "data" / "catalog.json"
INV = REPO / "sigma_ground" / "inventory" / "data"

VALVE = {"source": "Half-Life 2 / Counter-Strike: Source (Valve), Garry's Mod install", "license": "Valve-SSA", "author": "Valve",
         "url": None}
SHAPENET = {"source": "ShapeNetSem (Savva et al. 2015), derived aggregate facts via sigma_ground/inventory/data",
            "license": "ShapeNet Terms of Use (non-commercial research)", "author": "ShapeNet", "url": None}
FRONT_CURATED = {"checked by eye", "curated per model", "curated per kind", "from the collision mesh (confirmed by eye)", "symmetric"}


def _stem(model: str) -> str:
    return model.rsplit("/", 1)[-1].rsplit(".", 1)[0]


# ------------------------------------------------------------------------------------------------------------- 1. curated
def curated_cards(catalog: dict) -> list[dict]:
    out = []
    for kid in catalog["order"]:
        k = catalog["kinds"][kid]
        for i, v in enumerate(k["variants"]):
            mass = {"kg": v["mass"], "low": v.get("mass_low"), "high": v.get("mass_high"),
                    "basis": "estimated" if v["mass_status"] == "audited" else "guessed", "derived_from_size": True,
                    "source": v["mass_source"], "confidence": v.get("mass_confidence")}
            if mass["basis"] == "guessed":
                mass["source"] = "unaudited: the engine's own gameplay mass, not a real-world figure. " + mass["source"]
                mass["low"] = mass["high"] = None
            top = None
            if v.get("top"):
                t = v["top"]
                top = {key: t[key] * C.HU_M for key in ("z", "x0", "x1", "y0", "y1")}
            fsrc = "curated-by-eye" if v["front_source"] in FRONT_CURATED else "heuristic"
            if v["front_source"].startswith("unknown"):
                fsrc = "unknown"
            card = C.make_card(
                id=f"curated.{C.slug(kid)}.{i}", name=kid, kind=kid,
                aliases=[n for n in k["names"] if n != kid], approx_names=k.get("approx", []), category=k["category"],
                size={"w": v["w"] * C.HU_M, "d": v["d"] * C.HU_M, "h": v["h"] * C.HU_M, "basis": "measured",
                      "source": "Source-engine bounding box measured in-game (stage/gmod/addon/.../measure_models.lua)"},
                front={"yaw_deg": v["front"], "source": fsrc}, symmetric=v["symmetric"], top=top, mass=mass,
                seat=k["seat"], wall=k["wall"],
                visual={"class": "curated", "style": "game-model", "blocky": False, "local_model": v["model"],
                        "model_frame_hu": {"mins": v["mins"], "maxs": v["maxs"], "cx": v["cx"], "cy": v["cy"]}},
                provenance=dict(VALVE))
            C.assert_valid(card)
            out.append(card)
    return out


# ------------------------------------------------------------------------------------------------------------- 4. seed facts
SEED_NOTE = "second-hand from misc/SKIPPYVERSE_OBJECTS_DESIGN.md; the design cites Wikidata/Wikipedia, NOT re-checked here"
SEED = [
    dict(id="seed.car", name="car", aliases=["automobile", "motorcar", "auto"], category="vehicle/road",
         size=(1.8, 4.5, 1.45), size_src="design doc section 1: car 4.5 x 1.8 x 1.45 m (" + SEED_NOTE + ")",
         mass=dict(kg=1565, low=1100, high=2200, basis="estimated",
                   source="design doc section 9: Deckard prototype 1,565 kg for a mid-size sedan, inside a 'sourced band' 1,100-2,200 kg (" + SEED_NOTE + ")"),
         front=0.0),
    dict(id="seed.jumbo-jet", name="jumbo jet", aliases=["boeing 747", "747", "jetliner", "airliner", "jumbojet"],
         category="vehicle/aircraft", size=(64.44, 70.66, 19.40),
         size_src="design doc section 1: Wikidata dimensions 70.66 x 64.44 x 19.4 m (" + SEED_NOTE + ")",
         mass=dict(kg=183523, low=160000, high=220000, basis="sourced",
                   source="design doc section 4: 747-400 operating empty weight (Wikipedia table) (" + SEED_NOTE + ")"),
         front=0.0),
]


def seed_cards() -> list[dict]:
    out = []
    for s in SEED:
        w, d, h = s["size"]
        mass = dict(s["mass"], derived_from_size=False)
        card = C.make_card(
            id=s["id"], name=s["name"], aliases=s["aliases"], category=s["category"],
            size={"w": w, "d": d, "h": h, "basis": "sourced", "source": s["size_src"]},
            front={"yaw_deg": s["front"], "source": "unknown"}, symmetric=True, mass=mass,
            visual={"class": "placeholder", "style": "box", "blocky": True},
            provenance={"source": "design document facts, unverified", "license": "unverified", "author": None, "url": None},
            flags=["unverified_source"])
        C.assert_valid(card)
        out.append(card)
    return out


# --------------------------------------------------------------------------------------------- 2. ShapeNetSem / Deckard sizes
def _key(name: str) -> str:
    """Spaces and punctuation do not make a different name: 'floor lamp' = 'floorlamp'."""
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _names_in(cards_: list[dict]) -> dict:
    names = {}
    for c in cards_:
        for n in [c["name"], *c["aliases"]]:
            names.setdefault(_key(n), c)
    return names


def placeholder_cards(existing: list[dict]) -> tuple[list[dict], dict]:
    """ShapeNetSem names+sizes, object_sizes.json and object_dims.json -> placeholder cards. Names already held by an existing card
    are not added; their ShapeNet size is recorded on the existing card as a cross-check instead."""
    taken = _names_in(existing)
    rep = Counter()
    out: list[dict] = []
    sn = json.loads((INV / "shapenetsem_sizes.json").read_text(encoding="utf-8"))["categories"]
    # identical records under several names (sofa/couch) are one thing with aliases
    groups: dict[tuple, list[str]] = {}
    for name, rec in sn.items():
        groups.setdefault((tuple(rec["dims_m"]), rec["n"]), []).append(name)
    for (dims, n), names in sorted(groups.items(), key=lambda kv: sorted(kv[1])[0]):
        names = sorted(names, key=lambda s: (len(s), s))
        hit = next((taken[_key(x)] for x in names if _key(x) in taken), None)
        x, y, z = dims                                  # ShapeNetSem aligned dims are [x, y-up, z] (design section 9)
        if hit is not None:
            ratio = max(x, y, z) / max(hit["size_m"]["w"], hit["size_m"]["d"], hit["size_m"]["h"]) if hit.get("size_m") else None
            hit.setdefault("crosscheck", []).append({"source": "ShapeNetSem median", "dims_m": [x, y, z], "n": n, "longest_ratio": ratio})
            if ratio is not None and not (1 / 3 <= ratio <= 3):
                # ShapeNetSem scales are only `estimated`: if the held card is stronger, the OTHER source is the outlier
                if hit["size_m"]["basis"] in ("guessed", "estimated"):
                    if "size_crosscheck_disagrees" not in hit["flags"]:
                        hit["flags"].append("size_crosscheck_disagrees")
                    rep["crosscheck_disagree_held_card_flagged"] += 1
                else:
                    hit["crosscheck"][-1]["verdict"] = "ShapeNetSem value is the outlier (held card has a stronger basis)"
                    rep["crosscheck_disagree_shapenet_outlier"] += 1
            rep["shapenet_names_already_held"] += len(names)
            continue
        primary = names[0]
        size = {"w": x, "d": z, "h": y, "basis": "estimated",
                "source": f"ShapeNetSem per-name median of n={n} models (annotation scales, unverified; axis order [x, y-up, z])"}
        card = C.make_card(id=f"placeholder.shapenetsem.{C.slug(primary)}", name=primary, aliases=names[1:],
                           category=guess_category(" ".join(names)), size=size, mass=C.prior_mass(size),
                           visual={"class": "placeholder", "style": "box", "blocky": True},
                           provenance=dict(SHAPENET), flags=["shapenet_derived"])
        C.assert_valid(card)
        out.append(card)
        for nm in names:
            taken[_key(nm)] = card
        rep["shapenet_cards"] += 1
        rep["shapenet_names"] += len(names)
    # scalar typical sizes (object_sizes.json): one number, shape unknown -> a cube labelled guessed
    for rec in json.loads((INV / "object_sizes.json").read_text(encoding="utf-8")):
        names = [rec["object"], *rec.get("aliases", [])]
        if any(_key(n) in taken for n in names):
            rep["object_sizes_already_held"] += 1
            continue
        s = rec["size_m"]
        size = {"w": s, "d": s, "h": s, "basis": "guessed", "source": f"{rec['source']} (one scalar; cube assumed, shape unknown)"}
        card = C.make_card(id=f"placeholder.object-sizes.{C.slug(names[0])}", name=names[0], aliases=names[1:],
                           category=guess_category(" ".join(names)), size=size, mass=C.prior_mass(size),
                           visual={"class": "placeholder", "style": "box", "blocky": True},
                           provenance={"source": "sigma_ground/inventory/data/object_sizes.json", "license": "repo-original, unreviewed",
                                       "author": None, "url": None}, flags=["scalar_size_only"])
        C.assert_valid(card)
        out.append(card)
        for nm in names:
            taken[_key(nm)] = card
        rep["object_sizes_cards"] += 1
    return out, dict(rep)


def deckard_cards(existing: list[dict]) -> tuple[list[dict], dict]:
    """One card per PartNet exemplar category with a part layout, unless a curated card already owns the name."""
    taken = _names_in(existing)
    rep = Counter()
    dims = json.loads((INV / "object_dims.json").read_text(encoding="utf-8"))["categories"]
    sn = json.loads((INV / "shapenetsem_sizes.json").read_text(encoding="utf-8"))["categories"]
    by_cat: dict[str, dict] = {}
    for f in sorted(glob.glob(str(INV / "exemplars" / "*.json"))):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        by_cat.setdefault(d["category"], d)                        # the first exemplar file per category is the representative
    out = []
    for cat, ex in sorted(by_cat.items()):
        names = [cat, *ex.get("aliases", [])]
        if any(_key(n) in taken for n in names):
            rep["exemplar_categories_already_curated"] += 1
            continue
        size = None
        if cat in dims:
            x, y, z = dims[cat]["dims_m"]
            size = {"w": x, "d": y, "h": z, "basis": "sourced", "source": dims[cat]["source"]}
        else:
            key = next((n for n in [cat, cat.replace(" ", ""), *ex.get("aliases", [])] if n in sn), None)
            if key:
                x, y, z = sn[key]["dims_m"]
                size = {"w": x, "d": z, "h": y, "basis": "estimated",
                        "source": f"ShapeNetSem per-name median of n={sn[key]['n']} models (unverified; axis order [x, y-up, z])"}
        parts = [{"name": p["name"], "shape": p["shape"], "center_frac": p["center_frac"], "size_frac": p["size_frac"]} for p in ex["parts"]]
        card = C.make_card(id=f"deckard.{C.slug(cat)}", name=cat, aliases=ex.get("aliases", []), category=guess_category(cat),
                           size=size, mass=C.prior_mass(size) if size else None,
                           visual={"class": "deckard", "style": "parts", "blocky": True, "parts": parts},
                           provenance={"source": "PartNet exemplar via sigma_ground/inventory/data/exemplars (ShapeNet-derived)",
                                       "license": "ShapeNet/PartNet Terms of Use (non-commercial)", "author": "PartNet", "url": None},
                           flags=["shapenet_derived"])
        C.assert_valid(card)
        out.append(card)
        rep["deckard_cards"] += 1
        if size is None:
            rep["deckard_without_size"] += 1
    return out, dict(rep)


# ------------------------------------------------------------------------------------------------------------------------ all
def ingest_all() -> tuple[list[dict], dict]:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    cs = curated_cards(catalog)
    seed = seed_cards()
    held = cs + seed
    dk, rep_d = deckard_cards(held)
    ph, rep_p = placeholder_cards(held + dk)
    cards = cs + seed + dk + ph
    ids = [c["id"] for c in cards]
    assert len(ids) == len(set(ids)), "duplicate card ids"
    report = {"total": len(cards), "by_class": dict(Counter(c["visual"]["class"] for c in cards)),
              "redistributable": sum(c["redistributable"] for c in cards),
              "not_redistributable": sum(not c["redistributable"] for c in cards),
              "size_basis": dict(Counter((c["size_m"] or {}).get("basis", "none") for c in cards)),
              "mass_basis": dict(Counter((c["mass"] or {}).get("basis", "none") for c in cards)),
              "curated_kinds": len({c["kind"] for c in cs}), **rep_d, **rep_p}
    return cards, report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "build" / "cards.json"))
    a = ap.parse_args(argv)
    cards, report = ingest_all()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cards, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    print(f"wrote {len(cards)} cards -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
