"""Fetch targeted Objaverse LVIS categories + write the per-object license ledger.

Objaverse (Allen AI) fills the honest gaps ShapeNet cannot: LVIS's everyday-object
vocabulary has hammer / mallet / frying_pan / teakettle / wineglass / motor …
where ShapeNet/PartNet (furniture+household) has none. We pull ONLY the target
categories (selective per-UID download, not the 47k bulk), and — ToU doctrine —
record every object's license in a ledger CSV: Sketchfab licenses vary PER MODEL
(mostly CC-BY family), so provenance is per-object, never per-dataset.

Raw GLBs land under D:/Aaron/datasets/objaverse/ (local only, gitignored).
The ledger (a distilled provenance aggregate) is the committable artifact.

Run:  python tools/fetch_objaverse_lvis.py            # survey + ledger + download
      python tools/fetch_objaverse_lvis.py --ledger   # survey + ledger only
"""
import csv
import os
import sys

_DEST = "D:/Aaron/datasets/objaverse"
_LEDGER = os.path.join(os.path.dirname(__file__), "..", "sigma_ground",
                       "inventory", "data", "objaverse_ledger.csv")

# LVIS categories to pull (surveyed 2026-07-08; counts at survey time).
# Chosen for the flagship/gallery gaps + the actuation epic (motor).
CATEGORIES = [
    "hammer",                        # 59  — the Captain's canonical gap
    "mallet",                        # 79
    "frying_pan",                    # 31  — the skillet
    "saucepan",                      # 26
    "teakettle",                     # 69
    "wineglass",                     # 101
    "pitcher_(vessel_for_liquid)",   # 73
    "screwdriver",                   # 40
    "wrench",                        # 53
    "lightbulb",                     # 83
    "motor",                         # 54  — engine-adjacent (the actuation epic)
    "cup",                           # 70
    "mug",                           # 126 — cross-check vs the PartNet mug
]
# NOT in the LVIS vocabulary (verified): axe, skillet, anvil, feather, gear.


def _redirect_base(objaverse):
    """Point the package's cache at the datasets drive (default is ~/.objaverse)."""
    os.makedirs(_DEST, exist_ok=True)
    objaverse.BASE_PATH = _DEST
    if hasattr(objaverse, "_VERSIONED_PATH"):
        objaverse._VERSIONED_PATH = os.path.join(_DEST, "hf-objaverse-v1")


def _license_of(meta) -> str:
    lic = (meta or {}).get("license")
    if isinstance(lic, dict):
        return str(lic.get("label") or lic.get("slug") or lic.get("uid") or "")
    return str(lic or "")


def main() -> None:
    import objaverse
    _redirect_base(objaverse)
    lvis = objaverse.load_lvis_annotations()
    picks = {c: lvis[c] for c in CATEGORIES if c in lvis}
    missing = [c for c in CATEGORIES if c not in lvis]
    uids = [u for v in picks.values() for u in v]
    cat_of = {u: c for c, v in picks.items() for u in v}
    print(f"categories: {len(picks)} (missing: {missing or 'none'})  models: {len(uids)}")

    print("loading full annotations for license ledger …")
    anno = objaverse.load_annotations(uids)
    with open(os.path.abspath(_LEDGER), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["uid", "lvis_category", "license", "name", "source_uri"])
        for u in uids:
            m = anno.get(u, {})
            w.writerow([u, cat_of[u], _license_of(m),
                        (m.get("name") or "")[:80], m.get("uri") or ""])
    from collections import Counter
    lic_counts = Counter(_license_of(anno.get(u, {})) for u in uids)
    print("ledger:", os.path.abspath(_LEDGER))
    print("license mix:", dict(lic_counts))

    if "--ledger" in sys.argv:
        return
    print(f"downloading {len(uids)} GLBs -> {_DEST} …")
    paths = objaverse.load_objects(uids=uids, download_processes=8)
    # Windows `spawn` workers re-import objaverse and ignore the parent's
    # BASE_PATH monkey-patch, so GLBs land under ~/.objaverse — relocate them
    # to the datasets drive (idempotent; annotations already honour _DEST).
    import shutil
    stray = os.path.join(os.path.expanduser("~"), ".objaverse", "hf-objaverse-v1", "glbs")
    if os.path.isdir(stray):
        dest_glbs = os.path.join(_DEST, "hf-objaverse-v1", "glbs")
        os.makedirs(os.path.dirname(dest_glbs), exist_ok=True)
        for sub in os.listdir(stray):
            s, d = os.path.join(stray, sub), os.path.join(dest_glbs, sub)
            os.makedirs(d, exist_ok=True)
            for f in os.listdir(s):
                if not os.path.exists(os.path.join(d, f)):
                    shutil.move(os.path.join(s, f), os.path.join(d, f))
        print(f"relocated stray GLBs {stray} -> {dest_glbs}")
    ok = sum(1 for p in paths.values() if p and os.path.exists(p))
    print(f"downloaded {ok}/{len(uids)} objects")
    by_cat = Counter(cat_of[u] for u, p in paths.items() if p and os.path.exists(p))
    for c in CATEGORIES:
        if c in by_cat:
            print(f"  {c:32s} {by_cat[c]}")


if __name__ == "__main__":
    main()
