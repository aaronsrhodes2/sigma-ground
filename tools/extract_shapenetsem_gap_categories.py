"""Extract ShapeNetSem models for categories confirmed absent elsewhere.

Correction to an earlier session claim: "ShapeNet has no hammer" was true for
PartNet (the dataset Deckard's voxel path actually uses) but FALSE for the
ShapeNet family as a whole — ShapeNetSem (a separate, model-level-annotated
subset of ShapeNet, distinct from PartNet and ShapeNetCore) has a real
`Hammer` category with 22 models, plus ScrewDriver and Motorcycle. Found only
by opening ShapeNetSem's per-model metadata directly; the aggregate stats
(median dims, category material ratios) we'd already distilled gave no hint
of this.

wrench/axe/anvil/saw/drill/pliers/gear/engine were checked and are confirmed
ABSENT from all three ShapeNet-family sources (PartNet + ShapeNetCore +
ShapeNetSem) — a real, verified gap, not a PartNet-specific limitation.

ShapeNetSem's `fullId` (after stripping the `wss.` prefix) is the SAME model
hash scheme as ShapeNetCore/PartNet model_ids — confirmed concretely: PartNet
chair 44164's meta.json model_id (ee2ea12a2a2f8eb71335bcae6f5543ce) is
literally ShapeNetCore.v2/03001627/ee2ea12a2a2f8eb71335bcae6f5543ce/. This
extraction keeps that fullId as the join key for the main lane.

Each model gets: the real OBJ+MTL mesh (models-OBJ), the pre-computed SOLID
binvox (models-binvox-solid — an independent, trusted voxelization, same
kind as ShapeNetCore's own solid.binvox), and — when present — the model's
real per-model `up`/`front` orientation and `unit` real-world scale factor
from metadata.csv (only ~30%/77% of ShapeNetSem rows carry these; recorded
as null when absent, never guessed).

Skips Knife/Scissors/Stool/Barstool — ShapeNetSem has instances of these too,
but PartNet already covers them abundantly (knife 514, scissors 127 real
models with full part hierarchies); not worth the redundant extraction here.

Run:  python tools/extract_shapenetsem_gap_categories.py
"""
import csv
import json
import os
import zipfile

_META = "D:/Aaron/datasets/shapenetsem/metadata.csv"
_ZIP = "D:/Aaron/datasets/shapenet/ShapeNetSem-archive/ShapeNetSem.zip"
_ZIP_PREFIX = "ShapeNetSem-backup/"
_DEST = "D:/Aaron/datasets/shapenetsem/gap_categories"
_OUT = os.path.join(os.path.dirname(__file__), "..", "sigma_ground",
                    "inventory", "data", "shapenetsem_gap_models.json")

# category substring -> our canonical name. These are the ones confirmed
# ABSENT from PartNet (this session) and from our Objaverse LVIS pull.
TARGETS = {"hammer": "hammer", "screwdriver": "screwdriver",
          "motorcycle": "motorcycle"}


def main() -> None:
    with open(_META, encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))

    picks = []
    for row in rows:
        cat = (row.get("category") or "").lower()
        for needle, name in TARGETS.items():
            if needle in cat:
                picks.append((name, row))
                break

    print(f"{len(picks)} models across {len(TARGETS)} target categories")
    ledger = {"_meta": {
        "source": "ShapeNetSem (a per-model-annotated ShapeNet subset, "
                  "distinct from PartNet/ShapeNetCore) — ShapeNet ToU, "
                  "non-commercial research",
        "note": "fullId's hash is the SAME id scheme as ShapeNetCore/PartNet "
                "model_id (verified: PartNet chair 44164 model_id == "
                "ShapeNetCore.v2/03001627/<same hash>) -- usable as a join "
                "key across the three datasets.",
        "generated_by": "tools/extract_shapenetsem_gap_categories.py"}}

    with zipfile.ZipFile(_ZIP) as z:
        names = z.namelist()
        for name, row in picks:
            mid = row["fullId"].split(".", 1)[-1]           # strip "wss."
            members = [n for n in names if f"/{mid}." in n or f"/{mid}/" in n]
            obj_mtl = [n for n in members if "models-OBJ" in n]
            solid = [n for n in members if "models-binvox-solid" in n]
            wanted = obj_mtl + solid
            for m in wanted:
                z.extract(m, _DEST)
            up = (row.get("up") or "").replace("\\,", ",") or None
            front = (row.get("front") or "").replace("\\,", ",") or None
            unit = row.get("unit") or None
            ledger.setdefault(name, []).append({
                "full_id": mid, "up": up, "front": front,
                "unit_m_per_native": float(unit) if unit else None,
                "aligned_dims_cm": (row.get("aligned.dims") or "").replace("\\,", ","),
                "tags": row.get("tags") or "",
                "files_extracted": len(wanted)})
            print(f"  + {name:12s} {mid}  up={up!r} unit={unit!r}  "
                  f"{len(wanted)} files")

    for name in TARGETS.values():
        print(f"{name}: {len(ledger.get(name, []))} extracted")
    dest = os.path.abspath(_OUT)
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(ledger, fh, indent=1)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
