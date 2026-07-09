"""Selective extraction of target ShapeNetCore synsets + a census.

ShapeNetCore covers real ENGINEERING assemblies PartNet's furniture/household
scope does not: cars, airplanes, guitars, watercraft, motorcycles. Unlike
PartNet, ShapeNetCore has no per-part hierarchy — these are single-solid
whole-object meshes, useful for shape BREADTH (a real car body, a real
guitar), not per-part rigid decomposition.

Notable find worth flagging to the main lane: every model ships a
pre-computed model_normalized.solid.binvox — an ALREADY-SOLID voxel grid from
a trusted, well-established solid voxelizer (binvox/Patrick Min's tool),
sidestepping our fragile per-part-fill workaround entirely for these objects.
We extract .obj (+ .mtl + texture images) AND .solid.binvox per sampled model.

Source: the per-synset zips already on disk at
D:/Aaron/datasets/shapenet/ShapeNetCore/<synsetId>.zip (smaller/faster than
opening the full ShapeNetCore.v2.zip). Selective two-phase extraction (S1.1
pattern): list members first, then extract only a bounded SAMPLE of models
per synset (not the full ~500-4000 instances) — kept small deliberately;
this is shape-breadth seeding, not a bulk pull.

Output: models land under D:/Aaron/datasets/shapenet/ShapeNetCore/<synsetId>/
(local, gitignored). A census JSON (synset -> name -> sampled model ids ->
member counts) is the committable distilled artifact.

Run:  python tools/extract_shapenetcore_synsets.py
"""
import json
import os
import zipfile

_ROOT = "D:/Aaron/datasets/shapenet/ShapeNetCore"
_OUT = os.path.join(os.path.dirname(__file__), "..", "sigma_ground",
                    "inventory", "data", "shapenetcore_synset_census.json")

# (synsetId, human name, sample count) — real engineering assemblies PartNet
# lacks entirely.
TARGETS = [
    ("02958343", "car", 30),
    ("02691156", "airplane", 30),
    ("03467517", "guitar", 20),
    ("04530566", "watercraft", 20),
    ("03790512", "motorcycle", 20),
]
_WANT_SUFFIXES = (".obj", ".mtl", ".solid.binvox")


def _extract_synset(synset_id, name, n_sample):
    zpath = os.path.join(_ROOT, f"{synset_id}.zip")
    if not os.path.exists(zpath):
        print(f"  ! {name} ({synset_id}): zip not on disk — skipped")
        return None
    out_dir = os.path.join(_ROOT, synset_id)
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        model_ids = sorted({n.split("/")[1] for n in names
                            if "/" in n and len(n.split("/")) > 1
                            and n.split("/")[1]})
        sample = model_ids[:n_sample]
        extracted = {}
        for mid in sample:
            prefix = f"{synset_id}/{mid}/"
            members = [n for n in names if n.startswith(prefix)
                      and (n.endswith(_WANT_SUFFIXES) or "/images/" in n)]
            for m in members:
                z.extract(m, _ROOT)
            extracted[mid] = len(members)
    total_files = sum(extracted.values())
    print(f"  + {name:12s} ({synset_id}): {len(sample)}/{len(model_ids)} models "
          f"sampled, {total_files} files extracted -> {out_dir}")
    return {"synset_id": synset_id, "name": name, "total_models_in_zip": len(model_ids),
            "sampled_model_ids": sample, "files_extracted": total_files}


def main() -> None:
    out = {"_meta": {
        "source": "ShapeNetCore v2 (per-synset zips, ShapeNet ToU — "
                  "non-commercial research; local only)",
        "note": "each sampled model ships model_normalized.obj + .mtl + "
                "texture images + a pre-computed model_normalized.solid."
                "binvox (trusted solid voxelization, no fill heuristic "
                "needed for these objects)",
        "generated_by": "tools/extract_shapenetcore_synsets.py"}}
    for synset_id, name, n in TARGETS:
        got = _extract_synset(synset_id, name, n)
        if got:
            out[name] = got
    dest = os.path.abspath(_OUT)
    json.dump(out, open(dest, "w", encoding="utf-8"), indent=1)
    print(f"\nwrote {dest} ({len(out) - 1} synsets)")


if __name__ == "__main__":
    main()
