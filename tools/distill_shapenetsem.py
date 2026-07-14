"""Distill ShapeNetSem physical metadata into a Deckard fact-table.

ShapeNetSem (Savva et al. 2015; a ShapeNet subset) annotates ~12k models with
REAL-WORLD physical attributes. The archive's CSV export carries, per model:
consistently-aligned dimensions (``aligned.dims``, in cm, '\\,'-separated),
category tokens + WordNet lemmas — plus per-CATEGORY material composition
ratios (materials.csv) and a material->density table (densities.csv, g/cm3).
(The per-model weight/solidVolume/isContainerLike columns are empty in this
export — verified 2026-06-11 — so mass stays a derived quantity, not a cited
one.)

The data is access-gated under the ShapeNet Terms of Use (non-commercial
research; see docs/DATA_SOURCES.md — terms documented BEFORE pulling). We honor
the distill-don't-redistribute design:

  * ``fetch``   — pull ONLY the metadata CSVs, range-read out of the remote
                  12 GB ``ShapeNetSem.zip`` on Hugging Face (zipfile over a
                  seekable HfFileSystem handle transfers just the needed
                  members, never the archive). Requires approved access +
                  the ``HF_ID`` token in the dev-root vault.
  * ``distill`` — reduce the per-model CSVs to an AGGREGATE fact-table
                  (``shapenetsem_sizes.json``): per name (category token,
                  WordNet lemma, or synset alias) the MEDIAN aligned dims in
                  meters + sample count; per name the category material
                  ratios; and the material density table in kg/m3. Derived
                  facts only — no raw rows, no meshes, nothing redistributable.

    python tools/distill_shapenetsem.py fetch
    python tools/distill_shapenetsem.py distill [dir-with-csvs]

Attribution: "ShapeNetSem (Savva et al. 2015) / ShapeNet (Chang et al. 2015)".
"""
from __future__ import annotations

import csv
import json
import pathlib
import statistics
import sys

_REPO = "datasets/ShapeNet/ShapeNetSem-archive"
_ZIP = "ShapeNetSem.zip"
_ENV = pathlib.Path("D:/Aaron/development/.env")
_RAW_DIR = pathlib.Path("D:/datasets/shapenetsem")      # outside any git repo
_OUT = (pathlib.Path(__file__).resolve().parents[1]
        / "sigma_ground" / "inventory" / "data" / "shapenetsem_sizes.json")

_MIN_N = 3                  # names with fewer usable samples are dropped
_MAX_DIM_M = 100.0          # sanity clamp: reject mis-scaled rows
_MAX_WEIGHT_KG = 50_000.0
_MAX_MEMBER_MB = 200        # never extract a zip member bigger than this (meshes)
_MIN_RATIO = 0.02           # material ratios below this are noise


def _token() -> str | None:
    try:
        for line in _ENV.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.strip().startswith("HF_ID="):
                return line.strip().split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    return None


def fetch() -> list[pathlib.Path]:
    """Extract just the CSVs (+ README) from the remote zip via ranged reads."""
    import zipfile
    from huggingface_hub import HfFileSystem      # fetch-only optional dep

    fs = HfFileSystem(token=_token())
    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    got: list[pathlib.Path] = []
    with fs.open(f"{_REPO}/{_ZIP}") as remote:    # seekable -> HTTP range reads
        zf = zipfile.ZipFile(remote)
        wanted = [n for n in zf.namelist()
                  if n.lower().endswith(".csv") or n.lower().endswith("readme.txt")]
        if not wanted:
            raise SystemExit("no CSV/README members found — zip layout changed, inspect manually")
        for n in wanted:
            info = zf.getinfo(n)
            if info.file_size > _MAX_MEMBER_MB * 1024 * 1024:
                print(f"skip {n} ({info.file_size/1e6:.0f} MB > {_MAX_MEMBER_MB} MB cap)")
                continue
            out = _RAW_DIR / pathlib.Path(n).name
            with zf.open(n) as src:
                out.write_bytes(src.read())
            got.append(out)
            print(f"extracted {n} -> {out} ({info.file_size/1e6:.1f} MB)")
    return got


def _split_dims(raw: str) -> list[float]:
    """ShapeNetSem joins dims with an ESCAPED comma ('111.97\\,84.16\\,0.0')."""
    parts = raw.split("\\,") if "\\," in raw else raw.split(",")
    return [float(x) for x in parts[:3]]


def _names_of(row: dict, aliases: dict | None) -> tuple[set, set]:
    """(all name keys, category tokens) a metadata row contributes to."""
    names, cats = set(), set()
    for nm in (row.get("category") or "").split(","):
        nm = nm.strip().lower()
        if nm and not nm.startswith("_"):                 # '_Attributes' etc. are tags
            cats.add(nm)
            names.add(nm)
            for alias in (aliases or {}).get(nm, ()):
                names.add(alias)
    for nm in (row.get("wnlemmas") or "").split(","):
        nm = nm.strip().lower()
        if nm:
            names.add(nm)
    return names, cats


def aggregate(rows, aliases: dict | None = None) -> tuple[dict, dict]:
    """Per-name median real-world size (+ weight when present) from Sem rows.

    ``aligned.dims`` is centimeters (verified against real data) -> meters.
    A row contributes to every name in its ``category`` / ``wnlemmas`` columns
    plus any synset ``aliases`` of its categories. Junk rows are skipped.
    Returns (per-name stats, per-name contributing-category counts).
    """
    cats: dict[str, dict[str, list]] = {}
    name_cats: dict[str, dict[str, int]] = {}
    for r in rows:
        raw = (r.get("aligned.dims") or "").strip()
        if not raw:
            continue
        # scale-verified rows only: a model with no `unit` kept ShapeNetSem's
        # DEFAULT scale, and its aligned.dims are not real-world (the famous
        # 1.4-metre pushpin came from 7 unitless models). ~9.5k of 12.3k
        # models carry a verified unit — plenty, and honest.
        if "unit" in r and not (r.get("unit") or "").strip():
            continue
        try:
            dims = _split_dims(raw)
        except ValueError:
            continue
        if len(dims) != 3 or any(d <= 0.0 for d in dims):
            continue
        dims_m = [d / 100.0 for d in dims]                    # cm -> m
        if max(dims_m) > _MAX_DIM_M:
            continue
        weight = None
        w = (r.get("weight") or "").strip()
        if w:
            try:
                weight = float(w)
                if not (0.0 < weight < _MAX_WEIGHT_KG):
                    weight = None
            except ValueError:
                weight = None
        names, row_cats = _names_of(r, aliases)
        for nm in names:
            c = cats.setdefault(nm, {"dims": [], "weight": []})
            c["dims"].append(dims_m)
            if weight is not None:
                c["weight"].append(weight)
            nc = name_cats.setdefault(nm, {})
            for cat in row_cats:
                nc[cat] = nc.get(cat, 0) + 1

    out = {}
    for nm, c in sorted(cats.items()):
        if len(c["dims"]) < _MIN_N:
            continue
        med = [round(statistics.median(d[i] for d in c["dims"]), 4) for i in range(3)]
        entry = {"dims_m": med, "size_m": round(max(med), 4), "n": len(c["dims"])}
        if len(c["weight"]) >= _MIN_N:
            entry["weight_kg"] = round(statistics.median(c["weight"]), 4)
        out[nm] = entry
    return out, name_cats


def synset_aliases(path: pathlib.Path) -> dict:
    """category token (lower) -> its WordNet synset words ('1Shelves' -> shelf)."""
    out: dict[str, list] = {}
    try:
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            for r in csv.DictReader(f):
                cat = (r.get("category") or "").strip().lower()
                words = [w.strip().lower() for w in (r.get("synset words") or "").split(",")]
                words = [w for w in words if w]
                if cat and words:
                    out.setdefault(cat, [])
                    for w in words:
                        if w not in out[cat]:
                            out[cat].append(w)
    except OSError:
        pass
    return out


def material_ratios(path: pathlib.Path) -> dict:
    """category token (lower) -> {material (lower): ratio} from materials.csv."""
    out: dict[str, dict[str, float]] = {}
    try:
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            for r in csv.DictReader(f):
                cat = (r.get("Category") or "").strip().lower()
                mat = (r.get("Material") or "").strip().lower()
                try:
                    ratio = float((r.get("Ratio") or "").strip())
                except ValueError:
                    continue
                if cat and mat and ratio >= _MIN_RATIO:
                    out.setdefault(cat, {})[mat] = round(ratio, 4)
    except OSError:
        pass
    return out


def material_densities(path: pathlib.Path) -> dict:
    """material (lower) -> density kg/m3 (densities.csv is g/cm3)."""
    out: dict[str, float] = {}
    try:
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            for r in csv.DictReader(f):
                mat = (r.get("Material") or "").strip().lower()
                try:
                    out[mat] = round(float((r.get("Density") or "").strip()) * 1000.0, 1)
                except ValueError:
                    continue
    except OSError:
        pass
    return out


def _blend_materials(name_cats: dict, ratios: dict) -> dict:
    """Per name, the sample-count-weighted blend of its categories' materials."""
    out: dict[str, dict[str, float]] = {}
    for nm, ccounts in name_cats.items():
        acc: dict[str, float] = {}
        total = 0
        for cat, n in ccounts.items():
            mats = ratios.get(cat)
            if not mats:
                continue
            total += n
            for mat, ratio in mats.items():
                acc[mat] = acc.get(mat, 0.0) + ratio * n
        if total:
            blended = {m: round(v / total, 3) for m, v in acc.items()}
            out[nm] = dict(sorted(((m, v) for m, v in blended.items() if v >= _MIN_RATIO),
                                  key=lambda kv: -kv[1]))
    return out


def build(raw_dir: pathlib.Path) -> dict:
    aliases = synset_aliases(raw_dir / "categories.synset.csv")
    with open(raw_dir / "metadata.csv", newline="", encoding="utf-8", errors="replace") as f:
        sizes, name_cats = aggregate(csv.DictReader(f), aliases)
    mats = _blend_materials(name_cats, material_ratios(raw_dir / "materials.csv"))
    for nm, m in mats.items():
        if nm in sizes:
            sizes[nm]["materials"] = m
    return {
        "_source": "ShapeNetSem (Savva et al. 2015) / ShapeNet (Chang et al. 2015)",
        "_license": "ShapeNet Terms of Use — non-commercial research; "
                    "derived aggregate facts only, no raw rows",
        "_method": f"per-name median of aligned.dims (cm->m), n>={_MIN_N}; "
                   "category material ratios blended per name; "
                   "densities.csv g/cm3 -> kg/m3",
        "categories": sizes,
        "material_densities": material_densities(raw_dir / "densities.csv"),
    }


def main(argv: list[str]) -> None:
    cmd = argv[0] if argv else "distill"
    if cmd == "fetch":
        got = fetch()
        print(f"\n{len(got)} files in {_RAW_DIR} — now run: "
              f"python tools/distill_shapenetsem.py distill")
    elif cmd == "distill":
        raw = pathlib.Path(argv[1]) if len(argv) > 1 else _RAW_DIR
        doc = build(raw)
        _OUT.write_text(json.dumps(doc, indent=1, sort_keys=True), encoding="utf-8")
        cats = doc["categories"]
        with_mat = sum(1 for v in cats.values() if "materials" in v)
        print(f"wrote {_OUT}")
        print(f"  {len(cats)} names ({with_mat} with material priors), "
              f"{len(doc['material_densities'])} material densities")
    else:
        raise SystemExit("usage: distill_shapenetsem.py [fetch|distill [dir]]")


if __name__ == "__main__":
    main(sys.argv[1:])
