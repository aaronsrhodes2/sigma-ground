"""Distill ShapeNetSem physical metadata into a Deckard fact-table.

ShapeNetSem (Savva et al. 2015; a ShapeNet subset) annotates ~12k models with
REAL-WORLD physical attributes: consistently-aligned dimensions, weight, volume,
category + WordNet lemmas. That is exactly Deckard's missing grounding: typical
SIZE and MASS per nameable category.

The data is access-gated under the ShapeNet Terms of Use (non-commercial
research; see docs/DATA_SOURCES.md — terms documented BEFORE pulling). We honor
the distill-don't-redistribute design:

  * ``fetch``   — pull ONLY the metadata CSVs, range-read out of the remote
                  12 GB ``ShapeNetSem.zip`` on Hugging Face (zipfile over a
                  seekable HfFileSystem handle transfers just the needed
                  members, never the archive). Requires approved access +
                  the ``HF_ID`` token in the dev-root vault.
  * ``distill`` — reduce the per-model CSV to an AGGREGATE fact-table:
                  per category lemma, the MEDIAN aligned dims (m), median
                  weight (kg), and sample count. Derived facts only — no
                  raw rows, no meshes, nothing redistributable.

    python tools/distill_shapenetsem.py fetch
    python tools/distill_shapenetsem.py distill [metadata.csv]

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
_RAW_DIR = pathlib.Path("D:/Aaron/datasets/shapenetsem")      # outside any git repo
_OUT = (pathlib.Path(__file__).resolve().parents[1]
        / "sigma_ground" / "inventory" / "data" / "shapenetsem_sizes.json")

_MIN_N = 3                  # categories with fewer usable samples are dropped
_MAX_DIM_M = 100.0          # sanity clamp: reject mis-scaled rows
_MAX_WEIGHT_KG = 50_000.0
_MAX_MEMBER_MB = 200        # never extract a zip member bigger than this (meshes)


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


def aggregate(rows) -> dict:
    """Per-category median real-world size + weight from ShapeNetSem rows.

    ``aligned.dims`` is read as CENTIMETERS (the ShapeNetSem convention) and
    converted to meters — verify against the shipped README on the first real
    run. A row contributes to every name in its ``category`` and ``wnlemmas``
    columns. Junk rows (missing/non-positive/implausible) are skipped.
    """
    cats: dict[str, dict[str, list]] = {}
    for r in rows:
        raw = (r.get("aligned.dims") or "").strip()
        if not raw:
            continue
        try:
            dims = [float(x) for x in raw.split(",")[:3]]
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
        names = set()
        for col in ("category", "wnlemmas"):
            for nm in (r.get(col) or "").split(","):
                nm = nm.strip().lower()
                if nm:
                    names.add(nm)
        for nm in names:
            c = cats.setdefault(nm, {"dims": [], "weight": []})
            c["dims"].append(dims_m)
            if weight is not None:
                c["weight"].append(weight)

    out = {}
    for nm, c in sorted(cats.items()):
        if len(c["dims"]) < _MIN_N:
            continue
        med = [round(statistics.median(d[i] for d in c["dims"]), 4) for i in range(3)]
        entry = {"dims_m": med, "size_m": round(max(med), 4), "n": len(c["dims"])}
        if len(c["weight"]) >= _MIN_N:
            entry["weight_kg"] = round(statistics.median(c["weight"]), 4)
        out[nm] = entry
    return out


def write_table(agg: dict) -> pathlib.Path:
    doc = {
        "_source": "ShapeNetSem (Savva et al. 2015) / ShapeNet (Chang et al. 2015)",
        "_license": "ShapeNet Terms of Use — non-commercial research; "
                    "derived aggregate facts only, no raw rows",
        "_method": f"per-category median of aligned.dims (cm->m) + weight (kg); n>={_MIN_N}",
        "categories": agg,
    }
    _OUT.write_text(json.dumps(doc, indent=1, sort_keys=True), encoding="utf-8")
    return _OUT


def main(argv: list[str]) -> None:
    cmd = argv[0] if argv else "distill"
    if cmd == "fetch":
        got = fetch()
        print(f"\n{len(got)} files in {_RAW_DIR} — now run: "
              f"python tools/distill_shapenetsem.py distill")
    elif cmd == "distill":
        src = pathlib.Path(argv[1]) if len(argv) > 1 else _RAW_DIR / "metadata.csv"
        with open(src, newline="", encoding="utf-8", errors="replace") as f:
            agg = aggregate(csv.DictReader(f))
        p = write_table(agg)
        sized = sum(1 for v in agg.values() if "weight_kg" in v)
        print(f"wrote {p} — {len(agg)} categories ({sized} with weight)")
    else:
        raise SystemExit("usage: distill_shapenetsem.py [fetch|distill [metadata.csv]]")


if __name__ == "__main__":
    main(sys.argv[1:])
