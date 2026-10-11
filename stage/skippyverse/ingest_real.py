"""Real listings -> cards -> index. Fetch (official public endpoints, no keys) with --fetch, else read stage/skippyverse/cache/.

    python stage/skippyverse/ingest_real.py [--fetch]      # writes build/cards.json and build/index/

Sources (all verified against the live endpoints on 2026-10-10):
    Poly Haven   https://api.polyhaven.com/assets?t=models           521 models, CC0, real sizes in mm
    Gazebo Fuel  https://fuel.gazebosim.org/1.0/models?per_page=100  3387 models; CC0 + CC-BY-4.0 kept; no sizes in the listing
    NASA 3D      https://api.github.com/repos/nasa/NASA-3D-Resources/git/trees/master?recursive=1   256 glb folders; no sizes
Kenney has no machine-readable listing (its asset pages are script-rendered), so it is not ingested yet.
Local cards (curated Valve models, Deckard parts, ShapeNetSem placeholders) are never servable; they stay in the index so the
resolver can say what it knows and what it may not serve.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from skippyverse import ingest_local, index as idx  # noqa: E402
from skippyverse.sources import gazebo_fuel, nasa3d, polyhaven  # noqa: E402

CACHE = HERE / "cache"
UA = {"User-Agent": "SkippyVerse-ingest"}


def _get(url: str):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return json.load(r)


def fetch_all() -> None:
    CACHE.mkdir(exist_ok=True)
    (CACHE / "polyhaven_models.json").write_text(json.dumps(_get("https://api.polyhaven.com/assets?t=models")), encoding="utf-8")
    out, page = [], 1
    while True:
        d = _get(f"https://fuel.gazebosim.org/1.0/models?per_page=100&page={page}")
        out += d
        if len(d) < 100:
            break
        page += 1
        time.sleep(0.2)
    (CACHE / "fuel_models.json").write_text(json.dumps(out), encoding="utf-8")
    (CACHE / "nasa_tree.json").write_text(json.dumps(_get("https://api.github.com/repos/nasa/NASA-3D-Resources/git/trees/master?recursive=1")), encoding="utf-8")


def _load(name: str):
    return json.loads((CACHE / name).read_text(encoding="utf-8"))


def build(out_dir: Path | None = None) -> tuple[list[dict], dict]:
    cards, report = ingest_local.ingest_all()
    found = {}
    for label, got in (
        ("polyhaven", polyhaven.parse_listing_with_report(_load("polyhaven_models.json"))),
        ("fuel", gazebo_fuel.parse_listing_with_report(_load("fuel_models.json"))),
        ("nasa", nasa3d.parse_listing_with_report(nasa3d.from_github_tree(_load("nasa_tree.json")))),
    ):
        kept, rep = got
        found[label] = {"cards": len(kept), "skipped_by_licence_gate": len(rep["skipped"]), "problems": len(rep["problems"]),
                        "with_size": sum(1 for c in kept if c["size_m"])}
        cards += kept
    ids = Counter(c["id"] for c in cards)
    dup = [k for k, v in ids.items() if v > 1]
    if dup:
        raise SystemExit(f"duplicate card ids: {dup[:5]}")
    report["found"] = found
    report["total"] = len(cards)
    report["servable"] = sum(1 for c in cards if c["redistributable"])
    return cards, report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="download the listings first (official public endpoints, no keys)")
    ap.add_argument("--out", default=str(HERE / "build"))
    a = ap.parse_args(argv)
    if a.fetch:
        fetch_all()
    cards, report = build()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cards.json").write_text(json.dumps(cards), encoding="utf-8")
    ix = idx.build_index(cards, out / "index")
    print(json.dumps({k: report[k] for k in ("found", "total", "servable")}, indent=1))
    print("index:", ix if isinstance(ix, dict) else "built")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
