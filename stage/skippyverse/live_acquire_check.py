"""Live proof, run by hand (NOT part of pytest): acquire and measure 3 real assets, one per source, and compare the Poly Haven
measurement to Poly Haven's published dimensions. Downloads a few MB in total.

    python stage/skippyverse/live_acquire_check.py [--cache DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from skippyverse import acquire as A  # noqa: E402
from skippyverse import ingest_real  # noqa: E402

PICKS = {"nasa": "found.nasa-3d-resources.3d-models-hammer-hammer-glb",
         "fuel": "found.gazebo-fuel.openrobotics-fire-hydrant",
         "polyhaven": "found.poly-haven.barrel-03",
         "polyhaven_asymmetric": "found.poly-haven.dining-chair-02"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(HERE / "cache" / "assets"))
    a = ap.parse_args(argv)
    cards = {c["id"]: c for c in ingest_real.build()[0]}
    total, rows = 0, {}
    for label, cid in PICKS.items():
        card = cards.get(cid)
        if card is None:
            print(f"{label}: card {cid} not found"); continue
        r = A.acquire_and_measure(card, cache_dir=a.cache)
        total += r["bytes"]
        s = r["card"]["size_m"]
        rows[label] = {"id": cid, "mb": round(r["bytes"] / 1e6, 3), "size_m": {k: round(s[k], 4) for k in "wdh"}, "basis": s["basis"],
                       "published_size_m": {k: round(card["size_m"][k], 4) for k in "wdh"} if card["size_m"] else None,
                       "format": r["measurement"]["format"], "say": r["card"]["say"]}
        print(json.dumps(rows[label], indent=1))
    print(f"total downloaded: {total / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
