"""Build fixtures/v2physics_restated/SYSTEM_LOCAL_WITH_PLANETS.json from the C23a runs: the system-local moon model without (arm A) and with (arm B) the other planets and the Moon.

usage: python scripts/make_system_local_with_planets.py [--runs D:/datasets/sigma_runs] [--check]
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
OUT = _ROOT / "sigma_ground/field/interface/fixtures/v2physics_restated/SYSTEM_LOCAL_WITH_PLANETS.json"


def build(runs: Path) -> dict:
    entries = []
    for arm in ("A", "B"):
        per: dict = {}
        for f in sorted(glob.glob(str(runs / f"hl_{arm}" / "*.json"))):
            d = json.loads(Path(f).read_text(encoding="utf-8"))[0]
            for moon, v in d["err_km"].items():
                r = per.setdefault((d["system"], moon), {"period_d": d["period_days"][moon], "dt_d": d["dt_days"], "windows": {}})
                r["windows"][d["window"]] = v
        for (system, moon), r in sorted(per.items()):
            med = lambda i: statistics.median(w[i] for w in r["windows"].values())
            entries.append({"arm": arm, "system": system, "moon": moon, "period_d": r["period_d"], "dt_d": r["dt_d"], "e1": med(1), "e3": med(3), "e5": med(5), "windows": r["windows"]})
    return {"what": "C23a: the system-local moon model without (arm A: Sun, planet, moons) and with (arm B: A plus every other planet and the Moon as unscored perturbers) the other bodies; reduced membership, "
                    "r8 fixture, step shortest period / 300 of the reduced set, medians over windows j2010 and j2015",
            "fixture": "de440_state_vectors_v2physics_r8.json", "ledger": "misc/CANDIDATE_LEDGER.md C23a", "generator": "scripts/make_system_local_with_planets.py", "entries": entries}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("D:/datasets/sigma_runs"))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    doc = build(a.runs)
    if a.check:
        same = json.loads(OUT.read_text(encoding="utf-8")) == json.loads(json.dumps(doc))
        print("fixture matches the runs" if same else "fixture DIFFERS from the runs")
        return 0 if same else 1
    OUT.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print("wrote", OUT, len(doc["entries"]), "entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
