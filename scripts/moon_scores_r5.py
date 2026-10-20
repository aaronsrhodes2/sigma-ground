"""Score the moons of the r5 run (shoot-out windows J2010 and J2015, step 0.0125 d, candidate stack) against the DE440 fixture, by body and lead time.

Writes MOON_SCORES_R5.md and MOON_SCORES_R5.json beside the raw run. Errors are the norm of (predicted - truth) in km from the stored positions (the run's error_au field is not used).

usage: python scripts/moon_scores_r5.py [--run FILE.json] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parent.parent / "sigma_ground/field/interface/fixtures/v2physics_restated"
SLOW_NEW = ("Himalia", "Nereid", "Helene")
REFERENCE = ("Moon", "Titan", "Io", "Triton", "Pluto", "Jupiter")      # already-scored bodies, for scale


def scores(run: Path) -> dict:
    d = json.loads(run.read_text())
    cells = collections.defaultdict(dict)            # (body, window) -> {lead_yr: err_km}
    for r in d["samples"]:
        yrs = round((r["sample_jd"] - r["window_start_jd"]) / 365.25)
        cells[(r["body"], r["window_start_jd"])][yrs] = float(np.linalg.norm(np.array(r["predicted_km"]) - np.array(r["truth_km"])))
    out = {}
    for (body, w), v in cells.items():
        out.setdefault(body, {})[str(w)] = {str(k): v[k] for k in sorted(v)}
    return {"dt_days": d["metadata"]["dt_days"], "bodies": out}


def table(s: dict) -> str:
    lines = ["| body | window (JD) | +1 yr | +2 yr | +3 yr | +4 yr | +5 yr |", "|---|---|---|---|---|---|---|"]
    for body in SLOW_NEW + REFERENCE:
        for w, v in sorted(s["bodies"].get(body, {}).items()):
            lines.append(f"| {body} | {float(w):.1f} | " + " | ".join(f"{v[str(k)]:,.2f}" for k in range(1, 6)) + " |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, default=RES / "rolling_shootout_r5_moons_0.0125.json")
    ap.add_argument("--out-dir", type=Path, default=RES)
    a = ap.parse_args()
    s = scores(a.run)
    (a.out_dir / "MOON_SCORES_R5.json").write_text(json.dumps(s, indent=1))
    med5 = {b: statistics.median(v["5"] for v in s["bodies"][b].values()) for b in SLOW_NEW + REFERENCE}
    md = ("# Moons scored on the r5 fixture (step 0.0125 d)\n\nCandidate stack (DE440 asteroids, ring, seven KBOs, lunar C22), windows J2010 and J2015, each propagated 5 yr from the fixture state "
          "and compared with the fixture truth at yearly epochs. Position error in km.\n\n" + table(s) + "\n\nMedian of the two windows at +5 yr: " + ", ".join(f"{b} {m:,.1f} km" for b, m in med5.items()) + "\n")
    (a.out_dir / "MOON_SCORES_R5.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
