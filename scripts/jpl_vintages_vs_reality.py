"""JPL's prediction at the time against reality, with no integration: every past JPL ephemeris (DE403 ... DE440) at every yearly start after its release and every lead 1..5 yr, the
heliocentric distance from the truth kernel (DE442), plus reality's own spread (DE440 against DE442) and the start offset. Kernel lookups only; seconds to run.

Writes a jsonl in the fair contest's schema with no `eO` (ours comes from the integration, scripts/fair_jpl_contest.py --predictor): {vintage, start, jd, bodies: {body: {lead: {eV, eT, e0}}}, truth, kind}.
The portal's blanket builder merges it with a contest run.

usage: python scripts/jpl_vintages_vs_reality.py [--out D:/datasets/sigma_runs/contest_v2/jpl_vs_reality.jsonl] [--max-lead 5] [--last-target 2025]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))
BODIES = {"Mercury": 1, "Venus": 2, "Earth": 3, "Moon": 3, "Mars": 4, "Jupiter": 5, "Saturn": 6, "Uranus": 7, "Neptune": 8, "Pluto": 9}


def jd_of(y: int) -> float:
    return dt.date(y, 1, 1).toordinal() + 1721424.5


def hel(k, body: str, jd: float) -> np.ndarray | None:
    try:
        p = k[0, BODIES[body]].compute(jd) - k[0, 10].compute(jd)
        if body == "Earth":
            p = p + k[3, 399].compute(jd)
        elif body == "Moon":
            p = p + k[3, 301].compute(jd)
        return p
    except (KeyError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("D:/datasets/sigma_runs/contest_v2/jpl_vs_reality.jsonl"))
    ap.add_argument("--max-lead", type=int, default=5)
    ap.add_argument("--last-target", type=int, default=2025)
    a = ap.parse_args()
    from jplephem.spk import SPK
    from sigma_ground.field import datasets
    from vintage_deviance_mesh import V_DIR, VINTAGES
    truth_file, truth_name = datasets.truth_kernel()
    kt = SPK.open(str(truth_file))
    k440 = SPK.open(str(datasets.de440_kernel()))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with a.out.open("w", encoding="utf-8") as fh:
        for name, f, rel in VINTAGES:
            kv = SPK.open(str(V_DIR / f))
            kend = kv[0, 3].end_jd
            for y in range(rel.year + (1 if rel.month > 1 or rel.day > 1 else 0), a.last_target):
                jd0 = jd_of(y)
                if jd0 < rel.toordinal() + 1721424.5 or jd0 > kend:
                    continue
                bodies = {}
                for b in BODIES:
                    p0v, p0t = hel(kv, b, jd0), hel(kt, b, jd0)
                    if p0v is None or p0t is None:
                        continue
                    e0 = float(np.linalg.norm(p0v - p0t))
                    cells = {}
                    for L in range(1, a.max_lead + 1):
                        if y + L > a.last_target:
                            break
                        jd1 = jd_of(y + L)
                        if jd1 > kend:
                            break
                        pv, pt, p4 = hel(kv, b, jd1), hel(kt, b, jd1), hel(k440, b, jd1)
                        if pv is None or pt is None:
                            break
                        cells[str(L)] = {"eV": float(np.linalg.norm(pv - pt)), "eT": float(np.linalg.norm(p4 - pt)) if p4 is not None else None, "e0": e0}
                    if cells:
                        bodies[b] = cells
                if bodies:
                    fh.write(json.dumps({"vintage": name, "start": f"j{y}", "jd": jd0, "bodies": bodies, "truth": truth_name, "kind": "kernels_only"}) + "\n")
                    rows += 1
            kv.close()
    print(f"wrote {a.out}: {rows} vintage-start rows against {truth_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
