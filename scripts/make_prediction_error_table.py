"""Measure how far the forecast model (reduced_model, predictor `restated_zon`, step 0.25 d) lands from DE440 at each lead time, by hindcast: start it from DE440's own state at several past epochs and compare with DE440
at +1, +2 ... years. This is the error Mentat quotes with a prediction ("our integration is typically X km from DE440 at this lead"); it is measured with the same code that builds the forecast table.

Starts are chosen so that every target lies inside the span DE440 was fitted to (data end about 2020), so the comparison is with JPL's reconstruction and not with JPL's own extrapolation. Output: fixtures/prediction_error_table.json
(per body and lead: median, maximum and the number of starts). Heliocentric position error in km; Mars to Pluto are system barycentres, like the forecast.

usage: python scripts/make_prediction_error_table.py [--starts 2000 2002 2004 2006 2008 2010 2012 2015] [--max-lead 20]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
OUT = _ROOT / "sigma_ground/field/interface/fixtures/prediction_error_table.json"
DATA_END_YEAR = 2020.99                                  # the fit data end about 2020 (Park et al. 2021): targets stay at or before this


def jd_jan1(year: int) -> float:
    import datetime as dt
    return dt.date(year, 1, 1).toordinal() + 1721424.5


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--starts", nargs="+", type=int, default=[2000, 2002, 2004, 2006, 2008, 2010, 2012, 2015])
    ap.add_argument("--max-lead", type=int, default=20)
    ap.add_argument("--predictor", default="restated_zon")
    ap.add_argument("--dt-days", type=float, default=0.25)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    from jplephem.spk import SPK
    from sigma_ground.field import datasets
    from sigma_ground.field.interface import reduced_model as rm
    kernel = SPK.open(str(datasets.de440_kernel()))
    cells: dict[str, dict[int, list[float]]] = {}
    per_start = {}
    for y in a.starts:
        t0 = time.time()
        jd0 = jd_jan1(y)
        leads = [k for k in range(1, a.max_lead + 1) if y + k <= DATA_END_YEAR]
        if not leads:
            continue
        jd1 = jd0 + leads[-1] * 365.25
        system, order = rm.build_system(kernel, jd0, a.predictor, dt_days=a.dt_days, horizon_days=jd1 - jd0 + 10.0, fixture_key="j2010")
        every = int(round(365.25 / a.dt_days))
        d = rm.propagate_dense(system, jd0, jd1, a.dt_days, every=every)
        sun_i = order.index("Sun")
        per_start[y] = {}
        for row, jd in enumerate(d["jd"]):
            k = int(round((jd - jd0) / 365.25))
            if k < 1 or k not in leads:
                continue
            for i, n in enumerate(order):
                if n == "Sun":
                    continue
                truth = rm.kernel_state(kernel, n, jd)[0] - rm.kernel_state(kernel, "Sun", jd)[0]
                e = float(np.linalg.norm((d["pos_km"][row, i] - d["pos_km"][row, sun_i]) - truth))
                cells.setdefault(n, {}).setdefault(k, []).append(e)
                per_start[y].setdefault(n, {})[k] = e
        print(f"start {y}: leads 1..{leads[-1]} in {time.time() - t0:.0f} s", flush=True)
    table = {n: {str(k): {"median_km": statistics.median(v), "max_km": max(v), "n_starts": len(v)} for k, v in sorted(c.items())} for n, c in cells.items()}
    doc = {"metadata": {"what": "hindcast error of the forecast model against DE440 by lead, heliocentric position, km", "predictor": a.predictor, "dt_days": a.dt_days, "starts": list(per_start),
                        "targets_end": DATA_END_YEAR, "reference_points": "Mercury, Venus, Earth, Moon: centres; Mars..Pluto: system barycentres",
                        "generator": "scripts/make_prediction_error_table.py"}, "table": table, "per_start": per_start}
    a.out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print("wrote", a.out)
    for n in ("Mercury", "Earth", "Moon", "Mars", "Jupiter", "Neptune"):
        if n in table:
            print(f"  {n:8s} " + "  ".join(f"+{k}y {table[n][str(k)]['median_km']:.3g}" for k in (1, 5, 10, 20) if str(k) in table[n]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
