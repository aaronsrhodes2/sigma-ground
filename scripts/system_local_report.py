"""Tabulate the system-local moon runs (scripts/system_local_moons.py via run_system_local.py): every moon of every planet system, position error relative to its planet at +1..+5 yr, two windows,
with the measured zonals; and, where a control run with the table's zonals exists (Jupiter, Saturn), the paired change.

usage: python scripts/system_local_report.py RUN_DIR [--control-dir DIR] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

SYSTEMS = ["Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"]


def load(d: Path) -> dict:
    """{system: {moon: {"period": days, "dt": days, "windows": {window: [err km at +0..+5]}}}}"""
    out: dict = {}
    for f in sorted(d.glob("*_spo300.json")):
        for r in json.loads(f.read_text()):
            sysd = out.setdefault(r["system"], {})
            for m, e in r["err_km"].items():
                sysd.setdefault(m, {"period": r["period_days"][m], "dt": r["dt_days"], "windows": {}})["windows"][r["window"]] = e
    return out


def med(v):
    return statistics.median(v)


def fmt(x):
    return f"{x:,.0f}" if x >= 100 else (f"{x:.1f}" if x >= 1 else f"{x:.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--control-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    new = load(a.run_dir)
    ctrl = load(a.control_dir) if a.control_dir else {}
    lines = ["# Moons in system-local models (Sun + planet + all its moons), step = shortest period / 300, measured zonals", "",
             "Position error relative to the planet, km, median of windows J2010 and J2015. `period` = osculating period (d). Control = the table's zonals, same step.", "",
             "| system | moon | period d | +1 yr | +3 yr | +5 yr | control +5 yr | windows better than control |", "|---|---|---|---|---|---|---|---|"]
    rows = []
    for s in SYSTEMS:
        for m, d in sorted(new.get(s, {}).items(), key=lambda kv: kv[1]["period"]):
            w = d["windows"]
            e1, e3, e5 = (med([v[k] for v in w.values()]) for k in (1, 3, 5))
            c = ctrl.get(s, {}).get(m)
            ce5, better = None, ""
            if c and set(c["windows"]) == set(w):
                ce5 = med([v[5] for v in c["windows"].values()])
                better = f"{sum(w[k][5] < c['windows'][k][5] for k in w)}/{len(w)}"
            rows.append({"system": s, "moon": m, "period_d": d["period"], "dt_d": d["dt"], "e1": e1, "e3": e3, "e5": e5, "control_e5": ce5, "windows": w})
            lines.append(f"| {s} | {m} | {d['period']:.3f} | {fmt(e1)} | {fmt(e3)} | {fmt(e5)} | {fmt(ce5) if ce5 is not None else '-'} | {better or '-'} |")
    text = "\n".join(lines)
    print(text)
    if a.out:
        (a.out / "SYSTEM_LOCAL_MOONS.md").write_text(text + "\n", encoding="utf-8")
        (a.out / "SYSTEM_LOCAL_MOONS.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
