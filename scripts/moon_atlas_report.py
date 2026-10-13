"""Residual atlas of the Moon: what shape does our model's deficit have?

Input: the jsonl of scripts/fair_jpl_contest.py run with --sample-days (the Moon's gap, ours minus the vintage's own continuation, as radial / along-track / cross-track
components about the Earth, sampled every few days for 5 years from identical initial states). A missing force leaves a signature:
  * a smooth polynomial in time (a mean-motion error gives along-track growth ~ t and ~ t^2),
  * a periodic term at a known lunar or solar frequency (anomalistic month 27.55 d, synodic 29.53 d, draconic 27.21 d, evection 31.81 d, variation 14.77 d,
    the annual and semi-annual terms, the 8.85-yr apsidal and 18.6-yr nodal cycles are too long to resolve in 5 yr and show as polynomial).
For each component the report fits a quadratic polynomial plus a sinusoid scanned over periods 8-400 d, lists the strongest periodic terms with the fraction of the
remaining variance each explains, matches their periods to the known list, and prints the polynomial coefficients.

usage: python scripts/moon_atlas_report.py JSONL [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

KNOWN = {"anomalistic month": 27.5546, "synodic month": 29.5306, "draconic month": 27.2122, "sidereal month": 27.3217, "evection": 31.8119,
         "variation (half synodic)": 14.7653, "annual": 365.2596, "semi-annual": 182.6298, "anomalistic/2": 13.7773, "draconic/2": 13.6061,
         "parallactic inequality": 29.5306 * 1.0, "year/3": 121.75}
COMP = ("radial", "along-track", "cross-track")


def series(rows):
    out = []
    for r in rows:
        m = r["bodies"]["Moon"]
        pts = sorted((float(k) * 365.25, v["rtn_km"]) for k, v in m.items() if "rtn_km" in v)
        t = np.array([p[0] for p in pts])
        y = np.array([p[1] for p in pts])
        out.append((r["vintage"], r["start"], t, y))
    return out


def poly_fit(t, y, deg=2):
    X = np.vander(t / 365.25, deg + 1, increasing=True)
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    return c, y - X @ c


def periodic_scan(t, y, periods):
    best = []
    base = np.vander(t / 365.25, 3, increasing=True)
    var0 = float(np.var(y)) * len(y)
    for P in periods:
        w = 2 * np.pi / P
        X = np.column_stack([base, np.cos(w * t), np.sin(w * t)])
        c, res, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ c
        best.append((P, float(np.hypot(c[3], c[4])), 1.0 - float(r @ r) / max(var0, 1e-30)))
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    rows = [json.loads(ln) for ln in a.jsonl.read_text().splitlines() if ln.strip()]
    data = series(rows)
    periods = np.exp(np.linspace(np.log(8.0), np.log(400.0), 3000))
    report = {"cases": []}
    lines = [f"# Moon residual atlas ({len(data)} cases)", ""]
    for vint, start, t, y in data:
        lines.append(f"## {vint} from {start}: {len(t)} samples over {t[-1] / 365.25:.2f} yr")
        lines.append("| component | value at +1 yr (km) | +3 yr | +5 yr | quadratic coefficients (km, km/yr, km/yr^2) | rms after the quadratic (km) |")
        lines.append("|---|---|---|---|---|---|")
        entry = {"vintage": vint, "start": start, "components": {}}
        for j, name in enumerate(COMP):
            c, res = poly_fit(t, y[:, j])
            at = lambda yr: float(np.interp(yr * 365.25, t, y[:, j]))
            lines.append(f"| {name} | {at(1):+.3f} | {at(3):+.3f} | {at(5):+.3f} | {c[0]:+.3f}, {c[1]:+.3f}, {c[2]:+.3f} | {np.sqrt(np.mean(res ** 2)):.4f} |")
            scan = periodic_scan(t, y[:, j], periods)
            scan.sort(key=lambda x: -x[2])
            peaks = []
            for P, amp, frac in scan:
                if all(abs(P - q) / q > 0.04 for q, _, _ in peaks):
                    peaks.append((P, amp, frac))
                if len(peaks) == 3:
                    break
            entry["components"][name] = {"quad": c.tolist(), "rms_after_quad": float(np.sqrt(np.mean(res ** 2))),
                                         "peaks": [{"period_d": P, "amp_km": amp, "variance_fraction": frac} for P, amp, frac in peaks]}
        for name in COMP:
            pk = entry["components"][name]["peaks"]
            desc = []
            for p in pk:
                m = min(KNOWN.items(), key=lambda kv: abs(kv[1] - p["period_d"]) / kv[1])
                tag = f" ~ {m[0]} ({m[1]:.2f} d)" if abs(m[1] - p["period_d"]) / m[1] < 0.03 else ""
                desc.append(f"{p['period_d']:.1f} d, {p['amp_km'] * 1000:.0f} m, explains {p['variance_fraction'] * 100:.0f}% of the remaining variance{tag}")
            lines.append(f"- strongest periodic terms, {name}: " + "; ".join(desc))
        lines.append("")
        report["cases"].append(entry)
    text = "\n".join(lines)
    print(text)
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "MOON_ATLAS.md").write_text(text)
        (a.out / "MOON_ATLAS.json").write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
