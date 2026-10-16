"""Summarise scripts/comet_ladder.py output: median position error (km) at +1, +3, +5 yr per rung, in groups (far / near; with and without the time delay), and runs improved per rung step.

usage: python scripts/comet_ladder_report.py LADDER.json [--out FILE.md]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

RUNGS = [("R0", "Newton: Sun + nine planet systems"), ("R1", "+ Sun 1PN"), ("R2", "+ Marsden push (A1, A2, A3), no time delay"), ("R3", "+ time delay DT"),
         ("R3n", "R3 without the Sun's 1PN"), ("R4", "R3 + the pull of DE440's 343 asteroids")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ladder", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    rows = json.loads(a.ladder.read_text())
    groups = {"all": rows, "far (> 10 au at the start)": [r for r in rows if r["r0_au"] > 10], "near, no time delay": [r for r in rows if r["r0_au"] <= 10 and not r["DT"]],
              "near, with time delay": [r for r in rows if r["r0_au"] <= 10 and r["DT"]]}
    lines = [f"{len(rows)} (comet, window) runs from {len({r['comet'] for r in rows})} comets. Median position error against JPL's own solution, km.", ""]
    for gname, g in groups.items():
        if not g:
            continue
        lines += [f"## {gname} ({len(g)} runs)", "", "| rung | +1 yr | +3 yr | +5 yr | better than the previous rung (+5 yr) |", "|---|---|---|---|---|"]
        prev = None
        for key, label in RUNGS:
            vals = [r["err_km"].get(key) for r in g]
            if any(v is None for v in vals) or not vals:
                continue
            cells = [statistics.median(v[i] for v in vals) for i in (0, 2, 4)]
            better = "-"
            if prev is not None and key not in ("R3n",):
                better = f"{sum(v[4] < p[4] for v, p in zip(vals, prev))}/{len(g)}"
            lines.append(f"| {key}: {label} | {cells[0]:,.2f} | {cells[1]:,.2f} | {cells[2]:,.2f} | {better} |")
            if key != "R3n":
                prev = vals
        lines.append("")
    text = "\n".join(lines)
    print(text)
    if a.out:
        a.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
