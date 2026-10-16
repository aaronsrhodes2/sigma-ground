"""Full-system fine-step arm: r2 (0.05 d) vs the headline stack at 0.0125 d vs the same with DE440's asteroids, ring and the Moon's C22, all 33 bodies, windows J2008/J2014/J2020.

usage: python scripts/shootout_fine_report.py RESTATE_DIR [--out FILE]   (RESTATE_DIR holds rolling_shootout_short.json from run_shootout_parallel.py --plan ... --dt-days 0.0125)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shootout_full_restate_report as R   # noqa: E402

W = ["j2008", "j2014", "j2020"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("restate_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    new = R.load([a.restate_dir / "rolling_shootout_short.json"])
    r2 = R.load([R.V2 / "rolling_shootout_short_v2physics_r2.json"], {"de440_stack"})["de440_stack"]
    ctrl, cand = new["de440_stack"], new["restated_ast_ring_c22"]
    lines = [f"Median heliocentric position error (km) over windows {', '.join(W)}. r2 = 0.05 d, control = headline stack at 0.0125 d, candidate = control + DE440's 343 asteroids + 36-point ring + the Moon's C22.", "",
             "| body | lead | r2 | control | candidate | candidate better than control (windows) |", "|---|---|---|---|---|---|"]
    for b in R.INNER + R.GIANTS + [m for ms in R.MOONS.values() for m in ms]:
        for lead in (1, 5):
            x, c, d = (R.med(R.cells(s, W, b, lead)) for s in (r2, ctrl, cand))
            if x is None or d is None:
                continue
            lines.append(f"| {b} | +{lead} yr | {R.fmt(x)} | {R.fmt(c)} | **{R.fmt(d)}** | {R.paired(R.cells(cand, W, b, lead), R.cells(ctrl, W, b, lead))} |")
    text = "\n".join(lines)
    print(text)
    if a.out:
        a.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
