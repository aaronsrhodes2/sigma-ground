"""Consistency test for a moon's truth: is there ANY nearby initial relative velocity from which our dynamics reproduce the Horizons track to a few km?

Fits the moon's three initial velocity components (relative to the stored state) to the yearly positions with scripts/slow_moon_diagnosis.py's small system. This is a test of the truth, not a
correction: a small fitted change with tiny residuals means the track is a solution of our dynamics from a slightly different start (state or GM-definition mismatch); large residuals mean a missing force.
Nothing fitted is adopted into any predictor.

usage: python scripts/slow_moon_fit.py --moon Nereid --planet Neptune --extra Triton --window j2010 --years 3 [--dt 0.05]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))
import slow_moon_diagnosis as sd  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--moon", required=True)
    ap.add_argument("--planet", required=True)
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--window", default="j2010")
    ap.add_argument("--years", type=int, default=3)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    fix = json.loads(sd.FIXTURE.read_text())

    def resid(dv):
        r = sd.run(a.moon, a.planet, a.window, a.years, "base", fix, tuple(a.extra), dv_m_s=dv, vec=True, dt_days=a.dt)
        return np.concatenate(r[1:])                                            # km, the first epoch is the start

    r0 = resid(np.zeros(3))
    print("start residual norms (km):", [round(float(np.linalg.norm(r0[3 * i:3 * i + 3])), 2) for i in range(a.years)], flush=True)
    sol = least_squares(resid, np.zeros(3), diff_step=1e-6, x_scale=1e-3, xtol=1e-12, ftol=1e-12, max_nfev=12)
    r1 = sol.fun
    res = {"moon": a.moon, "window": a.window, "dt": a.dt, "dv_m_s": sol.x.tolist(), "dv_norm_mm_s": float(np.linalg.norm(sol.x) * 1e3),
           "start_km": [float(np.linalg.norm(r0[3 * i:3 * i + 3])) for i in range(a.years)], "fit_km": [float(np.linalg.norm(r1[3 * i:3 * i + 3])) for i in range(a.years)], "nfev": sol.nfev}
    print(json.dumps(res, indent=1))
    if a.out:
        a.out.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
