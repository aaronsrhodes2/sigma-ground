"""SPARC a0 test: fit the radial acceleration relation to the 2693 raw points and compare its acceleration scale with c H0 / 2 pi.

Data: Lelli, McGaugh, Schombert, Pawlowski 2017 (SPARC), D:/datasets/sparc/RAR.mrt (log10 g_bar, log10 g_obs with errors).
Relation fitted (McGaugh, Lelli & Schombert 2016): g_obs = g_bar / (1 - exp(-sqrt(g_bar / g_dagger))).
Prediction: a0 = c H0 / (2 pi) = 1.042e-10 m/s^2, from Mentat's mond_a0_constant (H0 = 67.4 km/s/Mpc inside Mentat).
Pre-registered criterion (UNIFICATION_PROGRAM.md, discriminator table): FAIL if the predicted a0 lies outside the fitted g_dagger by more than its scatter.
Run:  python -X utf8 scripts/unification/round3/SPARC_a0/sparc_a0_fit.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar, least_squares

DATA = Path("D:/datasets/sparc/RAR.mrt")
OUT = Path(__file__).parent / "sparc_a0_results.json"
A0_PRED = 1.042e-10          # Mentat mond_a0_constant, c H0 / (2 pi)


def load():
    rows = []
    for line in DATA.read_text().splitlines():
        parts = line.split()
        if len(parts) == 4:
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                pass
    a = np.array(rows)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3]       # log gbar, e_gbar, log gobs, e_gobs


def model_log_gobs(log_gbar, log_gd):
    gbar = 10.0 ** log_gbar
    gd = 10.0 ** log_gd
    return np.log10(gbar / (1.0 - np.exp(-np.sqrt(gbar / gd))))


def fit(lg, eg, lo, eo, idx=None, sig_int=None):
    if idx is not None:
        lg, eg, lo, eo = lg[idx], eg[idx], lo[idx], eo[idx]
    slope = lambda g: 0.5 * (model_log_gobs(g + 1e-3, np.log10(1.2e-10)) - model_log_gobs(g - 1e-3, np.log10(1.2e-10))) / 1e-3
    sig2 = eo ** 2 + (slope(lg) * eg) ** 2 + (0.0 if sig_int is None else sig_int ** 2)

    def chi(p):
        return (lo - model_log_gobs(lg, p[0])) / np.sqrt(sig2)
    sol = least_squares(chi, [math.log10(1.2e-10)])
    return 10.0 ** sol.x[0], sol


def main():
    lg, eg, lo, eo = load()
    n = len(lg)
    gd, sol = fit(lg, eg, lo, eo)
    resid = lo - model_log_gobs(lg, math.log10(gd))
    scatter = float(np.std(resid))
    # intrinsic scatter: increase sig_int until reduced chi2 = 1, then refit
    sig_int = 0.0
    for _ in range(50):
        gd_i, sol_i = fit(lg, eg, lo, eo, sig_int=sig_int)
        chi2 = float((sol_i.fun ** 2).sum()) / (n - 1)
        sig_int *= math.sqrt(max(chi2, 1e-6)) if sig_int > 0 else 1.0
        if sig_int == 0.0:
            sig_int = 0.05
        if abs(chi2 - 1.0) < 1e-3:
            break
    rng = np.random.default_rng(20261011)
    boots = []
    for _ in range(400):
        idx = rng.integers(0, n, n)
        boots.append(fit(lg, eg, lo, eo, idx, sig_int)[0])
    boots = np.array(boots)
    gd_err = float(np.std(boots))
    # galaxies are correlated points; the quoted statistical error treats points as independent, so also report the point scatter in log space
    out = {
        "n_points": n, "g_dagger_m_s2": gd_i, "g_dagger_stat_err": gd_err, "intrinsic_scatter_dex_fitted": sig_int,
        "rms_scatter_dex": scatter, "a0_pred_cH0_over_2pi_m_s2": A0_PRED,
        "log10_ratio_pred_over_fit": math.log10(A0_PRED / gd_i),
        "pred_over_fit": A0_PRED / gd_i,
        "stat_sigmas_away": (gd_i - A0_PRED) / gd_err,
        "difference_in_units_of_point_scatter": abs(math.log10(A0_PRED / gd_i)) / scatter,
        "published_g_dagger_Lelli2017": "1.20e-10 +/- 0.02 (random) +/- 0.24 (systematic) m/s^2 [LIT]",
    }
    OUT.write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(f"{k:42s} {v}")


if __name__ == "__main__":
    sys.exit(main())
