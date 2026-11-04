"""Term discovery, first cut (stage 2, phase 4; stage 3's seed): how each body's along-track residual grows with lead, and the candidates that growth nominates.

    python scripts/residual_growth.py [--min-nights 3] [--sigma 5] [--residuals DIR] [--out misc/CANDIDATE_QUEUE.md]

Per body with --min-nights or more nights of forward rows (lead > 0): the nightly mean along-track residual (arcsec) and its standard error; weighted least squares against
lead time for three models, constant, linear (a mean-motion offset: a start or constant error) and quadratic (a secular acceleration: an unmodelled transverse term); the
model with the lowest BIC is the body's growth law. The three physical classes are the only ones the fitter knows (the data-alone rule: no searches over constants).

Triggers, pre-registered in the plan of 2026-11-04:
  * quadratic coefficient above --sigma (5) sigma, and the quadratic chosen by BIC  -> class "unmodelled transverse acceleration"; test: a with/without run with the fitted
    value through the existing non-gravitational routine, kept only if the WITH arm is closer to the sky in the next window.
  * linear coefficient above 5 sigma, linear chosen                                 -> class "start/constant"; test: the refreshed-start control arm removes it or not.
  * the same-signed linear or quadratic term above 5 sigma in 50 or more bodies of one class with the same sign in the same month -> class "shared dynamics"; test: correlation
    with the perturber table's planets; reported to the Captain before any run.
Every candidate is written to the queue as [AUTO, UNTESTED] with its size, sigma and test. Nothing here is a result; a session runs the test and, if it passes, the ledger gets
the entry. The quadratic coefficient c (arcsec/day^2) is reported with the along-track acceleration it implies at the body's geocentric distance only when Mentat is asked.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

_ROOT = Path(__file__).resolve().parent.parent
SKY = Path("D:/datasets/mpc_sky")
MODELS = {"constant": 1, "linear": 2, "quadratic": 3}


def nightly_means(res: pd.DataFrame) -> pd.DataFrame:
    """One row per body per night: mean along-track residual, its standard error, mean lead, n."""
    g = res[res["lead_days"] > 0].groupby(["packed", "night"])
    out = g.agg(designation=("designation", "first"), orbit_class=("orbit_class", "first"), along=("along_arcsec", "mean"), sd=("along_arcsec", "std"), n=("along_arcsec", "size"), lead=("lead_days", "mean")).reset_index()
    out["sem"] = (out["sd"].fillna(0.3) / np.sqrt(out["n"])).clip(lower=0.02)                 # a single observation's night keeps a floor of 0.02 arcsec; 0.3 arcsec is a typical single-observation scatter
    return out


def fit_growth(lead: np.ndarray, y: np.ndarray, sem: np.ndarray) -> dict:
    """Weighted least squares for the three models; coefficients, sigmas (from the weighted normal matrix), chi2, BIC; the BIC-chosen model."""
    w = 1.0 / sem ** 2
    t = lead / 365.25                                                     # per year
    out = {}
    for name, k in MODELS.items():
        if len(t) < k + 1:
            continue
        X = np.stack([t ** p for p in range(k)], axis=1)
        XtW = X.T * w
        A = XtW @ X; bvec = XtW @ y
        try:
            cov = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            continue
        coef = cov @ bvec
        r = y - X @ coef
        chi2 = float((w * r * r).sum())
        n = len(t)
        bic = chi2 + k * math.log(n)
        out[name] = {"coef": [float(c) for c in coef], "sigma": [float(math.sqrt(max(cov[i, i], 0.0))) for i in range(k)], "chi2": chi2, "bic": bic, "n": n}
    if not out:
        return {}
    best = min(out, key=lambda m: out[m]["bic"])
    return {"models": out, "chosen": best}


def nominate(fits: dict, klass: str, sigma_min: float) -> dict | None:
    m = fits.get("models", {}); ch = fits.get("chosen")
    if ch == "quadratic":
        c, s = m["quadratic"]["coef"][2], m["quadratic"]["sigma"][2]
        if s > 0 and abs(c) / s >= sigma_min:
            return {"class": "unmodelled transverse acceleration", "coef_arcsec_per_yr2": c, "sigma": s, "nsig": abs(c) / s,
                    "test": "with/without run with the fitted value through the non-gravitational routine; keep only if the WITH arm is closer to the sky in the next window"}
    if ch == "linear":
        c, s = m["linear"]["coef"][1], m["linear"]["sigma"][1]
        if s > 0 and abs(c) / s >= sigma_min:
            return {"class": "start/constant", "coef_arcsec_per_yr": c, "sigma": s, "nsig": abs(c) / s,
                    "test": "the refreshed-start control arm: if it removes the slope the start was wrong, else a constant is"}
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-nights", type=int, default=3)
    ap.add_argument("--sigma", type=float, default=5.0)
    ap.add_argument("--residuals", type=Path, default=SKY / "residuals")
    ap.add_argument("--out", type=Path, default=_ROOT / "misc" / "CANDIDATE_QUEUE.md")
    ap.add_argument("--json", type=Path, default=SKY / "growth_fits.json")
    a = ap.parse_args()
    files = sorted(a.residuals.glob("*.parquet"))
    if not files:
        print("no residual files"); return 0
    res = pd.concat([pq.read_table(f, columns=["night", "packed", "designation", "orbit_class", "along_arcsec", "lead_days"]).to_pandas() for f in files])
    nm = nightly_means(res)
    fits, cands = {}, []
    for packed, g in nm.groupby("packed"):
        if g["night"].nunique() < a.min_nights:
            continue
        f = fit_growth(g["lead"].to_numpy(float), g["along"].to_numpy(float), g["sem"].to_numpy(float))
        if not f:
            continue
        f["designation"] = g["designation"].iloc[0]; f["orbit_class"] = g["orbit_class"].iloc[0]; f["nights"] = int(g["night"].nunique()); f["n_obs"] = int(g["n"].sum())
        fits[packed] = f
        c = nominate(f, f["orbit_class"], a.sigma)
        if c:
            cands.append({"packed": packed, **{k: f[k] for k in ("designation", "orbit_class", "nights", "n_obs")}, **c})
    # shared dynamics: the same-signed chosen slope/curvature above sigma in 50 or more bodies of one class
    shared = []
    for klass in sorted({f["orbit_class"] for f in fits.values()}):
        for model, idx in (("linear", 1), ("quadratic", 2)):
            sig = [(p, f["models"][model]["coef"][idx]) for p, f in fits.items() if f["orbit_class"] == klass and f["chosen"] == model and model in f["models"]
                   and f["models"][model]["sigma"][idx] > 0 and abs(f["models"][model]["coef"][idx]) / f["models"][model]["sigma"][idx] >= a.sigma]
            for sign in (1, -1):
                n = sum(1 for _, c in sig if np.sign(c) == sign)
                if n >= 50:
                    shared.append({"class": "shared dynamics", "orbit_class": klass, "model": model, "sign": sign, "bodies": n, "test": "correlation with the perturber table's planets; report to the Captain before any run"})
    a.json.parent.mkdir(parents=True, exist_ok=True)
    a.json.write_text(json.dumps({"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "bodies_fitted": len(fits), "candidates": cands, "shared": shared, "fits": fits}, indent=1), encoding="utf-8")
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines = [f"\n## {stamp} — growth fits over {len(fits)} bodies ({nm['night'].nunique()} nights): {len(cands)} body candidates, {len(shared)} shared signatures  [AUTO, UNTESTED]\n"]
    for c in cands[:200]:
        size = f"{c.get('coef_arcsec_per_yr2', c.get('coef_arcsec_per_yr')):+.4f} arcsec/yr{'^2' if 'coef_arcsec_per_yr2' in c else ''} ({c['nsig']:.1f} sigma)"
        lines.append(f"* [AUTO, UNTESTED] {c['designation']} ({c['orbit_class']}, {c['nights']} nights, {c['n_obs']} obs): **{c['class']}**, {size}. Test: {c['test']}.")
    if len(cands) > 200:
        lines.append(f"* ... {len(cands) - 200} more in {a.json}")
    for s in shared:
        lines.append(f"* [AUTO, UNTESTED] **shared dynamics**: {s['bodies']} {s['orbit_class']} bodies with a {'positive' if s['sign'] > 0 else 'negative'} {s['model']} term above {a.sigma:.0f} sigma. Test: {s['test']}.")
    if not cands and not shared:
        lines.append("* nothing above threshold")
    if not a.out.exists():
        a.out.write_text("# Candidate queue — automatic nominations from the whole-sky residuals (scripts/residual_growth.py)\n\nEvery line is [AUTO, UNTESTED]: a fitted growth law above the pre-registered threshold and the test it must pass. Nothing here is a result; a session runs the test and only a passed test enters misc/CANDIDATE_LEDGER.md.\n", encoding="utf-8")
    with a.out.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
