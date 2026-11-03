"""Our own orbit fit (misc/CANDIDATE_LEDGER.md C28): a body's barycentric state at a fixed epoch from MPC optical observations alone, by least-squares differential correction with our
propagation as the model, then three curves on the later observations: (a) the fitted state propagated forward, (b) our propagation from JPL's state at the epoch (the borrowed start),
(c) JPL's own Horizons vectors, all through the C24 pipeline (scripts/eros_sky_residuals.py).

Model: the body as a test particle in the DE440s planet field with the Sun's 1PN term and DE440's asteroids as prescribed sources (scripts/mpc_live_score.Engine); no non-gravitational
terms. Partials by finite differences (six perturbed propagations per iteration). Weights: one sigma per observation (0.5 arcsec default, every station). Rejection at 3 sigma on the
total residual after the first pass. Convergence: every state correction below a tenth of its formal sigma. Covariance: (J^T W J)^-1, and the same scaled by the unit-weight variance.

usage: python scripts/orbit_fit.py --body 433 --epoch-jd 2458849.5 --fit-from 2015-01-01 --fit-to 2019-12-31 --predict-from 2020-01-01 [--sigma-arcsec 0.5] [--md misc/ORBIT_FIT_EROS.md]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))
import eros_sky_residuals as esr  # noqa: E402
import mpc_live_score as mls  # noqa: E402

DAY = 86400.0
STEPS = np.array([100.0, 100.0, 100.0, 0.1, 0.1, 0.1])              # km, km/day. C24 used 0.005 km/day; with the integrator's 1e-6 km tolerance that step's partials were 10 % noise on a short arc, 0.1 km/day (1.2 m/s, 180 km over five years, still linear) makes them 0.5 %
OUT_ROOT = Path("D:/datasets/mpc_live/orbit_fit")


# ----------------------------------------------------------------------------- the fit (pure given a position function; tested on a synthetic case)
def computed_radec(obs: esr.Obs, x_ast: np.ndarray):
    d = x_ast - obs.x_obs
    return np.arctan2(d[:, 1], d[:, 0]), np.arcsin(d[:, 2] / np.linalg.norm(d, axis=1))


def synthesize(obs: esr.Obs, x_ast: np.ndarray, noise_arcsec: float, rng: np.random.Generator) -> None:
    """Overwrite obs.ra/obs.dec with the positions computed from x_ast plus Gaussian noise (arcsec, per component): a synthetic observation set for the tests."""
    ra, dec = computed_radec(obs, x_ast)
    obs.ra = ra + rng.normal(0.0, noise_arcsec, obs.n) / esr.ARC / np.cos(dec)
    obs.dec = dec + rng.normal(0.0, noise_arcsec, obs.n) / esr.ARC


def light_time_fixed(fn, obs: esr.Obs, passes: int = 2) -> np.ndarray:
    """Target times T = t - tau from the current model, held fixed through the fit (a state change of 100 km moves tau by 0.3 ms)."""
    tau = np.zeros(obs.n)
    for _ in range(passes):
        tau = np.linalg.norm(fn(obs.jd_tdb - tau / DAY) - obs.x_obs, axis=1) / esr.C_KM_S
    return obs.jd_tdb - tau / DAY


def partials(fn, obs: esr.Obs, y: np.ndarray, T: np.ndarray, steps: np.ndarray = STEPS) -> tuple[np.ndarray, np.ndarray]:
    """(r0, J): residuals O - C (2N, arcsec) at y and their derivative with respect to the six state components (2N x 6), one-sided finite differences."""
    r0 = esr.radec_residuals(obs, fn(y, T)).reshape(-1)
    J = np.empty((2 * obs.n, 6))
    for k in range(6):
        yk = y.copy(); yk[k] += steps[k]
        J[:, k] = (esr.radec_residuals(obs, fn(yk, T)).reshape(-1) - r0) / steps[k]
    return r0, J


def fit_state(fn, obs: esr.Obs, y_guess: np.ndarray, sigma_arcsec, *, reject_sigma: float = 3.0, max_iter: int = 10, steps: np.ndarray = STEPS, log=print) -> dict:
    """Differential correction of the state y (km, km/day at the epoch) to the observations. fn(y, T) -> positions (N, 3) km at target TDB times T.

    sigma_arcsec: one value or an array per observation. Returns the state, both covariances, the kept mask, the per-observation residuals and the iteration history.
    """
    sig = np.broadcast_to(np.asarray(sigma_arcsec, dtype=float), (obs.n,)).copy()
    T = light_time_fixed(lambda TT: fn(y_guess, TT), obs)
    y = np.array(y_guess, dtype=float)
    keep = np.ones(obs.n, dtype=bool)
    history = []
    for it in range(max_iter):
        t0 = time.time()
        r0, J = partials(fn, obs, y, T, steps)
        tot = np.hypot(r0[0::2], r0[1::2])
        if it >= 1:
            keep = tot / sig <= reject_sigma                                   # rejection only after the first pass
        w = np.repeat(keep / sig, 2)                                           # weight 1/sigma per component, 0 for rejected
        A = J * w[:, None]
        b = -(r0 * w)
        N = A.T @ A
        dp = np.linalg.solve(N, A.T @ b)
        sig_formal = np.sqrt(np.diag(np.linalg.inv(N)))
        rms = float(np.sqrt(np.mean(tot[keep] ** 2)))
        history.append({"iteration": it, "kept": int(keep.sum()), "rms_arcsec": rms, "step_km": [float(x) for x in dp[:3]], "step_km_day": [float(x) for x in dp[3:]],
                        "step_over_sigma": [float(x) for x in dp / sig_formal], "seconds": time.time() - t0})
        log(f"  iteration {it}: kept {int(keep.sum())}/{obs.n}, rms {rms:.4f} arcsec, step {np.linalg.norm(dp[:3]):.3f} km {np.linalg.norm(dp[3:]) * 1e6 / DAY:.3f} mm/s, "
            f"max step/sigma {np.abs(dp / sig_formal).max():.3f}  [{time.time() - t0:.0f} s]")
        y = y + dp
        converged = bool(np.all(np.abs(dp) < 0.1 * sig_formal))                # the correction is a tenth of the formal error in every component: the fit has stopped moving
        if converged and it >= 1:
            break
    r_fin = esr.radec_residuals(obs, fn(y, T))
    tot = np.hypot(r_fin[:, 0], r_fin[:, 1])
    r0, J = partials(fn, obs, y, T, steps)
    w = np.repeat(keep / sig, 2)
    A = J * w[:, None]
    cov = np.linalg.inv(A.T @ A)
    chi2 = float(np.sum((r0 * w) ** 2))
    dof = int(2 * keep.sum() - 6)
    return {"state_km_kmday": y.tolist(), "epoch_light_time_targets": T.tolist(), "kept": keep.tolist(), "n_kept": int(keep.sum()), "n_obs": int(obs.n),
            "rms_arcsec": float(np.sqrt(np.mean(tot[keep] ** 2))), "residuals_arcsec": r_fin.tolist(), "chi2": chi2, "dof": dof, "unit_weight_variance": chi2 / dof,
            "covariance_formal": cov.tolist(), "covariance_scaled": (cov * chi2 / dof).tolist(), "sigma_formal": np.sqrt(np.diag(cov)).tolist(),
            "sigma_scaled": np.sqrt(np.diag(cov) * chi2 / dof).tolist(), "iterations": history, "converged": converged}


def osculating_elements(r_km: np.ndarray, v_km_s: np.ndarray, gm_km3_s2: float) -> dict:
    """Heliocentric osculating elements (au, deg) from an equatorial heliocentric state: a two-body conversion, arithmetic on the state and the Sun's GM."""
    r = np.asarray(r_km, float); v = np.asarray(v_km_s, float)
    rn = np.linalg.norm(r); vn2 = float(v @ v)
    h = np.cross(r, v); hn = np.linalg.norm(h)
    evec = np.cross(v, h) / gm_km3_s2 - r / rn
    e = float(np.linalg.norm(evec))
    a = 1.0 / (2.0 / rn - vn2 / gm_km3_s2)
    inc = math.acos(h[2] / hn)
    n = np.cross([0.0, 0.0, 1.0], h); nn = np.linalg.norm(n)
    om = math.atan2(n[1], n[0]) % (2 * math.pi) if nn > 0 else 0.0
    w = math.acos(np.clip(float(n @ evec) / (nn * e), -1, 1)) if nn > 0 and e > 0 else 0.0
    if evec[2] < 0:
        w = 2 * math.pi - w
    nu = math.acos(np.clip(float(evec @ r) / (e * rn), -1, 1))
    if float(r @ v) < 0:
        nu = 2 * math.pi - nu
    E = 2 * math.atan2(math.sqrt(1 - e) * math.sin(nu / 2), math.sqrt(1 + e) * math.cos(nu / 2))
    M = (E - e * math.sin(E)) % (2 * math.pi)
    return {"a_au": a / esr.AU_KM, "e": e, "i_deg": math.degrees(inc), "om_deg": math.degrees(om), "w_deg": math.degrees(w), "ma_deg": math.degrees(M), "frame": "ICRF equatorial, heliocentric"}


# ----------------------------------------------------------------------------- the real body
def jpl_state_at(number: int, epoch_jd: float) -> tuple[np.ndarray, str]:
    files = sorted(glob.glob(str(mls.MPC_OLD / "eros" / "vec_*.json"))) if number == 433 else []
    files += sorted(glob.glob(str(Path("D:/datasets/jpl/horizons_asteroids") / f"horizons_ast_{number}.json")))
    for f in files:
        try:
            return mls.parse_horizons_vector_row(json.loads(Path(f).read_text(encoding="utf-8"))["result"], number, epoch_jd), f
        except ValueError:
            continue
    raise SystemExit(f"no JPL state for {number} at JD {epoch_jd} in the stored Horizons files (no fetch is made here)")


def subset(obs: esr.Obs, mask: np.ndarray) -> esr.Obs:
    o = esr.Obs.__new__(esr.Obs)
    o.__dict__.update({k: (v[mask] if isinstance(v, np.ndarray) and v.shape[:1] == (obs.n,) else v) for k, v in obs.__dict__.items()})
    o.n = int(mask.sum()); o.code = [c for c, f in zip(obs.code, mask) if f]
    return o


def year_table(jd_tdb: np.ndarray, epoch_jd: float, curves: dict[str, np.ndarray], ok: np.ndarray) -> list[dict]:
    years = np.array([esr.erfa.jd2cal(j, 0.0)[0] for j in jd_tdb])
    rows = []
    for yr in sorted(set(years[ok].tolist())):
        m = ok & (years == yr)
        row = {"year": int(yr), "n": int(m.sum()), "lead_years_median": float(np.median(jd_tdb[m] - epoch_jd) / 365.25)}
        for k, res in curves.items():
            row[k] = mls.robust(res[m])
        for k in curves:
            if k != "c":
                row[f"{k}_minus_c"] = mls.robust(curves[k][m] - curves["c"][m])
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--body", type=int, default=433)
    ap.add_argument("--epoch-jd", type=float, default=mls.JD_EROS)
    ap.add_argument("--fit-from", default="2015-01-01"); ap.add_argument("--fit-to", default="2019-12-31")
    ap.add_argument("--predict-from", default="2020-01-01")
    ap.add_argument("--sigma-arcsec", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--md", type=Path, default=None)
    a = ap.parse_args()
    out = a.out or OUT_ROOT / str(a.body)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "fit.log", "a", encoding="utf-8")

    def log(msg):
        line = f"{datetime.now(timezone.utc).strftime('%H:%M:%S')} {msg}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    log(f"orbit_fit body {a.body} epoch JD {a.epoch_jd} fit {a.fit_from}..{a.fit_to} predict from {a.predict_from} sigma {a.sigma_arcsec} arcsec")
    rows = esr.parse_obs80(mls.MPC_OLD / f"obs80_{a.body}.txt")
    codes = esr.parse_obscodes(mls.MPC_OLD / "ObsCodes.html")
    rows = [r for r in rows if r[5] in codes]
    t_of = lambda r: mls.obs_datetime(r).date().isoformat()
    fit_rows = [r for r in rows if a.fit_from <= t_of(r) <= a.fit_to]
    pred_rows = [r for r in rows if t_of(r) >= a.predict_from]
    pred_rows.sort(key=mls.obs_datetime)
    w0 = min(a.epoch_jd, mls.utc_jd(fit_rows).min()) - 10.0
    engine = mls.Engine(log, w0=math.floor(w0) + 0.5, horizon_days=max(mls.utc_jd(pred_rows).max(), a.epoch_jd) + 30.0 - (math.floor(w0) + 0.5))
    obs_fit = esr.Obs(fit_rows, codes, engine.kernel)
    obs_pred = esr.Obs(pred_rows, codes, engine.kernel)
    log(f"fit observations {obs_fit.n}, prediction observations {obs_pred.n}")
    y_jpl, jpl_file = jpl_state_at(a.body, a.epoch_jd)
    fn = lambda y, T: engine.positions(a.body, a.epoch_jd, y, np.asarray(T, float))
    t0 = time.time()
    fit = fit_state(fn, obs_fit, y_jpl, a.sigma_arcsec, log=log)
    log(f"fit done in {time.time() - t0:.0f} s: kept {fit['n_kept']}/{fit['n_obs']}, rms {fit['rms_arcsec']:.4f} arcsec, converged {fit['converged']}")
    y_fit = np.array(fit["state_km_kmday"])
    d = y_fit - y_jpl
    fit["jpl_state_km_kmday"] = y_jpl.tolist(); fit["jpl_state_file"] = jpl_file
    fit["difference_vs_jpl"] = {"km": d[:3].tolist(), "km_day": d[3:].tolist(), "position_km": float(np.linalg.norm(d[:3])), "speed_mm_s": float(np.linalg.norm(d[3:]) * 1e6 / DAY),
                                "in_formal_sigma": (d / np.array(fit["sigma_formal"])).tolist(), "in_scaled_sigma": (d / np.array(fit["sigma_scaled"])).tolist()}
    sp, sv = engine.kernel[0, 10].compute_and_differentiate(a.epoch_jd)
    gm_sun = float(engine.gmp[0])
    fit["elements_fit"] = osculating_elements(y_fit[:3] - sp, (y_fit[3:] - sv) / DAY, gm_sun)
    fit["elements_jpl"] = osculating_elements(y_jpl[:3] - sp, (y_jpl[3:] - sv) / DAY, gm_sun)
    fit["elements_source"] = f"two-body osculating elements from each state and the Sun's DE440 GM {gm_sun} km^3/s^2 (fixture de440_state_vectors_v2physics, via comet_ladder.gm_planets)"
    (out / f"fit_{a.body}.json").write_text(json.dumps(fit, indent=1), encoding="utf-8")

    # the three curves on the prediction window
    tj, pj, vj = esr.load_horizons()
    cfn = lambda T: esr.hermite(tj, pj, vj, T)
    Tc, _ = esr.light_time_targets(cfn, obs_pred)
    res_c = esr.radec_residuals(obs_pred, cfn(Tc))
    ok = np.hypot(res_c[:, 0], res_c[:, 1]) <= 5.0
    curves = {"c": res_c}
    for label, y in (("a", y_fit), ("b", y_jpl)):
        t0 = time.time()
        g = lambda T: fn(y, T)
        T, _ = esr.light_time_targets(g, obs_pred, iters=2)
        curves[label] = esr.radec_residuals(obs_pred, g(T))
        log(f"curve ({label}) in {time.time() - t0:.0f} s: median {mls.robust(curves[label][ok])['median_arcsec']:.4f} arcsec")
    table = year_table(obs_pred.jd_tdb, a.epoch_jd, curves, ok)
    overall = {k: mls.robust(v[ok]) for k, v in curves.items()}
    overall["a_minus_c"] = mls.robust((curves["a"] - curves["c"])[ok]); overall["b_minus_c"] = mls.robust((curves["b"] - curves["c"])[ok])
    result = {"ledger": "misc/CANDIDATE_LEDGER.md C28", "body": a.body, "epoch_jd_tdb": a.epoch_jd, "fit_window": [a.fit_from, a.fit_to], "predict_from": a.predict_from,
              "curves": {"a": "our fitted state propagated forward", "b": "our propagation from JPL's state at the epoch", "c": "JPL's Horizons vectors (C24 arm J)"},
              "n_pred": int(obs_pred.n), "n_dropped_5arcsec_on_c": int((~ok).sum()), "overall": overall, "per_year": table, "fit_summary": {k: fit[k] for k in ("n_kept", "n_obs", "rms_arcsec", "converged", "unit_weight_variance", "sigma_formal", "sigma_scaled", "difference_vs_jpl", "elements_fit", "elements_jpl")}}
    (out / f"curves_{a.body}.json").write_text(json.dumps({"jd_tdb": obs_pred.jd_tdb.tolist(), "code": obs_pred.code, "ok": ok.tolist(), **{k: v.tolist() for k, v in curves.items()}}), encoding="utf-8")
    (out / f"result_{a.body}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    log("per year: " + json.dumps([{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k in ("year", "n")} | {"a": round(r["a"]["median_arcsec"], 4), "b": round(r["b"]["median_arcsec"], 4), "c": round(r["c"]["median_arcsec"], 4), "a-c": round(r["a_minus_c"]["median_arcsec"], 4)} for r in table]))
    if a.md:
        a.md.write_text(markdown(result, fit), encoding="utf-8")
        log(f"wrote {a.md}")
    log(f"finished: fit rms {fit['rms_arcsec']:.4f}, curves a/b/c medians {overall['a']['median_arcsec']:.4f}/{overall['b']['median_arcsec']:.4f}/{overall['c']['median_arcsec']:.4f} arcsec")
    return 0


def markdown(result: dict, fit: dict) -> str:
    f = result["fit_summary"]; dv = f["difference_vs_jpl"]
    L = [f"# Orbit fit of {result['body']} from observations alone (C28)", "",
         f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} by `scripts/orbit_fit.py`; data `D:/datasets/mpc_live/orbit_fit/{result['body']}/`. Epoch JD {result['epoch_jd_tdb']} (TDB); fit window {result['fit_window'][0]} to {result['fit_window'][1]}; prediction window from {result['predict_from']}.",
         "", "## The fit", "",
         f"* Observations kept {f['n_kept']} of {f['n_obs']} (3-sigma rejection at 1.5 arcsec after the first pass); post-fit RMS **{f['rms_arcsec']:.4f} arcsec**; converged: {f['converged']}; unit-weight variance {f['unit_weight_variance']:.3f} (sigma assumed 0.5 arcsec).",
         "", "| component | fitted | JPL | fit minus JPL | formal sigma | scaled sigma | difference / scaled sigma |", "|---|---|---|---|---|---|---|"]
    names = ["x km", "y km", "z km", "vx km/day", "vy km/day", "vz km/day"]
    for i, nm in enumerate(names):
        L.append(f"| {nm} | {fit['state_km_kmday'][i]:.6f} | {fit['jpl_state_km_kmday'][i]:.6f} | {(dv['km'] + dv['km_day'])[i]:.6f} | {f['sigma_formal'][i]:.6f} | {f['sigma_scaled'][i]:.6f} | {dv['in_scaled_sigma'][i]:.2f} |")
    L += ["", f"Position difference {dv['position_km']:.2f} km, speed difference {dv['speed_mm_s']:.2f} mm/s.", "", "| element | fitted | JPL (same epoch, from its state) | difference |", "|---|---|---|---|"]
    for k in ("a_au", "e", "i_deg", "om_deg", "w_deg", "ma_deg"):
        L.append(f"| {k} | {f['elements_fit'][k]:.8f} | {f['elements_jpl'][k]:.8f} | {f['elements_fit'][k] - f['elements_jpl'][k]:.2e} |")
    L += ["", "## The three curves on the prediction window", "", f"{result['n_pred']} observations, {result['n_dropped_5arcsec_on_c']} dropped by the 5-arcsec rule on (c). Residuals in arcsec: median of the total residual (robust sigma); the last two columns are the per-observation differences.", "",
          "| year | n | lead yr | (a) fitted | (b) from JPL's state | (c) JPL | (a)-(c) median | (b)-(c) median |", "|---|---|---|---|---|---|---|---|"]
    for r in result["per_year"]:
        L.append(f"| {r['year']} | {r['n']} | {r['lead_years_median']:.2f} | {r['a']['median_arcsec']:.4f} ({r['a']['robust_sigma_arcsec']:.3f}) | {r['b']['median_arcsec']:.4f} ({r['b']['robust_sigma_arcsec']:.3f}) | {r['c']['median_arcsec']:.4f} ({r['c']['robust_sigma_arcsec']:.3f}) | {r['a_minus_c']['median_arcsec']:.4f} | {r['b_minus_c']['median_arcsec']:.4f} |")
    o = result["overall"]
    L += ["", f"All years: (a) {o['a']['median_arcsec']:.4f}, (b) {o['b']['median_arcsec']:.4f}, (c) {o['c']['median_arcsec']:.4f}; (a)-(c) {o['a_minus_c']['median_arcsec']:.4f}, (b)-(c) {o['b_minus_c']['median_arcsec']:.4f} arcsec (medians).", ""]
    return "\n".join(L)


if __name__ == "__main__":
    sys.exit(main())
