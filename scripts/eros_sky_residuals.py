"""Eros against raw sky positions (misc/CANDIDATE_LEDGER.md C24): MPC optical observations vs JPL's Eros (arm J), our engine from JPL's state (E1) and our engine from a fit to pre-2021 data (E2).

Everything here was fixed in C24 before any residual was computed. Observations: note-2 column C or B, stations with parallax constants in ObsCodes; pipeline: UTC -> TT -> TDB, Earth from DE440s,
station rotated to the celestial frame (erfa c2t06a, UT1 = UTC, no polar motion), light-time iteration, astrometric J2000 RA/Dec. Arm J drops any observation with total residual above 5 arcsec for every arm.

usage: python scripts/eros_sky_residuals.py [--arms J E1 E2] [--out FILE.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
import time
from pathlib import Path

import erfa
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

MPC = Path("D:/datasets/mpc")
C_KM_S = 299792.458
AU_KM = 149597870.7
DAY = 86400.0
R_EQ_KM = 6378.137
ARC = 206264.80624709636                      # arcsec per radian
JD0 = 2458849.5                               # 2020-01-01 00:00 TDB, the start of arms E1 and E2
TT_MINUS_UTC = 69.184                         # s, valid from 2017-01-01 (no leap second since)
PRE_FIRST, PRE_LAST, POST_FIRST = (1990, 1, 1), (2020, 12, 31), (2021, 1, 1)


def parse_obs80(path: Path):
    rows = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        if len(ln) < 80 or ln[14] not in "CB":
            continue
        try:
            y, m, d = int(ln[15:19]), int(ln[20:22]), float(ln[23:32])
            ra = (int(ln[32:34]) + int(ln[35:37]) / 60 + float(ln[38:44]) / 3600) * 15.0
            dec_s = -1.0 if ln[44] == "-" else 1.0
            dec = dec_s * (int(ln[45:47]) + int(ln[48:50]) / 60 + float(ln[51:56]) / 3600)
        except ValueError:
            continue
        rows.append((y, m, d, math.radians(ra), math.radians(dec), ln[77:80]))
    return rows


def parse_obscodes(path: Path):
    out = {}
    for ln in path.read_text(encoding="utf-8").splitlines():
        if len(ln) < 30 or not re.match(r"^[0-9A-Za-z]{3} ", ln):
            continue
        try:
            out[ln[:3]] = (float(ln[4:13]), float(ln[13:21]), float(ln[21:30]))        # east longitude deg, rho cos phi', rho sin phi' (Earth radii)
        except ValueError:
            continue
    return out


class Obs:
    """The observations as arrays, with the time scales and the observer's barycentric position at each."""

    def __init__(self, rows, codes, kernel):
        keep = [r for r in rows if r[5] in codes]
        self.n = len(keep)
        y = np.array([r[0] for r in keep]); mo = np.array([r[1] for r in keep]); dd = np.array([r[2] for r in keep])
        day = np.floor(dd)
        u1 = np.empty(self.n); u2 = np.empty(self.n)
        for i in range(self.n):
            u1[i], u2[i] = erfa.dtf2d("UTC", int(y[i]), int(mo[i]), int(day[i]), 0, 0, 0.0)
        u2 = u2 + (dd - day)
        t1, t2 = erfa.utctai(u1, u2)
        t1, t2 = erfa.taitt(t1, t2)
        g = math.radians(357.53) + 0.9856003 * np.radians(1) * ((t1 - 2451545.0) + t2)       # periodic TDB - TT term, 1.7 ms amplitude
        self.jd_tt = t1 + t2
        self.jd_tdb = self.jd_tt + 0.001657 * np.sin(g) / DAY
        self.year = y
        self.ra = np.array([r[3] for r in keep]); self.dec = np.array([r[4] for r in keep])
        self.code = [r[5] for r in keep]
        ut1a, ut1b = erfa.utcut1(u1, u2, 0.0)
        c2t = erfa.c2t06a(t1, t2, ut1a, ut1b, np.zeros(self.n), np.zeros(self.n))          # (N,3,3) celestial -> terrestrial
        lon = np.radians([codes[c][0] for c in self.code]); cx = np.array([codes[c][1] for c in self.code]); sx = np.array([codes[c][2] for c in self.code])
        itrs = R_EQ_KM * np.stack([cx * np.cos(lon), cx * np.sin(lon), sx], axis=1)
        station = np.einsum("nji,nj->ni", c2t, itrs)                                       # R^T x
        earth = (np.asarray(kernel[0, 3].compute(self.jd_tdb)) + np.asarray(kernel[3, 399].compute(self.jd_tdb))).T
        self.x_obs = earth + station


def hermite(tj, pj, vj, t):
    """Cubic Hermite position at times t from nodes (tj, pj km, vj km/day)."""
    i = np.clip(np.searchsorted(tj, t) - 1, 0, len(tj) - 2)
    h = tj[i + 1] - tj[i]
    s = ((t - tj[i]) / h)[:, None]
    h = h[:, None]
    h00 = 2 * s ** 3 - 3 * s ** 2 + 1; h10 = s ** 3 - 2 * s ** 2 + s; h01 = -2 * s ** 3 + 3 * s ** 2; h11 = s ** 3 - s ** 2
    return h00 * pj[i] + h10 * h * vj[i] + h01 * pj[i + 1] + h11 * h * vj[i + 1]


def load_horizons():
    jd, p, v = [], [], []
    for f in sorted(glob.glob(str(MPC / "eros" / "vec_*.json"))):
        r = json.load(open(f))["result"]
        for ln in r[r.find("$$SOE") + 6:r.find("$$EOE")].strip().splitlines():
            c = [x.strip() for x in ln.split(",")]
            jd.append(float(c[0])); p.append([float(x) for x in c[2:5]]); v.append([float(x) * DAY for x in c[5:8]])
    jd, p, v = np.array(jd), np.array(p), np.array(v)
    o = np.argsort(jd); jd, p, v = jd[o], p[o], v[o]
    keep = np.concatenate([[True], np.diff(jd) > 1e-9])
    return jd[keep], p[keep], v[keep]


def light_time_targets(pos_fn, obs: Obs, iters=4):
    """Target times T = t - tau and tau for an asteroid position function pos_fn(T) -> (N,3) km."""
    tau = np.zeros(obs.n)
    for _ in range(iters):
        tau = np.linalg.norm(pos_fn(obs.jd_tdb - tau / DAY) - obs.x_obs, axis=1) / C_KM_S
    return obs.jd_tdb - tau / DAY, tau


def radec_residuals(obs: Obs, x_ast, idx=None):
    """O - C in arcsec: (dRA cos dec, dDec) for the asteroid positions x_ast (N,3) at the light-time-corrected times."""
    d = x_ast - obs.x_obs
    ra_c = np.arctan2(d[:, 1], d[:, 0]); dec_c = np.arcsin(d[:, 2] / np.linalg.norm(d, axis=1))
    dra = np.angle(np.exp(1j * (obs.ra - ra_c)))
    return np.stack([dra * np.cos(obs.dec) * ARC, (obs.dec - dec_c) * ARC], axis=1)


def summarize(res):
    tot = np.hypot(res[:, 0], res[:, 1])
    return {"n": int(len(tot)), "rms_total_arcsec": float(np.sqrt(np.mean(tot ** 2))), "median_total_arcsec": float(np.median(tot)),
            "mean_dra_cosdec": float(res[:, 0].mean()), "mean_ddec": float(res[:, 1].mean()),
            "sem_dra": float(res[:, 0].std(ddof=1) / math.sqrt(len(tot))), "sem_ddec": float(res[:, 1].std(ddof=1) / math.sqrt(len(tot)))}


def engine_table(kernel, window_jd, horizon_days, cache: Path):
    from comet_ladder import gm_planets
    from sigma_ground.field.interface import perturber_paths as pp
    from sigma_ground.field.interface.asteroid_perturbers import Perturber
    from sigma_ground.field.interface.prescribed_sources import PrescribedSources
    fix = _ROOT / "sigma_ground/field/interface/fixtures"
    cat = [x for x in json.loads((fix / "de440_perturbers_v2physics.json").read_text())["asteroids"] if x["elements"]]
    pert = [Perturber(str(x["number"]), x["gm_km3_s2"], 0.0, *[x["elements"][k] for k in ("epoch_jd_tdb", "a_au", "e", "i_deg", "om_deg", "w_deg", "ma_deg")]) for x in cat]
    gm_all = np.array([x["gm_km3_s2"] for x in cat])
    gmp = gm_planets()
    if cache.exists():
        nodes = np.load(cache)
    else:
        t0 = time.time()
        nodes = pp.asteroid_window_paths(pert, kernel, np.array(gmp), window_jd, horizon_days, 1.0)
        np.save(cache, nodes)
        print(f"perturber paths: {nodes.shape} in {time.time() - t0:.0f}s", flush=True)
    gm_all = np.where(np.array([x["number"] for x in cat]) == 433, 0.0, gm_all)           # DE440 carries Eros itself: the target cannot pull on itself (as in asteroid_ladder.py)
    t = pp.to_prescribed_sources(nodes, gm_all, 1.0)
    return gmp, PrescribedSources(t.gm, t.t0, t.h, t.pos, t.vel)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", default=["J", "E1", "E2"])
    ap.add_argument("--out", type=Path, default=MPC / "eros" / "results.json")
    a = ap.parse_args()
    from jplephem.spk import SPK
    from sigma_ground.field.interface import comet_orbits as co
    kernel = SPK.open("D:/datasets/jpl/de440s.bsp")
    rows = parse_obs80(MPC / "obs80_433.txt")
    codes = parse_obscodes(MPC / "ObsCodes.html")
    print(f"optical C/B lines: {len(rows)}; stations with constants: {len(codes)}", flush=True)
    obs = Obs(rows, codes, kernel)
    post = obs.year >= POST_FIRST[0]
    pre = (obs.year >= PRE_FIRST[0]) & (obs.year <= PRE_LAST[0])
    print(f"observations kept {obs.n}: 2021+ {int(post.sum())}, 1990-2020 {int(pre.sum())}", flush=True)
    tj, pj, vj = load_horizons()
    result = {"pre_registration": "misc/CANDIDATE_LEDGER.md C24 (commit 5751049)", "n_obs_2021_plus": int(post.sum()), "n_obs_1990_2020": int(pre.sum())}

    # arm J: JPL's own positions through the same pipeline (2021 onwards only; Horizons covers 2019-12-31 onwards)
    opost = Obs.__new__(Obs); opost.__dict__.update({k: (v[post] if isinstance(v, np.ndarray) and v.shape[:1] == (obs.n,) else v) for k, v in obs.__dict__.items()})
    opost.n = int(post.sum()); opost.code = [c for c, f in zip(obs.code, post) if f]
    jfn = lambda T: hermite(tj, pj, vj, T)
    T, tau = light_time_targets(jfn, opost)
    resJ = radec_residuals(opost, jfn(T))
    ok = np.hypot(resJ[:, 0], resJ[:, 1]) <= 5.0
    result["J_all"] = summarize(resJ)
    result["outliers_dropped"] = int((~ok).sum())
    result["J"] = summarize(resJ[ok])
    print("arm J (JPL positions, same pipeline), all:", result["J_all"]["rms_total_arcsec"], " kept:", json.dumps(result["J"]), " dropped", int((~ok).sum()), flush=True)
    if "E1" not in a.arms and "E2" not in a.arms:
        a.out.write_text(json.dumps(result, indent=1)); return 0

    W0 = 2447880.5                                                                   # 1989-12-27, five days before the fit window
    gmp, src = engine_table(kernel, W0, 2461350.0 - W0, MPC / "eros" / "perturber_nodes.npy")
    i0 = int(np.argmin(np.abs(tj - JD0)))
    assert abs(tj[i0] - JD0) < 1e-6
    y_jpl = np.concatenate([pj[i0], vj[i0]])

    def run(y0, targets):
        out = co.propagate(list(y0), JD0, [float(t) for t in targets], kernel, gmp, gr=True, sources=src, sources_epoch_jd=W0)
        return np.array([out[float(t)][:3] for t in targets])

    # arm E1
    t0 = time.time()
    x1 = run(y_jpl, T)
    res1 = radec_residuals(opost, x1)
    diff = res1[ok] - resJ[ok]
    result["E1"] = summarize(res1[ok])
    result["E1_minus_J"] = {"mean_dra": float(diff[:, 0].mean()), "mean_ddec": float(diff[:, 1].mean()), "mean_offset_arcsec": float(math.hypot(diff[:, 0].mean(), diff[:, 1].mean())),
                            "rms_difference_arcsec": float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))}
    print(f"arm E1 [{time.time() - t0:.0f}s]:", json.dumps(result["E1"]), "\n  E1 - J:", json.dumps(result["E1_minus_J"]), flush=True)
    if "E2" not in a.arms:
        a.out.write_text(json.dumps(result, indent=1)); return 0

    # arm E2: six-element fit to 1990-2020 optical data only
    opre = Obs.__new__(Obs); opre.__dict__.update({k: (v[pre] if isinstance(v, np.ndarray) and v.shape[:1] == (obs.n,) else v) for k, v in obs.__dict__.items()})
    opre.n = int(pre.sum()); opre.code = [c for c, f in zip(obs.code, pre) if f]
    # light time for the fit epochs from the starting state (two passes), then held fixed
    tau = np.zeros(opre.n)
    for _ in range(2):
        xa = run(y_jpl, opre.jd_tdb - tau / DAY)
        tau = np.linalg.norm(xa - opre.x_obs, axis=1) / C_KM_S
    Tpre = opre.jd_tdb - tau / DAY
    steps = np.array([100.0, 100.0, 100.0, 5e-3, 5e-3, 5e-3])
    y = y_jpl.copy()
    weight = np.ones(opre.n)
    for it in range(6):
        r0 = radec_residuals(opre, run(y, Tpre))
        tot = np.hypot(r0[:, 0], r0[:, 1])
        weight = (tot < 3.0).astype(float)
        rms = float(np.sqrt(np.mean(tot[weight > 0] ** 2)))
        print(f"fit iteration {it}: used {int(weight.sum())}/{opre.n}, rms {rms:.4f} arcsec  [{time.time() - t0:.0f}s]", flush=True)
        J = np.empty((2 * opre.n, 6))
        for k in range(6):
            yk = y.copy(); yk[k] += steps[k]
            J[:, k] = ((radec_residuals(opre, run(yk, Tpre)) - r0) / steps[k]).reshape(-1)            # d(res)/dp of the residual O - C itself; solve J dp = -r0
        w2 = np.repeat(weight, 2)
        dp, *_ = np.linalg.lstsq(J * w2[:, None], -(r0.reshape(-1) * w2), rcond=None)
        y = y + dp
        if np.all(np.abs(dp) < 0.02 * steps):
            break
    r_fin = radec_residuals(opre, run(y, Tpre))
    result["E2_fit"] = {"used": int(weight.sum()), "of": int(opre.n), "rms_arcsec_on_fit_data": float(np.sqrt(np.mean(np.sum(r_fin[weight > 0] ** 2, axis=1)))),
                        "state_change_vs_jpl_km": [float(x) for x in (y - y_jpl)[:3]], "state_change_vs_jpl_kmday": [float(x) for x in (y - y_jpl)[3:]]}
    x2 = run(y, T)
    res2 = radec_residuals(opost, x2)
    diff2 = res2[ok] - resJ[ok]
    result["E2"] = summarize(res2[ok])
    result["E2_minus_J"] = {"mean_dra": float(diff2[:, 0].mean()), "mean_ddec": float(diff2[:, 1].mean()), "mean_offset_arcsec": float(math.hypot(diff2[:, 0].mean(), diff2[:, 1].mean())),
                            "rms_difference_arcsec": float(np.sqrt(np.mean(np.sum(diff2 ** 2, axis=1))))}
    result["E2_over_J_rms"] = result["E2"]["rms_total_arcsec"] / result["J"]["rms_total_arcsec"]
    print("arm E2:", json.dumps(result["E2"]), "\n  E2 - J:", json.dumps(result["E2_minus_J"]), "\n  E2/J rms:", result["E2_over_J_rms"], flush=True)
    a.out.write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
