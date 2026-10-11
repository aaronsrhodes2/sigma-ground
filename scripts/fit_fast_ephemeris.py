"""Fit the closed-form ephemeris (sigma_ground.field.interface.fast_ephemeris) to JPL DE440 and score it on dates it never saw.

Build time only: query time is a handful of trig calls. Per body (heliocentric, J2000 ecliptic frame; the Moon geocentric):
  1. osculating elements from the DE440 states -> linear-in-time fit -> least-squares refit of a drifting Keplerian ellipse;
  2. the residual is explained by a sum of sinusoids at integer combinations of the bodies' own mean motions, chosen greedily
     (orthogonal matching pursuit), recorded at a ladder of term counts so the error-versus-complexity curve is measured.
Scoring: Earth-view direction and distance on held-out dates (after the training span) against the DE440 kernel.

usage:  python scripts/fit_fast_ephemeris.py [--bodies Mars Moon ...] [--out DIR] [--quick]
Needs jplephem, scipy and the DE440s kernel at D:/datasets/jpl/de440s.bsp (the repo's usual dataset location).
"""
from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
from sigma_ground.field.interface import fast_ephemeris as fe          # noqa: E402

KERNEL = Path("D:/datasets/jpl/de440s.bsp")
OUT_DEFAULT = _ROOT / "sigma_ground" / "field" / "interface" / "fixtures" / "fast_ephemeris"


def jd_of(y: int, m: int = 1, d: int = 1) -> float:
    return dt.date(y, m, d).toordinal() + 1721424.5


RIDGE = 1e-4                     # relative ridge penalty in the sinusoid refit
TRAIN_END = jd_of(2020)          # overridden by --train-end-year (dry runs)
# body -> (kernel chain relative to the SSB, training start year, step in days, max dictionary frequencies, ladder)
PLAN_LADDER = (0, 4, 8, 16, 32, 64, 128)
MOON_LADDER = (0, 10, 20, 40, 80, 160, 320)
SUN_LADDER = (4, 8, 16, 32, 64)
# frame "helio": orbit about the Sun; "bary": orbit about the solar-system barycentre (the model subtracts the Sun's own wobble)
CONFIG = {
    "Mercury": dict(chain=((0, 1), (1, 199)), start=1950, step=2.0, mfreq=1500, ladder=PLAN_LADDER, deg=1, frame="helio"),
    "Venus":   dict(chain=((0, 2), (2, 299)), start=1950, step=2.0, mfreq=1500, ladder=PLAN_LADDER, deg=1, frame="helio"),
    "EMB":     dict(chain=((0, 3),),          start=1950, step=2.0, mfreq=1500, ladder=PLAN_LADDER, deg=1, frame="helio"),
    "Mars":    dict(chain=((0, 4),),          start=1950, step=2.0, mfreq=1500, ladder=PLAN_LADDER, deg=1, frame="helio"),
    "Jupiter": dict(chain=((0, 5),),          start=1850, step=5.0, mfreq=1500, ladder=PLAN_LADDER, deg=2, frame="bary"),
    "Saturn":  dict(chain=((0, 6),),          start=1850, step=10.0, mfreq=1500, ladder=PLAN_LADDER, deg=2, frame="bary"),
    "Uranus":  dict(chain=((0, 7),),          start=1850, step=20.0, mfreq=1200, ladder=PLAN_LADDER, deg=2, frame="bary"),
    "Neptune": dict(chain=((0, 8),),          start=1850, step=20.0, mfreq=1200, ladder=PLAN_LADDER, deg=2, frame="bary"),
    "Moon":    dict(chain=None,               start=1975, step=2.0, mfreq=2500, ladder=MOON_LADDER, deg=1, frame="helio"),
    "SunBary": dict(chain=((0, 10),),         start=1850, step=5.0, mfreq=800, ladder=SUN_LADDER, deg=None, frame="bary"),
}
ORDER = ["Mercury", "Venus", "EMB", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Moon", "SunBary"]


def low_priority() -> None:
    try:
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)   # BELOW_NORMAL
    except Exception:
        pass


# ---- DE440 access ---------------------------------------------------------------------------------------------------
class Truth:
    def __init__(self) -> None:
        from jplephem.spk import SPK
        self.k = SPK.open(str(KERNEL))

    def _sum(self, chain, jd, deriv=False):
        pos, vel = 0.0, 0.0
        for c, t in chain:
            if deriv:
                p, v = self.k[c, t].compute_and_differentiate(jd)
                pos, vel = pos + p, vel + v
            else:
                pos = pos + self.k[c, t].compute(jd)
        return (pos, vel) if deriv else pos

    def sun_relative(self, body: str, jd):
        """True heliocentric ICRF position (km, shape (3, N)) of any planet-system body, whatever frame it is modelled in."""
        return self._sum(CONFIG[body]["chain"], jd) - self._sum(((0, 10),), jd)

    def helio(self, body: str, jd, deriv=False):
        """Heliocentric (Moon: geocentric) ICRF position [and velocity] in km (km/day), shape (3, N)."""
        if body == "Moon":
            a = self._sum(((3, 301),), jd, deriv)
            b = self._sum(((3, 399),), jd, deriv)
            return (a[0] - b[0], a[1] - b[1]) if deriv else a - b
        if body == "Earth":
            e = self._sum(((0, 3), (3, 399)), jd, deriv)
        elif body in CONFIG and CONFIG[body]["frame"] == "bary":
            return self._sum(CONFIG[body]["chain"], jd, deriv)                 # about the barycentre, no Sun subtracted
        else:
            e = self._sum(CONFIG[body]["chain"], jd, deriv)
        s = self._sum(((0, 10),), jd, deriv)
        return (e[0] - s[0], e[1] - s[1]) if deriv else e - s




# ---- stage 1: drifting Kepler ellipse --------------------------------------------------------------------------------
def osculating(r: np.ndarray, v: np.ndarray, gm: float):
    rn = np.linalg.norm(r, axis=1)
    h = np.cross(r, v)
    hn = np.linalg.norm(h, axis=1)
    inc = np.arccos(h[:, 2] / hn)
    Om = np.arctan2(h[:, 0], -h[:, 1])
    ev = np.cross(v, h) / gm - r / rn[:, None]
    e = np.linalg.norm(ev, axis=1)
    a = 1.0 / (2.0 / rn - (v * v).sum(1) / gm)
    nh = np.stack([np.cos(Om), np.sin(Om), np.zeros_like(Om)], 1)
    hh = h / hn[:, None]
    q = np.cross(hh, nh)
    om = np.arctan2((ev * q).sum(1), (ev * nh).sum(1))
    vp = Om + om
    eh = ev / e[:, None]
    ph = np.cross(hh, eh)
    nu = np.arctan2((r * ph).sum(1), (r * eh).sum(1))
    E = 2.0 * np.arctan2(np.sqrt(1 - e) * np.sin(nu / 2), np.sqrt(1 + e) * np.cos(nu / 2))
    M = E - e * np.sin(E)
    return a, e, inc, Om, vp, vp + M


def fit_kepler(t: np.ndarray, r: np.ndarray, v: np.ndarray, deg: int = 1) -> np.ndarray:
    from scipy.optimize import least_squares
    # initial gravitational parameter from the orbit's own geometry: n from the unwrapped angle in the mean orbital plane
    h = np.cross(r, v).mean(0)
    hh = h / np.linalg.norm(h)
    p = r[0] - hh * np.dot(r[0], hh)
    p /= np.linalg.norm(p)
    q = np.cross(hh, p)
    th = np.unwrap(np.arctan2(r @ q, r @ p))
    n = (th[-1] - th[0]) / (t[-1] - t[0])
    rn = np.linalg.norm(r, axis=1)
    a_est = 0.5 * (rn.max() + rn.min())
    gm = n * n * a_est ** 3
    a, e, inc, Om, vp, L = osculating(r, v, gm)
    T = t / 36525.0
    p0 = np.zeros((6, deg + 1))
    for k, arr in enumerate((a, e, inc, np.unwrap(Om), np.unwrap(vp), np.unwrap(L))):
        p0[k] = np.polyfit(T, arr, deg)[::-1]
    scale = a_est

    def resid(x):
        return ((fe.kepler_ecliptic(x.reshape(6, deg + 1), t) - r) / scale).ravel()

    sol = least_squares(resid, p0.ravel(), x_scale="jac", method="trf", xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=200)
    return sol.x.reshape(6, deg + 1)


# ---- stage 2: sinusoids at combinations of the bodies' own frequencies ------------------------------------------------
def combos(bases: np.ndarray, max_sum: int, max_nz: int, wmin: float, wmax: float, cap: int, tol: float) -> np.ndarray:
    """Positive frequencies k.bases with small integers k, lowest order first, near-duplicates removed."""
    nb = len(bases)
    cand: list[tuple[int, float]] = []
    for nz in range(1, max_nz + 1):
        for idx in itertools.combinations(range(nb), nz):
            for ks in itertools.product([k for k in range(-max_sum, max_sum + 1) if k != 0], repeat=nz):
                s = sum(abs(k) for k in ks)
                if s > max_sum:
                    continue
                w = float(sum(k * bases[i] for k, i in zip(ks, idx)))
                if w > 0 and wmin <= w <= wmax:
                    cand.append((s, w))
    cand.sort()
    keep = np.empty(cap)
    nk = 0
    for _, w in cand:
        if nk >= cap:
            break
        if nk == 0 or float(np.min(np.abs(keep[:nk] - w))) > tol:
            keep[nk] = w
            nk += 1
    return keep[:nk].copy()


def omp(t: np.ndarray, resid: np.ndarray, freqs: np.ndarray, kmax: int):
    """Greedy selection of sinusoid frequencies (orthogonal matching pursuit) over the 3 axes jointly. Returns indices in order."""
    N = len(t)
    M = len(freqs)
    A = np.empty((N, 2 * M), dtype=np.float32)
    for j0 in range(0, M, 256):                              # chunked to keep memory small
        j1 = min(M, j0 + 256)
        ph = np.outer(t, freqs[j0:j1]).astype(np.float64)
        A[:, j0:j1] = np.cos(ph)
        A[:, M + j0:M + j1] = np.sin(ph)
    del ph
    Q = np.empty((N, 2 * kmax + 2))
    Q[:, 0] = 1.0 / math.sqrt(N)
    tc = (t - t.mean())
    tc = tc - Q[:, 0] * (Q[:, 0] @ tc)
    Q[:, 1] = tc / np.linalg.norm(tc)
    nq = 2
    R = resid - Q[:, :nq] @ (Q[:, :nq].T @ resid)
    chosen: list[int] = []
    for _ in range(kmax):
        c = A.T @ R.astype(np.float32)
        score = (c[:M] ** 2 + c[M:] ** 2).sum(1)
        score[chosen] = -1.0
        j = int(np.argmax(score))
        chosen.append(j)
        for col in (A[:, j].astype(np.float64), A[:, M + j].astype(np.float64)):
            for _ in range(2):
                col = col - Q[:, :nq] @ (Q[:, :nq].T @ col)
            nrm = np.linalg.norm(col)
            if nrm > 1e-8 * math.sqrt(N):
                Q[:, nq] = col / nrm
                nq += 1
        R = resid - Q[:, :nq] @ (Q[:, :nq].T @ resid)
    return chosen, A, M


def refit(t: np.ndarray, resid: np.ndarray, freqs: np.ndarray):
    """Least-squares amplitudes for a given frequency set: returns coef (K,2,3) km and poly (2,3) km (constant, per century)."""
    T = t / 36525.0
    cols = [np.ones_like(t), T - T.mean()]
    if len(freqs):
        ph = np.outer(t, freqs)
        cols_s = np.concatenate([np.cos(ph), np.sin(ph)], axis=1)
        D = np.column_stack(cols + [cols_s])
    else:
        D = np.column_stack(cols)
    # ridge penalty (relative to each column's energy) so near-collinear atoms cannot take huge cancelling amplitudes
    nrm = np.linalg.norm(D, axis=0)
    Dn = D / nrm
    lam = RIDGE
    aug = np.vstack([Dn, math.sqrt(lam) * np.eye(Dn.shape[1])])
    rhs = np.vstack([resid, np.zeros((Dn.shape[1], resid.shape[1]))])
    sol, *_ = np.linalg.lstsq(aug, rhs, rcond=None)
    sol = sol / nrm[:, None]
    poly = np.zeros((2, 3))
    poly[1] = sol[1]
    poly[0] = sol[0] - sol[1] * T.mean()                      # constant at T = 0
    K = len(freqs)
    coef = np.zeros((K, 2, 3))
    if K:
        coef[:, 0, :] = sol[2:2 + K]
        coef[:, 1, :] = sol[2 + K:2 + 2 * K]
    return coef, poly


def build_body(truth: Truth, name: str, bases: np.ndarray, cfg: dict, quick: bool, train_end: float) -> dict:
    t0 = time.time()
    jd = np.arange(jd_of(cfg["start"]), train_end, cfg["step"])
    pos, vel = truth.helio(name, jd, deriv=True)
    r = pos.T @ fe._ECL_TO_EQ                                   # equatorial -> ecliptic
    v = vel.T @ fe._ECL_TO_EQ
    t = jd - fe.J2000
    if cfg["deg"] is None:                                       # pure periodic body (the Sun's wobble about the barycentre)
        p, base = None, 0.0
    else:
        p = fit_kepler(t, r, v, cfg["deg"])
        base = fe.kepler_ecliptic(p, t)
    resid = r - base
    rms0 = float(np.sqrt((resid ** 2).sum(1).mean()))
    span = t[-1] - t[0]
    ladder_k = [k for k in cfg["ladder"] if k > 0]
    kmax = max(ladder_k) if not quick else min(max(ladder_k), 16)
    freqs = combos(bases, max_sum=5, max_nz=3 if name != "Moon" else 4, wmin=2 * math.pi / (span / 3.0),
                   wmax=math.pi / (1.2 * cfg["step"]) * 0.98, cap=cfg["mfreq"], tol=0.3 * 2 * math.pi / span)
    chosen, A, M = omp(t, resid, freqs, kmax)
    del A
    ladder = {}
    for K in cfg["ladder"]:
        if K > kmax:
            continue
        fs = freqs[chosen[:K]]
        coef, poly = refit(t, resid, fs)
        ladder[str(K)] = {"freq_rad_day": fs.tolist(), "coef_km": coef.ravel().tolist(), "poly_km": poly.ravel().tolist()}
    info = {"kepler_rms_km": rms0, "n_dictionary": int(len(freqs)), "fit_seconds": round(time.time() - t0, 1),
            "train_start_jd": float(jd[0]), "train_samples": int(len(jd))}
    print(f"  {name:8s} kepler residual rms {rms0:12.1f} km, dictionary {len(freqs)}, {time.time() - t0:.0f}s", flush=True)
    return {"elements": p.tolist() if p is not None else None, "frame": cfg["frame"], "ladder": ladder, "info": info}


def base_rates(truth: Truth, train_end: float) -> dict:
    """Mean motions (rad/day) of every body, from a coarse unwrapped-longitude fit, used to build frequency dictionaries."""
    rates = {}
    for name in ORDER:
        c = CONFIG[name]
        if c["deg"] is None:
            continue
        jd = np.arange(jd_of(c["start"]), train_end, max(c["step"], 5.0) if name != "Moon" else 1.0)
        pos, vel = truth.helio(name, jd, deriv=True)
        r = pos.T @ fe._ECL_TO_EQ
        v = vel.T @ fe._ECL_TO_EQ
        t = jd - fe.J2000
        p = fit_kepler(t, r, v, c["deg"])
        rates[name] = {"n": p[5, 1] / 36525.0, "varpi": p[4, 1] / 36525.0, "Omega": p[3, 1] / 36525.0}
    return rates


# ---- scoring --------------------------------------------------------------------------------------------------------
BUCKETS = ((0.0, 10.0), (10.0, 30.0), (30.0, 80.0))


def score(truth: Truth, doc: dict, terms: int | None, targets: list[str]) -> dict:
    eph = fe.FastEphemeris(doc=doc, terms=terms)
    out = {}
    start = TRAIN_END - 30 * 365.25
    ranges = {"in_sample": (start, TRAIN_END, 7.0)}
    for lo, hi in BUCKETS:
        ranges[f"{int(lo)}-{int(hi)}"] = (TRAIN_END + lo * 365.25 + (0.5 if lo == 0 else 0), TRAIN_END + hi * 365.25,
                                         2.0 if hi <= 10 else (7.0 if hi <= 30 else 15.0))
    for name in targets:
        out[name] = {}
        for lab, (a, b, st) in ranges.items():
            jd = np.arange(a, b, st)
            if name == "Moon":
                true_geo = truth.helio("Moon", jd).T
            else:
                tp = truth.sun_relative(name, jd).T if name != "Sun" else np.zeros((len(jd), 3))
                ep = truth.helio("Earth", jd).T
                true_geo = tp - ep
            if name == "Moon":
                pred = eph.geocentric_moon(jd)
            elif name == "Sun":
                pred = -eph.heliocentric("Earth", jd)
            else:
                pred = eph.heliocentric(name, jd) - eph.heliocentric("Earth", jd)
            ang = np.degrees(np.arccos(np.clip((true_geo * pred).sum(1) /
                                               (np.linalg.norm(true_geo, axis=1) * np.linalg.norm(pred, axis=1)), -1, 1))) * 3600.0
            km = np.linalg.norm(true_geo - pred, axis=1)
            out[name][lab] = {"median_arcsec": float(np.median(ang)), "p95_arcsec": float(np.percentile(ang, 95)),
                              "max_arcsec": float(ang.max()), "median_km": float(np.median(km)),
                              "p95_km": float(np.percentile(km, 95)), "n": int(len(jd))}
    return out


GIANTS = ("Jupiter", "Saturn", "Uranus", "Neptune")


def build_reflex(truth: Truth, train_end: float) -> dict:
    """The Sun's motion about the barycentre as a fixed linear combination of the giants' barycentric positions.

    Barycentre condition: M_sun r_sun + sum m_i r_i = 0, so r_sun = -sum (m_i / M_sun) r_i. The coefficients are fitted (they come out near
    minus the planet/Sun mass ratios, which is the check) on the true positions, then applied to the *modelled* giants at query time.
    """
    jd = np.arange(jd_of(CONFIG["SunBary"]["start"]), train_end, CONFIG["SunBary"]["step"])
    sun = truth.helio("SunBary", jd).T @ fe._ECL_TO_EQ
    cols = [truth.helio(g, jd).T @ fe._ECL_TO_EQ for g in GIANTS]
    A = np.stack([c.ravel() for c in cols], axis=1)
    coef, *_ = np.linalg.lstsq(A, sun.ravel(), rcond=None)
    resid = sun - sum(c * col for c, col in zip(coef, cols))
    print(f"  SunBary reflex: coefficients {np.round(coef, 8)}, residual rms {np.sqrt((resid ** 2).sum(1).mean()):.0f} km", flush=True)
    return {"elements": None, "frame": "bary", "reflex": {g: float(c) for g, c in zip(GIANTS, coef)},
            "ladder": {"0": {"freq_rad_day": [], "coef_km": [], "poly_km": [0.0] * 6}}, "default_terms": 0,
            "validation_median_km": {}, "info": {"reflex_residual_rms_km": float(np.sqrt((resid ** 2).sum(1).mean()))}}


def make_bodies(truth: Truth, names: list[str], train_end: float, quick: bool) -> dict:
    rates = base_rates(truth, train_end)
    planet_bases = np.array([rates[n]["n"] for n in ("Mercury", "Venus", "EMB", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune")])
    moon_bases = np.array([rates["Moon"]["n"], rates["Moon"]["varpi"], rates["Moon"]["Omega"], rates["EMB"]["n"],
                           rates["EMB"]["varpi"], rates["Jupiter"]["n"]])
    sun_bases = np.array([rates[n]["n"] for n in ("Jupiter", "Saturn", "Uranus", "Neptune")])
    out = {}
    for name in ORDER:
        if name not in names:
            continue
        if name == "SunBary" and all(g in names for g in GIANTS):
            out[name] = build_reflex(truth, train_end)
            continue
        bases = moon_bases if name == "Moon" else (sun_bases if name == "SunBary" else planet_bases)
        out[name] = build_body(truth, name, bases, CONFIG[name], quick, train_end)
    return out


def choose_terms(truth: Truth, names: list[str], quick: bool) -> dict:
    """Pick each body's term count on data inside the training span: fit on the first part, validate on the last 15 years."""
    sel_end = TRAIN_END - 15 * 365.25
    print(f"term selection: fit to {sel_end:.0f}, validate to {TRAIN_END:.0f}", flush=True)
    bodies = make_bodies(truth, names, sel_end, quick)
    doc = {"metadata": {"earth_from_emb_over_moon_geo": 0.0}, "bodies": bodies}
    eph = fe.FastEphemeris(doc=doc)
    chosen = {}
    for name, b in bodies.items():
        jd = np.arange(sel_end, TRAIN_END, 5.0)
        tr = truth.helio(name, jd).T @ fe._ECL_TO_EQ
        res = {}
        for K in b["ladder"]:
            eph._terms = int(K)
            eph._cache.clear()
            err = np.linalg.norm(eph._ecl(name, jd) - tr, axis=1)
            res[int(K)] = float(np.median(err))
        chosen[name] = {"default_terms": min(res, key=res.get), "validation_median_km": res}
        print(f"  {name:8s} best {chosen[name]['default_terms']:4d} terms  "
              + " ".join(f"{k}:{v:.0f}" for k, v in res.items()), flush=True)
    return chosen


def main() -> int:
    global TRAIN_END
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bodies", nargs="+", default=ORDER)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--train-end-year", type=int, default=2020, help="forecast origin: fit only to data before 1 Jan of this year")
    ap.add_argument("--quick", action="store_true", help="16 terms max, for a smoke test")
    a = ap.parse_args()
    TRAIN_END = jd_of(a.train_end_year)
    low_priority()
    truth = Truth()
    t0 = time.time()
    chosen = choose_terms(truth, a.bodies, a.quick)
    print("final fit ...", flush=True)
    bodies = make_bodies(truth, a.bodies, TRAIN_END, a.quick)
    for n, b in bodies.items():
        b["default_terms"] = chosen[n]["default_terms"]
        b["validation_median_km"] = chosen[n]["validation_median_km"]

    # derived constant: |Earth - EMB| / |Moon - Earth|, straight from the kernel (Earth = EMB - ratio * geocentric Moon)
    jd = np.array([jd_of(2000)])
    e_emb = truth._sum(((3, 399),), jd)
    moon_geo = truth.helio("Moon", jd)
    ratio = float(np.linalg.norm(e_emb) / np.linalg.norm(moon_geo))

    doc = {"metadata": {"model_name": "fast_ephemeris v1 (drifting Kepler ellipse + combination-frequency sinusoids)",
                        "fit_to": "JPL DE440s", "train_end_jd": TRAIN_END, "baseline_label": "fast_ephemeris_2026-10-11",
                        "earth_from_emb_over_moon_geo": ratio, "frame": "ICRF/J2000 (fit in the J2000 ecliptic)",
                        "obliquity_arcsec": fe.OBLIQUITY_J2000_ARCSEC, "generator": "scripts/fit_fast_ephemeris.py",
                        "generated": dt.datetime.now().isoformat(timespec="seconds"), "quick": a.quick,
                        "train_end_year": a.train_end_year,
                        "note": "Jupiter..Neptune are system barycentres, as in the DE440 planetary segments"},
           "bodies": bodies, "errors": {}}
    # the Earth's-view targets we can score from what was fitted
    avail = set(bodies)
    targets = [n for n in ("Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Moon", "Sun")
               if (n == "Sun" and {"EMB", "Moon"} <= avail) or (n == "Moon" and "Moon" in avail) or
               (n in avail and {"EMB", "Moon"} <= avail and (CONFIG[n]["frame"] != "bary" or "SunBary" in avail))]
    ladders = sorted({int(k) for b in bodies.values() for k in b["ladder"]})
    scores = {}
    for K in ladders:
        scores[str(K)] = score(truth, doc, K, targets)
    final = score(truth, doc, None, targets)
    scores["full"] = final
    for name in targets:
        doc["errors"][name] = {"buckets": [
            {"from_years": 0.0, "to_years": 0.0, "label": "inside the training span", **{k: final[name]["in_sample"][k] for k in
              ("median_arcsec", "p95_arcsec", "median_km", "p95_km")}}] + [
            {"from_years": lo, "to_years": hi, **{k: final[name][f"{int(lo)}-{int(hi)}"][k] for k in
              ("median_arcsec", "p95_arcsec", "median_km", "p95_km")}} for lo, hi in BUCKETS]}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "fast_ephemeris_coeffs.json").write_text(json.dumps(doc, separators=(",", ":")))
    (a.out / "fast_ephemeris_scores.json").write_text(json.dumps({"targets": targets, "scores": scores}, indent=1))
    print(f"done in {time.time() - t0:.0f}s -> {a.out}", flush=True)
    for n in targets:
        f = final[n]
        print(f"{n:8s} in-sample {f['in_sample']['median_arcsec']:8.2f}\"  0-10yr {f['0-10']['median_arcsec']:8.2f}\"  "
              f"10-30yr {f['10-30']['median_arcsec']:8.2f}\"  30-80yr {f['30-80']['median_arcsec']:8.2f}\"", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
