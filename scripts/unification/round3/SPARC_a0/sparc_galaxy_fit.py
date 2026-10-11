"""SPARC a0, per galaxy: fit the radial-acceleration-relation scale g_dagger to the 175 individual rotation curves with free stellar mass-to-light ratios.

Follows the earlier tabulated-RAR fit (sparc_a0_fit.py) by removing its main systematic: the fixed mass-to-light ratios. Per galaxy the disc and bulge ratios
are free with log-normal priors (0.5 and 0.7 solar, 0.1 dex: the SPARC convention, Lelli et al. 2016) and the model is
    V_model^2 = R g_bar / (1 - exp(-sqrt(g_bar / g_dagger))),   g_bar = (|V_gas| V_gas + Y_d |V_disk| V_disk + Y_b |V_bul| V_bul) / R,
fitted to V_obs with the tabulated errors (floor 3 km/s as in the SPARC analysis would be an extra choice; here the tabulated error plus a 2 % floor).
Quality cut (as Lelli et al. 2017): Q <= 2 and inclination >= 30 deg. Uncertainty on g_dagger: bootstrap over galaxies (the independent units).

Not modelled: distance and inclination uncertainties (they correlate galaxy by galaxy); the prediction is the one from Mentat (cH0/2pi = 1.042e-10 m/s^2, H0 = 67.4).
Run:  python -X utf8 scripts/unification/round3/SPARC_a0/sparc_galaxy_fit.py
"""
import io
import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

DATA = Path("D:/datasets/sparc")
OUT = Path(__file__).parent / "sparc_galaxy_results.json"
KPC_M = 3.0856775814913673e19
A0_PRED = 1.042e-10


def load_galaxies():
    table = {}
    for line in (DATA / "SPARC_Lelli2016c.mrt").read_text().splitlines():
        tok = line.split()
        # a data row: name, T, D, e_D, f_D, Inc, e_Inc, L, e_L, Reff, SBeff, Rdisk, SBdisk, MHI, RHI, Vflat, e_Vflat, Q, Ref
        if len(tok) >= 18 and tok[1].lstrip("-").isdigit() and tok[17].isdigit():
            table[tok[0]] = {"inc": float(tok[5]), "Q": int(tok[17])}
    z = zipfile.ZipFile(DATA / "Rotmod_LTG.zip")
    gals = []
    for name in z.namelist():
        gname = name.replace("_rotmod.dat", "")
        meta = table.get(gname)
        if meta is None or meta["Q"] > 2 or meta["inc"] < 30.0:
            continue
        rows = [[float(x) for x in ln.split()] for ln in z.read(name).decode().splitlines() if ln.strip() and not ln.startswith("#")]
        a = np.array(rows)
        if len(a) < 5:
            continue
        gals.append({"name": gname, "R": a[:, 0] * KPC_M, "V": a[:, 1] * 1e3, "eV": np.maximum(a[:, 2] * 1e3, 0.02 * a[:, 1] * 1e3),
                     "Vg": a[:, 3] * 1e3, "Vd": a[:, 4] * 1e3, "Vb": a[:, 5] * 1e3})
    return gals


def gbar(g, yd, yb):
    return (np.abs(g["Vg"]) * g["Vg"] + yd * np.abs(g["Vd"]) * g["Vd"] + yb * np.abs(g["Vb"]) * g["Vb"]) / g["R"]


def vmodel(g, yd, yb, gd):
    gb = np.maximum(gbar(g, yd, yb), 1e-16)
    return np.sqrt(g["R"] * gb / (1.0 - np.exp(-np.sqrt(gb / gd))))


def chi2_galaxy(g, ly, gd, has_bulge):
    yd = 10.0 ** ly[0]
    yb = 10.0 ** ly[1] if has_bulge else 0.7
    r = (g["V"] - vmodel(g, yd, yb, gd)) / g["eV"]
    c = float((r ** 2).sum()) + ((ly[0] - math.log10(0.5)) / 0.1) ** 2
    if has_bulge:
        c += ((ly[1] - math.log10(0.7)) / 0.1) ** 2
    return c


def best_ml(g, gd):
    has_b = bool(np.any(np.abs(g["Vb"]) > 0))
    if not has_b:
        res = minimize_scalar(lambda a: chi2_galaxy(g, (a, 0.0), gd, False), bounds=(math.log10(0.1), math.log10(2.0)), method="bounded")
        return (res.x, math.log10(0.7)), res.fun, False
    res = minimize_scalar(lambda a: minimize_scalar(lambda b: chi2_galaxy(g, (a, b), gd, True), bounds=(math.log10(0.2), math.log10(1.5)),
                                                    method="bounded").fun, bounds=(math.log10(0.1), math.log10(2.0)), method="bounded")
    a = res.x
    rb = minimize_scalar(lambda b: chi2_galaxy(g, (a, b), gd, True), bounds=(math.log10(0.2), math.log10(1.5)), method="bounded")
    return (a, rb.x), rb.fun, True


def total_chi2(gals, gd, cache=None):
    tot = 0.0
    ml = []
    for g in gals:
        m, c, hb = best_ml(g, gd)
        tot += c
        ml.append(m)
    return tot, ml


def main():
    gals = load_galaxies()
    print(f"{len(gals)} galaxies pass the cuts (Q <= 2, inclination >= 30 deg)")
    # fixed mass-to-light ratios (0.5, 0.7): the standard SPARC fit
    def fixed(gd):
        return sum(float((((g["V"] - vmodel(g, 0.5, 0.7, gd)) / g["eV"]) ** 2).sum()) for g in gals)
    fx = minimize_scalar(lambda lg: fixed(10.0 ** lg), bounds=(-10.6, -9.4), method="bounded", options={"xatol": 1e-6})
    gd_fixed = 10.0 ** fx.x
    free = minimize_scalar(lambda lg: total_chi2(gals, 10.0 ** lg)[0], bounds=(-10.6, -9.4), method="bounded", options={"xatol": 1e-4})
    gd_free = 10.0 ** free.x
    _, ml = total_chi2(gals, gd_free)
    yd = np.array([10.0 ** m[0] for m in ml])
    # bootstrap over galaxies at fixed per-galaxy best ratios (the independent units); both variants
    rng = np.random.default_rng(20261011)
    boot_fixed, boot_free = [], []
    for _ in range(150):
        idx = rng.integers(0, len(gals), len(gals))
        sub = [gals[i] for i in idx]
        r = minimize_scalar(lambda lg: sum(float((((g["V"] - vmodel(g, 0.5, 0.7, 10.0 ** lg)) / g["eV"]) ** 2).sum()) for g in sub),
                            bounds=(-10.6, -9.4), method="bounded", options={"xatol": 1e-5})
        boot_fixed.append(10.0 ** r.x)
        subml = [ml[i] for i in idx]
        r2 = minimize_scalar(lambda lg: sum(float((((g["V"] - vmodel(g, 10.0 ** m[0], 10.0 ** m[1], 10.0 ** lg)) / g["eV"]) ** 2).sum()) for g, m in zip(sub, subml)),
                             bounds=(-10.6, -9.4), method="bounded", options={"xatol": 1e-5})
        boot_free.append(10.0 ** r2.x)
    out = {
        "n_galaxies": len(gals),
        "g_dagger_fixed_ML_0.5_0.7": gd_fixed, "g_dagger_fixed_boot_sd": float(np.std(boot_fixed)),
        "g_dagger_free_ML_priors_0.1dex": gd_free, "g_dagger_free_boot_sd": float(np.std(boot_free)),
        "median_fitted_disc_ML": float(np.median(yd)), "disc_ML_16_84": [float(np.percentile(yd, 16)), float(np.percentile(yd, 84))],
        "a0_prediction_cH0_over_2pi": A0_PRED,
        "pred_over_fit_fixed": A0_PRED / gd_fixed, "pred_over_fit_free": A0_PRED / gd_free,
        "sigmas_fixed": (gd_fixed - A0_PRED) / float(np.std(boot_fixed)), "sigmas_free": (gd_free - A0_PRED) / float(np.std(boot_free)),
        "published_g_dagger_Lelli2017": "1.20e-10 +/- 0.02 (random) +/- 0.24 (systematic) [LIT]",
        "not_modelled": "distance and inclination uncertainties; bulge ratio free only for galaxies with a bulge",
    }
    OUT.write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(f"{k:36s} {v}")


if __name__ == "__main__":
    sys.exit(main())
