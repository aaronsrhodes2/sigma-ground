"""U5 v2 probes on the repo's own N-body force law (no integration, no engine change).

usage (repo root as cwd): python -X utf8 u5_probe_v2.py <outdir>      -> <outdir>/probe_v2.json

1. Force probe off perihelion. At true anomalies f = 0, pi/2, 2, 4 rad on the nominal Mercury orbit, the engine's
   relative-acceleration perturbation da = (a_mode[1]-a_mode[0]) - (a_newt[1]-a_newt[0]) is fitted as
   da = alpha * R_vec + beta * V_vec, with the IERS Conventions eq. 10.12 (beta_PPN = gamma = 1) pieces
     R_vec = GM/(c^2 r^3) (4GM/r - v^2) x,    V_vec = GM/(c^2 r^3) 4 (x.v) v     (x from Sun to Mercury).
   IERS itself is alpha = beta = +1. At perihelion x.v = 0, so only alpha is visible there; the other anomalies fix beta.
2. Orbit-averaged Gauss quadrature on the engine forces (a second method, independent of the integrator and of the
   LRL estimator): d(omega)/dt = (1/e) sqrt(p/mu) [-cos f R + (1 + r/p) sin f S] (Gauss equation as given in
   Hugentobler 2008 eq. 6), integrated over one unperturbed Kepler orbit with dt = r^2/h df, divided by
   6 pi GM/(c^2 p). Modes: IERS reference, engine gr_1pn, engine eih_cross, sigma = xi r_s/r, sigma = xi r_s/2r.
3. Static PN probe: Mercury held at rest at x = GM/(r c^2) in {1e-4, 1e-3, 1e-2}; outward radial da in units of
   GM/r^2 gives the engine's static 1PN coefficient (gr_1pn, eih_cross; harmonic Schwarzschild: +4) and 2PN
   coefficient (gr_2pn; harmonic Schwarzschild: -9, from Mentat series_expansion in leaves_v2.json).
Every physical number comes from leaves_v2.json (Mentat values + sources) or the engine itself.
"""
import sys, os, json, math
import numpy as np
sys.path.insert(0, r"D:/Aaron/development/sigma-ground")
from sigma_ground.field.interface.nbody import NBodySystem, CelestialBody, PhysicsToggles
from sigma_ground.inventory.core.sigma import XI, sigma_from_potential

D = sys.argv[1]
L = json.load(open(os.path.join(D, "leaves_v2.json")))
sun, mer = L["sun"]["value"], L["mercury"]["value"]
G, c, rs = L["G"]["value"], L["c"]["value"], L["r_s_sun"]["value"]
a, e = L["a_mercury_m"]["value"], L["e_mercury"]["value"]
M, m = sun["mass_kg"], mer["mass_kg"]
GM = G * M
mu = G * (M + m)
p = a * (1 - e * e)
h = math.sqrt(mu * p)
dwGR = 6 * math.pi * GM / (c * c * p)     # same expression Mentat evaluated in leaves_v2 (dw_GR_rad_per_orbit)
dwGR_mentat = float(L["dw_GR_rad_per_orbit"]["value"].replace("*pi", "")) * math.pi


def kepler_state(f):
    r = p / (1 + e * math.cos(f))
    x = np.array([r * math.cos(f), r * math.sin(f), 0.0])
    v = math.sqrt(mu / p) * np.array([-math.sin(f), e + math.cos(f), 0.0])
    return x, v


def bodies_at(x, v, sigma_sun=0.0):
    return [CelestialBody(M, np.zeros(3), np.zeros(3), sun["radius_m"], 0.0, sigma_field=sigma_sun),
            CelestialBody(m, x.copy(), v.copy(), mer["radius_m"], 0.0)]


def rel(acc):
    return acc[1] - acc[0]


def engine_da(x, v, mode):
    newt = rel(NBodySystem(bodies_at(x, v), softening_m=0.0).compute_accelerations())
    r = float(np.linalg.norm(x))
    if mode == "gr_1pn":
        S = NBodySystem(bodies_at(x, v), softening_m=0.0, toggles=PhysicsToggles(gr_1pn=True))
    elif mode == "eih_cross":
        S = NBodySystem(bodies_at(x, v), softening_m=0.0, toggles=PhysicsToggles(eih_cross=True))
    elif mode == "gr_2pn":
        S = NBodySystem(bodies_at(x, v), softening_m=0.0, toggles=PhysicsToggles(gr_2pn=True))
    elif mode == "sigma_field_eq":
        S = NBodySystem(bodies_at(x, v, XI * rs / r), softening_m=0.0)
    elif mode == "sigma_from_potential":
        S = NBodySystem(bodies_at(x, v, sigma_from_potential(r, M)), softening_m=0.0)
    else:
        raise ValueError(mode)
    return rel(S.compute_accelerations()) - newt


def iers_pieces(x, v):
    r = float(np.linalg.norm(x))
    k = GM / (c * c * r ** 3)
    return k * (4 * GM / r - v @ v) * x, k * 4 * (x @ v) * v


def iers_da(x, v):
    R, V = iers_pieces(x, v)
    return R + V


out = {"id": "U5_v2_probes", "leaves_file": os.path.join(D, "leaves_v2.json")}

# ---------------- 1. force probe off perihelion ----------------
F = [0.0, math.pi / 2, 2.0, 4.0]
probe = {}
for mode in ("gr_1pn", "eih_cross"):
    rows, rhs, per_f = [], [], []
    for f in F:
        x, v = kepler_state(f)
        da = engine_da(x, v, mode)
        R, V = iers_pieces(x, v)
        ref = R + V
        per_f.append({"f_rad": f, "xv_over_rv": float((x @ v) / (np.linalg.norm(x) * np.linalg.norm(v))),
                      "cos_engine_vs_IERS": float(da @ ref / (np.linalg.norm(da) * np.linalg.norm(ref))),
                      "norm_ratio_engine_over_IERS": float(np.linalg.norm(da) / np.linalg.norm(ref)),
                      "rel_residual_vs_minus_IERS": float(np.linalg.norm(da + ref) / np.linalg.norm(ref)),
                      "rel_residual_vs_IERS_cross_flipped": float(np.linalg.norm(da - (R - V)) / np.linalg.norm(ref))})
        for k in range(2):           # in-plane components
            rows.append([R[k], V[k]]); rhs.append(da[k])
    A_ = np.array(rows); b_ = np.array(rhs)
    scale = np.abs(A_).max()
    coef, res, rank, sv = np.linalg.lstsq(A_ / scale, b_ / scale, rcond=None)
    fit_res = float(np.linalg.norm(A_ @ coef - b_) / np.linalg.norm(b_))
    probe[mode] = {"alpha_radial_coef_vs_IERS": float(coef[0]), "beta_velocity_coef_vs_IERS": float(coef[1]),
                   "fit_rel_residual": fit_res, "per_anomaly": per_f}
out["force_probe"] = probe

# ---------------- 2. orbit-averaged Gauss quadrature ----------------
N = 2048                                    # uniform grid in f; periodic smooth integrand -> spectral convergence
fs = np.arange(N) * 2 * math.pi / N
gauss = {}
for mode in ("IERS_reference", "gr_1pn", "eih_cross", "sigma_field_eq", "sigma_from_potential"):
    tot = 0.0
    for f in fs:
        x, v = kepler_state(f)
        r = float(np.linalg.norm(x))
        da = iers_da(x, v) if mode == "IERS_reference" else engine_da(x, v, mode)
        eR = x / r
        eS = np.array([-eR[1], eR[0], 0.0])
        Rc, Sc = float(da @ eR), float(da @ eS)
        wdot = (1 / e) * math.sqrt(p / mu) * (-math.cos(f) * Rc + (1 + r / p) * math.sin(f) * Sc)
        tot += wdot * r * r / h
    dw = tot * 2 * math.pi / N
    gauss[mode] = {"rad_per_orbit": dw, "over_GR_analytic": dw / dwGR}
# convergence check: half the grid for the sigma hook
tot = 0.0
for f in fs[::2]:
    x, v = kepler_state(f); r = float(np.linalg.norm(x))
    da = engine_da(x, v, "sigma_field_eq"); eR = x / r; eS = np.array([-eR[1], eR[0], 0.0])
    tot += (1 / e) * math.sqrt(p / mu) * (-math.cos(f) * (da @ eR) + (1 + r / p) * math.sin(f) * (da @ eS)) * r * r / h
gauss["sigma_field_eq_halfgrid_over_GR"] = tot * 2 * math.pi / (N // 2) / dwGR
gauss["grid_points"] = N
gauss["dwGR_python_vs_mentat_rel"] = dwGR / dwGR_mentat - 1
out["gauss_quadrature"] = gauss

# ---------------- 3. static PN coefficients ----------------
static = {}
for mode in ("gr_1pn", "eih_cross", "gr_2pn"):
    rows = []
    for xpn in (1e-4, 1e-3, 1e-2):
        r = GM / (c * c * xpn)
        x = np.array([r, 0.0, 0.0]); v = np.zeros(3)
        da = engine_da(x, v, mode)
        dar = float(da @ (x / r)) / (GM / r ** 2)      # outward radial, units of GM/r^2
        power = 2 if mode == "gr_2pn" else 1
        rows.append({"x": xpn, "radial_da_over_GM_r2": dar, "coefficient": dar / xpn ** power,
                     "transverse_rel": float(np.linalg.norm(da - (da @ (x / r)) * (x / r)) / (abs(dar) * GM / r ** 2))})
    static[mode] = rows
static["harmonic_schwarzschild_expansion"] = {"value": L["harm_static_accel_series"]["value"],
                                              "source": L["harm_static_accel_series"]["source"]}
out["static_pn"] = static

json.dump(out, open(os.path.join(D, "probe_v2.json"), "w"), indent=1)
print(json.dumps({"force_probe": {k: {kk: vv for kk, vv in v.items() if kk != "per_anomaly"} for k, v in probe.items()},
                  "per_anomaly_gr_1pn": probe["gr_1pn"]["per_anomaly"],
                  "per_anomaly_eih": probe["eih_cross"]["per_anomaly"],
                  "gauss": gauss, "static": static}, indent=1))
