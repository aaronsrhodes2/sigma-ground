"""U7 — heat leg: arrival times of a point heat pulse on the repo's FVM heat law (diffuse_fvm, unmodified).

48^3 insulated grid, dx = 1, k = rho_cp = 1 (alpha = 1; scaled units, a disclosed Choice), one hot cell dT0 = 1.
Literal design: 40 log-spaced slices, threshold 1e-3 dT0, d = 2..10 dx.
Corrected: sampled every explicit substep; half-peak arrival (self-similar -> exactly d^2 in the continuum);
plus time-to-peak (continuum t_peak = d^2/(6 alpha)) and an absolute-threshold sweep as a diagnostic.
"""
import json, math, sys, time
import numpy as np
sys.path.insert(0, r"D:/Aaron/development/sigma-ground")
from sigma_ground.dynamics.fields.heat import diffuse_fvm, stable_dt

OUT = r"C:/Users/aaron/AppData/Local/Temp/claude/D--Aaron-development-sigma-ground/c6652910-47a9-478d-9c83-a8a8c8aaa457/scratchpad/round2/U7"
n, dx, kk, rcp = 48, 1.0, 1.0, 1.0
alpha = kk / rcp
c = 24
ds = np.arange(1, 11)
k = np.full((n, n, n), kk); R = np.full((n, n, n), rcp)

def fresh():
    T = np.zeros((n, n, n)); T[c, c, c] = 1.0
    return T

def ols(x, y):
    A = np.vstack([np.log(x), np.ones_like(x)]).T
    (p, b), *_ = np.linalg.lstsq(A, np.log(y), rcond=None)
    return float(p), float(b)

t0 = time.time()
# ── literal: 40 log-spaced slices ──
edges = np.logspace(math.log10(0.02), math.log10(60.0), 40)
T = fresh(); tprev = 0.0; lit_arr = {}
slice_vals = []
for te in edges:
    T, info = diffuse_fvm(T, k, R, dx, total_time=te - tprev, max_substeps=10 ** 7)
    tprev = te
    vals = T[c + ds, c, c]
    slice_vals.append(vals.tolist())
    for d, v in zip(ds, vals):
        if d >= 2 and int(d) not in lit_arr and v > 1e-3:
            lit_arr[int(d)] = float(te)
lit_d = np.array(sorted(lit_arr)); lit_t = np.array([lit_arr[d] for d in lit_d])
lit_p = ols(lit_d, lit_t)[0] if len(lit_d) >= 2 else None
energy_literal = float(T.sum())
# ── corrected: every substep ──
dt_sub = stable_dt(alpha, dx)               # the engine's own CFL step (safety 0.2)
t_end = 40.0
nsub = int(math.ceil(t_end / dt_sub))
T = fresh(); times = np.zeros(nsub + 1); series = np.zeros((nsub + 1, len(ds)))
series[0] = T[c + ds, c, c]
for i in range(1, nsub + 1):
    T, info = diffuse_fvm(T, k, R, dx, total_time=dt_sub, max_substeps=1)
    times[i] = i * dt_sub
    series[i] = T[c + ds, c, c]
energy_corr = float(T.sum())

def first_cross(tt, yy, thr):
    idx = np.nonzero(yy >= thr)[0]
    if len(idx) == 0:
        return None
    j = idx[0]
    if j == 0:
        return float(tt[0])
    return float(tt[j - 1] + (thr - yy[j - 1]) * (tt[j] - tt[j - 1]) / (yy[j] - yy[j - 1]))

half = {}; tpk = {}; peak = {}
for m, d in enumerate(ds):
    y = series[:, m]; jm = int(np.argmax(y)); peak[int(d)] = float(y[jm]); tpk[int(d)] = float(times[jm])
    half[int(d)] = first_cross(times, y, 0.5 * y[jm])
sel = [d for d in ds if d >= 2]
p_half, b_half = ols(np.array(sel, float), np.array([half[d] for d in sel]))
p_peak, b_peak = ols(np.array(sel, float), np.array([tpk[d] for d in sel]))
# continuum references (closed form, [DERIVED]): t_peak = d^2/(6 alpha); half-peak root of the self-similar profile
from scipy.optimize import brentq
def g(tau):   # dT(d,t)/peak(d) as a function of tau = alpha t / d^2 (3-D Gaussian)
    return (6 * tau) ** -1.5 * math.exp(-1.0 / (4 * tau) + 1.5)
tau_half = brentq(lambda tau: g(tau) - 0.5, 0.02, 1.0 / 6)
cont = {int(d): dict(t_peak=d * d / (6 * alpha), t_half=tau_half * d * d / alpha, peak=0.0736 * float(d) ** -3) for d in ds}
# exact 3-D continuum peak coefficient (3/2pi)^1.5 e^-1.5
peak_coef = (3 / (2 * math.pi)) ** 1.5 * math.exp(-1.5)
# absolute-threshold sweep (diagnostic only): p over the distances that cross, d >= 2
sweep = {}
for thr in [1e-3, 1e-4, 1e-6, 1e-8, 1e-12, 1e-20, 1e-40, 1e-80]:
    arr = {int(d): first_cross(times, series[:, m], thr) for m, d in enumerate(ds) if d >= 2}
    arr = {d: t for d, t in arr.items() if t is not None and t > 0}
    if len(arr) >= 2:
        dd = np.array(sorted(arr), float); tt = np.array([arr[int(d)] for d in dd])
        sweep[f"{thr:g}"] = dict(p=ols(dd, tt)[0], n_dist=len(dd), d_max=int(dd.max()), t=arr)
# numerical light cone of the explicit stencil: first substep with ANY nonzero value at distance d
lightcone = {int(d): float(times[np.nonzero(series[:, m] > 0)[0][0]]) for m, d in enumerate(ds)}
out = dict(grid=n, dx=dx, alpha=alpha, dt_substep=dt_sub, substeps=nsub,
           literal=dict(threshold=1e-3, slices=edges.tolist(), arrivals=lit_arr, p=lit_p,
                        distances_crossing=lit_d.tolist(), energy_conservation=energy_literal - 1.0),
           corrected=dict(half_peak_arrival=half, p_half=p_half, t_peak=tpk, p_peak=p_peak, peak=peak,
                          continuum=cont, tau_half=tau_half, peak_coef=peak_coef,
                          ratio_half_to_continuum={d: half[d] / cont[d]["t_half"] for d in half},
                          ratio_peak_to_continuum={d: tpk[d] / cont[d]["t_peak"] for d in tpk},
                          energy_conservation=energy_corr - 1.0),
           threshold_sweep=sweep, explicit_lightcone_first_nonzero=lightcone,
           series_t=times[::5].tolist(), series=series[::5].tolist(), wall_s=time.time() - t0)
json.dump(out, open(f"{OUT}/heat_results.json", "w"), indent=1)
print(json.dumps(dict(literal=out["literal"]["arrivals"], p_lit=lit_p, p_half=p_half, p_peak=p_peak,
                      half=half, tpk=tpk, peak_coef=peak_coef, tau_half=tau_half,
                      sweep={k: (v["p"], v["n_dist"]) for k, v in sweep.items()}, lightcone=lightcone,
                      E=(energy_literal - 1, energy_corr - 1), wall=out["wall_s"]), indent=1))
