"""U7 — heat leg, grid-refinement sentinel (post-hoc diagnostic, not a pass/fail input): the half-peak and
time-to-peak arrival at fixed PHYSICAL distances 2..10 (units of the original dx) on grids dx = 1 and dx = 0.5.
diffuse_fvm unmodified; one hot cell; insulated walls far away; sampled every explicit substep."""
import json, math, sys, time
import numpy as np
sys.path.insert(0, r"D:/Aaron/development/sigma-ground")
from sigma_ground.dynamics.fields.heat import diffuse_fvm, stable_dt
OUT = r"C:/Users/aaron/AppData/Local/Temp/claude/D--Aaron-development-sigma-ground/c6652910-47a9-478d-9c83-a8a8c8aaa457/scratchpad/round2/U7"

def ols(x, y):
    A = np.vstack([np.log(x), np.ones_like(x)]).T
    return float(np.linalg.lstsq(A, np.log(y), rcond=None)[0][0])

def first_cross(t, y, thr):
    j = int(np.nonzero(y >= thr)[0][0])
    return float(t[j - 1] + (thr - y[j - 1]) * (t[j] - t[j - 1]) / (y[j] - y[j - 1])) if j else float(t[0])

def run(dx, t_end=26.0, L=48.0):
    n = int(round(L / dx)); c = n // 2
    ks = np.arange(1, int(round(10 / dx)) + 1)
    k = np.ones((n, n, n)); R = np.ones((n, n, n))
    T = np.zeros((n, n, n)); T[c, c, c] = 1.0
    dts = stable_dt(1.0, dx); N = int(math.ceil(t_end / dts))
    times = np.arange(N + 1) * dts; ser = np.zeros((N + 1, len(ks))); ser[0] = T[c + ks, c, c]
    for i in range(1, N + 1):
        T, _ = diffuse_fvm(T, k, R, dx, total_time=dts, max_substeps=1)
        ser[i] = T[c + ks, c, c]
    res = {}
    for d in range(2, 11):
        m = int(np.nonzero(np.isclose(ks * dx, d))[0][0])
        y = ser[:, m]; j = int(np.argmax(y))
        res[d] = dict(t_half=first_cross(times[:j + 1], y[:j + 1], 0.5 * y[j]), t_peak=float(times[j]))
    dd = np.arange(2, 11, dtype=float)
    return dict(dx=dx, grid=n, dt_sub=dts, substeps=N, rows=res,
                p_half=ols(dd, np.array([res[d]["t_half"] for d in range(2, 11)])),
                p_peak=ols(dd, np.array([res[d]["t_peak"] for d in range(2, 11)])),
                p_half_4_10=ols(dd[2:], np.array([res[d]["t_half"] for d in range(4, 11)])))

t0 = time.time()
out = {f"dx={dx:g}": run(dx) for dx in (1.0, 0.5)}
tau_half = 0.07274356070405065   # from u7_heat.py (continuum 3-D Gaussian half-peak root, [DERIVED])
out["continuum"] = {d: dict(t_half=tau_half * d * d, t_peak=d * d / 6) for d in range(2, 11)}
out["wall_s"] = time.time() - t0
json.dump(out, open(f"{OUT}/heat_refine_results.json", "w"), indent=1)
print(json.dumps({k: (v["p_half"], v["p_peak"], v["p_half_4_10"]) for k, v in out.items() if k.startswith("dx")}, indent=1))
