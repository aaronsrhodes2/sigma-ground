"""U3 post-hoc decomposition (not a pass/fail input). Exact identity for each window, from 2K + W_s = (1/2) d2I/dt2 and E_s = K + U_s:
    <K> = -<E_s> + L + <U_s - W_s>,   L = (1/2)[Idot]/T
so  d<K>/dE = -1 (energy) + dL/dE (non-stationarity: breathing) + d<U_s - W_s>/dE (softening).
The design's 'LJ-corrected' K_LJ = <K> - L/2 = -<W_s>/2 removes HALF the breathing term; removing all of it returns
-1 + softening term by construction. Also: slope over the longest available window (post hoc)."""
import json, math, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
L_ = json.load(open(os.path.join(HERE, "mentat_leaves.json"), encoding="utf-8"))
G = float(L_["G"]["value"]); TR = float(L_["trelax_over_tcross_N128"]["value"])
tc = math.sqrt(5000.0 ** 3 / (G * 1e12)); dt = tc / 300; tr = TR * tc
N_PRE = int(3 * tr / dt); KMIN = next(k for k in range(N_PRE) if k * dt > 2 * tr)
DEL = (-0.10, -0.05, 0.05, 0.10)
out = {}
def trap(y): return float(np.sum(0.5 * (y[1:] + y[:-1])) * dt)
for s in (1, 2, 3):
    rows = {round(json.loads(l)["delta"], 2): json.loads(l) for l in open(os.path.join(HERE, f"seed{s}_branches.jsonl"))}
    br = {d: np.load(os.path.join(HERE, f"seed{s}_branch_{d:+.2f}.npz"))["rec"] for d in (0.0,) + DEL}
    nlast = min(len(r) for r in br.values()) - 1
    wins = dict(pre=(KMIN + 1, N_PRE), ext=(N_PRE + 1, nlast), long_posthoc=(KMIN + 1, nlast), first_tr_posthoc=(1, int(round(tr / dt))))
    res = {}
    for w, (a, b) in wins.items():
        q = {}
        for d in (0.0,) + DEL:
            r = br[d][a:b + 1]; T = (b - a) * dt
            K = trap(r[:, 0]) / T; U = trap(r[:, 1]) / T; W = trap(r[:, 2]) / T; E = K + U
            L = 0.5 * (r[-1, 4] - r[0, 4]) / T
            q[d] = dict(K=K, E=E, L=L, UmW=U - W, Kfull=K - L, Khalf=K - 0.5 * L)
        x = np.array([rows[d]["dE_formula"] for d in DEL])
        f = lambda key: float(np.polyfit(x, np.array([q[d][key] - q[0.0][key] for d in DEL]), 1)[0])
        sK, sE, sL, sS = f("K"), f("E"), f("L"), f("UmW")
        res[w] = dict(window_steps=[a, b], slope_K=sK, slope_E=sE, slope_L=sL, slope_UmW=sS,
                      identity_residual=sK - (-sE + sL + sS), slope_K_minus_L=f("Kfull"), slope_K_minus_halfL=f("Khalf"))
    out[s] = res
summ = {}
for w in ("pre", "ext", "long_posthoc", "first_tr_posthoc"):
    summ[w] = {k: dict(mean=float(np.mean([out[s][w][k] for s in out])), std=float(np.std([out[s][w][k] for s in out], ddof=1)),
                       per_seed=[out[s][w][k] for s in out])
               for k in ("slope_K", "slope_L", "slope_UmW", "slope_K_minus_L", "slope_K_minus_halfL", "identity_residual", "slope_E")}
json.dump(dict(per_seed=out, summary=summ), open(os.path.join(HERE, "u3_decomposition.json"), "w"), indent=1)
for w, v in summ.items():
    print(w, {k: (round(x["mean"], 4), round(x["std"], 4)) for k, x in v.items()})
