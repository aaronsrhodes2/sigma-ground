"""U3 analysis — reads the worker outputs, applies the pre-registered criterion (prereg.json), writes u3_results.json.

Run from the sigma-ground repo root:  python -X utf8 u3_analyze.py
"""
import glob, json, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
L = json.load(open(os.path.join(HERE, "mentat_leaves.json"), encoding="utf-8"))
PRE = json.load(open(os.path.join(HERE, "prereg.json"), encoding="utf-8"))
G = float(L["G"]["value"]); KB = float(L["k_B"]["value"])
TR_OVER_TC = float(L["trelax_over_tcross_N128"]["value"])
TCR_HM = float(L["t_cr_HM_over_unit"]["value"])
N, M, b = 128, 1.0e12, 5000.0
tc = math.sqrt(b ** 3 / (G * M)); dt = tc / 300; tr = TR_OVER_TC * tc
N_PRE = int(3 * tr / dt); N_EXT = int(6 * tr / dt)
KMIN = next(k for k in range(N_PRE) if k * dt > 2 * tr)
IDX = dict(pre=np.arange(KMIN + 1, N_PRE + 1), ext=np.arange(N_PRE + 1, N_EXT + 1))
F = dict(K=0, U_s=1, W_s=2, I=3, Idot=4, n_bound=5, K_bound=6, r_half=7)
DEL = (-0.10, -0.05, 0.05, 0.10)
C_GAS = float(L["C_gas_equipartition_J_per_K"]["value"])


def trap(y):
    return float(np.sum(0.5 * (y[1:] + y[:-1])) * dt)


def window_stats(rec, idx):
    r = rec[idx]; T = (len(idx) - 1) * dt
    K, W, U, Id = r[:, F["K"]], r[:, F["W_s"]], r[:, F["U_s"]], r[:, F["Idot"]]
    Kt, Wt = trap(K) / T, trap(W) / T
    lj_bound = 0.5 * (Id[-1] - Id[0]) / T                # (1/2)[Idot]/T
    return dict(K_mean=float(K.mean()), W_mean=float(W.mean()), U_mean=float(U.mean()),
                K_bound_mean=float(r[:, F["K_bound"]].mean()),
                K_LJ=float(K.mean()) - 0.5 * lj_bound,      # <K> - (1/4)[Idot]/T
                minus_half_W=-0.5 * float(W.mean()),
                virial_2K_over_W=2 * Kt / abs(Wt), LJ_term_over_W=lj_bound / abs(Wt),
                LJ_residual_over_W=abs(2 * Kt + Wt - lj_bound) / abs(Wt),
                w_W_over_U=float(W.mean() / U.mean()),
                escapers_min=int(N - r[:, F["n_bound"]].max()), escapers_max=int(N - r[:, F["n_bound"]].min()),
                r_half_mean_m=float(r[:, F["r_half"]].mean()))


def drift(rec, upto=None):
    E = rec[:, F["K"]] + rec[:, F["U_s"]]
    if upto is not None:
        E = E[: upto + 1]
    return dict(end=float(abs(E[-1] / E[0] - 1)), max=float(np.max(np.abs(E / E[0] - 1))))


seeds = sorted({int(os.path.basename(p)[4]) for p in glob.glob(os.path.join(HERE, "seed?_relax.npz"))})
per_seed = {}; series = {}; EXT_AVAILABLE = {}
# common extended length across seeds (the extension may be stopped early with a STOP file; use the same window for all)
N_COMMON = min(min(len(np.load(p_)["rec"]) for p_ in glob.glob(os.path.join(HERE, f"seed{s_}_branch_*.npz"))) - 1 for s_ in seeds)
for s in seeds:
    rel = np.load(os.path.join(HERE, f"seed{s}_relax.npz"))["rec"]
    g_idx = np.arange(1500, 3001)
    gate = window_stats(rel, g_idx)
    gate_pass = (0.97 <= gate["virial_2K_over_W"] <= 1.03) and gate["LJ_residual_over_W"] <= 1e-3
    rows = [json.loads(l) for l in open(os.path.join(HERE, f"seed{s}_branches.jsonl"), encoding="utf-8")]
    rows = {round(r["delta"], 2): r for r in rows}
    br = {}
    for d in (0.0,) + DEL:
        p = os.path.join(HERE, f"seed{s}_branch_{d:+.2f}.npz")
        if os.path.exists(p) and d in rows:
            br[d] = np.load(p)["rec"]
    n_av = min(min(len(r_) for r_ in br.values()) - 1, N_COMMON) if len(br) == 5 else 0
    br = {d_: r_[: n_av + 1] for d_, r_ in br.items()}
    have_ext = n_av >= N_PRE + int(round(tr / dt))          # at least one design t_relax beyond the pre-registered run
    idx_ext = np.arange(N_PRE + 1, min(n_av, N_EXT) + 1)
    EXT_AVAILABLE[s] = int(min(n_av, N_EXT))
    seg = dict(relax=drift(rel))
    for d, rec in br.items():
        seg[f"branch_{d:+.2f}_pre"] = drift(rec, N_PRE)
        if have_ext:
            seg[f"branch_{d:+.2f}_ext"] = drift(rec, int(min(n_av, N_EXT)))
    out = dict(gate=gate, gate_pass=bool(gate_pass), drift=seg, have_ext=bool(have_ext),
               drift_end_max_pre=max(v["end"] for k, v in seg.items() if not k.endswith("_ext")),
               drift_end_max_all=max(v["end"] for v in seg.values()),
               drift_excursion_max_all=max(v["max"] for v in seg.values()),
               K0=rows[0.0]["K0"] if 0.0 in rows else None, E_before=rows[0.0]["E_before"] if 0.0 in rows else None,
               branches_done=sorted(br.keys()))
    if 0.0 in br and all(d in br for d in DEL):
        for wname, idx in (("pre", IDX["pre"]), ("ext", idx_ext)):
            if wname == "ext" and not have_ext:
                continue
            st = {d: window_stats(br[d], idx) for d in (0.0,) + DEL}
            x = np.array([rows[d]["dE_formula"] for d in DEL])
            fits = {}
            for key in ("K_mean", "K_LJ", "K_bound_mean", "minus_half_W"):
                y = np.array([st[d][key] - st[0.0][key] for d in DEL])
                fits[key] = float(np.polyfit(x, y, 1)[0])
            w = st[0.0]["w_W_over_U"]
            out[wname] = dict(slope_raw=fits["K_mean"], slope_LJ=fits["K_LJ"], slope_bound=fits["K_bound_mean"],
                              slope_minus_half_W=fits["minus_half_W"], softened_virial_prediction=w / (w - 2),
                              dE=[float(v) for v in x],
                              dK_raw=[st[d]["K_mean"] - st[0.0]["K_mean"] for d in DEL],
                              dK_LJ=[st[d]["K_LJ"] - st[0.0]["K_LJ"] for d in DEL],
                              branch_stats={f"{d:+.2f}": st[d] for d in (0.0,) + DEL})
        # time-resolved slope: trailing running mean of K over one design t_relax, and the instantaneous slope
        nlast = min(len(r_) for r_ in br.values()) - 1
        nwin = int(round(tr / dt)); t = np.arange(nlast + 1) * dt
        Kd = {d: br[d][: nlast + 1, F["K"]] for d in (0.0,) + DEL}
        cs = {d: np.concatenate([[0.0], np.cumsum(Kd[d])]) for d in Kd}
        x = np.array([rows[d]["dE_formula"] for d in DEL])
        A = np.vstack([x, np.ones_like(x)]).T; P = np.linalg.pinv(A)[0]
        inst = P @ np.array([Kd[d] - Kd[0.0] for d in DEL])
        ends = np.arange(nwin, nlast + 1)
        run = {d: (cs[d][ends + 1] - cs[d][ends + 1 - nwin]) / nwin for d in Kd}
        runslope = P @ np.array([run[d] - run[0.0] for d in DEL])
        vir = {f"{d:+.2f}": (2 * br[d][: nlast + 1, F["K"]] / np.abs(br[d][: nlast + 1, F["W_s"]]))[::10].tolist() for d in (0.0, 0.10, -0.10)}
        series[s] = dict(t_over_tc_inst=(t[::10] / tc).tolist(), slope_inst=inst[::10].tolist(),
                         t_over_tc_run=(t[ends][::10] / tc).tolist(), slope_run=runslope[::10].tolist(), virial=vir)
    per_seed[s] = out


def agg(win, key):
    v = np.array([per_seed[s][win][key] for s in seeds if win in per_seed[s]])
    if len(v) == 0:
        return None
    return dict(per_seed=v.tolist(), mean=float(v.mean()), std=float(v.std(ddof=1)) if len(v) > 1 else None,
                sem=float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else None)


def verdict(mean, std, drift_):
    if mean is None:
        return "BLOCKED"
    if mean >= 0:
        return "SAME"
    if -1.10 <= mean <= -0.90 and std is not None and std <= 0.05 and drift_ <= 1e-4:
        return "SEPARATE"
    return "UNRESOLVED"


res = dict(id="U3", prereg=PRE, choices=dict(N=N, M_kg=M, b_m=b, eps_m=0.02 * b, eps_over_b=0.02, dt_s=dt, t_cross_design_s=tc,
                                              t_relax_design_s=tr, trelax_over_tcross=TR_OVER_TC,
                                              t_cross_HM_over_design_unit=TCR_HM, N_relax_steps=3000, N_pre=N_PRE, N_ext=N_EXT,
                                              pre_window_steps=[int(IDX['pre'][0]), int(IDX['pre'][-1])],
                                              ext_window_steps_planned=[int(IDX['ext'][0]), int(IDX['ext'][-1])],
                                              ext_last_step_available=EXT_AVAILABLE,
                                              seeds=seeds, deltas=[-0.10, -0.05, 0.0, 0.05, 0.10]),
           mentat_leaves=L, per_seed=per_seed, series=series)
summ = {}
for win in ("pre", "ext"):
    a = {k: agg(win, k) for k in ("slope_raw", "slope_LJ", "slope_bound", "slope_minus_half_W", "softened_virial_prediction")}
    if a["slope_raw"] is None:
        continue
    dr = max(per_seed[s]["drift_end_max_pre" if win == "pre" else "drift_end_max_all"] for s in seeds)
    a["drift_gate_value"] = dr
    a["verdict_raw"] = verdict(a["slope_raw"]["mean"], a["slope_raw"]["std"], dr)
    a["verdict_LJ"] = verdict(a["slope_LJ"]["mean"], a["slope_LJ"]["std"], dr)
    a["verdict_raw_ignoring_drift_gate"] = verdict(a["slope_raw"]["mean"], a["slope_raw"]["std"], 0.0)
    # pooled fit over all seeds' points (each seed's points relative to its own control) and its standard error
    for key, yk in (("raw", "dK_raw"), ("LJ", "dK_LJ")):
        X = np.concatenate([per_seed[s][win]["dE"] for s in seeds]); Y = np.concatenate([per_seed[s][win][yk] for s in seeds])
        A = np.vstack([X, np.ones_like(X)]).T; coef, *_ = np.linalg.lstsq(A, Y, rcond=None); res_ = Y - A @ coef
        cov = (res_ ** 2).sum() / (len(X) - 2) * np.linalg.inv(A.T @ A)
        a[f"pooled_slope_{key}"] = dict(slope=float(coef[0]), se=float(math.sqrt(cov[0, 0])), n_points=int(len(X)))
        a[f"t_mean_below_zero_{key}"] = float(-a[f"slope_{key}"]["mean"] / a[f"slope_{key}"]["sem"])
    a["C_over_C_gas_raw"] = 1.0 / a["slope_raw"]["mean"]
    a["C_grav_J_per_K_raw"] = C_GAS / a["slope_raw"]["mean"]
    summ[win] = a
res["summary"] = summ
res["gates"] = {s: dict(pass_=per_seed[s]["gate_pass"], virial=per_seed[s]["gate"]["virial_2K_over_W"],
                        LJ_residual=per_seed[s]["gate"]["LJ_residual_over_W"]) for s in seeds}
res["C_gas_J_per_K"] = C_GAS
res["process_notes"] = [
    "Relax phase (3000 steps per seed) ran in u3_worker.py; it was then stopped and the five branches per seed were run by "
    "u3_branches.py in lockstep from the saved relaxed positions/velocities (rebuild checked bit-for-bit against the saved "
    "diagnostics). Same engine, dt, eps, deltas and diagnostics; the switch only changes the order of work so the "
    "pre-registered window finished for all branches at once and the extension could be stopped at equal length.",
    "Extension planned to 6 design t_relax (4749 steps); stopped with a STOP file to keep the wall clock near the 90 min "
    "budget under a machine shared with other agents (0.20-0.25 s per step); the extended window actually used is "
    "choices.ext_last_step_available.",
    "An earlier sequential start (killed after ~500 relax steps) was discarded; the runs reported here started fresh.",
]
res["stage6a_reference"] = dict(slope=-0.905750710563272, err=0.028610277588712024,
                                source="D:/Aaron/development/sigma-ground/misc/unification/stage6_heat_capacity.json")
json.dump(res, open(os.path.join(HERE, "u3_results.json"), "w", encoding="utf-8"), indent=1)
print(json.dumps(dict(summary=summ, gates=res["gates"],
                      drift={s: dict(pre=per_seed[s]["drift_end_max_pre"], all=per_seed[s]["drift_end_max_all"],
                                     exc=per_seed[s]["drift_excursion_max_all"]) for s in seeds}), indent=1))
