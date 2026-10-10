"""U3 repair (v2) analysis. Reads the SAME saved simulation data as v1 (seed*_relax.npz, seed*_branch_*.npz,
seed*_branches.jsonl; no re-integration), re-applies the pre-registered rule from prereg.json unchanged, and adds the
post-hoc diagnostics the three skeptics asked for. Nothing here is a pass/fail input except the 'pre' and 'ext'
verdicts, which use the v1 code path verbatim.

New in v2 (all post hoc, labelled):
  * within-seed 4-point fit SE, between/within variance ratio (F(2,6) closed form from Mentat), chi-square CI on the
    3-seed std (closed form from Mentat), for every window;
  * post-hoc long window reported with the within-seed SE, not the 3-sample std;
  * pooled 12-point fit with naive AND seed-clustered SE, and the seed-mean t statistic kept separate;
  * time-resolved slope: zero crossings, running-mean minima / end values, window-end sensitivity;
  * moment-of-inertia, Idot and half-mass-radius responses across the four branches vs time (turnaround test);
  * decorrelation of the kicked branches from the control (4-point fit residual vs the control's own K fluctuation);
  * a finite-N noise model of the per-seed slope scatter (integrated autocorrelation time of K);
  * Lagrange-Jacobi decomposition (v1 u3_decompose.py, same code) with the consistent softened expectation.

Run from the sigma-ground repo root:  python -X utf8 u3_analyze_v2.py   -> u3_results_v2.json
"""
import glob, json, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
L = json.load(open(os.path.join(HERE, "mentat_leaves_v2.json"), encoding="utf-8"))
PRE = json.load(open(os.path.join(HERE, "prereg.json"), encoding="utf-8"))
lv = lambda k: float(L[k]["value"])
G = lv("G"); KB = lv("k_B"); TR_OVER_TC = lv("trelax_over_tcross_N128"); TCR_HM = lv("t_cr_HM_over_unit")
T_ORB_U = lv("T_orb_r_half_units"); C_GAS = float(L["C_gas_equipartition_J_per_K"]["value"])
Q025, Q975 = lv("chi2_2dof_q025"), lv("chi2_2dof_q975")
F26_SURV = lambda f: 81.0 / (3 * f ** 3 + 27 * f ** 2 + 81 * f + 81)          # Mentat integrate_expr (F_2_6_survival)
T2_SURV = lambda t: -t / (2 * math.sqrt(t ** 2 + 2)) + 0.5                    # Mentat integrate_expr (t2_survival)
assert abs(F26_SURV(0.0) - 1) < 1e-15 and abs(T2_SURV(0.0) - 0.5) < 1e-15
N, M, b = 128, 1.0e12, 5000.0
tc = math.sqrt(b ** 3 / (G * M)); dt = tc / 300; tr = TR_OVER_TC * tc
N_PRE = int(3 * tr / dt); N_EXT = int(6 * tr / dt)
KMIN = next(k for k in range(N_PRE) if k * dt > 2 * tr)
F = dict(K=0, U_s=1, W_s=2, I=3, Idot=4, n_bound=5, K_bound=6, r_half=7)
DEL = (-0.10, -0.05, 0.05, 0.10); ALL = (0.0,) + DEL
U = 300                                                     # steps per design unit (dt = t_cross_design/300)


def trap(y):
    return float(np.sum(0.5 * (y[1:] + y[:-1])) * dt)


def window_stats(rec, idx):                                 # v1, verbatim
    r = rec[idx]; T = (len(idx) - 1) * dt
    K, W, Us, Id = r[:, F["K"]], r[:, F["W_s"]], r[:, F["U_s"]], r[:, F["Idot"]]
    Kt, Wt = trap(K) / T, trap(W) / T
    lj_bound = 0.5 * (Id[-1] - Id[0]) / T
    return dict(K_mean=float(K.mean()), W_mean=float(W.mean()), U_mean=float(Us.mean()),
                K_bound_mean=float(r[:, F["K_bound"]].mean()),
                K_LJ=float(K.mean()) - 0.5 * lj_bound, minus_half_W=-0.5 * float(W.mean()),
                virial_2K_over_W=2 * Kt / abs(Wt), LJ_term_over_W=lj_bound / abs(Wt),
                LJ_residual_over_W=abs(2 * Kt + Wt - lj_bound) / abs(Wt),
                w_W_over_U=float(W.mean() / Us.mean()),
                escapers_min=int(N - r[:, F["n_bound"]].max()), escapers_max=int(N - r[:, F["n_bound"]].min()),
                r_half_mean_m=float(r[:, F["r_half"]].mean()))


def drift(rec, upto=None):                                  # v1, verbatim
    E = rec[:, F["K"]] + rec[:, F["U_s"]]
    if upto is not None:
        E = E[: upto + 1]
    return dict(end=float(abs(E[-1] / E[0] - 1)), max=float(np.max(np.abs(E / E[0] - 1))))


def verdict(mean, std, drift_):                             # v1, verbatim (prereg.json rule)
    if mean is None:
        return "BLOCKED"
    if mean >= 0:
        return "SAME"
    if -1.10 <= mean <= -0.90 and std is not None and std <= 0.05 and drift_ <= 1e-4:
        return "SEPARATE"
    return "UNRESOLVED"


def fit_se(x, y):
    A = np.vstack([x, np.ones_like(x)]).T; c, *_ = np.linalg.lstsq(A, y, rcond=None); r = y - A @ c
    cov = (r ** 2).sum() / (len(x) - 2) * np.linalg.inv(A.T @ A)
    return float(c[0]), float(math.sqrt(cov[0, 0])), float(c[1])


def tau_int(y):
    """Integrated autocorrelation time (in samples) of a linearly detrended series, ACF summed to its first zero."""
    t = np.arange(len(y)); y = y - np.polyval(np.polyfit(t, y, 1), t)
    n = len(y); f = np.fft.rfft(y, 2 * n); ac = np.fft.irfft(f * np.conj(f))[:n]; ac /= ac[0]
    k0 = int(np.argmax(ac <= 0)) if np.any(ac <= 0) else n
    return float(1 + 2 * ac[1:k0].sum()), float(np.std(y)), k0


seeds = sorted({int(os.path.basename(p)[4]) for p in glob.glob(os.path.join(HERE, "seed?_relax.npz"))})
N_COMMON = min(min(len(np.load(p_)["rec"]) for p_ in glob.glob(os.path.join(HERE, f"seed{s_}_branch_*.npz"))) - 1 for s_ in seeds)
WIN = dict(pre=(KMIN + 1, N_PRE), ext=(N_PRE + 1, min(N_COMMON, N_EXT)), long_posthoc=(KMIN + 1, min(N_COMMON, N_EXT)),
           early_3_5_posthoc=(900, 1500))
per_seed, series = {}, {}
for s in seeds:
    rel = np.load(os.path.join(HERE, f"seed{s}_relax.npz"))["rec"]
    gate = window_stats(rel, np.arange(1500, 3001))
    gate_pass = (0.97 <= gate["virial_2K_over_W"] <= 1.03) and gate["LJ_residual_over_W"] <= 1e-3
    rows = {round(json.loads(l)["delta"], 2): json.loads(l) for l in open(os.path.join(HERE, f"seed{s}_branches.jsonl"), encoding="utf-8")}
    br = {d: np.load(os.path.join(HERE, f"seed{s}_branch_{d:+.2f}.npz"))["rec"][: N_COMMON + 1] for d in ALL}
    K0 = rows[0.0]["K0"]; E0 = rows[0.0]["E_before"]
    x = np.array([rows[d]["dE_formula"] for d in DEL]); Sxx = float(((x - x.mean()) ** 2).sum())
    seg = dict(relax=drift(rel))
    for d in ALL:
        seg[f"branch_{d:+.2f}_pre"] = drift(br[d], N_PRE)
        seg[f"branch_{d:+.2f}_ext"] = drift(br[d], N_COMMON)
    out = dict(gate=gate, gate_pass=bool(gate_pass), K0=K0, E_before=E0, Sxx_J2=Sxx,
               dE_measured_over_formula=[rows[d]["dE_measured"] / rows[d]["dE_formula"] for d in DEL],
               drift=seg, drift_end_max_pre=max(v["end"] for k, v in seg.items() if not k.endswith("_ext")),
               drift_end_max_all=max(v["end"] for v in seg.values()),
               drift_excursion_max_all=max(v["max"] for v in seg.values()))
    vir_rel = 2 * rel[:, F["K"]] / np.abs(rel[:, F["W_s"]])
    vir0 = 2 * br[0.0][:, F["K"]] / np.abs(br[0.0][:, F["W_s"]])
    out["virial_inst_at_kick"] = float(vir_rel[-1])
    out["virial_inst_range_gate_window"] = [float(vir_rel[1500:].min()), float(vir_rel[1500:].max())]
    out["virial_inst_range_control_branch"] = [float(vir0.min()), float(vir0.max())]
    # ---- window slopes (v1 statistic = sample mean of K) + within-seed SE + other statistics ----
    for w, (a, bb) in WIN.items():
        idx = np.arange(a, bb + 1)
        st = {d: window_stats(br[d], idx) for d in ALL}
        res = {}
        for key in ("K_mean", "K_LJ", "K_bound_mean", "minus_half_W"):
            y = np.array([st[d][key] - st[0.0][key] for d in DEL])
            sl, se, ic = fit_se(x, y)
            res[key] = dict(slope=sl, within_seed_fit_se=se)
        wsoft = st[0.0]["w_W_over_U"]
        out[w] = dict(window_rows=[a, bb], window_units=(bb - a) / U,
                      slope_raw=res["K_mean"]["slope"], se_raw_within=res["K_mean"]["within_seed_fit_se"],
                      slope_LJ=res["K_LJ"]["slope"], slope_bound=res["K_bound_mean"]["slope"],
                      slope_minus_half_W=res["minus_half_W"]["slope"],
                      softened_virial_fixed_w=wsoft / (wsoft - 2),
                      dE=x.tolist(), dK_raw=[st[d]["K_mean"] - st[0.0]["K_mean"] for d in DEL],
                      normalised_response=[(st[d]["K_mean"] - st[0.0]["K_mean"]) / xd for d, xd in zip(DEL, x)],
                      n_branches_cooled_when_heated=int(sum((st[d]["K_mean"] - st[0.0]["K_mean"]) * xd < 0 for d, xd in zip(DEL, x))),
                      escapers_max=max(st[d]["escapers_max"] for d in ALL),
                      branch_stats={f"{d:+.2f}": st[d] for d in ALL})
    # ---- Lagrange-Jacobi decomposition (v1 u3_decompose.py, same code; trapezoid means) ----
    dec = {}
    for w, (a, bb) in WIN.items():
        q = {}
        for d in ALL:
            r = br[d][a:bb + 1]; T = (bb - a) * dt
            Kt = trap(r[:, 0]) / T; Ut = trap(r[:, 1]) / T; Wt = trap(r[:, 2]) / T
            Lt = 0.5 * (r[-1, 4] - r[0, 4]) / T
            q[d] = dict(K=Kt, E=Kt + Ut, L=Lt, UmW=Ut - Wt, Kfull=Kt - Lt, Khalf=Kt - 0.5 * Lt)
        f_ = lambda key: float(np.polyfit(x, np.array([q[d][key] - q[0.0][key] for d in DEL]), 1)[0])
        sK, sE, sL, sS = f_("K"), f_("E"), f_("L"), f_("UmW")
        dec[w] = dict(slope_K=sK, slope_E=sE, slope_L=sL, slope_UmW=sS, identity_residual=sK - (-sE + sL + sS),
                      slope_K_minus_L=f_("Kfull"), slope_K_minus_halfL=f_("Khalf"),
                      consistent_softened_expectation=-1 + sS)
    out["decomposition"] = dec
    # ---- time-resolved quantities across the four branches ----
    n = N_COMMON; t_u = np.arange(n + 1) / U
    A4 = np.vstack([x, np.ones_like(x)]).T
    P = np.linalg.pinv(A4)
    Pslope = P[0]
    D = lambda col: np.array([br[d][:, col] - br[0.0][:, col] for d in DEL])
    inst = Pslope @ D(F["K"])
    neg = np.where(inst < 0)[0]
    first_neg = float(t_u[neg[0]]) if len(neg) else None
    nwin = int(round(tr / dt))
    cs = {d: np.concatenate([[0.0], np.cumsum(br[d][:, F["K"]])]) for d in ALL}
    ends = np.arange(nwin, n + 1)
    run = {d: (cs[d][ends + 1] - cs[d][ends + 1 - nwin]) / nwin for d in ALL}
    runslope = Pslope @ np.array([run[d] - run[0.0] for d in DEL])
    # residual of the 4-point linear fit at every instant: decorrelation of the kicked branches
    Y = D(F["K"]); coef = P @ Y; resid = Y - A4 @ coef
    res_rms = np.sqrt((resid ** 2).sum(0) / 2) / K0
    k1 = np.ones(U) / U
    res_rms_1u = np.sqrt(np.convolve(res_rms ** 2, k1, mode="same"))
    tau_c, sig_c, _ = tau_int(br[0.0][U:, F["K"]])          # control branch after the first design unit
    tau_r, sig_r, _ = tau_int(rel[1000:, F["K"]])           # relax, last 2000 steps
    dec_idx = np.where(res_rms_1u[U:n - U // 2] > 0.5 * sig_c / K0)[0]
    t_decor = float(t_u[U + dec_idx[0]]) if len(dec_idx) else None
    # moment of inertia / Idot / half-mass radius responses (per unit dE/K0)
    I0 = br[0.0][0, F["I"]]; rh0 = br[0.0][0, F["r_half"]]
    sI = (Pslope @ D(F["I"])) * K0 / I0                      # d ln I / d(dE/K0)
    sI_E = (Pslope @ D(F["I"])) * abs(E0) / I0               # (dI/dE)|E|/I0, homologous re-virialised value 2
    sId = Pslope @ D(F["Idot"])
    srh = (Pslope @ D(F["r_half"])) * abs(E0) / rh0          # d ln r_half / d(dE/|E|); homologous re-virialised value 1
    srh_1u = np.convolve(srh, k1, mode="same")
    sI = sI_E                                                # report I on the same footing: d ln I / d(dE/|E|), homologous 2
    inner = slice(U // 2, n - U // 2)
    out["time_resolved"] = dict(
        inst_slope_first_negative_units=first_neg,
        inst_slope_max_after_2p3_units=float(inst[int(2.3 * U):].max()),
        inst_slope_at_end=float(inst[-1]),
        runmean_end=float(runslope[-1]), runmean_min=float(runslope.min()),
        runmean_min_at_units=float((ends[np.argmax(-runslope)]) / U),
        dlnI_ddE_over_K0_at_end=float(sI_E[-1] * K0 / abs(E0)), dIdE_absE_over_I0_at_end=float(sI_E[-1]),
        dIdE_absE_over_I0_at={str(k): float(sI_E[min(int(k * U), n)]) for k in (1, 2, 4, 6, 8, 10, 12, 12.49)},
        absE_over_K0=float(abs(E0) / K0),
        dlnI_ddE_increasing_fraction_after_1u=float(np.mean(np.diff(sI[U:]) > 0)),
        dlnI_ddE_global_max_at_units=float(t_u[np.argmax(sI)]),
        dIdot_dE_positive_fraction_after_0p1u=float(np.mean(sId[30:] > 0)),
        dIdot_dE_max_at_units=float(t_u[np.argmax(sId)]),
        dlnrh_ddE_1u_at={str(k): float(srh_1u[min(int(k * U), n - U // 2)]) for k in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)},
        dlnrh_ddE_1u_max_at_units=float(t_u[inner][np.argmax(srh_1u[inner])]),
        decorrelation_time_units=t_decor,
        fit_residual_rms_over_K0_at={str(k): float(res_rms_1u[min(int(k * U), n - U // 2)]) for k in (0.5, 1, 2, 3, 4, 5, 6, 8, 10, 12)},
        control_K_sigma_over_K0=float(sig_c / K0), control_K_tau_int_units=tau_c / U,
        relax_K_sigma_over_K0=float(sig_r / K0), relax_K_tau_int_units=tau_r / U)
    # finite-N noise model: per-seed slope scatter if the four kicked branches are decorrelated (independent window means;
    # the control's own noise is common to all four points and goes into the intercept, not the slope)
    nm = {}
    for w in ("pre", "ext", "long_posthoc"):
        T_u = out[w]["window_units"]
        for lab, tau_u, sig in (("control", tau_c / U, sig_c), ("relax", tau_r / U, sig_r)):
            sw = sig * math.sqrt(min(1.0, 2 * tau_u / T_u))
            nm[f"{w}_{lab}"] = float(sw / math.sqrt(Sxx))
    out["noise_model_slope_sigma"] = nm
    series[s] = dict(t_inst=t_u[::10].tolist(), slope_inst=inst[::10].tolist(), t_run=(ends[::10] / U).tolist(),
                     slope_run=runslope[::10].tolist(), dlnI=sI[::10].tolist(),
                     t_1u=t_u[inner][::10].tolist(), dlnrh_1u=srh_1u[inner][::10].tolist(),
                     fit_resid_1u=res_rms_1u[inner][::10].tolist(), control_sigma_over_K0=float(sig_c / K0),
                     vir_control=vir0[::10].tolist())
    per_seed[s] = out


def agg(vals):
    v = np.array(vals, float)
    return dict(per_seed=v.tolist(), mean=float(v.mean()), std=float(v.std(ddof=1)), sem=float(v.std(ddof=1) / math.sqrt(len(v))))


summ = {}
for w in WIN:
    a = {k: agg([per_seed[s][w][k] for s in seeds]) for k in ("slope_raw", "slope_LJ", "slope_bound", "slope_minus_half_W",
                                                               "softened_virial_fixed_w", "se_raw_within")}
    dr = max(per_seed[s]["drift_end_max_pre" if w in ("pre", "early_3_5_posthoc") else "drift_end_max_all"] for s in seeds)
    m, sd = a["slope_raw"]["mean"], a["slope_raw"]["std"]
    a["drift_gate_value"] = dr
    a["verdict_raw"] = verdict(m, sd, dr)
    a["verdict_is_pass_fail_input"] = w in ("pre", "ext")
    a["verdict_LJ"] = verdict(a["slope_LJ"]["mean"], a["slope_LJ"]["std"], dr)
    within = np.array([per_seed[s][w]["se_raw_within"] for s in seeds])
    var_w = float(np.mean(within ** 2)); Fr = sd ** 2 / var_w
    a["between_within"] = dict(rms_within_seed_se=math.sqrt(var_w), F_ratio=Fr, p_F_2_6=F26_SURV(Fr),
                               note="F(2,6) test of 'seed-to-seed scatter exceeds the within-seed 4-point scatter'; "
                                    "assumes the residual scatter measures the same noise that moves the slope")
    a["seed_std_95CI_chi2"] = [sd * math.sqrt(2 / Q975), sd * math.sqrt(2 / Q025)]
    a["sem_from_within_seed_se"] = math.sqrt(var_w / len(seeds))
    a["honest_error"] = max(a["slope_raw"]["sem"], a["sem_from_within_seed_se"])
    a["t_seed_mean_below_zero"] = -m / a["slope_raw"]["sem"]
    a["p_one_sided_t2"] = T2_SURV(a["t_seed_mean_below_zero"])
    # pooled 12-point fit: naive SE (v1) and seed-clustered sandwich SE
    X = np.concatenate([per_seed[s][w]["dE"] for s in seeds]); Y = np.concatenate([per_seed[s][w]["dK_raw"] for s in seeds])
    A = np.vstack([X, np.ones_like(X)]).T; coef, *_ = np.linalg.lstsq(A, Y, rcond=None); r_ = Y - A @ coef
    XtXi = np.linalg.inv(A.T @ A); naive = (r_ ** 2).sum() / (len(X) - 2) * XtXi
    meat = sum(np.outer(A[i:i + 4].T @ r_[i:i + 4], A[i:i + 4].T @ r_[i:i + 4]) for i in range(0, len(X), 4))
    g = len(seeds); clus = g / (g - 1) * (len(X) - 1) / (len(X) - 2) * XtXi @ meat @ XtXi
    a["pooled_12pt"] = dict(slope=float(coef[0]), se_naive=float(math.sqrt(naive[0, 0])), se_seed_clustered=float(math.sqrt(clus[0, 0])),
                            note="the 12 points are not independent: each seed's four share one control branch and one realisation; "
                                 "the seed-level SEM is the defensible uncertainty")
    a["noise_model_slope_sigma"] = ({k: [per_seed[s]["noise_model_slope_sigma"][f"{w}_{k}"] for s in seeds]
                                     for k in ("control", "relax")} if w in ("pre", "ext", "long_posthoc") else None)
    a["normalised_response_per_seed"] = {str(s): per_seed[s][w]["normalised_response"] for s in seeds}
    a["n_branches_cooled_when_heated_of_12"] = int(sum(per_seed[s][w]["n_branches_cooled_when_heated"] for s in seeds))
    a["n_branch_responses_below_minus1_of_12"] = int(sum(v < -1 for s in seeds for v in per_seed[s][w]["normalised_response"]))
    a["decomposition"] = {k: agg([per_seed[s]["decomposition"][w][k] for s in seeds])
                          for k in per_seed[seeds[0]]["decomposition"][w]}
    a["C_over_C_gas"] = dict(value=1 / m, err_from_sem=a["slope_raw"]["sem"] / m ** 2, err_from_std=sd / m ** 2,
                             tag="[BY CONSTRUCTION] C = C_gas/slope via the kinetic temperature T = 2K/(3 N k_B)")
    a["C_grav_J_per_K"] = dict(value=C_GAS / m, err_from_sem=C_GAS * a["slope_raw"]["sem"] / m ** 2)
    summ[w] = a

# window-end sensitivity of the seed-mean running slope (1-t_relax windows)
wend = {}
for e_u in (5.9, 6.9, 7.9, 8.9, 9.9, 10.9, 12.49):
    vals = []
    for s in seeds:
        sr = series[s]; tt = np.array(sr["t_run"]); vals.append(float(np.interp(e_u, tt, sr["slope_run"])))
    wend[str(e_u)] = dict(per_seed=vals, mean=float(np.mean(vals)))
inst_mean = np.mean([series[s]["slope_inst"] for s in seeds], axis=0); t_i = np.array(series[seeds[0]]["t_inst"])
inst_mean_at = {str(k): float(np.interp(k, t_i, inst_mean)) for k in (0.0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12.49)}

res = dict(id="U3", version="v2 (repair)", prereg=PRE,
           choices=dict(N=N, M_kg=M, b_m=b, eps_m=0.02 * b, dt_s=dt, t_cross_design_s=tc, t_relax_design_s=tr,
                        trelax_over_tcross=TR_OVER_TC, N_relax_steps=3000, N_pre=N_PRE, N_ext_planned=N_EXT,
                        N_ext_reached=N_COMMON, windows_rows=WIN, seeds=seeds, deltas=list(ALL)),
           mentat_leaves=L, per_seed={str(s): v for s, v in per_seed.items()}, series={str(s): v for s, v in series.items()},
           summary=summ, window_end_sensitivity_runmean=wend, inst_slope_seed_mean_at_units=inst_mean_at,
           gates={str(s): dict(pass_=per_seed[s]["gate_pass"], virial=per_seed[s]["gate"]["virial_2K_over_W"],
                               LJ_residual=per_seed[s]["gate"]["LJ_residual_over_W"]) for s in seeds},
           C_gas_J_per_K=C_GAS,
           stage6a_reference=dict(slope=-0.905750710563272, err=0.028610277588712024,
                                  source="D:/Aaron/development/sigma-ground/misc/unification/stage6_heat_capacity.json"))
# regression check against v1
v1 = json.load(open(os.path.join(HERE, "u3_results.json"), encoding="utf-8"))
res["v1_regression"] = {w: dict(v1=v1["summary"][w]["slope_raw"]["per_seed"], v2=summ[w]["slope_raw"]["per_seed"],
                                max_abs_diff=float(np.max(np.abs(np.array(v1["summary"][w]["slope_raw"]["per_seed"]) -
                                                                 np.array(summ[w]["slope_raw"]["per_seed"])))))
                        for w in ("pre", "ext")}
json.dump(res, open(os.path.join(HERE, "u3_results_v2.json"), "w", encoding="utf-8"), indent=1)

# ---- printout ----
for w in WIN:
    a = summ[w]
    print(f"\n== {w} rows {WIN[w]} ({per_seed[seeds[0]][w]['window_units']:.2f} units)")
    print(" raw", [round(v, 4) for v in a["slope_raw"]["per_seed"]], "mean %.4f std %.4f sem %.4f" % (a["slope_raw"]["mean"], a["slope_raw"]["std"], a["slope_raw"]["sem"]), a["verdict_raw"])
    print(" within SE", [round(v, 3) for v in a["se_raw_within"]["per_seed"]], "rms %.3f F %.2f p %.3f" % (a["between_within"]["rms_within_seed_se"], a["between_within"]["F_ratio"], a["between_within"]["p_F_2_6"]),
          "sem_within %.3f" % a["sem_from_within_seed_se"], "CI std", [round(v, 3) for v in a["seed_std_95CI_chi2"]])
    print(" pooled", {k: round(v, 4) for k, v in a["pooled_12pt"].items() if k != "note"}, "t %.2f p1 %.4f" % (a["t_seed_mean_below_zero"], a["p_one_sided_t2"]))
    print(" noise model", a["noise_model_slope_sigma"])
    print(" LJ %.4f +- %.4f bound %.4f +- %.4f" % (a["slope_LJ"]["mean"], a["slope_LJ"]["std"], a["slope_bound"]["mean"], a["slope_bound"]["std"]))
    dd = a["decomposition"]
    print(" decomp L %.4f+-%.4f UmW %.4f K-L %.4f+-%.4f resid %.1e E %.6f consistent %.4f" % (dd["slope_L"]["mean"], dd["slope_L"]["std"], dd["slope_UmW"]["mean"], dd["slope_K_minus_L"]["mean"], dd["slope_K_minus_L"]["std"], max(abs(v) for v in dd["identity_residual"]["per_seed"]), dd["slope_E"]["mean"], dd["consistent_softened_expectation"]["mean"]))
    print(" C/Cgas %.3f +- %.3f (sem) +- %.3f (std); softened fixed-w %.4f" % (a["C_over_C_gas"]["value"], a["C_over_C_gas"]["err_from_sem"], a["C_over_C_gas"]["err_from_std"], a["softened_virial_fixed_w"]["mean"]))
    print(" normalised", {k: [round(float(u), 2) for u in v] for k, v in a["normalised_response_per_seed"].items()},
          "cooled", a["n_branches_cooled_when_heated_of_12"], "below -1", a["n_branch_responses_below_minus1_of_12"])
for s in seeds:
    p = per_seed[s]
    print(f"\nseed {s}: kick vir {p['virial_inst_at_kick']:.4f} gate-window range {[round(v,3) for v in p['virial_inst_range_gate_window']]} control range {[round(v,3) for v in p['virial_inst_range_control_branch']]}")
    print(" drift end pre %.2e all %.2e exc %.2e" % (p["drift_end_max_pre"], p["drift_end_max_all"], p["drift_excursion_max_all"]))
    print(" ", json.dumps(p["time_resolved"]))
    print("  dE ratio", [round(v, 12) for v in p["dE_measured_over_formula"]])
print("\nwindow-end", {k: round(v["mean"], 3) for k, v in wend.items()})
print("inst mean", {k: round(v, 3) for k, v in inst_mean_at.items()})
print("v1 regression", {w: v["max_abs_diff"] for w, v in res["v1_regression"].items()})
