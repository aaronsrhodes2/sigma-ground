"""U6 v2 analysis (repair round). Evaluates the UNCHANGED pre-registered criteria (prereg.json) on results_v2_{A,B,C}.json, adds the
reviewer-requested post-hoc diagnostics (prereg_v2.json: D1 exact comparator, D2 S3 window sweep, D3 like-for-like slopes, D4 physics
numbers from physics_v2.json), checks the v2 re-run against v1 bit for bit, writes u6_results_v2.json and u6_figure_v2.png.
Run from the sigma-ground repo root after u6_mentat_leaves_v2.py, u6_physics_v2.py and the three u6_sigma_wave_v2.py sets."""
import sys, os, json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import u6_sigma_wave_v2 as U

L = U.L
PH = json.load(open(os.path.join(HERE, "physics_v2.json"), encoding="utf-8"))
def ph(k): return float(PH[k]["value"])
runs = []
for s in "ABC":
    runs += json.load(open(os.path.join(HERE, f"results_v2_{s}.json"), encoding="utf-8"))
runs_v1 = []
for s in "ABC":
    runs_v1 += json.load(open(os.path.join(HERE, f"results_{s}.json"), encoding="utf-8"))
tr, full = {}, {}
for s in "ABC":
    with np.load(os.path.join(HERE, f"traces_v2_{s}.npz")) as z:
        tr.update({k: z[k] for k in z.files})
    with np.load(os.path.join(HERE, f"fulltraces_v2_{s}.npz")) as z:
        full.update({k: z[k] for k in z.files})

WR = [0.01, 0.02, 0.04, 0.08]
def get(N, kind, mode, wR, rr=None):
    for r in (rr if rr is not None else runs):
        if r.get("label") != "ramp" and r["N"] == N and r["kind"] == kind and r["mode"] == mode and abs(r["wR"] - wR) < 1e-12:
            return r
    raise KeyError((N, kind, mode, wR))

def slope(As):
    return float(np.polyfit(np.log(WR), np.log(As), 1)[0])

def wrap(ph_):
    return float((ph_ + math.pi) % (2 * math.pi) - math.pi)

# ---------------- v2 vs v1: bit-for-bit reproducibility (prereg_v2 expectation) ----------------
keys = ("A_lockin", "phase_rad", "A_std_sqrt2", "A_2w", "res_max_over_A_at_cs", "res_max_over_A_control", "res_rms_over_A_all",
        "first_arrival_step")
ndiff, ncomp = 0, 0
for r in runs:
    if r.get("label") == "ramp":
        continue
    r1 = get(r["N"], r["kind"], r["mode"], r["wR"], runs_v1)
    for k in keys:
        if k in r or k in r1:
            ncomp += 1
            if r.get(k) != r1.get(k):
                ndiff += 1
ramp_same = all(np.array_equal(np.load(os.path.join(HERE, "traces_C.npz"))[f"ramp_N{N}"], tr[f"ramp_N{N}"]) for N in (8000, 16000))
repro = dict(fields_compared=ncomp, fields_differing=ndiff, ramp_traces_identical=bool(ramp_same), bit_for_bit=(ndiff == 0 and ramp_same))

# ---------------- pre-registered P1-P5 (unchanged logic) ----------------
def evaluate(kind, est, N=8000, Nf=16000):
    key = "A_lockin" if est == "lockin" else "A_std_sqrt2"
    out = {}
    for mode in ("pressure", "mass"):
        A = [get(N, kind, mode, q)[key] for q in WR]
        Af = [get(Nf, kind, mode, q)[key] for q in WR]
        pred = [get(N, kind, mode, q)["A_pred"] for q in WR]
        out[mode] = dict(A=A, A_fine=Af, A_pred=pred, ratio=[a / p for a, p in zip(A, pred)], ratio_fine=[a / p for a, p in zip(Af, pred)],
                         slope=slope(A), slope_fine=slope(Af), grid_change=[abs(f / a - 1) for a, f in zip(A, Af)],
                         arrival_offset_cells=[get(N, kind, mode, q)["arrival_offset_cells"] for q in WR])
    P1 = abs(out["pressure"]["slope"] - 0.0) <= 0.1
    P2 = abs(out["mass"]["slope"] - 2.0) <= 0.1
    P3 = all(abs(out[m]["ratio"][0] - 1.0) <= 0.10 for m in ("pressure", "mass"))
    P4 = all(abs(x) <= 2 for m in ("pressure", "mass") for x in out[m]["arrival_offset_cells"])
    P5 = all(g < 0.02 for m in ("pressure", "mass") for g in out[m]["grid_change"])
    out["criteria"] = dict(P1_pressure_slope=P1, P2_mass_slope=P2, P3_amplitude_at_0p01=P3, P4_first_arrival=P4, P5_grid_halving=P5,
                           PASS=bool(P1 and P2 and P3 and P4 and P5))
    return out

configs = {"primary_cell_lockin": evaluate("cell", "lockin"), "design_sketch_point_std": evaluate("point", "std"),
           "point_lockin": evaluate("point", "lockin"), "cell_std": evaluate("cell", "std")}
prim = configs["primary_cell_lockin"]

# ---------------- D3: like-for-like slopes ----------------
FFp = np.array([float(L[f"FF_pressure_wR{q}"]["value"]) for q in WR])
FFm_lin = np.array([float(L[f"FF_mass_wR{q}"]["value"]) for q in WR])
EXm = np.array([float(L[f"EXACT_mass_ratio_wR{q}"]["value"]) for q in WR])
D3 = dict(
    solver_pressure_LS_N8000=prim["pressure"]["slope"], solver_pressure_LS_N16000=prim["pressure"]["slope_fine"],
    solver_mass_LS_N8000=prim["mass"]["slope"], solver_mass_LS_N16000=prim["mass"]["slope_fine"],
    FF_pressure_LS=slope(FFp), FF_pressure_endpoint=float(math.log(FFp[-1] / FFp[0]) / math.log(8)),
    FF_mass_linear_LS=2 + slope(FFm_lin), FF_mass_linear_endpoint=float(2 + math.log(FFm_lin[-1] / FFm_lin[0]) / math.log(8)),
    EXACT_mass_LS=2 + slope(EXm), EXACT_mass_endpoint=float(2 + math.log(EXm[-1] / EXm[0]) / math.log(8)),
    note="v1 table quoted -1.42e-4 for the pressure slope: that was the endpoint slope of the form factor (-1.427e-4), not the pre-registered "
         "least-squares slope (-1.366e-4) actually computed by the analysis.")

# ---------------- D1: solver vs exact far-field amplitude (no solver in the comparator) ----------------
D1 = {}
for N in (4000, 8000, 16000):
    D1[str(N)] = dict(
        pressure_solver_over_exact_minus_1=[get(N, "cell", "pressure", q)["ratio_lockin"] / f - 1 for q, f in zip(WR, FFp)],
        mass_solver_over_exact_minus_1=[get(N, "cell", "mass", q)["ratio_lockin"] / e - 1 for q, e in zip(WR, EXm)],
        mass_solver_over_linear_FF_minus_1=[get(N, "cell", "mass", q)["ratio_lockin"] / f - 1 for q, f in zip(WR, FFm_lin)],
        mass_2w_over_w_solver=[get(N, "cell", "mass", q)["A_2w"] / get(N, "cell", "mass", q)["A_lockin"] for q in WR])
D1["linear_FF_over_exact_minus_1_mass"] = [f / e - 1 for f, e in zip(FFm_lin, EXm)]
D1["exact_2w_over_w_mass"] = [abs(float(L[f"EXACT_mass_2w_over_w_wR{q}"]["value"])) for q in WR]
D1["note"] = ("pressure mode is exactly linear in eps, so its exact amplitude is A_press x FF_pressure (Mentat Si closed form); the mass-mode "
              "exact amplitude is all orders in omega R/c AND eps (Mentat closed form + N(Integral, 30 digits)); the v1 'all-orders' mass form "
              "factor is first order in eps")
for N in ("4000", "8000", "16000"):
    D1[N]["max_abs_pressure"] = float(max(abs(x) for x in D1[N]["pressure_solver_over_exact_minus_1"]))
    D1[N]["max_abs_mass"] = float(max(abs(x) for x in D1[N]["mass_solver_over_exact_minus_1"]))
    D1[N]["max_abs_mass_vs_linear_FF"] = float(max(abs(x) for x in D1[N]["mass_solver_over_linear_FF_minus_1"]))

# ---------------- S1 phase (unchanged) ----------------
S1 = {m: [wrap(get(8000, "cell", m, q)["phase_rad"] - math.pi) for q in WR] for m in ("pressure", "mass")}
S1_pass = all(abs(x) < 0.05 for v in S1.values() for x in v)

# ---------------- S2 ramp (unchanged) ----------------
ramp = {}
rr = np.linspace(0, U.R, 400001)
x_ = np.maximum(rr / U.a, 1e-12)
rho0 = U.RHO_C * np.where(rr < U.R, np.sin(x_) / x_, 0.0)
s0 = U.COEF * (rho0 * U.C**2 - 3 * U.K * rho0**2)
w2 = rr**2 * s0
v_static_mentat = float(L["v_static_m"]["value"])
for Nn in (8000, 16000):
    rm = [r for r in runs if r.get("label") == "ramp" and r["N"] == Nn][0]
    pr = tr[f"ramp_N{Nn}"]; dr = rm["dr"]; ip = rm["ip"]
    n = np.arange(pr.size); xpos = (n - ip) * dr
    sel = (xpos > -1.5 * U.R) & (xpos < 1.5 * U.R)
    van = np.array([np.trapezoid(w2 * np.clip((xx + rr) / (2 * np.maximum(rr, 1e-30)), 0, 1), rr) for xx in xpos[sel]])
    dev = np.abs(pr[sel] - van) / v_static_mentat
    after = xpos > 1.0 * U.R + 2 * dr
    before = xpos < -1.0 * U.R - 2 * dr
    ramp[Nn] = dict(max_dev_over_vstatic=float(dev.max()), max_abs_before=float(np.max(np.abs(pr[before]))),
                    max_dev_after_over_vstatic=float(np.max(np.abs(pr[after] / v_static_mentat - 1))),
                    v_final=float(pr[-1]), v_static_mentat=v_static_mentat, v_final_rel_err=float(pr[-1] / v_static_mentat - 1),
                    first_arrival_offset_cells=rm["arrival_offset_cells"],
                    xs_over_R=(xpos[sel] / U.R).tolist()[::5], v_over_vs=(pr[sel] / v_static_mentat).tolist()[::5],
                    van_over_vs=(van / v_static_mentat).tolist()[::5],
                    tag="v_final exact to 1e-13 is [BY CONSTRUCTION] of the conservative source (discrete static Green's function min(i,j) dr); "
                        "zero before (r-R)/c is [BY CONSTRUCTION] of the CFL-1 domain of dependence")
S2_pass = all(r["max_dev_over_vstatic"] < 1e-2 and r["max_dev_after_over_vstatic"] < 1e-3 and abs(r["v_final_rel_err"]) < 1e-3
              and r["max_abs_before"] == 0.0 for r in ramp.values())

# ---------------- S3 pre-registered (unchanged) ----------------
S3 = {m: dict(at_cs=get(8000, "cell", m, 0.01)["res_max_over_A_at_cs"], control=get(8000, "cell", m, 0.01)["res_max_over_A_control"],
              t_cs_over_R_c=get(8000, "cell", m, 0.01)["t_cs_over_R_c"]) for m in ("pressure", "mass")}
S3_pass = all(v["at_cs"] < 1e-3 for v in S3.values())
S3_floor = {m: {str(N): dict(rms_all=get(N, "cell", m, 0.01)["res_rms_over_A_all"], max_at_cs=get(N, "cell", m, 0.01)["res_max_over_A_at_cs"],
                         max_control=get(N, "cell", m, 0.01)["res_max_over_A_control"]) for N in (4000, 8000, 16000)} for m in ("pressure", "mass")}
S3_other_q_mass_N8000 = {str(q): get(8000, "cell", "mass", q).get("res_max_over_A_at_cs") for q in WR}

# ---------------- D2: S3 window sweep on the full-resolution traces ----------------
D2, sweep_arrays = {}, {}
for m in ("pressure", "mass"):
    for N in (4000, 8000, 16000):
        meta = get(N, "cell", m, 0.01); probe = full[f"cell_{m}_0.01_N{N}"]
        dt, om, T, rp = meta["dt"], meta["om"], meta["T"], meta["r_probe"]
        t = np.arange(probe.size) * dt; tend = t[-1]
        wfit = t > tend - 3 * T
        coef, *_ = np.linalg.lstsq(U.fit_basis(t[wfit] - rp / U.C, om), probe[wfit], rcond=None)
        A = float(math.hypot(coef[1], coef[2]))
        assert abs(A / meta["A_lockin"] - 1) < 1e-12
        tss = (rp + U.R) / U.C + 10 * U.R / U.C
        sel = t > tss
        res = np.abs(probe[sel] - U.fit_basis(t[sel] - rp / U.C, om) @ coef) / A
        ts = t[sel] / (U.R / U.C)
        half = 20.0
        centres = np.arange(ts[0] + half, ts[-1] - half, 2.0)
        lo = np.searchsorted(ts, centres - half); hi = np.searchsorted(ts, centres + half)
        mx = np.array([res[i:j].max() for i, j in zip(lo, hi)])
        spec = np.abs(np.fft.rfft(mx - mx.mean())); fr = np.fft.rfftfreq(mx.size, d=2.0)
        kpk = int(np.argmax(spec[1:]) + 1)
        Tn = T / (U.R / U.C)
        phi = 2 * math.pi * (centres - rp / U.R) / Tn          # omega (t - r_p/c) at the window centre
        corr = float(np.corrcoef(mx, np.abs(np.cos(phi)))[0, 1])
        t_cs = meta["t_cs_over_R_c"]
        phi_cs = 2 * math.pi * (t_cs - rp / U.R) / Tn
        D2[f"{m}_N{N}"] = dict(A_lockin=A, prereg_window_value=meta["res_max_over_A_at_cs"], n_windows=int(mx.size),
                               window_max_min=float(mx.min()), window_max_median=float(np.median(mx)), window_max_max=float(mx.max()),
                               frac_windows_over_1em3=float(np.mean(mx > 1e-3)), envelope_period_over_Rc=float(1 / fr[kpk]),
                               T_over_Rc=Tn, half_T_over_Rc=Tn / 2, corr_winmax_with_abs_cos_phase=corr,
                               abs_cos_phase_at_sonic_window=float(abs(math.cos(phi_cs))),
                               median_winmax_turning_points=float(np.median(mx[np.abs(np.cos(phi)) < 0.2])),
                               median_winmax_peak_speed=float(np.median(mx[np.abs(np.cos(phi)) > 0.9])))
        sweep_arrays[f"{m}_N{N}"] = (centres, mx)
        print(m, N, json.dumps(D2[f"{m}_N{N}"]), flush=True)

# ---------------- S4 convergence order ----------------
S4 = {}
for m in ("pressure", "mass"):
    S4[m] = []
    for q in WR:
        a4, a8, a16 = (get(N, "cell", m, q)["A_lockin"] for N in (4000, 8000, 16000))
        d1, d2 = abs(a4 - a8), abs(a8 - a16)
        S4[m].append(dict(wR=q, A4000=a4, A8000=a8, A16000=a16, rel_8_16=d2 / a16, order=(math.log2(d1 / d2) if d2 > 0 and d1 > 0 else None)))
S4_range = {m: [min(x["order"] for x in S4[m]), max(x["order"] for x in S4[m])] for m in S4}

# ---------------- smoke: discrete monopole conservation (re-computed, no solver) ----------------
s = U.Source(8000, "cell", "mass", 0.05, 1.0); sp = U.Source(8000, "point", "mass", 0.05, 1.0)
lams = np.linspace(0.95, 1.05, 2001); mc, mp = [], []
x0 = np.maximum(sp.rj / U.a, 1e-12)
rho0p = U.RHO_C * np.where(sp.rj < U.R, np.sin(x0) / x0, 0.0)
for lam in lams:
    mc.append(s.mass_cell(lam).sum() / s.Mcell0.sum() - 1)
    xx = np.maximum(sp.rj / (U.a * lam), 1e-12)
    rho = U.RHO_C * np.where(sp.rj < U.R * lam, np.sin(xx) / xx, 0.0) / lam**3
    mp.append((sp.rj**2 * rho).sum() / (sp.rj**2 * rho0p).sum() - 1)
smoke = dict(lambda_grid="2001 values in [0.95, 1.05]",
             cell_monopole_max_abs_change=float(np.max(np.abs(mc))), point_monopole_max_abs_change=float(np.max(np.abs(mp))),
             point_monopole_rms_change=float(np.sqrt(np.mean(np.array(mp)**2))),
             point_monopole_std_about_mean=float(np.std(mp)),
             note_point="lambda = 1 sits at a B2 maximum (R/dr = 200 exactly), so the change relative to lambda = 1 is one-signed; its "
                        "peak-to-peak (dense scan) is the max |change|, and the fluctuation about its mean (std) is what prereg.json estimated (rms ~ 9e-6)",
             prereg_recorded="rms ~ 9e-6 xi r_s of spurious monopole (prereg.json); the v1 '1.97e-5 a priori' peak figure was NOT pre-registered "
                             "(label dropped in v2)",
             M_grid_kg=float(4 * np.pi * s.Mcell0.sum()), M_mentat_kg=float(L["M_kg"]["value"]),
             sumQ_over_vstatic_minus_1=float((U.COEF * (U.C**2 * s.Mcell0 - 3 * s.Pcell)).sum() / v_static_mentat - 1))

out = dict(
    id="U6", version="v2 (repair round)", prereg=json.load(open(os.path.join(HERE, "prereg.json"), encoding="utf-8")),
    prereg_v2=json.load(open(os.path.join(HERE, "prereg_v2.json"), encoding="utf-8")),
    mentat_leaves=L, physics_v2=PH, reproducibility_v2_vs_v1=repro,
    verdict_primary=prim["criteria"], verdict_design_sketch=configs["design_sketch_point_std"]["criteria"],
    configs=configs, D1_solver_vs_exact=D1, D2_S3_window_sweep=D2, D3_slopes=D3,
    S1_phase_minus_pi=S1, S1_pass=S1_pass, S2_ramp=ramp, S2_pass=S2_pass,
    S3_prereg_window=S3, S3_pass_as_preregistered=S3_pass, S3_residual_floor_vs_N=S3_floor, S3_mass_N8000_sonic_window_other_q=S3_other_q_mass_N8000,
    S3_reading="pressure mode: clean null (2e-9, no moving surface). mass mode: the pre-registered window passes (1.7e-4 < 1e-3) but the "
               "test is NON-DISCRIMINATING at N_r 8000 -- the O(dr) residual floor is periodic at T/2 (tracks surface speed |cos phi|) and "
               "the sonic window fell at a turning point; see D2. S3 is [BY CONSTRUCTION] anyway (c_s is not in the equation).",
    S4_convergence=S4, S4_order_range=S4_range, smoke=smoke,
    runs=[{k: v for k, v in r.items()} for r in runs],
)
json.dump(out, open(os.path.join(HERE, "u6_results_v2.json"), "w", encoding="utf-8"), indent=1)

# ======================= figure =======================
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
S = ["#2a78d6", "#eb6834", "#1baf7a", "#8b5cf6"]
plt.rcParams.update({"axes.unicode_minus": True, "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
fig, axs = plt.subplots(2, 2, figsize=(16, 11.5), facecolor=SURF)
ax = axs.ravel()
for a_ in ax:
    a_.set_facecolor(SURF); a_.grid(True, color=GRID, lw=0.8); a_.set_axisbelow(True)
    for sp_ in ("top", "right"):
        a_.spines[sp_].set_visible(False)
def mfmt(x, fmt):
    return format(x, fmt).replace("-", "−")
SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
def sci(x, nd=1):
    m, e = f"{x:.{nd}e}".split("e")
    return f"{m}×10{str(int(e)).translate(SUP)}"
xs = np.array(WR); w_c = L["choices"]["w_c"]; eps = L["choices"]["eps"]
rs = float(L["r_s_m"]["value"]); xir = U.XI * rs
p_ = prim["pressure"]; m_ = prim["mass"]; ds = configs["design_sketch_point_std"]["mass"]
# (a) amplitudes
ax[0].loglog(xs, np.array(p_["A"]) / xir, "o", color=S[0], ms=8, label="prescribed pressure mode, solver")
ax[0].loglog(xs, np.array(p_["A_pred"]) / xir, "--", color=S[0], lw=1.2, label="prediction ξ r_s w_c ε/2")
ax[0].loglog(xs, np.array(m_["A"]) / xir, "s", color=S[1], ms=8, label="prescribed homologous mass mode, solver")
ax[0].loglog(xs, np.array(m_["A_pred"]) / xir, "--", color=S[1], lw=1.2, label="prediction ξ r_s ε ω² I₂/(3Mc²)")
ax[0].loglog(xs, np.array(ds["A"]) / xir, "^", color=S[2], ms=8, label="mass mode, design sketch (point source, std·√2)")
lo_mw, hi_mw = ph("IV_metric_rhostar_over_Apress") * w_c * eps / 2, ph("IV_metric_flatrho_over_Apress") * w_c * eps / 2
ax[0].fill_between([0.008, 0.1], lo_mw, hi_mw, color=S[3], alpha=0.16, lw=0,
                   label="mass mode with the §2.1 metric weight: 2–4 × A_press [DERIVED, not run]")
qb = ph("III_breathing_omegaR_over_c_Gamma2")
ax[0].loglog([qb], [ph("III_amp_isolated_flat_over_xirs_eps") * eps], "D", mfc="none", mec=S[3], mew=1.8, ms=10,
             label="isolated star at its breathing frequency, flat □: (2/3) w_c ε [DERIVED]")
ax[0].set_xlim(0.008, 0.1); ax[0].set_ylim(1e-7, 2e-4)
ax[0].set_xlabel("ωR / c"); ax[0].set_ylabel("far-field amplitude of rσ at 30R  /  ξ r_s")
ax[0].set_title(f"flat □, prescribed sources: log slope pressure {mfmt(D3['solver_pressure_LS_N8000']*1e4, '+.2f')}×10⁻⁴ (pred. 0), "
                f"mass {D3['solver_mass_LS_N8000']:.4f} (pred. 2)", color=INK, fontsize=10)
ax[0].set_xticks(WR); ax[0].set_xticklabels(["0.01", "0.02", "0.04", "0.08"]); ax[0].minorticks_off()
ax[0].legend(fontsize=7.5, frameon=False, loc="lower right")
# (b) solver vs exact
for i, N in enumerate((4000, 8000, 16000)):
    ax[1].semilogy(xs * (1 + 0.03 * (i - 1)), np.abs(D1[str(N)]["pressure_solver_over_exact_minus_1"]), "o-", color=S[i], ms=6, lw=1,
                   label=f"pressure mode, N_r {N}")
    ax[1].semilogy(xs * (1 + 0.03 * (i - 1)), np.abs(D1[str(N)]["mass_solver_over_exact_minus_1"]), "s--", color=S[i], ms=6, lw=1,
                   mfc="none", label=f"mass mode, N_r {N}")
ax[1].semilogy(xs, np.abs(D1["linear_FF_over_exact_minus_1_mass"]), "x:", color=S[3], ms=8, lw=1.4,
               label="linear-in-ε form factor vs exact (the ε² term)")
ax[1].set_xscale("log"); ax[1].set_xticks(WR); ax[1].set_xticklabels(["0.01", "0.02", "0.04", "0.08"]); ax[1].minorticks_off()
ax[1].set_xlabel("ωR / c"); ax[1].set_ylabel("|A_solver / A_exact − 1|")
ax[1].set_title(f"solver vs an exact retarded amplitude (all orders in ωR/c and ε, no solver):" + chr(10) +
                f"agreement ≤ {sci(max(D1['8000']['max_abs_mass'], D1['8000']['max_abs_pressure']))} at N_r 8000, "
                f"≤ {sci(max(D1['16000']['max_abs_mass'], D1['16000']['max_abs_pressure']))} at N_r 16000", color=INK, fontsize=10)
ax[1].set_ylim(1e-11, 3e-6)
ax[1].legend(fontsize=7.5, frameon=False, loc="lower right", ncol=2)
# (c) ramp
rp8 = ramp[8000]
ax[2].plot(rp8["xs_over_R"], rp8["v_over_vs"], "-", color=S[0], lw=2.2, label="solver, static switch-on (N_r 8000)")
ax[2].plot(rp8["xs_over_R"], rp8["van_over_vs"], "--", color=S[1], lw=1.4, label="retarded integral (analytic)")
for xv in (-1, 1):
    ax[2].axvline(xv, color=INK2, lw=0.9, ls=":")
ax[2].text(-0.97, 0.5, "(r − R)/c", rotation=90, color=INK2, fontsize=8, va="center")
ax[2].text(1.03, 0.5, "(r + R)/c", rotation=90, color=INK2, fontsize=8, va="center")
ax[2].set_xlabel("(ct − r_p) / R   at r_p = 30R"); ax[2].set_ylabel("rσ(r_p, t) / [ξ r_s (1 − w_c/2)]")
ax[2].set_title(f"switch-on matches the retarded integral to {rp8['max_dev_over_vstatic']*100:.2f} % of v_static; "
                "light-cone arrival holds by construction (CFL-1 leapfrog)", color=INK, fontsize=10)
ax[2].legend(fontsize=8, frameon=False, loc="upper left")
# (d) S3 window sweep
for i, N in enumerate((4000, 8000, 16000)):
    cc, mx = sweep_arrays[f"mass_N{N}"]
    ax[3].semilogy(cc, mx, "-", color=S[i], lw=1.3, label=f"mass mode, N_r {N}")
cc, mx = sweep_arrays["pressure_N8000"]
ax[3].semilogy(cc, mx, "-", color=S[3], lw=1.3, label="pressure mode, N_r 8000")
ax[3].axhline(1e-3, color=INK, lw=1.0, ls="--", label="pre-registered threshold 10⁻³")
tcs = S3["mass"]["t_cs_over_R_c"]
ax[3].axvspan(tcs - 20, tcs + 20, color=INK2, alpha=0.18, label="pre-registered sonic window (r − R)/c_s ± 20 R/c")
ax[3].set_xlabel("window centre  t / (R/c)   (ωR/c = 0.01)"); ax[3].set_ylabel("max |rσ − steady-state fit| / A  in a ±20 R/c window")
ax[3].set_ylim(1e-10, 1e-1)
d8 = D2["mass_N8000"]
ax[3].set_title(f"mass-mode floor tracks surface speed (period ≈ T/2) and the sonic window sat at a turning point:\n"
                f"S3 is non-discriminating at N_r 8000 ({d8['frac_windows_over_1em3']*100:.0f} % of windows exceed 10⁻³); the pressure mode is a clean null",
                color=INK, fontsize=10)
ax[3].legend(fontsize=7.5, frameon=True, facecolor=SURF, edgecolor=GRID, loc="lower right", bbox_to_anchor=(1.0, 0.2))
verdict = "PASS" if prim["criteria"]["PASS"] else "FAIL"
fig.suptitle(f"U6 {verdict} as pre-registered, read only as a check of flat-□ algebra and solver: a prescribed pressure pulsation radiates σ at zeroth order, "
             f"a prescribed homologous mass pulsation at (ωR/c)²" + chr(10) +
             "the split belongs to the prescriptions and the flat truncation, not to a real star (energy–virial identity; dropped metric weight); "
             f"the design sketch misreads the mass mode by ×{configs['design_sketch_point_std']['mass']['ratio'][0]:.1f}",
             color=INK, fontsize=11.5, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.985))
fig.savefig(os.path.join(HERE, "u6_figure_v2.png"), dpi=140, facecolor=SURF)
print(json.dumps(dict(repro=repro, primary=prim["criteria"], design_sketch=configs["design_sketch_point_std"]["criteria"],
                      point_lockin=configs["point_lockin"]["criteria"], cell_std=configs["cell_std"]["criteria"],
                      S1=S1_pass, S2=S2_pass, S3=S3_pass, D3=D3, S4_range=S4_range, smoke=smoke,
                      D1={N: {k: D1[N][k] for k in ("max_abs_pressure", "max_abs_mass", "max_abs_mass_vs_linear_FF")} for N in ("4000", "8000", "16000")}),
                 indent=1))
