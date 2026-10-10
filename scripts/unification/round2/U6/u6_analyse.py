"""U6 analysis: evaluate the pre-registered criteria (prereg.json) on results_{A,B,C}.json, write u6_results.json and u6_figure.png."""
import sys, os, json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import u6_sigma_wave as U

L = U.L
runs = []
for s in "ABC":
    runs += json.load(open(os.path.join(HERE, f"results_{s}.json"), encoding="utf-8"))
tr = {}
for s in "ABC":
    with np.load(os.path.join(HERE, f"traces_{s}.npz")) as z:
        tr.update({k: z[k] for k in z.files})

WR = [0.01, 0.02, 0.04, 0.08]
def get(N, kind, mode, wR):
    for r in runs:
        if r.get("label") != "ramp" and r["N"] == N and r["kind"] == kind and r["mode"] == mode and abs(r["wR"] - wR) < 1e-12:
            return r
    raise KeyError((N, kind, mode, wR))

def slope(As):
    x = np.log(WR); y = np.log(As)
    p = np.polyfit(x, y, 1)
    return float(p[0])

def wrap(ph):
    return float((ph + math.pi) % (2 * math.pi) - math.pi)

def evaluate(kind, est, N=8000, Nf=16000):
    """The design's P1-P5 for one configuration (source discretisation 'kind', amplitude estimator 'est')."""
    key = "A_lockin" if est == "lockin" else "A_std_sqrt2"
    out = {}
    for mode in ("pressure", "mass"):
        A = [get(N, kind, mode, q)[key] for q in WR]
        Af = [get(Nf, kind, mode, q)[key] for q in WR]
        pred = [get(N, kind, mode, q)["A_pred"] for q in WR]
        out[mode] = dict(A=A, A_fine=Af, A_pred=pred, ratio=[a / p for a, p in zip(A, pred)], slope=slope(A),
                         slope_fine=slope(Af), grid_change=[abs(f / a - 1) for a, f in zip(A, Af)],
                         arrival_offset_cells=[get(N, kind, mode, q)["arrival_offset_cells"] for q in WR])
    P1 = abs(out["pressure"]["slope"] - 0.0) <= 0.1
    P2 = abs(out["mass"]["slope"] - 2.0) <= 0.1
    P3 = all(abs(out[m]["ratio"][0] - 1.0) <= 0.10 for m in ("pressure", "mass"))
    P4 = all(abs(x) <= 2 for m in ("pressure", "mass") for x in out[m]["arrival_offset_cells"])
    P5 = all(g < 0.02 for m in ("pressure", "mass") for g in out[m]["grid_change"])
    out["criteria"] = dict(P1_pressure_slope=P1, P2_mass_slope=P2, P3_amplitude_at_0p01=P3, P4_first_arrival=P4, P5_grid_halving=P5,
                           PASS=bool(P1 and P2 and P3 and P4 and P5))
    return out

configs = {
    "primary_cell_lockin": evaluate("cell", "lockin"),
    "design_sketch_point_std": evaluate("point", "std"),
    "point_lockin": evaluate("point", "lockin"),
    "cell_std": evaluate("cell", "std"),
}

# S1 phase (primary runs, N = 8000)
S1 = {m: [wrap(get(8000, "cell", m, q)["phase_rad"] - math.pi) for q in WR] for m in ("pressure", "mass")}
S1_pass = all(abs(x) < 0.05 for v in S1.values() for x in v)

# S2 ramp vs analytic
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
    n = np.arange(pr.size); xpos = (n - ip) * dr          # c t - r_p
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
                    van_over_vs=(van / v_static_mentat).tolist()[::5])
S2_pass = all(r["max_dev_over_vstatic"] < 1e-2 and r["max_dev_after_over_vstatic"] < 1e-3 and abs(r["v_final_rel_err"]) < 1e-3
              and r["max_abs_before"] == 0.0 for r in ramp.values())

# S3 no feature at c_s (primary, N = 8000, wR = 0.01)
S3 = {m: dict(at_cs=get(8000, "cell", m, 0.01)["res_max_over_A_at_cs"], control=get(8000, "cell", m, 0.01)["res_max_over_A_control"],
              t_cs_over_R_c=get(8000, "cell", m, 0.01)["t_cs_over_R_c"]) for m in ("pressure", "mass")}
S3_pass = all(v["at_cs"] < 1e-3 for v in S3.values())
S3_floor = {m: {str(N): dict(rms_all=get(N, "cell", m, 0.01)["res_rms_over_A_all"], max_at_cs=get(N, "cell", m, 0.01)["res_max_over_A_at_cs"],
                         max_control=get(N, "cell", m, 0.01)["res_max_over_A_control"]) for N in (4000, 8000, 16000)} for m in ("pressure", "mass")}

# S4 convergence order (cell, lock-in)
S4 = {}
for m in ("pressure", "mass"):
    S4[m] = []
    for q in WR:
        a4, a8, a16 = (get(N, "cell", m, q)["A_lockin"] for N in (4000, 8000, 16000))
        d1, d2 = abs(a4 - a8), abs(a8 - a16)
        S4[m].append(dict(wR=q, A4000=a4, A8000=a8, A16000=a16, rel_8_16=d2 / a16,
                          order=(math.log2(d1 / d2) if d2 > 0 and d1 > 0 else None)))

prim = configs["primary_cell_lockin"]
out = dict(
    id="U6", prereg=json.load(open(os.path.join(HERE, "prereg.json"), encoding="utf-8")),
    mentat_leaves=L,
    verdict_primary=prim["criteria"], verdict_design_sketch=configs["design_sketch_point_std"]["criteria"],
    configs=configs, S1_phase_minus_pi=S1, S1_pass=S1_pass, S2_ramp=ramp, S2_pass=S2_pass, S3_no_cs_feature=S3, S3_pass=S3_pass, S3_residual_floor_vs_N=S3_floor,
    S4_convergence=S4,
    runs=[{k: v for k, v in r.items()} for r in runs],
)
json.dump(out, open(os.path.join(HERE, "u6_results.json"), "w", encoding="utf-8"), indent=1)

# ---------------- figure ----------------
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
S = ["#2a78d6", "#eb6834", "#1baf7a", "#8b5cf6"]
plt.rcParams.update({"axes.unicode_minus": True, "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
fig, ax = plt.subplots(1, 3, figsize=(16, 5.2), facecolor=SURF)
for a_ in ax:
    a_.set_facecolor(SURF); a_.grid(True, color=GRID, lw=0.8); a_.set_axisbelow(True)
    for sp in ("top", "right"):
        a_.spines[sp].set_visible(False)
xs = np.array(WR)
rs = float(L["r_s_m"]["value"]); xir = U.XI * rs
p_ = prim["pressure"]; m_ = prim["mass"]; ds = configs["design_sketch_point_std"]["mass"]
ax[0].loglog(xs, np.array(p_["A"]) / xir, "o", color=S[0], ms=8, label="pressure mode, solver (lock-in)")
ax[0].loglog(xs, np.array(p_["A_pred"]) / xir, "--", color=S[0], lw=1.2, label="prediction  ξ r_s w_c ε / 2")
ax[0].loglog(xs, np.array(m_["A"]) / xir, "s", color=S[1], ms=8, label="mass mode, solver (lock-in)")
ax[0].loglog(xs, np.array(m_["A_pred"]) / xir, "--", color=S[1], lw=1.2, label="prediction  ξ r_s ε ω² I₂ / (3Mc²)")
ax[0].loglog(xs, np.array(ds["A"]) / xir, "^", color=S[2], ms=8, label="mass mode, design sketch (point source, std·√2)")
ax[0].set_xlabel("ωR / c"); ax[0].set_ylabel("far-field amplitude of rσ at 30R  /  ξ r_s")
ax[0].set_title(f"log slopes: pressure {p_['slope']*1e4:+.1f}×10⁻⁴ (pred. 0), mass {m_['slope']:.4f} (pred. 2)".replace("-", "−"), color=INK, fontsize=10)
ax[0].set_xticks(WR); ax[0].set_xticklabels(["0.01", "0.02", "0.04", "0.08"]); ax[0].minorticks_off()
ax[0].legend(fontsize=8, frameon=False, loc="lower right")

rp8 = ramp[8000]
ax[1].plot(rp8["xs_over_R"], rp8["v_over_vs"], "-", color=S[0], lw=2.2, label="solver, static switch-on")
ax[1].plot(rp8["xs_over_R"], rp8["van_over_vs"], "--", color=S[1], lw=1.4, label="retarded integral (analytic)")
for xv in (-1, 1):
    ax[1].axvline(xv, color=INK2, lw=0.9, ls=":")
ax[1].text(-0.97, 0.5, "(r − R)/c", rotation=90, color=INK2, fontsize=8, va="center")
ax[1].text(1.03, 0.5, "(r + R)/c", rotation=90, color=INK2, fontsize=8, va="center")
xcs = (S3["pressure"]["t_cs_over_R_c"]) - 30
ax[1].set_xlabel("(ct − r_p) / R   at r_p = 30R"); ax[1].set_ylabel("rσ(r_p, t) / [ξ r_s (1 − w_c/2)]")
ax[1].set_title(f"σ arrives on the light cone; sound would arrive at +{xcs:.0f} R", color=INK, fontsize=10)
ax[1].legend(fontsize=8, frameon=False, loc="upper left")

for i, m in enumerate(("pressure", "mass")):
    r8 = get(8000, "cell", m, 0.01)
    key = f"cell_{m}_0.01_N8000"; pr = tr[key]; dec = int(tr[key + "_dec"][0])
    t = np.arange(pr.size) * dec * r8["dt"]
    tss = (r8["r_probe"] + U.R) / U.C + 10 * U.R / U.C
    w = t > tss
    tend = r8["nsteps"] * r8["dt"]; T = r8["T"]
    wf = (np.arange(r8["nsteps"] + 1) * r8["dt"] > tend - 3 * T)
    # refit on the decimated samples of the last 3 periods (same basis as the analysis)
    wd = t > tend - 3 * T
    coef, *_ = np.linalg.lstsq(U.fit_basis(t[wd] - r8["r_probe"] / U.C, r8["om"]), pr[wd], rcond=None)
    res = pr[w] - U.fit_basis(t[w] - r8["r_probe"] / U.C, r8["om"]) @ coef
    ax[2].semilogy(t[w] / (U.R / U.C), np.maximum(np.abs(res) / r8["A_lockin"], 1e-13), ".", ms=1.5, color=S[i],
                   label=f"{m} mode, |residual| / A" + (" (floor: O(dr) surface aliasing)" if m == "mass" else ""))
ax[2].axvspan(S3["pressure"]["t_cs_over_R_c"] - 20, S3["pressure"]["t_cs_over_R_c"] + 20, color=S[3], alpha=0.18,
              label="sonic arrival (r − R)/c_s ± 20 R/c")
ax[2].set_xlabel("t / (R/c)   (ωR/c = 0.01, N_r = 8000)"); ax[2].set_ylabel("|rσ − steady-state fit| / amplitude")
ax[2].set_title("no feature at the sound-crossing time", color=INK, fontsize=10)
ax[2].set_ylim(1e-13, 1e0)
ax[2].legend(fontsize=8, frameon=True, facecolor=SURF, edgecolor=GRID, loc="center right", markerscale=6)
verdict = "PASS" if prim["criteria"]["PASS"] else "FAIL"
fig.suptitle(f"U6 {verdict} (solver + derivation check, not evidence for the thesis): a pressure pulsation radiates σ at zeroth order, "
             f"a mass redistribution only at (ωR/c)², and σ arrives on the light cone" + chr(10) +
             f"the design sketch's point-sampled source breaks discrete mass conservation and misreads the mass mode by ×{configs['design_sketch_point_std']['mass']['ratio'][0]:.0f} at ωR/c = 0.01",
             color=INK, fontsize=11.5, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(os.path.join(HERE, "u6_figure.png"), dpi=150, facecolor=SURF)
print(json.dumps(dict(primary=prim["criteria"], design_sketch=configs["design_sketch_point_std"]["criteria"],
                      point_lockin=configs["point_lockin"]["criteria"], cell_std=configs["cell_std"]["criteria"],
                      S1=S1_pass, S2=S2_pass, S3=S3_pass), indent=1))
