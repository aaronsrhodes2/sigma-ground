"""U2 analysis: evaluates the frozen criteria (original + corrected), writes u2_results.json and the figure."""
import json, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
def J(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return json.load(f)

pre = J("prereg.json")
nb = J("res_nbody.json"); main = J("res_sph_main.json"); dl = J("res_sph_delta.json")
ideal = J("res_sph_ideal.json"); fr = J("res_fourier.json")
leaves = J("mentat_leaves.json")

def pick(runs, **kw):
    out = [r for r in runs if all((r.get(k) == v) for k, v in kw.items())]
    assert len(out) == 1, (kw, len(out))
    return out[0]

NB = nb["runs"]; SM = main["runs"]; SD = dl["runs"]; SI = ideal["runs"]
eps = {}
for T in (0.3, 1.0):
    for n in (600, 1200):
        eps[("nbody", T, n)] = pick(NB, T_tff=T, n_per_tff=n, delta_R=0.0, N=64)["eps_rec_R"]
        for a in (0.0, 1.0):
            eps[(f"sph{int(a)}", T, n)] = pick(SM, alpha=a, T_tff=T, n_per_tff=n)["eps_rec_R"]
for n in (600, 1200):
    for a in (0.0, 1.0):
        eps[(f"ideal{int(a)}", 1.0, n)] = pick(SI, alpha=a, n_per_tff=n)["eps_rec_R"]
ratio = {k[0] + f"_T{k[1]}": eps[(k[0], k[1], 1200)] / eps[(k[0], k[1], 600)]
         for k in eps if k[2] == 600}

# delta control (T = 1, dt = t_ff/600)
deltas = (1e-13, 1e-11, 1e-9)
dctl = {"nbody": [pick(NB, T_tff=1.0, n_per_tff=600, delta_R=d)["eps_rec_R"] for d in deltas],
        "sph0": [pick(SD, alpha=0.0, delta_R=d)["eps_rec_R"] for d in deltas],
        "sph1": [pick(SD, alpha=1.0, delta_R=d)["eps_rec_R"] for d in deltas]}
d0 = {"nbody": eps[("nbody", 1.0, 600)], "sph0": eps[("sph0", 1.0, 600)], "sph1": eps[("sph1", 1.0, 600)]}

def lin_slope(e, e0):
    pts = [(d, x) for d, x in zip(deltas, e) if x >= 3 * e0 and x <= 1e-2]
    if len(pts) < 2:
        return None
    x = np.log10([p[0] for p in pts]); y = np.log10([p[1] for p in pts])
    return float(np.polyfit(x, y, 1)[0])
slopes = {k: lin_slope(v, d0[k]) for k, v in dctl.items()}
gain = {k: [x / d for x, d in zip(v, deltas)] for k, v in dctl.items()}
sph1_delta_indep = abs(dctl["sph1"][-1] / d0["sph1"] - 1.0)

# ledgers
r1 = {n: pick(SM, alpha=1.0, T_tff=1.0, n_per_tff=n) for n in (600, 1200)}
r1s = {n: pick(SM, alpha=1.0, T_tff=0.3, n_per_tff=n) for n in (600, 1200)}
r0 = {n: pick(SM, alpha=0.0, T_tff=1.0, n_per_tff=n) for n in (600, 1200)}
i1 = {n: pick(SI, alpha=1.0, n_per_tff=n) for n in (600, 1200)}
i0 = {n: pick(SI, alpha=0.0, n_per_tff=n) for n in (600, 1200)}
sdot_min = min(r["Sdot_led_min"] for r in SM + SI if r["alpha"] == 1.0)
sdot_neg = sum(r["Sdot_led_n_negative"] for r in SM + SI)
qv_min = min(r["q_visc_eng_step_min_J"] for r in SM + SI if r["alpha"] == 1.0)
qv_neg = sum(r["q_visc_eng_n_negative_steps"] for r in SM + SI)

def eng_closure(r):
    U0 = abs(r["U_soft0_J"]); loss = r["loss_corrected_end_over_absU0"] * U0
    return abs(r["Q_visc_eng_end_J"] - loss) / r["Q_visc_eng_end_J"]
post_hoc = {
    "richardson_corrected_closure_T1": 2 * r1[1200]["closure_corrected_end"] - r1[600]["closure_corrected_end"],
    "richardson_corrected_closure_T0.3": 2 * r1s[1200]["closure_corrected_end"] - r1s[600]["closure_corrected_end"],
    "engine_own_viscous_booking_vs_mech_loss_T1": {n: eng_closure(r1[n]) for n in (600, 1200)},
    "engine_own_viscous_booking_vs_mech_loss_T0.3": {n: eng_closure(r1s[n]) for n in (600, 1200)},
    "Q_led_vs_Q_visc_engine_T1": {n: r1[n]["closure_engine_vs_led_end"] for n in (600, 1200)},
}

# ---------------- original criterion (verbatim clauses, strict reading per addendum_1)
T = 1.0
o = {}
o["hamiltonian_le_1e-8"] = all(eps[(e, T, n)] <= 1e-8 for e in ("nbody", "sph0") for n in (600, 1200))
o["hamiltonian_shrink_under_dt_halving"] = all(eps[(e, T, 1200)] < eps[(e, T, 600)] for e in ("nbody", "sph0"))
o["viscous_ge_1e-3"] = all(eps[("sph1", T, n)] >= 1e-3 for n in (600, 1200))
o["viscous_does_not_shrink"] = eps[("sph1", T, 1200)] >= eps[("sph1", T, 600)]
o["min_Sdot_ge_0"] = sdot_min >= 0 and sdot_neg == 0
o["ledger_closure_le_1e-3_literal"] = r1[1200]["closure_literal_end"] <= 1e-3
o["SAME_all_within_10x"] = (max(eps[(e, T, 1200)] for e in ("nbody", "sph0", "sph1")) /
                            min(eps[(e, T, 1200)] for e in ("nbody", "sph0", "sph1"))) <= 10
sep_o = all(o[k] for k in ("hamiltonian_le_1e-8", "hamiltonian_shrink_under_dt_halving", "viscous_ge_1e-3",
                           "viscous_does_not_shrink", "min_Sdot_ge_0", "ledger_closure_le_1e-3_literal"))
verdict_original = "SEPARATE" if sep_o else ("SAME" if o["SAME_all_within_10x"] else "MIXED")
s8_orig = {
    "isothermal_sum_m_uint_eq_Qcum": {"internal_at_T": r1[1200]["internal_at_T"], "Q_led_at_T": r1[1200]["Q_led_at_T"],
                                      "pass": abs(r1[1200]["internal_at_T"] - r1[1200]["Q_led_at_T"]) <= 1e-3 * r1[1200]["Q_led_at_T"]},
    "ideal_sum_m_uint_eq_Qcum_relerr": i1[1200]["sentinel_literal_uint_vs_Qled_T"],
    "ideal_pass": i1[1200]["sentinel_literal_uint_vs_Qled_T"] <= 1e-3,
    "K_Wsoft_Uint_conserved_1e-4": {
        f"{lab}_n{n}": {"max_over_run": r["total_drift_max_over_absU0"], "at_end": r["total_drift_end_over_absU0"]}
        for lab, d in (("isothermal_a1_T1", r1), ("isothermal_a0_T1", r0), ("ideal_a1_T1", i1), ("ideal_a0_T1", i0))
        for n, r in d.items()},
}
s8_orig["K_Wsoft_Uint_conserved_1e-4"]["pass_at_dt_tff_1200_max"] = all(
    v["max_over_run"] <= 1e-4 for k, v in s8_orig["K_Wsoft_Uint_conserved_1e-4"].items() if k.endswith("n1200"))
s8_orig["K_Wsoft_Uint_conserved_1e-4"]["pass_at_dt_tff_1200_end"] = all(
    v["at_end"] <= 1e-4 for k, v in s8_orig["K_Wsoft_Uint_conserved_1e-4"].items() if k.endswith("n1200"))
s8_orig["K_Wsoft_Uint_conserved_1e-4"]["pass_at_dt_tff_600_max"] = all(
    v["max_over_run"] <= 1e-4 for k, v in s8_orig["K_Wsoft_Uint_conserved_1e-4"].items() if k.endswith("n600"))

# ---------------- corrected criterion (frozen in prereg.json before running)
c = {}
c["H1"] = o["hamiltonian_le_1e-8"]
c["H2"] = all(0.1 <= eps[(e, T, 1200)] / eps[(e, T, 600)] <= 10 for e in ("nbody", "sph0"))
c["D1"] = (o["viscous_ge_1e-3"] and abs(eps[("sph1", T, 1200)] / eps[("sph1", T, 600)] - 1) <= 0.1
           and sph1_delta_indep <= 0.1)
c["D2"] = o["min_Sdot_ge_0"] and qv_min >= 0 and qv_neg == 0
c["D3"] = r1[1200]["closure_corrected_end"] <= 1e-3
c["delta_linear_hamiltonian"] = {k: slopes[k] for k in ("nbody", "sph0")}
sep_c = all(c[k] for k in ("H1", "H2", "D1", "D2", "D3"))
same_c = o["SAME_all_within_10x"]
verdict_corrected = "SEPARATE" if sep_c else ("SAME" if same_c else "MIXED")
failed_c = [k for k in ("H1", "H2", "D1", "D2", "D3") if not c[k]]
sep_without_D3 = all(c[k] for k in ("H1", "H2", "D1", "D2"))

# Lyapunov estimates
lyap = {}
for e in ("nbody", "sph0"):
    lyap[e] = {"ratio_T1_over_T0.3_dt1200": eps[(e, 1.0, 1200)] / eps[(e, 0.3, 1200)],
               "round_trip_gain_from_delta": float(np.median(gain[e])),
               "lambda_per_tff_from_gain": math.log(float(np.median(gain[e]))) / 2.0}

results = {
    "id": "U2", "prereg": pre,
    "mentat_leaves": leaves,
    "choices": main["choices"],
    "eps_rec_R": {f"{k[0]}_T{k[1]}_n{k[2]}": v for k, v in eps.items()},
    "dt_halving_ratio": ratio,
    "delta_control": {"deltas_R": deltas, "eps_rec_R": dctl, "eps_rec_delta0_R": d0, "loglog_slope": slopes,
                      "round_trip_gain": gain, "sph1_rel_change_delta1e-9": sph1_delta_indep},
    "lyapunov": lyap,
    "smoke_N32": {"nbody": pick(NB, N=32)["eps_rec_R"],
                  "sph0": pick(SD, label="smoke_N32", alpha=0.0)["eps_rec_R"],
                  "sph1": pick(SD, label="smoke_N32", alpha=1.0)["eps_rec_R"],
                  "design_smoke": [1.7e-16, 2.3e-15, 1.5e-2]},
    "ledger": {
        "Sdot_led_min_alpha1": sdot_min, "Sdot_led_negative_steps_all": sdot_neg,
        "engine_q_visc_step_min_J": qv_min, "engine_q_visc_negative_steps": qv_neg,
        "isothermal_T1": {n: {k: r1[n][k] for k in ("Q_led_end_J", "Q_visc_eng_end_J", "Q_press_eng_end_J",
                          "Q_led_over_absU0", "closure_literal_end", "closure_corrected_end", "closure_literal_T",
                          "closure_corrected_T", "closure_engine_vs_led_end", "total_drift_max_over_absU0",
                          "total_drift_end_over_absU0", "engine_booking_split_max_relerr", "S_led_cum_J_per_K_end",
                          "dF_iso_end_over_absU0", "r_median_at_T_over_R")} for n in (600, 1200)},
        "isothermal_T0.3": {n: {k: r1s[n][k] for k in ("closure_literal_end", "closure_corrected_end",
                            "closure_engine_vs_led_end", "Q_led_over_absU0")} for n in (600, 1200)},
        "isothermal_alpha0_T1": {n: {k: r0[n][k] for k in ("bath_at_T", "Q_press_eng_end_J", "dF_iso_end_over_absU0",
                                 "total_drift_max_over_absU0", "total_drift_end_over_absU0")} for n in (600, 1200)},
        "post_hoc": post_hoc},
    "ideal_sentinel": {f"alpha{int(a)}_n{n}": {k: v for k, v in r.items() if k not in ("series", "backward_divergence_R")}
                       for a, d in ((1.0, i1), (0.0, i0)) for n, r in d.items()},
    "fourier": {k: v for k, v in fr.items() if k != "rows"},
    "criteria_original": {"clauses": o, "verdict": verdict_original, "S8_sentinel_literal": s8_orig},
    "criteria_corrected": {"clauses": c, "verdict": verdict_corrected, "failed": failed_c,
                           "separate_if_D3_dropped": sep_without_D3},
}

# ---------------- figure
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#8b5cf6"]
plt.rcParams.update({"axes.unicode_minus": True, "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.titlesize": 10})
fig, axs = plt.subplots(2, 3, figsize=(15, 9), facecolor=SURF)
for ax in axs.flat:
    ax.set_facecolor(SURF); ax.grid(True, color=GRID, lw=0.7); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

# (a) echo error, T = 1 t_ff
ax = axs[0, 0]
cats = [("nbody", "N-body\n(gravity)"), ("sph0", "SPH α = 0\nisothermal"), ("ideal0", "SPH α = 0\nideal gas"),
        ("sph1", "SPH α = 1\nisothermal"), ("ideal1", "SPH α = 1\nideal gas")]
x = np.arange(len(cats))
for j, (n, lab) in enumerate(((600, "dt = t_ff/600"), (1200, "dt = t_ff/1200"))):
    ax.scatter(x + (j - 0.5) * 0.22, [eps[(k, 1.0, n)] for k, _ in cats], s=46, color=COL[j], label=lab, zorder=3)
ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels([c_[1] for c_ in cats], fontsize=8)
ax.axhline(1e-8, color=INK2, ls="--", lw=1); ax.axhline(1e-3, color=INK2, ls=":", lw=1)
ax.text(4.45, 1.6e-8, "1e−8 R", color=INK2, fontsize=8, ha="right")
ax.text(4.45, 1.6e-3, "1e−3 R", color=INK2, fontsize=8, ha="right")
ax.annotate("halves with dt:\nfirst-order T-lag", xy=(2.1, 2.7e-4), xytext=(2.25, 1e-6), fontsize=7.5,
            color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
ax.set_ylim(1e-17, 3); ax.set_ylabel("echo error  rms|x(2T) − x(0)| / R")
ax.set_title("(a) Forward 1 t_ff, reverse every velocity, back 1 t_ff", loc="left")
ax.legend(frameon=False, loc="upper left", fontsize=8)

# (b) delta control
ax = axs[0, 1]
lbl = {"nbody": "N-body", "sph0": "SPH α = 0", "sph1": "SPH α = 1"}
for j, k in enumerate(("nbody", "sph0", "sph1")):
    xs = [1e-15] + list(deltas); ys = [d0[k]] + dctl[k]
    ax.plot(xs, ys, "o-", color=COL[j], label=lbl[k], ms=5, lw=1.3)
dd = np.array([1e-14, 1e-8])
ax.plot(dd, dd, ls="--", color=INK2, lw=1, label="error = δ")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks([1e-15, 1e-13, 1e-11, 1e-9]); ax.set_xticklabels(["0", "1e−13", "1e−11", "1e−9"])
ax.set_xlabel("displacement δ / R injected at the reversal"); ax.set_ylabel("echo error / R")
ax.set_title("(b) Perturbation control: Hamiltonian error ∝ δ, viscous flat", loc="left")
ax.legend(frameon=False, fontsize=8, loc="lower right")

# (c) backward-leg divergence
ax = axs[0, 2]
cur = {"nbody": pick(NB, T_tff=1.0, n_per_tff=1200, delta_R=0.0, N=64)["backward_divergence_R"],
       "sph0": r0[1200]["backward_divergence_R"], "sph1": r1[1200]["backward_divergence_R"]}
tf = nb["choices"]["tff"]
for j, k in enumerate(("nbody", "sph0", "sph1")):
    y = np.maximum(np.array(cur[k]), 1e-18)
    ax.plot(np.arange(1, len(y) + 1) / 1200.0, y, color=COL[j], lw=1.2, label=lbl[k])
ax.set_yscale("log"); ax.set_ylim(1e-18, 3)
ax.set_xlabel("time since reversal / t_ff"); ax.set_ylabel("rms |x_back − x_fwd| / R (same instant)")
ax.set_title("(c) Distance from the forward path on the way back", loc="left")
ax.legend(frameon=False, fontsize=8, loc="center right")

# (d) S_dot across the reversal
ax = axs[1, 0]
for j, (r, lab) in enumerate(((r1[1200], "SPH α = 1 isothermal (Q̇/300 K)"),)):
    s = r["series"]; t = np.array(s["t_s"]) / tf
    ax.plot(t, s["Sdot_led"], color=COL[2], lw=1.2, label=lab)
s = r0[1200]["series"]
ax.plot(np.array(s["t_s"]) / tf, s["Sdot_led"], color=COL[1], lw=1.2, label="SPH α = 0 (identically 0)")
ax.axvline(1.0, color=INK2, ls="--", lw=1); ax.text(1.03, 3.0, "velocities reversed", color=INK2, fontsize=8)
ax.set_xlabel("t / t_ff  (forward 0–1, reversed 1–2)"); ax.set_ylabel("viscous entropy production  Ṡ  (J/K/s)")
ax.set_title("(d) Dissipation does not run backward when v → −v", loc="left"); ax.set_ylim(-6, 150)
ax.legend(frameon=False, fontsize=8, loc="upper left")

# (e) ideal-gas state entropy
ax = axs[1, 1]
for j, (r, lab) in enumerate(((i1[1200], "α = 1: ΔS from (T_i, ρ_i)"), (i0[1200], "α = 0: ΔS from (T_i, ρ_i)"))):
    s = r["series"]; t = np.array(s["t_s"]) / tf
    ax.plot(t, np.array(s["S_state"]) - s["S_state"][0], color=COL[[2, 1][j]], lw=1.3, label=lab)
s = i1[1200]["series"]
ax.plot(np.array(s["t_s"]) / tf, s["dS_visc_eng"], color=COL[3], lw=1.0, ls="--", label="α = 1: Σ q_visc / T booked")
ax.axvline(1.0, color=INK2, ls="--", lw=1)
ax.set_xlabel("t / t_ff  (forward 0–1, reversed 1–2)"); ax.set_ylabel("gas entropy change  ΔS  (J/K)")
ax.set_title("(e) Ideal-gas SPH (eos=\"ideal\"): entropy only from viscosity", loc="left")
ax.legend(frameon=False, fontsize=8, loc="upper left")

# (f) Fourier leg
ax = axs[1, 2]
rows = fr["rows"]; tt = np.array([q["t_s"] for q in rows]); tt = tt / tt[-1]
S_ = np.array([q["S_J_per_K"] for q in rows]); Sd_ = np.array([q["Sdot_W_per_K"] for q in rows])
ax.plot(tt, (S_ - S_[0]) / (S_[-1] - S_[0]), "o-", color=COL[1], ms=4, lw=1.2, label="entropy produced (fraction of final)")
ax.plot(tt, Sd_ / Sd_[0], "s-", color=COL[0], ms=4, lw=1.2, label="Ṡ = Σ k ΔT² / (T_a T_b) dx  (fraction of initial)")
ax.set_ylim(-0.05, 1.08)
ax.set_xlabel("t / t_run  (diffuse_fvm, 32³, insulated)"); ax.set_ylabel("fraction")
ax.set_title("(f) Fourier conduction: Ṡ > 0 every slice, no reverse", loc="left")
ax.legend(frameon=False, fontsize=8, loc="center right")

head = ("Reversing every velocity undoes gravity and inviscid SPH to roundoff "
        f"({eps[('nbody', 1.0, 1200)]:.0e} R, {eps[('sph0', 1.0, 1200)]:.0e} R) but not viscous SPH "
        f"({eps[('sph1', 1.0, 1200)]:.2f} R):\nthe time arrow sits in the dissipative closures (viscosity, conduction), "
        "inside fluid dynamics, not between gravity and fluids")
fig.suptitle(head.replace("e-", "e−"), fontsize=12, color=INK, x=0.01, ha="left", y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.955))
fig_path = os.path.join(HERE, "u2_loschmidt_echo.png")
fig.savefig(fig_path, dpi=150, facecolor=SURF)
results["figure"] = fig_path
with open(os.path.join(HERE, "u2_results.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=1, default=lambda z: str(z))
print(json.dumps({k: results[k] for k in ("eps_rec_R", "dt_halving_ratio", "delta_control", "lyapunov", "smoke_N32",
                                          "criteria_original", "criteria_corrected")}, indent=1, default=str))
print(json.dumps(results["ledger"], indent=1, default=str))
print(json.dumps(results["ideal_sentinel"], indent=1, default=str)[:6000])
print(json.dumps(results["fourier"], indent=1, default=str))
