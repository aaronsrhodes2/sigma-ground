"""U5 aggregate: ratios vs pre-registered criteria, JSON of results with every Mentat source, one figure.

usage (repo root as cwd): python -X utf8 u5_aggregate.py <outdir>
"""
import sys, os, json, math
import numpy as np
import sympy
sys.path.insert(0, r"D:/Aaron/development/sigma-ground/tools")
from mentat_client import mentat
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = sys.argv[1]
L = json.load(open(os.path.join(D, "leaves.json")))


def num(k):
    v = L[k]["value"]
    return float(sympy.sympify(v).evalf(17)) if isinstance(v, str) else float(v)


dwGR = num("dw_GR_rad_per_orbit")
dwS_pred = num("dw_sigma_rad_per_orbit_pred")
ARC = num("rad_to_arcsec")
NCY = num("orbits_per_century")
XI3, XI6 = num("xi_over_3"), num("xi_over_6")


def load(mode, dt):
    return json.load(open(os.path.join(D, "runs", f"run_{mode}_dt{dt}_n50.json")))


R = {(m, dt): load(m, dt) for m, dt in [("A", 600), ("B", 600), ("C", 600), ("D", 600), ("E", 600),
                                         ("A", 300), ("C", 300)]}


def diff(m, dt, est):
    s, e = R[(m, dt)][est][0], R[(m, dt)][est][1]
    s0, e0 = R[("A", dt)][est][0], R[("A", dt)][est][1]
    return s - s0, math.hypot(e, e0)


out = {"id": "U5", "prereg_file": os.path.join(D, "PREREG.txt"), "leaves": L, "estimators": {}}
for est in ("lrl", "sketch", "interp"):
    E = {}
    dB, eB = diff("B", 600, est)
    dC, eC = diff("C", 600, est)
    dD, eD = diff("D", 600, est)
    dE, eE = diff("E", 600, est)
    dC3, eC3 = diff("C", 300, est)
    E["A_newtonian_rad_per_orbit_dt600"] = R[("A", 600)][est][0]
    E["A_newtonian_rad_per_orbit_dt300"] = R[("A", 300)][est][0]
    for k, (d, e) in {"B_minus_A": (dB, eB), "C_minus_A": (dC, eC), "D_minus_A": (dD, eD),
                      "E_minus_A": (dE, eE), "C_minus_A_dt300": (dC3, eC3)}.items():
        E[k] = {"rad_per_orbit": d, "fit_se": e, "arcsec_per_cy": d * ARC * NCY, "over_GR_analytic": d / dwGR}
    E["design_ratio_C_over_B"] = dC / dB
    E["design_ratio_se"] = abs(dC / dB) * math.hypot(eC / dC, eB / dB)
    E["dt_control_rel_change"] = (dC3 - dC) / dC
    out["estimators"][est] = E

P = out["estimators"]["lrl"]
ratio_design = P["design_ratio_C_over_B"]
ratio_corr = P["C_minus_A"]["over_GR_analytic"]
sig_arc = P["C_minus_A"]["arcsec_per_cy"]
out["criteria"] = {
    "design_engine_pass": bool(0.0497 <= ratio_design <= 0.0557),
    "design_ratio": ratio_design,
    "corrected_engine_pass_C": bool(0.0497 <= ratio_corr <= 0.0557),
    "corrected_ratio_C": ratio_corr,
    "B_over_GR_analytic": P["B_minus_A"]["over_GR_analytic"],
    "B_magnitude_within_1pct": bool(0.99 <= abs(P["B_minus_A"]["over_GR_analytic"]) <= 1.01),
    "B_sign": "retrograde (negative)" if P["B_minus_A"]["rad_per_orbit"] < 0 else "prograde (positive)",
    "P2_D_ratio": P["D_minus_A"]["over_GR_analytic"],
    "P2_D_pass": bool(abs(P["D_minus_A"]["over_GR_analytic"] - XI6) <= 0.0015),
    "P3_E_ratio": P["E_minus_A"]["over_GR_analytic"],
    "P3_E_pass": bool(abs(P["E_minus_A"]["over_GR_analytic"] + 5.0 / 3.0) <= 0.03),
    "P4_dt_control_rel_change": P["dt_control_rel_change"],
    "P4_dt_control_pass": bool(abs(P["dt_control_rel_change"]) < 0.01),
    "physics_threshold_arcsec_cy": 0.05,
    "sigma_advance_arcsec_cy": sig_arc,
    "physics_excluded": bool(sig_arc > 0.05),
}
# exclusion factors evaluated by Mentat (simplify_expr)
v, s = mentat("simplify_expr", {"expression": f"{sig_arc}/0.05"})
out["criteria"]["excess_over_design_threshold"] = {"value": float(sympy.sympify(v)), "source": s}
v2, s2 = mentat("simplify_expr", {"expression": f"{ratio_corr}/(3.9e-5/3)"})
out["criteria"]["excess_over_Park2017_beta_1sigma"] = {
    "value": float(sympy.sympify(v2)), "source": s2,
    "note": "[DERIVED] the sigma hook is an extra attractive 1/r^3 force, i.e. a PPN beta shift of -xi; compared with "
            "the 1-sigma of (2+2gamma-beta)/3 implied by Park+2017 beta-1 = (-2.7 +/- 3.9)e-5 [LIT]"}
v3, s3 = mentat("simplify_expr", {"expression": f"{dwGR}*{ARC}*{NCY}"})
out["GR_analytic_arcsec_cy"] = {"value": float(sympy.sympify(v3)), "source": s3}
out["probes"] = {m: R[(m, 600)]["probe"] for m in ("B", "C", "D", "E")}
out["anomalistic_period_s"] = {f"{m}_{dt}": R[(m, dt)]["anomalistic_period_s"] for (m, dt) in R}
out["wall_s"] = {f"{m}_{dt}": R[(m, dt)]["wall_s"] for (m, dt) in R}
out["n_steps"] = {f"{m}_{dt}": R[(m, dt)]["n_steps"] for (m, dt) in R}
out["fit_residual_rms_rad"] = {f"{m}_{dt}": R[(m, dt)]["lrl"][2] for (m, dt) in R}
json.dump(out, open(os.path.join(D, "u5_results.json"), "w"), indent=1)

# ---------------- figure ----------------
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#8b5cf6"
plt.rcParams.update({"axes.unicode_minus": True, "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2), facecolor=SURF)


def cum(m, dt=600):
    y = np.unwrap(np.array(R[(m, dt)]["peri_lrl"])) - np.unwrap(np.array(R[("A", dt)]["peri_lrl"]))
    return np.arange(1, len(y) + 1), (y - y[0]) * ARC   # arcsec, relative to first perihelion


k = np.arange(1, 51)
M = lambda t: t.replace('-', '−')   # true minus signs in figure text
for ax in (ax1, ax2):
    ax.set_facecolor(SURF)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

x, y = cum("C")
ax1.plot(x, y, "o", ms=4, color=C1,
         label=f"C: σ = ξ r_s/r (field equation), {P['C_minus_A']['arcsec_per_cy']:.3f}\u2033/cy")
ax1.plot(k, (k - 1) * dwS_pred * ARC, "-", color=C1, lw=1, alpha=0.7, label="ξ/3 of GR (prediction)")
x, y = cum("D")
ax1.plot(x, y, "s", ms=4, color=C2,
         label=f"D: σ = ξ r_s/2r (sigma_from_potential), {P['D_minus_A']['arcsec_per_cy']:.3f}\u2033/cy")
ax1.plot(k, (k - 1) * dwS_pred / 2 * ARC, "-", color=C2, lw=1, alpha=0.7, label="ξ/6 of GR (prediction)")
ax1.plot(k, (k - 1) * 0.05 / NCY, "--", color=INK2, lw=1.2,
         label="largest extra advance Mercury allows (0.05\u2033/cy)")
ax1.set_xlabel("orbit number")
ax1.set_ylabel("perihelion shift vs Newtonian run (arcsec)")
ax1.set_title("(a) the σ hook: points measured, lines predicted", fontsize=10, color=INK, loc="left")
ax1.legend(frameon=False, fontsize=8.3, loc="upper left")

x, y = cum("B")
ax2.plot(x, y, "o", ms=4, color=C3, label=M(f"B: engine gr_1pn, {P['B_minus_A']['arcsec_per_cy']:.2f}\u2033/cy"))
x, y = cum("E")
ax2.plot(x, y, "^", ms=4, color=C4, label=M(f"E: engine eih_cross, {P['E_minus_A']['arcsec_per_cy']:.2f}\u2033/cy"))
ax2.plot(k, (k - 1) * dwGR * ARC, "-", color=INK, lw=1.5,
         label=f"GR 6πGM/c²a(1\u2212e²): +{out['GR_analytic_arcsec_cy']['value']:.2f}\u2033/cy (what Mercury does)")
ax2.plot(k, -(k - 1) * dwGR * ARC, ":", color=C3, lw=1.4, label="\u22121 × GR (predicted for gr_1pn from the code)")
ax2.plot(k, -(k - 1) * dwGR * ARC * 5 / 3, ":", color=C4, lw=1.4,
         label="\u22125/3 × GR (predicted for eih_cross from the code)")
ax2.set_xlabel("orbit number")
ax2.set_ylabel("perihelion shift vs Newtonian run (arcsec)")
ax2.set_title("(b) the engine's own 1PN terms turn the perihelion backwards", fontsize=10, color=INK, loc="left")
ax2.legend(frameon=False, fontsize=8.3, loc="lower left")

fig.suptitle(f"The nbody σ hook advances Mercury's perihelion by {ratio_corr:.4f} of GR (ξ/3 = {XI3:.4f}): "
             f"{sig_arc:.2f}\u2033/cy, {sig_arc / 0.05:.0f}× the 0.05\u2033/cy Mercury allows, so the hook is excluded.\n"
             "Against the engine's own GR run the ratio is " + M(f"{ratio_design:+.4f}") + ", because that 1PN term has the wrong sign.",
             fontsize=11, color=INK, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(os.path.join(D, "u5_mercury_sigma_hook.png"), dpi=150, facecolor=SURF)

summary = {"criteria": out["criteria"], "GR_analytic": out["GR_analytic_arcsec_cy"],
           "lrl": {k2: (v if not isinstance(v, dict) else {kk: v[kk] for kk in ("arcsec_per_cy", "over_GR_analytic", "fit_se")})
                   for k2, v in P.items()},
           "sketch": {"design_ratio": out["estimators"]["sketch"]["design_ratio_C_over_B"],
                      "C_over_GR": out["estimators"]["sketch"]["C_minus_A"]["over_GR_analytic"],
                      "B_over_GR": out["estimators"]["sketch"]["B_minus_A"]["over_GR_analytic"]},
           "interp": {"design_ratio": out["estimators"]["interp"]["design_ratio_C_over_B"],
                      "C_over_GR": out["estimators"]["interp"]["C_minus_A"]["over_GR_analytic"],
                      "B_over_GR": out["estimators"]["interp"]["B_minus_A"]["over_GR_analytic"]},
           "anomalistic_period_s": out["anomalistic_period_s"], "residual_rms": out["fit_residual_rms_rad"]}
print(json.dumps(summary, indent=1))
