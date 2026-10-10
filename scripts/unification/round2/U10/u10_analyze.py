"""U10 analysis: scores the pre-registered criteria (u10_prereg.txt) on u10_n{32,40,48}.json, fits the convergence orders,
writes u10_results.json and u10_figure.png. Analytic interior profiles at off-centre radii are in-script quadrature of the
dimensionless n = 1 shapes (scipy.integrate.quad), cross-checked against the Mentat closed forms at r -> 0."""
import json, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from scipy.integrate import quad

OUT = os.path.dirname(os.path.abspath(__file__))
LV = json.load(open(os.path.join(OUT, "u10_leaves.json")))
PRED = {f"{p['w']:g}": p for p in LV["predictions"]}
NS = [32, 40, 48]
RUN = {n: json.load(open(os.path.join(OUT, f"u10_n{n}.json"))) for n in NS}
EXTRA = {}
if os.path.exists(os.path.join(OUT, "u10_n24.json")):
    EXTRA[24] = json.load(open(os.path.join(OUT, "u10_n24.json")))
WS = ["0.001", "0.01"]
PI = math.pi


# ---------- analytic potentials of the dimensionless sources (units: a = 1, rho_c = 1) ----------
def th(x):
    return math.sin(x) / x if x > 1e-12 else 1.0


def src(kind, w):
    mf = (lambda x: 1 + (4 * w / 3) * (1 + th(x))) if kind.endswith("PN") else (lambda x: 1.0)
    sgn = +1 if kind.startswith("T") else -1
    return lambda x: mf(x) * (th(x) + sgn * w * th(x) ** 2)


def pot(s, x):
    """v(x) = (1/x) int_0^x s x'^2 dx' + int_x^pi s x' dx'  (solves lap v = -s, v -> Q/4 pi r outside)."""
    inner = quad(lambda y: s(y) * y * y, 0, x, epsabs=0, epsrel=1e-13, limit=200)[0] / x if x > 0 else 0.0
    outer = quad(lambda y: s(y) * y, x, PI, epsabs=0, epsrel=1e-13, limit=200)[0]
    return inner + outer


def deficit_an(w, x, pn=True):
    k = "PN" if pn else ""
    return 1 - pot(src("S" + k, w), x) / pot(src("T" + k, w), x)


# sanity: quadrature at x -> 0 vs Mentat closed forms
quad_check = {}
for wk in WS:
    w = float(wk)
    quad_check[wk] = dict(quad_deficit_over_w_r0=deficit_an(w, 1e-9) / w, mentat_exact=PRED[wk]["deficit_over_w_weighted_centre_exact"])


def fit_order(dxs, errs):
    dxs, errs = np.asarray(dxs, float), np.abs(np.asarray(errs, float))
    p, lnC = np.polyfit(np.log(dxs), np.log(errs), 1)
    pair = [float(np.log(errs[i] / errs[i + 1]) / np.log(dxs[i] / dxs[i + 1])) for i in range(len(dxs) - 1)]
    return dict(order_lsq=float(p), pairwise=pair, C=float(np.exp(lnC)))


# ---------- per-grid scoring ----------
grid = {}
for n, d in {**EXTRA, **RUN}.items():
    a_cells = d["a_cells"]; x_in = d["r_inner_m"] / d["a_m"]
    g = dict(n=n, dx_km=d["dx_m"] / 1e3, a_cells=a_cells, T_leg_design_max=d["T_leg_design_err"]["max_abs_rel_to_peak"],
             T_leg_heat_max=d["T_leg_heat_err"]["max_abs_rel_to_peak"], T_leg_design_centre=d["T_leg_design_err"]["centre_signed_rel"],
             T_leg_heat_centre=d["T_leg_heat_err"]["centre_signed_rel"], N_flux_over_quad_minus_1=d["N_flux_over_quad_minus_1"],
             heat_flux_over_quad_minus_1=d["T_leg_heat"]["Q_flux_over_quad_minus_1"],
             split_measured_over_predicted=d["split_check"]["centre_diff_measured"] / d["split_check"]["centre_diff_predicted"],
             split_bias_over_v_centre=d["split_check"]["centre_diff_over_v_centre"], w={})
    # transient residual of the n_tau = 8 design solves, read off the heat-law run's own centre history (same box mode)
    H = np.array(d["T_leg_heat"]["centre_history_t_over_tau"]); Tfin = d["T_leg_heat"]["centre_T_K"]; Tinf = d["T_leg_heat"]["T_inf_K"]
    eps8 = float((Tfin - np.interp(8.0, H[:, 0], H[:, 1])) / (Tfin - Tinf))
    v_c = (Tfin - Tinf) / d["T_leg_heat"]["q0_W_m3"]
    g["transient_residual_at_8tau_frac"] = eps8
    g["split_measured_over_predicted_transient_corrected"] = d["split_check"]["centre_diff_measured"] / (d["split_check"]["centre_diff_predicted"] - eps8 * v_c)
    for wk in WS:
        w = float(wk); row = d["w"][wk]; p = PRED[wk]
        dfa = deficit_an(w, x_in, True); dfa_u = deficit_an(w, x_in, False)
        g["w"][wk] = dict(
            C1_dev=row["Q_S_over_Q_T"] - p["ratio_pressure_only"],
            C1_dev_vs_exact_weighted=row["Q_S_over_Q_T"] - p["ratio_weighted_exact"],
            C1_dev_flux=[x - p["ratio_pressure_only"] for x in row["Q_S_over_Q_T_flux"]],
            C2_literal_rel=(row["dQ_over_QN"] - w) / w,
            C2c_weighted_rel=(row["dQ_over_QN"] - p["dQ_over_QN_weighted_exact"]) / p["dQ_over_QN_weighted_exact"],
            C2c_unweighted_rel=(row["unweighted_dQ_over_QN"] - w) / w,
            C2_flux_literal_rel=[(x - w) / w for x in row["dQ_over_QN_flux"]],
            QT_over_QN=row["QT_over_QN"], QT_over_QN_exact=p["QT_over_QN_weighted_exact"], QT_over_QN_design_1p2_5w=1 + 2.5 * w,
            QT_over_QN_solver_err_rel_to_2w=(row["QT_over_QN"] - p["QT_over_QN_weighted_exact"]) / (2 * w),
            C3_deficit_over_w=row["deficit_over_w_weighted"], deficit_over_w_analytic_at_r_inner=dfa / w,
            deficit_over_w_analytic_r0=p["deficit_over_w_weighted_centre_exact"],
            deficit_err=row["deficit_over_w_weighted"] - dfa / w,
            deficit_unweighted_over_w=row["deficit_over_w_unweighted"], deficit_unweighted_analytic_at_r_inner=dfa_u / w,
            sigma_centre=row["sigma_centre"], sigma_centre_if_sourced_like_Phi_T=row["sigma_centre_if_sourced_like_Phi_T"],
            rs_over_R=row["rs_over_R"])
    grid[n] = g

dxs = [grid[n]["dx_km"] for n in NS]
orders = dict(
    T_leg_design_max=fit_order(dxs, [grid[n]["T_leg_design_max"] for n in NS]),
    T_leg_heat_max=fit_order(dxs, [grid[n]["T_leg_heat_max"] for n in NS]),
    deficit_err_w1e_2=fit_order(dxs, [grid[n]["w"]["0.01"]["deficit_err"] for n in NS]),
    deficit_err_w1e_3=fit_order(dxs, [grid[n]["w"]["0.001"]["deficit_err"] for n in NS]),
    QT_over_QN_solver_err_w1e_2=fit_order(dxs, [grid[n]["w"]["0.01"]["QT_over_QN_solver_err_rel_to_2w"] for n in NS]))

g40 = grid[40]
crit = {
    "C1 |Q_s/Q_T - (1-w/2)/(1+w/2)| < 1e-6 (w=1e-3)": dict(value=abs(g40["w"]["0.001"]["C1_dev"]), threshold=1e-6),
    "C1 |Q_s/Q_T - (1-w/2)/(1+w/2)| < 1e-4 (w=1e-2)": dict(value=abs(g40["w"]["0.01"]["C1_dev"]), threshold=1e-4),
    "C2 literal |(Q_T-Q_s)/Q_N - w|/w < 1e-3 (w=1e-3)": dict(value=abs(g40["w"]["0.001"]["C2_literal_rel"]), threshold=1e-3),
    "C2 literal |(Q_T-Q_s)/Q_N - w|/w < 1e-3 (w=1e-2)": dict(value=abs(g40["w"]["0.01"]["C2_literal_rel"]), threshold=1e-3),
    "C2' weighted vs 2B/I1, rel < 1e-3 (w=1e-3)": dict(value=abs(g40["w"]["0.001"]["C2c_weighted_rel"]), threshold=1e-3),
    "C2' weighted vs 2B/I1, rel < 1e-3 (w=1e-2)": dict(value=abs(g40["w"]["0.01"]["C2c_weighted_rel"]), threshold=1e-3),
    "C2' unweighted vs w, rel < 1e-3 (w=1e-3)": dict(value=abs(g40["w"]["0.001"]["C2c_unweighted_rel"]), threshold=1e-3),
    "C2' unweighted vs w, rel < 1e-3 (w=1e-2)": dict(value=abs(g40["w"]["0.01"]["C2c_unweighted_rel"]), threshold=1e-3),
    "C3 |deficit/w - 1.219| < 0.02 (w=1e-3)": dict(value=abs(g40["w"]["0.001"]["C3_deficit_over_w"] - 1.219), threshold=0.02),
    "C3 |deficit/w - 1.219| < 0.02 (w=1e-2)": dict(value=abs(g40["w"]["0.01"]["C3_deficit_over_w"] - 1.219), threshold=0.02),
    "C4 T leg max <= 1 % (design-literal, 40^3)": dict(value=g40["T_leg_design_max"], threshold=0.01),
    "C4 T leg max <= 1 % (heat law 1x kick, 40^3)": dict(value=g40["T_leg_heat_max"], threshold=0.01),
    "C5 order >= 1.2 (primary: design-literal T leg)": dict(value=orders["T_leg_design_max"]["order_lsq"], threshold=1.2, kind="min"),
}
for k, c in crit.items():
    c["pass"] = bool(c["value"] >= c["threshold"]) if c.get("kind") == "min" else bool(c["value"] < c["threshold"] or (c["value"] <= c["threshold"] and "<=" in k))
literal_keys = [k for k in crit if not k.startswith("C2'")]
corrected_keys = [k for k in crit if not k.startswith("C2 literal")]
summary = dict(all_literal_pass=all(crit[k]["pass"] for k in literal_keys), all_corrected_pass=all(crit[k]["pass"] for k in corrected_keys),
               literal_failures=[k for k in literal_keys if not crit[k]["pass"]], corrected_failures=[k for k in corrected_keys if not crit[k]["pass"]])

results = dict(experiment="U10", grids=grid, orders=orders, criteria_at_40=crit, summary=summary, quad_check=quad_check,
               mentat_leaves={k: v for k, v in LV.items() if isinstance(v, dict) and "source" in v},
               engine_leaves=dict(XI=LV["XI"]), engine_constants_equal_mentat=LV["engine_constants_equal_mentat"],
               predictions=LV["predictions"], timing_s={n: RUN[n]["timing_s"] for n in NS},
               bitwise_T_leg_equals_PhiN={n: RUN[n]["T_leg_design_identical_to_N_bitwise"] for n in NS})
results["reading"] = dict(
    outcome="MIXED: all pre-registered criteria pass except C2 as sketched, which fails exactly as pre-registered (design error); all corrected criteria pass",
    tags={"Q ratios (C1, C2)": "[BY CONSTRUCTION] source quadrature; only the O(w^2) metric-weight term touches the solver",
          "T leg = Phi_N (design call)": "[BY CONSTRUCTION] bitwise identical solve",
          "heat-law T leg vs analytic": "[VERIFIED numerics] 0.094 % max at 40^3, order 2.06",
          "central deficit/w": "[VERIFIED numerics] vs [DERIVED] exact 1PN pair (Mentat integrals)",
          "split-kick bias (dt/2)s": "[DERIVED] and [MEASURED] to 0.5 % after transient correction",
          "opposite-sign pressure coupling": "[DERIVED] put in by hand in the sources; zero evidential weight"},
    prereg_file="u10_prereg.txt")
json.dump(results, open(os.path.join(OUT, "u10_results.json"), "w"), indent=1)

# ---------------- figure ----------------
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
S = ["#2a78d6", "#eb6834", "#1baf7a", "#8b5cf6"]
plt.rcParams.update({"axes.unicode_minus": True, "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK})
F = np.load(os.path.join(OUT, "u10_fields_n40.npz"))
r, wall, theta = F["r"], F["wall"], F["theta"]
d40 = RUN[40]; a, R, dx = d40["a_m"], d40["R_m"], d40["dx_m"]
v_an = np.where(r < R, a * a * (1 + theta), PI * a**3 / np.maximum(r, 1e-12)); peak = 2 * a * a
m = ~wall
bins = np.linspace(0, r[m].max(), 34); idx = np.digitize(r[m], bins); rc = 0.5 * (bins[1:] + bins[:-1])
def binned(f):
    return np.array([np.mean(f[idx == i]) if np.any(idx == i) else np.nan for i in range(1, len(bins))])
fig, axs = plt.subplots(2, 2, figsize=(13, 9.4), dpi=170); fig.patch.set_facecolor(SURF)
for ax in axs.flat:
    ax.set_facecolor(SURF); ax.grid(color=GRID, lw=0.8)
# (a) T-leg residuals at 40^3
ax = axs[0, 0]
ax.plot(rc / R, 100 * binned(((F["v_N"] - v_an) / peak)[m]), color=S[0], lw=2, label="design-literal T leg (poisson_fvm, kick every 20 stable_dt)")
ax.plot(rc / R, 100 * binned(((F["v_heat"] - v_an) / peak)[m]), color=S[1], lw=2, label="heat-law T leg, kick every stable_dt, 16 τ")
dt20 = 20 * 0.2 * dx * dx / 6.0
ax.plot(rc / R, 100 * binned((dt20 / 2 * theta / peak)[m]), color=S[2], lw=1.5, ls="--", label="predicted split-kick bias (dt/2)·s  [derived]")
ax.axhline(1.0, color=INK2, lw=1, ls=":"); ax.text(1.55, 1.02, "1 % criterion (C4)", color=INK2, fontsize=8, va="bottom"); ax.set_ylim(-0.08, 1.2)
ax.axvline(1.0, color=GRID, lw=1.2)
ax.set_xlabel("r / R"); ax.set_ylabel("(T − T_analytic) / peak rise  (%)")
ax.set_title("(a) steady T leg vs the analytic n = 1 potential, 40³: the 0.7 % offset\nis the pilot solver's split-kick bias (dt/2)·s, not the heat law", fontsize=10)
ax.legend(frameon=False, fontsize=7.5, loc="center right")
# (b) deficit profile 1 - Psi_s/Phi_T over w
ax = axs[0, 1]
xs_an = np.linspace(0.02, 1.25, 60)
for j, wk in enumerate(WS):
    w = float(wk)
    num = (1 - F[f"v_S_{wk}"] / F[f"v_T_{wk}"]) / w
    ax.plot(rc / R, binned(num[m]), color=S[j], lw=0, marker="o", ms=3.5, label=f"engine, w_c = {w:g}")
    ax.plot(xs_an, [deficit_an(w, x * PI) / w if x < 1 else (1 - (pot(src('SPN', w), PI) * PI) / (pot(src('TPN', w), PI) * PI)) / w for x in xs_an],
            color=S[j], lw=1.3, label=f"exact 1PN pair, w_c = {w:g}")
ax.axhspan(1.219 - 0.02, 1.219 + 0.02, xmin=0, xmax=0.06, color=S[3], alpha=0.35, lw=0)
ax.text(0.08, 1.229, "design band 1.219 ± 0.02 at the centre", color=S[3], fontsize=8)
ax.axvline(1.0, color=GRID, lw=1.2)
ax.set_xlabel("r / R"); ax.set_ylabel("(1 − Ψ_σ / Φ_Tolman) / w_c")
ax.set_title("(b) pressure de-sources σ as much as it sources Φ:\nσ below the Tolman potential by 1.22·w_c at the centre, ≈ w_c outside", fontsize=10)
ax.legend(frameon=False, fontsize=7.5, loc="lower left")
# (c) convergence
ax = axs[1, 0]
dxa = np.array(dxs)
series = [("design-literal T leg, max error", [grid[n]["T_leg_design_max"] for n in NS], orders["T_leg_design_max"]),
          ("heat-law T leg, max error", [grid[n]["T_leg_heat_max"] for n in NS], orders["T_leg_heat_max"]),
          ("|central deficit/w − exact|, w_c = 0.01", [abs(grid[n]["w"]["0.01"]["deficit_err"]) for n in NS], orders["deficit_err_w1e_2"]),
          ("Q_T/Q_N solver error ÷ 2w, w_c = 0.01", [abs(grid[n]["w"]["0.01"]["QT_over_QN_solver_err_rel_to_2w"]) for n in NS], orders["QT_over_QN_solver_err_w1e_2"])]
for j, (lab, e, o) in enumerate(series):
    ax.loglog(dxa, e, color=S[j], marker="o", lw=1.6, label=f"{lab}: order {o['order_lsq']:.2f}")
ref = np.array([dxa.min(), dxa.max()])
ax.loglog(ref, series[0][1][1] * (ref / 1.0) ** 2, color=INK2, lw=1, ls=":", label="slope 2 reference")
ax.set_xticks(dxs); ax.set_xticklabels([f"{x:.3g}" for x in dxs]); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xlabel("dx (km)   [n = 48, 40, 32 in a fixed 40 km box, a = 5 km]"); ax.set_ylabel("error (fraction)")
ax.set_title("(c) convergence sentinel: every measured error falls as dx^2\n(design floor: order ≥ 1.2)", fontsize=10)
ax.legend(frameon=False, fontsize=7.5, loc="lower right")
# (d) criteria scorecard at 40^3
ax = axs[1, 1]
labels, ratios, cols = [], [], []
for k, c in crit.items():
    if c.get("kind") == "min":
        rr = c["threshold"] / c["value"]
    else:
        rr = c["value"] / c["threshold"]
    kk_ = k.replace("-", "−").replace("design−literal", "design-literal")
    labels.append(kk_.replace(" (", "\n(", 1) if len(kk_) > 46 else kk_); ratios.append(max(rr, 1e-6))
    cols.append(S[1] if k.startswith("C2 literal") else (S[0] if c["pass"] else S[3]))
yy = np.arange(len(labels))[::-1]
ax.barh(yy, ratios, color=cols, height=0.62)
ax.set_xscale("log"); ax.axvline(1.0, color=INK, lw=1.2)
ax.set_yticks(yy); ax.set_yticklabels(labels, fontsize=6.3)
ax.set_xlabel("measured ÷ allowed  (left of 1 = pass; C5 plotted as 1.2 ÷ order)")
ax.set_title("(d) pre-registered criteria at 40³: all pass except C2 as literally sketched,\nwhose prediction w_c omits the metric weight on the pressure term", fontsize=10)
from matplotlib.patches import Patch
ax.set_axisbelow(True)
ax.legend(handles=[Patch(color=S[0], label="pass"), Patch(color=S[1], label="C2 as sketched (design error, predicted to fail)")], frameon=False, fontsize=7.5, loc="lower right")
fig.suptitle("U10 — Φ_Tolman, σ and steady T from one polytrope on the repo's FVM heat law: one operator, second-order convergent;\n"
             "pressure enters Φ and σ with equal and opposite weight by construction — engine validation, zero evidential weight",
             fontsize=10.5, color=INK, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.965))
fig.savefig(os.path.join(OUT, "u10_figure.png"), facecolor=SURF)
print(json.dumps(dict(criteria=crit, summary=summary, orders=orders, quad_check=quad_check), indent=1))
for n in sorted(grid):
    g = grid[n]
    print(n, {k: g[k] for k in ("dx_km", "T_leg_design_max", "T_leg_heat_max", "T_leg_design_centre", "T_leg_heat_centre",
                                "N_flux_over_quad_minus_1", "heat_flux_over_quad_minus_1", "split_measured_over_predicted",
                                "transient_residual_at_8tau_frac", "split_measured_over_predicted_transient_corrected", "split_bias_over_v_centre")})
    for wk in WS:
        print("   ", wk, json.dumps(g["w"][wk]))
