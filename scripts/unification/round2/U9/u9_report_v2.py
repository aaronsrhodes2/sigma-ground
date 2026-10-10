"""U9 repair (v2) — aggregate the original parts A/B/C plus the repair runs, apply the pre-registered criteria
(unchanged), write u9_results_v2.json and u9_figure_v2.png.

Changes from u9_report.py (the original is kept unchanged):
  * Newtonian baselines at dt = 1200 s and 300 s are MEASURED (newton_v2_a/b.json, same script u9_orbits.py), not
    assumed as 16x / 1/16x the 600 s value.  Each 1PN run is referenced to the Newtonian run with the same dt and the
    same frame when that run exists (rest and x), otherwise to the rest-frame run with the same dt.
  * Engine-vs-Gauss deviations are listed run by run (the original 'max <= 5e-7' statement was wrong: <= 3e-6).
  * The independent DOP853 check and the reference laws (u9_dop853_v2.json) and the anchor/Sun-mass check
    (u9_anchor_v2.json) are folded in.
  * Pre-registration bookkeeping: file mtimes, the missed SPH magnitude prediction, and which Gauss predictions were
    computed after the first orbit runs (post hoc).
  * Figure: panel (a) says it cannot test 1PN correctness; panel (b) uses dt -> 0 (Richardson) values for the
    x-boost and adds the two reference laws; the title separates the gr_1pn and eih_cross frame results.
Usage: python -X utf8 u9_report_v2.py <folder>
"""
import datetime, glob, json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = sys.argv[1]
J = lambda n: json.load(open(os.path.join(D, n), encoding="utf-8"))
inst, gauss, dop, anc = J("u9_instant.json"), J("u9_gauss.json"), J("u9_dop853_v2.json"), J("u9_anchor_v2.json")
R, files = {}, {}
for fn in sorted(glob.glob(os.path.join(D, "orbits_p*.json"))) + sorted(glob.glob(os.path.join(D, "newton_v2_*.json"))):
    o = json.load(open(fn, encoding="utf-8"))
    R.update(o["runs"]); S = o["sources"]; A, E = o["a_m"], o["e"]
    files[os.path.basename(fn)] = {"runs": list(o["runs"]),
                                   "mtime": datetime.datetime.fromtimestamp(os.path.getmtime(fn)).strftime("%H:%M:%S")}
for fn in ("u9_instant.py", "u9_orbits.py", "u9_gauss.py", "u9_gauss.json", "u9_dop853_v2.py", "u9_dop853_v2.json"):
    files[fn] = {"mtime": datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(D, fn))).strftime("%H:%M:%S")}
G, C = S["G"]["value"], S["c"]["value"]
MU = G * (S["sun"]["value"]["mass_kg"] + S["mercury"]["value"]["mass_kg"])
p = A * (1 - E * E)
u = math.pi * MU / (C * C * p)                    # rad per orbit
ARCSEC = 180 * 3600 / math.pi                      # rad -> arcsec (pure math)
CENTURY_S = 36525 * 86400.0                        # Julian century (definition)
LIT_GR = 42.98                                     # [LIT] Will 2014, Living Rev. Relativ. 17, 4, eq. (65)
per_cy = lambda rad, P: rad * ARCSEC * CENTURY_S / P


def tag(dt):
    return "" if dt == 600.0 else f"_dt{int(dt)}"


def baseline(frame, dt):
    k = f"newton_{frame}{tag(dt)}"
    if k in R:
        return R[k]["domega_per_orbit_rad"], R[k]["de_per_orbit"], k
    k = f"newton_rest{tag(dt)}"
    return R[k]["domega_per_orbit_rad"], R[k]["de_per_orbit"], k


prec = {}
for nm, r in R.items():
    if r["physics"] == "newton":
        continue
    b, be, bk = baseline(r["frame"], r["dt_s"])
    d = r["domega_per_orbit_rad"] - b
    law = "gr_engine" if nm.startswith("gr") else "eih_engine"
    pu = gauss["pred_over_u"][f"{law}_{r['frame']}"]
    prec[nm] = {"baseline_run": bk, "rad_per_orbit": d, "over_u": d / u, "arcsec_per_cy": per_cy(d, r["period_s"]),
                "de_over_u": (r["de_per_orbit"] - be) / u, "period_s": r["period_s"], "dt_s": r["dt_s"],
                "predicted_over_u": pu, "predicted_arcsec_per_cy": per_cy(pu * u, r["period_s"]),
                "meas_over_pred_minus_1": (d / u) / pu - 1,
                "de_predicted_over_u": gauss["de_pred_over_u"][f"{law}_{r['frame']}"]}
# baseline change vs the original report (assumed dt^4 scaling)
base_note = {"newton_rest_dt1200_measured": R["newton_rest_dt1200"]["domega_per_orbit_rad"],
             "newton_rest_dt1200_assumed_v1": 16 * R["newton_rest"]["domega_per_orbit_rad"],
             "newton_rest_dt300_measured": R["newton_rest_dt300"]["domega_per_orbit_rad"],
             "newton_rest_dt300_assumed_v1": R["newton_rest"]["domega_per_orbit_rad"] / 16,
             "newton_x_dt1200_measured": R["newton_x_dt1200"]["domega_per_orbit_rad"],
             "newton_x_dt300_measured": R["newton_x_dt300"]["domega_per_orbit_rad"],
             "effect_on_gr_rest_dt1200_over_u": (16 * R["newton_rest"]["domega_per_orbit_rad"]
                                                 - R["newton_rest_dt1200"]["domega_per_orbit_rad"]) / u / 6.0}

# ── Richardson (first order: Forest-Ruth with velocity-dependent kicks converges at O(dt)) ──
rich = {}
for lo_, hi_ in (("gr_x_dt300", "gr_x"), ("eih_x", "eih_x_dt1200")):
    a, b = prec[lo_], prec[hi_]
    nm = lo_.split("_dt")[0]
    w0 = 2 * a["over_u"] - b["over_u"]; e0 = 2 * a["de_over_u"] - b["de_over_u"]
    rich[nm] = {"pair": [lo_, hi_], "domega_over_u_dt0": w0, "de_over_u_dt0": e0,
                "domega_pred_over_u": prec[nm]["predicted_over_u"], "de_pred_over_u": prec[nm]["de_predicted_over_u"],
                "domega_dev": w0 / prec[nm]["predicted_over_u"] - 1, "de_abs_dev_over_u": e0 - prec[nm]["de_predicted_over_u"],
                "arcsec_per_cy_dt0": per_cy(w0 * u, prec[nm]["period_s"])}
gx = [prec[k]["rad_per_orbit"] for k in ("gr_x_dt1200", "gr_x", "gr_x_dt300")]
conv_ratio = (gx[0] - gx[1]) / (gx[1] - gx[2])

# ── verdicts (pre-registered criteria unchanged) ──
rows = inst["nbody_rows"]
newton_max = max(r["newton"]["rel_da_mercury"] for r in rows)
sph_V1k = max(v["max_rel_da"] for v in inst["sph"]["boost"].values())
gr_rel = [r["gr_1pn"]["rel_da_mercury"] for r in rows]
first_mis = [r["gr_1pn_model"]["mercury"]["mismatch_first_order_vector"] for r in rows]
first_mis_mag = [r["gr_1pn_model"]["mercury"]["mismatch_first_order_magnitude"] for r in rows]
exact_mis = [r["gr_1pn_model"]["mercury"]["mismatch_exact"] for r in rows]
design_1pn_pass = max(first_mis) <= 0.10
vals = [max(newton_max, 1e-300), sph_V1k, min(gr_rel)]
anchor = {k: prec[k]["arcsec_per_cy"] for k in ("gr_rest", "eih_rest")}
frame_dt0 = {}
for law in ("gr", "eih"):
    r0 = prec[f"{law}_rest"]["over_u"]
    frame_dt0[law] = {"x": rich[f"{law}_x"]["domega_over_u_dt0"] / r0 - 1,
                      "y": prec[f"{law}_y"]["over_u"] / r0 - 1, "z": prec[f"{law}_z"]["over_u"] / r0 - 1}
dev600 = {k: v["meas_over_pred_minus_1"] for k, v in prec.items() if v["dt_s"] == 600.0}
dev600_noX = {k: v for k, v in dev600.items() if not k.endswith("_x")}
dc = dop["criteria"]; dp = dop["precession"]
verdict = {
    "design_SEPARATE_criterion_met": newton_max <= 1e-12 and sph_V1k <= 1e-12 and design_1pn_pass,
    "design_SAME_criterion_met": max(vals) / min(vals) <= 10,
    "corrected_SEPARATE_met": newton_max <= 1e-12 and sph_V1k <= 1e-12 and max(exact_mis) <= 1e-6,
    "newton_max_rel_da": newton_max, "sph_max_rel_da_V1km_s": sph_V1k,
    "sph_no_viscosity_bit_identical": inst["sph"]["no_viscosity_bit_identical"],
    "gr1pn_rel_da_range": [min(gr_rel), max(gr_rel)],
    "gr1pn_first_order_mismatch_vector_range": [min(first_mis), max(first_mis)],
    "gr1pn_first_order_mismatch_magnitude_range": [min(first_mis_mag), max(first_mis_mag)],
    "gr1pn_exact_mismatch_max": max(exact_mis), "design_1pn_first_order_10pct_pass": design_1pn_pass,
    "sign_probe": inst["sign_probe"],
    "anchor_arcsec_per_cy": anchor, "anchor_lit": LIT_GR,
    "anchor_pass_1pct": {k: abs(v / LIT_GR - 1) <= 0.01 for k, v in anchor.items()},
    "frame_test_rel_change_dt0": frame_dt0,
    "frame_test_pass_1pct_dt0": {law: all(abs(v) <= 0.01 for v in d.values()) for law, d in frame_dt0.items()},
    "frame_test_reading": {
        "gr_1pn": "fails in-plane, but BY CONSTRUCTION: absolute v_i (single-body Sun-frame formula); the sign-fixed "
                  "law has the same y/rest ratio (DOP853: %.6f vs %.6f), so this is not evidence of a defect beyond "
                  "the undocumented frame dependence" % (dc["ratio_y_over_rest"]["gr_engine"], dc["ratio_y_over_rest"]["gr_ref"]),
        "eih_cross": "fails for the y-boost (x%.4f) and drifts e under the x-boost; restoring only the cross-term sign "
                     "gives +6 u in all four frames (DOP853), so the frame dependence is the same defect as the sign"
                     % dc["ratio_y_over_rest"]["eih_engine"]},
    "engine_vs_gauss_rel_dev_dt600": dev600,
    "engine_vs_gauss_max_abs_rel_dev_dt600_rest_y_z": max(abs(v) for v in dev600_noX.values()),
    "richardson_x_boost": rich, "forest_ruth_convergence_ratio_gr_x": conv_ratio,
    "dt_halving_rel_change_gr": prec["gr_rest_dt1200"]["over_u"] / prec["gr_rest"]["over_u"] - 1,
    "dt_halving_rel_change_eih": prec["eih_rest_dt1200"]["over_u"] / prec["eih_rest"]["over_u"] - 1,
    "newton_boost_x_minus_rest_rad_per_orbit_dt600": R["newton_x"]["domega_per_orbit_rad"] - R["newton_rest"]["domega_per_orbit_rad"],
    "newton_baselines": base_note,
    "dop853_v2_criteria": dc, "dop853_v2_precession_over_u": {k: v["over_u"] for k, v in dp.items()},
    "dop853_v2_de_over_u": {k: v["de_over_u"] for k, v in dp.items()},
    "dop853_v2_coefficient_fits": dop["coefficient_fits"],
    "test_fixture_eih_vs_1pn": dop["test_fixture"],
    "anchor_formula_variants_arcsec_per_cy": anc["results"],
}
prereg_notes = {
    "file_mtimes_2026-10-10": files,
    "rest_frame_predictions": "gr_1pn -6 u and eih_cross -10 u were written in the u9_orbits.py docstring (03:01) "
                              "before any orbit ran: genuine advance predictions, confirmed.",
    "boosted_frame_gauss_predictions": "u9_gauss.py was last edited at 03:05 but its integrals were evaluated at 03:20, "
                                       "after orbits_p1-p3 (03:07-03:11, containing the rest/x/y/z runs) were written. The "
                                       "boosted-frame values (6.384, 35.31, -21.93 u, de -6.054 and -2.453 u) are therefore "
                                       "a post-hoc explanation, not advance predictions. The orbit-script criteria "
                                       "(anchor 1 %, frame 1 %, controls) were fixed before the runs and did not move.",
    "sph_prediction_c_missed": "u9_instant.py prediction (c) forecast SPH roundoff ~1e-12..1e-11 at V = 1 km/s and "
                               "said the design's 1e-12 line 'may be crossed by roundoff alone'. Measured %.2e..%.2e: "
                               "5-50x smaller, the line was not crossed at 1 km/s. The magnitude prediction missed; "
                               "the scaling prediction (proportional to V, 0 without viscosity) held."
                               % (min(v["max_rel_da"] for v in inst["sph"]["boost"].values()), sph_V1k),
    "repair_criteria_D1_D4": "fixed in the u9_dop853_v2.py docstring before that script ran (all met).",
}
out = {"experiment": "U9 Galilean boost + Mercury anchor (repaired v2)", "verdict": verdict, "prereg_notes": prereg_notes,
       "precession_forest_ruth": prec, "u_rad_per_orbit": u, "gauss": gauss, "orbit_runs": R, "dop853_v2": dop,
       "anchor_v2": anc, "instant": inst,
       "sources": {"orbits": S, "instant": inst["sources"], "gauss_integrals": gauss["sources"],
                   "dop853_v2": dop["sources"], "anchor_v2": anc["sources"],
                   "LIT": {"Mercury GR advance 42.98 arcsec/cy": "C. M. Will, Living Rev. Relativ. 17, 4 (2014), eq. (65), "
                           "doi:10.12942/lrr-2014-4, arXiv:1403.7377",
                           "(GM)_sun^N = 1.3271244e20 m^3 s^-2": "IAU 2015 Resolution B3, arXiv:1510.07674 (PDF fetched "
                           "2026-10-10, lit_v2/)",
                           "EIH velocity term with n_AB = (x_A - x_B)/r_AB": "Einstein-Infeld-Hoffmann equations, Wikipedia "
                           "(checked by the runner and by a reviewer 2026-10-10)",
                           "test-particle 1PN, outward r": "L. Iorio, arXiv:gr-qc/0405053, eq. (1); published in A. Reimer "
                           "(ed.), Quantum Gravity Research Trends, Nova (2006), pp. 297-318 (arXiv journal-ref, fetched "
                           "2026-10-10)",
                           "Mercury total precession 575.3100 +- 0.0015 arcsec/cy, J2 = (2.25 +- 0.09)e-7": "R. S. Park et al., "
                           "AJ 153, 121 (2017), doi:10.3847/1538-3881/aa5be2 (Crossref record fetched 2026-10-10)"}}}
with open(os.path.join(D, "u9_results_v2.json"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1)
print(json.dumps({k: verdict[k] for k in ("design_SEPARATE_criterion_met", "design_SAME_criterion_met",
                                          "corrected_SEPARATE_met", "anchor_arcsec_per_cy", "anchor_pass_1pct",
                                          "frame_test_rel_change_dt0", "engine_vs_gauss_rel_dev_dt600",
                                          "engine_vs_gauss_max_abs_rel_dev_dt600_rest_y_z", "richardson_x_boost",
                                          "forest_ruth_convergence_ratio_gr_x", "dt_halving_rel_change_gr",
                                          "dt_halving_rel_change_eih", "newton_baselines")}, indent=1))

# ── figure ──
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#8b5cf6"
plt.rcParams.update({"axes.unicode_minus": True, "font.size": 9.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 6.4), facecolor=SURF, gridspec_kw={"width_ratios": [1.0, 1.35]})
for ax in (ax1, ax2):
    ax.set_facecolor(SURF); ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
sc = inst["sph"]["scan_rand_dir"]
ax1.loglog([q["V"] for q in sc], [q["max_rel_da"] for q in sc], "o-", color=S1, ms=4, lw=1.6,
           label="SPH cloud N = 64 (max over parcels): roundoff, ∝ V")
ys = inst["nbody_scan"]["y"]
ax1.loglog([q["V"] for q in ys], [q["gr_1pn"] for q in ys], "s-", color=S2, ms=4, lw=1.6, label="N-body gr_1pn (Mercury, boost ∥ y)")
ax1.loglog([q["V"] for q in ys], [q["eih_cross"] for q in ys], "^-", color=S3, ms=4, lw=1.6, label="N-body eih_cross (Mercury, boost ∥ y)")
ax1.loglog([q["V"] for q in ys], [1e-18] * len(ys), "D", color=S4, ms=4, label="N-body Newtonian: δa = 0 exactly (drawn at floor)")
ax1.axhline(1e-12, color=INK2, ls="--", lw=1.0)
ax1.text(0.0013, 1.6e-12, "design line 10⁻¹²", color=INK2, fontsize=8.5)
ax1.axvline(1e3, color=GRID, lw=1.2); ax1.axvline(3e4, color=GRID, lw=1.2)
ax1.text(1.1e3, 3e-17, "SPH\nV = 1 km/s", color=INK2, fontsize=8)
ax1.text(3.3e4, 3e-17, "N-body\nV = 30 km/s", color=INK2, fontsize=8)
ax1.text(0.0013, 2.0e-21, "A correct 1PN law also changes by O(V·v/c²) at one instant under a Galilean boost,\n"
         "so the 1PN curves here validate the engine; they do not test 1PN correctness.",
         color=INK2, fontsize=7.8, va="bottom")
ax1.set_xlim(8e-4, 2e6); ax1.set_ylim(1e-21, 1e-6)
ax1.set_xlabel("Galilean boost V (m/s)"); ax1.set_ylabel("|a(boosted) − a(rest)| / |a|  (one instant)")
ax1.set_title("(a) One instant: Newtonian and SPH unchanged to roundoff (by construction)", loc="left", fontsize=10, color=INK)
ax1.legend(loc="upper left", fontsize=8, frameon=False)

frames = ["rest", "x", "y", "z"]
xs = np.arange(len(frames)); w = 0.2
Pk = dop["P_kepler_s"]; ud = dop["u_rad_per_orbit"]
g_m = [rich["gr_x"]["arcsec_per_cy_dt0"] if f == "x" else prec[f"gr_{f}"]["arcsec_per_cy"] for f in frames]
e_m = [rich["eih_x"]["arcsec_per_cy_dt0"] if f == "x" else prec[f"eih_{f}"]["arcsec_per_cy"] for f in frames]
g_r = [per_cy(dp[f"gr_ref_{f}"]["over_u"] * ud, Pk) for f in frames]
e_r = [per_cy(dp[f"eih_ref_{f}"]["over_u"] * ud, Pk) for f in frames]
series = [(g_m, S1, "engine gr_1pn (Forest–Ruth, 20 orbits)"),
          (e_m, S2, "engine eih_cross (Forest–Ruth, 20 orbits)"),
          (g_r, S3, "reference: gr_1pn with its sign restored (DOP853)"),
          (e_r, S4, "reference: eih_cross with its cross-term sign restored (DOP853)")]
for k, (vals_, col, lab) in enumerate(series):
    xpos = xs + (k - 1.5) * w
    ax2.bar(xpos, vals_, w, color=col, label=lab)
    for xp, v in zip(xpos, vals_):
        ax2.text(xp, v + (8 if v >= 0 else -8), f"{v:+.0f}".replace("-", "−"), ha="center",
                 va="bottom" if v >= 0 else "top", fontsize=7.5, color=INK)
ax2.axhline(LIT_GR, color=INK, lw=1.2, ls="--", label="GR: +42.98″/cy in every frame (Will 2014)")
ax2.axhline(0, color=INK2, lw=0.8)
ax2.set_xticks(xs)
ax2.set_xticklabels(["rest frame", "boost ∥ x\n(perihelion dir.; dt → 0)", "boost ∥ y\n(perihelion velocity)",
                     "boost ∥ z\n(orbit normal)"])
ax2.set_ylabel("perihelion advance minus Newtonian run (″ per century)")
ax2.set_title("(b) Whole orbits: Mercury precession at rest and boosted by 30 km/s", loc="left", fontsize=10, color=INK)
ax2.legend(loc="upper left", fontsize=7.8, frameon=False, ncol=1)
ax2.set_ylim(-330, 400)
fig.suptitle("U9: the engine's 1PN layers turn Mercury's perihelion backwards (gr_1pn −43″/cy, eih_cross −72″/cy; GR +43″/cy). "
             "Restoring eih_cross's cross-term sign gives +43″/cy in every frame;\n"
             "gr_1pn stays frame-dependent even with its sign restored (absolute velocities, by construction). "
             "Newtonian and SPH boost invariance holds by construction.",
             fontsize=10.5, color=INK, x=0.01, y=0.985, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.955))
fig.savefig(os.path.join(D, "u9_figure_v2.png"), dpi=150, facecolor=SURF)
print("figure written")
