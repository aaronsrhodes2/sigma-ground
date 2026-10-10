"""U9 — aggregate parts A/B/C, apply the pre-registered criteria, write u9_results.json and the figure.
Usage: python -X utf8 u9_report.py <folder>
"""
import glob, json, math, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = sys.argv[1]
inst = json.load(open(os.path.join(D, "u9_instant.json"), encoding="utf-8"))
gauss = json.load(open(os.path.join(D, "u9_gauss.json"), encoding="utf-8"))
orb = {"runs": {}}
for fn in sorted(glob.glob(os.path.join(D, "orbits_p*.json"))):
    o = json.load(open(fn, encoding="utf-8"))
    orb["runs"].update(o["runs"]); orb["sources"] = o["sources"]; orb["a_m"] = o["a_m"]; orb["e"] = o["e"]
S = orb["sources"]
G, C = S["G"]["value"], S["c"]["value"]
MU = G * (S["sun"]["value"]["mass_kg"] + S["mercury"]["value"]["mass_kg"])
A, E = orb["a_m"], orb["e"]
p = A * (1 - E * E)
u = math.pi * MU / (C * C * p)                    # rad per orbit
ARCSEC = 180 * 3600 / math.pi                      # rad -> arcsec (pure math)
CENTURY_S = 36525 * 86400.0                        # Julian century (definition)
LIT_GR = 42.98                                     # [LIT] Will 2014, Living Rev. Relativ. 17, 4 (arXiv:1403.7377)
R = orb["runs"]


def per_cy(rad_per_orbit, period_s):
    return rad_per_orbit * ARCSEC * CENTURY_S / period_s


base = {600.0: R["newton_rest"]["domega_per_orbit_rad"]}
# the dt1200 GR runs need a dt1200 Newtonian baseline; the Newtonian bias scales ~dt^4, measured at 600 s -> scale x16
base[1200.0] = 16.0 * base[600.0]
base[300.0] = base[600.0] / 16.0
prec = {}
for nm, r in R.items():
    if r["physics"] == "newton":
        continue
    d = r["domega_per_orbit_rad"] - base[r["dt_s"]]
    prec[nm] = {"rad_per_orbit": d, "arcsec_per_cy": per_cy(d, r["period_s"]), "over_u": d / u,
                "de_per_orbit": r["de_per_orbit"], "period_s": r["period_s"]}
gr_formula_cy = per_cy(6 * u, R["newton_rest"]["period_s"])
pred = gauss["pred_over_u"]
for nm, pr in prec.items():
    law = "gr_engine" if nm.startswith("gr") else "eih_engine"
    fr = nm.split("_")[1]
    pr["predicted_over_u"] = pred[f"{law}_{fr}"]
    pr["predicted_arcsec_per_cy"] = per_cy(pred[f"{law}_{fr}"] * u, pr["period_s"])
    pr["meas_over_pred_minus_1"] = pr["over_u"] / pred[f"{law}_{fr}"] - 1
    pr["de_over_u"] = pr["de_per_orbit"] / u
    pr["de_predicted_over_u"] = gauss["de_pred_over_u"][f"{law}_{fr}"]

# ── verdicts ──
rows = inst["nbody_rows"]
newton_max = max(r["newton"]["rel_da_mercury"] for r in rows)
newton_bit = all(r["newton"]["bit_identical"] for r in rows)
sph_V1k = max(v["max_rel_da"] for v in inst["sph"]["boost"].values())
gr_rel = [r["gr_1pn"]["rel_da_mercury"] for r in rows]
first_mis = [r["gr_1pn_model"]["mercury"]["mismatch_first_order_vector"] for r in rows]
first_mis_mag = [r["gr_1pn_model"]["mercury"]["mismatch_first_order_magnitude"] for r in rows]
exact_mis = [r["gr_1pn_model"]["mercury"]["mismatch_exact"] for r in rows]
inplane_xy = [r["gr_1pn_model"]["mercury"]["mismatch_first_order_vector"] for r in rows if r["dir"] in ("x", "y")]
design_1pn_pass = max(first_mis) <= 0.10
separate_design = (newton_max <= 1e-12 and sph_V1k <= 1e-12 and design_1pn_pass)
vals = [max(newton_max, 1e-300), sph_V1k, min(gr_rel)]
same = max(vals) / min(vals) <= 10
corrected_1pn_pass = max(exact_mis) <= 1e-6
anchor = {k: prec[k]["arcsec_per_cy"] for k in ("gr_rest", "eih_rest")}
anchor_pass = {k: abs(v / LIT_GR - 1) <= 0.01 for k, v in anchor.items()}
frame = {}
for law in ("gr", "eih"):
    r0 = prec[f"{law}_rest"]["rad_per_orbit"]
    frame[law] = {fr: prec[f"{law}_{fr}"]["rad_per_orbit"] / r0 - 1 for fr in ("x", "y", "z")}
frame_pass = {law: all(abs(v) <= 0.01 for v in d.values()) for law, d in frame.items()}
dt_conv = prec["gr_rest_dt1200"]["rad_per_orbit"] / prec["gr_rest"]["rad_per_orbit"] - 1
dt_conv_eih = prec["eih_rest_dt1200"]["rad_per_orbit"] / prec["eih_rest"]["rad_per_orbit"] - 1
newton_cov = (R["newton_x"]["domega_per_orbit_rad"] - R["newton_rest"]["domega_per_orbit_rad"])
pred_match = max(abs(pr["meas_over_pred_minus_1"]) for nm, pr in prec.items() if "_dt" not in nm)
# dt convergence of the x-boosted runs: Forest-Ruth with velocity-dependent forces converges at O(dt) here
# (differences halve with dt), so a first-order Richardson extrapolation 2*w(dt/2) - w(dt) gives the dt -> 0 value.
rich = {}
for lo_, hi_ in (("gr_x_dt300", "gr_x"), ("eih_x", "eih_x_dt1200")):
    a, b = prec[lo_], prec[hi_]
    w0 = 2 * a["rad_per_orbit"] - b["rad_per_orbit"]
    e0 = 2 * a["de_per_orbit"] - b["de_per_orbit"]
    nm = lo_.split("_dt")[0]
    rich[nm] = {"pair": [lo_, hi_], "domega_over_u_dt0": w0 / u, "de_over_u_dt0": e0 / u,
                "domega_pred_over_u": prec[nm]["predicted_over_u"], "de_pred_over_u": prec[nm]["de_predicted_over_u"],
                "domega_dev": (w0 / u) / prec[nm]["predicted_over_u"] - 1,
                "de_abs_dev_over_u": e0 / u - prec[nm]["de_predicted_over_u"]}
gx = [prec[k]["rad_per_orbit"] for k in ("gr_x_dt1200", "gr_x", "gr_x_dt300")]
conv_ratio_gr_x = (gx[0] - gx[1]) / (gx[1] - gx[2])
frame_dt0 = {"gr": dict(frame["gr"]), "eih": dict(frame["eih"])}
frame_dt0["gr"]["x"] = rich["gr_x"]["domega_over_u_dt0"] / prec["gr_rest"]["over_u"] - 1
frame_dt0["eih"]["x"] = rich["eih_x"]["domega_over_u_dt0"] / prec["eih_rest"]["over_u"] - 1

verdict = {
    "design_SEPARATE_criterion_met": separate_design,
    "design_SAME_criterion_met": same,
    "newton_max_rel_da": newton_max, "newton_bit_identical_all": newton_bit,
    "sph_max_rel_da_V1km_s": sph_V1k, "sph_no_viscosity_bit_identical": inst["sph"]["no_viscosity_bit_identical"],
    "gr1pn_rel_da_range": [min(gr_rel), max(gr_rel)],
    "gr1pn_first_order_mismatch_vector_range": [min(first_mis), max(first_mis)],
    "gr1pn_first_order_mismatch_magnitude_range": [min(first_mis_mag), max(first_mis_mag)],
    "gr1pn_exact_mismatch_max": max(exact_mis),
    "design_1pn_first_order_10pct_pass": design_1pn_pass,
    "corrected_1pn_exact_1e-6_pass": corrected_1pn_pass,
    "sign_probe": inst["sign_probe"],
    "anchor_arcsec_per_cy": anchor, "anchor_lit": LIT_GR, "anchor_formula_mentat_leaves": gr_formula_cy,
    "anchor_pass_1pct": anchor_pass,
    "frame_test_rel_change": frame, "frame_test_pass_1pct": frame_pass,
    "dt_halving_rel_change_gr": dt_conv, "dt_halving_rel_change_eih": dt_conv_eih,
    "newton_boost_x_minus_rest_rad_per_orbit": newton_cov,
    "engine_vs_gauss_prediction_max_rel_dev_dt600": pred_match,
    "richardson_x_boost": rich, "forest_ruth_convergence_ratio_gr_x_(d1200-d600)/(d600-d300)": conv_ratio_gr_x,
    "frame_test_rel_change_dt0": frame_dt0,
    "frame_test_pass_1pct_dt0": {law: all(abs(v) <= 0.01 for v in d.values()) for law, d in frame_dt0.items()},
    "de_engine_vs_gauss_max_abs_dev_over_u": max(abs(pr["de_over_u"] - pr["de_predicted_over_u"]) for pr in prec.values()),
    "eih_textbook_de_pred_over_u_all_frames": {fr: gauss["de_pred_over_u"][f"eih_textbook_{fr}"] for fr in ("rest", "x", "y", "z")},
    "eih_textbook_pred_over_u_all_frames": {fr: pred[f"eih_textbook_{fr}"] for fr in ("rest", "x", "y", "z")},
}
out = {"experiment": "U9 Galilean boost + Mercury anchor", "verdict": verdict, "precession": prec,
       "u_rad_per_orbit": u, "gauss": gauss, "orbit_runs": R, "instant": inst,
       "sources": {"orbits": S, "instant": inst["sources"], "gauss_integrals": gauss["sources"],
                   "LIT": {"Mercury GR advance 42.98 arcsec/cy": "C. M. Will, Living Rev. Relativ. 17, 4 (2014), "
                           "doi:10.12942/lrr-2014-4, arXiv:1403.7377 (verified by web search 2026-10-10)",
                           "EIH equations, sign of n_AB in the velocity term":
                           "Einstein-Infeld-Hoffmann equations, Wikipedia (n_AB = (x_A - x_B)/r_AB in "
                           "[n_AB.(4v_A - 3v_B)](v_A - v_B)), checked 2026-10-10",
                           "test-particle 1PN acceleration, outward r": "L. Iorio, arXiv:gr-qc/0405053: a_GE = GM/(c^2 r^3)"
                           "{[2(gamma+beta)GM/r - gamma v^2] r + 2(1+gamma)(r.v) v} -> (A,B,C) = (4,1,4) at gamma = beta = 1 "
                           "(verified by web search 2026-10-10)"}}}
with open(os.path.join(D, "u9_results.json"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1)
print(json.dumps(verdict, indent=1))
for nm, pr in sorted(prec.items()):
    print(f"{nm:18s} {pr['arcsec_per_cy']:+10.3f}\"/cy  over_u {pr['over_u']:+8.4f}  pred {pr['predicted_over_u']:+8.4f}"
          f"  dev {pr['meas_over_pred_minus_1']:+.2e}  de/u {pr['de_over_u']:+8.4f} pred {pr['de_predicted_over_u']:+8.4f}")

# ── figure ──
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#8b5cf6"
plt.rcParams.update({"axes.unicode_minus": True, "font.size": 9.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.6), facecolor=SURF,
                               gridspec_kw={"width_ratios": [1.0, 1.15]})
for ax in (ax1, ax2):
    ax.set_facecolor(SURF); ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
sc = inst["sph"]["scan_rand_dir"]
ax1.loglog([q["V"] for q in sc], [q["max_rel_da"] for q in sc], "o-", color=S1, ms=4, lw=1.6,
           label="SPH cloud N = 64 (max over parcels): roundoff, ∝ V")
for key, col, lab in (("gr_1pn", S2, "N-body gr_1pn (Mercury, boost ∥ y)"),
                      ("eih_cross", S3, "N-body eih_cross (Mercury, boost ∥ y)")):
    ys = inst["nbody_scan"]["y"]
    ax1.loglog([q["V"] for q in ys], [q[key] for q in ys], "s-" if key == "gr_1pn" else "^-", color=col, ms=4,
               lw=1.6, label=lab)
floor = 1e-18
ys = inst["nbody_scan"]["y"]
ax1.loglog([q["V"] for q in ys], [floor] * len(ys), "D", color=S4, ms=4, label="N-body Newtonian: δa = 0 exactly (drawn at floor)")
ax1.axhline(1e-12, color=INK2, ls="--", lw=1.0)
ax1.text(0.0013, 1.6e-12, "design line 10⁻¹²", color=INK2, fontsize=8.5)
ax1.axvline(1e3, color=GRID, lw=1.2); ax1.axvline(3e4, color=GRID, lw=1.2)
ax1.text(1.1e3, 3e-17, "SPH\nV = 1 km/s", color=INK2, fontsize=8)
ax1.text(3.3e4, 3e-17, "N-body\nV = 30 km/s", color=INK2, fontsize=8)
ax1.set_xlim(8e-4, 2e6); ax1.set_ylim(3e-19, 1e-6)
ax1.set_xlabel("Galilean boost V (m/s)"); ax1.set_ylabel("|a(boosted) − a(rest)| / |a|  (one instant)")
ax1.set_title("(a) Instantaneous response to a uniform boost", loc="left", fontsize=10, color=INK)
ax1.legend(loc="upper left", fontsize=8, frameon=False)

frames = ["rest", "x", "y", "z"]
xs = np.arange(len(frames))
w = 0.36
g_m = [prec[f"gr_{f}"]["arcsec_per_cy"] for f in frames]
e_m = [prec[f"eih_{f}"]["arcsec_per_cy"] for f in frames]
g_p = [prec[f"gr_{f}"]["predicted_arcsec_per_cy"] for f in frames]
e_p = [prec[f"eih_{f}"]["predicted_arcsec_per_cy"] for f in frames]
ax2.bar(xs - w / 2, g_m, w, color=S2, label="engine gr_1pn (measured, 20 orbits)")
ax2.bar(xs + w / 2, e_m, w, color=S3, label="engine eih_cross (measured, 20 orbits)")
ax2.plot(xs - w / 2, g_p, "_", color=INK, ms=22, mew=2.0)
ax2.plot(xs + w / 2, e_p, "_", color=INK, ms=22, mew=2.0, label="Gauss prediction from the engine's own coefficients")
ax2.axhline(LIT_GR, color=S1, lw=1.8, label="GR: +42.98″/cy, same in every frame (Will 2014)")
ax2.axhline(0, color=INK2, lw=0.8)
ax2.set_xticks(xs); ax2.set_xticklabels(["rest frame", "boost ∥ x\n(perihelion dir.)", "boost ∥ y\n(perihelion vel.)",
                                         "boost ∥ z\n(orbit normal)"])
ax2.set_ylabel("perihelion advance minus Newtonian run (″ per century)")
ax2.set_title("(b) Mercury anchor: whole-orbit precession, rest and boosted by 30 km/s", loc="left", fontsize=10, color=INK)
for i, (gm, em) in enumerate(zip(g_m, e_m)):
    ax2.text(i - w / 2, gm - 6 if gm < 0 else gm + 2, f"{gm:+.0f}".replace("-", "−"), ha="center",
             va="top" if gm < 0 else "bottom", fontsize=8, color=INK)
    ax2.text(i + w / 2, em - 6 if em < 0 else em + 2, f"{em:+.0f}".replace("-", "−"), ha="center",
             va="top" if em < 0 else "bottom", fontsize=8, color=INK)
ax2.legend(loc="upper center", ncol=2, fontsize=8, frameon=False)
lo = min(min(g_m), min(e_m))
ax2.set_ylim(lo * 1.18, 150)
fig.suptitle("U9: Newtonian and SPH forces are boost-invariant to roundoff, but the engine's 1PN terms are not Lorentz-covariant\n"
             "and advance Mercury's perihelion backwards (gr_1pn −43″/cy, eih_cross −72″/cy) instead of GR's +43″/cy",
             fontsize=11, color=INK, x=0.01, y=0.985, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.965))
fig.savefig(os.path.join(D, "u9_figure.png"), dpi=150, facecolor=SURF)
print("figure written")
