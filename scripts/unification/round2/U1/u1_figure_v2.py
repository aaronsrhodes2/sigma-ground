"""U1 v2 figure: one PNG from u1_results_v2.json, u1_gain_scan.json, u1_gw_precision_diag.json, u1_engine_defect2_v2.json.
No new physics here: every number in the titles and annotations is read back from the JSONs."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = os.path.dirname(os.path.abspath(__file__))
L = lambda f: json.load(open(os.path.join(D, f), encoding="utf-8"))
R, GS, DG, D2 = L("u1_results_v2.json"), L("u1_gain_scan.json"), L("u1_gw_precision_diag.json"), L("u1_engine_defect2_v2.json")

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#8b5cf6"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 9.5,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.unicode_minus": True,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def sci(x, nd=2):
    m, e = f"{x:.{nd}e}".split("e")
    return f"{m}×10{str(int(e)).translate(SUP)}"


ratio_eng = R["engine"]["ratio_vs_peters"]["Pbdot_energy_endpoint"]
eps2000 = DG["float64_resolution"][0]["eps_per_step"]
rows = R["gate_rows"]
pre = [r["log10_excess"] for r in rows[:4]] + list(rows[4]["log10_excess"])
fig, axs = plt.subplots(2, 2, figsize=(13.8, 9.6))
fig.suptitle(f"U1: as coded, the σ-bridge fails all five weak-field gates by {min(pre):.1f}–{max(pre):.1f} orders "
             f"(NO-GO, derived); nbody's GW loss carries r⁻⁴ for r⁻⁵ ({sci(ratio_eng)} × Peters)\n"
             f"and, with r⁻⁵, J1738+0333's per-step loss ({sci(eps2000)}) rounds to zero in float64 at the pre-registered dt = Pb/2000",
             fontsize=11.5, fontweight="bold", x=0.01, ha="left", y=0.99)

# ── A: engine leg ───────────────────────────────────────────────
ax = axs[0, 0]
g_rows = GS["rows"]
for batched, col, lab in ((False, S1, "r⁻⁵ × g, per-step rescale (engine method)"),
                          (True, S2, "r⁻⁵ × g, losses batched (scratch)")):
    g = np.array([r["gain"] for r in g_rows if r["batched"] == batched])
    y = np.array([r["energy_ratio"] for r in g_rows if r["batched"] == batched])
    ax.plot(g, y, "o-", color=col, lw=1.8, ms=5, label=lab)
a_m = R["engine"]["J1738"]["a_m"]
ratio_over_a = R["engine"]["defect_diagnosis"]["engine_over_peters_divided_by_a_per_m"]
ax.plot([a_m], [ratio_over_a], marker="D", ms=8, color=S3, ls="none",
        label=f"engine as coded: r⁻⁴ ≡ g = a = {sci(a_m)} m")
u = [r["fit_over_peters"] for r in D2["j1738_recompute_only"]["rows"] if r["steps_per_orbit"] == 2000]
ax.errorbar([1.0], [np.mean(u)], yerr=[[np.mean(u) - min(u)], [max(u) - np.mean(u)]], fmt="s", color=S4, ms=6, capsize=4,
            label=f"damp forced to 1, 5 phases: {min(u):.2f}–{max(u):.2f}")
ax.axhspan(0.98, 1.02, color=S3, alpha=0.12, lw=0)
ax.axhline(1, color=INK2, lw=0.9)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_ylim(0.4, 400)
ax.set_xlabel("gain g on the Peters loss (g = 1 is J1738+0333's real loss)")
ax.set_ylabel("engine Ṗb / (g × Peters)")
ax.set_title("A. Engine leg vs pre-registered ±2 % (band): BUG branch fired", loc="left", fontsize=10.5)
g1 = next(r["energy_ratio"] for r in g_rows if r["gain"] == 1.0 and not r["batched"])
ax.annotate(f"g = 1, dt = Pb/2000: damp = 1 − {sci(eps2000, 1)} rounds to exactly 1;\n"
            f"the {g1:.2f}× is velocity-recompute round-off", xy=(1, g1), xytext=(1.6, 90), fontsize=8.2, color=INK2,
            arrowprops=dict(arrowstyle="->", color=INK2))
ax.legend(loc="upper right", fontsize=7.6, bbox_to_anchor=(1.0, 0.78))

# ── B: the five gates ─────────────────────────────────────────────
ax = axs[0, 1]
labels, vals, alt, pred = [], [], [], []
for r in rows[:4]:
    labels.append(r["row"]); vals.append(r["log10_excess"]); alt.append(r["alt_log10_excess_central_plus_2sigma"])
    pred.append(r["predicted_log10_excess"])
for i, da in enumerate((0.056, 0.11)):
    labels.append(f"(Δα)² J1738 dipole, Δα = {da}"); vals.append(rows[4]["log10_excess"][i])
    alt.append(rows[4]["alt_log10_excess_central_plus_2sigma"][i]); pred.append(3.0)
labels = [l.replace("|gamma-1|", "|γ − 1|").replace("|eta_N|", "|η_N|").replace("eta(Ti,Pt)", "η(Ti,Pt)")
          .replace("|beta_LPI|", "|β_LPI|") for l in labels]
yy = np.arange(len(labels))[::-1]
ax.barh(yy, vals, color=S1, height=0.55, label="vs pre-registered bound (dipole: 1σ-based)")
ax.plot(alt, yy, marker="o", ms=6, ls="none", color=S2,
        label="vs |central| + 2σ (dipole: Freire Ṗb_D, 2σ)")
ax.plot(pred, yy, marker="|", ms=16, mew=2.2, ls="none", color=INK, label="pre-registered prediction (dipole ≥ 3)")
ax.axvline(0, color=INK, lw=1.2)
for y_, v_ in zip(yy, vals):
    ax.text(v_ + 0.25, y_ + 0.18, f"{v_:.2f}", va="center", fontsize=8.3, color=INK)
ax.set_yticks(yy); ax.set_yticklabels(labels, fontsize=8.5)
ax.set_xlim(-0.5, 13)
ax.set_xlabel("orders of magnitude above the bound: log₁₀(prediction / bound), Mentat (GO requires < 0)")
ax.set_title(f"B. Five gate rows, massless σ, ξ = {R['couplings']['XI']}: NO-GO on every row [DERIVED]", loc="left", fontsize=10.5)
ax.legend(loc="upper right", fontsize=7.6)

# ── C: route (a) ξ_max per row ─────────────────────────────────────
ax = axs[1, 0]
ra = R["routes"]["a_long_range_small_xi"]
keys = [("cassini", "Cassini γ"), ("llr", "LLR η_N"), ("clocks", "clocks β_LPI"), ("microscope", "MICROSCOPE η"),
        ("dipole_b10", "dipole, 10 % binding"), ("dipole_b20", "dipole, 20 % binding")]
xv = np.arange(len(keys))
ax.bar(xv, [ra[k] for k, _ in keys], color=S1, width=0.6, label="route (a): largest long-range ξ each row allows")
ax.axhspan(0.13, 0.17, color=S2, alpha=0.3, lw=0, label="the Captain's bet ξ = 0.13–0.17")
ax.set_yscale("log"); ax.set_ylim(1e-13, 1)
ax.set_xticks(xv); ax.set_xticklabels([l for _, l in keys], rotation=25, ha="right", fontsize=8.5)
ax.set_ylabel("ξ_max (massless σ)")
ax.set_title(f"C. Route (a): long range needs ξ ≲ {sci(ra['xi_max_all_rows'], 1)} "
             f"(≲ {sci(ra['xi_max_if_universal_coupling'], 1)} if the coupling is made universal)", loc="left", fontsize=10.5)
ax.legend(loc="upper left", fontsize=7.8)

# ── D: route (b) λ_max per row at the Captain's ξ ──────────────────
ax = axs[1, 1]
rb = R["routes"]["b_massive_sigma_at_captains_xi"]
keys = [("microscope", "MICROSCOPE η"), ("clocks", "clocks β_LPI"), ("cassini", "Cassini γ\n(GM☉ calibrated at 1 AU)"),
        ("llr", "LLR η_N"), ("dipole", "J1738 dipole\n(no radiation if λ < cPb/2π)")]
xv = np.arange(len(keys))
xi_keys = sorted(rb.keys(), key=float)
for off, xi_key, col in zip((-0.27, 0.0, 0.27), xi_keys, (S1, S2, S3)):
    ax.bar(xv + off, [rb[xi_key][k]["lambda_max_m"] for k, _ in keys], width=0.26, color=col, label=f"ξ = {xi_key}")
ax.plot([2], [rb[xi_keys[1]]["cassini_b1p6_superseded"]["lambda_max_m"]], marker="v", ms=8, color=S4, ls="none",
        label=f"v1 Cassini at b = 1.6 R☉ (superseded): {sci(rb[xi_keys[1]]['cassini_b1p6_superseded']['lambda_max_m'], 1)} m")
ax.set_yscale("log")
ax.set_xticks(xv); ax.set_xticklabels([l for _, l in keys], fontsize=8.3)
ax.set_ylabel("largest allowed Yukawa range λ (m)")
lam_mic = rb[xi_keys[1]]["microscope"]["lambda_max_m"]
ax.set_title(f"D. Route (b): these five rows allow at most λ ≈ {lam_mic / 1e3:.0f} km (MICROSCOPE binds);\n"
             f"   lab EP / inverse-square-law tests not evaluated (likely far lower)",
             loc="left", fontsize=10.2)
ax.text(0, lam_mic * 2.2, f"{lam_mic / 1e3:.1f} km", ha="center", fontsize=8.5, color=INK)
ax.legend(loc="upper left", fontsize=7.6)
rc = R["routes"]["c_notation"]
fig.text(0.01, 0.005, "Route (c) σ ≡ −2ξΦ/c² as notation: no observable, all rows pass by construction, but only if e^σ is a change "
         "of units;\n"
         f"a mass-only universal rescale gives γ − 1 = 4ξ/(1 − 2ξ) = {rc['caveat_gamma_minus_1_mass_only_universal']:.2f} "
         f"(Cassini excess 10^{rc['caveat_log10_excess']:.1f}). Routes recorded, not chosen.",
         fontsize=8.3, color=INK2)
fig.tight_layout(rect=(0, 0.035, 1, 0.945))
out = os.path.join(D, "u1_sigma_kill_test_v2.png")
fig.savefig(out, dpi=150, facecolor=SURF)
print(out)
