"""U1 figure: one PNG from u1_results.json, u1_gain_scan.json (no new physics here; numbers are read back)."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(D, "u1_results.json"), encoding="utf-8"))
GS = json.load(open(os.path.join(D, "u1_gain_scan.json"), encoding="utf-8"))

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#8b5cf6"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 9.5,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.unicode_minus": True,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})

fig, axs = plt.subplots(2, 2, figsize=(13.5, 9.2))
fig.suptitle("U1: the σ-bridge as coded fails all five weak-field gates by 2.9–11.1 orders (NO-GO); "
             "the nbody GW path is off by a factor r (r⁻⁴ for r⁻⁵)\nand, once fixed, sits below float64 resolution for PSR J1738+0333",
             fontsize=12, fontweight="bold", x=0.01, ha="left", y=0.99)

# ── A: engine check ──────────────────────────────────────────────
ax = axs[0, 0]
rows = GS["rows"]
for batched, col, lab in ((False, S1, "Peters r⁻⁵ × gain, per-step rescale (engine method)"),
                          (True, S3, "Peters r⁻⁵ × gain, losses batched (scratch)")):
    g = np.array([r["gain"] for r in rows if r["batched"] == batched])
    y = np.array([r["energy_ratio"] for r in rows if r["batched"] == batched])
    ax.plot(g, y, "o-", color=col, lw=1.8, ms=5, label=lab)
a_m = R["engine"]["J1738"]["a_m"]
ratio_coded = R["engine"]["defect_diagnosis"]["engine_over_peters_divided_by_a_per_m"]
ax.plot([a_m], [ratio_coded], marker="D", ms=8, color=S2, ls="none",
        label="engine as coded (r⁻⁴ ≡ gain a = 1.73×10⁹ m): Ṗb = 1.73×10⁹ × Peters")
ax.axhspan(0.98, 1.02, color=S3, alpha=0.12, lw=0)
ax.axhline(1, color=INK2, lw=0.9)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_ylim(0.5, 20)
ax.set_xlabel("gain g on the Peters loss (g = 1 is J1738+0333's real loss)")
ax.set_ylabel("engine Ṗb / (g × Peters)")
ax.set_title("A. Engine leg: pre-registered ±2 % (band) — BUG branch fired", loc="left", fontsize=10.5)
ax.annotate("g = 1: per-step factor 1 − 9×10⁻¹⁸ rounds to exactly 1;\nthe 8.8× is velocity-recompute round-off",
            xy=(1, 8.83), xytext=(3, 13), fontsize=8.3, color=INK2, arrowprops=dict(arrowstyle="->", color=INK2))
ax.legend(loc="upper right", fontsize=7.8, bbox_to_anchor=(1.0, 0.86))

# ── B: the five gates ─────────────────────────────────────────────
ax = axs[0, 1]
gr = R["gate_rows"]
labels, vals, pred = [], [], []
for r in gr[:4]:
    labels.append(r["row"]); vals.append(r["log10_excess"]); pred.append(r["predicted_log10_excess"])
for da, ex in zip((0.056, 0.11), gr[4]["log10_excess"]):
    labels.append(f"(Δα)² J1738 dipole, Δα = {da}"); vals.append(ex); pred.append(3.0)
labels = [l.replace("|gamma-1|", "|γ − 1|").replace("|eta_N|", "|η_N|").replace("eta(Ti,Pt)", "η(Ti,Pt)")
          .replace("|beta_LPI|", "|β_LPI|") for l in labels]
yy = np.arange(len(labels))[::-1]
ax.barh(yy, vals, color=S2, height=0.55, label="measured: log₁₀(prediction / 2σ bound), Mentat")
ax.plot(pred, yy, marker="|", ms=16, mew=2.2, ls="none", color=INK, label="pre-registered prediction (dipole: ≥ 3)")
ax.axvline(0, color=INK, lw=1.2)
for y_, v_ in zip(yy, vals):
    ax.text(v_ + 0.15, y_, f"{v_:.2f}", va="center", fontsize=8.5, color=INK)
ax.set_yticks(yy); ax.set_yticklabels(labels, fontsize=8.5)
ax.set_xlim(-0.5, 13)
ax.set_xlabel("orders of magnitude above the 2σ bound (GO requires < 0)")
ax.set_title("B. Five gate rows, massless σ, ξ = 0.1582: NO-GO on every row", loc="left", fontsize=10.5)
ax.legend(loc="upper right", fontsize=7.8)

# ── C: route (a) ξ_max per row ─────────────────────────────────────
ax = axs[1, 0]
ra = R["routes"]["a_long_range_small_xi"]
keys = [("cassini", "Cassini γ"), ("llr", "LLR η_N"), ("clocks", "clocks β_LPI"), ("microscope", "MICROSCOPE η"),
        ("dipole_b10", "J1738 dipole (10 % binding)"), ("dipole_b20", "J1738 dipole (20 % binding)")]
xv = np.arange(len(keys))
ax.bar(xv, [ra[k] for k, _ in keys], color=S1, width=0.6, label="route (a): largest long-range ξ each row allows")
ax.axhspan(0.13, 0.17, color=S4, alpha=0.25, lw=0, label="the Captain's bet ξ = 0.13–0.17")
ax.set_yscale("log"); ax.set_ylim(1e-13, 1)
ax.set_xticks(xv); ax.set_xticklabels([l for _, l in keys], rotation=18, ha="right", fontsize=8.5)
ax.set_ylabel("ξ_max (massless σ)")
ax.set_title("C. Route (a): long range needs ξ ≲ 1.3×10⁻¹² (≲ 1.2×10⁻⁵ if coupling made universal)", loc="left", fontsize=10.5)
ax.legend(loc="upper left", fontsize=7.8)

# ── D: route (b) λ_max per row at the Captain's ξ ──────────────────
ax = axs[1, 1]
rb = R["routes"]["b_massive_sigma_at_captains_xi"]
keys = [("microscope", "MICROSCOPE η"), ("cassini", "Cassini γ"), ("clocks", "clocks β_LPI"), ("llr", "LLR η_N"),
        ("dipole", "J1738 dipole\n(no radiation if λ < cPb/2π)")]
xv = np.arange(len(keys))
for off, (xi_key, col, lab) in zip((-0.27, 0.0, 0.27), (("0.13", S1, "ξ = 0.13"), ("0.1582", S2, "ξ = 0.1582"), ("0.17", S3, "ξ = 0.17"))):
    ax.bar(xv + off, [rb[xi_key][k]["lambda_max_m"] for k, _ in keys], width=0.26, color=col, label=lab)
ax.set_yscale("log")
ax.set_xticks(xv); ax.set_xticklabels([l for _, l in keys], fontsize=8.5)
ax.set_ylabel("largest allowed Yukawa range λ (m)")
ax.set_title("D. Route (b): at the Captain's ξ, MICROSCOPE caps σ's range at λ < 34 km", loc="left", fontsize=10.5)
ax.text(0, rb["0.1582"]["microscope"]["lambda_max_m"] * 2.2, "34 km", ha="center", fontsize=8.5, color=INK)
ax.legend(loc="upper left", fontsize=7.8)
fig.text(0.01, 0.005, "Route (c) σ ≡ −2ξΦ/c² as notation: no observable, all rows pass by construction — only if e^σ is a change of "
         "units;\n"
         "a mass-only universal rescale gives γ − 1 = 4ξ/(1 − 2ξ) = 0.93 (Cassini excess 10^4.3). Routes recorded, not chosen.",
         fontsize=8.3, color=INK2)
fig.tight_layout(rect=(0, 0.035, 1, 0.945))
out = os.path.join(D, "u1_sigma_kill_test.png")
fig.savefig(out, dpi=150, facecolor=SURF)
print(out)
