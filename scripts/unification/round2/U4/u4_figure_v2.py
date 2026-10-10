#!/usr/bin/env python
"""U4 v2 figure: reads U4_results_v2.json (written by u4_normalisation_table_v2.py); panel C calls the repo's
sigma_from_potential (engine) for a solar-mass body. No new physics numbers are introduced."""
import os, sys, json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = r"D:/Aaron/development/sigma-ground"
sys.path.insert(0, REPO)
from sigma_ground.inventory.core.sigma import XI, sigma_from_potential  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(OUT, "U4_results_v2.json"), encoding="utf8"))

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#8b5cf6"]
plt.rcParams.update({"axes.unicode_minus": True, "font.size": 10, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF})


def sup(v):
    m, e = f"{v:.1e}".split("e")
    e = int(e)
    return f"{m}×10{str(e).translate(str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹'))}"


fig = plt.figure(figsize=(15, 11.6))
gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1], width_ratios=[1.1, 1], hspace=0.36, wspace=0.42)
a = fig.add_subplot(gs[0, :])
b = fig.add_subplot(gs[1, 0])
c = fig.add_subplot(gs[1, 1])

GPA = d["lit"]["GPA_bound"]["value"]
WILL = d["lit"]["Will_2014"]["value"]
GAL = d["lit"]["Galileo_2018"]["value"]
CAS = d["lit"]["Cassini_gamma"]["value"]
norms = [("field_eq", "field equation\nσ = ξ r_s/r"),
         ("inventory", "sigma_from_potential\nσ = ξ GM/rc²"),
         ("docstring", "docstring value\nσ = GM/rc² (no ξ)")]

# ---------------- panel A: pre-registered decision table (clock-only coupling)
interp = [("extra_dilation", "extra dilation on top of GR:  |e^σ − 1| / z_GR", C[0]),
          ("notation", "replacement clock law ('notation'):  |e^σ − (1+z_GR)| / z_GR", C[1])]
ypos = {}
for i, (n, _) in enumerate(norms):
    for j, (it, _, _) in enumerate(interp):
        ypos[(n, it)] = 3 * (2 - i) + (0.45 if j == 0 else -0.45)
a.axvspan(GPA, 10, color=GRID, alpha=0.55, lw=0, zorder=0)
a.axvline(GPA, color=INK, lw=1.4, zorder=1)
a.axvline(WILL, color=INK2, lw=1.0, ls=":", zorder=1)
a.axvline(GAL, color=INK2, lw=1.0, ls="--", zorder=1)
a.text(WILL * 1.3, 8.75, "GP-A 1.4×10⁻⁴ (solid line, pre-registered) [LIT]", color=INK, fontsize=8.5, va="top")
a.text(WILL * 1.3, 8.15, "Will 2014 2×10⁻⁴ (dotted line) [LIT]", color=INK2, fontsize=8.5, va="top")
a.text(GAL / 1.3, 6.9, "Galileo 2018\n2.5×10⁻⁵ [LIT]", color=INK2, fontsize=8.5, va="top", ha="right")
for it, lab, col in interp:
    for body, mk, off in (("earth", "o", 0.12), ("sun", "s", -0.12)):
        xs, ys = [], []
        for n, _ in norms:
            r = next(x for x in d["table"] if x["normalisation"] == n and x["interpretation"] == it and x["body"] == body)
            xs.append(r["metric_stable"]); ys.append(ypos[(n, it)] + off)
        a.scatter(xs, ys, s=58, marker=mk, facecolor=col if body == "earth" else SURF, edgecolor=col,
                  lw=1.6, zorder=3, label=f"{lab.split(':')[0]} — {body.capitalize()}")
for n, _ in norms:
    for it, _, col in interp:
        r = next(x for x in d["table"] if x["normalisation"] == n and x["interpretation"] == it and x["body"] == "earth")
        if r["status"] == "excluded":
            extra = f" ({r['orders_vs_Will2014_2e-4']:+.2f} vs Will)" if n == "inventory" and it == "extra_dilation" else ""
            a.text(r["metric_stable"] / 3.2, ypos[(n, it)] + 0.12,
                   f"{r['orders_vs_GPA']:+.2f} orders{extra}".replace("-", "−"),
                   fontsize=8.5, color=INK2, va="center", ha="right")
        else:
            rs_ = next(x for x in d["table"] if x["normalisation"] == n and x["interpretation"] == it and x["body"] == "sun")
            a.text(1.3e-10, ypos[(n, it)] + 0.85,
                   f"allowed: residual = r_s/2R (Earth {sup(r['metric_stable'])}, Sun {sup(rs_['metric_stable'])})",
                   fontsize=8.5, color=INK2, va="center", ha="left")
a.set_xscale("log")
a.set_xlim(1e-10, 10)
a.set_ylim(-1.0, 8.9)
a.set_yticks([3 * (2 - i) for i in range(3)])
a.set_yticklabels([l for _, l in norms])
a.set_xlabel("clock-law mismatch relative to GR's surface redshift z_GR  (Mentat leaves; Earth ● filled, Sun ■ open)")
a.grid(axis="x", color=GRID, lw=0.7, zorder=0)
a.set_title("A. Pre-registered table: σ acts on clocks only, orbits (the measured GM) follow GR", loc="left",
            fontsize=10.5, color=INK)
h, l = a.get_legend_handles_labels()
a.legend(h, l, loc="upper left", fontsize=8, frameon=False, ncol=2)
for s in ("top", "right"):
    a.spines[s].set_visible(False)

# ---------------- panel B: universal (metric) coupling
U = d["coupling_universal_vs_clock_only"]
yb = {"field_eq": 2, "inventory": 1, "docstring": 0}
series_b = [("extra_universal_gpa_residual_vs_orbit_GM", "GP-A residual vs orbit GM, extra dilation", C[0], "o", 0.18),
            ("notation_universal_gpa_residual_vs_orbit_GM", "GP-A residual vs orbit GM, notation (ξ = G rescaling)", C[1], "D", 0.0),
            ("extra_universal_one_minus_gamma", "Cassini |γ − 1| = k/(1+k), extra dilation", C[2], "s", -0.18)]
b.axvline(GPA, color=INK, lw=1.4, zorder=1)
b.axvline(CAS, color=INK2, lw=1.0, ls="--", zorder=1)
b.text(GPA * 1.3, 2.62, "GP-A\n1.4×10⁻⁴", color=INK, fontsize=8.5, va="top")
b.text(CAS / 1.3, 2.62, "Cassini γ\n2.3×10⁻⁵ [LIT]", color=INK2, fontsize=8.5, va="top", ha="right")
for key, lab, col, mk, off in series_b:
    xs, ys = [], []
    for n, _ in norms:
        r = next(x for x in U["rows"] if x["normalisation"] == n and x["body"] == "earth")
        xs.append(r[key]); ys.append(yb[n] + off)
        if key == "extra_universal_one_minus_gamma":
            b.text(r[key], yb[n] + off + 0.09, f"{r['extra_universal_orders_vs_Cassini']:+.2f} orders vs Cassini",
                   fontsize=8, color=INK2, va="bottom", ha="center")
    b.scatter(xs, ys, s=52, marker=mk, facecolor=col, edgecolor=col, lw=1.2, zorder=3, label=lab)
b.text(1.5e-10, -0.62, "GP-A residuals are O(r_s/R) ≈ 10⁻¹⁰: GP-A cannot see a metric σ", fontsize=8.5,
       color=INK2, va="center")
b.set_xscale("log")
b.set_xlim(1e-10, 10)
b.set_ylim(-0.9, 3.4)
b.set_yticks([2, 1, 0])
b.set_yticklabels([l for _, l in norms])
b.set_xlabel("fractional deviation from GR, Earth  (each set against its own bound)")
b.grid(axis="x", color=GRID, lw=0.7, zorder=0)
b.legend(loc="upper left", fontsize=7.8, frameon=False, bbox_to_anchor=(0.0, 1.0))
b.set_title("B. If e^σ also sets g_tt for orbits (universal coupling)", loc="left", fontsize=10.5, color=INK)
for s in ("top", "right"):
    b.spines[s].set_visible(False)

# ---------------- panel C: horizon row
rs = d["leaves"]["sun"]["r_s_m"]
Msun = d["leaves"]["sun"]["mass_kg"]
u = np.logspace(math.log10(1.0002), math.log10(300.0), 800)        # r / r_s
lin_field = XI / u
inv_engine = np.array([sigma_from_potential(ui * rs, Msun) for ui in u])
u_in = np.linspace(0.3, 1.0, 50)
inv_inside = np.array([sigma_from_potential(ui * rs, Msun) for ui in u_in])
exact_irreg = -XI * np.log1p(-1.0 / u)
c.axvspan(0.3, 1.0, color=GRID, alpha=0.45, lw=0, zorder=0)
c.plot(u, exact_irreg, color=C[2], lw=2.0, label="vacuum branch B = −ξ: −ξ ln(1 − r_s/r), diverges at r_s")
c.plot(u, lin_field, color=C[0], lw=1.6, label="weak-field form ξ r_s/r (gives ξ at r_s)")
c.plot(np.r_[u_in, u], np.r_[inv_inside, inv_engine], color=C[1], lw=1.6,
       label="sigma_from_potential (engine)")
c.plot([0.3, 300], [0, 0], color=C[3], lw=1.6, label="only regular solution: σ = σ_∞ (= 0), no hair")
c.axvline(1.0, color=INK2, lw=1.0, ls=":")
c.text(0.95, 0.57, "r = r_s", color=INK2, fontsize=8.5, ha="right")
c.annotate("ξ/2 = 0.0791 at r_s: the 1/r formula\nevaluated at r_s (compactness = ½,\nclamp inactive); inside r_s the clamp\n"
           "(sigma.py:301) only caps it at ξ/2",
           xy=(1.0, XI / 2), xytext=(1.6, 0.20), fontsize=8, color=INK2,
           arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
c.set_xscale("log")
c.set_xlim(0.3, 300)
c.set_ylim(-0.03, 0.6)
c.set_xlabel("r / r_s  (solar-mass body; real stars sit at R/r_s = 2.4×10⁵ (Sun), 7.2×10⁸ (Earth))")
c.set_ylabel("σ")
c.grid(color=GRID, lw=0.7)
c.legend(loc="upper right", fontsize=7.8, frameon=False)
c.set_title("C. Horizon row: □σ = 0 outside the mass gives A + B ln(1 − r_s/r)", loc="left", fontsize=10.5, color=INK)
for s in ("top", "right"):
    c.spines[s].set_visible(False)

fig.suptitle("Read as a clock-only extra dilation, Gravity Probe A excludes every σ that carries ξ by 3.05–3.35 orders "
             "(fixed by construction once ξ = 0.1582);\nif σ also bends orbits, GP-A is blind and Cassini's γ excludes "
             "it instead. Only ξ-free σ = GM/rc² read as GR notation survives,\nand □σ = −ξR gives no ξ-dependent "
             "horizon value (the code's ξ/2 is the 1/r formula evaluated at r_s)",
             fontsize=12, color=INK, x=0.01, ha="left")
fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.06)
p = os.path.join(OUT, "U4_normalisation_table_v2.png")
fig.savefig(p, dpi=150, facecolor=SURF)
print(p)
