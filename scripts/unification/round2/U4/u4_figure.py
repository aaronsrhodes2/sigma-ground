#!/usr/bin/env python
"""U4 figure: reads U4_results.json (written by u4_normalisation_table.py); panel B calls the repo's
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
d = json.load(open(os.path.join(OUT, "U4_results.json"), encoding="utf8"))

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#8b5cf6"]
plt.rcParams.update({"axes.unicode_minus": True, "font.size": 10, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF})

fig, (a, b) = plt.subplots(1, 2, figsize=(14.5, 6.2), gridspec_kw=dict(width_ratios=[1.25, 1]))

# ---------------- panel A: decision table
GPA = d["lit"]["GPA_bound"]["value"]
GAL = d["lit"]["Galileo_2018"]["value"]
norms = [("field_eq", "field equation\nσ = ξ r_s/r"),
         ("inventory", "sigma_from_potential\nσ = ξ GM/rc²"),
         ("docstring", "docstring value\nσ = GM/rc² (no ξ)")]
interp = [("extra_dilation", "extra dilation on top of GR:  |e^σ − 1| / z_GR", C[0]),
          ("notation", "e^σ IS the GR factor:  |e^σ − (1+z_GR)| / z_GR", C[1])]
ypos = {}
for i, (n, _) in enumerate(norms):
    for j, (it, _, _) in enumerate(interp):
        ypos[(n, it)] = 3 * (2 - i) + (0.45 if j == 0 else -0.45)
a.axvspan(GPA, 10, color=GRID, alpha=0.55, lw=0, zorder=0)
a.axvline(GPA, color=INK, lw=1.4, zorder=1)
a.axvline(GAL, color=INK2, lw=1.0, ls="--", zorder=1)
a.text(GPA * 1.3, 8.05, "excluded by\nGravity Probe A\n(1.4×10⁻⁴) [LIT]", color=INK, fontsize=8.5, va="top")
a.text(GAL / 1.3, 8.05, "Galileo 2018\n2.5×10⁻⁵ [LIT]", color=INK2, fontsize=8.5, va="top", ha="right")
for it, lab, col in interp:
    for body, mk, off in (("earth", "o", 0.12), ("sun", "s", -0.12)):
        xs, ys = [], []
        for n, _ in norms:
            r = next(x for x in d["table"] if x["normalisation"] == n and x["interpretation"] == it
                     and x["body"] == body)
            xs.append(r["metric_stable"]); ys.append(ypos[(n, it)] + off)
        a.scatter(xs, ys, s=58, marker=mk, facecolor=col if body == "earth" else SURF, edgecolor=col,
                  lw=1.6, zorder=3, label=f"{lab.split(':')[0]} — {body.capitalize()}")
for n, _ in norms:
    for it, _, col in interp:
        r = next(x for x in d["table"] if x["normalisation"] == n and x["interpretation"] == it and x["body"] == "earth")
        if r["status"] == "excluded":
            a.text(r["metric_stable"] / 3.2, ypos[(n, it)] + 0.12, f"{r['orders_vs_GPA']:+.2f} orders",
                   fontsize=8.5, color=INK2, va="center", ha="right")
        else:
            rs_ = next(x for x in d["table"] if x["normalisation"] == n and x["interpretation"] == it
                       and x["body"] == "sun")
            a.text(1.3e-10, ypos[(n, it)] + 0.85,
                   f"allowed: residual = r_s/2R\n(Earth {r['metric_stable']:.1e}, Sun {rs_['metric_stable']:.1e})"
                   .replace("e-10", "×10⁻¹⁰").replace("e-06", "×10⁻⁶"),
                   fontsize=8.5, color=INK2, va="center", ha="left")
a.set_xscale("log")
a.set_xlim(1e-10, 10)
a.set_ylim(-1.0, 8.9)
a.set_yticks([3 * (2 - i) for i in range(3)])
a.set_yticklabels([l for _, l in norms])
a.set_xlabel("clock-law mismatch relative to GR's own surface redshift z_GR  (Mentat leaves, Earth ● / Sun ■)")
a.grid(axis="x", color=GRID, lw=0.7, zorder=0)
a.set_title("A. Three normalisations × two readings × two bodies", loc="left", fontsize=10.5, color=INK)
h, l = a.get_legend_handles_labels()
a.legend(h, l, loc="upper left", fontsize=8, frameon=False, ncol=1)
for s in ("top", "right"):
    a.spines[s].set_visible(False)

# ---------------- panel B: horizon row
rs = d["leaves"]["sun"]["r_s_m"]
Msun = d["leaves"]["sun"]["mass_kg"]
u = np.logspace(math.log10(1.0002), math.log10(300.0), 800)        # r / r_s
lin_field = XI / u
inv_engine = np.array([sigma_from_potential(ui * rs, Msun) for ui in u])
u_in = np.linspace(0.3, 1.0, 50)
inv_inside = np.array([sigma_from_potential(ui * rs, Msun) for ui in u_in])
exact_irreg = -XI * np.log1p(-1.0 / u)
b.plot(u, exact_irreg, color=C[2], lw=2.0, label="exact vacuum branch −ξ ln(1 − r_s/r): diverges")
b.plot(u, lin_field, color=C[0], lw=1.6, label="weak-field form ξ r_s/r (gives ξ at r_s)")
b.plot(np.r_[u_in, u], np.r_[inv_inside, inv_engine], color=C[1], lw=1.6,
       label="sigma_from_potential (engine), clamped")
b.plot([0.3, 300], [0, 0], color=C[3], lw=1.6, label="regular at horizon: σ = const (no hair)")
b.axvline(1.0, color=INK2, lw=1.0, ls=":")
b.text(0.93, 0.57, "r = r_s", color=INK2, fontsize=8.5, ha="right")
b.text(0.32, 0.092, "ξ/2 = 0.0791\nis a clamp\n(sigma.py:301),\nnot derived", fontsize=8, color=INK2, va="bottom")
b.set_xscale("log")
b.set_xlim(0.3, 300)
b.set_ylim(-0.03, 0.6)
b.set_xlabel("r / r_s  (solar-mass body; stars sit at R/r_s = 2.4×10⁵ (Sun), 7.2×10⁸ (Earth))")
b.set_ylabel("σ")
b.grid(color=GRID, lw=0.7)
b.legend(loc="upper right", bbox_to_anchor=(1.0, 0.66), fontsize=8, frameon=False)
b.set_title("B. Horizon row: □σ = 0 outside the mass gives A + B ln(1 − r_s/r)", loc="left", fontsize=10.5, color=INK)
for s in ("top", "right"):
    b.spines[s].set_visible(False)

fig.suptitle("Gravity Probe A excludes every σ clock law that carries ξ by more than 3 orders; only σ = GM/rc² read as "
             "GR notation survives,\nand □σ = −ξR yields no finite horizon value (the code's ξ/2 is a clamp)",
             fontsize=12, color=INK, x=0.01, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.92])
p = os.path.join(OUT, "U4_normalisation_table.png")
fig.savefig(p, dpi=150, facecolor=SURF)
print(p)
