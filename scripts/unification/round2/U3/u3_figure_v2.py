"""U3 repair (v2) figure - one PNG from u3_results_v2.json (Agg). Four panels, one y-axis each."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "u3_results_v2.json"), encoding="utf-8"))
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#8b5cf6"]
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.8, "axes.unicode_minus": True, "font.size": 10, "legend.frameon": False})
ch = R["choices"]; tr_u = ch["trelax_over_tcross"]; seeds = [str(s) for s in ch["seeds"]]
S = R["summary"]; pre, ext = S["pre"], S["ext"]
ext_end = ch["N_ext_reached"] / 300
T_orb = float(R["mentat_leaves"]["T_orb_r_half_units"]["value"])
fm = lambda x, n=2: f"{x:.{n}f}".replace("-", "−")

fig, axs = plt.subplots(2, 2, figsize=(15.5, 11.0))

# (a) dK vs dE, pre-registered window
ax = axs[0, 0]; allx = []
for i, s in enumerate(seeds):
    p = R["per_seed"][s]["pre"]
    xx_ = np.array(p["dE"]) / 1e9; yy_ = np.array(p["dK_raw"]) / 1e9; allx += list(xx_)
    ax.plot(xx_, yy_, "o", color=COL[i], ms=7, label=f"seed {s}: slope {fm(p['slope_raw'])}")
xx = np.linspace(min(allx) * 1.1, max(allx) * 1.1, 10)
ax.plot(xx, -xx, "--", color=INK, lw=1.2, label="virial theorem, slope −1 (gravity)")
ax.plot(xx, xx, ":", color=INK2, lw=1.4, label="ideal gas, slope +1 (formula)")
ax.axhline(0, color=INK2, lw=0.6); ax.axvline(0, color=INK2, lw=0.6)
ax.set_xlabel("injected energy ΔE (10⁹ J)")
ax.set_ylabel("Δ⟨K⟩ against the δ = 0 control (10⁹ J)")
ax.set_title("(a) pre-registered window (2–3 design t_relax): all 12 kicked branches cooled when heated",
             fontsize=10, color=INK, loc="left")
ax.legend(fontsize=8.5, loc="upper right")

# (b) slope vs time after injection
ax = axs[0, 1]
for i, s in enumerate(seeds):
    sr = R["series"][s]
    ax.plot(sr["t_run"], sr["slope_run"], color=COL[i], lw=1.8, label=f"seed {s}, running mean over 1 design t_relax")
inst = np.mean([R["series"][s]["slope_inst"] for s in seeds], axis=0)
ax.plot(R["series"][seeds[0]]["t_inst"], inst, color=INK2, lw=0.8, alpha=0.85, label="instantaneous, mean of 3 seeds")
ax.axhline(-1, color=INK, ls="--", lw=1.2, label="−1 (virial)")
ax.axhline(1, color=INK2, ls=":", lw=1.4, label="+1 (ideal gas)")
ax.axvspan(2 * tr_u, 3 * tr_u, color=COL[3], alpha=0.10, lw=0)
ax.axvspan(3 * tr_u, ext_end, color=COL[1], alpha=0.07, lw=0)
ax.text(2.5 * tr_u, 1.25, "pre-reg.\nwindow", ha="center", va="bottom", fontsize=8, color=INK2)
ax.text(0.5 * (3 * tr_u + ext_end), 1.25, "extended window\n(cut at 4.74 of 6 t_relax)", ha="center", va="bottom", fontsize=8, color=INK2)
ax.set_ylim(-2.2, 1.75); ax.set_xlim(0, 12.6)
ax.set_xlabel("time after injection (design units √(b³/GM) = %.0f s; T_orb(r_half) = %.1f units)" % (ch["t_cross_design_s"], T_orb))
ax.set_ylabel("slope d⟨K⟩/dE across the four kicked branches")
ax.set_title("(b) +1 at the kick, negative after 1.9–2.3 units, near −1 at 5–7, below −1 after 8",
             fontsize=10, color=INK, loc="left")
ax.legend(fontsize=8, loc="lower left", ncol=2)

# (c) expansion response: moment of inertia (halo-weighted) and half-mass radius, per unit dE/|E|
ax = axs[1, 0]
for i, s in enumerate(seeds):
    sr = R["series"][s]
    ax.plot(sr["t_inst"], sr["dlnI"], color=COL[i], lw=1.8)
    ax.plot(sr["t_1u"], sr["dlnrh_1u"], color=COL[i], lw=1.3, ls="--")
ax.axhline(2, color=INK2, ls=":", lw=1.2); ax.axhline(1, color=INK2, ls=":", lw=1.2)
ax.text(0.2, 2.04, "homologous re-virialised value for I (2)", fontsize=8, color=INK2, va="bottom")
ax.text(0.2, 1.04, "homologous value for r_half (1)", fontsize=8, color=INK2, va="bottom")
h = [Line2D([], [], color=COL[i], lw=2, label=f"seed {s}") for i, s in enumerate(seeds)]
h += [Line2D([], [], color=INK2, lw=1.8, label="d ln I / d(ΔE/|E|), moment of inertia"),
      Line2D([], [], color=INK2, lw=1.3, ls="--", label="d ln r_half / d(ΔE/|E|), 1-unit mean")]
ax.legend(handles=h, fontsize=8, loc="upper left")
ax.set_xlim(0, 12.6); ax.set_ylim(-0.1, 3.05)
ax.set_xlabel("time after injection (design units)")
ax.set_ylabel("logarithmic size response per unit ΔE/|E|")
ax.set_title("(c) moment of inertia still rising at 12.5 units (no turnaround); r_half levels off after 5–9 units",
             fontsize=10, color=INK, loc="left")

# (d) decorrelation of the kicked branches: residual of the 4-point linear fit, vs the control's own K fluctuation
ax = axs[1, 1]
for i, s in enumerate(seeds):
    sr = R["series"][s]
    ax.plot(sr["t_1u"], sr["fit_resid_1u"], color=COL[i], lw=1.6, label=f"seed {s}: fit residual (rms, 1-unit mean)")
    ax.axhline(sr["control_sigma_over_K0"], color=COL[i], ls=":", lw=1.2)
ax.plot([], [], color=INK2, ls=":", lw=1.2, label="control branch's own K scatter (same colour)")
ax.axvspan(2 * tr_u, 3 * tr_u, color=COL[3], alpha=0.10, lw=0)
ax.set_xlim(0, 12.6); ax.set_ylim(0, 0.16)
ax.set_xlabel("time after injection (design units)")
ax.set_ylabel("rms departure from a straight line in ΔE, in units of K0")
ax.set_title("(d) the kicked branches stay on one straight line in ΔE for about 2 units, then decorrelate",
             fontsize=10, color=INK, loc="left")
ax.legend(fontsize=8, loc="upper left")

m, sem, sd = pre["slope_raw"]["mean"], pre["slope_raw"]["sem"], pre["slope_raw"]["std"]
title = ("Heating a gate-passing N = 128 Plummer cluster cooled it in all 12 branches: d⟨K⟩/dE = %s ± %s "
         "(seed SEM; seed std %s) in the pre-registered window, against +1 for an ideal gas.\n"
         "The size of the response is not resolved: seed std %s > 0.05, so the verdict is %s; the extended window gives %s ± %s.\n"
         "The kicked clusters' moment of inertia is still growing at 12.5 units and the branches decorrelate after about 2–3 units, "
         "so no breathing period or damping was measured." % (
             fm(m), fm(sem), fm(sd), fm(sd), pre["verdict_raw"], fm(ext["slope_raw"]["mean"]), fm(ext["slope_raw"]["std"])))
fig.suptitle(title, fontsize=11.5, color=INK, x=0.01, y=0.99, ha="left", va="top")
fig.tight_layout(rect=(0, 0, 1, 0.935))
fig.subplots_adjust(top=0.895, hspace=0.24)
out = os.path.join(HERE, "u3_figure_v2.png")
fig.savefig(out, dpi=140, facecolor=SURF)
print(out)
