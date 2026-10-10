"""U3 figure — one PNG from u3_results.json (Agg)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "u3_results.json"), encoding="utf-8"))
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#8b5cf6"]
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.8, "axes.unicode_minus": True, "font.size": 10, "legend.frameon": False})
ch = R["choices"]; tc = ch["t_cross_design_s"]; tr_u = ch["trelax_over_tcross"]
seeds = [str(s) for s in ch["seeds"]]
S = R["summary"]
pre, ext = S["pre"], S.get("ext")
ext_end = min(ch["ext_last_step_available"].values()) * ch["dt_s"] / tc   # extended window end actually reached

fig, axs = plt.subplots(1, 3, figsize=(16.5, 5.4), gridspec_kw=dict(width_ratios=[1, 1.35, 1.1]))

# (a) dK vs dE, pre-registered window
ax = axs[0]
allx = []
for i, s in enumerate(seeds):
    p = R["per_seed"][s]["pre"]
    x = np.array(p["dE"]) / 1e9; y = np.array(p["dK_raw"]) / 1e9; allx += list(x)
    ax.plot(x, y, "o", color=COL[i], ms=7, label=f"seed {s}: slope {p['slope_raw']:.3f}".replace("-", "−"))
xx = np.linspace(min(allx) * 1.1, max(allx) * 1.1, 10)
ax.plot(xx, -xx, "--", color=INK, lw=1.2, label="virial theorem, slope −1 (gravity)")
ax.plot(xx, xx, ":", color=INK2, lw=1.4, label="ideal gas, slope +1")
ax.axhline(0, color=INK2, lw=0.6); ax.axvline(0, color=INK2, lw=0.6)
ax.set_xlabel("injected energy ΔE (10⁹ J)")
ax.set_ylabel("Δ⟨K⟩ vs δ = 0 control (10⁹ J)")
ax.set_title("(a) pre-registered window, 2–3 design t_relax", fontsize=10, color=INK)
ax.legend(fontsize=8.5, loc="upper right")

# (b) slope vs time after injection
ax = axs[1]
for i, s in enumerate(seeds):
    sr = R["series"][s]
    ax.plot(sr["t_over_tc_run"], sr["slope_run"], color=COL[i], lw=1.8, label=f"seed {s}, running mean over 1 t_relax")
inst = np.mean([R["series"][s]["slope_inst"] for s in seeds], axis=0)
ax.plot(R["series"][seeds[0]]["t_over_tc_inst"], inst, color=INK2, lw=0.7, alpha=0.8, label="instantaneous, mean of seeds")
ax.axhline(-1, color=INK, ls="--", lw=1.2, label="−1 (virial)")
ax.axhline(1, color=INK2, ls=":", lw=1.4, label="+1 (ideal gas)")
ax.axvspan(2 * tr_u, 3 * tr_u, color=COL[3], alpha=0.10, lw=0)
ax.axvspan(3 * tr_u, ext_end, color=COL[1], alpha=0.07, lw=0)
ax.text(2.5 * tr_u, 1.25, "pre-reg.\nwindow", ha="center", va="bottom", fontsize=8, color=INK2)
ax.text(0.5 * (3 * tr_u + ext_end), 1.25, "extended\nwindow", ha="center", va="bottom", fontsize=8, color=INK2)
ax.set_ylim(-2.2, 1.7)
ax.set_xlabel("time after injection (design t_cross = √(b³/GM) = %.0f s)" % tc)
ax.set_ylabel("slope d⟨K⟩/dE across the four injected branches")
ax.set_title("(b) +1 at the kick, through \u22121, then an overshoot: still breathing", fontsize=10, color=INK)
ax.legend(fontsize=8, loc="lower right", ncol=2)

# (c) exact decomposition per window: d<K>/dE = -1 + dL/dE (breathing, Lagrange-Jacobi) + d<U_s - W_s>/dE (softening)
ax = axs[2]
DC = json.load(open(os.path.join(HERE, "u3_decomposition.json"), encoding="utf-8"))["summary"]
wins = [("pre", "pre-registered\n2–3 t_relax"), ("ext", "extended\n3–%.2f t_relax" % (ext_end / tr_u)),
        ("long_posthoc", "post hoc\n2–%.2f t_relax" % (ext_end / tr_u))]
ax.axhspan(-1.10, -0.90, color=COL[2], alpha=0.12, lw=0, label="SEPARATE band [−1.10, −0.90]")
ax.axhline(-1, color=INK, ls="--", lw=1.0)
for j, (w, lab) in enumerate(wins):
    for k, (key, name, c, off) in enumerate([("slope_K", "raw d⟨K⟩/dE (the test)", COL[0], -0.18),
                                             ("slope_K_minus_halfL", "design's LJ-corrected (half the breathing term removed)", COL[1], 0.0),
                                             ("slope_K_minus_L", "breathing term fully removed (by construction)", COL[3], 0.18)]):
        v = DC[w][key]
        ax.errorbar(j + off, v["mean"], yerr=v["std"], fmt="o", color=c, ms=6, capsize=3, lw=1.3, label=name if j == 0 else None)
        for y in v["per_seed"]:
            ax.plot(j + off + 0.06, y, ".", color=c, ms=3, alpha=0.6)
ax.set_xticks(range(len(wins))); ax.set_xticklabels([l for _, l in wins], fontsize=8.5)
ax.set_xlim(-0.5, len(wins) - 0.5); ax.set_ylim(-1.75, -0.6)
ax.set_ylabel("slope d⟨K⟩/dE (mean ± seed std; dots = seeds)")
ax.set_title("(c) the miss from −1 is exactly the breathing term", fontsize=10, color=INK)
ax.legend(fontsize=7.5, loc="lower left")

def fm(x):
    return f"{x:.2f}".replace("-", "−")


m, sd = pre["slope_raw"]["mean"], pre["slope_raw"]["std"]
title = ("Heating the N\u2009=\u2009128 Plummer cluster cools it (a gas would warm): d\u27e8K\u27e9/dE = %s \u00b1 %s "
         "in the pre-registered window, negative in every seed" % (fm(m), fm(sd)))
if ext:
    title += ("\nbut its size is not resolved: seed std %s > 0.05 (pre-registered verdict %s); extended window %s \u00b1 %s.\n"
              "The cluster is still breathing, and the miss from \u22121 is exactly that Lagrange\u2013Jacobi term" % (
                  fm(sd), pre["verdict_raw"], fm(ext["slope_raw"]["mean"]), fm(ext["slope_raw"]["std"])))
fig.suptitle(title, fontsize=12, color=INK, x=0.01, y=0.985, ha="left", va="top")
fig.tight_layout(rect=(0, 0, 1, 0.85))
fig.subplots_adjust(top=0.80)
out = os.path.join(HERE, "u3_figure.png")
fig.savefig(out, dpi=150, facecolor=SURF)
print(out)
