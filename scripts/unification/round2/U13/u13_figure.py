"""U13 figure: one PNG, four panels, from u13_results.json and u13_supp_results.json."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import matplotlib.ticker  # noqa: E402

OUT = r"C:/Users/aaron/AppData/Local/Temp/claude/D--Aaron-development-sigma-ground/c6652910-47a9-478d-9c83-a8a8c8aaa457/scratchpad/round2/U13"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
SER = ["#2a78d6", "#eb6834", "#1baf7a", "#8b5cf6"]
plt.rcParams.update({"axes.unicode_minus": True, "font.size": 9.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.titlesize": 10.5,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False})

d = json.load(open(os.path.join(OUT, "u13_results.json"), encoding="utf-8"))
s = json.load(open(os.path.join(OUT, "u13_supp_results.json"), encoding="utf-8"))
A, B = d["analysis"]["runA"], d["analysis"]["runB"]
ALPHAS = ["0.25", "0.5", "1.0", "2.0"]

fig, axs = plt.subplots(2, 2, figsize=(12.5, 9.2))
for ax in axs.flat:
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)

# (a) ratio vs time, both protocols
ax = axs[0, 0]
t = np.array(B["t_over_tff"])
for k, a in enumerate(ALPHAS):
    r = np.array(B["ratio_by_alpha"][a], dtype=float)
    ax.plot(t[1:], r[1:], "-o", color=SER[k], lw=2, ms=5, label=f"α = {float(a):g}, dt = t_ff/600 (Run B)")
    runA = [x for x in d["runA"] if abs(x["alpha"] - float(a)) < 1e-12][0]
    ta = [row["t_over_tff"] for row in runA["rows"] if row["step"] > 0]
    ra = [row["ratio"] for row in runA["rows"] if row["step"] > 0]
    ax.plot(ta, ra, "s", mfc=SURF, mec=SER[k], mew=1.8, ms=8)
ax.plot([], [], "s", mfc=SURF, mec=INK2, mew=1.8, ms=8, label="same α, 100 cfl_timestep steps (Run A, design)")
for y, lab in ((0.1, "FORBID above 0.1"), (0.01, "ALLOW below 0.01")):
    ax.axhline(y, color=INK2, lw=1.2, ls="--")
    ax.text(1.3e-4, y * 1.12, lab, color=INK2, fontsize=8.5)
ax.plot([B["t_over_tff"][1] * 50 / 30], [B["ratio_alpha1_step50"]], "o", ms=13, mfc="none", mec=INK, mew=1.3)
ax.annotate("step 50, Run B: 0.045 (gray band)", (0.083, B["ratio_alpha1_step50"]), xytext=(2.2e-3, 0.18),
            fontsize=8.5, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
ax.annotate("step 50, Run A: 8.7e−5\n(cfl dt from iron's 5942 m/s)", (A["t_step50_over_tff"], 8.74e-5),
            xytext=(4e-4, 1.6e-3), fontsize=8.5, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(1e-4, 1.6); ax.set_ylim(1e-5, 1.5)
ax.set_xlabel("time since release, t / t_ff")
ax.set_ylabel("Σ m r·a_visc / Σ m r·a_press")
ax.set_title("(a) Artificial-viscosity share of the pressure virial, N = 64")
ax.legend(loc="lower right", fontsize=7.8, bbox_to_anchor=(1.0, 0.0))

# (b) alpha slope vs time
ax = axs[0, 1]
ax.axhspan(0.85, 1.15, color=GRID, alpha=0.6, lw=0)
ax.text(1.2e-4, 0.862, "design pass band 1.0 ± 0.15", color=INK2, fontsize=8.5, va="bottom")
sl = np.array([np.nan if v is None else v for v in B["alpha_slope_vs_time"]], dtype=float)
ax.plot(t[1:], sl[1:], "-o", color=SER[0], lw=2, ms=5, label="Run B, dt = t_ff/600")
ax.plot([A["t_step50_over_tff"], 2 * A["t_step50_over_tff"]], [A["step50"]["alpha_slope"], A["step100"]["alpha_slope"]],
        "s", mfc=SURF, mec=SER[1], mew=1.8, ms=8, label="Run A, steps 50 and 100 (1.000, by construction)")
ax.set_xscale("log"); ax.set_xlim(1e-4, 1.6); ax.set_ylim(0.6, 1.1)
ax.set_xlabel("time since release, t / t_ff")
ax.set_ylabel("d ln(ratio) / d ln α  (fit over α = 0.25…2)")
ax.set_title("(b) Linear in α until the flow feels the viscosity")
ax.legend(loc="lower left", fontsize=8)

# (c) h scan
ax = axs[1, 0]
hm = np.array([1.0, 1.5, 2.0])
s1_64 = s["S1"][0]
n64 = [x for x in s["S2"] if x["N"] == 64][0]
n512 = [x for x in s["S2"] if x["N"] == 512][0]


def norm(rows):
    r = np.array([x["ratio"] for x in rows])
    return r / r[1]


series = [("imposed homologous v = −k r, N = 64", s1_64["rows"], s1_64["h_slope"]),
          ("collapse, step 50, N = 64", n64["samples"][0]["rows"], n64["samples"][0]["h_slope"]),
          ("collapse, step 300 (0.5 t_ff), N = 64", n64["samples"][1]["rows"], n64["samples"][1]["h_slope"]),
          ("collapse, step 50, N = 512", n512["samples"][0]["rows"], n512["samples"][0]["h_slope"])]
for k, (lab, rows, slp) in enumerate(series):
    ax.plot(hm, norm(rows), "-o", color=SER[k], lw=2, ms=6, label=f"{lab}: slope {slp:.2f}")
ax.plot(hm, hm / 1.5, ls="--", color=INK2, lw=1.2, label="slope 1 (design: 1.0 ± 0.3)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks(hm); ax.set_xticklabels(["1.0", "1.5", "2.0"])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_yticks([0.7, 0.8, 0.9, 1.0, 1.2, 1.4]); ax.set_yticklabels(["0.7", "0.8", "0.9", "1.0", "1.2", "1.4"])
ax.set_xlabel("h / (V/N)^(1/3)  at fixed particle state")
ax.set_ylabel("ratio / ratio(h = 1.5 spacing)")
ax.set_title("(c) The operator scales as h; the collapsing flow does not")
ax.legend(loc="upper left", fontsize=8)

# (d) resolution
ax = axs[1, 1]
Ns = [x["N"] for x in s["S2"]]
r50 = [x["samples"][0]["ratio_alpha1"] for x in s["S2"]]
N300 = [x["N"] for x in s["S2"] if len(x["samples"]) > 1]
r300 = [x["samples"][1]["ratio_alpha1"] for x in s["S2"] if len(x["samples"]) > 1]
e50 = np.polyfit(np.log(Ns), np.log(r50), 1)[0]
e300 = np.polyfit(np.log(N300), np.log(r300), 1)[0]
ax.plot(N300, r300, "-o", color=SER[1], lw=2, ms=6, label=f"t = 0.5 t_ff (step 300): ∝ N^{e300:.2f}".replace("-", "−"))
ax.plot(Ns, r50, "-o", color=SER[0], lw=2, ms=6, label=f"t = 0.083 t_ff (step 50): ∝ N^{e50:.2f}".replace("-", "−"))
nn = np.array([64, 512])
ax.plot(nn, r50[0] * (nn / 64.0) ** (-1 / 3), ls=":", color=INK2, lw=1.2, label="∝ N^(−1/3) (h ∝ spacing)")
for y in (0.1, 0.01):
    ax.axhline(y, color=INK2, lw=1.2, ls="--")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks(Ns); ax.set_xticklabels([str(n) for n in Ns])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_ylim(5e-3, 0.6)
ax.set_xlabel("particle number N  (h = 1.5 spacing, α = 1)")
ax.set_ylabel("Σ m r·a_visc / Σ m r·a_press")
ax.set_title("(d) The share is a resolution artefact: it falls with N")
ax.text(70, 0.108, "FORBID above 0.1", color=INK2, fontsize=8.5)
ax.text(70, 0.0108, "ALLOW below 0.01", color=INK2, fontsize=8.5)
ax.legend(loc="upper right", fontsize=8)

fig.suptitle("Artificial viscosity is 5 % of the SPH pressure virial at step 50 and 43 % by one free-fall time (α = 1, N = 64):\n"
             "the scope rule forbids an SPH σ- or Tolman-source; the design's 100 CFL steps read 10⁻⁴ only because the cloud had not moved",
             fontsize=11.5, color=INK, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(os.path.join(OUT, "u13_av_share.png"), dpi=150)
print("saved")
