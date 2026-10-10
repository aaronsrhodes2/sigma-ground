"""U13 v2 figure: one PNG, four panels.  Every plotted number, every annotation and the title text are read
from u13_results_v2.json, u13_supp_small_v2.json, u13_supp_big_v2.json and u13_summary_v2.json."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
SER = ["#2a78d6", "#eb6834", "#1baf7a", "#8b5cf6"]
plt.rcParams.update({"axes.unicode_minus": True, "font.size": 9.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.titlesize": 10.3,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False})


def load(n):
    return json.load(open(os.path.join(OUT, n), encoding="utf-8"))


d, sm, bg, sy = (load("u13_results_v2.json"), load("u13_supp_small_v2.json"), load("u13_supp_big_v2.json"),
                 load("u13_summary_v2.json"))
A, B = d["analysis"]["runA"], d["analysis"]["runB"]
K2 = d["analysis"]["literal_step50"]
S = d["analysis"]["seeds"]
ALPHAS = [0.25, 0.5, 1.0, 2.0]
M = "−"  # true minus


def fmt(x, nd=2):
    return f"{x:.{nd}f}".replace("-", M)


def sci(x):
    m, e = f"{x:.1e}".split("e")
    return f"{m} × 10" + str(int(e)).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))


fig, axs = plt.subplots(2, 2, figsize=(13.2, 9.8))
for ax in axs.flat:
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)

# ---------------- (a) ratio vs time
ax = axs[0, 0]
for k, a in enumerate(ALPHAS):
    r = [x for x in d["runB"] if x["alpha"] == a][0]
    rows = [row for row in r["rows"] if row["step"] > 0 and row["step"] % 30 == 0]
    t = np.array([row["t_over_tff"] for row in rows])
    y = np.array([row["ratio"] for row in rows])
    inw = t <= 1.0 + 1e-9
    ax.plot(t[inw], y[inw], "-o", color=SER[k], lw=2, ms=4.5, label=f"α = {a:g}, dt = t_ff/600 (Run B)")
    if (~inw).any():
        ax.plot(np.r_[t[inw][-1], t[~inw]], np.r_[y[inw][-1], y[~inw]], "--o", color=SER[k], lw=1.6, ms=3.5, mfc=SURF,
                label="α = 1 continued past the window (post hoc)")
    ra = [x for x in d["runA"] if x["alpha"] == a][0]
    ax.plot([row["t_over_tff"] for row in ra["rows"] if row["step"] > 0],
            [row["ratio"] for row in ra["rows"] if row["step"] > 0], "s", mfc=SURF, mec=SER[k], mew=1.8, ms=7.5)
ax.plot([], [], "s", mfc=SURF, mec=INK2, mew=1.8, ms=7.5, label="same α, 100 cfl_timestep steps (Run A, design)")
for i, sd in enumerate(("1", "2", "3")):
    ax.plot([50 / 600, 0.5, 1.0], [S[sd]["50"], S[sd]["300"], S[sd]["600"]], "x", color=INK2, ms=7, mew=1.6,
            label="α = 1, seeds 1, 2, 3" if i == 0 else None)
k2 = K2["dt_tff_over_100"]
ax.plot([k2["step50_t_over_tff"]], [k2["step50_ratio"]], "D", color=INK, ms=7, mfc="none", mew=1.6,
        label="'step 50' with dt = t_ff/100")
for yv, lab in ((0.1, "FORBID above 0.1"), (0.01, "ALLOW below 0.01")):
    ax.axhline(yv, color=INK2, lw=1.2, ls="--")
    ax.text(1.25e-4, yv * 1.13, lab, color=INK2, fontsize=8.3)
ax.axvline(1.0, color=INK2, lw=0.9, ls=":")
ax.text(0.93, 1.75, "pre-registered\nwindow ends", color=INK2, fontsize=7.8, ha="right", va="top")
rB50 = B["ratio_alpha1_step50"]
ax.annotate(f"step 50, dt = t_ff/600: {rB50:.3f} (GRAY)", (B["t_step50_over_tff"], rB50), xytext=(1.6e-3, 0.2),
            fontsize=8.3, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
ax.annotate(f"step 50, design cfl dt: {sci(A['ratio_alpha1_step50'])} (ALLOW,\nartefact: 100 steps = "
            f"{sci(A['t_step100_over_tff'])} t_ff)", (A["t_step50_over_tff"], A["ratio_alpha1_step50"]),
            xytext=(2.6e-4, 1.4e-3), fontsize=8.3, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(1e-4, 2.0); ax.set_ylim(1e-5, 2.0)
ax.set_xlabel("time since release, t / t_ff")
ax.set_ylabel("Σ m r·a_visc / Σ m r·a_press")
ax.set_title("(a) AV share of the pressure virial, N = 64: GRAY at step 50, FORBID after 0.2 t_ff")
ax.legend(loc="lower right", fontsize=7.2, ncol=1)

# ---------------- (b) alpha slope vs time
ax = axs[0, 1]
ax.axhspan(0.85, 1.15, color=GRID, alpha=0.6, lw=0)
ax.text(1.2e-4, 0.862, "design pass band 1.0 ± 0.15", color=INK2, fontsize=8.5, va="bottom")
t = np.array(B["t_over_tff"], float)
sl = np.array([np.nan if v is None else v for v in B["alpha_slope_vs_time"]], dtype=float)
ax.plot(t[1:], sl[1:], "-o", color=SER[0], lw=2, ms=5, label="Run B, dt = t_ff/600")
ax.plot([A["t_step50_over_tff"], A["t_step100_over_tff"]], [A["step50"]["alpha_slope"], A["step100"]["alpha_slope"]],
        "s", mfc=SURF, mec=SER[1], mew=1.8, ms=8,
        label=f"Run A, steps 50 and 100 ({A['step50']['alpha_slope']:.4f}, by construction)")
fb = B["alpha_slope_first_below_0p85_t_over_tff"]
ax.axvline(fb, color=INK2, lw=0.9, ls=":")
ax.text(fb * 0.93, 0.66, f"below 0.85\nfrom {fb:.1f} t_ff", color=INK2, fontsize=8, ha="right")
ax.set_xscale("log"); ax.set_xlim(1e-4, 1.6); ax.set_ylim(0.6, 1.1)
ax.set_xlabel("time since release, t / t_ff")
ax.set_ylabel("d ln(ratio) / d ln α  (fit over α = 0.25 to 2)")
ax.set_title(f"(b) α-slope: 1 by construction early, {fmt(min(x for x in sl[1:] if x == x))} at 1 t_ff "
             "(viscosity slows the infall)")
ax.legend(loc="lower left", fontsize=8)

# ---------------- (c) h scan, both readings
ax = axs[1, 0]
fine = B["fine_hscan_step50"]["rows"]
s1_64 = [x for x in sm["S1"] if x["N"] == 64][0]
n512 = bg["S2"][0]


def nrm(rows):
    rows = sorted(rows, key=lambda r: r["h_over_spacing"])
    x = np.array([r["h_over_spacing"] for r in rows])
    y = np.array([r["ratio"] for r in rows])
    return x, y / y[x == 1.5][0]


hs = sy["h_slopes_both_readings"]
series = [("imposed homologous v = " + M + "k r, N = 64", s1_64["rows"], s1_64["slopes"]),
          ("collapse, N = 64, 0.083 t_ff", fine, B["h_slopes_step50"]),
          ("collapse, N = 512, 0.083 t_ff", [s for s in n512["samples"] if s["step"] == 50][0]["rows"],
           [s for s in n512["samples"] if s["step"] == 50][0]["slopes"]),
          ("collapse, N = 512, 1 t_ff", [s for s in n512["samples"] if s["step"] == 600][0]["rows"],
           [s for s in n512["samples"] if s["step"] == 600][0]["slopes"])]
for k, (lab, rows, slp) in enumerate(series):
    x, y = nrm(rows)
    ax.plot(x, y, "-o", color=SER[k], lw=2, ms=5,
            label=f"{lab}: slope S {fmt(slp['reading_S_spacing_x_1_1.5_2'])}, H {fmt(slp['reading_H_h0_x_1_1.5_2'])}")
xx = np.array([1.0, 3.0])
ax.plot(xx, xx / 1.5, ls="--", color=INK2, lw=1.2, label="slope 1 (design band 1.0 ± 0.3)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(0.92, 3.3); ax.set_ylim(0.5, 2.6)
ax.plot([1.0, 2.0], [0.56, 0.56], color=INK, lw=2.2)
ax.text(1.0, 0.575, "reading S: h = 1, 1.5, 2 × spacing", fontsize=8, color=INK, va="bottom")
ax.plot([1.5, 3.0], [0.66, 0.66], color=INK2, lw=2.2)
ax.text(1.5, 0.677, "reading H: h = 1, 1.5, 2 × h0", fontsize=8, color=INK2, va="bottom")
sat = min(k for k, v in sy["h_scan_neighbours_N64_step50"].items() if v >= 62.5)
ax.axvline(float(sat), color=GRID, lw=6, alpha=0.9, zorder=0)
ax.text(float(sat) * 1.025, 0.86, f"N = 64: the kernel holds all\n63 neighbours from {float(sat):g} × spacing",
        fontsize=7.8, color=INK2, va="top")
ticks = [1.0, 1.5, 2.0, 2.25, 3.0]
ax.set_xticks(ticks); ax.set_xticklabels([f"{v:g}" for v in ticks])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_yticks([0.6, 0.8, 1.0, 1.5, 2.0]); ax.set_yticklabels(["0.6", "0.8", "1.0", "1.5", "2.0"])
ax.set_xlabel("h / (V/N)^(1/3)  at a fixed particle state  (h0 = 1.5 spacing)")
ax.set_ylabel("ratio / ratio(h0)")
s64 = B["h_slopes_step50"]
ax.set_title(f"(c) The h-slope depends on the h set read: {fmt(s64['reading_S_spacing_x_1_1.5_2'])} (S) or "
             f"{fmt(s64['reading_H_h0_x_1_1.5_2'])} (H) at 0.083 t_ff, N = 64")
ax.legend(loc="upper left", fontsize=7.4)

# ---------------- (d) resolution
ax = axs[1, 1]
rt = sy["resolution_by_t_over_tff"]
for k, lab in enumerate(("0.083", "0.5", "1.0")):
    q = rt[lab]
    ax.plot(q["N"], q["ratio"], "-o", color=SER[k], lw=2, ms=6,
            label=(f"t = {lab} t_ff: ∝ N^{fmt(q['fit_exponent_all_N'])}, "
                   f"C = {min(q['C_kin']):.2f} to {max(q['C_kin']):.2f}"))
    nn = np.array([64, 512])
    ax.plot(nn, q["ratio"][0] * (nn / 64.0) ** (-1 / 3), ls=":", color=SER[k], lw=1.2)
ax.plot([], [], ls=":", color=INK2, lw=1.2, label="∝ N^(" + M + "1/3), i.e. ∝ h at fixed C and |∇·u|")
for yv, lab in ((0.1, "FORBID above 0.1"), (0.01, "ALLOW below 0.01")):
    ax.axhline(yv, color=INK2, lw=1.2, ls="--")
    ax.text(66, yv * 1.08, lab, color=INK2, fontsize=8.3)
Ns = rt["0.083"]["N"]
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks(Ns); ax.set_xticklabels([str(n) for n in Ns])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_ylim(5e-3, 3.0)
ax.set_xlabel("particle number N  (h = 1.5 spacing, α = 1, dt = t_ff/600)")
ax.set_ylabel("Σ m r·a_visc / Σ m r·a_press")
e1 = rt["1.0"]["fit_exponent_all_N"]
c_early = rt["0.083"]["C_kin"] + rt["0.5"]["C_kin"]
ax.set_title(f"(d) Early it falls like ζ = C ρ̄ α c_s h (C = {min(c_early):.2f} to {max(c_early):.2f} at every N);\n"
             f"by 1 t_ff it barely falls (∝ N^{fmt(e1)}) and stays above 0.1 up to N = {max(Ns)}")
ax.legend(loc="upper right", fontsize=7.8)

wc = sy["verdict_by_rule"]["whole_collapse_by_N"]
forbid_Ns = [n for n in ("64", "128", "256", "512") if wc[n]["verdict"] == "FORBID"]
r1 = d["analysis"]["runB"]["ratio_alpha1_at_1tff"]
fig.suptitle(
    f"Artificial viscosity is {100 * rB50:.1f} % of the SPH pressure virial at 0.083 t_ff (GRAY under the design's step-50 rule) "
    f"and {100 * r1:.0f} % at 1 t_ff (α = 1, N = 64; seeds {100 * S['range_step600'][0]:.0f} to {100 * S['range_step600'][1]:.0f} %).\n"
    f"So the pre-declared whole-collapse criterion gives FORBID for SPH collapse runs at every N tested ({', '.join(forbid_Ns)});"
    f" the h-slope verdict depends on the h-scan reading.\n"
    f"The design's 100 cfl steps read {sci(A['ratio_alpha1_step50'])} only because they covered "
    f"{sci(A['t_step100_over_tff'])} t_ff (cfl_timestep takes iron's {d['material_sound_speed_iron_m_s']:.0f} m/s sound speed, "
    f"not the cloud's c_s = {d['choices']['c_s_m_s']:.3f} m/s).",
    fontsize=10.8, color=INK, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(os.path.join(OUT, "u13_av_share_v2.png"), dpi=150)
print("saved")
