"""U8 figure: one PNG, three panels, read from u8_results.json (no physics computed here)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = r"C:/Users/aaron/AppData/Local/Temp/claude/D--Aaron-development-sigma-ground/c6652910-47a9-478d-9c83-a8a8c8aaa457/scratchpad/round2/U8"
R = json.load(open(os.path.join(HERE, "u8_results.json"), encoding="utf8"))
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#8b5cf6"
plt.rcParams.update({"axes.unicode_minus": True, "font.size": 9.5, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.facecolor": SURF, "figure.facecolor": SURF})
MINUS = "\u2212"


def fmt(x, nd=3):
    s = f"{x:.{nd}f}"
    return s.replace("-", MINUS)


A, B = R["partA"], R["partB"]
N = np.array(A["N"], float)
fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4))

# (a) extensivity
ax = axes[0]
def norm(v):
    v = np.abs(np.asarray(v, float))
    return v / v[0]
ax.plot(N, norm(A["U_mentat"]), "-", color=C1, lw=1.6,
        label=f"|U| Mentat (3/5)GM²/R  slope {fmt(A['slope_U_mentat'])}")
ax.plot(N, norm(A["U_engine_pe_j"]), "o", color=C1, ms=6, mfc="none", mew=1.6,
        label=f"|U| engine pair sum (seed 7)  slope {fmt(A['slope_U_engine_pe_j'])}")
ax.plot(N, norm(A["E_P_Mcs2"]), "-", color=C2, lw=1.6,
        label=f"pressure energy M c_s² = Σ mP/ρ  slope {fmt(A['slope_E_P_Mcs2'])}")
ax.plot(N, norm(A["kernel_pair_energy_sum_m_rho"]), "s", color=C3, ms=5,
        label=f"control: kernel pair energy Σ m ρ_SPH  slope {fmt(A['slope_kernel_pair_energy_sum_m_rho'])}")
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xlabel("number of parcels N at fixed density (R ∝ N^1/3)")
ax.set_ylabel("energy relative to its N = 32 value")
ms = A["multiseed"]
ax.set_title(f"(a) gravity's energy is non-extensive\n10-seed |U| slope {fmt(ms['slope_U_mean'])} ± {ms['slope_U_sd']:.3f}",
             fontsize=10, color=INK)
ax.grid(True, color=GRID, lw=0.7, which="both"); ax.legend(fontsize=7.6, loc="upper left", frameon=False)

# (b), (c) range
FLOOR = 1e-9
for ax, key, lab in ((axes[1], "N256", "(b) N = 256 (design run)"), (axes[2], "N2048", "(c) N = 2048 (declared supplement)")):
    b = B[key]
    r = np.array(b["r_over_h"])
    sel = r > 0
    g0 = np.array(b["eps0"]["gravity_dA"]); g5 = np.array(b["eps0.5h"]["gravity_dA"])
    pi_ = np.array(b["eps0"]["pressure_isothermal"]["dA"])
    o = np.argsort(r[sel])
    ax.plot(r[sel][o], (g0[sel] / g0[sel].max())[o], ".", color=C1, ms=3,
            label=f"gravity Δa, ε = 0: slope [2h,R] {fmt(b['eps0']['gravity_slope_2h_R'])}")
    lab5 = f"gravity Δa, ε = 0.5h: slope [2h,R] {fmt(b['eps0.5h']['gravity_slope_2h_R'])}"
    if b["eps0.5h"]["gravity_slope_4h_R"] is not None:
        lab5 += f"; [4h,R] {fmt(b['eps0.5h']['gravity_slope_4h_R'])}"
    ax.plot(r[sel][o], (g5[sel] / g5[sel].max())[o], ".", color=C4, ms=3, label=lab5)
    pn = pi_ / pi_.max()
    nz = pn > 0
    ax.plot(r[nz], pn[nz], ".", color=C2, ms=4, label="SPH pressure Δa (isothermal), non-zero")
    z = (~nz) & sel
    if z.any():
        ax.plot(r[z], np.full(z.sum(), FLOOR), "x", color=C2, ms=4, mew=0.9,
                label=f"SPH pressure Δa exactly 0.0 ({z.sum()} parcels; plotted at floor)")
    for xv, t in ((2, "2h"), (4, "4h")):
        ax.axvline(xv, color=INK2, lw=0.9, ls="--")
        ax.text(xv, 2.2, t, color=INK2, ha="center", va="bottom", fontsize=8.5)
    ax.axvline(b["R_over_h"], color=INK2, lw=0.9, ls=":")
    ax.text(b["R_over_h"], 2.2, "R", color=INK2, ha="center", va="bottom", fontsize=8.5)
    pz = b["eps0"]["pressure_isothermal"]
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(FLOOR / 3, 6)
    ticks = [t_ for t_ in (0.25, 0.5, 1, 2, 3, 4, 6) if t_ <= r.max() * 1.05]
    ax.set_xticks(ticks); ax.set_xticklabels([f"{t_:g}" for t_ in ticks]); ax.minorticks_off()
    ax.set_xlabel("distance from the perturbed central parcel, r / h")
    ax.set_ylabel("|Δa| / max |Δa|  (each response normalised to itself)")
    ax.set_title(f"{lab}, central mass +10 %\npressure Δa ≠ 0 in {pz['n_nonzero_beyond_2h']}/{pz['n_beyond_2h']} parcels beyond 2h, "
                 f"{pz['n_nonzero_beyond_4h']}/{pz['n_beyond_4h']} beyond 4h", fontsize=9.5, color=INK)
    ax.grid(True, color=GRID, lw=0.7, which="both"); ax.legend(fontsize=7.4, loc="lower left", frameon=False)

fig.suptitle("U8: gravity's binding energy grows as N^5/3 while pressure energy grows as N; a central mass change is felt as r⁻² "
             "across the whole cloud by gravity,\nbut by SPH pressure only out to 4h (two kernel hops), not the design's 2h. "
             "All of it follows from the engines' force laws: engine validation, not evidence about unification.",
             fontsize=10.5, color=INK, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.945))
out = os.path.join(HERE, "u8_range_extensivity.png")
fig.savefig(out, dpi=150, facecolor=SURF)
print(out)
