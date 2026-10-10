"""U8 figure v2: one PNG, four panels, read from u8_results_v2.json (no physics computed here).

Changes from v1: series colours in the mandated order in every panel; formula curves drawn as
secondary-ink reference lines (by construction) so that only engine measurements carry series
colours; the two-hop reach is in the range-panel titles; panel (d) shows local exponents of the
measured energies out to N = 8192; the title states both verdicts.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = r"C:/Users/aaron/AppData/Local/Temp/claude/D--Aaron-development-sigma-ground/c6652910-47a9-478d-9c83-a8a8c8aaa457/scratchpad/round2/U8"
R = json.load(open(os.path.join(HERE, "u8_results_v2.json"), encoding="utf8"))
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#8b5cf6"
plt.rcParams.update({"axes.unicode_minus": True, "font.size": 9.5, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.facecolor": SURF, "figure.facecolor": SURF})
MINUS = "\u2212"


def fmt(x, nd=3):
    return f"{x:.{nd}f}".replace("-", MINUS)


A, A2, B = R["partA"], R["partA2_posthoc_extension"], R["partB"]
N = np.array(A["N"], float)
fig, axes = plt.subplots(2, 2, figsize=(14.5, 11.0))


def norm(v):
    v = np.abs(np.asarray(v, float))
    return v / v[0]


# (a) extensivity, N = 32..512 (pre-registered range)
ax = axes[0, 0]
ms = A["multiseed"]
ax.plot(N, norm(A["U_mentat"]), "--", color=INK2, lw=1.2,
        label=f"reference: Mentat (3/5)GM²/R, slope {fmt(A['slope_U_mentat'])} (by construction)")
ax.plot(N, norm(A["E_P_Mcs2"]), ":", color=INK2, lw=1.6,
        label=f"reference: M c_s² = isothermal Σ mP/ρ, slope {fmt(A['slope_E_P_Mcs2'])} (by construction)")
ax.plot(N, norm(A["U_engine_pe_j"]), "o", color=C1, ms=6.5,
        label=f"engine |U| pair sum, seed 7: slope {fmt(A['slope_U_engine_pe_j'])} "
              f"(10 seeds {fmt(ms['slope_U_mean'])}, SD {ms['slope_U_sd']:.3f})")
ax.plot(N, norm(A["kernel_pair_energy_sum_m_rho"]), "s", color=C2, ms=6,
        label=f"engine kernel pair energy Σ m ρ_SPH, seed 7: slope {fmt(A['slope_kernel_pair_energy_sum_m_rho'])} "
              f"(10 seeds {fmt(ms['slope_kernel_mean'])}, SD {ms['slope_kernel_sd']:.3f})")
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xlabel("number of parcels N at fixed iron density (R ∝ N^1/3)")
ax.set_ylabel("energy relative to its N = 32 value")
ax.set_title("(a) gravity's pair energy is non-extensive (N^5/3);\nthe engine's short-range energy is close to extensive",
             fontsize=10, color=INK)
ax.grid(True, color=GRID, lw=0.7, which="both"); ax.legend(fontsize=7.4, loc="upper left", frameon=False)

# (b), (c) range
FLOOR = 1e-9
for ax, key, lab in ((axes[0, 1], "N256", "(b) N = 256, the design's run"),
                     (axes[1, 0], "N2048", "(c) N = 2048, supplement declared before the run")):
    b = B[key]
    th = b["two_hop_set_check"]
    r = np.array(b["r_over_h"])
    sel = r > 0
    g0 = np.array(b["eps0"]["gravity_dA"]); g5 = np.array(b["eps0.5h"]["gravity_dA"])
    pi_ = np.array(b["eps0"]["pressure_isothermal"]["dA"])
    o = np.argsort(r[sel])
    ax.plot(r[sel][o], (g0[sel] / g0[sel].max())[o], ".", color=C1, ms=3,
            label=f"gravity Δa, ε = 0: slope over [2h, R] {fmt(b['eps0']['gravity_slope_2h_R'])}")
    lab5 = f"gravity Δa, ε = 0.5h (engine default): [2h, R] {fmt(b['eps0.5h']['gravity_slope_2h_R'])}"
    if b["eps0.5h"]["gravity_slope_4h_R"] is not None:
        lab5 += f"; [4h, R] {fmt(b['eps0.5h']['gravity_slope_4h_R'])}"
    ax.plot(r[sel][o], (g5[sel] / g5[sel].max())[o], ".", color=C2, ms=3, label=lab5)
    pn = pi_ / pi_.max()
    nz = pn > 0
    ax.plot(r[nz], pn[nz], ".", color=C3, ms=4,
            label=f"SPH pressure Δa (isothermal), non-zero: {int(nz.sum())} parcels")
    z = (~nz) & sel
    if z.any():
        ax.plot(r[z], np.full(z.sum(), FLOOR), "x", color=C4, ms=4, mew=0.9,
                label=f"SPH pressure Δa exactly 0.0: {int(z.sum())} parcels (drawn at the floor)")
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
    eq = "equals" if th["all_sets_equal"] else "differs from"
    ax.set_title(f"{lab}, central mass +10 %\npressure reaches {th['n_predicted']} of {th['n_total']} parcels "
                 f"({100 * th['fraction_of_parcels_reached']:.0f} %),\nthe non-zero set {eq} the two-hop set; "
                 + (f"{pz['n_nonzero_beyond_4h']} of {pz['n_beyond_4h']} beyond 4h" if pz["n_beyond_4h"]
                    else f"no parcel lies beyond 4h (R = {b['R_over_h']:.2f}h)"), fontsize=9.5, color=INK)
    ax.grid(True, color=GRID, lw=0.7, which="both"); ax.legend(fontsize=7.4, loc="lower left", frameon=False)

# (d) local exponents of the measured energies
ax = axes[1, 1]
mid = lambda Ns: [float(np.sqrt(Ns[i] * Ns[i + 1])) for i in range(len(Ns) - 1)]
ax.axhline(5 / 3, color=INK2, lw=1.0, ls="--")
ax.text(0.99, 5 / 3 - 0.012, "5/3", color=INK2, ha="right", va="top", fontsize=8.5,
        transform=ax.get_yaxis_transform())
ax.axhline(1.0, color=INK2, lw=1.0, ls=":")
ax.text(0.99, 1.0 + 0.012, "1 (extensive)", color=INK2, ha="right", va="bottom", fontsize=8.5,
        transform=ax.get_yaxis_transform())
ax.plot(mid(A["N"]), A["local_slopes_U_engine_pe_j"], "o-", color=C1, ms=5, lw=1.2,
        label="engine |U| pair sum, seed 7")
s7 = A2["per_seed"]["7"]
Nk7 = A["N"] + [n for n in s7["N"] if n > A["N"][-1]]
# seed-7 kernel series: Part A (N = 32..512) followed by the post hoc extension (N > 512)
kvals = A["kernel_pair_energy_sum_m_rho"] + [rw["kernel_sum_m_rho"] for rw in s7["rows"] if rw["N"] > A["N"][-1]]
loc7 = [float(np.log(kvals[i + 1] / kvals[i]) / np.log(Nk7[i + 1] / Nk7[i])) for i in range(len(Nk7) - 1)]
ax.plot(mid(Nk7), loc7, "s-", color=C2, ms=5, lw=1.2, label="engine kernel pair energy Σ m ρ_SPH, seed 7")
for sd, col in (("1", C3), ("2", C4)):
    v = A2["per_seed"][sd]
    ax.plot(mid(v["N"]), v["kernel_local_slopes"], "s-", color=col, ms=4.5, lw=1.0,
            label=f"engine kernel pair energy, seed {sd} (post hoc)")
ax.set_xscale("log", base=2)
ax.set_xlabel("N (local exponent plotted at the geometric mid-point of each N pair)")
ax.set_ylabel("local exponent  d ln E / d ln N")
tt = [rw["tait_over_Mcs2"] for rw in s7["rows"]]
sc = s7["tait_sign_change_between"]
txt = ("Tait Σ mP/ρ ÷ M c_s², seed 7, N = " + ", ".join(str(n) for n in s7["N"]) + ":\n"
       + ", ".join(fmt(v, 4) for v in tt)
       + (f"\nchanges sign between N = {sc[0][0]} and {sc[0][1]}: no power law" if sc else ""))
ax.text(0.98, 0.42, txt, transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK2)
ax.set_ylim(0.9, 1.85)
ax.set_title("(d) local exponents: gravity stays at 5/3; the short-range energy\napproaches 1 slowly (surface terms); "
             "N > 512 is a post hoc extension", fontsize=10, color=INK)
ax.grid(True, color=GRID, lw=0.7, which="both"); ax.legend(fontsize=7.6, loc="upper right", frameon=False)

vo = R["verdict_original"]["literal_outcomes_over_all_readings"]
fig.suptitle("U8: gravity's pair energy grows as N^5/3 and reaches every parcel as r⁻²; SPH pressure energy is "
             "close to extensive and its response stops at the two-hop set (4h), not the design's 2h.\n"
             f"Design's literal criterion: {' / '.join(vo)}; pre-registered corrected criterion: "
             f"{R['verdict_corrected']['outcome']}. All of it follows from how the engines are built: "
             "engine validation, not evidence about unification.",
             fontsize=10.5, color=INK, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.97))
out = os.path.join(HERE, "u8_range_extensivity_v2.png")
fig.savefig(out, dpi=150, facecolor=SURF)
print(out)
