"""U14G figure: one PNG from u14g_results.json + u14g_supp_convergence.json."""
import os, json, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

OUT = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(OUT, "u14g_results.json"), encoding="utf8"))
S = json.load(open(os.path.join(OUT, "u14g_supp_convergence.json"), encoding="utf8"))

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#8b5cf6"]
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.unicode_minus": True, "font.size": 10, "legend.frameon": False})

fig, axs = plt.subplots(2, 2, figsize=(13, 9.2))
fig.suptitle("Both suspected continuum.py defects confirmed: buoyancy is 1/3 of Archimedes (linear α used for 3α),\n"
             "and SPH conduction runs 1/ρ_sph too slow (≈ 7,900× for iron in SI) and leaks heat when densities differ",
             fontsize=12.5, fontweight="bold", color=INK, y=0.995)

# ── panel A: buoyancy ratio per material ───────────────────────────────────
ax = axs[0, 0]
rows = R["D1"]["rows"]
mats = []
for r in rows:
    if r["material"] not in mats:
        mats.append(r["material"])
xpos = {m: i for i, m in enumerate(mats)}
for k, (dT, lab) in enumerate([(1.0, "ΔT = +1 K"), (10.0, "ΔT = +10 K"), (-10.0, "ΔT = −10 K"),
                               (100.0, "ΔT = +100 K")]):
    xs = [xpos[r["material"]] + (k - 1.5) * 0.15 for r in rows if r["dT_K"] == dT]
    ys = [r["R1_e2e_over_exact"] for r in rows if r["dT_K"] == dT]
    ax.plot(xs, ys, "o", ms=5, color=C[k], label=lab)
ax.axhline(1.0, color=INK, lw=1.2, ls="-")
ax.axhline(1 / 3, color=INK2, lw=1.0, ls="--")
ax.text(len(mats) - 0.6, 1.0 + 0.02, "correct physics (ratio 1)", ha="right", va="bottom", color=INK, fontsize=9)
ax.text(len(mats) - 0.6, 1 / 3 + 0.02, "1/3 = α_L / (3α_L)", ha="right", va="bottom", color=INK2, fontsize=9)
ax.set_xticks(range(len(mats)))
ax.set_xticklabels(mats, rotation=35, ha="right")
ax.set_ylim(0, 1.15)
ax.set_ylabel("engine buoyancy / exact Archimedes buoyancy")
ax.set_title("A  Buoyancy, via continuum_step on a lone parcel (cubic crystals)", loc="left", fontsize=10.5)
ax.legend(loc="center right", fontsize=8.5)

# ── panel B: conduction rate vs Fourier, as a function of resolution ─────────
ax = axs[0, 1]
kh = [r["kh"] for r in S["rows"]]
ax.semilogy(kh, [r["Q_reference"] for r in S["rows"]], "o-", color=C[0], label="SPH formula without the extra 1/ρ_i (Jubelgas+2004 eq. 17)")
ax.semilogy(kh, [r["Q_engine"] for r in S["rows"]], "s-", color=C[1], label="engine _heat_conduction")
rho = S["rows"][0]["rho_sph_mean"]
ax.axhline(1.0, color=INK, lw=1.2)
ax.axhline(1.0 / rho, color=INK2, lw=1.0, ls="--")
ax.text(max(kh), 1.0 / rho * 1.5, f"1/ρ_sph = 1/{rho:.0f} (SI)", ha="right", va="bottom", color=INK2, fontsize=9)
ax.text(max(kh), 1.25, "Fourier rate −κk²δT/(ρc_p)", ha="right", va="bottom", color=INK, fontsize=9)
ax.set_xlabel("k h  (sinusoid on a 7×7×N iron lattice; smaller = better resolved)")
ax.set_ylabel("fitted dT/dt amplitude / analytic Fourier amplitude")
ax.set_ylim(3e-5, 5)
ax.set_title("B  Conduction rate against the analytic heat equation", loc="left", fontsize=10.5)
ax.legend(loc="center left", fontsize=8.5, bbox_to_anchor=(0.0, 0.42))

# ── panel C: density-scaling exponent ─────────────────────────────────────
ax = axs[1, 0]
pts = R["D2"]["b_density_scaling"]["points"]
rs = [p["rho_sph_interior_mean"] for p in pts]
ax.loglog(rs, [abs(p["engine_amp_K_s"]) for p in pts], "s-", color=C[1],
          label=f"engine: slope {R['D2']['b_density_scaling']['slope_engine']:.3f}".replace("-", "−"))
refp = [p for p in pts if "reference_amp_K_s" in p]
a1 = R["D2"]["a_operator_lattice"]
refx = sorted([(p["rho_sph_interior_mean"], abs(p["reference_amp_K_s"])) for p in refp]
              + [(a1["rho_sph_interior_mean"], abs(a1["reference_amp_K_s"]))])
ax.loglog([x for x, _ in refx], [y for _, y in refx], "o-", color=C[0],
          label=f"formula without extra 1/ρ_i: slope {R['D2']['b_density_scaling']['slope_reference']:.3f}".replace("-", "−"))
ax.set_xlabel("SPH density ρ_sph (kg/m³), parcel masses scaled ×0.25 … ×4, κ and c_p fixed")
ax.set_ylabel("|dT/dt| amplitude (K/s)")
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xticks([2000, 4000, 8000, 16000, 32000])
ax.set_title("C  Heat equation says dT/dt ∝ 1/ρ (slope −1); engine gives slope −2", loc="left", fontsize=10.5)
ax.legend(loc="lower left", fontsize=9)

# ── panel D: two-parcel energy balance ──────────────────────────────────────
ax = axs[1, 1]
tp = R["D2"]["c_two_parcel"]
he, hr = tp["heat_rate_engine_W"], tp["heat_rate_reference_W"]
ratio_e = he[1] / -he[0]
ratio_r = hr[1] / -hr[0]
bars = ax.bar([0, 1], [ratio_r, ratio_e], color=[C[0], C[1]], width=0.55)
ax.axhline(1.0, color=INK, lw=1.2)
ax.set_xticks([0, 1])
ax.set_xticklabels(["formula without extra 1/ρ_i", "engine _heat_conduction"])
ax.set_ylim(0, 1.15)
ax.set_ylabel("heat gained by cold parcel / heat lost by hot parcel")
for b, v in zip(bars, [ratio_r, ratio_e]):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.4f}", ha="center", va="bottom", fontsize=10)
ax.text(1, 0.35, f"ρ_sph,hot / ρ_sph,cold\n= {tp['rho_sph_1']:.0f} / {tp['rho_sph_2']:.0f} = "
        f"{tp['rho_sph_1']/tp['rho_sph_2']:.4f}\n{abs(tp['net_heat_created_by_engine_W']):.2f} W of "
        f"{-he[0]:.2f} W vanishes", ha="center", va="center", fontsize=9, color=SURF)
ax.set_title("D  Two iron parcels, unequal masses (10 and 12.5 mm), 1000 K vs 300 K", loc="left", fontsize=10.5)

for a in axs.flat:
    a.set_axisbelow(True)
fig.tight_layout(rect=(0, 0, 1, 0.955))
fig.savefig(os.path.join(OUT, "u14g_defects.png"), dpi=130)
print("saved", os.path.join(OUT, "u14g_defects.png"))
