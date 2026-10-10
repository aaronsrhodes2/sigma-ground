"""U14G v2 figure: one PNG from u14g_results_v2.json (no hard-coded results)."""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

OUT = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(OUT, "u14g_results_v2.json"), encoding="utf8"))

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#8b5cf6"]
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.unicode_minus": True, "font.size": 10, "legend.frameon": False})


def m(s):
    return s.replace("-", "−")


fig, axs = plt.subplots(2, 2, figsize=(13.5, 9.6))
fig.suptitle("Both continuum.py defects confirmed: buoyancy is 1/3 of the value from literature α (linear α used where 3α is needed);\n"
             "SPH conduction carries an extra 1/ρ_sph (1.2×10⁻⁴ of the Fourier rate for iron in SI) "
             "and does not conserve heat when densities differ",
             fontsize=12, fontweight="bold", color=INK, y=0.995)

# ── panel A: buoyancy against literature linear alpha (the test that could fail) ──
ax = axs[0, 0]
D1 = R["D1"]
mats = []
for r in D1["rows_as_coded"]:
    if r["material"] not in mats:
        mats.append(r["material"])
for k, (key, lab) in enumerate([("rows_as_coded", "engine as coded"),
                                ("rows_counterfactual_volumetric_cascade",
                                 "control: cascade replaced in memory by 3α (no file edited)")]):
    xs, ys, lo, hi = [], [], [], []
    for i, mat in enumerate(mats):
        v = [r["R_LIT_e2e_over_3alphaL_wiki"] for r in D1[key] if r["material"] == mat and abs(r["dT_K"]) <= 10]
        mid = sum(v) / len(v)
        xs.append(i + (k - 0.5) * 0.18); ys.append(mid); lo.append(mid - min(v)); hi.append(max(v) - mid)
    ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", ms=5, color=C[k], ecolor=C[k], capsize=2, label=lab)
ax.axhline(1.0, color=INK, lw=1.2)
ax.axhline(1 / 3, color=INK2, lw=1.0, ls="--")
ax.text(-0.3, 1.0 - 0.03, "correct Boussinesq (β = 3α_L)", ha="left", va="top", color=INK, fontsize=9)
ax.text(len(mats) - 0.6, 1 / 3 + 0.025, "1/3: α_L used for β", ha="right", va="bottom", color=INK2, fontsize=9)
oc, cf = D1["original_criterion"]["R1_small_dT_range"], D1["counterfactual_R1_small_dT_range"]
ax.text(0.02, 0.60, f"v1 ratio to 'exact Archimedes' built from the cascade α:\n"
                    f"{oc[0]:.3f}–{oc[1]:.3f} as coded, {cf[0]:.3f}–{cf[1]:.3f} under the control\n"
                    f"→ 1/3 by construction, so not evidence on its own",
        transform=ax.transAxes, fontsize=8.5, color=INK2, va="center")
ax.set_xticks(range(len(mats)))
ax.set_xticklabels(mats, rotation=35, ha="right")
ax.set_ylim(0, 1.25)
ax.set_ylabel("engine buoyancy / (3 α_L[lit] ΔT g)")
ax.set_title("A  Lone-parcel buoyancy vs literature α_L (Wikipedia, 20 °C), |ΔT| ≤ 10 K", loc="left", fontsize=10.5)
ax.legend(loc="upper left", fontsize=8.5)

# ── panel B: conduction rate vs Fourier, as a function of resolution ──
ax = axs[0, 1]
conv = R["D2"]["e_convergence_supplementary"]["rows"]
kh = [r["kh"] for r in conv]
ax.semilogy(kh, [r["Q_reference_independent"] for r in conv], "o-", color=C[0],
            label="SPH form without the extra 1/ρ_i (independent Jubelgas+2004 eq. 17 code)")
ax.semilogy(kh, [r["Q_engine"] for r in conv], "s-", color=C[1], label="engine _heat_conduction, iron")
cu = R["D2"]["a_operator_lattice_copper"]
ax.semilogy([cu["kh"]], [cu["Q_engine"]], "D", color=C[2], ms=6, label="engine _heat_conduction, copper")
rho = conv[0]["rho_sph_interior_mean"]
Mm = R["D2"]["e_convergence_supplementary"]["lattice_moment_M_independent_kernel"]
ax.axhline(1.0, color=INK, lw=1.2)
ax.axhline(Mm, color=C[0], lw=0.8, ls=":")
ax.axhline(1.0 / rho, color=INK2, lw=1.0, ls="--")
ax.text(min(kh), 1.0 / rho * 0.62, f"1/ρ_sph = 1/{rho:.0f} (iron, SI)", ha="left", va="top", color=INK2, fontsize=9)
ax.text(min(kh), 1.25, "Fourier rate −κk²δT/(ρc_p)", ha="left", va="bottom", color=INK, fontsize=9)
ax.text(max(kh), Mm * 0.82, f"lattice limit k h → 0: {Mm:.3f}", ha="right", va="top", color=C[0], fontsize=8.5)
ax.set_xlabel("k h  (sinusoid on a 7×7×N lattice; smaller = better resolved)")
ax.set_ylabel("fitted dT/dt amplitude / analytic Fourier amplitude")
ax.set_ylim(3e-5, 6)
ax.set_title("B  Conduction rate against the analytic heat equation", loc="left", fontsize=10.5)
ax.legend(loc="center left", fontsize=8.3, bbox_to_anchor=(0.0, 0.40))

# ── panel C: density-scaling exponent ──
ax = axs[1, 0]
b = R["D2"]["b_density_scaling"]
pts = b["points"]
rs = [p["rho_sph_interior_mean"] for p in pts]
ax.loglog(rs, [abs(p["engine_amp_K_s"]) for p in pts], "s-", color=C[1], label=m(f"engine: slope {b['slope_engine']:.3f}"))
refp = sorted([(p["rho_sph_interior_mean"], abs(p["reference_amp_K_s"])) for p in pts if "reference_amp_K_s" in p])
ax.loglog([x for x, _ in refp], [y for _, y in refp], "o-", color=C[0],
          label=m(f"form without the extra 1/ρ_i: slope {b['slope_reference']:.3f}"))
ax.set_xlabel("SPH density ρ_sph (kg/m³), parcel masses scaled ×0.25 … ×4, κ and c_p fixed")
ax.set_ylabel("|dT/dt| amplitude (K/s)")
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xticks([2000, 4000, 8000, 16000, 32000])
ax.set_title(m("C  Heat equation: dT/dt ∝ 1/ρ (slope -1); engine slope -2 (same factor, read off line 682)"),
             loc="left", fontsize=10.2)
ax.legend(loc="lower left", fontsize=9)

# ── panel D: two-parcel energy balance, both orientations ──
ax = axs[1, 1]
tp = R["D2"]["c_two_parcel"]
cases = [("hot parcel less dense\n(pre-registered)", tp["hot_less_dense_preregistered"]),
         ("hot parcel denser\n(swapped)", tp["hot_denser_new"])]
w = 0.34
for k, (key, lab) in enumerate([("gain_over_loss_reference", "form without the extra 1/ρ_i"),
                                ("gain_over_loss_engine", "engine _heat_conduction")]):
    vals = [c[key] for _, c in cases]
    bars = ax.bar([i + (k - 0.5) * w for i in range(2)], vals, width=w, color=C[k], label=lab)
    for bb, v in zip(bars, vals):
        ax.text(bb.get_x() + bb.get_width() / 2, v + 0.02, f"{v:.4f}", ha="center", va="bottom", fontsize=9.5)
for i, (_, c) in enumerate(cases):
    ax.text(i + 0.5 * w, 0.30, m(f"net {c['net_heat_engine_W']:+.3f} W\nhot loses {abs(min(c['heat_rate_engine_W'])):.2f} W\n"
                                 f"ρ_hot/ρ_cold\n= {c['rho_hot_over_rho_cold']:.4f}"),
            ha="center", va="center", fontsize=8.5, color=SURF)
ax.axhline(1.0, color=INK, lw=1.2)
ax.set_xticks([0, 1])
ax.set_xticklabels([lab for lab, _ in cases])
ax.set_ylim(0, 1.55)
ax.set_ylabel("heat gained by cold / heat lost by hot parcel")
ax.set_title("D  Two iron parcels (10 and 12.5 mm), 1000 K and 300 K: heat destroyed or created",
             loc="left", fontsize=10.2)
ax.legend(loc="upper left", fontsize=9)

for a in axs.flat:
    a.set_axisbelow(True)
fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=2.0)
fig.savefig(os.path.join(OUT, "u14g_defects_v2.png"), dpi=130)
print("saved", os.path.join(OUT, "u14g_defects_v2.png"))
