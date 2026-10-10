"""J6 figure: reads jacobson_results.json (Mentat numbers) and draws the one-figure reading.
Panel a is pure geometry (the exact Raychaudhuri solution, no physical constants).
"""
import os, json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "jacobson_results.json"), encoding="utf-8"))
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
SER = ["#2a78d6", "#eb6834", "#1baf7a", "#8b5cf6"]
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "font.size": 10, "axes.unicode_minus": True, "axes.titlesize": 10.5,
                     "legend.frameon": False})

fig, axs = plt.subplots(1, 3, figsize=(18, 6.8), constrained_layout=True,
                        gridspec_kw={"width_ratios": [1, 1, 1.15]})

# (a) focusing: exact Raychaudhuri area change over the leading-order one, with and without shear at p
ax = axs[0]
x = np.logspace(-3, 0.15, 400)            # eps * sqrt(R_kk)
def dA_exact(x, s2):                      # R_kk = 1 units; theta' = -theta^2/2 - (1 + s2), theta(0)=0
    Rp = 1.0 + s2
    return -2.0 * np.log(np.cos(np.sqrt(Rp / 2.0) * x))
lin = x**2 / 2.0
for i, s2 in enumerate([0.0, 0.1, 0.3]):
    lab = ("σ(p) = 0: exact focusing (θ²/2 kept)" if s2 == 0 else f"σ(p)² = {s2:g} R_kk held along the patch")
    ax.plot(x, dA_exact(x, s2) / lin, color=SER[i], lw=2.0, label=lab)
ax.plot(x, 1 + x**2 / 12.0, color=SER[3], lw=1.3, ls="--", label="series 1 + ε²R_kk/12 (sympy = Mentat)")
ax.set_xscale("log")
ax.set_xlabel("horizon patch size  ε·√R_kk")
ax.set_ylabel("δA / (ε² R_kk A / 2)   (Clausius-inferred 8πG over true)")
ax.set_ylim(0.95, 1.45)
ax.set_title("a  Local limit: the θ²/2 term drops out as ε → 0;\n"
             "    shear at p does not, so σ(p) = 0 must be assumed\n"
             "    (equilibrium) or booked as viscosity (EGJ 2006)", loc="left")
ax.legend(loc="upper left", fontsize=8.6)

# (b) Unruh line with Hawking temperatures placed at the Schwarzschild surface gravity
ax = axs[1]
N3, N4 = D["mentat_numbers"]["N3"], D["mentat_numbers"]["N4"]
hb, c, kB = D["constants"]["hbar"][0], D["constants"]["c"][0], D["constants"]["k_B"][0]
aa = np.logspace(-1, 46, 200)
ax.plot(aa, hb * aa / (2 * math.pi * c * kB), color=SER[0], lw=1.6,
        label="ħa/(2πck_B), Mentat G, ħ, c, k_B")
ax.scatter([r["a_m_s2"] for r in N3["rows"]], [r["T_mentat_K"] for r in N3["rows"]], s=40, color=SER[1],
           zorder=3, label="Mentat unruh_temperature(a)")
ax.scatter([r["kappa_m_s2"] for r in N4["rows"]], [r["T_hawking_K"] for r in N4["rows"]], s=70, marker="D",
           facecolors="none", edgecolors=SER[2], linewidths=1.6, zorder=4,
           label="Mentat hawking_temperature(M) at a = κ = c²/2r_s")
ge = N3["g_earth_m_s2"]; Te = N3["T_unruh_earth_K"]
ax.annotate(f"Earth's g: {Te:.2e} K", (ge, Te), xytext=(ge * 30, Te * 1e-4), color=INK2, fontsize=8.6,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
for r in N4["rows"]:
    if abs(r["M_kg"] - 1.989e30) / 1.989e30 < 1e-6:
        ax.annotate(f"1 M☉ hole: {r['T_hawking_K']:.3e} K", (r["kappa_m_s2"], r["T_hawking_K"]),
                    xytext=(r["kappa_m_s2"] * 1e3, r["T_hawking_K"] * 1e-6), color=INK2, fontsize=8.6,
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("acceleration / surface gravity  a  (m s⁻²)")
ax.set_ylabel("temperature  T  (K)")
ax.set_title("b  The temperature input is one line, T = ħa/2πck_B:\n"
             "    Mentat's Unruh tool equals the formula exactly;\n"
             f"    Hawking T equals Unruh T at κ = c²/2r_s to {N4['max_rel']:.1e}", loc="left")
ax.legend(loc="upper left", fontsize=8.6)

# (c) closing-number residuals against the pre-registered tolerances
ax = axs[2]
nm = D["mentat_numbers"]
items = [("N1  η/k_B = c³/4Għ  vs  1/(4 L_P²)", nm["N1"]["rel"], nm["N1"]["tol"]),
         ("N2  S_BH/(k_B A) (8 masses)  vs  η/k_B", nm["N2"]["max_rel"], nm["N2"]["tol"]),
         ("N3  unruh_temperature  vs  ħa/2πck_B", nm["N3"]["max_rel"], nm["N3"]["tol"]),
         ("N4  hawking_temperature  vs  Unruh(κ)", nm["N4"]["max_rel"], nm["N4"]["tol"]),
         ("N5  first law Δ(Mc²)/(T_H ΔS)  − 1", nm["N5"]["max_rel"], nm["N5"]["tol"])]
floor = 1e-18
ypos = np.arange(len(items))[::-1]
vals = [max(v, floor) for _, v, _ in items]
ax.barh(ypos, vals, color=SER[0], height=0.55, label="measured max |relative deviation|")
ax.scatter([t for _, _, t in items], ypos, marker="|", s=420, color=SER[1], linewidths=2.2, zorder=3,
           label="pre-registered tolerance")
for yp, (_, v, _) in zip(ypos, items):
    ax.text(max(v, floor) * 1.8, yp, "0 (exact)" if v == 0 else f"{v:.1e}", va="center", fontsize=8.6, color=INK)
ax.set_yticks(ypos); ax.set_yticklabels([s for s, _, _ in items], fontsize=8.6)
ax.set_xscale("log"); ax.set_xlim(floor / 3, 1e-5)
ax.set_xlabel("|relative deviation|")
eta = nm["N1"]["eta_over_kB_m^-2"]
ax.set_title(f"c  Closing numbers from Mentat: η/k_B = 1/(4 l_P²) = {eta:.4e} m⁻²;\n"
             "    all five inside the pre-registered tolerance\n"
             "    (code-path consistency, not independent evidence)", loc="left")
ax.set_ylim(-0.7, len(items) + 0.4)
ax.legend(loc="upper right", fontsize=8.6)

nclose = sum(1 for s in D["steps"] if s["status"] == "CLOSES")
fig.suptitle(f"Jacobson 1995 closes in sympy ({nclose}/{len(D['steps'])} steps, every residual 0): δQ = T dS on every "
             "local Rindler horizon forces G_ab + Λg_ab = 8πG T_ab with η = 1/(4Għ).\n"
             "It is an equation of state given horizon entropy ∝ area, the Unruh temperature and σ(p) = 0; "
             "it is not gravity derived from fluid flow.", fontsize=12.5, color=INK, x=0.01, ha="left")
out = os.path.join(HERE, "J6_jacobson_derivation.png")
fig.savefig(out, dpi=150)
print(out)
