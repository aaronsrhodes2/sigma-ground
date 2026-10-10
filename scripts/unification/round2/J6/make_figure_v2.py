"""J6 v2 figure: reads jacobson_results_v2.json (with mutation_controls merged by combine_v2.py).
Panel a: mutation controls.  Panel b: leading-order scope (sympy expressions from D1, geometry only, R_kk = 1 units).
Panel c: Mentat closing numbers vs tolerance, baseline and mutated runs."""
import os, json
import numpy as np
import sympy as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "jacobson_results_v2.json"), encoding="utf-8"))
MC = D["mutation_controls"]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
SER = ["#2a78d6", "#eb6834", "#1baf7a", "#8b5cf6"]
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "font.size": 10, "axes.unicode_minus": True, "axes.titlesize": 10.5, "legend.frameon": False})

fig, axs = plt.subplots(1, 3, figsize=(20, 7.6), constrained_layout=True, gridspec_kw={"width_ratios": [1.0, 1.05, 1.15]})

# ---------------- (a) mutation controls ----------------
ax = axs[0]; ax.grid(False)
cols = [("baseline", "baseline"), ("unruh_x2", "A5 broken:\nT = ħκ/π"), ("chi_sign", "A6 broken:\nχ = +κλk")]
SHORT = {"[VERIFIED numerics]": "live", "[VERIFIED numerics] + [BY CONSTRUCTION]": "live + by constr.",
         "[LIT reproduced]": "LIT reproduced", "[LIT reproduced] + [BY CONSTRUCTION]": "LIT repr. + by constr."}
rows, mat, txt = [], [], []
for sid, r in MC["steps"].items():
    rows.append(f"{sid}  ({SHORT.get(r['tag'], r['tag'])})")
    mat.append([0 if r[c] == "CLOSES" else 1 for c, _ in cols]); txt.append([r[c].lower() if r[c] == "CLOSES" else "FAILS" for c, _ in cols])
tol = {k: D["mentat_numbers"][k]["tol"] for k in ("N1", "N2", "N3", "N4", "N5")}
for k, v in MC["numbers"].items():
    rows.append(f"{k}  (Mentat, code path)")
    mat.append([0 if v[c] <= tol[k] else 1 for c, _ in cols]); txt.append(["pass" if v[c] <= tol[k] else "FAILS" for c, _ in cols])
mat = np.array(mat)
ax.imshow(mat, cmap=ListedColormap([SER[0], SER[1]]), vmin=0, vmax=1, aspect="auto")
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        ax.text(j, i, txt[i][j], ha="center", va="center", fontsize=8.4, color="white",
                fontweight="bold" if mat[i, j] else "normal")
ax.set_xticks(range(len(cols))); ax.set_xticklabels([c[1] for c in cols], fontsize=8.8)
ax.xaxis.tick_top()
ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=8.4)
ax.axhline(len(MC["steps"]) - 0.5, color=INK, lw=1.4)
for j in range(1, len(cols)):
    ax.axvline(j - 0.5, color=SURF, lw=2)
ax.legend(handles=[Patch(color=SER[0], label="closes / passes"), Patch(color=SER[1], label="FAILS")],
          loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2, fontsize=8.6)
ax.set_title("a  Could it fail? Breaking the Unruh factor (A5) or the boost-field\n"
             "    sign (A6) fails every step that uses them; S1, S1b, S4, S5, S6 do\n"
             "    not use them, S3b holds for any factor, N1–N5 do not move at all", loc="left")

# ---------------- (b) leading-order scope ----------------
ax = axs[1]
FE = D["diagnostic_D1_next_order"]["figure_expressions"]
xs, gs, s0s, Rks = sp.symbols("x g s0 R_kk", positive=True)
f_grad = sp.lambdify((xs, gs), sp.sympify(FE["gradient_rel_gap(x,g)"], locals={"x": xs, "g": gs}), "numpy")
f_th2 = sp.lambdify(xs, sp.sympify(FE["theta2_rel_gap(x)"], locals={"x": xs}), "numpy")
f_sh = sp.lambdify((s0s, Rks), sp.sympify(FE["shear_rel_gap(s0,R_kk)"], locals={"s0": s0s, "R_kk": Rks}), "numpy")
x = np.logspace(-3, 0, 300)
ax.plot(x, np.abs(f_grad(x, 1.0)), color=SER[0], lw=2.0,
        label="gradient along the generator, |∂λR_kk| = R_kk^1.5:  εR₁/3R₀")
ax.plot(x, np.abs(f_th2(x)), color=SER[2], lw=2.0, label="θ²/2 focusing term (exact Raychaudhuri):  ε²R_kk/12")
ax.plot(x, np.full_like(x, f_sh(np.sqrt(0.1), 1.0)), color=SER[1], lw=2.0, ls="--",
        label="shear at p, σ(p)² = 0.1 R_kk:  σ²/R_kk (does not shrink)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_ylim(1e-8, 1)
ax.set_xlabel("horizon patch size  ε·√R_kk")
ax.set_ylabel("|relative Clausius gap|  with Einstein's equation imposed")
ax.legend(loc="lower right", fontsize=8.4)
gap3 = D["diagnostic_D1_next_order"]["gap_eps3_with_Einstein"]
ax.set_title("b  Leading order only (A8): the gap → 0 as ε → 0 for the gradient\n"
             "    (∝ ε, the largest neglected term; −A R₁ηħκ ε³/12π) and θ²/2 (∝ ε²);\n"
             "    shear at p stays, so σ(p) = 0 is chosen via P or booked as viscosity", loc="left")

# ---------------- (c) Mentat closing numbers ----------------
ax = axs[2]
nm = D["mentat_numbers"]
labels = ["N1  η/k_B = c³/4Għ  vs  1/(4 L_P²)", "N2  S_BH/(k_B A), 9 masses  vs  η/k_B",
          "N3  unruh_temperature  vs  ħa/2πck_B", "N4  hawking_temperature  vs  Unruh(κ)",
          "N5  first law Δ(Mc²)/(T_H ΔS) − 1"]
keys = ["N1", "N2", "N3", "N4", "N5"]
floor = 1e-18
ypos = np.arange(len(keys))[::-1]
base = [max(MC["numbers"][k]["baseline"], floor) for k in keys]
ax.barh(ypos, base, color=SER[0], height=0.5, label="baseline run: max |relative deviation|")
ax.scatter([max(MC["numbers"][k]["unruh_x2"], floor) for k in keys], ypos + 0.12, marker="o", s=46,
           facecolors="none", edgecolors=SER[2], linewidths=1.6, zorder=4, label="run with A5 broken (T doubled)")
ax.scatter([max(MC["numbers"][k]["chi_sign"], floor) for k in keys], ypos - 0.12, marker="s", s=40,
           facecolors="none", edgecolors=SER[3], linewidths=1.6, zorder=4, label="run with A6 broken (χ sign)")
ax.scatter([tol[k] for k in keys], ypos, marker="|", s=420, color=SER[1], linewidths=2.2, zorder=3,
           label="pre-registered tolerance")
for yp, k, b in zip(ypos, keys, base):
    v = MC["numbers"][k]["baseline"]
    ax.text(b * 2.2, yp + 0.3, "0 (bit-identical)" if v == 0 else f"{v:.1e}", va="center", fontsize=8.4, color=INK)
ax.set_yticks(ypos); ax.set_yticklabels(labels, fontsize=8.4)
ax.set_xscale("log"); ax.set_xlim(floor / 3, 1e-4)
ax.set_xlabel("|relative deviation|")
ax.set_ylim(-0.7, len(keys) + 0.9)
ax.legend(loc="upper right", fontsize=8.2)
eta = nm["N1"]["eta_over_kB_m^-2"]
ax.set_title(f"c  Mentat: η/k_B = 1/(4 l_P²) = {eta:.4e} m⁻²; all five inside\n"
             "    tolerance and identical when the derivation is broken, so they\n"
             "    check Mentat's code paths, not the derivation", loc="left")

fig.suptitle("Jacobson 1995 closes in sympy at leading order (13/13 steps): on horizons with θ(p) = σ(p) = 0, δQ = T dS "
             "for every null k leaves only G_ab + Λg_ab = (2π/ħη) T_ab;\n"
             "naming 2π/ħη = 8πG gives η = 1/(4Għ). An equation of state given entropy ∝ area, the Unruh temperature and "
             "heat = boost-energy flux: not gravity from fluid flow, not a test of the thesis.",
             fontsize=12.2, color=INK, x=0.01, ha="left")
out = os.path.join(HERE, "J6_jacobson_derivation_v2.png")
fig.savefig(out, dpi=150)
print(out)
