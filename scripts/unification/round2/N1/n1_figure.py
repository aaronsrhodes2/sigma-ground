"""N1 figure: why the dense-matter work is needed (3P/rho c^2 ladder), one defect it fixes (SEMF), and the work order.
Reads n1_survey.json (all numbers there carry Mentat sources); writes n1_plan_figure.png."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
d = json.load(open(HERE / "n1_survey.json", encoding="utf8"))
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
S = ["#2a78d6", "#eb6834", "#1baf7a", "#8b5cf6"]
plt.rcParams.update({"axes.unicode_minus": True, "font.size": 10, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": INK2, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF})
fig, ax = plt.subplots(1, 3, figsize=(17, 5.6), gridspec_kw={"width_ratios": [0.8, 1.0, 1.35]})

# (a) 3P/(rho c^2) for the repo's own P6 [LIT] rows, Mentat c
rows = d["p6_rows_3P_over_rho_c2"]
labs = ["Sun core", "white dwarf centre", "neutron star centre (~2 n0)"]
vals = [rows[k]["3P_over_rho_c2"] for k in labs]
a = ax[0]
a.bar(range(3), vals, color=[S[0], S[0], S[1]], width=0.6, zorder=3)
a.set_yscale("log"); a.set_ylim(1e-7, 10)
a.axhline(1.0, color=INK2, ls="--", lw=1, zorder=2)
a.text(-0.4, 1.35, "3P = ρc²: the trace ε − 3P changes sign", color=INK2, fontsize=8.5)
for i, v in enumerate(vals):
    m, e = f"{v:.1e}".split("e"); a.text(i, v * 1.6, "$" + m + r"\times10^{" + str(int(e)) + "}$", ha="center", fontsize=9.5)
a.set_xticks(range(3)); a.set_xticklabels(["Sun core", "white dwarf\ncentre", "neutron star\ncentre"])
a.set_ylabel("3P / ρc²  (pressure share of the σ source)")
a.set_title("(a) only neutron stars reach order one\n(P6's [LIT] rows; c from Mentat)", fontsize=10, loc="left")
a.grid(axis="y", color=GRID, zorder=0)

# (b) the two existing SEMF implementations vs AME2020
b = ax[1]
r = d["semf_comparison"]["rows"]
nz = [x["A"] - 2 * x["Z"] for x in r]
b.plot(nz, [x["resid_element_py_pct"] for x in r], "o", color=S[1], label="element.py as written (asymmetry ÷ 4)", zorder=3)
b.plot(nz, [x["resid_element_py_consistent_pct"] for x in r], "s", color=S[2], label="element.py, asymmetry as its own stable_mass_number assumes", zorder=3)
b.plot(nz, [x["resid_mcp_SEMF_pct"] for x in r], "^", color=S[0], label="Mentat nuclear_binding_energy (second coefficient set)", zorder=3)
b.axhline(0, color=INK2, lw=0.8)
#
#
for x in r:
    if (x["Z"], x["A"]) in [(82, 208), (92, 238), (26, 56), (2, 4)]:
        b.annotate({(82, 208): "²⁰⁸Pb", (92, 238): "²³⁸U", (26, 56): "⁵⁶Fe", (2, 4): "⁴He"}[(x["Z"], x["A"])],
                   (x["A"] - 2 * x["Z"], x["resid_element_py_pct"]), textcoords="offset points", xytext=(4, 6), fontsize=8.5, color=INK2)
b.set_ylim(-21, 14)
b.text(2.0, -9.5, "⁴He (A = 4) sits outside any SEMF's domain;\nthe rms excludes A < 12", fontsize=8, color=INK2)
b.set_xlabel("neutron excess N − Z"); b.set_ylabel("binding energy: SEMF − AME2020 (%)")
b.set_title("(b) a defect the plan fixes: element.py's asymmetry term is 4× too weak\n(rms 5.9 % → 0.8 % for A ≥ 12; flagged, not patched)", fontsize=10, loc="left")
b.legend(fontsize=8, frameon=False, loc="lower right"); b.grid(color=GRID, zorder=0)

# (c) the work order
c = ax[2]
phases = [("P0 kernel/ode.py + cite saturation constants", 0.5, 0), ("P1 M1 Fermi gas (e, n, p, μ; T = 0 exact)", 1.25, 0),
          ("P2 M2 Lane–Emden + white dwarfs", 1.0, 1), ("P3 M5 TOV + EOS protocol", 1.5, 1),
          ("P4 M8 trace diagnostics; P7 beyond weak field", 1.0, 1), ("P5 M3 one SEMF fitted to AME2020", 0.75, 2),
          ("P6 M4 nuclear matter + β-equilibrium → NS", 2.5, 2), ("P7 M6 composition layers (crust, drip, core)", 2.0, 2),
          ("P8 M7 bag model, deconfinement, hybrids", 1.25, 2), ("P9 MCP sweep, docs, push, redeploy Mentat", 1.0, 3)]
t = 0.0
for i, (name, dur, g) in enumerate(phases):
    c.barh(i, dur, left=t, color=S[g], height=0.62, zorder=3)
    c.text(t + dur + 0.12, i, name, va="center", fontsize=8.6)
    t += dur
c.set_yticks([]); c.set_xlim(0, 22); c.set_ylim(9.7, -1.7); c.set_xlabel("working days (rough)")
for x, lab in [(4.25, "MS1 white dwarfs + free-neutron star"), (5.25, "MS2 P7 rerun on the TOV metric"),
               (8.5, "MS3 3P/ρc² ~ 1 computed, not borrowed"), (12.75, "MS5 live in Mentat")]:
    c.axvline(x, color=INK2, ls=":", lw=1, zorder=2)
c.text(4.2, -1.1, "MS1", fontsize=8, color=INK2, ha="right"); c.text(5.3, -1.1, "MS2", fontsize=8, color=INK2)
c.text(8.6, -1.1, "MS3: σ-bridge pressure term scoreable", fontsize=8, color=INK2)
c.text(12.85, -0.4, "MS5: live in Mentat", fontsize=8, color=INK2)
from matplotlib.patches import Patch
c.legend(handles=[Patch(color=S[0], label="foundations"), Patch(color=S[1], label="degenerate stars + GR"),
                  Patch(color=S[2], label="nuclear and quark matter"), Patch(color=S[3], label="integration")],
         fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4)
c.set_title("(c) the plan: 10 phases, ≈ 12.75 days; tests land with every module", fontsize=10, loc="left")
c.grid(axis="x", color=GRID, zorder=0)
for s in ["top", "right"]:
    for a_ in ax: a_.spines[s].set_visible(False)

fig.suptitle("Pressure is an order-one share of the σ source only inside neutron stars, and Mentat cannot build one yet:\n"
             "the plan goes Fermi gas → white dwarfs → TOV → nuclear EOS → layers → quark cores",
             fontsize=12.5, x=0.01, ha="left", y=1.0)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(HERE / "n1_plan_figure.png", dpi=150)
print("wrote", HERE / "n1_plan_figure.png")
