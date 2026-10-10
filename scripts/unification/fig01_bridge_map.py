"""Figure 1 — the literature map as a bridge list: one row per bridge, a bar spanning the two domains it connects,
coloured + styled by status (theorem / derivation in a framework / analogy-duality / conjecture-critiqued), a dot where
it touches measured data. Every reference verified externally 2026-10-10 (locators in misc/UNIFICATION_PROGRAM.md §2)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
STATUS = {"theorem / exact": ("#2a78d6", "solid"), "conjecture / critiqued": ("#eb6834", (0, (2, 2))),
          "derivation in a framework": ("#1baf7a", (0, (6, 3))), "analogy / duality": ("#eda100", (0, (1, 2)))}
DOMAINS = ["solid\nmechanics", "fluid\ndynamics", "gas\nthermodynamics", "gravity\n(Newton / GR)", "horizon\nthermodynamics", "nuclear /\ndegenerate matter"]
SP = 2.8
col = {d.split("\n")[0]: SP * i for i, d in enumerate(DOMAINS)}
# (label, from, to, status, touches data)
BRIDGES = [
    ("Bekenstein 73 · Hawking 74 — S ∝ A, T ∝ κ, first law, Smarr", "gravity", "horizon", "theorem / exact", False),
    ("Jacobson 95 — δQ = TdS on local Rindler horizons ⇒ Einstein equation; Padmanabhan 10", "horizon", "gravity", "derivation in a framework", False),
    ("Verlinde 11 / 17 — entropic / emergent gravity; critiques: Kobakhidze 11, Visser 11; obstruction: Weinberg–Witten 80", "horizon", "gravity", "conjecture / critiqued", True),
    ("Damour 78 · Price–Thorne 86 — membrane paradigm: horizon as viscous conducting fluid, η = 1/16πG", "gravity", "fluid", "derivation in a framework", False),
    ("BHMR 08 · BKLS 12 — fluid/gravity duality (AdS, near-horizon): Einstein ⇔ Navier–Stokes; KSS 05 bound", "gravity", "fluid", "analogy / duality", True),
    ("Unruh 81 · Visser 98 · BLV 11 — analogue gravity: sound on an acoustic metric (kinematics only)", "fluid", "gravity", "analogy / duality", True),
    ("Weinfurtner 11 · Steinhauer 16 · Muñoz de Nova 19 — stimulated / thermal analogue Hawking emission measured", "fluid", "gravity", "analogy / duality", True),
    ("Boltzmann → Chapman–Enskog → Navier–Stokes–Fourier — local equilibrium, σ_s ≥ 0", "gas", "fluid", "theorem / exact", False),
    ("virial · Jeans 02 · Lynden-Bell–Wood 68 · Thirring 70 · Padmanabhan 90 — negative C_V, no max-entropy state", "gas", "gravity", "theorem / exact", False),
    ("Bondi 52 — exact transonic accretion; TOV / Chandrasekhar: Mentat gap", "nuclear /", "gravity", "theorem / exact", True),
    ("viscoelasticity (Maxwell) · Deborah number · Navier–Cauchy = vector Poisson", "solid", "fluid", "theorem / exact", False),
    ("Sakharov 67 — induced gravity: Einstein–Hilbert action as vacuum response", "gas", "horizon", "conjecture / critiqued", False),
    ("McGaugh–Lelli–Schombert 16 — radial acceleration relation, g† = 1.20e-10 m/s² (SPARC)", "gravity", "gravity", "theorem / exact", True),
]
fig, ax = plt.subplots(figsize=(14, 9), dpi=200); fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
n = len(BRIDGES); xs = [SP * i for i in range(len(DOMAINS))]
for i, x in enumerate(xs):
    ax.axvline(x, color=GRID, lw=1, zorder=1)
    ax.text(x, n + 0.6, DOMAINS[i], ha="center", va="bottom", fontsize=8.4, color=INK, fontweight="bold")
for row, (label, a, b, status, data) in enumerate(BRIDGES):
    y = n - row; c, ls = STATUS[status]; xa, xb = col[a], col[b]
    if xa == xb:   # a data anchor on one domain
        ax.plot([xa], [y], marker="D", ms=7, color=c, zorder=3)
    else:
        ax.plot([xa, xb], [y, y], color=c, linestyle=ls, lw=3, solid_capstyle="round", zorder=3)
        ax.plot([xa, xb], [y, y], marker="o", ms=6, color=c, linestyle="none", zorder=4)
    if data:
        ax.plot([(xa + xb) / 2], [y], marker="o", ms=5, color=INK, zorder=5)
    ax.text(SP * (len(DOMAINS) - 1) + 0.55, y, label, ha="left", va="center", fontsize=8.3, color=INK2)
for s, (c, ls) in STATUS.items():
    ax.plot([], [], color=c, linestyle=ls, lw=3, label=s)
ax.plot([], [], marker="o", ms=5, color=INK, linestyle="none", label="touches measured data")
ax.legend(loc="lower left", bbox_to_anchor=(0.0, -0.09), ncol=5, frameon=False, fontsize=8.5, labelcolor=INK2)
ax.set_xlim(-1.0, SP * (len(DOMAINS) - 1) + 10.5); ax.set_ylim(0.2, n + 1.8); ax.axis("off")
ax.set_title("Fig. 1 — What the literature has already built between the domains (every reference verified 2026-10-10)\n"
             "Only the solid blue bridges are theorems; every route that would make gravity itself thermodynamic or fluid is a framework derivation, an analogy, or a contested conjecture",
             fontsize=10.5, color=INK, pad=12, loc="left")
fig.savefig("misc/unification/fig01_bridge_map.png", bbox_inches="tight", facecolor=SURFACE); print("saved")
