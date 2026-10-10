"""Figure 2 — the five-column similarity/difference matrix (Stage 2). Verdict per row is ordinal
(identical > analogous > different) and is encoded by an ordinal single-hue ramp plus the verdict word itself."""
import textwrap, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
VERDICT = {"identical": "#1c5cab", "analogous": "#5598e7", "different": "#b7d3f6"}   # ordinal ramp steps 550 / 350 / 150
COLS = ["fluid dynamics", "gas thermodynamics", "gravity (Newton / GR)", "solid mechanics", "nuclear / degenerate matter"]
ROWS = [
 ("governing structure", ["balance laws + EOS & transport closure", "fundamental relation S(U,V,N) + local equilibrium", "∇·T = 0 (Bianchi) + Einstein eq.; Poisson in the limit", "momentum balance + Hooke / flow rule", "hydrostatic (TOV) + nuclear EOS"], "analogous"),
 ("static / steady operator", ["pressure Poisson ∇²p = −ρ∂ᵢuⱼ∂ⱼuᵢ", "steady Fourier ∇²T = −q/k", "∇²Φ = 4πGρ (+3P/c² at 1PN)", "Navier–Cauchy: vector Poisson", "∇²Φ with ρ(P)"], "identical"),
 ("Green's function", ["−1/4πr", "−1/4πr", "−1/4πr", "Kelvin solution (tensor 1/r)", "−1/4πr"], "identical"),
 ("PDE class of the dynamics", ["hyperbolic (Euler); parabolic (viscous)", "parabolic", "elliptic (Newton, instantaneous); hyperbolic (GR, waves at c)", "hyperbolic, two speeds", "ODE (hydrostatic)"], "different"),
 ("time structure", ["reversible ideal + dissipative", "irreversible (2nd law)", "reversible Hamiltonian; horizons the only entropy", "reversible elastic + irreversible plastic", "equilibrium"], "different"),
 ("variational principle", ["Hamilton (ideal); Rayleigh dissipation (viscous)", "entropy maximum", "Hamilton (N-body); Einstein–Hilbert", "elastic energy minimum", "energy minimum at fixed baryon number"], "analogous"),
 ("symmetry group", ["Galilean; scale-free (Euler) broken by viscosity", "none spatial; extensivity", "Galilean / diffeomorphism; Kepler t² ∝ r³", "Galilean + material symmetry", "—"], "different"),
 ("conserved charges", ["mass, p, E, circulation", "E", "E, p, L, Runge–Lenz", "mass, p, E", "baryon number"], "analogous"),
 ("entropy & 2nd law", ["local σ_s ≥ 0", "extensive S", "no max-S state unbounded (Antonov); BH S ∝ area", "configurational / plastic", "degenerate: S → 0"], "different"),
 ("heat capacity, extensivity", ["positive, extensive", "positive, extensive", "NEGATIVE, non-extensive", "positive", "positive (Fermi gas)"], "different"),
 ("range & sign", ["contact; pressure", "—", "1/r², attractive only, unscreened", "short-range elastic", "short-range strong; degeneracy repulsion"], "different"),
 ("characteristic speed", ["c_s", "∞ (parabolic)", "∞ (Newton) / c (GR)", "c_P, c_S", "c_s ≤ c/√3"], "different"),
 ("dimensionless crossing", ["Re, Ma, Fr", "Pe, Ra", "Jeans λ_J, virial 2K/|U|, r_s/R", "Deborah De, Weissenberg", "3P/ρc², compactness"], "analogous"),
 ("equilibrium", ["hydrostatic", "thermal", "virial; Lane–Emden; Bonnor–Ebert", "static", "TOV"], "analogous"),
 ("linear response", ["sound ω = c_s k", "diffusive ω = −iαk²", "Jeans ω² = c_s²k² − 4πGρ; GW ω = ck", "P / S waves", "—"], "analogous"),
]
nr, nc = len(ROWS), len(COLS)
fig, ax = plt.subplots(figsize=(15.5, 10.5), dpi=180); fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
ax.set_xlim(0, nc + 1.9); ax.set_ylim(-0.6, nr + 1.2); ax.axis("off")
for j, col in enumerate(COLS):
    ax.text(1.9 + j + 0.5, nr + 0.55, col, ha="center", va="center", fontsize=9.5, color=INK, fontweight="bold")
for i, (prop, cells, verdict) in enumerate(ROWS):
    y = nr - i - 0.5
    ax.add_patch(plt.Rectangle((0, y - 0.5), 1.9, 1, fc=VERDICT[verdict], ec=SURFACE, lw=2))
    ax.text(0.08, y + 0.17, prop, fontsize=8.6, color=INK if verdict == "different" else "#ffffff", va="center", fontweight="bold")
    ax.text(0.08, y - 0.2, verdict, fontsize=8.2, color=INK if verdict == "different" else "#ffffff", va="center")
    for j, cell in enumerate(cells):
        ax.add_patch(plt.Rectangle((1.9 + j, y - 0.5), 1, 1, fc=SURFACE, ec=GRID, lw=1))
        ax.text(1.9 + j + 0.5, y, "\n".join(textwrap.wrap(cell, 24)), ha="center", va="center", fontsize=7.3, color=INK2)
for v, col in VERDICT.items():
    ax.plot([], [], marker="s", ms=10, color=col, linestyle="none", label=v)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.045), ncol=3, frameon=False, fontsize=9, labelcolor=INK2)
ax.set_title("Fig. 2 — Five domains, fifteen properties: where the mathematics is the same and where it is not\n"
             "Identical only in the static operator and its Green's function (a property of every static linear field); seven structural differences are what a one-phenomenon theory must produce",
             fontsize=11, color=INK, pad=14)
fig.savefig("misc/unification/fig02_matrix.png", bbox_inches="tight", facecolor=SURFACE); print("saved misc/unification/fig02_matrix.png")
