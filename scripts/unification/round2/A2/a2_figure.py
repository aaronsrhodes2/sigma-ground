"""A2 figure: what each sigma-action option does to the section-6 bounds (reads a2_results.json)."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "a2_results.json"), encoding="utf-8"))
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#8b5cf6"
plt.rcParams.update({"font.size": 9, "axes.unicode_minus": True, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})

order = [("O1_T_healthy", "O1 σT\nhealthy"), ("O2_T_ghost", "O2 σT\nGHOST\n(repo sign)"), ("O1n_T_healthy_inv_norm", "O1 σT\nξ/2 norm."),
         ("O3_R_zeta1", "O3 σR\nζ = 1"), ("O3_R_zeta_cassini", "O3 σR\nζ = 1.1e−4"), ("O4_RT_zeta2f", "O4 σR+σT\nζ = 2f"),
         ("O4_RT_zeta_f_plus_1e-4", "O4 σR+σT\nζ = f+10⁻⁴"), ("O5_cancel_Hsource", "O5 cancel\nζ = f(Fe),\nH source")]
X = list(range(len(order)))
FLOOR = 1e-3

fig, axes = plt.subplots(1, 3, figsize=(16.5, 7.2), gridspec_kw=dict(width_ratios=[1.25, 1.25, 1.0]))

# (a) composition-dependent bounds
ax = axes[0]
for key, lab, col, mk in (("clock", "Yb⁺/Cs clocks (Lange 2021)", S1, "o"), ("micro", "MICROSCOPE Ti–Pt (Touboul 2022)", S2, "s"),
                          ("eotwash", "Eöt-Wash Be–Ti (Schlamminger 2008)", S3, "^"), ("wd_G2938", "white-dwarf H₂ G29-38 (Bagdonaite 2014)", S4, "D")):
    ys = [D["options"][o]["exclusion_factor"][key] for o, _ in order]
    pts = [(i, y) for i, y in zip(X, ys) if y > 0]
    ax.plot([a for a, _ in pts], [b for _, b in pts], marker=mk, ls="none", ms=7, color=col, label=lab)
for i, (o, _) in enumerate(order):
    if D["options"][o]["f"] == 0:
        ax.text(i, FLOOR * 2.2, "no mass\nresponse:\nall 0", ha="center", fontsize=7, color=INK2)
ax.axhline(1, color=INK, lw=1.1)
ax.text(len(X) - 0.6, 1.4, "bound (2σ)", fontsize=8, color=INK, ha="right")
ax.set_yscale("log"); ax.set_ylim(FLOOR * 0.5, 1e12)
ax.set_xticks(X); ax.set_xticklabels([l for _, l in order], fontsize=7.2)
ax.set_ylabel("prediction ÷ measured bound  (> 1 = excluded)")
ax.set_title("(a) composition-dependent tests: set by ξ_eff and Δf alone\n(identical for O1, O2, O4 at |ξ_eff| = 0.158)", fontsize=9.5, color=INK)
ax.grid(True, color=GRID, lw=0.6); ax.legend(fontsize=7.4, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)

# (b) universal bounds
ax = axes[1]
for key, lab, col, mk in (("cassini", "Cassini |γ − 1| (Bertotti 2003)", S1, "o"), ("nordtvedt", "lunar ranging |η_N| (Hofmann & Müller 2018)", S2, "s"),
                          ("pulsar_dalpha_composition_only", "pulsar |Δα|, composition part only (Freire 2012)", S3, "^")):
    ys = [D["options"][o]["exclusion_factor"][key] for o, _ in order]
    pts = [(i, y) for i, y in zip(X, ys) if y > 0]
    ax.plot([a for a, _ in pts], [b for _, b in pts], marker=mk, ls="none", ms=7, color=col, label=lab)
ax.axhline(1, color=INK, lw=1.1)
for i, (o, _) in enumerate(order):
    if D["options"][o]["f"] == 0:
        ax.text(i, 1.3e-2, "pulsar Δα:\nno mass\nresponse", ha="center", fontsize=7, color=INK2)
ax.set_yscale("log"); ax.set_ylim(1e-2, 1e5)
ax.set_xticks(X); ax.set_xticklabels([l for _, l in order], fontsize=7.2)
ax.set_ylabel("prediction ÷ measured bound  (> 1 = excluded)")
ax.set_title("(b) universal tests: these are what the action choice moves\n(ghost flips γ − 1 to +0.91; tuning ζ → f trades Cassini for LLR)", fontsize=9.5, color=INK)
ax.grid(True, color=GRID, lw=0.6); ax.legend(fontsize=7.4, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)

# (c) range
ax = axes[2]
lam = [l / 1e3 for l in D["microscope_curve"]["lam_m"]]
ax.plot(lam, D["microscope_curve"]["ratio_field"], color=S1, lw=2, label="ξ_eff = 0.1582 (σ = 2ξGM/rc², field reading)")
ax.plot(lam, D["microscope_curve"]["ratio_inventory"], color=S2, lw=2, ls="--", label="ξ_eff = 0.0791 (σ = ξGM/rc², inventory reading)")
ax.axhline(1, color=INK, lw=1.1)
lmax = D["range_lambda_max_m"]["xi_eff=0.1582"] / 1e3
ax.axvline(lmax, color=INK2, lw=0.8, ls=":")
ax.text(lmax * 1.15, 1e-6, "λ < %.1f km\n(ξ = 0.13–0.17:\n34.4–34.8 km)" % lmax, fontsize=7.5, color=INK2)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-12, 1e12)
ax.set_xlabel("range of σ, λ = ħ/(m_σ c)   (km)")
ax.set_ylabel("MICROSCOPE prediction ÷ bound")
ax.set_title("(c) the one axis that rescues long-range σ: its range\n(paper point λ = 0.91 fm is 10¹⁹ × further left)", fontsize=9.5, color=INK)
ax.grid(True, color=GRID, lw=0.6); ax.legend(fontsize=7.4, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1, frameon=False)

fig.suptitle("A2 — No choice of σ action rescues a long-range σ with ξ ≈ 0.16: the kinetic sign, frame and σR/σT mix move only the universal "
             "tests and the ghost; the composition tests stay excluded by 10⁵–10¹¹; only a range below ~35 km escapes",
             fontsize=10.5, color=INK, wrap=True)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = os.path.join(HERE, "a2_sigma_action_options.png")
fig.savefig(out, dpi=150, facecolor=SURF)
print(out)
