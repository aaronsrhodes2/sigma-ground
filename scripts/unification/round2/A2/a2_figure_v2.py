"""A2 v2 figure (reads a2_results_v2.json only): what each sigma-action option does to the section-6 bounds, the range axis
with the finite-range Eot-Wash indicator, and the ghost onset of the repo-sign option O4 around a background sigma."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "a2_results_v2.json"), encoding="utf-8"))
V2 = D["v2"]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#8b5cf6"
plt.rcParams.update({"font.size": 9, "axes.unicode_minus": True, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})

order = [("O1_T_healthy", "O1 σT\nhealthy"), ("O2_T_ghost", "O2 σT\nghost\n(repo sign)"), ("O1n_T_healthy_inv_norm", "O1 σT\nξ/2 norm."),
         ("O3_R_zeta1", "O3 σR\nζ = 1"), ("O3_R_zeta_cassini", "O3 σR\nζ = 1.1e−4"), ("O4_RT_zeta2f", "O4 σR+σT\nζ = 2f"),
         ("O4_RT_zeta_f_plus_1e-4", "O4 σR+σT\nζ = f+10⁻⁴"), ("O5", "O5 cancel\nζ = f(Fe),\nper-test source")]
X = list(range(len(order)))


def factor(opt, key, observed=False):
    if opt == "O5":
        return V2["O5_per_test_source"]["exclusion_factor"][key]
    o = D["options"][opt]
    return (o["exclusion_factor_observed_potential"] if observed else o["exclusion_factor"])[key]


def nomass(opt):
    return opt != "O5" and D["options"][opt]["f"] == 0


fig, axes = plt.subplots(2, 2, figsize=(16.5, 13.6))

# (a) composition-dependent tests
ax = axes[0, 0]
FLOOR = 1e-4
for key, lab, col, mk in (("clock", "Yb⁺/Cs clocks (Lange 2021)", S1, "o"), ("micro", "MICROSCOPE Ti–Pt (Touboul 2022)", S2, "s"),
                          ("eotwash", "Eöt-Wash Be–Ti, λ = ∞ (Schlamminger 2008)", S3, "^"), ("wd_G2938", "white-dwarf H₂ G29-38 (Bagdonaite 2014)", S4, "D")):
    pts = [(i, factor(o, key)) for i, (o, _) in enumerate(order) if factor(o, key) > 0]
    ax.plot([a for a, _ in pts], [b for _, b in pts], marker=mk, ls="none", ms=7, color=col, label=lab)
    if key in ("clock", "micro"):
        pob = [(i + 0.18, factor(o, key, True)) for i, (o, _) in enumerate(order) if o != "O5" and factor(o, key, True) > 0]
        ax.plot([a for a, _ in pob], [b for _, b in pob], marker=mk, ls="none", ms=6, mfc="none", mec=col)
for i, (o, _) in enumerate(order):
    if nomass(o):
        ax.text(i, FLOOR * 2.5, "no mass\nresponse:\nall 0", ha="center", fontsize=7.2, color=INK2)
ax.axhline(1, color=INK, lw=1.1)
ax.text(len(X) - 0.55, 1.5, "2σ bound", fontsize=8, color=INK, ha="right")
ax.set_yscale("log"); ax.set_ylim(FLOOR * 0.5, 1e12)
ax.set_xticks(X); ax.set_xticklabels([l for _, l in order], fontsize=7.4)
ax.set_ylabel("prediction ÷ 2σ bound  (> 1 = excluded)")
ax.set_title("(a) composition tests: every option that changes masses is excluded, by ≥ 6 × 10²;\n"
             "O3 passes only because σ changes no mass (filled: bare G; hollow: against the observed potential)", fontsize=9.5)
ax.grid(True, color=GRID, lw=0.6)
h, l = ax.get_legend_handles_labels()
h.append(Line2D([], [], marker="o", ls="none", mfc="none", mec=INK2, ms=6)); l.append("hollow: clocks and MICROSCOPE × 1/(1 + α²)")
ax.legend(h, l, fontsize=7.4, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False)

# (b) universal tests
ax = axes[0, 1]
for key, lab, col, mk in (("cassini", "Cassini |γ − 1| (Bertotti 2003)", S1, "o"),
                          ("nordtvedt", "lunar ranging, Nordtvedt part |η_N| (Hofmann & Müller 2018)", S2, "s")):
    pts = [(i, factor(o, key)) for i, (o, _) in enumerate(order) if factor(o, key) > 0]
    ax.plot([a for a, _ in pts], [b for _, b in pts], marker=mk, ls="none", ms=7, color=col, label=lab)
key = "pulsar_dalpha_composition_only"
pts = [(i, factor(o, key)) for i, (o, _) in enumerate(order) if factor(o, key) > 0]
ax.plot([a for a, _ in pts], [b for _, b in pts], marker="^", ls="none", ms=8, mfc="none", mec=S3,
        label="pulsar |Δα|: composition part only, a LOWER BOUND, not a pass\n(program §6: ≥ 10³ with neutron-star sensitivities, not recomputed)")
for a, b in pts:
    ax.annotate("", xy=(a, b * 6), xytext=(a, b * 1.25), arrowprops=dict(arrowstyle="->", color=S3, lw=1))
ax.axhline(1, color=INK, lw=1.1)
ax.set_yscale("log"); ax.set_ylim(1e-3, 1e5)
ax.set_xticks(X); ax.set_xticklabels([l for _, l in order], fontsize=7.4)
ax.set_ylabel("prediction ÷ 2σ bound  (> 1 = excluded)")
ax.set_title("(b) universal tests: what the action choice moves (bare-G reading)\n"
             "the ghost flips γ − 1 to +0.91; tuning ζ → f passes Cassini but fails lunar ranging", fontsize=9.5)
ax.grid(True, color=GRID, lw=0.6); ax.legend(fontsize=7.4, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1, frameon=False)

# (c) range
ax = axes[1, 0]
lam = D["microscope_curve"]["lam_m"]
cf, ci, ce = D["microscope_curve"]["ratio_field"], D["microscope_curve"]["ratio_inventory"], V2["eotwash_finite_range_indicator"]["curve"]


def pos(xs, ys, cond=lambda x: True):
    p = [(x, y) for x, y in zip(xs, ys) if y > 0 and cond(x)]
    return [a for a, _ in p], [b for _, b in p]


ax.plot(*pos(lam, cf), color=S1, lw=2, label="MICROSCOPE, ξ_eff = 0.1582 (field reading)")
ax.plot(*pos(lam, ci), color=S2, lw=2, ls="--", label="MICROSCOPE, ξ_eff = 0.0791 (inventory reading)")
ax.plot(*pos(lam, ce, lambda x: x >= 1), color=S3, lw=2,
        label="Eöt-Wash Be–Ti at the ground, ξ_eff = 0.1582: smooth-Earth\nindicator (not a bound; Schlamminger 2008 covers 1 m to ∞)")
ax.plot(*pos(lam, ce, lambda x: x < 1), color=S3, lw=1.2, ls=":")
ax.axhline(1, color=INK, lw=1.1)
lmax = D["range_lambda_max_m"]["xi_eff=0.1582"]
ax.axvline(lmax, color=INK2, lw=0.8, ls=":")
ew_l = V2["eotwash_finite_range_indicator"]["ratio_at_lambda_star"]["uniform"]
ax.plot([lmax], [ew_l], marker="o", color=S3, ms=7)
ax.text(lmax * 0.6, ew_l * 4, "λ = 34.5 km: MICROSCOPE edge;\nEöt-Wash indicator ×%.0e" % ew_l, fontsize=7.6, color=INK2, ha="right")
ax.axvline(38.6e-6, color=S4, lw=1.2)
ax.text(38.6e-6 * 1.4, 1e-9, "inverse-square-law tests:\n|α| = 1 needs λ < 38.6 μm\n(Lee 2020); every option with\na mass response has α² ≈ 0.16–0.46",
        fontsize=7.4, color=S4)
ax.axvspan(1e-5, 1.0, color=GRID, alpha=0.35, lw=0)
ax.text(2e-3, 3e10, "below 1 m: not covered by\nSchlamminger 2008 (not mapped)", fontsize=7.4, color=INK2)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-12, 1e12); ax.set_xlim(1e-5, 1e9)
ax.set_xlabel("range of σ, λ = ħ/(m_σ c)   (m)")
ax.set_ylabel("prediction ÷ 2σ bound")
ax.set_title("(c) range: λ < 34.5 km is necessary for MICROSCOPE, not sufficient;\n"
             "the ground-level signal falls only as ~1.5λ/R⊕, so the surviving range is far shorter and unmapped", fontsize=9.5)
ax.grid(True, color=GRID, lw=0.6); ax.legend(fontsize=7.4, loc="lower right", frameon=False)

# (d) ghost onset of O4 around a background sigma
ax = axes[1, 1]
G = V2["ghost_onset"]
sg = G["sigma_grid"]
for kk, lab, cl, ce_ in (("O4_RT_zeta2f", "ζ = 2f", S1, S2), ("O4_RT_zeta_f_plus_1e-4", "ζ = f + 10⁻⁴ (tuned)", S3, S4)):
    ax.plot(sg, G["curves"][kk]["lin"], color=cl, lw=2, label=f"O4 {lab}, F = 1 + 2ζσ (the memo's Lagrangian)")
    ax.plot(sg, G["curves"][kk]["exp"], color=ce_, lw=2, ls="--", label=f"O4 {lab}, F = e^(2ζσ)")
ax.axhline(0, color=INK, lw=1.1)
NICE = {"sigma_conv = -ln xi": "σ_conv = −ln ξ", "horizon (r = r_s)": "horizon (r = r_s)"}
for i, (nm, v) in enumerate(G["landmarks"].items()):
    ax.axvline(v, color=INK2, lw=0.8, ls=":")
    ax.text(v * 1.08, 0.97 - 0.075 * i, NICE.get(nm, nm), transform=ax.get_xaxis_transform(), color=INK2, fontsize=7.4, va="top")
ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=1); ax.set_ylim(-2e4, 2e4); ax.set_xlim(1e-8, 10)
ax.set_xlabel("background σ near matter (repo sign, ξ_eff = 0.1582)")
ax.set_ylabel("K(σ) / K(0)   (σ's kinetic coefficient; < 0 = ghost)")
ax.set_title("(d) the repo's sign without a ghost (O4) holds only near σ = 0 with the linear F:\n"
             "ghost beyond σ = 0.091 (ζ = 2f, below the horizon value) and σ = 5.4 × 10⁻⁵ (tuned, inside white dwarfs)", fontsize=9.5)
ax.grid(True, color=GRID, lw=0.6); ax.legend(fontsize=7.4, loc="lower left", frameon=False)

fig.suptitle("A2 v2 — No σ action rescues a long-range σ that changes masses: every such option is excluded by ≥ 6 × 10², pure σR survives only by "
             "changing no mass,\nthe repo's sign is ghost-free only near σ = 0, and λ < 34.5 km escapes MICROSCOPE but not the finite-range "
             "Eöt-Wash indicator", fontsize=11.5, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = os.path.join(HERE, "a2_sigma_action_options_v2.png")
fig.savefig(out, dpi=140, facecolor=SURF)
print(out)
