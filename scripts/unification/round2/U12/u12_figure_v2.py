"""U12 figure v2: reads u12_results_v2.json (written by u12_material_chain_v2.py) and draws one PNG."""
import json, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

OUT = os.path.dirname(os.path.abspath(__file__))
r = json.load(open(os.path.join(OUT, "u12_results_v2.json"), encoding="utf-8"))
D = r["dlnX_dsigma"]

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#8b5cf6"
plt.rcParams.update({"axes.unicode_minus": True, "font.size": 10, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF})
mn = lambda t: t.replace("-", "−")  # noqa: E731  true minus signs


def clean(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


fig = plt.figure(figsize=(17.5, 12.0))
gs = GridSpec(2, 2, figure=fig, width_ratios=[1.0, 1.0], height_ratios=[1.0, 1.0], hspace=0.38, wspace=0.42)
a1, a3, a2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])

# ---- panel A: d ln X / d sigma in the audited chain, as coded vs consistent chain
labels = ["ρ", "K = G = E", "v_L\n(acoustics)", "v_D\n(thermal)", "Θ_D", "T_melt\n(Lindemann)", "c_p\n(300 K)", "air c_s"]
coded = [D["rho"], D["K"], D["vL"], D["vD_therm"], D["ThD"], D["Tm"], D["cp"], D["cs_air"]]
cons = [D["rho"], D["K"], D["vL"], D["vD_acou"], D["ThD_cons"], D["Tm_cons"], D["cp_cons"], D["cs_air_cons"]]
x = range(len(labels))
w = 0.38
a1.bar([i - w / 2 for i in x], coded, w, color=S1, label="engine as coded")
a1.bar([i + w / 2 for i in x], cons, w, color=S2, label="consistent chain (ρ(σ), M(σ), Ar coupled) [DERIVED]")
a1.axhline(0, color=INK2, lw=0.8)
a1.set_xticks(list(x))
a1.set_xticklabels(labels, fontsize=9)
a1.set_ylabel("d ln X / dσ at σ = 0 (iron; dry air, 288 K)")
a1.set_ylim(-1.05, 1.2)
a1.grid(axis="y", color=GRID, lw=0.8)
a1.set_axisbelow(True)
gap = r["identities"]["I3_like_for_like_dlnvD_therm_minus_dlnvD_acou"]
a1.annotate(f"thermal.sound_velocity keeps ρ₀:\nI3 = {gap:.4f} = f_QCD/2,\nsame in every material [BY CONSTRUCTION]",
            xy=(3 - w / 2, D["vD_therm"]), xytext=(2.05, 0.42), fontsize=8.5, color=INK,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
a1.text(0, D["rho"] + 0.03, f"{D['rho']:+.4f}", ha="center", fontsize=8.5, color=INK2)
a1.text(2, D["vL"] - 0.09, mn(f"{D['vL']:+.4f}"), ha="center", fontsize=8.5, color=INK2)
a1.text(6 + w / 2, D["cp_cons"] - 0.09, mn(f"{D['cp_cons']:+.3f}"), ha="center", fontsize=8.5, color=INK2)
a1.text(5, 0.10, f"{D['Tm']:+.4f}\n(two omissions\ncancel)", ha="center", fontsize=8.5, color=INK2)
a1.set_title("A. Audited elastic/thermal/acoustic chain: I1, I2 hold to 2 × 10⁻⁹ (by construction);\n"
             "    I3 (thermal vs acoustics) fails by f_QCD/2, as predicted", fontsize=10.5, loc="left", color=INK)
a1.legend(frameon=False, loc="lower left", fontsize=8.5)
clean(a1)

# ---- panel C: every sigma coupling found in the engine vs the coupling needed to reach the best floor
ext = r["coupling_conventions_extended"]
pick = lambda prefix: abs(next(v["dln_dsigma"] for k, v in ext.items() if k.startswith(prefix)))  # noqa: E731
rowsC = [
    ("ρ (acoustics, QCD-fraction mass)", abs(D["rho"]), S1),
    ("v_L, v_T, v_D (consistent)", abs(D["vD_acou"]), S1),
    ("c_p (consistent)", abs(D["cp_cons"]), S1),
    ("K, T_melt (ZPE term)", abs(D["K"]), S1),
    ("air c_s", abs(D["cs_air_cons"]), S1),
    ("quartz resonator f (piezoelectricity)", pick("quartz AT-cut"), S1),
    ("E_coh, ZPE law (mechanical, surface)", pick("E_coh, ZPE law with derived"), S3),
    ("E_coh, ZPE law f_zpe = 0.01 (fluid, diffusion)", pick("E_coh, ZPE law with fixed"), S3),
    ("E_coh, full f_QCD law (corrosion, hysteresis)", pick("E_coh, full"), S3),
    ("Fe-56 atom mass (inventory law)", pick("inventory atom-level"), S2),
    ("yield stress, Hall–Petch slope (× e^σ)", pick("yield stress"), S2),
    ("Curie T_C (hysteresis)", pick("Curie T_C"), S2),
    ("oxidation rate, 300 K (Arrhenius)", pick("oxidation rate"), S2),
    ("grain growth K_g, 900 K (Arrhenius)", pick("grain growth K_g, grain_structure.grain_growth_rate_constant (Q x e^s, Arrhenius), 900"), S2),
    ("grain growth K_g, 300 K (Arrhenius)", pick("grain growth K_g, grain_structure.grain_growth_rate_constant (Q x e^s, Arrhenius), 300"), S2),
]
yC = list(range(len(rowsC)))[::-1]
a3.barh(yC, [v for _, v, _ in rowsC], 0.62, color=[c for _, _, c in rowsC])
a3.set_xscale("log")
a3.set_xlim(1e-3, 1e5)
a3.set_yticks(yC)
a3.set_yticklabels([n for n, _, _ in rowsC], fontsize=8.3)
need = r["needed_min_field"]
a3.axvline(need, color=S4, lw=1.4, ls="--")
a3.text(need * 1.08, yC[0] + 0.1, f"needed to reach the\nbest floor (ρ, 1.2 × 10⁻⁸)\nin one year: {need:.0f}", fontsize=8.5, color=INK, va="top")
for yy, (_, v, _) in zip(yC, rowsC):
    a3.text(v * 1.12, yy, f"{v:.3g}", va="center", fontsize=8, color=INK2)
a3.set_xlabel("|d ln X / dσ| at σ = 0, iron unless named (log scale)")
a3.grid(axis="x", color=GRID, lw=0.8, which="major")
a3.set_axisbelow(True)
from matplotlib.patches import Patch  # noqa: E402
a3.legend(handles=[Patch(color=S1, label="audited chain"), Patch(color=S3, label="E_coh(σ) laws"),
                   Patch(color=S2, label="other modules")],
          frameon=False, loc="center right", fontsize=8.5)
a3.set_title("C. The engine couples σ in many incompatible ways (its two E_coh laws differ × 187),\n"
             f"    but the largest coupling ({r['max_engine_coupling_found']:.0f}) is 4× short of what the best property\n"
             "    floor needs: unobservability is largely by construction",
             fontsize=10.5, loc="left", color=INK)
clean(a3)

# ---- panel B: orders of magnitude below the floor (field-equation normalisation, consistent chain)
rows = [("ρ", "rho"), ("K, G", "K"), ("v_L, v_T, v_D", "vL"), ("Θ_D", "ThD"), ("T_melt", "Tm"), ("c_p", "cp"), ("air c_s", "cs_air")]
obs = [o for o in r["observability"] if o["normalisation"] == "field_eq" and o["chain"] == "consistent"]
get = lambda p, ch: next(o["orders_below_floor"] for o in obs if o["property"] == p and o["channel"] == ch)  # noqa: E731
ann = [get(k, "dsig_annual") for _, k in rows]
km = [get(k, "dsig_km") for _, k in rows]
res_o = r["acoustic_resonator_row_v2"]["field_eq"]["optical clock (sigma-invariant: prop. m_e alpha^2)"]["log10_floor_over_effect_short_term"]
clk = r["clock_row"]["field_eq"]
clk_o = math.log10(clk["annual_dlnmu_floor"] / clk["annual_dlnmu_effect"])
ep = r["ep_row"]["rows"]
ep_e = ep["electron-mass term alone, pure Ti vs pure Pt|field_eq"]["log10_floor_over_effect"]
ep_s = ep["repo inventory law, TA6V vs PtRh10 alloys (S9a record misc/unification/sigma_exclusion.json)|field_eq"]["log10_floor_over_effect"]
other = [("quartz resonator vs optical clock\n(short-term floor only) [SPECULATIVE]", res_o, "annual"),
         ("Yb⁺ / Cs clocks, μ coupling\n(Lange+2021)", clk_o, "annual"),
         ("free fall, MICROSCOPE: electron\nterm alone (Touboul+2022)", ep_e, "static ∇σ"),
         ("free fall, MICROSCOPE: repo\natom law, S9a Δf", ep_s, "static ∇σ")]
names = [n for n, _ in rows] + [n for n, _, _ in other]
y = list(range(len(names)))[::-1]
h = 0.38
a2.barh([yy + h / 2 for yy in y[:len(rows)]], ann, h, color=S1, label="annual Δσ = 5.2 × 10⁻¹¹")
a2.barh([yy - h / 2 for yy in y[:len(rows)]], km, h, color=S2, label="Δσ per km = 3.5 × 10⁻¹⁴")
a2.barh(y[len(rows):], [v for _, v, _ in other], 0.55, color=S3, label="frequency and free-fall channels")
a2.axvline(5, color=S4, lw=1.4, ls="--", label="design criterion: ≥ 5 orders")
a2.axvline(0, color=INK, lw=1.0)
a2.set_yticks(y)
a2.set_yticklabels(names, fontsize=9)
a2.set_xlabel("log₁₀(floor / predicted effect)   [floors and bounds: LIT]\n"
              "> 0: signal below the best measurement;  < 0: already measured beyond it")
a2.grid(axis="x", color=GRID, lw=0.8)
a2.set_axisbelow(True)
a2.set_xlim(-12.5, 11.8)
a2.set_ylim(-0.8, y[0] + 1.2)
for yy, v in zip(y[:len(rows)], ann):
    a2.text(v + 0.15, yy + h / 2, f"{v:.2f}", va="center", fontsize=8.5, color=INK2)
for yy, (_, v, kind) in zip(y[len(rows):], other):
    a2.text(0.25, yy, mn(f"{v:.1f}") + f"  ({kind}; × {10 ** (-v):.1e} beyond)".replace("e+0", "e").replace("e+", "e"),
            va="center", ha="left", fontsize=8.5, color=INK2)
a2.text(-0.2, y[0] + 0.85, "← measured beyond the signal", ha="right", fontsize=9, color=INK2)
a2.text(0.2, y[0] + 0.85, "below floor →", ha="left", fontsize=9, color=INK2)
a2.set_title("B. Property metrology misses σ_Earth by 2.4–7.3 orders (annual channel),\n"
             "    so the design's ≥ 5-order criterion FAILS there (per km it passes, ≥ 5.5).\n"
             "    Clocks (× 3.2 × 10⁵) and free-fall EP tests (× 1.5 × 10⁹ to 9.4 × 10¹⁰) already\n"
             "    exclude the σ-bridge; the resonator beats only short-term stability",
             fontsize=10.5, loc="left", color=INK)
a2.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.95), fontsize=8.5)
clean(a2)

fig.suptitle("U12: in the audited material chain e^σ acts only through nuclear mass, and the thermal module drops it (I3 = f_QCD/2, chain check FAILS as predicted).\n"
             "No property measurement could see σ_Earth (2.4–7.3 orders short, so the ≥ 5-order criterion FAILS on the annual channel);\n"
             "clocks and free-fall equivalence-principle tests already exclude the long-range σ-bridge by 5.5 to 11 orders",
             fontsize=11.5, color=INK, x=0.01, ha="left")
fig.subplots_adjust(left=0.185, right=0.975, top=0.885, bottom=0.075)
fig.savefig(os.path.join(OUT, "u12_material_chain_v2.png"), dpi=140, facecolor=SURF)
print("saved", os.path.join(OUT, "u12_material_chain_v2.png"))
