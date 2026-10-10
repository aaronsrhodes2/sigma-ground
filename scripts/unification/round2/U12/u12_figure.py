"""U12 figure: reads u12_results.json (written by u12_material_chain.py) and draws one PNG."""
import json, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
r = json.load(open(os.path.join(OUT, "u12_results.json"), encoding="utf-8"))
D = r["dlnX_dsigma"]

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#8b5cf6"
plt.rcParams.update({"axes.unicode_minus": True, "font.size": 10, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF})

mn = lambda t: t.replace("-", "−")  # noqa: E731  true minus signs
fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 7.0), gridspec_kw={"width_ratios": [1.05, 1.0]})

# ---- panel A: d ln X / d sigma, as coded vs consistent chain
labels = ["ρ", "K = G = E", "v_L\n(acoustics)", "v_D\n(thermal)", "Θ_D", "T_melt\n(Lindemann)", "c_p\n(300 K)",
          "air c_s"]
coded = [D["rho"], D["K"], D["vL"], D["vD_therm"], D["ThD"], D["Tm"], D["cp"], D["cs_air"]]
cons = [D["rho"], D["K"], D["vL"], D["vD_acou"], D["ThD_cons"], D["Tm_cons"], D["cp_cons"], D["cs_air_cons"]]
x = range(len(labels))
w = 0.38
a1.bar([i - w / 2 for i in x], coded, w, color=S1, label="engine as coded")
a1.bar([i + w / 2 for i in x], cons, w, color=S2, label="consistent chain (ρ(σ), M(σ), Ar coupled) [DERIVED]")
a1.axhline(0, color=INK2, lw=0.8)
a1.set_xticks(list(x))
a1.set_xticklabels(labels)
a1.set_ylabel("d ln X / dσ at σ = 0 (iron; dry air at 288 K)")
a1.set_ylim(-1.05, 1.15)
a1.grid(axis="y", color=GRID, lw=0.8)
a1.set_axisbelow(True)
gap = r["identities"]["I3_like_for_like_dlnvD_therm_minus_dlnvD_acou"]
a1.annotate(f"thermal.sound_velocity keeps ρ₀:\nresidual {gap:.4f} = f_QCD/2 (all 25 materials)",
            xy=(3 - w / 2, D["vD_therm"]), xytext=(2.2, 0.55), fontsize=9, color=INK,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
a1.text(0, D["rho"] + 0.03, f"{D['rho']:+.4f}", ha="center", fontsize=8.5, color=INK2)
a1.text(2, D["vL"] - 0.08, mn(f"{D['vL']:+.4f}"), ha="center", fontsize=8.5, color=INK2)
a1.text(6 + w / 2, D["cp_cons"] - 0.08, mn(f"{D['cp_cons']:+.3f}"), ha="center", fontsize=8.5, color=INK2)
a1.text(5, 0.12, f"{D['Tm']:+.4f}\n(two omissions\ncancel)", ha="center", fontsize=8.5, color=INK2)
a1.set_title("A. How e^σ moves each property: chain identities I1, I2 hold to 2 × 10⁻⁹;\n    I3 (thermal vs acoustics) fails by f_QCD/2",
             fontsize=10.5, loc="left", color=INK)
a1.legend(frameon=False, loc="lower left", fontsize=9)
for sp in ("top", "right"):
    a1.spines[sp].set_visible(False)

# ---- panel B: orders of magnitude below the metrology floor (field-equation normalisation, consistent chain)
rows = [("ρ", "rho"), ("K, G", "K"), ("v_L, v_T", "vL"), ("Θ_D", "ThD"), ("T_melt", "Tm"), ("c_p", "cp"),
        ("air c_s", "cs_air")]
obs = [o for o in r["observability"] if o["normalisation"] == "field_eq" and o["chain"] == "consistent"]
get = lambda p, ch: next(o["orders_below_floor"] for o in obs if o["property"] == p and o["channel"] == ch)  # noqa: E731
ann = [get(k, "dsig_annual") for _, k in rows]
km = [get(k, "dsig_km") for _, k in rows]
res_o = r["acoustic_resonator_row"]["field_eq"]["orders_below_floor"]
clk = r["clock_row"]["field_eq"]
clk_o = math.log10(clk["annual_dlnmu_floor"] / clk["annual_dlnmu_effect"])
names = [n for n, _ in rows] + ["acoustic resonator\nvs clock", "Yb⁺/Cs clocks\n(Lange+2021)"]
y = list(range(len(names)))[::-1]
h = 0.38
a2.barh([yy + h / 2 for yy in y[:len(rows)]], ann, h, color=S1, label="annual Δσ\n= 5.2 × 10⁻¹¹")
a2.barh([yy - h / 2 for yy in y[:len(rows)]], km, h, color=S2, label="Δσ per km\n= 3.5 × 10⁻¹⁴")
a2.barh([y[len(rows)] + h / 2, y[len(rows) + 1] + h / 2], [res_o, clk_o], h, color=S1)
a2.axvline(5, color=S4, lw=1.4, ls="--", label="design:\n≥ 5 orders")
a2.axvline(0, color=INK, lw=1.0)
a2.set_yticks(y)
a2.set_yticklabels(names)
a2.set_xlabel("log₁₀(floor / effect): orders of magnitude below the best measurement   [floors: LIT]")
a2.grid(axis="x", color=GRID, lw=0.8)
a2.set_axisbelow(True)
a2.set_xlim(-6.5, 11.5)
a2.set_ylim(-0.8, y[0] + 1.1)
for yy, v in zip(y[:len(rows)], ann):
    a2.text(v + 0.15, yy + h / 2, f"{v:.1f}", va="center", fontsize=8.5, color=INK2)
a2.text(0.2, y[len(rows)] + h / 2, mn(f"{res_o:.1f}") + "  (effect 2.6 × 10⁻¹¹ vs 2.5 × 10⁻¹⁴ stability)", va="center", ha="left", fontsize=8.5, color=INK2)
a2.text(0.2, y[len(rows) + 1] + h / 2, mn(f"{clk_o:.1f}") + "  (σ-bridge excluded × 3.2 × 10⁵)",
        va="center", ha="left", fontsize=8.5, color=INK2)
a2.text(-0.2, y[0] + 0.75, "← measurable", ha="right", fontsize=9, color=INK2)
a2.text(0.2, y[0] + 0.75, "below floor →", ha="left", fontsize=9, color=INK2)
a2.set_title("B. Property metrology misses σ by 2.4–7.3 orders (annual Δσ);\n    frequency metrology (resonator or clock) does not",
             fontsize=10.5, loc="left", color=INK)
a2.legend(frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.55), fontsize=8.5)
for sp in ("top", "right"):
    a2.spines[sp].set_visible(False)

fig.suptitle("U12: e^σ reaches matter only through nuclear mass (d ln ρ/dσ = 0.99); the engine's thermal chain drops it (residual 0.495),\n"
             "and no property measurement could see σ_Earth — only frequency comparisons could, and clocks already exclude it",
             fontsize=11.5, color=INK, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(os.path.join(OUT, "u12_material_chain.png"), dpi=150, facecolor=SURF)
print("saved", os.path.join(OUT, "u12_material_chain.png"))
