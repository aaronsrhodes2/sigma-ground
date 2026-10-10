"""Figure 8 — Stage 8: gas expansion, the first law and the second, on the repo's SPH engine (misc/unification/stage8_gas_expansion.json)."""
import json, math
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("misc/unification/stage8_gas_expansion.json"))
ex, co = d["expansion"], d["collision"]; gam, cv = d["gamma"], d["c_v"]
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), dpi=180); fig.patch.set_facecolor(SURFACE)

ax = axes[0]; ax.set_facecolor(SURFACE)
H = ex["history"]; t = [h["t"] * 1e3 for h in H]
ax.plot(t, [h["K"] / 1e6 for h in H], color=S1, lw=2.2, marker="o", ms=4, label="bulk kinetic energy K")
ax.plot(t, [h["U"] / 1e6 for h in H], color=S2, lw=2.2, marker="s", ms=4, linestyle=(0, (6, 3)), label="internal energy U_int (heat)")
ax.plot(t, [h["total"] / 1e6 for h in H], color=INK, lw=1.4, label="K + U_int — drift %.1e of K" % ex["ledger_drift_rel_to_K"])
ax.axhline(0, color=MUTED, lw=0.8)
ax.set_xlabel("time (ms)"); ax.set_ylabel("energy change (MJ)")
ax.set_title("(a) first law: an expanding gas turns heat into motion,\njoule for joule (N₂, 400 parcels, free expansion)", fontsize=10, color=INK)
ax.legend(frameon=False, fontsize=8.5, loc="center left"); ax.grid(color=GRID, lw=0.6)

ax = axes[1]; ax.set_facecolor(SURFACE)
rho = np.array([p[0] for p in ex["rho_T_pairs"]]); T = np.array([p[1] for p in ex["rho_T_pairs"]])
ax.scatter(rho, T, s=14, color=S1, alpha=0.7, edgecolors="none", label="parcels (ρ/ρ₀, T/T₀) at r_med × %.2f" % ex["r_med_ratio"])
xx = np.linspace(min(rho.min(), 0.2), max(rho.max(), 1.1), 100)
ax.plot(xx, xx ** (gam - 1), color=S2, lw=2.2, label="adiabat T/T₀ = (ρ/ρ₀)^(γ−1), γ = %.3f (repo gas.py)" % gam)
ax.set_xlabel("density ρ/ρ₀ (SPH)"); ax.set_ylabel("temperature T/T₀")
ax.set_title("(b) expansion is reversible: parcels follow the adiabat\n(mass-weighted Δs = %+.3f c_v; scatter = SPH density estimate)" % ex["mass_weighted_mean_ds_over_cv"], fontsize=10, color=INK)
ax.legend(frameon=False, fontsize=8.2, loc="upper left"); ax.grid(color=GRID, lw=0.6)

ax = axes[2]; ax.set_facecolor(SURFACE)
m_total = co["N"] * (4 / 3) * math.pi * 0.05 ** 3 * 920.0
ds_col = co["total_entropy_change_J_per_K"] / (m_total * cv)
vals = [ex["mass_weighted_mean_ds_over_cv"], ds_col]
bars = ax.bar(["free expansion\n(reversible)", "two clouds collide\nat Mach %.1f" % co["mach"]], vals, color=[S1, S2], width=0.55)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + (0.01 if v >= 0 else -0.03), "%+.3f" % v, ha="center", fontsize=10, color=INK)
ax.axhline(0, color=INK, lw=0.8)
ax.set_ylabel("mean entropy change Δs / c_v"); ax.set_ylim(-0.08, 0.34)
ax.set_title("(c) second law: the collision shock-heats to %.0f K and makes\nentropy (energy still conserved to %.0e)" % (co["T_max"], abs(co["ledger_drift_rel_to_K0"]) or 1e-16), fontsize=10, color=INK)
ax.grid(color=GRID, lw=0.6, axis="y")
fig.suptitle("Fig. 8 — Gas expansion on the repo's own fluid engine, after the first-law fix: heat ↔ motion exactly, reversible when it should be, irreversible when it must be",
             fontsize=11.5, color=INK)
fig.tight_layout(); fig.savefig("misc/unification/fig08_first_law.png", bbox_inches="tight", facecolor=SURFACE)
print("collision mean ds/cv:", ds_col)
