"""Figure 6a — the decisive difference, measured: ⟨K⟩ against E for relaxed self-gravitating clusters at five genuinely
different energies (fixed radius, four seeds each), from stage6_heat_capacity.py. Prediction: slope −1 (virial,
C = −(3/2)Nk_B); an ideal gas has slope +1 (C = +(3/2)Nk_B)."""
import json
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("misc/unification/stage6_heat_capacity.json"))
rows = d["rows"]
E = np.array([r["E0"] for r in rows]); K = np.array([r["K_mean"] for r in rows]); Kb = np.array([r["K_bound_mean"] for r in rows])
Q0 = np.array([r["Q0"] for r in rows])
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
fig, ax = plt.subplots(figsize=(10.5, 6), dpi=180); fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
sc = 1e9
ax.scatter(E / sc, K / sc, s=46, color=S1, edgecolors=INK, linewidths=0.4, zorder=5, label="⟨K⟩, all 64 bodies (4 seeds per energy)")
ax.scatter(E / sc, Kb / sc, s=26, marker="s", color=S3, edgecolors=INK, linewidths=0.4, zorder=5, label="⟨K⟩, bound bodies only")
xx = np.linspace(E.min() * 1.08, E.max() * 0.92, 50)
b_all = np.polyfit(E, K, 1)
ax.plot(xx / sc, np.polyval(b_all, xx) / sc, color=S1, lw=2,
        label=("fit: d⟨K⟩/dE = %.3f ± %.3f  →  C/C_gas = %.2f" % (d["slope_dK_dE_all"], d["slope_err_all"], d["C_over_Cgas_all"])).replace("-", "−"))
ax.plot(xx / sc, -xx / sc, color=INK, lw=1.2, linestyle=":", label="virial prediction ⟨K⟩ = −E  (slope −1)")
K_mid = np.mean(K); E_mid = np.mean(E)
ax.plot(xx / sc, (K_mid + (xx - E_mid)) / sc, color=S2, lw=1.6, linestyle=(0, (6, 3)), label="an ideal gas would go this way (slope +1)")
ax.set_xlabel("total energy E = K + U_softened  (10⁹ J; more negative = more tightly bound)")
ax.set_ylabel("time-averaged kinetic energy ⟨K⟩  (10⁹ J)  ∝  temperature")
ax.set_title("Fig. 6a — Gravity's thermodynamics runs backwards, measured: take energy out of a star cluster and it gets hotter\n"
             + ("slope negative at %.0fσ (N = 64, 10 t_ff relax + 10 t_ff average, E drift ≤ %.0e); the 9 %% shortfall from −1 is the measured virial-ratio drift" %
             (d["significance_negative_sigma"], max(r["E_drift"] for r in rows))).replace("e-0", " × 10^").replace("^1", "⁻¹").replace("^2", "⁻²").replace("^3", "⁻³").replace("^4", "⁻⁴").replace("^5", "⁻⁵").replace("^6", "⁻⁶"), fontsize=10.5, color=INK)
ax.legend(frameon=False, fontsize=8.5, loc="upper right"); ax.grid(color=GRID, lw=0.6)
fig.tight_layout(); fig.savefig("misc/unification/fig06a_heat_capacity.png", bbox_inches="tight", facecolor=SURFACE)
print({k: d[k] for k in ("slope_dK_dE_all", "slope_err_all", "C_over_Cgas_all", "slope_dK_dE_bound", "slope_err_bound", "C_over_Cgas_bound", "significance_negative_sigma")})
