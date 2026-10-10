"""Stage 7 — the fourth column: how solid becomes fluid in the repo's matter model.

What the code actually implements (field/interface/viscoelasticity.py, read 2026-10-10): `storage_modulus` and `loss_modulus`
are the STANDARD LINEAR SOLID (Zener), not the pure Maxwell element the module docstring's section 5 shows:
    E′(ω) = E_R + (E_U − E_R)·De²/(1 + De²),   E″(ω) = (E_U − E_R)·De/(1 + De²),   De = ωτ,
    τ = (1/ν_D)·exp(E_a/k_BT)  (Debye attempt frequency, measured diffusion activation energy),
    E_R/E_U = 1 − 0.7·(T/T_melt)²  (a heuristic floor, uncited — flagged for the gap ledger).
Two consequences, both shown:
  (a) one dimensionless number: every material and temperature collapses onto ONE curve in De once normalised by the
      relaxing part (E_U − E_R) — the material's clock against the observer's clock;
  (b) but the floor E_R ≥ 0.3·E_U means a crystalline metal never reaches the fluid end below T_melt in this model: for
      crystals, solid → liquid is a first-order phase transition (latent heat, a thermodynamic boundary), not a Deborah
      crossover. The Maxwell limit E_R → 0 is the behaviour of amorphous matter (glasses, ice sheets, the mantle).
"""
import json, math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
from mentat_client import mentat
import numpy as np
from sigma_ground.field.interface import viscoelasticity as ve
from sigma_ground.field.interface.surface import MATERIALS

MATS = ["aluminum", "copper", "iron", "titanium", "gold"]
# DIAGNOSTIC (filed in KNOWN_GAPS.md): _relaxation_ratio reads MATERIALS[m]["melting_point_K"], a key surface.MATERIALS does
# not have, so every metal silently gets T_melt = 1800 K. The CRC melting points come from Mentat for the corrected positions.
TMELT = {m: mentat("melting_point", {"material": m})[0] for m in MATS}
TEMPS = [300.0, 600.0, 900.0]
CLOCKS = {"1 s": 1.0, "1 year": 2 * math.pi / (365.25 * 86400)}
pts = []
for m in MATS:
    for T in TEMPS:
        tau = ve.relaxation_time(m, T); ratio = ve._relaxation_ratio(m, T)
        E_U = ve.youngs_modulus(m); E_R = ratio * E_U; dE = E_U - E_R
        for lab, w in CLOCKS.items():
            Ep = ve.storage_modulus(m, w, T); Epp = ve.loss_modulus(m, w, T); De = w * tau
            pts.append(dict(material=m, T=T, T_melt=MATERIALS[m].get("melting_point_K", 1800), clock=lab, tau_s=tau, De=De,
                            ER_over_EU=ratio, y_store=(Ep - E_R) / dE if dE > 0 else None, y_loss=Epp / dE if dE > 0 else None,
                            Ep_over_EU=Ep / E_U))
collapse = max(max(abs(p["y_store"] - p["De"] ** 2 / (1 + p["De"] ** 2)), abs(p["y_loss"] - p["De"] / (1 + p["De"] ** 2)))
               for p in pts if p["y_store"] is not None and p["De"] < 1e150)
json.dump(dict(points=pts, collapse_max_deviation=collapse, relax_ratio_max=ve._RELAX_RATIO_MAX),
          open("misc/unification/stage7_maxwell.json", "w"), indent=1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.8), dpi=180); fig.patch.set_facecolor(SURFACE)
ax = axes[0]; ax.set_facecolor(SURFACE)
De = np.logspace(-8, 8, 600)
ax.plot(De, De ** 2 / (1 + De ** 2), color=S1, lw=2.4, label="stored part (E′ − E_R)/(E_U − E_R)")
ax.plot(De, De / (1 + De ** 2), color=S2, lw=2.4, linestyle=(0, (6, 3)), label="dissipated part E″/(E_U − E_R)")
ax.axvline(1.0, color=INK, lw=1, linestyle=":"); ax.text(1.4, 1.03, "De = 1", fontsize=8.5, color=INK2)
mk = {"1 s": "o", "1 year": "s"}
for p in pts:
    if 1e-8 < p["De"] < 1e8:
        ax.plot(p["De"], p["y_store"], marker=mk[p["clock"]], ms=6, color=S1, mec=INK, mew=0.5, zorder=5)
        ax.plot(p["De"], p["y_loss"], marker=mk[p["clock"]], ms=6, color=S2, mec=INK, mew=0.5, zorder=5)
ax.plot([], [], marker="o", color=MUTED, linestyle="none", label="Al, Cu, Fe, Ti, Au at 300–900 K, watched for 1 s")
ax.plot([], [], marker="s", color=MUTED, linestyle="none", label="the same, watched for 1 year")
ax.set_xscale("log"); ax.set_xlim(1e-8, 1e8); ax.set_ylim(-0.03, 1.1)
ax.set_xlabel("Deborah number De = ωτ  (material's clock ÷ observer's clock)"); ax.set_ylabel("fraction of the relaxing modulus")
ax.set_title("(a) one number decides elastic vs viscous: every material collapses\nonto one curve (max deviation %.0e)" % collapse, fontsize=10.5, color=INK)
ax.legend(frameon=False, fontsize=8.2, loc="center left"); ax.grid(color=GRID, lw=0.6)
ax = axes[1]; ax.set_facecolor(SURFACE)
h = np.linspace(0, 1, 200)
ax.plot(h, 1 - ve._RELAX_RATIO_MAX * h ** 2, color=S1, lw=2.4, label="E_R/E_U = 1 − 0.7 (T/T_melt)²  (the repo's heuristic floor)")
ax.axhline(0.0, color=S2, lw=2, linestyle=(0, (6, 3)), label="a true fluid (Maxwell limit, E_R = 0): glasses, ice sheets, mantle")
for m in MATS:
    for T in TEMPS:
        ax.plot(T / 1800.0, ve._relaxation_ratio(m, T), marker="o", ms=5, color=S1, mec=INK, mew=0.5)
        if TMELT[m] and T < TMELT[m]:          # titanium: not in Mentat's melting-point table (gap ledger) — no corrected point
            hc = T / TMELT[m]
            ax.plot(hc, 1 - ve._RELAX_RATIO_MAX * hc ** 2, marker="o", ms=6, mfc="none", mec=S2, mew=1.4)
ax.plot([], [], marker="o", color=S1, mec=INK, linestyle="none", label="as the code computes it (T_melt defaults to 1800 K — bug, filed)")
ax.plot([], [], marker="o", mfc="none", mec=S2, linestyle="none", label="with CRC melting points (Mentat melting_point)")
ax.annotate("never reaches the fluid end\nbelow melting: for crystals, solid → liquid\nis a phase transition (latent heat),\nnot a Deborah crossover",
            xy=(1.0, 0.3), xytext=(0.35, 0.12), fontsize=8.5, color=INK, arrowprops=dict(arrowstyle="->", color=INK2))
ax.set_xlim(0, 1.05); ax.set_ylim(-0.08, 1.08)
ax.set_xlabel("homologous temperature T / T_melt"); ax.set_ylabel("relaxed / unrelaxed modulus")
ax.set_title("(b) …but in the repo's model a crystal keeps ≥ 30 % of its stiffness\nuntil it melts — the fluid end needs a thermodynamic transition", fontsize=10.5, color=INK)
ax.legend(frameon=False, fontsize=8.2, loc="upper right"); ax.grid(color=GRID, lw=0.6)
fig.suptitle("Fig. 7 — The fourth column: solid ↔ fluid is one number for amorphous matter, and a phase transition for crystals", fontsize=12, color=INK)
fig.tight_layout(); fig.savefig("misc/unification/fig07_maxwell_crossover.png", bbox_inches="tight", facecolor=SURFACE)
print("collapse max deviation:", collapse); print("CRC T_melt via Mentat:", TMELT)
for p in pts:
    if p["clock"] == "1 s": print(f"{p['material']:9s} {int(p['T'])} K  T/Tm={p['T']/p['T_melt']:.2f}  tau={p['tau_s']:.2e} s  E_R/E_U={p['ER_over_EU']:.3f}")
