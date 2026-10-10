"""Stage 9b — where the fluid/gravity duality touches data: viscosity per unit entropy, η/s, on one axis.

  KSS bound (Kovtun, Son & Starinets 2005, PRL 94, 111601): η/s ≥ ħ/(4π k_B) for theories with black-hole duals,
      conjectured as a universal lower bound; ħ, k_B from Mentat.
  Black-hole horizon (membrane paradigm, Damour 1978; Price & Thorne 1986): the horizon fluid has η = 1/(16πG)
      per unit area and entropy s = 1/(4Għ)·k_B... i.e. η/s = ħ/(4πk_B) exactly — the horizon SITS on the bound.
  Quark–gluon plasma: closest known substance to the bound (KSS 2005; Bernhard, Moreland & Bass 2019, Nat. Phys. 15,
      1113) — plotted as an [LIT, order-of-magnitude] band of 1–3× the bound; the exact posterior was not re-fetched.
  Water at 25 °C: η from the repo's NIST-anchored correlation (liquid_water.water_viscosity), s from the standard molar
      entropy 69.95 J/(mol·K) [LIT, NIST-JANAF; not re-fetched] and the repo's density.
Liquid metals are left off on purpose: the repo's Eyring viscosity is 10³–10⁶× too high (gap filed 2026-10-10).
"""
import json, math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
from mentat_client import mentat
from sigma_ground.field.interface import liquid_water as lw

HBAR, hs = mentat("lookup_constant", {"name": "hbar"}); KB, ks = mentat("lookup_constant", {"name": "k_B"})
KSS = HBAR / (4 * math.pi * KB)                                  # K·s
T = 298.15
eta_w = lw.water_viscosity(T)
rho_w = lw.water_density(T)
S_molar = 69.95                                                  # J/(mol K) [LIT, NIST-JANAF standard entropy of liquid water]
M_w = 0.018015
s_w = S_molar * rho_w / M_w                                      # J/(K m³)
ratio_w = (eta_w / s_w) / KSS
out = dict(KSS_K_s=KSS, sources=dict(hbar=hs, k_B=ks), water=dict(T=T, eta=eta_w, rho=rho_w, s_volumetric=s_w, eta_over_s=eta_w / s_w, over_KSS=ratio_w),
           horizon_over_KSS=1.0, qgp_over_KSS_band_LIT=[1.0, 3.0])
json.dump(out, open("misc/unification/stage9b_eta_over_s.json", "w"), indent=1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
fig, ax = plt.subplots(figsize=(11, 3.8), dpi=180); fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
ax.axvspan(0.1, 1.0, color=MUTED, alpha=0.15); ax.text(0.32, 0.86, "forbidden if the KSS\nbound is universal", fontsize=8.5, color=INK2, ha="center", transform=ax.get_xaxis_transform())
ax.axvline(1.0, color=INK, lw=1.4)
ax.plot([1.0], [0.5], marker="D", ms=12, color=S1, zorder=5); ax.text(1.12, 0.62, "black-hole horizon\n(membrane paradigm): exactly 1", fontsize=9, color=INK)
ax.plot([1.0, 3.0], [0.25, 0.25], color=S2, lw=8, solid_capstyle="butt", alpha=0.85); ax.text(3.4, 0.22, "quark–gluon plasma ≈ 1–3 [LIT]", fontsize=9, color=INK)
ax.plot([ratio_w], [0.5], marker="o", ms=11, color=S3, zorder=5); ax.text(ratio_w * 0.42, 0.66, "water, 25 °C: %.0f×\n(repo η, NIST-anchored)" % ratio_w, fontsize=9, color=INK)
ax.set_xscale("log"); ax.set_xlim(0.1, 3000); ax.set_ylim(0, 1); ax.set_yticks([])
ax.set_xlabel("η / s in units of the KSS bound ħ/(4πk_B) = %.2e K·s (Mentat ħ, k_B)" % KSS)
ax.set_title("Fig. 9b — Where fluid/gravity duality touches data: a black-hole horizon is the most perfect fluid the theory allows,\n"
             "the hottest matter ever made comes within a factor of ~2 of it, and ordinary water is %.0f× more viscous per unit entropy" % ratio_w,
             fontsize=10, color=INK)
ax.grid(color=GRID, lw=0.6, axis="x", which="both")
fig.tight_layout(); fig.savefig("misc/unification/fig09b_eta_over_s.png", bbox_inches="tight", facecolor=SURFACE)
print(json.dumps(out, indent=1))
