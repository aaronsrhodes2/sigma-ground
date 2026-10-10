"""Figure 5 — Stage 5 (L3, same dictionary): a falling fluid makes a horizon that sound cannot cross.
Reads misc/unification/stage5_sonic_horizon.json (from stage5_sonic_horizon.py) and applies two corrections found on
reading the run:
  (1) pulse tracking needs an amplitude floor — after the inside pulse leaves the grid through the inner (supersonic)
      boundary, argmax(δρ/ρ) wanders over 1e-14 noise; only samples with amplitude > 1 % of the injected 1e-3 count;
  (2) the accretion eigenvalue must use ρ_∞ from Bernoulli, ρ_∞ = ρ(r)·exp(−GM/rc_s² + v²/2c_s²), because the outer
      boundary pins ρ = 1 at 12 r_sonic where the Bondi profile is still e^{1/6} ≈ 1.18 above its value at infinity.
"""
import json, math
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("misc/unification/stage5_sonic_horizon.json"))
prof = d["profile"]; r = np.array(prof["r"]); mach = np.array(prof["mach"]); rho = np.array(prof["rho"])
GMd, cs = 2.0, 1.0
rho_inf = rho * np.exp(-GMd / r + 0.5 * mach ** 2)                    # Bernoulli invariant (isothermal)
sel = (r > 0.5) & (r < 5)
mdot = 4 * math.pi * rho * mach * r ** 2                               # |v| = mach·c_s
lam = float(np.median(mdot[sel] / (4 * math.pi * GMd ** 2 * rho_inf[sel])))
lam_bondi = math.exp(1.5) / 4
floor = 1e-5
def clean(track):
    return [(t, rr, a) for t, rr, a in track if a > floor]
ins = clean(d["pulses"]["inside"]["track"]); out = clean(d["pulses"]["outside"]["track"])
ins_max = max(p[1] for p in ins); out_max = max(p[1] for p in out)
res = dict(lambda_c_bernoulli=lam, lambda_c_bondi=lam_bondi, lambda_rel_err=lam / lam_bondi - 1,
           rho_inf_spread=float(np.std(rho_inf[sel]) / np.mean(rho_inf[sel])),
           r_sonic_num_over_bondi=d["steady"]["r_sonic_num_over_bondi"],
           inside_pulse_max_r_above_floor=ins_max, inside_crossed_horizon=ins_max > 1.0,
           outside_pulse_max_r_above_floor=out_max, outside_escaped_past_4=out_max > 4.0)
json.dump(res, open("misc/unification/stage5_corrected.json", "w"), indent=1)

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), dpi=180); fig.patch.set_facecolor(SURFACE)
ax = axes[0]; ax.set_facecolor(SURFACE)
x = np.logspace(np.log10(0.15), np.log10(12), 300)
# exact transonic Bondi branch: solve u²/2 − ln u = 2 ln x + 2/x − 3/2, u < 1 for x > 1, u > 1 for x < 1
from scipy.optimize import brentq
def u_exact(xx):
    rhs = 2 * math.log(xx) + 2 / xx - 1.5
    f = lambda u: 0.5 * u * u - math.log(u) - rhs
    if abs(xx - 1) < 1e-9: return 1.0
    return brentq(f, 1e-8, 1.0 - 1e-12) if xx > 1 else brentq(f, 1.0 + 1e-12, 50.0)
ax.plot(x, [u_exact(xx) for xx in x], color=MUTED, lw=4, alpha=0.5, label="Bondi 1952, exact transonic solution")
ax.plot(r, mach, color=S1, lw=2, label="finite-volume solver, relaxed from rest")
ax.axhline(1.0, color=INK, lw=0.8, linestyle=":"); ax.axvline(1.0, color=S2, lw=1.4, linestyle="--")
ax.text(1.05, 3.2, "sonic horizon\nr = GM/2c_s²", fontsize=8.5, color=INK)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.15, 12)
ax.set_xlabel("r / r_sonic"); ax.set_ylabel("infall Mach number |v| / c_s")
ax.set_title("(a) the flow: horizon at %.3f r_sonic; accretion eigenvalue %.3f vs Bondi %.3f (%+.1f %%)" %
             (res["r_sonic_num_over_bondi"], lam, lam_bondi, 100 * res["lambda_rel_err"]), fontsize=10, color=INK)
ax.legend(frameon=False, fontsize=8.5); ax.grid(color=GRID, lw=0.6, which="both")
ax = axes[1]; ax.set_facecolor(SURFACE)
ax.plot([p[0] for p in ins], [p[1] for p in ins], color=S2, marker="o", ms=4, lw=2, label="sound launched outward at 0.5 r_sonic (inside)")
ax.plot([p[0] for p in out], [p[1] for p in out], color=S3, marker="s", ms=4, lw=2, label="sound launched outward at 2 r_sonic (outside)")
ax.axhline(1.0, color=S2, lw=1.4, linestyle="--"); ax.text(4.0, 1.08, "horizon", fontsize=8.5, color=INK)
ax.axhline(0.15, color=MUTED, lw=1, linestyle=":"); ax.text(4.0, 0.16, "inner edge of the grid", fontsize=8, color=MUTED)
ax.set_yscale("log"); ax.set_xlabel("time  (r_sonic / c_s)"); ax.set_ylabel("pulse position  r / r_sonic")
ax.set_title("(b) the test: pointed outward, sound inside is swept in (max %.2f r_sonic);\noutside, it escapes (reaches %.1f r_sonic)" % (ins_max, out_max),
             fontsize=10, color=INK)
ax.legend(frameon=False, fontsize=8.5, loc="lower right"); ax.grid(color=GRID, lw=0.6, which="both")
fig.suptitle("Fig. 5 — L3, same dictionary, now run: a fluid falling onto a solar mass forms a sonic horizon sound cannot cross "
             "(c_s = 10 km/s: r_sonic = %.2e m, κ = %.2e s⁻¹, T_analog = %.1e K via Mentat)" %
             (d["dimensional"]["r_sonic_m"], d["dimensional"]["kappa_1_s"], d["dimensional"]["T_analog_K_from_mentat"]), fontsize=11, color=INK)
fig.tight_layout(); fig.savefig("misc/unification/fig05_sonic_horizon.png", bbox_inches="tight", facecolor=SURFACE)
print(json.dumps(res, indent=1))
