"""
Phase V — Joint Γ × f_rd Bayesian posterior.

Marginalises over the dominant f_rd systematic (ringdown energy fraction) to
obtain the true posterior on the amplitude ratio Γ = A_obs / A_GR.

Since ρ_syn ∝ sqrt(f_rd) (from the A₂₂ energy-conservation formula), the
effective signal SNR per detector-event at arbitrary f_rd is:

    s_i(Γ, f_rd) = Γ × ρ_syn_i × sqrt(f_rd / f_rd_ref)

where ρ_syn_i is the Phase-III injection SNR computed at f_rd_ref = 0.15.

Prior:
    Γ  ~ Uniform[0, 3]
    f_rd ~ LogUniform[0.03, 0.20]   (NR estimates span this range)

Likelihood: independent Rician per detector-event
    p(ρ_obs_i | Γ, f_rd) = Rician(ρ_obs_i; s_i(Γ, f_rd), σ=1)

Output:
    - 2D posterior heat-map (Γ, f_rd)
    - Marginal p(Γ | data)   (f_rd integrated out)
    - Marginal p(f_rd | data)
    - 68 % / 95 % credible contours in 2D
    - Summary statistics printed to stdout
    - Figure: misc/bh_phase_v_joint_gamma_frd.png
    - Verdict doc: misc/bh_phase_v_joint_gamma_frd_results.md
"""

import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import i0 as bessel_i0

# ---------------------------------------------------------------------------
# Phase-III data: (event, detector, rho_obs, rho_syn @ f_rd=0.15)
# ---------------------------------------------------------------------------
F_RD_REF = 0.15   # reference f_rd used in Phase III

PHASE_III_DATA = [
    ("GW150914", "H1", 1.000, 1.5430),
    ("GW150914", "L1", 0.734, 0.1764),
    ("GW151226", "H1", 1.458, 0.0114),
    ("GW151226", "L1", 1.205, 0.0016),
    ("GW170814", "H1", 1.216, 1.1302),
    ("GW170814", "L1", 1.059, 0.2132),
    ("GW170814", "V1", 1.402, 0.0196),
    ("GW170104", "H1", 0.392, 0.4493),
    ("GW170104", "L1", 0.875, 0.0566),
    ("GW190521", "H1", 2.164, 1.0741),
    ("GW190521", "L1", 0.395, 1.5491),
]

RHO_OBS  = np.array([d[2] for d in PHASE_III_DATA])
RHO_SYN  = np.array([d[3] for d in PHASE_III_DATA])

# ---------------------------------------------------------------------------
# σ_conv predictions
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sigma_ground.field.constants import XI, ETA, SIGMA_CONV

GAMMA_PREDS = {
    "GR":           1.0000,
    "exp":          0.8395,                                # Phase G phenomenology (hardcoded)
    "sigma_coh":    round(1.0 - ETA / 2.0, 4),            # ≈ 0.7924
    "linear_cbrt":  round(ETA ** (1.0 / 3.0), 4),         # ≈ 0.7461
}

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
N_GAMMA  = 400
N_FRD    = 300
GAMMA_MIN, GAMMA_MAX = 0.0, 3.0
FRD_MIN,   FRD_MAX   = 0.03, 0.20

Gamma_grid = np.linspace(GAMMA_MIN, GAMMA_MAX, N_GAMMA)
frd_grid   = np.linspace(FRD_MIN,   FRD_MAX,   N_FRD)

# ---------------------------------------------------------------------------
# Rician log-pdf  p(A | s, sigma=1)
# ---------------------------------------------------------------------------
def rician_logpdf_arr(A, s):
    """Vectorised: A scalar, s ndarray → ndarray of log-pdf values."""
    log_A   = math.log(A) if A > 0 else -1e30
    quad    = -(A * A + s * s) / 2.0
    log_I0  = np.log(bessel_i0(A * s) + 1e-300)
    return log_A + quad + log_I0

# ---------------------------------------------------------------------------
# 2-D log-posterior  (N_GAMMA × N_FRD)
# ---------------------------------------------------------------------------
print("Building 2-D log-posterior grid …", flush=True)

# Log-uniform prior on f_rd: log π(f_rd) = -log(f_rd) - log(log(FRD_MAX/FRD_MIN))
LOG_PRIOR_FRD_NORM = math.log(math.log(FRD_MAX / FRD_MIN))

logpost_2d = np.zeros((N_GAMMA, N_FRD))

for j, frd in enumerate(frd_grid):
    scale = math.sqrt(frd / F_RD_REF)
    rho_syn_eff = RHO_SYN * scale          # shape (11,)
    log_prior_frd = -math.log(frd) - LOG_PRIOR_FRD_NORM

    for i, G in enumerate(Gamma_grid):
        s_arr = G * rho_syn_eff             # shape (11,)
        ll = sum(rician_logpdf_arr(float(A), float(s))
                 for A, s in zip(RHO_OBS, s_arr))
        logpost_2d[i, j] = ll + log_prior_frd

# Numerical stability
logpost_2d -= logpost_2d.max()
post_2d     = np.exp(logpost_2d)

# ---------------------------------------------------------------------------
# Normalise on the 2-D grid
# ---------------------------------------------------------------------------
dG   = Gamma_grid[1] - Gamma_grid[0]
dfrd = frd_grid[1]   - frd_grid[0]
norm_2d = np.trapezoid(np.trapezoid(post_2d, frd_grid, axis=1), Gamma_grid)
post_2d /= norm_2d

# ---------------------------------------------------------------------------
# Marginals
# ---------------------------------------------------------------------------
marg_gamma = np.trapezoid(post_2d, frd_grid, axis=1)   # shape (N_GAMMA,)
marg_frd   = np.trapezoid(post_2d, Gamma_grid, axis=0)  # shape (N_FRD,)

# renorm marginals
marg_gamma /= np.trapezoid(marg_gamma, Gamma_grid)
marg_frd   /= np.trapezoid(marg_frd,   frd_grid)

# ---------------------------------------------------------------------------
# Credible intervals from marginals
# ---------------------------------------------------------------------------
def hdi_from_marginal(x_grid, pdf, levels=(0.68, 0.95)):
    cdf = np.cumsum(pdf) * (x_grid[1] - x_grid[0])
    cdf /= cdf[-1]
    results = []
    for p in levels:
        lo = float(x_grid[np.searchsorted(cdf, (1 - p) / 2)])
        hi = float(x_grid[np.searchsorted(cdf, 1 - (1 - p) / 2)])
        results.append((lo, hi))
    return results

hdis_gamma = hdi_from_marginal(Gamma_grid, marg_gamma)
hdis_frd   = hdi_from_marginal(frd_grid,   marg_frd)

map_gamma = float(Gamma_grid[np.argmax(marg_gamma)])
map_frd   = float(frd_grid[np.argmax(marg_frd)])

# ---------------------------------------------------------------------------
# Posterior probabilities for model predictions
# ---------------------------------------------------------------------------
cdf_gamma = np.cumsum(marg_gamma) * dG
cdf_gamma /= cdf_gamma[-1]

def p_less_than(val):
    idx = np.searchsorted(Gamma_grid, val)
    if idx >= len(cdf_gamma):
        return 1.0
    return float(cdf_gamma[idx])

# ---------------------------------------------------------------------------
# 2-D credible contours  (68 % / 95 %)
# ---------------------------------------------------------------------------
# Sort post_2d values descending, find cumulative mass thresholds
flat = post_2d.flatten()
isort = np.argsort(flat)[::-1]
cumsum = np.cumsum(flat[isort]) * dG * dfrd
thresh_68 = float(flat[isort[np.searchsorted(cumsum, 0.68)]])
thresh_95 = float(flat[isort[np.searchsorted(cumsum, 0.95)]])

# ---------------------------------------------------------------------------
# Log-likelihood ratios at each model prediction (marginalised over f_rd)
# ---------------------------------------------------------------------------
def marginal_loglike(G_val):
    """ln ∫ p(data|G, frd) π(frd) dfrd  (unnormalised, for ratios)."""
    lls = []
    for j, frd in enumerate(frd_grid):
        scale = math.sqrt(frd / F_RD_REF)
        rho_syn_eff = RHO_SYN * scale
        s_arr = G_val * rho_syn_eff
        ll = sum(rician_logpdf_arr(float(A), float(s))
                 for A, s in zip(RHO_OBS, s_arr))
        log_prior = -math.log(frd) - LOG_PRIOR_FRD_NORM
        lls.append(ll + log_prior)
    lls = np.array(lls)
    lls -= lls.max()
    return float(np.log(np.trapezoid(np.exp(lls), frd_grid)))

print("Computing marginal log-likelihoods for model predictions …", flush=True)
marg_ll = {name: marginal_loglike(G) for name, G in GAMMA_PREDS.items()}
delta_ln_bf = {name: marg_ll[name] - marg_ll["GR"] for name in GAMMA_PREDS}

# ---------------------------------------------------------------------------
# Summary printout
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("Phase V — Joint Γ × f_rd Posterior")
print("=" * 60)
print(f"MAP Γ (marginal)   = {map_gamma:.3f}")
print(f"MAP f_rd (marginal)= {map_frd:.3f}")
print(f"68% HDI Γ          = [{hdis_gamma[0][0]:.3f}, {hdis_gamma[0][1]:.3f}]")
print(f"95% HDI Γ          = [{hdis_gamma[1][0]:.3f}, {hdis_gamma[1][1]:.3f}]")
print(f"68% HDI f_rd       = [{hdis_frd[0][0]:.3f}, {hdis_frd[0][1]:.3f}]")
print(f"95% HDI f_rd       = [{hdis_frd[1][0]:.3f}, {hdis_frd[1][1]:.3f}]")
print()
print("Posterior tail probabilities (marginal on Γ):")
for name, G in GAMMA_PREDS.items():
    p = p_less_than(G)
    print(f"  P(Γ < {G:.4f}) [{name:12s}] = {p:.3f}")
print()
print("Marginal Bayes factors vs GR (f_rd integrated out):")
for name, dlnl in delta_ln_bf.items():
    if name == "GR":
        continue
    bf = math.exp(dlnl)
    print(f"  B({name:12s} / GR) = exp({dlnl:+.3f}) = {bf:.3f}")
print("=" * 60)

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(14, 10), facecolor="#0d1117")

ax_2d  = fig.add_axes([0.08, 0.38, 0.56, 0.54])
ax_mg  = fig.add_axes([0.68, 0.38, 0.28, 0.54])
ax_mf  = fig.add_axes([0.08, 0.08, 0.56, 0.24])

DARK_BG   = "#0d1117"
LIGHT_TEXT = "#e6edf3"
ACCENT     = "#58a6ff"
ORANGE     = "#f0883e"
GREEN      = "#3fb950"
PURPLE     = "#bc8cff"
RED        = "#ff7b72"

PRED_COLORS = {
    "GR":          RED,
    "exp":         ORANGE,
    "sigma_coh":   GREEN,
    "linear_cbrt": PURPLE,
}

for ax in [ax_2d, ax_mg, ax_mf]:
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=LIGHT_TEXT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

# --- 2D heat-map ---
GG, FF = np.meshgrid(Gamma_grid, frd_grid, indexing="ij")
ax_2d.pcolormesh(GG, FF, post_2d, cmap="inferno", shading="auto")
ax_2d.contour(GG, FF, post_2d,
               levels=[thresh_95, thresh_68],
               colors=["white", "white"],
               linestyles=["--", "-"],
               linewidths=[0.8, 1.2])

for name, G in GAMMA_PREDS.items():
    ax_2d.axvline(G, color=PRED_COLORS[name], lw=1.2, ls=":", alpha=0.9)
    ax_2d.text(G + 0.01, FRD_MAX * 0.97, name,
               color=PRED_COLORS[name], fontsize=7, va="top")

ax_2d.set_xlabel("Γ  (amplitude ratio A_obs / A_GR)", color=LIGHT_TEXT, fontsize=9)
ax_2d.set_ylabel("f_rd  (ringdown energy fraction)", color=LIGHT_TEXT, fontsize=9)
ax_2d.set_title("2-D posterior  p(Γ, f_rd | data)\n"
                "— / ·· = 68 % / 95 % credible regions",
                color=LIGHT_TEXT, fontsize=9)
ax_2d.set_xlim(GAMMA_MIN, GAMMA_MAX)
ax_2d.set_ylim(FRD_MIN, FRD_MAX)

# --- Marginal on Γ ---
ax_mg.plot(marg_gamma, Gamma_grid, color=ACCENT, lw=1.5)
ax_mg.fill_betweenx(Gamma_grid, 0, marg_gamma, alpha=0.15, color=ACCENT)
ax_mg.axhline(hdis_gamma[0][0], color=ACCENT, lw=0.7, ls="--")
ax_mg.axhline(hdis_gamma[0][1], color=ACCENT, lw=0.7, ls="--")
ax_mg.axhline(hdis_gamma[1][0], color=ACCENT, lw=0.5, ls=":")
ax_mg.axhline(hdis_gamma[1][1], color=ACCENT, lw=0.5, ls=":")

for name, G in GAMMA_PREDS.items():
    ax_mg.axhline(G, color=PRED_COLORS[name], lw=1.0, ls=":")

ax_mg.set_xlabel("p(Γ | data)", color=LIGHT_TEXT, fontsize=9)
ax_mg.set_title("Marginal on Γ\n(f_rd integrated)", color=LIGHT_TEXT, fontsize=9)
ax_mg.set_ylim(GAMMA_MIN, GAMMA_MAX)
ax_mg.set_xlim(left=0)
ax_mg.tick_params(labelleft=False)

# --- Marginal on f_rd ---
ax_mf.plot(frd_grid, marg_frd, color=ORANGE, lw=1.5)
ax_mf.fill_between(frd_grid, 0, marg_frd, alpha=0.15, color=ORANGE)
ax_mf.axvline(hdis_frd[0][0], color=ORANGE, lw=0.7, ls="--")
ax_mf.axvline(hdis_frd[0][1], color=ORANGE, lw=0.7, ls="--")
ax_mf.axvline(hdis_frd[1][0], color=ORANGE, lw=0.5, ls=":")
ax_mf.axvline(hdis_frd[1][1], color=ORANGE, lw=0.5, ls=":")
ax_mf.axvline(F_RD_REF, color=LIGHT_TEXT, lw=0.8, ls="--", alpha=0.6,
              label=f"f_rd ref = {F_RD_REF}")
ax_mf.legend(fontsize=7, facecolor=DARK_BG, labelcolor=LIGHT_TEXT, framealpha=0.6)

ax_mf.set_xlabel("f_rd  (ringdown energy fraction)", color=LIGHT_TEXT, fontsize=9)
ax_mf.set_ylabel("p(f_rd | data)", color=LIGHT_TEXT, fontsize=9)
ax_mf.set_title("Marginal on f_rd  (Γ integrated)", color=LIGHT_TEXT, fontsize=9)
ax_mf.set_xlim(FRD_MIN, FRD_MAX)
ax_mf.set_ylim(bottom=0)

# Legend block
legend_lines = [
    plt.Line2D([0], [0], color=PRED_COLORS[n], ls=":", lw=1.2, label=f"{n}  Γ={G:.4f}")
    for n, G in GAMMA_PREDS.items()
]
ax_2d.legend(handles=legend_lines, fontsize=7,
             facecolor=DARK_BG, labelcolor=LIGHT_TEXT, framealpha=0.7,
             loc="lower right")

fig.suptitle(
    "Phase V — Joint Γ × f_rd Posterior  |  11 detector-events (Phases I–IV data)",
    color=LIGHT_TEXT, fontsize=11, y=0.97,
)

out_dir = os.path.join(os.path.dirname(__file__), "..", "misc")
fig_path = os.path.join(out_dir, "bh_phase_v_joint_gamma_frd.png")
fig.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close(fig)
print(f"\nFigure saved → {fig_path}")
print(f"Size: {os.path.getsize(fig_path) / 1024:.1f} KB")
