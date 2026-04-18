"""
Phase VII — Corrected amplitude-ratio + Bayesian analysis.

Phase VI identified that Phases III–V used h_inj = A_det × exp(-t/τ) cos(2πft)
without the exp(-t_start/τ) decay that the actual GR QNM carries by t_start ms
after merger.  This overestimated ρ_syn by exp(+t_start/τ) per event, biasing
all Γ values low and creating a spurious preference for Γ < 1.

Correction (analytical — exact for the amplitude factor):

    ρ_syn_corrected = ρ_syn_Phase_III × exp(−t_start / τ_QNM_i)

ρ_obs is unchanged (it is measured from data, not affected by the injection).

This phase applies that correction to the Phase III/IV dataset and reruns:
  - Corrected per-event amplitude ratios (Phase III equivalent)
  - Corrected Bayesian Rician posterior (Phase IV equivalent)
  - Corrected joint Γ × f_rd posterior (Phase V equivalent)

Output:
  - Figure:  misc/bh_phase_vii_corrected_analysis.png
  - Verdict: misc/bh_phase_vii_corrected_analysis_results.md
"""

import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import i0 as bessel_i0

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))
from sigma_ground.field.constants import ETA, XI, SIGMA_CONV

# ── σ_conv model predictions ──────────────────────────────────────────────────
ETA_VAL = float(ETA)
GAMMA_PREDS = {
    "GR":           1.0000,
    "exp":          0.8395,
    "sigma_coh":    round(1.0 - ETA_VAL / 2.0, 4),
    "linear_cbrt":  round(ETA_VAL ** (1.0 / 3.0), 4),
}
PRED_COLORS = {
    "GR": "#ff7b72", "exp": "#f0883e",
    "sigma_coh": "#3fb950", "linear_cbrt": "#bc8cff",
}

# ── Phase III / IV input data (ρ_obs, ρ_syn_biased at f_rd=0.15) ─────────────
# ρ_syn_biased: Phase III injection — does NOT include exp(-t_start/τ) factor.
# τ_qnm_ms: QNM decay time per event (from EVENTS dict / Phase III params).
# t_start_ms: 3.0 ms for all events (RINGDOWN_START_MS in Phase III).
T_START_MS = 3.0

PHASE_III_RAW = [
    # (event, detector, rho_obs, rho_syn_biased, tau_qnm_ms)
    ("GW150914", "H1", 1.000, 1.5430, 4.00),
    ("GW150914", "L1", 0.734, 0.1764, 4.00),
    ("GW151226", "H1", 1.458, 0.0114, 1.20),
    ("GW151226", "L1", 1.205, 0.0016, 1.20),
    ("GW170814", "H1", 1.216, 1.1302, 3.28),
    ("GW170814", "L1", 1.059, 0.2132, 3.28),
    ("GW170814", "V1", 1.402, 0.0196, 3.28),
    ("GW170104", "H1", 0.392, 0.4493, 2.94),
    ("GW170104", "L1", 0.875, 0.0566, 2.94),
    ("GW190521", "H1", 2.164, 1.0741, 8.75),
    ("GW190521", "L1", 0.395, 1.5491, 8.75),
]

# ── Apply correction ──────────────────────────────────────────────────────────
CORRECTED = []
print(f"{'Event/Det':18s}  {'τ(ms)':6s}  {'K=exp(-t/τ)':12s}  "
      f"{'ρ_obs':7s}  {'ρ_syn_bias':10s}  {'ρ_syn_corr':10s}  {'Γ_bias':8s}  {'Γ_corr':8s}")
print("-" * 96)

for ev, det, rho_obs, rho_syn_b, tau_ms in PHASE_III_RAW:
    K       = math.exp(-T_START_MS / tau_ms)          # correction factor < 1
    rho_syn_c = rho_syn_b * K
    gamma_b   = rho_obs / rho_syn_b if rho_syn_b > 0 else float("nan")
    gamma_c   = rho_obs / rho_syn_c if rho_syn_c > 0 else float("nan")
    CORRECTED.append((ev, det, rho_obs, rho_syn_b, rho_syn_c, tau_ms, K))
    print(f"{ev}/{det:2s}          τ={tau_ms:.2f}  K={K:.4f}  "
          f"ρ_obs={rho_obs:.3f}  ρ_syn_b={rho_syn_b:.4f}  ρ_syn_c={rho_syn_c:.4f}  "
          f"Γ_b={gamma_b:.3f}  Γ_c={gamma_c:.3f}")

RHO_OBS   = np.array([r[2] for r in CORRECTED])
RHO_SYN_B = np.array([r[3] for r in CORRECTED])   # biased (Phase III)
RHO_SYN_C = np.array([r[4] for r in CORRECTED])   # corrected (Phase VII)
F_RD_REF  = 0.15

# ── Rician helpers ────────────────────────────────────────────────────────────
def rician_logpdf(A, s):
    if A <= 0:
        return -1e30
    return math.log(A) - (A**2 + s**2) / 2.0 + float(np.log(bessel_i0(A * s) + 1e-300))


# ── Corrected Phase IV: Bayesian Rician on Γ alone ───────────────────────────
def compute_gamma_posterior(rho_obs_arr, rho_syn_arr, n_pts=3000):
    G_arr = np.linspace(0.0, 4.0, n_pts)
    logp  = np.array([
        sum(rician_logpdf(float(A), float(G * s))
            for A, s in zip(rho_obs_arr, rho_syn_arr))
        for G in G_arr
    ])
    logp -= logp.max()
    post  = np.exp(logp)
    norm  = np.trapezoid(post, G_arr)
    post /= norm
    cdf   = np.cumsum(post) * (G_arr[1] - G_arr[0])
    cdf  /= cdf[-1]
    return G_arr, post, cdf


def hdi(G_arr, post, level=0.95):
    cdf = np.cumsum(post) * (G_arr[1] - G_arr[0])
    cdf /= cdf[-1]
    lo = float(G_arr[np.searchsorted(cdf, (1 - level) / 2)])
    hi = float(G_arr[np.searchsorted(cdf, 1 - (1 - level) / 2)])
    return lo, hi


def p_less_than(cdf, G_arr, val):
    return float(cdf[min(np.searchsorted(G_arr, val), len(cdf) - 1)])


def per_event_delta_lnl(rho_obs_arr, rho_syn_arr, G1, G2):
    """Sum of ln p(rho_obs | G1*rho_syn) - ln p(rho_obs | G2*rho_syn) over events."""
    return sum(
        rician_logpdf(float(A), float(G1 * s)) - rician_logpdf(float(A), float(G2 * s))
        for A, s in zip(rho_obs_arr, rho_syn_arr)
    )


print("\n── Corrected Phase IV (Γ posterior) ─────────────────────────────────────")
G_corr, post_corr, cdf_corr = compute_gamma_posterior(RHO_OBS, RHO_SYN_C)
G_bias, post_bias, cdf_bias = compute_gamma_posterior(RHO_OBS, RHO_SYN_B)

map_corr = float(G_corr[np.argmax(post_corr)])
map_bias = float(G_bias[np.argmax(post_bias)])
hdi68_corr = hdi(G_corr, post_corr, 0.68)
hdi95_corr = hdi(G_corr, post_corr, 0.95)
hdi68_bias = hdi(G_bias, post_bias, 0.68)
hdi95_bias = hdi(G_bias, post_bias, 0.95)

print(f"  BIASED   (Phases III–V)  MAP={map_bias:.3f}  68%=[{hdi68_bias[0]:.3f},{hdi68_bias[1]:.3f}]  "
      f"95%=[{hdi95_bias[0]:.3f},{hdi95_bias[1]:.3f}]")
print(f"  CORRECTED (Phase VII)    MAP={map_corr:.3f}  68%=[{hdi68_corr[0]:.3f},{hdi68_corr[1]:.3f}]  "
      f"95%=[{hdi95_corr[0]:.3f},{hdi95_corr[1]:.3f}]")
print()
for name, G_val in GAMMA_PREDS.items():
    pb  = p_less_than(cdf_bias,  G_bias,  G_val)
    pc  = p_less_than(cdf_corr, G_corr, G_val)
    print(f"  P(Γ < {G_val:.4f}) [{name:12s}]: biased={pb:.3f}  corrected={pc:.3f}")

print()
dlnl_corr = per_event_delta_lnl(RHO_OBS, RHO_SYN_C, GAMMA_PREDS["sigma_coh"], GAMMA_PREDS["GR"])
dlnl_bias = per_event_delta_lnl(RHO_OBS, RHO_SYN_B, GAMMA_PREDS["sigma_coh"], GAMMA_PREDS["GR"])
print(f"  ΔlnL sigma_coh/GR  biased  = {dlnl_bias:+.3f}  B={math.exp(dlnl_bias):.3f}")
print(f"  ΔlnL sigma_coh/GR corrected= {dlnl_corr:+.3f}  B={math.exp(dlnl_corr):.3f}")


# ── Corrected Phase V: Joint Γ × f_rd posterior ───────────────────────────────
print("\n── Corrected Phase V (joint Γ × f_rd) ────────────────────────────────────")
N_G, N_F = 300, 200
FRD_MIN, FRD_MAX = 0.03, 0.20
Gamma_g  = np.linspace(0.0, 4.0,  N_G)
frd_g    = np.linspace(FRD_MIN, FRD_MAX, N_F)
LOG_PRIOR_NORM = math.log(math.log(FRD_MAX / FRD_MIN))

logpost_2d = np.zeros((N_G, N_F))
for j, frd in enumerate(frd_g):
    scale      = math.sqrt(frd / F_RD_REF)
    rho_syn_eff = RHO_SYN_C * scale
    lp_frd     = -math.log(frd) - LOG_PRIOR_NORM
    for i, G in enumerate(Gamma_g):
        s_arr = G * rho_syn_eff
        ll    = sum(rician_logpdf(float(A), float(s)) for A, s in zip(RHO_OBS, s_arr))
        logpost_2d[i, j] = ll + lp_frd

logpost_2d -= logpost_2d.max()
post_2d     = np.exp(logpost_2d)
dG, df      = Gamma_g[1] - Gamma_g[0], frd_g[1] - frd_g[0]
norm_2d     = np.trapezoid(np.trapezoid(post_2d, frd_g, axis=1), Gamma_g)
post_2d    /= norm_2d

marg_g_corr = np.trapezoid(post_2d, frd_g, axis=1)
marg_f_corr = np.trapezoid(post_2d, Gamma_g, axis=0)
marg_g_corr /= np.trapezoid(marg_g_corr, Gamma_g)
marg_f_corr /= np.trapezoid(marg_f_corr, frd_g)

map_g_v5c  = float(Gamma_g[np.argmax(marg_g_corr)])
map_f_v5c  = float(frd_g[np.argmax(marg_f_corr)])
hdi68_gv5c = hdi(Gamma_g, marg_g_corr, 0.68)
hdi95_gv5c = hdi(Gamma_g, marg_g_corr, 0.95)
hdi68_fv5c = hdi(frd_g,   marg_f_corr, 0.68)
hdi95_fv5c = hdi(frd_g,   marg_f_corr, 0.95)

cdf_gv5c = np.cumsum(marg_g_corr) * dG
cdf_gv5c /= cdf_gv5c[-1]

print(f"  MAP Γ   = {map_g_v5c:.3f}   68%=[{hdi68_gv5c[0]:.3f},{hdi68_gv5c[1]:.3f}]  "
      f"95%=[{hdi95_gv5c[0]:.3f},{hdi95_gv5c[1]:.3f}]")
print(f"  MAP f_rd= {map_f_v5c:.3f}   68%=[{hdi68_fv5c[0]:.3f},{hdi68_fv5c[1]:.3f}]  "
      f"95%=[{hdi95_fv5c[0]:.3f},{hdi95_fv5c[1]:.3f}]")
print()
for name, G_val in GAMMA_PREDS.items():
    pc = p_less_than(cdf_gv5c, Gamma_g, G_val)
    print(f"  P(Γ < {G_val:.4f}) [{name:12s}] corrected+f_rd marginal = {pc:.3f}")


# ── Per-event ΔlnL table (corrected) ─────────────────────────────────────────
print("\n── Per-event ΔlnL (corrected, sigma_coh vs GR) ─────────────────────────")
G_sc, G_gr = GAMMA_PREDS["sigma_coh"], GAMMA_PREDS["GR"]
total_dlnl = 0.0
print(f"{'Event/Det':18s}  {'ρ_obs':7s}  {'ρ_syn_c':8s}  {'ΔlnL':8s}  {'Direction':12s}")
for ev, det, rho_obs, _, rho_syn_c, tau_ms, K in CORRECTED:
    dl = rician_logpdf(float(rho_obs), float(G_sc * rho_syn_c)) - \
         rician_logpdf(float(rho_obs), float(G_gr * rho_syn_c))
    total_dlnl += dl
    direction = "sigma_coh" if dl > 0 else "GR" if dl < 0 else "—"
    info = "★" if rho_syn_c > 0.5 else ""
    print(f"{ev}/{det:2s}         {rho_obs:5.3f}    {rho_syn_c:.4f}   {dl:+.4f}   {direction} {info}")
print(f"{'Combined':18s}                         {total_dlnl:+.4f}   B={math.exp(total_dlnl):.3f}")


# ── Full summary ──────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("Phase VII — Corrected Analysis Summary")
print("=" * 65)
print(f"Injection correction applied: ρ_syn × exp(−{T_START_MS:.1f}ms / τ_QNM)")
print()
print("Corrected Phase IV (Bayesian Rician on Γ, f_rd fixed):")
print(f"  MAP Γ = {map_corr:.3f}")
print(f"  68% HDI Γ = [{hdi68_corr[0]:.3f}, {hdi68_corr[1]:.3f}]")
print(f"  95% HDI Γ = [{hdi95_corr[0]:.3f}, {hdi95_corr[1]:.3f}]")
print(f"  P(Γ < 1.0) = {p_less_than(cdf_corr, G_corr, 1.0):.3f}")
print(f"  ΔlnL sigma_coh/GR = {dlnl_corr:+.3f}  B = {math.exp(dlnl_corr):.3f}")
print()
print("Corrected Phase V (joint Γ × f_rd, f_rd marginalised):")
print(f"  MAP Γ = {map_g_v5c:.3f}")
print(f"  68% HDI Γ = [{hdi68_gv5c[0]:.3f}, {hdi68_gv5c[1]:.3f}]")
print(f"  95% HDI Γ = [{hdi95_gv5c[0]:.3f}, {hdi95_gv5c[1]:.3f}]")
print(f"  P(Γ < 1.0) = {p_less_than(cdf_gv5c, Gamma_g, 1.0):.3f}")
print(f"  MAP f_rd = {map_f_v5c:.3f}  68%=[{hdi68_fv5c[0]:.3f},{hdi68_fv5c[1]:.3f}]")
print("=" * 65)


# ── Figure ────────────────────────────────────────────────────────────────────
DARK_BG    = "#0d1117"
LIGHT_TEXT = "#e6edf3"
ACCENT     = "#58a6ff"
ORANGE     = "#f0883e"
GREEN      = "#3fb950"
RED        = "#ff7b72"
YELLOW     = "#e3b341"

fig = plt.figure(figsize=(16, 11), facecolor=DARK_BG)

ax_rho = fig.add_axes([0.05, 0.58, 0.38, 0.36])   # per-event ρ comparison
ax_gam = fig.add_axes([0.50, 0.58, 0.45, 0.36])   # Γ posteriors overlay
ax_2d  = fig.add_axes([0.05, 0.08, 0.45, 0.40])   # joint 2D corrected
ax_frd = fig.add_axes([0.57, 0.08, 0.38, 0.40])   # marginal f_rd comparison

for ax in [ax_rho, ax_gam, ax_2d, ax_frd]:
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=LIGHT_TEXT, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")

labels = [f"{ev}/{det}" for ev, det, *_ in CORRECTED]
x_pos  = np.arange(len(labels))

# --- Per-event ρ_syn comparison ---
ax_rho.bar(x_pos - 0.2, RHO_SYN_B, 0.35, color=RED,   alpha=0.7, label="ρ_syn biased (Phase III)")
ax_rho.bar(x_pos + 0.2, RHO_SYN_C, 0.35, color=GREEN, alpha=0.9, label="ρ_syn corrected (Phase VII)")
ax_rho.plot(x_pos, RHO_OBS, "o", color=YELLOW, ms=5, zorder=5, label="ρ_obs (data)")
ax_rho.axhline(1.0, color=LIGHT_TEXT, lw=0.7, ls="--", alpha=0.5)
ax_rho.set_xticks(x_pos)
ax_rho.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
ax_rho.set_ylabel("SNR", color=LIGHT_TEXT, fontsize=9)
ax_rho.set_title("Per-event ρ_syn: biased vs corrected", color=LIGHT_TEXT, fontsize=9)
ax_rho.legend(fontsize=7, facecolor=DARK_BG, labelcolor=LIGHT_TEXT, framealpha=0.7)

# --- Γ posteriors overlay ---
ax_gam.plot(G_bias, post_bias, color=RED,   lw=1.5, ls="--", alpha=0.8,
            label="Phase III/IV (biased)")
ax_gam.fill_between(G_bias, 0, post_bias, alpha=0.06, color=RED)
ax_gam.plot(G_corr, post_corr, color=GREEN, lw=2.0,
            label="Phase VII (corrected)")
ax_gam.fill_between(G_corr, 0, post_corr, alpha=0.10, color=GREEN)

for name, G_val in GAMMA_PREDS.items():
    ax_gam.axvline(G_val, color=PRED_COLORS[name], lw=0.9, ls=":", alpha=0.85)
    ax_gam.text(G_val + 0.02, ax_gam.get_ylim()[1] * 0.98 if ax_gam.get_ylim()[1] > 0 else 0.5,
                name, color=PRED_COLORS[name], fontsize=7, va="top")

ax_gam.set_xlabel("Γ  (amplitude ratio A_obs / A_GR)", color=LIGHT_TEXT, fontsize=9)
ax_gam.set_ylabel("p(Γ | data)", color=LIGHT_TEXT, fontsize=9)
ax_gam.set_title("Γ posterior: Phase III/IV biased vs Phase VII corrected",
                 color=LIGHT_TEXT, fontsize=9)
ax_gam.legend(fontsize=8, facecolor=DARK_BG, labelcolor=LIGHT_TEXT, framealpha=0.7)
ax_gam.set_xlim(0.0, 3.5)
ax_gam.set_ylim(bottom=0)

# --- 2D joint posterior (corrected) ---
GG, FF = np.meshgrid(Gamma_g, frd_g, indexing="ij")
ax_2d.pcolormesh(GG, FF, post_2d, cmap="inferno", shading="auto")
flat = post_2d.flatten()
isort = np.argsort(flat)[::-1]
cs_2d = np.cumsum(flat[isort]) * dG * df
t68 = float(flat[isort[np.searchsorted(cs_2d, 0.68)]])
t95 = float(flat[isort[np.searchsorted(cs_2d, 0.95)]])
ax_2d.contour(GG, FF, post_2d, levels=[t95, t68],
              colors=["white", "white"], linestyles=["--", "-"], linewidths=[0.8, 1.2])
for name, G_val in GAMMA_PREDS.items():
    ax_2d.axvline(G_val, color=PRED_COLORS[name], lw=1.0, ls=":", alpha=0.85)
ax_2d.set_xlabel("Γ", color=LIGHT_TEXT, fontsize=9)
ax_2d.set_ylabel("f_rd", color=LIGHT_TEXT, fontsize=9)
ax_2d.set_title("Corrected joint Γ × f_rd posterior\n—/·· = 68%/95% credible contours",
                color=LIGHT_TEXT, fontsize=9)
ax_2d.set_xlim(0, 4)
ax_2d.set_ylim(FRD_MIN, FRD_MAX)

# --- Marginal f_rd ---
# Load Phase V biased f_rd marginal for comparison (recompute quickly)
logpost_b = np.zeros((N_G, N_F))
for j, frd in enumerate(frd_g):
    scale     = math.sqrt(frd / F_RD_REF)
    rho_syn_e = RHO_SYN_B * scale
    lp_frd    = -math.log(frd) - LOG_PRIOR_NORM
    for i, G in enumerate(Gamma_g):
        s_arr       = G * rho_syn_e
        ll          = sum(rician_logpdf(float(A), float(s)) for A, s in zip(RHO_OBS, s_arr))
        logpost_b[i, j] = ll + lp_frd
logpost_b -= logpost_b.max()
post_b = np.exp(logpost_b)
norm_b = np.trapezoid(np.trapezoid(post_b, frd_g, axis=1), Gamma_g)
post_b /= norm_b
marg_f_bias = np.trapezoid(post_b, Gamma_g, axis=0)
marg_f_bias /= np.trapezoid(marg_f_bias, frd_g)

ax_frd.plot(frd_g, marg_f_bias, color=RED,   lw=1.5, ls="--", label="Biased (Phase V)")
ax_frd.plot(frd_g, marg_f_corr, color=GREEN, lw=2.0, label="Corrected (Phase VII)")
ax_frd.fill_between(frd_g, 0, marg_f_corr, alpha=0.12, color=GREEN)
ax_frd.axvline(F_RD_REF, color=LIGHT_TEXT, lw=0.8, ls="--", alpha=0.6, label=f"f_rd ref={F_RD_REF}")
ax_frd.set_xlabel("f_rd  (ringdown energy fraction)", color=LIGHT_TEXT, fontsize=9)
ax_frd.set_ylabel("p(f_rd | data)", color=LIGHT_TEXT, fontsize=9)
ax_frd.set_title("Marginal f_rd posterior: biased vs corrected\n(Γ integrated out)",
                 color=LIGHT_TEXT, fontsize=9)
ax_frd.legend(fontsize=8, facecolor=DARK_BG, labelcolor=LIGHT_TEXT, framealpha=0.7)
ax_frd.set_xlim(FRD_MIN, FRD_MAX)
ax_frd.set_ylim(bottom=0)

fig.suptitle(
    "Phase VII — Corrected Amplitude-Ratio + Bayesian Analysis  |  exp(−t_start/τ) bias removed",
    color=LIGHT_TEXT, fontsize=11, y=0.98,
)

out_dir  = os.path.join(os.path.dirname(__file__), "..", "misc")
fig_path = os.path.join(out_dir, "bh_phase_vii_corrected_analysis.png")
fig.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close(fig)
print(f"\nFigure saved → {fig_path}")
print(f"Size: {os.path.getsize(fig_path)/1024:.1f} KB")
