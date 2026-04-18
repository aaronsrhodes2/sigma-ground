"""Phase IV — Bayesian amplitude ratio inference using Rician likelihood.

The simple ratio Γ = ρ_obs/ρ_syn (Phase III) is biased when ρ_syn << 1:
pure Rayleigh noise gives E[ρ_obs] = sqrt(π/2) regardless of ρ_syn, so
Γ_noise → sqrt(π/2)/ρ_syn → ∞ for small denominators.

Correct framework
-----------------
The matched-filter amplitude A = sqrt(ρ_cos² + ρ_sin²) is Rician-distributed
under a signal with SNR s:

    Under H₀ (no signal):  ρ_cos, ρ_sin ~ N(0, 1)  → A ~ Rayleigh(1)
    Under H₁ (signal SNR s): A ~ Rician(s, 1)        → p(A|s) = A exp(-(A²+s²)/2) I₀(As)

For a putative QNM amplitude suppression Γ relative to GR:
    s_i = Γ × ρ_syn_i   (per-event signal SNR)

Combined log-posterior (uniform prior on Γ ≥ 0):
    ln p(Γ | data) ∝ Σ_i ln p(ρ_obs_i | Γ × ρ_syn_i)

This is proper: events with ρ_syn_i ≈ 0 contribute a Γ-flat term and are
automatically down-weighted.  Events with ρ_syn_i ~ O(1) drive the posterior.

Input data (Phase III results, f_rd = 0.15, published LVK MAP parameters)
--------------------------------------------------------------------------
GW150914/H1 and L1 use the 16 kHz files (better PSD estimation at 251 Hz);
all other events use 4 kHz as in Phase III.

γ-model predictions tested
---------------------------
  GR           : Γ = 1.0000
  exp          : Γ = 0.8395   (Phase G phenomenology)
  sigma_coh    : Γ = 0.7924   (1 − η/2)
  linear/cbrt  : Γ = 0.7461   (η^(1/3) = Θ)
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
from scipy.special import i0 as bessel_i0

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..')))
from sigma_ground.field.constants import ETA

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ETA_VAL = ETA   # 0.4153

GAMMA_PREDICTIONS = {
    'GR':          1.0000,
    'exp':         0.8395,
    'sigma_coh':   round(1.0 - ETA_VAL / 2.0, 4),
    'linear_cbrt': round(ETA_VAL ** (1.0 / 3.0), 4),
}
PRED_COLORS = {
    'GR': '#f55', 'exp': '#fa0', 'sigma_coh': '#6af', 'linear_cbrt': '#8f8',
}

# ── Phase III results (rho_obs, rho_syn) per detector-event ───────────────────
# GW150914 uses 16 kHz (better PSD); all others use 4 kHz from Phase III.
# rho_obs = Phase-II matched-filter SNR at QNM window (calibrated vs noise).
# rho_syn = Matched-filter response to GR-predicted injection (same pipeline).
PHASE_III_DATA = [
    # event         det   rho_obs   rho_syn
    ('GW150914',   'H1',  1.000,   1.5430),   # 16 kHz
    ('GW150914',   'L1',  0.734,   0.1764),   # 16 kHz
    ('GW151226',   'H1',  1.458,   0.0114),   # 4 kHz (short τ, decayed)
    ('GW151226',   'L1',  1.205,   0.0016),   # 4 kHz (essentially no signal)
    ('GW170814',   'H1',  1.216,   1.1302),   # 4 kHz
    ('GW170814',   'L1',  1.059,   0.2132),   # 4 kHz
    ('GW170814',   'V1',  1.402,   0.0196),   # 4 kHz
    ('GW170104',   'H1',  0.392,   0.4493),   # 4 kHz
    ('GW170104',   'L1',  0.875,   0.0566),   # 4 kHz
    ('GW190521',   'H1',  2.164,   1.0741),   # 4 kHz
    ('GW190521',   'L1',  0.395,   1.5491),   # 4 kHz
]


# ── Rician likelihood ─────────────────────────────────────────────────────────

def rician_logpdf(A, s):
    """Log-PDF of Rician(s, σ=1) evaluated at A ≥ 0.

    p(A | s) = A exp(-(A² + s²)/2) I₀(As)

    For s=0 this reduces to the Rayleigh(1) distribution.
    Numerically stable: use log-scaled Bessel function for large As.
    """
    if A <= 0:
        return -1e300
    s = max(s, 0.0)
    log_A   = math.log(A)
    quad    = -(A * A + s * s) / 2.0
    log_I0  = float(np.log(bessel_i0(A * s) + 1e-300))
    return log_A + quad + log_I0


def log_posterior(Gamma, rho_obs_list, rho_syn_list):
    """Combined log-posterior (log-likelihood with uniform prior Γ ∈ [0, 3])."""
    if Gamma < 0:
        return -1e300
    ll = 0.0
    for A, rho_syn in zip(rho_obs_list, rho_syn_list):
        s = Gamma * rho_syn
        ll += rician_logpdf(A, s)
    return ll


# ── Posterior computation ─────────────────────────────────────────────────────

def compute_posterior(rho_obs_list, rho_syn_list,
                      gamma_min=0.0, gamma_max=3.0, n_pts=2000):
    """Evaluate the Bayesian posterior p(Γ | data) on a dense grid."""
    Gamma_grid = np.linspace(gamma_min, gamma_max, n_pts)
    logpost = np.array([log_posterior(G, rho_obs_list, rho_syn_list)
                        for G in Gamma_grid])
    logpost -= logpost.max()   # shift for numerical stability
    post = np.exp(logpost)
    norm = np.trapezoid(post, Gamma_grid)
    post /= norm
    return Gamma_grid, post


def credible_interval(Gamma_grid, post, level=0.68):
    """Highest-posterior-density credible interval at given probability level."""
    cdf  = np.cumsum(post) * (Gamma_grid[1] - Gamma_grid[0])
    # MAP
    map_idx = np.argmax(post)
    Gamma_map = float(Gamma_grid[map_idx])
    # HDI: expand from MAP outward
    lo_i, hi_i = map_idx, map_idx
    prob = 0.0
    while prob < level and (lo_i > 0 or hi_i < len(Gamma_grid) - 1):
        expand_lo = post[lo_i - 1] if lo_i > 0 else -1
        expand_hi = post[hi_i + 1] if hi_i < len(Gamma_grid) - 1 else -1
        if expand_lo >= expand_hi and lo_i > 0:
            lo_i -= 1
        elif hi_i < len(Gamma_grid) - 1:
            hi_i += 1
        elif lo_i > 0:
            lo_i -= 1
        else:
            break
        prob = np.trapezoid(post[lo_i: hi_i + 1], Gamma_grid[lo_i: hi_i + 1])
    return Gamma_map, float(Gamma_grid[lo_i]), float(Gamma_grid[hi_i])


def marginal_probability(Gamma_grid, post, val, side='above'):
    """P(Γ > val) or P(Γ < val) from the posterior."""
    dG = Gamma_grid[1] - Gamma_grid[0]
    if side == 'above':
        mask = Gamma_grid >= val
    else:
        mask = Gamma_grid < val
    return float(np.sum(post[mask]) * dG)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    names    = [f'{ev}/{det}' for ev, det, _, _ in PHASE_III_DATA]
    rho_obs  = [r  for _, _, r, _ in PHASE_III_DATA]
    rho_syn  = [s  for _, _, _, s in PHASE_III_DATA]

    print('Phase IV — Bayesian Rician amplitude ratio inference')
    print(f'Model predictions: { {k: f"{v:.4f}" for k, v in GAMMA_PREDICTIONS.items()} }')
    print()
    print(f'{"Event/Det":<15} {"ρ_obs":>7} {"ρ_syn":>7} '
          f'{"Rician(Γ=1)":>12} {"Rician(Γ=0.79)":>15}')
    print('-' * 60)
    for name, A, s in zip(names, rho_obs, rho_syn):
        ll_gr   = rician_logpdf(A, 1.0 * s)
        ll_sc   = rician_logpdf(A, 0.7924 * s)
        print(f'{name:<15} {A:>7.3f} {s:>7.4f} {ll_gr:>12.3f} {ll_sc:>15.3f}')

    # Full posterior
    Gamma_grid, post = compute_posterior(rho_obs, rho_syn)
    Gamma_map, lo68, hi68 = credible_interval(Gamma_grid, post, level=0.68)
    _, lo95, hi95         = credible_interval(Gamma_grid, post, level=0.95)

    prob_below_gr   = marginal_probability(Gamma_grid, post, 1.0, side='below')
    prob_below_sc   = marginal_probability(Gamma_grid, post, 0.7924, side='below')

    print(f'\n{"="*70}')
    print(f'Bayesian posterior on Γ = A_obs / A_GR')
    print(f'  MAP Γ         = {Gamma_map:.3f}')
    print(f'  68% HDI       = [{lo68:.3f}, {hi68:.3f}]')
    print(f'  95% HDI       = [{lo95:.3f}, {hi95:.3f}]')
    print()
    print(f'  P(Γ < 1.000 | data)  = {prob_below_gr:.3f}  '
          f'(supports amplitude suppression)')
    print(f'  P(Γ < 0.792 | data)  = {prob_below_sc:.3f}  '
          f'(consistent with sigma_coh)')
    print()
    print(f'  {"Model":<15} {"Γ_pred":>8} {"P(Γ>Γ_pred)":>13} {"P(Γ<Γ_pred)":>13}')
    print('  ' + '-' * 52)
    for name, val in GAMMA_PREDICTIONS.items():
        pa = marginal_probability(Gamma_grid, post, val, side='above')
        pb = marginal_probability(Gamma_grid, post, val, side='below')
        print(f'  {name:<15} {val:>8.4f} {pa:>13.3f} {pb:>13.3f}')
    print(f'{"="*70}')

    # ── Event-level likelihoods ───────────────────────────────────────────────
    print(f'\nPer-event log-likelihood ratio ln[p(Γ=0.79)/p(Γ=1.0)]:')
    for name, A, s in zip(names, rho_obs, rho_syn):
        lr = rician_logpdf(A, 0.7924 * s) - rician_logpdf(A, 1.0 * s)
        print(f'  {name:<15}  ρ_syn={s:.4f}  ΔlnL = {lr:+.3f}  '
              f'{"→ sigma_coh" if lr > 0 else "→ GR"}')

    # ── Figure ────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        'figure.facecolor': '#111', 'axes.facecolor': '#181818',
        'savefig.facecolor': '#111', 'text.color': '#eee',
        'axes.labelcolor': '#ccc', 'font.family': 'DejaVu Sans',
    })
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        r'Phase IV — Bayesian $\Gamma$ inference via Rician likelihood'
        '\n'
        r'$p(A_\mathrm{obs} | \Gamma) = \mathrm{Rician}(A_\mathrm{obs};\, '
        r'\Gamma \cdot \rho_\mathrm{syn},\, 1)$'
        f'\n MAP Γ = {Gamma_map:.3f}   68% HDI = [{lo68:.3f}, {hi68:.3f}]',
        fontsize=10.5, color='#eee', y=0.99)

    # Left: posterior
    ax1.fill_between(Gamma_grid, post, color='#6af', alpha=0.30)
    ax1.plot(Gamma_grid, post, color='#6af', lw=1.5, label='p(Γ | data)')
    # 68% and 95% HDI shading
    for lo, hi, alpha, lbl in [(lo68, hi68, 0.25, '68% HDI'),
                                (lo95, hi95, 0.12, '95% HDI')]:
        mask = (Gamma_grid >= lo) & (Gamma_grid <= hi)
        ax1.fill_between(Gamma_grid[mask], post[mask],
                         color='#9cf', alpha=alpha, label=lbl)
    ax1.axvline(Gamma_map, color='#fff', lw=1.4, ls='-', label=f'MAP={Gamma_map:.3f}')
    for name, val in GAMMA_PREDICTIONS.items():
        ax1.axvline(val, color=PRED_COLORS[name], lw=1.2, ls='--', alpha=0.85,
                    label=f'{name}: Γ={val:.4f}')
    ax1.set_xlabel(r'$\Gamma = A_\mathrm{obs}/A_\mathrm{GR}$', fontsize=10)
    ax1.set_ylabel('Posterior density', fontsize=9)
    ax1.set_xlim(0, 2.5)
    ax1.set_title('Combined posterior (11 detector-events)\nRician likelihood, uniform prior',
                  fontsize=9, color='#ddd')
    ax1.legend(fontsize=8, framealpha=0.15, labelcolor='#ddd')
    ax1.tick_params(colors='#bbb')
    for sp in ax1.spines.values():
        sp.set_color('#444')

    # Right: per-event Rician likelihood profiles p(Γ) ∝ Rician(ρ_obs; Γ·ρ_syn, 1)
    ax2.set_title('Per-event likelihood profile  p(Γ | ρ_obs_i)\n'
                  '(normalised to unit peak; only events with ρ_syn > 0.1 shown)',
                  fontsize=9, color='#ddd')
    colors_evt = plt.cm.plasma(np.linspace(0.1, 0.9, len(PHASE_III_DATA)))

    Gamma_fine = np.linspace(0, 3, 1000)
    for i, (name, A, s) in enumerate(zip(names, rho_obs, rho_syn)):
        if s < 0.1:
            continue   # skip events with negligible GR signal (flat likelihood)
        ll_evt = np.array([rician_logpdf(A, G * s) for G in Gamma_fine])
        ll_evt -= ll_evt.max()
        lik_evt = np.exp(ll_evt)
        ax2.plot(Gamma_fine, lik_evt, color=colors_evt[i], lw=1.1, alpha=0.8,
                 label=f'{name}  (ρ_syn={s:.2f})')

    # Show combined
    logpost_fine = np.array([log_posterior(G, rho_obs, rho_syn)
                              for G in Gamma_fine])
    logpost_fine -= logpost_fine.max()
    post_fine = np.exp(logpost_fine)
    ax2.plot(Gamma_fine, post_fine / post_fine.max(), color='#fff', lw=2.0,
             ls='-', alpha=0.9, label='Combined (normalised)')

    for name, val in GAMMA_PREDICTIONS.items():
        ax2.axvline(val, color=PRED_COLORS[name], lw=1.0, ls='--', alpha=0.7)
    ax2.set_xlabel(r'$\Gamma = A_\mathrm{obs}/A_\mathrm{GR}$', fontsize=9)
    ax2.set_ylabel('Normalised likelihood', fontsize=9)
    ax2.set_xlim(0, 2.5)
    ax2.legend(fontsize=7.5, framealpha=0.15, labelcolor='#ddd', loc='upper right')
    ax2.tick_params(colors='#bbb')
    for sp in ax2.spines.values():
        sp.set_color('#444')

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = os.path.normpath(
        os.path.join(os.path.dirname(__file__),
                     '..', 'misc', 'bh_phase_iv_bayesian_gamma.png'))
    fig.savefig(out, dpi=125, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f'\nwrote {out} ({os.path.getsize(out) / 1024:.1f} KB)')

    return Gamma_map, lo68, hi68, lo95, hi95


if __name__ == '__main__':
    main()
