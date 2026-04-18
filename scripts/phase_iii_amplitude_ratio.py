"""Phase III — QNM amplitude ratio test (σ_conv vs GR).

Tests whether the observed (2,2,0) QNM amplitude at ringdown onset
is consistent with GR (Γ = 1.0) or the σ_conv entanglement-decoherence
predictions (Γ = γ < 1).

Observable
----------
    Γ = A_obs / A_GR = ρ_obs / ρ_syn

where:
  ρ_obs  = Phase-II matched-filter SNR at the QNM window (from data)
  ρ_syn  = Matched-filter SNR of the GR-predicted signal injected through
            the same whitening + matched-filter pipeline (synthetic)

Because the whitening operator W is linear and σ_noise divides both:
    Γ = amp_obs / amp_syn     (σ_noise cancels exactly)

Source-frame amplitude (energy conservation, (2,2,0) mode, SVEA)
-----------------------------------------------------------------
    E_ring = ε_rd × M_f c² = (πc³/5G) × A₂₂² × f_QNM × Q

    →  A₂₂ = sqrt(5G ε_rd M_f / (πc f_QNM Q))   [metres]

where Q = π f_QNM τ_QNM and ε_rd = f_rd × E_rad / (M_f c²).

Derivation:  GW luminosity (SVEA, sky-integrated over (2,2) angular pattern):
    dE/dt = (4π²c³/20G) × A₂₂² × f² × exp(−2t/τ)
    E_ring = ∫₀^∞ dE/dt dt = (π²c³/5G) × A₂₂² × f² × τ
           = (πc³/5G) × A₂₂² × f_QNM × Q   [with Q = πfτ]

Detector amplitude at ringdown start
--------------------------------------
    A_det = (A₂₂ / d_L) × R_rms(ι)

    R_rms(ι) = sqrt[(1/5) × ((1+cos²ι)²/4 + cos²ι)]
    Sky-average: ⟨F₊²⟩ = ⟨F×²⟩ = 1/5, ⟨F₊F×⟩ = 0.
    Inclination ι (= θ_JN) from LVK published MAP posteriors.

γ predictions to test
----------------------
    GR           : Γ_pred = 1.0000
    exp          : Γ_pred = 0.8395   (Phase G phenomenology)
    sigma_coh    : Γ_pred = 1 − η/2  = 0.7924
    linear/cbrt  : Γ_pred = η^(1/3)  = 0.7461  (Θ-endpoint)

Cross-references
----------------
  Phase I.4  : time-domain catalog null, p_F = 0.609
  Phase I.5  : frequency-domain comb null, R = 0.916
  Phase II   : amplitude bound, ξ_UL = 1.0 (untestable)
  This phase : scripts/phase_iii_amplitude_ratio.py
  Result doc : misc/bh_phase_iii_amplitude_ratio_results.md
  Figure     : misc/bh_phase_iii_amplitude_ratio.png
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..')))
from sigma_ground.field.constants import ETA, XI, SIGMA_CONV
from sigma_ground.field.interface.ligo_echo_search import (
    EVENTS, fetch_strain, read_strain, find_merger_sample,
    estimate_psd, bandpass, whiten,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── Physical constants (consistent with ligo_echo_search.py) ─────────────────
G_SI    = 6.67430e-11       # m³ kg⁻¹ s⁻²
C_SI    = 299792458.0       # m/s
M_SUN   = 1.98892e30        # kg
MPC_M   = 3.085677581e22    # metres per megaparsec

# ── σ_conv model predictions ──────────────────────────────────────────────────
ETA_VAL = ETA   # 0.4153

GAMMA_PREDICTIONS = {
    'GR':          1.0000,
    'exp':         0.8395,                    # Phase G phenomenology
    'sigma_coh':   round(1.0 - ETA_VAL / 2.0, 4),   # 0.7924
    'linear_cbrt': round(ETA_VAL ** (1.0 / 3.0), 4), # Θ = η^(1/3) ≈ 0.7461
}

# ── Analysis parameters (identical to Phase I & II) ───────────────────────────
RINGDOWN_START_MS   = 3.0
TEMPLATE_DURATION_MS = 20.0
F_RD = 0.15   # ringdown energy fraction f_rd ∈ [0.10, 0.20]; uncertainty ~±30 %

# ── Published LVK parameters per event ───────────────────────────────────────
# E_rad_msun : total GW-radiated energy [M☉c²], source frame, MAP
# d_L_mpc    : luminosity distance [Mpc], MAP
# iota_deg   : inclination θ_JN [°], MAP
#
# Sources: GW150914 → PRL 116, 061102 (2016)
#          GW151226 → PRL 116, 241103 (2016)
#          GW170814 → PRL 119, 141101 (2017)
#          GW170104 → PRL 118, 221101 (2017)
#          GW190521 → ApJL 900, L13 (2020)
PHASE_III_PARAMS = {
    'GW150914': {'E_rad_msun': 3.0,  'd_L_mpc':  410.0, 'iota_deg': 163.0},
    'GW151226': {'E_rad_msun': 1.0,  'd_L_mpc':  440.0, 'iota_deg':  88.0},
    'GW170814': {'E_rad_msun': 2.7,  'd_L_mpc':  540.0, 'iota_deg':  54.0},
    'GW170104': {'E_rad_msun': 2.0,  'd_L_mpc':  880.0, 'iota_deg': 135.0},
    'GW190521': {'E_rad_msun': 8.0,  'd_L_mpc': 5300.0, 'iota_deg': 153.0},
}

# Same 11 detector-events as Phase II
CATALOG = [
    ('GW150914', 'H1',  4),
    ('GW150914', 'L1',  4),
    ('GW151226', 'H1', 16),
    ('GW151226', 'L1', 16),
    ('GW170814', 'H1',  4),
    ('GW170814', 'L1',  4),
    ('GW170814', 'V1',  4),
    ('GW170104', 'H1',  4),
    ('GW170104', 'L1',  4),
    ('GW190521', 'H1',  4),
    ('GW190521', 'L1',  4),
]


# ── Physics helpers ───────────────────────────────────────────────────────────

def a22_source_m(M_f_kg, eps_rd, f_qnm_hz, tau_qnm_s):
    """Source-frame QNM amplitude A₂₂ [metres].

    From GW energy conservation for the (2,2,0) Kerr QNM (SVEA):
        E_ring = ε_rd M_f c²  =  (πc³/5G) A₂₂² f_QNM Q
    with Q = π f_QNM τ_QNM.
    """
    Q = math.pi * f_qnm_hz * tau_qnm_s
    return math.sqrt((5.0 * G_SI * eps_rd * M_f_kg)
                     / (math.pi * C_SI * f_qnm_hz * Q))


def r_rms_sky_avg(iota_deg):
    """Sky-averaged antenna amplitude factor at known inclination.

    For (2,2) mode: h₊ ∝ (1+cos²ι)/2, h× ∝ cosι.
    Averaging over polarisation angle ψ (⟨F₊²⟩ = ⟨F×²⟩ = 1/5, ⟨F₊F×⟩ = 0):
        R²_rms = (1/5) × [(1+cos²ι)²/4 + cos²ι]
    """
    c = math.cos(math.radians(iota_deg))
    return math.sqrt((1.0 / 5.0) * ((1.0 + c * c) ** 2 / 4.0 + c * c))


# ── Core computation ──────────────────────────────────────────────────────────

def compute_amplitude_ratio(event_name, detector, fs_khz,
                             f_rd=F_RD, verbose=True):
    """Compute Γ = ρ_obs / ρ_syn for one detector-event.

    ρ_obs : Phase-II matched-filter SNR (observed QNM amplitude / noise).
    ρ_syn : Same pipeline applied to the GR-predicted injection signal.
    Γ     : amp_obs / amp_syn — σ_noise cancels in the ratio.

    Returns a dict or None if data unavailable / segment too short.
    """
    meta   = EVENTS[event_name]
    params = PHASE_III_PARAMS[event_name]

    path = fetch_strain(event_name, detector, fs_khz=fs_khz)
    strain, t0_gps, fs = read_strain(path)
    merger_idx = find_merger_sample(event_name, t0_gps, fs)
    rd_start = merger_idx + int(RINGDOWN_START_MS * 1e-3 * fs)

    psd_len   = int(8.0 * fs)
    psd_start = max(0, merger_idx - int(2.0 * fs) - psd_len)
    psd_freqs, psd_vals = estimate_psd(strain, fs, psd_start, psd_len)

    bp = bandpass(strain, fs, lo=35.0, hi=min(fs / 2.0 * 0.45, 1500.0))

    f_qnm   = meta['f_qnm_hz']
    tau_qnm = meta['tau_qnm_s']
    n_tmpl  = int(TEMPLATE_DURATION_MS * 1e-3 * fs)

    # Quadrature whitened templates — identical to Phase II
    t_tmpl  = np.arange(n_tmpl) / fs
    env     = np.exp(-t_tmpl / tau_qnm)
    tc_raw  = env * np.cos(2.0 * np.pi * f_qnm * t_tmpl)
    ts_raw  = env * np.sin(2.0 * np.pi * f_qnm * t_tmpl)
    tc_wht  = whiten(tc_raw, fs, psd_freqs, psd_vals)
    ts_wht  = whiten(ts_raw, fs, psd_freqs, psd_vals)
    norm_sq = float(np.dot(tc_wht, tc_wht) + np.dot(ts_wht, ts_wht))
    norm    = math.sqrt(norm_sq)
    if norm <= 0:
        return None

    def quad_amp(sig_wht):
        rc = float(np.dot(sig_wht, tc_wht)) / norm
        rs = float(np.dot(sig_wht, ts_wht)) / norm
        return math.sqrt(rc * rc + rs * rs)

    # Observed amplitude at QNM window
    seg = bp[rd_start: rd_start + n_tmpl]
    if len(seg) < n_tmpl:
        return None
    amp_obs = quad_amp(whiten(seg, fs, psd_freqs, psd_vals))

    # Background calibration: 300 pre-merger samples [−12 s, −2 s] (seed 42)
    rng     = np.random.default_rng(seed=42)
    bg_vals = []
    for _ in range(300):
        idx = merger_idx - int(rng.uniform(2.0, 12.0) * fs)
        if idx >= 0:
            bg_seg = bp[idx: idx + n_tmpl]
            if len(bg_seg) == n_tmpl:
                bg_vals.append(quad_amp(whiten(bg_seg, fs, psd_freqs, psd_vals)))
    bg_mean     = float(np.mean(bg_vals)) if bg_vals else math.sqrt(math.pi / 2.0)
    sigma_noise = bg_mean / math.sqrt(math.pi / 2.0)
    rho_obs     = amp_obs / sigma_noise if sigma_noise > 0 else float('nan')

    # ── GR-predicted amplitude ─────────────────────────────────────────────
    M_f_kg  = meta['M_rem_msun'] * M_SUN
    eps_rd  = f_rd * params['E_rad_msun'] / meta['M_rem_msun']
    d_L_m   = params['d_L_mpc'] * MPC_M
    R_rms   = r_rms_sky_avg(params['iota_deg'])

    A22     = a22_source_m(M_f_kg, eps_rd, f_qnm, tau_qnm)
    A_det   = (A22 / d_L_m) * R_rms   # detector amplitude at ringdown start

    # Synthetic injection: h_inj = A_det × exp(−t/τ) × cos(2πft), phase φ=0.
    # Whitening is linear → amp_syn = A_det × (pipeline response to unit template).
    # Phase-independence: quadrature filter gives amp = A_det × C regardless of φ.
    h_inj    = A_det * env * np.cos(2.0 * np.pi * f_qnm * t_tmpl)
    amp_syn  = quad_amp(whiten(h_inj, fs, psd_freqs, psd_vals))
    rho_syn  = amp_syn / sigma_noise if sigma_noise > 0 else float('nan')

    # Γ = amp_obs / amp_syn (σ_noise cancels; same as rho_obs / rho_syn)
    gamma = amp_obs / amp_syn if amp_syn > 0 else float('nan')

    # Decay fraction at t_start (informational)
    decay_frac = math.exp(-RINGDOWN_START_MS * 1e-3 / tau_qnm)

    if verbose:
        print(f'  {event_name}/{detector} fs={fs_khz}kHz | '
              f'ρ_obs={rho_obs:.3f}  ρ_syn={rho_syn:.4f}  Γ={gamma:.3f} | '
              f'A22={A22:.0f}m  A_det={A_det:.2e}  R_rms={R_rms:.3f} | '
              f'decay@{RINGDOWN_START_MS}ms={decay_frac:.2f}')

    return {
        'event':      event_name,
        'detector':   detector,
        'fs_khz':     fs_khz,
        'rho_obs':    rho_obs,
        'rho_syn':    rho_syn,
        'gamma':      gamma,
        'A22_m':      A22,
        'A_det':      A_det,
        'R_rms':      R_rms,
        'eps_rd':     eps_rd,
        'decay_frac': decay_frac,
        'sigma_noise': sigma_noise,
    }


# ── Combined estimate ─────────────────────────────────────────────────────────

def combined_gamma(rows):
    """ρ_syn²-weighted mean Γ and weighted scatter.

    Events with larger GR-predicted signal dominate the combined estimate,
    reducing the bias from noise-dominated low-ρ_syn events.
    """
    finite = [r for r in rows
              if r is not None and np.isfinite(r['gamma']) and r['rho_syn'] > 0]
    if not finite:
        return float('nan'), float('nan')
    w  = np.array([r['rho_syn'] ** 2 for r in finite])
    gv = np.array([r['gamma']        for r in finite])
    w_sum     = float(w.sum())
    g_mean    = float((w * gv).sum()) / w_sum
    g_std     = float(np.sqrt((w * (gv - g_mean) ** 2).sum() / w_sum))
    return g_mean, g_std


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f'\nPhase III — QNM amplitude ratio test')
    print(f'f_rd = {F_RD}  (ringdown energy fraction; ±30% systematic)')
    print(f'Predictions: { {k: f"{v:.4f}" for k, v in GAMMA_PREDICTIONS.items()} }')
    print()

    rows = []
    for event_name, detector, fs_khz in CATALOG:
        print(f'[{event_name}/{detector}]')
        result = compute_amplitude_ratio(event_name, detector, fs_khz)
        rows.append(result)

    # ── Summary table ─────────────────────────────────────────────────────────
    header = f'\n{"Event/Det":<18} {"ρ_obs":>7} {"ρ_syn":>8} {"Γ":>8} ' \
             f'{"A_det":>12} {"R_rms":>7} {"decay":>7}'
    print(f'\n{"="*80}')
    print(header)
    print('-' * 80)
    for r in rows:
        if r is None:
            continue
        print(f'{r["event"]}/{r["detector"]:<10} '
              f'{r["rho_obs"]:>7.3f} {r["rho_syn"]:>8.4f} {r["gamma"]:>8.3f} '
              f'{r["A_det"]:>12.2e} {r["R_rms"]:>7.3f} {r["decay_frac"]:>7.2f}')

    # ── Combined statistics ───────────────────────────────────────────────────
    gamma_mean, gamma_std = combined_gamma(rows)
    print(f'\n{"="*80}')
    print(f'Combined Γ (ρ_syn²-weighted) = {gamma_mean:.3f} ± {gamma_std:.3f}')
    print()
    print(f'{"Mode":<15} {"Γ_pred":>8} {"|Γ_obs−pred|":>14} {"σ-tension":>12}')
    print('-' * 52)
    for name, val in GAMMA_PREDICTIONS.items():
        diff = abs(gamma_mean - val)
        sig  = diff / gamma_std if gamma_std > 0 else float('nan')
        print(f'{name:<15} {val:>8.4f} {diff:>14.3f} {sig:>12.1f}σ')
    print(f'{"="*80}')

    # ── Sensitivity note ──────────────────────────────────────────────────────
    finite = [r for r in rows if r is not None and np.isfinite(r['rho_syn'])]
    rho_syn_arr = np.array([r['rho_syn'] for r in finite])
    rho_combined_syn = math.sqrt(float(np.sum(rho_syn_arr ** 2)))
    delta_gamma_needed = abs(GAMMA_PREDICTIONS['GR'] - GAMMA_PREDICTIONS['sigma_coh'])
    snr_needed = delta_gamma_needed / gamma_std if gamma_std > 0 else float('nan')
    print(f'\nSensitivity check:')
    print(f'  Combined (coherent) ρ_syn = {rho_combined_syn:.3f}')
    print(f'  GR−sigma_coh separation   = {delta_gamma_needed:.4f}')
    print(f'  Current 1σ on Γ           = {gamma_std:.3f}')
    print(f'  Sensitivity to sigma_coh  = {snr_needed:.1f}σ  '
          f'(need >2σ to distinguish from GR)')

    # ── Figure ────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        'figure.facecolor': '#111', 'axes.facecolor': '#181818',
        'savefig.facecolor': '#111', 'text.color': '#eee',
        'axes.labelcolor': '#ccc', 'font.family': 'DejaVu Sans',
    })
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        r'Phase III — QNM Amplitude Ratio  $\Gamma = A_\mathrm{obs}/A_\mathrm{GR}$'
        '\n'
        r'σ_conv prediction: $\Gamma_\mathrm{sigma\_coh} = 1 - \eta/2 = 0.792$   '
        r'vs GR: $\Gamma_\mathrm{GR} = 1.000$',
        fontsize=11, color='#eee', y=0.98)

    PRED_COLORS = {
        'GR': '#f55', 'exp': '#fa0', 'sigma_coh': '#6af', 'linear_cbrt': '#8f8',
    }

    finite_rows = [r for r in rows if r is not None and np.isfinite(r['gamma'])]
    labels   = [f'{r["event"][:7]}\n/{r["detector"]}' for r in finite_rows]
    gammas   = [r['gamma']   for r in finite_rows]
    rho_syns = [r['rho_syn'] for r in finite_rows]
    x = np.arange(len(finite_rows))

    # Left panel: per-detector-event Γ (marker size ∝ ρ_syn)
    ax = axes[0]
    sizes = np.clip(np.array(rho_syns) / max(rho_syns + [1e-9]) * 140 + 20, 10, 200)
    sc = ax.scatter(x, gammas, s=sizes, c=rho_syns, cmap='plasma',
                    vmin=0, vmax=max(rho_syns + [1e-9]),
                    edgecolors='#ccc', linewidths=0.5, zorder=3)
    plt.colorbar(sc, ax=ax, label=r'$\rho_\mathrm{syn}$ (GR-predicted SNR)',
                 pad=0.01, fraction=0.04)
    for name, val in GAMMA_PREDICTIONS.items():
        ax.axhline(val, color=PRED_COLORS[name], lw=1.2, ls='--', alpha=0.85,
                   label=f'{name}: Γ={val:.4f}')
    if np.isfinite(gamma_mean) and np.isfinite(gamma_std):
        ax.axhline(gamma_mean, color='#fff', lw=1.5, ls='-',
                   label=f'combined: {gamma_mean:.3f}±{gamma_std:.3f}')
        ax.axhspan(gamma_mean - gamma_std, gamma_mean + gamma_std,
                   color='#fff', alpha=0.07)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5, color='#ccc')
    ax.set_ylabel(r'$\Gamma = A_\mathrm{obs}/A_\mathrm{GR}$', fontsize=10)
    ax.set_title('Per-detector-event amplitude ratio\n'
                 '(marker size ∝ ρ_syn; dim = noise-dominated)',
                 fontsize=9, color='#ddd')
    ax.legend(fontsize=8, framealpha=0.15, labelcolor='#ddd', loc='upper right')
    ax.tick_params(colors='#bbb')
    for sp in ax.spines.values():
        sp.set_color('#444')

    # Right panel: Γ vs ρ_syn — shows which events are informative
    ax2 = axes[1]
    ax2.scatter(rho_syns, gammas, s=55, c='#6af',
                edgecolors='#9cf', linewidths=0.5, zorder=3)
    for r in finite_rows:
        ax2.annotate(
            f'{r["event"][:7]}/{r["detector"]}',
            (r['rho_syn'], r['gamma']),
            fontsize=6.5, color='#bbb',
            xytext=(4, 2), textcoords='offset points')
    for name, val in GAMMA_PREDICTIONS.items():
        ax2.axhline(val, color=PRED_COLORS[name], lw=1.0, ls='--', alpha=0.8,
                    label=f'{name}: {val:.4f}')
    if np.isfinite(gamma_mean):
        ax2.axhline(gamma_mean, color='#fff', lw=1.3, ls='-',
                    label=f'combined: {gamma_mean:.3f}')
    ax2.axvline(1.0, color='#888', lw=0.7, ls=':', alpha=0.5,
                label='ρ_syn = 1 (detection threshold)')
    ax2.set_xlabel(r'$\rho_\mathrm{syn}$ (GR-predicted matched-filter SNR)',
                   fontsize=9)
    ax2.set_ylabel(r'$\Gamma = A_\mathrm{obs}/A_\mathrm{GR}$', fontsize=9)
    ax2.set_title('Γ vs GR-predicted SNR\n'
                  '(events left of ρ_syn=1 are noise-dominated)',
                  fontsize=9, color='#ddd')
    ax2.legend(fontsize=8, framealpha=0.15, labelcolor='#ddd')
    ax2.tick_params(colors='#bbb')
    for sp in ax2.spines.values():
        sp.set_color('#444')

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = os.path.normpath(
        os.path.join(os.path.dirname(__file__),
                     '..', 'misc', 'bh_phase_iii_amplitude_ratio.png'))
    fig.savefig(out, dpi=125, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f'\nwrote {out} ({os.path.getsize(out) / 1024:.1f} KB)')

    return rows, gamma_mean, gamma_std


if __name__ == '__main__':
    main()
