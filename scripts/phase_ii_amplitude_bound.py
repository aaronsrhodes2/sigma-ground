"""Phase II — Amplitude upper bound and falsifiability bound.

Translates the Phase I null results into a constraint on the ξ-shell echo
reflectivity and compares that constraint to the σ_conv prediction.

Key identity
------------
The Phase I matched-filter evaluates the n-th echo at time
    t_eval = t_merger + ringdown_start_ms + n·Δt₁

The echo was launched at t_merger + n·Δt₁ and has therefore been ringing
for exactly ringdown_start_ms = t_start by the evaluation time.  The QNM
template matched at delay 0 also evaluates ringdown_start_ms after the QNM
launched.  Both carry the same exponential decay factor e^{−t_start/τ}.
Therefore the predicted ratio of echo SNR to in-window QNM SNR is simply ξ:

    ρ_echo_predicted = ξ · ρ_QNM_in_window

This is exact for the Phase I template shape (same exponential, same window);
any amplitude prediction for the echoes reduces to knowing ρ_QNM from the data.

Output
------
- Per-event table: ρ_QNM, ρ_echo_pred = ξ·ρ_QNM, single-event threshold
- Combined incoherent upper limit on ξ (from Fisher χ² at 95 % CL)
- Combined coherent sensitivity (hypothetical phase-preserving stack)
- Figure: ρ_echo_pred bars vs threshold + ξ sensitivity curve
- Verdict doc: misc/bh_phase_ii_amplitude_bound_results.md
"""
from __future__ import annotations

import os
import sys
import math

import numpy as np
from scipy import signal as scipy_signal
from scipy.stats import chi2, norm

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..')))
from sigma_ground.field.constants import XI, SIGMA_CONV
from sigma_ground.field.interface.ligo_echo_search import (
    EVENTS, fetch_strain, read_strain, find_merger_sample,
    estimate_psd, bandpass, whiten,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

XI_VAL = XI           # ≈ 0.1582
SIGMA_CONV_VAL = SIGMA_CONV  # ≈ 1.844

# Events + detectors used in Phase I.4
CATALOG = [
    ('GW150914', 'H1', 4),
    ('GW150914', 'L1', 4),
    ('GW151226', 'H1', 16),
    ('GW151226', 'L1', 16),
    ('GW170814', 'H1', 4),
    ('GW170814', 'L1', 4),
    ('GW170814', 'V1', 4),
    ('GW170104', 'H1', 4),
    ('GW170104', 'L1', 4),
    ('GW190521', 'H1', 4),
    ('GW190521', 'L1', 4),
]

RINGDOWN_START_MS = 3.0   # same as Phase I
TEMPLATE_DURATION_MS = 20.0  # match Phase I default
BACKGROUND_P99 = 2.3  # calibrated background |SNR| p99 from Phase I.4


# ── Core computation ────────────────────────────────────────────────────────

def compute_qnm_snr(event_name, detector, fs_khz=4,
                    ringdown_start_ms=RINGDOWN_START_MS,
                    template_ms=TEMPLATE_DURATION_MS,
                    verbose=True):
    """Phase-independent QNM matched-filter amplitude.

    Uses quadrature (cos + sin) templates so the result is independent of the
    unknown QNM initial phase.  Returns the Rayleigh-distributed amplitude

        A = sqrt(ρ_cos² + ρ_sin²)

    where ρ_c/s = <h_wht | T_c/s_wht> / ||T_quad||, ||T_quad||² = ||T_cos||²
    + ||T_sin||².  Under H₀ (Gaussian noise, σ²_noise = 1), A ~ Rayleigh(σ=1)
    with E[A] = √(π/2) ≈ 1.25 and background p99 ≈ 3.03.
    The calibration uses a pre-merger quiet window so non-Gaussian post-merger
    noise does not inflate the denominator.
    """
    meta = EVENTS[event_name]
    path = fetch_strain(event_name, detector, fs_khz=fs_khz)
    strain, t0_gps, fs = read_strain(path)
    merger_idx = find_merger_sample(event_name, t0_gps, fs)
    rd_start = merger_idx + int(ringdown_start_ms * 1e-3 * fs)

    psd_len = int(8.0 * fs)
    psd_start = max(0, merger_idx - int(2.0 * fs) - psd_len)
    psd_freqs, psd_vals = estimate_psd(strain, fs, psd_start, psd_len)
    bp = bandpass(strain, fs, lo=35.0, hi=min(fs / 2 * 0.45, 1500.0))

    f_qnm = meta['f_qnm_hz']
    tau_qnm = meta['tau_qnm_s']
    n_tmpl = int(template_ms * 1e-3 * fs)

    # Quadrature whitened templates
    t_tmpl = np.arange(n_tmpl) / fs
    env = np.exp(-t_tmpl / tau_qnm)
    tmpl_cos = env * np.cos(2.0 * np.pi * f_qnm * t_tmpl)
    tmpl_sin = env * np.sin(2.0 * np.pi * f_qnm * t_tmpl)
    tc_wht = whiten(tmpl_cos, fs, psd_freqs, psd_vals)
    ts_wht = whiten(tmpl_sin, fs, psd_freqs, psd_vals)
    norm = math.sqrt(float(np.dot(tc_wht, tc_wht) + np.dot(ts_wht, ts_wht)))
    if norm <= 0:
        return float('nan'), float('nan')

    def quadrature_amplitude(start_idx):
        seg = bp[start_idx: start_idx + n_tmpl]
        if len(seg) < n_tmpl:
            return float('nan')
        sw = whiten(seg, fs, psd_freqs, psd_vals)
        rc = float(np.dot(sw, tc_wht)) / norm
        rs = float(np.dot(sw, ts_wht)) / norm
        return math.sqrt(rc * rc + rs * rs)

    # Signal at ringdown start
    amp_signal = quadrature_amplitude(rd_start)

    # Pre-merger calibration: background from [-12 s, -2 s] relative to merger
    # (quiet noise, uncontaminated by the signal).
    bg_vals = []
    rng = np.random.default_rng(seed=42)
    for _ in range(300):
        idx = merger_idx - int(rng.uniform(2.0, 12.0) * fs)
        if idx >= 0:
            v = quadrature_amplitude(idx)
            if np.isfinite(v):
                bg_vals.append(v)
    # For Rayleigh-distributed amplitude: E[A] = sqrt(pi/2) * sigma_noise.
    # Infer sigma_noise from background mean: sigma_noise = E_bg / sqrt(pi/2).
    bg_mean = float(np.mean(bg_vals)) if bg_vals else math.sqrt(math.pi / 2.0)
    sigma_noise = bg_mean / math.sqrt(math.pi / 2.0)

    # QNM amplitude in units of sigma_noise → this is the "SNR"
    snr_cal = amp_signal / sigma_noise if sigma_noise > 0 else amp_signal

    if verbose:
        print(f'  {event_name}/{detector}: ρ_QNM={snr_cal:.3f}  '
              f'(amp={amp_signal:.4f}, bg_mean={bg_mean:.4f}, σ_noise={sigma_noise:.4f})')

    return snr_cal, sigma_noise


# ── Upper-limit computation ─────────────────────────────────────────────────

def incoherent_xi_upper_limit(rho_qnm_list, alpha=0.05):
    """95 % CL upper limit on ξ from incoherent Fisher combination.

    Under H₀ + ξ-signal with per-event echo SNR ρ_echo_i = ξ·ρ_QNM_i,
    the expected Fisher χ² = Σ_i [-2 ln p(ρ_echo_i)] where p is the
    two-sided normal p-value.

    The 95 % CL upper limit solves:
        expected_chi2(ξ_UL) = chi2.ppf(1-alpha, dof=2N)
    """
    N = len(rho_qnm_list)
    dof = 2 * N
    chi2_thresh = chi2.ppf(1.0 - alpha, dof)

    def expected_chi2(xi):
        total = 0.0
        for rho_q in rho_qnm_list:
            rho_echo = xi * abs(rho_q)
            p_val = 2.0 * (1.0 - norm.cdf(rho_echo))
            p_val = max(p_val, 1e-300)
            total += -2.0 * math.log(p_val)
        return total

    # Binary search for ξ_UL
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if expected_chi2(mid) < chi2_thresh:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def coherent_xi_sensitivity(rho_qnm_list, sigma=3.0):
    """ξ detectable at sigma·σ with coherent (phase-preserving) stacking.

    Combined SNR = √(Σ ρ_echo_i²) = ξ · √(Σ ρ_QNM_i²)
    Set equal to sigma → ξ_sens = sigma / √(Σ ρ_QNM_i²)
    """
    rho_sq_sum = sum(r**2 for r in rho_qnm_list if np.isfinite(r))
    if rho_sq_sum <= 0:
        return float('nan')
    return sigma / math.sqrt(rho_sq_sum)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print(f'\nξ = {XI_VAL:.4f}  σ_conv = {SIGMA_CONV_VAL:.4f}')
    print(f'Phase I threshold (background p99): |ρ| = {BACKGROUND_P99}')
    print()

    rows = []
    for event_name, detector, fs_khz in CATALOG:
        print(f'[{event_name}/{detector}]')
        rho_q, bg_std = compute_qnm_snr(event_name, detector, fs_khz=fs_khz)
        rho_echo_pred = XI_VAL * abs(rho_q) if np.isfinite(rho_q) else float('nan')
        detectable = rho_echo_pred > BACKGROUND_P99 if np.isfinite(rho_echo_pred) else False
        rows.append({
            'event': event_name,
            'det': detector,
            'rho_qnm': rho_q,
            'rho_echo_pred': rho_echo_pred,
            'detectable': detectable,
        })

    # Summary table
    print(f'\n{"="*70}')
    print(f'{"Event/Det":<18} {"ρ_QNM":>8} {"ρ_echo_pred":>12} '
          f'{"threshold":>10} {"status":>12}')
    print(f'{"-"*70}')
    for r in rows:
        stat = 'DETECTABLE' if r['detectable'] else 'below thresh'
        print(f'{r["event"]}/{r["det"]:<10} {r["rho_qnm"]:>8.3f} '
              f'{r["rho_echo_pred"]:>12.3f} {BACKGROUND_P99:>10.1f} {stat:>12}')

    # Combined statistics
    rho_qnm_list = [abs(r['rho_qnm']) for r in rows if np.isfinite(r['rho_qnm'])]
    xi_ul_incoherent = incoherent_xi_upper_limit(rho_qnm_list)
    xi_sens_coherent = coherent_xi_sensitivity(rho_qnm_list, sigma=3.0)
    combined_echo_snr_coherent = XI_VAL * math.sqrt(sum(r**2 for r in rho_qnm_list))

    print(f'\n{"="*70}')
    print(f'Phase II — Amplitude falsifiability summary')
    print(f'  ξ_predicted (σ_conv)         = {XI_VAL:.4f}')
    print(f'  ξ_UL incoherent (95% CL)     = {xi_ul_incoherent:.4f}')
    print(f'  ξ_sens coherent (3σ thresh)  = {xi_sens_coherent:.4f}')
    print(f'  Combined coherent echo SNR   = {combined_echo_snr_coherent:.2f}σ')
    print(f'  Model status:')
    if XI_VAL < xi_ul_incoherent:
        print(f'    ξ_pred < ξ_UL_incoherent — CONSISTENT with Phase I.4 null')
    else:
        print(f'    ξ_pred > ξ_UL_incoherent — TENSION with Phase I.4 null')
    if XI_VAL > xi_sens_coherent:
        print(f'    ξ_pred > ξ_sens_coherent — TESTABLE with coherent 11-event stack')
    else:
        print(f'    ξ_pred < ξ_sens_coherent — coherent stack still insufficient')
    print(f'{"="*70}')

    # ── Figure ──────────────────────────────────────────────────────────────
    plt.rcParams.update({
        'figure.facecolor': '#111', 'axes.facecolor': '#181818',
        'savefig.facecolor': '#111', 'text.color': '#eee',
        'axes.labelcolor': '#ccc', 'font.family': 'DejaVu Sans',
    })
    fig, (ax_bars, ax_xi) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        'Phase II — ξ-shell Echo Amplitude Falsifiability\n'
        r'$\rho_\mathrm{echo}^\mathrm{pred} = \xi \cdot \rho_\mathrm{QNM}$'
        f'   (ξ = {XI_VAL:.4f},  σ_conv = {SIGMA_CONV_VAL:.3f})',
        fontsize=12, color='#eee', y=0.98)

    # Left: per-event bar chart
    labels = [f'{r["event"][:7]}\n/{r["det"]}' for r in rows]
    rho_preds = [r['rho_echo_pred'] for r in rows]
    rho_qnms  = [abs(r['rho_qnm']) for r in rows]
    colors = ['#f88' if r['detectable'] else '#6af' for r in rows]

    x = np.arange(len(rows))
    bars = ax_bars.bar(x, rho_preds, color=colors, alpha=0.85, width=0.55,
                       label=r'$\rho_\mathrm{echo}^\mathrm{pred} = \xi \cdot \rho_\mathrm{QNM}$')
    # Overlay ρ_QNM as step outline
    ax_bars.step(x - 0.275, rho_qnms, where='post',
                 color='#fa0', lw=1.0, alpha=0.7, label=r'$\rho_\mathrm{QNM}$ (in-window)')
    ax_bars.axhline(BACKGROUND_P99, color='#f55', lw=1.2, ls='--',
                    label=f'Phase I threshold ({BACKGROUND_P99}σ)')
    ax_bars.axhline(XI_VAL * np.mean(rho_qnms), color='#adf', lw=0.8, ls=':',
                    alpha=0.6, label=f'mean predicted echo ({XI_VAL:.3f} × ρ̄_QNM)')
    ax_bars.set_xticks(x)
    ax_bars.set_xticklabels(labels, fontsize=7, color='#ccc')
    ax_bars.set_ylabel('Matched-filter SNR', fontsize=9)
    ax_bars.set_title('Per-detector-event echo SNR', fontsize=10, color='#ddd')
    ax_bars.legend(fontsize=7.5, framealpha=0.15, labelcolor='#ddd')
    ax_bars.tick_params(colors='#bbb')
    for sp in ax_bars.spines.values():
        sp.set_color('#444')

    # Right: ξ sensitivity curve
    n_events_range = np.arange(1, 51)
    rho_qnm_mean = float(np.nanmean(rho_qnm_list))

    def xi_incoherent_ul(n, alpha=0.05):
        rho_list = [rho_qnm_mean] * int(n)
        return incoherent_xi_upper_limit(rho_list, alpha=alpha)

    def xi_coherent_sens(n, sigma=3.0):
        return sigma / math.sqrt(n * rho_qnm_mean**2)

    xi_ul_curve = [xi_incoherent_ul(n) for n in n_events_range]
    xi_coh_curve = [xi_coherent_sens(n) for n in n_events_range]

    ax_xi.plot(n_events_range, xi_ul_curve, color='#6af', lw=1.5,
               label='Incoherent 95% CL upper limit on ξ')
    ax_xi.plot(n_events_range, xi_coh_curve, color='#fa0', lw=1.5, ls='--',
               label='Coherent 3σ sensitivity (ξ_detectable)')
    ax_xi.axhline(XI_VAL, color='#f88', lw=1.3, ls='-.',
                  label=f'ξ_predicted = {XI_VAL:.4f}')
    ax_xi.axvline(len(rho_qnm_list), color='#888', lw=0.8, ls=':',
                  label=f'Current N = {len(rho_qnm_list)}')
    ax_xi.set_xlabel('Number of detector-events', fontsize=9)
    ax_xi.set_ylabel('ξ', fontsize=9)
    ax_xi.set_title('Sensitivity vs catalog size\n'
                    '(assuming average ρ_QNM per event)', fontsize=9.5, color='#ddd')
    ax_xi.set_ylim(0, 0.5)
    ax_xi.legend(fontsize=8, framealpha=0.15, labelcolor='#ddd')
    ax_xi.tick_params(colors='#bbb')
    for sp in ax_xi.spines.values():
        sp.set_color('#444')

    # Annotation boxes
    ax_xi.annotate(
        f'ξ_UL(incoherent, N={len(rho_qnm_list)}) = {xi_ul_incoherent:.3f}',
        xy=(len(rho_qnm_list), xi_ul_incoherent),
        xytext=(len(rho_qnm_list) + 3, xi_ul_incoherent + 0.04),
        fontsize=7.5, color='#6af',
        arrowprops=dict(arrowstyle='->', color='#6af', lw=0.8))
    ax_xi.annotate(
        f'ξ_sens(coherent, N={len(rho_qnm_list)}) = {xi_sens_coherent:.3f}',
        xy=(len(rho_qnm_list), xi_sens_coherent),
        xytext=(len(rho_qnm_list) + 3, xi_sens_coherent - 0.05),
        fontsize=7.5, color='#fa0',
        arrowprops=dict(arrowstyle='->', color='#fa0', lw=0.8))

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.normpath(
        os.path.join(os.path.dirname(__file__),
                     '..', 'misc', 'bh_phase_ii_amplitude_bound.png'))
    fig.savefig(out, dpi=125, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f'\nwrote {out} ({os.path.getsize(out) / 1024:.1f} KB)')

    return {
        'rows': rows,
        'xi_ul_incoherent': xi_ul_incoherent,
        'xi_sens_coherent': xi_sens_coherent,
        'combined_coherent_snr': combined_echo_snr_coherent,
    }


if __name__ == '__main__':
    main()
