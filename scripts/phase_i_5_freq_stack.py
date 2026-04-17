"""Phase I.5 — frequency-domain sideband stacking across 5-event catalog.

If echoes arrive at spacing Δt_1, the whitened post-merger power spectrum
near f_QNM shows interference sidebands at offsets ±k/Δt_1 from f_QNM
(k=1,2,3,...).  Shifting the frequency axis by −f_QNM and rescaling by
Δt_1 maps all events' sidebands to the same integer positions ±1, ±2, ...
regardless of event mass or spin.  Averaging across N=11 detector-events
improves S/N by √N ≈ 3.3×.

The test is model-agnostic in f_QNM and τ_QNM for the stack itself —
it only uses the Δt_1 prediction (fixed by σ_conv = −ln(ξ) from M_rem).
f_QNM is used only to centre the extraction band.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy import signal as scipy_signal

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..')))
from sigma_ground.field.interface.ligo_echo_search import (
    EVENTS, fetch_strain, read_strain, find_merger_sample,
    estimate_psd, bandpass, whiten, echo_delay_n,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


DETECTOR_PLAN = [
    ('GW151226', 'H1', 16),
    ('GW151226', 'L1', 16),
    ('GW150914', 'H1', 16),  # 16 kHz — unlocks k=1–7 sidebands
    ('GW150914', 'L1', 16),
    ('GW170814', 'H1', 4),
    ('GW170814', 'L1', 4),
    ('GW170814', 'V1', 4),
    ('GW170104', 'H1', 4),
    ('GW170104', 'L1', 4),
    ('GW190521', 'H1', 4),
    ('GW190521', 'L1', 4),
]

# Window length for the FFT.  A short window (e.g. 10×Δt_1 ≈ 30 ms) gives
# only ~26 in-band frequency bins at 4 kHz — far too few for a stable median
# normalisation.  Using 1 s gives ~865 bins (35–900 Hz at 4 kHz) so that the
# per-event normalisation and on/off-comb comparison are well-averaged.  The
# QNM decays within the first ≪1% of the window, leaving a narrow Lorentzian
# spike at f_r = 0 that is well-separated from the k ≥ 1 sideband positions.
WINDOW_FACTOR = 10
MIN_WINDOW_MS = 1000.0

# Rescaled frequency grid: (f − f_QNM) × Δt_1
# Sidebands from echoes live at ±1, ±2, ±3, ...
# GW150914 at 16 kHz reaches f_r ≈ 7.8, so extend to 8.
F_RESCALED_MAX = 8.0
F_RESCALED_NPTS = 4000


def get_sideband_spectrum(event_name, detector, fs_khz=4,
                           ringdown_start_ms=3.0, verbose=True):
    """Compute the post-merger power spectrum rescaled to sideband coordinates.

    Returns
    -------
    f_rescaled : ndarray
        (f − f_QNM) × Δt_1 — sidebands from echoes appear at ±1, ±2, …
    power_norm : ndarray
        Normalised power (median = 1) on f_rescaled grid.
    delta_t1_s : float
    f_qnm_hz : float
    """
    meta = EVENTS[event_name]
    path = fetch_strain(event_name, detector, fs_khz=fs_khz)
    strain, t0_gps, fs = read_strain(path)
    merger_idx = find_merger_sample(event_name, t0_gps, fs)
    rd_start = merger_idx + int(ringdown_start_ms * 1e-3 * fs)

    psd_len = int(8.0 * fs)
    psd_start = max(0, merger_idx - int(2.0 * fs) - psd_len)
    psd_freqs, psd_vals = estimate_psd(strain, fs, psd_start, psd_len)

    bp = bandpass(strain, fs, lo=35.0, hi=fs / 2 * 0.45)

    delta_t1 = echo_delay_n(meta['M_rem_msun'], 1)
    f_qnm = meta['f_qnm_hz']

    window_s = max(MIN_WINDOW_MS * 1e-3,
                   WINDOW_FACTOR * delta_t1)
    n_win = min(int(window_s * fs), len(bp) - rd_start - 10)
    if n_win < 32:
        return None, None, delta_t1, f_qnm

    seg = bp[rd_start: rd_start + n_win]
    wht = whiten(seg, fs, psd_freqs, psd_vals)

    # Hann-windowed FFT
    win = scipy_signal.windows.hann(n_win)
    X = np.fft.rfft(wht * win)
    freqs = np.fft.rfftfreq(n_win, d=1.0 / fs)
    power = np.abs(X) ** 2

    # Restrict to in-band frequencies only.  The bandpass attenuates out-of-band
    # bins but does not zero them; whitening then amplifies that residual to
    # ~flat noise.  np.interp would treat those out-of-band bins as valid data,
    # making all events appear to cover the full f_rescaled grid.
    bp_lo = 35.0
    bp_hi = fs / 2.0 * 0.45
    in_band = (freqs >= bp_lo) & (freqs <= bp_hi)
    freqs = freqs[in_band]
    power = power[in_band]

    # Rescale: f_rescaled = (f − f_QNM) × Δt_1
    f_rescaled = (freqs - f_qnm) * delta_t1

    if verbose:
        df_phys = 1.0 / (n_win / fs)
        df_resc = df_phys * delta_t1
        print(f'  {event_name}/{detector}: Δt_1={delta_t1*1000:.3f} ms, '
              f'window={n_win/fs*1000:.1f} ms, '
              f'Δf_rescaled={df_resc:.4f}')

    return f_rescaled, power, delta_t1, f_qnm


def stack_spectra(f_rescaled_list, power_list, f_grid):
    """Average normalised power spectra onto a common rescaled grid.

    Normalises each spectrum by the median over positive f_r (above the QNM
    notch at |f_r| < 0.5), excluding windows around each sideband position.
    This handles events where the upper bandpass falls short of f_r = 1.5 —
    negative f_r corresponds to frequencies below f_QNM and is never used for
    sideband detection.  Events with fewer than 5 usable normalisation points
    are skipped so they cannot corrupt the stack with unscaled power.
    """
    stack = np.zeros(len(f_grid))
    n_valid = np.zeros(len(f_grid))
    for f_resc, pwr in zip(f_rescaled_list, power_list):
        pwr_interp = np.interp(f_grid, f_resc, pwr,
                               left=np.nan, right=np.nan)
        valid = np.isfinite(pwr_interp)
        # Normalise by median of valid points above the QNM notch,
        # excluding windows around positive sideband positions k=1,2,3.
        norm_mask = valid & (f_grid > 0.5)
        for k in (1, 2, 3):
            norm_mask &= (np.abs(f_grid - k) > 0.25)
        if np.sum(norm_mask) < 5:
            # Fallback: widen to include all valid points outside QNM notch
            norm_mask = valid & (np.abs(f_grid) > 0.3)
        if np.sum(norm_mask) < 5:
            continue  # cannot normalise — skip this detector-event
        pwr_interp[valid] /= np.nanmedian(pwr_interp[norm_mask])
        stack[valid] += pwr_interp[valid]
        n_valid[valid] += 1
    with np.errstate(invalid='ignore'):
        stack = np.where(n_valid > 0, stack / n_valid, np.nan)
    return stack, n_valid


def comb_excess(stack, f_grid, n_valid=None, half_width=0.25, sideband_k=(1, 2, 3)):
    """Ratio of mean power at positive sideband positions to off-comb baseline.

    Only positive sidebands are tested: echo interference sidebands appear at
    f = f_QNM + k/Δt_1, i.e. f_rescaled = +k.  The corresponding negative
    positions (f_rescaled = -k) require f < f_QNM - k/Δt_1 < 0, which is
    always below DC and therefore inaccessible in strain data.
    """
    on_comb = []
    n_contributors = []
    for k in sideband_k:
        mask = np.abs(f_grid - k) < half_width
        finite_mask = mask & np.isfinite(stack)
        vals = stack[finite_mask]
        if len(vals):
            # Use median: the comb position covers hundreds of bins so the
            # median is a reliable estimator and is robust to spectral lines
            # or non-Gaussian glitches present in only one or two events.
            on_comb.append(float(np.median(vals)))
            if n_valid is not None:
                n_contributors.append(int(np.max(n_valid[finite_mask])))

    # Off-comb baseline: all finite points above the QNM notch,
    # excluding windows around each positive sideband position.
    off_mask = np.isfinite(stack) & (f_grid > 0.5)
    for k in sideband_k:
        off_mask &= (np.abs(f_grid - k) > half_width)
    off_vals = stack[off_mask]
    off_median = float(np.median(off_vals)) if len(off_vals) else 1.0

    if not on_comb:
        return float('nan'), float('nan'), off_median, []
    on_median = float(np.median(on_comb))
    ratio = (on_median / off_median) if off_median > 0 else float('nan')
    return ratio, on_median, off_median, n_contributors


def main():
    f_grid = np.linspace(-F_RESCALED_MAX, F_RESCALED_MAX, F_RESCALED_NPTS)

    f_resc_list, pwr_list = [], []
    per_det = []  # for individual traces

    for event_name, detector, fs_khz in DETECTOR_PLAN:
        print(f'\n[{event_name}/{detector}] loading...')
        f_resc, pwr, dt1, fqnm = get_sideband_spectrum(
            event_name, detector, fs_khz=fs_khz)
        if f_resc is None:
            print('  SKIP (window too short)')
            continue
        f_resc_list.append(f_resc)
        pwr_list.append(pwr)
        per_det.append((event_name, detector, f_resc, pwr, dt1, fqnm))

    stack, n_valid = stack_spectra(f_resc_list, pwr_list, f_grid)
    ratio, on_mean, off_mean, n_contribs = comb_excess(
        stack, f_grid, n_valid=n_valid, sideband_k=tuple(range(1, 8)))

    print(f'\n{"="*60}')
    print(f'Phase I.5 — sideband comb excess (positive k=1..7)')
    print(f'  on-comb median power: {on_mean:.4f}')
    print(f'  off-comb median power: {off_mean:.4f}')
    print(f'  ratio (null → 1.0, signal → >1): {ratio:.4f}')
    print(f'  contributors per sideband k=1..7: {n_contribs}')
    print(f'{"="*60}')

    # ── Figure ──────────────────────────────────────────────────────────────
    plt.rcParams.update({
        'figure.facecolor': '#111', 'axes.facecolor': '#181818',
        'savefig.facecolor': '#111', 'text.color': '#eee',
        'axes.labelcolor': '#ccc', 'font.family': 'DejaVu Sans',
    })
    fig, axes = plt.subplots(3, 4, figsize=(18, 11))
    fig.suptitle(
        'Phase I.5 — frequency-domain sideband stacking, 5-event catalog\n'
        r'$f_\mathrm{rescaled} = (f - f_\mathrm{QNM}) \cdot \Delta t_1$'
        ' — echo sidebands expected at ±1, ±2, ±3 (dashed lines)',
        fontsize=12, color='#eee', y=0.98)

    # Individual event traces (first 11 panels)
    axes_flat = axes.flatten()
    DET_COL = {'H1': '#6cf', 'L1': '#fc8', 'V1': '#b8f'}

    for idx, (ev, det, f_resc, pwr, dt1, fqnm) in enumerate(per_det):
        ax = axes_flat[idx]
        # Normalise for display
        pwr_interp = np.interp(f_grid, f_resc, pwr,
                               left=np.nan, right=np.nan)
        bl = ((np.abs(f_grid) > 1.5) & (np.abs(f_grid) < F_RESCALED_MAX)
              & np.isfinite(pwr_interp))
        if np.sum(bl) > 5:
            pwr_interp = pwr_interp / np.nanmedian(pwr_interp[bl])

        ax.plot(f_grid, pwr_interp, color=DET_COL.get(det, '#aaa'),
                lw=0.7, alpha=0.85)
        ax.axhline(1.0, color='#666', lw=0.6, ls='--', alpha=0.6)
        for k in range(1, 8):
            ax.axvline(k, color='#f6b26b', lw=0.7, ls=':', alpha=0.55)
        ax.set_xlim(-F_RESCALED_MAX, F_RESCALED_MAX)
        ax.set_ylim(0, None)
        ax.set_title(f'{ev} / {det}', fontsize=8.5,
                     color=DET_COL.get(det, '#aaa'), pad=2)
        ax.set_xlabel(r'$(f-f_\mathrm{QNM})\cdot\Delta t_1$', fontsize=7.5)
        ax.set_ylabel('norm. power', fontsize=7.5)
        ax.tick_params(labelsize=7, colors='#bbb')
        for spine in ax.spines.values():
            spine.set_color('#444')

    # Last panel: stacked spectrum
    ax_stack = axes_flat[11]
    ax_stack.plot(f_grid, stack, color='#fff', lw=1.2, alpha=0.9,
                  label=f'stack (N={len(per_det)})')
    ax_stack.axhline(1.0, color='#666', lw=0.8, ls='--', alpha=0.7,
                     label='noise baseline')
    for k in range(1, 8):
        ax_stack.axvline(k, color='#f6b26b', lw=1.0, ls='--', alpha=0.7,
                         label='echo comb (k=1..7)' if k == 1 else None)
    ax_stack.set_xlim(-F_RESCALED_MAX, F_RESCALED_MAX)
    ax_stack.set_ylim(0, None)
    ax_stack.set_title(
        f'Stacked (N={len(per_det)} det-events)\n'
        f'comb ratio = {ratio:.3f}  (null → 1.0)',
        fontsize=9, color='#fff', pad=3)
    ax_stack.set_xlabel(r'$(f-f_\mathrm{QNM})\cdot\Delta t_1$', fontsize=8)
    ax_stack.set_ylabel('norm. power', fontsize=8)
    ax_stack.tick_params(labelsize=7.5, colors='#bbb')
    ax_stack.legend(fontsize=7, framealpha=0.0, labelcolor='#ccc')
    for spine in ax_stack.spines.values():
        spine.set_color('#555')

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.normpath(
        os.path.join(os.path.dirname(__file__),
                     '..', 'misc', 'bh_phase_i_5_freq_stack.png'))
    fig.savefig(out, dpi=125, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f'wrote {out} ({os.path.getsize(out) / 1024:.1f} KB)')
    return ratio


if __name__ == '__main__':
    main()
