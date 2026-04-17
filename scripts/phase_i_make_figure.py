"""
Phase I — matplotlib visualisation of ξ-shell echo search results.

2x2 grid: rows = events (GW151226, GW150914), cols = detectors (H1, L1).
Each panel shows whitened residual + lollipop bars at Δt_n with calibrated
SNR heights; background |SNR| histogram inset on the side.

Written to misc/bh_phase_i_echo_search.png.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sigma_ground.field.interface.ligo_echo_search import (
    EVENTS, fetch_strain, read_strain, estimate_psd, bandpass, whiten,
    fit_qnm, qnm_damped_sinusoid, make_echo_template,
    correlate_at_delay, background_distribution,
    echo_delay_n, find_merger_sample,
)


PLAN = [
    ('GW151226', 'H1', 16),
    ('GW151226', 'L1', 16),
    ('GW150914', 'H1',  4),
    ('GW150914', 'L1',  4),
]


def analyse(event_name, detector, fs_khz,
            ringdown_start_ms=3.0, qnm_fit_duration_ms=20.0,
            template_length_ms=20.0, n_echoes=5, n_bg=2000):
    meta = EVENTS[event_name]
    path = fetch_strain(event_name, detector, fs_khz=fs_khz)
    strain, t0, fs = read_strain(path)
    merger_idx = find_merger_sample(event_name, t0, fs)
    ringdown_start_idx = merger_idx + int(ringdown_start_ms * 1e-3 * fs)

    psd_seg_len = int(8.0 * fs)
    psd_start = max(0, merger_idx - int(2.0 * fs) - psd_seg_len)
    psd_freqs, psd_vals = estimate_psd(strain, fs, psd_start, psd_seg_len)

    bp = bandpass(strain, fs, lo=35.0, hi=min(fs / 2 * 0.45, 1500.0))

    popt, qnm_model, qnm_start, qnm_end = fit_qnm(
        bp, fs, ringdown_start_idx, duration_ms=qnm_fit_duration_ms,
        f_guess=meta['f_qnm_hz'], tau_guess=meta['tau_qnm_s'],
    )
    A_fit, f_qnm, tau_qnm, phi_fit = popt

    residual = bp.copy()
    full_len = int(min(10.0 * tau_qnm * fs, len(bp) - qnm_start))
    t_full = np.arange(full_len) / fs
    qnm_full = qnm_damped_sinusoid(t_full, A_fit, f_qnm, tau_qnm, phi_fit)
    residual[qnm_start:qnm_start + full_len] -= qnm_full

    whitened = whiten(residual, fs, psd_freqs, psd_vals)

    template = make_echo_template(fs, f_qnm, tau_qnm,
                                   length_seconds=template_length_ms * 1e-3)

    delta_t1 = echo_delay_n(meta['M_rem_msun'], 1)
    echo_snrs_raw = []
    delays = []
    for n in range(1, n_echoes + 1):
        dt = echo_delay_n(meta['M_rem_msun'], n)
        snr = correlate_at_delay(whitened, template, fs,
                                  ringdown_start_idx, dt)
        echo_snrs_raw.append(snr)
        delays.append(dt)

    bg = background_distribution(whitened, template, fs,
                                  ringdown_start_idx, delta_t1,
                                  n_samples=n_bg)
    bg_std_raw = float(np.std(bg))
    if bg_std_raw > 0:
        bg = bg / bg_std_raw
        echo_snrs = [s / bg_std_raw for s in echo_snrs_raw]
    else:
        echo_snrs = echo_snrs_raw

    plot_start = merger_idx - int(1.0e-3 * fs)
    plot_end = merger_idx + int(26e-3 * fs)
    plot_t_ms = (np.arange(plot_end - plot_start) - (merger_idx - plot_start)) / fs * 1000.0
    plot_wh = whitened[plot_start:plot_end]
    if bg_std_raw > 0:
        plot_wh = plot_wh / bg_std_raw  # Put the trace in the same calibrated units.

    return {
        'event': event_name,
        'detector': detector,
        'fs': fs,
        'plot_t_ms': plot_t_ms,
        'plot_wh': plot_wh,
        'delays_ms': np.array([d * 1000.0 for d in delays]),
        'echo_snrs': np.array(echo_snrs),
        'bg': bg,
        'bg_p99': float(np.percentile(np.abs(bg), 99)),
        'qnm_f': f_qnm,
        'qnm_tau_ms': tau_qnm * 1000.0,
        'qnm_phi': phi_fit,
        'delta_t1_ms': delta_t1 * 1000.0,
    }


def main():
    results = [analyse(ev, det, fs_khz) for ev, det, fs_khz in PLAN]

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(15, 10), constrained_layout=False)
    fig.patch.set_facecolor('#0b0e12')

    fig.suptitle(
        'Phase I — ξ-shell echo search • GWOSC strain • '
        'matched-filter at Δt$_n$ = 2·r$_s$·n·σ$_{conv}$/c',
        fontsize=14, color='white', y=0.98,
    )
    fig.text(
        0.5, 0.945,
        'sigma-ground RODM-A prediction: linear-in-n echo train at σ$_{conv}$ = −ln ξ ≈ 1.844',
        fontsize=10, color='#aaaaaa', ha='center',
    )

    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.18,
                           left=0.06, right=0.98, top=0.90, bottom=0.07)

    event_ylims = {}
    for r in results:
        key = r['event']
        snr_max = max(6.0, np.max(np.abs(r['echo_snrs'])) * 1.15)
        event_ylims[key] = max(event_ylims.get(key, 6.0), snr_max)

    for i, r in enumerate(results):
        row, col = i // 2, i % 2
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor('#121821')

        # Calibrated whitened residual.
        ax.plot(r['plot_t_ms'], r['plot_wh'],
                color='#5bb6e6', lw=0.7, alpha=0.55,
                label='whitened residual (σ units)', zorder=2)

        # Horizontal p99 band.
        ax.axhspan(-r['bg_p99'], r['bg_p99'], color='#888888',
                   alpha=0.08, zorder=1, label=f'bg |SNR| ≤ p99 ({r["bg_p99"]:.2f}σ)')
        ax.axhline(0, color='#666666', lw=0.5, alpha=0.4, zorder=1)
        ax.axhline(r['bg_p99'], color='#888888', ls=':', lw=0.7, alpha=0.5)
        ax.axhline(-r['bg_p99'], color='#888888', ls=':', lw=0.7, alpha=0.5)

        # Lollipop bars at each Δt_n.
        for n, (dt_ms, snr) in enumerate(zip(r['delays_ms'], r['echo_snrs']),
                                          start=1):
            if abs(snr) > r['bg_p99'] * 1.3:
                color = '#ff9f43'
                alpha = 0.9
            elif abs(snr) > r['bg_p99']:
                color = '#ffd166'
                alpha = 0.7
            else:
                color = '#7ac2e6'
                alpha = 0.55
            ax.vlines(dt_ms, 0, snr, color=color, lw=3.5,
                      alpha=alpha, zorder=4)
            ax.plot(dt_ms, snr, 'o', color=color, markersize=7,
                    alpha=alpha, zorder=5,
                    markeredgecolor='#0b0e12', markeredgewidth=0.8)
            label_y = snr + np.sign(snr) * 0.6 + (0.6 if snr == 0 else 0)
            ax.annotate(
                f'n={n}\n{snr:+.2f}σ',
                xy=(dt_ms, snr), xytext=(dt_ms, label_y),
                ha='center', va='bottom' if snr >= 0 else 'top',
                fontsize=8, color=color,
                bbox=dict(boxstyle='round,pad=0.15',
                          fc='#0b0e12', ec='none', alpha=0.6),
            )

        title = (
            f'{r["event"]} / {r["detector"]}  —  '
            f'f$_s$={int(r["fs"]/1000)} kHz  •  '
            f'f$_{{QNM}}$={r["qnm_f"]:.0f} Hz  •  '
            f'τ$_{{QNM}}$={r["qnm_tau_ms"]:.2f} ms  •  '
            f'φ$_{{fit}}$={r["qnm_phi"]:+.2f} rad  •  '
            f'Δt$_1$={r["delta_t1_ms"]:.3f} ms'
        )
        ax.set_title(title, fontsize=9.5, color='#eeeeee', pad=8, loc='left')
        ax.set_xlabel('time since merger (ms)', color='#bbbbbb', fontsize=9)
        ax.set_ylabel('calibrated SNR (σ)', color='#bbbbbb', fontsize=9)
        ax.set_xlim(-1, 26)
        ax.set_ylim(-event_ylims[r['event']], event_ylims[r['event']])
        ax.grid(alpha=0.12, color='#666666')
        ax.axvline(0, color='#cccccc', lw=0.5, alpha=0.35)
        ax.tick_params(colors='#bbbbbb', which='both')
        for spine in ax.spines.values():
            spine.set_color('#444444')

        # Background histogram inset.
        ax_inset = ax.inset_axes([0.70, 0.66, 0.28, 0.30])
        abs_bg = np.abs(r['bg'])
        ax_inset.hist(abs_bg, bins=40, color='#5bb6e6',
                      alpha=0.55, density=True, edgecolor='none')
        ax_inset.axvline(r['bg_p99'], color='#ff6666', ls=':',
                          lw=0.8, alpha=0.85)
        for snr in r['echo_snrs']:
            ax_inset.axvline(abs(snr), color='#ff9f43',
                              lw=0.8, alpha=0.7)
        ax_inset.set_xlabel('|SNR|', fontsize=7, color='#aaaaaa')
        ax_inset.set_ylabel('density', fontsize=7, color='#aaaaaa')
        ax_inset.set_title(
            f'bg: μ={np.mean(r["bg"]):+.2f}, σ={np.std(r["bg"]):.2f}, '
            f'p99={r["bg_p99"]:.2f}',
            fontsize=7, color='#aaaaaa', pad=2,
        )
        ax_inset.tick_params(labelsize=6, colors='#999999')
        ax_inset.set_facecolor('#0f1520')
        for spine in ax_inset.spines.values():
            spine.set_color('#333333')

    fig.text(
        0.5, 0.018,
        'orange = |SNR| above background p99 (nominally significant). '
        'shading = ±p99 band (null range). '
        'Background = 2000 off-source delays excluding ±1.5 ms around predicted Δt$_n$.',
        fontsize=8, color='#888888', ha='center',
    )

    out_path = os.path.join('misc', 'bh_phase_i_echo_search.png')
    fig.savefig(out_path, dpi=140, facecolor='#0b0e12',
                bbox_inches='tight')
    print(f'wrote {out_path}')
    print(f'  {os.path.getsize(out_path) / 1024:.1f} KB')


if __name__ == '__main__':
    main()
