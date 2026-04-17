"""
Phase I.2 — matplotlib visualisation of ξ-shell echo search + coherence.

3x2 grid:
  row 1: GW151226 H1, GW151226 L1  (whitened residual + lollipop SNRs)
  row 2: GW150914 H1, GW150914 L1  (same)
  row 3: coherence panel GW151226,   coherence panel GW150914
          - scatter of (ρ_H1[n], ρ_L1[n]) per echo
          - shaded unit-diagonal band (|L1/H1| ∈ [0.5, 2]) showing the
            "astrophysically-plausible magnitude ratio" region
          - sign-concordance & binomial p annotated

Written to misc/bh_phase_i_2_overtone_coherence.png.
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
    echo_delay_n, find_merger_sample, cross_detector_coherence,
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
        plot_wh = plot_wh / bg_std_raw

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


def draw_snr_panel(ax, r, event_ylim):
    ax.set_facecolor('#121821')

    ax.plot(r['plot_t_ms'], r['plot_wh'],
            color='#5bb6e6', lw=0.7, alpha=0.55,
            label='whitened residual (σ units)', zorder=2)

    ax.axhspan(-r['bg_p99'], r['bg_p99'], color='#888888',
               alpha=0.08, zorder=1)
    ax.axhline(0, color='#666666', lw=0.5, alpha=0.4, zorder=1)
    ax.axhline(r['bg_p99'], color='#888888', ls=':', lw=0.7, alpha=0.5)
    ax.axhline(-r['bg_p99'], color='#888888', ls=':', lw=0.7, alpha=0.5)

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
        # Cap the visible lollipop at the axis limit for legibility
        visible_y = max(-event_ylim * 0.95, min(event_ylim * 0.95, snr))
        ax.vlines(dt_ms, 0, visible_y, color=color, lw=3.5,
                  alpha=alpha, zorder=4)
        ax.plot(dt_ms, visible_y, 'o', color=color, markersize=7,
                alpha=alpha, zorder=5,
                markeredgecolor='#0b0e12', markeredgewidth=0.8)
        if abs(snr) > event_ylim * 0.95:
            label = f'n={n}\n{snr:+.1f}σ\n↕'
        else:
            label = f'n={n}\n{snr:+.2f}σ'
        label_y = visible_y + np.sign(visible_y) * 0.6 + (0.6 if visible_y == 0 else 0)
        ax.annotate(
            label,
            xy=(dt_ms, visible_y), xytext=(dt_ms, label_y),
            ha='center', va='bottom' if visible_y >= 0 else 'top',
            fontsize=8, color=color,
            bbox=dict(boxstyle='round,pad=0.15',
                      fc='#0b0e12', ec='none', alpha=0.6),
        )

    title = (
        f'{r["event"]} / {r["detector"]}  —  '
        f'f$_s$={int(r["fs"]/1000)} kHz  •  '
        f'f$_{{QNM}}$={r["qnm_f"]:.0f} Hz  •  '
        f'τ$_{{QNM}}$={r["qnm_tau_ms"]:.2f} ms  •  '
        f'Δt$_1$={r["delta_t1_ms"]:.3f} ms'
    )
    ax.set_title(title, fontsize=9.5, color='#eeeeee', pad=8, loc='left')
    ax.set_xlabel('time since merger (ms)', color='#bbbbbb', fontsize=9)
    ax.set_ylabel('calibrated SNR (σ)', color='#bbbbbb', fontsize=9)
    ax.set_xlim(-1, 26)
    ax.set_ylim(-event_ylim, event_ylim)
    ax.grid(alpha=0.12, color='#666666')
    ax.axvline(0, color='#cccccc', lw=0.5, alpha=0.35)
    ax.tick_params(colors='#bbbbbb', which='both')
    for spine in ax.spines.values():
        spine.set_color('#444444')


def draw_coherence_panel(ax, event_name, r_h1, r_l1, coh):
    ax.set_facecolor('#121821')
    h = r_h1['echo_snrs']
    l = r_l1['echo_snrs']

    # Plausibility band: |L1/H1| ∈ [1/3, 3] — a rough envelope for
    # antenna-pattern amplitude ratio across detector orientations.
    xs = np.linspace(-40, 40, 2)
    ax.fill_between(xs, xs / 3.0, 3.0 * xs,
                     where=(xs >= 0), color='#3b6e9b', alpha=0.12,
                     label='|L1/H1| ∈ [1/3, 3] (astrophys-plausible)')
    ax.fill_between(xs, 3.0 * xs, xs / 3.0,
                     where=(xs < 0), color='#3b6e9b', alpha=0.12)
    ax.plot(xs, xs, color='#3b6e9b', lw=0.6, ls='--', alpha=0.5,
             label='|L1| = |H1| (ideal)')
    ax.plot(xs, -xs, color='#9b3b6e', lw=0.5, ls=':', alpha=0.4,
             label='antenna-flipped')

    ax.axhline(0, color='#666666', lw=0.5, alpha=0.4)
    ax.axvline(0, color='#666666', lw=0.5, alpha=0.4)

    for n, (hx, ly) in enumerate(zip(h, l), start=1):
        match = 'yes' if np.sign(hx) * np.sign(ly) > 0 else 'no'
        color = '#6dd47e' if match == 'yes' else '#ff6b6b'
        ax.plot(hx, ly, 'o', color=color, markersize=9,
                 markeredgecolor='#0b0e12', markeredgewidth=1.2, zorder=5)
        ax.annotate(f'n={n}', xy=(hx, ly), xytext=(hx + 0.4, ly + 0.8),
                     fontsize=9, color=color, zorder=6,
                     bbox=dict(boxstyle='round,pad=0.15',
                                fc='#0b0e12', ec='none', alpha=0.7))

    # Symmetric limits driven by the bigger-magnitude detector.
    lim = max(4, float(np.max(np.abs(l))) * 1.15,
               float(np.max(np.abs(h))) * 1.15)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel('ρ$_{H1}$ (σ)', color='#bbbbbb', fontsize=9)
    ax.set_ylabel('ρ$_{L1}$ (σ)', color='#bbbbbb', fontsize=9)

    title = f'{event_name}  —  inter-detector coherence'
    ax.set_title(title, fontsize=9.5, color='#eeeeee', pad=8, loc='left')
    ax.grid(alpha=0.12, color='#666666')
    ax.tick_params(colors='#bbbbbb', which='both')
    for spine in ax.spines.values():
        spine.set_color('#444444')

    # Annotation box with coherence stats.
    txt = (
        f'sign matches: {coh["sign_matches"]}/5  (p={coh["sign_p_value_binomial"]:.3f} binomial)\n'
        f'Σ ρ$_{{H1}}$·ρ$_{{L1}}$ = {coh["coherence_statistic"]:+.2f}  (z={coh["z_score"]:+.2f})\n'
        f'null μ, σ = {coh["null_mean"]:+.2f}, {coh["null_std"]:.2f}\n'
        f'|L1|/|H1| range: '
        f'{min(coh["magnitude_ratio_l_over_h"]):.1f}–{max(coh["magnitude_ratio_l_over_h"]):.1f}'
    )
    ax.text(0.02, 0.98, txt, transform=ax.transAxes,
             fontsize=8, color='#dddddd', va='top', ha='left',
             family='monospace',
             bbox=dict(boxstyle='round,pad=0.4',
                        fc='#0b0e12', ec='#444444', alpha=0.88))

    leg = ax.legend(loc='lower right', fontsize=7, framealpha=0.75,
                     facecolor='#0b0e12', edgecolor='#444444',
                     labelcolor='#cccccc')


def main():
    results = {}
    for ev, det, fs_khz in PLAN:
        results[(ev, det)] = analyse(ev, det, fs_khz)

    # Coherence per event.
    coherence = {}
    for ev in ('GW151226', 'GW150914'):
        rh = results[(ev, 'H1')]
        rl = results[(ev, 'L1')]
        coherence[ev] = cross_detector_coherence(
            rh['echo_snrs'].tolist(),
            rl['echo_snrs'].tolist(),
            rh['bg'], rl['bg'],
        )

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 14), constrained_layout=False)
    fig.patch.set_facecolor('#0b0e12')

    fig.suptitle(
        'Phase I.2 — ξ-shell echo search on GWOSC strain + cross-detector coherence',
        fontsize=14, color='white', y=0.985,
    )
    fig.text(
        0.5, 0.963,
        'sigma-ground RODM-A prediction: linear-in-n echo train at σ$_{conv}$ = −ln ξ ≈ 1.844.  '
        'H1 calibrated and quiet; L1 detector-specific pathology.  Sign coherence at chance → no signal.',
        fontsize=10, color='#aaaaaa', ha='center',
    )

    gs = fig.add_gridspec(3, 2, hspace=0.36, wspace=0.22,
                           left=0.06, right=0.98, top=0.925, bottom=0.05,
                           height_ratios=[1.0, 1.0, 1.05])

    # Event-specific y-limits for SNR panels.
    event_ylims = {
        'GW151226': max(4.0, float(np.max(np.abs(
            np.concatenate([results[('GW151226', 'H1')]['echo_snrs'],
                             # Cap L1 at 20 for legibility — L1 has
                             # 30+ artefact spikes that dwarf the scale;
                             # we cap for display, but the numeric
                             # value is shown in the annotation.
                             np.clip(results[('GW151226', 'L1')]['echo_snrs'],
                                      -20, 20)])))) * 1.15),
        'GW150914': max(4.0, float(np.max(np.abs(
            np.concatenate([results[('GW150914', 'H1')]['echo_snrs'],
                             np.clip(results[('GW150914', 'L1')]['echo_snrs'],
                                      -15, 15)])))) * 1.15),
    }

    plan_by_row = [
        [('GW151226', 'H1'), ('GW151226', 'L1')],
        [('GW150914', 'H1'), ('GW150914', 'L1')],
    ]
    for row_idx, row in enumerate(plan_by_row):
        for col_idx, (ev, det) in enumerate(row):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            r = results[(ev, det)]
            draw_snr_panel(ax, r, event_ylims[ev])

    # Row 3: coherence scatter panels.
    for col_idx, ev in enumerate(('GW151226', 'GW150914')):
        ax = fig.add_subplot(gs[2, col_idx])
        rh = results[(ev, 'H1')]
        rl = results[(ev, 'L1')]
        draw_coherence_panel(ax, ev, rh, rl, coherence[ev])

    fig.text(
        0.5, 0.018,
        'orange = |SNR| > bg p99 (nominally significant).  '
        'green = H1·L1 signs agree at that n;  red = disagree.  '
        'blue band = astrophysically-plausible |L1/H1| ratio.',
        fontsize=8, color='#888888', ha='center',
    )

    out_path = os.path.join('misc', 'bh_phase_i_2_overtone_coherence.png')
    fig.savefig(out_path, dpi=140, facecolor='#0b0e12',
                bbox_inches='tight')
    print(f'wrote {out_path}')
    print(f'  {os.path.getsize(out_path) / 1024:.1f} KB')


if __name__ == '__main__':
    main()
