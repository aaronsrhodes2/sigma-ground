"""Phase I.4 figure — 5-event catalog echo search summary.

Produces misc/bh_phase_i_4_catalog.png:
  - Top section: per-event per-detector lollipop SNR panels (no-subtract)
  - Bottom row: coherence scatter plots (H1×L1 for all; H1×V1, L1×V1 for GW170814)
  - Inset Fisher summary bar

Data hard-coded from misc/bh_phase_i_4_catalog_output.txt (2026-04-17).
"""
from __future__ import annotations

import os
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# ─── SNR data (no-subtract, Phase I.4) ───────────────────────────────────────

DELAYS = {
    'GW151226': [0.756, 1.512, 2.267, 3.023, 3.779],
    'GW150914': [2.260, 4.520, 6.781, 9.041, 11.301],
    'GW170814': [1.933, 3.866, 5.799, 7.733, 9.666],
    'GW170104': [1.784, 3.568, 5.353, 7.137, 8.921],
    'GW190521': [5.160, 10.320, 15.480, 20.640, 25.800],
}

SNR = {
    ('GW151226', 'H1'): [+0.070, -0.409, +0.202, +0.338, +0.916],
    ('GW151226', 'L1'): [+0.405, -1.211, -0.323, -0.282, +0.199],
    ('GW150914', 'H1'): [+0.777, -2.038, -0.965, -1.679, +1.293],
    ('GW150914', 'L1'): [+0.618, -1.570, +0.032, +0.586, +1.080],
    ('GW170814', 'H1'): [-1.291, +0.034, +0.283, -0.357, +0.284],
    ('GW170814', 'L1'): [+0.990, +1.308, -1.169, -1.175, +0.616],
    ('GW170814', 'V1'): [-1.181, -1.377, +0.643, +0.835, -1.086],
    ('GW170104', 'H1'): [+1.558, -1.509, -0.488, -0.827, -1.277],
    ('GW170104', 'L1'): [-2.190, -0.380, +1.351, -1.199, +0.608],
    ('GW190521', 'H1'): [+1.727, -0.382, -0.570, +0.369, -0.509],
    ('GW190521', 'L1'): [-0.627, +0.114, -1.157, +0.073, -0.727],
}

BG_P99 = {
    ('GW151226', 'H1'): 2.489, ('GW151226', 'L1'): 2.475,
    ('GW150914', 'H1'): 2.651, ('GW150914', 'L1'): 2.559,
    ('GW170814', 'H1'): 2.393, ('GW170814', 'L1'): 2.532,
    ('GW170814', 'V1'): 2.610,
    ('GW170104', 'H1'): 2.759, ('GW170104', 'L1'): 2.892,
    ('GW190521', 'H1'): 2.439, ('GW190521', 'L1'): 2.420,
}

COMBINED_P = {
    ('GW151226', 'H1'): 0.952, ('GW151226', 'L1'): 0.879,
    ('GW150914', 'H1'): 0.079, ('GW150914', 'L1'): 0.516,
    ('GW170814', 'H1'): 0.865, ('GW170814', 'L1'): 0.357,
    ('GW170814', 'V1'): 0.353,
    ('GW170104', 'H1'): 0.206, ('GW170104', 'L1'): 0.144,
    ('GW190521', 'H1'): 0.615, ('GW190521', 'L1'): 0.825,
}

# Cross-detector coherence per (event, pair_label): (sign_k, binomial_p, z, one_sided_p)
COHERENCE = {
    ('GW151226', 'H1×L1'): (3, 0.500, +0.242, 0.385),
    ('GW150914', 'H1×L1'): (3, 0.500, +1.827, 0.037),
    ('GW170814', 'H1×L1'): (3, 0.500, -0.430, 0.693),
    ('GW170814', 'H1×V1'): (2, 0.812, +0.469, 0.291),
    ('GW170814', 'L1×V1'): (0, 1.000, -2.407, 0.986),
    ('GW170104', 'H1×L1'): (2, 0.812, -1.473, 0.935),
    ('GW190521', 'H1×L1'): (3, 0.500, -0.033, 0.511),
}

EVENTS_ORDER = ['GW151226', 'GW150914', 'GW170814', 'GW170104', 'GW190521']
DETS = {
    'GW151226': ('H1', 'L1'),
    'GW150914': ('H1', 'L1'),
    'GW170814': ('H1', 'L1', 'V1'),
    'GW170104': ('H1', 'L1'),
    'GW190521': ('H1', 'L1'),
}

DET_COLOUR = {'H1': '#6cf', 'L1': '#fc8', 'V1': '#b8f'}


def _snr_col(snr_abs, cap=3.0):
    x = min(snr_abs / cap, 1.0)
    return (0.25 + 0.75 * x, 0.55 - 0.35 * x, 0.85 - 0.45 * x)


def draw_snr(ax, event, det):
    snrs = SNR[(event, det)]
    delays = DELAYS[event]
    p99 = BG_P99[(event, det)]
    p_comb = COMBINED_P[(event, det)]
    y_max = max(3.2, 1.15 * max(abs(s) for s in snrs), 1.1 * p99)

    ax.axhline(0, color='#666', lw=0.7, alpha=0.5)
    ax.axhline(+p99, color='#f6b26b', lw=0.9, ls='--', alpha=0.65)
    ax.axhline(-p99, color='#f6b26b', lw=0.9, ls='--', alpha=0.65)

    dc = DET_COLOUR[det]
    for i, (dt, snr) in enumerate(zip(delays, snrs)):
        col = _snr_col(abs(snr))
        ax.plot([dt, dt], [0, snr], color=col, lw=2.4, solid_capstyle='round')
        ax.scatter([dt], [snr], s=52, color=col, edgecolors='white',
                   linewidths=0.6, zorder=5)
        label_y = snr + (0.06 * y_max if snr >= 0 else -0.06 * y_max)
        va = 'bottom' if snr >= 0 else 'top'
        ax.text(dt, label_y, f'{snr:+.2f}', ha='center', va=va,
                fontsize=7, color='#ddd')
        ax.text(dt, -y_max * 0.94, f'n={i+1}', ha='center', va='bottom',
                fontsize=6.5, color='#999')

    p_str = f'p={p_comb:.3f}' if p_comb >= 0.001 else f'p<0.001'
    ax.set_title(f'{event}  {det}  ({p_str})', fontsize=9,
                 color=dc, pad=3)
    ax.set_xlim(0, max(delays) * 1.15)
    ax.set_ylim(-y_max, y_max)
    ax.set_xlabel('Δt_n [ms]', fontsize=8)
    ax.set_ylabel('SNR [σ]', fontsize=8)
    ax.tick_params(labelsize=7.5, colors='#bbb')
    for spine in ax.spines.values():
        spine.set_color('#444')


def draw_coherence(ax, event, pair_label):
    det_a, det_b = pair_label.split('×')
    h = np.array(SNR[(event, det_a)])
    l = np.array(SNR[(event, det_b)])
    k, bin_p, z_c, p_c = COHERENCE[(event, pair_label)]

    lim = max(3.2, 1.2 * max(np.max(np.abs(h)), np.max(np.abs(l))))
    xs = np.linspace(-lim, lim, 200)
    ax.fill_between(xs, xs * 0.33, xs * 3.0, color='#3a5', alpha=0.08)
    ax.fill_between(xs, xs * -3.0, xs * -0.33, color='#3a5', alpha=0.08)
    ax.axhline(0, color='#666', lw=0.5, alpha=0.4)
    ax.axvline(0, color='#666', lw=0.5, alpha=0.4)

    n_echoes = len(h)
    for i in range(n_echoes):
        same = (h[i] * l[i]) > 0
        col = '#6c6' if same else '#e66'
        ax.scatter([h[i]], [l[i]], s=95, color=col, edgecolors='white',
                   linewidths=0.7, zorder=5)
        ax.text(h[i], l[i] + 0.06 * lim, f'n={i+1}', ha='center',
                va='bottom', fontsize=6.5, color='#eee', zorder=6)

    sign_col = '#6c6' if k >= 3 else '#e66'
    txt = (f'sign {k}/5 (p={bin_p:.3f})\n'
           f'z = {z_c:+.2f}  p_c = {p_c:.3f}')
    ax.text(0.04, 0.97, txt, transform=ax.transAxes, ha='left', va='top',
            fontsize=7.5, color=sign_col,
            bbox={'boxstyle': 'round,pad=0.3', 'facecolor': '#1a1a1a',
                  'edgecolor': '#444', 'alpha': 0.9})

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(f'{det_a} SNR [σ]', fontsize=8)
    ax.set_ylabel(f'{det_b} SNR [σ]', fontsize=8)
    ax.set_title(f'{event}  {pair_label}', fontsize=9, color='#aaa', pad=3)
    ax.tick_params(labelsize=7.5, colors='#bbb')
    for spine in ax.spines.values():
        spine.set_color('#444')


def draw_fisher_summary(ax):
    """Mini bar chart: per-detector-event combined p-value."""
    labels = [
        'GW151226/H1', 'GW151226/L1',
        'GW150914/H1', 'GW150914/L1',
        'GW170814/H1', 'GW170814/L1', 'GW170814/V1',
        'GW170104/H1', 'GW170104/L1',
        'GW190521/H1', 'GW190521/L1',
    ]
    pvals = [COMBINED_P[(ev.replace('/H1','').replace('/L1','').replace('/V1',''),
                          det)] for ev, det in
             [('GW151226', 'H1'), ('GW151226', 'L1'),
              ('GW150914', 'H1'), ('GW150914', 'L1'),
              ('GW170814', 'H1'), ('GW170814', 'L1'), ('GW170814', 'V1'),
              ('GW170104', 'H1'), ('GW170104', 'L1'),
              ('GW190521', 'H1'), ('GW190521', 'L1')]]
    x = range(len(labels))
    cols = [DET_COLOUR.get(lbl.split('/')[1], '#aaa') for lbl in labels]
    bars = ax.bar(x, pvals, color=cols, edgecolor='#333', linewidth=0.5,
                  alpha=0.85)
    ax.axhline(0.05, color='#f88', lw=1.0, ls='--', alpha=0.7,
               label='p=0.05')
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=7.5,
                       color='#ccc')
    ax.set_ylabel('combined p-value', fontsize=8, color='#ccc')
    ax.set_ylim(0, 1.05)
    ax.set_title(
        'Catalog: per-detector-event combined p  |  Fisher χ²=19.6, dof=22, p_F=0.609',
        fontsize=9, color='#aaa', pad=4)
    ax.tick_params(labelsize=7.5, colors='#bbb')
    ax.legend(fontsize=7.5, framealpha=0.0, labelcolor='#ccc')
    for spine in ax.spines.values():
        spine.set_color('#444')


def main():
    plt.rcParams.update({
        'figure.facecolor': '#111', 'axes.facecolor': '#1a1a1a',
        'savefig.facecolor': '#111', 'text.color': '#eee',
        'axes.labelcolor': '#ccc', 'font.family': 'DejaVu Sans',
    })

    # Layout: 4 rows × 6 cols
    #   Row 0: GW151226 H1, GW151226 L1, GW150914 H1, GW150914 L1, [coherence ×2]
    #   Row 1: GW170814 H1, GW170814 L1, GW170814 V1, GW170104 H1, GW170104 L1, GW190521 H1
    #   Row 2: GW190521 L1, coh(151226), coh(150914), coh(170814-H1L1), coh(170104), coh(190521)
    #   Row 3: GW170814 H1×V1, GW170814 L1×V1, Fisher summary (spans 4 cols)
    fig = plt.figure(figsize=(21, 16))
    gs = gridspec.GridSpec(4, 6, figure=fig,
                           hspace=0.55, wspace=0.40,
                           left=0.04, right=0.98, top=0.94, bottom=0.08)

    # Row 0: O1 event SNR panels
    for ci, (ev, det) in enumerate([('GW151226','H1'),('GW151226','L1'),
                                     ('GW150914','H1'),('GW150914','L1')]):
        draw_snr(fig.add_subplot(gs[0, ci]), ev, det)
    # Row 0 coherence (O1 pair)
    draw_coherence(fig.add_subplot(gs[0, 4]), 'GW151226', 'H1×L1')
    draw_coherence(fig.add_subplot(gs[0, 5]), 'GW150914', 'H1×L1')

    # Row 1: GW170814 (3 det) + GW170104 H1, L1 + GW190521 H1
    for ci, (ev, det) in enumerate([('GW170814','H1'),('GW170814','L1'),
                                     ('GW170814','V1'),('GW170104','H1'),
                                     ('GW170104','L1'),('GW190521','H1')]):
        draw_snr(fig.add_subplot(gs[1, ci]), ev, det)

    # Row 2: GW190521 L1 + catalog coherence panels
    draw_snr(fig.add_subplot(gs[2, 0]), 'GW190521', 'L1')
    draw_coherence(fig.add_subplot(gs[2, 1]), 'GW170814', 'H1×L1')
    draw_coherence(fig.add_subplot(gs[2, 2]), 'GW170104', 'H1×L1')
    draw_coherence(fig.add_subplot(gs[2, 3]), 'GW190521', 'H1×L1')

    # Row 3: GW170814 3-det coherence pairs + Fisher bar
    draw_coherence(fig.add_subplot(gs[3, 0]), 'GW170814', 'H1×V1')
    draw_coherence(fig.add_subplot(gs[3, 1]), 'GW170814', 'L1×V1')
    draw_fisher_summary(fig.add_subplot(gs[3, 2:6]))

    fig.suptitle(
        'Phase I.4 — ξ-shell echo search, 5-event catalog (O1/O2/O3a)\n'
        'subtraction_mode=none  |  σ_conv = −ln(ξ) ≈ 1.844  |  '
        'Fisher p_F = 0.609 (consistent with null)',
        fontsize=13, color='#eee', y=0.975)

    out = os.path.normpath(
        os.path.join(os.path.dirname(__file__), '..', 'misc',
                     'bh_phase_i_4_catalog.png'))
    fig.savefig(out, dpi=125, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f'wrote {out} ({os.path.getsize(out) / 1024:.1f} KB)')


if __name__ == '__main__':
    main()
