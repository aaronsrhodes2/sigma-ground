"""Phase I.3 figure — subtraction-mode ablation (n0 vs none).

Produces misc/bh_phase_i_3_no_subtraction.png: a 3×2 grid showing
per-event per-detector matched-filter SNR at each predicted Δt_n under
n0-subtract (red, left column) and no-subtract (blue, right column),
with per-panel |SNR| colour scale, bg p99 reference line, and a bottom
row of cross-detector coherence scatter plots under each mode.

Data is hard-coded from misc/bh_phase_i_3_no_subtract_output.txt to
make the figure deterministic and re-rerunnable without a 4-minute
pipeline call.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# ─────────────────────────────────────────────────────────────────────────
# Data from misc/bh_phase_i_3_no_subtract_output.txt (captured 2026-04-17)
# Per-echo calibrated SNRs.  Sign preserved.
# ─────────────────────────────────────────────────────────────────────────

DELAYS_MS = {
    'GW151226': [0.756, 1.512, 2.267, 3.023, 3.779],
    'GW150914': [2.260, 4.520, 6.781, 9.041, 11.301],
}

# (event, detector, mode) → [SNR per n=1..5]
SNR = {
    ('GW151226', 'H1', 'n0'):   [+1.215,  +3.321,  -3.239,  +2.428,  -0.334],
    ('GW151226', 'H1', 'none'): [+0.070,  -0.409,  +0.202,  +0.338,  +0.916],
    ('GW151226', 'L1', 'n0'):   [-32.567, +34.687, -25.381, +11.569, -5.722],
    ('GW151226', 'L1', 'none'): [+0.405,  -1.211,  -0.323,  -0.282,  +0.199],
    ('GW150914', 'H1', 'n0'):   [+0.470,  -1.428,  -1.494,  -1.381,  +1.176],
    ('GW150914', 'H1', 'none'): [+0.777,  -2.038,  -0.965,  -1.679,  +1.293],
    ('GW150914', 'L1', 'n0'):   [-12.025, -4.434,  +5.246,  -4.039,  +4.354],
    ('GW150914', 'L1', 'none'): [+0.618,  -1.570,  +0.032,  +0.586,  +1.080],
}

# Background p99 per (event, detector, mode)
BG_P99 = {
    ('GW151226', 'H1', 'n0'):   2.531,
    ('GW151226', 'H1', 'none'): 2.489,
    ('GW151226', 'L1', 'n0'):   2.422,
    ('GW151226', 'L1', 'none'): 2.475,
    ('GW150914', 'H1', 'n0'):   2.650,
    ('GW150914', 'H1', 'none'): 2.651,
    ('GW150914', 'L1', 'n0'):   2.558,
    ('GW150914', 'L1', 'none'): 2.559,
}

# Cross-detector coherence diagnostics per (event, mode).
# sign_matches, binomial_p, magnitude-weighted z-score, 1-sided p.
COHERENCE = {
    ('GW151226', 'n0'):   {'sign': (4, 0.188), 'z': 83.128, 'p': 0.0000},
    ('GW151226', 'none'): {'sign': (3, 0.500), 'z': 0.242,  'p': 0.3852},
    ('GW150914', 'n0'):   {'sign': (3, 0.500), 'z': 1.575,  'p': 0.0556},
    ('GW150914', 'none'): {'sign': (3, 0.500), 'z': 1.827,  'p': 0.0366},
}


def _snr_colour(snr_abs: float, saturate_at: float = 5.0) -> tuple[float, float, float]:
    """Colour coding: small |SNR| grey, large |SNR| hot."""
    x = min(snr_abs / saturate_at, 1.0)
    return (0.3 + 0.7 * x, 0.3 + 0.3 * (1 - x), 0.3 * (1 - x))


def draw_snr_panel(ax, event: str, detector: str, mode: str):
    """Lollipop SNR panel for one (event, detector, mode) combination."""
    snrs = SNR[(event, detector, mode)]
    delays = DELAYS_MS[event]
    p99 = BG_P99[(event, detector, mode)]

    ax.axhline(0.0, color='#888', lw=0.8, alpha=0.6)
    ax.axhline(+p99, color='#f6b26b', lw=0.8, ls='--', alpha=0.7,
               label=f'bg |SNR| p99 = {p99:.2f}')
    ax.axhline(-p99, color='#f6b26b', lw=0.8, ls='--', alpha=0.7)

    y_max = max(6.0, 1.1 * max(abs(s) for s in snrs), 1.1 * p99)

    for i, (dt, snr) in enumerate(zip(delays, snrs)):
        col = _snr_colour(abs(snr), saturate_at=max(5.0, 0.6 * y_max))
        ax.plot([dt, dt], [0, snr], color=col, lw=2.2, solid_capstyle='round')
        ax.scatter([dt], [snr], s=55, color=col,
                   edgecolors='white', linewidths=0.6, zorder=5)
        label_y = snr + (0.04 * y_max if snr >= 0 else -0.04 * y_max)
        va = 'bottom' if snr >= 0 else 'top'
        ax.text(dt, label_y, f'{snr:+.2f}', ha='center', va=va, fontsize=7.5,
                color='#eee')
        ax.text(dt, -y_max * 0.96, f'n={i+1}', ha='center', va='bottom',
                fontsize=7, color='#aaa')

    ax.set_xlim(0, max(delays) * 1.12)
    ax.set_ylim(-y_max, y_max)
    ax.set_xlabel('Δt_n [ms]', fontsize=9)
    ax.set_ylabel('matched-filter SNR [σ]', fontsize=9)
    mode_label = 'n0-subtract' if mode == 'n0' else 'no subtraction'
    title_colour = '#e88' if mode == 'n0' else '#8be'
    ax.set_title(f'{event}  {detector}  —  {mode_label}',
                 fontsize=10, color=title_colour, pad=4)
    ax.tick_params(colors='#ccc', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#555')
    ax.legend(loc='lower right', fontsize=7, framealpha=0.0,
              labelcolor='#ccc')


def draw_coherence_panel(ax, event: str, mode: str):
    """H1×L1 scatter with |L1/H1| plausibility band and sign-match colouring."""
    h1 = np.array(SNR[(event, 'H1', mode)])
    l1 = np.array(SNR[(event, 'L1', mode)])
    coh = COHERENCE[(event, mode)]

    lim = max(6.0, 1.15 * max(np.max(np.abs(h1)), np.max(np.abs(l1))))

    # |L1|/|H1| ∈ [1/3, 3] plausibility band (shaded).
    xs = np.linspace(-lim, lim, 200)
    ax.fill_between(xs, xs * 0.33, xs * 3.0, color='#3a5', alpha=0.08,
                    label='|L1|/|H1| ∈ [⅓, 3]')
    ax.fill_between(xs, xs * -3.0, xs * -0.33, color='#3a5', alpha=0.08)

    ax.axhline(0, color='#777', lw=0.6, alpha=0.5)
    ax.axvline(0, color='#777', lw=0.6, alpha=0.5)
    ax.plot([-lim, lim], [-lim, lim], color='#888', lw=0.5, ls=':', alpha=0.5)
    ax.plot([-lim, lim], [lim, -lim], color='#888', lw=0.5, ls=':', alpha=0.5)

    # Scatter: green if sign agrees, red if not.
    for i in range(len(h1)):
        same_sign = (h1[i] * l1[i]) > 0
        col = '#6c6' if same_sign else '#e66'
        ax.scatter([h1[i]], [l1[i]], s=110, color=col, edgecolors='white',
                   linewidths=0.8, zorder=5)
        ax.text(h1[i], l1[i] + 0.05 * lim, f'n={i+1}', ha='center',
                va='bottom', fontsize=7.5, color='#eee', zorder=6)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel('H1 SNR [σ]', fontsize=9)
    ax.set_ylabel('L1 SNR [σ]', fontsize=9)
    mode_label = 'n0-subtract' if mode == 'n0' else 'no subtraction'
    title_colour = '#e88' if mode == 'n0' else '#8be'
    ax.set_title(f'{event}  coherence  —  {mode_label}',
                 fontsize=10, color=title_colour, pad=4)
    ax.tick_params(colors='#ccc', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#555')

    # Stats box.
    sign_k, sign_p = coh['sign']
    txt = (f'sign match: {sign_k}/5 (p={sign_p:.3f} binomial)\n'
           f'mag-weighted z = {coh["z"]:+.2f}\n'
           f'one-sided p = {coh["p"]:.4f}')
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, ha='left', va='top',
            fontsize=7.5, color='#ddd',
            bbox={'boxstyle': 'round,pad=0.35',
                  'facecolor': '#222', 'edgecolor': '#555', 'alpha': 0.85})
    ax.legend(loc='lower right', fontsize=7, framealpha=0.0,
              labelcolor='#ccc')


def main():
    plt.rcParams.update({
        'figure.facecolor': '#111',
        'axes.facecolor': '#181818',
        'savefig.facecolor': '#111',
        'text.color': '#eee',
        'axes.labelcolor': '#ccc',
        'axes.titlecolor': '#eee',
        'font.family': 'DejaVu Sans',
    })

    # Layout: 3 rows × 4 cols.
    #   Row 1: GW151226 H1 (n0), GW151226 H1 (none), GW151226 L1 (n0), GW151226 L1 (none)
    #   Row 2: GW150914 H1 (n0), GW150914 H1 (none), GW150914 L1 (n0), GW150914 L1 (none)
    #   Row 3: GW151226 coh (n0), GW151226 coh (none), GW150914 coh (n0), GW150914 coh (none)
    fig, axes = plt.subplots(3, 4, figsize=(19, 13))

    for col_i, (det, mode) in enumerate([
            ('H1', 'n0'), ('H1', 'none'),
            ('L1', 'n0'), ('L1', 'none')]):
        draw_snr_panel(axes[0, col_i], 'GW151226', det, mode)
        draw_snr_panel(axes[1, col_i], 'GW150914', det, mode)

    for col_i, (ev, mode) in enumerate([
            ('GW151226', 'n0'), ('GW151226', 'none'),
            ('GW150914', 'n0'),  ('GW150914', 'none')]):
        draw_coherence_panel(axes[2, col_i], ev, mode)

    fig.suptitle(
        'Phase I.3 — ξ-shell echo search, subtraction-mode ablation\n'
        'n=0-subtract (red titles) vs no subtraction (blue titles); '
        '4 detector-events × 2 modes + cross-detector coherence',
        fontsize=13, color='#eee', y=0.995)

    fig.text(0.5, 0.013,
             'n=0-subtract (Phase I/I.2 default) injects fit-amplitude-scaled '
             'noise at f_QNM into the residual — matched filter then correlates '
             'template against its own image (L1 worst-affected: A_fit ≈ 30× '
             'literature).  No-subtract gives clean nulls on 3/4 detector-events; '
             'GW150914 no-subtract has a single-point n=2 excess (|SNR|=2.04) but '
             'sign-coherence at chance (3/5).  σ_conv = −ln(ξ) ≈ 1.844 prediction '
             'neither detected nor falsified; pipeline floor ≈ 2.5σ per echo.',
             ha='center', va='bottom', fontsize=8.5, color='#aaa',
             wrap=True)

    fig.tight_layout(rect=(0.005, 0.045, 0.995, 0.965))

    out = os.path.join(os.path.dirname(__file__),
                       '..', 'misc', 'bh_phase_i_3_no_subtraction.png')
    out = os.path.normpath(out)
    fig.savefig(out, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'wrote {out} ({os.path.getsize(out) / 1024:.1f} KB)')


if __name__ == '__main__':
    main()
