"""
Black-hole merger predictions from sigma-ground and the duality ellipse.

Combines three pieces that were already separately derived:

    1. ξ-shell geometry (from RODM_hypothesis.md / run_nested_bh_chain.py):
       Black-hole interiors have nested pressure layers at radii
           r_n = r_s · ξⁿ         (inward nesting)
       where ξ ≈ 0.1582 is the per-event matter-conversion fraction.

    2. Tortoise-coordinate echo delays (ECO / Cardoso-Pani framework):
       A perturbation reflecting off a shell at fractional distance ε_n = ξⁿ
       from the would-be horizon returns to the external observer after a
       round-trip coordinate-time delay
           Δt_n = 2·r_s·|ln ε_n|/c  =  2·r_s·n·σ_conv/c
       since σ_conv = −ln ξ.  So the SAME constant that sets matter-
       conversion thresholds sets the echo spacing.

    3. γ(σ_conv) ringdown-amplitude modulation (Phase G):
       The remnant's environment-marginal coherence γ at the conversion
       horizon multiplies the GR ringdown strain envelope:
           h(t) = A_GR · γ(σ_conv) · exp(−t/τ_QNM) · cos(2π·f_QNM·t + φ)
       Different σ→γ modes predict different amplitude suppressions:
           linear, cbrt → Θ       ≈ 0.7461   (H4 endpoint)
           sigma_coh    → 1 − η/2 ≈ 0.7924   (parsimony recommendation)
           exp          → Θ+(1−Θ)/e ≈ 0.8395 (slowest decay)
           H1 / pure GR → 1.0               (null hypothesis)

──────────────────────────────────────────────────────────────────────────────
PREDICTIONS
──────────────────────────────────────────────────────────────────────────────

For a remnant of mass M_kg with Schwarzschild radius r_s = 2GM/c²:

    Echo delays:     Δt_n = 2·r_s·n·σ_conv/c,    n = 1, 2, 3, ...
    Strain supp.:    h_measured / h_GR  =  γ(σ_conv) under chosen mode.

Both are falsifiable against LIGO/Virgo high-SNR events.  See companion
predictions table in misc/bh_merger_predictions.md.

The INSIDE-shell interpretation (r_n = r_s·ξⁿ inside the horizon, reflective
under ECO-like soft-horizon models) is the primary one here; an OUTSIDE
interpretation (r_n = r_s/ξⁿ, nested-outward) is also physically natural
under RODM and gives Δt_n = 2·r_s·(1/ξⁿ − 1)/c — geometrically growing
rather than arithmetically growing delays.  Both are implemented so
empirical data can discriminate.
"""

import math

from sigma_ground.field.constants import (
    C, G, M_SUN_KG, SIGMA_CONV, XI,
)
from sigma_ground.field.interface.duality_ellipse import coherence_gamma_from_sigma


# Named LIGO/Virgo events with published remnant-mass estimates (solar masses).
# Sources: LIGO-Virgo catalog papers (GWTC-1, GWTC-2.1, GWTC-3).
NAMED_EVENTS = {
    'GW150914': {'M_remnant_solar': 62.2, 'M_total_solar': 65.3,  'f_QNM_Hz': 251.0, 'tau_QNM_ms': 4.0},
    'GW151226': {'M_remnant_solar': 20.8, 'M_total_solar': 21.8,  'f_QNM_Hz': 760.0, 'tau_QNM_ms': 1.2},
    'GW170104': {'M_remnant_solar': 48.7, 'M_total_solar': 50.7,  'f_QNM_Hz': 320.0, 'tau_QNM_ms': 3.1},
    'GW170814': {'M_remnant_solar': 53.2, 'M_total_solar': 55.9,  'f_QNM_Hz': 295.0, 'tau_QNM_ms': 3.4},
    'GW190521': {'M_remnant_solar': 142.0, 'M_total_solar': 150.0, 'f_QNM_Hz': 110.0, 'tau_QNM_ms': 9.0},
}


def schwarzschild_radius(M_kg):
    """r_s = 2GM/c².

    Args:
        M_kg: mass in kilograms; must be > 0.

    Returns:
        Schwarzschild radius in metres.
    """
    if M_kg <= 0:
        raise ValueError("M_kg must be > 0")
    return 2.0 * G * M_kg / (C * C)


def shell_radii(M_kg, n_max=5, direction='inward'):
    """Nested pressure-shell radii, in metres.

    direction='inward':   r_n = r_s · ξⁿ    (n = 1..n_max; inside horizon)
    direction='outward':  r_n = r_s · ξ⁻ⁿ   (n = 1..n_max; RODM nesting-up)

    FIRST_PRINCIPLES: ξ ≈ 0.1582 is the per-conversion baryonic fraction
    (XI in constants.py), derived from the Planck 2018 baryon-to-total
    matter ratio and taken as the geometric ratio between successive
    gravitational-pressure layers in sigma-ground.

    Args:
        M_kg:       remnant mass in kg
        n_max:      how many shells to return
        direction:  'inward' (default) or 'outward'

    Returns:
        list of r_n in metres, n = 1..n_max
    """
    if n_max < 1:
        return []
    r_s = schwarzschild_radius(M_kg)
    if direction == 'inward':
        return [r_s * (XI ** n) for n in range(1, n_max + 1)]
    if direction == 'outward':
        return [r_s * (XI ** (-n)) for n in range(1, n_max + 1)]
    raise ValueError(f"unknown direction '{direction}'; expected 'inward' or 'outward'")


def echo_delays(M_kg, n_max=5, direction='inward'):
    """Round-trip coordinate-time echo delays at each ξ-shell, in seconds.

    direction='inward':
        Δt_n = 2·r_s·n·σ_conv/c
        Derivation: tortoise coordinate r* = r + r_s·ln|r/r_s − 1| diverges
        logarithmically as r → r_s.  A shell at r_n = r_s·(1 + ξⁿ), i.e.
        fractional displacement ε_n = ξⁿ above the would-be horizon, has
        r*(r_n) ≈ r_s·ln(ξⁿ) = −r_s·n·σ_conv.  Round-trip to external
        observer = 2·r_s·n·σ_conv/c.  Delays grow LINEARLY in n.

    direction='outward':
        Δt_n = 2·r_s·(ξ⁻ⁿ − 1)/c
        Propagation from shell at r_n = r_s·ξ⁻ⁿ to horizon through
        asymptotically flat region.  Delays grow GEOMETRICALLY in n.

    Args:
        M_kg:       remnant mass in kg
        n_max:      how many echoes to predict
        direction:  'inward' (default) or 'outward'

    Returns:
        list of Δt_n in seconds, n = 1..n_max
    """
    if n_max < 1:
        return []
    r_s = schwarzschild_radius(M_kg)
    if direction == 'inward':
        return [2.0 * r_s * n * SIGMA_CONV / C for n in range(1, n_max + 1)]
    if direction == 'outward':
        return [2.0 * r_s * ((XI ** (-n)) - 1.0) / C for n in range(1, n_max + 1)]
    raise ValueError(f"unknown direction '{direction}'; expected 'inward' or 'outward'")


def gamma_at_merger(mode='sigma_coh'):
    """γ(σ_conv) — environment-marginal coherence at the remnant horizon.

    This is the single-number ringdown suppression factor predicted by
    each σ→γ candidate mode.  The merger remnant's local σ pegs at
    σ_conv during the ringdown (maximum-γ-suppression regime).

    Args:
        mode:  'linear', 'exp', 'cbrt', or 'sigma_coh' (default).

    Returns:
        γ in [0, 1] — strain-envelope multiplier vs pure-GR template.
    """
    return coherence_gamma_from_sigma(SIGMA_CONV, mode=mode)


def ringdown_amplitude_suppression(mode='sigma_coh'):
    """Alias for gamma_at_merger; named for clarity in waveform contexts.

    h_measured(t) / h_GR(t) = ringdown_amplitude_suppression(mode)
    holds across the full ringdown envelope under the sigma-ground
    ansatz that γ is approximately constant at γ(σ_conv) during the
    damping time τ_QNM.
    """
    return gamma_at_merger(mode=mode)


def predict_event(M_remnant_solar, name=None, n_max=5, direction='inward'):
    """Compute full sigma-ground prediction bundle for a BH-merger remnant.

    Args:
        M_remnant_solar:  remnant mass in solar masses
        name:             optional event tag (e.g. 'GW150914')
        n_max:            how many echo shells to predict
        direction:        'inward' (default) or 'outward'

    Returns:
        dict with keys:
            name                          (str or None)
            M_remnant_kg                  (float)
            r_s_m, r_s_km                 (float)
            light_crossing_ms             (float, r_s/c in ms)
            shell_radii_m                 (list of n_max floats)
            echo_delays_s                 (list of n_max floats)
            echo_delays_ms                (list of n_max floats)
            direction                     (str)
            gamma_by_mode                 (dict mode -> γ(σ_conv))
            amplitude_suppression_by_mode (same, reworded for clarity)
    """
    if M_remnant_solar <= 0:
        raise ValueError("M_remnant_solar must be > 0")
    M_kg = M_remnant_solar * M_SUN_KG
    r_s = schwarzschild_radius(M_kg)
    shells = shell_radii(M_kg, n_max=n_max, direction=direction)
    delays_s = echo_delays(M_kg, n_max=n_max, direction=direction)
    modes = ('linear', 'exp', 'cbrt', 'sigma_coh')
    gamma_map = {m: gamma_at_merger(mode=m) for m in modes}
    return {
        'name': name,
        'M_remnant_solar': M_remnant_solar,
        'M_remnant_kg': M_kg,
        'r_s_m': r_s,
        'r_s_km': r_s / 1000.0,
        'light_crossing_ms': (r_s / C) * 1000.0,
        'shell_radii_m': shells,
        'echo_delays_s': delays_s,
        'echo_delays_ms': [d * 1000.0 for d in delays_s],
        'direction': direction,
        'gamma_by_mode': gamma_map,
        'amplitude_suppression_by_mode': dict(gamma_map),
    }


def predict_named_event(event_name, n_max=5, direction='inward'):
    """Convenience: look up a published LIGO/Virgo event by name and predict.

    Args:
        event_name:  one of NAMED_EVENTS keys (e.g. 'GW150914')
        n_max:       how many echoes to predict
        direction:   'inward' (default) or 'outward'

    Returns:
        same dict as predict_event, with name filled in.
    """
    if event_name not in NAMED_EVENTS:
        raise ValueError(
            f"unknown event '{event_name}'; known: {sorted(NAMED_EVENTS.keys())}"
        )
    M_solar = NAMED_EVENTS[event_name]['M_remnant_solar']
    out = predict_event(M_solar, name=event_name, n_max=n_max, direction=direction)
    out['f_QNM_Hz'] = NAMED_EVENTS[event_name]['f_QNM_Hz']
    out['tau_QNM_ms'] = NAMED_EVENTS[event_name]['tau_QNM_ms']
    return out
