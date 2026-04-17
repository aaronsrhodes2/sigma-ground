"""
BH mass-function cutoff analysis — tests Hypothesis B2 from
misc/bh_conversion_mass_hypothesis.md.

──────────────────────────────────────────────────────────────────────────────
B2 RECAP
──────────────────────────────────────────────────────────────────────────────

B2: BH conversion ignites when accumulated mass exceeds a critical
threshold M_crit.  Each conversion event sheds ξ·M of mass (the
"merger-triggered" B1 variant is a special case of B2 at very low
M_crit).  Predicted signatures in the BH mass function:

  1. **Cutoff** — no BHs persist at M > M_crit; dN/dM → 0 above M_crit.
  2. **Pile-up** — BHs land at (1 − ξ)·M_crit after conversion,
     producing a local over-density in dN/dM at the landing zone.
  3. **IMR triggering** — any merger whose remnant mass exceeds M_crit
     should trigger conversion *at* the merger, producing the same
     ε_M ≈ 0.18 signature as B1 for that event.

B2 is thus constrained by two independent channels:
  (a) existence of observed BHs with m > M_crit  → rules out low M_crit
  (b) IMR consistency of events with M_f > M_crit → rules out mid M_crit

──────────────────────────────────────────────────────────────────────────────
PUBLISHED DATA
──────────────────────────────────────────────────────────────────────────────

Highest-confidence primary-BH mass in GWTC-3: **GW190521 primary** at
85 +21/-14 M_sun (90 % CL).  Source: Abbott et al. 2020, PRL 125:101102.
This establishes m_1 ≥ 71 M_sun at 90 % CL — a hard lower bound on
M_crit (a BH with m_1 = 85 M_sun cannot exist under B2 with
M_crit < m_1, because the BH would have converted before reaching that
mass).

Remnant masses of the Phase H.3 event set are reused from
NAMED_EVENTS in bh_merger.py.

──────────────────────────────────────────────────────────────────────────────
PAIR-INSTABILITY GAP (theoretical, for degeneracy discussion)
──────────────────────────────────────────────────────────────────────────────

Stellar-evolution theory predicts BHs cannot form directly from massive-
star collapse in the mass range ~ 45–135 M_sun due to pulsational
pair-instability supernovae (Woosley 2017; Farmer et al. 2019).  BHs in
this range only exist via hierarchical mergers.  Any B2 M_crit inside
this range is therefore degenerate with the pair-instability
explanation; discriminating B2 from the conventional PI-gap story at
M_crit ∈ [45, 135] M_sun requires additional observables (IMR
consistency of individual events, not population statistics).
"""

import math

from sigma_ground.field.constants import XI
from sigma_ground.field.interface.bh_merger import NAMED_EVENTS
from sigma_ground.field.interface.imr_consistency import exclusion_sigma


# GWTC-3 primary-mass 90 % CL intervals for the highest-mass detections.
# Format: {event: (median, lower_90, upper_90, source)}
# Sources:
#   GW190521 primary  — Abbott 2020 PRL 125:101102
#   GW190519_153544   — Abbott 2021 PRX 11:021053 (GWTC-2 catalog)
#   GW170729 primary  — Abbott 2019 PRX 9:031040 (GWTC-1 catalog)
GWTC3_PRIMARY_MASSES = {
    'GW190521_primary':        (85.0, 71.0, 106.0, 'Abbott 2020 PRL 125:101102'),
    'GW190519_153544_primary': (66.0, 46.0,  86.0, 'Abbott 2021 PRX 11:021053'),
    'GW170729_primary':        (50.2, 34.0,  72.0, 'Abbott 2019 PRX 9:031040'),
}

# Pair-instability mass gap (theoretical, for degeneracy discussion).
PAIR_INSTABILITY_LOW_MSUN = 45.0
PAIR_INSTABILITY_HIGH_MSUN = 135.0


def minimum_M_crit_90CL():
    """Hardest lower bound on M_crit from observed primary BH masses.

    Under B2, any BH with m > M_crit would have converted before
    reaching that mass.  Therefore the 90 % CL lower bound on the
    highest observed primary mass is also a 90 % CL lower bound on
    M_crit.

    Returns:
        (M_crit_min_solar, source_event) — the 90 % CL lower bound
        and the event name that sets it.
    """
    best = None
    best_event = None
    for event, (median, lower, upper, _source) in GWTC3_PRIMARY_MASSES.items():
        if best is None or lower > best:
            best = lower
            best_event = event
    return best, best_event


def b2_features(M_crit_solar):
    """B2 structural features for a given M_crit.

    Args:
        M_crit_solar:  threshold mass in solar units

    Returns:
        dict with keys:
            cutoff_M_solar        — mass above which BHs cannot persist
            pileup_M_solar        — landing zone after conversion, (1−ξ)·M_crit
            depletion_range_solar — (pileup_M, cutoff_M)
            drop_fraction         — ξ
    """
    if M_crit_solar <= 0:
        raise ValueError("M_crit_solar must be > 0")
    return {
        'cutoff_M_solar': M_crit_solar,
        'pileup_M_solar': M_crit_solar * (1.0 - XI),
        'depletion_range_solar': (M_crit_solar * (1.0 - XI), M_crit_solar),
        'drop_fraction': XI,
    }


def events_triggered_by_B2(M_crit_solar):
    """Named LIGO events whose remnant mass exceeds M_crit.

    Under B2, these events would have triggered a conversion event at
    the merger, producing the same ε_M ≈ 0.18 IMR signature as B1.

    Returns:
        list of event names (sorted by remnant mass, ascending)
    """
    triggered = [
        (name, meta['M_remnant_solar'])
        for name, meta in NAMED_EVENTS.items()
        if meta['M_remnant_solar'] > M_crit_solar
    ]
    triggered.sort(key=lambda pair: pair[1])
    return [name for name, _ in triggered]


def b2_imr_exclusion(M_crit_solar):
    """Combined IMR exclusion σ of B2 at given M_crit.

    Only events with M_f > M_crit contribute (they would trigger
    conversion under B2, producing ε_M ≈ 0.18).  Individual exclusion
    sigmas combine in quadrature under the Gaussian-independence
    approximation.

    Args:
        M_crit_solar: threshold mass in solar units

    Returns:
        (sigma_combined, list_of_triggered_events)
    """
    triggered = events_triggered_by_B2(M_crit_solar)
    sum_sq = 0.0
    for event in triggered:
        sigma_i = exclusion_sigma(event, hypothesis='B1')
        sum_sq += sigma_i * sigma_i
    return (math.sqrt(sum_sq), triggered)


def b2_viable(M_crit_solar, imr_threshold_sigma=3.0):
    """Is B2 at this M_crit viable given current data?

    Args:
        M_crit_solar:       threshold under test
        imr_threshold_sigma: above this IMR-combined σ, treat as excluded

    Returns:
        dict with keys:
            M_crit_solar       — echoed
            primary_mass_ok    — bool, passes primary-mass lower bound
            primary_mass_bound — M_crit_min_90CL in solar units
            imr_exclusion_sigma — combined σ from triggered events
            imr_ok             — bool, below imr_threshold_sigma
            viable             — bool, both checks pass
            triggered_events   — events that would trigger under B2
            in_PI_gap          — bool, M_crit inside pair-instability gap
    """
    primary_min, _ = minimum_M_crit_90CL()
    primary_ok = M_crit_solar >= primary_min
    imr_sigma, triggered = b2_imr_exclusion(M_crit_solar)
    imr_ok = imr_sigma < imr_threshold_sigma
    in_PI = PAIR_INSTABILITY_LOW_MSUN <= M_crit_solar <= PAIR_INSTABILITY_HIGH_MSUN
    return {
        'M_crit_solar': M_crit_solar,
        'primary_mass_ok': primary_ok,
        'primary_mass_bound': primary_min,
        'imr_exclusion_sigma': imr_sigma,
        'imr_ok': imr_ok,
        'viable': primary_ok and imr_ok,
        'triggered_events': triggered,
        'in_PI_gap': in_PI,
    }


def sweep_M_crit(M_crit_grid=None, imr_threshold_sigma=3.0):
    """Sweep M_crit across a grid and report viability at each point.

    Args:
        M_crit_grid:  iterable of M_crit_solar values to test; default
                      covers 20 → 250 M_sun in 13 steps
        imr_threshold_sigma:  exclusion threshold

    Returns:
        list of b2_viable() dicts, one per grid point, sorted ascending
    """
    if M_crit_grid is None:
        M_crit_grid = (20.0, 30.0, 45.0, 55.0, 65.0, 75.0, 85.0, 100.0,
                       135.0, 150.0, 175.0, 200.0, 250.0)
    return [b2_viable(M, imr_threshold_sigma=imr_threshold_sigma)
            for M in sorted(M_crit_grid)]
