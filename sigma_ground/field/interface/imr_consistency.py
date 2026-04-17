"""
LIGO IMR (inspiral-merger-ringdown) consistency test primitive.

Phase H.3 of the duality-ellipse / BH-merger program.  Tests whether
the Hypothesis B1 ("merger-triggered mass shedding") prediction is
compatible with published LIGO IMR consistency bounds.

──────────────────────────────────────────────────────────────────────────────
BACKGROUND
──────────────────────────────────────────────────────────────────────────────

LIGO's IMR consistency test (Ghosh et al. 2016, arXiv:1602.03841) compares:
  - (M_f, a_f)_insp  — final mass/spin *predicted* from inspiral-only
                       parameters via numerical-relativity fitting formulas
  - (M_f, a_f)_MR    — final mass/spin *measured* from merger-ringdown only

Standardised fractional deviation parameter:
    ε_M  =  2·(M_f_insp − M_f_MR) / (M_f_insp + M_f_MR)

Under GR (Hypothesis A): ε_M = 0 identically.
Under Hypothesis B1: the ringdown sees a lighter remnant because the
conversion fraction ξ has gravitationally decoupled.  If f_rad is the
fraction of initial mass-energy radiated as GWs, then:

    M_f_insp = M_tot·(1 − f_rad)         (GR prediction ignoring B1)
    M_f_MR   = M_tot·(1 − f_rad − ξ)     (B1: extra ξ has decoupled)

    ε_M_B1   =  2·ξ / (2·(1 − f_rad) − ξ)

For f_rad ∈ [0.04, 0.10] typical of stellar-mass BH mergers and ξ =
0.1582, this gives ε_M_B1 ∈ [0.18, 0.19].

──────────────────────────────────────────────────────────────────────────────
PUBLISHED BOUNDS
──────────────────────────────────────────────────────────────────────────────

Per-event and combined bounds on ε_M at 90 % CL.  Sources:
  - Abbott et al. 2016, PRL 116, 221101 (GW150914 TGR)
  - Abbott et al. 2019, PRD 100, 104036 (GWTC-1 combined TGR)
  - Abbott et al. 2021, PRD 103, 122002 (GWTC-2 TGR)
  - Abbott et al. 2021, arXiv:2112.06861 (GWTC-3 TGR)

We report only the well-established medians and 90 % CL half-widths;
asymmetric tails are handled by taking the worse-for-B1 (upper) side.
"""

import math

from sigma_ground.field.constants import XI


NOMINAL_F_RAD = 0.05  # typical radiated-energy fraction for stellar-mass BH mergers


# Published LIGO IMR consistency bounds on ε_M.
# Format: {event: (median, lower_90, upper_90, source)}
# Bounds extracted from published TGR analyses; values rounded to 2 dp.
# Upper_90 is the quantity relevant to excluding a POSITIVE B1 prediction.
LIGO_IMR_BOUNDS = {
    'GW150914':        (-0.04, -0.14,  0.06, 'Abbott 2016 PRL 116:221101'),
    'GW151226':        (+0.09, -0.19,  0.38, 'Abbott 2016 PRX 6:041015'),
    'GW170104':        (+0.02, -0.17,  0.21, 'Abbott 2017 PRL 118:221101'),
    'GW170814':        (-0.03, -0.15,  0.09, 'Abbott 2019 PRD 100:104036'),
    'GW190521':        (-0.08, -0.48,  0.32, 'Abbott 2021 arXiv:2112.06861'),
    # Catalog-wide combined posterior (GWTC-3 TGR, ~15 high-quality events)
    'COMBINED_GWTC3':  (-0.01, -0.06,  0.04, 'Abbott 2021 arXiv:2112.06861'),
}


def predict_epsilon_M(hypothesis='A', f_rad=NOMINAL_F_RAD):
    """ε_M predicted under the given hypothesis.

    Args:
        hypothesis:  'A' (GR / mass conservation), 'B1' (merger-triggered
                     mass shedding of fraction ξ).
        f_rad:       radiated GW energy fraction of M_tot.  Default
                     0.05 is typical for stellar-mass equal-mass mergers.

    Returns:
        ε_M in [−1, +1] — expected value of the IMR consistency
        parameter.
    """
    if hypothesis == 'A':
        return 0.0
    if hypothesis == 'B1':
        denom = 2.0 * (1.0 - f_rad) - XI
        if denom <= 0.0:
            raise ValueError(
                f"Unphysical: f_rad={f_rad} gives non-positive denominator "
                f"for XI={XI}; f_rad must be < 1 − XI/2 = {1.0 - XI/2.0}"
            )
        return 2.0 * XI / denom
    raise ValueError(f"unknown hypothesis '{hypothesis}'; expected 'A' or 'B1'")


def published_bound(event_name):
    """Published ε_M bound for an event.

    Returns:
        (median, lower_90, upper_90, source_string)

    Raises:
        ValueError if event unknown.
    """
    if event_name not in LIGO_IMR_BOUNDS:
        raise ValueError(
            f"unknown event '{event_name}'; known: "
            f"{sorted(LIGO_IMR_BOUNDS.keys())}"
        )
    return LIGO_IMR_BOUNDS[event_name]


def _approx_sigma_from_90ci(median, lower_90, upper_90, side='upper'):
    """Approximate 1-σ from a 90 % CL interval.

    Args:
        side:  'upper' or 'lower' — which half-width to use, since the
               published bounds are often asymmetric.
    """
    if side == 'upper':
        half_width_90 = upper_90 - median
    else:
        half_width_90 = median - lower_90
    # 90 % CL for a Gaussian corresponds to ±1.645·σ.
    return half_width_90 / 1.645


def exclusion_sigma(event_name, hypothesis='B1', f_rad=NOMINAL_F_RAD):
    """Approximate σ-level at which the hypothesis is excluded by this event.

    Uses the upper-tail half-width of the 90 % CL interval (since B1
    predicts positive ε_M, the upper bound is what constrains it).

    Returns:
        σ-level (float).  Positive means hypothesis is in tension with
        data; negative means prediction sits inside the bound.
    """
    median, lower, upper, _ = published_bound(event_name)
    predicted = predict_epsilon_M(hypothesis, f_rad=f_rad)
    if predicted == 0.0:
        return 0.0
    side = 'upper' if predicted > 0 else 'lower'
    sigma = _approx_sigma_from_90ci(median, lower, upper, side=side)
    if sigma <= 0.0:
        return math.inf
    return (predicted - median) / sigma


def exclusion_report(hypothesis='B1', f_rad=NOMINAL_F_RAD):
    """Per-event exclusion sigmas for every catalogued event.

    Returns:
        list of (event_name, predicted_eps_M, published_median,
                 published_upper_90, sigma_level, source) tuples,
        sorted by event name.
    """
    predicted = predict_epsilon_M(hypothesis, f_rad=f_rad)
    rows = []
    for name in sorted(LIGO_IMR_BOUNDS.keys()):
        median, lower, upper, source = LIGO_IMR_BOUNDS[name]
        sigma = exclusion_sigma(name, hypothesis=hypothesis, f_rad=f_rad)
        rows.append((name, predicted, median, upper, sigma, source))
    return rows
