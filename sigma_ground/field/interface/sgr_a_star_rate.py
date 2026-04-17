"""
Sgr A* astrometric constraint on B3 (rare spontaneous conversion).

Phase H.5.  Tests whether GRAVITY-collaboration mass stability of Sgr A*
over ~30 years is strong enough to bound the baseline B3 rate.

──────────────────────────────────────────────────────────────────────────────
B3 RECAP
──────────────────────────────────────────────────────────────────────────────

B3: conversion is a spontaneous event (not merger-triggered, not
critical-mass-triggered) happening at roughly R = 1/τ_Hubble per BH.
Each event drops external mass by ξ ≈ 0.1582 of the BH's current mass.

──────────────────────────────────────────────────────────────────────────────
THE CONSTRAINT
──────────────────────────────────────────────────────────────────────────────

For a Poisson process with rate R [events/yr/BH] observed over
τ_obs years across N_BH objects, the expected number of events is
   λ = R · τ_obs · N_BH
At 90 % CL, null observation (k = 0) rules out λ > −ln(0.10) = 2.303.
So: R < 2.303 / (τ_obs · N_BH)

Sgr A* alone: N_BH = 1, τ_obs ≈ 30 yr.  R_max = 0.0768 /yr.
Baseline B3 rate: R_base = 1/τ_Hubble ≈ 7.2e−11 /yr.
Ratio R_max / R_base ≈ 1.06e9 — **Sgr A* is ~9 orders of magnitude
too weak to test B3 at the baseline rate.**

A stronger bound comes from stellar-mass BH X-ray binary monitoring:
~20 confirmed BH-XRB systems with ~30-year mass stability at ≲10 %
precision.  That gives ~600 object-years.  Rate bound R < 3.8e−3 /yr,
still ~5e7 × baseline B3.

──────────────────────────────────────────────────────────────────────────────
WHAT THIS MODULE DOES
──────────────────────────────────────────────────────────────────────────────

Provides rate-bound primitives.  The deliberate finding is that Sgr A*
alone cannot falsify B3 at any physically-motivated rate; this is
recorded as a quantitative result rather than treated as a failure.
"""

import math

from sigma_ground.field.constants import H0, XI, YEAR_S


# Published observational parameters.
# Source: Abuter et al. (GRAVITY Collaboration) 2020, A&A 636:L5.
SGR_A_STAR_MASS_SOLAR = 4.152e6  # M_sun, GRAVITY best-fit
SGR_A_STAR_MASS_FRACTIONAL_STABILITY = 0.003  # ~0.3 % per decade drift bound
GRAVITY_OBSERVATION_YEARS = 30.0  # S-star orbit monitoring since ~1995
# Population-level stellar-mass BH-XRB baseline (rough).
# Source: Remillard & McClintock 2006 catalog + updates; Corral-Santana 2016
XRAY_BINARY_BH_OBJECT_YEARS = 600.0  # ~20 objects × ~30-year histories


# Hubble time in years.
TAU_HUBBLE_YEARS = (1.0 / H0) / YEAR_S  # ≈ 1.44e10 yr
BASELINE_B3_RATE_PER_YEAR = 1.0 / TAU_HUBBLE_YEARS  # ≈ 6.96e−11 /yr/BH


def rate_bound_null_observation(
    object_years,
    confidence_level=0.90,
):
    """90 % CL upper bound on rate R from zero-event observation.

    Poisson: P(k = 0 | λ) = exp(−λ).  For 90 % CL upper bound,
    λ_max = −ln(1 − CL) = −ln(0.10) = 2.303.
    So R_max = λ_max / (object_years).

    Args:
        object_years:    integrated BH-years of observation
        confidence_level: CL (default 0.90)

    Returns:
        R_max in events per BH per year
    """
    if object_years <= 0:
        raise ValueError("object_years must be > 0")
    if not (0.0 < confidence_level < 1.0):
        raise ValueError("confidence_level must be in (0, 1)")
    lambda_max = -math.log(1.0 - confidence_level)
    return lambda_max / object_years


def sgr_a_star_rate_bound(
    observation_years=GRAVITY_OBSERVATION_YEARS,
    confidence_level=0.90,
):
    """B3 rate upper bound from Sgr A* alone.

    Uses the conservative "no single conversion event observed in the
    GRAVITY monitoring window" criterion.  A 15.8 % mass drop would be
    unambiguously detectable given 0.3 %-per-decade stability.
    """
    return rate_bound_null_observation(
        object_years=observation_years,
        confidence_level=confidence_level,
    )


def xray_binary_population_rate_bound(
    object_years=XRAY_BINARY_BH_OBJECT_YEARS,
    confidence_level=0.90,
):
    """B3 rate upper bound from combined stellar-mass BH-XRB population.

    Tighter than Sgr A* alone because it integrates over many objects.
    """
    return rate_bound_null_observation(
        object_years=object_years,
        confidence_level=confidence_level,
    )


def baseline_rate_test(rate_bound_per_year, baseline=BASELINE_B3_RATE_PER_YEAR):
    """Does this observational rate bound actually test B3 at baseline R?

    Args:
        rate_bound_per_year:  observational upper bound on R
        baseline:             physical R under test (default 1/τ_Hubble)

    Returns:
        dict with keys:
            rate_bound      — echoed
            baseline        — R_base
            ratio           — rate_bound / baseline (how many × too weak)
            sensitive       — bool, True if rate_bound ≤ baseline
    """
    ratio = rate_bound_per_year / baseline
    return {
        'rate_bound_per_year': rate_bound_per_year,
        'baseline_rate_per_year': baseline,
        'ratio': ratio,
        'sensitive': rate_bound_per_year <= baseline,
    }


def minimum_object_years_to_test_baseline(
    baseline=BASELINE_B3_RATE_PER_YEAR,
    confidence_level=0.90,
):
    """Object-years needed to reach baseline B3 sensitivity.

    Inverts rate_bound_null_observation: solve R_max = baseline
    for object_years.
    """
    lambda_max = -math.log(1.0 - confidence_level)
    return lambda_max / baseline


def summary():
    """Compact summary bundle for reports / verdict doc."""
    sgr = sgr_a_star_rate_bound()
    xrb = xray_binary_population_rate_bound()
    return {
        'sgr_a_star': baseline_rate_test(sgr),
        'xrb_population': baseline_rate_test(xrb),
        'baseline_rate_per_year': BASELINE_B3_RATE_PER_YEAR,
        'tau_Hubble_years': TAU_HUBBLE_YEARS,
        'minimum_object_years_for_baseline_test':
            minimum_object_years_to_test_baseline(),
        'xi_conversion_fraction': XI,
    }
