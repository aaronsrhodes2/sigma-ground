# Phase H.4 verdict: BH mass-function constraints on Hypothesis B2

**Date:** 2026-04-16
**Phase:** H.4 — BH mass-function cutoff analysis for B2 (critical-mass threshold)
**Module:** `sigma_ground/field/interface/bh_mass_function.py`
**Tests:** `sigma_ground/field/interface/test_bh_mass_function.py` — 24/24 green.
**Hypothesis under test:** B2 from `misc/bh_conversion_mass_hypothesis.md`.

## Variable glossary (name[symbol])

| Name | Symbol | Meaning |
|------|--------|---------|
| critical mass threshold | M_crit | mass above which B2 triggers conversion |
| primary BH mass | m_1 | heavier pre-merger BH mass |
| remnant mass | M_f | post-merger BH mass |
| pair-instability gap | [M_PI_low, M_PI_high] | ~45–135 M_sun; stellar evolution predicts no direct BH formation here |
| landing zone | (1 − ξ)·M_crit | BH mass immediately after B2 conversion event |

## Verdict: **B2 is not falsified, but viable only in a degenerate zone**

Three mutually-consistent constraints:

1. **Primary-mass floor (hard):** M_crit ≥ 71 M_sun at 90 % CL, from
   the existence of GW190521's primary BH at m_1 = 85 +21/-14 M_sun.
   A BH cannot exist at 71 M_sun if the B2 threshold is below that —
   it would have converted en route.
2. **IMR-triggering ceiling (soft):** For M_crit ≲ 142 M_sun, at least
   one named LIGO event (GW190521, M_f = 142 M_sun) would have
   triggered conversion at the merger and should show the
   ε_M ≈ 0.18 signature.  IMR data for GW190521 alone is 1.08σ from B1 —
   too weak to exclude.
3. **Pair-instability degeneracy (serious):** M_crit ∈ [71, 135] M_sun
   lies inside the conventional pair-instability mass gap.  An observed
   "cutoff" in this range is equally well explained by
   pulsational-pair-instability supernovae shutting off direct stellar
   BH formation — B2 and PI are mutually degenerate on population data
   alone.

| M_crit range (M_sun) | Viable? | Notes |
|----------------------|---------|-------|
| < 71 | **RULED OUT** | violates GW190521 primary-mass 90 % CL lower bound |
| 71 – 135 | viable but degenerate | hides inside pair-instability gap; can't be distinguished from PI cutoff |
| 135 – 142 | viable, weak IMR pressure | GW190521 remnant exceeds; 1.08σ tension only |
| > 142 | viable, untested | above current LIGO sensitivity horizon for mass-function cutoff |

## M_crit sweep (numeric output)

Generated via `sweep_M_crit()` with IMR exclusion threshold = 3σ:

| M_crit | Primary-mass OK? | IMR σ | Events triggered | In PI gap? | Viable? |
|--------|-----------------|-------|------------------|------------|---------|
|  20.0  | N | 5.01 | all 5 | N | N |
|  30.0  | N | 4.98 | 4 (no GW151226) | N | N |
|  45.0  | N | 4.98 | 4 | **Y** | N |
|  55.0  | N | 3.80 | GW150914 + GW190521 | **Y** | N |
|  65.0  | N | 1.08 | GW190521 | **Y** | N |
|  75.0  | Y | 1.08 | GW190521 | **Y** | Y (but degenerate) |
|  85.0  | Y | 1.08 | GW190521 | **Y** | Y (but degenerate) |
| 100.0  | Y | 1.08 | GW190521 | **Y** | Y (but degenerate) |
| 135.0  | Y | 1.08 | GW190521 | **Y** | Y (boundary of PI gap) |
| 150.0  | Y | 0.00 | (none) | N | **Y (clean)** |
| 175.0  | Y | 0.00 | (none) | N | **Y (clean)** |
| 200.0  | Y | 0.00 | (none) | N | **Y (clean)** |
| 250.0  | Y | 0.00 | (none) | N | **Y (clean)** |

## Structural signatures B2 predicts (if real)

For any given M_crit, B2 predicts three features in dN/dM:

| Feature | Location | Magnitude |
|---------|----------|-----------|
| Hard cutoff | M_crit | dN/dM → 0 above |
| Pile-up (landing zone) | (1 − ξ)·M_crit = 0.8418·M_crit | factor ~1/ξ·(flux into threshold) over smooth continuation |
| Depletion gap | (0.8418·M_crit, M_crit) | BHs transit but don't pile up |

**None of these features are observed cleanly in GWTC-3 dN/dM.**  The
best-fit population models (Power-Law-Plus-Peak, Flexible Mixtures,
Truncated Power Law) show a smooth primary-mass distribution extending
to ~80 M_sun with no sharp cutoff and no distinct pile-up.  This is
additional (though weaker) pressure against B2 at any
detector-sensitive M_crit.

## What each Hypothesis B variant now looks like after H.3 + H.4

| Hypothesis | Post-H.3 status | Post-H.4 status |
|------------|-----------------|------------------|
| **A** (mass conservation) | Consistent | Consistent |
| **B1** (merger-triggered, universal) | **Falsified (6.3σ)** | — |
| **B2** (critical-mass threshold) | Not addressed | Viable only at M_crit ≳ 135 M_sun, otherwise degenerate with pair-instability |
| **B3** (rare spontaneous) | Not addressed | Not addressed |
| **B4** (continuous slow leak) | Not addressed | Not addressed |

## Net result

B2 is **not killed** but is squeezed.  Its remaining viable window is:

- **[135, ~200] M_sun** (M_crit above pair-instability gap, below
  current detector horizon) — the "clean" survival zone; B2 predicts
  a dN/dM cutoff in a regime where LIGO has only a handful of events
  and no clean mass-function measurement.  Future O4 / O5 observations
  should tighten this significantly.

- **[71, 135] M_sun** (M_crit inside pair-instability gap) — the
  "degenerate" survival zone; any observed cutoff in this range can
  be attributed to either B2 or pair-instability and the data cannot
  distinguish.  Only event-by-event IMR consistency of individual
  GW190521-class mergers could break the degeneracy, and current
  single-event σ for GW190521 is only 1.08.

## Caveats

1. **σ combination is quadrature, not Bayesian.**  We combine
   individual event σ's as √(Σσᵢ²).  A proper analysis would compute
   joint likelihoods with the correlated posteriors; this is a
   first-order approximation.
2. **Primary-mass lower bound uses 90 % CL from a single waveform
   analysis.**  Different waveform families (NRSur7dq4, SEOBNRv4PHM,
   IMRPhenomXPHM) give slightly different posteriors for GW190521.
   Our 71 M_sun floor is representative but not family-independent.
3. **"Pair-instability gap" has its own uncertainty.**  Theoretical
   predictions for M_PI_low and M_PI_high depend on stellar-metallicity,
   convective overshoot, and nuclear reaction rate assumptions.
   Farmer et al. 2019 give [45, 135] M_sun; other works push the
   upper edge as high as 160 M_sun.  The "degenerate zone" boundary is
   therefore fuzzy.
4. **B2 conversion rate within the viable window is unconstrained.**
   If B2's conversion rate is much less than one-per-merger even for
   M > M_crit, its IMR signature is weaker than B1's 0.18 and our
   combined exclusion weakens proportionally.  We assumed B2 at the
   100 %-trigger limit; softer rates are even less constrained.

## Recommended Phase H.5

Two natural continuations, different target hypotheses:

**Option 1 — B3 test via Sgr A\* astrometry.**
Pull GRAVITY collaboration bounds on M_SgrA drift over the last decade
(~0.3 % precision).  Translate into an upper bound on R_conv for
spontaneous B3 conversion events.  Single well-monitored object, so
constrains only R_conv·τ_obs — needs additional objects to sum up
population rates.

**Option 2 — Sharpen B2 survival zone with future LIGO observing runs.**
Write a forward-looking doc predicting what O4 (2023–2025, now
completed) and O5 (scheduled 2027–2030) should see if B2 with M_crit
∈ [135, 200] M_sun is real.  Identify which events would be
discriminating.  Cheap doc-only work, no data pull.

**Option 3 — Return to Phase H.1 echo predictions.**
With B1 dead and B2 constrained, revise the Phase H.1 echo search
plan.  The ξ-shell echo signatures remain B-hypothesis-agnostic (they
come from Hypothesis A's ringdown structure); so the Phase H.1 plan
stands but can be prioritised more confidently.

**Preferred:** Option 1.  B3 is now the only B variant that has *not*
been tested by Phase H.3 or H.4; closing that gap completes the
B-hypothesis matrix.  Sgr A\* data is public and well-documented.

## Cross-references

- Hypothesis formalisation: `misc/bh_conversion_mass_hypothesis.md`
- Phase H.3 verdict (B1 falsification): `misc/bh_imr_verdict.md`
- Phase H.1 echo predictions: `misc/bh_merger_predictions.md`
- Phase G γ(σ) verdict: `misc/duality_ellipse_verdict.md`

## Files

- **New:** `sigma_ground/field/interface/bh_mass_function.py` — 6 primitives + GWTC-3 primary-mass bounds
- **New:** `sigma_ground/field/interface/test_bh_mass_function.py` — 24 tests (all pass)
- **New:** `misc/bh_mass_function_verdict.md` — this file

## Test evidence

```
Phase H.4 validation:
  sigma_ground/field/interface/test_bh_mass_function.py ... 24 passed in 0.28s
  Full regression (duality + bh_merger + imr + mass_function) ... 105 passed in 0.27s
  M_crit 90 % CL lower bound: 71 M_sun (from GW190521 primary)
  Pair-instability degeneracy zone: 71–135 M_sun
  Clean B2 survival zone: 135–200+ M_sun (untestable with current detectors)
```
