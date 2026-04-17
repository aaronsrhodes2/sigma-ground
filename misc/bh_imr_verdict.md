# Phase H.3 verdict: LIGO IMR consistency vs Hypothesis B1

**Date:** 2026-04-16
**Phase:** H.3 — LIGO IMR consistency pull against B1 (merger-triggered mass shedding)
**Module:** `sigma_ground/field/interface/imr_consistency.py`
**Tests:** `sigma_ground/field/interface/test_imr_consistency.py` — 19/19 green.
**Hypothesis under test:** B1 from `misc/bh_conversion_mass_hypothesis.md`.

## Variable glossary (name[symbol])

| Name | Symbol | Meaning |
|------|--------|---------|
| fractional IMR deviation | ε_M | 2·(M_f_insp − M_f_MR)/(M_f_insp + M_f_MR); LIGO's standard IMR consistency parameter |
| inspiral-predicted final mass | M_f_insp | remnant mass computed from pre-merger parameters via NR fits |
| ringdown-measured final mass | M_f_MR | remnant mass fit from merger-ringdown signal alone |
| radiated-energy fraction | f_rad | E_rad/M_tot·c²; nominal 0.05 for stellar-mass BH mergers |
| conversion fraction | ξ | 0.1582 (baryonic conversion ratio per event) |
| catalog-combined posterior | COMBINED_GWTC3 | GWTC-3 TGR aggregate over ~15 high-quality events |

## Verdict: **B1 falsified at ≥ 6σ by published LIGO data**

### Gate signals

| Signal | Result | Pass? |
|--------|--------|-------|
| B1 prediction derivable | ε_M_B1 = 2·ξ / (2(1 − f_rad) − ξ) = **0.1817** at f_rad = 0.05 | **PASS** |
| Individual event exclusion (GW150914) | 3.65σ | **PASS (> 2σ)** |
| Combined catalog exclusion (GWTC-3) | 6.31σ | **PASS (> 4σ)** |
| Regression: full pytest | 81 passed across duality/bh_merger/imr | **PASS** |

**B1 (merger-triggered conversion with 15.82 % external mass drop) is
inconsistent with already-published LIGO data at the > 6σ level on the
combined catalog.** No new data pull was required; the test is a
prediction-vs-published-bound comparison.

## Derivation (one line)

If the ringdown sees a remnant lighter than the inspiral-predicted
remnant by ξ·M_tot (because the converted fraction has gravitationally
decoupled), then

    ε_M_B1 = 2·ξ / (2·(1 − f_rad) − ξ)

For f_rad ∈ [0.03, 0.10] this gives ε_M_B1 ∈ [0.179, 0.193] — B1's
prediction is essentially flat across the plausible radiation-fraction
range, so event-by-event f_rad variation does not rescue it.

## Per-event exclusion table

At nominal f_rad = 0.05, B1 predicts ε_M = **0.1817**.  σ computed from
the upper tail of the published 90 % CL interval (Gaussian-approx; 90 %
CL = 1.645·σ).

| Event | Published ε_M (median, 90% CL upper) | σ excluded | Source |
|-------|--------------------------------------|------------|--------|
| **COMBINED_GWTC3** | **−0.01, +0.04** | **6.31σ** | Abbott 2021 arXiv:2112.06861 |
| GW150914  | −0.04, +0.06 | 3.65σ | Abbott 2016 PRL 116:221101 |
| GW170814  | −0.03, +0.09 | 2.90σ | Abbott 2019 PRD 100:104036 |
| GW170104  | +0.02, +0.21 | 1.40σ | Abbott 2017 PRL 118:221101 |
| GW190521  | −0.08, +0.32 | 1.08σ | Abbott 2021 arXiv:2112.06861 |
| GW151226  | +0.09, +0.38 | 0.52σ | Abbott 2016 PRX 6:041015 |

GW150914 alone puts B1 in 3.65σ tension.  GW170814 adds another 2.90σ.
GW151226, GW170104, and GW190521 have wide posteriors (high masses,
short signals, or borderline SNR) and individually cannot exclude B1 —
but they also contribute positively to the combined posterior, which
drives exclusion to 6.31σ.

## What survives (from the A-vs-B menu)

From `misc/bh_conversion_mass_hypothesis.md` there were five hypotheses:

| Hypothesis | Phase H.3 status |
|------------|------------------|
| **A** — mass conservation (RODM default) | **Consistent with data.**  Predicts ε_M = 0; published medians near zero. |
| **B1** — merger-triggered shedding | **Falsified at ≥ 6σ.**  Dead. |
| **B2** — critical-mass threshold shedding | Not addressed by this test (trigger is not merger). |
| **B3** — rare spontaneous shedding | Not addressed (rate too low for merger sample). |
| **B4** — continuous slow leak | Not addressed (indistinguishable from A in IMR). |

**Net:** Aaron's stronger alternative (observable mass drops per merger)
is cleanly ruled out.  The softer variants (B2, B3, B4) remain live and
need different observational tests — population-level mass function,
Sgr A* astrometry, and stochastic GW background, respectively.

## Caveats and limitations

1. **Published summaries, not re-derived posteriors.**  We did not
   re-run parameter estimation with a conversion-drop prior.  A proper
   Bayesian analysis would compute a log-evidence ratio ln(Z_B1/Z_A)
   rather than a Gaussian z-score.  The 6σ is suggestive-grade; a
   publication-grade result would be 2–3σ tighter or looser depending
   on the prior, but the sign and order-of-magnitude are robust.
2. **Systematic errors on NR fitting formulas.**  The IMR consistency
   test assumes the inspiral→final-state mapping from NR simulations is
   exact.  Published analyses bound the NR systematic at ≪ 1 %, much
   smaller than B1's 18 % prediction, so this does not rescue B1.
3. **Asymmetric posteriors handled by upper-tail half-width.**  We used
   the Gaussian-approximation 1.645-factor to translate 90 % CL → 1σ.
   The published posteriors are often non-Gaussian at the tails; a
   more careful analysis would integrate the actual posterior tail
   above ε_M = 0.18.
4. **GW190521-class mergers.**  Intermediate-mass merger signals are
   short and have wide posteriors; these do not individually constrain
   B1, but they do not rescue it either.

## Recommended Phase H.4

**Option 1 (next test against B2):** BH mass-function cutoff analysis.
Pull GWTC-3 mass posteriors, fit dN/dM, check for a sharp cutoff not
explained by pair-instability.  B2 predicts a distinctive cliff at
M_crit.

**Option 2 (next test against B3):** Sgr A\* astrometric constraint.
GRAVITY has bounds on M_SgrA drift at ~0.3 % per decade.  Translate
into a bound on R_conv (events per BH per unit time) for rare
spontaneous conversion.

**Option 3 (sharpen Phase H.1):**  Return to the Phase H.1 ringdown
echo search, now with B1 removed from contention.  The echoes in
`bh_merger_predictions.md` were derived under A (r_s of full M); that
derivation is now the uncontested choice rather than a coin flip with B1.

**Preferred:** Option 1.  Mass-function analysis is population-level
(more statistical power than single-object astrometry) and uses the
same GWTC-3 data already being worked with.

## Cross-references

- Hypothesis formalisation: `misc/bh_conversion_mass_hypothesis.md`
- Phase H.1 predictions: `misc/bh_merger_predictions.md`
- Phase G verdict: `misc/duality_ellipse_verdict.md`

## Files

- **New:** `sigma_ground/field/interface/imr_consistency.py` — 4 primitives + published bounds dict
- **New:** `sigma_ground/field/interface/test_imr_consistency.py` — 19 tests (all pass)
- **New:** `misc/bh_imr_verdict.md` — this file

## Test evidence

```
Phase H.3 validation:
  sigma_ground/field/interface/test_imr_consistency.py ... 19 passed in 0.08s
  Combined regression (duality + bh_merger + imr)    ... 81 passed in 0.14s
  B1 prediction: ε_M = 0.1817 at f_rad = 0.05
  Combined catalog exclusion: 6.31σ (Gaussian approximation)
  GW150914 individual exclusion: 3.65σ
```
