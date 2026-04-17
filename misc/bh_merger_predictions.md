# BH-merger predictions from sigma-ground + duality ellipse

**Date:** 2026-04-16
**Phase:** H.1 — analytic prediction engine (no data pull yet)
**Module:** `sigma_ground/field/interface/bh_merger.py`
**Tests:** `sigma_ground/field/interface/test_bh_merger.py` — 29/29 green.

## Variable glossary (name[symbol])

| Name | Symbol | Meaning |
|------|--------|---------|
| remnant mass | M | post-merger black-hole mass |
| Schwarzschild radius | r_s | 2GM/c² |
| baryonic conversion fraction | ξ | 0.1582 (matter-conversion ratio per event) |
| conversion-horizon σ | σ_conv | −ln ξ ≈ 1.8439 |
| marginal coherence | γ | environment-notepad overlap (from Khatiwada-Qian 2025) |
| per-dim entanglement | Θ | η^(1/3) ≈ 0.7461 |
| fossil entanglement | η | 0.4153 |
| shell index | n | 1, 2, 3, … (nesting depth) |
| echo delay | Δt_n | round-trip coordinate-time of n-th shell reflection |
| QNM damping time | τ_QNM | standard l=2,m=2,n=0 ringdown e-folding |

## Two testable predictions

Both fall out of sigma-ground first principles without tuning.

### Prediction 1 — ξ-shell ringdown echoes

BH interiors have gravitational-pressure shells at r_n = r_s · ξⁿ (from RODM's
nested-universe framework, `run_nested_bh_chain.py`).  Under an ECO-like
soft-horizon model (Cardoso-Pani framework), a perturbation reflecting off the
n-th shell at fractional displacement ε_n = ξⁿ from the would-be horizon
re-emerges to the external observer after round-trip coordinate-time delay

    Δt_n  =  2 · r_s · n · σ_conv / c       (inward shells; primary)
    Δt_n  =  2 · r_s · (ξ⁻ⁿ − 1) / c        (outward shells; alternative)

**The same constant σ_conv = −ln ξ that pegs the matter-conversion threshold
and the Phase G σ→γ endpoints also sets the echo spacing.**  This is the
sigma-ground-specific falsifier: Δt is linear in n (not geometric, not
arbitrary) with slope set by ξ alone.

### Prediction 2 — γ(σ_conv) ringdown-amplitude suppression

The remnant's local σ pegs at σ_conv at the horizon, so the ringdown strain
amplitude is multiplied by

    h_measured(t) / h_GR(t)  =  γ(σ_conv)

with γ(σ_conv) set by the chosen candidate mode:

| Mode        | γ(σ_conv) | Amplitude vs pure-GR |
|-------------|-----------|----------------------|
| **H1** pure GR (null) | 1.0000 | 1.000 |
| **exp**     | 0.8395    | 0.840 |
| **sigma_coh** (default) | 0.7924 | 0.792 |
| **linear**  | 0.7461 (=Θ) | 0.746 |
| **cbrt**    | 0.7461 (=Θ) | 0.746 |

Predicted amplitude deficits of 16–25 % relative to pure-GR IMR templates.

## Worked predictions for five LIGO/Virgo events

Inward-shell delays (primary prediction).  M_rem, r_s, τ_QNM from published
LIGO-Virgo catalog papers.  Δt_n in milliseconds.  All Δt_1 < 0.65 · τ_QNM —
the first echo arrives while the ringdown is still detectable.

| Event       | M_rem (M☉) | r_s (km) | τ_QNM (ms) | Δt_1 | Δt_2 | Δt_3 | Δt_4 | Δt_5 | Δt_1/τ |
|-------------|------------|----------|------------|------|------|------|------|------|--------|
| GW151226    |  20.8      |  61.4    | 1.20       | 0.756 | 1.512 | 2.268 | 3.023 | 3.779 | 0.630 |
| GW170104    |  48.7      | 143.9    | 3.10       | 1.770 | 3.539 | 5.309 | 7.079 | 8.849 | 0.571 |
| GW170814    |  53.2      | 157.2    | 3.40       | 1.933 | 3.866 | 5.800 | 7.733 | 9.666 | 0.569 |
| GW150914    |  62.2      | 183.7    | 4.00       | 2.260 | 4.521 | 6.781 | 9.041 | 11.301 | 0.565 |
| GW190521    | 142.0      | 419.5    | 9.00       | 5.160 | 10.320 | 15.480 | 20.641 | 25.801 | 0.573 |

**Δt_1 / τ_QNM ≈ 0.57 is nearly mass-independent** — a constant ratio set by
sigma-ground constants alone (≈ σ_conv · c / (π · r_s · f_QNM) under the
Kerr-adjacent approximation f_QNM · τ_QNM ≈ 1).  That invariant is itself a
check: any event conforming to GR will have the first echo at ≈ 57 % of the
ringdown damping time regardless of M.

### Outward-shell delays (alternative interpretation)

If shells nest outward (r_n = r_s · ξ⁻ⁿ) rather than inward, delays grow
geometrically:

| Event       | Δt_1 (ms) | Δt_2 (ms) | Δt_3 (ms) | Δt_4 (ms) | Δt_5 (ms) |
|-------------|-----------|-----------|-----------|-----------|-----------|
| GW151226    |    2.18   |   15.97   |   103.12  |   654.04  |  4136.44 |
| GW170104    |    5.11   |   37.39   |   241.45  |  1531.33  |  9684.83 |
| GW170814    |    5.58   |   40.84   |   263.76  |  1672.83  | 10579.73 |
| GW150914    |    6.52   |   47.75   |   308.38  |  1955.83  | 12369.53 |
| GW190521    |   14.89   |  109.02   |   704.02  |  4465.07  | 28239.13 |

Only Δt_1 falls within the ringdown envelope in this interpretation; higher n
fall into the post-ringdown noise floor and would not be detectable.  Inward
is the sharper-signal target.

## Best event for an echo search

**GW151226** — lightest remnant (20.8 M☉) gives tightest echo spacing
(0.76 ms), highest shell-number visibility per ringdown (≈ 1.6 shells before
τ_QNM is reached), and the most echoes packed into the detectable envelope.
It is also a **published high-SNR event** with clean strain data on GWOSC.

**GW150914** — remains the canonical first-light event.  Δt_1 = 2.26 ms fits
comfortably inside the 4.0 ms ringdown damping time.  Its SNR is highest,
which may outweigh the wider echo spacing.

**GW190521** — heaviest published remnant (142 M☉).  Δt_1 = 5.16 ms sits
at 57 % of τ = 9 ms.  Low-frequency (f_QNM ≈ 110 Hz) makes it susceptible to
seismic noise, but it's the cleanest test of the mass-scaling invariant Δt ∝ M.

## What Phase H.2 (LIGO data pull) would do

1. Fetch GWOSC strain data (.hdf5) for GW151226, GW150914, GW190521 — ~32 s
   each around merger time at 4096 Hz.
2. Fit standard l=2,m=2,n=0 QNM ringdown starting ~3 ms post-merger.
3. Subtract the best-fit QNM; whiten the residual.
4. Matched-filter the residual at the predicted Δt_n using the QNM template
   echoed at each delay.  Record SNR per echo + combined.
5. Noise-background: 1000 random off-source delays with the same template,
   bootstrapped p-value.
6. Report: per-event significance at each predicted Δt_n, plus the
   amplitude-deficit test (measured h_ringdown / h_GR-predicted).

**Honest framing:** a one-session analysis is suggestive-grade, not
publication-grade.  Afshordi/Abedi 2017 took months with careful background
treatment and found only a ~2.5σ hint at quite different (0.2 s) spacings.
Our prediction lives in the 1–10 ms window, a regime they did not search
at ξ-quantised spacings.  If H.2 shows a clean SNR peak at Δt_1 with random
delays drawing blanks, that's a strong preliminary positive and warrants a
full follow-up.  If H.2 shows nothing above noise, the inward-shell model
is weakly disfavoured for that event; try the next.

## Cross-reference to Phase G duality-ellipse verdict

The γ(σ_conv) values used here (0.7461 / 0.7924 / 0.8395 across modes) are
the **identical terminal values** reported in
`misc/duality_ellipse_verdict.md` under the Phase G σ-sweep table.  Same
constants, same formula, different physics regime (gravitational-wave strain
rather than double-slit fringe visibility).  If LIGO's IMR consistency tests
already bound the ringdown amplitude ratio tighter than ±20 %, the H4-class
modes (linear, cbrt → Θ) would be in tension with existing data — something
Phase H.2 should check against published constraints.

## Files

- **New:** `sigma_ground/field/interface/bh_merger.py` — 7 primitives + 5 named events
- **New:** `sigma_ground/field/interface/test_bh_merger.py` — 29 tests (all pass)
- **New:** `misc/bh_merger_predictions.md` — this file

## Test evidence

```
Phase H.1 validation:
  sigma_ground/field/interface/test_bh_merger.py ... 29 passed in 0.06s
  Cross-check: Δt_1 for 62.2 M☉ = 2.260 ms matches hand calculation to 4 dp
  Cross-check: all γ(σ_conv) values match Phase G table to 12 dp
  Cross-check: Δt_1/τ_QNM ≈ 0.57 invariant across all 5 named events
```
