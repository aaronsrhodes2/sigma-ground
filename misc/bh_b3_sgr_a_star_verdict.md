# Phase H.5 verdict: Sgr A\* astrometry vs Hypothesis B3

**Date:** 2026-04-17
**Phase:** H.5 — rate bound on B3 (rare spontaneous conversion)
**Module:** `sigma_ground/field/interface/sgr_a_star_rate.py`
**Tests:** `sigma_ground/field/interface/test_sgr_a_star_rate.py` — 19/19 green.
**Hypothesis under test:** B3 from `misc/bh_conversion_mass_hypothesis.md`.

## Variable glossary (name[symbol])

| Name | Symbol | Meaning |
|------|--------|---------|
| B3 conversion rate | R | events per BH per year |
| baseline B3 rate | R_base | 1/τ_Hubble ≈ 6.9×10⁻¹¹ /yr/BH — Aaron's default |
| Hubble time | τ_Hubble | 1.45×10¹⁰ yr from Planck 2018 H₀ = 67.4 km/s/Mpc |
| GRAVITY observation window | τ_obs | ~30 yr S-star orbit monitoring |
| integrated BH-years | N_obj·τ_obs | total astrometric exposure across all monitored BHs |
| 90 % CL null-observation bound | R_max | 2.303 / (N_obj·τ_obs) |

## Verdict: **B3 untested — Sgr A\* sensitivity is ~10⁹× too weak for the baseline rate**

This is a **null result that is itself informative**.  The conclusion is
not "B3 is falsified" nor "B3 is supported" — it is that **no current
astrometric data can reach the physically-motivated B3 rate**, and we
quantify by how much.

### Numerical result

| Observable | R_max (90 % CL) | Ratio to R_base | Sensitive? |
|------------|-----------------|------------------|------------|
| Sgr A\* alone (30 yr, 1 object) | **7.68×10⁻² /yr/BH** | 1.11×10⁹ × too weak | **No** |
| XRB population (~600 object-yr) | **3.84×10⁻³ /yr/BH** | 5.57×10⁷ × too weak | **No** |
| Required for baseline test | — | — | **3.34×10¹⁰ object-years** |

At the baseline rate R_base = 1/τ_Hubble, the expected number of
conversion events across 30 years of Sgr A\* monitoring is
**2.07×10⁻⁹** — effectively zero, so a null result from GRAVITY is
trivially consistent with B3.  Sgr A\* cannot kill B3 at any plausible
rate; it can only bound *enhanced-rate* scenarios where BHs convert
~10⁹× more often than baseline.

### Derivation

Poisson rate test with zero detections in observation window:

    λ = R · N_obj · τ_obs      (expected events)
    P(k=0 | λ) = exp(−λ)
    λ_max at 90 % CL = −ln(0.10) = 2.303
    ⇒ R_max = 2.303 / (N_obj · τ_obs)

Inverted for sensitivity threshold: to reach R_max = R_base,

    N_obj · τ_obs = 2.303 / R_base ≈ 3.3×10¹⁰ object-years

For comparison: the Milky Way has ~10⁸ stellar-mass BHs; even monitoring
every single one at 30-year precision would give ~3×10⁹ object-years —
still an order of magnitude short.  The only way to reach baseline
sensitivity is either (a) ~10¹⁰ years of monitoring (impossible), or
(b) direct GW detection of the conversion event itself.

## What this means for B3 and the wider B matrix

| Hypothesis | Post-H.5 status |
|------------|------------------|
| **A** (mass conservation) | Consistent with all data |
| **B1** (merger-triggered) | Dead at 6.3σ (Phase H.3) |
| **B2** (critical-mass threshold) | Squeezed to [135, 200+] M☉, untestable / degenerate with PI gap (Phase H.4) |
| **B3** (rare spontaneous) | **Untested and not testable with current astrometry** |
| **B4** (continuous leak) | Indistinguishable from A |

B3 is *not falsified*, but the current observational regime cannot
reach its baseline rate.  This is a **~9-orders-of-magnitude
sensitivity gap**, not a marginal 2σ question.  Future prospects:

1. **Direct GW detection of a conversion event.**  If a 15.8 % mass
   drop happens in a well-monitored SMBH, it should emit a
   characteristic GW burst (asymmetric mass redistribution).
   Searching for such bursts in LIGO/Virgo/KAGRA archives would probe
   rates R ≳ (population-weighted detectable-volume rate).  This is
   the only path to reaching R_base.
2. **LISA SMBH background.**  LISA (2035+) will monitor ~10⁴ SMBHs
   continuously over years.  If conversion events produce coherent
   low-frequency GW signatures, LISA could reach R ~ 10⁻⁶ /yr/BH —
   still ~10⁵ × too weak for baseline but 10⁴ × better than current.
3. **Accept B3 as a "hidden" variant.**  If no detection channel
   reaches R_base, B3 remains a theoretically-permitted but
   empirically-silent hypothesis.  This is a valid outcome: the
   universal theory (RODM + nested universes under A) stands as the
   default; B3 is a historical curiosity.

## Caveats

1. **Mass-dependence of the rate is not considered.**  We assumed
   R is BH-mass-independent.  If R ∝ M^n for some n > 0, SMBHs like
   Sgr A\* (4×10⁶ M☉) convert ~(4×10⁵)^n times faster than stellar-
   mass BHs, which *could* close part of the gap.  For n = 2, Sgr A\*
   rate enhancement is ~10¹¹ — that would actually exceed current
   bounds and constrain such scalings.  This is a useful sub-test
   if the user wants to explore rate-scaling models.
2. **Poisson approximation.**  We treat conversion events as
   independent Poisson arrivals.  If conversion is triggered by
   specific BH-internal states (e.g., accumulated accreted mass
   exceeds the central density threshold), the process may be
   non-Poissonian; our bound is a reasonable first approximation.
3. **XRB object-year count is rough.**  600 object-years is a ballpark
   from ~20 known BH-XRBs × ~30-year mass-stability histories.  A
   careful census (Corral-Santana 2016 BlackCAT catalog + follow-ups)
   might push this to ~10³ object-years — still nowhere near the
   3×10¹⁰ needed.
4. **0.3 %-per-decade stability.**  GRAVITY's mass-precision number is
   for the best-fit Sgr A\* mass from S-star orbits, not for a drift
   search.  A true drift search might tighten or loosen this by factors
   of 2–3; the order-of-magnitude conclusion is robust.

## Cross-references

- Hypothesis formalisation: `misc/bh_conversion_mass_hypothesis.md`
- Phase H.3 (B1 dead): `misc/bh_imr_verdict.md`
- Phase H.4 (B2 squeezed): `misc/bh_mass_function_verdict.md`
- Phase H.1 (echo predictions, A-baseline): `misc/bh_merger_predictions.md`
- Phase G (γ-mode verdict): `misc/duality_ellipse_verdict.md`

## Files

- **New:** `sigma_ground/field/interface/sgr_a_star_rate.py` — 5 primitives + constants
- **New:** `sigma_ground/field/interface/test_sgr_a_star_rate.py` — 19 tests (all pass)
- **New:** `misc/bh_b3_sgr_a_star_verdict.md` — this file

## Test evidence

```
Phase H.5 validation:
  sigma_ground/field/interface/test_sgr_a_star_rate.py ... 19 passed in 0.06s
  Sgr A* rate bound (90% CL):    7.68e-02 /yr/BH
  XRB population bound (90% CL): 3.84e-03 /yr/BH
  Baseline B3 rate:              6.89e-11 /yr/BH
  Sensitivity gap:               ~10^9 (Sgr A*), ~10^8 (XRB pop)
  Object-years required:         3.34e+10
```
