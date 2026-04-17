# BH conversion-event mass hypothesis: A vs B

**Date:** 2026-04-16
**Phase:** H.2 — hypothesis formalisation (no data pull yet)
**Scope:** Does a BH's internal "conversion event" (the Black-Hole-Nova
mechanism, RODM's posited Big-Bang analogue) leave the parent BH's
external gravitational mass unchanged (A), or does it drop by the
conversion fraction ξ ≈ 0.1582 (B)?
**Status:** open question, formalised for falsification.  No commitment
made here — goal is to pin down what each hypothesis predicts so the
sharpest discriminator can be chosen.

## Variable glossary (name[symbol])

| Name | Symbol | Meaning |
|------|--------|---------|
| parent BH gravitational mass (external) | M_ext | mass as seen by external observer (orbits, lensing, GW inspiral) |
| parent BH total enclosed mass | M_tot | total matter inside horizon |
| conversion fraction | ξ | 0.1582 — fraction converted to baryons in child universe (= Ω_b/(Ω_b+Ω_c) from Planck 2018) |
| conversion rate | R_conv | expected events per BH per unit time |
| conversion trigger | T_conv | the physical precondition (merger / critical mass / spontaneous / continuous) |
| IMR consistency bound | Δ_IMR | LIGO's catalog-wide bound on (M_pre − M_post)/M_pre |
| Sgr A* astrometric bound | Δ_SgrA | GRAVITY/S-star bound on M_SgrA fractional drift per decade |
| BH mass function | dN/dM | population-level black-hole mass distribution from GWTC / X-ray surveys |

## The question in one sentence

When the innermost core of an accumulated BH ignites a Chapman–Jouguet
detonation and births a child universe, does the converted mass
(fraction ξ) stay gravitationally coupled to the parent's exterior
spacetime (**A**), or does it decouple — dropping M_ext by ξ (**B**)?

## Hypothesis A — mass conservation (RODM default)

**Claim:** M_ext is unchanged by conversion events.  The parent BH's
exterior Kerr/Schwarzschild metric remains sourced by the full enclosed
mass-energy, converted plus remnant.

**Derivation from standing RODM commitments:**
- `matter-shaper/theory/outline.md:319` — "Outer population: unconverted
  SSBM → electromagnetically dark, **gravitationally active** → dark
  matter halos"
- `RODM_hypothesis.md:18-20` — "…persists as *remnant matter*:
  **gravitationally active** but inert under Standard Model gauge
  interactions."
- `RODM_hypothesis.md:1157-1158` — "irreversible phase conversion that
  changes the matter content and gauge-compatibility of the entire
  causal domain."  (Gauge structure changes; gravitational coupling
  does not.)
- GR no-hair theorem — external observer sees only (M, J, Q).  Internal
  reshuffling of matter content (SM baryons ↔ SSBM remnant) cannot
  change M_ext unless mass-energy actually leaves the causal envelope.

**External predictions:**
- No mass discontinuity in BH orbits across any rate/trigger scenario.
- Ringdown amplitude modulated by γ(σ_conv) (Phase G / H.1), but mass
  input to ringdown template is still the full M_tot.
- ξ-shell echoes at Δt_n = 2·r_s·n·σ_conv/c (Phase H.1), computed from
  r_s of the full M_ext.

**External signature:** only the sigma-ground-specific Phase H.1
signatures (echoes, γ suppression).  No mass-loss signal.

## Hypothesis B — mass shedding on conversion

**Claim:** At a conversion event, the fraction ξ of parent mass fully
decouples from the parent's exterior spacetime — converted matter now
sources a new universe's metric and no longer appears in the parent's
g_μν.  External observer sees M_ext → M_ext·(1 − ξ), a ~15.8 % drop.

**Operational definition of "gravitational decoupling":**
We take the strongest version — complete g_μν decoupling of the
converted fraction, instantaneous from the external observer's view.
Weaker variants (partial decoupling, gradual fade over τ_decouple) are
strict subsets that reduce the signal amplitude and smear its spectrum;
any null result for strong B therefore bounds all weaker B variants.

B is underdetermined without pinning R_conv and T_conv.  Four sub-variants:

### B1 — merger-triggered

Conversion is ignited by the merger event itself (shock heating of
dense BH interior crosses the phase-transition threshold).

- R_conv: ≈ 1 per BH-BH merger
- Mass drop: ΔM = ξ·M_post ≈ 0.158·M_post, visible at the merger itself
- Predicted signature: systematic **M_post_GR − M_post_measured ≈ ξ·M_post**
  residual in LIGO IMR consistency tests.  GW inspiral fits pre-merger
  M_tot; post-merger ringdown fits M_ext.  Under B1, M_ext = (1−ξ)·M_tot.
- **LIGO status:** catalog-wide Δ_IMR is already bounded to order ≤ 5–10 %
  at 90 % CL for high-SNR events (GWTC-3).  **B1 is disfavoured** at
  the 15.8 % level for merger-triggered conversion, pending careful
  re-analysis.  This is the quickest kill.

### B2 — critical-mass threshold

Conversion ignites when accumulated mass exceeds M_crit (the density
threshold RODM posits for the central core).  Mergers raise M past
M_crit only sometimes; most conversions happen during slow accretion.

- R_conv: BH-dependent, peaks at mass-function cutoff M_crit
- Mass drop: ΔM = ξ·M at the trigger moment
- Predicted signature: sharp **population-level cutoff** in observed BH
  mass function dN/dM at M_crit — BHs simply cannot persist above this
  mass without shedding ξ·M.  After each event BH lands at M·(1−ξ) and
  begins re-accreting toward M_crit again.
- **Status:** testable against GWTC / X-ray mass functions.  Current BH
  mass function extends smoothly to ~150 M☉ (GW190521) with no clean
  cutoff.  Pair-instability gap exists at ~60–120 M☉ for different
  reasons; B2 would need M_crit outside that gap.  Mildly disfavoured
  but not cleanly killed.

### B3 — rare spontaneous

Conversion is a spontaneous quantum/thermal fluctuation that happens
roughly once per Hubble time per BH, uncorrelated with mergers.

- R_conv: ≈ 1 per Hubble time per BH (≈ 7×10⁻¹¹ yr⁻¹)
- Mass drop: ΔM = ξ·M at the trigger moment
- Predicted signature: extremely rare per object; population effect
  detectable only by **statistical mass-function drift** across cosmic
  epochs or **direct orbital discontinuity** in a single well-monitored
  object.
- **Status:** LIGO IMR *cannot* test — merger-triggered signal absent.
  **Sgr A\* 30-year astrometry** is the sharpest constraint — GRAVITY
  fits M_SgrA to ~0.3 % precision per decade.  At R_conv = 1/Hubble,
  expectation is ~3×10⁻⁹ events over 30 years — null result is
  consistent with B3, so Sgr A\* alone cannot kill it.  Need larger
  object population (e.g. all BHs in Milky Way via pulsar timing) to
  integrate up a detectable rate.

### B4 — continuous slow leak

Conversion is continuous at rate ξ/τ_Hubble rather than discrete.  Mass
leaks out gradually rather than in one event.

- R_conv: continuous, fractional dM/dt ≈ −ξ/τ_Hubble per BH
- Mass drop: gradual, ΔM(t)/M ≈ (1 − exp(−t·ξ/τ_Hubble)) → ξ at t = ∞
- Predicted signature: **stochastic GW background** from asymmetric
  slow-conversion emission; secular drift of BH masses on cosmic
  timescales; no single-event signature.
- **Status:** indistinguishable from A for any timescale longer than
  observational window.  Weakest falsifier; deprioritise.

## RODM current commitment — aligned with A

The existing RODM documents are explicit on one side: converted matter
remains "gravitationally active" in the child universe's dark-matter
halo, and unconverted remnant matter is the dark matter component.
However, RODM is *silent* on whether the converted fraction remains
coupled to the *parent's* exterior spacetime after the conversion event.
A plain reading of "gravitationally active" and no-hair theorem is
Hypothesis A.  Hypothesis B is a deliberate departure from the standing
model; Aaron raised it as a sharper-falsifier alternative.

## Prediction matrix (hypothesis × observable)

| Observable | A (mass conservation) | B1 (merger-trig) | B2 (crit-mass) | B3 (rare spont.) | B4 (cont. leak) |
|------------|-----------------------|------------------|----------------|-------------------|-----------------|
| LIGO IMR consistency residual | ≲ 1 % (noise only) | **~15.8 % systematic** | 0–15.8 % event-dependent | ~0 (rate too low) | ≲ 1 % |
| BH mass-function cutoff | no cutoff from conversion | no cutoff | **sharp cutoff at M_crit** | no cutoff | no cutoff |
| Sgr A\* ΔM/M over 30 yr | ≲ 10⁻⁴ (accretion only) | ~0 (no merger in window) | ~0 (below M_crit) | **ΔM ∝ R_conv·τ_obs** | smooth drift ~10⁻⁹ |
| Stochastic GW background | standard compact-binary | standard + echo component | standard + conversion spikes | negligible | **distinctive spectral feature** |
| Ringdown echoes (Phase H.1) | **Δt_n = 2·r_s·n·σ_conv/c with r_s of full M** | echoes with r_s of (1−ξ)·M | as A until event | as A | as A |
| Ringdown γ suppression | γ(σ_conv) per mode | γ(σ_conv) per mode | same | same | same |

## Discriminator ranking (sharpest first)

1. **LIGO IMR consistency (against B1).**  Already-published catalog
   bounds Δ_IMR at ~5–10 % across high-SNR events — B1 predicts ~15.8 %.
   Directly falsifies merger-triggered mass shedding.  Cost: low — pull
   GWTC-3 posterior samples, compute M_pre vs M_post per event, check
   distribution.  Data already public on GWOSC.
2. **BH mass function cutoff (against B2).**  Population-level check.
   GWTC-3 has ~90 events; look for sharp high-mass cutoff not explained
   by pair-instability.  Cost: moderate — needs careful selection-effect
   modelling.
3. **Sgr A\* astrometry (against B3).**  GRAVITY published bounds on
   M_SgrA drift.  Single object only, so constrains R_conv·N_obs.
   Cost: low-moderate — pull GRAVITY papers and compute bound on R_conv.
4. **Stochastic GW background (against B4 and as a cross-check).**
   LIGO/Virgo stochastic search upper limits.  Cost: high — requires
   careful template for conversion-event spectrum.  Low priority.

## Recommended Phase H.3 — LIGO IMR consistency pull

**Rationale:** sharpest cut, cheapest data pull, already-published
posteriors.  If B1 survives, it tells us conversion is not merger-
triggered and narrows the remaining space to B2/B3/B4.  If B1 dies
(expected), B2 remains the next sharpest target.

**Procedure sketch:**
1. Fetch posterior samples for ~5 high-SNR GWTC-3 events (GW150914,
   GW190521, GW170817's partner, GW170814, GW190408).
2. For each event, extract M_total_source (from inspiral phase) and
   M_final_source (from ringdown-only analysis, where available).
3. Compute observed residual ρ = (M_total − M_final) / M_total.
4. Compare distribution of ρ against:
   - A's prediction: ρ ≈ 0.03–0.05 (radiated GW energy; standard)
   - B1's prediction: ρ ≈ 0.15–0.20 (standard radiation + ξ drop)
5. Report per-event significance and combined likelihood ratio.

**Honest caveat:** a single-session pull is suggestive-grade.  A
publication-grade analysis would re-run parameter estimation with
conversion-drop priors and compare log-evidences.  That's Phase H.4+.

## What this doc does NOT commit to

- Which hypothesis is correct.  The goal is to make both falsifiable.
- A specific R_conv or T_conv for B.  The four sub-variants cover the
  plausible trigger space; if data eliminates a subset we narrow.
- Whether "gravitational decoupling" is instantaneous or gradual.  We
  took the strong-B limit; weaker variants are bounded by null results
  at strong B.
- Revising RODM's dark-matter attribution.  Even if B holds, the
  unconverted 84.18 % can still be dark matter; B only speaks to where
  the 15.82 % goes after conversion.

## Cross-reference

- Phase G verdict: `misc/duality_ellipse_verdict.md` — γ(σ_conv)
  predictions per mode.  γ modulation is orthogonal to A-vs-B: both
  predict the same γ suppression at the horizon.
- Phase H.1 predictions: `misc/bh_merger_predictions.md` — ξ-shell
  echoes.  Echo formula uses r_s of external mass M_ext, which differs
  between A (full M) and B1 (0.842·M_tot) — a secondary
  discriminator if H.2 data is inconclusive.

## Files

- **New:** `misc/bh_conversion_mass_hypothesis.md` — this file.
- **No code changes.** Phase H.2 is a pure-formalisation deliverable.
  Phase H.3 (recommended next) would add `sigma_ground/field/interface/
  imr_consistency.py` with a `predict_imr_residual(M_total, hypothesis)`
  primitive and a GWOSC data fetcher.

## Status

**A**: coherent with standing RODM commitments, predicts no external
mass-loss signal.
**B1**: already disfavoured by existing LIGO IMR bounds at ~15.8 % level
(to be confirmed by Phase H.3 direct pull).
**B2**: mildly disfavoured by absence of BH mass-function cutoff, but
not cleanly killed.
**B3**: not constrained by current data; Sgr A\* limits are too weak
alone.
**B4**: indistinguishable from A at current observational timescales.

**Recommended test:** Phase H.3 — LIGO IMR consistency pull (sharpest,
cheapest, kills or confirms B1 definitively).
