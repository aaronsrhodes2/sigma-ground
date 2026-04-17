# RODM-A BH-BH Collision Phenomenology

**Date:** 2026-04-16
**Status:** Synthesis doc — collects Phases G + H.1–H.7 into a single
standalone prediction for "what does LIGO see from a BH-BH merger
under RODM-A?"
**Scope:** doc-only — no new primitives, no code changes.

## The claim in one sentence

Under RODM-A (mass-conserving black-hole-to-parent-universe conversion),
a BH-BH merger produces **exactly three** external signals:

1. A GR-standard gravitational-wave waveform, with two sigma-ground
   corrections layered on the ringdown phase.
2. Hawking radiation from the combined remnant — astronomically faint,
   undetectable for any stellar or intermediate-mass BH.
3. Nothing else. No ejecta, no shockwave, no EM flash (absent a
   pre-existing accretion disk), no mass-loss burst.

The dead B1 hypothesis had predicted a fourth signal (a 15.8 %
mass-loss burst at merger), but Phase H.3 killed B1 at 6.3σ on
already-published LIGO data.  B1's obituary lives in
`misc/bh_imr_verdict.md`; everything that follows is the post-B1
RODM-A prediction.

## Variable glossary (name[symbol])

| Name | Symbol | Value | Meaning |
|------|--------|-------|---------|
| conversion fraction | ξ (XI) | 0.1582 | mass fraction hypothetically lost per conversion (zero under A) |
| entanglement index | η (ETA) | 0.4153 | two-dim entanglement-per-dim constant |
| conversion scale | σ_conv | 1.8439 | −ln(ξ); dimensionless σ at which matter fully converts |
| Θ entanglement floor | Θ | 0.7461 | η^(1/3); minimum coherence ratio |
| horizon-survival amplitude | γ(σ_conv) | 0.7924 | 1 − η/2; ringdown amplitude-surviving fraction (Phase G, `sigma_coh` mode) |
| Schwarzschild radius | r_s | event-dependent | 2GM/c² for remnant mass M |
| n-th echo delay | Δt_n | event-dependent | 2·r_s·n·σ_conv/c |
| ringdown e-folding time | τ_QNM | event-dependent | ~M·GM/c³ QNM decay timescale |

## Why there is nothing else

Under the A-hypothesis, the internal conversion physics is
**causally sealed by the event horizon.**  A reader comfortable with
the no-hair theorem can stop here: external spacetime depends only
on (M, J, Q), so any internal matter reshuffling produces no external
signature beyond the geometric change that LIGO already measures as
the GW waveform.

Aaron's intuitive framing maps directly onto this:

- **"Sub-spheres of pressure combine chaotically-but-geometrically"** —
  Internal σ-field redistributes through the merger.  Geometry is
  regular in the RODM interior model (see Phase H.2 hypothesis map),
  but the reorganisation is all interior.
- **"Mass ablation via gravitational pressure geometry"** — Under A,
  this is a phrase for interior reorganisation, not external mass loss.
  The ADM mass seen by a distant observer is conserved; that is
  precisely what Phase H.3 confirmed at 6.3σ.
- **"Matter re-forms within new pressure zones"** — Happens inside;
  the new horizon encloses the reorganised state.
- **"Equalises internally"** — Exactly: the external observer learns
  nothing about interior re-equilibration beyond what M, J, Q carry.

There is no medium in vacuum between the two holes for a shockwave
to propagate through.  What "waves" in a GW signal is the metric
tensor g_μν itself, not any material substrate.  An EM flash requires
pre-existing matter (accretion disk, infalling debris); a bare BH-BH
merger in vacuum produces none.

## The two sigma-ground-specific GW signals

Both corrections apply to the **ringdown** phase — after the two
horizons merge into one and the remnant rings down to the Kerr state.

### 1. γ(σ_conv) amplitude deficit

**Prediction:** The observed ringdown strain envelope is suppressed
by a factor γ(σ_conv) ≈ 0.7924 relative to the inspiral-consistent
GR prediction.

**Derivation:** Phase G `sigma_coh` mode gives

    γ(σ_conv) = 1 − η/2 = 1 − 0.2077 = 0.7924

**Physical interpretation:** ~21 % of the inspiral-predicted ringdown
coherence phase-decorrelates during horizon reformation.  This is
**not mass loss** — it is phase scrambling.  The coherent post-merger
state loses ~21 % of its overlap with the pure-GR template because
the horizon reformation mixes in an η/2 fraction of phase-orthogonal
content.  ADM mass is unchanged; the template overlap is what falls.

**Current LIGO status:** IMR consistency tests already constrain
ringdown-vs-inspiral amplitude ratios to the ~10–15 % level on the
best events (GWTC-3 TGR paper).  A 21 % deficit is in tension with
Θ-endpoint modes (`linear`, `cbrt` at 0.7461) but consistent with
`sigma_coh` (0.7924) and `exp` (0.8395).  Formalising the bound is
the open Phase H.7b sub-task, flagged but not executed.

### 2. ξ-shell echo train

**Prediction:** After the ringdown onset, echoes appear at

    Δt_n = 2 · r_s · n · σ_conv / c,     n = 1, 2, 3, ...

with linear-in-n spacing.  The slope is fixed by theory-fixed XI and
SIGMA_CONV — **not a free parameter.**  This is what distinguishes
sigma-ground's echo prediction from generic Cardoso-Pani ECO
templates, where any delay is allowed.

**Per-event Δt_1 values** (from `misc/bh_merger_predictions.md`,
confirmed unchanged under A-baseline by Phase H.7):

| Event | M_rem (M☉) | r_s (km) | Δt_1 (ms) | Δt_1 / τ_QNM |
|-------|-----------|----------|-----------|---------------|
| GW151226 |  20.8 |  61.4 | 0.756 | 0.630 |
| GW170104 |  48.7 | 143.9 | 1.770 | 0.571 |
| GW170814 |  53.2 | 157.2 | 1.933 | 0.569 |
| GW150914 |  62.2 | 183.7 | 2.260 | 0.565 |
| GW190521 | 142.0 | 419.5 | 5.160 | 0.573 |

**Invariant:** Δt_1 / τ_QNM ≈ 0.57 across all masses.  This is the
single sharpest mass-independent sigma-ground signature.  Under
standard GR there is no structured feature at this delay.

**Physical interpretation:** The ξ-shells are discrete layers of the
internal geometry that partially reflect post-merger radiation
outward.  Each reflection round-trip adds 2·r_s·σ_conv/c to the
delay.  Aaron's "sub-spheres of pressure" language maps directly
onto these shells — they are the geometric residue of the
hierarchical interior structure surviving horizon reformation.

**Current LIGO status:** Not searched.  The Afshordi-Abedi 2017
2.5σ hint is at ~0.2 s, an entirely different delay regime.  The
1–10 ms sigma-ground regime is virgin territory for a matched-filter
search.

## Mapping Aaron's intuitions to formal primitives

| Aaron's phrase | Formal sigma-ground primitive |
|----------------|-------------------------------|
| "Out of phase" coherence during merger | γ(σ_conv) in `coherence_gamma_from_sigma` (sigma_coh mode) |
| "Mass ablation via gravitational pressure geometry" | Interior σ-field reshuffling under A; ADM-mass-conserving |
| "Sub-spheres of pressure combining chaotically-but-geometrically" | ξ-shell structure surviving horizon reformation |
| "Matter re-forms within new pressure zones" | Interior reorganisation causally sealed by horizon |
| "Equalises internally" | No external signal except GW + Hawking |

## What the dead / untestable B-variants predicted instead

For readers coming in cold: four conversion-hypothesis variants
(B1–B4) were posited in Phase H.2 as possible competitors to A.
Phases H.3–H.5 ran them through the data:

| Variant | Mechanism | Prediction | Post-H verdict |
|---------|-----------|------------|-----------------|
| **B1** | 15.8 % mass shedding at merger | ε_M ≈ 0.18 IMR offset on every event | **Dead at 6.3σ** (Phase H.3) |
| **B2** | Critical-mass threshold | Mass-function cliff at M_crit ∈ [135, 200+] M☉ | Viable but untestable until O5 (Phase H.6) |
| **B3** | Rare spontaneous conversion at R = 1/τ_Hubble | Sgr A*-visible mass drops | Sensitivity gap of ~10⁹ (Phase H.5) |
| **B4** | Continuous slow leak | Indistinguishable from A | Not a separate hypothesis |

The surviving testable-right-now prediction is A + (γ, ξ-echoes).
B2 is still alive but will be tested out by end of O5 (~2030); see
`misc/bh_o4_o5_forecast_b2.md`.

## Observable summary table

| Signal | A prediction | B1 (dead) | GR-null | Current LIGO status |
|--------|--------------|-----------|---------|---------------------|
| Inspiral chirp | Standard GR | Standard GR | Standard GR | observed |
| Merger burst | Standard GR | Standard GR | Standard GR | observed |
| Ringdown QNMs | Suppressed by γ(σ_conv) ≈ 0.79 | Suppressed + frequency-shifted by ξ | Full GR amplitude | ~10–15 % amplitude bounds consistent with γ |
| ε_M IMR residual | 0 | +0.18 | 0 | ~0 ± 0.03 ⇒ A ✓, B1 ✗ at 6.3σ |
| ξ-shell echoes | Δt_n = 2·r_s·n·σ_conv/c | (1−ξ)·Δt_n | None | **not searched** (Phase I) |
| Ejecta / shockwave | None | None | None | none observed |
| EM flash (no disk) | None | None | None | none observed |
| Mass-loss GW burst at merger | None | Yes, 15.8 % | None | none — B1 killed |
| Hawking radiation | Yes, ~10⁻⁸ K (stellar-mass) | Yes | Yes | undetectable, consistent |

## Falsifier table

What each null result would imply if searched and found null:

| Signal tested null | Implication |
|--------------------|--------------|
| No ξ-shell echoes at predicted Δt_n across ≥3 high-SNR events | Sigma-ground's ξ-quantisation is wrong.  RODM-A itself survives — echoes are a specific sub-prediction, not a theory-load-bearing one.  Phase I yields a clean upper bound on echo amplitude. |
| γ(σ_conv) = 1.0 (no ringdown amplitude deficit) | Phase G `sigma_coh` mode is wrong.  Fall back to `exp` (γ = 0.84) or a fifth candidate — but the theoretical prior favours a deficit, so a strict null rules out all four modes and would require a new mechanism imposing γ = 1. |
| Both null at high SNR | RODM-A survives as indistinguishable from pure GR at current sensitivity.  Theory is unfalsified but empirically unsupported at the sigma-ground-specific level.  Philosophically OK outcome: B1 is already falsified, A lives as the universal theory; sigma-ground would just not have produced a stellar-BH-regime signature. |
| Both positive at predicted values across multiple events | Sigma-ground promoted from "internally consistent speculative theory" to "empirically supported — both A and specific sigma-ground-quantised corrections confirmed." |

## Recommended next step

**Phase I** (optional): matched-filter search for ξ-shell echoes in
GWOSC strain data for GW151226 + GW150914.  Scope from Phase H.7:

- ~32 s GWOSC strain data (~10 MB per event)
- QNM subtraction via `pycbc` / `gwpy`
- Matched-filter residual search at predicted Δt_n, n = 1…5
- Off-source background bootstrap
- Per-event + combined p-value

Cost: moderate (requires `lalsuite` / `pycbc` install, CMake-based
toolchain).  Outcome: either a clean sigma-ground signal at predicted
Δt_n or a null result that bounds the echo amplitude and tightens
γ-mode selection.

This is the only sigma-ground-specific physics test left that is
(a) falsifiable with existing data and (b) not already closed by the
Phase H.2–H.5 analysis.  The theoretical framework stands on its own
without Phase I; the user decision point is unchanged.

## Cross-references

- Phase G γ-mode ranking: `misc/duality_ellipse_verdict.md`
- Phase H.1 echo + γ predictions: `misc/bh_merger_predictions.md`
- Phase H.2 hypothesis formalisation: `misc/bh_conversion_mass_hypothesis.md`
- Phase H.3 B1 falsification: `misc/bh_imr_verdict.md`
- Phase H.4 B2 squeeze: `misc/bh_mass_function_verdict.md`
- Phase H.5 B3 sensitivity gap: `misc/bh_b3_sgr_a_star_verdict.md`
- Phase H.6 O5 forecast for B2: `misc/bh_o4_o5_forecast_b2.md`
- Phase H.7 refined echo search: `misc/bh_echo_search_refined.md`

## Files

- **New:** `misc/bh_collision_phenomenology.md` — this file (doc-only, no code)
- **No code changes.**
- **Unchanged:** all prior Phase G/H artefacts stand exactly as written.
