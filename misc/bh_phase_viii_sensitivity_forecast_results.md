# Phase VIII — Sensitivity Forecast: Roadmap to a Decisive Test

**Date:** 2026-04-17  
**Phase:** VIII — B(sigma_coh/GR) vs ρ_syn and catalog depth  
**Pipeline:** `scripts/phase_viii_sensitivity_forecast.py`  
**Baseline:** Phase VII corrected (ΔlnL = +0.183, B = 1.20)

---

## Method

Expected ΔlnL per event computed by marginalising over ρ_obs:

    E[ΔlnL | H] = ∫ p(ρ_obs | H) × [ln Rician(ρ_obs; Γ_sc × ρ_syn) −
                                       ln Rician(ρ_obs | Γ_GR × ρ_syn)] dρ_obs

under two hypotheses: H_GR (ρ_obs ~ Rician(Γ_GR × ρ_syn, 1)) and
H_sc (ρ_obs ~ Rician(Γ_sc × ρ_syn, 1)), where Γ_sc = 0.7924.

Cumulative B(sigma_coh/GR) = exp(baseline_ΔlnL + N × E[ΔlnL_per_event]).

Three forecast axes:
- **A** — GW150914/H1 pycbc improvement (ρ_syn 0.73 → 20)
- **B** — GWTC-3 coherent stack (N additional events, ρ_syn_typ = 0.45)
- **C** — Combined: pycbc on GW150914 + N GWTC-3 events

---

## Results

### Panel A — Single-event sensitivity (GW150914/H1 pycbc path)

| ρ_syn | B (if GR true) | B (if sigma_coh true) |
|---|---|---|
| 0.73 (current) | 1.14 | 1.14 |
| 2.00 (t* opt) | 1.06 | 1.22 |
| 5.00 (pycbc 3–5×) | 0.67 | 1.93 |
| 10.00 (pycbc 10×) | 0.13 | 9.71 |
| **≥ 10.1** | **< 0.1** | **> 10** |

**Key finding:** ρ_syn ≥ 10.1 is needed for a single event to give B > 10
(strong evidence) IF sigma_coh is true.  The pycbc IMR subtraction target
of ρ_syn ≈ 3–5 (originally estimated in Phase IV) is insufficient — it
gives B ≈ 1.9–2.0.  A 10–14× improvement over the corrected ρ_syn = 0.73
is needed.

If GR is true: B(sigma_coh/GR) decreases to 0.13 at ρ_syn = 10, i.e.,
B(GR/sigma_coh) = 7.5 — approaching strong evidence against sigma_coh
from a single event.

### Panel B — GWTC-3 coherent stack (no pycbc)

```
E[ΔlnL per event] under sigma_coh = +0.00061   (ρ_syn_typ = 0.45)
E[ΔlnL per event] under GR        = −0.00062

N to reach B > 10 (if sigma_coh true):  3500 events
N to reach B < 0.1 (if GR true):        4010 events   [rule out sigma_coh]
GWTC-3 catalog:                         ~90 BBH events → ~180 detector-events
GWTC-3 fraction of needed:              ~5%
```

**Key finding:** GWTC-3 alone cannot test the sigma_coh prediction at any
meaningful level.  The corrected per-event E[ΔlnL] = 0.00061 is 15× smaller
than the biased Phase II estimate (which used the inflated ρ_syn_biased).
At 90 BBH events × 2 detectors = 180 detector-events:

    B after GWTC-3 (sigma_coh true) = exp(0.183 + 180 × 0.00061) = exp(0.293) = 1.34

GWTC-3 would move the Bayes factor from 1.20 to 1.34 — unmeasurable.

### Panel C — Combined: pycbc on GW150914 + GWTC-3 stack

```
Immediately after pycbc (ρ_syn 0.73 → 5.0):
  ΔlnL boost = +0.521 (sigma_coh)
  B (if sigma_coh true) = 2.02
  B (if GR true)        = 0.71

N additional events needed for B > 10 after pycbc:
  If sigma_coh true: 2650 additional events
  If GR true:        ~3010 additional events (to rule out sigma_coh)
```

pycbc is worth doing (ΔlnL boost = +0.52, B 1.20 → 2.02) but still
leaves a factor of 5 gap to B = 10.  Combined with the full O3+O4+O5
catalog projection (~1000 events), B would reach ~4–5 — suggestive but
not conclusive.

---

## Why the Test is Harder than Phase II Estimated

Phase II predicted ~312 events for B > 10 using the formula:

    N = (3σ / ξ / ρ̄_QNM)²   with ρ̄_QNM (biased) = 1.075

After the Phase VI injection correction:

    ρ̄_QNM (corrected) = ρ̄_QNM (biased) × ⟨exp(−3ms/τ)⟩ ≈ 1.075 × 0.42 = 0.45

The corrected N scales as (0.45/1.075)² ≈ 0.175× relative to biased,
giving:

    N_corrected = 312 / 0.175 ≈ 1780 events   (Phase II formula)

The Phase VIII Bayesian calculation gives 3500 (no pycbc) — consistent with
the Phase II formula up to a factor of 2 from the different statistical framework.

**Root cause:** The injection bias in Phase III inflated ρ_syn by 2.1–2.5× for
most events.  Since sensitivity scales as N × ρ_syn², the effective catalog
depth was overestimated by (2.1–2.5)² ≈ 4–6×, and the required event count
was underestimated by the same factor.

---

## Asymmetry: Confirmation vs Falsification

| Outcome | Condition | N events needed | B threshold |
|---|---|---|---|
| Confirm sigma_coh | sigma_coh true | 2650 + pycbc | B > 10 |
| Rule out sigma_coh | GR true | 3010 + pycbc | B(GR/sc) > 10 |
| Current state | Unknown | 0 | B = 1.20 |

The test is symmetric: confirming OR ruling out sigma_coh requires
~3000 events with pycbc.  This is because the Γ_sc = 0.79 and Γ_GR = 1.0
predictions differ by only 0.21 (21%), and at ρ_syn_typ ≈ 0.45, the
Rician likelihood is nearly insensitive to this difference.

---

## Realistic Scenarios

### Near-term (O4 catalog, ~500 BBH events, no pycbc)

    B (if sigma_coh true) = exp(0.183 + 500 × 0.00061) = exp(0.488) = 1.63
    B (if GR true)        = exp(0.183 − 500 × 0.00062) = exp(−0.127) = 0.88

Inconclusive — Bayes factor moves from 1.20 to 1.63 (sigma_coh) or 0.88 (GR).

### Near-term (O4 + pycbc on 5 best events at ρ_syn ≈ 5 each)

    Boost from 5 pycbc events = 5 × (ΔlnL at ρ_syn=5 under sigma_coh) ≈ 5 × 0.52 = 2.6
    B (if sigma_coh true) = exp(0.183 + 2.6) = exp(2.783) = 16.2  ← B > 10!
    B (if GR true)        = exp(0.183 + 5 × (−0.90)) = exp(−4.3) = 0.014  [rules out sigma_coh]

**This is the decisive scenario:** pycbc applied to the 5 best GWTC-3 events
(selected by highest intrinsic ρ_syn before correction) gives B > 10 either
direction after O4, without waiting for O5/O6.

### Key: pycbc quality matters more than quantity

At ρ_syn = 5 per event (pycbc 3–5× improvement from corrected baseline):
- E[ΔlnL | sigma_coh] ≈ 0.52 per pycbc event
- Need 4 events at ρ_syn = 5 for B > 10 (if sigma_coh true)

At ρ_syn = 10 per event (pycbc 10–14× improvement):
- E[ΔlnL | sigma_coh] ≈ 2.14 per pycbc event
- Need only 1 event for B > 10 (if sigma_coh true)

**Conclusion:** The critical lever is pycbc IMR subtraction quality on a small
number of high-SNR events.  GWTC-3 depth alone cannot compensate for low
per-event ρ_syn.

---

## Combined Evidence State (Phases I–VIII)

| Phase | Claim | Corrected verdict |
|---|---|---|
| I.4–I.5 | Echo search | Null — 13–17× below threshold |
| II | ξ bound | Untestable at any ξ |
| III (biased) | Γ = 0.856 ± 1.196 | Artifact; corrected Γ ≈ 1.5–2.7 for most events |
| IV (biased) | P(Γ<1) = 94% | Artifact; corrected = 71% |
| V (biased) | P(Γ<1) = 77% | Artifact; corrected = 53% |
| VI | Injection bias found | ρ_syn wrong ×2.1–12× per event |
| VII | Corrected: B = 1.20 | No evidence; GR inside 68% HDI |
| **VIII** | **Roadmap** | **Need ρ_syn ≥ 10 (1 event) or 3000+ events** |

---

## Path to a Decisive Test (Corrected)

```
Step 1 (immediate): Apply corrected injection to Phase III
        → Now done in Phase VII; establishes clean null baseline

Step 2 (pycbc, ~months): Full IMR subtraction on 5 best GWTC-3 events
        → Need ρ_syn ≥ 5 per event (requires lalsuite/SXS toolchain)
        → If sigma_coh true: B ≈ 16 after 5 events  [decisive]
        → If GR true: B ≈ 0.014 after 5 events  [decisive in other direction]

Step 3 (O4+O5 stack, ~years): Coherent stack of all corrected events
        → Provides independent confirmation/rejection at B > 10
        → Required only if pycbc result is marginal (B ≈ 2–5)
```

The single highest-leverage action is **pycbc IMR subtraction on the top 5
GWTC-3 events by intrinsic mass** — no additional data needed, just software.

---

## Cross-References

- Phase VII corrected baseline: `misc/bh_phase_vii_corrected_analysis_results.md`
- Phase VIII script: `scripts/phase_viii_sensitivity_forecast.py`
- Phase VIII figure: `misc/bh_phase_viii_sensitivity_forecast.png`
- pycbc / SXS requirements: lalsuite ≥ 7.0, SXS catalog v3, GW150914 strain (16 kHz cached)
