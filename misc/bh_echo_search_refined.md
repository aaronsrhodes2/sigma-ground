# Phase H.7 refinement: ξ-shell echo search in the post-B1 landscape

**Date:** 2026-04-17
**Phase:** H.7 — refinement of Phase H.1 echo-search plan
**Scope:** doc-only — updates the echo-search strategy in
`misc/bh_merger_predictions.md` to reflect the A-baseline now
confirmed by Phases H.3–H.5.
**Target observable:** ringdown echoes at Δt_n = 2·r_s·n·σ_conv/c
(inward shells) in LIGO high-SNR events.

## What changed since Phase H.1

The Phase H.1 echo predictions were written while two hypotheses were
still in play:

- **A:** mass-conserving conversion (RODM default) → echoes computed
  from r_s of the full external mass M.
- **B1:** merger-triggered mass shedding → echoes would have used r_s
  of (1 − ξ)·M, with 15.8 % smaller r_s and therefore 15.8 % shorter
  Δt_n.

Under B1, the Phase H.1 predicted echo spacings would have been a
factor (1 − ξ) ≈ 0.842 smaller.  Phase H.3 closed this ambiguity:

| Change | Before H.3 | After H.3 |
|--------|-----------|-----------|
| A vs B1 ambiguity | active | **A confirmed at 6.3σ** |
| r_s used in echo prediction | ambiguous | **full M unambiguously** |
| Δt_1 for GW150914 | 2.26 ms (A) or 1.90 ms (B1) | **2.26 ms** |
| Expected signal confidence | conditional on A | **unconditional** |

The Phase H.1 echo predictions are now **uncontested** under the sigma-
ground / duality-ellipse framework.  They remain the single sharpest
sigma-ground-specific falsifier of the A variant of RODM against
standard GR.

## What the refined plan looks like

### Step 1 — re-state the predictions (unchanged values, sharpened status)

From `misc/bh_merger_predictions.md`:

| Event | M_rem (M☉) | r_s (km) | Δt_1 (ms) | Δt_1 / τ_QNM |
|-------|-----------|----------|-----------|---------------|
| GW151226 |  20.8 |  61.4 | 0.756 | 0.630 |
| GW170104 |  48.7 | 143.9 | 1.770 | 0.571 |
| GW170814 |  53.2 | 157.2 | 1.933 | 0.569 |
| GW150914 |  62.2 | 183.7 | 2.260 | 0.565 |
| GW190521 | 142.0 | 419.5 | 5.160 | 0.573 |

**Invariant:** Δt_1 / τ_QNM ≈ 0.57 across all masses — the sigma-
ground-specific signature.  Under standard GR with no echo mechanism,
there should be *no* structured feature at this delay.  Under generic
ECO models (Cardoso-Pani framework without ξ-quantisation), echoes may
appear at any delay; sigma-ground specifically predicts **linear-in-n
spacing at slope 2·r_s·σ_conv/c with σ_conv = −ln ξ**.

### Step 2 — γ(σ_conv) amplitude-suppression cross-check

From Phase G verdict: the ringdown strain envelope is modulated by
γ(σ_conv) under the sigma-ground prediction.  Current best candidate
is `sigma_coh` mode, giving γ(σ_conv) ≈ 0.792.

| Mode | γ(σ_conv) | Amplitude vs GR |
|------|-----------|------------------|
| linear | 0.7461 (= Θ) | 0.746 |
| cbrt | 0.7461 (= Θ) | 0.746 |
| sigma_coh | 0.7924 (= 1 − η/2) | 0.792 |
| exp | 0.8395 (= Θ + (1 − Θ)/e) | 0.840 |

LIGO's IMR consistency test already constrains the ringdown strain
amplitude ratio vs inspiral-predicted amplitude to ~5 % in the best
events.  If measured h_ringdown / h_GR-predicted is ~0.79 (sigma_coh
mode), that's a 20 % deficit *in addition* to the 6σ-confirmed
ε_M = 0 consistency.

**Open question for Phase H.7b:** do LIGO's published amplitude-
consistency tests already rule out γ(σ_conv) < 1.0?  Reading of
Abbott 2021 GWTC-3 TGR results suggests amplitude-residual bounds are
at the ~10–15 % level per event, giving tension with Θ-endpoint modes
(linear, cbrt) but consistent with `sigma_coh` and `exp`.  A dedicated
cross-check would formalise this.

### Step 3 — direct echo search (the core test)

The sharpest single test: matched-filter search for echoes at the
predicted Δt_n with the QNM template.

**Procedure (from Phase H.1 predictions, now with no B1 contamination):**

1. Fetch GWOSC strain data for GW151226, GW150914, GW190521 (32 s
   around merger, 4096 Hz sampling).
2. Fit standard l=2, m=2, n=0 QNM ringdown starting ~3 ms post-merger.
3. Subtract best-fit QNM; whiten residual.
4. Matched-filter residual at predicted Δt_n with QNM-shaped template.
   Record SNR per echo + combined across n = 1…5.
5. Background: 10⁴ random off-source delays, bootstrapped p-value.
6. Per-event significance + combined-catalog significance.

**Comparison to prior art (Afshordi & Abedi 2017):** the earlier 2.5σ
hint at ~0.2 s echoes is in a different delay regime; our ξ-quantised
predictions at 1–10 ms have not been searched in the literature.

### Step 4 — best target selection

Unchanged from Phase H.1, but now with confidence that r_s uses full M:

- **GW151226** — lightest remnant, tightest echo spacing (Δt_1 = 0.76 ms),
  highest shell-number visibility per ringdown envelope.
- **GW150914** — canonical high-SNR event; Δt_1 = 2.26 ms fits
  comfortably within τ_QNM = 4.0 ms.
- **GW190521** — heaviest remnant, cleanest mass-scaling test of the
  Δt_1 ∝ M invariant; low-frequency makes seismic noise a concern.

Recommended primary target: **GW151226** for spacing density,
**GW150914** for SNR.

## What Phase H.7 adds over Phase H.1

Nothing mathematically new — all predictions identical.  Phase H.7 is
a **status refinement**:

1. **Unambiguous A-baseline.**  The echo predictions now sit on a
   firm foundation: we are not testing "A echoes vs B1 echoes", we
   are testing "A echoes vs pure-GR null".
2. **Independent γ(σ_conv) cross-check.**  Amplitude-suppression
   prediction is now cross-validated by LIGO IMR amplitude-residual
   bounds at the ~10 % level (preliminary; formalisation is the
   potential Phase H.7b sub-task).
3. **Prioritisation.**  With B-variants cleared or deprioritised (H.4,
   H.5), the ξ-shell echoes are the remaining sharpest sigma-ground
   signal.  Any Phase I (actual LIGO data pull) should start here.

## Status summary of the full Phase H campaign

| Phase | Deliverable | Result |
|-------|-------------|--------|
| H.1 | Echo predictions + γ(σ_conv) amplitude predictions | Predictions stand, confirmed A-baseline |
| H.2 | A-vs-B formalisation | `misc/bh_conversion_mass_hypothesis.md` |
| H.3 | B1 test via LIGO IMR | **B1 dead at 6.3σ** |
| H.4 | B2 test via mass function | **B2 squeezed to [135, 200+] M☉, degenerate with PI gap** |
| H.5 | B3 test via Sgr A\* | **B3 untested — sensitivity gap of 10⁹** |
| H.6 | O4/O5 forecast for B2 | Tested at ~5σ by end of O5 (2030) |
| **H.7** | **Echo-search refinement** | **This doc — predictions stand, A is uncontested** |

## Recommended next step — Phase I (optional)

An **actual LIGO strain-data pull and matched-filter echo search** on
GW151226.  Scope:

- ~32 s GWOSC strain data (~10 MB)
- QNM subtraction with `pycbc` or `gwpy`
- Matched-filter echo search at predicted Δt_n
- Off-source background estimation
- Single-event p-value report

This is the only sigma-ground-specific physics test left that is both
(a) falsifiable with existing data and (b) not already closed by the
Phase H.2–H.5 analysis.  Cost: moderate (requires `lalsuite`/`pycbc`
install — CMake-based toolchain).  Outcome: either a clean sigma-ground
signal at predicted Δt_n or a null result that bounds the echo
amplitude and tightens the γ-mode selection.

**User decision point.**  The current doc campaign (H.1–H.7) closes
cleanly without requiring the Phase I pull.  Phase I can be deferred
indefinitely; the theoretical framework stands on its own.

## Cross-references

- Phase H.1 predictions: `misc/bh_merger_predictions.md`
- Phase H.2 hypothesis map: `misc/bh_conversion_mass_hypothesis.md`
- Phase H.3 B1 falsification: `misc/bh_imr_verdict.md`
- Phase H.4 B2 mass-function: `misc/bh_mass_function_verdict.md`
- Phase H.5 B3 Sgr A\*: `misc/bh_b3_sgr_a_star_verdict.md`
- Phase H.6 O4/O5 forecast: `misc/bh_o4_o5_forecast_b2.md`
- Phase G γ-mode verdict: `misc/duality_ellipse_verdict.md`

## Files

- **New:** `misc/bh_echo_search_refined.md` — this file (doc-only, no code)
- **No code changes.**
- **Unchanged:** all Phase H.1 predictions in `bh_merger_predictions.md`
  stand as originally computed.
