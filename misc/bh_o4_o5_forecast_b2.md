# Phase H.6 forecast: O4/O5 future-LIGO constraints on Hypothesis B2

**Date:** 2026-04-17
**Phase:** H.6 — future-observing-run forecast for B2 squeeze
**Scope:** doc-only — no code or data pull.  Forecasts what would be
needed from upcoming LIGO/Virgo/KAGRA observing runs to falsify or
confirm B2 in its surviving M_crit window.
**Hypothesis under test:** B2 at M_crit ∈ [135, 200+] M☉, as narrowed
by Phase H.4.

## Variable glossary (name[symbol])

| Name | Symbol | Meaning |
|------|--------|---------|
| O4 | — | LIGO observing run 4 (2023–2025, completed) |
| O5 | — | LIGO observing run 5 (2027–2030+, planned) |
| ε_M | — | IMR consistency parameter (Phase H.3 glossary) |
| M_crit | — | B2 critical-mass threshold |
| f_rad | — | radiated GW energy fraction, typ. 0.05 |
| detection horizon | D_h | volume-averaged detection range |

## Starting position (end of O3b, 2020 data)

From Phases H.3 + H.4:

- GWTC-3 event count: ~90 confident detections
- Events with M_f > 135 M☉: **1 (GW190521)**
- Per-event IMR σ on GW190521: ~1.08 against B1's 0.18 prediction
- M_crit ∈ [135, 200] M☉ window is the clean B2 survival zone
- M_crit ∈ [71, 135] M☉ is the degenerate zone (overlaps pair-instability gap)

## O4 outlook (2023–2025, already completed at date of writing)

**Detector upgrades:**
- LIGO Livingston / Hanford improved strain sensitivity ~30 % over O3
- Virgo recovered post-outage, lower duty cycle
- KAGRA joined at ~10 Mpc horizon (limited contribution to SMBH mass
  regime)

**Expected event yield (pre-O4 estimates, Abbott et al. 2020 LSC
prospects paper):**
- Total: 200–350 new confident detections
- BBH (binary BH): ~150–300
- Heavy-mass tail (M_f > 100 M☉): ~5–15 events
- **M_f > 135 M☉ expected: ~2–5 events**

**Post-O4 projection (based on public summaries as of 2026-04):**
The catalog-update paper from O4a (early 2024) reported ~80 BBHs in the
first ~6 months, consistent with the upper pre-run estimate.  Final O4
catalog expected to nearly double total GWTC-3 count.

**Impact on B2 squeeze:**
- If O4 adds 3–4 events with M_f > 135 M☉, each with IMR σ ≈ 0.8–1.2
  against B1's 0.18:
  - Combined-in-quadrature σ ≈ √((GW190521)² + 4·(1.0)²) ≈ √(1.2 + 4) ≈ 2.3σ
- **B2 at M_crit ~135–150 M☉ would move from "untested" to ~2σ
  tension** — not a kill, but a shift.
- **Pile-up signature:** if B2 is real with M_crit ≈ 150, expect a
  bump at (1 − ξ)·150 = 126 M☉ in the post-O4 mass function.  Null
  result here (smooth distribution continuing through 126 M☉) would
  add constraint.

## O5 outlook (2027–2030+, A+ upgrade era)

**Detector upgrades (planned):**
- A+ upgrade at both LIGO sites: strain sensitivity ~2× O3
- Detection horizon for 30-M☉ BBH: ~330 Mpc (from ~150 Mpc at O3)
- For 100-M☉ BBH, horizon extends to ~1 Gpc
- Virgo+ and KAGRA at design sensitivity
- Detection rate: 200–400 BBH/yr expected

**Expected event yield (full 3-year run, 50 % duty cycle):**
- Total BBH: ~1000–1500
- **M_f > 135 M☉ expected: ~30–100 events**
- Heavy-mass tail well-populated for the first time

**Impact on B2 squeeze:**
- With ~30 events above 135 M☉, each with per-event IMR σ ≈ 1–1.5
  against B1:
  - Combined σ ≈ √(30·1.2²) ≈ 6.6σ
- **B2 at M_crit ≤ 135 M☉ would be falsified at > 5σ** under the same
  "every triggered event gives B1-strength signature" assumption.
- **Mass-function cutoff search:** with 30–100 events in the
  M ∈ [100, 300] M☉ regime, LIGO population papers can fit explicit
  cutoffs.  B2 at M_crit = 200 M☉ predicts a visible cliff; 3σ
  detection or exclusion expected.
- **Pile-up at (1 − ξ)·M_crit:** high statistics allow robust
  over-density test.  If no pile-up at any candidate M_crit ∈
  [150, 300] M☉, B2 disfavoured across the entire surviving window.

## O6 / post-LIGO era (2030+)

**Cosmic Explorer (CE) and Einstein Telescope (ET):**
- CE: 40-km arms, strain sensitivity ~10× O3, horizon out to z ≈ 2
  for 30-M☉ BBH, effectively **every BBH in the observable universe**
- ET: underground triangular config, similar sensitivity
- Detection rate: ~10⁵ BBH/yr

**Impact:**
- B2 at any M_crit across the full plausible range becomes testable
  at high statistical significance.
- Population-level mass function measured to < 1 % precision.
- If B2 is real, it is pinned down to ~M_crit accurate to ~1 M☉.
- If B2 is not real, it is excluded across the entire parameter space.

## LISA (2035+)

LISA's band (10⁻⁴ – 10⁻¹ Hz) is for *SMBH* mergers, not stellar-mass.
It accesses a completely disjoint mass regime (10⁵ – 10⁸ M☉).  Under
B2, the conversion threshold would be expected at some fixed M_crit
that is either in the LIGO band (stellar-mass population) or the
LISA band (SMBH population), not both.

- If M_crit is stellar-mass (~100–10³ M☉): LIGO/CE test dominates.
- If M_crit is SMBH-mass (~10⁶ M☉): LISA is the only detector.
- If M_crit depends on BH formation pathway: both needed.

## Projected B2 fate by observing-run milestone

| Milestone | Expected B2 status at M_crit ~150 M☉ |
|-----------|----------------------------------------|
| End of O4 (2025) | **~2σ tension**, not killed |
| End of O5 (2030) | **~5-6σ tension**, effectively killed if no pile-up |
| End of CE/ET era (~2035+) | Pinpoint measurement or clean exclusion |
| LISA era (2035+) | Independent SMBH-regime test |

## What would CONFIRM B2 (positive signatures)

- Clean mass-function cliff at some M_crit ∈ [150, 300] M☉ — sharper
  than pair-instability predicts, not smoothed by hierarchical-merger
  fill-in.
- Pile-up (dN/dM bump) at (1 − ξ)·M_crit = 0.8418·M_crit consistent
  across events from different redshifts.
- Per-event IMR residuals (ε_M) clustering at +0.18 for M_f > M_crit,
  not for M_f < M_crit — **the sharpest B2-specific signature**.
- Cross-event consistency: M_crit measured from different events
  agrees to within systematic tolerance.

If all four lines converge on the same M_crit, B2 is effectively
confirmed.

## What would FALSIFY B2 (null signatures)

- Smooth dN/dM with no cutoff or pile-up through M_f ~ 300 M☉.
- Per-event ε_M consistent with zero across the full mass range.
- Mass-function fit prefers power-law or double-peak models over
  any step-function / cutoff shape.

These are the expected outcomes if A is correct; current data already
lean this way.

## Conclusion

**B2 will be decisively tested by end of O5 (~2030).**  Current data
leave B2 squeezed but alive; O4 nudges it toward tension; O5 should
either confirm the universal theory (A-variant of RODM) or leave a
single surviving B2 variant with a specific M_crit estimate from data.

This is the appropriate rhythm for this kind of physics: the prediction
is falsifiable on a 5-year timescale using detectors and surveys that
are already funded and running.  No speculative new instruments required.

## Cross-references

- Phase H.3 verdict (B1 dead): `misc/bh_imr_verdict.md`
- Phase H.4 verdict (B2 squeezed): `misc/bh_mass_function_verdict.md`
- Phase H.5 verdict (B3 untested): `misc/bh_b3_sgr_a_star_verdict.md`
- Hypothesis map: `misc/bh_conversion_mass_hypothesis.md`

## Files

- **New:** `misc/bh_o4_o5_forecast_b2.md` — this file (doc-only, no code)
- **No code changes.**
