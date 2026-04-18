# Unpublished Predictions — sigma-ground + arXiv integrations

**Purpose:** Log predictions that emerge from combining sigma-ground's physics with
a specific arXiv paper, that **neither the engine alone nor the paper alone predicts**,
and that no other paper we have surveyed claims. Each prediction carries a clear
falsifier so it can be either confirmed, refuted, or promoted to a paper.

**Format:** one entry per prediction. Each is `[SPECULATIVE]` until an observation or
dedicated simulation moves it to `[VERIFIED]` or `[FALSIFIED]`.

---

## UP-001 — η-triple-coincidence at 0.4153 ± 0.005

**Combines:** sigma-ground (ρ_DE-derived η) + arXiv:2411.08639 (DESI DR2 HDE c²) +
Phase XI's ETA_FORMULA = exp(−φ/σ_conv).

**Prediction:** DESI DR3 HDE c² will land within 1% of 0.4153, and ETA_FORMULA will
reproduce the DR3 central value to the same tolerance. All three independent derivations
of η will overlap within the measurement uncertainty.

**Falsifier:** If DESI DR3 publishes c² with a central value outside [0.410, 0.420] at
<1σ, the triple-coincidence dissolves and the η-identification with c² is coincidence
or wrong.

**Status:** `[SPECULATIVE]` — awaiting DR3 central value.

---

## UP-002 — η-shifted Page time for evaporating BHs

**Combines:** sigma-ground (η as entanglement fraction) + arXiv:2502.04430 (Page time
t_P(M) in SM + BSM).

**Prediction:** Sigma-ground's Page time for a Schwarzschild BH is shifted from the
SM value by a factor (1 − η/2) ≈ 0.792, reflecting the fraction of Hawking-radiation
entanglement that never thermalizes with the σ=0 exterior.

**Falsifier:** Dedicated LIGO-era analysis of inspiraling primordial-BH candidates
with direct entropy accounting (if ever feasible). Short-term, the prediction is
consistency-testable against the Page curve computed in `sigma_ground/field/entanglement.py`.

**Status:** `[SPECULATIVE]` — Phase XII.b integration task.

---

## UP-003 — H₀ tension magnitude correlates with local σ-conv-crossing event count

**Combines:** sigma-ground (σ-conv crossings = BH formations + primordial bubble events)
+ arXiv:2511.09467 (Hubble tension requires second component beyond HDE).

**Prediction:** The H₀ tension is larger along sightlines with higher density of
σ-conv-crossing events (e.g., through galaxy clusters with more stellar-mass BHs) than
along void-sightlines. Effect size: parts-in-10⁴ of the late-time H₀ value — within
next-decade precision-cosmology reach.

**Falsifier:** Direction-dependent H₀ maps from Euclid + LSST showing zero correlation
with BH-density maps.

**Status:** `[SPECULATIVE]` — requires a dedicated analysis pipeline we have not built.

---

## UP-004 — S₈ anti-correlates with η across cosmological ensembles

**Combines:** sigma-ground (γ(σ) flattens small-scale power) + arXiv:2505.23382
(EDE + interacting DE joint H₀ + S₈ fit).

**Prediction:** In a parameter scan over modified-σ cosmologies with varying η, S₈ and η
trace out an approximately linear anti-correlation with slope ≈ −0.5 ± 0.2.

**Falsifier:** Output of a ~100-cosmology sim grid where the anti-correlation is either
present (confirm) or absent (falsify).

**Status:** `[SPECULATIVE]` — Phase XII.c candidate.

---

## UP-005 — ξ ↔ asymptotic-safety RG fixed point

**Combines:** arXiv:2410.22412 (Kobakhidze ξ from N=8 QCD β-functions) + arXiv:2505.01422
(neutrino mass from asymptotic-safety RG flow).

**Prediction:** The asymptotic-safety RG fixed point coupling, evaluated at the neutrino
mass scale, equals the Kobakhidze N=8 ξ = 0.1572 to within 1%.

**Falsifier:** Direct RG calculation from 2505.01422's methodology giving a value outside
[0.155, 0.160].

**Status:** `[SPECULATIVE]` — most adventurous Batch-3 claim; would unify
DM/baryon ratio with neutrino-mass generation under one RG framework.

---

## UP-006 — Intra-horizon LIV without external LIV

**Combines:** arXiv:2502.18256 (external LIV bound Λ₂ > 5 × 10¹⁹ GeV) + sigma-ground's
non-zero σ inside event horizons.

**Prediction:** Photons propagating in the bond-failure layer (σ ∈ [ξ/2, σ_conv)) should
exhibit effective dispersion at an energy scale derivable from σ(r), while σ=0 extragalactic
photons show none. The intra-horizon Λ₂_eff(r) is a specific function of r/r_s.

**Falsifier:** Future intra-shadow EHT fluctuation measurements inconsistent with the
predicted Λ₂_eff(r).

**Status:** `[SPECULATIVE]` — observationally distant but mathematically concrete.

---

## UP-007 — η as bit-thread fraction on σ-conv surface

**Combines:** arXiv:2508.18941 (differential entropy = Bekenstein-Hawking via bit threads)
+ sigma-ground's σ_conv = −ln(ξ) bond-failure surface.

**Prediction:** The bit-thread density through a σ = σ_conv iso-surface equals η × (1 Planck
area)⁻¹ to leading order, giving a geometric interpretation of η as the "active fraction"
of Planck cells on the conversion surface.

**Falsifier:** Direct bit-thread computation on a Schwarzschild-AdS background with a
sigma-ground-specified σ-field profile, yielding a density inconsistent with η/ℓ_P².

**Status:** `[SPECULATIVE]` — requires an entanglement-entropy computation we have not
yet set up.

---

## Entry conventions

- New entries append; do not edit past entries except to update **Status**.
- When an entry moves to `[VERIFIED]` or `[FALSIFIED]`, leave the original text in place
  and add a timestamped **Update:** block.
- Cross-reference the Phase doc or commit that moved the status.
