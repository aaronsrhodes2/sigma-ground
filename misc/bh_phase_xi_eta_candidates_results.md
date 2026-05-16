# Phase XI — η Candidate Comparison

> **STATUS (2026-05-15): SUPERSEDED.** The "best candidate" verdict at the
> bottom of this document — `ETA_FORMULA = exp(−φ/σ_conv)` — was **REJECTED**
> by a 2026-05-15 audit as formula-search numerology. The formula was found
> by searching {ξ, σ_conv, π, e, φ} for an expression matching the working
> ETA = 0.4153; that target was itself derived from a "golden-spiral"
> heuristic, so the φ-match was circular by construction. The 0.125% residual
> gap is the signature of a near-miss in a wide search, not of a near-derivation.
>
> **Current resolution:** see `misc/eta_empirical_verdict_2026-05-15.md`. η is
> now [EMPIRICAL-INPUT] anchored at DESI Union3 c² ≈ 0.4122. This document
> is retained for historical traceability; do not cite it as a current verdict.

**Date:** 2026-04-17
**Phase:** XI — systematic evaluation of η = 0.4153 replacement candidates
**Pipeline:** `scripts/phase_xi_eta_candidates.py`
**Label:** `[SPECULATIVE]` — no candidate is yet confirmed
**Status (2026-05-15):** SUPERSEDED — see banner above

---

## Background

η = 0.4153 is the cosmic entanglement fraction in sigma-ground, originally derived heuristically from a dark-energy constraint and acknowledged to be "related to the golden spiral." This phase evaluates four independent lines of evidence for what η's true value might be.

---

## Candidates

| Label | Formula | Value | Error from 0.4153 | Source |
|-------|---------|-------|-------------------|--------|
| η_working | (heuristic) | 0.415300 | 0.000% | dark-energy constraint |
| **η_formula** | **exp(−φ/σ_conv)** | **0.415818** | **0.125%** | formula search 2026-04-17 |
| η_HDE_U3 | c² (Union3) | 0.412164 | 0.755% | DESI 2024, arXiv:2411.08639 |
| η_HDE_D5 | c² (DESY5) | 0.491401 | 18.3% | DESI 2024 (excluded) |
| η_Barrow | Δ_max | ≤ 0.43 | 3.5% | arXiv:2503.18230 |

---

## Key result: triple coincidence

η_working (0.4153), η_formula (0.4158), and η_HDE_U3 (0.4122) all cluster within **0.88%** of each other. The DESI DESY5 value (0.491) and Barrow upper bound (0.43) are excluded as primary candidates.

DESI Union3 1-σ band: c = 0.642 ± 0.028 → c² ∈ [0.377, 0.449]. All three primary candidates fall within this band.

---

## Best candidate: exp(−φ/σ_conv)

**η_formula = exp(−φ / σ_conv) ≈ 0.4158**, within 0.125% of the working value.

Properties:
- Expressed purely in terms of model constants: φ (golden ratio) and σ_conv = −ln(ξ)
- Consistent with the original "related to the golden spiral" heuristic — φ appears explicitly
- Sits at +0.89% from the DESI Union3 central c² = 0.4122 (within 1σ)
- Under Kobakhidze ξ, shifts to 0.4170 (+0.001 from Planck value)

The formula **exp(−φ/σ_conv)** connects the golden ratio directly to the matter-conversion horizon. This is equivalent to exp(φ/ln ξ), showing ξ appears through its log (i.e., through σ_conv).

---

## Interpretation

If the HDE identification c² ≡ η holds, the triple coincidence (formula + heuristic + DESI) is non-trivial: exp(−φ/σ_conv) predicts 0.4158 without fitting to DESI data, yet lands inside the observational band.

The three lines of evidence are not fully independent (all connect to ρ_DE), but their convergence narrows η to 0.412–0.416 with high confidence.

**Verdict: ETA_FORMULA = exp(−φ/σ_conv) is the best current candidate for η.**

---

## Constants added

In `sigma_ground/field/constants.py`:
```python
PHI = (1.0 + math.sqrt(5.0)) / 2.0          # golden ratio ≈ 1.6180
ETA_FORMULA = math.exp(-PHI / SIGMA_CONV)    # ≈ 0.4158 [SPECULATIVE]
C_HDE_UNION3 = 0.642                          # DESI HDE c (Union3)
ETA_HDE_UNION3 = C_HDE_UNION3 ** 2          # ≈ 0.412
C_HDE_DESY5 = 0.701                           # DESI HDE c (DESY5)
ETA_HDE_DESY5 = C_HDE_DESY5 ** 2            # ≈ 0.491
```

---

## Cross-references

- Script: `scripts/phase_xi_eta_candidates.py`
- Constants: `sigma_ground/field/constants.py` (PHI, ETA_FORMULA, C_HDE_*, ETA_HDE_*)
- DESI HDE paper: arXiv:2411.08639
- Barrow HDE paper: arXiv:2503.18230
- Kobakhidze ξ: `misc/kobakhidze_xi_check.md`
- Survey: `misc/arxiv_pluggable_survey_2025.md`
