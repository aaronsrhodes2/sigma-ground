# Kobakhidze–Liang ξ Derivation Check

**Date:** 2026-04-17
**Paper:** Kobakhidze & Liang 2024, arXiv:2410.22412 — "Predicting the Dark Matter–Baryon Abundance Ratio"
**Label:** `[SPECULATIVE]`

## What the paper claims

Ω_DM/Ω_B emerges as a ratio of QCD beta-functions in an N=8 composite-axion gauge
group model.  For N=8: Ω_DM/Ω_B = 5.36, giving:

    ξ_Kobakhidze = 1 / (1 + 5.36) = 0.157233...

compared to sigma-ground's empirical anchor:

    ξ_Planck = 0.1582  (Planck 2018, Ω_b/(Ω_b + Ω_c))

## Numerical comparison

| Quantity | Planck 2018 (sigma-ground anchor) | Kobakhidze N=8 | Δ (%) |
|----------|-----------------------------------|----------------|--------|
| ξ | 0.158200 | 0.157233 | −0.611% |
| σ_conv = −ln(ξ) | 1.843895 | 1.850028 | +0.332% |

The 0.6% gap in ξ produces a 0.33% gap in σ_conv (logarithmic compression: d(σ)/dξ = −1/ξ, so Δσ/σ ≈ Δξ/ξ × 1/σ_conv).

## What shifts downstream if ξ = XI_KOBAKHIDZE is adopted

- σ_conv shifts from 1.843895 → 1.850028 (+0.33%)
- THETA = ETA^(1/3): unchanged (ETA is independent of ξ in the current model)
- All γ(σ) mode terminators: unchanged (they are ratios of σ/σ_conv, which cancel)
- QNM corrections: τ_QNM ∝ 1/f_QNM, independent of σ_conv — unchanged
- decoherence_at_horizon: t_Page ∝ M³ η⁻², no direct ξ dependence — unchanged
- Existing test thresholds: all expressed in σ/σ_conv normalised units → **zero tests break**

## Conclusion

The 0.6% match between Kobakhidze N=8 and Planck 2018 is the tightest first-principles
derivation of ξ found in the 2022–2025 arXiv literature.  If the N=8 gauge-group
identification can be justified physically (i.e., sigma-ground's σ-field couples to a
composite-axion sector with 8 colours), then ξ ceases to be an empirical input and
becomes a prediction from an integer.

**Current status:** The identification is speculative.  sigma-ground's σ is a classical
scale field, not a gauge-theory coupling.  The matching could be numerological coincidence.
The constants `XI_KOBAKHIDZE` and `SIGMA_CONV_KOBAKHIDZE` are exported from
`sigma_ground/field/constants.py` for future comparison but do not replace the canonical
`XI = 0.1582` used in all calculations.

## Why ξ = 0.1582 is the model's most promising anchor (per author note)

The baryon fraction ξ is the single free parameter sigma-ground introduces.  Unlike η
(which the author notes is a "hobby number" related to the golden spiral), ξ comes
directly from CMB observations and is the physical basis for σ_conv = −ln(ξ).  Any
theoretical derivation of ξ from first principles would convert sigma-ground's one
empirical input into a prediction — the most significant possible upgrade to the model's
status.  Kobakhidze N=8 is the strongest candidate for that derivation found so far.

## Cross-references

- Constants: `sigma_ground/field/constants.py` (XI_KOBAKHIDZE, SIGMA_CONV_KOBAKHIDZE)
- Survey: `misc/arxiv_pluggable_survey_2025.md`
