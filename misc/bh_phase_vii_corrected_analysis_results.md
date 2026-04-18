# Phase VII — Corrected Pipeline Verdict

**Date:** 2026-04-17  
**Phase:** VII — exp(−t_start/τ) injection bias corrected across all events  
**Pipeline:** `scripts/phase_vii_corrected_analysis.py`  
**Input:** Phase III raw data + per-event K = exp(−t_start/τ_QNM) correction

---

## Correction Applied

Phase VI identified that Phases III–V injected h_inj = A_det × exp(−t/τ) cos(2πft)
without the exp(−t_start/τ) amplitude decay that the physical GR QNM carries by
the time the observation window opens at t_start = 3.0 ms after merger.

**Correction:**

    ρ_syn_corrected = ρ_syn_Phase_III × K_i
    K_i = exp(−t_start / τ_QNM_i)

ρ_obs is unchanged (measured from data, unaffected by injection convention).

### Per-event correction factors and Γ values

| Event / Det | τ (ms) | K | ρ_obs | ρ_syn_bias | ρ_syn_corr | Γ_bias | Γ_corr |
|---|---|---|---|---|---|---|---|
| GW150914 / H1 | 4.00 | 0.472 | 1.000 | 1.5430 | 0.7289 | 0.648 | **1.37** |
| GW150914 / L1 | 4.00 | 0.472 | 0.734 | 0.1764 | 0.0833 | 4.16 | noise |
| GW151226 / H1 | 1.20 | 0.082 | 1.458 | 0.0114 | 0.0009 | noise | noise |
| GW151226 / L1 | 1.20 | 0.082 | 1.205 | 0.0016 | 0.0001 | noise | noise |
| GW170814 / H1 | 3.28 | 0.401 | 1.216 | 1.1302 | 0.4528 | 1.076 | **2.69** |
| GW170814 / L1 | 3.28 | 0.401 | 1.059 | 0.2132 | 0.0854 | 4.97 | noise |
| GW170814 / V1 | 3.28 | 0.401 | 1.402 | 0.0196 | 0.0079 | noise | noise |
| GW170104 / H1 | 2.94 | 0.360 | 0.392 | 0.4493 | 0.1619 | 0.872 | 2.42 |
| GW170104 / L1 | 2.94 | 0.360 | 0.875 | 0.0566 | 0.0204 | noise | noise |
| GW190521 / H1 | 8.75 | 0.710 | 2.164 | 1.0741 | 0.7623 | 2.015 | **2.84** |
| GW190521 / L1 | 8.75 | 0.710 | 0.395 | 1.5491 | 1.0995 | 0.255 | **0.359** |

Events with ρ_syn_corr > 0.5 (marked ★ in script) are the only informative ones:
GW150914/H1, GW190521/H1, GW190521/L1.

**Key shift:** After correction, only ONE event (GW190521/L1) shows ρ_obs < ρ_syn
and Γ < 1. All other informative events show Γ > 1. The biased analysis had
GW150914/H1 (Γ=0.648) driving the Γ < 1 signal; corrected, it shows Γ=1.37.

---

## Corrected Phase IV Results (Bayesian Rician on Γ)

```
Biased (Phases III–V):
  MAP Γ = 0.000   68% HDI = [0.115, 0.759]   95% HDI = [0.017, 1.156]
  P(Γ < 1.0) = 0.943   ΔlnL sigma_coh/GR = +0.679   B = 1.97

Corrected (Phase VII):
  MAP Γ = 0.000   68% HDI = [0.199, 1.299]   95% HDI = [0.031, 1.967]
  P(Γ < 1.0) = 0.710   ΔlnL sigma_coh/GR = +0.183   B = 1.20
```

### What changed

| Quantity | Biased | Corrected | Change |
|---|---|---|---|
| 68% HDI width | 0.644 | 1.100 | 1.7× wider |
| GR inside 68% HDI? | No (barely outside) | **Yes** | Qualitative flip |
| P(Γ < 1.0) | 94.3% | 71.0% | −23 pp |
| B(sigma_coh/GR) | 1.97 | 1.20 | −39% |

GR (Γ=1.0) is now well within the 68% credible interval.

### Per-event ΔlnL (corrected)

| Event / Det | ρ_syn_corr | ΔlnL | Direction |
|---|---|---|---|
| GW150914 / H1 ★ | 0.729 | +0.052 | sigma_coh |
| GW170814 / H1 | 0.453 | +0.011 | sigma_coh |
| GW170104 / H1 | 0.162 | +0.005 | sigma_coh |
| GW190521 / H1 ★ | 0.762 | **−0.094** | GR |
| GW190521 / L1 ★ | 1.099 | **+0.208** | sigma_coh |
| All others | < 0.09 | ≈ 0 | Γ-flat |
| **Combined** | — | **+0.183** | B = 1.20 |

The entire corrected signal rests on GW190521/L1 alone (+0.208). GW190521/H1
nearly cancels it (−0.094). GW150914/H1 — previously the most informative
event — now contributes only +0.052 because ρ_syn_corr = 0.729 < 1.

---

## Corrected Phase V Results (Joint Γ × f_rd)

```
MAP Γ   = 0.000   68% HDI = [0.281, 1.993]   95% HDI = [0.040, 3.184]
P(Γ < 1.0) = 0.529   (biased was 0.772)

MAP f_rd = 0.030   68% HDI f_rd = [0.037, 0.127]   (unchanged from Phase V)
```

**P(Γ < 1) = 0.529 is statistically indistinguishable from a coin flip.**

The f_rd marginal posterior is essentially unchanged from Phase V — the data
still prefers f_rd ≈ 0.04–0.09, well below the reference 0.15. This is a
robust result independent of the injection convention.

---

## Summary: Full Phase I–VII Arc

| Phase | Key number | Verdict |
|---|---|---|
| I.4 (echoes, time) | p_F = 0.609 | Null; below threshold |
| I.5 (echoes, freq) | R = 0.916 | Null; below noise floor |
| II (amplitude bound) | ξ_UL = 1.0 | Untestable; pipeline too weak |
| III (amplitude ratio) | Γ = 0.856 ± 1.196 | **Biased** by K per event |
| IV (Bayesian Rician) | P(Γ<1)=94%, B=1.97 | **Inherited bias**; corrected here |
| V (joint Γ×f_rd) | P(Γ<1)=77% | **Inherited bias**; corrected here |
| VI (injection audit) | ρ_syn wrong by ×2.1–12 | Bias identified and quantified |
| **VII (corrected)** | **P(Γ<1)=53%, B=1.20** | **No evidence for σ_conv** |

### Final corrected statement

**After removing the injection bias, the 11-event LIGO/Virgo dataset provides
no statistically significant evidence for or against the σ_conv / γ-mode
prediction (Γ < 1).  The corrected Bayes factor sigma_coh/GR = 1.20 is
"not worth mentioning" (Jeffreys scale).  P(Γ < 1) = 53% after f_rd
marginalisation is statistically equivalent to a coin flip.**

The sole remaining source of non-zero ΔlnL is GW190521/L1 (ρ_obs = 0.395 vs
ρ_syn_corr = 1.099), which is a single noise-floor observation against a
ρ_syn ~ 1 prediction.  The contradicting GW190521/H1 (ΔlnL = −0.094) nearly
cancels it from the same physical event.

The prior preference for Γ < 1 across Phases III–V was entirely explained by
the injection convention mismatch (K = 0.40–0.47 for most events), which
systematically shrunk all Γ values by those factors before comparison against
the σ_conv predictions (0.75–0.84).

---

## f_rd Constraint (robust, independent of bias)

The marginal posterior on f_rd is unchanged by the injection correction:

```
68% HDI f_rd = [0.037, 0.127]
95% HDI f_rd = [0.031, 0.185]
f_rd_reference = 0.15  →  sits at 86th percentile of corrected marginal
```

This result is robust: it comes from the shape of the joint likelihood in
(Γ, f_rd) space, where the Γ–f_rd degeneracy (Γ × √f_rd = const) is
independent of the injection convention. The data independently prefers
f_rd ≈ 0.04–0.09, which is physically consistent with NR expectations for
the mass ratios and spins in this catalog. This is a genuine observational
constraint on the ringdown energy fraction, separate from the Γ question.

---

## Path Forward

The corrected pipeline establishes a clean null: no evidence for σ_conv in the
current 5-event, 11-detector-event dataset. To make a positive test:

1. **pycbc IMR subtraction (GW150914/H1)**: Subtract the inspiral-merger template
   before applying the QNM matched filter.  In the residual, the QNM starts
   at near-peak amplitude (small effective t_start) → ρ_syn ≈ 3–5.  This is
   the only way to break the ρ_syn ceiling without new events.

2. **Full GWTC-3 coherent stack (~90 BBH events)**: With corrected injection,
   the per-event ρ_syn_corr is 2–5× smaller than before.  The required stack
   size grows by 4–25×, pushing the needed catalog from ~312 events (Phase II
   estimate) to ~600–7800 events — likely requiring O3+O4 combined.

3. **NR f_rd priors per event**: If f_rd is independently constrained from NR
   (per event, Gaussian σ ~ 0.02), the Γ–f_rd banana degeneracy contracts and
   the remaining B = 1.20 Bayes factor can be properly interpreted.

---

## Cross-References

- Phase VI injection audit: `misc/bh_phase_vi_gw150914_optimization_results.md`
- Phase IV (biased): `misc/bh_phase_iv_bayesian_gamma_results.md`
- Phase V (biased): `misc/bh_phase_v_joint_gamma_frd_results.md`
- Phase VII script: `scripts/phase_vii_corrected_analysis.py`
- Phase VII figure: `misc/bh_phase_vii_corrected_analysis.png`
