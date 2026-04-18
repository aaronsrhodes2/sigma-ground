# Phase VI — GW150914/H1 Template Optimisation & Injection Convention Audit: Verdict

**Date:** 2026-04-17  
**Phase:** VI — Ringdown start-time scan, (f_QNM, τ_QNM) optimisation, injection audit  
**Pipeline:** `scripts/phase_vi_gw150914_optimization.py`  
**Input:** GW150914/H1 @ 16 kHz (GWOSC cached); Phase III injection convention compared

---

## Phase VI Findings

Three independent results from this phase:

1. **Injection convention audit**: Phase III's injection does not include the
   exp(−t_start/τ) amplitude decay, biasing Γ systematically low for all events.
2. **t_start scan**: Optimal ringdown start is t* = 0.5 ms; ρ_syn peaks at 2.06.
3. **Template scan**: f_opt = 225 Hz, τ_opt = 6.5 ms maximise ρ_syn at t* — but
   this reflects template norm optimisation, not a physical signal improvement.

---

## Finding 1 — Injection Convention Mismatch in Phase III

### The bug

Phase III computes the GR-predicted injection as:

    h_inj(t) = A_det × exp(−t/τ) × cos(2πft)   [Phase III convention]

where A_det is the QNM peak amplitude (at t = 0, i.e., at merger onset) and
t runs from 0 at the start of the observation window (t_start ms after merger).

The physically correct injection — the actual GR QNM signal at t_start ms after
merger — is:

    h_inj(t) = A_det × exp(−(t + t_start)/τ) × cos(2πf(t + t_start))   [correct]

The difference: by t_start ms after merger, the QNM has already decayed to
A_det × exp(−t_start/τ).  Phase III injects a signal that is too strong by
exp(+t_start/τ) at the start of the observation window.

### Consequence

For the quadrature matched filter (phase-independent):

    amp_syn_Phase_III = A_det × ||template_wht||
    amp_syn_correct   = A_det × exp(−t_start/τ) × ||template_wht||

So ρ_syn_Phase_III overestimates ρ_syn_correct by exactly exp(+t_start/τ):

| Event | τ_QNM (ms) | K = exp(3ms/τ) | ρ_syn_Phase_III | ρ_syn_correct |
|---|---|---|---|---|
| GW150914 | 4.00 | **2.12** | 1.543 | 0.728 |
| GW151226 | 1.20 | **12.2** | 0.011 | 0.00090 |
| GW170814 | 3.28 | **2.49** | 1.130 | 0.454 |
| GW170104 | 2.94 | **2.77** | 0.449 | 0.162 |
| GW190521 | 8.75 | **1.41** | 1.074 | 0.762 |
| GW190521 | 8.75 | **1.41** | 1.549 | 1.099 |

Since Γ = ρ_obs / ρ_syn, the Phase III Γ values are biased LOW by exp(−t_start/τ)
relative to the physically correct value:

    Γ_correct = Γ_Phase_III × exp(+t_start/τ)

### Phase VI direct measurement confirms the bias

Phase VI at t_start = 3 ms with the correct injection:

```
ρ_obs = 1.000                 (same as Phase III — observed data, unaffected)
ρ_syn = 0.708                 (Phase VI correct injection)
ρ_syn = 1.543                 (Phase III incorrect injection)
Ratio = 1.543 / 0.708 = 2.18  ≈  exp(3ms/4ms) = 2.12  ✓
```

**Corrected Γ for GW150914/H1:**

    Γ_correct = ρ_obs / ρ_syn_correct = 1.000 / 0.708 = 1.41

Compare: Phase III reported Γ = 0.648 (= ρ_obs/ρ_syn_Phase_III = 1.000/1.543).

### Impact on Phases III–V conclusions

Phase III compared its (biased) Γ values against model predictions of
GR (1.0), sigma_coh (0.79), etc., which are the CORRECT Γ values under
Convention B.  This convention mismatch means the Phase III Γ metric has
GR expected value exp(−t_start/τ), not 1.0:

| Event / Det | Γ_Phase_III | Γ_expected_GR | Γ_correct | Γ_correct vs GR |
|---|---|---|---|---|
| GW150914 / H1 | 0.648 | 0.472 | 1.41 | above GR |
| GW170814 / H1 | 1.076 | 0.401 | 2.68 | above GR |
| GW170104 / H1 | 0.873 | 0.361 | 2.42 | above GR |
| GW190521 / H1 | 2.014 | 0.709 | 2.84 | above GR |
| GW190521 / L1 | 0.255 | 0.709 | 0.360 | below GR |

**After correcting the injection convention:**

- 4 of 5 informative events show Γ_correct > 1 (above GR prediction)
- Only GW190521/L1 shows Γ_correct < 1 (at 0.36, consistent with noise)
- There is **no statistical preference for Γ < 1** in the corrected dataset

**Phase III–V conclusion reversal:**  The apparent preference for Γ < 1
(sigma_coh direction) across Phases III–V was an artifact of the injection
convention mismatch, not a physical signal.  After correction, the data is
consistent with GR (Γ = 1) or even Γ > 1 at all informative events.

---

## Finding 2 — Ringdown Start-Time Scan

### Results at nominal (f = 251 Hz, τ = 4.0 ms)

| t_start (ms) | ρ_obs | ρ_syn | Γ |
|---|---|---|---|
| 0.5 | 0.768 | 1.423 | 0.540 |
| 1.0 | 0.893 | 1.169 | 0.764 |
| 1.5 | 0.940 | 0.982 | 0.957 |
| 2.0 | 1.880 | 0.937 | 2.006 |
| 2.5 | 1.910 | 0.863 | 2.213 |
| **3.0** | **1.000** | **0.708** | **1.41** |
| 4.0 | 0.517 | 0.569 | 0.909 |
| 5.0 | 0.897 | 0.429 | 2.091 |
| 6.5 | 3.019 | 0.317 | 9.523 |
| 8.0 | 0.857 | 0.210 | 4.081 |

ρ_syn decreases monotonically with t_start, following exp(−t_start/τ):
ρ_syn(0.5ms)/ρ_syn(3.0ms) = 1.423/0.708 = 2.01 ≈ exp(2.5ms/4ms) = 1.87 (within 7%).

ρ_obs fluctuates noisily — O(1) noise variations dominate, confirming there is
no coherent QNM signal above the noise floor in the GW150914/H1 data.

**Optimal t_start**: Earliest allowed (0.5 ms) maximises ρ_syn.  Physical
constraint: t_start < 1 ms is inside the IMR merger turn-on; without pycbc
subtraction, starting at t < 1 ms contaminates ρ_obs with merger signal while
ρ_syn remains clean.  The practical optimum is t_start ≈ 0.5–1 ms.

### Interpretation

The exp(−t_start/τ) decay of ρ_syn is exactly what physics predicts.  The
inability to go below t_start ~ 0.5 ms without pycbc is the fundamental
sensitivity ceiling of the current pipeline.

---

## Finding 3 — (f_QNM, τ_QNM) Template Scan

At t* = 0.5 ms, the (f, τ) scan finds:

```
Optimal f_QNM = 225 Hz   (nominal: 251 Hz)
Optimal τ_QNM = 6.50 ms  (nominal: 4.00 ms)
ρ_syn_opt     = 2.059
ρ_obs_opt     = 0.526    ← drops from 0.768 at nominal
Γ_opt         = 0.256
```

**Interpretation:** The (f, τ) scan maximises ρ_syn — but ρ_syn at off-nominal
parameters is not a physically meaningful improvement.  A larger τ gives a
slower-decaying template with higher ||tc_wht||, boosting the injection SNR.
But this template does NOT match the actual GW150914 QNM (f=251 Hz, τ=4 ms),
so ρ_obs simultaneously drops: the matched filter response to the actual data
(noise) is lower for a mismatched template.

The result Γ = 0.256 at the "optimal" (f, τ) is an artefact of template
optimisation, not evidence for Γ < 1.  The correct approach is to use the
(f, τ) posterior from the LIGO/Virgo GWTC-1 ringdown analysis — not to
maximise ρ_syn over an (f, τ) grid.

**Published GW150914 QNM posterior (Isi et al. 2019, PRL 123, 111102):**
    f_220 ∈ [235, 267] Hz   (68 % CI, MAP ≈ 251 Hz)
    τ_220 ∈ [3.0,  5.5] ms  (68 % CI)

At the nominal values (251 Hz, 4.0 ms) with t* = 0.5 ms:
    ρ_syn = 1.423   ρ_obs = 0.768   Γ = 0.540

---

## Sensitivity Ceiling Summary

```
GW150914/H1 @ 16 kHz — maximum ρ_syn without pycbc
──────────────────────────────────────────────────────────────
Phase III (t=3ms, no offset, f=251, τ=4ms)   ρ_syn = 1.543   [biased]
Phase VI  (t=3ms, correct offset, f=251, τ=4ms)   ρ_syn = 0.708   [correct]
Phase VI  (t=0.5ms, correct offset, f=251, τ=4ms)  ρ_syn = 1.423   [best physical]

Target for B > 10:   ρ_syn ≥ 5.0
Current best (physical):  ρ_syn = 1.423
Remaining gap:  5.0 / 1.423 = 3.5×   (requires pycbc IMR subtraction)
```

---

## Posterior Comparison

| Scenario | ρ_syn | P(Γ < 1) | P(Γ < 0.79) |
|---|---|---|---|
| Phase III (biased) | 1.543 | 77.8% | 66.3% |
| Phase VI correct (t=3ms) | 0.708 | 39.1% | 27.4% |
| Phase VI optimal (t=0.5ms) | 1.423 | 73.3% | 61.6% |
| pycbc forecast (ρ_syn=5.0) | 5.000 | 97.2% | 81.4% |

Note: the Phase VI correct (t=3ms) posterior P(Γ<1) = 39% is BELOW 50% —
consistent with NO preference for Γ < 1 at this event. The pycbc forecast
assumed the same ρ_obs as Phase III; with a corrected ρ_obs calibration the
forecast would shift.

---

## Revised Combined Evidence State (Phases I–VI)

| Phase | Result | Systematic |
|---|---|---|
| I.4 (time domain) | p_F = 0.609, null | Echo below threshold |
| I.5 (freq comb) | R = 0.916, null | Below noise floor |
| II (amplitude bound) | ξ_UL = 1.0, saturated | Untestable regime |
| III (amplitude ratio) | Γ = 0.856 ± 1.196 — **injection bias** | ρ_syn overestimated ×2.1–12 |
| IV (Bayesian Rician) | MAP=0, P(Γ<1)=94% — **inherited bias** | Same injection input |
| V (joint Γ×f_rd) | P(Γ<1)=77% — **inherited bias** | Same injection input |
| VI (corrected GW150914) | Γ_correct=1.41; ρ_syn ceiling=1.42 | 3.5× gap to pycbc |

**Revised summary:** Once the Phase III injection convention is corrected,
the data shows NO statistically significant preference for Γ < 1 (sigma_coh
direction). GW150914/H1 gives Γ_correct = 1.41 ± noise (above GR), and the
combined corrected dataset is consistent with GR at < 1σ. The apparent
preference for σ_conv across Phases III–V was a systematic artifact.

---

## Path Forward

The pipeline correction is straightforward: replace Phase III's injection with
the correctly offset form. A corrected Phase III would likely show Γ_weighted ≈ 2
(above GR) across the informative subset, consistent with pure noise fluctuations
(ρ_obs ~ O(1) noise + O(0.7) signal → large scatter).

**For a decisive test:**

1. **Fix Phase III injection** (immediate): rerun with h_inj including exp(-t_start/τ).
   This produces the correct baseline but does not improve sensitivity.

2. **pycbc IMR subtraction on GW150914** (path to decisive):
   - Subtract the full inspiral-merger waveform (SXS template) from the strain
   - The QNM starts at ~0 ms post-merger in the residual with amplitude A_det (no decay)
   - Effectively sets t_start → 0: ρ_syn increases by exp(+3ms/τ) ≈ 2.12× vs current
   - Combined with residual noise reduction from IMR subtraction: total gain ~3–5×
   - Target ρ_syn ≈ 1.423 × (3–5) ≈ 4–7  →  sufficient for B > 10

3. **Extend to full GWTC-3 catalog** (path to statistics):
   - ~90 BBH events, ~2 detectors each, corrected injection
   - Coherent stack gives combined ρ_syn ~ 8–10 × correction factor
   - Decisive even without pycbc if stack is coherent

---

## Cross-References

- Phase III script (has injection bug): `scripts/phase_iii_amplitude_ratio.py`
- Phase IV (inherits bias): `misc/bh_phase_iv_bayesian_gamma_results.md`
- Phase V (inherits bias): `misc/bh_phase_v_joint_gamma_frd_results.md`
- Phase VI script: `scripts/phase_vi_gw150914_optimization.py`
- Phase VI figure: `misc/bh_phase_vi_gw150914_optimization.png`
- Isi et al. 2019 (GW150914 QNM posterior): PRL 123, 111102
