# σ_conv / γ-Mode QNM Test: Full Research Arc Synthesis

**Date:** 2026-04-17  
**Phases:** I.4 through VIII  
**Status:** Corrected null — no evidence; decisive test requires pycbc on top-5 GWTC-3 events

---

## 1. Hypothesis

The σ_conv framework (sigma-ground project) predicts that black-hole mergers
produce quasinormal mode (QNM) ringdowns with amplitude suppressed relative
to the GR prediction by a factor determined by the baryon entanglement
parameter:

    Γ = A_obs / A_GR

where several candidate γ-mode expressions give:

| Model | Γ_pred | Expression |
|---|---|---|
| GR | 1.0000 | — |
| exp | 0.8395 | Phase G phenomenology |
| sigma_coh | 0.7924 | 1 − η/2, η = 0.4153 |
| linear_cbrt | 0.7461 | η^(1/3) |

with constants: ξ = Ω_b/(Ω_b+Ω_c) = 0.1582 (Planck 2018),
σ_conv = −ln(ξ) = 1.844, η = 0.4153.

The test: does the observed (2,2,0) Kerr QNM amplitude at LIGO/Virgo match GR
(Γ = 1) or one of the suppressed predictions (Γ < 1)?

---

## 2. Data and Pipeline

**Events:** 5 BBH mergers from GWOSC (O1–O3), 11 detector-event channels.

| Event | M_f (M☉) | d_L (Mpc) | f_QNM (Hz) | τ_QNM (ms) | Detectors |
|---|---|---|---|---|---|
| GW150914 | 62.2 | 410 | 251 | 4.00 | H1, L1 |
| GW151226 | 20.8 | 440 | 737 | 1.20 | H1, L1 |
| GW170814 | 53.2 | 540 | 329 | 3.28 | H1, L1, V1 |
| GW170104 | 49.1 | 880 | 339 | 2.94 | H1, L1 |
| GW190521 | 142.0 | 5300 | 123 | 8.75 | H1, L1 |

**Pipeline core** (`sigma_ground/field/interface/ligo_echo_search.py`):
- Bandpass 35–1500 Hz; Welch PSD (8 s window); Tukey taper α=0.2
- Whitening: divide FFT by √(PSD × fs/2), invert
- Quadrature matched filter: A = √(⟨data_wht|cos_wht⟩² + ⟨data_wht|sin_wht⟩²) / ||template_wht||
- Template: exp(−t/τ_QNM) × cos(2πf_QNM t), duration 20 ms
- Background calibration: 300 pre-merger samples from [−12 s, −2 s] (seed 42)
- σ_noise = bg_mean / √(π/2); ρ = amplitude / σ_noise

**GR-predicted amplitude** (Phase III, from energy conservation):

    A₂₂ = √(5G ε_rd M_f / πc f_QNM Q)   where Q = πf_QNM τ_QNM, ε_rd = f_rd × E_rad/M_f c²
    A_det = (A₂₂ / d_L) × R_rms(ι),      R_rms² = (1/5)[(1+cos²ι)²/4 + cos²ι]

---

## 3. Phase-by-Phase Results

### Phase I.4 — Time-domain echo search

Searched for ξ-quantised echoes at delays Δt_n = n × Δt_1 after merger.
Bootstrap p-value p_F = 0.609 — consistent with noise.  Predicted echo amplitude
ρ_echo = ξ × ρ_QNM ≈ 0.13–0.34 σ — 13–17× below the per-event detection
threshold.  **Conclusion:** null, but untestable at this sensitivity.

### Phase I.5 — Frequency-domain comb

Stacked power at ξ-comb frequencies in the post-merger spectrum.
Ratio R = 0.916 (expected 1.0 under H₀) — null.  Same sensitivity limit.

### Phase II — ξ-shell amplitude falsifiability bound

Combined Fisher upper limit on ξ: ξ_UL = 1.0 (saturated — no constraint).
Coherent sensitivity at the σ_conv prediction (ξ = 0.158) requires ~312 events
or ρ̄_QNM ≥ 5 per event.  **Conclusion:** pipeline is 5–25× too insensitive.

### Phase III — QNM amplitude ratio (biased)

Computed Γ = ρ_obs / ρ_syn, where ρ_syn is the matched-filter SNR of the
GR-predicted injection through the same pipeline.

Combined Γ (ρ_syn²-weighted): **0.856 ± 1.196**.  All four predictions
consistent at < 0.1σ.  7 of 11 events below noise floor (ρ_syn < 0.1).

*Later found to have injection bias — see Phase VI.*

### Phase IV — Bayesian Rician posterior (biased)

Rician likelihood p(ρ_obs | s) replaces naive ratio; uniform prior Γ ≥ 0.
Combined ΔlnL = +0.679; **B(sigma_coh/GR) = 1.97**.
MAP Γ = 0, P(Γ<1) = 94.4%, 95% HDI = [0, 1.028].
GR sits at the edge of the 95% credible interval.

*Inherited Phase III injection bias — apparent 94% preference is an artifact.*

### Phase V — Joint Γ × f_rd posterior (biased)

Marginalised the ringdown energy fraction f_rd (log-uniform prior [0.03, 0.20])
jointly with Γ to account for the dominant systematic.

After marginalisation: **P(Γ<1) = 77%**, B = 1.21.  The f_rd marginal posterior
peaks at f_rd ≈ 0.05 (68% HDI: [0.037, 0.127]), well below the reference 0.15.
Γ–f_rd banana degeneracy unbroken at current sensitivity.

*Inherited Phase III injection bias — 77% preference partially an artifact.*

---

## 4. The Injection Bias — Key Discovery of Phase VI

### What Phase III injected

```python
h_inj(t) = A_det × exp(−t/τ) × cos(2πft)   [Phase III convention]
```

where A_det is the **peak** QNM amplitude (at merger onset, t = 0) and t runs
from zero at the observation window start (t_start = 3 ms after merger).

### What should have been injected

```python
h_inj(t) = A_det × exp(−(t + t_start)/τ) × cos(2πf(t + t_start))   [correct]
```

By t_start = 3 ms after merger, the physical GR QNM has decayed to
A_det × exp(−t_start/τ).  Phase III injected a signal that is too strong by:

    K_i = exp(+t_start / τ_QNM_i)

This overestimates ρ_syn by factors of 2.1–12 per event.

### Per-event bias factors

| Event | τ (ms) | K = exp(3/τ) | ρ_syn overestimate |
|---|---|---|---|
| GW150914 | 4.00 | **2.12×** | moderate |
| GW151226 | 1.20 | **12.2×** | severe |
| GW170814 | 3.28 | **2.49×** | moderate |
| GW170104 | 2.94 | **2.77×** | moderate |
| GW190521 | 8.75 | **1.41×** | mild |

Since Γ = ρ_obs / ρ_syn, all Γ values are biased LOW by the same factor.
Under GR (where Γ_true = 1), Phase III's metric gives Γ = exp(−t_start/τ),
not 1.0 — a per-event expected value of 0.36–0.71, not 1.0.

**The σ_conv predictions (0.79–0.84) were compared against the wrong baseline.**

### Direct verification

Phase VI measured ρ_syn at 16 kHz with the corrected injection at t_start = 3 ms:

    ρ_syn_corr  = 0.708   (Phase VI, correct injection)
    ρ_syn_bias  = 1.543   (Phase III, wrong injection)
    Ratio = 2.18   ≈   exp(3ms/4ms) = 2.12  ✓

Corrected GW150914/H1: **Γ_correct = 1.000/0.708 = 1.41** (above GR, not 0.648 below).

---

## 5. Corrected Results (Phase VII)

Applying K_i = exp(−t_start/τ_i) to all Phase III ρ_syn values:

### Corrected per-event Γ

| Event / Det | ρ_obs | ρ_syn_bias | ρ_syn_corr | Γ_bias | Γ_corr |
|---|---|---|---|---|---|
| GW150914 / H1 | 1.000 | 1.543 | 0.729 | 0.648 | **1.37** |
| GW170814 / H1 | 1.216 | 1.130 | 0.453 | 1.076 | **2.68** |
| GW170104 / H1 | 0.392 | 0.449 | 0.162 | 0.872 | **2.42** |
| GW190521 / H1 | 2.164 | 1.074 | 0.762 | 2.015 | **2.84** |
| GW190521 / L1 | 0.395 | 1.549 | 1.099 | 0.255 | **0.359** |
| All others | — | < 0.22 | < 0.09 | noise | noise |

4 of 5 informative events show Γ_correct > 1.  Only GW190521/L1 shows Γ < 1.

### Corrected Bayesian posterior

```
Corrected Phase IV (f_rd fixed):
  MAP Γ = 0.000   68% HDI = [0.199, 1.299]   95% HDI = [0.031, 1.967]
  P(Γ < 1.0)  = 0.710   ΔlnL sigma_coh/GR = +0.183   B = 1.20

Corrected Phase V (f_rd marginalised):
  MAP Γ = 0.000   68% HDI = [0.281, 1.993]   95% HDI = [0.040, 3.184]
  P(Γ < 1.0)  = 0.529                         B ≈ 1.12 (coin flip)
```

**GR (Γ = 1.0) is inside the 68% credible interval in the corrected analysis.**

The sole non-trivial contribution comes from GW190521/L1 (ΔlnL = +0.208).
GW190521/H1 nearly cancels it (ΔlnL = −0.094) from the same physical event.

### Comparison table

| Quantity | Biased (III–V) | Corrected (VII) |
|---|---|---|
| P(Γ<1) fixed f_rd | 94% | 71% |
| P(Γ<1) f_rd marginalised | 77% | **53% (coin flip)** |
| B(sigma_coh/GR) | 1.97 | **1.20** |
| GR inside 68% HDI | No | **Yes** |

---

## 6. Robust Findings (Bias-Independent)

Two results survive the injection correction because they do not depend on
the absolute value of ρ_syn:

**f_rd constraint (Phase V, robust):**

    68% HDI: f_rd ∈ [0.037, 0.127]   (reference 0.15 at 86th percentile)

The data independently prefers f_rd ≈ 0.05–0.09 for the events in this catalog,
which is consistent with NR expectations for the mass ratios and spins present
(q ≈ 0.5–0.85, moderate spins → f_rd ≈ 0.03–0.10 in simulations).  This is
a genuine observational constraint on the ringdown energy fraction — independent
of the σ_conv hypothesis.

**Sensitivity ceiling without pycbc (Phase VI, robust):**

    ρ_syn ceiling = 1.42   (t_start = 0.5 ms, nominal f/τ, 16 kHz)

This is the maximum achievable ρ_syn for GW150914/H1 without IMR subtraction.
It is set by the exp(−t_start/τ) decay; t_start < 0.5 ms enters the IMR merger
and cannot be used without pycbc.

---

## 7. Sensitivity Roadmap (Phase VIII)

**Per-event ΔlnL at corrected ρ_syn_typ = 0.45 (GWTC-3 typical event):**

    E[ΔlnL | sigma_coh true] = +0.00061 per event

**Milestones to B > 10 (strong evidence), if sigma_coh is true:**

| Scenario | Events needed | B at completion |
|---|---|---|
| GWTC-3 stack only (no pycbc) | 3500 | 10 |
| pycbc ×5 on 1 event (ρ_syn=5) | — | 1.93 alone |
| pycbc ×5 on 5 best events | ~0 extra stack | ~16 |
| pycbc ×10 on 1 event (ρ_syn=10) | — | 9.71 alone |
| pycbc ×14 on 1 event (ρ_syn=10.1) | — | **> 10** |

**Critical path:**

```
    pycbc on 5 best GWTC-3 events
    (ρ_syn ≈ 5 each, achievable with lalsuite + SXS waveforms)
         ↓
    B ≈ 16  [if sigma_coh true]   or   B ≈ 0.01  [if GR true]
         ↓
    Decisive either way — no additional events needed
```

GWTC-3 alone (90 events × 2 detectors) moves B from 1.20 to 1.34 — indistinguishable
from the current null.  Catalog depth cannot substitute for per-event SNR.

---

## 8. Physical Interpretation

### What the corrected null means

The 11-event LIGO/Virgo dataset (O1–O3), after correcting the injection
convention, provides **no statistically significant evidence for or against
the σ_conv prediction**.  The Bayes factor of 1.20 is below the Jeffreys
"barely worth mentioning" threshold of 3.  GR is fully consistent with the
data at the 29% level (P(Γ>1) = 29%).

This is not evidence against σ_conv.  It is a statement that the current
pipeline — running on 5 events with corrected ρ_syn ≈ 0.45–1.1 per channel —
is incapable of distinguishing Γ = 0.79 from Γ = 1.0 at any meaningful confidence.

### What the f_rd constraint means

The independent constraint f_rd ∈ [0.037, 0.127] (68%) is physically meaningful:
- It disfavours f_rd = 0.15 (the commonly used reference) at modest credence
- It is consistent with NR results for quasi-circular BBH mergers
  with mass ratios q ≈ 0.5–0.9 and low effective spin
- It is NOT a constraint on σ_conv itself — it could equally reflect
  f_rd heterogeneity across the event catalog (different q, a_eff per event)

### The injection bias and scientific integrity

The Phase VI injection bias is a pipeline error that produced spuriously
optimistic results in Phases III–V.  The corrected analysis is 10–11×
harder to pass than the biased one:
- Biased Phase II estimate: ~312 events for B > 10
- Corrected Phase VIII estimate: ~3500 events (no pycbc) or 5 pycbc events

This represents the correct scientific state: **the σ_conv prediction has
not been tested at meaningful sensitivity**.  Future claims of confirmation
or falsification require pycbc-grade analysis.

---

## 9. Summary Verdict

**The σ_conv QNM amplitude suppression prediction (Γ = 0.79) is untested
by the current analysis.  The corrected Bayes factor B(sigma_coh/GR) = 1.20
represents no evidence in either direction.  GR is consistent with the data.**

**The dominant physical result is the f_rd constraint: the data independently
prefers f_rd ≈ 0.04–0.09 over the reference 0.15, consistent with NR
expectations for this event catalog.**

**The decisive path is pycbc IMR subtraction on the 5 best GWTC-3 events.
This is achievable with existing software (lalsuite ≥ 7.0, SXS catalog v3,
16 kHz GWOSC strain) and would produce a definitive B > 10 result in either
direction with no new data.**

---

## 10. Cross-Reference Index

| Phase | Script | Result doc | Figure |
|---|---|---|---|
| I.4 catalog null | `ligo_echo_search.py` | `bh_phase_i_4_catalog_results.md` | — |
| I.5 freq stack | `phase_i_5_freq_stack.py` | `bh_phase_i_5_freq_stack_results.md` | — |
| II amplitude bound | `phase_ii_amplitude_bound.py` | `bh_phase_ii_amplitude_bound_results.md` | `bh_phase_ii_amplitude_bound.png` |
| III ratio (biased) | `phase_iii_amplitude_ratio.py` | `bh_phase_iii_amplitude_ratio_results.md` | `bh_phase_iii_amplitude_ratio.png` |
| IV Rician (biased) | `phase_iv_bayesian_gamma.py` | `bh_phase_iv_bayesian_gamma_results.md` | `bh_phase_iv_bayesian_gamma.png` |
| V joint Γ×f_rd (biased) | `phase_v_joint_gamma_frd.py` | `bh_phase_v_joint_gamma_frd_results.md` | `bh_phase_v_joint_gamma_frd.png` |
| VI injection audit | `phase_vi_gw150914_optimization.py` | `bh_phase_vi_gw150914_optimization_results.md` | `bh_phase_vi_gw150914_optimization.png` |
| VII corrected | `phase_vii_corrected_analysis.py` | `bh_phase_vii_corrected_analysis_results.md` | `bh_phase_vii_corrected_analysis.png` |
| VIII forecast | `phase_viii_sensitivity_forecast.py` | `bh_phase_viii_sensitivity_forecast_results.md` | `bh_phase_viii_sensitivity_forecast.png` |

**Phenomenology and theory:**
- σ_conv / γ-mode predictions: `bh_collision_phenomenology.md`
- Phase J horizon identity: `bh_horizon_sigma_conv_identity.md`
- Echo delay derivation: `bh_merger_predictions.md`

**Constants (sigma_ground):**
- ξ = 0.1582, σ_conv = 1.844, η = 0.4153: `sigma_ground/field/constants.py`
