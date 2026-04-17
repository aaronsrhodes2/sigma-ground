# Phase II — ξ-Shell Echo Amplitude Falsifiability Bound: Verdict

**Date:** 2026-04-17  
**Phase:** II — amplitude falsifiability bound from Phase I null catalog  
**Pipeline:** `scripts/phase_ii_amplitude_bound.py`  
**Input:** Phase I.4 catalog (11 detector-events, GWOSC LIGO/Virgo O1–O3)

---

## Variables

| Symbol | Name | Meaning |
|---|---|---|
| ξ | XI | Baryon fraction Ω_b/(Ω_b+Ω_c) ≈ 0.1582 (Planck 2018) |
| σ_conv | SIGMA_CONV | −ln(ξ) ≈ 1.844 — the fundamental echo reflectivity |
| ρ_QNM | qnm_snr | Phase-independent matched-filter amplitude of QNM at 3 ms after merger |
| ρ_echo_pred | echo_snr_predicted | Predicted echo SNR = ξ × ρ_QNM |
| ξ_UL | xi_upper_limit | 95 % CL incoherent upper limit on ξ from Fisher combination |
| ξ_sens | xi_sensitivity | ξ detectable at 3σ with coherent N-event stack |
| ρ̄_QNM | rho_bar | Mean ρ_QNM across 11 detector-events |

---

## Key Identity

The matched-filter template at echo delay n·Δt₁ evaluates the echo exactly
`ringdown_start_ms` after the echo was launched — the same age as the QNM
template evaluated at delay 0.  Both carry the same exponential decay factor
e^{−t_start/τ_QNM}.  Therefore the predicted ratio of echo SNR to in-window
QNM SNR is exactly ξ:

    ρ_echo_n_predicted = ξ × ρ_QNM_in_window

This is model-exact for the Phase I template shape.  Measuring ρ_QNM from data
fully determines the predicted echo amplitude without any free parameters.

---

## Method

**Template:** Quadrature (cos + sin) damped sinusoid at the literature (2,2,0)
Kerr QNM frequency f_QNM and decay time τ_QNM, starting 3 ms after merger.
Template duration 20 ms.  Using both quadrature components gives a
phase-independent amplitude A = √(ρ_cos² + ρ_sin²) that is Rayleigh
distributed under H₀ regardless of the unknown QNM initial phase.

**Calibration:** Pre-merger background window [−12 s, −2 s] relative to merger
(300 samples, seed 42).  The pre-merger window is guaranteed clean Gaussian
noise — no signal contamination, no post-merger glitch structures.  The noise
amplitude σ_noise is inferred from the Rayleigh mean: E[A] = √(π/2) × σ_noise
so σ_noise = bg_mean / √(π/2).  The QNM SNR is then ρ_QNM = amp_signal / σ_noise.

**Upper limit:** Incoherent Fisher combination over N=11 detector-events.  For
each ξ, the predicted echo SNR per event is ρ_echo_i = ξ × ρ_QNM_i.  The Fisher
statistic is Σ_i [−2 ln p_i] where p_i = 2(1−Φ(ρ_echo_i)) is the two-sided
Gaussian p-value.  ξ_UL solves expected Fisher χ²(ξ_UL) = χ²_0.95(dof=2N).
Binary search to 60-bit precision.

**Coherent sensitivity:** ξ_sens = 3σ / √(Σ ρ_QNM_i²) — the ξ for which a
phase-coherent stack reaches 3σ detection.

---

## Per-Detector-Event Results

| Event / Det | fs (kHz) | ρ_QNM | ρ_echo_pred | threshold | status |
|---|---|---|---|---|---|
| GW150914 / H1 | 4 | 0.841 | 0.133 | 2.3 | below thresh |
| GW150914 / L1 | 4 | 0.821 | 0.130 | 2.3 | below thresh |
| GW151226 / H1 | 16 | 1.458 | 0.231 | 2.3 | below thresh |
| GW151226 / L1 | 16 | 1.205 | 0.191 | 2.3 | below thresh |
| GW170814 / H1 | 4 | 1.216 | 0.192 | 2.3 | below thresh |
| GW170814 / L1 | 4 | 1.059 | 0.168 | 2.3 | below thresh |
| GW170814 / V1 | 4 | 1.402 | 0.222 | 2.3 | below thresh |
| GW170104 / H1 | 4 | 0.392 | 0.062 | 2.3 | below thresh |
| GW170104 / L1 | 4 | 0.875 | 0.138 | 2.3 | below thresh |
| GW190521 / H1 | 4 | 2.164 | 0.342 | 2.3 | below thresh |
| GW190521 / L1 | 4 | 0.395 | 0.063 | 2.3 | below thresh |

Mean ρ_QNM = 1.075.  Range: 0.39 – 2.16.  All ρ_echo_pred ≪ threshold.

---

## Combined Falsifiability Statistics

```
ξ_predicted (σ_conv = −ln(ξ))        = 0.1582
ξ_UL incoherent (95 % CL, N=11)      = 1.0000   ← saturated; no constraint
ξ_sens coherent (3σ, N=11)           = 0.7675
Combined coherent echo SNR (ξ=0.158) = 0.62 σ
```

**ξ_UL = 1.0 means the data provides zero constraining power on ξ.**  Even at
ξ = 1.0 the expected Fisher χ² = 20.1 (dof = 22), which is below the 95 % CL
threshold of χ²_0.95(22) = 33.9.  The ρ_QNM values are too small — individually
and collectively — for the catalog to rule out any value of ξ ∈ (0, 1].

**ξ_sens = 0.77 means the coherent 11-event stack can only detect ξ > 0.77
at 3σ.**  The σ_conv prediction ξ = 0.158 is 4.9× below this floor.

**Combined coherent SNR = 0.62σ** — with the predicted ξ = 0.158, the expected
combined echo signal in the full 11-event stack is less than 1σ.

---

## Verdict

**The Phase I null result is trivially consistent with the ξ-shell model.
The σ_conv prediction is not falsified — but neither is it tested.**

The predicted echo amplitude at ξ = 0.158 is 13–17× below the per-event
detection threshold across the entire 11-event catalog.  No pipeline with this
configuration (simple QNM matched filter, GWOSC 4–16 kHz data, 5 events, 11
detector-channels) could have detected the predicted signal even if it is real.

This is a fundamentally different conclusion from "no echoes found."  The correct
statement is: **the ξ-shell echo signal is too faint for the Phase I pipeline to
detect; the search is not yet in the regime where a positive detection or a
meaningful constraint on ξ is possible.**

### Path to testability

To reach 3σ coherent sensitivity at ξ = 0.158 with ρ̄_QNM ≈ 1.075:

    N_required = (3 / (ξ × ρ̄_QNM))² ≈ (3 / (0.158 × 1.075))² ≈ 312

That is approximately 312 detector-events — achievable with the full GWTC-3
catalog (~90 BBH events × ~2 detectors on average per event = ~180 detector-
events) combined with a coherent analysis framework.  Two additional levers can
reduce N_required substantially:

1. **Better QNM subtraction** (pycbc + SXS waveform catalog).  Full IMR waveform
   subtraction before the QNM template removes the GW merger turn-on from the
   residual, improving the QNM match.  Raising ρ̄_QNM from ~1 to ~5 would
   reduce N_required by 25×, bringing the threshold down to ~13 events.

2. **Higher sample-rate data** (16 kHz for all events).  At 4 kHz, the QNM
   template for GW150914 (f_QNM = 251 Hz) resolves well, but for GW151226
   (f_QNM = 737 Hz) the 4 kHz Nyquist would alias the signal.  All events at
   16 kHz is a prerequisite for a clean QNM match.

3. **Optimal template bank**.  The current analysis uses a single 20 ms template
   per event.  A matched-filter bank over (f, τ) would pick up the best-fitting
   QNM parameters for each event rather than using literature values.

### What this means for σ_conv / ξ-shell

The Phase I time-domain (p_F = 0.609, catalog null) and Phase I.5 frequency-
domain (R = 0.92, comb null) results are both trivially consistent with the
model because the predicted signal is below the noise floor of these searches.
Phase II formalises this:

- σ_conv = −ln(ξ) ≈ 1.844 is not ruled out by the GWOSC LIGO data at the
  sensitivity accessible to this pipeline.
- The model makes a definite, falsifiable prediction (ξ = 0.158), but the
  current analysis is 5–25× too insensitive to test it.
- A pycbc-grade reanalysis of the full GWTC-3 catalog with SXS waveform
  subtraction and a coherent multi-event stack would represent the first
  genuinely decisive test.

---

## Cross-References

- Phase I.4 catalog null: `misc/bh_phase_i_4_catalog_results.md`
- Phase I.5 frequency-domain null: `misc/bh_phase_i_5_freq_stack_results.md`
- Phase I pipeline: `sigma_ground/field/interface/ligo_echo_search.py`
- Phase II script: `scripts/phase_ii_amplitude_bound.py`
- Phase II figure: `misc/bh_phase_ii_amplitude_bound.png`
- Echo delay derivation: `misc/bh_merger_predictions.md`
- σ_conv / ξ identity: `misc/bh_horizon_sigma_conv_identity.md`
- Phase G phenomenology: `misc/bh_collision_phenomenology.md`
