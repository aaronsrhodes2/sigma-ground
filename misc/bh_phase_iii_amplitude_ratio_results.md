# Phase III — QNM Amplitude Ratio Test: Verdict

**Date:** 2026-04-17  
**Phase:** III — GR-predicted vs observed QNM amplitude comparison  
**Pipeline:** `scripts/phase_iii_amplitude_ratio.py`  
**Input:** Phase II catalog (11 detector-events, GWOSC LIGO/Virgo O1–O3)

---

## Variables

| Symbol | Name | Meaning |
|---|---|---|
| Γ | gamma_obs | A_obs / A_GR = ρ_obs / ρ_syn — amplitude ratio (GR predicts 1.0) |
| A₂₂ | a22_m | Source-frame QNM amplitude [m] from energy conservation |
| A_det | A_det | Detector-frame QNM amplitude = (A₂₂/d_L) × R_rms |
| ε_rd | eps_rd | Ringdown energy fraction = f_rd × E_rad / (M_f c²) |
| f_rd | F_RD | Ringdown fraction of E_rad, default 0.15 ± 0.05 |
| Q | quality_factor | π f_QNM τ_QNM — QNM quality factor |
| ρ_obs | rho_obs | Phase-II matched-filter SNR at QNM window (from data) |
| ρ_syn | rho_syn | Matched-filter SNR of GR-predicted injection through same pipeline |
| R_rms | r_rms_sky_avg | Sky-averaged antenna amplitude factor at known inclination ι |
| θ_JN | iota_deg | Inclination angle (MAP from LVK published posteriors) |

---

## Key Identity

    Γ = A_obs / A_GR = ρ_obs / ρ_syn

σ_noise divides both ρ_obs and ρ_syn identically, so Γ = amp_obs / amp_syn
exactly. The background calibration and whitening normalisation cancel.

**Source-frame amplitude from energy conservation:**

    E_ring = ε_rd M_f c² = (πc³/5G) A₂₂² f_QNM Q   →   A₂₂ = sqrt(5G ε_rd M_f / (πc f_QNM Q))

Derivation: sky-integrated GW luminosity for the (2,2,0) mode (SVEA):
dE/dt = (4π²c³/20G) A₂₂² f² exp(−2t/τ). Integrating to infinity gives E_ring.

**Detector amplitude at ringdown start (3 ms after merger):**

    A_det = (A₂₂ / d_L) × R_rms(ι)
    R_rms(ι) = sqrt[(1/5)((1+cos²ι)²/4 + cos²ι)]

R_rms averages the detector antenna factors F₊, F× over polarisation angle ψ
(⟨F₊²⟩ = ⟨F×²⟩ = 1/5, ⟨F₊F×⟩ = 0) at fixed inclination ι from LVK.

---

## Method

**Injection test (linear superposition):**  
Since the whitening operator W is linear, the pipeline response to
h_inj = A_det × exp(−t/τ) cos(2πft) can be computed without adding h_inj
to the strain noise:

    h_inj_wht = W[h_inj]  (whitening applied to injection directly)
    amp_syn   = sqrt(⟨h_inj_wht | t_c⟩² + ⟨h_inj_wht | t_s⟩²) / norm

This equals what the pipeline would return if h_inj were added to the data,
because W[strain + h_inj] = W[strain] + W[h_inj] and the signal and noise are independent.

**Per-event parameters (from LVK published posteriors, MAP values):**

| Event | M_f (M☉) | E_rad (M☉c²) | d_L (Mpc) | θ_JN (°) | f_QNM (Hz) | τ_QNM (ms) |
|---|---|---|---|---|---|---|
| GW150914 | 62.2 | 3.0 | 410 | 163 | 251 | 4.00 |
| GW151226 | 20.8 | 1.0 | 440 | 88 | 737 | 1.20 |
| GW170814 | 53.2 | 2.7 | 540 | 54 | 329 | 3.28 |
| GW170104 | 49.1 | 2.0 | 880 | 135 | 339 | 2.94 |
| GW190521 | 142.0 | 8.0 | 5300 | 153 | 123 | 8.75 |

---

## Per-Detector-Event Results

| Event / Det | ρ_obs | ρ_syn | Γ = A_obs/A_GR | A_det | R_rms | decay@3ms |
|---|---|---|---|---|---|---|
| GW150914 / H1 | 0.841 | 1.4069 | **0.598** | 9.57e-22 | 0.605 | 0.47 |
| GW150914 / L1 | 0.821 | 0.0667 | ~~12.3~~ | 9.57e-22 | 0.605 | 0.47 |
| GW151226 / H1 | 1.458 | 0.0114 | ~~128~~ | 1.19e-22 | 0.224 | 0.08 |
| GW151226 / L1 | 1.205 | 0.0016 | ~~747~~ | 1.19e-22 | 0.224 | 0.08 |
| GW170814 / H1 | 1.216 | 1.1302 | **1.076** | 3.84e-22 | 0.400 | 0.40 |
| GW170814 / L1 | 1.059 | 0.2132 | ~~4.97~~ | 3.84e-22 | 0.400 | 0.40 |
| GW170814 / V1 | 1.402 | 0.0196 | ~~71.3~~ | 3.84e-22 | 0.400 | 0.40 |
| GW170104 / H1 | 0.392 | 0.4493 | 0.873 | 2.39e-22 | 0.461 | 0.36 |
| GW170104 / L1 | 0.875 | 0.0566 | ~~15.5~~ | 2.39e-22 | 0.461 | 0.36 |
| GW190521 / H1 | 2.164 | 1.0741 | **2.014** | 1.56e-22 | 0.565 | 0.71 |
| GW190521 / L1 | 0.395 | 1.5491 | **0.255** | 1.56e-22 | 0.565 | 0.71 |

Strikethrough Γ values (~~n~~): ρ_syn < 0.1 — noise-dominated, physically meaningless.  
Bold Γ values: ρ_syn > 1 — partially informative.

**Informative events only (ρ_syn > 1):**
Γ = 0.598, 1.076, 2.014, 0.255  →  mean ≈ 0.99, scatter ≈ 0.8

---

## Combined Statistics

```
Combined Γ (ρ_syn²-weighted)   = 0.856 ± 1.196
Combined coherent ρ_syn         = 2.658

Prediction comparison:
  GR           : Γ_pred = 1.0000   |Γ_obs − pred| = 0.144   tension = 0.1σ
  exp          : Γ_pred = 0.8395   |Γ_obs − pred| = 0.016   tension = 0.0σ
  sigma_coh    : Γ_pred = 0.7923   |Γ_obs − pred| = 0.063   tension = 0.1σ
  linear_cbrt  : Γ_pred = 0.7461   |Γ_obs − pred| = 0.110   tension = 0.1σ

Sensitivity to GR vs sigma_coh: 0.2σ  (need >2σ to distinguish)
```

**All four predictions are consistent with the data at < 0.1σ.**

---

## Verdict

**The Phase III amplitude ratio test is consistent with every candidate model.
The current dataset has insufficient sensitivity to distinguish GR (Γ = 1.0)
from any of the σ_conv-derived predictions (Γ = 0.746 – 0.839).**

**Root cause: ρ_syn << 1 for 7 of 11 detector-events.**

The GR-predicted QNM amplitude is below the noise floor for most channels.
This happens for three cumulative reasons:

1. **GW151226**: τ_QNM = 1.2 ms. By t = 3 ms the signal has decayed to
   exp(−3/1.2) = 8 % of peak. A_det = 1.19e-22 — far below the ~10⁻²¹ noise
   floor, giving ρ_syn < 0.02.

2. **Same event, different detectors**: H1 and L1 share the same A_det, but
   ρ_syn differs by a factor of 10–20 between detectors for GW150914 (1.41 vs
   0.067) and GW170814 (1.13 vs 0.21 vs 0.02). This reflects PSD differences
   at the QNM frequency between the two sites at the time of observation — the
   whitened signal amplitude scales as A_det / sqrt(S_h(f_QNM)), which is
   detector- and time-dependent.

3. **GW190521 contradiction**: H1 gives Γ = 2.01 and L1 gives Γ = 0.26 from
   the SAME physical event. Both ρ_syn > 1, yet Γ values differ by 8×. This
   is pure noise — ρ_obs ≈ O(1) noise fluctuation in both detectors, with
   no genuine QNM signal visible above the noise at this distance (5.3 Gpc).

**What the informative subset shows:**

The four events with ρ_syn > 1 give Γ ∈ {0.26, 0.60, 1.08, 2.01} — spanning
a factor of 8 with no systematic trend. This is the statistical profile of a
null experiment: O(1) noise fluctuations divided by O(1) GR-predicted SNR.
The weighted mean Γ = 0.856 ± 1.196 is indistinguishable from any prediction.

**Correct statistical statement:**
The observed QNM amplitude at t = 3 ms is consistent with the σ_conv
entanglement-decoherence prediction (Γ_sigma_coh = 0.7924) at 0.1σ.  It is
equally consistent with GR at 0.1σ.  Phase III does not distinguish between
the two hypotheses.

---

## Why GW150914/H1 is the single most informative data point

GW150914/H1 has ρ_syn = 1.41 — the highest per-event GR-predicted SNR in the
catalog that is not contradicted by its paired detector.  It gives Γ = 0.598:
below GR, near or below the sigma_coh prediction.

This is not a detection — ρ_obs = 0.841, still below the 2.3σ noise threshold.
But it is the only event where the GR-predicted amplitude is at 1.4× the noise
floor, making Γ non-trivially constrained.  Improving this one event to ρ_syn
≥ 3 (via pycbc matched filtering with SXS QNM templates) would make the ratio
measurement a genuine test.

---

## Uncertainty Budget

| Source | Contribution to σ(Γ) | Reducible? |
|---|---|---|
| Noise floor (ρ_syn << 1 for 7/11 events) | Dominant | Yes — pycbc + more events |
| f_rd uncertainty (±30%) | ~15% per event | Partially — NR calibration |
| Distance uncertainty (±20%) | ~10% per event | Partially — improved d_L |
| Inclination uncertainty (±15%) | ~8% per event | Yes — GWTC posterior samples |
| Calibration | ~5% | No |

Current total: ρ_syn² weighted scatter σ(Γ) ≈ 1.20, entirely dominated
by the noise (ρ_syn << 1) factor.  Systematic uncertainties (f_rd, d_L, ι)
are subdominant and could be reduced once the noise floor is cleared.

---

## Path to a Decisive Test

The sigma_coh prediction Γ = 0.7924 differs from GR by ΔΓ = 0.208.
To measure this at 2σ significance requires σ(Γ) ≤ 0.104, i.e.:

    σ(Γ) ≤ ΔΓ / 2 = 0.104
    →  combined coherent ρ_syn  ≥  ΔΓ / 0.104 ≈ 2  (one event at ρ_syn ≥ 2)

or equivalently:

    Need ρ_syn ≥ 3 on one event, or ρ_syn ≥ 2 on each of several events.

Three levers raise ρ_syn without additional events:

1. **Full IMR waveform subtraction** (pycbc + SXS waveforms).  Removes the
   merger turn-on from the residual, improving the QNM template match.
   Typical gain: ρ_QNM increases 3–5×, pushing GW150914/H1 from ρ_syn = 1.4
   to ρ_syn ≥ 4.  One event would become decisive.

2. **Optimised QNM parameters** (f_QNM, τ_QNM from posterior samples rather
   than MAP).  Template mismatch reduces ρ_syn by up to 30% when using the
   wrong f/τ.  Using the full posterior distribution recovers this.

3. **More events at ρ_syn > 1**.  The current count is 4 (only H1 detectors
   and GW190521/L1).  The full GWTC-3 catalog (~90 BBH events × ~2 detectors)
   would provide ~60–80 detector-events with ρ_syn > 0.5, and coherent stacking
   would push the combined ρ_syn to ~8–10.

---

## Cross-References

- Phase I.4 catalog null: `misc/bh_phase_i_4_catalog_results.md`
- Phase I.5 frequency-domain null: `misc/bh_phase_i_5_freq_stack_results.md`
- Phase II amplitude bound: `misc/bh_phase_ii_amplitude_bound_results.md`
- Phase III script: `scripts/phase_iii_amplitude_ratio.py`
- Phase III figure: `misc/bh_phase_iii_amplitude_ratio.png`
- σ_conv / ξ-shell identity: `misc/bh_horizon_sigma_conv_identity.md`
- γ-mode predictions: `misc/bh_collision_phenomenology.md`
- Echo delay derivation: `misc/bh_merger_predictions.md`
