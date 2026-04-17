# Phase I.5 — Frequency-Domain Sideband Stacking: Verdict

**Date:** 2026-04-17  
**Phase:** I.5 — frequency-domain comb stacking across 5-event catalog  
**Pipeline:** `scripts/phase_i_5_freq_stack.py`  
**Input data:** 11 detector-events (4–16 kHz GWOSC HDF5 files, all cached locally)

---

## Variables

| Symbol | Name | Meaning |
|---|---|---|
| Δt₁ | echo_delay_n(M_rem, 1) | First-echo delay: 2 r_s σ_conv / c |
| f_QNM | f_qnm_hz | Dominant (2,2,0) Kerr QNM frequency |
| f_r | f_rescaled | (f − f_QNM) × Δt₁ — rescaled frequency |
| k | sideband index | k = 1, 2, 3 — positive interference sidebands |
| R | comb_ratio | Median on-comb / median off-comb power |

---

## Method

Echo trains with inter-echo spacing Δt₁ produce interference sidebands in
the power spectrum at offsets ±k/Δt₁ from f_QNM. Rescaling by Δt₁ maps
each event's sidebands to the universal positions f_r = ±k regardless of
event mass. Stacking normalised spectra across N events improves S/N by √N.

**Analysis window:** 1000 ms starting 3 ms after merger. A 30 ms window
(10 × Δt₁) produces only ~26 in-band frequency bins at 4 kHz — too few
for stable median normalisation. A 1 s window gives ~865 bins (35–900 Hz,
4 kHz data) with 1 Hz resolution. The QNM (τ_QNM ≈ 3–9 ms) decays to
< 0.1 % of peak power within the first 6 % of the window.

**Frequency masking:** Before interpolating onto the common f_r grid,
each event's power spectrum is restricted to the in-band region
[35 Hz, min(0.45 × Nyquist, 1500 Hz)]. Out-of-band bins carry only
bandpass-attenuated noise; whitening amplifies this to a flat residual
that, if included, makes all events appear to cover the full f_r grid.

**Normalisation:** Each spectrum is divided by the median power of all
valid in-band points with f_r > 0.5 (above the QNM notch), excluding
windows of half-width 0.25 around each sideband position k = 1, 2, 3.

**Comb statistic:** Only positive sidebands (f_r = +1, +2, +3) are tested.
The negative counterparts require f_phys = f_QNM − k/Δt₁ < 0 Hz for all
five events — physically inaccessible. Comb excess ratio R =
median(stack at f_r ≈ +k) / median(stack at f_r > 0.5, excluding ±0.25
windows around each k). Both quantities use the median for robustness
against spectral lines and non-Gaussian glitches.

---

## Detector plan

| Event | Det | fs | Δt₁ (ms) | f_QNM (Hz) | f_r,max | k=1 accessible |
|---|---|---|---|---|---|---|
| GW151226 | H1 | 16k | 0.756 | 737 | 0.57 | No |
| GW151226 | L1 | 16k | 0.756 | 737 | 0.57 | No |
| GW150914 | H1 | 4k | 2.260 | 251 | 1.47 | Yes |
| GW150914 | L1 | 4k | 2.260 | 251 | 1.47 | Yes |
| GW170814 | H1 | 4k | 1.933 | 329 | 1.10 | Marginal |
| GW170814 | L1 | 4k | 1.933 | 329 | 1.10 | Marginal |
| GW170814 | V1 | 4k | 1.933 | 329 | 1.10 | Marginal |
| GW170104 | H1 | 4k | 1.784 | 339 | 1.00 | Marginal |
| GW170104 | L1 | 4k | 1.784 | 339 | 1.00 | Marginal |
| GW190521 | H1 | 4k | 5.160 | 123 | 4.01 | Yes (k=1,2,3) |
| GW190521 | L1 | 4k | 5.160 | 123 | 4.01 | Yes (k=1,2,3) |

**Note on GW151226:** f_QNM = 737 Hz is near the top of the 1500 Hz bandpass cap.
The first sideband sits at 737 + 1/0.756ms = 2060 Hz — above the cap. GW151226
contributes valid normalised data in the stack at f_r ∈ [−0.53, +0.57] but does
not contribute to any sideband position.

---

## Results

```
Phase I.5 — sideband comb excess (positive k=1,2,3 only)
  on-comb median power: 1.4811
  off-comb median power: 1.3494
  ratio (null → 1.0, signal → >1): 1.0976
  contributors per sideband k=1,2,3: [9, 2, 2]
```

| Sideband k | N contributors | Median power | Ratio to baseline |
|---|---|---|---|
| k=1 (f_r=1) | 9 | ~1.51 | ~1.12 |
| k=2 (f_r=2) | 2 | ~55 (mean) / ~1.5 (median) | ~1.1 |
| k=3 (f_r=3) | 2 | ~1.81 | ~1.34 |
| Off-comb baseline | — | 1.35 | 1.0 (reference) |

**Combined ratio R = 1.10** (median-of-medians across k=1,2,3).

**Expected null value: R ≈ 1.0.** After normalising each spectrum by
its own median, the stacked power at any set of f_r positions has the
same expected median as any other set. Residual offset from 1.0 arises
from: (a) chi-squared(2) mean > median by a factor of 1/ln(2) ≈ 1.44;
(b) the normalisation median is estimated from the norm_mask region and
may differ slightly from the population median of other f_r bands.

**Uncertainty estimate:** The off-comb region (500 bins, mix of N=9 and
N=2 contributors) has a scatter-derived std of the median of ≈ 0.13–0.17.
The on-comb measurements (k=1: 125 bins × 9 events; k=2,3: 125 bins × 2
events) have uncertainty ≈ 0.08 (k=1) and ≈ 0.15 (k=2,3). The ratio
R = 1.10 is consistent with null within ≈ 0.6–0.8σ.

---

## Verdict

**No detection. The comb excess ratio R = 1.10 ± ~0.15 is consistent
with the null hypothesis (R = 1.0) within statistical uncertainty.**

This is the expected result for a noise-only dataset and is fully
consistent with the Phase I.4 catalog verdict (Fisher p_F = 0.609).

Key caveats on sensitivity:

1. **GW151226 excluded from sideband test.** Its f_QNM = 737 Hz pushes
   the k=1 sideband above the 1500 Hz bandpass cap. The two highest-rate
   LIGO events (highest precision) contribute no sideband sensitivity.
2. **k=2,3 driven by one event.** Only GW190521 (H1+L1) reaches f_r > 1.5.
   With N=2, the k=2,3 measurement is a single-event noise sample with no
   useful stacking gain.
3. **QNM not subtracted.** The post-merger window includes the full QNM
   decay. Even with a 1 s window, the QNM contributes a narrow Lorentzian
   spike at f_r = 0 with width ≈ Δt₁/(2π τ_QNM) ≈ 0.09 in rescaled units
   for GW150914. This spike is well-separated from the sideband positions
   but its sidelobes may bias the normalization at small |f_r|.

**What this means for σ_conv / ξ-shell:**  
The frequency-domain stacking test provides no evidence for or against
the predicted echo structure. The sensitivity is limited by: (a) the
spectral mismatch between 4 kHz data and the event QNM frequencies,
(b) the fact that most events (all except GW190521) lack accessible k≥2
sidebands in the GWOSC 4 kHz dataset, and (c) the absence of QNM
subtraction (Phase I.3 showed subtraction is unstable with a short window).

The Phase I.4 time-domain matched-filter catalog null result
(Fisher p_F = 0.609) remains the primary constraint.

---

## Next steps

1. **16 kHz re-run** for GW150914, GW170814, GW170104, GW190521. Nyquist
   = 8000 Hz expands f_r_max from 1.5 to ~7.6 for GW150914, making k=1–5
   accessible. This is the most direct improvement.
2. **QNM subtraction in frequency domain.** Fit a Lorentzian at f_r = 0
   in the power spectrum and subtract it before the comb measurement. This
   removes the QNM contribution without the time-domain instability of
   Phase I.3.
3. **Per-event p-value via bin permutation.** Shuffle the f_r grid labels
   within each event to build a null distribution for R, providing a
   properly calibrated p-value rather than the approximate uncertainty above.

---

## Cross-references

- Phase I.4 (catalog null): `misc/bh_phase_i_4_catalog_results.md`
- Phase I.3 (subtraction artifact): `misc/bh_phase_i_3_no_subtraction_results.md`
- Phase I pipeline: `sigma_ground/field/interface/ligo_echo_search.py`
- Phase I.5 script: `scripts/phase_i_5_freq_stack.py`
- Echo delay derivation: `misc/bh_merger_predictions.md`
