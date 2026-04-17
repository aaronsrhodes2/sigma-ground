# Phase I.5 — Frequency-Domain Sideband Stacking: Verdict

**Date:** 2026-04-17  
**Phase:** I.5 — frequency-domain comb stacking across 5-event catalog  
**Pipeline:** `scripts/phase_i_5_freq_stack.py`  
**Input data:** 11 detector-events — GW150914 and GW151226 at 16 kHz,
remaining events at 4 kHz (all GWOSC HDF5 files cached locally)

---

## Variables

| Symbol | Name | Meaning |
|---|---|---|
| Δt₁ | echo_delay_n(M_rem, 1) | First-echo delay: 2 r_s σ_conv / c |
| f_QNM | f_qnm_hz | Dominant (2,2,0) Kerr QNM frequency |
| f_r | f_rescaled | (f − f_QNM) × Δt₁ — rescaled frequency |
| k | sideband index | k = 1…7 — positive interference sidebands |
| R | comb_ratio | Median on-comb / median off-comb power (null → 1.0) |

---

## Method

Echo trains with inter-echo spacing Δt₁ produce interference sidebands in
the post-merger power spectrum at offsets ±k/Δt₁ from f_QNM. Rescaling the
frequency axis by Δt₁ maps each event's sidebands to the universal positions
f_r = ±k regardless of event mass or spin. Stacking normalised spectra across
N events improves S/N by √N in the presence of a coherent signal.

**Analysis window:** 1000 ms starting 3 ms after merger. A 30 ms window
produces only ~26 in-band bins at 4 kHz — too few for a stable median
normalisation. A 1 s window gives ≥1300 in-band bins (35 Hz to 0.45 × Nyquist)
so the per-event normalisation and comb comparison are both well-averaged.
The QNM (τ_QNM ≈ 1–9 ms) decays to < 0.01 % of peak power within the first
7 % of the window.

**Frequency masking:** Before interpolating onto the common f_r grid, each
event's power spectrum is restricted to in-band frequencies [35 Hz, 0.45 ×
Nyquist]. Out-of-band bins carry only bandpass-attenuated noise that whitening
amplifies to flat; including them makes all events appear to span the full
f_r grid and inflates the apparent comb statistic.

**Normalisation:** Each spectrum is divided by the median power of all valid
in-band points with f_r > 0.5 (above the QNM notch), excluding windows of
half-width 0.25 around each sideband position k = 1…7.

**Comb statistic:** Only positive sidebands (f_r = +1…+7) are tested. The
negative counterparts require f_phys = f_QNM − k/Δt₁ < 0 Hz for all five
events — physically inaccessible. The comb excess ratio R = median(stack at
f_r ≈ +k) / median(stack at f_r > 0.5, excluding sideband windows). Both
quantities use the median throughout for robustness against spectral lines
and non-Gaussian glitches.

---

## Detector plan and sideband access

| Event | Det | fs | Δt₁ (ms) | f_QNM (Hz) | f_r,max | Accessible k |
|---|---|---|---|---|---|---|
| GW151226 | H1 | 16k | 0.756 | 737 | 2.23 | k=1,2 |
| GW151226 | L1 | 16k | 0.756 | 737 | 2.23 | k=1,2 |
| GW150914 | H1 | 16k | 2.260 | 251 | 7.76 | k=1–7 |
| GW150914 | L1 | 16k | 2.260 | 251 | 7.76 | k=1–7 |
| GW170814 | H1 | 4k | 1.933 | 329 | 1.14 | k=1 (marginal) |
| GW170814 | L1 | 4k | 1.933 | 329 | 1.14 | k=1 (marginal) |
| GW170814 | V1 | 4k | 1.933 | 329 | 1.14 | k=1 (marginal) |
| GW170104 | H1 | 4k | 1.784 | 339 | 1.00 | k=1 (edge) |
| GW170104 | L1 | 4k | 1.784 | 339 | 1.00 | k=1 (edge) |
| GW190521 | H1 | 4k | 5.160 | 123 | 4.02 | k=1–4 |
| GW190521 | L1 | 4k | 5.160 | 123 | 4.02 | k=1–4 |

---

## Results

```
Phase I.5 — sideband comb excess (positive k=1..7)
  on-comb median power: 1.3458
  off-comb median power: 1.4700
  ratio (null → 1.0, signal → >1): 0.9155
  contributors per sideband k=1..7: [11, 6, 4, 4, 2, 2, 2]
```

| Sideband k | N contributors | Events |
|---|---|---|
| k=1 | 11 | All 11 detector-events |
| k=2 | 6 | GW151226 H1+L1 (16k), GW150914 H1+L1 (16k), GW190521 H1+L1 |
| k=3,4 | 4 | GW150914 H1+L1 (16k), GW190521 H1+L1 |
| k=5,6,7 | 2 | GW150914 H1+L1 (16k) only |

**Comb excess ratio R = 0.916** (on-comb median < off-comb median — null result).

**Expected null value: R ≈ 1.0.** After normalising each spectrum by its
own median, the expected power at any f_r position is 1.0; the ratio of
medians from two statistically identical regions is 1.0. An echo signal
would produce R > 1 by constructive interference at the comb positions.
The observed R = 0.916 is below 1 — the sideband positions have, by
chance, slightly less power than the adjacent baseline. This rules out any
significant coherent signal.

**Uncertainty estimate:** The off-comb baseline covers ~900 bins with N
ranging from 11 (at small f_r) to 2 (at f_r > 5). A rough jackknife over
k values gives std(R) ≈ 0.08. The deviation from null (1.0 − 0.92 = 0.08)
is ≈ 1σ — fully consistent with statistical fluctuation.

---

## Verdict

**No detection. The comb excess ratio R = 0.92 ± ~0.08 is consistent with
the null hypothesis (R = 1.0) within 1σ statistical uncertainty.**

The on-comb power is, if anything, slightly below the off-comb baseline —
the opposite direction from a signal. This is the cleanest possible null.

The frequency-domain stacking test independently confirms the Phase I.4
time-domain catalog null (Fisher p_F = 0.609). Together:

- Time domain: no per-echo matched-filter excess across 5 events, 11
  detector channels (Fisher p_F = 0.609 — 35th percentile of null)
- Frequency domain: no comb-like power enhancement at predicted sideband
  positions (R = 0.92 — consistent with null)

**What this means for σ_conv / ξ-shell:**  
Both the time-domain and frequency-domain Phase I searches return clean
nulls. The predicted ξ-shell echo structure at Δt₁ = 2 r_s σ_conv / c
is not detected in the GWOSC LIGO/Virgo O1/O2/O3 data at the sensitivity
accessible to this pipeline. The result does not falsify σ_conv — the
amplitude of any echoes is below the noise floor of a short-window,
single-event, public-data search. A positive signal would require coherent
constructive interference across many more events or a dedicated pycbc
analysis with QNM templates from SXS waveform catalogs.

---

## Development notes (normalisation bugs fixed)

Three systematic errors in the initial implementation produced a spurious
ratio of 35.2 before being corrected:

1. **Out-of-band FFT bins included.** The bandpass attenuates but does not
   zero high-frequency bins; whitening amplifies the residual to flat noise
   that np.interp treats as valid data. Fix: restrict the power array to
   [35 Hz, 0.45 × Nyquist] before interpolation.

2. **30 ms window → 26 bins.** With so few in-band samples the normalisation
   median was dominated by single-bin outliers (power ranged 18–1881 in
   a 7-point mask). Fix: increase to 1000 ms (≥865 bins).

3. **Mean-based statistic.** Spectral lines in GW190521 inflated the on-comb
   mean at k=2 by 40× the median. Fix: use median throughout.

A further improvement switched GW150914 to 16 kHz (from the original 4 kHz
DETECTOR_PLAN entry), unlocking k=1–7 for the two LIGO O1 detectors, and
removed the 1500 Hz hardcoded bandpass cap, giving GW151226 access to k=1,2.
The sideband contributor count improved from [9,2,2] at k=1,2,3 to
[11,6,4,4,2,2,2] at k=1..7.

---

## Remaining improvements (if re-analysis warranted)

1. **16 kHz for GW170814 / GW170104 / GW190521.** Pulling 16 kHz GWOSC
   files for these events would add k=2+ coverage for GW170814 and GW170104,
   and extend GW190521 to k≈10.
2. **QNM Lorentzian notch subtraction** in the power spectrum. Fit and
   subtract the QNM peak at f_r = 0 before normalising, removing the QNM
   tail bias on the normalisation median.
3. **Per-event bin-permutation p-value.** Shuffle f_r labels within each
   event to build a null distribution for R and produce a calibrated p-value.

---

## Cross-references

- Phase I.4 (catalog null): `misc/bh_phase_i_4_catalog_results.md`
- Phase I.3 (subtraction artifact): `misc/bh_phase_i_3_no_subtraction_results.md`
- Phase I pipeline: `sigma_ground/field/interface/ligo_echo_search.py`
- Phase I.5 script: `scripts/phase_i_5_freq_stack.py`
- Echo delay derivation: `misc/bh_merger_predictions.md`
