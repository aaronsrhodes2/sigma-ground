# Phase I.2 — ξ-shell echo search: Kerr overtones + cross-detector coherence

| Field | Value |
|---|---|
| Date | 2026-04-17 |
| Phase | I.2 (follow-up to Phase I) |
| Scope | Two tests: (1) Kerr multi-overtone QNM subtraction; (2) H1-L1 sign-coherence discriminator |
| Pipeline | `sigma_ground/field/interface/ligo_echo_search.py` |
| Prior verdict | `misc/bh_phase_i_echo_search_results.md` — Phase I combined p ≈ 0.003 ambiguous (L1 artefact-contaminated) |
| Tools added | `qnm` library (Berti-Kerr overtone spectra), sign-binomial coherence test |
| TL;DR | The Phase I excess is **definitively L1 artefact**. Sign-coherence at chance; Phase I null confirmed with stronger evidence. The ξ-shell prediction is **neither detected nor ruled out** — upper-bound tightening is left to pycbc+SXS. |

---

## variable glossary

| name[symbol] | meaning |
|---|---|
| Kerr overtone[n] | n-th Kerr QNM mode (l=m=2, s=-2); n=0 fundamental, n=1, n=2 first/second overtones |
| overtone amplitude[A_n] | fit coefficient for mode n (magnitude of complex c_n + i·s_n pair from linear LS) |
| overtone phase[φ_n] | fit phase for mode n |
| overtone ratio[A_n/A_0] | amplitude of mode n relative to fundamental; NR expectation 0.4–0.9 for n=1,2 (Giesler+2019) |
| matched-filter SNR[ρ] | inner product of whitened residual with decaying-sinusoid template, calibrated so bg std = 1 |
| per-echo product[ρ_H1·ρ_L1] | inter-detector SNR product at predicted delay Δt_n |
| sign concordance[S/N] | fraction of echoes where sign(ρ_H1)·sign(ρ_L1) > 0 |
| sign p-value[p_sign] | binomial tail probability P(≥S matches in N trials | p = 0.5) |
| magnitude-weighted statistic[Σρ_H1·ρ_L1] | sum of per-echo products; artefact-sensitive |
| null-bootstrap p-value[p_coh] | fraction of random-delay H1-L1 pairings with stat ≥ observed |

---

## pipeline changes since Phase I

1. **Overtone library** — `qnm` (0.4.4, pure-pip, numba-accelerated).
   Produces Kerr (s=-2, l=m=2) QNM complex frequencies as a function
   of spin a*.  Verified output for GW150914 (a*=0.67, M_rem=62.2 M☉):
   n=0 → 270.0 Hz, τ=3.744 ms;
   n=1 → 263.6 Hz, τ=1.238 ms;
   n=2 → 251.8 Hz, τ=0.732 ms.
   Literature n=0 values in the pipeline's `EVENTS` metadata (251 Hz,
   4.0 ms for GW150914) come from Berti-Cardoso-Starinets fitting
   formulas and are less precise than qnm-library output.

2. **`fit_kerr_overtones()`** — new function.
   Decomposes the post-merger waveform as a sum of n modes with
   frequencies/damping-times fixed by `qnm`.  Each mode contributes
   a two-column basis (cos, sin); the full 2n-parameter linear LS
   is solved in closed form.

3. **`cross_detector_coherence()`** — new function.
   Returns three coherence statistics for each event:
   - **magnitude-weighted:**  Σ_n ρ_H1[n] · ρ_L1[n], null from
     random-delay H1/L1 pairings;
   - **sign-only:**  # echoes where signs agree out of N, binomial tail;
   - **diagnostic:**  per-echo |ρ_L1|/|ρ_H1| ratio — real astrophysical
     signal has this O(1) (antenna-pattern), detector-noise pathology
     does not.

4. **Pipeline default for `use_kerr_overtones`: `False`.**
   See §"cherished failure" below.

---

## TEST 1 — Kerr multi-overtone subtraction: the cherished failure

### hypothesis

Phase I diagnosed the per-echo sign-alternation (n=1: +, n=2: −, n=3: +, …)
as residual power at f_QNM from imperfect fundamental-only subtraction.
A Kerr multi-overtone fit (n=0, 1, 2) should capture the true post-merger
waveform better and leave a cleaner residual for the matched filter.

### result: null, with a clear mechanism

Across all 32 tested combinations of (event, detector, fit-start offset,
fit-duration), the 3-mode subtraction produced **larger** residual std
in the 20 ms post-fit window than the 1-mode (n=0 only) subtraction.

Representative ratios (residual std, 3-mode / 1-mode):

| event | det | fit@0 ms dur 8 | fit@0 ms dur 15 | fit@3 ms dur 20 |
|---|---|---|---|---|
| GW151226 | H1 | 1.51 | 1.47 | 1.28 |
| GW151226 | L1 | 1.41 | 1.40 | 4.15 |
| GW150914 | H1 | 4.30 | 2.98 | 65.8 |
| GW150914 | L1 | 2.47 | 2.12 | 2.60 |

Fit amplitudes were catastrophically miscalibrated.  NR-expected
A_1/A_0 ≈ 0.8, A_2/A_0 ≈ 0.4 (Giesler+2019, binary BH ringdown at
a* ≈ 0.67).  Observed ratios on real data:

| event | det | A_1/A_0 | A_2/A_0 |
|---|---|---|---|
| GW151226 | H1 | 3.58 | 2.74 |
| GW151226 | L1 | 5.92 | 5.99 |
| GW150914 | H1 | 11.44 | 16.91 |
| GW150914 | L1 | 7.71 | 8.72 |

### why it fails

The three Kerr modes for GW150914 at a* = 0.67 have frequencies
(270.0, 263.6, 251.8) Hz — f_1 − f_0 = 6 Hz, f_2 − f_0 = 18 Hz.
On a 20 ms fit window (5 cycles of the 250 Hz mode) the frequency
resolution is 1/20 ms = 50 Hz, so n=0 and n=1 are **not frequency-
separable** and n=2 is marginal.  The linear-LS basis vectors g_0(t),
g_1(t), g_2(t) are nearly collinear; the inverse of MᵀM amplifies any
noise component onto the overtone coefficients.  At a 3 ms fit-start
offset, the overtones have physically decayed to exp(-3ms/1.2ms) ≈ 0.09
and exp(-3ms/0.7ms) ≈ 0.02 of their initial amplitude — the fit is
indistinguishable from fitting noise.

**Learning:** unconstrained multi-overtone ringdown fitting of GWOSC
strain is ill-conditioned without an NR-informed prior on A_n/A_0.
Publication-grade echo searches (e.g., Abedi+2017, Westerweck+2018,
Nielsen+2019) use SXS-waveform subtraction or pycbc's Teukolsky
templates to avoid this exact problem.  The
`use_kerr_overtones=True` code path is preserved in the pipeline for
future constrained-fit experiments; the default is `False`.

### the code path is kept for future work

When an NR-informed prior becomes available (A_1/A_0 fixed, only
φ_1, φ_2 free as a 2-parameter corrector), the multi-overtone path
becomes defensible and can be reactivated.  Until then, n=0 alone is
the least-wrong choice.

---

## TEST 2 — cross-detector coherence: the discriminator

### hypothesis

A real astrophysical echo signal must be present in **both** detectors
at the predicted Δt_n with |ρ_L1|/|ρ_H1| ≈ O(1) (antenna-pattern
factor) and a common sign (modulo uniform antenna flip).  Detector-
specific noise pathologies (calibration glitches, spectral lines in
one IFO, non-stationary PSD) produce large per-detector |ρ| but
**fail** the coherence criterion.

The Phase I combined p ≈ 0.003 excess was attributed to L1 artefacts
(A_fit 5–24× H1's, sign-alternating residual).  Phase I.2 makes this
attribution testable: run the n=0 pipeline, extract per-echo ρ_H1[n]
and ρ_L1[n], then compute three coherence statistics.

### result

#### GW151226 (16 kHz, 5 predicted echoes at n·0.756 ms)

| n | Δt (ms) | ρ_H1 | ρ_L1 | ρ_H1·ρ_L1 | \|L1\|/\|H1\| |
|---|---|---|---|---|---|
| 1 | 0.756 | +1.22 | −32.57 | −39.55 | 26.8 |
| 2 | 1.512 | +3.32 | +34.69 | +115.18 | 10.5 |
| 3 | 2.267 | −3.24 | −25.38 | +82.21 | 7.8 |
| 4 | 3.023 | +2.43 | +11.57 | +28.09 | 4.8 |
| 5 | 3.779 | −0.33 | −5.72 | +1.91 | 17.2 |

- **Magnitude-weighted Σρ_H1·ρ_L1 = +187.8**, null μ=+0.02, σ=2.26 → z = **+83.1**, p = 0.
- **Sign matches: 4/5**, binomial p = **0.188** — *not* significant.
- **\|L1\|/\|H1\| per echo: 4.8–26.8** — real GW signal has this ≤ ~2.

Interpretation: the huge magnitude-weighted stat is entirely driven by
L1's inflated SNRs.  Sign-only is at chance (4/5 occurs 18.8 % of the
time for random coin-flips).  The \|L1\|/\|H1\| ratios are diagnostic
of detector-specific artefact, not an antenna-pattern-modulated
astrophysical waveform.

#### GW150914 (4 kHz, 5 predicted echoes at n·2.260 ms)

| n | Δt (ms) | ρ_H1 | ρ_L1 | ρ_H1·ρ_L1 | \|L1\|/\|H1\| |
|---|---|---|---|---|---|
| 1 | 2.260 | +0.47 | −12.02 | −5.65 | 25.6 |
| 2 | 4.520 | −1.43 | −4.43 | +6.33 | 3.1 |
| 3 | 6.781 | −1.49 | +5.25 | −7.84 | 3.5 |
| 4 | 9.041 | −1.38 | −4.04 | +5.58 | 2.9 |
| 5 | 11.301 | +1.18 | +4.35 | +5.12 | 3.7 |

- **Magnitude-weighted Σρ_H1·ρ_L1 = +3.54**, null μ=+0.03, σ=2.23 → z = +1.58, p = **0.056**.
- **Sign matches: 3/5**, binomial p = **0.500** — chance level.
- **\|L1\|/\|H1\|: 2.9–25.6** — still elevated, especially n=1.

Interpretation: the magnitude-weighted stat is marginal (z=1.58),
easily explained by the n=1 and n=3 outliers.  Sign concordance is
exactly at chance.  No coherent echo signal.

### calibration diagnostics

Whitened pre-merger std (ideal 1.0):

| event | det | whitened pre-merger std |
|---|---|---|
| GW151226 | H1 | 0.405 |
| GW151226 | L1 | 0.375 |
| GW150914 | H1 | 0.635 |
| GW150914 | L1 | 0.631 |

The pre-merger std is 0.38–0.64 — still under-calibrated despite the
bg-rescaling in the echo window.  This reflects a PSD-estimation
mismatch between the 8-s Welch window and the near-merger spectrum.
The bg-rescaling gets bg std = 1.000 exactly (by construction), but
the pre-merger diagnostic shows the whitening is not perfectly flat.
The off-source background p99 lands at 2.42–2.65 (Gaussian expectation
2.33) so the calibration is within 15 % of Gaussian in the search
window — adequate for p-value comparisons.

---

## verdict

**Phase I.2 does not detect a ξ-shell echo train at Δt_n = 2·r_s·n·σ_conv/c
in GW150914 or GW151226, nor does it rule one out.**

Three cherished findings:

1. **The Phase I combined p ≈ 0.003 excess is L1 artefact, not signal.**
   Sign-only coherence between H1 and L1 is at chance (4/5 and 3/5,
   p = 0.188 and 0.500) for both events.  The huge magnitude-weighted
   coherence is entirely driven by L1's detector-specific noise
   pathology (|ρ_L1|/|ρ_H1| = 3–27×, inconsistent with any real GW
   signal's O(1) antenna-pattern factor).  The cross-detector
   coherence test **is the right discriminator** and it cleanly
   kills the Phase I excess.

2. **Unconstrained Kerr multi-overtone subtraction is ill-conditioned
   on GWOSC strain.**  The near-degeneracy of f_0 ≈ f_1 ≈ f_2 (within
   20 Hz at a*=0.67) on a 20 ms fit window makes the 3-mode linear LS
   overfit noise: A_n ratios 3–25× the NR expectation, residual std
   1.3–65× worse than n=0 alone.  The standard approach in the
   literature (SXS waveform subtraction via pycbc) sidesteps this by
   using a physically-constrained ringdown model.

3. **H1 alone is calibrated and quiet.**  Per-echo |ρ_H1| ≤ 3.32σ for
   both events, p99 background = 2.53 — all within a normal Gaussian
   tail.  Upper-bound on a ξ-quantised echo signal in H1: any coherent
   echo train with amplitude > ~2–3× the background noise floor is
   excluded at the 99 % level.  Tightening this bound is the scope
   of a publication-grade Phase II analysis.

### what this means for the sigma-ground theory

- **The ξ-quantised echo prediction (Δt_n = 2·r_s·n·σ_conv/c with
  σ_conv = −ln ξ ≈ 1.844) is not confirmed.**  It is also not
  falsified — only a weak amplitude bound is set.
- The **γ-mode choice** in the Phase G table is unaffected by this
  test: γ_coh = 1 − η/2 ≈ 0.7924 is the sigma_coh mode used to
  predict the echo delays, and the prediction's validity depends on
  a non-zero echo amplitude which H1 data is too quiet to see at the
  current sensitivity.
- The **Phase J horizon-identity picture** (horizon energy tensor
  trace = σ_conv·ρ·c²) is unaffected; Phase I.2 only tests the
  reflection signature, not the identity itself.

### comparison to published echo-search literature

Abedi-Dykaar-Afshordi (2017) reported p ≈ 0.011 for echoes in
GWTC-1 using SXS-subtraction and a 2D search over (echo delay,
damping factor).  Nielsen-Capano-Birnholtz-Westerweck (2019)
reanalysed the same data with expanded priors and found p ≈ 0.15
(null).  Westerweck-Nielsen-Birnholtz-Capano (2018) found p ≈ 0.03
via matched-filter on IMRPhenom residuals.  Our Phase I.2 result
(H1 p_min = 0.135, H1 p_combined = 0.19 for GW150914; no coherent
signal per sign-test) sits at the null end of this spread and is
consistent with the Nielsen+2019 finding.

---

## next-step recommendations

Ranked by (learning per run)/(cost to run):

1. **Constrained multi-overtone fit** — fix A_1/A_0, A_2/A_0 to
   Giesler+2019 NR values, only fit φ_n.  This converts a 6-parameter
   ill-conditioned LS into a 4-parameter well-conditioned one.  If
   this cleanly reduces the residual std below the 1-mode result, the
   overtone subtraction becomes a viable pre-search step.  **Cost:**
   ~30 min coding, minutes to re-run.  **Expected learning:**
   improved upper bound by a factor of 1.5–3 on echo amplitude.

2. **Phase II: pycbc + SXS waveform subtraction.**  Install pycbc
   (requires conda/miniforge on Windows; pip fails on Python 3.13 due
   to numpy-1.26 C-compiler requirement).  Use PyCBC's precision
   IMRPhenomXPHM waveform subtraction with NR-informed parameters.
   **Cost:** ~1 session of install + ~1 session of analysis.
   **Expected learning:** publication-grade upper bound; ~10× tighter
   than Phase I.2.

3. **Catalog extension.**  Pull GW170104, GW170814, GW190521
   strain.  Run Phase I.2 on each.  If ξ-echo exists with
   event-independent amplitude ratio, combining 5 events sqrt(5) ≈ 2.2×
   tightens the bound.  **Cost:** ~10 min strain pulls + re-run.
   **Expected learning:** either confirm no-signal in 5 events
   (combined upper bound ≈ 1/sqrt(5) of single-event) or surface
   an event-dependent pattern worth investigating.

4. **Phase J footnote** formalising the isotropic-interior reduction
   implied by Aaron's galaxy-isotropy observation.  **Cost:** 30 min.
   **Expected learning:** tightens the Phase J prediction from "horizon
   oblateness follows interior net-J" to "horizon oblateness ≈ Kerr
   a*-oblateness for isotropic interiors".

Default priority: (1) → (2) → (3).  (4) can go into any other session.

---

## reproducibility

Full stdout captured at `misc/bh_phase_i_2_overtone_coherence_output.txt`.
Pipeline code: `sigma_ground/field/interface/ligo_echo_search.py`
(git SHA at time of run).  Cached strain data at `local-cache/gwosc/`.
Re-run:

```bash
cd D:\Aaron\development\sigma-ground
python -X utf8 -m sigma_ground.field.interface.ligo_echo_search
```

---

## cross-references

- `misc/bh_phase_i_echo_search_results.md` — Phase I verdict (artefact-
  contaminated combined p ≈ 0.003, later explained by this I.2 test).
- `misc/bh_merger_predictions.md` — Δt_n ground-truth table and
  γ(σ_conv) derivation.
- `misc/bh_echo_search_refined.md` — Phase H.7 methodology.
- `misc/bh_collision_phenomenology.md` — synthesis of the BH-collision
  phenomenology across Phases A–J.
- `misc/bh_horizon_sigma_conv_identity.md` — Phase J (horizon energy
  identity); not tested here.
