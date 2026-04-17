# Phase I.4 — ξ-shell echo search, catalog extension (5 events)

| Field | Value |
|---|---|
| Date | 2026-04-17 |
| Phase | I.4 (ξ-echo pipeline, O1/O2 + O3a catalog) |
| Pipeline module | `sigma_ground/field/interface/ligo_echo_search.py` |
| Events | GW150914, GW151226, GW170814 (H1+L1+V1), GW170104, GW190521 |
| subtraction_mode | `none` (established as clean baseline in Phase I.3) |
| Prior docs | `bh_phase_i_3_no_subtraction_results.md` |
| Captured stdout | `misc/bh_phase_i_4_catalog_output.txt` |

## Variable glossary

- `Δt_n[ms]` — predicted n-th ξ-shell echo delay, Δt_n = 2·r_s·n·σ_conv/c,
  σ_conv = −ln(ξ) ≈ 1.8439
- `SNR_n[σ]` — calibrated matched-filter SNR at Δt_n; bg_std normalised to 1.0
- `p_comb` — combined p-value across n=1..5 (quadrature SNR vs bootstrap)
- `sign[k/5]` — number of echoes with sign(H)=sign(X) across two detectors
- `z_c` — magnitude-weighted cross-detector coherence statistic z-score
- `p_c` — one-sided p-value of coherence statistic vs off-source null
- `χ²_F` — Fisher combination: −2·Σ ln(p_comb_i) across detector-event pairs
- `p_F` — Fisher p-value under χ²(2N dof), N = number of p-values combined

## New events added to pipeline

Three events beyond the Phase I/I.3 pair (GW150914, GW151226):

| event | M_rem [M☉] | a* | Δt_1 [ms] | detectors | f_qnm [Hz] | τ_qnm [ms] |
|---|---|---|---|---|---|---|
| GW170814 | 53.2 | 0.72 | 1.933 | H1+L1+V1 | 329.0 | 3.28 |
| GW170104 | 49.1 | 0.66 | 1.784 | H1+L1 | 339.4 | 2.94 |
| GW190521 | 142.0 | 0.72 | 5.160 | H1+L1 | 123.3 | 8.75 |

f_qnm and τ_qnm computed via `qnm` library at (M_rem, a*) from GWTC LVC papers,
Kerr l=m=2, n=0 fundamental mode.  GW190521 is the IMBH-scale event with the
longest predicted echo delay in the O1–O3a catalog sample.

## Per-event results

### GW170814 (H1+L1+V1)

| det | n | Δt_n [ms] | SNR_n [σ] | p_n |
|---|---|---|---|---|
| H1 | 1 | 1.933 | −1.29 | 0.207 |
| H1 | 2 | 3.866 | +0.03 | 0.971 |
| H1 | 3 | 5.799 | +0.28 | 0.776 |
| H1 | 4 | 7.733 | −0.36 | 0.709 |
| H1 | 5 | 9.666 | +0.28 | 0.775 |
| **H1 combined** | | | **1.40** | **0.865** |
| L1 | 1 | 1.933 | +0.99 | 0.326 |
| L1 | 2 | 3.866 | +1.31 | 0.197 |
| L1 | 3 | 5.799 | −1.17 | 0.245 |
| L1 | 4 | 7.733 | −1.18 | 0.245 |
| L1 | 5 | 9.666 | +0.62 | 0.535 |
| **L1 combined** | | | **2.41** | **0.357** |
| V1 | 1 | 1.933 | −1.18 | 0.248 |
| V1 | 2 | 3.866 | −1.38 | 0.178 |
| V1 | 3 | 5.799 | +0.64 | 0.514 |
| V1 | 4 | 7.733 | +0.84 | 0.407 |
| V1 | 5 | 9.666 | −1.09 | 0.293 |
| **V1 combined** | | | **2.36** | **0.353** |

Three-detector coherence:

| pair | sign matches | binomial p | z_c | p_c (1-sided) |
|---|---|---|---|---|
| H1×L1 | 3/5 | 0.500 | −0.43 | 0.693 |
| H1×V1 | 2/5 | 0.812 | +0.47 | 0.291 |
| L1×V1 | **0/5** | **1.000** | **−2.41** | **0.986** |

The L1×V1 result (0/5 sign matches) is a downward fluctuation: ALL five echoes
have opposite sign between L1 and V1.  The two-sided binomial probability for
{0 or 5 out of 5} is 2/32 = 0.0625.  With 7 pairs tested across the three
events in this phase, the expected number of results with |sign − 2.5| ≥ 2.5
is ≈ 0.22 (7 × 0.0625 / 2), so this is consistent with chance under the
global null.  Crucially, the anti-coherent direction (product z-score = −2.41)
is incompatible with a signal: a real echo train would bias the product
statistic *positive*, not negative.

### GW170104 (H1+L1)

| det | n | Δt_n [ms] | SNR_n [σ] | p_n |
|---|---|---|---|---|
| H1 | 1 | 1.784 | +1.56 | 0.118 |
| H1 | 2 | 3.568 | −1.51 | 0.126 |
| H1 | 3 | 5.353 | −0.49 | 0.618 |
| H1 | 4 | 7.137 | −0.83 | 0.392 |
| H1 | 5 | 8.921 | −1.28 | 0.198 |
| **H1 combined** | | | **2.69** | **0.206** |
| L1 | 1 | 1.784 | −2.19 | 0.037 |
| L1 | 2 | 3.568 | −0.38 | 0.670 |
| L1 | 3 | 5.353 | +1.35 | 0.156 |
| L1 | 4 | 7.137 | −1.20 | 0.210 |
| L1 | 5 | 8.921 | +0.61 | 0.524 |
| **L1 combined** | | | **2.93** | **0.144** |

H1×L1 coherence: sign 2/5 (p=0.812), z_c = −1.47, p_c = 0.935 (null).

L1 n=1 at 2.19σ (p=0.037) is the largest single-echo excess in the catalog
extension.  However it is ANTI-coherent with H1 n=1 (+1.56σ): the two
detectors have opposite signs at n=1, ruling out a common astrophysical origin.
Under the signal hypothesis, sign-anti-coherence at n=1 requires a detector
antenna-pattern sign flip of ≈ −1, which for GW170104 sky position is possible
(LIGO has non-zero negative projection terms) but does not explain why the
combined L1 statistic (p=0.144) without sign-alignment is still not significant.

### GW190521 (H1+L1)

| det | n | Δt_n [ms] | SNR_n [σ] | p_n |
|---|---|---|---|---|
| H1 | 1 | 5.160 | +1.73 | 0.080 |
| H1 | 2 | 10.320 | −0.38 | 0.729 |
| H1 | 3 | 15.480 | −0.57 | 0.605 |
| H1 | 4 | 20.640 | +0.37 | 0.736 |
| H1 | 5 | 25.800 | −0.51 | 0.651 |
| **H1 combined** | | | **1.96** | **0.615** |
| L1 | 1 | 5.160 | −0.63 | 0.543 |
| L1 | 2 | 10.320 | +0.11 | 0.915 |
| L1 | 3 | 15.480 | −1.16 | 0.263 |
| L1 | 4 | 20.640 | +0.07 | 0.933 |
| L1 | 5 | 25.800 | −0.73 | 0.490 |
| **L1 combined** | | | **1.51** | **0.825** |

H1×L1 coherence: sign 3/5 (p=0.500), z_c = −0.03, p_c = 0.511 (null).

H1 n=1 at 1.73σ (p=0.080) is the most significant single echo in GW190521 —
not significant, and un-corroborated by L1.  The IMBH-scale echoes (Δt_1 =
5.16 ms, 21 samples at 4 kHz) are well-resolved by the pipeline; no aliasing
artifacts apply.

### Phase I/I.3 reference (no-subtract)

| det | combined p |
|---|---|
| GW151226/H1 | 0.952 |
| GW151226/L1 | 0.879 |
| GW150914/H1 | 0.079 |
| GW150914/L1 | 0.516 |

## Fisher combination — all 11 detector-event p-values

| det-event | p_comb |
|---|---|
| GW151226/H1 | 0.952 |
| GW151226/L1 | 0.879 |
| GW150914/H1 | 0.079 |
| GW150914/L1 | 0.516 |
| GW170814/H1 | 0.865 |
| GW170814/L1 | 0.357 |
| GW170814/V1 | 0.353 |
| GW170104/H1 | 0.206 |
| GW170104/L1 | 0.144 |
| GW190521/H1 | 0.615 |
| GW190521/L1 | 0.825 |
| **Fisher χ²_F** | **19.58** |
| **dof** | **22** |
| **p_F** | **0.609** |

χ²_F = 19.58 on 22 degrees of freedom.  The 5th and 95th percentiles of
χ²(22) are 12.3 and 33.9 respectively; the observed statistic sits at the
35th percentile — centrally distributed under the null, not an outlier in
either direction.

## Verdict

**The sigma-ground ξ-shell echo prediction is consistent with null across the
full 5-event, 11-detector-event O1–O3a catalog sample examined.**

Specifics:
1. No single detector-event produces a per-echo |SNR| > 2.5σ (background p99
   threshold) at any predicted Δt_n.  The largest single-echo excess is GW170104
   L1 n=1 at 2.19σ (p=0.037), which is ANTI-coherent with H1's n=1 (+1.56σ
   opposite sign), ruling out a common astrophysical origin.
2. No event shows sign-coherent excess across two detectors at the predicted
   delays.  Sign matches at chance (3/5, p=0.5 or worse) in every case.
3. GW170814 L1×V1 = 0/5 sign matches is the single most anomalous coherence
   result, but it is anti-coherent (negative product statistic) and consistent
   with chance after multiple-testing correction.
4. Fisher combination: χ²_F = 19.58, dof = 22, p_F = 0.609 — the 11
   p-values are statistically indistinguishable from uniform[0,1].  The catalog
   contains no aggregate signal at this pipeline sensitivity.

**What the null teaches:**
- σ_conv = −ln(ξ) ≈ 1.844 as the Δt_n spacing parameter is not refuted:
  the pipeline sensitivity floor (|SNR| ≈ 2.5σ per echo, no template bank)
  cannot detect echoes at theoretically-expected amplitude A_echo ≲ 10⁻⁴·A_ring.
- γ_coh = 0.7924 open-input knob remains unconstrained; echo amplitude is
  the unknown, not σ_conv.
- The no-subtract pipeline is healthy: whitened-residual std 0.37–0.64 (sub-
  unity due to PSD overestimate, corrected by bootstrap calibration), bg p99
  2.4–2.9σ (Gaussian expectation ≈ 2.3σ for 2000 samples), Fisher p = 0.61
  (consistent with null).

## Cross-detector coherence summary — all pairs

| event | pair | sign | binomial p | z_c | p_c | verdict |
|---|---|---|---|---|---|---|
| GW151226 | H1×L1 | 3/5 | 0.500 | +0.24 | 0.385 | null |
| GW150914 | H1×L1 | 3/5 | 0.500 | +1.83 | 0.037 | marginal (1.8σ) |
| GW170814 | H1×L1 | 3/5 | 0.500 | −0.43 | 0.693 | null |
| GW170814 | H1×V1 | 2/5 | 0.812 | +0.47 | 0.291 | null |
| GW170814 | L1×V1 | 0/5 | 1.000 | −2.41 | 0.986 | anti-coherent (noise) |
| GW170104 | H1×L1 | 2/5 | 0.812 | −1.47 | 0.935 | null |
| GW190521 | H1×L1 | 3/5 | 0.500 | −0.03 | 0.511 | null |

The GW150914 H1×L1 marginal (z_c = +1.83, p_c = 0.037) from Phase I.3 is
an isolated excess: sign-coherence is still at chance (3/5), no corroborating
excess on any other event or pair, and it carries no Fisher weight beyond
its 0.037 single-event p-value.

## Next-step recommendations

1. **pycbc + SXS matched-filter bank** (highest priority for sensitivity).
   Current pipeline floor: single-delay matched-filter at point Δt_n with
   20ms template, sensitivity ≈ 2.5σ per echo.  pycbc with SXS NR-derived
   echo templates and a full template-bank search across (M, a*) could lower
   the floor by a factor of ~5 through optimal filtering.  Install path:
   `conda install -c conda-forge pycbc lalsuite` (prebuilt wheels, bypasses
   Windows CMake issue).  Estimated gain: detect echoes at A_echo ≳ 10⁻⁵·A_ring.

2. **Frequency-domain stacking across the 5-event catalog**.  Even without
   SXS templates, coherent stacking of post-merger FFTs (phase-aligning each
   event at its Δt_1·f_QNM) can pull a coherent echo signal below the per-
   event 2.5σ floor by √N ≈ 2.2× for N=5 events.  Pure-numpy, fits in the
   current pipeline, ~1 session to implement and run.

3. **Catalog growth** (GWTC-3 events: GW200202_154313, GW191230_180458, etc.)
   The three O1/O2/O3a IMBH events (GW170729, GW190814, GW190412) have
   different mass regimes and could be added with the same `search_event`
   call pattern.  Each is a one-line EVENTS dict entry + strain pull.

4. **Phase J footnote** (low effort).  Aaron's galaxy-isotropy observation
   (isotropic interior universe → NET J ≈ 0 → parent BH horizon is pure-Kerr,
   no additional disc-flattening) tightens the Phase J prediction from "horizon
   deforms toward a platter" to "horizon oblateness = Kerr(a*) exactly."  EHT
   shadow measurements of M87* and Sgr A* constrain this at ~10% level; a
   one-section addendum to `bh_horizon_sigma_conv_identity.md` would formalise
   this reduction.

## Cross-references

- `misc/bh_phase_i_3_no_subtraction_results.md` — Phase I.3 (subtraction
  ablation diagnostic; established no-subtract as clean baseline)
- `misc/bh_phase_i_2_overtone_coherence_results.md` — Phase I.2 (overtone
  subtraction cherished failures + cross-detector coherence framework)
- `misc/bh_phase_i_echo_search_results.md` — Phase I (original pipeline)
- `misc/bh_merger_predictions.md` — Δt_n ground-truth table
- `misc/bh_horizon_sigma_conv_identity.md` — Phase J
- `sigma_ground/field/interface/ligo_echo_search.py` — pipeline source
