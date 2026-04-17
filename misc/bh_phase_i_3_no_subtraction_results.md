# Phase I.3 — ξ-shell echo search, no-subtraction diagnostic

| Field | Value |
|---|---|
| Date | 2026-04-17 |
| Phase | I.3 (ξ-echo pipeline, subtraction-ablation diagnostic) |
| Pipeline module | `sigma_ground/field/interface/ligo_echo_search.py` |
| Events | GW150914 (H1+L1 4 kHz), GW151226 (H1+L1 16 kHz) |
| Prior docs | `bh_phase_i_echo_search_results.md`, `bh_phase_i_2_overtone_coherence_results.md` |
| Captured stdout | `misc/bh_phase_i_3_no_subtract_output.txt` |

## Variable glossary

- `subtraction_mode[m]` — pipeline branch controlling ringdown removal before
  matched-filter: `'none'` skips subtraction; `'n0'` subtracts the literature
  Kerr 2,2,0 fundamental via 2-parameter linear LS (Phase I/I.2 default);
  `'kerr_constrained'` and `'kerr_unconstrained'` retain multi-overtone
  subtraction for reference — both documented-failed in Phase I.2.
- `|SNR|_m[σ]` — calibrated matched-filter |SNR| at predicted Δt_n under
  subtraction mode m.
- `|L1|/|H1|[r]` — per-echo magnitude ratio; real astrophysical signal has
  this O(1) from antenna-pattern geometry, detector-specific noise leaves
  it unbounded.
- `sign_matches[k]` — number of predicted echoes where sign(SNR_H1) ==
  sign(SNR_L1); k/5 with binomial p = P(≥k | p=0.5, n=5).
- `coherence_z[z_c]` — magnitude-weighted Σρ_H1·ρ_L1 z-score against off-
  source null; `coherence_p[p_c]` is the one-sided p-value.
- `A_fit[A_f]` — fit amplitude of the damped-sinusoid QNM template; Kerr
  waveform literature puts A_22 ≈ 1–3 × 10⁻²¹ at GW150914 distance.

## Pipeline change since Phase I.2

Added `subtraction_mode` parameter to `search_event()`.  All four paths
(`'none'`, `'n0'`, `'kerr_constrained'`, `'kerr_unconstrained'`) share the
same bandpass / PSD / whitening / matched-filter / bootstrap machinery and
diverge only in how (if at all) the ringdown template gets built and
subtracted before whitening.

`'none'` builds the matched-filter template from literature Berti–Cardoso–
Starinets (f, τ) rather than from a fit — this is the ablation step that
isolates whether the Phase I combined p ≈ 0.003 excess comes from the
predicted Δt_n structure or from the subtraction operator itself.

## Results

### GW151226 (16 kHz)

Per-echo |SNR|, comparing n0 vs none:

| det | n | Δt_n [ms] | \|SNR\|_n0 [σ] | \|SNR\|_none [σ] | Δ [σ] |
|---|---|---|---|---|---|
| H1 | 1 | 0.756 | 1.22 | 0.07 | +1.14 |
| H1 | 2 | 1.512 | 3.32 | 0.41 | +2.91 |
| H1 | 3 | 2.267 | 3.24 | 0.20 | +3.04 |
| H1 | 4 | 3.023 | 2.43 | 0.34 | +2.09 |
| H1 | 5 | 3.779 | 0.33 | 0.92 | −0.58 |
| L1 | 1 | 0.756 | **32.57** | 0.41 | +32.16 |
| L1 | 2 | 1.512 | **34.69** | 1.21 | +33.48 |
| L1 | 3 | 2.267 | **25.38** | 0.32 | +25.06 |
| L1 | 4 | 3.023 | **11.57** | 0.28 | +11.29 |
| L1 | 5 | 3.779 | **5.72** | 0.20 | +5.52 |

No-subtract per-echo |SNR| all fall below the per-detector background
p99 (2.49 H1, 2.48 L1).  The Phase I.2 combined p = 0.003 excess is
entirely a property of the subtraction operator — removing the operator
removes the signal.

### GW150914 (4 kHz)

| det | n | Δt_n [ms] | \|SNR\|_n0 [σ] | \|SNR\|_none [σ] | Δ [σ] |
|---|---|---|---|---|---|
| H1 | 1 | 2.260 | 0.47 | 0.78 | −0.31 |
| H1 | 2 | 4.520 | 1.43 | 2.04 | −0.61 |
| H1 | 3 | 6.781 | 1.49 | 0.97 | +0.53 |
| H1 | 4 | 9.041 | 1.38 | 1.68 | −0.30 |
| H1 | 5 | 11.301 | 1.18 | 1.29 | −0.12 |
| L1 | 1 | 2.260 | **12.03** | 0.62 | +11.41 |
| L1 | 2 | 4.520 | **4.43** | 1.57 | +2.86 |
| L1 | 3 | 6.781 | **5.25** | 0.03 | +5.21 |
| L1 | 4 | 9.041 | **4.04** | 0.59 | +3.45 |
| L1 | 5 | 11.301 | **4.35** | 1.08 | +3.27 |

H1 behaves similarly under both modes (n0 fit is well-behaved on H1 — the
residual is approximately the raw strain modulo a small QNM template).
L1 produces 4–12σ spurious excess under n0-subtract but drops to < 1.6σ
under no-subtract.  Again, the "excess" is operator-induced.

### Cross-detector coherence under no-subtract

| event | Σρ_H1·ρ_L1 | null μ,σ | z_c | p_c | sign matches | binomial p |
|---|---|---|---|---|---|---|
| GW151226 | +0.545 | +0.009, 2.214 | +0.24 | 0.385 | 3/5 | 0.500 |
| GW150914 | +4.063 | +0.009, 2.219 | +1.83 | 0.037 | 3/5 | 0.500 |

GW151226 is a clean null under every coherence diagnostic.

GW150914 shows a modest 1.83σ magnitude-weighted coherence, driven primarily
by n=2 (product +3.20 = 2.04×1.57, both negative) and n=5 (+1.40 =
1.29×1.08, both positive).  But:
- Sign-coherence is at chance (3/5, p = 0.5).
- n=3 has \|L1\|/\|H1\| = 0.03 (L1 essentially zero; H1 = −0.97).
- n=1 flips sign between detectors (H1 +0.78, L1 +0.62 — actually same
  sign but small magnitudes).

A real echo train would produce sign-coherence well above chance across
the full n-series, not a single-point magnitude excess with sign flips
elsewhere.  The correct interpretation is a chance magnitude coincidence
on one detector pair, not a weak signal.

### Magnitude-ratio diagnostic

|L1|/|H1| under no-subtract:

- GW151226: 5.75, 2.96, 1.60, 0.84, 0.22 — range compatible with noise
  (antenna-pattern ratio ≈ 1, but wide scatter at these low \|SNR\|
  values is expected).
- GW150914: 0.80, 0.77, 0.03, 0.35, 0.83 — tight around 0.8 except n=3.

Compare to n0-subtract, where these ratios ran 5–27× — diagnostic of
fit-amplitude-driven noise injection, not astrophysical signal.

## Mechanism of the subtraction-induced pathology

The n=0 fit (2-parameter linear LS on a 20 ms window starting 3 ms past
merger) finds the coefficient (c, s) that minimises
`||y - (c·env·cos(2πft) + s·-env·sin(2πft))||²` with f = f_QNM and
τ = τ_QNM fixed from literature.  On GW151226 L1, where the local SNR is
low and the noise has power in the 700–800 Hz band, this fit returns
A_fit = 8.7 × 10⁻²⁰ — a factor of ~30 higher than the physical QNM
amplitude at that event's distance (expected A_22 ≈ 2–4 × 10⁻²¹).

Evaluating that inflated template forward in time (10τ ≈ 12 ms) and
subtracting it from the residual INJECTS the noise pattern at frequencies
near f_QNM into the matched-filter window.  The template the matched-
filter correlates against is ALSO tuned to f_QNM (and τ_QNM for envelope
weighting).  The matched-filter then reports high |SNR| because it's
correlating its own template against an amplified image of itself.

Evidence this is the mechanism:
- Per-echo SNRs alternate sign — a broadband frequency-matched noise
  injection at random phase produces exactly this pattern when sampled
  at Δt_n = n·0.756 ms (GW151226) for n = 1..5.
- L1 A_fit / H1 A_fit = 8.7e-20 / 1.5e-20 = 5.7× — consistent with L1's
  higher detector noise in the 737 Hz band at GW151226 time.
- The spurious |SNR| scales with A_fit: GW151226 L1 at 35σ, GW150914 L1
  at 12σ (4× lower A_fit, ~3× lower spurious excess).
- Removing the subtraction operator collapses all |SNR| < 2.5σ (well
  within Gaussian bg p99).

## Verdict

**The sigma-ground ξ-shell echo prediction (Δt_n = 2·r_s·n·σ_conv/c) is
neither detected nor falsified by this pipeline against GW150914 and
GW151226 strain.**

What *is* established:

1. The Phase I combined p ≈ 0.003 excess on GW151226 L1, and the
   catalog-combined 5.4σ in Phase I.2, are **artefacts of the
   subtraction operator**, not weak signals.  With subtraction disabled
   the excess vanishes entirely (min p-value goes from 0.0000 to 0.879
   on the worst-affected detector-event).
2. The pipeline **is adequate** to produce a clean null on both events
   when operated in no-subtract mode.  Whitening calibration (bg_std
   normalised to 1.0) and background bootstrap (2000 off-source samples,
   p99 ≈ 2.5) are functioning correctly.
3. The pipeline is **inadequate** as a detection instrument: with only
   matched-filter-at-predicted-delay and no template bank, the noise
   floor is the 2.5σ p99.  A weak echo (peak \|SNR\| ≈ 3–5σ in each
   detector) would be visible but a faint one (\|SNR\| ≲ 2σ) would not
   be distinguishable from noise.
4. Cross-detector sign-coherence (3/5 on both events, p = 0.5 binomial)
   is at chance — no hint of a shared astrophysical signal at the
   ξ-predicted delays in either event.

What this means for the sigma-ground Phase G/H/J doc set:

- σ_conv = −ln(ξ) ≈ 1.844 as the conversion factor between r_s and echo
  delay is **not refuted** — the pipeline cannot resolve signals below
  ~3σ per echo, and the prediction is a shape (Δt_n ∝ n), not an
  amplitude.  A model with A_echo ~ 10⁻⁴·A_ringdown (theoretically
  motivated for wormhole-style echoes) is below this pipeline's floor.
- The γ_coh = 0.7924 open-input knob in Phase G — which was tied to
  echo amplitude through the "γ-mode" table — is not constrained by
  Phase I.3.  Further constraints require either (a) a detector with
  better low-\|SNR\| sensitivity, or (b) a full template bank
  matched-filter in pycbc, or (c) joint N-event stacking in the
  frequency domain.
- The Phase J horizon-identity prediction (that the parent BH's horizon
  shape reflects its a*-set oblateness with NO additional flattening
  from interior-J accumulation, per the galaxy-isotropy reduction
  Aaron observed) is untouched by Phase I.3 — that test lives in
  horizon imaging (EHT), not ringdown echoes.

## Learnings cherished from Phase I.3

- **Always run a no-subtraction ablation when a signal is spatially
  coincident with the subtracted template's frequency band.**  The
  matched-filter + subtracted-residual combination is only trustworthy
  when the subtraction is well-calibrated against injections; with
  real broadband noise in the f_QNM band the subtraction amplifies
  rather than removes.
- **Sign-coherence beats magnitude-coherence for detector-pair
  validation.**  The magnitude-weighted Σρ_H1·ρ_L1 was z = 83 on
  GW151226 under n0-subtract — entirely driven by L1's 30σ artefact
  magnitudes.  Sign-only binomial at 4/5 (p = 0.19) correctly refused
  to call that a detection.
- **A_fit / A_literature > 2 is a pipeline-health flag.**  On GW151226
  L1, A_fit was 30× literature — the pipeline should have refused to
  subtract and instead flagged "fit inconsistent with astrophysical
  prior."  That guardrail did not exist in Phase I/I.2 and created the
  subtraction pathology.

## Next-step recommendations, priority ordered

1. **Catalog extension — highest learning/run ratio.**  GW170814
   (H1+L1+V1, three-detector coherence), GW170104 (H1+L1, similar to
   GW150914 mass regime), GW190521 (IMBH-scale, longer Δt_1). Each run
   adds an independent null/excess check without further pipeline
   development.  Per-event strain pull is ~20 MB × 4 files ≈ 80 MB per
   event.  With no-subtract mode now default-safe, the pipeline is
   ready — `search_event_both_detectors('GW170814', ...,
   subtraction_mode='none')` is a one-line extension.

2. **pycbc + SXS template bank.**  Deferred in Phase I.2 due to
   Windows/CMake compiler install blocker; the route forward is
   `conda install -c conda-forge pycbc lalsuite` which ships prebuilt
   wheels and bypasses the source build.  A proper IMR-consistency
   residual + SXS echo template bank would let us detect \|SNR\| ≈ 1σ
   signals via template-stacking that the current matched-filter-at-
   point cannot resolve.  Estimated session time: 1-2 hours for install
   + 1-2 hours for the pipeline adaptation + 2-4 hours for the catalog
   re-run.

3. **Frequency-domain stacking across catalog.**  Even without SXS
   templates, stacking N events' post-merger FFTs at their per-event
   Δt_n·f_QNM phase offsets should pull coherent echo signals out
   below the per-event 2.5σ floor.  This is pure-numpy, fits in
   ligo_echo_search.py, and could be an afternoon's work after the
   catalog extension.

4. **Phase J horizon-imaging follow-up.**  Aaron's galaxy-isotropy
   observation — that NET interior J of an isotropic universe is zero,
   so the parent BH's horizon should be pure-Kerr with no additional
   disc-flattening — tightens the Phase J prediction to "horizon
   oblateness exactly matches a* from ringdown."  EHT measurements of
   M87* and Sgr A* shadow shapes already constrain this at ~10% level;
   a directed sigma-ground-vs-Kerr comparison would be a ~1-session
   doc write-up.

## Cross-references

- `misc/bh_phase_i_echo_search_results.md` — original Phase I verdict
  (combined p = 0.003 claimed, now identified as artefact)
- `misc/bh_phase_i_2_overtone_coherence_results.md` — Phase I.2
  constrained/unconstrained overtone cherished failures + cross-
  detector coherence framework
- `misc/bh_merger_predictions.md` — ground-truth Δt_n table used by
  this pipeline
- `misc/bh_horizon_sigma_conv_identity.md` — Phase J; candidate for
  isotropy-reduction footnote
- `sigma_ground/field/constants.py` — SIGMA_CONV, XI, γ_coh definitions
- `sigma_ground/field/interface/ligo_echo_search.py` — pipeline source
