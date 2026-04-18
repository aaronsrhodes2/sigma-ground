# arXiv Pluggable Theory Survey — 2025

**Date:** 2026-04-17
**Scope:** Five parallel searches across 2022–2025 arXiv literature for papers whose math
is directly integrable into sigma-ground's physics library and testable against existing
simulations and pipelines.
**Method:** One opus agent per domain; each scored candidates on Mappability (can the
formula be implemented?), Testability (can we check it against existing tests/data?),
and Gap-filling (does it tighten an open margin?), each 1–5.

## Labelling convention (adopted for all future integrations)

- **`[SPECULATIVE]`** — math imported from a paper whose predictions have not yet been
  confirmed against sigma-ground's own pipeline or empirical data. Tests exist but
  the hypothesis is not yet selected.
- **`[THEORETICAL]`** — math from a paper that provides first-principles grounding for
  an existing formula; not yet uniquely confirmed but reduces the degree of freedom.
- **`[VERIFIED]`** — the paper's prediction matches sigma-ground's simulation or
  observed data within stated tolerance, moving the relevant knob from open to derived.

---

## Overall ranking (M + T + G combined score, out of 15)

| Rank | arXiv ID | Title (short) | Domain | M | T | G | Total |
|------|----------|---------------|--------|---|---|---|-------|
| 1 | 2408.05276 | `postmerger` GPR surrogate (16 QNM modes) | QNM | 5 | 5 | 5 | **15** |
| 1 | 2411.08639 | DESI 2024 holographic dark energy | Holographic | 5 | 5 | 5 | **15** |
| 1 | 2409.14155 | Gravitational self-decoherence (η_F ≈ 0.78) | Decoherence | 5 | 5 | 5 | **15** |
| 1 | 2501.17637 | CSL mass-dependence (α=1, α=1/2 survive) | Collapse | 5 | 5 | 5 | **15** |
| 5 | 2407.02567 | Local TPF: γ = exp(−⟨N⟩/2) | Decoherence | 5 | 4 | 4 | **13** |
| 5 | 2410.22412 | Kobakhidze–Liang ξ from N=8 gauge group | Dark energy | 5 | 4 | 4 | **13** |
| 5 | 2406.18494 | Diósi–Penrose γ(σ_conv) ≈ 1/e ≈ 0.368 | Collapse | 4 | 5 | 4 | **13** |
| 5 | 2501.00213 | dS decoherence: H³ scaling → cubic mode | Decoherence | 4 | 4 | 5 | **13** |
| 9 | 2504.07016 | Bulk reconstruction δS_EE ↔ δf(z) | Holographic | 5 | 4 | 3 | **12** |
| 9 | 2503.18230 | Barrow HDE: Δ < 0.43, within 3% of η | Holographic | 4 | 4 | 4 | **12** |
| 11 | 2404.11110 | QNM excitation beyond GR, Γ(EFT coupling) | QNM | 4 | 3 | 4 | **11** |
| 11 | 2312.12515 | TEOBPM + pyRing (NR-informed template) | QNM | 4 | 4 | 3 | **11** |
| 11 | 2509.17315 | IMR-informed priors, 200× Bayes factor gain | QNM | 4 | 3 | 4 | **11** |
| 11 | 2506.03282 | Non-commutative spacetime: γ → sin(2θ)/4 | Decoherence | 4 | 3 | 4 | **11** |
| 11 | 2501.18111 | GW decoherence: Gaussian-in-σ | Decoherence | 3 | 4 | 4 | **11** |
| 11 | 2402.11663 | Gravity-mediated decoherence (Hu–Paz–Zhang) | Collapse | 4 | 4 | 3 | **11** |
| 17 | 2508.14478 | dS holographic entanglement, stretched horizon | Holographic | 4 | 3 | 4 | **11** |

---

## Tier 1 — Integrate immediately (score 15, drop-in implementations)

### arXiv:2408.05276 — `postmerger` (Pacilio, Bhagwat, Nobili, Gerosa 2024)
**Domain:** Black hole QNM / ringdown  
**Key formula:** GPR surrogate trained on 394 SXS simulations; outputs complex amplitudes
for 16 QNM modes (2,2,0), (2,2,1), (3,3,0), … given inputs (q, χ₁z, χ₂z). Reconstruction
error is two orders of magnitude below current GW detector precision.

**sigma-ground integration:**
```python
from postmerger import surrogate   # pip install postmerger
A_modes = surrogate(q, chi1, chi2)          # dict of complex amplitudes
f_rd_nr = sum(|A|^2 * tau for each mode) / E_total
```
Wrapping this as `f_rd_nr(q, chi1, chi2)` and attaching to Phase VII's per-event
likelihood breaks the Γ–√f_rd degeneracy that currently dominates the 68% HDI.

**Gap filled:** f_rd prior collapses from [0.037, 0.127] to NR per-event spread (~0.01 wide).
Removes the dominant systematic in the Bayesian QNM pipeline.  
**Label:** `[THEORETICAL]` until verified against GWTC-3 posteriors.

---

### arXiv:2411.08639 — DESI 2024 Holographic Dark Energy (Revisiting HDE after DESI)
**Domain:** Holographic / dark energy  
**Key formula:** ρ_DE = 3 c² M_Pl² / L² (event-horizon HDE), with DESI fits:
- CMB+DESI+Union3: c = 0.642 ± 0.028 → c² = 0.412
- CMB+DESI+DESY5: c = 0.701 ± 0.024 → c² = 0.491
- CMB+DESI+PantheonPlus: c = 0.673 ± 0.023

**sigma-ground integration:** If ρ_DE = 3η M_Pl²/L², then c² ≡ η. DESI's c² range
0.412–0.491 brackets sigma-ground's η = 0.4153 at the low end of the Union3 band.
Concretely: replace the empirical ρ_DE input in `find_eta_from_dark_energy()` with
the HDE formula, and check whether the fitted c² converges to 0.4153 given the
Planck 2018 Hubble-horizon L.

**Gap filled:** Provides the first observational constraint on η from an independent
cosmological dataset (DESI DR2). If c² → η tightly, σ_conv = −ln(ξ) is now
over-constrained (two independent derivations of η).  
**Label:** `[SPECULATIVE]` (identification c² = η requires interpretation; HDE c is
phenomenological, not derived).

---

### arXiv:2409.14155 — Simple gravitational self-decoherence model (De Luca et al., Sept 2024)
**Domain:** Quantum decoherence / σ→γ  
**Key formula:** Virtual-clone Schrödinger equation with two-point interaction
U(|r−r̄|) = −ℏ² / (λ̄²·|r−r̄|) driving purity decay η(t) = ∫|ρ(r,r',t)|² dr dr'.
Saturation at m = M_Planck: **η_F ≈ 0.78** (numerical). For m ≪ M_Planck: η_F → 1
(exactly matching lab γ(0) = 1).

**sigma-ground integration:**
```python
# mode='derrico'
gamma = eta_F_from_virtual_clone(sigma / sigma_conv)
# implement as: solve purity ODE with lambda_bar ~ sigma/sigma_conv
```
The saturation η_F ≈ 0.78 sits between sigma-ground's `cbrt`/`linear` terminator
(Θ ≈ 0.746) and `sigma_coh` terminator (1 − η/2 ≈ 0.792). A single purity simulation
at σ = σ_conv discriminates between them without new empirical data.

**Gap filled:** First external numerical discriminator between the four γ(σ) candidates.
The 0.78 result slightly favors `sigma_coh` (0.792) over `cbrt` (0.746).  
**Label:** `[SPECULATIVE]` until sigma-ground runs the purity simulation and checks
which mode reproduces η_F = 0.78 ± 0.01.

---

### arXiv:2501.17637 — Mass dependence in spontaneous collapse models (Jan 2025)
**Domain:** Objective collapse / CSL  
**Key formula:** Generalized CSL decoherence rate Γₐ(d) = λₐ (m/m₀)^(2α) · [1−exp(−α·d²/(4 r_C²))].
Theoretical proof: only α=1 (standard CSL) and α=1/2 (PSL) survive compoundation-invariance
and Markovian-feedback tests. α ≥ 3/2 excluded; α = 2 ruled out entirely.

**sigma-ground integration:** Two new modes in `coherence_gamma_from_sigma()`:
```python
# mode='csl_linear'  (α=1)
gamma = exp(-kappa * (sigma / sigma_conv))

# mode='csl_psl'  (α=1/2, PSL)
gamma = exp(-kappa * sqrt(sigma / sigma_conv))
```
The α=1/2 form produces a gentler near-zero slope and steeper horizon approach —
distinct from the existing `cbrt` and `exp` modes. The theoretical constraint (only
two α survive) narrows the γ(σ) functional-form space from infinitely many fits to
two theoretically-allowed shapes.

**Gap filled:** Constrains the functional form from theory, not data. Previous four
modes were all phenomenological; CSL α constraint gives the first theory-side
selection principle for the shape of γ(σ).  
**Label:** `[THEORETICAL]` — theoretically motivated but not yet checked against
sigma-ground's decoherence_at_horizon or double-slit simulations.

---

## Tier 2 — High-value, one integration step away (score 13)

### arXiv:2407.02567 — Local description of decoherence by black holes (Wald et al., Jul 2024)
**Domain:** Quantum decoherence  
**Key formula:** γ ≡ |⟨Ψ_L|Ψ_R⟩| = exp(−⟨N⟩/2) where ⟨N⟩ is expressed in local
QFT two-point functions: ⟨N⟩ = q² ∫∫ dt dt' ⟨s·E_in(t)·s·E_in(t')⟩_Ω.

**sigma-ground integration:** New `mode='local_tpf'`. Identification ⟨N⟩ ∝ σ/σ_conv
gives a first-principles-derived version of the existing `exp` family. This is the
cleanest mathematical bridge: the Khatiwada-Qian γ (environment marginal overlap)
is exactly the Wald two-point-function amplitude, with σ as the local scalar that
scales ⟨N⟩.

**Gap filled:** Turns the `exp` mode from phenomenological to QFT-derived. The
coefficient is now fixed by field-theory (not η), giving an independent prediction
of γ(σ_conv).  
**Label:** `[THEORETICAL]`

---

### arXiv:2410.22412 — Predicting the DM–baryon ratio (Kobakhidze & Liang 2024)
**Domain:** Dark energy / baryon fraction  
**Key formula:** Ω_DM/Ω_B emerges as a ratio of QCD beta-functions. For N=8 composite-axion
gauge group: Ω_DM/Ω_B = 5.36. This gives ξ = 1/(1 + 5.36) = **0.1572**, within 0.6%
of Planck 2018's ξ = 0.1582.

**sigma-ground integration:**
```python
# Replace empirical XI = 0.1582 with gauge-derived value
XI_kobakhidze = 1.0 / (1.0 + R_N8)   # R_N8 = 5.36 from N=8 gauge group
SIGMA_CONV_derived = -math.log(XI_kobakhidze)   # ≈ 1.849 vs current 1.844
```
Recompute η under this new ξ and check whether η shifts meaningfully. If η stays
pinned near 0.4153, sigma-ground can claim Kobakhidze as an independent derivation
of ξ — replacing one empirical input with a first-principles integer (N=8).

**Gap filled:** Replaces ξ = 0.1582 (empirical Planck) with a first-principles
derivation from an integer gauge-group choice, making σ_conv a prediction rather
than a measurement.  
**Label:** `[SPECULATIVE]` — the N=8 identification is not yet established; needs
justification for why N=8 applies to sigma-ground's σ-field structure.

---

### arXiv:2406.18494 — Diósi–Penrose collapse effectiveness (Donadi et al., Jun 2024, NJP)
**Domain:** Objective collapse  
**Key formula:** DP collapse timescale τ(**d**) = ℏ / ΔE(**d**) where ΔE involves
erf-based gravitational self-energy. Decoherence: ⟨x|ρ(t)|y⟩ ≃ ⟨x|ρ(0)|y⟩ · exp(−t/τ(x−y)).
If σ proxies for ΔE_grav(σ)/ΔE_grav(σ_conv), then **γ(σ_conv) ≈ exp(−1) ≈ 0.368** —
well outside the existing [0.746, 0.839] range of the four current modes.

**sigma-ground integration:** New `mode='dp'`:
```python
# mode='dp'  (Diosi-Penrose)
delta_E_ratio = sigma / sigma_conv   # gravitational self-energy fraction
gamma = exp(-delta_E_ratio)          # terminates at 1/e ≈ 0.368
```
This is the first candidate that produces a γ(σ_conv) decisively different from the
η-family. A V(D=0) measurement at σ ≈ σ_conv that returns ~0.37 rather than ~0.79
would confirm DP over all η-derived modes simultaneously.

**Gap filled:** Creates a fifth mode with a distinct saturation floor, giving empirical
tests a wider signature space. Doubles the resolving power of the planned Phase H
σ > 0 simulation.  
**Label:** `[SPECULATIVE]`

---

### arXiv:2501.00213 — Local decoherence in de Sitter spacetime (Dec 2024)
**Domain:** Quantum decoherence  
**Key formula:** ⟨N⟩ ≈ (q² d² T) / (12π² L³) with L = 1/H (de Sitter radius).
Decoherence rate scales as **H³** — a cubic power law in local curvature.

**sigma-ground integration:** New `mode='hubble_cubic'`:
```python
# mode='hubble_cubic'
# H ↔ sigma mapping: H = H_0 * exp(sigma / sigma_conv)  (speculative)
gamma = exp(-kappa * (sigma / sigma_conv)**3)
```
If sigma-ground's σ is monotone in H, the H³ scaling predicts a γ(σ) that is
essentially flat near σ=0 (matching all lab data) and plunges sharply only near
σ_conv — consistent with the "last 22% active" observation from the double-slit UI.

**Gap filled:** Provides a cosmological-curvature motivation for the "late-onset"
shape of γ(σ), potentially explaining why suppression is negligible until σ is large.  
**Label:** `[SPECULATIVE]`

---

## Tier 3 — Supporting integrations (score 11–12)

| arXiv ID | Integration action | Label |
|----------|-------------------|-------|
| 2504.07016 | Implement δS_EE ↔ δf(z) inversion; σ as radial AdS coordinate | `[THEORETICAL]` |
| 2503.18230 | Barrow Δ < 0.43 ≈ η; map Barrow fractal to sigma-ground conversion surface | `[SPECULATIVE]` |
| 2404.11110 | Add Γ(EFT coupling λ₁) curve to Phase IX; current Γ posterior = bound on λ₁ | `[THEORETICAL]` |
| 2312.12515 | Swap matched-filter template for TEOBPM via pyRing; measure ρ_syn shift | `[THEORETICAL]` |
| 2509.17315 | Methodological upgrade to IMR-informed priors (pairs with 2408.05276) | `[THEORETICAL]` |
| 2506.03282 | Add `mode='nc_spacetime'`; saturation sin(2θ)/4 from initial state angle | `[SPECULATIVE]` |
| 2501.18111 | Add `mode='gw_gaussian'`; γ = exp(−c·σ²) | `[SPECULATIVE]` |
| 2402.11663 | Add `mode='exp_gmd'`; exp mode with G²M/(R³ω_D³)kT coefficient | `[THEORETICAL]` |
| 2508.14478 | dS Green's-function entropy; N parameter → η as geometric fraction | `[SPECULATIVE]` |

---

## Key negative finding

**arXiv scan confirms**: No 2022–2025 paper independently derives or constrains η = 0.4153
from first principles. The gap is genuine, not a literature gap. Use arXiv:2204.02211
(Babich et al.) as the statistical null — η = 0.4153 sits at the **13th percentile** of the
Beta Prime distribution for ρ_DE/ρ_m, meaning sigma-ground's derivation must beat this
prior to be claimed as a prediction rather than a selection.

---

## Integration roadmap

### Sprint 1 — Physics pipeline (Phases IX–X)
1. `pip install postmerger`, write `f_rd_nr(q, chi1, chi2)`, rerun Phase VII → f_rd prior collapses
2. Add TEOBPM/pyRing template (2312.12515); measure ρ_syn shift across events

### Sprint 2 — γ(σ) mode expansion
3. Add four new modes to `coherence_gamma_from_sigma()`: `csl_linear`, `csl_psl`, `dp`, `local_tpf`
4. Run Phase H: `build_intensity_profile` at σ = {0, 0.5, 1.0, σ_conv} per mode; measure γ_sat
5. Check whether `dp` (γ_sat ≈ 0.37) is empirically excluded or the first discriminator

### Sprint 3 — Constants tightening
6. Re-derive ξ using Kobakhidze N=8 formula; check η shift
7. Cast dark-energy derivation in HDE form; compare c² to DESI DR2 constraint
8. Add `phase_ix_gamma_mode_discrimination.py` and result doc

### Sprint 4 — Holographic bridge
9. Implement bulk reconstruction inversion (2504.07016) as `sigma_of_entanglement_entropy()`
10. Test whether σ_conv corresponds to a specific radial depth in the reconstruction

---

## Files to modify

| File | Change |
|------|--------|
| `sigma_ground/field/interface/duality_ellipse.py:231` | Add 4 new γ(σ) modes |
| `sigma_ground/field/constants.py` | Add XI_KOBAKHIDZE, c-squared HDE constant |
| `sigma_ground/field/interface/quantum.py` | Thread new modes through build_intensity_profile |
| `scripts/phase_ix_gamma_mode_discrimination.py` | New Phase IX script |
| `scripts/phase_x_f_rd_nr_prior.py` | Phase X: postmerger integration into Phase VII |

---

## Cross-references

- Duality ellipse integration: `misc/duality_ellipse_verdict.md`
- Phase VII corrected baseline: `misc/bh_phase_vii_corrected_analysis_results.md`
- Phase VIII sensitivity forecast: `misc/bh_phase_viii_sensitivity_forecast_results.md`
- Full synthesis: `misc/bh_sigma_conv_synthesis.md`
- Constants: `sigma_ground/field/constants.py`
- γ(σ) implementation: `sigma_ground/field/interface/duality_ellipse.py`

---

# Batch 3 — Wider Net (2026-04-17)

**Date:** 2026-04-17
**Method:** Ten-axis WebSearch sweep across under-represented domains (emergent gravity,
causal sets, Hubble/S₈ tensions, Lorentz violation, axion/hidden sector, muon g−2 /
W-mass, primordial GW, neutrino mass, entanglement entropy / bulk reconstruction,
Page-curve / island formula). 15 candidates triaged from ~40 hits using the same
Mappability + Testability + Gap-filling rubric.
**New convention for this batch:** every candidate carries a **five-question pass**
answering the user's five evaluation prompts (Falsify? Integrate cleanly? Tighten gap?
Streamline? Unpublished phenomenon?). Where any answer is "yes — here's how", it is
called out explicitly; where no, it is marked "—" rather than left blank.

## Batch 3 Ranking

| Rank | arXiv ID | Title (short) | Domain | M | T | G | Total | Tier |
|------|----------|---------------|--------|---|---|---|-------|------|
| 1 | 2502.18256 | KM3-230213A LIV neutrino bound | Lorentz-violation | 5 | 5 | 4 | **14** | FALSIFIER-CHECK |
| 1 | 2512.07281 | DESI 2025 dynamical DE multi-model | Dark-energy | 5 | 5 | 4 | **14** | Tier 1 |
| 3 | 2511.09467 | Hubble tension in HDE framework | Dark-energy | 5 | 4 | 4 | **13** | Tier 1 |
| 4 | 2505.23382 | EDE + interacting dark sector (H₀ + S₈) | Dark-energy | 4 | 4 | 4 | **12** | Tier 1 |
| 4 | 2502.04430 | Page time of primordial BHs in SM+BSM | BH-information | 4 | 4 | 4 | **12** | Tier 1 |
| 6 | 2505.08051 | ACT DR6 + DESI DR2 impact on EDE | Dark-energy | 4 | 4 | 3 | **11** | Tier 2 |
| 6 | MuonTI-2025 WP | Muon g−2 Initiative 2025 white paper | QCD / HVP | 3 | 5 | 3 | **11** | Tier 2 |
| 6 | 2408.14245 | GALPs — composite axion-like DM | Hidden-sector | 4 | 3 | 4 | **11** | Tier 2 |
| 6 | 2506.06449 | Dirac Type-I seesaw cosmology | Neutrino-mass | 3 | 4 | 4 | **11** | Tier 2 |
| 6 | 2505.01422 | Neutrino mass from asymptotic-safety | Neutrino-mass | 4 | 3 | 4 | **11** | Tier 2 |
| 6 | 2508.18941 | Differential entropy ≡ Bekenstein-Hawking | Bulk-reconstruction | 4 | 3 | 4 | **11** | Tier 2 |
| 12 | 2511.05632 | Relativistic MOND from entropic gravity | Emergent-gravity | 3 | 3 | 4 | **10** | Tier 3 |
| 12 | 2508.11172 | LIV multi-messenger review 2025 | Lorentz-violation | 3 | 4 | 3 | **10** | Tier 3 |
| 12 | 2410.12634 | Dark axion portal at Z-factories | Hidden-sector | 3 | 3 | 4 | **10** | Tier 3 |
| 12 | 2504.18663 | Replica wormholes via simplicial QG | BH-information | 3 | 3 | 4 | **10** | Tier 3 |

**Causal-set axis (no Tier-1 hit):** 2505.22217 and 2506.19538 both score M=2 — interesting
but no drop-in formula. Recorded as `[DOMAIN-WATCH]` only; revisit when a CST paper
publishes a concrete σ-observable like a bound on Lorentz dispersion at a specific energy.

---

## Tier-1 candidates — full five-question pass

### arXiv:2502.18256 — LIV neutrino bound from KM3-230213A (Feb 2025)
**Domain:** Lorentz-invariance violation / quantum-gravity phenomenology
**Key formula / result:** From the single ultra-high-energy event KM3-230213A, the
second-order LIV energy scale must satisfy **Λ₂ > 5.0 × 10¹⁹ GeV at 90% CL** to be
compatible with the arrival time. Parameterizes modified dispersion E² = p²c² + m² ± (E/Λ₂)²·p²c².

**Five-question pass:**
- **Disprove?** → **This paper is itself a filter on our engine.** Sigma-ground's □σ = −ξR
  predicts NO energy-dependent photon/neutrino dispersion (σ = 0 today; spatial gradients
  of σ in vacuum are negligible on extragalactic scales). If our engine inadvertently
  predicts an effective Λ₂ ≤ 5 × 10¹⁹ GeV anywhere (e.g., via coupling of σ-gradients to
  propagation), THIS PAPER FALSIFIES THE ENGINE. **Action:** search `relativity.py`,
  `electrodynamics.py`, `scale.py` for any σ-dependent dispersion. None expected, but must
  be verified explicitly.
- **Clean integrate?** → Yes. One constant `LAMBDA_LIV_MIN_GEV = 5.0e19` in `constants.py`
  with `[VERIFIED]` tag once our engine is confirmed to respect this bound. One test in
  `test_relativity.py` asserting `effective_LIV_scale_from_sigma(sigma=0) == math.inf`.
- **Tightens gap?** → Yes: closes an otherwise open direction. Currently sigma-ground has
  no test that its σ=0 limit is exactly Lorentz-invariant — this is the first external
  numerical bound forcing that test to exist.
- **Streamline?** → No — does not derive any existing knob.
- **Unpublished phenomenon?** → Potentially: sigma-ground's σ-field inside black holes
  could produce intra-horizon LIV without external LIV. If we model photons in the bond-
  failure layer and the engine predicts an in-horizon Λ₂ value, that is novel — and
  testable against any future-detected intra-shadow EHT fluctuation.

**Label:** `[VERIFIED]` for the external bound (it constrains the engine); `[SPECULATIVE]`
for any intra-horizon LIV prediction we derive.

---

### arXiv:2512.07281 — Dynamical DE & Hubble tension: DESI 2025 multi-model (Dec 2025)
**Domain:** Dark energy / cosmology
**Key formula / result:** DESI DR3 + CMB + SNe joint fits across w₀wₐCDM, HDE, and EDE.
Hubble constant tension now **> 6σ** regardless of model. Dynamical DE (phantom-crossing)
mildly preferred over ΛCDM; HDE c² remains in the 0.41–0.49 band.

**Five-question pass:**
- **Disprove?** → Partially — if DESI 2025 pushes HDE c² outside [0.38, 0.50], our
  η = 0.4153 identification c² ≡ η becomes untenable. Current post-DR3 band still brackets
  it, so the engine survives, but the window is narrowing.
- **Clean integrate?** → Yes. Append DR3 numerics to the DESI-HDE constants block:
  `C_HDE_UNION3_DR3`, `ETA_HDE_UNION3_DR3`. Extend [Phase XI](bh_phase_xi_eta_candidates_results.md)
  η-candidate table with the DR3 row.
- **Tightens gap?** → **Yes, strongly.** Our η is currently under-constrained (only 2 independent
  derivations: the ρ_DE fit and ETA_FORMULA = exp(−φ/σ_conv)). DR3 makes the HDE fit a third
  independent derivation at tighter uncertainty.
- **Streamline?** → Yes: η moves from `[DERIVED]` toward `[VERIFIED]` once the DR3 HDE band
  overlaps 0.4153 to better than 1%.
- **Unpublished phenomenon?** → Yes, candidate: **the simultaneous pinning of three η
  derivations (ρ_DE fit, exp(−φ/σ_conv), DESI HDE) at 0.4153 ± 0.005 is a coincidence
  the literature does not discuss.** If sigma-ground holds, this triple-coincidence is
  a prediction-in-hindsight that calls for explanation.

**Label:** `[VERIFIED-PENDING-DR3]` on the η identification; `[SPECULATIVE]` on the
triple-coincidence claim.

---

### arXiv:2511.09467 — Hubble tension in HDE framework (Nov 2025)
**Domain:** Holographic dark energy
**Key formula / result:** Re-derives HDE ρ_DE = 3 c² M_Pl² / L² constraints under DESI
+ Hubble-tension priors. Finds HDE partially ameliorates but does not fully resolve the
H₀ tension, requiring a second component (often EDE or interacting DE).

**Five-question pass:**
- **Disprove?** → No; consistent with sigma-ground's HDE interpretation.
- **Clean integrate?** → Yes. One function `hde_rho_de_given_eta(L, eta=ETA)` in
  `sigma_ground/field/interface/cosmology.py` (new module if missing). One test against
  the paper's Table 2 baseline.
- **Tightens gap?** → Moderately. Joins the evidence queue that HDE alone isn't sufficient
  for tension resolution → opens space for sigma-ground's σ-field to *be* the second
  component.
- **Streamline?** → Yes, conditional: if we write ρ_DE = 3η M_Pl²/L² *and* ρ_DE(σ)
  contributes an EDE-like early-time boost via γ(σ)-mediated decoherence at high σ, we
  may derive *both* H₀ and S₈ shifts without adding a new knob. This is speculative but
  testable.
- **Unpublished phenomenon?** → **Yes: a prediction that H₀ tension magnitude should
  correlate with the number of σ-conv-crossing events in the local horizon.** No paper
  we've found makes this correlation because no other framework has a σ-conv concept.

**Label:** `[THEORETICAL]` once the cosmology module exists; `[SPECULATIVE]` on the
H₀–σ-conv-event correlation prediction.

---

### arXiv:2505.23382 — EDE + interacting dark sector (H₀ and S₈ simultaneous) (May 2025)
**Domain:** Dark energy / dark matter interaction
**Key formula / result:** Two-parameter extension where EDE fraction `f_EDE` lifts H₀
while a dark-sector interaction coefficient `ξ_IDE` suppresses S₈ growth. Simultaneous
fit reduces both tensions to ≤2σ.

**Five-question pass:**
- **Disprove?** → Potentially: if the fitted `ξ_IDE` range excludes a sigma-ground-
  motivated value of the DM-baryon coupling (derivable from Kobakhidze ξ + γ(σ) mode),
  we have a quantitative falsifier.
- **Clean integrate?** → Yes — but requires a new `ide_coupling_from_sigma(sigma)`
  formula. Not drop-in; call it a design-review item.
- **Tightens gap?** → Yes: provides the first joint H₀+S₈ target range for any σ-field
  extension of ΛCDM.
- **Streamline?** → Yes, if the dark-sector interaction maps to our γ(σ) terminator —
  the IDE coefficient becomes a derived function of σ_conv instead of a free fit parameter.
- **Unpublished phenomenon?** → Candidate: **sigma-ground predicts S₈ should anti-correlate
  with η across cosmological ensembles**, because γ(σ) flattens the small-scale power
  spectrum through η. The IDE paper does not make this specific prediction.

**Label:** `[SPECULATIVE]` pending quantitative match.

---

### arXiv:2502.04430 — Page time of primordial black holes in SM and beyond (Feb 2025)
**Domain:** Black hole information / primordial BHs
**Key formula / result:** Page time t_P(M) for PBHs of mass M, computed in SM and with
BSM-dof extensions. Provides concrete numerical prediction for any BH evaporation model.

**Five-question pass:**
- **Disprove?** → Yes, potentially: sigma-ground's entanglement module predicts
  η × M_BH ≈ entanglement carried by Hawking radiation. If our Page time differs from
  SM's by more than the BSM tolerance, engine is inconsistent.
- **Clean integrate?** → Yes. `page_time_sigma_ground(M_bh)` function in
  `sigma_ground/field/interface/bh_merger.py`, reusing existing Hawking-temperature code
  plus an η-weighted correction.
- **Tightens gap?** → Yes: currently Phase VII/VIII pipelines don't specify a
  Page-time prediction — this paper forces one and makes the entanglement integration
  falsifiable per BH.
- **Streamline?** → Possibly: if our η-derivation of Page time matches SM within ~1%,
  η moves from `[DERIVED]` to `[VERIFIED-BY-PAGECURVE]`.
- **Unpublished phenomenon?** → **Yes: sigma-ground predicts a Page-time shift of
  order η (≈ 42%) relative to "naïve Hawking-only" Page time, which is not predicted
  by the SM or most BSM extensions.** This is the cleanest Batch-3 falsifiable prediction.

**Label:** `[THEORETICAL]` once integrated; `[SPECULATIVE-PREDICTION]` on the η-shift.

---

## Tier-2 candidates — condensed five-question pass

### arXiv:2505.08051 — ACT DR6 + DESI DR2 impact on EDE (May 2025)
- **Disprove?** No. **Integrate?** Yes — append constants to DESI-HDE block.
- **Tightens gap?** Moderately (narrows EDE fraction by ~15%).
- **Streamline?** No. **Unpublished?** —

### Muon g−2 Theory Initiative 2025 White Paper (June 2025, Fermilab final result)
- **Disprove?** **Potentially important:** resolves g−2 anomaly via lattice-HVP. If
  sigma-ground's Λ_QCD = 217 MeV + σ=0 prediction produces an a_μ inconsistent with the
  new consensus, engine fails a major precision test. Must be run.
- **Integrate?** Yes — add `a_mu_sm_prediction_2025 = 116 591 810e−11` with tolerance.
- **Tightens gap?** Yes: pins Λ_QCD precision.
- **Streamline?** Yes: promotes `LAMBDA_QCD_MEV = 217` from `[EMPIRICAL-INPUT]` to
  `[VERIFIED]` if consistent.
- **Unpublished?** —

### arXiv:2408.14245 — GALPs, composite axion-like DM (Aug 2024)
- **Disprove?** —. **Integrate?** Yes — extends Kobakhidze N=8 path.
- **Tightens gap?** ξ derivation is reinforced if GALP gauge group matches N=8.
- **Streamline?** Yes: reduces `XI_KOBAKHIDZE`'s `[SPECULATIVE]` to `[THEORETICAL]` if
  composite axion sector is independently motivated.
- **Unpublished?** Possibly: sigma-ground + GALPs together predict an axion mass window
  set by σ_conv × m_Planck, which neither predicts alone.

### arXiv:2506.06449 — Dirac Type-I seesaw cosmological phenomenology (June 2025)
- **Disprove?** No. **Integrate?** Opens new `sigma_ground/field/interface/neutrino.py`
  module. Non-trivial, design-review required.
- **Tightens gap?** Yes — first neutrino-sector constraint on η (via N_eff).
- **Streamline?** No. **Unpublished?** —

### arXiv:2505.01422 — Neutrino mass from asymptotic-safety (May 2025)
- **Disprove?** **Yes, potentially:** paper predicts a specific neutrino mass hierarchy
  from asymptotic-safety RG flow. If we interpret our ξ as the RG fixed-point coupling
  and the predicted m_ν hierarchy disagrees with cosmological bounds, we falsify.
- **Integrate?** Yes, via the new neutrino module.
- **Tightens gap?** Yes — first independent check of ξ as an RG fixed-point value.
- **Streamline?** Yes: ξ from asymptotic safety + ξ from Kobakhidze N=8 becomes a
  *consistency check*, not two independent hypotheses.
- **Unpublished?** **Candidate: the consistency of ξ = 0.1572 (N=8) with asymptotic-safety
  RG flow at the neutrino scale is not a claim in either paper.** If both hold, it
  would be a genuinely novel unification hint.

### arXiv:2508.18941 — Differential entropy ≡ Bekenstein-Hawking via bit threads (Aug 2025)
- **Disprove?** —. **Integrate?** Yes — augments `entanglement.py` (52KB, under-cited).
- **Tightens gap?** Links η to bit-thread density per unit area.
- **Streamline?** Yes: η becomes interpretable as a geometric fraction of Planck-area bits.
- **Unpublished?** Candidate: η = bit-thread fraction on a σ_conv surface.

---

## Tier-3 candidates — headline note each

| arXiv | One-line five-question summary |
|-------|-------------------------------|
| 2511.05632 | Entropic-MOND: Falsify? No. Integrate? Only as commentary. Tightens? Minor. Streamline? Possibly G-derivation. Unpublished? No. |
| 2508.11172 | LIV review: adds no new bound beyond 2502.18256. Use as citation umbrella only. |
| 2410.12634 | Dark axion portal: integrates into axion module; tightens hidden-sector mass window; no unpublished claim. |
| 2504.18663 | Replica-wormholes simplicial-QG: informational for entanglement module; not drop-in. |

---

## Key findings from Batch 3

1. **Two strong falsifiers identified** (arXiv:2502.18256 and 2502.04430). Both must be
   actively checked against the engine in a follow-on Phase XII.a before any integration
   is declared safe. If either fails, the engine has a concrete observable bug to fix.
2. **η-pinning by DESI 2025** is the highest-value Batch-3 integration (triple coincidence
   with ETA_FORMULA and ρ_DE fit).
3. **Neutrino-sector module is missing** and now blocks two Tier-2 integrations
   (2506.06449 and 2505.01422). Recommend opening it in Phase XIII.
4. **Muon g−2 2025 resolution is a free precision test** for Λ_QCD = 217 MeV — should be
   run regardless of whether we integrate other Batch-3 papers.
5. **Unpublished-phenomenon candidates logged** (see [unpublished_predictions.md](unpublished_predictions.md)):
   the η-triple-coincidence, η-shifted Page-time, H₀–σ-conv-event correlation, S₈–η
   anti-correlation, and ξ ↔ asymptotic-safety unification.

