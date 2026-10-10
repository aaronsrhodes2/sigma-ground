# Dark Energy, Cosmic Expansion, the Hubble Tension and Cosmological-Parameter Constraints (window: 2026-06-10 to 2026-10-10)

Conventions used below:
- Tags: **[IN WINDOW]** means first released or published between 10 Jun and 10 Oct 2026. **[BACKGROUND]** means older and included for context, with its numbers.
- Every item gives the title, the collaboration or authors, the ID, the date, the numbers, the status and a **Hook**. A hook is a quantity a Python cosmology engine can reproduce or test against.
- The w0–wa (CPL) equation of state is w(a) = w0 + wa(1−a), so ρ_DE(a)/ρ_DE0 = a^(−3(1+w0+wa)) · exp(−3wa(1−a)).
- Units: H0 is in km s⁻¹ Mpc⁻¹, D_M is the transverse comoving distance, D_H = c/H(z), and r_d is the sound horizon at the drag epoch.
- Source access: arxiv.org and desi.lbl.gov could not be fetched directly because DNS failed in this environment. Their content was read through the Exa fetch/search mirrors of the same pages, including full PDF text for arXiv 2607.27410 and 2503.14738.

---

## Q1. DESI and evolving dark energy: DR2 follow-ups, full-shape/DR3 status, current w0/wa and significance per dataset combination

### Takeaway
The main DESI release in the window is the DR2 Lyman-α full-shape Alcock–Paczyński (AP) analysis (arXiv 2607.27410, 29–30 Jul 2026). It moved Ω_m up to 0.325 ± 0.018, which is 1.4σ above the DESI BAO value and toward Planck ΛCDM. As a result, the evolving-dark-energy preference softened slightly: 3.1σ to 2.7σ for DESI+CMB, and 3.2σ (DES-Dovekie), 3.0σ (Pantheon+) and 3.5σ (Union3.1) once supernovae are added. The best fit is still w0 ≈ −0.82, wa ≈ −0.65, a "phantom-crossing" pattern. The supernova side has converged downward as well: the DES-SN5YR → DES-Dovekie recalibration cut 4.2σ to 3.2σ. DR3 BAO results and the first full five-year (Y5) dark-energy results are expected in 2027, not in this window.

### Cited Findings

**Primary new DESI result [IN WINDOW]**
- **DESI DR2 Results IV: Alcock–Paczyński Measurements from the Lyman Alpha Forest and Cosmological Constraints.** DESI Collaboration (A. G. Adame et al.), arXiv:2607.27410, submitted 29 Jul 2026. The DESI press post is dated 30 Jul 2026. — [arXiv abs](https://arxiv.org/abs/2607.27410); [arXiv HTML](https://arxiv.org/html/2607.27410); [DESI news](https://www.desi.lbl.gov/2026/07/30/new-desi-dr2-lyman-alpha-results-shed-light-on-dark-energy/)
  - The AP effect is measured to 1% at z_eff = 2.33, which is twice as tight as the DR2 Lyα BAO-only constraint. Joint Lyα AP+BAO gives **D_H(2.33)/r_d = 8.600 ± 0.066** and **D_M(2.33)/r_d = 39.32 ± 0.33**. — [arXiv abs](https://arxiv.org/abs/2607.27410)
  - The Lyα full shape alone gives **Ω_m = 0.325 ± 0.018** and **h·r_d = 97.9 ± 2.1 Mpc**, with correlation ρ = −0.96. Lyα FS + BBN (Ω_b h² = 0.02218 ± 0.00055) gives **H0 = 66.5 ± 1.3**. — [arXiv HTML](https://arxiv.org/html/2607.27410)
  - The DESI–CMB discrepancy in ΛCDM drops from 2.4σ to 2.2σ. The tension with DES-Dovekie supernovae on Ω_m drops from 1.9σ to 1.7σ, and with Union from 1.4σ to 1.3σ. It was originally 2.9σ against DES-Y5. — [arXiv HTML](https://arxiv.org/html/2607.27410)
  - **w0waCDM, DESI (BAO+Lyα FS) + CMB + DES-Dovekie: w0 = −0.821 ± 0.054, wa = −0.65 ± 0.20; 3.2σ (Δχ²_MAP = −12.9).** With updated Pantheon+ the preference is 3.0σ (Δχ² = −11.5). With Union3.1 it is 3.5σ (Δχ² = −15.4). DESI+CMB alone gives 2.7σ. — [arXiv HTML](https://arxiv.org/html/2607.27410)
  - Table 3 rows from the same paper:
    - ΛCDM DESI+CMB: Ω_m = 0.3042 ± 0.0033, H0 = 68.11 ± 0.24.
    - ΛCDM DESI+BBN: Ω_m = 0.3020 ± 0.0075, H0 = 68.47 ± 0.57.
    - w0wa DESI only: Ω_m = 0.363 (+0.033/−0.017), w0 = −0.43 (+0.31/−0.14), wa < −1.68.
    - **w0wa DESI+CMB: Ω_m = 0.343 (+0.019/−0.022), H0 = 64.5 ± 1.9, w0 = −0.54 (+0.19/−0.21), wa = −1.39 (+0.61/−0.50).**
    - w0wa DESI+CMB+Dovekie: Ω_m = 0.3134 ± 0.0052, H0 = 67.37 ± 0.54.
    - ΛCDM+Ω_K, DESI+CMB: 10³Ω_K = 2.10 ± 1.1.
    - ΛCDM+Σm_ν, DESI+CMB: Σm_ν < 0.0592 eV (95%). Adding Dovekie gives < 0.0654 eV.
    - w0wa+Σm_ν, DESI+CMB: Σm_ν < 0.166 eV. Adding Dovekie gives < 0.128 eV, with w0 = −0.827 ± 0.056 and wa = −0.62 (+0.25/−0.20).
    - [arXiv HTML](https://arxiv.org/html/2607.27410)
  - Pivot points: **w_p = −1.030 ± 0.040 at z_p = 0.55** (DESI+CMB) and **w_p = −0.981 ± 0.022 at z_p = 0.32** (DESI+CMB+Dovekie). The w0–wa figure of merit is 227.4. The paper says w(z) "still favors a phantom-crossing-like behaviour", with w < −1 in matter domination and w > −1 at low z. — [arXiv HTML](https://arxiv.org/html/2607.27410)
  - High-τ test: dropping low-ℓ EE gives Ω_m = 0.2998 ± 0.0035 and τ = 0.0905 ± 0.0096. That makes DESI and the CMB consistent but conflicts with SNe and with Planck low-ℓ EE. — [arXiv HTML](https://arxiv.org/html/2607.27410)
  - DESI's own framing: the new Lyα result "agree[s] with the standard model, which could indicate that the current hints of evolving dark energy may fade away, or alternatively, that a more complex model may be needed." — [DESI news](https://www.desi.lbl.gov/2026/07/30/new-desi-dr2-lyman-alpha-results-shed-light-on-dark-energy/)
  - **Hook:** reproduce D_H/r_d = 8.600 and D_M/r_d = 39.32 at z = 2.33. Use r_d ≈ 147.07 Mpc (the CMB-SPA value below) or r_d(h) from h·r_d = 97.9 Mpc. Check the pivot w(z_p = 0.32) = −0.981 ± 0.022 in the CPL model.
- **Supporting papers [IN WINDOW]**, all 29 Jul 2026:
  - Herbold et al., validation, arXiv:2607.27411. The growth-rate parameter fσ8 was removed because mocks showed a 10% bias at ≫5σ. — [arXiv HTML](https://arxiv.org/html/2607.27411)
  - Ruiz-Herrera et al., CoLoRe-2LPT mocks, arXiv:2607.27412. — [DESI news](https://www.desi.lbl.gov/2026/07/30/new-desi-dr2-lyman-alpha-results-shed-light-on-dark-energy/)
  - Chaves-Montero et al., P1D→3D "holography" priors, arXiv:2607.27413. — [DESI news](https://www.desi.lbl.gov/2026/07/30/new-desi-dr2-lyman-alpha-results-shed-light-on-dark-energy/)

**DESI survey status**
- DESI finished its planned five-year survey in April 2026 with >47 million galaxies and quasars. The first dark-energy results from the full five-year dataset are expected in 2027. — [Fermilab news, Apr 2026](https://news.fnal.gov/2026/04/desi-completes-planned-3d-map-of-the-universe-and-continues-exploring/); [DESI, 15 Apr 2026](https://www.desi.lbl.gov/2026/04/15/desi-reaches-mapping-milestone-surpassing-expectations/). This was relayed through a search summary; the pages were not fully fetched.
- DR3 BAO results are expected in 2027. — [A&A DR2 Lyα paper context (search summary)](https://www.aanda.org/articles/aa/full_html/2026/05/aa57820-25/aa57820-25.html)

**DESI DR2 BAO baseline [BACKGROUND, Mar 2025; PRD Oct 2025, DOI 10.1103/tr6y-kpc6]**
- **DESI DR2 Results II**, arXiv:2503.14738. — [arXiv HTML v3](https://arxiv.org/html/2503.14738v3)
  - BAO-only ΛCDM: **Ω_m = 0.2975 ± 0.0086, h·r_d = 101.54 ± 0.73 Mpc** (r = −0.92). This is a 2.3σ tension with the CMB.
  - DESI+BBN: **H0 = 68.51 ± 0.58**. DESI+CMB: **H0 = 68.17 ± 0.28, Ω_m = 0.3027 ± 0.0036**. CMB alone: H0 = 67.14 ± 0.47, Ω_m = 0.3169 ± 0.0065.
- **DR2 w0wa table (posterior means) [BACKGROUND]:** — [arXiv HTML v3](https://arxiv.org/html/2503.14738v3)
  - DESI+CMB: w0 = −0.42 ± 0.21, wa = −1.75 ± 0.58, Ω_m = 0.353 ± 0.021, H0 = 63.6 (+1.6/−2.1); **3.1σ**.
  - +Pantheon+: **w0 = −0.838 ± 0.055, wa = −0.62 (+0.22/−0.19)**, Ω_m = 0.3114 ± 0.0057, H0 = 67.51 ± 0.59; **2.8σ** (Δχ² = −10.7).
  - +Union3: **w0 = −0.667 ± 0.088, wa = −1.09 (+0.31/−0.27)**, Ω_m = 0.3275 ± 0.0086, H0 = 65.91 ± 0.84; **3.8σ** (Δχ² = −17.4).
  - +DESY5: **w0 = −0.752 ± 0.057, wa = −0.86 (+0.23/−0.20)**, Ω_m = 0.3191 ± 0.0056, H0 = 66.74 ± 0.56; **4.2σ** (Δχ² = −21.0).
  - Neutrino mass: Σm_ν < 0.064 eV (ΛCDM) and < 0.16 eV (w0wa).
- **DR2 BAO distances per tracer (z_eff; D_M/r_d; D_H/r_d) [BACKGROUND]:** — [arXiv HTML v3](https://arxiv.org/html/2503.14738v3)
  - BGS: z = 0.295, D_V/r_d = 7.942 ± 0.075.
  - LRG1: z = 0.510; 13.588 ± 0.167; 21.863 ± 0.425.
  - LRG2: z = 0.706; 17.351 ± 0.177; 19.455 ± 0.330.
  - LRG3+ELG1: z = 0.934; 21.576 ± 0.152; 17.641 ± 0.193.
  - ELG2: z = 1.321; 27.601 ± 0.318; 14.176 ± 0.221.
  - QSO: z = 1.484; 30.512 ± 0.760; 12.817 ± 0.516.
  - Lyα (BAO only): z = 2.330; 38.988 ± 0.531; 8.632 ± 0.101.
  - **Hook:** this is a direct regression test for any H(z) and D_M(z) implementation. Cross-correlations r_M,H are about −0.40 to −0.50.
- Cortês & Liddle argue that the three supernova significances can be neither chosen among nor averaged. A principled combination gives **3.1σ**, and the most secure figure is DESI+CMB alone at 3.1σ. — [ULisboa record (Nov 2025)](https://researchportal.ulisboa.pt/en/publications/on-desis-dr2-exclusion-of-%CE%BBcdm/)
- DESI DR1 full shape + DR2 BAO (ShapeFit; Forero-Sánchez, Gil-Marín, Verde et al.; arXiv:2602.18761; Feb 2026). Results: ΛCDM Ω_m = 0.3035 ± 0.0085, h = 0.6876 ± 0.0059, σ8 = 0.822 ± 0.034. **w0wa (DESI only + BBN): w0 = 0.49 ± 0.25 [sic, positive], wa = −1.52 ± 0.77**, a 1.4σ discrepancy with ΛCDM. — [arXiv Science](https://arxiv.science/abs/2602.18761). The positive w0 is quoted as printed in the abstract.

**Supernova recalibrations that move the significance**
- **DES-Dovekie** (Popovic, Shah, Kenworthy, Kessler, Davis et al.; MNRAS 548, stag632; published 14 Apr 2026) [BACKGROUND, near window]:
  - About 1820 SNe (1623 DES and 197 low-z). ΛCDM Ω_m = 0.330 ± 0.015, a shift of −0.022 from DES-SN5YR.
  - With CMB (Planck+ACT+SPT) + DESI DR2: **w0 = −0.803 ± 0.054, wa = −0.72 ± 0.21; 3.2σ** (down from 4.2σ). The Bayesian odds are about 5:1, described as "only a weak preference".
  - Original DES-SN5YR in the same table: w0 = −0.770 ± 0.060, wa = −0.740 ± 0.220, Ω_m = 0.318 ± 0.006.
  - [MNRAS](https://academic.oup.com/mnras/article/548/4/stag632/8653925)
- Union3 has been updated to "Union3.1". Its latest ΛCDM Ω_m = 0.334 (+0.025/−0.024) as quoted in DESI DR2-IV. — [arXiv HTML 2607.27410](https://arxiv.org/html/2607.27410)

**Independent re-analyses [IN WINDOW]**
- **Lozano Torres, "Dynamical dark energy from DESI DR2 and DES Year 6 BAO, supernovae, and cosmic chronometers: a 3σ preference without decisive Bayesian evidence"**, MNRAS advance article, 26 Aug 2026. — [MNRAS](https://academic.oup.com/mnras/advance-article/doi/10.1093/mnras/stag1607/8771483)
  - Full data: ΛCDM H0 = 68.41 ± 0.30, Ω_m = 0.301 ± 0.004, **r_drag = 147.55 ± 0.20 Mpc**. wCDM w = −0.992 ± 0.028.
  - **w0 = −0.740 ± 0.082, wa = −0.893 ± 0.285 (3σ)** with Δχ² ≈ −12, but ln B ≈ −0.5 (inconclusive).
- **Kandel et al., "Revisiting the evidence for dynamical dark energy: from DES-SN5Y to DES-Dovekie"**, EPJC, 3 Jul 2026. With a compressed CMB likelihood + DESI DR2 + Dovekie, the preference falls below 1σ for most parametrizations (CPL, BA, JBP, Efstathiou: ~2.3σ → ~0.95σ; GEDE: 2.15σ → 0.28σ). — [EPJC](https://link.springer.com/article/10.1140/epjc/s10052-026-16010-6)
- Chaudhary, Capozziello et al., EPJC (22 May 2026; arXiv 2508.10514) [BACKGROUND]:
  - CMB(compressed)+DESI DR2 ΛCDM: H0 = 68.19 ± 0.46, Ω_m = 0.3035 ± 0.0056.
  - wCDM: w = −1.037 ± 0.041 (+DESI), −0.986 ± 0.026 (+Pantheon+), −0.983 ± 0.024 (+Dovekie), −0.979 ± 0.031 (+Union3).
  - Phantom crossing appears near z ≈ 0.5, with Nσ ≈ 1.1–2.3. The evidence is driven by the LRG1–2 tracers.
  - [arXiv HTML](https://arxiv.org/html/2508.10514); [EPJC](https://link.springer.com/article/10.1140/epjc/s10052-026-15806-w)
- Which CMB is used matters [BACKGROUND]:
  - ACT-DR6 + DESI-DR2 + DES-all: w0 = −0.769 ± 0.061, wa = −0.75 (+0.28/−0.25), ~3.9σ.
  - With DES-SN (z>0.1) only: ~1.0σ.
  - Planck+lensing + DESI DR2 + DES-all reaches ~5σ.
  - [arXiv 2507.09981](https://arxiv.org/pdf/2507.09981)
- **DES Y6 dark-energy combination** (DES Collaboration, 2026) [BACKGROUND]:
  - 3×2pt + SN + BAO: w0 = −0.84 ± 0.10, wa = −0.44 (+0.60/−0.55), 2.2σ.
  - +DESI DR2: w0 = −0.84 (+0.06/−0.07), wa = −0.53 (+0.33/−0.28), 2.3σ.
  - **+CMB: w0 = −0.82 ± 0.05, wa = −0.63 (+0.21/−0.18), 3.0σ.**
  - Without SN it is 2.6σ.
  - [DES Y6 papers page](https://www.darkenergysurvey.org/des-y6-cosmology-results-papers/)

### Inferences
- In the window, the DESI-centred evidence for evolving dark energy did not grow. It held at roughly 3σ: 2.7σ for DESI+CMB and 3.0–3.5σ with supernovae. Every recalibration (DES-Dovekie, Union3.1, Lyα full shape) moved toward ΛCDM. The best-fit direction stays fixed in the w0 > −1, wa < 0 quadrant.
- The most robust testable hook is the pair (w0, wa) = (−0.821, −0.65) with Ω_m = 0.3134 and H0 = 67.37. In CPL the phantom divide is crossed at a_× = 1 + (1+w0)/wa, which gives z_× ≈ 0.38 for this pair. This is my derivation. A code that hard-wires w = −1, or that refuses w < −1, cannot reproduce this best fit.
- For a scalar-field (σ-field) dark-energy model, the DESI best fit needs w < −1 at z ≳ 0.4. A single minimally coupled canonical scalar cannot do this (see Q5). A σ-field model should therefore either (a) match the pivot value w(z ≈ 0.3) ≈ −0.98 ± 0.02 with w ≥ −1 throughout, or (b) state explicitly how it crosses the phantom divide.

### Gaps
- No DESI DR2 galaxy full-shape key paper (as opposed to the DR1 FS + DR2 BAO combination) and no DR3 BAO result was found in the window. DR3 BAO and Y5 dark-energy results are stated for 2027.
- The DESI DR2-IV Table 3 values with Pantheon+ and Union3.1 (as opposed to Dovekie) were not extracted; only their significances (3.0σ and 3.5σ) were.
- No standalone "Union3.1" paper with w0/wa numbers was retrieved.

---

## Q2. Euclid, Rubin/LSST, DES final, KiDS, HSC and eROSITA in this window; S8 tension status

### Takeaway
Neither Euclid nor Rubin published cosmological-parameter results in the window. The Euclid DR1 public release is scheduled for October 2026 or later, and ESA pages give conflicting dates. Rubin is in early data previews. The S8 picture is now mostly "resolved toward the CMB" for cosmic shear:
- KiDS-Legacy 2+3-point gives S8 = 0.806 (+0.025/−0.023) (June 2026).
- HSC Y3 recalibrated with DESI gives 0.805 ± 0.018.
- DES Y6 3×2pt is the main holdout at 0.789 ± 0.012 (2.6σ in S8 vs combined CMB).
- eROSITA clusters sit high at 0.86 ± 0.01.
- SPT-3G D1 CMB lensing (31 Aug 2026) gives S8 = 0.811 ± 0.011 with DES Y3.

### Cited Findings

**Euclid**
- **Euclid DR1 timing conflict:**
  - The Euclid Q1 overview (A&A, published 30 Jun 2026) says "The first complete data release (e.g. DR1) will take place in October 2026" covering ~1900 deg². — [A&A Q1-I](https://www.aanda.org/articles/aa/full_html/2026/07/aa54610-25/aa54610-25.html); [ESA DR1 memo](https://www.cosmos.esa.int/documents/10647/12245842/DR1-memo_250624.pdf/)
  - The ESA Cosmos DR1 page lists **"DR1-Foundation: November 2026; DR1: mid-2027"**. — [ESA Cosmos DR1](https://www.cosmos.esa.int/web/euclid/data-release-dr1)
  - The two sources conflict, and no Euclid S8 or w0wa result was found.
- **[IN WINDOW] Euclid Q1 LensMC shear catalogue** (arXiv:2606.20829, June 2026). It covers 63 deg² at 26 arcmin⁻² (I_E < 24.5). The DR1 shear-bias requirements are σ_m = 5.7×10⁻³ and σ_c = 8.6×10⁻⁴ for 1700 deg². Cluster lensing at 10¹⁴ M_⊙ is measured out to z ≈ 2. This is not a cosmology result. — [arXiv HTML](https://arxiv.org/html/2606.20829)

**Rubin/LSST**
- No cosmology results. An August 2026 release marked Rubin's "Early Data Preview 2 (EDP2)". The search summary also states that LSST formally began on the night of 29 Jun 2026; that date was not verified from a fetched primary page. — [phys.org, Aug 2026](https://phys.org/news/2026-08-rubin-observatory-deep-window-famous.html); [Rubin commissioning, arXiv 2606.09938](https://arxiv.org/pdf/2606.09938)

**DES (final)** [BACKGROUND, Jan–Feb 2026]
- **DES Y6 3×2pt** (arXiv:2601.14559, 29 Jan 2026):
  - **S8 = 0.789 ± 0.012, Ω_m = 0.333 (+0.023/−0.028)** in ΛCDM. That is 1.8σ from CMB-SPA in the full parameter space and **2.6σ projected onto S8**.
  - wCDM: S8 = 0.782 (+0.021/−0.020), w = −1.12 (+0.26/−0.20).
  - Joint Y6 + CMB + low-z: **S8 = 0.806 (+0.006/−0.007), Ω_m = 0.302 ± 0.003, h = 0.683 (+0.003/−0.002), Σm_ν < 0.14 eV**, w = −0.981 (+0.021/−0.022).
  - [arXiv](https://arxiv.org/abs/2601.14559)
- DES Y6 cosmic shear (arXiv:2602.10065): S8 = 0.798 (+0.014/−0.015) with NLA intrinsic alignment, or 0.783 (+0.019/−0.015) with TATT. That is 2.0σ (2.3σ) from the CMB in S8. — [Arizona Cosmology Lab](https://cosmolab.arizona.edu/news/dark-energy-survey-year-6-results)

**KiDS**
- **[IN WINDOW] KiDS-Legacy joint second- and third-order cosmic shear** (arXiv:2606.12389, June 2026). Results: **Ω_m = 0.297 (+0.056/−0.040), S8 = 0.806 (+0.025/−0.023)**, "no evidence for an S8 tension", consistent with Planck within 1σ. — [arXiv HTML](https://arxiv.org/html/2606.12389)
- [BACKGROUND] KiDS-Legacy two-point (Wright et al., A&A Nov 2025): **S8 = 0.815 (+0.016/−0.021)**, 0.73σ from Planck. Σ8 = σ8(Ω_m/0.3)^0.58 = 0.821 (+0.014/−0.016). — [A&A](https://www.aanda.org/articles/aa/full_html/2025/11/aa54908-25/aa54908-25.html)

**HSC** [BACKGROUND]
- HSC Y3 shear recalibrated with DESI clustering redshifts (arXiv:2511.18134): **S8 = 0.805 ± 0.018**, up from 0.769 (+0.031/−0.034).
- HSC+KiDS+DES combined: S8 = 0.813 (+0.009/−0.010). Adding CMB lensing gives S8 = 0.818 ± 0.007, against Planck×ACT×DESI DR2 at 0.812 ± 0.007.
- The authors conclude the S8 tension was "primarily systematics driven".
- [ar5iv](https://ar5iv.labs.arxiv.org/html/2511.18134)

**eROSITA** [BACKGROUND]
- eRASS1 clusters: **S8 = 0.86 ± 0.01**, above the CMB at −1.5σ. SPT clusters give 0.795 ± 0.029. — [S8 review, arXiv 2602.12238](https://arxiv.org/html/2602.12238v2)
- Artis et al. eRASS1 growth index: **γ = 1.19 ± 0.21**, against the GR value γ = 0.55. — [Artis et al. 2024](https://exa.ai/library/publication/jsnpx3zd04z)

**S8 review baseline** [BACKGROUND, Feb 2026]
- The "Status of the S8 Tension: A 2026 Review" (arXiv:2602.12238) uses a Combined CMB (Planck+ACT DR6+SPT-3G) value **S8 = 0.836 (+0.012/−0.013)**. It finds DES Y6 2.4–2.7σ, KiDS-Legacy 0.9σ and HSC Y3 (DESI n(z)) 1.4σ. — [arXiv HTML](https://arxiv.org/html/2602.12238v2)

**SPT-3G CMB lensing [IN WINDOW]**
- **SPT-3G D1: Quadratic-Estimator CMB Lensing Reconstruction and Cosmology**, arXiv:2608.31136, 31 Aug 2026.
  - **σ8 Ω_m^0.25 = 0.6046 ± 0.0096** (SPT-3G lensing alone). With ACT DR6 + Planck PR4 lensing: 0.6020 ± 0.0084.
  - With DES Y3 3×2pt: **S8 = 0.811 ± 0.011**.
  - With CMB-SPA + DESI DR2: **Σm_ν < 0.072 eV (95%)**. Preferences for curvature and for (w0, wa) ≠ (−1, 0) are reduced to about 2σ.
  - [arXiv abs](https://arxiv.org/abs/2608.31136)

### Inferences
- The S8 tension is shrinking. Weak-lensing surveys whose redshift calibration improved (KiDS-Legacy, HSC+DESI) moved up to about 0.81. DES Y6 is the only stage-III result that still sits about 2.5σ low in S8, and its result depends on the intrinsic-alignment model.
- **Hook:** S8 = σ8 (Ω_m/0.3)^0.5. A reasonable "consensus" check band is S8 ∈ [0.80, 0.84], with CMB-SPA σ8 = 0.8137 and Ω_m ≈ 0.31. A model that suppresses growth by >5% at z < 1 relative to ΛCDM is now disfavoured by the lensing+CMB-lensing combination (0.818 ± 0.007).
- No stage-IV (Euclid or Rubin) cosmology number exists yet. Any σ-field claim about S8 can only be tested against stage-III and CMB-lensing values for now.

### Gaps
- No Euclid DR1 cosmology papers were found in the window. Whether DR1 actually went public in early October 2026 could not be confirmed, and the ESA pages conflict.
- No 2026 HSC final (Y5) cosmology result was found.
- No new eROSITA cosmology result (e.g. eRASS:5) was found in the window.

---

## Q3. Current H0 values (SH0ES, CCHP TRGB/JAGB, H0DN distance network, TDCOSMO, standard sirens, masers): is the tension growing or shrinking?

### Takeaway
The local distance-ladder consensus is now **H0DN: 73.50 ± 0.81**, published in A&A in April 2026. It sits 7.1σ from CMB ΛCDM (67.24 ± 0.35) and 5.0σ from DESI+BBN (68.51 ± 0.58). CCHP/Freedman's JWST TRGB/JAGB values (about 68–70) remain the main dissent. Independent late-time probes are too imprecise to arbitrate:
- TDCOSMO 2025: 71.6 (+3.9/−3.3).
- Sirens: about 70–77 with ±4 to ±13.

In the window, the tension did not shrink. The review literature (Cai & Wang, June 2026) calls it "a real crisis". One meta-analysis reframes it as "distance ladder vs everything else" at 6.5σ.

### Cited Findings

**Distance-ladder consensus**
- **H0DN Collaboration (Casertano et al.), "The Local Distance Network: A community consensus report on the measurement of the Hubble constant at ∼1% precision"** — arXiv:2510.23823 (27 Oct 2025); A&A, April 2026, DOI 10.1051/0004-6361/202557993 [BACKGROUND but the definitive current value].
  - **H0 = 73.50 ± 0.81** (baseline, 1.1%; 0.9% with all estimators). χ²/dof = 0.988.
  - It is **7.1σ** from Planck+SPT+ACT ΛCDM (67.24 ± 0.35) and **5.0σ** from DESI DR2 BBN+BAO (68.51 ± 0.58).
  - Removing either Cepheids or TRGB has minimal effect. Replacing SNe Ia with galaxy-based indicators changes H0 by less than 0.1.
  - [A&A abstract](https://www.aanda.org/articles/aa/abs/2026/04/aa57993-25/aa57993-25.html); [arXiv Science](https://arxiv.science/abs/2510.23823); [ISSI press release](https://workshops.issibern.ch/hubble-constant/press-release/)
- SH0ES baseline: **73.04 ± 1.04** (Riess et al. 2022), as cited throughout. — [CCHP 2026 ApJ intro](https://iopscience.iop.org/article/10.3847/1538-4357/ae29eb)

**CCHP / Freedman**
- **Freedman et al., "Status Report on the CCHP…"**, ApJ 985, 203 (27 May 2025), arXiv:2408.06153 [BACKGROUND]. — [ApJ](https://iopscience.iop.org/article/10.3847/1538-4357/adce78)
  - **H0 = 70.39 ± 1.22 (stat) ± 1.33 (sys) ± 0.70 (σ_SN)** from TRGB with 24 SN calibrators (HST+JWST).
  - JWST only: **TRGB 68.81 ± 1.79 ± 1.32; JAGB 67.80 ± 2.17 ± 1.64**.
- **Hoyt et al., "The CCHP: Improving the Calibration of Type Ia Supernovae with JWST Measurements of the TRGB"**, ApJ 1002, 11 (21 Apr 2026), arXiv:2503.11769 [BACKGROUND]. — [ApJ](https://iopscience.iop.org/article/10.3847/1538-4357/ae29eb)
  - JWST 11 SNe: **H0 ≈ 69** with 0.6 dispersion across four SN calibrations (68.4–69.6).
  - Expanding to 24 SNe shifts the Pantheon+ calibration up by +2.0 (3.1σ) and CSP-II up by +0.8 (1.4σ).

**Strong-lensing time delays**
- **TDCOSMO 2025: Cosmological constraints from strong lensing time delays** (TDCOSMO Collaboration), arXiv:2506.03023; A&A 704, A63 (Dec 2025) [BACKGROUND]. — [arXiv HTML v4](https://arxiv.org/html/2506.03023v4); [A&A](https://www.aanda.org/articles/aa/full%5Fhtml/2025/12/aa55801-25/aa55801-25.html)
  - 8 lensed quasars + Pantheon+ Ω_m, flat ΛCDM: **H0 = 71.6 (+3.9/−3.3)**.
  - Adding SLACS+SL2S tightens the precision to 4.6%.
  - Hook: H0 ∝ D_Δt⁻¹, and δH0 = 2(δσ/σ)·H0.
- **[IN WINDOW] GWTC-4 sirens + TDCOSMO2025, model-independent via distance sum rule** (arXiv:2606.03634, June 2026; exact day not confirmed). Results: **H0 = 83.78 (+12.53/−10.23)** with TDCOSMO-only lensing, and **75.42 (+3.74/−4.66)** with the H0LiCOW MST treatment. Both are "consistent with both Planck and SH0ES". — [arXiv HTML](https://arxiv.org/html/2606.03634)

**Standard sirens**
- LVK GWTC-4.0 cosmology (O4a, 142 CBCs), dark + bright (GW170817) [BACKGROUND, Sep 2025]: **H0 = 76.6 (+13.0/−9.5)**. — [LIGO science summary](https://www.ligo.org/wp-content/uploads/2025/09/O4a_Cosmology.pdf)
- Spectral sirens, GWTC-4.0 [BACKGROUND]:
  - Gaussian Process + GW170817: H0 = 69 (+7/−6). — [arXiv 2509.03607](https://arxiv.org/html/2509.03607)
  - Flexible secondary-mass model: 71.4 (+13.8/−13.4) alone; **73.5 (+9.2/−7.2)** with GW170817. — [arXiv 2605.11474](https://arxiv.org/html/2605.11474)
- DES Y6 Gold dark sirens (McMahon et al., arXiv:2602.04766) [BACKGROUND]: 70.9 (+22.3/−18.6) for dark sirens alone, **73.1 (+11.7/−8.6)** with GW170817. — [arXiv PDF](https://arxiv.org/pdf/2602.04766)
- **[IN WINDOW] Alfradique et al., "Improved constraint on the Hubble constant from dark sirens with LVK O4a"**, JCAP08(2026)068, 28 Aug 2026.
  - 17 well-localized dark sirens: 78.2 (+11.8/−10.8) with r-band weighting.
  - Combined with GW170817: **H0 = 69.9 (+4.2/−4.0)**, about 11% better than GW170817 alone.
  - [JCAP](https://iopscience.iop.org/article/10.1088/1475-7516/2026/08/068)
- Masers: the NGC 4258 megamaser geometric distance (1.5%) remains the zero-point anchor for both CCHP and SH0ES. Megamasers are included as "Tracers" in the H0DN baseline. No new standalone maser-H0 (MCP) value was found in the window. — [Hoyt et al. 2026](https://iopscience.iop.org/article/10.3847/1538-4357/ae29eb); [H0DN](https://www.aanda.org/articles/aa/pdf/2026/04/aa57993-25.pdf)

**Reviews and meta-analyses**
- **[IN WINDOW] Cai & Wang, "The Hubble tension: A decade review"**, arXiv:2606.20434 (June 2026). It calls the tension "likely… a real crisis". It notes that local-void/Hubble-bubble solutions have "long been ruled out". It reviews interacting dark-energy resolutions. — [arXiv abs](https://arxiv.org/abs/2606.20434)
- Pantos & Perivolaropoulos, "Dissecting the Hubble tension" (arXiv:2601.00650, Jan 2026) [BACKGROUND]. — [arXiv abs](https://arxiv.org/abs/2601.00650)
  - 30 distance-ladder measurements: **72.73 ± 0.39**.
  - 58 sound-horizon-free / ladder-independent measurements: **69.37 ± 0.34**. That is 6.5σ, with a minimum of 3.9σ after correlations.
  - Sub-classes: local-ΛCDM 67.61 ± 0.96; pure-local 71.03 ± 0.69; CMB sound-horizon-free 69.07 ± 0.44.

**CMB / BAO side, for the comparison numbers**
- SPT-3G D1 alone: **66.66 ± 0.60** (6.2σ from SH0ES). SPT+ACT: 66.59 ± 0.46. **CMB-SPA: 67.24 ± 0.35** (6.4σ from SH0ES). Versions differ: the current arXiv abstract and SPT PDF give 67.24 ± 0.35 with n_s = 0.9684 ± 0.0030, while the arXiv HTML table and INSPIRE give 67.19 ± 0.38 with n_s = 0.9679 ± 0.0033. — [arXiv 2506.20707](https://arxiv.org/abs/2506.20707); [SPT PDF](https://pole.uchicago.edu/public/data/camphuis25/C25.pdf)
  - A 1 Apr 2026 LAMBDA note reports a bug fix in the CMB-SPA chains: "0.2–0.3 sigma change in the numerical values". — [LAMBDA](https://lambda.gsfc.nasa.gov/product/spt/spt3g_d1_bandp_likelihood_info.html)
- ACT DR6 P-ACT-LB ΛCDM: **H0 = 68.22 ± 0.36**. — [PoS ACT DR6 proceedings](https://pos.sissa.it/485/094/pdf)
- DESI DR2 Lyα FS + BBN (z ≈ 2.3 alone): **66.5 ± 1.3** [IN WINDOW]. — [arXiv 2607.27410](https://arxiv.org/abs/2607.27410)

### Inferences
- The tension is not shrinking. The local side has consolidated (H0DN at ±0.81), and the early side has tightened (CMB-SPA ±0.35; DESI+CMB ±0.24). The CCHP dissent (about 68.8–70.4) is now the main source of doubt.
- The H0DN/CCHP difference is mostly the SN Ia calibration branch. Pantheon+ moves CCHP's 24-SN H0 up by +2.0 at 3.1σ.
- **Hooks:**
  - c/H0 = 4458.5 Mpc (14.54 Gly) for H0 = 67.24.
  - c/H0 = 4397.7 Mpc (14.34 Gly) for H0 = 68.17.
  - c/H0 = 4078.8 Mpc (13.30 Gly) for H0 = 73.50.
  - 1/H0 in Gyr equals the same numbers.
  - These are my computations with c = 299792.458 km/s and 1 Mpc = 3.0857×10²² m.
- A codebase should carry at least two H0 "truths": early (67.2–68.5) and late (73.0–73.5). It should report which one it uses. Hard-coding one value silently picks a side of a >5σ dispute.

### Gaps
- No new SH0ES (Riess et al.) H0 paper dated June–October 2026 was found.
- No new CCHP (Freedman) H0 value was published in the window. The Owens et al. JWST Cepheid paper was "in preparation".
- No new megamaser (MCP) H0 value was found.
- No new TDCOSMO H0 (beyond TDCOSMO 2025) was found in the window.

---

## Q4. CMB experiments (ACT DR6, SPT-3G, Simons Observatory, Planck reanalyses): N_eff, Σm_ν (including the "negative neutrino mass" preference) and early dark energy

### Takeaway
In the window the CMB news is incremental:
- SPT-3G D1 quadratic-estimator lensing (31 Aug 2026) gives Σm_ν < 0.072 eV with CMB-SPA + DESI DR2, and reduced (~2σ) preferences for w0wa and curvature.
- An ACT DR6 + DESI DR2 isocurvature paper (June 2026) gives Σm_ν < 0.052–0.057 eV.
- Simons Observatory has only commissioning and "Initial Science Observations" (no parameters yet).

Background constraints remain the reference:
- N_eff = 2.86 ± 0.13 (ACT DR6, P-ACT-LB), ΔN_eff < 0.17.
- No early-dark-energy detection (f_EDE < 0.09).
- Σm_ν,eff peaks negative in ΛCDM (≈ −0.075 to −0.16 eV), at roughly 3σ against the oscillation minimum of 0.058 eV.

### Cited Findings

**SPT-3G**
- **[IN WINDOW] SPT-3G D1 QE lensing** (arXiv:2608.31136, 31 Aug 2026): Σm_ν < 0.072 eV (95%) with CMB-SPA + DESI DR2 in ΛCDM. σ8 Ω_m^0.25 = 0.6046 ± 0.0096. Preferences for (w0, wa) and Ω_K are reduced to about 2σ. — [arXiv abs](https://arxiv.org/abs/2608.31136)
- [BACKGROUND] SPT-3G D1 T&E (Camphuis et al., arXiv:2506.20707, 25 Jun 2025): the CMB-SPA ΛCDM table. — [SPT PDF](https://pole.uchicago.edu/public/data/camphuis25/C25.pdf)
  - **100θ* = 1.04162 ± 0.00023; Ω_b h² = 0.022381 ± 0.000093; Ω_c h² = 0.12009 ± 0.00086; n_s = 0.9684 ± 0.0030** (10.5σ from 1).
  - **ln(10¹⁰A_s) = 3.0479 ± 0.0099; τ = 0.0559 ± 0.0055; H0 = 67.24 ± 0.35; r_d = 147.07 ± 0.22 Mpc; σ8 = 0.8137 ± 0.0038**.
  - CMB vs DESI DR2 ΛCDM shows a 2.8σ difference.
  - SPT-3G 2019–2020 polarization (MUSE) lensing + BAO gives Σm_ν < 0.075 eV. — [PRD 111, 083534](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.111.083534)

**ACT DR6** [BACKGROUND; JCAP 2025]
- Extended models (Calabrese et al., arXiv:2503.14454): — [Cambridge repository](https://www.repository.cam.ac.uk/items/9a1242d7-0031-4767-8380-a7dc8c9d5a7e)
  - **N_eff = 2.86 ± 0.13** (2.89 ± 0.11 with He+D); **Σm_ν < 0.089 eV**; N_idr < 0.134.
  - **α_EM/α_EM,0 = 1.0043 ± 0.0017; m_e/m_e,0 = 1.0063 ± 0.0056**; Y_p = 0.2312 ± 0.0092.
  - **T_CMB = 2.698 ± 0.016 K**; w = −0.986 ± 0.025; growth index **γ = 0.663 ± 0.052**; running dn_s/dlnk = 0.0062 ± 0.0052.
  - H0 in EDE / PMF / modified recombination: **69.9 (+0.8/−1.5)**, 69.1 ± 0.5 and 69.6 ± 1.0. None is preferred over ΛCDM.
- P-ACT-LB: **H0 = 68.22 ± 0.36; ΔN_eff < 0.17 (95%)**, against Planck-LB < 0.36. — [PoS](https://pos.sissa.it/485/094/pdf)
- Dark-radiation follow-up (arXiv:2512.19633): P-ACT gives ΔN_eff < 0.14. WZDR N_IR < 0.246; nuADaM N_idr < 0.237. The mean H0 is about 68.6–68.75 without a SH0ES prior. — [arXiv PDF](https://arxiv.org/pdf/2512.19633)
- **[IN WINDOW] "Constraints on the Sum of Neutrino Masses from ACT DR6 and DESI DR2 Considering Isocurvature Initial Conditions"** (arXiv:2606.17994, June 2026). It combines Planck+ACT DR6+SPT-3G with DESI DR2 and DES-Y5 SNe:
  - **Σm_ν < 0.052 eV (95%, ΛCDM, adiabatic)**; < 0.057 eV with neutrino isocurvature (no detection); about 0.092 eV with a hierarchy-informed prior.
  - [arXiv abs](https://arxiv.org/abs/2606.17994). Numbers are from the abstract as summarized by search; a second mirror confirms the < 0.057 eV figure.

**Early dark energy** [BACKGROUND]
- Axion-like EDE with DESI BAO + ACT DR6 lensing: **f_EDE < 0.091 (95%)** for Planck+lensing+DESI, and < 0.070 for Planck+lensing alone. Σm_ν < 0.096 eV with EDE. — [PRD (xhh6-9v62)](https://link.aps.org/doi/10.1103/xhh6-9v62)

**Negative-neutrino-mass preference**
- [BACKGROUND, near window] **"Negative neutrino mass or negative dark energy?"** (arXiv:2605.21456, May 2026). — [arXiv HTML](https://arxiv.org/html/2605.21456v1)
  - DESI DR2 + CMB + DES-Dovekie: **Σm_ν,eff = −0.075 (+0.039/−0.053) eV in ΛCDM**.
  - Sign-switching cosmological constant (Λ_sCDM): Σm_ν,eff = +0.055 ± 0.050 eV, with transition redshift **z† > 2.4 (95%)** and best fit z† ≈ 2.78.
  - The CMB-only geometric degeneracy needs Σm_ν,eff ≈ −0.11 eV to match DESI's H0 r_d. The tension reaches about 3σ.
- [BACKGROUND] Elbers et al. (arXiv:2407.10965): Planck+ACT+DESI gives a 2.8–3.3σ tension with oscillation minima. Full posterior Σm_ν,eff = −0.125 (+0.058/−0.070) eV. In w0wa it becomes +0.06 (+0.15/−0.10) eV. — [arXiv PDF](https://arxiv.org/pdf/2407.10965)
- [BACKGROUND] Craig et al. (arXiv:2405.00836): Σm̃_ν = −160 ± 90 meV. Σm_ν = 58 meV suppresses lensing power by 2–3%, while −160 meV enhances it by 6–9%. — [arXiv PDF](https://arxiv.org/pdf/2405.00836)
- [BACKGROUND] Green & Meyers (arXiv:2407.07878): the negative-mass preference "is a consequence of larger than expected lensing of the CMB". — [arXiv HTML](https://arxiv.org/html/2407.07878)
- [BACKGROUND] DESI DR2 neutrino PRD (DOI 10.1103/w9pk-xsk7) reports Feldman–Cousins limits and Σm_ν,eff in tension with oscillations. — [PRD](https://link.aps.org/doi/10.1103/w9pk-xsk7). The exact numbers were stripped from the retrieved abstract.

**Simons Observatory**
- LAT first light came in Feb–Mar 2025. "Initial Science Observations" (unblinded 2025 data, ~1% of the sky at ACT-like depth) were shown on 2 Jun 2026; power spectra are seen but no parameters were reported. — [GdR CoPhy talk, 2 Jun 2026](https://indico.ijclab.in2p3.fr/event/13091/contributions/41862/attachments/27824/41604/GDRCophy%20010626.pdf)
- **[IN WINDOW] LAT readout performance paper** (arXiv:2609.39100, 30 Sep 2026): 13 optics tubes installed by early 2026; 81.4% of 63,000 TES detectors show transitions; median readout noise 68.1 pA/√Hz. Instrument only. — [arXiv HTML](https://arxiv.org/html/2609.39100)
- Forecast, not a measurement: σ(Σm_ν) ≈ 0.03 eV from SO CMB alone, about 0.015 eV with DESI and better τ. — [SO forecasts (JCAP 2025)](https://google.iopscience.iop.org/article/10.1088/1475-7516/2025/08/034)

### Inferences
- Cosmological upper limits (Σm_ν < 0.052–0.072 eV) now press against or below the normal-ordering minimum of about 0.059 eV. That is either a sign of new physics (excess CMB lensing, or evolving or negative-density dark energy) or of systematics.
- **Hook for a σ-field model:** if the field changes H(z) at 1.5 ≲ z ≲ 10, it changes the predicted CMB-lensing amplitude. Its "effective" Σm_ν can be computed and compared with −0.075 (+0.039/−0.053) eV (ΛCDM) and +0.055 ± 0.050 eV (Λ_sCDM). A sign-switch in the vacuum energy near z ≈ 2.4–2.8 is the specific late-time structure that restores positive masses.
- N_eff = 2.86 ± 0.13 against the standard 3.044 means any extra early radiation from a σ-field must satisfy ΔN_eff < 0.14–0.17 (95%).
- ACT bounds early-universe variation of m_e and α_EM to about 0.5% and 0.2% (1σ). This directly constrains σ-field couplings to particle masses at recombination.

### Gaps
- No Planck "PR5" or new official Planck reanalysis was found in the window.
- No ACT "DR6 final / legacy" cosmology paper dated June–October 2026 was found; the DR6 parameter papers are from March 2025.
- No Simons Observatory cosmological parameters had been released.
- The exact Feldman–Cousins numbers from the DESI DR2 neutrino PRD were not retrieved.

---

## Q5. New theory premises: phantom crossing, interacting DE, quintessence, cosmologically coupled black holes (k = 3) and tests, black-hole cosmology, r_s ≈ R_H, timescape/backreaction, varying G or constants

### Takeaway
- **Cosmologically coupled black holes (CCBH, k = 3):** the strongest test so far appeared in the window. Lei et al. (A&A Letter, 1 Oct 2026) find **k = −0.16 ± 0.39**, which excludes k = 3 at **8.1σ** using JWST high-z AGN in red-sequence hosts.
- **Phantom crossing:** this remains the phenomenological reading of DESI. Single canonical quintessence fits only marginally and cannot cross w = −1.
- **Black-hole-universe / r_s = R_H:** active only in the Gaztañaga line (r_S = r_Λ, Λ = 3/r_S²) and in non-peer-reviewed preprints. The identity r_s(M_crit) = c/H is exact for a flat FLRW critical-density Hubble sphere, so it is a tautology rather than a test.
- **Timescape:** the Pantheon+ preference reported by Seifert et al. failed to carry over to DES-Dovekie in an independent replication (Sep 2026, non-peer-reviewed).
- **Varying G:** no detection.

### Cited Findings

**Cosmologically coupled black holes (Farrah/Croker k = 3)**
- **[IN WINDOW] Lei, Wang, Wang, Feng & Fan, "Black holes in red-sequence galaxies at z ∼ 0.7–2.5: Not a source of dark energy, but remnants of little red dots"** — A&A Letter, published 1 Oct 2026, DOI 10.1051/0004-6361/202661516; arXiv:2506.19589 (v1 claimed >10σ, about 11σ with k = 0.13 ± 0.27). — [A&A](https://www.aanda.org/articles/aa/full_html/2026/10/aa61516-26/aa61516-26.html); [arXiv v1](https://arxiv.org/html/2506.19589v1)
  - Model: M_BH(a) = M_BH(a_i)(a/a_i)^k.
  - **Published value: k = −0.16 ± 0.39 (68%), which excludes k = 3 at 8.1σ.** Without bias corrections k = −0.48 ± 0.27 (9.3σ). With an extended B_g prior k = −0.12 ± 0.40 (7.7σ).
  - Little red dots evolved from z ~ 6 reproduce Farrah's z ~ 0.7–2.5 sample without coupling.
  - Cosmology used: H0 = 67.36, Ω_m = 0.3135, Ω_Λ = 0.6847.
  - Status: peer-reviewed. Earlier tests (Rodriguez 2023 NGC 3201 at 3.8σ; Andrae & El-Badry 2023) assumed a TOV lower mass bound, which may not apply to vacuum-energy interiors. This test does not rely on that assumption.
- [BACKGROUND] Claims in favour of coupling:
  - Farrah et al. 2023 report k = 3.09 ± 0.76 and exclude zero coupling at 99.98%. With w = −k/3, k = 3 behaves like vacuum energy and accounts for Ω_Λ ≈ 0.68 through stellar-remnant BHs. — [arXiv Science 2302.07878](https://arxiv.science/abs/2302.07878)
  - Croker et al. (arXiv:2405.12282) fit CCBHs to DESI: **H0 = 69.94 ± 0.81**, reducing the SH0ES tension to 2.7σ, with "two fewer parameters than w0wa". The DE density tracks star formation and plateaus at z ≲ 1. — [arXiv HTML v3](https://arxiv.org/html/2405.12282v3)
- [BACKGROUND] Amendola et al. (MNRAS 528, 2377, 2024): LVK minimum-mass test gives k < 2.1 (log-uniform delay) or k < 1.1 (CCBH-corrected) at 2σ. — [ADS](https://ui.adsabs.harvard.edu/abs/2024MNRAS.528.2377A/abstract)
- [BACKGROUND] Ghodla et al.: k = 3 would raise binary-BH merger rates by orders of magnitude over LVK observations. — [arXiv Science 2306.08199](https://arxiv.science/abs/2306.08199)
- [BACKGROUND] Wu, Chu & Hu (arXiv:2603.24609; EPJC 86, 288, 2026): an exact BH-in-anisotropic-dark-sector solution in which the coupling exponent is radius-dependent and set by the dark-halo profile. — [arXiv abs](https://arxiv.org/abs/2603.24609)

**Black-hole cosmology, and the r_s ≈ R_H relation**
- [BACKGROUND] **Gaztañaga, "The Cosmological Constant as Event Horizon"** (arXiv:2202.00641; Symmetry 14, 300, 2022) and the BHU papers. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2202.00641); [HAL BHU](https://hal.science/hal-03344159v2/file/BHU_MNRAS.pdf)
  - Claim: **r_Λ = √(3/Λ) = 1/H_Λ is an event horizon, and the universe expands inside its own Schwarzschild radius r_S = r_Λ = 2GM**. Cancelling the GHY boundary term requires **Λ = 3/r_S²**.
  - The BH density ρ_BH = 3r_S⁻²/(8πG) = ρ_Λ. Gaztañaga states this is about 25% below the critical density today.
  - The mass inside the future event horizon is M ≈ 5×10²² M_⊙. Collapse happened "25 Gyrs ago".
  - A 2025 PRD "Black-Hole Universe" bounce model predicts small positive spatial curvature. This is reported secondhand via a review page; the PRD itself was not retrieved. — [Sadasivam review page (non-academic)](https://prabhusadasivam.com/blackhole)
- **[IN WINDOW, non-peer-reviewed] Phanvilai, "On the Possibility of Our Universe as the Interior of a Black Hole"** (Zenodo preprint, 21 Jun 2026). — [Exa library record](https://exa.ai/library/publication/19x2p3pbgn4)
  - Claims a Kantowski–Sachs interior gives **Λ_eff = (32/5π) r_S⁻² ≈ 2.04 r_S⁻²**, and Λ_eff/Λ_obs ≈ 0.97.
  - Claims a BHU accretion fit to DESI DR1 with ε ≈ 0.38, giving **w0 = −0.747, wa = −0.282** and Δχ² = −10.5. Low reliability.
- **[IN WINDOW, fringe / non-peer-reviewed] Wojnow, "R_H = ct Thermodynamic Cosmology…"** (July 2026). It builds on the Haug–Tatum "black-hole Hubble sphere" with R_H = r_s = ct and T_CMB = √(T_Hawking,Hubble · T_Hawking,Planck). — [PDF](https://constantecosmologique.fr/wp-content/uploads/2026/07/001-FHU-HTC-en.pdf); [Haug 2025](https://exa.ai/library/publication/vx7bk9ll3m0); [Haug & Tatum entropy (JAMP)](https://www.scirp.net/pdf/jamp_1724348.pdf)
  - Haug–Tatum statement of the identity: "In a critical Friedmann universe, the mass is M = c²R_H/(2G)… the Hubble radius and the Schwarzschild radius are identical." — [Haug & Tatum](https://www.scirp.net/pdf/jamp_1724348.pdf)
- [BACKGROUND, EPJC issue June 2026] Mehrotra compares ΛCDM with **R_h = ct** using DESI DR2 + CC + SNe. ΛCDM is preferred by AIC/BIC. Ages: **t0(ΛCDM) = 13.676 (+0.92/−0.81) Gyr; t0(R_h = ct) = 16.035 (+1.09/−0.98) Gyr**. — [EPJC](https://epjc.epj.org/articles/epjc/abs/2026/06/10052_2026_Article_15890/10052_2026_Article_15890.html)

**Phantom crossing, quintessence and interacting DE**
- The DESI DR2-IV statement (Jul 2026) that w(z) "still favors a phantom-crossing-like behaviour" is quoted under Q1. — [arXiv HTML 2607.27410](https://arxiv.org/html/2607.27410)
- No-go: "a single canonical scalar field or a single perfect fluid model cannot reproduce such quintom-like behavior". Crossing needs non-minimal coupling, modified gravity or EFT. — [Chaudhary et al., EPJC 2026](https://link.springer.com/article/10.1140/epjc/s10052-026-15806-w); [Pantos & Perivolaropoulos](https://arxiv.org/abs/2601.00650)
- [BACKGROUND] Adam, Hertzberg et al., JCAP04(2026)052. They test quadratic, quartic hilltop, double-well, cosine, Gaussian and inverse-power potentials and find "only modest" improvement over Λ. Non-minimally coupled scalars can give a temporary w < −1, but they produce a fifth force and a time-varying G_eff that are "tightly constrained". Only a "very narrow range of carefully selected non-minimal couplings" survives. — [JCAP](https://iopscience.iop.org/article/10.1088/1475-7516/2026/04/052)
- [BACKGROUND] Exponential quintessence V = V0 e^(−λφ) with DESI DR2: **λ = 0.698 (+0.173/−0.202)** (flat) or 0.722 (+0.182/−0.208) with Ω_K = 0.003 ± 0.001, a 3.3–3.8σ preference in that paper. — [arXiv 2504.04226](https://arxiv.org/html/2504.04226)
- [BACKGROUND] Thawing quintessence (Padé-w parameterization) is "consistently preferred" over Λ with DESI DR2 + Planck/ACT + any SN sample including Dovekie. The preference is prior-sensitive. — [arXiv 2512.20832](https://arxiv.org/pdf/2512.20832)
- [BACKGROUND] A Gaussian-process reconstruction of V(φ) and kinetic energy from DESI DR2 + Pantheon+ finds a monotonically decreasing potential consistent with thawing. — [arXiv 2603.21125](https://arxiv.org/html/2603.21125v1)
- [BACKGROUND, near window] Sign-switching Λ (Λ_sCDM, "negative dark energy" at z > z†): z† > 2.4 (95%). — [arXiv 2605.21456](https://arxiv.org/html/2605.21456v1)
- Interacting dark energy is reviewed as the leading surviving Hubble-tension route in Cai & Wang [IN WINDOW]. No quantitative coupling bound was extracted. — [arXiv 2606.20434](https://arxiv.org/abs/2606.20434)

**Timescape / backreaction**
- [BACKGROUND] Seifert et al. (arXiv:2412.15143): Pantheon+ gives ln B > 5 for timescape over ΛCDM on the full sample, and ln B > 1 for z_min ≥ 0.075. Void fraction **f_v0 = 0.737 ± 0.029**. — [arXiv HTML](https://arxiv.org/html/2412.15143v1)
- **[IN WINDOW, non-peer-reviewed deposit] Nguyen, "Timescape versus ΛCDM with the Seifert et al. supernova method: a replication, an application to DES-Dovekie and a measurement-error test"** (22 Sep 2026). — [Exa library record](https://exa.ai/library/publication/r2q5m8f182d)
  - Reproduces the Pantheon+ Bayes factors to within 0.08.
  - On DES-Dovekie: **ln B < 0 at every cut (−2.70 to −0.71)**.
  - Including stretch and colour measurement errors turns the Pantheon+ z_min = 0.075 preference into a likelihood ratio of about 2.

**Varying G and constants**
- [Just before window, 9 Jun 2026] Colaço, Holanda & Ferreira use cluster gas mass fractions + Pantheon+ (Chandrasekhar-mass luminosity scaling) with a Gaussian process. "A constant gravitational coupling remains broadly consistent with the data, although mild low-redshift departures are allowed." — [Exa library record](https://exa.ai/library/publication/bpx1zcylnx0)
- ACT DR6: α_EM/α_EM,0 = 1.0043 ± 0.0017 and m_e/m_e,0 = 1.0063 ± 0.0056. SPT-3G+DESI show a "mild preference" for varying m_e in a non-flat universe. — [ACT](https://www.repository.cam.ac.uk/items/9a1242d7-0031-4767-8380-a7dc8c9d5a7e); [SPT-3G D1](https://arxiv.org/abs/2506.20707)

### Inferences
These bullets matter most for sigma-ground.
- **The r_s / R_H identity is a tautology of flat FLRW, not a discovery.**
  - Take ρ_c = 3H²/(8πG) and M = ρ_c (4π/3)(c/H)³. Then M = c³/(2GH) and r_s = 2GM/c² = c/H exactly, for any H.
  - My computation: M = 4.658×10²² M_⊙ for H0 = 67.24; 4.595×10²² for 68.17; 4.262×10²² for 73.50. In each case r_s/R_H = 1.000.
  - Unit-test hook: assert r_s(M_crit(H)) == c/H to machine precision. Report that it holds for every H0, so it cannot discriminate between cosmologies.
- **Gaztañaga's version is not the same identity.** It sets r_S = r_Λ = c/H_Λ = c/(H0√Ω_Λ), not c/H0.
  - My computation: r_Λ = 5388 Mpc and M = 5.63×10²² M_⊙ (H0 = 67.24, Ω_Λ = 0.6847); Λ = 3/r_Λ² = 1.085×10⁻⁵² m⁻².
  - With H0 = 68.17 and Ω_Λ = 0.6973: r_Λ = 5266 Mpc, M = 5.50×10²² M_⊙, Λ = 1.136×10⁻⁵² m⁻².
  - A code implementing the "r_s = R_H identity" should state which horizon (Hubble c/H0 or de Sitter c/H_Λ) it equates to r_s. They differ by a factor of 1/√Ω_Λ ≈ 1.21.
  - Under evolving dark energy (w0 ≠ −1), r_Λ is not constant, which breaks the BHU's Λ = 3/r_S² as a fixed constant.
- **The Haug–Tatum CMB-temperature relation is a computable hook, but it is fringe.**
  - Formula as I reconstructed it from their stated geometric mean: T = ħc/(4π k_B √(2 l_P R_H)).
  - My computation: T = 2.7325 K for H0 = 67.24, 2.7513 K for H0 = 68.17, 2.8569 K for H0 = 73.50.
  - Inverting with T_CMB = 2.72548 K gives **H0 ≈ 66.89**. This value is disfavoured by SH0ES/H0DN and sits about 1σ below CMB-SPA.
- **CCBH at k = 3 (w = −k/3 = −1) is now strongly disfavoured** (8.1σ, with independent 2–4σ LVK, globular-cluster and Gaia tests). If the codebase couples black-hole mass growth to dark energy, the test number is k = −0.16 ± 0.39 at z ≲ 6. A σ-field model producing BH mass growth ∝ a^k must keep k ≲ 1 (2σ ≈ 0.6 from this result).
- **For the σ-field as scalar dark energy:** canonical quintessence can match w_p ≈ −0.98 near z ≈ 0.3 and the thawing shape, but cannot cross into w < −1. Matching the DESI CPL best fit literally needs non-minimal coupling. That brings fifth-force and Ġ/G constraints, which are not quantified in the sources retrieved here.

### Gaps
- No peer-reviewed 2026 paper giving a quantitative observational test of r_s = c/H0 (as opposed to the tautology) was found.
- No 2026 Gaztañaga BHU arXiv paper with new quantitative curvature predictions was retrieved. The predicted Ω_K value is unknown.
- No June–October 2026 interacting-DE paper with a numerical coupling constraint (e.g. ξ or β from DESI DR2+CMB) was retrieved.
- No 2026 bound on Ġ/G in units of yr⁻¹ was extracted. The Colaço et al. result is qualitative.
- No 2026 PTA/GWB-based CCBH test was retrieved, although Lei et al. note that PTA analyses have favoured k ≈ 3.

---

## Q6. Cosmic age, q0, transition redshift, and any slowing of acceleration

### Takeaway
The "universe is no longer accelerating" claim (Son et al. 2025 progenitor-age bias; Sah, Rameez & Sarkar, MNRAS June 2026) was directly rebutted:
- Wiseman, Popovic, Sullivan, Riess & Scolnic, "Still accelerating", MNRAS June 2026.
- A revised Pantheon+ decomposition (Aug 2026), which gives q_m = −0.490 without the correction and −0.267 with it, so still accelerating.

Standard fits give t0 ≈ 13.7–13.8 Gyr, q0 ≈ −0.55 (ΛCDM) and z_t ≈ 0.66. Notably, the DESI+CMB-only w0wa posterior means imply q0 ≈ 0 to +0.09. A "slowing acceleration" therefore follows from the DESI best fit itself, but not from the SN-inclusive fits.

### Cited Findings

**The age-bias / deceleration debate**
- [BACKGROUND] **Son, Lee, Chung et al. ("Strong progenitor age bias in supernova cosmology")**, MNRAS 544, 975 (2025); summary in Universe 12, 22 (15 Jan 2026). — [MDPI Universe](https://www.mdpi.com/2218-1997/12/1/22)
  - Reports a 5.5σ host-age–Hubble-residual correlation.
  - After correction, SN+BAO+CMB give a "~9σ" discordance with ΛCDM and a "currently non-accelerating universe".
  - Adopted DESI w0wa values: "r_d H0 = 93.9 ± 2.9 Mpc" as printed (presumably r_d·h), and H0(BAO+CMB) = 63.7 ± 2.0.
- **[IN WINDOW] Wiseman, Popovic, Sullivan, Riess & Scolnic, "Still accelerating: type Ia supernova cosmology is robust to host galaxy age evolution"**, MNRAS 549, stag797 (10 Jun 2026). — [MNRAS](https://academic.oup.com/mnras/article/549/3/stag797/8703725)
  - With the standard host-mass step there is no dependence on host age.
  - DES measures host-step evolution of **−0.028 ± 0.034 mag z⁻¹**, which changes w by less than 0.01.
  - The claimed ~5 Gyr progenitor-age difference is overstated by a factor of 3–5.
- **[IN WINDOW] Sah, Rameez & Sarkar, "Pantheon+ supernovae corrected for progenitor age indicate the universe is decelerating"**, MNRAS 549, stag844 (11 Jun 2026); arXiv:2606.09650. — [MNRAS](https://academic.oup.com/mnras/article/549/3/stag844/8701424); [arXiv HTML](https://arxiv.org/html/2606.09650)
  - Locally, q0 is mainly a dipole aligned with the bulk flow, dying out by z ~ 0.1.
  - Applying the age correction shifts the q0 monopole to positive values (deceleration).
  - Status: contested.
- **[IN WINDOW] Park, Lee, Chung, Yoon & Son, "Strong progenitor age bias III"**, MNRAS 549(2), 1–17 (26 Jun 2026). It claims the host mass/sSFR steps are "projected manifestations" of a progenitor-age dependence. — [Yonsei record](https://yonsei.elsevierpure.com/en/publications/strong-progenitor-age-bias-in-supernova-cosmology-iii-progenitor-/)
- [Near window] Son et al. reply, "Still non-accelerating" (arXiv:2605.21586, May 2026): the correction is "largely unchanged". — [arXiv HTML](https://arxiv.org/html/2605.21586)
- **[IN WINDOW] "Accelerating expansion and isotropic sky-hemisphere consistency in Pantheon+ supernovae: a revised analysis"** (arXiv:2607.20570, 11 Aug 2026). — [arXiv HTML](https://arxiv.org/html/2607.20570)
  - 1564 SNe.
  - **q_m = −0.490 without age correction** (ΛCDM expects −0.55). **q_m = −0.267 with the Son et al. correction**, so still accelerating.
  - CMB-dipole hemisphere gives q_m = −0.527 and the anti-dipole hemisphere −0.464, which supports isotropy.
  - "Do[es] not support… a decelerating universe."
- **[IN WINDOW] Nguyen, "Progenitor age-bias-corrected Type Ia supernovae favor a logarithmic luminosity-distance relation"** (arXiv:2608.00806, 1 Aug 2026). — [arXiv abs](https://arxiv.org/abs/2608.00806)
  - Fits d_L = (c/H0)(1+z) ln(1+z).
  - Joint Δχ² = 17.1 against flat ΛCDM on age-corrected Pantheon+ and DES (ΔAIC ≈ 19.1, ΔBIC ≈ 25.2).
  - Einstein–de Sitter is rejected.
  - Conditional on the contested correction.

**Kinematic q(z) reconstructions**
- [BACKGROUND, May 2026] Verdugo et al., EPJC. CC+Pantheon+ + HII galaxies + QSO give **q0 = −0.25 ± 0.04, z_T ≈ 0.80, h = 0.729 ± 0.006**, and "possible reduction of the late-time acceleration". The authors caution that this depends on the parameterization. — [EPJC](https://link.springer.com/article/10.1140/epjc/s10052-026-15766-1)
- [BACKGROUND] Model-independent reconstruction (CC + DESI DR2 + DESY5): compatible with ΛCDM within 1σ for 0 ≲ z ≲ 1.2, with "no statistically significant evidence for phantom crossing". CC-only H0 = 64.99 ± 6.94. — [ApJ (2026)](https://google.iopscience.iop.org/article/10.3847/1538-4357/ae56f5)
- [BACKGROUND] DESI DR1 cosmic chronometers: **H(z = 0.46) = 88.48 ± 0.57 (stat) ± 12.32 (sys); H(0.67) = 119.45 ± 6.39 ± 16.64; H(0.83) = 108.28 ± 10.07 ± 15.08**. Combined H0 = 72.82 ± 6.03. — [arXiv 2511.02730](https://arxiv.org/html/2511.02730)
- [BACKGROUND] Transition-redshift compilations: CC+Pantheon+ give **0.61 < z_t < 0.79** across ΛCDM/XCDM variants. — [RAA 2023](https://iopscience.iop.org/article/10.1088/1674-4527/ace17a/meta)

**Cosmic age**
- [BACKGROUND, June 2026 issue] ΛCDM t0 = 13.676 (+0.92/−0.81) Gyr from DESI DR2+CC+SNe; R_h = ct gives 16.035 (+1.09/−0.98) Gyr. — [EPJC Mehrotra](https://epjc.epj.org/articles/epjc/abs/2026/06/10052_2026_Article_15890/10052_2026_Article_15890.html)
- The q(z) paper notes current estimates of t0 ≈ 13.8 Gyr, and that ages "significantly below ∼13 Gyr would be difficult to reconcile with stellar population data". — [EPJC Verdugo](https://link.springer.com/article/10.1140/epjc/s10052-026-15766-1)

### Inferences
These are my numerical derivations. They use flat cosmology, Ω_r h² ≈ 4.18×10⁻⁵, massive neutrinos folded into Ω_m, and the published posterior means, so treat them as indicative to about 0.01 in q0 and about 20 Myr in t0. They are useful as regression targets.
- **ΛCDM DESI+CMB** (H0 = 68.17, Ω_m = 0.3027): **t0 = 13.79 Gyr, q0 = −0.546, z_t = 0.663**.
- DESI DR2-IV ΛCDM (68.11, 0.3042): t0 = 13.78 Gyr, q0 = −0.544, z_t = 0.659.
- DES Y6 all-probe ΛCDM (68.3, 0.302): t0 = 13.77 Gyr, q0 = −0.547, z_t = 0.665.
- **w0wa DESI+CMB+Dovekie** (67.37, 0.3134, −0.821, −0.65): **t0 = 13.75 Gyr, q0 = −0.345, z_t = 0.739**, phantom crossing at z_× ≈ 0.38.
- w0wa DESI+CMB+Pantheon+ (DR2): t0 = 13.76, q0 = −0.365, z_t = 0.737, z_× = 0.35.
- w0wa DESI+CMB+Union3 (DR2): t0 = 13.76, q0 = −0.173, z_t = 0.781, z_× = 0.44.
- w0wa DESI+CMB+DESY5 (DR2): t0 = 13.76, q0 = −0.268, z_t = 0.760, z_× = 0.41.
- **w0wa DESI+CMB only (DR2: −0.42, −1.75, Ω_m = 0.353, H0 = 63.6): q0 = +0.09**, which is decelerating today, with z_t = 0.85 and z_× = 0.50.
- **w0wa DESI+CMB only (DR2-IV: −0.54, −1.39, Ω_m = 0.343, H0 = 64.5): q0 = −0.03**, with z_t = 0.80 and z_× = 0.50.
- Reading of these numbers:
  - The "slowing acceleration" or even "currently non-accelerating" reading is built into the DESI+CMB-only w0wa posterior.
  - Adding any SN sample restores q0 ≈ −0.17 to −0.37.
  - The age is remarkably stable at 13.75–13.79 Gyr across all of these fits, because the CMB fixes it.
  - **Hook:** q0 = ½[1 + 3w0Ω_DE0] (neglecting radiation). Check, for example, q0(ΛCDM, Ω_m = 0.3027) = −0.546, and that t0 lands in [13.74, 13.80] Gyr for every DESI-family fit.
- If the codebase has a scalar field governing "scale transitions", its predicted z_t and q0 can be checked against:
  - z_t ≈ 0.66 (ΛCDM) to 0.74 (DESI+SN w0wa);
  - q0 ≈ −0.55 to −0.35;
  - q0 ≥ −0.27 only if the contested SN age correction is adopted (see q_m = −0.267 above).

### Gaps
- No June–October 2026 primary paper giving a CMB-anchored t0 with error bar was found. Planck/ACT/SPT ages were not extracted.
- The redshift range of the "3–4% increase in expansion rate" in the PRD on DESI DR2 expansion-history preferences was stripped from the retrieved abstract. — [PRD (zypq-s6nl)](https://link.aps.org/doi/10.1103/zypq-s6nl)
- No independent (non-Yonsei) confirmation of a > 5σ progenitor-age–luminosity slope was found. The deceleration claim remains a two-group dispute.
