# Astrophysics (non-DM/DE/BH/GW) and Other Headline Physics, ~2026-06-10 to 2026-10-10: Testable Benchmarks

Legend: **[IN WINDOW]** = published/announced 2026-06-10..2026-10-10. **[BACKGROUND]** = earlier, included for context or because it is still the current benchmark. "Hook" = a number or relation a Python engine (sigma-ground) could reproduce or test. Every number below comes from the linked source unless it is in an "Inferences" block, where I computed it myself and say so.

## Q1. JWST early galaxies: redshift records, overabundant bright galaxies vs ΛCDM, early quenched galaxies, cosmic-dawn timeline

### Takeaway
No galaxy beyond MoM-z14 (z_spec = 14.44) was spectroscopically confirmed during the window. The frontier is stuck near z ≈ 14.5, and deeper spectroscopy of fainter z > 15 candidates keeps turning up interlopers. That produced two in-window lines of work: claims of a sharp "Cosmic Daybreak" at z ≈ 15, and ultradeep follow-up of the z ≈ 14 galaxies, which shows fast metal enrichment and very hard ionizing spectra under 300 Myr after the Big Bang. The strongest Pop III candidate yet ("Hebe", z = 10.6) went from claim to interpretation papers in mid-2026. The overabundance of bright z ≈ 14–15 galaxies (×182 over pre-JWST models) and quenched galaxies that formed within about 300 Myr are the hardest numbers for ΛCDM timelines.

### Cited Findings
**Redshift frontier / record holders**
- **[BACKGROUND, 2025 preprint; NASA release 2026-01-28] MoM-z14 ("Mirage or Miracle" survey), arXiv:2505.11263.** z_spec = 14.44 ± 0.02, about 280 Myr after the Big Bang. M_UV = −20.2, r_e = 74 (+15/−12) pc, β = −2.5 ± 0.2, axis ratio b/a = 0.25. SFR_5Myr/SFR_50Myr = 9.9 (+3.0/−5.8). The number density of bright z ≈ 14–15 sources is **182 (+329/−105)× pre-JWST consensus models**. Status: published. **Hook:** cosmic age at z = 14.44 under Planck ΛCDM (NASA quotes ~280 Myr; lookback ~13.5 of 13.8 Gyr). Also the predicted abundance of halos able to host M_UV ≈ −20 at z ≈ 14.5 vs the observed ×182 excess. — [alphaXiv abstract](https://www.alphaxiv.org/abs/2505.11263); [NASA Webb release](https://science.nasa.gov/missions/webb/nasa-webb-pushes-boundaries-of-observable-universe-closer-to-big-bang/)
- **[BACKGROUND] JADES-GS-z14-0.** The NIRSpec prism gave z = 14.32 (+0.08/−0.20), r ≈ 260 pc ([Carniani et al., Nature 2024](https://www.nature.com/articles/s41586-024-07860-9)). ALMA [OIII]88µm then refined it to **z = 14.1793 ± 0.0007**, M_UV = −20.81 ± 0.16, Z ≈ 0.05–0.2 Z☉, dust-to-stellar mass < 2×10⁻³ ([ESO science paper eso2507](https://eso.org/public/archives/releases/sciencepapers/eso2507/eso2507b.pdf)). **Hook:** the redshift from the [OIII] 88.356 µm rest wavelength is a clean λ_obs = (1+z)λ_rest test.
- **[IN WINDOW, 2026-06-15] Helton et al., "Ionizing Photon Production Efficiencies and Chemical Abundances at Cosmic Dawn… JADES-GS-z14-0", ApJL 1004 L30, DOI 10.3847/2041-8213/ae7446.** This is the longest MIRI spectrum ever taken (≈51 h). [O III]λλ4959,5007 detected at ≈14σ and Hα at ≈4σ. SFR ≈ 8 ± 2 M☉/yr, Σ_SFR ≈ 20 ± 5 M☉/yr/kpc², ξ_ion ≈ 10^(25.3±0.1) Hz/erg, [C/O] ≈ −0.4 ± 0.2, n_H ≈ 690 ± 200 cm⁻³. The oxygen abundance is ≈6% Z☉ from strong-line diagnostics but ≈60% Z☉ from photoionization modelling, a 10× method systematic. The authors say existing simulations cannot reproduce these properties. **Hook:** Hα-to-SFR conversion and the ξ_ion definition (ionizing photon rate / L_UV,ν); metallicity buildup within ~290 Myr. — [IOPscience](https://iopscience.iop.org/article/10.3847/2041-8213/ae7446/meta)
- **[BACKGROUND, 2026-05-04] Donnan et al., PAN-z14-1, ApJ 1002 134, DOI 10.3847/1538-4357/ae5c05 (arXiv:2601.11515).** z_spec = 13.53 (+0.05/−0.06), the fourth most distant known. M_UV = −20.6 ± 0.2, β = −2.26 ± 0.08, r_c = 233 ± 10 pc, no UV lines at ≥2σ. The authors argue for two z > 10 populations, compact strong-line emitters vs extended "normal" galaxies. — [IOPscience](https://google.iopscience.iop.org/article/10.3847/1538-4357/ae5c05); [arXiv Science](https://arxiv.science/abs/2601.11515)
- **[BACKGROUND, 2026-03] EGS-z11-R0, arXiv:2603.15841.** z_spec = 11.452 ± 0.021, the most distant spectroscopically confirmed red ("red monster") galaxy. β_UV ≈ −1.0, log M* ≈ 9.2–9.6, SFR ≈ 10–40 M☉/yr. — [arXiv](https://arxiv.org/abs/2603.15841)

**Cosmic-dawn timeline / UV luminosity function**
- **[IN WINDOW, 2026-09-23, preprint] Zhang, Broadhurst, Chiueh, Umetsu, Schive et al., "JWST evidence for a sharp 'Cosmic Daybreak' at z=15", arXiv:2609.28257.** The paper notes JWST finds luminous galaxies easily out to z ≈ 14.5, while deeper spectroscopy of fainter candidates "now finds only interlopers". It also reports that measured stellar ages converge to zero by z = 15. The authors read this as a sharp transition, unlike ΛCDM's gradual cosmic dawn, and fit wave dark matter with Jeans mass 4×10⁹ M☉ at z = 15 → m_ψ = 2.2 (+0.4/−0.3)×10⁻²² eV. Status: unrefereed, and the interpretation is model-dependent; the dark-matter side belongs to the DM researcher. **Hook:** age of the universe at z = 15 vs summed stellar ages of the z ≈ 14 galaxies. Would a "first light at z ≈ 15" be consistent with ΛCDM halo collapse? — [arXiv](https://arxiv.org/abs/2609.28257)
- **[BACKGROUND, 2026-04] arXiv:2604.16666 (UV LF at 12.5 < z < 18.5).** Based on >0.6 deg² of NIRCam imaging over >150 sightlines. The decline in ρ_SFR at z > 12 is **≈4× steeper** than at z < 12. ρ_SFR at z ≃ 15.5 lies significantly below the log-linear extrapolation, and there is a "notable lack" of candidates at z > 14.5. — [arXiv HTML](https://arxiv.org/html/2604.16666v1)
- **[IN WINDOW, 2026-08-13, preprint] SPURS: GN-z11 (z = 10.6) ultradeep rest-UV, arXiv:2608.12699.** P-Cygni wind features and broad He II are fit by very massive stars (>100 M☉) at low Z with ages ≲3 Myr. Broad N IV]λ1486 FWHM = 1670 km/s, a ~500 km/s ionized outflow, Lyα EW = 5.6 Å, f_esc(Lyα) = 2.7%, and 44% of the Lyα flux at >500 km/s. — [Emergent Mind summary of arXiv:2608.12699](https://www.emergentmind.com/papers/2608.12699)

**Population III ("Hebe", near GN-z11)**
- **[BACKGROUND→IN WINDOW] Maiolino et al., arXiv:2603.20362 (v4 2026-04-06).** NIRSpec-IFU G235H confirms a He II λ1640 emitter 3 pkpc from GN-z11 at z = 10.6. It has two components separated by 120 km/s, EW(He II) ≫ 20 Å, and no metal lines. The authors argue Pop III is the most plausible source. — [arXiv HTML](https://arxiv.org/html/2603.20362v4)
- **[BACKGROUND, 2026-05-18] Rusta et al., ApJL, DOI 10.3847/2041-8213/ae64e1 (arXiv:2603.20363).** Both components need >50% of their stellar mass in Pop III stars, with log O/H < 7.4. L_HeII(C1) = (5.1 ± 0.9)×10⁴⁰ erg/s and Pop III stellar mass 2×10⁴ < M*/M☉ < 6×10⁵. A top-heavy IMF is favoured: Salpeter-slope IMFs with M_ch < 75 M☉ are disfavoured at 1σ for ages ≤1 Myr. — [IOPscience](https://iopscience.iop.org/article/10.3847/2041-8213/ae64e1); [arXiv PDF](https://arxiv.org/pdf/2603.20363)
- **[IN WINDOW, 2026-07-13] Jeon, Jeong, Zhang & Bromm, "What is Powering the Enigmatic He II Emitter Hebe", ApJ, DOI 10.3847/1538-4357/ae7bea.** From simulations the Pop III starburst mass is ~2.5×10⁵ M☉ initially and 6.6×10⁵ M☉ at maximum. An accreting SMBH (10⁴–10⁵ M☉, direct-collapse BH or PBH) gives a "less natural" fit. A Pop II cluster would need M* ≈ 10⁷ M☉ and would show a detectable continuum. **Hook:** He II λ1640 / Hγ recombination ratios vs stellar effective temperature (T_eff ~10⁵ K Pop III photospheres). — [IOPscience](https://iopscience.iop.org/article/10.3847/1538-4357/ae7bea)

**Early quenched / "too old too early" galaxies**
- **[BACKGROUND, 2026-04-27] "Eridu", ApJ, DOI 10.3847/1538-4357/ae5233.** A massive quiescent galaxy at z = 2.675. Its star-formation histories imply formation **within ~300 Myr of the Big Bang** and quenching ~2 Gyr before observation (z ~ 10). That requires an M* ≈ 10¹¹ M☉ progenitor at z > 10, "nearly 2 orders of magnitude" above the most massive high-z observations. It is strongly α-enhanced, and metallicity varies ~0.4 dex between solar-scaled and α-enhanced models. — [IOPscience](https://iopscience.iop.org/article/10.3847/1538-4357/ae5233)
- **[BACKGROUND, 2026-02] GS-z5-Q1 (z = 5.39) and COS-z5-Q1 (z = 5.11), A&A.** Low-mass quiescent galaxies with M* = 10^9.6 and 10^9.5 M☉. COS-z5-Q1 quenched >300 Myr before observation (z ~ 7) and formed at z ~ 11. — [A&A](https://www.aanda.org/articles/aa/full_html/2026/02/aa57207-25/aa57207-25.html)
- **[BACKGROUND, 2026-01-27] RUBIES number densities, ApJ 997 252.** Massive quiescent galaxies are "surprisingly common", ≳10⁻⁵ Mpc⁻³ by 4 < z < 5. — [IOPscience](https://iopscience.iop.org/article/10.3847/1538-4357/ae24e1)
- **[BACKGROUND] RUBIES-EGS-QG-1, z = 4.90.** 10¹¹ M☉ formed in a ~200 Myr burst ([Nature Astronomy 2024](https://www.nature.com/articles/s41550-024-02424-3)). GS-9209 at z = 4.658 has M* = 3.8 ± 0.2×10¹⁰ M☉ and quenched at z = 6.5 (+0.2/−0.5) ([Nature 2023](https://www.nature.com/articles/s41586-023-06158-6)).

### Inferences
- **Engine benchmark (my computation, flat ΛCDM: H0 = 67.66, Ωm = 0.3111, Ωr = 9.15×10⁻⁵).** t₀ = 13.786 Gyr. Ages at key redshifts: t(z=20) = 178 Myr; t(15) = 268 Myr; t(14.44, MoM-z14) = 283 Myr, which matches NASA's "280 Myr"; t(14.18, GS-z14-0) = 290 Myr; t(13.53) = 310 Myr; t(11.45) = 391 Myr; t(10.6, GN-z11/Hebe) = 435 Myr; t(5.124) = 1.135 Gyr; t(4.90) = 1.200 Gyr; t(2.675) = 2.430 Gyr. These are good regression targets for any cosmology module, and any σ-field modification of early expansion would shift them.
- The "Eridu" timeline (10¹¹ M☉ assembled by z ~ 10, i.e. in under ~470 Myr) is the most acute stellar-mass-budget test. An engine can compare M* to f_b × M_halo,max(z) using Ωb/Ωm ≈ 0.157 and a halo mass function. That is the cleanest "challenge to the standard timeline" item in this list.
- The empirical picture (bright galaxies to z ≈ 14.5, then a steep or absent population beyond) gives a falsifiable timeline. A model predicting a gradual ρ_SFR(z) out to z ~ 18 is in tension with arXiv:2604.16666 and 2609.28257.

### Gaps
- No spectroscopic confirmation above z = 14.44 was found for June–Oct 2026. The 2609.28257 preprint itself says the frontier is z ≈ 14.5, so this absence appears real rather than a search artefact.
- No in-window record for the earliest quenched galaxy (higher z or earlier quench) was found. The newest quenched results are Feb–Apr 2026.
- AAS 248 (June 2026) JWST highlights were not retrieved.

## Q2. Neutron stars and dense-matter EOS: NICER mass–radius, magnetars, FRB sources/progenitors, kilonovae

### Takeaway
The headline in-window EOS result is NICER's first radius for the heavy MSP PSR J1614−2230: M = 1.937 M☉ and R = 10.06 (+1.25/−0.87) km, the most compact NICER star so far. Folding it into χEFT+pQCD inference pulls heavy-NS radii down by ~0.4 km. A 6-yr re-analysis of PSR J0030+0451 (published July 2026) removed the old 14.4 km "stiff" solution. All NICER stars now cluster at R ≈ 11.5–12.7 km for ~1.3–1.4 M☉. FRB work produced a metallicity split between repeater and non-repeater hosts. Magnetar and kilonova work in the window is mostly simulation and theory.

### Cited Findings
- **[IN WINDOW, 2026-08-31, preprint] "A NICER view of PSR J1614−2230: a massive and compact millisecond pulsar", arXiv:2609.00172 (NICER pulse-profile team; author list not retrieved).** Combined NICER + XMM-Newton + Chandra data give **R_eq = 10.06 (+1.25/−0.87) km and M = 1.937 (+0.012/−0.013) M☉** (68% equal-tailed). The pulse significance is 8.5σ. Across all geometries and atmospheres, the 90th-percentile radius is ≲12.08 km. The fit has one hot spot near the pole and one near the equator, and no non-thermal component. **Hook:** solve the TOV equations for a candidate EOS and check R(1.937 M☉) against 10.06 km. Compactness GM/(Rc²). — [arXiv](https://arxiv.org/abs/2609.00172)
- **[IN WINDOW, 2026-08-31, preprint] Mendes, Svensson, Gottling, Hebeler, Schwenk, Rutherford, Mauviard, Kazantsev, Kini, Gonzalez-Caniulef, Guillot & Watts, arXiv:2609.00173.** Adds the J1614−2230, J2124−3358 and 47 Tuc X7 constraints to χEFT + pQCD EOS inference. J1614 "shifts heavy neutron stars to smaller radii by around 0.4 km". It significantly tightens the pressure and speed-of-sound posteriors at ~3 n_sat. **Hook:** P(ε) and c_s²(n) near 3n₀ (n₀ = 0.16 fm⁻³). — [arXiv](https://arxiv.org/abs/2609.00173)
- **[IN WINDOW publication, 2026-07-06; arXiv:2602.23743] Kini, Mauviard, Salmi, Watts et al., "A NICER View of PSR J0030+0451: Updated Constraints from 6 yr of NICER Observations", ApJ 1005, 201, DOI 10.3847/1538-4357/ae733e.** Uses ~50% more counts. Preferred PDT-U model: **M = 1.43 (+0.20/−0.17) M☉, R_eq = 12.68 (+1.31/−1.04) km**, down from 14.44 km and 1.70 M☉ in the 2024 analysis. ST+PDT model: M = 1.29 (+0.13/−0.12) M☉, R = 11.63 (+0.95/−0.89) km. Spin period 4.87 ms. This "reduces the tension" with J0437, J0614 and GW170817. — [arXiv abstract](https://arxiv.org/abs/2602.23743); [arXiv HTML](https://arxiv.org/html/2602.23743); [IOPscience](https://google.iopscience.iop.org/article/10.3847/1538-4357/ae733e)
- **[BACKGROUND, 2026-03-27] Miller et al., "The Radius of PSR J0437−4715 from NICER Data", ApJL 1000 L48.** Including the modulated non-thermal component gives a 68% radius range of **11.9–15.5 km at M = 1.418 ± 0.044 M☉**. — [IOPscience](https://iopscience.iop.org/article/10.3847/2041-8213/ae5057)
- **[IN WINDOW, 2026-09-28, review] "What We Have Learned from NICER About Neutron Star Radii", arXiv:2609.36169.** Argues that the systematic errors explored so far "have at most a minor effect" on inferred NICER radii. — [arXiv HTML](https://arxiv.org/html/2609.36169)
- **[IN WINDOW, 2026-09-01] A&A, "NICER constraints on low density matching and high density continuation…".** NICER posteriors favour a χEFT matching density n_m ≃ 0.18 fm⁻³ and width Δ_m ≃ 0.021 fm⁻³. — [A&A](https://www.aanda.org/articles/aa/full_html/2026/09/aa59810-26/aa59810-26.html)
- **[BACKGROUND] Systematics study (arXiv:2507.12540).** Adding PSR J0614−3329 lowers R_1.4 by ~100 m. Bayes factor log₁₀BF ≈ 1.58 for ST+PDT over PDT-U in a multi-messenger fit. — [arXiv HTML](https://arxiv.org/html/2507.12540)

**FRBs**
- **[IN WINDOW, 2026-09-23, preprint] CHIME/FRB Outriggers dwarf hosts, arXiv:2609.28629.** Eight hosts, five newly classed as dwarfs (log M* ≈ 8.1–9.1). Repeaters prefer metal-poor dwarfs, 12+log(O/H) ≲ 8.35. Repeaters with persistent radio sources sit at ≲8.15. Non-repeaters are at ≳8.35, a statistically distinct distribution. — [arXiv](https://arxiv.org/abs/2609.28629)
- **[BACKGROUND, 2026-03] Second CHIME/FRB Catalog, ApJS.** 4539 FRBs from 3641 sources, including 981 bursts from 83 repeaters (2018-07-25 to 2023-09-15), 400–800 MHz at 0.983 ms. — [IOPscience](https://beta.iopscience.iop.org/article/10.3847/1538-4365/ae3828)
- **[BACKGROUND, 2026-05] 30 new CHIME repeaters, arXiv:2605.08410.** Only **2.4 ± 0.4%** of sources repeat. Extragalactic DMs span 99.4–1446.0 pc cm⁻³. Four repeaters show linear DM drifts over years. — [arXiv PDF](https://arxiv.org/pdf/2605.08410)
- **[BACKGROUND, 2026-02] FRB 20241228A, ApJ.** Host at z = 0.1614 ± 0.0002. Persistent X-ray limit < 1.2×10⁴³ erg/s (0.3–10 keV); scattering ~1 ms and scintillation ~15 kHz at 600 MHz. **Hook:** Macquart DM–z relation at z = 0.1614. — [IOPscience](https://iopscience.iop.org/article/10.3847/1538-4357/ae2ea0)
- **[IN WINDOW, 2026-09-04, preprint] FRB–BNS merger rate comparison, arXiv:2609.05209.** A population-synthesis model; no new measurement. — [arXiv Science](https://arxiv.science/abs/2609.05209)

**Magnetars / kilonovae**
- **[IN WINDOW, 2026-07-24] Chatterjee, Philippov, Beloborodov, Parfrey & Ripperda, ApJL 1006 L45, DOI 10.3847/2041-8213/ae8997.** The first relativistic-MHD simulation of a shear-driven magnetar giant flare. It ejects a plasmoid carrying up to **~9% of magnetospheric magnetic energy**, which drives a blast wave relevant to FRBs. Context from the paper: three known giant flares (1979, 1998, 2004). Initial spikes ~0.2–0.5 s with 10⁴⁴–10⁴⁵ erg (10⁴⁶ erg for SGR 1806−20); pulsating tails ~200–400 s with ~10⁴⁴ erg. **Hook:** magnetospheric energy E ~ B²R³/12 for a dipole, as a fraction released. — [IOPscience](https://iopscience.iop.org/article/10.3847/2041-8213/ae8997)
- **[BACKGROUND, 2026-04-08] Bransgrove et al., ApJL 1001 L13.** Magnetars with internal B ≳ 10¹⁶ G erupt through ambipolar diffusion on ~10⁹ s timescales. SGR 1806−20 (2004) had L > 10⁴⁷ erg/s for ~0.5 s, P = 7.56 s, ejecta 10²⁵–10²⁷ g at ~c/2, and dipole moment μ ≈ 2×10³³ G cm³. — [IOPscience](https://google.iopscience.iop.org/article/10.3847/2041-8213/ae50f2/meta)
- **[IN WINDOW, 2026-07-27 arXiv:2607.24531; A&A 2026-09] Zhong et al., NS–WD mergers with magnetar giant flares.** The proposed mechanism for r-process kilonovae in GRB 211211A and 230307A (theory). Per-spike ejecta 10⁻⁸–10⁻⁶ M☉ at 0.1–0.4c; total M_ej ≳ 10⁻⁵–10⁻³ M☉; post-merger B_s ≈ 4–7×10¹⁴ G. — [A&A](https://www.aanda.org/articles/aa/full_html/2026/09/aa60826-26/aa60826-26.html); [arXiv Science](https://arxiv.science/abs/2607.24531)
- **[BACKGROUND, 2025] Patel et al., ApJ 985 234.** Giant-flare r-process yield ≲8×10⁻⁷ M☉ per flare, 1–10% of Galactic r-process. The "nova brevis" transient peaks at L ~ 10³⁹–10⁴⁰ erg/s about 10–15 min after the flare. — [IOPscience](https://iopscience.iop.org/article/10.3847/1538-4357/adceb7)

### Inferences
- **Compactness (my computation, GM☉/c² = 1.4766 km).** C = GM/Rc² is 0.284 for J1614−2230 (1.937 M☉, 10.06 km). It is 0.166 for J0030 PDT-U (1.43 M☉, 12.68 km) and 0.164 for J0030 ST+PDT (1.29 M☉, 11.63 km). It spans 0.135–0.176 for J0437 (1.418 M☉, 11.9–15.5 km). J1614's C ≈ 0.28 is near the regime where many stiff hadronic EOS fail, and its 90th-percentile R ≲ 12.1 km at 1.94 M☉ is a sharp TOV test.
- A test set for a TOV + EOS module: {J1614: 1.937 M☉ / 10.1 km; J0030: 1.43 / 12.7 (or 1.29 / 11.6); J0437: 1.418 / 11.9–15.5}, plus the ~0.4 km downshift of heavy-star radii from 2609.00173.
- The FRB metallicity split (≈8.35) is a clean empirical threshold. It would favour repeaters coming from young, metal-poor magnetar-forming environments, though it is only a population trend so far.

### Gaps
- Author list and final status of arXiv:2609.00172 (J1614 NICER) were not retrieved.
- No new observed kilonova or new magnetar giant flare in the window was found; in-window work is simulation and theory.
- No NICER results for PSR J2124−3358 or 47 Tuc X7 themselves were retrieved; they appear only as inputs to 2609.00173.
- The numerical maximum-mass posteriors in the ApJ July 2026 population paper (DOI 10.3847/1538-4357/ae75e4) were stripped from the retrieved abstract.

## Q3. Rubin first-year science, Gaia DR4, Roman, SPHEREx, Euclid data releases, and interstellar object 3I/ATLAS

### Takeaway
In-window milestones were operational rather than science-heavy. Rubin formally started LSST (2026-06-29/30) and released Early Data Preview 2 at the end of July: 790 M objects and ~300 k Solar System objects. Roman launched on 2026-08-30 and its Wide Field Instrument took first photons in September. Gaia DR4 (2026-12-02), Euclid DR1-Foundation (2026-11-12) and SPHEREx's first annual release (Nov 2026) all fall after the window. Several 3I/ATLAS composition papers appeared in the window: a CO-dominated coma post-perihelion, NH₂ depletion, and metals tracking CO. Its hyperbolic orbit (e ≈ 6.14, v∞ ≈ 57.94 km/s) is an exact orbital-mechanics benchmark.

### Cited Findings
**Rubin / LSST**
- **[IN WINDOW, 2026-06-30] NOIRLab release noirlab2616: the 10-year LSST "has officially started".** Each point in the sky will be revisited ~800 times over the decade. During early optimization surveys Rubin found >11,000 new asteroids, including 33 NEOs and 380 TNOs, in about 1.5 months. — [NOIRLab](https://noirlab.edu/public/news/noirlab2616/)
- **[IN WINDOW, released 2026-07-27/31] Rubin Early Data Preview 2 (EDP2), RTN-115 (Rubin Observatory Team, dated 2026-07-30).**
  - Data: 29,756 science exposures over 158 nights; formal LSST start was 2026-06-29.
  - Coverage: ~3,000 deg² of coadds and ~15,000 deg² of total footprint.
  - Content: **790 million objects, 299,819 Solar System objects**; 59.15 TB.
  - Image quality: median PSF FWHM 1.17″ (z) to 1.26″ (g, r).
  - COSMOS 5σ coadd depths: u = 26, g = 26.8, r = 26.3, i = 26.1, z = 25.3, y = 23.9.
  - Known systematics: a mis-calibrated focus (corrected December 2025) and a chromatic PSF.
  - The full DP2 release is expected Oct–Dec 2026.

  — [RTN-115 PDF](https://rtn-115.lsst.io/RTN-115.pdf); [Rubin news](https://rubinobservatory.org/news/rubin-new-window-cosmos-field); [dp2.lsst.io](https://dp2.lsst.io/index.html)
- **[BACKGROUND, 2026-05-22] Rubin DP1 paper, AJ 171 360.** 1792 exposures over 48 nights, ~15 deg², median PSF 1.14″, ~2.3 M objects, 431 Solar System objects (93 new). — [arXiv HTML](https://arxiv.org/html/2603.23786); [IOPscience](https://google.iopscience.iop.org/article/10.3847/1538-3881/ae521f)
- **[BACKGROUND, 2026-01-07] Fastest-spinning >500 m asteroid, 2025 MN45: D = 710 m, P = 1.88 min.** Part of 19 super- and ultra-fast rotators. Others: 2025 MJ71 (1.9 min), MK41 (3.8 min), MV71 (13 min), MG56 (16 min). Paper DOI 10.3847/2041-8213/ae2a30. **Hook:** spin-barrier physics, centrifugal acceleration ω²R vs self-gravity (4/3)πGρR, and the cohesion needed. — [Rubin news](https://rubinobservatory.org/news/rubin-record-breaking-asteroid-pre-survey)
- **[BACKGROUND] Rubin alert system.** 800,000 alerts on the night of 2026-02-24, scaling to ~7 million alerts per night. — [Rubin news](https://rubinobservatory.org/news/first-alerts)
- **[BACKGROUND] Rubin and 3I/ATLAS.** Rubin's initial data had recorded 3I/ATLAS 10 days before the ATLAS discovery. Expected Rubin interstellar-object yield is "five to 500". — [Quanta, 2026-05-15](https://www.quantamagazine.org/rubin-tracks-skyscraper-size-asteroids-failed-supernovas-and-interstellar-visitors-20260515/)

**Roman, Gaia, Euclid, SPHEREx**
- **[IN WINDOW, 2026-08-30] Roman launched** at 07:26 EDT on a Falcon Heavy from LC-39A. It is on a ~3-month cruise to L2 with a 5-yr planned mission. — [NASA](https://www.nasa.gov/news-release/nasas-dark-universe-seeking-nancy-grace-roman-space-telescope-launches/); [Reuters](https://www.reuters.com/science/nasa-launches-powerful-new-roman-space-telescope-florida-2026-08-30/)
- **[IN WINDOW, 2026-09-15 and 09-30] Roman commissioning.** The WFI (300 Mpx, 18 detectors cooled to about −183 °C) was activated. Fine-guidance tests ran Sept 15–21 and the Coronagraph saw first light on Sept 22. First science images are expected early 2027. — [NASA Roman blog 09-15](https://science.nasa.gov/blogs/roman/2026/09/15/nasa-activates-romans-primary-instrument-checks-out-coronagraph/); [blog 09-30](https://science.nasa.gov/blogs/roman/2026/09/30/nasa-checks-roman-guidance-system-takes-first-coronagraph-observation/)
- **[NOT YET RELEASED] Gaia DR4** is scheduled for **2 Dec 2026, ~12:00 CET**. It is based on 66 months of data (2014-07-25 to 2020-01-20, vs 34 months for DR3). An astrometric time-series pre-release for a small sample came out on 29 June. — [ESA Gaia DR4](https://www.cosmos.esa.int/web/gaia/data-release-4); [Gaia DR4 content](https://www.cosmos.esa.int/web/gaia/dr4)
- **[NOT YET RELEASED] Euclid DR1-Foundation** is scheduled for **12 Nov 2026** (~1900 deg² of raw and calibrated images, catalogues and spectra), with the full DR1 in mid-2027. — [Euclid timeline](https://www.cosmos.esa.int/web/euclid/timeline); [DR1 timeline note](https://www.cosmos.esa.int/web/euclid/dr1-timeline)
- **[IN WINDOW, Sept 2026] SPHEREx Quick Release 3 (QR3)** began. The first annual release (102-channel all-sky cubes) is planned for Nov 2026, and the High-Reliability Source Catalog for Aug 2027. The first all-sky map was completed in Dec 2025. — [IRSA SPHEREx](https://irsa.ipac.caltech.edu/Missions/spherex.html); [SPHEREx data products](https://spherex.caltech.edu/page/data-products); [JPL Dec 2025](https://www.jpl.nasa.gov/news/nasas-spherex-observatory-completes-first-cosmic-map-like-no-other/)

**3I/ATLAS (C/2025 N1)**
- **[BACKGROUND, orbit] Seligman et al., ApJL 2025 (arXiv:2507.02757).**
  - Heliocentric orbit: q = 1.3558541 ± 0.0019855 au, e = 6.1329072 ± 0.0162018, i = 175.11239°, T_p = MJD 60977.51085 (2025-10-29).
  - Barycentric incoming orbit: e_b = 6.144 ± 0.016, q_b = 1.3611 au, **v∞ = 57.942 ± 0.049 km/s**.
  - The incoming asymptote points to RA ~295°, Dec ~−19°.

  **Hook:** v∞ = √(GM☉(e−1)/q). — [arXiv HTML](https://arxiv.org/html/2507.02757v3); [IOPscience](https://iopscience.iop.org/article/10.3847/2041-8213/adf49a/meta)
- **[BACKGROUND] Kinematic age.** 3–11 Gyr ([Taylor & Seligman, ApJL 2025](https://iopscience.iop.org/article/10.3847/2041-8213/adfa28)), or 7.6–14 Gyr at 68% (likely >7.6 Gyr) in the Ōtautahi–Oxford model ([arXiv:2507.05318](https://arxiv.org/html/2507.05318v3)). Galactic velocity (U, V, W) = (−51.1, −19.3, 18.9) km/s.
- **[BACKGROUND, 2026-03-06] Hui et al., ApJL 999 L37.** HST nucleus detection: p_V × cross-section = 0.22 ± 0.07 km² → **R_eff = 1.3 ± 0.2 km** at p_V = 0.04. Axis ratio ≥2:1 if the variation is rotational. Outbound activity index 4.5 ± 0.3; opposition surge ~0.2 mag, width 3° ± 1°. — [IOPscience](https://iopscience.iop.org/article/10.3847/2041-8213/ae471c)
- **[IN WINDOW, 2026-06-24] Roth et al., "Coma Physics of an Interstellar Object: JWST Spatial–Spectral Mapping of 3I/ATLAS", ApJL 1005 L5, DOI 10.3847/2041-8213/ae7443.** Observed at r_H = 2.37–2.41 au (2025-12-22/23). **CO is the most abundant volatile.** H₂O/CO = (40.5 ± 3.1)% and CO₂/CO = (41.6 ± 0.3)%. H₂O ortho-to-para ratio = 2.7 ± 0.2, below the equilibrium value of 3. **Hook:** invert the H₂O OPR(T) relation (from rotational partition functions of the ortho/para ladders) to get a nuclear-spin temperature; the source gives only OPR = 2.7 ± 0.2, so the T_spin value has to be computed. — [IOPscience](https://google.iopscience.iop.org/article/10.3847/2041-8213/ae7443)
- **[IN WINDOW, 2026-06-12] Zhao et al., ApJL 1004 L24, DOI 10.3847/2041-8213/ae73fe.** Post-perihelion optical spectroscopy of CN, C₃, C₂, CH, Fe I and Ni I at 1.8–3.3 au. The coma is "metal rich relative to its H₂O" but follows the metal–CO correlation, suggesting metal carbonyls. There is an unexplained [O I]6300 residual. — [IOPscience](https://iopscience.iop.org/article/10.3847/2041-8213/ae73fe)
- **[IN WINDOW, 2026-06-12] Medler et al., AJ 172 25, DOI 10.3847/1538-3881/ae6b71.** Optical slope 21–27%/100 nm (0.4–0.55 µm) and 6–10% (0.65–0.9 µm). NIR slope −0.9% (0.9–1.5 µm) and −2.3% (1.9–2.5 µm), with a turnover at ~1.1 µm. No water-ice features; perihelion 2025-10-29 at 1.356 au. — [IOPscience](https://iopscience.iop.org/article/10.3847/1538-3881/ae6b71)
- **[IN WINDOW, 2026-06, preprint] arXiv:2606.18751, dust modelling from 4 au inbound to 4 au outbound.** Size index κ ≈ −3.5, r_min = 10 µm, r_max = 1–10 cm. Maximum dust loss at perihelion (0.5–1.8)×10⁴ kg/s. Opposition surge 0.1–0.4 mag over a 1–3° width; linear phase coefficient 0.02–0.04 mag/deg. — [arXiv HTML](https://arxiv.org/html/2606.18751)
- **[IN WINDOW, 2026-09-04, preprint] Gray et al., arXiv:2609.04932 (VLT/NOT polarimetry).** Minimum polarization **−2.9% at phase angle 5.5°**, unchanged across perihelion. — [arXiv](https://arxiv.org/abs/2609.04932)
- **[date not retrieved] JUICE/MAJIS, ApJL DOI 10.3847/2041-8213/ae8a47.** Observations 2025-11-02..25 at 1.36–1.68 au: Q(H₂O) fell from 8×10²⁸ to 4×10²⁸ s⁻¹ and CO₂/H₂O ≈ 10%. 3.2–3.6 µm aliphatic C–H features are tentative. — [IOPscience PDF](https://beta.iopscience.iop.org/article/10.3847/2041-8213/ae8a47/pdf)
- **[BACKGROUND, 2026-03-31] Kawakita et al., ApJL 1000 L60.** NH₂ clearly depleted post-perihelion. CN, C₃ and C₂ relative to H₂O are normal for comets. Dust slope 20–23%/100 nm vs ~10% typical. — [IOPscience](https://iopscience.iop.org/article/10.3847/2041-8213/ae4c5b)

### Inferences
- **3I/ATLAS v∞ check (my computation, GM☉ = 1.32712440018×10²⁰ m³/s², 1 au = 1.495978707×10¹¹ m).** e_b = 6.144 with q_b = 1.3611 au gives v∞ = 57.90 km/s; e = 6.1329 with q = 1.35585 au gives 57.95 km/s. Both are within ~0.05 km/s of the published 57.942 ± 0.049. This makes a good exact-orbit unit test.
- **2025 MN45 (my order-of-magnitude estimate, assuming ρ ≈ 2500 kg/m³).** Centrifugal acceleration at the equator ω²R ≈ (2π/112.8 s)²×355 m ≈ 1.1 m/s², vs self-gravity (4/3)πGρR ≈ 2.5×10⁻⁴ m/s². That is a ratio of ~4000, so the body must be monolithic or cohesive, not a rubble pile. The rubble-pile spin barrier is ~2.2 h.
- The in-window survey "results" are mostly data-volume and depth benchmarks, not physics claims. Gaia DR4 and Euclid DR1 will land after the window and cannot be used yet.

### Gaps
- No in-window peer-reviewed Rubin LSST science result with physics content beyond EDP2 statistics was found.
- No Roman science data yet; first images are expected early 2027.
- No in-window 3I/ATLAS non-gravitational acceleration value was retrieved.

## Q4. Solar and stellar physics: solar-cycle-25 peak, Parker Solar Probe and Solar Orbiter results, stellar nucleosynthesis, the lithium problem, oldest-star ages vs cosmic age

### Takeaway
SC25 peaked in Oct 2024 at a smoothed sunspot number of ≈160.9–161. That is ~40% above the 2019 panel forecast (115) but 10% below an average cycle. The best in-window heliophysics result is Solar Orbiter's heavy-ion evidence that magnetic switchbacks come from closed coronal loops via interchange reconnection (Nature Astronomy, 2026-10-08). On the age question, a July 2026 analysis of ~150,000 old LAMOST/Gaia stars finds the oldest star at 13.73–13.80 (+0.2/−0.16, ±0.4 sys) Gyr. That is consistent with ΛCDM's 13.6 Gyr expectation and disfavours pre-recombination Hubble-tension fixes. The lithium problem is unresolved, with ~3–4σ between SBBN and the Spite plateau.

### Cited Findings
- **[BACKGROUND, confirmed] SC25 maximum.**
  - SILSO 13-month smoothed SN maximum **160.9 (≈161) in Oct 2024**. Neighbouring months: 159 (Sep) and 157 (Nov).
  - Monthly maximum 216.0 in Aug 2024, the highest since Sep 2001. Daily maximum 290 on 2024-07-18.
  - Comparisons: an earlier SC25 smoothed local maximum of 125.3 (June 2023); SC24 max 116.4; SC23 max 180.3.
  - LYRA/GOES smoothed irradiance peaked on 2024-10-01.

  — [SIDC](https://www.sidc.be/article/solar-cycle-25-reached-its-maximum-october-2024); [STCE SC25 tracking](https://www.stce.be/content/sc25-tracking)

  Ranking: 15th of 25 cycles, 10% below the average cycle amplitude of 178.7 — [Gallego et al., RNAAS](https://iopscience.iop.org/article/10.3847/2515-5172/adf326). The 2019 panel had predicted 115 ± 10 in July 2025 — [NOAA SWPC](https://www.spaceweather.gov/products/solar-cycle-progression).
- **[IN WINDOW, 2026-10-08] Coburn, Yardley, Ngampoopun, Dewey, Rivera et al., "On the coronal origin of magnetic switchbacks in the solar wind", Nature Astronomy, DOI 10.1038/s41550-026-02928-0.** Solar Orbiter at 0.55 AU saw elevated O⁷⁺/O⁶⁺ and C⁶⁺/C⁴⁺ ratios and He depletion inside a switchback, pointing to hot closed loops. Coronal loop lengths of ~7.97–12.4 Mm were traced (13, 11 and 18 arcsec). The mechanism is interchange reconnection, with the switchback propagating as an outward Alfvén wave. **Hook:** frozen-in charge-state ratios → coronal electron temperature (ionization equilibrium). — [Nature Astronomy](https://www.nature.com/articles/s41550-026-02928-0)
- **[IN WINDOW, 2026-06, preprint] PSP switchback wave activity, arXiv:2606.28614.** Encounters E01 and E17–21 at 11–50 R☉. Transverse power δB⊥ is enhanced inside switchbacks. Spectral indices: B∥ ≈ −5/3 and B⊥ ≈ −3/2. The |B| index is −1 at small deflection, steepening to ~−3/2. — [arXiv HTML](https://arxiv.org/html/2606.28614)
- **[IN WINDOW, 2026-09-17, preprint] arXiv:2609.20760.** Two closely spaced PSP switchbacks (2020-01-24) with different microphysics: one shows higher T_p, steeper spectrum and a transient β > 1. — [arXiv](https://arxiv.org/abs/2609.20760)
- **[BACKGROUND, 2026] PSP switchback statistics.** Switchbacks occur "almost exclusively" in super-Alfvénic wind, and radial velocity fluctuations saturate at M_A ~ 4–6 (Kelvin–Helmholtz interpretation) — [Goodwill et al., NSF PAR](https://par.nsf.gov/biblio/10696483-parker-solar-probe-analysis-across-alfvenic-transition-velocity-shear-magnetic-deflection-switchback-formation). Their dependence on solar activity is weak; PSP's closest perihelion is ~10 R☉ — [Sneha et al., A&A](https://www.aanda.org/articles/aa/pdf/forth/aa61400-26.pdf).

**Oldest stars vs age of the universe**
- **[IN WINDOW, 2026-07 (v1); revised version listed 2026-10-08] "The age of the Universe from a large sample of the oldest Galactic stars", arXiv:2607.00764.** Starts from the Xiang & Rix LAMOST DR7 + Gaia eDR3 sample (247,103 stars), cut to metal-poor, α-rich stars within 5 kpc. The two versions differ:
  - v1: 155,600 stars, **A★ = 13.73 (+0.18/−0.15) Gyr**, range 13.31–14.02 Gyr under different cuts.
  - Revised: 146,810 stars, **A★ = 13.80 (+0.20/−0.16) Gyr ± 0.4 Gyr systematic**.

  Allowing 0.2 Gyr for first-star formation, this is consistent with the 13.6 Gyr ΛCDM expectation, which "casts doubt on solutions to the Hubble tension solely through new physics prior to recombination". **Hook:** t₀(H0, Ωm) vs A★ + 0.2 Gyr. A high-H0 (≈73) ΛCDM universe is ~12.7 Gyr old and conflicts with these stars. — [arXiv HTML v1](https://arxiv.org/html/2607.00764v1); [arXiv Science (revised)](https://arxiv.science/abs/2607.00764)
- **[BACKGROUND, 2026-03] Globular-cluster ages IV, arXiv:2603.04872 (69 HST GCs, multiple populations).** t_GC = 13.61 ± 0.25 (stat) ± 0.23 (sys) Gyr → t_U = 13.81 ± 0.25 ± 0.23 Gyr. The earlier paper III gave t_GC = 13.39 ± 0.10 ± 0.23 and t_U ≤ 13.92 Gyr (95%) ([JCAP 2025](https://iopscience.iop.org/article/10.1088/1475-7516/2025/10/030/meta)). — [arXiv HTML](https://arxiv.org/html/2603.04872v1)
- **[BACKGROUND, 2026-03] A&A, 160 Gaia DR3 MSTO/subgiant stars.** Age peak 13.6 ± 1.0 (stat) ± 1.4 (sys) Gyr → t_U ≥ 13.8 Gyr (formation at z_f = 20). — [A&A](https://www.aanda.org/articles/aa/full_html/2026/03/aa57038-25/aa57038-25.html)
- Claims of globular clusters aged 15–21 Gyr from blue-straggler methods (a "26.7 Gyr universe") come from a low-impact venue ([Science Publishing Group 2024](https://sciencepublishinggroup.com/article/10.11648/j.ajaa.20241101.11)). They contradict all isochrone-based results above and should be treated as unreliable.

**Lithium problem / BBN**
- **[BACKGROUND, 2026-04] Bipartite-decay paper, arXiv:2604.01324.** SBBN gives **⁷Li/H = (5.464 ± 0.220)×10⁻¹⁰**, against an observed (1.45 ± 0.25)×10⁻¹⁰ (~4σ). D/H = (25.08 ± 0.29)×10⁻⁶ and Y_p = 0.245 ± 0.003. Photon-injection fixes need E_γ between 1.59 MeV (the ⁷Be photodissociation threshold) and 2.22 MeV (the D threshold). — [arXiv HTML](https://arxiv.org/html/2604.01324v1)
- **[BACKGROUND, 2026-03] BBN sensitivity atlas, arXiv:2603.22414 (PRyMordial; 14 parameters, 63 rates; new LBT Y_p halves the error).** The lithium problem persists at ~3–4σ, with "no nuclear fix… at 1σ". It would need several jointly >3σ rate shifts. — [Emergent Mind summary](https://api.emergentmind.com/papers/2603.22414)
- Older comparison values: (⁷Li/H)_BBN+CMB = (4.94 ± 0.72)×10⁻¹⁰ vs Spite plateau (1.6 ± 0.3)×10⁻¹⁰ ([Miranda, A&A 2025](https://www.aanda.org/articles/aa/full_html/2025/09/aa54482-25/aa54482-25.html)).
- **[IN WINDOW, 2026-08-18; low-authority venue, a high-school journal] GALAH DR4 Spite-plateau fit.** A(Li) ≈ 1.90–2.17, a flat slope, vs BBN A(Li) ≈ 2.72, a gap of 0.55–0.82 dex. Use only as corroboration. — [NHSJS](https://nhsjs.com/2026/investigating-the-cosmological-lithium-problem-the-spite-plateau-in-warm-metal-poor-dwarf-stars-with-gaia-dr3-galah-dr4-and-bayesian-regression/)

**Stellar nucleosynthesis (cross-reference)**
- In-window nucleosynthesis-relevant items appear under Q1 and Q2:
  - GN-z11 very massive stars and N enrichment (arXiv:2608.12699).
  - The [C/O] ≈ −0.4 of GS-z14-0 (ApJL 1004 L30).
  - Pop III top-heavy IMF constraints (Hebe).
  - Giant-flare r-process (Zhong et al.).

### Inferences
- **Age benchmark.** My flat-ΛCDM t₀ = 13.786 Gyr (Planck-like parameters) minus a ~0.2 Gyr formation delay gives ~13.6 Gyr. That is 0.1–0.2 Gyr below the 13.73–13.80 Gyr central values of the oldest stars, but within the ±0.4 Gyr systematics. For a σ-field or alternative cosmology module, the oldest-star ages are a hard floor on t₀. Any model with t₀ < ~13.3 Gyr (≈1σ low end) should be flagged.
- **⁷Li/H.** SBBN/observed ≈ 3.8 (5.464/1.45). A BBN network module could reproduce the ~5.5×10⁻¹⁰ SBBN number at η₁₀ ≈ 6.1 as a regression test.
- SC25 numbers are settled benchmarks rather than new discoveries: smoothed SN 160.9 in Oct 2024, monthly SN 216.0 in Aug 2024.

### Gaps
- No in-window PSP perihelion-specific headline numbers were retrieved. The record perihelion (≈6.1 million km, ≈10 R☉, Dec 2024) is background and was not re-sourced here.
- No in-window primary-literature lithium measurement was found, other than the low-authority GALAH note.
- No new stellar-nucleosynthesis lab result (e.g., a ¹²C(α,γ) rate) was searched for.

## Q5. Cosmic rays and high-energy astrophysics: UHE events, KM3NeT ~220 PeV neutrino, LHAASO PeVatrons, the knee

### Takeaway
No new Amaterasu-class (>100 EeV) event or new UHE neutrino was found in the window. The in-window literature is interpretation of KM3-230213A and of LHAASO's proton knee: a review, Lorentz-invariance-violation tests, and a superheavy-dark-matter fit to the knee that its own photon constraints rule out. The benchmark numbers remain:
- KM3-230213A: E_ν ≈ 220 PeV (72 PeV–2.6 EeV at 90% CL), muon 120 PeV.
- LHAASO proton knee: 3.3 ± 0.6 PeV, with Δγ ≈ −1.
- Auger dipole above 8 EeV: 6.8σ.
- Amaterasu: ~244 EeV.

### Cited Findings
- **[BACKGROUND event 2023-02-13; 2026 analyses] KM3-230213A (KM3NeT/ARCA, 21-line configuration).**
  - Energy: muon 120 (+110/−60) PeV; neutrino ≈220 (+570/−110) PeV, assuming E⁻².
  - Direction: RA = 94.3°, Dec = −7.8°, R68 = 1.5°, R99 = 3.0°; l = 216.1°, b = −11.1°.
  - Atmospheric background: 1–5×10⁻⁵ events/yr above 100 PeV.
  - A Galactic origin is disfavoured.

  — [KM3NeT proceedings arXiv:2605.07641](https://arxiv.org/html/2605.07641); [KM3NeT, ApJ 1003 157 (2026-05-22)](https://iopscience.iop.org/article/10.3847/1538-4357/ade3b0)
- **Tension with IceCube and Auger.** The probability of one KM3NeT event and none in IceCube/Auger is ~0.4% for a steady diffuse flux, and the global joint fit gives p = 0.5% (2.6σ). The diffuse cosmogenic scenario is in ≳3σ tension, reducible to ~2.3–2.4σ for a flaring source (MRC 0614−083 accretion-flare model). — [Alhebsi et al., ApJ 1003 235](https://iopscience.iop.org/article/10.3847/1538-4357/ae6452); [JETP Letters 2026](https://link.springer.com/article/10.1134/S0021364025610061); [ApJL accretion flare](https://iopscience.iop.org/article/10.3847/2041-8213/ae3a73)
- **[BACKGROUND, 2026-05-28] Alhebsi et al.** The UHECR proton fraction is ~20% at 20 EeV. A cosmogenic interpretation that uses only KM3NeT requires strongly evolving sources such as high-luminosity AGN. — [IOPscience](https://iopscience.iop.org/article/10.3847/1538-4357/ae6452)
- **[BACKGROUND] KM3NeT GRB population study.** Baryon loading ≤ 392 (90%) for ISM density (A&A 710 A168, 2026). The blazar-population best fit has baryon loading ~10 and proton index ~1.8. — [arXiv:2605.07641](https://arxiv.org/html/2605.07641)
- **[IN WINDOW, 2026-09-04] Review, "KM3-230213A: The Highest-Energy Cosmic Neutrino".** No unambiguous electromagnetic counterpart. Tension with other detectors persists, and beyond-Standard-Model scenarios remain open. — [SciLTP PAC](https://www.sciltp.com/journals/pac/articles/2609005071)
- **[IN WINDOW, 2026-09-11] Sasse, Lang & Anjos, MNRAS, DOI 10.1093/mnras/stag1733.** Uses KM3-230213A to probe Lorentz-invariance violation in cosmogenic neutrino propagation. The bounds themselves were not retrieved. — [MNRAS](https://academic.oup.com/mnras/advance-article/doi/10.1093/mnras/stag1733/8790751)
- **[BACKGROUND, published 2025-12-30] LHAASO proton spectrum across the knee, Science Bulletin 70, 4173, DOI 10.1016/j.scib.2025.10.048 (arXiv:2505.14447).**
  - Measured from 0.15 to 12 PeV.
  - Spectral index ≈ −2.51 in 0.3–1.5 PeV, with hardening Δγ ~ +0.2 near ~0.1 PeV.
  - **Proton knee E_k = 3.3 PeV (R_knee = 3.3 ± 0.6 PV), Δγ = δ_knee = 0.99 ± 0.29.**
  - The all-particle knee is at 3.67 PeV.
  - Context: 43 UHE γ-ray sources, with photons up to 2.5 PeV.
  - Comparison points: DAMPE softening at 13.6 (+4.1/−1.8) TeV (index −2.85 ± 0.07), CALET at 9.3 TeV, ISS-CREAM at 9.0 ± 1.3 TeV.

  **Hook:** broken-power-law fit and rigidity scaling of the knee for heavier nuclei (E_knee,Z ≈ Z × 3.3 PeV; iron knee ~80 PeV per KASCADE-Grande). — [arXiv HTML](https://arxiv.org/html/2505.14447); [CUHK record](https://research.cuhk.edu.hk/en/publications/precise-measurements-of-the-cosmic-ray-proton-energy-spectrum-in-/)
- **[BACKGROUND, 2026-02-16] Yue et al., ApJ 998 264.** A joint spectrum-plus-anisotropy fit rejects a propagation origin for the knee at 95% CL, using anisotropy data from 2.7 PeV to 2.48 EeV. This favours acceleration-limit origins. — [IOPscience](https://iopscience.iop.org/article/10.3847/1538-4357/ae3d2d)
- **[IN WINDOW, 2026-09-15, preprint] Akl, Iovino, Kantzas & Macciò, arXiv:2609.17315.** A 5×10⁷ GeV superheavy relic decaying to nucleons reproduces both the 0.34 PeV hardening and the 3.3 PeV knee. Its own photon cascade rules this out by ~2 orders of magnitude against the CR photon fraction and ~1 order against LHAASO's diffuse emission. — [arXiv](https://arxiv.org/abs/2609.17315)
- **[BACKGROUND] UHECR anisotropy.**
  - Auger dipole above 8 EeV at 6.8σ, pointing ~115° from the Galactic centre, from 123,000 km² sr yr and ~50,000 events.
  - Auger+TA full-sky analysis: dipole 4.6σ in the angular power spectrum and starburst-galaxy correlation 4.4σ.
  - The combined analysis also reports a hint of a quadrupole along the supergalactic plane.

  — [Auger ApJ 2024](https://iopscience.iop.org/article/10.3847/1538-4357/ad843b); [Auger+TA PoS UHECR2024](https://pos.sissa.it/484/009/pdf); [arXiv:2507.10441](https://arxiv.org/html/2507.10441)
- **[BACKGROUND] Amaterasu (TA, 2021-05-27).** ~244 EeV with 29 EeV statistical resolution. The arrival direction lies near the Local Void and no source is identified. Average propagation distance at 244 EeV is ~30 Mpc. 28 TA events above 100 EeV are isotropic. — [Science 2023](https://www.science.org/doi/10.1126/science.abo5095)

### Inferences
- **Engine hooks.**
  - Neutrino energy from muon energy: E_ν ≈ 1.8 × E_μ here (220/120).
  - GZK horizon: tens of Mpc at >200 EeV, consistent with ~30 Mpc for Amaterasu.
  - Rigidity-dependent knee: Peters cycle, E_knee(Z) = Z × R_knee.
  - Head-on p+γ_CMB → Δ⁺/pion threshold: E_p ≈ m_π(m_p + m_π/2)/(2E_γ) ≈ 6.8×10¹⁹ eV for E_γ ≈ 1 meV.
- The biennial ICRC (2025, 2027) did not fall in this window. That likely explains the lack of new experimental CR data releases (my inference).

### Gaps
- No new >100 EeV UHECR event or new UHE neutrino event was found for June–Oct 2026.
- No new LHAASO PeVatron/microquasar paper in the window was retrieved. The LHAASO microquasar PeVatron results are from 2024–2025 and were not re-sourced here.
- The LIV bound values in the MNRAS Sept 2026 paper were not retrieved.

## Q6. Other physics headlines with clean numbers: G, clocks, nuclear clocks, antimatter, superconductivity, new phases of matter, plasma/fusion

### Takeaway
The window's best "clean-number" headlines:
- **Two Nature papers (2026-10-07) demonstrating operating ²²⁹Th nuclear clocks.** The transition is near 2.020407 PHz (148.38 nm, 8.356 eV), with instabilities of 5×10⁻¹³ to 3×10⁻¹²/√τ.
- **¹⁷⁶Lu⁺ optical references verified at ~1×10⁻¹⁹** (Nature, 2026-09-23).
- The first international optical-clock agreement below 10⁻¹⁷ (7.7×10⁻¹⁸).
- A milligram-scale Cavendish G measurement (Nat. Commun., 2026-10-07).
- The NIST replication of the BIPM G apparatus: 6.67387(38)×10⁻¹¹, ~1σ below CODATA (published April; covered July).
- ALPHA's 4 ppm antihydrogen hyperfine splitting (published May 27; covered June/July).
- La₃₋ₓNdₓNi₂O₇ superconductivity signatures up to 100.5 K under pressure.
- A plasmonic photonic time crystal (Nature, July 29).
- NIF's 11th ignition shot: 7.9 MJ, gain ≈3.8 (2026-06-20). NIF's record is still 8.6 MJ, gain 4.13 (April 2025).

### Cited Findings
**Gravitational constant G**
- **[BACKGROUND pub. 2026-04-16; IN-WINDOW coverage 2026-07-28] Schlamminger, Chao, Lee, Shakarji, Possolo, Newell, Stirling, Cochrane & Speake, "Redetermination of the gravitational constant with the BIPM torsion balance at NIST", Metrologia 63 025012, DOI 10.1088/1681-7575/ae570f.**
  - Published version: **G = (6.67387 ± 0.00038)×10⁻¹¹ m³ kg⁻¹ s⁻²**, u_r = 5.7×10⁻⁵, lower than the BIPM result by 2.5×10⁻⁴ (relative).
  - It uses the same hexadecapole torsion balance, whose torque scales as R_s⁻⁵.
  - CODATA is 6.67430(15). IEEE Spectrum quotes 6.67387 and "0.0235%" below BIPM.
  - **Conflict:** the NIST publication-page abstract gives G = (6.67366 ± 0.00020)×10⁻¹¹ and 2.81×10⁻⁴ below BIPM; its stated 3.0×10⁻⁶ relative uncertainty is inconsistent with those numbers. This is likely a pre-final version, so treat 6.67387(38) as the published value.

  — [IOP Metrologia PDF](https://beta.iopscience.iop.org/article/10.1088/1681-7575/ae570f/pdf); [NIST pub page](https://www.nist.gov/publications/redetermination-gravitational-constant-bipm-torsion-balance-nist); [IEEE Spectrum](https://spectrum.ieee.org/universal-gravitational-constant-nist-schlamminger)
- **[IN WINDOW, 2026-10-07] An, Tan, Dong, Qin, Deng et al., "Test of the gravitational interaction between milligram masses using a miniature torsion pendulum", Nature Communications, DOI 10.1038/s41467-026-78187-4.** Two **7.6 mg** Si wafers at separations of 0.654–1.554 mm; maximum uncertainty 50 aN (2σ). **G = 6.67(35)×10⁻¹¹** (1σ), with Yukawa constraints. **Hook:** Newtonian force between two thin plates at mm separation, and Yukawa α–λ exclusion. — [Nature Communications](https://www.nature.com/articles/s41467-026-78187-4)
- **[IN WINDOW, 2026-10-08] Hu & Zhang, "The third measurement of the gravitational constant between massive bodies", AIP Advances 16, 105206.** The value was not retrieved. — [AIP Advances](https://pubs.aip.org/aip/adv/article/16/10/105206/3406194/The-third-measurement-of-the-gravitational)

**Nuclear clocks (²²⁹Th)**
- **[IN WINDOW, 2026-10-07] Toscani De Col, Riebner, Morawetz, Schneider, Sempelmann et al. (TU Wien/PTB lineage), "A thorium-229 optical nuclear clock with feedback loop", Nature, DOI 10.1038/s41586-026-11084-4 (arXiv:2606.04997).**
  - Setup: a CW 148-nm laser locked by absorption spectroscopy in a room-temperature ²²⁹Th:CaF₂ crystal, compared with a Yb⁺ ion clock.
  - Instability: **3×10⁻¹²/√τ**, approaching 10⁻¹⁵ over one day.
  - f₀ = 2.0204×10¹⁵ Hz with a linewidth of ~100 kHz.
  - Line centres varied by up to 1.7 kHz between crystal positions, attributed to strain.
  - Ultralight-dark-matter coupling limits are competitive for photons and best-to-date for the strong force.

  — [Nature](https://www.nature.com/articles/s41586-026-11084-4)
- **[IN WINDOW, 2026-10-07] Huang, Yan, Xiao, Bu, Zhao et al. (Tsinghua, Ding Shiqian group), "A nuclear clock synchronized to ²²⁹Th", Nature, DOI 10.1038/s41586-026-11122-1 (arXiv:2606.08870).**
  - Laser: a 10 µW CW 148.4 nm source from four-wave mixing in Cd vapour, sub-Hz estimated linewidth.
  - Crystal: grown from 1.4 µg (10 kBq) of ²²⁹Th.
  - Locked to "line b". Instability **5×10⁻¹³/√τ**.
  - Two independent crystals agree at the 10⁻¹³ level.

  — [Nature](https://www.nature.com/articles/s41586-026-11122-1); [Tsinghua news 2026-10-09](https://www.phys.tsinghua.edu.cn/phyen/info/1154/2344.htm)
- **[IN WINDOW, 2026-10-07, preprint] arXiv:2610.10056, "A solid-state nuclear clock based on VUV absorption spectroscopy of ²²⁹Th".**
  - Operation: 30 h locked on line b; instability 1.29×10⁻¹¹/√τ, reaching 1.24×10⁻¹³ at 10⁴ s.
  - **ν_b = 2,020,407,298,796.17(54) kHz** and ν_c = 2,020,407,444,524.78(91) kHz, both temperature-referenced. These are partial uncertainty budgets.

  **Hook:** E = hν, λ = c/ν; line splittings (ν_c − ν_b ≈ 145.73 MHz) from the electric quadrupole interaction. — [arXiv HTML](https://arxiv.org/html/2610.10056)

**Optical clocks / redefinition of the second**
- **[IN WINDOW, 2026-09-23] "Lu⁺ optical frequency references with accuracy verified at the 19th digit", Nature, DOI 10.1038/s41586-026-11072-8.** Two ¹⁷⁶Lu⁺ references with evaluated systematics of 1.2×10⁻¹⁹ and 1.3×10⁻¹⁹. Measured difference **(−0.1 ± 5.7 stat ± 1.0 sys)×10⁻¹⁹** after 200 h. Single-run instability 4.8×10⁻¹⁶/√τ. The SI-redefinition roadmap thresholds are ≲2×10⁻¹⁸ evaluated and ≲5×10⁻¹⁸ comparison agreement. Geopotential knowledge limits remote comparisons to ≳2×10⁻¹⁸ (a few cm of height). **Hook:** gravitational redshift Δf/f = gΔh/c² ≈ 1.09×10⁻¹⁶ per metre, so 10⁻¹⁹ ↔ ~1 mm. — [Nature](https://www.nature.com/articles/s41586-026-11072-8)
- **[IN WINDOW, 2026-09-01] Pizzocaro et al., "International optical clock comparison using the European optical fiber network", APS journal, DOI 10.1103/l4bh-ryxs.**
  - Campaign: 7 clocks at INRIM, LNE-OP, NPL and PTB over 65 days in 2023, via fibre links of 785, 1370 and 1023 km.
  - The NPL and PTB ¹⁷¹Yb⁺(E3) clocks agree within **7.7×10⁻¹⁸**, the first international verification below 10⁻¹⁷.
  - Ratio example: Yb⁺(E3)/Sr = 1.495991618544900560(22).

  — [APS harvest](https://harvest.aps.org/v2/journals/articles/10.1103/l4bh-ryxs/fulltext)
- **[IN WINDOW, preprint 2026-08-03] PTB transportable Sr4 vs Yb⁺(E3), arXiv:2608.01916.** ν_Yb⁺/ν_Sr = 1.4959916185449005881(65), a fractional uncertainty of 4.3×10⁻¹⁸. — [arXiv HTML](https://arxiv.org/html/2608.01916)
- **[IN WINDOW, preprint 2026-06-22, v2 2026-08-24] PTB ²⁷Al⁺ clock, arXiv:2606.23151.** Systematic uncertainty 1.6×10⁻¹⁸. ν_Al⁺ = 1,121,015,393,207,859.19(24) Hz (Cs fountains). Al⁺/Sr = 2.611701431781462668(36), which differs by **8.6σ** from the BACON 2021 ratio and by 1.2σ from BACON 2025. — [arXiv](https://arxiv.org/abs/2606.23151v2)
- **[BACKGROUND, PRL accepted 2026-05-28] NIST/JILA (BACON) Al⁺/Yb/Sr frequency ratios ≤3.2×10⁻¹⁸.** Sr ratios moved by ~1×10⁻¹⁶ and Al⁺/Yb by 1.5–1.6×10⁻¹⁷ relative to previous measurements. A 3.6 km fibre link is used and the lattice-clock ratio instability is 1.3×10⁻¹⁶ at 1 s. — [NIST](https://www.nist.gov/publications/optical-clock-frequency-ratios-uncertainty-32-10-18); [PRL accepted](https://journals.aps.org/prl/accepted/10.1103/g865-9mk1)

**Antimatter**
- **[BACKGROUND pub. 2026-05-27; IN-WINDOW coverage APS Physics 2026-06-03, CERN Courier 2026-07-23] ALPHA Collaboration, "Four ppm measurement of the antihydrogen ground-state hyperfine splitting", Nature, DOI 10.1038/s41586-026-10556-x.**
  - Main result: **a₁S/h = 1,420,404.8 ± 1.1 (stat) ± 5.6 (sys) kHz** at ~1 T, from ~24,000 anti-atoms. This is a 100× improvement on 2017.
  - The antiproton structure (Zemach) contribution is ~40 ppm.
  - Derived: a₂S/h = 177,563 ± 18 kHz. Sternheim interval (8a₂S − a₁S)/h = 100 ± 150 kHz, vs a hydrogen measurement of 48.9592 ± 0.0068 kHz and a QED prediction of 48.953 ± 0.003 kHz.

  **Hook:** the Fermi-contact hydrogen HFS formula plus QED and Zemach corrections, and Breit–Rabi transition frequencies at 1 T. — [Nature](https://www.nature.com/articles/s41586-026-10556-x); [APS Physics](https://link.aps.org/doi/10.1103/Physics.19.79); [CERN Courier](https://cerncourier.com/antihydrogen-toes-the-line/)
- **[BACKGROUND] ALPHA-g 2023: a_ḡ = (0.75 ± 0.13 ± 0.16) g**, downward; repulsive gravity is excluded. A g-equivalent magnetic gradient is 1.77×10⁻³ T/m for ground-state H. — [Nature 2023](https://www.nature.com/articles/s41586-023-06527-1)

**Superconductivity / quantum materials**
- **[IN WINDOW, 2026-08-12] "High-temperature superconductivity at 100 K in La₃₋ₓNdₓNi₂O₇", Nature Communications, DOI 10.1038/s41467-026-76534-z.** Polycrystals with 0 ≤ x ≤ 2.4, measured under high pressure. Resistive onset T_c ~93 K; the dR/dT signature reaches 96–97 K for x = 2.1 and 2.4. **Radio-frequency transmission shows signatures at 100.5 K for x = 2.1.** Nd compresses the lattice and raises the spin-density-wave T and the pressure of the orthorhombic→tetragonal transition. — [Nature Communications](https://www.nature.com/articles/s41467-026-76534-z)
- **[BACKGROUND, 2026-03-10] Zhou, …, Xue & Chen, "Superconductivity onset above 60 K in ambient-pressure nickelate films", Natl. Sci. Rev. 13 nwag151.** Ambient-pressure (La,Pr)₃Ni₂O₇ on SrLaAlO₄: **T_c,onset ≈ 63 K, T_c,zero ≈ 37 K**, Meissner onset ≈ 23 K (previous record ~10 K). The normal-state exponent α goes from ~2 to 1.11 as T_c rises (strange metal), based on 90 samples. — [NSR](https://academic.oup.com/nsr/advance-article/doi/10.1093/nsr/nwag151/8512895); [SUSTech news](https://newshub.sustech.edu.cn/en/?p=48641)
- **[BACKGROUND, 2026-04-08] Nickelate superstructures, Nature, DOI 10.1038/s41586-026-10352-7.** 1212 and 2323 stackings superconduct at ambient pressure with onsets of 50 K and 46 K; 1313 is metallic. A γ-band Fermi pocket is the discriminator. — [NSFC news](https://www.nsfc.gov.cn/english/site%5F1/news/A2/2026/07-02/542.html)
- **[BACKGROUND, 2025-12-02] Li et al., Nature.** La₂SmNi₂O₇ shows bulk T_c,onset = 92 K and T_c,zero = 73 K at 21.6 GPa, Meissner at 60 K (20.6 GPa), and 96 K onset in La₁.₅₇Sm₁.₄₃Ni₂O₇. — [Nature](https://www.nature.com/articles/s41586-025-09954-4)
- **[BACKGROUND, 2026-02-27] Nature Communications, strained films under pressure.** T_c,onset rises from ~30 K to 48.5 K at 7 GPa (dT_c/dP ~3.3 K/GPa), then falls to 42.8 K at 13 GPa. — [Nature Communications](https://www.nature.com/articles/s41467-026-69660-1)

**New phases of matter / time crystals**
- **[IN WINDOW, 2026-07-29] Guo, Sueiro, Andolina et al., "Plasmonic metamaterial time crystal", Nature, DOI 10.1038/s41586-026-10825-9.** An all-optical photonic time crystal in a THz surface-plasmon cavity metamaterial. The carrier effective mass is modulated by up to **80% of rest mass**. The transition passes through an exceptional point, and emergent gain cuts plasmonic losses by >50%. — [Nature](https://www.nature.com/articles/s41586-026-10825-9)
- **[IN WINDOW, 2026-07-22] Greilich, Kopteva, Korenev et al., "Non-local synchronization of continuous time crystals in a semiconductor", Nat. Commun. 17, 6971, DOI 10.1038/s41467-026-75714-1.** In InGaAs, auto-oscillating electron–nuclear spin systems up to **40 µm** apart synchronize. That matches the electron spin diffusion length of ~38 µm. Frequencies differing by up to 40% lock, across ~10⁹ oscillators. Each donor has a Bohr radius of ~11 nm and couples to ~10⁶ nuclei. — [Nature Communications](https://www.nature.com/articles/s41467-026-75714-1)
- **[BACKGROUND, 2026] Earlier 2026 phases of matter.** Emergent topological semimetal from quantum criticality in CeRu₄Sn₆ (Nature Physics, Jan 2026; spontaneous Hall effect below 1 K) — [phys.org](https://phys.org/news/2026-01-state-quantum-material.html). 2D discrete time crystal on IBM hardware (Nat. Commun., Jan 2026) — [Nature Communications](https://www.nature.com/articles/s41467-025-67787-1). Classical liquid-crystal space-time crystals (Nat. Commun., Mar 2026) — [Nature Communications](https://www.nature.com/articles/s41467-026-70880-8).

**Plasma / fusion**
- **[IN WINDOW, 2026-06-20] NIF ignition #11: yield 7.9 ± 0.4 MJ, target gain ≈3.8.** The record is still the 2025-04-07 shot: **8.6 ± 0.45 MJ from 2.08 MJ** of laser energy (456 TW peak), gain **4.13**. Other shots: 2025-02-23, 5.0 MJ from 2.05 MJ (gain 2.44); 2025-10-01, 3.5 ± 0.17 MJ from 2.065 MJ (gain 1.74). — [LLNL LMF ignition page](https://lmf.llnl.gov/science/achieving-fusion-ignition); [LLNL news 2026-04-15](https://lasers.llnl.gov/news/target-breakthrough-enabled-fusion-record-nif)
- The first gain > 1 shot was 2.05 MJ → 3.1 ± 0.16 MJ (G = 1.5 ± 0.1) ([Phys. Rev. E 109, 025203](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.109.025203)). A local news article misreports 3.88 MJ for that shot ([Independent](https://www.independentnews.com/news/livermore_news/livermore-lab-breaks-fusion-record-again/article_926d3f7f-59c4-4eb0-b6fa-33119b5784cd.html)); LLNL attributes 3.88 MJ to July 30, 2023.
- **[IN WINDOW, 2026-09-17, design study] arXiv:2609.19752, indirect-drive IFE.** A 10 MJ laser driver; gains G ≈ 26–43; yields 265–427 MJ; ~40% burn fraction; ρR ≈ 3 g/cm²; fuel mass >10× current NIF; ignition margin 2–4×. — [arXiv](https://arxiv.org/abs/2609.19752)
- **[BACKGROUND, 2025] Magnetic-confinement records** — [IAEA FEC 2025 poster](https://conferences.iaea.org/event/392/contributions/36141/attachments/20114/36116/Litaudon-IAEA-Poster%20vf.pdf); [IPP](https://www.ipp.mpg.de/5532945/w7x):
  - EAST: 1066 s fully non-inductive H-mode.
  - WEST: 1337 s, 2.6 GJ injected.
  - W7-X: nTτ_E = 0.077 atm·s for up to 24 s and 0.06 atm·s for 40 s (a 43 s record discharge); 1.8 GJ over 360 s.
- **EAST 1066 s shot details** (EPS 2026 paper): 3.05 GJ injected; β_p = 2.0; f_BS ≈ 50%; H₉₈ = 1.3; n/n_G = 0.63; driven by LHCD 1.1 MW + ECRH 1.9 MW. — [UKAEA Indico EPS 2026](https://indico.ukaea.uk/event/449/contributions/2097/attachments/717/1226/EPS2026_Rev_JHuang.pdf)
- **W7-X 360 s discharge details:** detached at 5 MW, n_e ~5×10¹⁹, diamagnetic energy ~500 kJ, f_rad > 80%. — [IAEA long-pulse meeting abstract (Oct 20–23, 2026)](https://conferences.iaea.org/event/452/contributions/42916/)

### Inferences
- **Benchmarks (my computations).**
  - **²²⁹Th:** ν_b = 2.02040729879617×10¹⁵ Hz gives λ = 148.382 nm and E = 8.35573 eV, using exact SI h and c. This is a clean atomic/nuclear-inventory regression value.
  - **Antihydrogen vs hydrogen HFS:** 1,420,404.8 vs 1,420,405.751768 kHz, a difference of −0.95 kHz (−6.7×10⁻⁷, −0.17σ), so CPT holds at 4 ppm.
  - **G:** NIST 6.67387(38) is −6.4×10⁻⁵ relative to CODATA 6.67430(15), −1.05σ. The NIST-page variant 6.67366(20) would be −2.6σ, a material conflict worth flagging.
  - **NIF:** 8.6/2.08 = 4.13; 7.9 MJ at gain 3.8 implies ~2.08 MJ on target.
- For a "Wolfram-Alpha accuracy" engine, the clock ratios are 18–19 significant-figure constants (Yb⁺(E3)/Sr, Al⁺/Sr). The ²²⁹Th line frequencies are the most reproducible "new" numbers, but labs disagree at the 10⁻¹⁷ level (the 8.6σ Al⁺/Sr discrepancy). Store them with provenance rather than as single truths.
- The nickelate T_c record depends on definition (onset, zero-resistance, RF signature) and on pressure. An engine should store T_c with definition and pressure fields: 100.5 K is an RF signature under pressure, 63 K is the ambient onset, 37 K is ambient zero-resistance.

### Gaps
- No in-window ALPHA-g precision measurement of antimatter gravity, and no in-window BASE antiproton result, was found.
- The pressure value for the 100.5 K La₃₋ₓNdₓNi₂O₇ signature was not in the retrieved abstract.
- The G value from the AIP Advances "third measurement" (2026-10-08) was not retrieved.
- No in-window record from a new tokamak or stellarator (KSTAR, JT-60SA, SPARC, W7-X OP2.4) was found. The IAEA long-pulse meeting (Oct 20–23, 2026) is just after the window.
- Author lists for the Lu⁺ Nature paper and the plasmonic time-crystal paper beyond the first authors were not retrieved.
- No in-window "speed of light" measurement was found; c is exact in the SI, so clock-comparison results are the relevant precision items.
