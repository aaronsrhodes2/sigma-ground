# Particle and Nuclear Physics: Testable Results, ~10 June – 10 October 2026

Compiled 2026-10-10 from live web searches and fetched pages (arXiv, CERN/ATLAS/CMS/LHCb pages, Nature, PDG 2026, conference slides). Tags used on every item:

- **[IN WINDOW]**: first released or published between about 10 June and 10 October 2026.
- **[BACKGROUND]**: released before the window. It is included because it is the current reference value a physics engine should check against.
- **[σ-FLAG]**: bears on scalar fields coupling to mass, the split between QCD binding and Higgs contributions to mass, varying constants, or nucleon mass decomposition. These matter for the speculative σ field in sigma-ground.
- "Hook" marks the concrete number, relation or formula a Python engine can reproduce or be validated against. Hooks marked "(derived)" are my own arithmetic from the cited inputs, not quoted from a source.

Global context: LHC Run 3 is over. Long Shutdown 3 began on 29 June 2026. CMS reports 355 fb⁻¹ delivered in Run 3, and ATLAS reports 332 fb⁻¹ recorded in Run 3 (505 fb⁻¹ over its lifetime). ALICE's Run-3 Pb–Pb sample is 6.9 nb⁻¹. These figures come from search-result summaries of the [CMS "End of Run 3" page](https://cms.cern/news/end-run-3), the [ATLAS HL-LHC era statement](https://atlas.cern/Updates/Press-Statement/ATLAS-Enters-HiLumi-Era) and the [ALICE Run-3 finale page](https://alice-collaboration.web.cern.ch/2026-ALICE-Run3-Finale). I did not fetch those pages, so treat the luminosity figures as approximate. The **PDG 2026 edition** is out (citation F. Takahashi et al., Int. J. Mod. Phys. A 41, 2630011 (2026); its reviews are dated 1 June 2026). It is the natural reference table to re-baseline a codebase against: [PDG 2026 top listing](https://pdg.lbl.gov/2026/listings/rpp2026-list-t-quark.pdf).

---

## Q1. What new LHC Run 3 results came out (ATLAS, CMS, LHCb, ALICE)? Coverage: Higgs couplings and self-coupling, W mass, top mass, rare decays, LFU, new resonances/excesses, toponium/bound states

### Takeaway
The headline in-window LHC results are the following:
- New Run-2+3 di-Higgs (bbττ) results. ATLAS sees a 2.6σ HH excess, μ_HH = 2.6^{+1.4}_{-1.0}, and constrains κ_λ ∈ [−3.4, 1.6] ∪ [5.5, 10.1]. CMS excludes μ_HH > 4.0 and constrains κ_λ to (−2.5, 9.4).
- An independent >5σ confirmation of toponium in the single-lepton channel: σ(η_t) = 5.1 ± 0.9 pb at 6.1σ, with the paper released in August 2026.
- LHCb's discovery of an unexpectedly light, narrow B_s0*(5700) at 5698.9 ± 1.6 MeV, more than 7σ global.
- CMS limits on rare W decays and a |V_cb| measurement from top decays.

There was no new W-mass or top-mass measurement in the window. The reference values remain the May 2026 W world average, 80362.5 ± 7.7 MeV excluding CDF, and the LHC Run-1 top combination, 172.52 ± 0.33 GeV.

### Cited Findings

#### Higgs pair production and self-coupling (κ_λ) [σ-FLAG: these measure the shape of the Higgs scalar potential]
- **[IN WINDOW] ATLAS, "Improved analysis of non-resonant HH production in the bb̄τ⁺τ⁻ final state with 196 fb⁻¹ at 13 and 13.6 TeV", arXiv:2607.26879 (submitted 29 Jul 2026; preprint; shown at ICHEP 2026).**
  - Data: 140 fb⁻¹ at 13 TeV plus 56 fb⁻¹ at 13.6 TeV.
  - μ_HH = 2.6^{+1.4}_{-1.0}. Observed (expected) significance is 2.6 (1.2)σ over background-only and 1.65σ from the SM.
  - κ_λ at 95% CL is observed in [−3.4, 1.6] ∪ [5.5, 10.1], with expected [−1.7, 8.5].
  - Validation channels in the same final state: ZH at 3.5 (2.4)σ, which is first ATLAS evidence for ZH→bbττ, and ZZ at 1.8 (2.7)σ.
  - Expected sensitivity improves about 60% over the previous ATLAS bbττ analysis.
  - Hook: κ_λ intervals. Sources: [arXiv:2607.26879](https://arxiv.org/abs/2607.26879v1); [ATLAS briefing, 5 Aug 2026](https://atlas.cern/Updates/Briefing/DiHiggs-bbtautau)
- **[IN WINDOW] CMS, "Search for nonresonant HH production in the bbττ final state at √s = 13.6 TeV", CMS-HIG-25-008 (ICHEP 2026; preliminary PAS).**
  - Run 3 alone uses 172 fb⁻¹. The observed (expected) 95% CL upper limit is 6.6 (3.9)× SM for inclusive production and 81 (85)× SM for VBF.
  - Combined with 138 fb⁻¹ of Run 2 (310 fb⁻¹ total), the limits are 4.0 (3.0)× SM inclusive and 62 (70)× SM for VBF.
  - −2.5 < κ_λ < 9.4 observed (expected −1.5 to 8.4). 0.02 < κ_2V < 2.1 observed (expected −0.1 to 2.2).
  - Sources: [CMS ICHEP-2026 list](https://cms.cern/news/cms-ichep-2026); [abstract via Exa library](https://exa.ai/library/publication/24r0djpnrsl); [CERN news, 27 Aug 2026](https://home.cern/atlas-and-cms-narrow-in-on-double-higgs-production/)
- **[IN WINDOW, journal publication] CMS Run-2 HH combination (138 fb⁻¹), J. Phys. G (IOPscience, online 24 Jul 2026).**
  - HH < 3.5 (2.5)× SM. −1.35 ≤ κ_λ ≤ 6.37 at 95% CL. 0.64 ≤ κ_2V ≤ 1.40, and κ_2V = 0 is excluded at more than 5σ.
  - **Conflict:** the HEPData/PAS version (CMS-HIG-20-011) quotes −1.39 < κ_λ < 7.02 and 0.62 < κ_2V < 1.42. The journal version appears updated, so use the IOP numbers.
  - Sources: [IOP J. Phys. G](https://iopscience.iop.org/article/10.1088/1361-6471/ae8ff1); [HEPData](https://www.hepdata.net/record/ins3065780)
- **[BACKGROUND, Feb 2026] First ATLAS+CMS HH combination (Run 2), arXiv:2602.23991.**
  - μ_HH = 0.8^{+0.9}_{-0.7}, with an upper limit of 2.5 (1.7)× SM.
  - −0.71 < κ_λ < 6.1 (expected −1.3 to 6.7). 0.73 < κ_2V < 1.3.
  - This is still the most stringent κ_λ constraint. — [arXiv:2602.23991](https://arxiv.org/abs/2602.23991)
- **[BACKGROUND, 2 Jun 2026] ATLAS first ttHH search (Run 2 + early Run 3).** The HEFT coefficient is constrained to −3.9 < c_ttHH < 3.3 at 95% CL, against an SM value of 0. The limit on the ttHH cross section is only about 8× weaker than the ATLAS+CMS HH combination, even though the rate is about 40× smaller. — [ATLAS briefing](https://atlas-public.web.cern.ch/index%2Ephp/Updates/Briefing/ttHH-Frontier)
- The SM rate is about 1 HH pair per 1500 single Higgs bosons at the LHC. The HL-LHC will record about 6× more data. — [CERN news](https://home.cern/atlas-and-cms-narrow-in-on-double-higgs-production/)

#### Rare Higgs decays and couplings
- **[IN WINDOW as preprint; PAS 7 Apr 2026] CMS, "Search for the rare Higgs boson decay H→Zγ at √s = 13 and 13.6 TeV", arXiv:2609.04402.**
  - Data: 200 fb⁻¹ (138 fb⁻¹ Run 2 + 62 fb⁻¹ Run 3).
  - μ = 1.10^{+0.52}_{-0.61} (expected 1.00^{+0.49}_{-0.46}) at m_H = 125.38 GeV. Observed (expected) significance is 1.9 (2.3)σ.
  - Hook: the SM prediction B(H→Zγ) = (1.5 ± 0.1)×10⁻³.
  - The previous ATLAS+CMS Run-2 combination gave μ = 2.2 ± 0.7, B = (3.4 ± 1.1)×10⁻³ at 3.4σ. The new results pull the rate back toward the SM.
  - Sources: [arXiv:2609.04402](https://arxiv.org/pdf/2609.04402); [CMS PAS HIG-25-010](https://cms-results.web.cern.ch/cms-results/public-results/preliminary-results/HIG-25-010/index.html)
- **[BACKGROUND, 2025] ATLAS H→Zγ, Run 2+3 (305 fb⁻¹).** μ = 1.3^{+0.6}_{-0.5} at 2.5 (1.9)σ. SM inputs quoted: BR(H→Zγ) = (1.54^{+0.10}_{-0.11})×10⁻³ and BR(H→γγ) = (2.27^{+0.07}_{-0.06})×10⁻³ at m_H = 125.09 GeV. — [arXiv:2507.12598](https://arxiv.org/pdf/2507.12598)
- **[BACKGROUND, PRL 135, 231802 (2025)] ATLAS evidence for H→μμ (Run 2 + 165 fb⁻¹ Run 3).** 3.4 (2.5)σ, μ = 1.4 ± 0.4, B(H→μμ) = (3.0 ± 0.9)×10⁻⁴ against an SM value of 2.17×10⁻⁴. This is the first evidence for a second-generation Yukawa coupling. — [arXiv:2507.03595](https://arxiv.org/html/2507.03595)
- **[IN WINDOW] Other ATLAS ICHEP-2026 Higgs/EW preprints (numbers not extracted):**
  - ttH/tH with H→ττ at 13 + 13.6 TeV (arXiv:2607.27276)
  - H→ℓℓγ at low dilepton mass (arXiv:2608.03369)
  - Observation of longitudinally polarized ZZ pairs (arXiv:2607.12879)
  - Observation of electroweak γγjj (arXiv:2607.25869)
  - Source: [ATLAS ICHEP-2026 summary](https://atlas.cern/Updates/News/Summary-ICHEP-2026)

#### W boson
- **[BACKGROUND, 7 May 2026, adopted in PDG 2026] LHC–Tevatron W-mass working-group world average: m_W = 80362.5 ± 7.7 MeV, CDF excluded.**
  - Including CDF, the compatibility probability is only 0.06%. Excluding it, compatibility is 96.9%.
  - Input values:
    - CMS: 80360.2 ± 9.9 MeV
    - ATLAS (2024 reanalysis): 80366.5 ± 15.9 MeV
    - LHCb: 80354 ± 32 MeV
    - CDF 2022 (not used in the average): 80433.5 ± 9.4 MeV
  - W width: PDG average Γ_W = 2.14 ± 0.05 GeV (scale factor 1.7). The ATLAS value alone is 2.202 ± 0.047 GeV.
  - Sources: [PDG 2026 W review](https://pdg.lbl.gov/2026/reviews/rpp2026-rev-w-mass.pdf); [PDG 2026 W listing](https://pdg.lbl.gov/2026/listings/rpp2026-list-w-boson.pdf)
- **[BACKGROUND, Nature, 8 Apr 2026] CMS first W mass: m_W = 80360.2 ± 2.4 (stat) ± 9.6 (syst) MeV.** It uses 117M W→μν events at 13 TeV. The EW-fit prediction is 80353 ± 6 MeV, and m_Z = 91188.0 ± 2.0 MeV. — [Nature s41586-026-10168-5](https://www.nature.com/articles/s41586-026-10168-5)
- **[BACKGROUND, 2024] ATLAS Γ_W = 2202 ± 47 MeV against an SM value of 2088 ± 1 MeV, a 2.4σ difference.** — [EPJC 84 (2024) 1309](https://link.springer.com/article/10.1140/epjc/s10052-024-13190-x)
- **[IN WINDOW] CMS, "Search for W boson decays to three charged hadrons", CMS-PAS-SMP-25-011 (8 Jul 2026; preliminary; ICHEP highlight).**
  - Data: 138 fb⁻¹ at 13 TeV plus 308 fb⁻¹ at 13.6 TeV, including 2026 data.
  - B(W±→3h±) < 2.4×10⁻⁷ at 95% CL, about 4× better than before. The best fit is 0.5^{+1.0}_{-0.5}×10⁻⁷.
  - Source: [CMS PAS SMP-25-011](https://cms-results.web.cern.ch/cms-results/public-results/preliminary-results/SMP-25-011/index.html)
- **[IN WINDOW] CMS first observation of high-energy lepton–lepton scattering (SMP-26-008; uses 2026 data).** No numbers were extracted. — [CMS ICHEP-2026](https://cms.cern/news/cms-ichep-2026)

#### Top quark, |V_cb| and toponium
- **[IN WINDOW] CMS, "Measurement of |V_cb| from top quark pair events at 13 TeV", CMS-PAS-TOP-26-001 (30 Jul 2026; preliminary).**
  - |V_cb| = 4.59^{+0.44}_{-0.48} (stat) ^{+0.64}_{-0.76} (syst) ×10⁻², from B(W→cb) in tt̄ lepton+jets events with 138 fb⁻¹.
  - It is independent of the B-decay form factors.
  - Source: [CMS PAS TOP-26-001](https://cms-results.web.cern.ch/cms-results/public-results/preliminary-results/TOP-26-001/index.html)
- **[Likely just before or at the window start, LHCP 2026] ATLAS |V_cb| from tt̄ decays.** |V_cb| = (50^{+7}_{-9} (stat) ^{+9}_{-10} (syst))×10⁻³, which the slide also quotes as (50^{+11}_{...})×10⁻³. — [ATLAS LHCP-2026 top slides](https://indico.cern.ch/event/1537360/contributions/6952149/attachments/3278875/5858971/toplhcp2026jtv4.pdf)
- **[BACKGROUND] Top mass references.**
  - The ATLAS+CMS Run-1 combination is m_t = 172.52 ± 0.14 (stat) ± 0.30 (syst) GeV, i.e. ± 0.33 GeV, still the most precise. — [arXiv:2402.08713](https://arxiv.org/abs/2402.08713)
  - ATLAS boosted top: 172.95 ± 0.53 GeV. CMS ℓ+jets: 171.77 ± 0.37 GeV. — [PDG 2026 t listing](https://pdg.lbl.gov/2026/listings/rpp2026-list-t-quark.pdf)
  - Indirect pole mass from an NNPDF global fit, including toponium corrections: m_t = 172.80 ± 0.26 GeV (arXiv:2603.28865, Mar 2026). — [arXiv:2603.28865](https://www.arxiv.org/pdf/2603.28865)
  - No new in-window m_t measurement was found.
- **[IN WINDOW as full paper; PAS TOP-25-002 at Moriond, Mar 2026] CMS, "Observation of an excess at the top quark pair production threshold in the single-lepton channel", arXiv:2608.19546.** [σ-FLAG: the heaviest QCD bound state, and a clean Coulombic-binding test]
  - Data: 138 fb⁻¹ at 13 TeV.
  - σ(η_t) = 5.1 ± 0.9 pb. Observed (expected) significance is 6.1 (7.3)σ. μ = 0.80^{+0.14}_{-0.13} relative to a 6.43 pb model.
  - With the Green's-function (GFRW) signal model instead, the significance is 5.9σ observed and 13σ expected.
  - Using the relative velocity β*_t instead of m(tt̄) raises the expected sensitivity from 2.1σ to 6.3σ.
  - Hooks:
    - η_t model mass 343 GeV and width 2.8 GeV, about 2Γ_t.
    - Binding energy about 2 GeV below 2m_t.
    - Predicted σ = 6.43 pb.
    - Top lifetime 5×10⁻²⁵ s, against a hadronization time 1/Λ_QCD ≈ 10⁻²⁴ s.
  - Sources: [arXiv:2608.19546](https://export.arxiv.org/pdf/2608.19546); [CMS Moriond 2026](https://cms.cern/index.php/news/cms-moriond-2026); [CMS news, 25 Mar 2026](https://cms.cern/news/cms-observes-another-indication-momentary-union-heaviest-quarks-now-lepton-jets-final-state)
- **[BACKGROUND; published in Rep. Prog. Phys.; CERN-EP-2026-002] ATLAS dilepton toponium.**
  - Excess over the pQCD-only model at more than 8σ (expected 6σ).
  - σ(tt̄_GFRW) = 9.3^{+1.4}_{-1.3} pb, which is 45^{+21}_{-20}% above the calculated 6.43 pb.
  - With the η_t simplified model, σ = 13.1^{+1.9}_{-1.7} pb. With the bb4l baseline, 8.5^{+1.2}_{-1.1} pb.
  - Sources: [arXiv:2601.11780](http://arxiv.org/pdf/2601.11780); [IOP Rep. Prog. Phys.](https://iopscience.iop.org/article/10.1088/1361-6633/ae60a0/pdf); [arXiv:2604.17138 proceedings](https://ar5iv.labs.arxiv.org/html/2604.17138)
  - **Model dependence:** extracted cross sections differ by up to about 1.6× between NRQCD models (η_t vs GFRW). A codebase should not hard-code a single "toponium cross section".

#### Flavour, LFU and rare decays (LHCb, Belle II)
- **[IN WINDOW] LHCb, "Discovery of an unexpectedly light and narrow beauty-strange state", arXiv:2609.25178 (LHCb-PAPER-2026-021, 21 Sep 2026; announced at ICHEP, 1 Aug 2026; preprint).** [σ-FLAG: a mass deficit attributed to chiral-symmetry-breaking dynamics]
  - B_s0*(5700)⁰ → B_s⁰π⁰. m = 5698.9 ± 1.5 (stat) ± 0.6 (syst) MeV. Γ < 9.8 (11.8) MeV at 90 (95)% CL.
  - 343^{+44}_{-41} signal events. Local significance above 8σ, global above 7σ. Data: 9 fb⁻¹ from Run 1+2.
  - Interpretation: the beauty analogue of D_s0*(2317), and the first non-conventional single-b hadron, about 130 MeV below quark-model predictions.
  - Predicted related peaks: 5649.5 MeV (B_s*γ) and 5748.3/5698.9 MeV (1⁺ partner), each ± 1.6 MeV.
  - The press account gives m(B_s0*) − m(B_s) ≈ 332 MeV, against ≈ 349 MeV in the charm sector.
  - Sources: [arXiv:2609.25178](https://arxiv.org/abs/2609.25178); [LHCb outreach, 1 Aug 2026](https://lhcb-outreach.web.cern.ch/2026/08/01/discovery-of-a-beauty-strange-particle-with-unexpected-properties/); [news summary](https://thenationview.com/lhcb-finds-new-particle-130-mev-lighter-than-predicted/)
- **[IN WINDOW] LHCb, "Improved measurement of CP violation in B_s⁰→J/ψπ⁺π⁻", arXiv:2608.14119 (LHCb-PAPER-2026-017, 14 Aug 2026).**
  - Run 2 (6 fb⁻¹): φ_s = −0.077 ± 0.034 ± 0.007 rad, |λ| = 0.993 ± 0.026 ± 0.007, Γ_H = 0.610 ± 0.002 ± 0.004 ps⁻¹.
  - LHCb all-mode combination: φ_s = −0.041 ± 0.017 rad.
  - World average φ_s(cc̄s) = −0.052 ± 0.013 rad. Consistent with the SM.
  - Source: [arXiv:2608.14119](https://arxiv.org/html/2608.14119)
- **[IN WINDOW] LHCb, "Observation of several sources of CP violation in B⁺→K⁺π⁺π⁻", arXiv:2608.12614 (17 Aug 2026).**
  - First observation of quasi-two-body CPV in B⁺→ρ(770)⁰K⁺: A_CP = +0.280 ± 0.022 ± 0.029, at least 5.9σ.
  - CPV in the S–P interference is observed at about 5.4σ.
  - Evidence for CPV in ρ₃(1690)⁰K⁺ (A_CP ≈ +0.60 ± 0.15, at least 3.3σ), the first such evidence in a decay involving a spin-3 resonance.
  - Source: [arXiv:2608.12614](https://arxiv.org/html/2608.12614)
- **[IN WINDOW, journal] LHCb, first decay-time-integrated CP asymmetry in B_s⁰→D_s⁻π⁺: ⟨A_s untagged⟩ = (−1.4 ± 5.9 ± 1.1)×10⁻³ (PRL, 21 Aug 2026).** This excludes generic BSM tree-level CP asymmetries outside −1.9% to +1.6% at 95% CL. — [PRL via Birmingham repository](https://pure-oai.bham.ac.uk/ws/portalfiles/portal/319930312/AaijR2026First.pdf)
- **[Likely BACKGROUND, arXiv:2604.08631] LHCb R(K*⁰) at high q².** R(K*⁰) = 1.08^{+0.14}_{-0.12} (stat) ± 0.07 (syst) for q² > 14 GeV², consistent with LFU and the first hadron-collider measurement in this region. — [arXiv:2604.08631](https://arxiv.science/abs/2604.08631)
- **[BACKGROUND, reported at 2026 conferences] R(D(*)).**
  - HFLAV world averages: R(D) = 0.358 ± 0.024 and R(D*) = 0.281 ± 0.011, with correlation ρ = −0.374. SM values: R(D) = 0.296 ± 0.004 and R(D*) = 0.254 ± 0.005.
  - The combined tension is about 3.8σ.
  - A new LHCb B_c→J/ψℓν measurement moves the R(J/ψ) average to 2.4σ from the SM.
  - Belle II preliminary (hadronic tag): R(D) = 0.439 ± 0.055 ± 0.046 and R(D*) = 0.242 ± 0.019 ± 0.016.
  - Sources: [Beauty 2026, LHCb semileptonic slides](https://indico.cern.ch/event/1595459/contributions/7068642/attachments/3288622/5879664/4.3%20Beauty2026-MCalvi.pdf); [BEACH 2026, Belle II slides](https://indico.cern.ch/event/1537617/contributions/7010256/attachments/3289696/5881815/Warburton_BEACH2026_LFU_SL_BelleII_v5.pdf)
- **[IN WINDOW, ICHEP] CMS flavour PAS results (no numbers extracted):**
  - rare b→sμμ decays with partial Run 3 data (BPH-24-005)
  - search for K_S→μμ at 13.6 TeV (BPH-25-005)
  - observation of η→μ⁺μ⁻π⁺π⁻ (BPH-26-004)
  - Source: [CMS ICHEP-2026](https://cms.cern/news/cms-ichep-2026)

#### New resonances and excesses
- **[IN WINDOW, v2 dated 26 Aug 2026] CMS diphoton search, 10–70 GeV (54.4 fb⁻¹, 2018 data), arXiv:2603.03250v2.** The largest excess is at 13.6 GeV, with 3.5σ local and 1.9 ± 0.1σ global significance. ALP decay-constant limits are 4–15 TeV. — [arXiv:2603.03250v2](https://arxiv.org/html/2603.03250v2)
- **[IN WINDOW] ATLAS high-mass dilepton resonance search at 13.6 TeV combined with 13 TeV, arXiv:2607.28334.** No excess numbers were extracted. — [ATLAS ICHEP-2026](https://atlas.cern/Updates/News/Summary-ICHEP-2026)

#### Heavy ions at the LHC: see Q6.

### Inferences
- **κ_λ hooks (derived from standard SM relations).**
  - λ_SM = m_H²/(2v²) ≈ 0.129–0.130, using m_H = 125.09–125.38 GeV and v = 246.22 GeV.
  - The SM trilinear coupling is g_hhh = 3m_H²/v ≈ 191 GeV.
  - Current data allow κ_λ ≈ −0.7 to 6.1. An engine that modifies the Higgs potential (for example through σ–Higgs mixing) must stay inside this band.
  - The ATLAS bbττ excess (μ_HH ≈ 2.6) is only 1.65σ from the SM. Do not treat it as evidence of an enhanced self-coupling.
- **W mass hook (derived).**
  - The world average minus the EW fit is 80362.5 − 80353 = 9.5 ± 9.8 MeV, about 1σ.
  - CDF minus the world average is about 71 ± 12 MeV, roughly 6σ if treated as uncorrelated.
  - The SM relation m_W²(1 − m_W²/m_Z²) = πα/(√2 G_F)(1 + Δr) should reproduce about 80.355 GeV when given m_t, m_H and α(m_Z). The CDF value should be flagged as an outlier.
- **Toponium hook (derived).** A Coulombic NRQCD estimate gives E_B ≈ ½ μ (C_F α_s)² with μ = m_t/2 ≈ 86 GeV and C_F = 4/3. This yields about 1.7–2.0 GeV for α_s ≈ 0.15–0.16, consistent with the "about 2 GeV" binding the experiments assume. The Bohr radius is about 0.011 fm, far inside the confinement scale. A σ-field that rescales m_t or α_s would move this bound state directly. The binding energy is comparable to Γ_t ≈ 1.4 GeV, which is why the state is only "quasi-bound".
- **B_s0*(5700) hook (derived).** If heavy-quark flavour symmetry holds, the chiral-partner splitting should be nearly flavour-independent. The measured splittings are about 332 MeV (b sector) and about 349 MeV (c sector). This is a mass-generation test from spontaneous chiral-symmetry breaking, not from the Higgs, and a σ-field model of "QCD binding" should reproduce a splitting of roughly 0.33–0.35 GeV.

### Gaps
- I could not find the per-collaboration combined (all-channel) Run-2+3 κ_λ results. These are probably not released yet; CERN says the "upcoming analyses combining full Run 2 and Run 3" are still to come.
- There is no new in-window W-mass or top-mass measurement. An ATLAS low-pile-up m_W result is expected "in approximately one year", per [W-mass combination slides, Mar 2026](https://indico.cern.ch/event/1615651/contributions/6818997/attachments/3229241/5756209/AP_CMSPrecisionEWKWorkshop_20260302.pdf).
- I found no 2026 Run-3 update of the 95 GeV diphoton excess (see Q7).
- I did not extract numbers for the CMS lepton–lepton scattering observation, the CMS Higgs-compositeness search (EXO-25-001), or the ATLAS longitudinal-ZZ observation.
- ALICE-specific in-window particle-physics (non-heavy-ion) numbers were not found.

---

## Q2. What came out on the muon g−2 (final Fermilab result, lattice vs data-driven HVP), the electron g−2, and the fine-structure constant?

### Takeaway
No new muon g−2 measurement appeared in the window. The final Fermilab value from June 2025 stands, with a world average of a_μ(exp) = 116 592 0715(145)×10⁻¹² (124 ppb). It agrees with the lattice-based 2025 White Paper SM value, so the "muon g−2 anomaly" is effectively resolved.

The live problem is now inside the SM: the data-driven e⁺e⁻→hadrons HVP disagrees with lattice QCD. New in-window results deepen this split:
- A first sub-percent lattice NLO-HVP result (JHEP, Sept 2026) shows a 4.6σ tension with data-driven evaluations that exclude CMD-3.
- A July 2026 dispersive fit with inflated errors gives an SM value 2σ below experiment.

No new α or electron g−2 measurement was found in the window. The Cs–Rb α discrepancy (>5σ) remains.

### Cited Findings
- **[BACKGROUND, PRL 135, 101802, published 2 Sep 2025] Muon g−2 Collaboration (FNAL E989), "Measurement of the positive muon anomalous magnetic moment to 127 ppb".**
  - Runs 4–6: a_μ = 116 592 0710(162)×10⁻¹² (139 ppb).
  - Runs 1–6 combined: a_μ = 116 592 0705(148)×10⁻¹² (127 ppb).
  - World average with BNL: a_μ(exp) = 116 592 0715(145)×10⁻¹² (124 ppb).
  - Sources: [arXiv:2506.03069](https://arxiv.org/abs/2506.03069); [FNAL PDF](https://muon-g-2.fnal.gov/result2025.pdf); [APS Physics](https://physics.aps.org/articles/v18/150)
- **[BACKGROUND, Physics Reports 1143 (2025)] Muon g−2 Theory Initiative White Paper 2025.**
  - a_μ(SM) = 116 592 033(62)×10⁻¹¹ (530 ppb). It uses only the lattice LO-HVP average (about 0.9% precision), because the CMD-3 data make data-driven inputs impossible to combine.
  - Against the older experimental average, Δa_μ = 26(66)×10⁻¹¹.
  - Source: [arXiv:2505.21476](https://arxiv.org/abs/2505.21476v2)
- **[BACKGROUND, Nature, 22 Apr 2026] BMW/DMZ hybrid lattice + data LO-HVP (Boccaletti, Borsanyi, Davier, Fodor, Lellouch, Malaescu et al.).**
  - a_μ^{LO-HVP} = 715.1(2.5)(2.3)[3.4]×10⁻¹⁰, i.e. 0.48%, with lattice spacing down to 0.048 fm.
  - The long-distance tail (t ≥ 2.8 fm, under 5% of the total) comes from e⁺e⁻ data below the ρ, where experiments agree.
  - The resulting SM prediction differs from experiment by only 0.5σ.
  - Source: [Nature s41586-026-10449-z](https://www.nature.com/articles/s41586-026-10449-z)
- **[IN WINDOW, journal: JHEP 09 (2026) 038, published 2 Sep 2026; arXiv:2603.06806] First sub-percent lattice computation of the NLO HVP contribution.**
  - It is 1.4σ below the WP2025 estimate and 2× more precise.
  - It shows a **4.6σ tension with data-driven evaluations based on hadronic cross sections excluding CMD-3**.
  - The numerical value was not captured in my fetch.
  - Sources: [JHEP](https://link.springer.com/article/10.1007/JHEP09(2026)038); [arXiv:2603.06806](https://arxiv.org/html/2603.06806)
- **[IN WINDOW, preprint, 7 Jul 2026] "Dispersive estimation of the LO hadronic contribution to the muon g−2: fitting incompatible e⁺e⁻ data", arXiv:2607.06352.**
  - BaBar, KLOE-2 and CMD-3 ππ data are mutually incompatible at about 3–5σ. Inflating the errors gives a_μ(had, LO) = (697.7 ± 9.8_{e⁺e⁻} ± 3.6_sys)×10⁻¹⁰.
  - This yields a_μ(SM) = 11659185.5(10.6)×10⁻¹⁰, about 2σ below experiment.
  - Source: [arXiv:2607.06352](https://arxiv.org/html/2607.06352)
- **[BACKGROUND, Jan 2026, preliminary] BaBar new ππ(γ) cross-section analysis on 460 fb⁻¹.** It is consistent with BaBar 2009. KLOE vs CMD-3 is a more than 5σ tension at the ρ peak. — [arXiv:2601.16587](https://arxiv.org/html/2601.16587v1); [DIS2026 slides](https://agenda.infn.it/event/47074/contributions/289308/attachments/149590/228792/DIS2026-garziav2.pdf)
- **[IN WINDOW, meeting] 9th Plenary Workshop of the Muon g−2 Theory Initiative, 3–7 Aug 2026, UConn.** Its aim is to resolve the lattice vs data-driven HVP discrepancy. I found no published outcome or updated SM number. — [UConn Indico timetable](https://indico.phys.uconn.edu/event/3/timetable/?print=1&view=standard_numbered)
- **[BACKGROUND] Fine-structure constant references.**
  - Cs recoil (Berkeley, 2018): α⁻¹ = 137.035999046(27), 0.20 ppb. — [Science 2018](https://www.science.org/doi/10.1126/science.aap7706)
  - Rb recoil (LKB, 2020): α⁻¹ = 137.035999206(11), 81 ppt. — [Nature 2020](https://www.nature.com/articles/s41586-020-2964-7)
  - They differ by more than 5σ.
  - A Feb 2026 Nature paper states that the SM test of the electron magnetic moment "is at present limited to 0.7 ppt by discrepant measurements of α". — [Maisenbacher et al., Nature 2026](https://preview-www.nature.com/articles/s41586-026-10124-3)

### Inferences
- **Hook (derived).** a_μ(exp) − a_μ(SM, WP25) = 116592071.5 − 116592033 = 38.5 ± 63.7 ×10⁻¹¹, about 0.6σ. Any σ-field or new scalar that couples to muons must contribute |Δa_μ| ≲ 1.3×10⁻⁹ at 2σ. For a scalar of mass M and Yukawa y, Δa_μ ~ (y²/8π²)(m_μ²/M²) × O(1). The muon therefore no longer provides room for a light scalar coupling to it.
- The real open quantity is the HVP itself. Lattice gives about 715×10⁻¹⁰ (hybrid), while data-driven fits range down to about 698×10⁻¹⁰.
  - A codebase reproducing a_μ should carry HVP as a parameter with this spread, not a single number.
  - The about 2% spread corresponds to about 1.7×10⁻⁹ in a_μ (derived: 715.1 − 697.7 = 17.4×10⁻¹⁰).
- α: the engine should know the Cs/Rb split. The relative difference is (137.035999206 − 137.035999046)/137.036 ≈ 1.2 ppb (derived), about 6× the Cs uncertainty. Choosing CODATA 2022 (which averages them) hides this.

### Gaps
- I found no in-window electron g−2 or α measurement. The 2023 Northwestern electron g−2 value (0.13 ppt) was not re-fetched.
- The numerical value of the 2026 lattice NLO-HVP result (JHEP 09 (2026) 038) was not extracted.
- I found no MUonE first-physics result or J-PARC g−2 status update in the window.
- I found no outcome document from the August 2026 Theory Initiative workshop.

---

## Q3. What neutrino results appeared (KATRIN mass bound, JUNO first results, NOvA/T2K, IceCube, KM3NeT, sterile searches, 0νββ)?

### Takeaway
JUNO dominated the window. Its first paper appeared in Nature on 10 June 2026. At Neutrino 2026 (UC Irvine, 22–26 June) it released a 207-day analysis, posted as arXiv:2609.37895 on 29 Sep 2026. This gave:
- the first reactor-only Δm²₃₁ at about 1%: +2.509^{+0.027}_{-0.025}×10⁻³ eV² for normal ordering;
- improved solar parameters: sin²θ₁₂ = 0.3036 ± 0.0064 and Δm²₂₁ = (7.388 ± 0.078)×10⁻⁵ eV²;
- a preference for normal ordering of at least 2.1σ (Bayes factor about 10) when combined with long-baseline data.

KATRIN released no new mass bound; m_β < 0.45 eV still stands and the final dataset is due in 2027. Light-sterile-neutrino explanations kept losing ground (MicroBooNE, KATRIN, IceCube). No new 0νββ limits were found in the window.

### Cited Findings
- **[IN WINDOW, journal: Nature 654, 343–348, published 10 Jun 2026; DOI 10.1038/s41586-026-10538-z] JUNO Collaboration, "Measurement of reactor neutrino oscillation with the first JUNO data".**
  - 59.1 days of data. sin²θ₁₂ = 0.3092 ± 0.0087 and Δm²₂₁ = (7.50 ± 0.12)×10⁻⁵ eV² for normal ordering.
  - This is a factor-1.6 precision gain over all previous measurements combined: 1.55% on Δm²₂₁ (KamLAND had 2.5%) and 2.81% on sin²θ₁₂ (SK+SNO had 4.6%).
  - Detector: 20 kt liquid scintillator at a 52.5 km baseline, energy resolution about 3% at 1 MeV.
  - Sources: [Nature](https://www.nature.com/articles/s41586-026-10538-z); [arXiv:2511.14593](https://arxiv.org/html/2511.14593v1); [CAS news](https://english.cas.cn/newsroom/headlines/202606/t20260611_1161560.shtml)
- **[IN WINDOW, preprint, 29 Sep 2026; first shown at Neutrino 2026] JUNO, "Measurement of Neutrino Oscillation Parameters at JUNO and Indication of the Neutrino Mass Ordering", arXiv:2609.37895.**
  - Data: 207.2 live days (30 Aug 2025 – 18 May 2026), 8,294 IBD candidates.
  - **Δm²₃₁ = +2.509^{+0.027}_{-0.025}×10⁻³ eV² (NO) and −2.482^{+0.025}_{-0.026}×10⁻³ eV² (IO)**, about 1%. The Daya Bay constraint is used only to select the correct local minimum.
  - **sin²θ₁₂ = 0.3036 ± 0.0064 (2.1%) and Δm²₂₁ = (7.388 ± 0.078)×10⁻⁵ eV² (1.1%).**
  - With the NuFIT long-baseline Δm²₃₁ constraint, NO is favoured with a one-sided significance of at least 2.1σ for all δ_CP. The Bayes factor is 10.18, and the frequentist value about 2.3σ.
  - The analysis includes the first TAO near-detector spectrum: about 2900 IBD per day reactor-on and 419 per day reactor-off.
  - First geoneutrino measurement at about 15% precision.
  - Sources: [arXiv:2609.37895](https://arxiv.org/html/2609.37895); [JUNO 207-day slides (ICHEP)](https://indico.cern.ch/event/1522800/contributions/6985975/attachments/3322412/5948712/JUNO_reactor_oscillation_chenzy.pdf); [JUNO PASCOS slides](https://indico.global/event/14705/contributions/156876/attachments/72533/140875/xlu_20260623_PASCOS_Sheffield_JUNO_202606NewResults.pdf); [JUNO Erice slides](https://indico.gsi.de/event/23757/contributions/101461/attachments/56869/87503/Ludhova_JUNO_ERICE.pdf)
- **[IN WINDOW, Neutrino 2026 talk] NOvA, "Probing the 3-flavor paradigm".**
  - NOvA with JUNO's Δm²₃₂ gives a Bayes factor of 15.74 (94% vs 6%) for NO, "a 1.88σ indication".
  - Quoted Δm²₃₂ (NO): JUNO 2.435 (1.05%) and NOvA 2.431 (1.4%), both ×10⁻³ eV².
  - The NOvA+T2K joint fit weakly prefers IO (Bayes factor 1.3).
  - Source: [NOvA Neutrino 2026 slides](https://indico.global/event/15740/contributions/155585/attachments/72595/140743/202606_Neutrino_NOvA.pdf)
- **[BACKGROUND] NOvA 10-year result (PRL 136, 011802 (2026)).** Δm²₃₂ = 2.431^{+0.036}_{-0.034}×10⁻³ eV² (NO). Δm²₃₂ = −2.479 ± 0.036×10⁻³ eV² (IO). Posterior probability for NO is 87% (1.6σ) with Daya Bay constraints. — [arXiv:2606.14121](https://arxiv.org/html/2606.14121v1)
- **[BACKGROUND, Nature 646, 818 (Oct 2025)] T2K+NOvA joint analysis.**
  - Δm²₃₂ = 2.43^{+0.04}_{-0.03}×10⁻³ eV² (NO) and −2.48^{+0.03}_{-0.04}×10⁻³ eV² (IO). sin²θ₂₃ = 0.56^{+0.03}_{-0.05}.
  - δ_CP 3σ intervals: [−1.38π, 0.30π] (NO) and [−0.92π, −0.04π] (IO).
  - Source: [Nature](https://link.springer.com/article/10.1038/s41586-025-09599-3)
- **[BACKGROUND, no new mass result in window] KATRIN.**
  - m_β < 0.45 eV (90% CL, Lokhov–Tkachov) from 259 days and 36M electrons [Science 388, 180 (2025)]. The best fit is m² = −0.14^{+0.13}_{-0.15} eV², and the Feldman–Cousins limit is 0.31 eV.
  - Data taking ended in October 2025: 1000 days, about 231M electrons. Statistical-only sensitivity is 0.28 eV, with a target below 0.3 eV.
  - **Final release is aimed at 2027.** The TRISTAN keV-sterile phase runs 2026–2027.
  - Sources: [KATRIN Neutrino 2026 slides (23 Jun 2026)](https://indico.global/event/15740/contributions/155591/attachments/72631/140789/1_20260624_KATRIN_Neutrino.pdf); [KATRIN PASCOS 2026](https://indico.global/event/14705/contributions/155014/attachments/72499/140586/Erhardt_PASCOS26_KATRIN.pdf); [CosPA 2026](https://indico.global/event/16063/contributions/149214/attachments/68062/142480/KATRIN%20talk-CosPA2026.pdf)
- **[BACKGROUND, Nature, Dec 2025] KATRIN sterile search.**
  - The BEST+GALLEX+SAGE best fit (Δm²₄₁ ≈ 1.25 eV², sin²2θ ≈ 0.34) is excluded at 96.56% CL.
  - The Neutrino-4 best fit (Δm²₄₁ = 7.3 eV², sin²2θ ≈ 0.36) is excluded at 99.99% CL.
  - The gallium deficit itself remains at more than 5σ.
  - Sources: [Nature s41586-025-09739-9](https://www.nature.com/articles/s41586-025-09739-9); [CERN Courier](https://cerncourier.com/two-strikes-for-the-light-sterile-neutrino/)
- **[IN WINDOW, Neutrino 2026 talk, 22 Jun 2026] MicroBooNE.** The full 3+1 program (ν_e appearance, ν_e disappearance and ν_μ disappearance in one detector) is "all consistent with no sterile neutrino". The new ν_μ-disappearance result excludes part of the IceCube allowed region. MicroBooNE also excludes the dark-neutrino e⁺e⁻ interpretation of the MiniBooNE excess at more than 95% CL. — [MicroBooNE Neutrino 2026 slides](https://indico.global/event/15740/contributions/155581/attachments/72459/140525/20260622_JHJ_Neutrino26_MicroBooNE_nobackup.pdf)
- **[IN WINDOW, preprint, 22 Sep 2026] Joint MiniBooNE + MicroBooNE 3+1 fit with simulation-based inference, arXiv:2609.26965.** The preference for 3+1 is 2.7σ (trials-based). The parameter-goodness-of-fit tension between the two experiments is at least 2.5σ. — [arXiv:2609.26965](https://arxiv.org/html/2609.26965)
- **[IN WINDOW, PhD thesis, not a collaboration result, 11 Sep 2026] IceCube unstable-sterile (ν₄ decay) search with 10.67 years of data, arXiv:2609.12941.** It finds no preference over 3+1 and excludes most of the short-baseline global-fit region at 90% CL. — [arXiv:2609.12941](https://arxiv.org/abs/2609.12941)
- **[IN WINDOW, preprint, 10 Sep 2026; submitted to PRL] IceCube point-source search toward KM3-230213A, arXiv:2609.10931.**
  - It uses 15 years of data and finds no emission in steady or flaring tests.
  - A steady or transient astrophysical point source explaining the event (assuming an E⁻² spectrum) is "strongly constrained".
  - KM3-230213A: E_ν ≈ 220 PeV (90%: 72 PeV – 2.6 EeV), muon energy 120^{+110}_{-60} PeV.
  - Joint UHE flux E²Φ = 7.5^{+13.1}_{-4.7}×10⁻¹⁰ GeV cm⁻² s⁻¹ sr⁻¹, with KM3NeT vs IceCube/Auger tension about 2.5–3σ.
  - Sources: [arXiv:2609.10931](https://arxiv.org/abs/2609.10931); [IceCube news, 23 Sep 2026](https://icecube.wisc.edu/news/research/2026/09/icecube-search-for-the-source-of-km3nets-record-breaking-event/); [KM3 review, arXiv:2605.07641](https://arxiv.org/html/2605.07641)
- **[BACKGROUND] 0νββ best limits.**
  - KamLAND-Zen (full 800 dataset, PRL, published 29 Dec 2025): T½(¹³⁶Xe) > 3.8×10²⁶ yr (90% CL), m_ββ < 28–122 meV, m_lightest < 84–353 meV. — [OSTI PRL](https://www.osti.gov/pages/servlets/purl/3012943)
  - LEGEND-200 + GERDA + MJD (PRL, Jan 2026): T½(⁷⁶Ge) > 1.9×10²⁶ yr, m_ββ < 75–200 meV. Q_ββ = 2039.061 ± 0.007 keV. — [OSTI](https://www.osti.gov/biblio/3010197); [arXiv:2603.12884](https://arxiv.org/pdf/2603.12884)

### Inferences
- **JUNO consistency check (derived).** Δm²₃₂ = Δm²₃₁ − Δm²₂₁ = 2.509 − 0.0739 = 2.435×10⁻³ eV². This matches the 2.435 value on the NOvA slide, so the two documents use consistent conventions. The JUNO (2.435) and NOvA (2.431) values agree to 0.2% under NO, which is the physical origin of the NO preference.
- **Hook.** A reactor survival-probability engine is
  P_ee = 1 − cos⁴θ₁₃ sin²2θ₁₂ sin²Δ₂₁ − sin²2θ₁₃ (cos²θ₁₂ sin²Δ₃₁ + sin²θ₁₂ sin²Δ₃₂), with Δ_ij = 1.267 Δm²_ij[eV²] L[m]/E[MeV].
  Evaluated with JUNO's 207-day parameters at L = 52.5 km, it should reproduce the oscillation pattern. This is the standard formula, not quoted from the sources above.
- The neutrino mass scale is still bounded only at 0.45 eV (direct) and m_ββ ≲ 28–122 meV (Majorana-dependent). Any σ-field mechanism that generates neutrino mass must sit below these numbers.

### Gaps
- I found no in-window 0νββ limit updates (CUORE/CUPID, LEGEND-200 second release, NEXT, SNO+). They may have been shown at Neutrino 2026, but I did not locate them.
- I found no new in-window IceCube or KM3NeT oscillation or atmospheric parameter values, and no KM3NeT event re-localisation results.
- The SBND first oscillation results and the JUNO-TAO standalone sterile results were not located.

---

## Q4. Were there new precision measurements of fundamental constants or particle masses (CODATA, proton radius, neutron lifetime beam-vs-bottle, G)?

### Takeaway
There were no new CODATA, G or neutron-lifetime results in the window. The key reference values all date from early 2026 or earlier:
- The 2026 hydrogen 2S–6P measurement effectively closed the proton-radius puzzle: r_p = 0.8406(15) fm, matching muonic hydrogen at 0.84060(39) fm.
- The neutron-lifetime beam–bottle split persists at about 5σ: 888.1 ± 2.0 s vs 877.8 ± 0.3 s.
- PDG 2026 (1 June 2026) gives first-row CKM unitarity of 0.9983(7), about 2σ low.

### Cited Findings
- **[BACKGROUND, Nature, Feb 2026] Maisenbacher et al. (MPQ), "Sub-part-per-trillion test of the Standard Model with atomic hydrogen".**
  - ν(2S–6P) = 730 690 248 610.79(48) kHz, against an SM prediction of 730 690 248 610.79(23) kHz. This tests the SM to 0.7 ppt and bound-state QED to 0.5 ppm.
  - **r_p = 0.8406(15) fm.** It is 2.5× more precise than any other atomic-H result, matches muonic H at 0.84060(39) fm, and lies 5.5σ from CODATA 2014.
  - 1S Lamb shift: 8 172 744.1(3.6) kHz measured, 8 172 744.1(1.7) kHz predicted.
  - CODATA 2022 gives r_p = 0.84075(64) fm.
  - Sources: [Nature](https://preview-www.nature.com/articles/s41586-026-10124-3); [PMC copy](https://pmc.ncbi.nlm.nih.gov/articles/PMC12935534/)
- **[BACKGROUND, PRL, Apr 2026] Bullis et al. (2S–nS transitions).** r_p = 0.8433 fm and R∞ = 3 289 841 960 252.9 kHz. — [phys.org summary](https://phys.org/news/2026-04-year-proton-mystery-ultra-precise.html)
- **[BACKGROUND, arXiv:2604.15546] G. A. Miller, "The Proton Radius Puzzle".**
  - The average of electronic-H results is r_p = 0.843 ± 0.0013 fm, and the review concludes "the proton radius puzzle is no more".
  - PRad gave 0.831 ± 0.007 ± 0.012 fm.
  - **PRad-II ran 26 Mar – 5 Jul 2026, targeting 0.43% total uncertainty. No result yet.**
  - Source: [arXiv:2604.15546](https://arxiv.org/abs/2604.15546)
- **[BACKGROUND, CERN Courier, Jan 2026] Neutron lifetime.**
  - Beam average 888.1 ± 2.0 s vs magnetic-bottle 877.8 ± 0.3 s, about 5σ.
  - UCNτ final trap-door result: 877.82 ± 0.22 (stat) ^{+0.20}_{-0.17} (sys) s.
  - J-PARC electron-counting beam: 877.2 ± 1.7 (stat) ^{+4.0}_{-3.6} (sys) s, which is bottle-like and 2.3σ from proton-counting beams.
  - Exotic-decay explanations are "largely excluded". NIST BL3 is commissioning in 2026, and τSPECT (PSI) has started data taking.
  - Sources: [CERN Courier](https://cerncourier.com/the-beam-bottle-debate-at-psi/); [UCNτ arXiv:2409.05560](https://arxiv.org/pdf/2409.05560); [J-PARC arXiv:2412.19519](https://arxiv.org/pdf/2412.19519); [Wietfeldt review](https://www.mdpi.com/2073-8994/16/8/956)
- **[BACKGROUND, PDG 2026 review dated 1 Jun 2026] V_ud, V_us and CKM unitarity.**
  - |V_ud|² + |V_us|² + |V_ub|² = 0.9983(7), about 2σ. It would be about 3σ if nuclear-structure uncertainties were ignored.
  - V_us(kaons) = 0.22431(85), with a scale factor of 2.5 from the K_ℓ3 vs K_μ2 tension. V_us(τ) = 0.2207(14). G_μ = 1.1663787(6)×10⁻⁵ GeV⁻².
  - Source: [PDG 2026 V_ud/V_us review](https://pdg.lbl.gov/2026/reviews/rpp2026-rev-vud-vus.pdf)
- **[BACKGROUND] PDG 2026 masses quoted above:** m_W = 80.3625 ± 0.0077 GeV; m_t (LHC Run-1 combination) = 172.52 ± 0.33 GeV; m_Z = 91188.0 ± 2.0 MeV (quoted in CMS's Nature paper). — [PDG W listing](https://pdg.lbl.gov/2026/listings/rpp2026-list-w-boson.pdf); [CMS Nature](https://www.nature.com/articles/s41586-026-10168-5)
- **[IN WINDOW, journal published 1 Sep 2026; DOI 10.1103/l4bh-ryxs; arXiv:2604.27963] International optical clock comparison over the European fibre network (INRIM, LNE-OP, NPL, PTB; 7 clocks).**
  - Frequency ratios are measured at 7.7×10⁻¹⁸ to 6.1×10⁻¹⁷.
  - Two independently built ¹⁷¹Yb⁺(E3) clocks (NPL vs PTB) agree to 7.7×10⁻¹⁸: ratio 0.999 999 999 999 999 992 9(77). This is the first international sub-10⁻¹⁷ verification.
  - Example ratios: Yb⁺(E3)/⁸⁷Sr = 1.495 991 618 544 900 560(22); ¹⁹⁹Hg/Yb⁺(E3) = 1.757 572 821 468 313 043(99); Yb⁺(E2)/Yb⁺(E3) = 1.072 007 373 634 205 480(29).
  - Sources: [APS full text](https://harvest.aps.org/v2/journals/articles/10.1103/l4bh-ryxs/fulltext); [arXiv:2604.27963](https://arxiv.org/html/2604.27963)
- **[BACKGROUND, Apr 2026] NIST/JILA: ²⁷Al⁺, ¹⁷¹Yb and ⁸⁷Sr frequency ratios with uncertainty ≤ 3.2×10⁻¹⁸**, meeting the criteria for redefining the SI second. They also report discrepancies of 1×10⁻¹⁶ (Sr ratios) and 1.5×10⁻¹⁷ (Al⁺/Yb) relative to the group's own earlier measurements. — [NIST](https://www.nist.gov/publications/optical-clock-frequency-ratios-uncertainty-32-10-18)

### Inferences
- **Hooks.**
  - Use r_p = 0.8406–0.8408 fm. Do not use the pre-2018 value of about 0.875 fm.
  - The finite-size shift of S levels scales as r_p²/n³, so the 2S–6P frequency is a sub-ppt consistency test of any atomic-structure module.
  - τ_n: any σ-field mechanism that changes the neutron–proton mass difference or g_A shifts τ_n. The roughly 10 s beam–bottle gap (about 1.2%) is the largest unexplained "lifetime" discrepancy. The bottle value of about 878 s is the one consistent with BBN and CKM fits, per the [Aug 2026 triangulation preprint](https://figshare.com/articles/online_resource/Examining_the_Neutron_Lifetime_Puzzle/33261597). That preprint is not peer-reviewed.
- Optical clock ratios at about 10⁻¹⁸ are the most precise dimensionless numbers in physics. Any model in which α or μ = m_p/m_e drifts or oscillates must keep these ratios constant at that level over the campaign timescales (see Q5).

### Gaps
- I found no in-window CODATA update. CODATA 2022 (Mohr et al. 2025) remains current.
- I found no new determination of Newton's G in the window.
- I found no new neutron-lifetime numbers in the window, from τSPECT, UCNτ+, BL3 or J-PARC.
- The PRad-II result has not been released.

---

## Q5. Were there new tests of varying fundamental constants (α, μ = m_p/m_e) from atomic clocks, quasar spectra, or the thorium-229 nuclear clock? [σ-FLAG — highest relevance]

### Takeaway
This is the most σ-field-relevant area of the window. **Two independent groups demonstrated working ²²⁹Th nuclear clocks** (Nature, 7 Oct 2026), stabilising a laser to the 148-nm nuclear transition at f₀ ≈ 2.0204×10¹⁵ Hz:
- TU Wien/PTB/BEV used the clock to constrain ultralight scalar dark matter coupling to photons (d_e) and to the strong sector (d_g). The d_g limits go beyond previous measurements.
- The nuclear transition energy results from a near-cancellation of MeV-scale Coulomb and strong-force energies. Its sensitivity to α is therefore enhanced: K ≈ 5900(2300), versus about 1 for atomic clocks. It is also enhanced for Λ_QCD and m_q.

I did not find new quasar (ESPRESSO) α or μ results in the window.

### Cited Findings
- **[IN WINDOW, journal: Nature, published 7 Oct 2026, DOI 10.1038/s41586-026-11084-4] Toscani De Col, Riebner, …, Peik, Schumm (TU Wien / PTB / BEV / CMI), "A thorium-229 optical nuclear clock with feedback loop".**
  - A CW 148-nm laser is locked to the ²²⁹Th:CaF₂ transition (|I_g, m = ±5/2⟩→|I_is, m = ±3/2⟩, D-centre) by absorption spectroscopy, and compared to a Yb⁺ ion clock over a fibre link.
  - Instability is **3×10⁻¹²/√(τ/s)**, approaching 10⁻¹⁵ over a day. Linewidth is about 100 kHz. **f₀ = 2.0204×10¹⁵ Hz.**
  - Temperature coefficient is 1.8 kHz/K at 295 K. Run-to-run reproducibility is about 5×10⁻¹³, and the line centre varies by up to 1.7 kHz between crystal positions, attributed to strain. The isomer lifetime in the crystal is about 10 min.
  - **Ultralight-dark-matter constraints** from oscillations on 20 s – 1 day timescales (t_tot ≈ 23 h). The coupling formula is
    d_e = (A_f / k_Th^α) · (m_φ / 2.40×10⁻³¹ eV/c²),
    where A_f is the measured fractional oscillation amplitude. For sensitivity factors k ~ 10³–10⁵, the d_e limits "compete with the best atomic clocks", and the Λ_QCD (d_g) limits "go beyond previous measurements".
  - The paper states that "couplings to the electromagnetic force, strong force or quarks would lead to an apparent oscillation of α, Λ_QCD and m_q", and that the isomer's low energy arises from "a coincidental near-cancellation of the mega-electronvolt-scale electromagnetic and strong nuclear force effects".
  - Source: [Nature s41586-026-11084-4](https://www.nature.com/articles/s41586-026-11084-4)
- **[IN WINDOW, journal: Nature, published 7 Oct 2026, DOI 10.1038/s41586-026-11122-1] Huang, Yan, Xiao, Bu, …, Ding (Tsinghua et al.), "A nuclear clock synchronized to ²²⁹Th".**
  - The source is a 10 µW CW 148.4 nm VUV laser made by four-wave mixing in Cd vapour.
  - Instability is **5×10⁻¹³/√(τ/s)**.
  - **Clock frequencies from two independently fabricated crystals agree at the 10⁻¹³ level**, consistent with earlier VUV-comb measurements.
  - Source: [Nature s41586-026-11122-1](https://www.nature.com/articles/s41586-026-11122-1)
- **[IN WINDOW, journal: Nature, 16 Sep 2026, DOI 10.1038/s41586-026-11011-7] "Continuous-wave laser absorption spectroscopy of the thorium-229 nucleus".** It observes six absorption lines (five from a dihedral thorium-dimer D-centre) and finds some line centres more than 3σ from earlier values, attributed to doping variation. — [Nature s41586-026-11011-7](https://www.nature.com/articles/s41586-026-11011-7)
- **[BACKGROUND, Nat. Commun., Oct 2025] Beeks et al.** Sensitivity of the ²²⁹Th transition to α: **K = 5900(2300)** in δν/ν = K δα/α. This comes from ΔQ₀/Q₀ = 1.791(2)% and is limited by the measured charge-radius change Δ⟨r²⟩. It implies about a 10³× enhancement over electronic clocks. — [Nat. Commun.](https://www.nature.com/articles/s41467-025-64191-7); [arXiv:2407.17300](https://arxiv.org/html/2407.17300)
- **[BACKGROUND, PRA 102, 052833 (2020)] Earlier estimate K = −(0.82 ± 0.25)×10⁴.** The paper says that for δα/α ≲ 10⁻¹⁷/yr the shift would be about 200 Hz/yr. — [arXiv:2007.00408](https://ar5iv.labs.arxiv.org/html/2007.00408)
- **[IN WINDOW, INSPIRE, 16 Sep 2026] "Sensitivity of the ²²⁹ᵐTh clock transition to the fine-structure constant in a Skyrme–Hartree–Fock–BCS approach".** A new nuclear-theory K estimate; I could not extract the value. — [INSPIRE 3204565](https://inspirehep.net/literature/3204565)
- **[IN WINDOW] European fibre-linked clock campaign** (see Q4). It provides 11 new optical frequency ratios with correlation matrices at 7.7×10⁻¹⁸ – 6.1×10⁻¹⁷. These are inputs to α-variation fits, because Yb⁺(E3) is strongly α-sensitive. — [arXiv:2604.27963](https://arxiv.org/html/2604.27963)
- **[BACKGROUND] Laboratory drift limits (pre-window reference values).**
  - From Yb⁺ E3/E2 vs Cs: (1/α)dα/dt = −0.20(20)×10⁻¹⁶/yr and (1/μ)dμ/dt = −0.5(1.6)×10⁻¹⁶/yr. — [Huntemann et al. 2014](https://www.emergentmind.com/papers/1407.4408)
  - From a 12-year Sr-vs-Cs record: μ̇/μ = −6.9(6.5)×10⁻¹⁷/yr. Gravitational-potential coupling k_μ = 3.5(5.9)×10⁻⁷. Sensitivity coefficients K_α(Sr, Cs) = −2.77, K_μ(Sr, Cs) = 1, K_Xq(Sr, Cs) = −0.002. — [Schwarz et al., PRR 2, 033242](https://www.ptb.de/cms/fileadmin/internet/fachabteilungen/abteilung_4/4.4_zeit_und_frequenz/pdf/PhysRevResearch.2.033242.pdf)

### Inferences
- **Direct σ-field hook (derived).** For a field that shifts α by δα/α, the Th-229 clock frequency shifts by δν ≈ K f₀ (δα/α) ≈ 5900 × 2.02×10¹⁵ Hz × (δα/α) ≈ 1.2×10¹⁹ Hz × (δα/α).
  - So δα/α = 10⁻¹⁸ gives about 12 Hz. With the current reproducibility of about 5×10⁻¹³ (about 1 kHz), the clock already tests δα/α at about 10⁻¹⁷-equivalent per run, before systematics, within about 2.5× (K uncertainty).
  - Because the isomer energy (about 8.36 eV) is a difference of MeV-scale Coulomb and nuclear terms, a σ-field that shifts hadron masses through Λ_QCD or m_q is also strongly amplified. Th-229 is the single most sensitive laboratory probe of "scale-transition" scalar fields.
- **Ratio test hook.** Any σ-field that changes α or μ must leave the Yb⁺(E3)/Sr, Hg/Yb⁺ and Al⁺/Yb ratios fixed at 10⁻¹⁷–10⁻¹⁸ over months. The pre-window drift limits (|α̇/α| ≲ 2×10⁻¹⁷/yr, |μ̇/μ| ≲ 10⁻¹⁶/yr) should be hard-coded as constraints.
- **Scalar–matter coupling convention.** The dark-matter clock literature parameterises a scalar φ by d_e (photon), d_g (gluon/Λ_QCD), d_mq (quark mass) and d_me (electron mass). A σ field that "shifts particle/hadron masses" maps onto d_g and d_mq. These are exactly the couplings the TU Wien result says it constrains "beyond previous measurements".

### Gaps
- I found no in-window quasar or ESPRESSO α or μ measurements. A targeted search might find some, but none surfaced.
- The numerical d_e and d_g exclusion values (as functions of m_φ) from the Th-229 clock were not extractable as text; they appear only in the Nature paper's Fig. 4.
- The 2026 Skyrme–HF–BCS K value was not obtained.
- I found no updated combined global fit of α̇/α or μ̇/μ dated 2026. The values above are 2014/2020 reference limits.

---

## Q6. Were there nuclear physics results on nuclear EOS, exotic hadrons (tetraquarks, pentaquarks, glueballs), or the quark-gluon plasma?

### Takeaway
- **QGP:** ATLAS observed jet quenching (dijet imbalance, above 5σ) in O+O and Ne+Ne, the smallest systems in which it has been seen. CMS published (PRL, Aug 2026) geometry-driven flow in O+O and Ne+Ne that is sensitive to the "bowling-pin" ²⁰Ne shape. All four LHC experiments now report QGP signatures in light-ion collisions.
- **Exotic hadrons:** the B_s0*(5700) discovery (Q1) and LHCb's "six new hadrons in 2026".
- **EOS:** new NICER mass-radius constraints (August 2026) shrink heavy-neutron-star radii by about 0.4 km.
- **Hadron mass decomposition:** an August 2026 PRL gives the first lattice validation of the trace-anomaly mass sum rules (for charmonium). The lattice–phenomenology σ_πN tension (about 42 vs 59 MeV) persists.

### Cited Findings

#### Quark–gluon plasma / light ions
- **[IN WINDOW, ATLAS briefing, 28 Jun 2026] ATLAS observes jet quenching in O+O and Ne+Ne at √s_NN = 5.36 TeV.** The dijet momentum balance x_J = p_T,2/p_T,1 is modified in central collisions at more than 5σ. The most central light-ion events resemble Pb+Pb with a similar QGP size. — [ATLAS briefing](https://www.atlas.cern/index%2ephp/Updates/Briefing/OO-NeNe-Jet-Quenching); [arXiv:2606.20463 (ICHEP list)](https://atlas.cern/Updates/News/Summary-ICHEP-2026)
- **[IN WINDOW, journal: PRL 137, 082302, published 17 Aug 2026; DOI 10.1103/26wx-tg6f] CMS, "Observation of long-range collective flow in O+O and Ne+Ne collisions".**
  - Data: 7 nb⁻¹ O+O and 0.8 nb⁻¹ Ne+Ne at 5.36 TeV.
  - Significant v₂ and v₃ are seen. The Ne/O v₂ ratio rises strongly toward central collisions, qualitatively consistent with a deformed ²⁰Ne, while the v₃ ratio falls.
  - Hydrodynamics with ab initio (NLEFT/PGCM) nuclear structure reproduces the trends but not the full magnitudes.
  - Sources: [OSTI PRL](https://www.osti.gov/servlets/purl/3484604); [CMS HIN-25-009](https://cms-results.web.cern.ch/cms-results/public-results/publications/HIN-25-009/index.html)
- **[IN WINDOW (approx.), ALICE news "one year on" from the July 2025 O+O run] All four LHC experiments report QGP signs in O+O/Ne+Ne:**
  - CMS charged-hadron suppression and sequential Υ suppression
  - LHCb stronger charm-meson suppression in Ne+Ne than in O+O
  - ALICE π⁰ suppression in O+O vs p+O, called "unambiguous evidence for parton energy loss"
  - ALICE baryon-vs-meson anisotropic flow (preliminary)
  - Source: [ALICE news](https://alice-collaboration.web.cern.ch/2026-oxygen-collisions)
- **[BACKGROUND] ALICE R_OO(π⁰)** is suppressed below unity, 2.4σ beyond cold-nuclear-matter expectations using EPPS21. — [arXiv:2604.07184](https://arxiv.org/html/2604.07184v1)
- **[IN WINDOW, ICHEP] Other new heavy-ion results:**
  - CMS Z+jet quenching in pp and PbPb at 5.36 TeV (HIN-25-017)
  - ATLAS J/ψ–jet correlations (arXiv:2606.31375)
  - ATLAS forward-jet suppression (arXiv:2608.03261)
  - ATLAS O+O/Ne+Ne charged-particle dN/dη (arXiv:2606.20257)
  - Numbers were not extracted. Sources: [CMS ICHEP-2026](https://cms.cern/news/cms-ichep-2026); [ATLAS ICHEP-2026](https://atlas.cern/Updates/News/Summary-ICHEP-2026)

#### Exotic and new hadrons
- **[IN WINDOW] LHCb B_s0*(5700)⁰** (see Q1): the first non-conventional single-b hadron candidate, at 5698.9 ± 1.6 MeV with Γ < 9.8 MeV. — [arXiv:2609.25178](https://arxiv.org/abs/2609.25178)
- **[IN WINDOW (talk)] LHCb hadron-spectroscopy summary: "six new hadrons observed by LHCb in 2026".** These include a new decay of Z_c(4430)⁻, a new D_s1(2933)⁺ state, the B_s0*(5700)⁰, and D_s1(2460)⁺/D_s1(2536)⁺ → D_s⁺π⁺π⁻ studies. — [LHCb spectroscopy slides](https://indico.global/event/17070/contributions/162377/attachments/77116/149842/lhcb_hadron_spectroscopy_chenchen.pdf)
- **[BACKGROUND, 17 Mar 2026] LHCb discovers a new doubly charmed baryon (ccd) at 7σ with Run-3 data.** It is about 4× the proton mass and brings the LHC hadron count to 80. — [CERN press release](https://home.cern/lhcb-collaboration-discovers-new-proton-particle/)
- **[BACKGROUND, submitted to Sci. Adv., Feb 2026] CMS, "Observation of a family of all-charm tetraquarks", arXiv:2602.02252 (315 fb⁻¹).**
  - X(6600), X(6900) and X(7100) are each above 5σ; X(7100) is at 7.7σ. Their mutual interference is above 5σ, implying common J^PC.
  - Their squared masses lie on a linear (Regge) trajectory in the radial index, and their widths decrease with n.
  - J^PC = 2⁺⁺ is favoured (Nature, Dec 2025), pointing to aligned spin-1 [cc][c̄c̄] diquarks.
  - Sources: [CMS BPH-24-003](https://cms-results.web.cern.ch/cms-results/public-results/publications/BPH-24-003/index.html); [Nature spin-parity paper](https://www.nature.com/articles/s41586-025-09711-7)

#### Hadron mass decomposition (QCD vs Higgs) [σ-FLAG]
- **[IN WINDOW, journal: PRL, published 7 Aug 2026; DOI 10.1103/5n46-717z] "Lattice-QCD Validation of Hadron Mass and Trace-Anomaly Decomposition Sum Rules".**
  - This is the first lattice computation of all quark and gluon EMT components, with gradient-flow renormalisation, three lattice spacings and two-loop MS-bar matching.
  - For η_c and J/ψ:
    - The gluonic contribution to the mass is about 15% in the Lorcé and Metz–Pasquini–Rodini decompositions.
    - The trace-anomaly term in the Ji sum rule is about 6%.
    - The gluonic component of the trace anomaly in the Hatta–Rajan–Tanaka sum rule is about 35%.
  - It also gives the first lattice result for the gravitational form factor C̄.
  - Source: [PRL](https://link.aps.org/doi/10.1103/5n46-717z)
- **[IN WINDOW, preprint, Sep 2026] Barca, Bali, Collins, Rodekamp, "Nucleon sigma terms at m_π = 222 MeV with a variational analysis from lattice QCD", arXiv:2609.05989.**
  - It states the open tension:
    - phenomenology (Roy–Steiner): σ_πN = 59.1(3.5) MeV
    - FLAG N_f = 2+1: 42.2(2.4) MeV
    - FLAG N_f = 2+1+1: 60.9(6.5) MeV, dominated by PNDME
  - The isospin-convention correction is Δ = 3.1(5) MeV.
  - Sigma terms set the Higgs/scalar–nucleon coupling at zero momentum transfer.
  - Source: [arXiv:2609.05989](https://arxiv.org/pdf/2609.05989)
- **[BACKGROUND] χQCD (PRL 121, 212001, 2018) proton mass decomposition at MS-bar 2 GeV.** Quark energy 32(4)(4)%, gluon field energy 36(5)(4)%, a quarter of the trace anomaly 23(1)(1)%, and u, d, s scalar condensates (quark-mass/Higgs part) 9(2)(1)%. — [PRL 121, 212001](https://link.aps.org/doi/10.1103/PhysRevLett.121.212001)
- **[BACKGROUND, arXiv:2411.18402] Lattice baryon-mass origin study.** Ground-state baryon masses are reproduced within 1%. The flavour-insensitive gluon trace-anomaly contribution is 0.8–1.2 GeV. The Higgs-mechanism (σ-term) contribution is enhanced 4–8× over bare masses for light quarks, 2–3× for strange and 1.2–1.3× for charm. — [arXiv:2411.18402](https://arxiv.org/html/2411.18402)

#### Nuclear equation of state / neutron stars
- **[IN WINDOW, preprint, 31 Aug 2026] Mendes, Svensson, Hebeler, Schwenk, Watts et al., arXiv:2609.00173.**
  - It adds new NICER mass-radius constraints for PSR J1614−2230, PSR J2124−3358 and 47 Tuc X7 to a χEFT + pQCD inference.
  - **Heavy neutron-star radii shift about 0.4 km smaller**, driven by J1614−2230. Pressure and speed-of-sound posteriors tighten, especially near 3n₀.
  - Source: [arXiv:2609.00173](https://arxiv.org/abs/2609.00173)
- **[IN WINDOW, A&A, 1 Sep 2026] NICER-informed low-density matching.** It favours a χEFT matching density n_m ≈ 0.18 fm⁻³ (Δ_m ≈ 0.021 fm⁻³). — [A&A](https://www.aanda.org/articles/aa/full_html/2026/09/aa59810-26/aa59810-26.html)
- **[BACKGROUND, ApJL, Mar 2026] PSR J0437−4715 reanalysis including the non-thermal component.** R = 11.9–15.5 km (68%) at M = 1.418 ± 0.044 M☉. — [ApJL](https://iopscience.iop.org/article/10.3847/2041-8213/ae5057)
- **[BACKGROUND, PRX 15, 021014 (2025)] TOV maximum-mass upper limit of 2.39 M☉**, from the GW170817 remnant. — [PRX](https://link.aps.org/doi/10.1103/PhysRevX.15.021014)

### Inferences
- **σ-field and nucleon-mass hook (derived, Feynman–Hellmann).** If a scalar field rescales the light quark masses by a fraction ε, then δm_N ≈ σ_πN · ε, i.e. ≈ 42–60 MeV × ε depending on which σ_πN is right.
  - If it instead rescales Λ_QCD by ε, then δm_N ≈ (m_N − σ_terms) · ε ≈ 0.85–0.9 GeV × ε.
  - So about 90% of nucleon mass tracks Λ_QCD (gluon/trace anomaly) and only about 9% tracks quark (Higgs) masses, per χQCD.
  - This roughly 10:1 ratio is the key number for any σ-field that distinguishes "QCD binding" from "Higgs mass".
- The σ_πN tension (42 vs 59 MeV, about a 40% spread) is a real input uncertainty. The codebase should carry both values, not average them silently.
- **QGP hook.** v₂{2}(Ne+Ne)/v₂{2}(O+O) > 1 in central collisions is an imprint of α-cluster nuclear geometry: a tetrahedral ¹⁶O and a bowling-pin ²⁰Ne (¹⁶O + α). An engine with nuclear-shape modelling can test whether its ¹⁶O/²⁰Ne densities reproduce the ratio's direction.

### Gaps
- I found no in-window glueball result (for example BESIII X(2370) updates) or new pentaquark observation.
- I found no in-window PREX/CREX symmetry-energy (L) update or new GW (O4) tidal-deformability value.
- I did not extract the numeric R_1.4 from arXiv:2609.00173.
- I found no in-window lattice proton (as opposed to charmonium) mass-decomposition result.

---

## Q7. Which anomalies are currently live, and which were resolved in this window?

### Takeaway
**No anomaly was decisively resolved inside the window**, but several moved:
- The muon g−2 tension, effectively resolved in 2025, is now an internal lattice-vs-e⁺e⁻ conflict. That conflict got sharper (4.6σ for NLO HVP).
- Light-sterile-neutrino explanations of LSND/MiniBooNE/gallium were further disfavoured, though the gallium deficit itself (above 5σ) and a 2.7σ MiniBooNE+MicroBooNE 3+1 preference remain.
- The proton-radius puzzle is regarded as closed as of early 2026.

Live anomalies include:
- R(D(*)) at about 3.8σ
- the CDF W mass, disfavoured at 0.06% compatibility
- the neutron-lifetime beam–bottle gap at about 5σ
- the Cabibbo-angle (CKM first-row) deficit at about 2σ
- Belle II B⁺→K⁺νν̄ at 2.7σ
- the 95 GeV diphoton/ditau hints at about 3σ local, with no 2026 update found
- the KM3NeT UHE event tension at about 2.5–3σ
- the Cs/Rb α discrepancy above 5σ
- the σ_πN lattice/phenomenology split

### Cited Findings
- **Muon g−2:** experiment and the lattice SM agree (Δa_μ = 26(66)×10⁻¹¹ in WP25), but the lattice NLO-HVP is in 4.6σ tension with data-driven evaluations. Status: **moved from "BSM anomaly" to "SM-internal tension"**. — [WP25](https://arxiv.org/abs/2505.21476v2); [JHEP 09 (2026) 038](https://link.springer.com/article/10.1007/JHEP09(2026)038)
- **R(D)/R(D*):** world average about 3.8σ from the SM. R(J/ψ) is now 2.4σ. Belle II R(D*) = 0.242 ± 0.025 is SM-like. Status: **live**. — [Beauty 2026 slides](https://indico.cern.ch/event/1595459/contributions/7068642/attachments/3288622/5879664/4.3%20Beauty2026-MCalvi.pdf)
- **b→sℓℓ angular/branching tensions persist**, for example "reinforcing existing tensions … in branching fractions and angular observables" in B⁰→K*⁰μμ, while LFU ratios (R_K*, R(K*⁰) at high q²) are SM-like. Status: **live but LFU-clean**. — [LHCb rare-decays proceedings](https://cds.cern.ch/record/2958657); [arXiv:2604.08631](https://arxiv.science/abs/2604.08631)
- **Belle II B⁺→K⁺νν̄:** B = (2.3 ± 0.5 ^{+0.5}_{-0.4})×10⁻⁵. That is 3.5σ evidence for the decay and 2.7σ above the SM. A reinterpretation favours an invisible X at m_X = 2.1^{+0.2}_{-0.1} GeV with B·P_inv = 9.2^{+1.8}_{-3.4}×10⁻⁶, at 3.0σ (not a collaboration result). Status: **live, no in-window update found**. — [arXiv:2602.09666](https://arxiv.org/html/2602.09666v1); [Belle II evidence paper](https://arxiv.science/abs/2311.14647)
- **CDF W mass:** excluded from the PDG 2026 world average because combined compatibility is 0.06%. CMS, ATLAS and LHCb all agree with the SM. Status: **effectively resolved against CDF, although not formally explained**. — [PDG 2026 W review](https://pdg.lbl.gov/2026/reviews/rpp2026-rev-w-mass.pdf)
- **Cabibbo-angle anomaly:** first-row sum 0.9983(7), about 2σ (about 3σ ignoring nuclear-structure uncertainties). V_us from K_ℓ3 and K_μ2 disagree, with a scale factor of 2.5. Status: **live**. — [PDG 2026](https://pdg.lbl.gov/2026/reviews/rpp2026-rev-vud-vus.pdf)
- **Neutron lifetime:** beam 888.1 ± 2.0 s vs bottle 877.8 ± 0.3 s, about 5σ. Status: **live**; new BL3/τSPECT data are expected. — [CERN Courier](https://cerncourier.com/the-beam-bottle-debate-at-psi/)
- **Proton radius:** considered resolved in favour of the small radius, r_p ≈ 0.8406 fm. — [Nature 2026](https://preview-www.nature.com/articles/s41586-026-10124-3); [Miller, arXiv:2604.15546](https://arxiv.org/abs/2604.15546)
- **Light sterile neutrinos:**
  - MicroBooNE completed its full 3+1 program with null results (June 2026).
  - KATRIN excludes the gallium best fit at 96.6% and Neutrino-4 at 99.99%.
  - The joint MiniBooNE+MicroBooNE fit still prefers 3+1 at 2.7σ, with at least 2.5σ inter-experiment tension (Sep 2026).
  - Status: **the single light sterile neutrino is strongly disfavoured, while the underlying gallium and MiniBooNE excesses are unexplained**.
  - Sources: [MicroBooNE slides](https://indico.global/event/15740/contributions/155581/attachments/72459/140525/20260622_JHJ_Neutrino26_MicroBooNE_nobackup.pdf); [arXiv:2609.26965](https://arxiv.org/html/2609.26965); [CERN Courier](https://cerncourier.com/two-strikes-for-the-light-sterile-neutrino/)
- **95 GeV scalar hints (background):**
  - μ_γγ(ATLAS+CMS) = 0.24^{+0.09}_{-0.08}, about 3.1σ local, combined by phenomenologists while neglecting correlations.
  - CMS ττ at 95 GeV: μ = 1.2 ± 0.5 (2.6σ local).
  - LEP bb̄: μ = 0.117 ± 0.057 (2.3σ).
  - I found no 2026 Run-3 experimental update. Status: **live but stale**.
  - Sources: [PRD 109, 035005](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.109.035005); [arXiv:2509.26155](https://arxiv.org/pdf/2509.26155v2/mailto:changqin@htu.edu.cn)
- **New small excess:** CMS diphoton at 13.6 GeV, 3.5σ local / 1.9σ global (v2 of the paper dated Aug 2026). — [arXiv:2603.03250v2](https://arxiv.org/html/2603.03250v2)
- **Toponium:** what was once a "tt̄ threshold excess" is now an established SM QCD bound-state effect, at more than 5σ in both channels. The ATLAS cross section is 45% above the NRQCD prediction. Status: **resolved as SM physics, with residual normalisation tension**. — [arXiv:2608.19546](https://export.arxiv.org/pdf/2608.19546); [ATLAS arXiv:2601.11780](http://arxiv.org/pdf/2601.11780)
- **KM3-230213A (220 PeV):** about 2.5–3σ tension with IceCube/Auger non-observation. The in-window IceCube search finds no counterpart. Status: **live**. — [arXiv:2609.10931](https://arxiv.org/abs/2609.10931)
- **Electron EDM:** no new result. The best limit is still |d_e| < 4.1×10⁻³⁰ e·cm (90% CL, JILA HfF⁺, 2023), and the next-generation ThF⁺ effort is in development as of DAMOP, June 2026. — [arXiv:2212.11841](https://arxiv.org/html/2212.11841); [DAMOP 2026 abstract](https://meetings-archive.aps.org/damop/2026/s00/159/)

### Inferences
- For a codebase validation suite, the useful classes are:
  - **Pass/fail anchors:** W mass (excluding CDF), proton radius, muon g−2 vs lattice SM, Higgs couplings, CKM to about 0.1%.
  - **Known tensions to reproduce rather than "fix":** R(D(*)), τ_n beam/bottle, Cs/Rb α, σ_πN, e⁺e⁻ vs lattice HVP.
  - Mentat's "scientifically honest" policy fits naturally here. If the engine uses CODATA averages that hide the Cs/Rb or beam/bottle splits, a diagnostic should flag that.
- **The σ-field-relevant anomaly set is small.**
  - None of the current anomalies requires a universal scalar shifting hadron masses.
  - The strongest direct constraints on such a field are now the Th-229 clock and the optical-clock ratios (Q5), the κ_λ bounds on the Higgs potential (Q1), and the absence of any g−2 room (Q2).

### Gaps
- I could not confirm any Run-3 update to the 95 GeV excess, the 650 GeV hints, or the ATLAS/CMS ditau excess in the window.
- I found no in-window PADME X17 or ATOMKI update.
- I found no in-window Belle II update to B→Kνν̄ with more data.
- A dedicated 2026 "anomaly scoreboard" review (for example from CERN Courier or HFLAV 2026) was not located.
