# Fluid dynamics and gas-expansion dynamics: new results, ~June 10 – Oct 10, 2026 (with testable hooks for sigma-ground)

Conventions used below:
- **[IN-WINDOW]** means the item first appeared between about June 10 and October 10, 2026. **[BACKGROUND]** means it is older, or I could not confirm an in-window date. These are given only as reference values and benchmarks.
- **Status** values: peer-reviewed (journal), preprint (arXiv or other), or announcement.
- **Hook** gives the concrete formula, dimensionless number or benchmark value that a Python engine could implement and check.
- Research method: live web search and page fetches on 2026-10-10. Most arXiv pages were reached through Exa fetch, because direct WebFetch to arxiv.org and phys.org failed with DNS errors. Where only a secondary source was available, I say so.

---

## Q1. What is new in turbulence and in Navier–Stokes/Euler regularity? (Kolmogorov corrections, intermittency, pipe-flow transition, drag crisis, wall turbulence, blow-up proofs)

### Takeaway
The biggest event in the window is the week of 7–9 September 2026. Four results appeared in three days:
- Alpöge and Buckmaster, assisted by LLMs and building on Córdoba–Martínez-Zoroa, proved finite-time blow-up with smooth forcing for IPM, 2D Boussinesq and 3D Euler.
- OpenAI announced an AI-generated proof, formalized in Lean, of finite-time blow-up for the *forced* 3D Navier–Stokes equations (Clay statements C/D) and for the *unforced* 3D Euler equations.
- The Caltech group published a PINN-found candidate for a stable self-similar Euler singularity at λ = 1/2.

The unforced Navier–Stokes problem (Clay statements A/B) remains open, and none of these results was peer-reviewed as of mid/late September. On the classical side, the new results with crisp numbers are:
- a Re⁻³ scaling for the minimal transition energy in pipe flow, with prefactor 6.09×10⁵;
- a proposed onset of homogeneous pipe turbulence at Re ≈ 2900 ("anti-puffs");
- the first direct observation of Kelvin-wave turbulence on a classical vortex (PRL, 21 Sep 2026).

### Cited Findings

**Navier–Stokes / Euler singularities (Sept 2026 cluster)**
- [IN-WINDOW] **OpenAI, "On the Navier–Stokes Millennium Prize Problem" (announcement plus write-up and Lean code), 8 Sep 2026.**
  - Claim: an internal system "produced an analytical proof and a Lean formalization that an initially smooth fluid at rest can develop a singularity in a finite time". A smooth force is applied and energy stays finite throughout. This establishes Clay statement "C" (and also "D").
  - The construction is "a vortex… that spirals inward and gets increasingly elongated, like spaghetti".
  - The model used is described as "significantly more capable than GPT‑6 Astra". The effort began 1 Sep 2026 after rumors about Alpöge/Buckmaster, and the Lean verification was completed 6 Sep. The NS group involved "on the order of 10,000 concurrent agents".
  - Source: [OpenAI](https://openai.com/index/navier-stokes-solution/). Lean repo: [openai/NavierStokesAndEuler](https://github.com/openai/NavierStokesAndEuler).
- [IN-WINDOW] **Process numbers reported by press.** The run took 88 hours, about 2.7 million agent messages and about 130 billion tokens, and the Lean translation took a further 17 hours. Karniadakis (Brown) told Nature that "for air, the singularity appears when [the] vortex becomes around 70 nanometres wide". Status: the proof was still being checked by mathematicians as of 24 Sep 2026. Source: [Tufts Daily, 24 Sep 2026](https://www.tuftsdaily.com/article/2026/09/mathematicians-still-checking-the-navier-stokes-proof-that-openai-claims-to-have-solved). This is a secondary source.
- [IN-WINDOW] **Clay Institute response.** Three days after the announcement, the Clay Institute said the problem had "apparently been settled". Its review is "deliberately unhurried", and recognition requires peer-reviewed publication. Sources: [ScienceABC](https://www.scienceabc.com/pure-sciences/navier-stokes-openai-singularity-what-was-proved-what-is-open); [Tufts Daily](https://www.tuftsdaily.com/article/2026/09/mathematicians-still-checking-the-navier-stokes-proof-that-openai-claims-to-have-solved). Both are secondary, and I did not find the Clay statement itself.
- [IN-WINDOW] **OpenAI, "Finite time blowup for the Euler equation" (preprint PDF), Sep 2026.**
  - Claim: there exists u₀ ∈ C_c^∞(ℝ³), divergence-free, with 0 < T*(u₀) < ∞ for the *unforced* incompressible Euler equations. ‖∇u(t)‖_∞ is unbounded as t → T*, and ∫₀^{T*}‖curl u‖_∞ dt = ∞.
  - The construction is an iterative "parent/child" scheme of localized oscillations amplified by the parent shear.
  - About 100 agents worked for about 50 hours on this result.
  - Sources: [OpenAI Euler PDF](https://cdn.openai.com/pdf/315b36cd-ec98-4023-8342-93345194ece1/euler.pdf); [OpenAI announcement](https://openai.com/index/navier-stokes-solution/).
- [IN-WINDOW] **Alpöge and Buckmaster, three manuscripts released 7–8 Sep 2026: finite-time blow-up with smooth forcing for IPM, 2D inviscid Boussinesq and 3D incompressible Euler.**
  - Euler (Theorem 1.1): axisymmetric data with swirl, supported in a solid torus. Circulation and meridional velocity stay bounded, while ‖∇Γ‖_∞ and ‖ω‖_∞ → ∞ and ∫‖ω‖_∞dt = ∞ (Beale–Kato–Majda).
  - The work is Lean-formalized in tristanbuckmaster/fluid_lean. The repository notes that the build needs about 100 GB of RAM.
  - Sources: [VibeMathed record](https://vibemathed.com/problem/euler-blowup-smooth-forcing); [Buckmaster statement](https://cims.nyu.edu/~tristanb/statement.pdf).
  - LLM use: Claude, plus OpenAI Codex/GPT‑5.6 Sol. "Astra" was used only for write-ups and audits. The first blow-up was found 15 Aug 2026 and Lean-verified 22 Aug. Credit for the program is given to Córdoba and Martínez-Zoroa. Source: [Buckmaster statement](https://cims.nyu.edu/~tristanb/statement.pdf).
- [IN-WINDOW] **Alpöge, Buckmaster and Coiculescu, "Extending the Córdoba–Martínez-Zoroa IPM Blow-Up to Uniformly Space-Time Smooth Forcing", arXiv:2609.16470.** On 𝕋², with a smooth odd initial density and a force F ∈ C^∞([0,1]×𝕋²), the density gradient and the velocity gradient diverge in L^∞ as t↑1. Nevertheless, ρ(t) converges in C^η for every η < 1. Source: [arXiv:2609.16470](https://arxiv.org/pdf/2609.16470).
- [IN-WINDOW] **Terence Tao's assessment (blog, 7 Sep 2026).** Tao calls the work a breakthrough on model equations and says forced and then unforced NS look "very feasible… in the near future". In the Boussinesq ansatz the solution is linear in space plus a high-frequency plane wave, and the problem reduces to explicit ODE modulation equations. Source: [Tao blog](https://terrytao.wordpress.com/2026/09/07/finite-time-blowup-with-smooth-forcing-term-for-the-incompressible-porous-medium-boussinesq-and-incompressible-euler-equations/).
- [IN-WINDOW] **Priority dispute.**
  - Buckmaster alleges that OpenAI pressured him over authorship and credit.
  - OpenAI says its effort began only after it heard rumors on 1 Sep. It also says that its investigation confirmed Buckmaster's Codex prompts "could not have influenced the system". OpenAI acknowledges Alpöge/Buckmaster's priority on forced Euler.
  - Sources: [Buckmaster statement](https://cims.nyu.edu/~tristanb/statement.pdf); [OpenAI](https://openai.com/index/navier-stokes-solution/). An aggregator reports that OpenAI's Bubeck called the allegations "false and inflammatory": [postcutoff.com](https://postcutoff.com/p/2026-09-08-tristanbuckmaster-blowup-results-statement/). That is a secondary source.
- [IN-WINDOW] **Ganeshram, Duruisseaux and Anandkumar, "Stable Singularity of the Euler Equations on ℝ³", arXiv:2609.10867 (9 Sep 2026), plus companion stability framework arXiv:2609.10860.**
  - A PINN with a traveling-wave self-similar ansatz finds an approximate *unforced*, boundary-free Euler blow-up profile at the critical rate λ = 0.5.
  - The profile is converted to splines, and Arb-certified residuals are O(10⁻⁵) in L² and O(10⁻³) in L^∞. The PINN residual MSE reaches about 10⁻¹⁰ with SS-Broyden.
  - When λ is left free, it converges near 0.5. For the convection-weighted family it tends to 0.5 as ε→1; Hou et al. had found singular behavior only for ε ∈ [0, 0.3).
  - Status: evidence plus a *conditional* proof framework. The constants are not yet certified.
  - Sources: [arXiv:2609.10867](https://arxiv.org/abs/2609.10867); [paper PDF](https://anima-ai.org/wp-content/uploads/2026/09/Euler.pdf); [arXiv:2609.10860](https://arxiv.org/html/2609.10860v1).
- [IN-WINDOW] **Petrillo and Glimm, "The Positive Defect Problem…", arXiv:2609.23868 (20 Sep 2026).**
  - The paper states that the *unforced* NS statements (A)/(B) "stand open". It recasts them as a positive energy-defect target, equivalent to a floor on Littlewood–Paley shell energy flux.
  - Its pseudo-spectral searches at 128³ and 256³ find that the fine-shell flux floor "holds to within 11–6% of the ceiling", persists for a third of a turnover time, and "fails in scale at the Kolmogorov wavenumber".
  - The implications are proven in Lean 4/Mathlib.
  - Source: [arXiv:2609.23868](https://arxiv.org/abs/2609.23868).
- [BACKGROUND] **Chen and Hou: computer-assisted proof of Luo–Hou blow-up for 3D Euler with a boundary (2022).** Source: [anima-ai blog summary](https://anima-ai.org/2026/09/07/stable-singularity-of-the-euler-equations-on-r3-without-forcing/).

**Kelvin-wave turbulence (classical vortex as quantum-turbulence analog)**
- [IN-WINDOW] **Barckicke, Gissinger and Falcon, "Experimental Evidence of Kelvin-Wave Turbulence along a Vortex Core", Phys. Rev. Lett. 137, 134001, DOI 10.1103/t3bt-m431, published 21 Sep 2026; arXiv:2607.07535.**
  - Setup: a randomly shaken draining-tank vortex.
  - Result: energy cascades over "roughly a factor of 100" in scale, and six-wave resonant interactions are identified as the transfer mechanism.
  - Sources: [Phys.org](https://phys.org/news/2026-09-whirlpool-tank-reveals-turbulence.html); [arXiv:2607.07535](https://arxiv.org/abs/2607.07535).
  - Competing theory spectra: the six-wave Kozik–Svistunov prediction E_KS ∝ k^(−7/5), and the effective four-wave L'vov–Nazarenko prediction E_LN ∝ k^(−5/3). Source: [arXiv HTML](https://arxiv.org/html/2607.07535).
  - **Hook:** Rankine-vortex Kelvin-wave (m=1) dispersion in the long-wave limit, ω = −(Γ/4π) k² [ln(2/(k a₀)) − γ_E + 1/4], with γ_E ≈ 0.577 and Ω₀ = Γ/(2π a₀²). Also compare spectral slopes −7/5 and −5/3. Source: [arXiv HTML](https://arxiv.org/html/2607.07535).

**Transition to turbulence in pipe flow**
- [IN-WINDOW] **"Scaling of the minimal energy for turbulence transition in pipe flow", arXiv:2606.23269 (22 Jun 2026).**
  - Method: nonlinear optimal perturbations up to Re = 5000, with Re = R U_c/ν.
  - Optimal perturbations thin as δ_r ∝ Re^(−1/3).
  - Transition requires the perturbation to reach E_tr ≳ 0.13, independent of Re.
  - Minimal-seed threshold: **E₀^crit ≈ 6.09×10⁵ Re⁻³**, in good agreement for Re ≳ 1750.
  - Literature exponents quoted for comparison: −2.0 to −3.0 generally, and −2.7 for plane Couette.
  - Background constant: sustained turbulence for Re ≳ 2040 (Avila et al. 2011).
  - **Hook:** E₀^crit(Re) = 6.09e5·Re⁻³ and Re_c ≈ 2040.
  - Source: [arXiv:2606.23269](https://arxiv.org/html/2606.23269).
- [IN-WINDOW] **"Laminar gaps mirror turbulent puffs in pipe flow", arXiv:2608.04823 (5 Aug 2026).**
  - DNS at Re = 2400–2550.
  - Laminar gaps ("anti-puffs") have exponentially distributed lifetimes.
  - The authors propose a sharp transition to homogeneous turbulence at **Re ≈ 2900**. The puff regime is Re ∈ (1750, 2300), and the critical point of directed-percolation onset is Re ≈ 2040.
  - **Hook:** a regime map for pipe flow: Re < ~2040 decaying puffs; 2040–2300 puffs; 2300–2900 slugs plus anti-puffs; > 2900 uniform turbulence.
  - Source: [arXiv:2608.04823](https://arxiv.org/html/2608.04823).
- [IN-WINDOW] **Gallorini, Massaro, Schlatter and Quadrio, "Drag reduction and subcritical turbulence in controlled pipe flows", arXiv:2608.30452 (31 Aug 2026).** DNS at Re_b = 4900 with streamwise-travelling waves of wall velocity. Pipe flow relaminarizes but channel flow does not. Controlled puffs persist up to 3× the uncontrolled bulk Re, and their fronts can travel faster than the bulk. Source: [arXiv:2608.30452](https://arxiv.org/abs/2608.30452).
- [IN-WINDOW] **Kamil and Hof, "Inertialess turbulence in viscoelastic pipe flow", arXiv:2609.16492 (15 Sep 2026).** They trace a polymer-pipe instability to Re "four orders of magnitude below the theoretically predicted minimum", so the turbulence is purely elastic. Source: [arXiv:2609.16492](https://arxiv.org/abs/2609.16492).
- [IN-WINDOW] **Kamil, Suresh, Lopez and Hof, "Low inertia limit of elasto-inertial turbulence", arXiv:2609.11581 (10 Sep 2026).** They map the EIT threshold over 3 decades of elasticity number and find center-mode structures at onset. This is the first experimental traveling wave in viscoelastic pipe flow, consistent with the "arrowhead" solution. Source: [arXiv:2609.11581](https://arxiv.org/abs/2609.11581).
- [IN-WINDOW] **Garg and Rosti, "The ultimate state of elastic turbulence", arXiv:2608.26479 (27 Aug 2026).** Triperiodic DNS with Oldroyd-B and FENE-P models shows that dissipation, polymer energy transfer and elastic stresses become independent of the Deborah number as it grows. Source: [arXiv:2608.26479](https://arxiv.org/abs/2608.26479).
- [BACKGROUND, Apr 2026] **"Discontinuous transition to shear flow turbulence" (Yang et al.).** Body forces suppress laminar–turbulent coexistence. In a helical pipe, the coexistence window shrinks to ΔRe ≤ 40. In reduced Re, the transition width falls from Δε ≈ 0.5 (unforced) to < 0.04. Source: [alphaXiv summary](https://www.alphaxiv.org/abs/2311.11474).
- [IN-WINDOW, accepted] **PRL accepted, 14 Sep 2026: "Exact coherent structures as building blocks of turbulence on spatially extended domains".** No numbers were retrieved. Source: [PRL accepted list via search](https://journals.aps.org/prl/accepted).

**Intermittency / Kolmogorov corrections**
- [IN-WINDOW] **"Disentangling intermittent flow structure contributions to anomalous scaling and multifractality in turbulence", arXiv:2607.26896 (29 Jul 2026).** Thresholding vorticity and inverting Biot–Savart leaves the energy-spectrum scaling intact but flattens the bottleneck. Structure-function exponents move toward K41 ζ_p = p/3, faster for transverse than longitudinal exponents. Source: [arXiv:2607.26896](https://arxiv.org/html/2607.26896).
- [IN-WINDOW] **Schmitt, Fuchs, Peinke and Obligado, "A universal relation between intermittency and dissipation within and beyond homogeneous isotropic turbulence", Phys. Fluids 38, 105138 (9 Oct 2026).** Only the title and metadata were retrieved, with no numbers. Source: [AIP](https://pubs.aip.org/aip/pof/article/38/10/105138/3406264/A-universal-relation-between-intermittency-and).
- [BACKGROUND; revised version dated 24 Aug 2026] **Eling and Oz KPZ-type formula for anomalous exponents: ξ_n − n/3 = γ²(d) ξ_n (1 − ξ_n).** It has one parameter per dimension, and ξ_n ∝ √n at large n. **Hook:** a closed-form ζ_p to compare against She–Leveque or DNS. Source: [arXiv:1502.03069v3](https://arxiv.org/html/1502.03069v3).
- [BACKGROUND] **Unified longitudinal/transverse multifractal framework validated with DNS at R_λ up to 1300**, arXiv:2601.12528 (Jan 2026). Source: [arXiv:2601.12528](https://ar5iv.labs.arxiv.org/html/2601.12528).
- [BACKGROUND] **Yeung et al., exascale DNS of isotropic turbulence on a 32768³ periodic cube** (JFM, Sep 2025). A snapshot is in the Johns Hopkins Turbulence Database. Source: [JFM](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/smallscale-properties-from-exascale-computations-of-turbulence-on-a-mathbf327683-periodic-cube/ACBB7FFE296106B0F2E9CA1B50A1A456).

**Wall-bounded turbulence and drag**
- [BACKGROUND, 2025/26] **CICLoPE pipe (Re_τ up to about 50,000).**
  - Centerline von Kármán constant **κ_CL = 0.44**.
  - Centerline turbulence intensity plateaus at about 2.9% beyond Re_τ ≈ 25,000.
  - Skewness and kurtosis are −0.5 and 3.5, Re-independent down to Re_τ = 5000.
  - Source: [Nagib et al.](https://cris.unibo.it/handle/11585/1029411).
- [BACKGROUND] **Superpipe re-analysis: κ = 0.433 ± 0.01 with a linear overlap term S₀ = 2.5,** giving U⁺ = (1/0.433) ln Y + 5.9 in outer scaling. Source: [Monkewitz, JFM](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/F9DA84F93A1E268BFC5F9E1CB5622E5E/S0022112023004482a.pdf/div-class-title-the-hunt-for-the-karman-constant-revisited-div.pdf).
- [IN-WINDOW] **Flow Turbulence Combust. (7 Sep 2026), explicit layer-wise law-of-the-wall for pipe/channel.** It uses κ = (2π)^(−1/2) ≈ 0.40 and Re_τ = 4000 as the "high-Re" boundary for channel flow. Source: [Springer](https://link.springer.com/article/10.1007/s10494-026-00788-2).
- [BACKGROUND, Jan 2026] **Drag on a hollow porous sphere (JFM).** Drag *increases* with porosity, up to 80%, at supercritical Re of 1.54–3.83×10⁵, unlike plates and cylinders. 2% porosity lowers the critical Re much as roughness does. Source: [JFM](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/drag-on-a-hollow-sphere-can-increase-with-porosity/96AC0BED872B41FDE7C447214028BA77).

### Inferences
- None of the Sept-2026 blow-up results gives a numerical benchmark a physics engine can "reproduce" directly. The engine-relevant hooks are diagnostics:
  - Beale–Kato–Majda: blow-up implies ∫‖ω‖_∞dt = ∞.
  - A self-similar rate λ = 1/2 for the Euler candidate.
  - Continuum breakdown: Karniadakis's ~70 nm figure is of the same order as the molecular mean free path of air at STP. That mean free path is a standard value of about 65–70 nm, which I did not take from these sources. So a sigma-ground "Knudsen check" (Kn = λ_mfp/ℓ) would flag the NS model as invalid at roughly that scale.
- Mentat/sigma-ground should treat the NS claim as "announced, Lean-formalized, under review". Per the CLAUDE.md honesty rule, the engine should not report Navier–Stokes as "solved" without that qualifier. It should also be explicit that the *unforced* problem (Clay A/B) remains open.
- The most implementable classical items are the pipe-flow numbers: Re_c ≈ 2040, the puff/slug/anti-puff map up to 2900, and E₀^crit = 6.09×10⁵ Re⁻³. Others are the Kelvin-wave dispersion relation, κ ≈ 0.43–0.44, and the transverse-vs-longitudinal intermittency asymmetry.

### Gaps
- I did not retrieve the OpenAI NS write-up PDF itself, so its blow-up rate, solution class and force regularity class are unverified. I also did not retrieve a primary Clay Institute statement.
- I found no in-window *experimental* drag-crisis result with new critical-Re numbers; the porous-sphere paper is January 2026.
- I found no in-window measurement of a new value for the Kolmogorov constant C_K or for ζ_p exponents with error bars.
- The Kelvin-wave PRL's fitted spectral exponent, and whether it matches −7/5 or −5/3, was not stated in the sources I could access.
- I found no DeepMind Navier–Stokes or Boussinesq announcement in this window, so its unstable-singularity λ values could not be refreshed.

---

## Q2. What is new in gas expansion? (free/Joule–Thomson expansion, supersonic jets, shocks and Mach reflection, Sedov–Taylor, rarefied gas/DSMC, nozzles, plume into vacuum, cavitation/sonoluminescence/Rayleigh–Plesset)

### Takeaway
In-window primary results in classical gas expansion were sparse. The new items are:
- unsteady Mach-reflection/regular-reflection (MR–RR) transitions over a vibrating wedge, which follow the sonic criterion with hysteresis;
- several Physics of Fluids and RSI shock papers that have no extractable numbers yet;
- an Aug-2026 revision of an analytic Rayleigh–Plesset mapping (Q = R^{5/2}). It gives a "no rebound below polytropic index k = 1/3" criterion and an acoustic-loss formula that explains about 90% of collapse energy loss.

Most crisp benchmarks for this area come from background sources: the Sedov–Taylor constants, the Rayleigh collapse time 0.915 R√(ρ/Δp), near-wall Rayleigh prolongation and jet speeds, and rarefied-jet scaling.

### Cited Findings

**Shocks, Mach reflection, blast waves**
- [IN-WINDOW] **Chen, Ye, Wang and Ye, "The evolution of asymmetric shock reflection with a vibrating wedge", Phys. Fluids 38, 096105 (2 Sep 2026).**
  - CFD with harmonic wedge motion.
  - The instantaneous MR↔RR transition stays consistent with the **sonic criterion**.
  - The flow lags the wall motion, causing hysteresis. Larger amplitude delays MR→RR and advances RR→MR, while higher frequency postpones both.
  - **Hook:** a steady sonic and detachment criterion for MR↔RR as a function of Mach number and wedge angle, compared against unsteady shifts.
  - Source: [AIP](https://pubs.aip.org/aip/pof/article/38/9/096105/3403642/The-evolution-of-asymmetric-shock-reflection-with).
- [IN-WINDOW] **Other Oct 2026 shock papers with no numbers retrieved:**
  - Righi et al., "Measurements of ripple decay in reflecting shocks in epoxy under high energy density conditions", Rev. Sci. Instrum. 97, 103902 (8 Oct 2026). Source: [AIP](https://pubs.aip.org/aip/rsi/article/97/10/103902/3406193/Measurements-of-ripple-decay-in-reflecting-shocks).
  - Xie et al., near-seabed underwater-explosion shock-pressure zonal model, Phys. Fluids 38, 106110 (7 Oct 2026). Source: [AIP](https://pubs.aip.org/aip/pof/article/38/10/106110/3406122/Zonal-evolution-and-prediction-model-of-shock-wave).
- [BACKGROUND] **Sedov–Taylor–von Neumann blast:** R = (25γ/(4K₀))^{1/5} (E₀t²/ρ₀)^{1/5}. The exact similarity constant is S(1.4) = 1.033 (Chernyi approximation 1.014). The strong-shock validity bound is z ≲ K₀^{−1/3} = 0.50. The time-of-arrival method gave a Beirut yield of 514 (+41/−43) t TNT. **Hook:** R ∝ (E t²/ρ)^{1/5} with prefactor S(γ). Source: [Díaz & Rigby, arXiv:2110.09488](https://ar5iv.labs.arxiv.org/html/2110.09488).
- [BACKGROUND] **LANL "Reflection of a blast wave by a very intense explosion".** The best-fit γ = 1.2265 for 43 atmospheric tests gives S_γ = 0.91952432. A universal ground-reflection function exists, with fit RMS relative error 0.0048. The Mohawk yield is reconciled to about 248 kt instead of about 360 kt. Source: [OSTI 1812582](https://www.osti.gov/servlets/purl/1812582).
- [BACKGROUND, Mar 2026] **Bauer and Johnson, Sci. Rep. 16, 12841.**
  - Small-charge blasts give a shock velocity of 400 ± 1.7 m/s.
  - The Mach stem travels at 411 ± 4 m/s, against 399 ± 2 m/s for the incident shock.
  - Confinement in a shock tube raises positive-phase impulse by 67.5% and lowers peak pressure by 2%.
  - Source: [Sci. Rep.](https://preview-www.nature.com/articles/s41598-026-42282-9).
- [BACKGROUND, Mar 2026] **Shock-tube LES (arXiv:2603.10706).**
  - Mark's bifurcation criterion is P₀,BL/P₅ < 1. N₂ and CO₂ bifurcate for M_s ≥ 1.5, while argon does not.
  - Axial temperature gradients in region 5 reach 524 K/m (N₂) and 1127 K/m (CO₂).
  - Source: [arXiv:2603.10706](https://arxiv.org/abs/2603.10706).
- [BACKGROUND] **Mach-stem bifurcation (triple Mach-White reflection)** occurs at 2.5 < M_i ≤ 3 for γ = 1.06 and θ_w = 30°, and none occurs above γ ≈ 1.3. Source: [arXiv:2008.11234](https://arxiv.org/pdf/2008.11234).

**Supersonic jets, rarefied gas, DSMC**
- [BACKGROUND] **Rarefied supersonic free jet (JFM).**
  - After the Mach disk (x* ≈ 8), the jet spreading rate is **0.03**, matching laminar 12/Re_L with Re_L = 400. The turbulent value is about 0.1.
  - Rebrov's flow regimes: turbulent for Re_L > 10⁴; transition for 10³–10⁴; laminar for 10²–10³; scattering for < 10².
  - Source: [JFM](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/experimental-investigation-of-turbulent-and-laminar-flow-regimes-in-a-supersonic-free-jet-under-nearvacuum-conditions/40D207C8F295656D80BAC89D22359C7D).
- [BACKGROUND, 2024] **LEMPUS-2 hybrid NS–DSMC jets.** A Mach disk forms only for Re_L > 15. No shocks form at Re_L ≈ 4. Agreement with experiment is within 20%. Source: [Dubrovin et al.](https://exa.ai/library/publication/dbk7p9l8cks).
- [BACKGROUND, 2005] **Parallel DSMC of free orifice jets:** centerline density decays ∝ z⁻² across Kn₀ = 0.0005–0.1. **Hook:** ρ(z)/ρ₀ ∝ (z/D)⁻² in the far field. Source: [Wu et al.](https://exa.ai/library/publication/z769v8z8ss4).

**Cavitation, bubble collapse, sonoluminescence**
- [IN-WINDOW revision, v3 dated 24 Aug 2026] **"Mapping Driven Oscillations in the Size of a Bubble to the Dynamics of a Newtonian Particle in a Potential", arXiv:2409.05961v3.**
  - The substitution Q = R^{5/2} turns Rayleigh–Plesset into Newtonian motion in a potential. That potential has a hard wall for polytropic index k > 1/3, so no sharp rebound is expected for k < 1/3.
  - Acoustic loss in collapse: E_acoustic = (4/3)π(p₀ − p_V) R_max³ [1 − (R_min/R_E)^{3−D}]. With M = −0.9, the effective dimension D ≈ 2.05.
  - Experiments show about 99% of the energy is lost between the first and second maxima, and the model accounts for about 90% of that.
  - **Hook:** integrate Rayleigh–Plesset or Keller–Miksis and compare the energy lost per bounce.
  - Source: [arXiv:2409.05961v3](https://arxiv.org/html/2409.05961v3).
- [BACKGROUND] **Rayleigh collapse time t_R = 0.915 t\*, where t\* = R₀√(ρ/Δp),** for an empty spherical bubble. Near-wall jet-impact pressures are about 10–15 p\*. Source: [arXiv:2104.06298](https://arxiv.org/abs/2104.06298).
- [BACKGROUND, 2022] **Rayleigh prolongation factor near a rigid wall.** k = T_L/T_Rayleigh peaks at about 1.2 at stand-off γ = d/R_max = 0.69, and k_fit(γ=0) = 0.978. A 7th-degree polynomial fit is given for 0 < γ < 1 (RMSE 0.0068). Source: [Reuter, Zeng & Ohl](https://exa.ai/library/publication/2zcrhb43mz1).
- [BACKGROUND] **Near-wall jet regimes.**
  - Needle jets exceed 850 m/s for γ < 0.053, and up to 2400 m/s at γ = 0.06.
  - Tip collapse occurs for 1.14 < γ < 3.23, and torus collapse for γ < 1.14 (Supponen et al.).
  - Water-hammer pressure is p_h = ½ ρ c U_jet.
  - Sources: [arXiv:2506.01286](https://arxiv.org/html/2506.01286); [arXiv:2408.03479](https://arxiv.org/html/2408.03479v2).
- [BACKGROUND, Feb 2026] **Nakamura and Yamamoto, JJAP 65, 03SP04.** Raising static pressure increases sonoluminescence intensity: it shortens collapse time, shrinks the bubble and shifts the trapping position. No in-window SL paper with numbers was found. Source: [IOP](https://iopscience.iop.org/article/10.35848/1347-4065/ae3773/meta).

### Inferences
- For sigma-ground, the most robust gas-dynamics validation set is still the canonical one:
  - Sedov–Taylor with S(γ) (1.033 at γ = 1.4);
  - Rayleigh collapse with prefactor 0.915;
  - the jet Mach-disk location, a standard correlation x_M/D ≈ 0.67√(p₀/p_b) that I did not take from these sources;
  - z⁻² density decay in free jets;
  - the Re_L thresholds (Mach disk for Re_L > 15; laminar spreading 12/Re_L).
- The 2026 papers mostly refine *unsteady* or *non-ideal* versions of these benchmarks. Examples are moving-wall MR–RR hysteresis and real-gas γ (1.2265) for nuclear-test blast fits.
- The k > 1/3 rebound criterion from the Q = R^{5/2} mapping is a cheap analytic unit test for any Rayleigh–Plesset integrator.

### Gaps
- I found no in-window (Jun–Oct 2026) primary paper with new quantitative results on Joule–Thomson inversion, free (Joule) expansion, adiabatic nozzle expansion or plume expansion into vacuum. I also found none for DSMC benchmarks or sonoluminescence temperatures.
- I found no new laser-induced cavitation-collapse dataset dated in-window.
- The Physics of Fluids shock-reflection paper's numeric MR–RR boundary shifts were not accessible because of a paywall.

---

## Q3. What is new in astrophysical and cosmological fluids? (supernova hydro, RT/RM, accretion, relativistic hydro, QGP η/s vs KSS bound)

### Takeaway
New RT/RM work in the window offers implementable closures:
- a unified buoyancy–drag model across density ratios (arXiv:2609.02128);
- an RTI model for time-dependent and shock-impulsive acceleration with < 20% error (Phys. Plasmas, Aug 2026);
- a new controlled-acceleration RT rig (Exp. Fluids, Jul 2026).

3D supernova studies disagree on whether multi-D progenitor perturbations matter:
- 3D FLASH runs find +35–50% non-radial kinetic energy and a shock radius 5–10% larger.
- 2D Chimera runs find no discernible effect.

For the QGP, CMS published (PRL, Aug 2026) flow in O+O and Ne+Ne collisions that hydrodynamic models reproduce. A July-2026 PRC Bayesian analysis finds an η/s minimum plateau over 150–220 MeV, consistent with η/s ≈ 0.08–0.2, i.e. about 1–2.5× the KSS value 1/4π ≈ 0.0796.

### Cited Findings

**Rayleigh–Taylor / Richtmyer–Meshkov**
- [IN-WINDOW] **Zhang, Li, Xiao and Wang, "Bridging the local and the global: a physically constrained buoyancy–drag model…", arXiv:2609.02128 (2 Sep 2026).**
  - Separate bubble and spike buoyancy–drag equations, with six effective coefficients fixed by RT/RM similarity, mass conservation and asymptotics, without case-by-case fitting.
  - It recovers bubble–spike symmetry at low Atwood number, free-fall RT spikes and ballistic RM spikes at high density ratio.
  - It is validated against RT, RM and Nova laser data.
  - Source: [arXiv:2609.02128](https://arxiv.org/abs/2609.02128).
- [IN-WINDOW] **Fu, Wang, Zhang, Ni and Lin, "A linear model for Rayleigh–Taylor instability induced by time-dependent accelerations…", Phys. Plasmas 33, 082707 (25 Aug 2026).**
  - Linear RTI ḧ = g(t) A_T k h works for non-impulsive acceleration.
  - The "RTI-TA" model adds a shock compression factor, h⁺ = h⁻(1 − ΔV/V_s).
  - Prediction error stays **< 20%** across Mach number. For comparison, the plain RTI model reaches up to 300% error and the impulsive (Richtmyer) model about 70%.
  - **Hook:** integrate ḧ = g(t) A k h with the post-shock amplitude jump.
  - Source: [AIP](https://pubs.aip.org/aip/pop/article-abstract/33/8/082707/3403085).
- [IN-WINDOW] **Horne-Jones, Glinnan, Lawrie and Williams (AWE), "Controlled accelerations for Rayleigh–Taylor instability", Exp. Fluids (1 Jul 2026), CAMPI rig.**
  - Atwood number up to 0.23, using NaBr brine and 2-propanol.
  - Accel–decel–accel histories at +20 m/s² and −30 m/s².
  - First experimental demonstration of mixing-zone *shrinking* during deceleration.
  - Self-similar law: **h = α A_T g t²**.
  - Source: [Springer](https://link.springer.com/article/10.1007/s00348-026-04249-4).
- [IN-WINDOW] **Li et al., J. Appl. Phys. 140, 045907 (28 Jul 2026).** Explosively driven RT on copper at 34–37 GPa, with single- and dual-mode perturbations. The Steinberg–Guinan strength model predicts amplitude to within 5%. Source: [AIP](https://pubs.aip.org/aip/jap/article/140/4/045907/3399589/Experimental-study-on-the-Rayleigh-Taylor).
- [BACKGROUND] **NIF self-similar RT growth constant α_B = 0.038 ± 0.008.** Theory for 3D immiscible fluids gives about 0.05. Source: [arXiv:2210.06631](https://export.arxiv.org/pdf/2210.06631v1.pdf).
- [BACKGROUND, Jan 2026] **Three-layer RMI (He/air/SF₆, Physica D 489).** No significant change in lower-interface mixing width compared with two-layer air/SF₆. Source: [OSTI 3016117](https://www.osti.gov/pages/biblio/3016117).

**Core-collapse supernova hydrodynamics**
- [IN-WINDOW] **Andresen, O'Connor, Fields and Couch, arXiv:2608.28358 (28 Aug 2026).**
  - 3D simulations of the last ~10–15 min of shell burning for 20, 21.5, 24.5, 26 and 29 M☉ progenitors (MESA → FLASH). O-shell convective velocities reach several hundred km/s.
  - For 24.5 M☉, a 3D progenitor gives **35–50% higher non-radial kinetic energy** and a **5–10% larger mean shock radius** than the angle-averaged case, and the earliest shock revival.
  - Source: [arXiv:2608.28358](https://arxiv.org/abs/2608.28358).
- [IN-WINDOW] **Chen et al., ApJ 1008, 74 (26 Aug 2026), 2D Chimera, 15 M☉, seven progenitor variants.** They find "no discernible impact" of multi-D progenitor perturbations on shock revival. The diagnostic energy E⁺ turns positive between 454 and 566 ms post-bounce, and the overburden is about 0.2 B. This contradicts the 3D result above, although the dimensionality differs. Source: [ApJ](https://iopscience.iop.org/article/10.3847/1538-4357/ae8dcb).
- [IN-WINDOW] **Neopane et al., ApJ 1008, 36 (24 Aug 2026), 9.6 M☉ zero-metallicity progenitor through to a young remnant.** The diagnostic explosion energy is 1.67–1.68×10⁵⁰ erg at mapping. 24.4% of radioactive heating comes from chains other than ⁵⁶Ni, and Ni/Fe > 1 makes it resemble an electron-capture SN. Decay heating in 1D suppresses the reverse shock. Source: [ApJ](https://iopscience.iop.org/article/10.3847/1538-4357/ae8a44).
- [IN-WINDOW, A&A forthcoming] **Kovalenko, O'Connor, Andresen and Couch, 39 M☉ rotating/magnetized 3D core collapse.** Net gain-region heating peaks at about 4×10⁵² erg/s at about 0.25 s (non-rotating). The non-rotating model forms a black hole about 1 s after bounce. The magnetized model revives first, with a bipolar outflow. Source: [A&A preprint](https://www.aanda.org/articles/aa/pdf/forth/aa60993-26.pdf).

**QGP as near-perfect fluid**
- [IN-WINDOW] **CMS Collaboration, "Observation of Long-Range Collective Flow in O+O and Ne+Ne Collisions…", Phys. Rev. Lett. 137, 082302, published 17 Aug 2026, DOI 10.1103/26wx-tg6f.**
  - √s_NN = 5.36 TeV, with 7 nb⁻¹ (O+O) and 0.8 nb⁻¹ (Ne+Ne).
  - Significant v₂ and v₃ are seen. The Ne/O v₂ ratio rises in central collisions, consistent with a deformed ²⁰Ne.
  - Hydro (Trajectum, IP-Glasma+MUSIC) captures the trends but "fail[s] to fully describe the v_n magnitude".
  - Sources: [OSTI](https://www.osti.gov/servlets/purl/3484604); [CMS-HIN-25-009](https://cms-results.web.cern.ch/cms-results/public-results/publications/HIN-25-009/index.html).
- [IN-WINDOW] **ATLAS, dN/dη and ⟨p_T⟩ in O+O and Ne+Ne at 5.36 TeV, arXiv:2606.20257 (18 Jun 2026).** It restates that hydrodynamics needs η/s ≈ 0.1–0.2 (in units of ħ), close to the conjectured bound. Source: [arXiv:2606.20257](http://www.arxiv.org/abs/2606.20257).
- [IN-WINDOW] **Auvinen, Eskola, Hirvonen and Niemi, "Neural-network-enhanced Bayesian global analysis of relativistic heavy-ion collisions", Phys. Rev. C 114, 014904 (10 Jul 2026).**
  - EKRT plus 2+1D viscous hydro, fit to RHIC Au+Au 200 GeV and LHC Pb+Pb 2.76/5.02 TeV and Xe+Xe 5.44 TeV.
  - η/s has a **minimum-value plateau from about 150 MeV to at least 220–230 MeV**. ζ/s is non-zero at 90% credibility near the crossover.
  - The numeric (η/s)_min was garbled in my retrieval; see Gaps.
  - Source: [PRC](https://link.aps.org/doi/10.1103/13dy-4g64).
- [BACKGROUND, May 2026] **Bayesian analysis including Λ longitudinal spin polarization (Pb+Pb 5.02 TeV), arXiv:2605.29383.** η/s is unchanged, while the median ζ/s rises by about 2×. Source: [arXiv:2605.29383](https://arxiv.org/html/2605.29383).
- [BACKGROUND] **JETSCAPE analysis with model discrepancy:** η/s ≈ 0.1–0.2 with a minimum near T ≈ 220 MeV, and T_sw = 144.5 ± 4.2 MeV. Source: [arXiv:2509.19759](https://arxiv.org/html/2509.19759v2).
- [BACKGROUND] **φ/Ω-based Bayesian analysis:** η/s = (2.08 +1.10/−1.09)/4π and ζ/s = 0.06 ± 0.04 at about 160 MeV. Source: [arXiv:2207.13534](https://arxiv.org/html/2207.13534).
- [BACKGROUND, Jan 2026] **Lattice-QCD gluon plasma:** η/s ≈ 0.1–0.6 above T_c, rising strongly below T_c. Source: [arXiv:2601.14967](https://arxiv.org/html/2601.14967).
- [BACKGROUND, Apr 2026] **Chapman–Enskog pQCD:** η/s ≈ 0.15 at T_c, sensitive to the scale choice Q = 3T. Source: [pith.science 2604.25059](https://pith.science/paper/2604.25059).
- [IN-WINDOW, no number retrieved] **Parton-transport vs Chapman–Enskog shear viscosity of quark-gluon matter, arXiv:2606.21722 (Jun 2026).** Source: [arXiv:2606.21722](https://arxiv.org/html/2606.21722).
- [BACKGROUND] **ALICE O–O multiplicity:** ⟨dN_ch/dη⟩ = 129.7 ± 4.1 (0–5%) and 11.7 ± 1.2 (60–90%). Source: [arXiv:2604.07184](https://arxiv.org/html/2604.07184v1).

### Inferences
- **Hook for the QGP:** sigma-ground should hold KSS η/s = 1/(4π) ≈ 0.0796 as the lower reference. The data-driven range is η/s ≈ 0.08–0.2, with a minimum plateau around T ≈ 150–230 MeV. The 2026 analyses mainly tighten the temperature dependence rather than move the minimum.
- **RT/RM hooks to implement:** h = α A g t² with α_B ≈ 0.03–0.05; linear growth ḧ = g A k h; and an RM impulsive post-shock jump. The <20% error figure for RTI-TA is a quantitative target if the engine adds a shock-driven interface module.
- The supernova results conflict in direction and differ in dimensionality (3D vs 2D). The engine should not adopt a single "progenitor-perturbation effect" number. The 5–10% shock-radius boost is model-specific.

### Gaps
- I could not read the numeric (η/s)_min of the July 2026 PRC (Auvinen et al.) from the fetched abstract because the math rendering was garbled.
- I found no in-window results with crisp numbers on accretion-flow hydrodynamics or relativistic hydrodynamics benchmarks (for example Bondi or relativistic shock tubes).
- I found no in-window RHIC/sPHENIX η/s update.

---

## Q4. What is new in quantum fluids and analog systems? (superfluid He, BEC expansion/TOF, quantum turbulence, fluids of light, analog black holes and cosmology)

### Takeaway
The window brought:
- direct visualization of inertial and Kelvin waves on a rotating He-II vortex lattice, with a Donnelly–Glaberson threshold of 31.4 mW and a spectral slope of about −1.6;
- a Nature paper (1 Jul 2026) observing the *backreaction* of stimulated Hawking radiation in a fibre-optical horizon;
- a PRL (7 Jul 2026) showing that Kibble–Zurek quenches generate quantum turbulence with Kolmogorov scaling;
- the Cambridge homogeneous-BEC result, Vinen decay dL/dt = −B L² with B = 1.0(2) ħ/m. It was first posted in April 2026 and has a version dated August 2026.

I found no in-window *experiment* on analog cosmological expansion.

### Cited Findings
- [IN-WINDOW] **"Visualization of Inertial and Kelvin Waves on the Quantum Vortex Lattice in Superfluid Helium", arXiv:2607.00821 (1 Jul 2026).**
  - Rotating He II at T = 2.088 K, perturbed by a heat flux with P = 0–50 mW. Above 50 mW the lattice breaks into turbulence.
  - Inertial-wave signatures appear for ω* = ω/2Ω ≤ 1, following E_z/E_x = 2(1−ω*²)/(1+ω*²).
  - Wave energy above ω* > 1 rises past P ≈ 30 mW, matching the Donnelly–Glaberson threshold **P_DG = 2Aρ_s s T √(2ν_sΩ)/r ≈ 31.4 mW**.
  - The high-frequency spectral slope is **≈ −1.6**, against −4/3 theory for weak Kelvin-wave turbulence.
  - Source: [arXiv:2607.00821](https://arxiv.org/html/2607.00821).
- [IN-WINDOW, v dated 24 Aug 2026; first posted Apr 2026] **Morris, Gazo, Fischer, Zhang, Ho, Cooper, Eigen and Hadzibabic, "Observation of Vinen turbulence during far-from-equilibrium Bose–Einstein condensation", arXiv:2604.28191.**
  - Homogeneous ³⁹K box with V ≈ 9×10⁴ μm³, n ≈ 2.2 μm⁻³ and ξ ≈ 0.9 μm at a = 430 a₀.
  - Vortex line density obeys **dL/dt = −B L²** with **B = 1.0(2) ħ/m**, independent of interaction strength, with one-body decay τ ≈ 300 ms.
  - Coarsening follows dℓ²/dt ≈ 3.4 ħ/m.
  - B ≈ ħ/m matches superfluid ⁴He.
  - **Hook:** L(t) = L₀/(1 + B L₀ t), with B = ħ/m.
  - Source: [arXiv:2604.28191](https://arxiv.org/abs/2604.28191).
- [IN-WINDOW] **"Spontaneous Quantum Turbulence in a Newborn Bose-Einstein Condensate via the Kibble-Zurek Mechanism", Phys. Rev. Lett. 137, 020402 (7 Jul 2026), DOI 10.1103/b16v-hwp2.**
  - 2D stochastic projected GPE simulations.
  - The incompressible energy spectrum follows Kolmogorov k^(−5/3), with a k^(−3) vortex-core range beyond k ~ 1/ξ.
  - Spectra collapse when rescaled by the Kibble–Zurek correlation length.
  - Structure-function exponents deviate from K41 and follow K62.
  - Source: [PRL](https://link.aps.org/doi/10.1103/b16v-hwp2).
- [IN-WINDOW] **"Study of quantum turbulence by vortex-antivortex dynamics in dipolar BECs", arXiv:2610.00857 (1 Oct 2026).** These are simulations. A k^(−5/3) range spans about one decade at 5–6 ms, and no vortex–antivortex annihilation occurs up to 100 ms, which the authors attribute to dipolar suppression of decay. Source: [arXiv:2610.00857](https://arxiv.org/html/2610.00857).
- [IN-WINDOW] **Procopio, Aguero-Santacruz, Bermudez and Leonhardt, "Backreaction of stimulated Hawking radiation in an optical analogue", Nature 655, 336–341 (published 1 Jul 2026), DOI 10.1038/s41586-026-10720-3; arXiv:2607.01118.**
  - Fibre-optical horizon: an 8 fs, 800 nm, 80 MHz pump in a 7 mm photonic-crystal fibre.
  - IR probes at 1100–1600 nm stimulate Hawking partners at about **233 nm**.
  - They identify a direct (bi-quadratic Kerr) generation process and measure its backreaction on the pump. The spectrum stays thermal even in the strong-dispersion regime, where the temperature is no longer set by the surface gravity.
  - Sources: [Nature](https://www.nature.com/articles/s41586-026-10720-3); [arXiv:2607.01118](https://arxiv.org/abs/2607.01118).
- [BACKGROUND, May 2026] **Felipe-Elizarraras et al., "Measurement of analogue Hawking radiation stimulated by a single photon", Nat. Commun. (2026), DOI 10.1038/s41467-026-73812-8.** Sources: [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13392050/); citation in [Nature ref. list](https://www.nature.com/articles/s41586-026-10720-3).
- [IN-WINDOW, review] **Jacquet and Giacobino, "Acoustic horizons and the Hawking effect in polariton fluids of light", Gen. Relativ. Gravit. (24 Jul 2026).** Source: [Exa library](https://exa.ai/library/publication/3phycffx0hw).
- [BACKGROUND theory; revised Aug 2026] **"Efimovian Phonon Production for an Analog Coasting Universe in BECs", arXiv:2603.16095.** For a(t) ∝ t in a quasi-2D BEC, phonon production is log-periodic for kl > 1/2 (sub-horizon) and follows power-law growth for kl < 1/2 (super-horizon). This is a prediction only. Source: [arXiv:2603.16095](https://arxiv.org/pdf/2603.16095).
- [BACKGROUND] **Superfluid helium and fluid-of-light reference values:**
  - Intervortex spacing in He-II coflow turbulence scales as ℓ ∝ Re_κ^(−3/4), with κ = h/m ≈ 9.97×10⁻⁸ m²/s and T_λ = 2.1768 K. Source: [arXiv:2504.21416](https://arxiv.org/html/2504.21416).
  - Reconnection in He II shows universal δ ∝ t^{1/2} scaling with pre/post asymmetry. Source: [PNAS 2025](https://www.pnas.org/doi/10.1073/pnas.2426064122).
  - Steinhauer group, spontaneous analog cosmological particle creation in a 3D fluid of light. Source: [Nat. Commun. 2022](http://www.nature.com/articles/s41467-022-30603-1.pdf).

### Inferences
- The Vinen law with B ≈ ħ/m holds for both BECs and ⁴He. It is a clean, mass-scaled hook: B(⁴He) = ħ/m_He ≈ 1.6×10⁻⁸ m²/s, and B(³⁹K) is about 10× smaller. I computed these values from the stated B = ħ/m.
- Kelvin-wave spectra now have two experimental platforms, classical water (PRL Sep 2026) and He-II lattices (arXiv Jul 2026). So the engine could compare its filament-model predictions (−7/5 vs −5/3 vs −4/3 in ω) against both.
- For analog gravity, the optical-fibre numbers are an exact kinematic check: an 800 nm pump and 1100–1600 nm probes give a 233 nm partner. Any sigma-ground horizon/Doppler (phase-matching) module should reproduce this.

### Gaps
- I found no in-window *experimental* BEC time-of-flight or analog-expanding-universe result. The proposal 2411.09596 and the 2018 Eckel et al. experiment are older.
- I found no in-window polariton or fluid-of-light experiment with numbers.
- The Hawking-temperature or horizon-gradient values in the Nature paper were not extracted.

---

## Q5. What is new in soft matter and everyday fluids with crisp numbers? (splashing thresholds, Leidenfrost, capillarity, granular flow)

### Takeaway
The cleanest in-window numbers come from a FluidX3D simulation study of oblique drop impact on a deep pool (arXiv:2608.29762):
- the deposition→front-splash boundary is We·cosθ ≈ 120;
- the onset of lateral splash is at We_N ≈ 318;
- secondary drops follow d_med/D ~ We^(−3/5) and N_s ~ We^(9/5).

Leidenfrost work in the window was a humidity/internal-circulation model (Lohse group, Aug 2026). The headline capillary-Leidenfrost result (Nature Physics, levitation at 110 °C instead of about 200 °C) is from April 2026, just before the window. I found no granular-flow results with numbers.

### Cited Findings
- [IN-WINDOW] **"Splashing-regime transitions and secondary-droplet scaling in oblique drop impacts on a deep pool", arXiv:2608.29762 (posted late Aug/Sep 2026).**
  - Range: water, 41 ≤ We ≤ 1790, 10° ≤ θ ≤ 90°, 0.0017 ≤ Oh ≤ 0.0027.
  - Five regimes: deposition, front splash, side splash, side-front splash and crown splash.
  - **Deposition→front splash: K_s = We cos θ, with K_s^c ≈ 120.** This comes from crown-rim inertia against Taylor–Culick retraction. The film-impact value is 128.
  - **Front→side-front splash: We_N^c ≈ 318.** The film-impact value is 450.
  - Splash criterion K = We·Oh^(−2/5) with K_c = 2100, which corresponds to We_c ≈ 182 at Oh = 0.0022.
  - Secondary drops: **d_med/D ~ We^(−3/5)** and **N_s ~ We^(9/5)**.
  - Source: [arXiv:2608.29762](http://arxiv.org/pdf/2608.29762).
- [IN-WINDOW] **de Wildt, Prosperetti, Diddens and Lohse, "Leidenfrost droplets: the roles of ambient humidity and internal droplet circulation", arXiv:2608.01828 (Aug 2026).** Humidity, together with internal flow, significantly changes droplet geometry and drying kinetics. This is a preprint, and no numbers were retrieved. Source: [arXiv:2608.01828](https://arxiv.org/abs/2608.01828), via the search summary.
- [BACKGROUND, 14 Apr 2026] **Zhang et al. (Steven Wang group, CityU HK), "Capillary Leidenfrost effect", Nature Physics 22(5), DOI 10.1038/s41567-026-03255-x.**
  - A solid capillary structure levitates at **110 °C**, against about **200 °C** for a water droplet on the same substrate.
  - The Leidenfrost point is tunable from 110 to 160 °C, set by capillary radius and porosity. The critical porosity is about 0.025–0.1.
  - The vapor film is **50–80 μm** (about 63 μm typical), and the critical heat flux is reduced 5.6×.
  - The structure self-propels for 8 m.
  - Sources: [eScholarship PDF](https://escholarship.org/content/qt3w95b1df/qt3w95b1df.pdf); [CityU news](https://www.cityu.edu.hk/en/mne/news-publications-seminars/capillary-leidenfrost-effect-unlocking-low-temperature-frictionless-solid-levitation).
- [BACKGROUND, Feb 2026] **Drop impact on a pool (Exp. Fluids):**
  - We < 100: coalescence plus vortex ring.
  - 100–200: swell plus thin Worthington jet.
  - 200–2000: crown plus thick jet.
  - > 2000: bubble canopy.
  - These thresholds are unchanged by particulate loading ≲ 10%, and surfactant shifts them up.
  - Source: [Springer](https://link.springer.com/article/10.1007/s00348-026-04178-2).
- [BACKGROUND, Feb 2026] **Bouncing→coalescence on a moving bath (Harris, Cimpeanu et al.).** The critical normal Weber number drops with bath speed. The data collapse onto **We_c = ρ(V cos φ + U sin φ)² R/σ with φ ≈ 25°**. Source: [Exa library](https://exa.ai/library/publication/lh8y653c00k).
- [IN-WINDOW, no numbers] **Coalescence–rupture threshold for micron droplets colliding with dust, Phys. Fluids 38, 063331 (15 Jun 2026).** Source: [AIP](https://pubs.aip.org/aip/pof/article/38/6/063331/3395145/Coalescence-rupture-threshold-in-head-on).
- [BACKGROUND] **Riboux and Gordillo (JFM 2016)** is the standard reference for the splash threshold, including in the dynamic Leidenfrost regime. Source: [search summary pointing to JFM "Inverse Leidenfrost impacting drops" and related](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/inverse-leidenfrost-impacting-drops/CD740A160413544A886F86CCC4DF2BDD).

### Inferences
- **Hooks:** We = ρU²D/σ, Oh = μ/√(ρσD), K = We·Oh^(−2/5) with K_c ≈ 2100; oblique impact K_s = We cos θ ≈ 120; and the pool-impact regime boundaries at We ≈ 100, 200 and 2000. The engine also needs the Taylor–Culick velocity u_TC = √(2σ/(ρh)). That formula is standard and is not given in the cited abstract.
- The capillary-Leidenfrost model modifies the Arpaci vapor-film balance. The paper's supplementary material gives an analytic film-thickness expression in Jakob, Archimedes and Prandtl numbers, which could be implemented. Source: [SI PDF](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41567-026-03255-x/MediaObjects/41567_2026_3255_MOESM1_ESM.pdf).

### Gaps
- I found no in-window granular-flow (μ(I) rheology, Beverloo discharge) or capillary-rise result with numbers.
- I found no in-window *experimental* splash-threshold paper; the oblique-impact work is numerical.
- The Leidenfrost-humidity preprint's quantitative results were not retrieved.

---

## Q6. Were there new measurements of transport coefficients or equations of state? (viscosity, thermal conductivity, sound speed at extreme conditions, supercritical fluids, warm dense matter)

### Takeaway
In-window items:
- a PRE (8 Sep 2026) arguing that the supercritical region is a distinct "supercritical fluid" state, bounded by two crossover lines L± with distinct viscosity, diffusion and structure signatures (argon MD);
- a new time-of-flight speed-of-sound apparatus with 0.1% uncertainty (Int. J. Thermophys., 5 Oct 2026);
- a 2026 Nature Communications measurement of electrical conductivity in expanded warm dense aluminium, with a metal-to-nonmetal minimum near 1 g/cm³. Its exact date is unconfirmed.

QGP η/s results are covered in Q3. I found no in-window viscosity measurement at warm-dense-matter conditions.

### Cited Findings
- [IN-WINDOW] **"Physical signatures of supercritical fluid boundaries", Phys. Rev. E 114, 035410 (8 Sep 2026), DOI 10.1103/3mxm-gqh5.** The radial distribution function, diffusion coefficient, shear viscosity and VACF are qualitatively different in the supercritical fluid and change on crossing the L± lines, which obey universal scaling laws. The study uses argon MD. Source: [PRE](https://link.aps.org/doi/10.1103/3mxm-gqh5).
- [IN-WINDOW] **Itaiwi, Sameen, Miyazaki, Thu and Higashi, "A New Speed-of-Sound Instrument Based on the Time-of-Flight Method…", Int. J. Thermophys. (5 Oct 2026).** Saturated-liquid sound speeds of R-1224yd(Z), R-1233zd(E) and R-1336mzz(E), with uncertainties of 0.1% (speed of sound), 30 mK and 1.0 kPa. Calibration used R-1336mzz(Z) over 278–343 K. Source: [Springer](https://link.springer.com/article/10.1007/s10765-026-03837-0).
- [2026, exact date unconfirmed] **Expanded warm dense aluminium (Nat. Commun., pulsed Joule heating plus DFT-MD).**
  - Electrical conductivity is 0.28–5 MS/m from ambient density down to 0.6 g/cm³, with T > 37 kK.
  - σ₀ = 40 MS/m at 2.7 g/cm³.
  - Along the high-energy path, σρ₀/(σ₀ρ) has a minimum of about 0.02 near 1.0 g/cm³, which marks the onset of the metal-to-nonmetal transition.
  - The two thermodynamic paths differ by up to about 100% at 1.7 g/cm³.
  - The electron relaxation time, not ionization, dominates.
  - Source: [Nat. Commun. reference PDF](https://www.nature.com/articles/s41467-026-77435-x_reference.pdf).
- [BACKGROUND, May 2026] **Supercritical property rig (J. Propulsion & Energy 6(1)).** Validated against NIST for water and CO₂ within ±5.44% (density), ±7.45% (viscosity) and ±9.25% (thermal conductivity). Source: [JPNE](https://www.jpne.org/articles/article/9qRv/).
- [BACKGROUND, Oct 2025] **Laser-shocked diamond-anvil-cell scaling.** P_qtz/(1 GPa) = 47 (I/1 TW cm⁻²)^0.49 exp[−0.0019 (h/1 μm)(τ/1 ns)^0.60], within about 38–46%. Source: [Phys. Plasmas](https://pubs.aip.org/aip/pop/article/32/10/102704/3366159/Investigation-of-warm-dense-matter-by-combining).
- QGP transport (η/s, ζ/s): see Q3. The ATLAS 2026 paper restates η/s ≈ 0.1–0.2. Source: [arXiv:2606.20257](http://www.arxiv.org/abs/2606.20257).

### Inferences
- For sigma-ground's materials/fluids layer, the supercritical-boundary PRE suggests a testable feature: viscosity and self-diffusion of argon versus (P, T) should show crossover kinks along two lines (L±), not a single Widom line. NIST argon data would be the natural reference.
- The warm-dense-Al conductivity minimum (σ/σ₀ ≈ 0.02 at ρ ≈ 1 g/cm³) is a sharp benchmark against Drude/Ziman-type models.

### Gaps
- I found no in-window measurements of viscosity or thermal conductivity in warm dense matter, of the sound speed in dense hydrogen or helium, or of supercritical CO₂ or water transport with new numbers.
- I could not confirm the exact publication date of the Nat. Commun. warm-dense-Al paper.
- I could not extract numeric L± line parameters from the PRE abstract.
