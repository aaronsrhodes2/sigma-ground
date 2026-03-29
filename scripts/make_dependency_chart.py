#!/usr/bin/env python3
"""
Generate the sigma-ground physics dependency chart.

Output: docs/dependency_chart.html  (self-contained, no external deps)

Usage:
    cd /path/to/sigma-ground
    python scripts/make_dependency_chart.py
"""

import json
from pathlib import Path

# ── Node Types ─────────────────────────────────────────────────────────
# measured  = directly measured physical constant (NIST/PDG)
# derived   = computed by formula from measured constants
# ssbm      = SSBM-specific parameter (ξ, η, σ_conv)
# higgs     = Higgs-origin mass — σ-INVARIANT (doesn't change with σ)
# qcd       = QCD-origin quantity — σ-DEPENDENT (scales with Λ_eff)
# field     = σ-field value at a location
# api       = exposed physics function

# ── Edge Types ─────────────────────────────────────────────────────────
# formula   = target derived by formula from source
# ssbm      = SSBM-specific derivation (dark energy matching etc.)
# qcd       = QCD sets/governs target
# sigma     = σ-field coupling modifies target
# uses      = API function uses this constant

LAYER_NAMES = [
    "Cosmological",
    "Fundamental Measured Constants",
    "SSBM Field Parameters",
    "Derived Fundamentals",
    "QCD / σ-Field",
    "Quark & Nucleon",
    "Nuclear & Electromagnetic",
    "Astronomical",
    "Physics API — Core Field",
    "Physics API — Interface Layer A",
    "Physics API — Interface Layer B",
]

# (id, display_label, layer, node_type, tooltip_text)
NODES = [

    # ── Layer 0: Cosmological ──────────────────────────────────────────
    ("H0", "H₀", 0, "measured",
     "Hubble constant\n= 67.4 km/s/Mpc\nPlanck 2018 CMB power spectrum\nSets the cosmological energy scale"),

    # ── Layer 1: Fundamental Measured ─────────────────────────────────
    ("G",        "G",   1, "measured",
     "Gravitational constant\n= 6.67430×10⁻¹¹ m³ kg⁻¹ s⁻²\nCavendish torsion balance"),
    ("C",        "c",   1, "measured",
     "Speed of light\n= 2.99792458×10⁸ m/s  [exact by SI 2019]\nDefines the metre"),
    ("HBAR",     "ℏ",   1, "measured",
     "Reduced Planck constant\n= 1.054571817×10⁻³⁴ J·s  [exact by SI 2019]"),
    ("E_CHARGE", "e",   1, "measured",
     "Elementary charge\n= 1.602176634×10⁻¹⁹ C  [exact by SI 2019]"),
    ("EPS_0",    "ε₀",  1, "measured",
     "Vacuum permittivity\n= 8.854187817×10⁻¹² F/m  (CODATA 2018)"),
    ("MU_0",     "μ₀",  1, "measured",
     "Vacuum permeability\n= 1.25664×10⁻⁶ H/m\nDerived from SI exact values"),
    ("K_B",      "k_B", 1, "measured",
     "Boltzmann constant\n= 1.380649×10⁻²³ J/K  [exact by SI 2019]\nLinks energy to temperature"),
    ("AMU_KG",   "m_u", 1, "measured",
     "Atomic mass unit = 1.66054×10⁻²⁷ kg\n(CODATA 2018)\n1/12 of ¹²C mass.\nUsed for atomic volume, number density."),
    ("N_AVOGADRO","N_A", 1, "measured",
     "Avogadro constant\n= 6.02214076×10²³ mol⁻¹\n[exact by SI 2019]\nLinks per-atom to per-mole."),

    # ── Layer 2: SSBM Parameters ───────────────────────────────────────
    ("XI", "ξ", 2, "ssbm",
     "Baryon fraction Ω_b/(Ω_b+Ω_c) = 0.1582\n"
     "Planck 2018.  THE single new SSBM free parameter.\n"
     "Ω_b h² = 0.02237,  Ω_c h² = 0.1200\n"
     "Drives the entire scale transition framework."),
    ("SIGMA_CONV", "σ_conv", 2, "ssbm",
     "Critical σ for nuclear bond failure\n"
     "= −ln(ξ) ≈ 1.849\n"
     "Formula derivation: σ_conv = −ln(ξ)\n"
     "The σ at which QCD binding energy overwhelms nuclear structure."),
    ("ETA", "η", 2, "ssbm",
     "Cosmic entanglement fraction = 0.4153\n"
     "DERIVED from dark energy constraint:\n"
     "  η × ρ_released = ρ_DE(observed)\n"
     "Fraction of particles with cross-hadron quantum entanglement."),

    # ── Layer 3: Derived Fundamentals ─────────────────────────────────
    ("L_PLANCK",    "l_P",     3, "derived",
     "Planck length = √(ℏG/c³) ≈ 1.616×10⁻³⁵ m\n"
     "Smallest length with physical meaning.\n"
     "Used as spatial 'effectively zero' floor."),
    ("M_PLANCK_KG", "m_P",     3, "derived",
     "Planck mass = √(ℏc/G) ≈ 2.18×10⁻⁸ kg"),
    ("ALPHA",       "α",       3, "derived",
     "Fine structure constant\n"
     "= e²/(4πε₀ℏc) ≈ 1/137.036\n"
     "Strength of electromagnetic interaction.\n"
     "σ-INVARIANT: e is Higgs-origin, ε₀ and ℏ are fundamental."),
    ("M_ELECTRON_KG","mₑ (kg)", 3, "higgs",
     "Electron mass = 9.1094×10⁻³¹ kg  (CODATA 2018)\n"
     "Higgs origin — σ-INVARIANT\n"
     "Does not change with σ-field compression."),
    ("SIGMA_FLOOR", "σ_floor", 3, "derived",
     "σ computational floor\n"
     "= l_P × H₀/c ≈ 1.18×10⁻⁶¹\n"
     "Planck/Hubble ratio.\n"
     "Smallest σ the classical field description can sustain.\n"
     "Universe's own minimum — not a magic number."),
    ("SIGMA_HERE", "σ_here", 3, "field",
     "Observer frame σ — OUR spacetime\n"
     "= σ_floor ≈ 1.18×10⁻⁶¹\n"
     "Set to σ_floor (not exact 0.0) to prevent\n"
     "floating-point catastrophes.\n"
     "exp(σ_here) = 1.0 exactly in double precision.\n"
     "Change THIS to couple to extreme astrophysics."),
    ("M_HUBBLE_KG", "M_H",     3, "derived",
     "Hubble mass = c³/(2GH₀) ≈ 9.3×10⁵² kg\n"
     "Total mass within the Hubble radius."),
    ("H_PLANCK",  "h",          3, "derived",
     "Planck constant = 2πℏ\n= 6.626×10⁻³⁴ J·s\nDERIVED from ℏ.\nUsed in Planck radiation, photon energy."),
    ("STEFAN_BOLTZMANN","σ_SB", 3, "derived",
     "Stefan-Boltzmann constant\n= π²k_B⁴/(60ℏ³c²)\n= 5.670×10⁻⁸ W m⁻² K⁻⁴\nDERIVED — never stored.\nBlackbody total radiated power."),
    ("BOHR_RADIUS","a₀",       3, "derived",
     "Bohr radius = ℏ/(mₑcα)\n= 5.292×10⁻¹¹ m\nDERIVED from ℏ, mₑ, c, α.\nAtomic length scale."),
    ("MU_BOHR",   "μ_B",       3, "derived",
     "Bohr magneton = eℏ/(2mₑ)\n= 9.274×10⁻²⁴ J/T\nDERIVED from e, ℏ, mₑ.\nMagnetic moment unit."),
    ("R_GAS",     "R",         3, "derived",
     "Gas constant = k_B × N_A\n= 8.314 J/(mol·K)\nDERIVED — never stored."),
    ("EV_TO_J",   "eV→J",     3, "derived",
     "Energy conversion\n= e = 1.602×10⁻¹⁹ J/eV\nDERIVED: 1 eV ≡ e × 1 V.\nSingle source of truth for eV↔J."),

    # ── Layer 4: QCD / σ-Field ─────────────────────────────────────────
    ("LAMBDA_QCD_MEV", "Λ_QCD", 4, "qcd",
     "QCD confinement scale = 217 MeV  (PDG)\n"
     "Sets the strong force energy scale.\n"
     "σ-DEPENDENT: Λ_eff(σ) = Λ_QCD × e^σ\n"
     "~99% of nucleon mass originates here."),
    ("SIGMA_FIELD", "σ(r)", 4, "field",
     "σ-field value at location r\n"
     "= ξ·GM/(rc²)  [from Newtonian potential]\n"
     "Field equation: □σ = −ξR\n"
     "σ = σ_here ≈ 0 in flat spacetime (observer frame)\n"
     "σ > 0 inside matter, black holes, at Big Bang\n"
     "σ_here = σ_floor (Planck/Hubble, not exact 0)"),

    # ── Layer 5: Quark & Nucleon ───────────────────────────────────────
    ("M_UP_MEV",        "m_u",     5, "higgs",
     "Up quark bare mass = 2.16 MeV  (PDG 2020)\n"
     "Higgs origin — σ-INVARIANT\n"
     "Only ~0.2% of proton mass."),
    ("M_DOWN_MEV",      "m_d",     5, "higgs",
     "Down quark bare mass = 4.67 MeV  (PDG 2020)\n"
     "Higgs origin — σ-INVARIANT"),
    ("PROTON_BARE_MEV", "p_bare",  5, "derived",
     "Proton Higgs-origin mass\n"
     "= 2m_u + m_d = 8.99 MeV\n"
     "Only ~1% of total proton mass!\n"
     "σ-INVARIANT — set entirely by Higgs."),
    ("PROTON_TOTAL_MEV","p",       5, "measured",
     "Proton total mass = 938.272 MeV  (PDG 2022)\n"
     "~99% is QCD binding energy, only ~1% Higgs.\n"
     "Direct measurement."),
    ("PROTON_QCD_MEV",  "p_QCD",   5, "qcd",
     "Proton QCD binding mass\n"
     "= p_total − p_bare = 929.28 MeV\n"
     "~99% of proton mass — sets by gluon dynamics.\n"
     "σ-DEPENDENT: scales with Λ_QCD × e^σ"),
    ("NEUTRON_TOTAL_MEV","n",      5, "measured",
     "Neutron total mass = 939.565 MeV  (PDG 2022)\n"
     "~99% QCD binding, ~1% Higgs quark masses."),
    ("M_ELECTRON_MEV",  "mₑ (MeV)",5, "higgs",
     "Electron mass = 0.51100 MeV\n"
     "Higgs origin — σ-INVARIANT\n"
     "Not a quark; lives outside the nucleus."),
    ("PROTON_QCD_FRAC", "f_QCD",   5, "derived",
     "Proton QCD fraction\n"
     "= p_QCD/p_total ≈ 0.9904\n"
     "DERIVED — never stored separately.\n"
     "~99% of proton mass is QCD binding.\n"
     "Key multiplier in all σ-corrections."),
    ("DELTA_NP",  "Δm_np",    5, "measured",
     "Neutron−proton mass difference\n"
     "= 1.29333236 MeV (PDG 2020)\n"
     "MEASURED directly — NOT computed from\n"
     "m_n − m_p (which loses 3 sig figs).\n"
     "Avoids catastrophic cancellation."),

    # ── Layer 6: Nuclear & EM ──────────────────────────────────────────
    ("R0_FM",       "r₀",    6, "measured",
     "Nuclear charge radius parameter = 1.215 fm\n"
     "Hofstadter electron scattering experiments.\n"
     "R(A) = r₀ × A^(1/3)"),
    ("KE_E2_MEV_FM","ke²",   6, "derived",
     "EM coupling in nuclear units\n"
     "= e²/(4πε₀) ≈ 1.43996 MeV·fm\n"
     "Bridge between SI EM and nuclear energy scales."),
    ("A_C_MEV",     "a_C",   6, "derived",
     "Coulomb coefficient (Bethe-Weizsäcker)\n"
     "= (3/5)·ke²/r₀ ≈ 0.7111 MeV\n"
     "NOT from SEMF — derived from first principles.\n"
     "σ-INVARIANT: pure electrostatics."),
    ("N0_FM3",      "n₀",    6, "measured",
     "Nuclear saturation density = 0.16 fm⁻³\n"
     "Where nuclear attraction and repulsion balance.\n"
     "Measured from electron scattering on heavy nuclei."),
    ("E_SAT_MEV",   "E_sat", 6, "measured",
     "Binding energy per nucleon at saturation = −16 MeV\n"
     "Measured from nuclear mass spectrometry."),
    ("K_SAT_MEV",   "K",     6, "measured",
     "Nuclear incompressibility = 230 MeV\n"
     "Stiffness of nuclear matter at saturation.\n"
     "Measured from giant monopole resonances.\n"
     "σ-DEPENDENT: K(σ) = K₀ × e^σ"),
    ("J_SYM_MEV",   "J",     6, "measured",
     "Symmetry energy at saturation = 32 MeV\n"
     "Energy cost of neutron-proton imbalance.\n"
     "Measured from nuclear mass differences."),

    # ── Layer 7: Astronomical ──────────────────────────────────────────
    ("M_SUN_KG","M_☉", 7, "measured",
     "Solar mass = 1.989×10³⁰ kg\n"
     "Reference mass for stellar/galactic scales."),
    ("L_SUN_W", "L_☉", 7, "measured",
     "Solar luminosity = 3.828×10²⁶ W\n"
     "IAU 2015 nominal solar luminosity."),
    ("AU_M",    "AU",  7, "measured",
     "Astronomical unit = 1.4960×10¹¹ m\n"
     "IAU 2012 exact definition."),

    # ── Layer 8: Physics API — Core Field ─────────────────────────────
    ("api_lorentz",    "lorentz_factor\n(v)",          8, "api",
     "field.relativity\nγ = 1/√(1−v²/c²)\nUses: c"),
    ("api_rest_energy","rest_energy\n(m₀)",            8, "api",
     "field.relativity\nE₀ = m₀c²\nUses: c"),
    ("api_sr_sigma",   "sigma_time\n_dilation(σ,t₀)", 8, "api",
     "field.relativity\nt = t₀ × e^σ\nUses: σ-field (scale_ratio)"),
    ("api_coulomb",    "coulomb_force\n(q₁,q₂,r)",    8, "api",
     "field.electrodynamics\nF = ke·q₁q₂/r²\nUses: e, ε₀"),
    ("api_larmor",     "larmor_power\n(q,a)",          8, "api",
     "field.electrodynamics\nP = q²a²/(6πε₀c³)\nUses: e, ε₀, c"),
    ("api_alpha_fn",   "fine_structure\n_constant()",  8, "api",
     "field.electrodynamics\nα = e²/(4πε₀ℏc)\nUses: e, ε₀, ℏ, c → verifies α"),
    ("api_schwarz",    "schwarzschild\n_radius(M)",    8, "api",
     "field.gr_basics / scale\nr_s = 2GM/c²\nUses: G, c"),
    ("api_hawking",    "hawking_temp\n(M)",            8, "api",
     "field.gr_basics\nT_H = ℏc³/(8πGMk_B)\nUses: ℏ, c, G, k_B"),
    ("api_sigma_hz",   "sigma_at\n_horizon(M)",        8, "api",
     "field.gr_basics\nAlways = ξ/2 ≈ 0.079\nMass-independent! Uses: ξ only\nKey SSBM consistency check."),
    ("api_binding",    "binding_energy\n_mev(Z,A)",    8, "api",
     "field.binding\nBethe-Weizsäcker + QCD terms\nUses: a_C, K, n₀, J"),
    ("api_scale",      "scale\n(e^σ,Λ_eff)",          8, "api",
     "field.scale\nscale_ratio(σ), lambda_eff(σ), sigma_from_potential\n"
     "Uses: ξ, Λ_QCD, G, c\nThe fundamental σ-field scaling engine"),
    ("api_bounds",     "bounds\n(check,clamp)",        8, "api",
     "field.bounds\nDomain validity checks for all σ quantities\n"
     "check_sigma, clamp_sigma, safe_proton_mass\n"
     "Uses: σ_conv, ξ, p_bare, Λ_QCD"),
    ("api_entangle",   "entanglement\n(η,decoherence)", 8, "api",
     "field.entanglement\nη-fraction, dark energy constraint\n"
     "decoherence_time, rendering connectivity\n"
     "Uses: η, ξ, G, c, k_B, ℏ"),

    # ── Layer 9: Physics API — Interfaces ─────────────────────────────
    ("api_proton_mev","proton_mass\n_mev(σ)",         9, "api",
     "field.nucleon\np(σ) = p_bare + p_QCD × e^σ\nUses: PROTON_BARE, Λ_QCD, σ\nShows σ-dependence of nucleon mass"),
    ("api_decay",     "decay\n(α,β,Gamow,σ)",         8, "api",
     "field.decay\nGamow tunneling factor, Geiger-Nuttall α-decay rates\n"
     "Q-value from mass excess, σ-coupling λ_eff = λ₀ × e^σ\n"
     "Uses: HBAR, C, e, α, AMU_KG"),
    ("api_plasma_f",  "plasma_freq\n(n_e)",           9, "api",
     "field.interface.plasma\nω_p = √(n_e e²/ε₀mₑ)\nUses: e, ε₀, mₑ"),
    ("api_debye",     "debye_length\n(n_e,T)",        9, "api",
     "field.interface.plasma\nλ_D = √(ε₀k_BT/n_e e²)\nUses: ε₀, k_B, e"),
    ("api_fermi",     "fermi_dirac\n(E,μ,T)",         9, "api",
     "field.interface.statistical\n1/(e^{ΔE/kT}+1)\nUses: k_B"),
    ("api_rms",       "rms_speed\n(m,T)",             9, "api",
     "field.interface.statistical\n√(3k_BT/m)\nUses: k_B"),
    ("api_stoq",      "stoq\n(structure)",            9, "api",
     "inventory.stoq\nResolve material → quarks\nUses: p_total, n_total"),

    # ── New interface modules ──────────────────────────────────────────
    ("api_acoustics", "acoustics\n(v_sound,Z)",       9, "api",
     "field.interface.acoustics\nv = √(B/ρ)\nUses: k_B (Debye model)\nσ-dep: bulk modulus shifts"),
    ("api_magnetism", "magnetism\n(χ,M,B)",           9, "api",
     "field.interface.magnetism\nCurie/Langevin/diamag\nUses: k_B, μ₀, μ_B\nσ-dep: Curie temperature"),
    ("api_nucleosynth","nucleosynthesis\n(Q,σ_cross)", 9, "api",
     "field.interface.nucleosynthesis\nGamow peak, pp-chain\nUses: ℏ, k_B, e, m_p\nσ-dep: Coulomb barrier"),
    ("api_rad_decay", "radioactive_decay\n(A(t),t½)",  9, "api",
     "field.interface.radioactive_decay\nBateman equations, α/β/γ\nUses: ℏ, e, α\nσ-dep: α-decay barrier"),
    ("api_elasticity","elasticity\n(σ,ε,E,ν)",        9, "api",
     "field.interface.elasticity\nLamé, Hooke, von Mises\nUses: mechanical K,E,G\nσ-dep: elastic modulus shift"),
    ("api_diffusion", "diffusion\n(D,J,c)",            9, "api",
     "field.interface.diffusion\nArrhenius, Fick's laws\nUses: k_B, thermal κ\nσ-dep: activation energy"),
    ("api_viscosity", "viscosity\n(F_drag,v_t)",       9, "api",
     "field.interface.viscosity\nStokes drag, Poiseuille\nUses: k_B"),
    ("api_electrochem","electrochemistry\n(E°,ΔG)",   9, "api",
     "field.interface.electrochemistry\nNernst, Faraday, Tafel\nUses: k_B, e, N_A\nσ-dep: thermal shift"),
    ("api_photonics", "photonics\n(NA,Bragg,SHG)",     9, "api",
     "field.interface.photonics\nWave optics, nonlinear χ²\nUses: ε₀, c, ℏ\nσ-dep: Bragg wavelength"),
    ("api_supercon",  "superconductivity\n(Δ,λ_L,H_c)",9, "api",
     "field.interface.superconductivity\nBCS gap, London depth, Abrikosov\n"
     "Uses: ℏ, k_B, e, μ₀, mₑ\n53 elements + 9 compounds\nσ-dep: T_c through Θ_D"),
    ("api_piezo",     "piezoelectricity\n(d,k,f_res)",  9, "api",
     "field.interface.piezoelectricity\nDirect/converse effect, coupling\nUses: ε₀\nσ-dep: resonant freq shift"),

    # ── New quantum physics modules ──────────────────────────────────
    ("api_atomic_spectra", "atomic_spectra\n(E_n,λ,Zeeman)", 9, "api",
     "field.interface.atomic_spectra\nRydberg formula, hydrogen levels, fine structure\n"
     "Uses: α, ℏ, mₑ, c, e, ε₀, k_B, μ_B\n"
     "Zeeman splitting, selection rules, QHO\n"
     "σ-dep: Rydberg energy through mₑ coupling"),
    ("api_quantum_wells", "quantum_wells\n(E_n,QD,DOS)", 9, "api",
     "field.interface.quantum_wells\nParticle-in-box, finite well, Brus equation\n"
     "Uses: ℏ, mₑ, e, ε₀\n11 QD materials (MEASURED)\n"
     "DOS in 0D-3D, quantum wire subbands\nσ-INVARIANT: EM confinement"),
    ("api_tunneling", "tunneling\n(T,WKB,STM)", 9, "api",
     "field.interface.tunneling\nRectangular barrier, WKB, double barrier\n"
     "Uses: ℏ, mₑ, e\nFowler-Nordheim, STM, Gamow factor\n"
     "10 metal work functions (MEASURED)\nσ-dep: barrier height"),
    ("api_angular_mom", "angular_momentum\n(J,CG,SO)", 9, "api",
     "field.interface.angular_momentum\nClebsch-Gordan (Racah), term symbols\n"
     "Uses: ℏ, α, mₑ, c, μ_B\nHund's rules, spin-orbit coupling\n"
     "Landé g-factor, Pauli matrices\nσ-INVARIANT: pure QM algebra"),

    # ── New interface modules (24) ───────────────────────────────────
    # Layer 9 — only constants, no inter-module deps
    ("api_acid_base",  "acid_base\n(pKa,pH,buffer)",    9, "api",
     "field.interface.acid_base\nKa from ΔG, Henderson-Hasselbalch, titration\n"
     "Uses: k_B, N_A, R_GAS\nσ-INVARIANT: EM equilibria"),
    ("api_solution",   "solution\n(Ksp,γ±,π)",          9, "api",
     "field.interface.solution\nDebye-Hückel, colligative props, Ksp\n"
     "Uses: k_B, N_A, R_GAS, ε₀, e\nσ-INVARIANT: EM equilibria"),
    ("api_phosphor",   "phosphor\n(I(t),screen)",       9, "api",
     "field.interface.phosphor\nPhosphorescent screen model\n"
     "Exponential decay, hit accumulation\nσ-INVARIANT: EM transition"),
    ("api_nbody",      "nbody\n(gravity,GW)",           9, "api",
     "field.interface.nbody\nN-body Verlet/Forest-Ruth integrator\n"
     "Uses: G, c, L_☉\nσ-dep: mass scaling via scale_ratio"),
    ("api_orbital",    "orbital\n(a,e,i,Ω)",            9, "api",
     "field.interface.orbital\nKepler orbit fitting from state vectors\n"
     "Uses: G (via GM anchors)\nσ-INVARIANT: pure gravity + geometry"),
    ("api_stat_full",  "statistical\n(f_FD,f_BE,v_rms)", 9, "api",
     "field.interface.statistical\nFermi-Dirac, Bose-Einstein, Maxwell-Boltzmann\n"
     "Uses: k_B, ℏ\nσ-dep: through mass in partition function"),

    # Layer 10 — depend on other interface modules
    ("api_adhesion",   "adhesion\n(W_adh,θ)",           10, "api",
     "field.interface.adhesion\nDupré work of adhesion, Young contact angle\n"
     "Uses: surface γ values\nσ-dep: through surface energy"),
    ("api_atmosphere", "atmosphere\n(P(z),Γ,RH)",       10, "api",
     "field.interface.atmosphere\nBarometric, lapse rate, Clausius-Clapeyron\n"
     "Uses: k_B, R_GAS, N_A\nσ-dep: molecular mass shift"),
    ("api_chem_rxn",   "chemical_rxns\n(ΔH,k(T),Keq)", 10, "api",
     "field.interface.chemical_reactions\nHess's law, Evans-Polanyi, Arrhenius\n"
     "Uses: N_A, k_B, ℏ\nσ-dep: activation energy shift"),
    ("api_cigar",      "cigar\n(combustion,flow)",      10, "api",
     "field.interface.cigar\nCarbon cigar gas-phase physics test\n"
     "Darcy flow, Hess combustion, σ-spectroscopy\nσ-dep: nuclear mass → flow"),
    ("api_composites", "composites\n(Voigt,Reuss,HS)",  10, "api",
     "field.interface.composites\nVoigt-Reuss-Hashin-Shtrikman bounds\n"
     "Uses: mechanical K,E,G + thermal κ\nσ-dep: through elastic moduli"),
    ("api_dielectric", "dielectric\n(ε_r,Drude,CM)",    10, "api",
     "field.interface.dielectric\nDrude/Clausius-Mossotti permittivity\n"
     "Uses: ε₀, e\nσ-INVARIANT: all EM"),
    ("api_hardness",   "hardness\n(HV,HB,Mohs)",       10, "api",
     "field.interface.hardness\nVickers/Brinell/Knoop from yield stress\n"
     "Uses: plasticity σ_y, mechanical E\nσ-dep: through yield stress"),
    ("api_ignition",   "ignition\n(T_auto,flash,q̇)",   10, "api",
     "field.interface.ignition\nAutoignition, flash point, burn rate\n"
     "Uses: k_B, N_A\nσ-dep: activation energy shift"),
    ("api_impact",     "impact\n(COR,E_diss)",          10, "api",
     "field.interface.impact\nCoeff of restitution from elastic-plastic contact\n"
     "Uses: mechanical E, plasticity σ_y\nσ-dep: through moduli"),
    ("api_liq_water",  "liquid_water\n(ρ,c_p,γ,η)",    10, "api",
     "field.interface.liquid_water\nTwo-state model + H-bond network\n"
     "Uses: k_B, N_A, R_GAS, h\nσ-dep: nuclear mass → anomalies"),
    ("api_mobius",     "mobius\n(L,Z,shield)",           10, "api",
     "field.interface.mobius\nMöbius conductor topology\n"
     "Inductance collapse, impedance → R\nσ-INVARIANT: pure EM"),
    ("api_organic",    "organic_materials\n(ΔH,T_b)",   10, "api",
     "field.interface.organic_materials\nHydrocarbon combustion, wood, bone\n"
     "Uses: N_A, k_B, ε₀\nσ-dep: nuclear mass → ZPE shift"),
    ("api_semicon_opt","semiconductor_optics\n(E_g,RGB)", 10, "api",
     "field.interface.semiconductor_optics\nVarshni band gap, Fresnel reflectance\n"
     "Uses: h, c from optics cascade\nσ-INVARIANT: all EM"),
    ("api_subsurface", "subsurface\n(μ_s,BSSRDF)",      10, "api",
     "field.interface.subsurface\nRayleigh/Mie scattering, diffusion approx\n"
     "Uses: optics n,k + photonics\nσ-INVARIANT: all EM"),
    ("api_texture",    "texture\n(Ra,BRDF,step)",       10, "api",
     "field.interface.texture\nAtomic step height, thermal roughness, Beckmann\n"
     "Uses: surface γ, k_B\nσ-dep: through surface energy"),
    ("api_therm_exp",  "thermal_expansion\n(α,Grüneisen)", 10, "api",
     "field.interface.thermal_expansion\nGrüneisen relation α = γC_v/(3V_mK)\n"
     "Uses: k_B, mechanical K, thermal C_v\nσ-dep: Θ_D and K shift"),
    ("api_viscoelast", "viscoelasticity\n(Maxwell,SLS)", 10, "api",
     "field.interface.viscoelasticity\nMaxwell, Kelvin-Voigt, Zener models\n"
     "Uses: k_B, ℏ, h, mechanical E\nσ-dep: relaxation time shift"),

    # ── Layer 10: Physics API — Interface Layer B ────────────────────
    ("api_surface",   "surface\n(γ,W_adh)",         10, "api",
     "field.interface.surface\nSurface energy from broken bonds\nUses: E_coh, k_B, Θ_D\nσ-dep: cohesive energy shift\nf_ZPE derived per material (not guessed)"),
    ("api_mechanical","mechanical\n(K,E,G,τ)",       10, "api",
     "field.interface.mechanical\nBulk/Young's/shear modulus, Frenkel strength\nUses: E_coh, AMU, k_B, Θ_D\nσ-dep: through cohesive energy\nf_ZPE derived per material from Debye T"),
    ("api_thermal",   "thermal\n(c_v,κ,Θ_D,v_D)",   10, "api",
     "field.interface.thermal\nDebye model, specific heat, conductivity\nv_D = Debye avg of v_L, v_T (1L+2T modes)\nΘ_D = ℏv_D(6π²n)^(1/3)/k_B\nUses: k_B, ℏ, K, G from mechanical\nσ-dep: Debye temperature shift"),
    ("api_optics",    "optics\n(n,k,R,T)",           10, "api",
     "field.interface.optics\nDrude model, Fresnel, Beer-Lambert\nUses: e, ε₀, ℏ, mₑ\nσ-INVARIANT: all EM"),
    ("api_electronics","electronics\n(ρ,σ_e,R_H)",  10, "api",
     "field.interface.electronics\nBloch-Grüneisen resistivity, Hall effect\nUses: ℏ, k_B, e, mₑ\nσ-dep: Debye temperature shift"),
    ("api_element",   "element\n(Z,A,props)",        10, "api",
     "field.interface.element\nPeriodic table: 118 elements\nSEMF binding from a_C (first principles)\nUses: A_C_MEV (derived Coulomb coeff)"),
    ("api_gas",       "gas\n(ρ,η,c_v)",              10, "api",
     "field.interface.gas\nIdeal gas, molecular properties\nUses: k_B, AMU, N_A\nσ-dep: reduced mass shift"),
    ("api_fluid",     "fluid\n(Re,v,Δp)",            10, "api",
     "field.interface.fluid\nNavier-Stokes, Reynolds, Poiseuille\nUses: k_B\nσ-dep: viscosity shift"),
    ("api_phase",     "phase_transition\n(T_m,T_b)", 10, "api",
     "field.interface.phase_transition\nMelting/boiling from Lindemann/Trouton\nUses: k_B, AMU\nσ-dep: through binding energy"),
    ("api_quantum",   "quantum\n(ψ,Δy,λ_dB)",       10, "api",
     "field.interface.quantum\nDouble-slit, tunneling, de Broglie\nUses: ℏ, mₑ\nσ-dep: neutron mass shift → fringe compression"),
    ("api_therm_em",  "thermal_emission\n(B,ε,RGB)", 10, "api",
     "field.interface.thermal_emission\nPlanck B(λ,T) × Kirchhoff ε(λ)\nUses: h, c, k_B, n+ik\nσ-INVARIANT: all EM/QED"),
    ("api_grain",     "grain_structure\n(σ_y,H)",    10, "api",
     "field.interface.grain_structure\nHall-Petch, Taylor factor per structure\nUses: K,E,G from mechanical\nσ-dep: through elastic moduli\nTaylor M: FCC=3.06, BCC=2.75, HCP=4.5"),
    ("api_crystal",   "crystal_field\n(Δ,colour)",   10, "api",
     "field.interface.crystal_field\nTanabe-Sugano, d-orbital splitting\nUses: ε₀, BOHR_RADIUS\nσ-INVARIANT: EM crystal field"),
    ("api_mol_bonds", "molecular_bonds\n(k,ν,E_d)",  10, "api",
     "field.interface.molecular_bonds\nBadger's rule force constants\nUses: AMU, ℏ\nDERIVED k from bond length (not guessed)"),
    ("api_friction",  "friction\n(μ,F_f)",           10, "api",
     "field.interface.friction\nBowden-Tabor adhesion model\nUses: surface γ, shear G\nσ-dep: through surface energy"),
    ("api_stress",    "stress\n(σ_y,K_IC)",          10, "api",
     "field.interface.stress\nYield, fracture toughness, fatigue\nUses: E, G from mechanical\nσ-dep: through elastic moduli"),
    ("api_thermo_e",  "thermoelectric\n(S,ZT,η)",    10, "api",
     "field.interface.thermoelectric\nSeebeck, Peltier, ZT figure of merit\nUses: k_B, e\nσ-dep: through Debye T"),
    ("api_corrosion", "corrosion\n(v_corr,Δm)",      10, "api",
     "field.interface.corrosion\nPilling-Bedworth, Arrhenius oxidation\nUses: k_B, R_GAS\nσ-dep: activation energy shift"),
    ("api_wear",      "wear\n(V,k_w)",               10, "api",
     "field.interface.wear\nArchard equation, abrasive/adhesive\nUses: hardness H from grain_structure\nσ-dep: through hardness"),
    ("api_h_bond",    "hydrogen_bonding\n(E_HB)",    10, "api",
     "field.interface.hydrogen_bonding\nHydrogen bond energies, water properties\nUses: k_B, e, ε₀\nσ-INVARIANT: EM interaction"),
    ("api_hysteresis","hysteresis\n(M,H_c,B_r)",     10, "api",
     "field.interface.hysteresis\nFerromagnetic hysteresis loops\nUses: k_B, μ₀\nσ-dep: Curie temperature shift"),
    ("api_plasticity","plasticity\n(σ_flow,ε_p)",    10, "api",
     "field.interface.plasticity\nVoce/Ludwik hardening, strain rate\nUses: k_B, mechanical K,E\nσ-dep: through elastic moduli"),

    # ── Quantum computing stack (Layer 10 — builds on Layer 9) ───────
    ("api_qc",       "quantum_computing\n(gates,qubits)", 10, "api",
     "field.interface.quantum_computing\nDecoherence-free state-vector simulator\n"
     "Uses: ℏ, e, k_B, mₑ, μ_B, h\n"
     "16 gates (X,Y,Z,H,S,T,Rx,Ry,Rz,CNOT,CZ,SWAP,iSWAP,Toffoli,Fredkin)\n"
     "4 qubit types from cascade:\n"
     "  Transmon: BCS gap → E_J → ω₀₁\n"
     "  Spin: Zeeman → Larmor freq\n"
     "  QD: confinement → level spacing\n"
     "  NV center: D=2.87 GHz (MEASURED)\n"
     "σ-dep: qubit frequency through BCS gap"),
    ("api_qo",       "quantum_output\n(measure,sample)", 10, "api",
     "field.interface.quantum_output\nBorn rule measurement, sampling, extraction\n"
     "Uses: quantum_computing state vectors\n"
     "Pauli expectation values, fidelity, entropy\n"
     "Schmidt decomposition, Bloch sphere\n"
     "5 algorithms: Bell, Deutsch-Jozsa, Grover,\n"
     "  teleportation, Bernstein-Vazirani\n"
     "Classical extraction chain → useable computation"),
    ("api_qa",       "quantum_algorithms\n(10 algorithms)", 11, "api",
     "field.interface.quantum_algorithms\n10 quantum algorithms:\n"
     "QFT, QPE, Shor(15), Simon, QAOA MaxCut,\n"
     "Ising VQE, Heisenberg VQE, HeH+ VQE,\n"
     "QEC bit-flip, Quantum Walk\n"
     "Cascade: J from T_C, B_c prediction"),
    ("api_qm",       "quantum_matter\n(Mott,crystal_field)", 11, "api",
     "field.interface.quantum_matter\nMaterial-specific quantum predictions:\n"
     "Mott phase diagram from cascade (U/t)\n"
     "Crystal field → spin Hamiltonian → VQE\n"
     "Tanabe-Sugano crossover = Mott transition\n"
     "Nephelauxetic β = metallicity ranking\n"
     "Itinerant vs localized magnetism (J_super/J_Curie)"),
]

# (source_id, target_id, edge_type, formula_label)
EDGES = [
    # Derived fundamentals
    ("HBAR",     "L_PLANCK",     "formula", "√(ℏG/c³)"),
    ("G",        "L_PLANCK",     "formula", ""),
    ("C",        "L_PLANCK",     "formula", ""),
    ("HBAR",     "M_PLANCK_KG",  "formula", "√(ℏc/G)"),
    ("G",        "M_PLANCK_KG",  "formula", ""),
    ("C",        "M_PLANCK_KG",  "formula", ""),
    ("E_CHARGE", "ALPHA",        "formula", "e²/(4πε₀ℏc)"),
    ("EPS_0",    "ALPHA",        "formula", ""),
    ("HBAR",     "ALPHA",        "formula", ""),
    ("C",        "ALPHA",        "formula", ""),
    ("E_CHARGE", "M_ELECTRON_KG","formula", "Higgs coupling"),
    ("L_PLANCK", "SIGMA_FLOOR",  "formula", "l_P·H₀/c"),
    ("H0",       "SIGMA_FLOOR",  "formula", ""),
    ("C",        "SIGMA_FLOOR",  "formula", ""),
    ("SIGMA_FLOOR","SIGMA_HERE", "formula", "σ_here = σ_floor"),
    ("G",        "M_HUBBLE_KG",  "formula", "c³/(2GH₀)"),
    ("C",        "M_HUBBLE_KG",  "formula", ""),
    ("H0",       "M_HUBBLE_KG",  "formula", ""),
    # New derived constants
    ("HBAR",         "H_PLANCK",        "formula", "h = 2πℏ"),
    ("K_B",          "STEFAN_BOLTZMANN", "formula", "π²k_B⁴/(60ℏ³c²)"),
    ("HBAR",         "STEFAN_BOLTZMANN", "formula", ""),
    ("C",            "STEFAN_BOLTZMANN", "formula", ""),
    ("HBAR",         "BOHR_RADIUS",     "formula", "ℏ/(mₑcα)"),
    ("M_ELECTRON_KG","BOHR_RADIUS",     "formula", ""),
    ("C",            "BOHR_RADIUS",     "formula", ""),
    ("ALPHA",        "BOHR_RADIUS",     "formula", ""),
    ("E_CHARGE",     "MU_BOHR",         "formula", "eℏ/(2mₑ)"),
    ("HBAR",         "MU_BOHR",         "formula", ""),
    ("M_ELECTRON_KG","MU_BOHR",         "formula", ""),
    ("K_B",          "R_GAS",           "formula", "k_B × N_A"),
    ("N_AVOGADRO",   "R_GAS",           "formula", ""),
    ("E_CHARGE",     "EV_TO_J",         "formula", "1 eV ≡ e × 1 V"),

    # SSBM
    ("XI", "SIGMA_CONV", "ssbm", "−ln(ξ)"),
    ("XI", "ETA",        "ssbm", "dark energy constraint"),

    # σ-field
    ("XI", "SIGMA_FIELD", "formula", "σ = ξGM/rc²"),
    ("G",  "SIGMA_FIELD", "formula", ""),
    ("C",  "SIGMA_FIELD", "formula", ""),

    # QCD coupling
    ("LAMBDA_QCD_MEV", "SIGMA_FIELD",     "qcd",   "Λ_eff scales with σ"),
    ("LAMBDA_QCD_MEV", "PROTON_QCD_MEV",  "qcd",   "sets ~99% of nucleon mass"),
    ("SIGMA_FIELD",    "PROTON_QCD_MEV",  "sigma", "Λ_eff = Λ_QCD·e^σ"),
    ("SIGMA_FIELD",    "K_SAT_MEV",       "sigma", "K(σ) = K₀·e^σ"),

    # QCD fraction and mass differences
    ("PROTON_QCD_MEV",  "PROTON_QCD_FRAC", "formula", "p_QCD/p_total"),
    ("PROTON_TOTAL_MEV","PROTON_QCD_FRAC", "formula", ""),
    ("NEUTRON_TOTAL_MEV","DELTA_NP",       "formula", "m_n − m_p (measured directly)"),
    ("PROTON_TOTAL_MEV", "DELTA_NP",       "formula", ""),

    # Nucleon derivation
    ("M_UP_MEV",        "PROTON_BARE_MEV", "formula", "2m_u + m_d"),
    ("M_DOWN_MEV",      "PROTON_BARE_MEV", "formula", ""),
    ("PROTON_TOTAL_MEV","PROTON_QCD_MEV",  "formula", "p_total − p_bare"),
    ("PROTON_BARE_MEV", "PROTON_QCD_MEV",  "formula", ""),

    # Nuclear EM
    ("E_CHARGE",    "KE_E2_MEV_FM", "formula", "e²/(4πε₀)"),
    ("EPS_0",       "KE_E2_MEV_FM", "formula", ""),
    ("KE_E2_MEV_FM","A_C_MEV",      "formula", "(3/5)·ke²/r₀"),
    ("R0_FM",       "A_C_MEV",      "formula", ""),

    # API uses
    ("C",              "api_lorentz",    "uses", ""),
    ("C",              "api_rest_energy","uses", ""),
    ("SIGMA_FIELD",    "api_sr_sigma",   "uses", "e^σ"),
    ("E_CHARGE",       "api_coulomb",    "uses", ""),
    ("EPS_0",          "api_coulomb",    "uses", ""),
    ("E_CHARGE",       "api_larmor",     "uses", ""),
    ("EPS_0",          "api_larmor",     "uses", ""),
    ("C",              "api_larmor",     "uses", ""),
    ("ALPHA",          "api_alpha_fn",   "uses", ""),
    ("E_CHARGE",       "api_alpha_fn",   "uses", ""),
    ("EPS_0",          "api_alpha_fn",   "uses", ""),
    ("HBAR",           "api_alpha_fn",   "uses", ""),
    ("C",              "api_alpha_fn",   "uses", ""),
    ("G",              "api_schwarz",    "uses", ""),
    ("C",              "api_schwarz",    "uses", ""),
    ("HBAR",           "api_hawking",    "uses", ""),
    ("C",              "api_hawking",    "uses", ""),
    ("G",              "api_hawking",    "uses", ""),
    ("K_B",            "api_hawking",    "uses", ""),
    ("XI",             "api_sigma_hz",   "uses", "ξ/2 always"),
    ("A_C_MEV",        "api_binding",    "uses", ""),
    ("K_SAT_MEV",      "api_binding",    "uses", ""),
    ("N0_FM3",         "api_binding",    "uses", ""),
    ("J_SYM_MEV",      "api_binding",    "uses", ""),
    # scale
    ("XI",              "api_scale",     "uses", ""),
    ("LAMBDA_QCD_MEV",  "api_scale",     "uses", ""),
    ("G",               "api_scale",     "uses", ""),
    ("C",               "api_scale",     "uses", ""),
    ("SIGMA_FIELD",     "api_scale",     "sigma","e^σ scaling"),
    # bounds
    ("SIGMA_CONV",      "api_bounds",    "uses", ""),
    ("XI",              "api_bounds",    "uses", ""),
    ("PROTON_BARE_MEV", "api_bounds",    "uses", ""),
    ("LAMBDA_QCD_MEV",  "api_bounds",    "uses", ""),
    # entanglement
    ("ETA",             "api_entangle",  "uses", ""),
    ("XI",              "api_entangle",  "uses", ""),
    ("G",               "api_entangle",  "uses", ""),
    ("C",               "api_entangle",  "uses", ""),
    ("K_B",             "api_entangle",  "uses", ""),
    ("HBAR",            "api_entangle",  "uses", ""),
    ("SIGMA_FIELD",     "api_entangle",  "sigma","decoherence"),
    ("PROTON_BARE_MEV", "api_proton_mev","uses", ""),
    ("LAMBDA_QCD_MEV",  "api_proton_mev","uses", ""),
    ("SIGMA_FIELD",     "api_proton_mev","uses", "e^σ"),
    ("HBAR",            "api_decay",     "uses", ""),
    ("C",               "api_decay",     "uses", ""),
    ("E_CHARGE",        "api_decay",     "uses", ""),
    ("ALPHA",           "api_decay",     "uses", ""),
    ("AMU_KG",          "api_decay",     "uses", ""),
    ("SIGMA_FIELD",     "api_decay",     "sigma","λ_eff = λ₀·e^σ"),
    ("E_CHARGE",        "api_plasma_f",  "uses", ""),
    ("EPS_0",           "api_plasma_f",  "uses", ""),
    ("M_ELECTRON_KG",   "api_plasma_f",  "uses", ""),
    ("EPS_0",           "api_debye",     "uses", ""),
    ("K_B",             "api_debye",     "uses", ""),
    ("E_CHARGE",        "api_debye",     "uses", ""),
    ("K_B",             "api_fermi",     "uses", ""),
    ("K_B",             "api_rms",       "uses", ""),
    ("PROTON_TOTAL_MEV","api_stoq",      "uses", ""),
    ("NEUTRON_TOTAL_MEV","api_stoq",     "uses", ""),

    # New interface modules — edges
    ("K_B",             "api_acoustics",   "uses", ""),
    ("SIGMA_FIELD",     "api_acoustics",   "sigma","bulk modulus shift"),
    ("K_B",             "api_magnetism",   "uses", ""),
    ("MU_0",            "api_magnetism",   "uses", ""),
    ("SIGMA_FIELD",     "api_magnetism",   "sigma","Curie T shift"),
    ("HBAR",            "api_nucleosynth", "uses", ""),
    ("K_B",             "api_nucleosynth", "uses", ""),
    ("E_CHARGE",        "api_nucleosynth", "uses", ""),
    ("SIGMA_FIELD",     "api_nucleosynth", "sigma","Coulomb barrier"),
    ("HBAR",            "api_rad_decay",   "uses", ""),
    ("E_CHARGE",        "api_rad_decay",   "uses", ""),
    ("ALPHA",           "api_rad_decay",   "uses", ""),
    ("SIGMA_FIELD",     "api_rad_decay",   "sigma","α barrier shift"),
    ("K_B",             "api_diffusion",   "uses", ""),
    ("SIGMA_FIELD",     "api_diffusion",   "sigma","E_a shift"),
    ("K_B",             "api_viscosity",   "uses", ""),
    ("K_B",             "api_electrochem", "uses", ""),
    ("E_CHARGE",        "api_electrochem", "uses", ""),
    ("SIGMA_FIELD",     "api_electrochem", "sigma","thermal shift"),
    ("EPS_0",           "api_photonics",   "uses", ""),
    ("C",               "api_photonics",   "uses", ""),
    ("HBAR",            "api_photonics",   "uses", ""),
    ("SIGMA_FIELD",     "api_photonics",   "sigma","Bragg shift"),
    ("HBAR",            "api_supercon",    "uses", ""),
    ("K_B",             "api_supercon",    "uses", ""),
    ("E_CHARGE",        "api_supercon",    "uses", ""),
    ("MU_0",            "api_supercon",    "uses", ""),
    ("M_ELECTRON_KG",   "api_supercon",    "uses", ""),
    ("SIGMA_FIELD",     "api_supercon",    "sigma","T_c through Θ_D"),
    ("EPS_0",           "api_piezo",       "uses", ""),
    ("SIGMA_FIELD",     "api_piezo",       "sigma","resonant freq"),

    # ── Layer 10 edges — Interface Layer B ────────────────────────────
    # surface
    ("K_B",             "api_surface",     "uses", ""),
    ("SIGMA_FIELD",     "api_surface",     "sigma","cohesive energy"),
    # mechanical
    ("K_B",             "api_mechanical",  "uses", ""),
    ("AMU_KG",          "api_mechanical",  "uses", ""),
    ("SIGMA_FIELD",     "api_mechanical",  "sigma","bulk modulus"),
    # thermal (uses K and G from mechanical for Debye average v_D)
    ("K_B",             "api_thermal",     "uses", ""),
    ("HBAR",            "api_thermal",     "uses", ""),
    ("N_AVOGADRO",      "api_thermal",     "uses", ""),
    ("api_mechanical",  "api_thermal",     "uses", "K, G → v_L, v_T → v_D"),
    ("SIGMA_FIELD",     "api_thermal",     "sigma","Debye Θ_D"),
    # optics
    ("E_CHARGE",        "api_optics",      "uses", ""),
    ("EPS_0",           "api_optics",      "uses", ""),
    ("M_ELECTRON_KG",   "api_optics",      "uses", ""),
    # electronics
    ("HBAR",            "api_electronics", "uses", ""),
    ("K_B",             "api_electronics", "uses", ""),
    ("E_CHARGE",        "api_electronics", "uses", ""),
    ("SIGMA_FIELD",     "api_electronics", "sigma","Debye Θ_D"),
    # element
    ("A_C_MEV",         "api_element",     "uses", ""),
    # gas
    ("K_B",             "api_gas",         "uses", ""),
    ("AMU_KG",          "api_gas",         "uses", ""),
    ("SIGMA_FIELD",     "api_gas",         "sigma","reduced mass"),
    # fluid
    ("K_B",             "api_fluid",       "uses", ""),
    ("SIGMA_FIELD",     "api_fluid",       "sigma","viscosity"),
    # phase_transition
    ("K_B",             "api_phase",       "uses", ""),
    ("AMU_KG",          "api_phase",       "uses", ""),
    ("SIGMA_FIELD",     "api_phase",       "sigma","binding energy"),
    # quantum
    ("HBAR",            "api_quantum",     "uses", ""),
    ("SIGMA_FIELD",     "api_quantum",     "sigma","neutron mass → fringe"),
    # thermal_emission
    ("H_PLANCK",        "api_therm_em",    "uses", ""),
    ("C",               "api_therm_em",    "uses", ""),
    ("K_B",             "api_therm_em",    "uses", ""),
    # grain_structure
    ("SIGMA_FIELD",     "api_grain",       "sigma","elastic moduli"),
    # crystal_field
    ("EPS_0",           "api_crystal",     "uses", ""),
    ("BOHR_RADIUS",     "api_crystal",     "uses", ""),
    # molecular_bonds
    ("AMU_KG",          "api_mol_bonds",   "uses", ""),
    ("HBAR",            "api_mol_bonds",   "uses", ""),
    # friction
    ("SIGMA_FIELD",     "api_friction",    "sigma","surface energy"),
    # stress
    ("SIGMA_FIELD",     "api_stress",      "sigma","elastic moduli"),
    # thermoelectric
    ("K_B",             "api_thermo_e",    "uses", ""),
    ("E_CHARGE",        "api_thermo_e",    "uses", ""),
    ("SIGMA_FIELD",     "api_thermo_e",    "sigma","Debye Θ_D"),
    # corrosion
    ("K_B",             "api_corrosion",   "uses", ""),
    ("R_GAS",           "api_corrosion",   "uses", ""),
    ("SIGMA_FIELD",     "api_corrosion",   "sigma","activation energy"),
    # wear
    ("SIGMA_FIELD",     "api_wear",        "sigma","hardness"),
    # hydrogen_bonding
    ("K_B",             "api_h_bond",      "uses", ""),
    ("E_CHARGE",        "api_h_bond",      "uses", ""),
    # hysteresis
    ("K_B",             "api_hysteresis",  "uses", ""),
    ("MU_0",            "api_hysteresis",  "uses", ""),
    ("SIGMA_FIELD",     "api_hysteresis",  "sigma","Curie T"),
    # plasticity
    ("K_B",             "api_plasticity",  "uses", ""),
    ("SIGMA_FIELD",     "api_plasticity",  "sigma","elastic moduli"),
    # ── New quantum physics modules — edges ────────────────────────────
    # atomic_spectra
    ("ALPHA",           "api_atomic_spectra", "uses", ""),
    ("HBAR",            "api_atomic_spectra", "uses", ""),
    ("M_ELECTRON_KG",   "api_atomic_spectra", "uses", ""),
    ("C",               "api_atomic_spectra", "uses", ""),
    ("E_CHARGE",        "api_atomic_spectra", "uses", ""),
    ("K_B",             "api_atomic_spectra", "uses", ""),
    ("MU_BOHR",         "api_atomic_spectra", "uses", ""),
    ("SIGMA_FIELD",     "api_atomic_spectra", "sigma", "Rydberg energy"),
    # quantum_wells
    ("HBAR",            "api_quantum_wells",  "uses", ""),
    ("M_ELECTRON_KG",   "api_quantum_wells",  "uses", ""),
    ("E_CHARGE",        "api_quantum_wells",  "uses", ""),
    ("EPS_0",           "api_quantum_wells",  "uses", ""),
    # tunneling
    ("HBAR",            "api_tunneling",      "uses", ""),
    ("M_ELECTRON_KG",   "api_tunneling",      "uses", ""),
    ("E_CHARGE",        "api_tunneling",      "uses", ""),
    ("SIGMA_FIELD",     "api_tunneling",      "sigma", "barrier height"),
    # angular_momentum
    ("HBAR",            "api_angular_mom",    "uses", ""),
    ("ALPHA",           "api_angular_mom",    "uses", ""),
    ("M_ELECTRON_KG",   "api_angular_mom",    "uses", ""),
    ("C",               "api_angular_mom",    "uses", ""),
    ("MU_BOHR",         "api_angular_mom",    "uses", ""),
    # quantum_computing (Layer 10 — uses Layer 9 modules)
    ("HBAR",            "api_qc",             "uses", ""),
    ("E_CHARGE",        "api_qc",             "uses", ""),
    ("K_B",             "api_qc",             "uses", ""),
    ("M_ELECTRON_KG",   "api_qc",             "uses", ""),
    ("MU_BOHR",         "api_qc",             "uses", ""),
    ("H_PLANCK",        "api_qc",             "uses", ""),
    ("api_supercon",    "api_qc",             "uses", "BCS gap → transmon"),
    ("api_atomic_spectra","api_qc",           "uses", "Zeeman → spin qubit"),
    ("api_quantum_wells","api_qc",            "uses", "confinement → QD qubit"),
    ("SIGMA_FIELD",     "api_qc",             "sigma", "qubit frequency"),
    # quantum_output (Layer 10 — uses quantum_computing)
    ("api_qc",          "api_qo",             "uses", "state vectors + gates"),

    # quantum_algorithms (Layer 11 — uses qc + qo + magnetism)
    ("api_qc",          "api_qa",             "uses", "gate engine"),
    ("api_qo",          "api_qa",             "uses", "measurement + expectation"),
    ("api_magnetism",   "api_qa",             "uses", "T_C → J coupling"),

    # quantum_matter (Layer 11 — bridges material DB to quantum Hamiltonians)
    ("api_qc",          "api_qm",             "uses", "gate engine"),
    ("api_qo",          "api_qm",             "uses", "Pauli expectations"),
    ("api_surface",     "api_qm",             "uses", "E_coh, ρ, a, Z"),
    ("api_magnetism",   "api_qm",             "uses", "T_C, n_unpaired"),
    ("api_crystal",     "api_qm",             "uses", "10Dq, B, β"),
    ("api_supercon",    "api_qm",             "uses", "λ_ep, T_c"),
    ("K_B",             "api_qm",             "uses", ""),
    ("E_CHARGE",        "api_qm",             "uses", ""),
    ("M_ELECTRON_KG",   "api_qm",             "uses", ""),

    # ── New interface modules (24) — edges ──────────────────────────
    # acid_base (Layer 9)
    ("K_B",             "api_acid_base",   "uses", ""),
    ("N_AVOGADRO",      "api_acid_base",   "uses", ""),
    ("R_GAS",           "api_acid_base",   "uses", ""),
    # solution (Layer 9)
    ("K_B",             "api_solution",    "uses", ""),
    ("N_AVOGADRO",      "api_solution",    "uses", ""),
    ("R_GAS",           "api_solution",    "uses", ""),
    ("EPS_0",           "api_solution",    "uses", ""),
    ("E_CHARGE",        "api_solution",    "uses", ""),
    # phosphor (Layer 9) — no constants imports, pure math
    # nbody (Layer 9)
    ("G",               "api_nbody",       "uses", ""),
    ("C",               "api_nbody",       "uses", ""),
    ("L_SUN_W",         "api_nbody",       "uses", ""),
    ("SIGMA_FIELD",     "api_nbody",       "sigma","mass scaling"),
    # orbital (Layer 9)
    ("G",               "api_orbital",     "uses", ""),
    # statistical full (Layer 9)
    ("K_B",             "api_stat_full",   "uses", ""),
    ("HBAR",            "api_stat_full",   "uses", ""),
    ("SIGMA_FIELD",     "api_stat_full",   "sigma","mass in partition fn"),
    # adhesion (Layer 10)
    ("api_surface",     "api_adhesion",    "uses", "γ values"),
    ("SIGMA_FIELD",     "api_adhesion",    "sigma","surface energy"),
    # atmosphere (Layer 10)
    ("K_B",             "api_atmosphere",  "uses", ""),
    ("R_GAS",           "api_atmosphere",  "uses", ""),
    ("N_AVOGADRO",      "api_atmosphere",  "uses", ""),
    ("api_gas",         "api_atmosphere",  "uses", "cp, γ, η"),
    ("api_liq_water",   "api_atmosphere",  "uses", "L_vap"),
    ("SIGMA_FIELD",     "api_atmosphere",  "sigma","molecular mass"),
    # chemical_reactions (Layer 10)
    ("N_AVOGADRO",      "api_chem_rxn",    "uses", ""),
    ("K_B",             "api_chem_rxn",    "uses", ""),
    ("HBAR",            "api_chem_rxn",    "uses", ""),
    ("api_mol_bonds",   "api_chem_rxn",    "uses", "bond energies"),
    ("api_organic",     "api_chem_rxn",    "uses", "combustion ΔH"),
    # cigar (Layer 10)
    ("api_gas",         "api_cigar",       "uses", "gas properties"),
    ("api_chem_rxn",    "api_cigar",       "uses", "combustion"),
    ("SIGMA_FIELD",     "api_cigar",       "sigma","nuclear mass"),
    # composites (Layer 10)
    ("api_mechanical",  "api_composites",  "uses", "K,E,G"),
    ("api_thermal",     "api_composites",  "uses", "κ"),
    ("SIGMA_FIELD",     "api_composites",  "sigma","elastic moduli"),
    # dielectric (Layer 10)
    ("EPS_0",           "api_dielectric",  "uses", ""),
    ("E_CHARGE",        "api_dielectric",  "uses", ""),
    ("api_electronics", "api_dielectric",  "uses", "n_e, ωp"),
    ("api_optics",      "api_dielectric",  "uses", "n, Cauchy"),
    # hardness (Layer 10)
    ("api_plasticity",  "api_hardness",    "uses", "σ_y"),
    ("api_mechanical",  "api_hardness",    "uses", "E"),
    ("SIGMA_FIELD",     "api_hardness",    "sigma","yield stress"),
    # ignition (Layer 10)
    ("K_B",             "api_ignition",    "uses", ""),
    ("N_AVOGADRO",      "api_ignition",    "uses", ""),
    ("api_chem_rxn",    "api_ignition",    "uses", "E_a"),
    ("api_organic",     "api_ignition",    "uses", "ΔH_comb"),
    ("SIGMA_FIELD",     "api_ignition",    "sigma","activation energy"),
    # impact (Layer 10)
    ("api_mechanical",  "api_impact",      "uses", "E"),
    ("api_plasticity",  "api_impact",      "uses", "σ_y"),
    ("SIGMA_FIELD",     "api_impact",      "sigma","moduli"),
    # liquid_water (Layer 10)
    ("K_B",             "api_liq_water",   "uses", ""),
    ("N_AVOGADRO",      "api_liq_water",   "uses", ""),
    ("R_GAS",           "api_liq_water",   "uses", ""),
    ("H_PLANCK",        "api_liq_water",   "uses", ""),
    ("api_h_bond",      "api_liq_water",   "uses", "H-bond energy"),
    ("SIGMA_FIELD",     "api_liq_water",   "sigma","nuclear mass"),
    # mobius (Layer 10) — no constants imports, pure EM math
    # organic_materials (Layer 10)
    ("N_AVOGADRO",      "api_organic",     "uses", ""),
    ("K_B",             "api_organic",     "uses", ""),
    ("EPS_0",           "api_organic",     "uses", ""),
    ("api_mol_bonds",   "api_organic",     "uses", "bond energies"),
    ("api_h_bond",      "api_organic",     "uses", "intermolecular"),
    # semiconductor_optics (Layer 10)
    ("api_optics",      "api_semicon_opt", "uses", "Fresnel"),
    # subsurface (Layer 10)
    ("api_optics",      "api_subsurface",  "uses", "n,k"),
    ("api_photonics",   "api_subsurface",  "uses", "absorption"),
    # texture (Layer 10)
    ("api_surface",     "api_texture",     "uses", "γ, lattice"),
    ("K_B",             "api_texture",     "uses", ""),
    ("SIGMA_FIELD",     "api_texture",     "sigma","surface energy"),
    # thermal_expansion (Layer 10)
    ("api_mechanical",  "api_therm_exp",   "uses", "K"),
    ("api_thermal",     "api_therm_exp",   "uses", "C_v, Θ_D"),
    ("K_B",             "api_therm_exp",   "uses", ""),
    ("SIGMA_FIELD",     "api_therm_exp",   "sigma","Θ_D and K shift"),
    # viscoelasticity (Layer 10)
    ("K_B",             "api_viscoelast",  "uses", ""),
    ("HBAR",            "api_viscoelast",  "uses", ""),
    ("H_PLANCK",        "api_viscoelast",  "uses", ""),
    ("api_mechanical",  "api_viscoelast",  "uses", "E"),
    ("api_thermal",     "api_viscoelast",  "uses", "Θ_D"),
    ("api_diffusion",   "api_viscoelast",  "uses", "E_a"),
    ("SIGMA_FIELD",     "api_viscoelast",  "sigma","relaxation time"),

    # SIGMA_HERE → all σ-dependent APIs (observer frame)
    ("SIGMA_HERE",      "SIGMA_FIELD",     "formula", "reference frame"),
    # f_QCD fraction → all σ-correction functions
    ("PROTON_QCD_FRAC", "api_surface",     "uses", "σ-correction"),
    ("PROTON_QCD_FRAC", "api_mechanical",  "uses", "σ-correction"),
    ("PROTON_QCD_FRAC", "api_proton_mev",  "uses", "σ-correction"),
]


def build_html(nodes, edges, layer_names):
    nodes_data = [
        {"id": n[0], "label": n[1], "layer": n[2], "type": n[3], "tip": n[4]}
        for n in nodes
    ]
    edges_data = [
        {"src": e[0], "tgt": e[1], "type": e[2], "label": e[3]}
        for e in edges
    ]
    graph_json = json.dumps(
        {"nodes": nodes_data, "edges": edges_data, "layers": layer_names},
        ensure_ascii=False
    )
    return HTML_TEMPLATE.replace("__GRAPH_DATA__", graph_json)


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>sigma-ground: Physics Dependency Chart</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #1a1a2e; color: #e0e0e0; font-family: 'Segoe UI', sans-serif; }
#header { padding: 18px 28px 10px; }
#header h1 { font-size: 1.35rem; font-weight: 600; color: #a8d8ea; letter-spacing: 0.04em; }
#header p  { font-size: 0.78rem; color: #888; margin-top: 4px; }
#legend { display: flex; flex-wrap: wrap; gap: 8px 18px; padding: 6px 28px 12px; }
.leg { display: flex; align-items: center; gap: 6px; font-size: 0.72rem; color: #bbb; }
.leg-dot { width: 12px; height: 12px; border-radius: 3px; border: 1.5px solid; }
.leg-line { width: 22px; height: 3px; border-radius: 2px; }
#scroll { overflow-x: auto; padding: 0 16px 24px; }
svg { display: block; }
.node rect { rx: 7; ry: 7; cursor: pointer; transition: opacity 0.15s; }
.node text { pointer-events: none; font-size: 10.5px; dominant-baseline: middle; }
.node.dim rect { opacity: 0.18; }
.node.dim text { opacity: 0.18; }
.node.hi  rect { filter: drop-shadow(0 0 6px rgba(255,255,180,0.7)); }
.edge { fill: none; transition: opacity 0.15s; }
.edge.dim { opacity: 0.06; }
.edge.hi  { opacity: 1 !important; stroke-width: 2.5px !important; }
.band { opacity: 0.07; }
.band-label { font-size: 10px; fill: #aaa; font-weight: 500; letter-spacing: 0.05em; }
#tooltip {
  position: fixed; display: none; max-width: 280px;
  background: #0d1b2a; border: 1px solid #445; border-radius: 8px;
  padding: 10px 13px; font-size: 0.76rem; line-height: 1.55;
  color: #dde; pointer-events: none; z-index: 100;
  box-shadow: 0 4px 18px rgba(0,0,0,0.6);
  white-space: pre-wrap;
}
#tooltip strong { color: #a8d8ea; display: block; margin-bottom: 4px; font-size: 0.82rem; }
</style>
</head>
<body>
<div id="header">
  <h1>sigma-ground — Physics Dependency Chart</h1>
  <p>Hover a node to trace its ancestry. Hover an edge to see the formula. Click to pin.</p>
</div>
<div id="legend">
  <span class="leg"><span class="leg-dot" style="background:#D6EAF8;border-color:#2980B9"></span>Measured</span>
  <span class="leg"><span class="leg-dot" style="background:#D5F5E3;border-color:#27AE60"></span>Derived by formula</span>
  <span class="leg"><span class="leg-dot" style="background:#FDEBD0;border-color:#E67E22"></span>SSBM parameter</span>
  <span class="leg"><span class="leg-dot" style="background:#E8DAEF;border-color:#8E44AD"></span>Higgs-origin (σ-invariant)</span>
  <span class="leg"><span class="leg-dot" style="background:#FADBD8;border-color:#C0392B"></span>QCD-origin (σ-dependent)</span>
  <span class="leg"><span class="leg-dot" style="background:#FFFDE7;border-color:#F9A825"></span>σ-field</span>
  <span class="leg"><span class="leg-dot" style="background:#F0F0F0;border-color:#607D8B"></span>API call</span>
  <span class="leg"><span class="leg-line" style="background:#2980B9"></span>Formula derivation</span>
  <span class="leg"><span class="leg-line" style="background:#E67E22;background:repeating-linear-gradient(90deg,#E67E22 0,#E67E22 5px,transparent 5px,transparent 8px)"></span>SSBM</span>
  <span class="leg"><span class="leg-line" style="background:#C0392B"></span>QCD origin</span>
  <span class="leg"><span class="leg-line" style="background:#F57F17;background:repeating-linear-gradient(90deg,#F57F17 0,#F57F17 4px,transparent 4px,transparent 6px)"></span>σ-coupling</span>
  <span class="leg"><span class="leg-line" style="background:#90A4AE;background:repeating-linear-gradient(90deg,#90A4AE 0,#90A4AE 3px,transparent 3px,transparent 6px)"></span>API uses</span>
</div>
<div id="scroll"><svg id="chart"></svg></div>
<div id="tooltip"></div>

<script>
const RAW = __GRAPH_DATA__;

const NODE_STYLE = {
  measured: {fill:"#D6EAF8", stroke:"#2980B9", text:"#154360"},
  derived:  {fill:"#D5F5E3", stroke:"#27AE60", text:"#145a32"},
  ssbm:     {fill:"#FDEBD0", stroke:"#E67E22", text:"#784212"},
  higgs:    {fill:"#E8DAEF", stroke:"#8E44AD", text:"#4a235a"},
  qcd:      {fill:"#FADBD8", stroke:"#C0392B", text:"#641e16"},
  field:    {fill:"#FFFDE7", stroke:"#F9A825", text:"#7d6608"},
  api:      {fill:"#F0F0F0", stroke:"#607D8B", text:"#263238"},
};

const EDGE_STYLE = {
  formula: {color:"#2980B9", w:1.4, dash:"none"},
  ssbm:    {color:"#E67E22", w:1.4, dash:"6,3"},
  qcd:     {color:"#C0392B", w:1.8, dash:"none"},
  sigma:   {color:"#F57F17", w:1.8, dash:"4,2"},
  uses:    {color:"#78909C", w:1.0, dash:"3,3"},
};

const BAND_COLORS = [
  "#1a3a5c","#1a2a4c","#1a2a3c","#162540","#162030",
  "#1a1a30","#161a28","#161620","#2a1a30","#2a1828","#1a1828",
];

const W = 2200, ROW_H = 148, TOP_PAD = 52, SIDE_PAD = 60;
const NW = 118, NH = 50;

function layout(nodes) {
  const byLayer = {};
  nodes.forEach(n => { (byLayer[n.layer] = byLayer[n.layer]||[]).push(n); });
  nodes.forEach(n => {
    const peers = byLayer[n.layer];
    const idx   = peers.indexOf(n);
    const count = peers.length;
    const usable = W - 2 * SIDE_PAD;
    n._x = count === 1 ? W/2 : SIDE_PAD + idx * usable / (count - 1);
    n._y = TOP_PAD + n.layer * ROW_H;
    n._w = NW; n._h = NH;
  });
}

function svgEl(tag, attrs, parent) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k,v] of Object.entries(attrs)) el.setAttribute(k, v);
  if (parent) parent.appendChild(el);
  return el;
}

function bezier(x1,y1,x2,y2) {
  const cy = (y1+y2)/2;
  return `M${x1},${y1} C${x1},${cy} ${x2},${cy} ${x2},${y2}`;
}

let pinned = null;
const tooltip = document.getElementById("tooltip");

function showTip(evt, html) {
  tooltip.innerHTML = html;
  tooltip.style.display = "block";
  moveTip(evt);
}
function moveTip(evt) {
  const x = evt.clientX + 14, y = evt.clientY - 10;
  tooltip.style.left = Math.min(x, window.innerWidth - 290) + "px";
  tooltip.style.top  = (y < 10 ? 10 : y) + "px";
}
function hideTip() { tooltip.style.display = "none"; }

function render() {
  const nodes = RAW.nodes;
  const edges = RAW.edges;
  const layers = RAW.layers;
  const nodeMap = Object.fromEntries(nodes.map(n => [n.id, n]));

  layout(nodes);

  const totalLayers = Math.max(...nodes.map(n=>n.layer)) + 1;
  const svgH = TOP_PAD + totalLayers * ROW_H + 40;
  const svg = document.getElementById("chart");
  svg.setAttribute("width",  W);
  svg.setAttribute("height", svgH);
  svg.setAttribute("viewBox", `0 0 ${W} ${svgH}`);

  // ── Arrowhead markers ──────────────────────────────────────────────
  const defs = svgEl("defs", {}, svg);
  const markerTypes = Object.entries(EDGE_STYLE);
  markerTypes.forEach(([type, st]) => {
    const m = svgEl("marker", {
      id:`arr-${type}`, markerWidth:8, markerHeight:6,
      refX:7, refY:3, orient:"auto"
    }, defs);
    svgEl("polygon", {
      points:"0 0, 8 3, 0 6", fill:st.color, opacity:0.85
    }, m);
  });

  // ── Layer bands ────────────────────────────────────────────────────
  const bandG = svgEl("g", {}, svg);
  for (let i = 0; i < totalLayers; i++) {
    const y = i * ROW_H;
    svgEl("rect", {
      x:0, y:TOP_PAD + y - ROW_H/2 + NH/2,
      width:W, height:ROW_H,
      fill: BAND_COLORS[i] || "#111",
      class:"band"
    }, bandG);
    const label = layers[i] || `Layer ${i}`;
    svgEl("text", {
      x:6, y: TOP_PAD + y - ROW_H/2 + NH/2 + 13,
      class:"band-label"
    }, bandG).textContent = label.toUpperCase();
  }

  // ── Edges ──────────────────────────────────────────────────────────
  const edgeG = svgEl("g", {}, svg);
  const edgeEls = [];

  edges.forEach((e,i) => {
    const src = nodeMap[e.src], tgt = nodeMap[e.tgt];
    if (!src || !tgt) return;
    const st = EDGE_STYLE[e.type] || EDGE_STYLE.uses;

    const x1 = src._x, y1 = src._y + NH/2 + 1;
    const x2 = tgt._x, y2 = tgt._y - NH/2 - 1;

    const pathD = bezier(x1,y1,x2,y2);
    const path = svgEl("path", {
      d:pathD, stroke:st.color,
      "stroke-width": st.w,
      "stroke-dasharray": st.dash === "none" ? "" : st.dash,
      "marker-end": `url(#arr-${e.type})`,
      opacity: 0.65,
      class:"edge"
    }, edgeG);

    path.dataset.src  = e.src;
    path.dataset.tgt  = e.tgt;
    path.dataset.etype = e.type;
    path.dataset.label = e.label;
    path.dataset.idx   = i;
    edgeEls.push(path);

    // Fat invisible hit area
    const hit = svgEl("path", {
      d:pathD, stroke:"transparent", "stroke-width":10, fill:"none", cursor:"pointer"
    }, edgeG);
    hit.addEventListener("mousemove", evt => {
      if (pinned) return;
      const lbl = e.label ? `<strong>${e.label}</strong>${e.src} → ${e.tgt}` : `${e.src} → ${e.tgt}`;
      showTip(evt, lbl);
    });
    hit.addEventListener("mouseleave", () => { if (!pinned) hideTip(); });
  });

  // ── Nodes ──────────────────────────────────────────────────────────
  const nodeG = svgEl("g", {}, svg);
  const nodeEls = {};

  nodes.forEach(n => {
    const st = NODE_STYLE[n.type] || NODE_STYLE.api;
    const g  = svgEl("g", {class:"node", transform:`translate(${n._x - NW/2},${n._y - NH/2})`}, nodeG);
    g.dataset.id = n.id;

    svgEl("rect", {
      width:NW, height:NH, rx:7, ry:7,
      fill:st.fill, stroke:st.stroke, "stroke-width":1.5
    }, g);

    // Multi-line label
    const lines = n.label.split("\n");
    if (lines.length === 1) {
      const t = svgEl("text", {x:NW/2, y:NH/2, "text-anchor":"middle", fill:st.text}, g);
      t.textContent = n.label;
    } else {
      const t = svgEl("text", {x:NW/2, y:NH/2 - 7, "text-anchor":"middle", fill:st.text}, g);
      t.textContent = lines[0];
      const t2 = svgEl("text", {x:NW/2, y:NH/2 + 8, "text-anchor":"middle", fill:st.text, "font-size":"9.5px"}, g);
      t2.textContent = lines[1] || "";
    }

    nodeEls[n.id] = g;

    // Hover / click
    g.addEventListener("mousemove", evt => {
      if (pinned && pinned !== n.id) return;
      highlightNode(n.id, nodes, edges, edgeEls, nodeEls);
      const tipText = `<strong>${n.label.replace(/\n/g," ")}</strong>${n.tip}`;
      showTip(evt, tipText);
      evt.stopPropagation();
    });
    g.addEventListener("mouseleave", () => {
      if (!pinned) { clearHighlight(nodes, edgeEls, nodeEls); hideTip(); }
    });
    g.addEventListener("click", evt => {
      if (pinned === n.id) {
        pinned = null;
        clearHighlight(nodes, edgeEls, nodeEls);
        hideTip();
      } else {
        pinned = n.id;
        highlightNode(n.id, nodes, edges, edgeEls, nodeEls);
      }
      evt.stopPropagation();
    });
  });

  // Click elsewhere to unpin
  svg.addEventListener("click", () => {
    if (pinned) { pinned = null; clearHighlight(nodes, edgeEls, nodeEls); hideTip(); }
  });
}

function highlightNode(id, nodes, edges, edgeEls, nodeEls) {
  // Find all connected node IDs (immediate parents + children)
  const connected = new Set([id]);
  edges.forEach(e => {
    if (e.src === id) connected.add(e.tgt);
    if (e.tgt === id) connected.add(e.src);
  });

  // Dim/highlight nodes
  nodes.forEach(n => {
    const el = nodeEls[n.id];
    if (!el) return;
    el.classList.toggle("dim", !connected.has(n.id));
    el.classList.toggle("hi",  n.id === id);
  });

  // Dim/highlight edges
  edgeEls.forEach(path => {
    const s = path.dataset.src, t = path.dataset.tgt;
    const relevant = s === id || t === id;
    path.classList.toggle("dim", !relevant);
    path.classList.toggle("hi",   relevant);
    path.style.opacity = relevant ? "1" : "";
  });
}

function clearHighlight(nodes, edgeEls, nodeEls) {
  nodes.forEach(n => {
    const el = nodeEls[n.id];
    if (el) { el.classList.remove("dim","hi"); }
  });
  edgeEls.forEach(p => {
    p.classList.remove("dim","hi");
    p.style.opacity = "";
  });
}

render();
document.addEventListener("mousemove", evt => {
  if (tooltip.style.display === "block") moveTip(evt);
});
</script>
</body>
</html>
"""


def main():
    out = Path(__file__).parent.parent / "docs" / "dependency_chart.html"
    out.parent.mkdir(exist_ok=True)
    html = build_html(NODES, EDGES, LAYER_NAMES)
    out.write_text(html, encoding="utf-8")
    n_nodes = len(NODES)
    n_edges = len(EDGES)
    print(f"Written: {out}")
    print(f"  {n_nodes} nodes across {len(LAYER_NAMES)} layers")
    print(f"  {n_edges} edges")
    print(f"  File size: {len(html)//1024} KB")
    print(f"\nOpen in browser: file:///{out.resolve()}")


if __name__ == "__main__":
    main()
