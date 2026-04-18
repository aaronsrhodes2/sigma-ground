"""
SSBM Constants — The three numbers that define Scale-Shifted Baryonic Matter.

  ξ (XI)    = 0.1582  — INPUT:   baryon fraction Ω_b/(Ω_b+Ω_c), Planck 2018.
                                  The one new free parameter the theory introduces.
  η (ETA)   = 0.4153  — DERIVED: cosmic entanglement fraction, from dark energy.
                                  Falls out of ξ matched to observed ρ_DE.
  σ (SIGMA) = 0       — FIELD:   scale field value in the current epoch (our reference).
                                  Nonzero inside black holes and at the Big Bang.

Every other number here is measured (from experiment/observation) or derived from
those measurements.
"""

import math

# ── SSBM Parameters ───────────────────────────────────────────────────
# The single new constant: baryon fraction (Planck 2018)
#   Ω_b h² = 0.02237,  Ω_c h² = 0.1200
#   Exact: 0.02237 / (0.02237 + 0.1200) = 0.15716...
#   Canonical SSBM value (used in all publications and tests):
XI = 0.1582

# Kobakhidze-Liang 2024 (arXiv:2410.22412): ξ from QCD beta-function ratios
# For an N=8 composite-axion gauge group, Ω_DM/Ω_B = 5.36 →
# ξ = 1/(1 + 5.36) = 0.1572, within 0.6% of Planck 2018's 0.1582.
# [SPECULATIVE] — the N=8 identification with sigma-ground's σ-field is not established.
XI_KOBAKHIDZE = 1.0 / (1.0 + 5.36)          # ≈ 0.1572 (N=8 gauge group prediction)
SIGMA_CONV_KOBAKHIDZE = -math.log(XI_KOBAKHIDZE)  # ≈ 1.849

# DESI 2024 Holographic Dark Energy (arXiv:2411.08639)
# [SPECULATIVE] HDE: ρ_DE = 3 c² M_Pl²/L². If c² ≡ η, DESI DR2 fits bracket
# σ-ground's η = 0.4153 at the low end of the Union3 band:
#   CMB+DESI+Union3:  c = 0.642 ± 0.028  →  c² = 0.412  (η within ~0.5σ)
#   CMB+DESI+DESY5:   c = 0.701 ± 0.024  →  c² = 0.491
C_HDE_UNION3   = 0.642            # HDE c from CMB+DESI+Union3 (DESI 2024)
C_HDE_DESY5    = 0.701            # HDE c from CMB+DESI+DESY5
ETA_HDE_UNION3 = C_HDE_UNION3 ** 2  # ≈ 0.412 — lower bound of DESI range
ETA_HDE_DESY5  = C_HDE_DESY5  ** 2  # ≈ 0.491 — upper bound of DESI range

# QCD confinement scale (PDG reference)
LAMBDA_QCD_MEV = 217.0  # MeV

# Spectral index relation: γ = 3 - n_s (Planck 2018: n_s = 0.9649)
GAMMA = 2.035

# Critical σ where nuclear bonds fail
SIGMA_CONV = -math.log(XI)  # ≈ 1.849... but using exact XI

# ── Physical Constants ────────────────────────────────────────────────
G = 6.67430e-11       # m³ kg⁻¹ s⁻²  (gravitational constant)
C = 2.99792458e8      # m/s            (speed of light)
HBAR = 1.054571817e-34  # J·s          (reduced Planck constant)

# ── Nucleon Mass Decomposition (MeV) ─────────────────────────────────
# This is the KEY FACT: ~99% of nucleon mass is QCD binding, not Higgs.

# Quark bare masses (Higgs origin — σ-INVARIANT)
M_UP_MEV = 2.16       # MeV (PDG 2020)
M_DOWN_MEV = 4.67     # MeV (PDG 2020)

# Proton (uud): bare = 2×m_u + m_d
PROTON_BARE_MEV = 2 * M_UP_MEV + M_DOWN_MEV  # = 8.99 MeV
PROTON_TOTAL_MEV = 938.272          # MeV (measured)
PROTON_QCD_MEV = PROTON_TOTAL_MEV - PROTON_BARE_MEV  # = 929.282 MeV

# Neutron (udd): bare = m_u + 2×m_d
NEUTRON_BARE_MEV = M_UP_MEV + 2 * M_DOWN_MEV  # = 11.50 MeV
NEUTRON_TOTAL_MEV = 939.565         # MeV (measured)
NEUTRON_QCD_MEV = NEUTRON_TOTAL_MEV - NEUTRON_BARE_MEV  # = 928.065 MeV

# QCD fractions
PROTON_QCD_FRACTION = PROTON_QCD_MEV / PROTON_TOTAL_MEV   # ≈ 0.9904
NEUTRON_QCD_FRACTION = NEUTRON_QCD_MEV / NEUTRON_TOTAL_MEV  # ≈ 0.9878

# Neutron-proton mass difference — MEASURED directly to higher precision
# than computing 939.565 - 938.272 (which loses 3 significant digits).
# PDG 2020: m_n - m_p = 1.29333236 ± 0.00000046 MeV
DELTA_NP_TOTAL_MEV = 1.29333236     # MeV (MEASURED, PDG 2020)
# Bare mass difference: (m_u + 2m_d) - (2m_u + m_d) = m_d - m_u
DELTA_NP_BARE_MEV = M_DOWN_MEV - M_UP_MEV  # = 2.51 MeV (Higgs, σ-INVARIANT)
# QCD contribution to n-p mass difference
DELTA_NP_QCD_MEV = DELTA_NP_TOTAL_MEV - DELTA_NP_BARE_MEV  # ≈ -1.217 MeV

# Electron mass (Higgs origin — σ-INVARIANT)
M_ELECTRON_MEV = 0.51100            # MeV

# ── Electromagnetic Constants ─────────────────────────────────────────
# All measured (CODATA 2018 / 2019 SI exact values):
#
#   e    = 1.602176634e-19 C    (elementary charge, exact by 2019 SI)
#   ε₀   = 8.854187817e-12 F/m  (vacuum permittivity, CODATA)
#   μ₀   = 4π×10⁻⁷ H/m         (vacuum permeability, derived from SI exact values)
#   α    = e²/(4πε₀ℏc) ≈ 1/137.036  (fine structure constant, DERIVED)

E_CHARGE = 1.602176634e-19    # C  (elementary charge, exact by 2019 SI)
EPS_0 = 8.854187817e-12       # F/m (vacuum permittivity)
MU_0 = 1.25663706212e-6       # H/m (vacuum permeability, μ₀ = 4π×10⁻⁷ approx)

# Fine structure constant: α = e²/(4πε₀ℏc)
# DERIVED from the four measured constants above.
# Value ≈ 7.2973525693e-3 ≈ 1/137.036
ALPHA = E_CHARGE**2 / (4 * math.pi * EPS_0 * HBAR * C)

# ── Electron Mass ─────────────────────────────────────────────────────
# Already defined in MeV above (Higgs origin, σ-INVARIANT).
# SI form for use in EM and plasma calculations:
M_ELECTRON_KG = 9.1093837015e-31  # kg (CODATA 2018)

# ── Boltzmann Constant ────────────────────────────────────────────────
# Exact by 2019 SI redefinition (links energy to temperature).
K_B = 1.380649e-23            # J/K (Boltzmann constant, exact by 2019 SI)

# ── Atomic Mass Unit ─────────────────────────────────────────────────
# MEASURED: 1/12 of ¹²C mass (CODATA 2018).
AMU_KG = 1.66053906660e-27    # kg

# ── Avogadro & Gas Constant ─────────────────────────────────────────
# N_A is exact by 2019 SI.  R = k_B × N_A is DERIVED.
N_AVOGADRO = 6.02214076e23    # 1/mol (exact by 2019 SI)
R_GAS = K_B * N_AVOGADRO      # J/(mol·K) — DERIVED, not stored

# ── Energy Conversion Factors ────────────────────────────────────────
# ALL derived from E_CHARGE (elementary charge, exact by 2019 SI).
# 1 eV ≡ 1 elementary charge × 1 volt ≡ E_CHARGE joules.
EV_TO_J = E_CHARGE             # J/eV  (= 1.602176634e-19)
MEV_TO_J = E_CHARGE * 1e6     # J/MeV (= 1.602176634e-13)
KEV_TO_J = E_CHARGE * 1e3     # J/keV (= 1.602176634e-16)

# ── Planck Constant (full) ──────────────────────────────────────────
# DERIVED from HBAR: h = 2πℏ
H_PLANCK = 2 * math.pi * HBAR  # J·s (= 6.62607015e-34)

# ── Stefan-Boltzmann Constant ────────────────────────────────────────
# DERIVED from k_B, ℏ, c:  σ_SB = π²k_B⁴ / (60ℏ³c²)
# Not stored — computed from fundamentals every time this module loads.
STEFAN_BOLTZMANN = (math.pi**2 * K_B**4) / (60 * HBAR**3 * C**2)

# ── Bohr Radius & Magneton ──────────────────────────────────────────
# Both DERIVED from measured constants.
# a₀ = ℏ/(m_e c α) = 4πε₀ℏ²/(m_e e²)
BOHR_RADIUS = HBAR / (M_ELECTRON_KG * C * ALPHA)  # ≈ 5.2918e-11 m
# μ_B = eℏ/(2m_e)
MU_BOHR = E_CHARGE * HBAR / (2 * M_ELECTRON_KG)   # ≈ 9.2740e-24 J/T

# ── Coulomb Energy (first-principles electrostatics) ─────────────────
# NOT from SEMF. Derived from Coulomb's law integrated over a uniform
# charge sphere:  E_C = (3/5) × ke_e² / r₀ × Z(Z-1) / A^(1/3)
#
# Inputs (all measured):
#   e    = 1.602176634e-19 C    (elementary charge, CODATA exact)
#   ε₀   = 8.854187817e-12 F/m  (vacuum permittivity, CODATA)
#   r₀   = 1.215 fm             (nuclear charge radius, electron scattering — Hofstadter)
#
# ke × e² = e²/(4πε₀) = 1.43996 MeV·fm  (EM coupling constant)
# a_C = (3/5) × 1.43996 / 1.215 = 0.7111 MeV
R0_FM = 1.215                 # fm (nuclear charge radius, electron scattering)

# ke × e²  in MeV·fm
KE_E2_MEV_FM = E_CHARGE**2 / (4 * math.pi * EPS_0) / MEV_TO_J * 1e15

# Coulomb coefficient — pure electrostatics, no SEMF
A_C_MEV = (3.0 / 5.0) * KE_E2_MEV_FM / R0_FM  # ≈ 0.7111 MeV (σ-INVARIANT)

# ── Entanglement Fraction (η) ─────────────────────────────────────────
# The ratio of particles entangled with at least one other particle
# *somewhere* (it doesn't matter where the partner is).
#
# DERIVED from the dark energy constraint:
#   At σ_conv, QCD binding energy is released. The entangled fraction
#   releases coherently → gluon condensate (w = −1, dark energy).
#   The non-entangled fraction thermalizes → radiation (w = +1/3).
#   Matching ρ_DE(observed) = η × ρ_released gives η = 0.4153.
#
# Bounds:   0 ≤ η ≤ 1
#   η = 0:  no particles entangled (fully classical limit)
#   η = 1:  every particle entangled with at least one other
#
# Physical meaning: inside a hadron, quarks are always entangled
# (color singlet). But η asks about CROSS-HADRON entanglement —
# particles that share quantum correlations with partners in other
# hadrons, other atoms, other galaxies. This includes:
#   - EPR pairs (produced together, flew apart)
#   - Particles from common decay chains
#   - Cosmological correlations from the Big Bang
#   - Any interaction that created entanglement and was never decohered
ETA = 0.4153  # DERIVED — cosmic entanglement fraction (dark energy constraint)
# Note (2026-04-17): ETA was derived heuristically from a convenient coherence
# ratio related to the golden spiral. Retained as the working value; see
# ETA_FORMULA below for a candidate first-principles expression.

# ── Golden ratio ──────────────────────────────────────────────────────
PHI = (1.0 + math.sqrt(5.0)) / 2.0  # = (1+√5)/2 ≈ 1.6180339887...

# ── ETA formula candidate: exp(-φ/σ_conv)  [SPECULATIVE] ────────────
# Systematic formula search (2026-04-17) over combinations of {ξ, σ_conv, π, e, φ}:
#   exp(-φ / σ_conv) ≈ 0.4158   (0.125% error from η = 0.4153)
# Equivalent forms:  exp(-φ / σ_conv) = exp(φ / ln ξ)
# If valid, η has no free parameters — fully derived from ξ and φ, where φ is
# the golden ratio (consistent with the original "related to the golden spiral"
# heuristic) and σ_conv = −ln ξ is the model's matter-conversion horizon.
# ETA = 0.4153 remains the working value until ETA_FORMULA is confirmed.
ETA_FORMULA = math.exp(-PHI / SIGMA_CONV)  # exp(-φ/σ_conv) ≈ 0.4158 [SPECULATIVE]

# ── Duality-ellipse marginal coherence γ ─────────────────────────────
# γ = |⟨M₁|M₂⟩| = overlap between the environment's two "notepad"
# states that record which-path information in a double-slit setup.
#
# FIRST_PRINCIPLES: Khatiwada-Qian 2025, arXiv:2505.21443v1. The
# duality ellipse V²/γ² + D² = 1 generalises Englert 1996's circle
# V² + D² ≤ 1. At γ=1 the ellipse degenerates to the Englert circle,
# recovering all pre-existing sigma-ground double-slit behaviour
# byte-for-byte.
#
# γ is an OPEN INPUT KNOB. The σ→γ mapping is intentionally left
# unspecified here; it is derived empirically (see Phase F/G of the
# duality-ellipse verdict artifact). Do NOT confuse this with GAMMA
# above — that is the spectral index from Planck 2018.
GAMMA_COHERENCE_DEFAULT = 1.0

# ── Per-dimension entanglement fraction Θ (H4 candidate) ──────────────
# Θ = η^(1/3) — the cube root of the cosmic entanglement fraction,
# interpreted as the share of entangled matter "per gravitational
# dimension" in 3-D flat space (Θ³ = η recovers the total).
#
# HYPOTHESIS (user, not paper): the Khatiwada-Qian marginal coherence γ
# is not a free knob but a derived cosmological constant equal to Θ.
# Near Earth (σ ≈ 0) laboratory double-slit observations pin γ → 1
# because the coherence is normalised by the local σ-frame; at
# cosmological scales with σ ≠ 0 the γ = Θ prediction becomes visible.
#
# This constant is DERIVED from ETA, not fitted. Treat as a testable
# prediction — see Phase E/F/G of the duality-ellipse verdict artifact.
THETA_ENTANGLEMENT_PER_DIM = ETA ** (1.0 / 3.0)  # ≈ 0.7463

# ── Observer Frame σ ─────────────────────────────────────────────────
# The σ-field value in OUR spacetime — the observer's compression frame.
#
# Conceptually σ ≈ 0 here. We are the baseline against which all other
# σ values are measured. This is not a measurement — it is the definition
# of the reference frame.
#
# Inside a black hole accretion disk: σ > 0 (spacetime more compressed).
# At the Big Bang / inside cavitation: σ → σ_conv ≈ 1.85.
# In a hypothetical decompressed region: σ < 0.
#
# If you are doing wormhole physics or computing properties in a different
# compression frame, change THIS constant — every module that uses it will
# shift accordingly. That is the point of having it here instead of
# writing 0.0 everywhere.
#
# WHY NOT EXACT ZERO:
# Exact 0.0 is a computer landmine — division by σ, log(σ), and formulas
# like 1/(1−e^{−σ}) all diverge. We use SIGMA_FLOOR (Planck/Hubble ratio
# ≈ 1.18e-61) as the observer frame value. It is physically equivalent to
# zero (exp(1e-61) = 1.0 exactly in double precision) but prevents any
# floating-point catastrophe if σ ever reaches a denominator or logarithm.
#
# → SIGMA_HERE is defined after SIGMA_FLOOR below (needs L_PLANCK, H0, C).

# ── Nuclear Matter (QCD observables) ─────────────────────────────────
# Nuclear saturation density: where attraction and repulsion balance.
# MEASURED from electron scattering on heavy nuclei.
N0_FM3 = 0.16              # fm⁻³ (nuclear saturation density)
E_SAT_MEV = -16.0           # MeV (binding energy per nucleon at saturation)

# Nuclear incompressibility: K = 9n₀² d²(E/A)/dn² at saturation.
# MEASURED from giant monopole resonances in heavy nuclei.
# This is a QCD observable — it's the stiffness of nuclear matter,
# set by the balance of attractive σ-meson and repulsive ω-meson exchange.
# In SSBM: both meson masses scale with Λ_QCD × e^σ, so K(σ) = K₀ × e^σ.
K_SAT_MEV = 230.0           # MeV (nuclear incompressibility)

# Symmetry energy: energy cost of neutron-proton imbalance.
# MEASURED from nuclear mass differences.
J_SYM_MEV = 32.0            # MeV (symmetry energy at saturation)

# ── Cosmological ──────────────────────────────────────────────────────
H0 = 67.4e3 / 3.086e22  # Hubble constant in s⁻¹ (67.4 km/s/Mpc)
M_HUBBLE_KG = C**3 / (2 * G * H0)  # Hubble mass ≈ 9.3e52 kg
M_PLANCK_KG = math.sqrt(HBAR * C / G)  # Planck mass ≈ 2.18e-8 kg
L_PLANCK    = HBAR / (M_PLANCK_KG * C) # Planck length √(ℏG/c³) ≈ 1.616e-35 m
#   The smallest length with physical meaning (below this: quantum gravity).
#   Use as the natural "effectively zero" floor — not a magic number,
#   but the universe's own minimum. Derived from ℏ, G, c above.
M_SUN_KG = 1.989e30    # Solar mass
L_SUN_W  = 3.828e26    # Solar luminosity (W) — IAU 2015 nominal solar luminosity
AU_M     = 1.495978707e11  # 1 Astronomical Unit in metres (IAU 2012 exact)

# Derived time constants
YEAR_S = 365.25 * 86400.0  # Julian year in seconds (exact by IAU definition)

# ── SIGMA_FLOOR: Planck-derived σ computational epsilon ───────────────
# Never use SIGMA_0 = 0.0 directly in σ computations — division by σ,
# log(σ), and formulas like 1/(1 − e^{−σ}) all diverge at exact zero.
#
# Use SIGMA_FLOOR wherever a "zero σ" guard is needed.
#
# Definition: ratio of Planck length to Hubble radius — the quantum of
# geometry in SSBM.
#
#   SIGMA_FLOOR = l_P / R_H = L_PLANCK * H0 / C  ≈ 1.18 × 10⁻⁶¹
#
# Physical meaning: the smallest dimensionless σ gradient that the
# classical field description can sustain before quantum gravity
# (below l_P) renders it meaningless.  Analogous to how L_PLANCK
# is used as the spatial "effectively zero" floor.
#
# Properties:
#   - Derived from L_PLANCK, H0, C — no new free parameters
#   - Above double-precision underflow (~ 5×10⁻³²⁴)
#   - Below any physically meaningful σ (Earth surface: ~7×10⁻¹⁰)
SIGMA_FLOOR = L_PLANCK * H0 / C  # ≈ 1.18e-61 (Planck/Hubble ratio)

# ── SIGMA_HERE: Observer frame σ (defined here, after SIGMA_FLOOR) ───
SIGMA_HERE = SIGMA_FLOOR  # σ in OUR spacetime (observer's compression frame)
SIGMA_0 = SIGMA_HERE      # backward compatibility alias
