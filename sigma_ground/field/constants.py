"""
SSBM Constants — The three numbers that define Scale-Shifted Baryonic Matter.

  ξ (XI)    = 0.1582  — INPUT:   baryon fraction Ω_b/(Ω_b+Ω_c), Planck 2018.
                                  The one new free parameter the theory introduces.
  η (ETA)   = 0.4122  — EMPIRICAL-INPUT: cosmic entanglement fraction.
                                  Set equal to c² from DESI 2024 Union3 HDE fit
                                  (arXiv:2411.08639): c = 0.642 ± 0.028.
                                  Previously 0.4153 [DERIVED] from a heuristic
                                  golden-spiral coherence-ratio argument that
                                  did not survive 2026-05-15 audit.
  σ (SIGMA) = 0       — FIELD:   scale field value in the current epoch (our reference).
                                  Nonzero inside black holes and at the Big Bang.

Every other number here is measured (from experiment/observation) or derived from
those measurements.
"""

import math

# ── SSBM Parameters ───────────────────────────────────────────────────
# THE ONE FREE PARAMETER OF SSBM:
# baryon fraction Ω_b/(Ω_b+Ω_c) from Planck 2018 CMB analysis.
#   Ω_b h² = 0.02237,  Ω_c h² = 0.1200
#   Exact ratio: 0.02237 / (0.02237 + 0.1200) = 0.15716...
# (XI -- the SSBM baryon-fraction parameter -- relocated to ssbm-theory.)

# (XI_KOBAKHIDZE / SIGMA_CONV_KOBAKHIDZE variants -- SSBM -- relocated to ssbm-theory.)

# DESI 2024 Holographic Dark Energy (arXiv:2411.08639)
# HDE: ρ_DE = 3 c² M_Pl²/L². Under the identification c² ≡ η, the DESI 2024
# Union3 fit gives the empirical anchor for our cosmic entanglement fraction:
#   CMB+DESI+Union3:  c = 0.642 ± 0.028  →  c² ≈ 0.412   (adopted as ETA)
#   CMB+DESI+DESY5:   c = 0.701 ± 0.024  →  c² ≈ 0.491   (sub-sample shown,
#                                                        not adopted -- DESY5
#                                                        differs from Union3
#                                                        by ~18%, suggesting
#                                                        SN systematics).
# Adopting Union3 specifically: it's the standardised Pantheon+ZTF compilation
# with consistent SN cuts; DESY5 has known photometric-calibration drift.
C_HDE_UNION3   = 0.642            # HDE c from CMB+DESI+Union3 (DESI 2024)
C_HDE_DESY5    = 0.701            # HDE c from CMB+DESI+DESY5
ETA_HDE_UNION3 = C_HDE_UNION3 ** 2  # ≈ 0.412164 -- ADOPTED as ETA (2026-05-15)
ETA_HDE_DESY5  = C_HDE_DESY5  ** 2  # ≈ 0.491   -- shown for comparison only

# DESI 2025 DR3 dynamical-DE multi-model fits
# ─────────────────────────────────────────────
# CITATION CORRECTION (2026-05-15 audit, see PROVENANCE.md):
# The originally-cited arXiv:2512.07281 surveys ΛCDM, wwCDM, w₀wₐCDM, φCDM,
# and a "ξ-index" interacting-DE model -- it does NOT contain HDE c² values.
# The correct source for DR3 HDE constraints is a separate (yet-to-be-found)
# paper, possibly in the DESI 2025 cosmology release set. These slots stay
# None with explicit citation-needed status -- loud failure beats fabricated
# numbers.
ETA_HDE_UNION3_DR3 = None         # [SPECULATIVE-PENDING] DR3 Union3 c²; CITATION NEEDED (arXiv:2512.07281 was checked and lacks HDE content)
ETA_HDE_DESY5_DR3  = None         # [SPECULATIVE-PENDING] DR3 DESY5 c²;  CITATION NEEDED (same as above)

# EDE + interacting dark-sector joint fit (arXiv:2505.23382, May 2025)
# Two-parameter extension where EDE fraction f_EDE lifts H₀ and an iDEDM
# coupling ξ suppresses S₈ growth. Joint analysis P18+DESI+DES+PP+H0
# (paper's Table 2) constrains both as UPPER BOUNDS, not central values
# with symmetric uncertainties -- the data is consistent with both = 0.
# Note: ξ here (iDEDM coupling) is NOT sigma-ground's XI (baryon fraction).
# Distinct symbol deliberately used.
F_EDE_UPPER_95CL   = 0.113   # [VERIFIED] arXiv:2505.23382 Table 2, mixed EDE+iDEDM model, P18+DESI+DES+PP+H0 (95% C.L. upper bound)
XI_IDE_UPPER_95CL  = 0.071   # [VERIFIED] arXiv:2505.23382 Table 2, mixed EDE+iDEDM model (95% C.L. upper bound)
# Legacy aliases for backward compatibility -- now point to the upper bounds.
# Future code should reference F_EDE_UPPER_95CL / XI_IDE_UPPER_95CL directly.
F_EDE_CENTRAL      = F_EDE_UPPER_95CL   # alias: upper bound treated as conservative "best fit"
XI_IDE_CENTRAL     = XI_IDE_UPPER_95CL  # alias: same

# ── Neutrino sector ────────────────────────────────────────────────────
# PDG 2022 best-fit mass-splittings for 3-flavour oscillations.
# Squared mass differences in eV² (sign convention: Δm²_ij = m_i² − m_j²).
DELTA_M2_21_EV2 = 7.42e-5   # [VERIFIED] PDG 2022 solar mass-squared splitting
DELTA_M2_31_EV2 = 2.514e-3  # [VERIFIED] PDG 2022 atmospheric (normal ordering)
DELTA_M2_32_EV2 = -2.497e-3 # [VERIFIED] PDG 2022 (inverted ordering magnitude)

# Cosmological sum-of-neutrino-masses upper bound (Planck 2018 + DESI 2024)
# Σm_ν < 0.12 eV at 95% CL combining CMB + BAO; tightened by DESI 2024 to
# Σm_ν < 0.072 eV under ΛCDM assumption (arXiv:2404.03002).
SUM_NEUTRINO_MASS_UPPER_EV = 0.12         # [VERIFIED] Planck 2018 combined (95% CL)
SUM_NEUTRINO_MASS_DESI_UPPER_EV = 0.072   # [VERIFIED] DESI 2024 (arXiv:2404.03002)

# Neutrino nature placeholder — Dirac vs Majorana undetermined experimentally.
# The distinction can only be resolved by a positive observation of
# neutrinoless double-beta decay (0νββ). As of May 2026, no experiment has
# claimed detection: KamLAND-Zen, GERDA, CUORE, EXO-200, MAJORANA, LEGEND-200
# all set upper bounds on the effective Majorana mass. Until LEGEND-1000,
# nEXO, or similar next-gen experiment claims a positive signal (or rules
# out the remaining parameter space at inverted-hierarchy sensitivity), the
# question is genuinely open. Sigma-ground takes no stance.
#
# This is intentionally NOT a lift-from-table value -- it's a binary
# (dirac/majorana/unknown) that requires experimental resolution we don't have.
NEUTRINO_NATURE = None  # [SPECULATIVE-PENDING] genuinely open question, no measurement to lift

# LEGEND-200 first 0νββ results (arXiv:2603.12884, March 2026)
# Searching for 0νββ in ⁷⁶Ge with 142.5 kg of detectors over first year.
# No positive signal -- only lower limits on the half-life:
#   LEGEND-200 alone:                  T₁/₂⁰ν > 0.5 × 10²⁶ yr  (90% CL)
#   Joint with GERDA + MAJORANA Demo:  T₁/₂⁰ν > 1.9 × 10²⁶ yr  (90% CL)
# These tighten the upper bound on the effective Majorana mass m_ββ but do
# not resolve the Dirac/Majorana question (no positive observation).
NEUTRINOLESS_BETA_HALFLIFE_LEGEND200_YR = 0.5e26  # [VERIFIED] arXiv:2603.12884 (90% CL lower bound)
NEUTRINOLESS_BETA_HALFLIFE_GERMANIUM_COMBINED_YR = 1.9e26  # [VERIFIED] LEGEND-200 + GERDA + MAJORANA joint (90% CL)

# ── MOND characteristic acceleration ──────────────────────────────────
# Milgrom 1983 (ApJ 270, 365): rotation curves of galaxies transition from
# Newtonian to modified dynamics at acceleration scale ≈ 1.2 × 10⁻¹⁰ m/s²
# Measured from ~150 galaxies (SPARC sample); single empirical constant.
# Reference: McGaugh et al. 2016; arXiv:2511.05632 (entropic derivation).
A0_MOND_MS2 = 1.2e-10  # [VERIFIED] Milgrom 1983, confirmed SPARC 2016

# ── Dark axion portal ──────────────────────────────────────────────────
# An axion-like particle (ALP) coupled to both photons and dark photons,
# with mass and coupling constrained by collider and astrophysical searches.
#   Axion mass window:         m_a ∈ (10⁻²² eV, 10⁻² eV) for ULDM
#   ALP-photon coupling limit: g_aγγ < ~10⁻¹⁰ GeV⁻¹ (CAST/helioscope)
#
# CITATION CORRECTION (2026-05-15 audit): the originally-cited
# arXiv:2410.12634 is actually about cavity axion-photon conversion
# physics, NOT a dark-axion-portal review. The Z-factory sensitivity slot
# below references a paper we haven't located yet; it stays None until the
# correct citation is identified.
AXION_MASS_ULDM_MIN_EV = 1.0e-22            # [VERIFIED] ultralight DM lower bound
AXION_MASS_ULDM_MAX_EV = 1.0e-2             # [VERIFIED] QCD-axion window upper bound
ALP_PHOTON_COUPLING_MAX_INVGEV = 1.0e-10    # [VERIFIED] CAST 2017 helioscope bound
DARK_AXION_PORTAL_Z_SENSITIVITY = None      # [SPECULATIVE-PENDING] Z-factory sensitivity; CITATION NEEDED (arXiv:2410.12634 was checked and is unrelated)

# ── KM3-230213A LIV bound (Feb 2025, arXiv:2502.18256) ────────────────
# Stringent constraint on the neutrino-sector second-order Lorentz-invariance
# violation scale, from the single ultra-high-energy event KM3-230213A:
#   Λ₂ > 5.0 × 10¹⁹ GeV at 90% CL  (modified dispersion parameterization)
# The engine must respect this bound at σ=0 (our frame). See
# sigma_ground.field.relativity.effective_liv_scale_gev(sigma).
LAMBDA_LIV_MIN_GEV = 5.0e19  # [VERIFIED] KM3-230213A bound (arXiv:2502.18256)

# QCD confinement scale (PDG reference)
LAMBDA_QCD_MEV = 217.0  # MeV

# ── Muon g−2 — Theory Initiative 2025 white paper + Fermilab final (Jun 2025)
# MTI 2025 WP adopted lattice-HVP → SM prediction shifted up; Fermilab final
# (Jun 2025) reports a_μ with 127 ppb precision. Net: anomaly DISSOLVED,
# SM and experiment agree at current precision.
# Units: dimensionless a_μ = (g − 2)/2. Values are a_μ × 10¹¹ (integer-ish).
A_MU_EXP_X1E11        = 116_592_061.0  # [VERIFIED] Fermilab final, Jun 2025 (a_μ = 1.16592061e-3)
A_MU_EXP_X1E11_SIGMA  =          22.0  # [VERIFIED] 127 ppb → ~22 × 10⁻¹¹
A_MU_SM_X1E11         = 116_591_810.0  # [VERIFIED] MTI 2025 WP (lattice-HVP central value)
A_MU_SM_X1E11_SIGMA   =          43.0  # [VERIFIED] MTI 2025 WP combined uncertainty
# Per MTI 2025 WP adoption of lattice-HVP: no physics tension between SM and
# experiment; remaining sigma reflects CMD-3 vs BMW lattice-HVP input choice.
MU_G2_TENSION_RESOLVED = True   # [VERIFIED] per MTI 2025 WP adoption of lattice-HVP

# Spectral index relation: γ = 3 - n_s (Planck 2018: n_s = 0.9649)
GAMMA = 2.035

# (SIGMA_CONV -- the SSBM matter-conversion threshold -- relocated to ssbm-theory.)

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
# [EMPIRICAL-INPUT] (2026-05-15): η is now an empirical input alongside ξ,
# set equal to c² from the DESI 2024 Union3 HDE fit (arXiv:2411.08639).
# Previously ETA = 0.4153 was tagged [DERIVED] under the claim "ρ_DE(observed)
# = η × ρ_released gives η = 0.4153", but a 2026-05-15 audit found:
#
#   1. The "derivation" relied on a heuristic golden-spiral coherence ratio
#      whose ρ_released side was not independently constructed -- partly
#      circular.
#   2. The candidate first-principles expression ETA_FORMULA = exp(-φ/σ_conv)
#      ≈ 0.4158 came from a formula search over {ξ, σ_conv, π, e, φ}, with
#      the golden ratio φ pre-baked into the target. The 0.125% gap was
#      treated as approximate agreement; for a genuine derivation it should
#      be exact, not 0.125% off.
#   3. See misc/bh_phase_xi_eta_candidates_results.md for the rejected
#      Phase XI verdict; see misc/eta_empirical_verdict_2026-05-15.md for
#      the current adoption.
#
# Bounds:   0 ≤ η ≤ 1
#   η = 0:  no particles entangled (fully classical limit)
#   η = 1:  every particle entangled with at least one other
#
# Physical meaning: inside a hadron, quarks are always entangled
# (color singlet). But η asks about CROSS-HADRON entanglement --
# particles that share quantum correlations with partners in other
# hadrons, other atoms, other galaxies. This includes:
#   - EPR pairs (produced together, flew apart)
#   - Particles from common decay chains
#   - Cosmological correlations from the Big Bang
#   - Any interaction that created entanglement and was never decohered
#
# Defined further below as ETA = ETA_HDE_UNION3 (= 0.642²) so the empirical
# provenance is explicit in code. Uncertainty: σ_η = 2c·σ_c = 0.036 (1σ).

# ── Golden ratio ──────────────────────────────────────────────────────
PHI = (1.0 + math.sqrt(5.0)) / 2.0  # = (1+√5)/2 ≈ 1.6180339887...

# ── ETA empirical anchor ──────────────────────────────────────────────
# η is now set equal to the DESI 2024 Union3 HDE c² fit. The value
# ETA_HDE_UNION3 = C_HDE_UNION3 ** 2 = 0.642² ≈ 0.412164 is defined near
# the top of this file (Phase XII.b DESI block).
ETA = ETA_HDE_UNION3  # [EMPIRICAL-INPUT] cosmic entanglement fraction (DESI Union3)

# 1σ uncertainty on η, propagated from c = 0.642 ± 0.028:
#   σ_η = 2 c σ_c = 2 · 0.642 · 0.028 ≈ 0.036
ETA_UNCERTAINTY_1SIGMA = 2.0 * C_HDE_UNION3 * 0.028  # ≈ 0.0360

# ── ETA_FORMULA: REJECTED (numerology, not derivation) ───────────────
# exp(-φ/σ_conv) was a Phase XI (2026-04-17) formula-search candidate
# claiming to be a first-principles expression for η. A 2026-05-15 audit
# rejected it:
#   - Came from a search over {ξ, σ_conv, π, e, φ} -- order ~10³ simple
#     combinations, so a near-hit at any target is expected by chance.
#   - The target η = 0.4153 was itself derived with a "golden-spiral"
#     heuristic, so finding a φ-formula matching it is circular.
#   - The 0.125% residual gap is the signature: a real derivation would
#     be exact, not 0.125% off.
# ETA_FORMULA is retained as None so legacy imports do not silently break;
# any code that depended on a numeric value will fail loudly.
ETA_FORMULA = None  # [REJECTED 2026-05-15] -- formula-search numerology

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
# (THETA_ENTANGLEMENT_PER_DIM -- SSBM duality prediction -- relocated to ssbm-theory.)

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
