"""
Stellar nucleosynthesis from Gamow tunneling and reaction Q-values.

Derivation chain:
  σ → nuclear mass → Coulomb barrier → Gamow peak → reaction rate
  σ → binding energy → Q-value → energy generation rate

This module implements the thermonuclear reaction rates for the three
principal stellar burning stages:

  1. pp-chain (Hydrogen → Helium)
     The dominant energy source in stars like the Sun.
     Rate-limiting step: p + p → d + e⁺ + ν_e (weak interaction)
     Followed by: d + p → ³He + γ, then ³He + ³He → ⁴He + 2p

  2. CNO cycle (Hydrogen → Helium, catalyzed by Carbon)
     Dominant in massive stars (T > 17 million K).
     ¹²C + p → ¹³N → ¹³C + e⁺ + ν, then ¹³C + p → ¹⁴N + γ, ...
     Net: 4p → ⁴He + 2e⁺ + 2ν + γ (carbon is catalyst)

  3. Triple-alpha (Helium → Carbon)
     ⁴He + ⁴He → ⁸Be* (unstable, ~10⁻¹⁶ s lifetime)
     ⁸Be* + ⁴He → ¹²C* → ¹²C + γ (Hoyle state resonance)

──────────────────────────────────────────────────────────────────────────────
REACTION RATE FORMALISM
──────────────────────────────────────────────────────────────────────────────

For charged-particle reactions, the rate per unit volume is:
  r₁₂ = (n₁ n₂ / (1 + δ₁₂)) × ⟨σv⟩

where ⟨σv⟩ is the thermally-averaged cross section × velocity.

The cross section for tunneling through a Coulomb barrier:
  σ(E) = S(E) / E × exp(−2πη)

where:
  S(E) = astrophysical S-factor (nuclear physics, MEASURED)
  η = Z₁Z₂e² / (4πε₀ℏv) = Sommerfeld parameter
  E = center-of-mass kinetic energy

Thermally-averaged rate (Gamow peak integration):
  ⟨σv⟩ = √(8/(πμ)) × (k_BT)^(−3/2) × ∫₀^∞ S(E) × exp(−E/k_BT − b/√E) dE

where b = π Z₁Z₂e² √(2μ) / (ε₀h) is the Gamow penetration parameter.

The integrand peaks at the Gamow energy:
  E_G = (b k_BT / 2)^(2/3)

The integral is well approximated by a Gaussian around E_G:
  ⟨σv⟩ ≈ Δ × √(2/(μ k_BT)) × S(E_G)/E_G × exp(−3E_G/k_BT)
  where Δ = 4√(E_G k_BT / 3) is the Gamow window width.

FIRST_PRINCIPLES: all of this follows from quantum tunneling (WKB)
+ Maxwell-Boltzmann thermal distribution. The S-factor is MEASURED.

──────────────────────────────────────────────────────────────────────────────
σ-DEPENDENCE
──────────────────────────────────────────────────────────────────────────────

The Gamow energy depends on the reduced mass μ of the reacting nuclei:
  E_G ∝ μ^(1/3)

Since μ(σ) = μ_bare + μ_QCD × e^σ, the Gamow peak shifts with σ.
Higher σ → heavier nuclei → higher Gamow peak → LOWER reaction rate
at the same temperature.

Equivalently: to maintain the same reaction rate, a star in a high-σ
region needs HIGHER core temperature. This shifts:
  - Main sequence lifetime
  - Stellar luminosity
  - Element abundances

SSBM prediction: isotope ratios in matter processed through black hole
accretion disks (high σ) differ from standard stellar nucleosynthesis.
This is testable via spectroscopy of AGN jets and accretion disk winds.

Origin tags:
  - Gamow peak: FIRST_PRINCIPLES (WKB + thermal average)
  - S-factors: MEASURED (nuclear experiment)
  - Reaction rates: FIRST_PRINCIPLES (Gamow peak approximation)
  - Energy generation: FIRST_PRINCIPLES (Q × rate)
  - σ-dependence: CORE (through □σ = −ξR)
"""

import math
from ..constants import (
    HBAR, C, E_CHARGE, EPS_0,
    PROTON_TOTAL_MEV, PROTON_BARE_MEV, PROTON_QCD_MEV,
    PROTON_QCD_FRACTION,
)
from ..scale import scale_ratio

# ── Constants ─────────────────────────────────────────────────────
_K_BOLTZMANN = 1.380649e-23       # J/K
_MEV_TO_JOULE = 1.602176634e-13   # MeV → J
_KEV_TO_JOULE = 1.602176634e-16   # keV → J
_AMU_KG = 1.66053906660e-27       # atomic mass unit in kg
_YEAR_S = 365.25 * 86400.0        # Julian year in seconds

# Proton mass in kg at σ=0
_M_PROTON_KG = PROTON_TOTAL_MEV * _MEV_TO_JOULE / C**2

# Coulomb constant × e² in MeV·fm
_KE_E2_J = E_CHARGE**2 / (4.0 * math.pi * EPS_0)  # in J·m

# ── Nuclear Reaction Database ─────────────────────────────────────
# Astrophysical S-factors at zero energy: S(0) in keV·barn.
# Source: Adelberger et al. (2011) Rev. Mod. Phys. 83:195,
#         NACRE II compilation (2013), Solar Fusion Cross Sections III.
#
# Q-values from AME2020 mass tables.
#
# For each reaction:
#   Z1, A1, Z2, A2: reactant nuclei
#   S0_keV_barn: astrophysical S-factor at E=0 (MEASURED)
#   Q_MeV: total energy released
#   f_Q_strong: fraction of Q from strong-force binding differences
#               (rest is Coulomb; used for σ-scaling of Q)

REACTIONS = {
    'pp': {
        'name': 'p + p -> d + e+ + nu_e',
        'description': 'pp-chain step 1: proton-proton fusion (weak)',
        'Z1': 1, 'A1': 1, 'Z2': 1, 'A2': 1,
        'S0_keV_barn': 4.01e-22,    # MEASURED — tiny! (weak interaction)
        'Q_MeV': 1.442,             # includes positron annihilation
        'f_Q_strong': 0.8,          # deuteron binding is mostly strong
    },
    'dp': {
        'name': 'd + p -> 3He + gamma',
        'description': 'pp-chain step 2: deuterium burning',
        'Z1': 1, 'A1': 2, 'Z2': 1, 'A2': 1,
        'S0_keV_barn': 2.14e-4,     # MEASURED (electromagnetic, fast)
        'Q_MeV': 5.493,
        'f_Q_strong': 0.9,
    },
    'He3_He3': {
        'name': '3He + 3He -> 4He + 2p',
        'description': 'pp-chain step 3: helium-3 fusion',
        'Z1': 2, 'A1': 3, 'Z2': 2, 'A2': 3,
        'S0_keV_barn': 5.21e3,      # MEASURED (strong, but higher barrier)
        'Q_MeV': 12.860,
        'f_Q_strong': 0.95,
    },
    'C12_p': {
        'name': '12C + p -> 13N + gamma',
        'description': 'CNO cycle step 1',
        'Z1': 6, 'A1': 12, 'Z2': 1, 'A2': 1,
        'S0_keV_barn': 1.34,        # MEASURED (LUNA experiment)
        'Q_MeV': 1.943,
        'f_Q_strong': 0.7,
    },
    'N14_p': {
        'name': '14N + p -> 15O + gamma',
        'description': 'CNO cycle bottleneck (slowest step)',
        'Z1': 7, 'A1': 14, 'Z2': 1, 'A2': 1,
        'S0_keV_barn': 1.66,        # MEASURED (LUNA)
        'Q_MeV': 7.297,
        'f_Q_strong': 0.85,
    },
    'triple_alpha': {
        'name': '3 4He -> 12C',
        'description': 'Triple-alpha process (Hoyle state resonance)',
        'Z1': 2, 'A1': 4, 'Z2': 2, 'A2': 4,
        # S-factor not meaningful for resonance reaction.
        # We use the NACRE rate formula directly.
        'S0_keV_barn': 0.0,         # placeholder — uses special rate
        'Q_MeV': 7.275,             # for the full 3α → ¹²C
        'f_Q_strong': 0.95,
    },
}


# ── Reduced Mass ──────────────────────────────────────────────────

def reduced_mass_kg(A1, A2, sigma=0.0):
    """Reduced mass of two nuclei in kg.

    μ = m₁ m₂ / (m₁ + m₂)

    Each nuclear mass scales with σ:
      m(σ) = A × [m_bare_per_nucleon + m_QCD_per_nucleon × e^σ]

    For simplicity, we use:
      m(σ) ≈ A × m_proton(σ)

    (Ignoring small neutron-proton mass difference for reduced mass.)

    Args:
        A1, A2: mass numbers
        sigma: σ-field value

    Returns:
        Reduced mass in kg.
    """
    f_qcd = PROTON_QCD_FRACTION
    m_p_sigma = _M_PROTON_KG * ((1.0 - f_qcd) + f_qcd * scale_ratio(sigma))

    m1 = A1 * m_p_sigma
    m2 = A2 * m_p_sigma
    return m1 * m2 / (m1 + m2)


def reduced_mass_mev(A1, A2, sigma=0.0):
    """Reduced mass in MeV/c²."""
    return reduced_mass_kg(A1, A2, sigma) * C**2 / _MEV_TO_JOULE


# ── Gamow Peak ────────────────────────────────────────────────────

def gamow_energy_keV(Z1, A1, Z2, A2, T_K, sigma=0.0):
    """Gamow peak energy in keV.

    E_G = (b × k_B T / 2)^(2/3)

    where b = π × Z₁Z₂ × e² × √(2μ) / (ε₀ × h)
    (often written as b = √(E_G_0) where E_G_0 = (π Z₁Z₂ α)² × μc²/2)

    FIRST_PRINCIPLES: the energy where the Coulomb tunneling
    probability (increases with E) and the Maxwell-Boltzmann tail
    (decreases with E) overlap maximally.

    This is the "sweet spot" for thermonuclear reactions.

    For pp in the Sun (T ≈ 15.7 MK): E_G ≈ 6 keV.
    The Sun burns at a temperature where k_BT ≈ 1.35 keV,
    but the reactions happen at 6 keV thanks to tunneling.

    Args:
        Z1, A1, Z2, A2: reactant nuclei
        T_K: temperature in Kelvin
        sigma: σ-field value

    Returns:
        Gamow peak energy in keV.
    """
    if T_K <= 0:
        return 0.0

    mu = reduced_mass_kg(A1, A2, sigma)
    kT = _K_BOLTZMANN * T_K  # in Joules

    # Gamow parameter b (in Joules^(1/2)):
    # b = π Z₁Z₂ e² √(2μ) / (2 ε₀ h)
    # But more cleanly: b² = 2μ × (π Z₁Z₂ e²/(4πε₀))² / ℏ²
    # So: E_G = (b² kT²/4)^(1/3)

    b_squared = 2.0 * mu * (math.pi * Z1 * Z2 * _KE_E2_J)**2 / HBAR**2
    E_G = (b_squared * kT**2 / 4.0) ** (1.0 / 3.0)

    return E_G / _KEV_TO_JOULE  # convert to keV


def gamow_window_keV(Z1, A1, Z2, A2, T_K, sigma=0.0):
    """Width of the Gamow window in keV.

    Δ = 4 × √(E_G × k_BT / 3)

    The Gamow window is the energy range where most reactions occur.
    Wider window → more of the thermal distribution participates.

    Args:
        Z1, A1, Z2, A2: reactant nuclei
        T_K: temperature in Kelvin
        sigma: σ-field value

    Returns:
        Gamow window width in keV.
    """
    E_G = gamow_energy_keV(Z1, A1, Z2, A2, T_K, sigma)
    kT_keV = _K_BOLTZMANN * T_K / _KEV_TO_JOULE
    return 4.0 * math.sqrt(E_G * kT_keV / 3.0)


# ── Thermonuclear Reaction Rate ───────────────────────────────────

def reaction_rate_sigma_v(reaction_key, T_K, sigma=0.0):
    """Thermally-averaged reaction rate ⟨σv⟩ in cm³/s.

    Uses the Gamow peak approximation:
      ⟨σv⟩ ≈ (8/9)^(1/2) × S(E_G) / (μ E_G) × Δ × exp(−3E_G/k_BT)

    This integral is dominated by a narrow Gaussian around E_G.

    For the triple-alpha process, we use the NACRE parametrization
    since it's a sequential resonance, not a simple two-body tunneling.

    FIRST_PRINCIPLES: WKB tunneling × thermal average.
    S-factor: MEASURED.

    Args:
        reaction_key: key into REACTIONS dict
        T_K: temperature in Kelvin
        sigma: σ-field value

    Returns:
        ⟨σv⟩ in cm³/s.
    """
    rxn = REACTIONS[reaction_key]

    if T_K <= 0:
        return 0.0

    # Triple-alpha uses special rate formula
    if reaction_key == 'triple_alpha':
        return _triple_alpha_rate(T_K, sigma)

    Z1, A1, Z2, A2 = rxn['Z1'], rxn['A1'], rxn['Z2'], rxn['A2']
    S0 = rxn['S0_keV_barn']

    if S0 <= 0:
        return 0.0

    # Convert S-factor: keV·barn → keV·cm²
    S0_keV_cm2 = S0 * 1e-24  # 1 barn = 10⁻²⁴ cm²

    mu_kg = reduced_mass_kg(A1, A2, sigma)
    kT_keV = _K_BOLTZMANN * T_K / _KEV_TO_JOULE
    E_G_keV = gamow_energy_keV(Z1, A1, Z2, A2, T_K, sigma)

    if E_G_keV <= 0 or kT_keV <= 0:
        return 0.0

    # Gamow window width
    delta_keV = gamow_window_keV(Z1, A1, Z2, A2, T_K, sigma)

    # ⟨σv⟩ in the Gamow peak approximation:
    # From Clayton "Principles of Stellar Evolution and Nucleosynthesis":
    #
    # ⟨σv⟩ = √(2/(μ kT)) × (2/√3) × S(E_G) × exp(−3E_G/kT) × Δ/(2kT)
    #
    # More precisely:
    # ⟨σv⟩ = (2/μ)^(1/2) × (2/(3kT))^(1/2) × S_eff × Δ × exp(-τ)
    #
    # where τ = 3 E_G / kT and S_eff ≈ S(0) for slowly varying S.
    #
    # Standard form from Iliadis "Nuclear Physics of Stars":
    # ⟨σv⟩ = (8/(9√3)) × 1/√(μ (kT)³) × S₀ × τ² × exp(-τ)
    # where τ = 3 E_G / kT

    tau = 3.0 * E_G_keV / kT_keV

    # Convert everything to CGS for the standard formula
    mu_g = mu_kg * 1e3  # kg → g
    kT_erg = _K_BOLTZMANN * T_K * 1e7  # J → erg
    S0_erg_cm2 = S0_keV_cm2 * _KEV_TO_JOULE * 1e7  # keV·cm² → erg·cm²

    # ⟨σv⟩ = (8/(9√3 π))^(1/2) × 1/(μ kT)^(1/2) × S₀ × (τ²/3) × exp(-τ)
    # Wait, let me use the clean form:
    #
    # ⟨σv⟩ = √(8/(π μ kT)) × Δ_E/(2kT) × S(E_G) × exp(-τ)
    #
    # where Δ_E = 4√(E_G kT / 3)
    #
    # But the canonical formula from Fowler is:
    # ⟨σv⟩ = (8/(π μ))^(1/2) × (kT)^(-3/2) × S₀ ×
    #         ∫ exp(-E/kT - (E_G_0/E)^(1/2)) dE
    # ≈ (4/√3) × √(2/(μ (kT)³)) × S₀ × E_G × exp(-τ)
    #
    # Let me just use the dimensional formula directly:

    # Gamow peak height: exp(-τ)
    # τ = 3 E_G / kT
    # For pp at T_sun: τ ≈ 3 × 6 / 1.35 ≈ 13.3 → exp(-τ) ≈ 1.7 × 10⁻⁶

    # Clean formula (Iliadis Eq. 3.107):
    # ⟨σv⟩ = (2/(μ_amu))^(1/2) × 7.8327 × 10⁹ × T₉^(-2/3) ×
    #         S_eff(keV·barn) × exp(-τ) cm³ s⁻¹ mol⁻¹
    # where μ_amu = A₁A₂/(A₁+A₂), T₉ = T/10⁹ K

    mu_amu = A1 * A2 / (A1 + A2)
    T9 = T_K / 1e9

    # τ = 4.2487 × (Z₁² Z₂² μ_amu / T₉)^(1/3)
    tau_formula = 4.2487 * (Z1**2 * Z2**2 * mu_amu / T9) ** (1.0 / 3.0)

    # σ-correction to τ: μ_amu(σ) = μ_amu(0) × mass_ratio
    f_qcd = PROTON_QCD_FRACTION
    mass_ratio = (1.0 - f_qcd) + f_qcd * scale_ratio(sigma)
    tau_sigma = 4.2487 * (Z1**2 * Z2**2 * mu_amu * mass_ratio / T9) ** (1.0 / 3.0)

    # ⟨σv⟩ per particle pair (not per mole):
    # N_A ⟨σv⟩ = 7.8327e9 × (μ_amu × mass_ratio)^(-1/2) × T₉^(-2/3) ×
    #              S₀(keV·barn) × exp(-τ_σ) cm³ mol⁻¹ s⁻¹
    #
    # Divide by N_A to get per-particle rate:
    N_A = 6.02214076e23
    NA_sv = (7.8327e9 / (mu_amu * mass_ratio)**0.5 *
             T9**(-2.0/3.0) * S0 * math.exp(-tau_sigma))

    sigma_v = NA_sv / N_A  # cm³/s per particle pair

    return sigma_v


def _triple_alpha_rate(T_K, sigma=0.0):
    """Triple-alpha reaction rate ⟨σv⟩_3α (cm⁶/s).

    The triple-alpha process is a sequential two-body reaction:
      ⁴He + ⁴He ↔ ⁸Be* (quasi-equilibrium, Saha equation)
      ⁸Be* + ⁴He → ¹²C* → ¹²C + γ (Hoyle state resonance)

    The effective three-body rate from NACRE (Angulo et al. 1999):
      r₃α = n_α³ × (N_A²/6) × ⟨σv⟩²₃α

    We use the Fynbo et al. (2005) / NACRE II parametrization:
      log₁₀(N_A² ⟨σv⟩_3α) ≈ A + B/T₉ + C/T₉^(1/3) + D×T₉^(1/3) + E×log₁₀(T₉)

    For the σ-dependence: the Hoyle state energy (7.654 MeV above ¹²C
    ground state) has both strong and Coulomb components. The resonance
    energy shifts with σ, dramatically affecting the rate.

    MEASURED: rate parametrization from nuclear experiment.
    σ-dependence: CORE (resonance energy shifts through QCD binding).

    Returns:
        Effective ⟨σv⟩²_3α in cm⁶ s⁻¹ mol⁻² (for r = n³ N_A² ⟨σv⟩²/6).
    """
    if T_K <= 0:
        return 0.0

    T9 = T_K / 1e9

    if T9 < 0.01 or T9 > 10.0:
        return 0.0  # outside validity range

    # NACRE II parametrization for triple-alpha:
    # Dominant contribution: Hoyle state resonance at E_r = 379.47 keV
    # (above the 3α threshold)
    #
    # Rate: r₃α = 3.04 × 10⁻⁵⁶ × T₉⁻³ × exp(−4.4027/T₉) cm⁶ s⁻¹ mol⁻²
    # This is the Nomoto (1982) / Caughlan & Fowler (1988) rate.

    # σ-correction: the resonance energy E_r scales with the strong
    # component. E_r = 379.47 keV, mostly QCD binding.
    f_strong = 0.9  # 90% of resonance energy is strong-force
    e_sig = scale_ratio(sigma)
    E_r_keV_sigma = 379.47 * ((1.0 - f_strong) + f_strong * e_sig)

    # The rate goes as exp(−E_r / kT), so shifting E_r shifts the rate
    # exponentially.
    E_r_keV_0 = 379.47
    # exp(-E_r(σ)/kT) / exp(-E_r(0)/kT) = exp(-(E_r(σ)-E_r(0))/kT)
    kT_keV = _K_BOLTZMANN * T_K / _KEV_TO_JOULE

    # Base rate (Caughlan & Fowler 1988)
    rate_0 = 3.04e-56 * T9**(-3) * math.exp(-4.4027 / T9)

    # σ-correction factor
    delta_E_keV = E_r_keV_sigma - E_r_keV_0
    sigma_correction = math.exp(-delta_E_keV / kT_keV) if kT_keV > 0 else 0.0

    return rate_0 * sigma_correction


# ── Energy Generation Rate ────────────────────────────────────────

def pp_chain_energy_rate(T_K, rho_kg_m3, X_H=0.70, sigma=0.0):
    """Energy generation rate from the pp-chain (W/kg).

    ε_pp = ρ X² × (N_A²/2) × ⟨σv⟩_pp × Q_pp_eff / m_H²

    The pp reaction (p+p → d+e⁺+ν) is rate-limiting. The subsequent
    reactions (d+p → ³He, ³He+³He → ⁴He+2p) are much faster and
    process the deuterium almost instantly.

    Effective Q for the full pp-I chain: Q_eff ≈ 26.73 MeV per ⁴He produced.
    But 2% goes to neutrinos (lost). Net: Q_eff ≈ 26.2 MeV.
    Each pp reaction produces 1/2 of a ⁴He eventually (need 2 pp to make one ⁴He).

    So: ε = (1/2) × ⟨σv⟩_pp × n_H² × Q_eff / ρ

    FIRST_PRINCIPLES: reaction rate × energy per reaction / mass.

    For the Sun: ε_pp ≈ 1.7 × 10⁻³ W/kg (core average) to
    ~30 W/kg (center). Our central temperature gives the peak rate.

    Args:
        T_K: temperature in Kelvin
        rho_kg_m3: mass density in kg/m³
        X_H: hydrogen mass fraction (default: solar 0.70)
        sigma: σ-field value

    Returns:
        Energy generation rate in W/kg (specific power).
    """
    if T_K <= 0 or rho_kg_m3 <= 0 or X_H <= 0:
        return 0.0

    sv_pp = reaction_rate_sigma_v('pp', T_K, sigma)  # cm³/s

    # Number density of hydrogen (particles/cm³)
    rho_cgs = rho_kg_m3 * 1e-3  # kg/m³ → g/cm³
    n_H = rho_cgs * X_H / (_M_PROTON_KG * 1e3)  # protons/cm³

    # Q effective for full pp-I chain: 26.2 MeV per ⁴He
    # Each pp reaction → 1 deuterium → eventually 1/2 of a ⁴He
    # So energy per pp reaction (counting full chain):
    Q_eff_MeV = 26.2 / 2.0  # MeV per pp reaction
    Q_eff_J = Q_eff_MeV * _MEV_TO_JOULE

    # σ-correction to Q:
    f_strong = REACTIONS['pp']['f_Q_strong']
    e_sig = scale_ratio(sigma)
    Q_sigma_J = Q_eff_J * ((1.0 - f_strong) + f_strong * e_sig)

    # Reaction rate per unit volume: r = n_H² × ⟨σv⟩ / 2
    # (factor 2: identical particles)
    r_pp = n_H**2 * sv_pp / 2.0  # reactions/cm³/s

    # Energy rate per unit volume
    eps_vol = r_pp * Q_sigma_J  # W/cm³

    # Convert to specific power (W/kg)
    eps_specific = eps_vol / rho_cgs * 1e-3  # W/cm³ / (g/cm³) × (1e-3 kg/g) oops
    # Actually: eps_vol [J/(cm³·s)] / rho [g/cm³] → J/(g·s) = W/g
    # Then × 1000 g/kg → W/kg
    eps_specific = eps_vol / rho_cgs * 1e3  # W/kg

    return eps_specific


def cno_energy_rate(T_K, rho_kg_m3, X_H=0.70, X_CNO=0.01, sigma=0.0):
    """Energy generation rate from the CNO cycle (W/kg).

    The CNO cycle is catalyzed by ¹²C, ¹³C, ¹⁴N, ¹⁵N, ¹⁵O, ¹³N.
    The slowest step is ¹⁴N + p → ¹⁵O + γ (bottleneck).

    ε_CNO = n_H × n_14N × ⟨σv⟩_N14p × Q_eff / ρ

    Q_eff ≈ 25.0 MeV per cycle (similar to pp, but different ν losses).

    At T > 17 MK, CNO dominates over pp (steeper T dependence).
    CNO ∝ T^~16 vs pp ∝ T^~4.

    Args:
        T_K: temperature in Kelvin
        rho_kg_m3: mass density in kg/m³
        X_H: hydrogen mass fraction
        X_CNO: CNO element mass fraction (default: solar ~0.01)
        sigma: σ-field value

    Returns:
        Energy generation rate in W/kg.
    """
    if T_K <= 0 or rho_kg_m3 <= 0 or X_H <= 0 or X_CNO <= 0:
        return 0.0

    sv_N14p = reaction_rate_sigma_v('N14_p', T_K, sigma)  # cm³/s

    rho_cgs = rho_kg_m3 * 1e-3
    n_H = rho_cgs * X_H / (_M_PROTON_KG * 1e3)

    # ¹⁴N number density: most CNO is in ¹⁴N at equilibrium
    # (bottleneck accumulates material as ¹⁴N)
    m_N14_g = 14.0 * _AMU_KG * 1e3
    n_N14 = rho_cgs * X_CNO / m_N14_g

    Q_eff_MeV = 25.0  # MeV per CNO cycle
    Q_eff_J = Q_eff_MeV * _MEV_TO_JOULE

    f_strong = REACTIONS['N14_p']['f_Q_strong']
    e_sig = scale_ratio(sigma)
    Q_sigma_J = Q_eff_J * ((1.0 - f_strong) + f_strong * e_sig)

    r_cno = n_H * n_N14 * sv_N14p
    eps_vol = r_cno * Q_sigma_J
    eps_specific = eps_vol / rho_cgs * 1e3

    return eps_specific


# ── Stellar Burning Temperatures ──────────────────────────────────

def pp_cno_crossover_temperature(X_H=0.70, X_CNO=0.01, sigma=0.0):
    """Temperature where CNO rate equals pp rate (in Kelvin).

    Above this temperature, CNO dominates. Below, pp dominates.
    For the Sun (σ=0): T_cross ≈ 17 MK.

    σ-dependence: higher σ → heavier nuclei → higher Coulomb barriers
    → both rates decrease, but CNO (higher Z) is more affected.
    So T_cross shifts upward with σ.

    Uses bisection search (no scipy needed).

    Args:
        X_H: hydrogen mass fraction
        X_CNO: CNO mass fraction
        sigma: σ-field value

    Returns:
        Crossover temperature in Kelvin.
    """
    rho = 100e3  # kg/m³ — typical stellar core (doesn't affect crossover much)

    T_lo = 5e6    # 5 MK
    T_hi = 50e6   # 50 MK

    for _ in range(60):  # bisection iterations (convergence: 2⁻⁶⁰ ≈ 10⁻¹⁸)
        T_mid = (T_lo + T_hi) / 2.0
        eps_pp = pp_chain_energy_rate(T_mid, rho, X_H, sigma)
        eps_cno = cno_energy_rate(T_mid, rho, X_H, X_CNO, sigma)
        if eps_cno > eps_pp:
            T_hi = T_mid
        else:
            T_lo = T_mid

    return (T_lo + T_hi) / 2.0


def pp_temperature_exponent(T_K, sigma=0.0):
    """Power-law exponent ν for pp-chain: ε ∝ T^ν.

    Computed numerically: ν = d(ln ε)/d(ln T).

    For the Sun: ν ≈ 4.0 at T = 15.7 MK.
    This is why stellar luminosity is so sensitive to core temperature.

    Args:
        T_K: temperature in Kelvin
        sigma: σ-field value

    Returns:
        Power-law exponent ν.
    """
    rho = 100e3  # arbitrary (cancels in ratio)
    dT = T_K * 0.01  # 1% step

    eps_lo = pp_chain_energy_rate(T_K - dT/2, rho, sigma=sigma)
    eps_hi = pp_chain_energy_rate(T_K + dT/2, rho, sigma=sigma)

    if eps_lo <= 0 or eps_hi <= 0:
        return 0.0

    return math.log(eps_hi / eps_lo) / math.log((T_K + dT/2) / (T_K - dT/2))


# ── Nagatha Export ────────────────────────────────────────────────

def reaction_properties(reaction_key, T_K=15.7e6, sigma=0.0):
    """Export reaction properties in Nagatha-compatible format.

    Args:
        reaction_key: key into REACTIONS dict
        T_K: temperature in Kelvin (default: solar core)
        sigma: σ-field value

    Returns:
        Dict with all reaction quantities and origin tags.
    """
    rxn = REACTIONS[reaction_key]
    Z1, A1, Z2, A2 = rxn['Z1'], rxn['A1'], rxn['Z2'], rxn['A2']

    E_G = gamow_energy_keV(Z1, A1, Z2, A2, T_K, sigma)
    delta = gamow_window_keV(Z1, A1, Z2, A2, T_K, sigma)
    sv = reaction_rate_sigma_v(reaction_key, T_K, sigma)
    mu = reduced_mass_mev(A1, A2, sigma)

    Q_MeV = rxn['Q_MeV']
    f_strong = rxn['f_Q_strong']
    e_sig = scale_ratio(sigma)
    Q_sigma = Q_MeV * ((1.0 - f_strong) + f_strong * e_sig)

    result = {
        'reaction': reaction_key,
        'name': rxn['name'],
        'description': rxn['description'],
        'temperature_K': T_K,
        'temperature_MK': T_K / 1e6,
        'sigma': sigma,
        'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2,
        'reduced_mass_MeV': mu,
        'gamow_energy_keV': E_G,
        'gamow_window_keV': delta,
        'sigma_v_cm3_s': sv,
        'Q_MeV': Q_MeV,
        'Q_at_sigma_MeV': Q_sigma,
        'S0_keV_barn': rxn['S0_keV_barn'],
        'origin': (
            "Gamow peak: FIRST_PRINCIPLES (WKB tunneling + Maxwell-Boltzmann). "
            "S-factor: MEASURED (nuclear experiment). "
            "Reaction rate: FIRST_PRINCIPLES (Gamow peak approximation). "
            "Q-value: MEASURED (AME2020). "
            "σ-dependence: CORE (μ and Q shift through QCD mass)."
        ),
    }

    return result


def stellar_burning_summary(T_K=15.7e6, rho_kg_m3=150e3, sigma=0.0):
    """Summary of stellar energy generation at given conditions.

    Default: solar core (T = 15.7 MK, ρ = 150 g/cm³).

    Returns:
        Dict with pp and CNO rates, crossover, and σ-comparison.
    """
    eps_pp = pp_chain_energy_rate(T_K, rho_kg_m3, sigma=sigma)
    eps_cno = cno_energy_rate(T_K, rho_kg_m3, sigma=sigma)
    T_cross = pp_cno_crossover_temperature(sigma=sigma)

    return {
        'temperature_K': T_K,
        'temperature_MK': T_K / 1e6,
        'density_kg_m3': rho_kg_m3,
        'sigma': sigma,
        'epsilon_pp_W_kg': eps_pp,
        'epsilon_cno_W_kg': eps_cno,
        'epsilon_total_W_kg': eps_pp + eps_cno,
        'dominant_chain': 'pp' if eps_pp >= eps_cno else 'CNO',
        'pp_cno_crossover_MK': T_cross / 1e6,
        'pp_fraction': eps_pp / (eps_pp + eps_cno) if (eps_pp + eps_cno) > 0 else 0,
        'origin': (
            "pp rate: FIRST_PRINCIPLES (Gamow) + MEASURED (S-factor). "
            "CNO rate: FIRST_PRINCIPLES (Gamow) + MEASURED (S-factor, LUNA). "
            "σ-dependence: CORE (Gamow peak shifts through nuclear mass)."
        ),
    }
