"""
Radioactive decay from first-principles nuclear physics.

Derivation chain:
  σ → nuclear mass → Coulomb barrier → tunneling probability → decay rate
  σ → binding energy → Q-value → decay energetics

Three decay modes, each derived from fundamentals:

  1. Alpha Decay (Gamow theory, 1928)
     The alpha particle (⁴He nucleus) tunnels through the Coulomb barrier.

     Decay rate:
       λ = f × P_tunnel
     where:
       f = v / (2R)     — assault frequency (FIRST_PRINCIPLES: classical
                           oscillation of alpha in nuclear well)
       v = √(2 Q / m_α) — alpha velocity from Q-value
       R = r₀ × A^(1/3) — nuclear radius

     Tunneling probability (Gamow):
       P_tunnel = exp(−2G)
       G = (Z_d × Z_α × e² / (4πε₀ℏ)) × √(2m_α / Q) × [arccos(√x) − √(x(1−x))]
       where x = R / R_turn, R_turn = Z_d × Z_α × e² / (4πε₀ × Q)

     FIRST_PRINCIPLES: WKB approximation to quantum tunneling through
     Coulomb potential. The Gamow factor is exact for a pure Coulomb barrier.

     σ-dependence: Q-value has strong + Coulomb components.
       - Strong binding scales with e^σ → shifts Q
       - Coulomb barrier is EM → σ-INVARIANT
       - m_α(σ) = m_α_bare + m_α_QCD × e^σ → shifts tunneling mass
       Combined: decay rates CHANGE in strong σ fields.
       This is a TESTABLE SSBM prediction.

  2. Beta Decay (Fermi theory, 1934)
     Neutron → proton + electron + antineutrino (β⁻)
     Proton → neutron + positron + neutrino (β⁺)

     Decay rate (simplified Sargent's rule):
       λ ∝ G_F² × Q⁵ / (60π³ℏ)
     where:
       G_F = Fermi coupling constant (weak force, σ-INVARIANT)
       Q = endpoint energy (mass difference − electron mass)

     FIRST_PRINCIPLES: Fermi's golden rule applied to weak interaction.
     APPROXIMATION: Sargent's rule (Q⁵ scaling) is the leading term.
     Real rates need Coulomb correction (Fermi function) and nuclear
     matrix elements.

     σ-dependence:
       Q = (m_n − m_p − m_e)c² for free neutron decay
       Since m_n and m_p shift differently with σ (different bare masses),
       Q shifts → λ shifts as Q⁵.
       At large σ: Q can vanish → beta decay turns off!

  3. Gamma Decay (electromagnetic transition)
     Excited nucleus → ground state + photon

     Rate: depends on multipole order and transition energy.
     σ-dependence: nuclear level spacing shifts through nuclear mass,
     but the transition itself is EM → σ-INVARIANT coupling.

     We include gamma for completeness but focus on alpha and beta,
     which have the richest σ-physics.

──────────────────────────────────────────────────────────────────────────────
σ-DEPENDENCE SUMMARY
──────────────────────────────────────────────────────────────────────────────

Alpha decay:
  Strong binding → Q-value shifts → tunneling probability shifts EXPONENTIALLY.
  The Gamow factor depends on √(m_α/Q), and both shift with σ.
  Small σ changes → large lifetime changes (exponential sensitivity).

Beta decay:
  Q = m_n(σ) − m_p(σ) − m_e
  Since m_n and m_p have different bare masses but similar QCD masses,
  the Q-value shifts predictably. λ ∝ Q⁵ → steep dependence.

SSBM prediction: radioactive half-lives are different inside black hole
accretion disks. This is in principle measurable via spectroscopy of
accreting matter (isotope ratios reveal decay rates).

──────────────────────────────────────────────────────────────────────────────

Origin tags:
  - Gamow tunneling: FIRST_PRINCIPLES (WKB + Coulomb barrier)
  - Assault frequency: FIRST_PRINCIPLES (classical nuclear oscillation)
  - Sargent's rule: FIRST_PRINCIPLES (Fermi golden rule) +
    APPROXIMATION (leading-order Q⁵, no Coulomb correction)
  - Q-values: MEASURED (from nuclear mass tables)
  - Half-lives: MEASURED (for validation)
  - σ-dependence: CORE (through □σ = −ξR)
"""

import math
from ..constants import (
    HBAR, C, E_CHARGE, EPS_0, R0_FM, MEV_TO_J, AMU_KG as _AMU_KG,
    PROTON_TOTAL_MEV, NEUTRON_TOTAL_MEV,
    PROTON_BARE_MEV, PROTON_QCD_MEV,
    NEUTRON_BARE_MEV, NEUTRON_QCD_MEV,
    DELTA_NP_TOTAL_MEV, DELTA_NP_BARE_MEV, DELTA_NP_QCD_MEV,
    M_ELECTRON_MEV, A_C_MEV,
    PROTON_QCD_FRACTION,
    SIGMA_HERE,
)
from ..scale import scale_ratio
from ..binding import coulomb_energy_mev

# ── Constants ─────────────────────────────────────────────────────
_MEV_TO_JOULE = MEV_TO_J

# Alpha particle properties at σ = 0
# ⁴He nucleus: 2p + 2n, binding energy 28.296 MeV
ALPHA_Z = 2
ALPHA_A = 4
ALPHA_BE_MEV = 28.296              # MEASURED (AME2020)
ALPHA_MASS_MEV = (2 * PROTON_TOTAL_MEV + 2 * NEUTRON_TOTAL_MEV
                  - ALPHA_BE_MEV)  # ≈ 3727.38 MeV

# Bare (Higgs) mass of alpha: 2×m_p_bare + 2×m_n_bare
ALPHA_BARE_MEV = 2 * PROTON_BARE_MEV + 2 * NEUTRON_BARE_MEV  # ≈ 40.98 MeV
# QCD mass of alpha (at σ=0): total − bare − binding removed from QCD
# Actually: QCD mass = (2×m_p_QCD + 2×m_n_QCD) − BE_strong
# For simplicity, track the total mass shift:
ALPHA_QCD_MEV = ALPHA_MASS_MEV - ALPHA_BARE_MEV  # ≈ 3686.40 MeV
ALPHA_QCD_FRACTION = ALPHA_QCD_MEV / ALPHA_MASS_MEV  # ≈ 0.989

# Fermi coupling constant (weak interaction — σ-INVARIANT)
# G_F/(ℏc)³ = 1.1663788 × 10⁻⁵ GeV⁻² (PDG 2024)
G_FERMI_GEV2 = 1.1663788e-5       # GeV⁻²

# ── Nuclear Isotope Database ──────────────────────────────────────
# All values MEASURED. Sources: NNDC (Brookhaven), NUBASE2020,
# AME2020 (atomic mass evaluation).
#
# Fields:
#   Z, A: proton number, mass number
#   decay_mode: 'alpha', 'beta_minus', 'beta_plus', 'stable'
#   Q_value_MeV: total decay energy (kinetic + mass difference)
#   half_life_s: measured half-life in seconds
#   daughter_Z, daughter_A: daughter nucleus identity
#   be_per_nucleon_MeV: binding energy per nucleon (for σ-scaling)

ISOTOPES = {
    'U238': {
        'name': 'Uranium-238',
        'Z': 92, 'A': 238,
        'decay_mode': 'alpha',
        'Q_value_MeV': 4.270,       # alpha kinetic energy + recoil
        'half_life_s': 1.410e17,     # 4.468 × 10⁹ years
        'daughter_Z': 90, 'daughter_A': 234,
        'daughter_name': 'Th-234',
        'be_per_nucleon_MeV': 7.570,
    },
    'Ra226': {
        'name': 'Radium-226',
        'Z': 88, 'A': 226,
        'decay_mode': 'alpha',
        'Q_value_MeV': 4.871,
        'half_life_s': 5.049e10,     # 1600 years
        'daughter_Z': 86, 'daughter_A': 222,
        'daughter_name': 'Rn-222',
        'be_per_nucleon_MeV': 7.662,
    },
    'Po210': {
        'name': 'Polonium-210',
        'Z': 84, 'A': 210,
        'decay_mode': 'alpha',
        'Q_value_MeV': 5.407,
        'half_life_s': 1.196e7,      # 138.4 days
        'daughter_Z': 82, 'daughter_A': 206,
        'daughter_name': 'Pb-206',
        'be_per_nucleon_MeV': 7.834,
    },
    'Po212': {
        'name': 'Polonium-212',
        'Z': 84, 'A': 212,
        'decay_mode': 'alpha',
        'Q_value_MeV': 8.954,
        'half_life_s': 2.99e-7,      # 299 ns (very fast — high Q)
        'daughter_Z': 82, 'daughter_A': 208,
        'daughter_name': 'Pb-208',
        'be_per_nucleon_MeV': 7.805,
    },
    'C14': {
        'name': 'Carbon-14',
        'Z': 6, 'A': 14,
        'decay_mode': 'beta_minus',
        'Q_value_MeV': 0.156,       # very low Q → very long half-life
        'half_life_s': 1.808e11,     # 5730 years
        'daughter_Z': 7, 'daughter_A': 14,
        'daughter_name': 'N-14',
        'be_per_nucleon_MeV': 7.520,
    },
    'Co60': {
        'name': 'Cobalt-60',
        'Z': 27, 'A': 60,
        'decay_mode': 'beta_minus',
        'Q_value_MeV': 2.824,
        'half_life_s': 1.663e8,      # 5.27 years
        'daughter_Z': 28, 'daughter_A': 60,
        'daughter_name': 'Ni-60',
        'be_per_nucleon_MeV': 8.747,
    },
    'K40': {
        'name': 'Potassium-40',
        'Z': 19, 'A': 40,
        'decay_mode': 'beta_minus',
        'Q_value_MeV': 1.311,
        'half_life_s': 3.938e16,     # 1.248 × 10⁹ years
        'daughter_Z': 20, 'daughter_A': 40,
        'daughter_name': 'Ca-40',
        'be_per_nucleon_MeV': 8.557,
    },
    'free_neutron': {
        'name': 'Free neutron',
        'Z': 0, 'A': 1,
        'decay_mode': 'beta_minus',
        'Q_value_MeV': DELTA_NP_TOTAL_MEV - M_ELECTRON_MEV,  # avoid 939-938 cancellation
        'half_life_s': 611.0,        # ~10.2 minutes (MEASURED)
        'daughter_Z': 1, 'daughter_A': 1,
        'daughter_name': 'Proton',
        'be_per_nucleon_MeV': 0.0,
    },
}


# ── Alpha Decay: Gamow Theory ────────────────────────────────────

def alpha_mass_mev(sigma=SIGMA_HERE):
    """Alpha particle mass in MeV at given σ.

    m_α(σ) = m_α_bare + m_α_QCD × e^σ

    The alpha's QCD mass content scales with σ just like individual
    nucleons, minus the binding energy (which also has QCD scaling).
    """
    return ALPHA_BARE_MEV + ALPHA_QCD_MEV * scale_ratio(sigma)


def nuclear_radius_fm(A):
    """Nuclear radius R = r₀ × A^(1/3) in femtometers.

    FIRST_PRINCIPLES: liquid drop model. Nuclear matter has constant
    density, so volume ∝ A → radius ∝ A^(1/3).

    r₀ = 1.215 fm (from electron scattering — Hofstadter).
    """
    return R0_FM * A ** (1.0 / 3.0)


def gamow_factor(Z_daughter, Q_MeV, m_alpha_MeV, A_parent):
    """Gamow tunneling factor G for alpha decay.

    The tunneling probability is P = exp(−2G).

    G = (π × Z_d × Z_α × e² / (4πε₀ℏc)) × √(2 m_α c² / Q)
        × [arccos(√η) − √(η(1−η))] / π

    where η = R / R_turn (ratio of nuclear radius to classical
    turning point).

    For the standard Gamow approximation (R << R_turn, η << 1):
      G ≈ (Z_d × Z_α × e²) / (4πε₀ℏ) × √(2m_α/Q) × (π/2 − 2√η)

    We use the full formula for accuracy.

    FIRST_PRINCIPLES: WKB approximation applied to Coulomb barrier.
    Exact for a pure Coulomb potential outside the nuclear radius.

    Args:
        Z_daughter: proton number of daughter nucleus
        Q_MeV: Q-value (total kinetic energy released) in MeV
        m_alpha_MeV: alpha particle mass in MeV
        A_parent: mass number of parent nucleus

    Returns:
        Gamow factor G (dimensionless). Tunneling P = exp(−2G).
    """
    if Q_MeV <= 0 or Z_daughter <= 0:
        return float('inf')  # no tunneling possible

    Z_alpha = ALPHA_Z

    # Sommerfeld parameter: dimensionless measure of Coulomb strength
    # η_S = Z_d × Z_α × e² / (4πε₀ × ℏ × v_α)
    # where v_α = √(2Q/m_α)

    # Nuclear radius in meters
    R_m = nuclear_radius_fm(A_parent) * 1e-15

    # Classical turning point: R_turn = Z_d × Z_α × ke²/Q
    ke_e2_joule = E_CHARGE**2 / (4.0 * math.pi * EPS_0)  # Coulomb constant × e²
    Q_joule = Q_MeV * _MEV_TO_JOULE
    m_alpha_kg = m_alpha_MeV * _MEV_TO_JOULE / C**2

    R_turn = Z_daughter * Z_alpha * ke_e2_joule / Q_joule

    if R_turn <= R_m:
        return 0.0  # above barrier, no tunneling needed

    # eta = R / R_turn
    eta = R_m / R_turn

    # Gamow integral (exact for Coulomb barrier):
    # I = arccos(√η) − √(η(1−η))
    sqrt_eta = math.sqrt(eta)
    integral = math.acos(sqrt_eta) - math.sqrt(eta * (1.0 - eta))

    # G = √(2 m_α / Q) × (Z_d Z_α e² / (4πε₀ℏ)) × integral
    # Factor: √(2m_α c²/Q) × (Z_d Z_α α / (2)) × something...
    # Let me be explicit in SI:
    v_alpha = math.sqrt(2.0 * Q_joule / m_alpha_kg)
    sommerfeld = Z_daughter * Z_alpha * ke_e2_joule / (HBAR * v_alpha)

    # G = sommerfeld × (R_turn / R_m)^(1/2) × integral... no.
    # Actually: G = (1/ℏ) × ∫[R to R_turn] √(2m(V(r)−Q)) dr
    # For Coulomb V(r) = Z_d Z_α e²/(4πε₀ r):
    # G = sommerfeld × [arccos(√η) − √(η(1−η))]
    # Factor of 2 from the WKB integral substitution:
    # G = (2/ℏ) × ∫ √(2μ(V-E)) dr = 2η × [arccos(√x) − √(x(1−x))]
    G = 2.0 * sommerfeld * integral

    return G


def alpha_Q_decomposition(isotope_key):
    """Decompose alpha decay Q-value into Coulomb and strong components.

    Q = BE_daughter + BE_alpha − BE_parent
    Each BE = BE_strong − E_Coulomb (binding.py decomposition).

    Therefore:
      Q = (BE_strong_d + BE_strong_α − BE_strong_p)     [strong: scales with e^σ]
        − (E_C_d + E_C_α − E_C_p)                       [Coulomb: σ-INVARIANT]

    Define:
      Q_coulomb = E_C_parent − E_C_daughter − E_C_alpha  [EM push on alpha]
      Q_strong  = Q_measured − Q_coulomb                  [strong pull on alpha]

    The Q-value is a knife-edge balance: Coulomb pushes the alpha out
    (Q_coulomb >> 0) while the strong force holds it in (Q_strong << 0).
    For U-238: Q_coulomb ≈ +36 MeV, Q_strong ≈ −31 MeV, net Q = +4.3 MeV.

    At higher σ, Q_strong grows more negative → Q decreases → decay slows.
    At critical σ, Q goes negative → alpha decay is FORBIDDEN.

    DERIVED from coulomb_energy_mev() — no guesswork.

    Returns:
        (Q_coulomb_MeV, Q_strong_MeV)
    """
    iso = ISOTOPES[isotope_key]
    Z_p, A_p = iso['Z'], iso['A']
    Z_d, A_d = iso['daughter_Z'], iso['daughter_A']
    Q_measured = iso['Q_value_MeV']

    E_C_parent = coulomb_energy_mev(Z_p, A_p)
    E_C_daughter = coulomb_energy_mev(Z_d, A_d)
    E_C_alpha = coulomb_energy_mev(ALPHA_Z, ALPHA_A)

    Q_coulomb = E_C_parent - E_C_daughter - E_C_alpha
    Q_strong = Q_measured - Q_coulomb

    return (Q_coulomb, Q_strong)


def alpha_Q_at_sigma(isotope_key, sigma=SIGMA_HERE):
    """Alpha decay Q-value at arbitrary σ.

    Q(σ) = Q_strong × e^σ + Q_coulomb

    Strong binding scales with σ (pulls alpha in tighter).
    Coulomb repulsion is EM (σ-invariant, pushes alpha out).

    DERIVED from binding decomposition — not estimated.

    Args:
        isotope_key: key into ISOTOPES dict
        sigma: σ-field value

    Returns:
        Q-value in MeV at given σ.
    """
    Q_coulomb, Q_strong = alpha_Q_decomposition(isotope_key)
    e_sig = scale_ratio(sigma)
    return Q_strong * e_sig + Q_coulomb


def alpha_decay_constant(isotope_key, sigma=SIGMA_HERE):
    """Alpha decay constant λ (s⁻¹) from Gamow theory.

    λ = f × exp(−2G)

    where:
      f = v_α / (2R) — assault frequency
      G = Gamow tunneling factor

    The alpha particle bounces back and forth inside the nucleus
    (frequency f), and each time it hits the barrier, it has
    probability exp(−2G) of tunneling through.

    σ-dependence: both m_α and Q shift. The Gamow factor is
    exponentially sensitive to √(m_α/Q), so even small σ changes
    can dramatically alter decay rates.

    Q(σ) is DERIVED from the Coulomb/strong decomposition:
      Q(σ) = Q_strong × e^σ + Q_coulomb
    where Q_coulomb and Q_strong are computed from coulomb_energy_mev().

    At high σ: Q_strong (negative) grows → Q decreases → decay slows.
    At critical σ: Q → 0 → alpha decay turns off entirely.

    Args:
        isotope_key: key into ISOTOPES dict
        sigma: σ-field value

    Returns:
        Decay constant λ in s⁻¹.
    """
    iso = ISOTOPES[isotope_key]
    if iso['decay_mode'] != 'alpha':
        return 0.0

    A_parent = iso['A']
    Z_daughter = iso['daughter_Z']

    # Q-value at this σ — DERIVED from Coulomb/strong decomposition
    Q_sigma = alpha_Q_at_sigma(isotope_key, sigma)

    if Q_sigma <= 0:
        return 0.0

    # Alpha mass at σ
    m_alpha = alpha_mass_mev(sigma)

    # Nuclear radius
    R_fm = nuclear_radius_fm(A_parent)
    R_m = R_fm * 1e-15

    # Gamow factor
    G = gamow_factor(Z_daughter, Q_sigma, m_alpha, A_parent)

    if G == float('inf'):
        return 0.0

    # Assault frequency: f = v_α / (2R)
    Q_joule = Q_sigma * _MEV_TO_JOULE
    m_alpha_kg = m_alpha * _MEV_TO_JOULE / C**2
    v_alpha = math.sqrt(2.0 * Q_joule / m_alpha_kg)
    f_assault = v_alpha / (2.0 * R_m)

    # Decay constant
    lam = f_assault * math.exp(-2.0 * G)
    return lam


def alpha_half_life(isotope_key, sigma=SIGMA_HERE):
    """Alpha decay half-life in seconds.

    t½ = ln(2) / λ

    Args:
        isotope_key: key into ISOTOPES dict
        sigma: σ-field value

    Returns:
        Half-life in seconds. Returns inf for non-alpha decayers.
    """
    lam = alpha_decay_constant(isotope_key, sigma)
    if lam <= 0:
        return float('inf')
    return math.log(2) / lam


# ── Beta Decay: Sargent's Rule ────────────────────────────────────

def beta_Q_decomposition(isotope_key):
    """Decompose beta decay Q-value into σ-invariant and σ-dependent parts.

    For β⁻: parent (Z, A) → daughter (Z+1, A) + e⁻ + ν̄

    Q = (m_n − m_p)c² + (BE_daughter − BE_parent)
      = (m_n − m_p)c² + ΔBE

    The nucleon mass difference decomposes:
      (m_n − m_p) = (m_n_bare − m_p_bare) + (m_n_QCD − m_p_QCD) × e^σ
                   = Δm_bare + Δm_QCD × e^σ

    The binding energy difference decomposes via Coulomb:
      ΔBE = ΔBE_strong × e^σ − ΔE_C
      where ΔE_C = E_C(Z+1, A) − E_C(Z, A) (daughter has more Coulomb repulsion)

    So: Q(σ) = [Δm_bare − m_e − ΔE_C]                    [σ-INVARIANT]
             + [Δm_QCD + ΔBE_strong_at_sigma_0] × e^σ     [σ-DEPENDENT]

    DERIVED from coulomb_energy_mev() and nucleon mass decomposition.

    Returns:
        (Q_invariant_MeV, Q_sigma_coefficient_MeV)
        such that Q(σ) = Q_invariant + Q_sigma_coeff × e^σ
    """
    iso = ISOTOPES[isotope_key]

    if isotope_key == 'free_neutron':
        # No nuclear binding — pure nucleon mass decomposition
        # Q = (m_n_bare - m_p_bare - m_e) + (m_n_QCD - m_p_QCD) × e^σ
        # Uses DELTA_NP_* to avoid 939-938 cancellation (saves 3 sig digits)
        Q_invariant = DELTA_NP_BARE_MEV - M_ELECTRON_MEV
        Q_sigma_coeff = DELTA_NP_QCD_MEV
        return (Q_invariant, Q_sigma_coeff)

    Z_p, A_p = iso['Z'], iso['A']
    Z_d, A_d = iso['daughter_Z'], iso['daughter_A']
    Q_measured = iso['Q_value_MeV']

    # Coulomb energy change: daughter has one more proton → more repulsion
    E_C_parent = coulomb_energy_mev(Z_p, A_p)
    E_C_daughter = coulomb_energy_mev(Z_d, A_d)
    delta_E_C = E_C_daughter - E_C_parent  # positive (more repulsion)

    # At σ=0: Q_measured = Q_invariant + Q_sigma_coeff × 1.0
    # Q_invariant = (Δm_bare - m_e) - ΔE_C  (these are the EM parts)
    # Q_sigma_coeff = Δm_QCD + ΔBE_strong_0
    #
    # Since Q_measured = Q_invariant + Q_sigma_coeff:
    #   Q_sigma_coeff = Q_measured - Q_invariant
    #
    # Uses DELTA_NP_BARE_MEV to avoid 939-938 cancellation
    Q_invariant = DELTA_NP_BARE_MEV - M_ELECTRON_MEV - delta_E_C
    Q_sigma_coeff = Q_measured - Q_invariant

    return (Q_invariant, Q_sigma_coeff)


def beta_Q_value_mev(isotope_key, sigma=SIGMA_HERE):
    """Beta decay Q-value at arbitrary σ.

    Q(σ) = Q_invariant + Q_sigma_coeff × e^σ

    Where Q_invariant and Q_sigma_coeff are DERIVED from:
      - Coulomb energy difference (from coulomb_energy_mev)
      - Nucleon bare/QCD mass decomposition (from constants)

    For free neutron: exact tracking of m_n(σ) − m_p(σ) − m_e.
    For complex nuclei: Coulomb/strong decomposition via binding.py.

    σ-dependence: small for light nuclei where Coulomb change is small,
    larger for heavy nuclei where the Coulomb step from Z→Z+1 is significant.

    Returns:
        Q-value in MeV at given σ.
    """
    iso = ISOTOPES[isotope_key]
    if iso['decay_mode'] not in ('beta_minus', 'beta_plus'):
        return 0.0

    Q_invariant, Q_sigma_coeff = beta_Q_decomposition(isotope_key)
    e_sig = scale_ratio(sigma)
    return Q_invariant + Q_sigma_coeff * e_sig


def beta_decay_constant(isotope_key, sigma=SIGMA_HERE):
    """Beta decay constant λ (s⁻¹) from Sargent's rule.

    λ ∝ G_F² × Q⁵

    FIRST_PRINCIPLES: Fermi's golden rule gives transition rate
    proportional to phase space volume × matrix element squared.
    For allowed transitions, the phase space integral gives Q⁵
    dependence (Sargent's rule, 1933).

    APPROXIMATION: We calibrate the proportionality constant from
    the measured half-life at σ=0, then use the Q⁵ scaling to
    predict how λ changes with σ. This avoids needing nuclear
    matrix elements (which require shell-model calculations).

    σ-dependence: Q(σ) shifts → λ shifts as Q(σ)⁵/Q(0)⁵.

    Args:
        isotope_key: key into ISOTOPES dict
        sigma: σ-field value

    Returns:
        Decay constant λ in s⁻¹.
    """
    iso = ISOTOPES[isotope_key]
    if iso['decay_mode'] not in ('beta_minus', 'beta_plus'):
        return 0.0

    Q_0 = iso['Q_value_MeV']
    Q_sigma = beta_Q_value_mev(isotope_key, sigma)

    if Q_0 <= 0 or Q_sigma <= 0:
        return 0.0

    # Calibrate from measured half-life at σ=0
    t_half_0 = iso['half_life_s']
    lambda_0 = math.log(2) / t_half_0

    # Sargent's rule: λ ∝ Q⁵
    return lambda_0 * (Q_sigma / Q_0) ** 5


def beta_half_life(isotope_key, sigma=SIGMA_HERE):
    """Beta decay half-life in seconds.

    t½ = ln(2) / λ

    Returns:
        Half-life in seconds. Returns inf for non-beta decayers.
    """
    lam = beta_decay_constant(isotope_key, sigma)
    if lam <= 0:
        return float('inf')
    return math.log(2) / lam


# ── General Interface ─────────────────────────────────────────────

def decay_constant(isotope_key, sigma=SIGMA_HERE):
    """Decay constant λ (s⁻¹) for any isotope, dispatching to the
    appropriate decay mode.

    Args:
        isotope_key: key into ISOTOPES dict
        sigma: σ-field value

    Returns:
        Decay constant λ in s⁻¹.
    """
    iso = ISOTOPES[isotope_key]
    mode = iso['decay_mode']
    if mode == 'alpha':
        return alpha_decay_constant(isotope_key, sigma)
    elif mode in ('beta_minus', 'beta_plus'):
        return beta_decay_constant(isotope_key, sigma)
    return 0.0


def half_life(isotope_key, sigma=SIGMA_HERE):
    """Half-life in seconds for any isotope.

    t½ = ln(2) / λ
    """
    lam = decay_constant(isotope_key, sigma)
    if lam <= 0:
        return float('inf')
    return math.log(2) / lam


def half_life_human(isotope_key, sigma=SIGMA_HERE):
    """Half-life in human-readable units.

    Returns (value, unit) tuple.
    """
    t = half_life(isotope_key, sigma)
    if t == float('inf'):
        return (float('inf'), 'stable')

    year_s = 365.25 * 86400.0
    if t > 1e9 * year_s:
        return (t / (1e9 * year_s), 'Gyr')
    elif t > year_s:
        return (t / year_s, 'years')
    elif t > 86400:
        return (t / 86400, 'days')
    elif t > 3600:
        return (t / 3600, 'hours')
    elif t > 60:
        return (t / 60, 'minutes')
    elif t > 1:
        return (t, 'seconds')
    elif t > 1e-3:
        return (t * 1e3, 'ms')
    elif t > 1e-6:
        return (t * 1e6, 'μs')
    else:
        return (t * 1e9, 'ns')


def activity_becquerel(isotope_key, n_atoms, sigma=SIGMA_HERE):
    """Radioactive activity in Becquerel (disintegrations/second).

    A = λ × N

    FIRST_PRINCIPLES: activity is the number of decays per unit time.

    Args:
        isotope_key: key into ISOTOPES dict
        n_atoms: number of radioactive atoms
        sigma: σ-field value

    Returns:
        Activity in Bq (s⁻¹).
    """
    lam = decay_constant(isotope_key, sigma)
    return lam * n_atoms


def remaining_fraction(isotope_key, time_s, sigma=SIGMA_HERE):
    """Fraction of atoms remaining after time t.

    N(t)/N₀ = exp(−λt)

    FIRST_PRINCIPLES: exponential decay law. Each atom has a constant
    probability of decaying per unit time. This gives exponential
    decay for the ensemble.

    Args:
        isotope_key: key into ISOTOPES dict
        time_s: time elapsed in seconds
        sigma: σ-field value

    Returns:
        Fraction remaining (0 to 1).
    """
    lam = decay_constant(isotope_key, sigma)
    return math.exp(-lam * time_s)


# ── Geiger-Nuttall Relation (Validation) ─────────────────────────

def geiger_nuttall_check():
    """Verify our Gamow theory reproduces the Geiger-Nuttall relation.

    The empirical Geiger-Nuttall law (1911):
      log(λ) = a + b/√(Q)

    This falls out naturally from the Gamow theory:
      λ = f × exp(−2G), and G ∝ 1/√Q for fixed Z.

    Returns a list of (isotope, log10_lambda_predicted, log10_lambda_measured)
    for all alpha emitters in our database.
    """
    results = []
    for key, iso in ISOTOPES.items():
        if iso['decay_mode'] != 'alpha':
            continue
        lam_pred = alpha_decay_constant(key, sigma=SIGMA_HERE)
        lam_meas = math.log(2) / iso['half_life_s']
        if lam_pred > 0 and lam_meas > 0:
            results.append({
                'isotope': key,
                'Q_MeV': iso['Q_value_MeV'],
                'log10_lambda_predicted': math.log10(lam_pred),
                'log10_lambda_measured': math.log10(lam_meas),
                'log10_ratio': math.log10(lam_pred / lam_meas),
            })
    return results


# ── Nagatha Export ────────────────────────────────────────────────

def isotope_decay_properties(isotope_key, sigma=SIGMA_HERE):
    """Export decay properties in Nagatha-compatible format.

    Returns a dict with all decay quantities and honest origin tags.
    """
    iso = ISOTOPES[isotope_key]
    mode = iso['decay_mode']

    t_half = half_life(isotope_key, sigma)
    t_half_0 = iso['half_life_s']
    t_val, t_unit = half_life_human(isotope_key, sigma)

    result = {
        'isotope': isotope_key,
        'name': iso['name'],
        'Z': iso['Z'], 'A': iso['A'],
        'decay_mode': mode,
        'sigma': sigma,
        'Q_value_MeV': iso['Q_value_MeV'],
        'half_life_s': t_half,
        'half_life_measured_s': t_half_0,
        'half_life_human': f"{t_val:.3g} {t_unit}",
        'decay_constant_per_s': decay_constant(isotope_key, sigma),
        'daughter': iso['daughter_name'],
    }

    if mode == 'alpha':
        result['gamow_factor'] = gamow_factor(
            iso['daughter_Z'], iso['Q_value_MeV'],
            alpha_mass_mev(sigma), iso['A'])
        result['origin'] = (
            "Gamow tunneling: FIRST_PRINCIPLES (WKB + Coulomb barrier). "
            "Assault frequency: FIRST_PRINCIPLES (v_α/2R). "
            "Nuclear radius: FIRST_PRINCIPLES (r₀A^(1/3), r₀ MEASURED). "
            "Q-value: MEASURED. "
            "σ-dependence: CORE (m_α and Q shift through QCD mass)."
        )
    elif mode in ('beta_minus', 'beta_plus'):
        result['Q_value_at_sigma_MeV'] = beta_Q_value_mev(isotope_key, sigma)
        result['origin'] = (
            "Sargent's rule: FIRST_PRINCIPLES (Fermi golden rule, Q⁵ scaling) + "
            "APPROXIMATION (leading-order, no Coulomb correction). "
            "Calibration: MEASURED (half-life at σ=0). "
            "Q-value: MEASURED. "
            "σ-dependence: CORE (Q shifts through nucleon mass difference)."
        )
    else:
        result['origin'] = 'Stable isotope.'

    return result
