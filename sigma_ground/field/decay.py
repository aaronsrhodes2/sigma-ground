"""
Nuclear Decay — σ-grounded radioactive decay physics.

Radioactive decay rates, Q-values, and Gamow tunneling built on
sigma_ground.field.constants.  Q-value functions accept masses in
atomic mass units (u) or MeV/c² — caller's choice via the mass_in_mev
flag.

References:
  Krane, "Introductory Nuclear Physics" (Wiley, 1988)
  PDG Review: "Radioactive Decay" (pdg.lbl.gov)
  Geiger & Nuttall 1911; Gamow 1928

σ-connection
------------
At the σ-transition threshold, nuclear bond failure layers alter
effective binding energies and therefore decay rates.  See
sigma_decay_shift().

Dependency: HBAR, C, E_CHARGE, ALPHA from constants; scale.py for σ
"""

import math

from .constants import HBAR, C, E_CHARGE, ALPHA, AMU_KG
from .scale import scale_ratio

# Conversion factor: 1 atomic mass unit in kg
_AMU_KG = AMU_KG
_AMU_MEV = 931.494102         # MeV/c² per u  (CODATA 2018)
_LN2 = math.log(2.0)


# ── Basic Decay Rates ──────────────────────────────────────────────────

def decay_constant(half_life_s):
    """Decay constant λ = ln(2) / t½.

    Args:
        half_life_s: half-life (s), must be > 0

    Returns:
        decay constant λ (s⁻¹)
    """
    if half_life_s <= 0:
        raise ValueError(f"half_life_s={half_life_s} ≤ 0: must be positive")
    return _LN2 / half_life_s


def half_life(lambda_s):
    """Half-life t½ = ln(2) / λ.

    Args:
        lambda_s: decay constant (s⁻¹), must be > 0

    Returns:
        half-life (s)
    """
    if lambda_s <= 0:
        raise ValueError(f"lambda_s={lambda_s} ≤ 0: must be positive")
    return _LN2 / lambda_s


def activity(N, half_life_s):
    """Activity A = λN (Becquerel).

    Args:
        N: number of radioactive nuclei
        half_life_s: half-life (s), must be > 0

    Returns:
        activity (Bq = decays/s)
    """
    return decay_constant(half_life_s) * N


def remaining_nuclei(N0, t, half_life_s):
    """Remaining nuclei after time t: N(t) = N₀ × e^(−λt).

    Args:
        N0: initial number of nuclei
        t: elapsed time (s)
        half_life_s: half-life (s), must be > 0

    Returns:
        N(t) (may be fractional for continuous approximation)
    """
    lam = decay_constant(half_life_s)
    return N0 * math.exp(-lam * t)


# ── Q-Values ───────────────────────────────────────────────────────────

def q_value_mev(M_parent_mev, M_products_mev):
    """Generic Q-value from mass excess: Q = (M_parent − ΣM_products) × c².

    Masses in MeV/c² (rest mass energy in MeV).

    Q > 0: exothermic (energy released — decay is spontaneous).
    Q < 0: endothermic (energy required — decay is forbidden at rest).

    Args:
        M_parent_mev: parent rest mass (MeV/c²)
        M_products_mev: iterable of product rest masses (MeV/c²)

    Returns:
        Q (MeV)
    """
    return M_parent_mev - sum(M_products_mev)


def q_value_alpha(M_parent_mev, M_daughter_mev, M_alpha_mev=None):
    """Q-value for alpha decay: Q = (M_P − M_D − M_α) × c² in MeV.

    Standard alpha mass: M_α = 3727.379 MeV/c² (He-4 atomic mass).
    Caller may supply their own M_alpha_mev for non-standard particles.

    Args:
        M_parent_mev: parent atomic rest mass (MeV/c²)
        M_daughter_mev: daughter atomic rest mass (MeV/c²)
        M_alpha_mev: alpha particle rest mass (MeV/c²).
                     Default: He-4 atomic mass = 3727.379 MeV/c²

    Returns:
        Q_alpha (MeV)
    """
    if M_alpha_mev is None:
        M_alpha_mev = 3727.379  # He-4 atomic mass in MeV/c² (PDG 2022)
    return M_parent_mev - M_daughter_mev - M_alpha_mev


def q_value_beta_minus(M_parent_mev, M_daughter_mev):
    """Q-value for β⁻ decay: Q = (M_P − M_D) × c² in MeV.

    Using atomic masses: electron masses cancel (parent has Z electrons,
    daughter has Z+1 → beta electron comes from atom's electrons in
    the atomic mass formulation).

    Args:
        M_parent_mev: parent atomic rest mass (MeV/c²)
        M_daughter_mev: daughter atomic rest mass (MeV/c²)

    Returns:
        Q_beta (MeV)
    """
    return M_parent_mev - M_daughter_mev


def q_value_beta_plus(M_parent_mev, M_daughter_mev, M_ELECTRON_MEV=0.51100):
    """Q-value for β⁺ decay: Q = (M_P − M_D − 2m_e) × c² in MeV.

    Using atomic masses: positron emission costs 2m_e (one from the
    emitted positron, one because daughter has one fewer electron).

    Args:
        M_parent_mev: parent atomic rest mass (MeV/c²)
        M_daughter_mev: daughter atomic rest mass (MeV/c²)
        M_ELECTRON_MEV: electron rest mass (MeV). Default: 0.51100 MeV.

    Returns:
        Q_beta_plus (MeV)
    """
    return M_parent_mev - M_daughter_mev - 2.0 * M_ELECTRON_MEV


# ── Gamow Tunneling ────────────────────────────────────────────────────

def gamow_factor(Z_daughter, Z_alpha, Q_MeV, A_daughter):
    """Gamow factor G for alpha decay tunneling.

    G = π Z_d Z_α α / β_α

    where β_α = v_α/c = √(2 Q / M_α c²) is the alpha velocity at
    infinity (classical kinematic energy at Q).

    The Gamow tunneling probability: T ~ e^(−2G).

    Reference: Gamow 1928, Z. Phys. 51, 204; Krane §6.3

    Args:
        Z_daughter: proton number of daughter nucleus
        Z_alpha: proton number of alpha particle (= 2)
        Q_MeV: Q-value of the decay (MeV), must be > 0
        A_daughter: mass number of daughter

    Returns:
        Gamow factor G (dimensionless)
    """
    if Q_MeV <= 0:
        raise ValueError(f"Q_MeV={Q_MeV} ≤ 0: alpha decay requires Q > 0")
    # Alpha kinetic energy at infinity = Q_MeV (energy conservation)
    # M_alpha * c² ≈ 3727.379 MeV
    M_alpha_mev = 3727.379
    # β_α = v/c = √(2 Q / M_α c²)
    beta_alpha = math.sqrt(2.0 * Q_MeV / M_alpha_mev)
    return math.pi * Z_daughter * Z_alpha * ALPHA / beta_alpha


def alpha_decay_rate_geiger_nuttall(Z, A, Q_MeV, r0_fm=1.215):
    """Geiger-Nuttall estimate of alpha decay rate.

    Uses the Gamow model: λ ≈ f₀ × e^(−2G)

    where f₀ is the assault frequency (alpha bouncing inside the nucleus)
    and G is the Gamow factor.

    Reference: Geiger & Nuttall 1911; Krane §6.3

    Args:
        Z: proton number of parent nucleus
        A: mass number of parent
        Q_MeV: Q-value (MeV), must be > 0
        r0_fm: nuclear radius parameter (fm). Default: 1.215 fm.

    Returns:
        estimated decay rate λ (s⁻¹)
    """
    Z_daughter = Z - 2
    # Nuclear radius in metres
    R_nuc = r0_fm * 1e-15 * A**(1.0/3.0)
    # Alpha velocity inside the nucleus (rough estimate from Q + barrier)
    M_alpha_mev = 3727.379
    v_alpha = math.sqrt(2.0 * Q_MeV * 1e6 * E_CHARGE / (M_alpha_mev * 1e6 * E_CHARGE / C**2))
    # Assault frequency: f₀ = v_α / (2 R_nuc)
    f0 = v_alpha / (2.0 * R_nuc)
    # Gamow factor for tunneling
    G = gamow_factor(Z_daughter, 2, Q_MeV, A - 4)
    return f0 * math.exp(-2.0 * G)


# ── σ-Connection ───────────────────────────────────────────────────────

def sigma_decay_shift(sigma, lambda_0):
    """Modified decay constant in σ-compressed spacetime.

    At the σ-transition, nuclear bond failure layers alter the effective
    binding energies, changing the potential barrier height for tunneling.
    The Gamow factor G depends on Λ_QCD (through r₀ scaling), so the
    decay rate changes as:

        λ_eff(σ) = λ₀ × e^σ

    Physical interpretation: in compressed spacetime (σ > 0), the nuclear
    radius shrinks and the potential barrier shortens — tunneling is
    easier and decay happens faster.

    At σ = 0: λ_eff = λ₀ (standard decay rate).
    At σ_conv ≈ 1.849: λ_eff ≈ 6.35 × λ₀.

    Uses scale_ratio(σ) = e^σ from sigma_ground.field.scale.

    Args:
        sigma: σ-field value (dimensionless, ≥ 0)
        lambda_0: standard decay constant (s⁻¹)

    Returns:
        effective decay constant (s⁻¹)
    """
    return lambda_0 * scale_ratio(sigma)
