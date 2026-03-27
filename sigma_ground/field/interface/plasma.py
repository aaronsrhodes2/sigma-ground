"""
Plasma Physics — σ-grounded plasma parameters.

Key dimensionless and dimensional quantities for plasma physics, built
on sigma_ground.field.constants.  No external dependencies.

References:
  Goldston & Rutherford, "Introduction to Plasma Physics" (IOP, 1995)
  Bittencourt, "Fundamentals of Plasma Physics" 3rd ed. (Springer, 2004)
  NRL Plasma Formulary (Naval Research Laboratory, 2019)

σ-connection
------------
At the σ-transition, the effective electron mass (and hence QCD-scale
nuclear mass) shifts.  sigma_plasma_transition() gives the modified
plasma frequency at a given σ-field value.

Dependency: E_CHARGE, EPS_0, MU_0, M_ELECTRON_KG, K_B from constants
"""

import math

from sigma_ground.field.constants import (
    E_CHARGE, EPS_0, MU_0, M_ELECTRON_KG, K_B, C,
)
from sigma_ground.field.scale import scale_ratio


# ── Plasma Frequency ───────────────────────────────────────────────────

def plasma_frequency(n_e):
    """Angular plasma frequency ω_p = √(n_e e² / ε₀ m_e).

    The natural oscillation frequency of free electrons.  EM waves
    with ω < ω_p cannot propagate (the plasma is opaque).

    Args:
        n_e: free electron number density (m⁻³), must be > 0

    Returns:
        ω_p (rad/s)
    """
    if n_e <= 0:
        raise ValueError(f"n_e={n_e} ≤ 0: electron density must be positive")
    return math.sqrt(n_e * E_CHARGE**2 / (EPS_0 * M_ELECTRON_KG))


def plasma_frequency_hz(n_e):
    """Plasma frequency f_p = ω_p / (2π) in Hz.

    Args:
        n_e: free electron number density (m⁻³)

    Returns:
        f_p (Hz)
    """
    return plasma_frequency(n_e) / (2.0 * math.pi)


# ── Debye Length ───────────────────────────────────────────────────────

def debye_length(n_e, T_e):
    """Debye screening length λ_D = √(ε₀ k_B T_e / n_e e²).

    The characteristic scale over which charge is screened in a plasma.

    A valid plasma requires the Debye number N_D = n_e × (4π/3) λ_D³ ≫ 1
    (many particles in the Debye sphere).

    Reference: Goldston & Rutherford §1.2

    Args:
        n_e: electron number density (m⁻³), must be > 0
        T_e: electron temperature (K), must be > 0

    Returns:
        λ_D (m)
    """
    if n_e <= 0:
        raise ValueError(f"n_e={n_e} ≤ 0: density must be positive")
    if T_e <= 0:
        raise ValueError(f"T_e={T_e} K ≤ 0: temperature must be positive")
    return math.sqrt(EPS_0 * K_B * T_e / (n_e * E_CHARGE**2))


def debye_number(n_e, T_e):
    """Number of particles in a Debye sphere N_D = (4π/3) n_e λ_D³.

    Validity criterion: N_D ≫ 1 (collective behaviour dominates).
    N_D < 1: strongly coupled plasma (not described by this module).

    Args:
        n_e: electron number density (m⁻³)
        T_e: electron temperature (K)

    Returns:
        N_D (dimensionless)
    """
    lam = debye_length(n_e, T_e)
    return (4.0 / 3.0) * math.pi * n_e * lam**3


# ── Magnetic Field Parameters ──────────────────────────────────────────

def alfven_speed(B, rho):
    """Alfvén wave speed v_A = B / √(μ₀ ρ).

    The speed at which magnetic tension propagates along field lines.

    Args:
        B: magnetic field strength (T)
        rho: mass density (kg/m³), must be > 0

    Returns:
        v_A (m/s)
    """
    if rho <= 0:
        raise ValueError(f"rho={rho} ≤ 0: density must be positive")
    return B / math.sqrt(MU_0 * rho)


def cyclotron_radius(m, v_perp, B):
    """Larmor (cyclotron) radius r_c = mv_⊥ / |q|B.

    Radius of circular motion of a charged particle in a magnetic field.

    Args:
        m: particle mass (kg)
        v_perp: speed component perpendicular to B (m/s)
        B: magnetic field strength (T), must be > 0

    Returns:
        r_c (m)
    """
    if B <= 0:
        raise ValueError(f"B={B} ≤ 0: magnetic field must be positive")
    return m * v_perp / (E_CHARGE * B)


def plasma_beta(n, T, B):
    """Plasma beta β = nkT / (B²/2μ₀).

    Ratio of thermal pressure to magnetic pressure.
    β ≫ 1: thermally dominated (fluid-like).
    β ≪ 1: magnetically dominated.

    Args:
        n: particle number density (m⁻³)
        T: temperature (K)
        B: magnetic field strength (T)

    Returns:
        β (dimensionless)
    """
    p_thermal = n * K_B * T
    p_magnetic = B**2 / (2.0 * MU_0)
    return p_thermal / p_magnetic


# ── Resistivity ────────────────────────────────────────────────────────

def coulomb_logarithm(n_e, T_e):
    """Coulomb logarithm ln Λ ≈ ln(λ_D / b_min).

    b_min = max(b_classical, b_quantum)
    b_classical = k_e e² / (3/2 kT) = e²/(4πε₀ × 3/2 kT) (classical closest approach)
    b_quantum  = ℏ / (m_e v_th)     (de Broglie wavelength)

    Reference: NRL Plasma Formulary 2019, §§ "Frequencies and lengths"

    Args:
        n_e: electron number density (m⁻³)
        T_e: electron temperature (K)

    Returns:
        ln Λ (dimensionless, typically 10–20 for lab plasmas)
    """
    lam_D = debye_length(n_e, T_e)
    # Classical closest approach for 90° scattering
    b_classical = E_CHARGE**2 / (4.0 * math.pi * EPS_0 * (1.5 * K_B * T_e))
    # Quantum de Broglie limit (thermal velocity)
    from sigma_ground.field.constants import HBAR
    v_th = math.sqrt(2.0 * K_B * T_e / M_ELECTRON_KG)
    b_quantum = HBAR / (M_ELECTRON_KG * v_th)
    b_min = max(b_classical, b_quantum)
    ratio = lam_D / b_min
    if ratio <= 1.0:
        return 1.0  # degenerate / strongly coupled — ln Λ → 0, clamp at 1
    return math.log(ratio)


def spitzer_resistivity(T_e, Z_eff=1.0):
    """Spitzer electrical resistivity η_S ∝ Z_eff ln Λ / T_e^(3/2).

    η_S = (π m_e)^(1/2) Z_eff e² ln Λ / (3 (2 k_B T_e)^(3/2) ε₀²)

    Reference: Spitzer 1962; NRL Plasma Formulary

    Note: this uses a reference electron density of n_e = 1e18 m⁻³
    for computing ln Λ.  Pass n_e explicitly if you need a more accurate
    Coulomb logarithm.

    Args:
        T_e: electron temperature (K), must be > 0
        Z_eff: effective charge state of ions (default 1.0 for hydrogen)

    Returns:
        η_S (Ω·m)
    """
    if T_e <= 0:
        raise ValueError(f"T_e={T_e} K ≤ 0: temperature must be positive")
    from sigma_ground.field.constants import HBAR
    lnL = coulomb_logarithm(1e18, T_e)  # reference density
    numerator = math.sqrt(math.pi * M_ELECTRON_KG) * Z_eff * E_CHARGE**2 * lnL
    denominator = 3.0 * (2.0 * K_B * T_e) ** 1.5 * EPS_0**2
    # Dimensional prefactor: SI units
    # η has units Ω·m = kg·m³/(A²·s³)
    # The standard form: η = (m_e)^{1/2} Z e² ln Λ / [3 (2πε₀)^{1/2} (kT)^{3/2}]
    # Using the NRL form corrected to SI:
    return numerator / denominator * (1.0 / (4.0 * math.pi * EPS_0))


# ── σ-Connection ───────────────────────────────────────────────────────

def sigma_plasma_transition(sigma):
    """Effective plasma frequency at a σ-field value.

    The electron mass is Higgs-origin and σ-invariant, but nucleon masses
    scale with Λ_QCD × e^σ.  In a fully ionized plasma at σ-transition,
    the ion contribution to the total plasma frequency shifts.

    For a pure electron plasma the effect is indirect: the σ-field
    compresses the pocket and raises the ambient density:
        n_eff = n₀ × e^(3σ)   (density scales as inverse volume ~ e^(3σ))

    Therefore:  ω_p,eff = ω_p × e^(3σ/2)

    Args:
        sigma: σ-field value (dimensionless, ≥ 0)

    Returns:
        Callable that maps n_e → effective ω_p at this σ.
        Or, equivalently, the scale factor e^(3σ/2) to apply to ω_p(n_e).
    """
    # Return the scale factor — caller applies it to plasma_frequency(n_e)
    return scale_ratio(1.5 * sigma)
