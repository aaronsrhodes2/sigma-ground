"""
Statistical Mechanics — σ-grounded thermodynamic distributions.

Classical and quantum statistical mechanics built on the measured
constants K_B and HBAR from sigma_ground.field.constants.

References:
  Reif, "Fundamentals of Statistical and Thermal Physics" (McGraw-Hill, 1965)
  Pathria & Beale, "Statistical Mechanics" 3rd ed. (Elsevier, 2011)

σ-connection
------------
Inside a σ-compressed spacetime pocket, the effective thermal energy
scale shifts with the QCD energy scale Λ_eff = Λ_QCD × e^σ.
sigma_partition_shift() gives the modified temperature seen by
infalling matter approaching a σ-transition.

Dependency: K_B, HBAR from constants; scale_ratio from field.scale
"""

import math

from sigma_ground.field.constants import K_B, HBAR
from sigma_ground.field.scale import scale_ratio


# ── Boltzmann Distributions ────────────────────────────────────────────

def boltzmann_factor(E_j, T):
    """Boltzmann factor e^(−E_j / kT).

    Relative probability of a microstate with energy E_j at temperature T.

    Args:
        E_j: energy of the state (J)
        T: temperature (K), must be > 0

    Returns:
        Boltzmann factor (dimensionless, 0 < result ≤ 1)
    """
    if T <= 0:
        raise ValueError(f"T={T} K ≤ 0: temperature must be positive")
    return math.exp(-E_j / (K_B * T))


def partition_function(energies, T):
    """Canonical partition function Z = Σ_i e^(−ε_i / kT).

    Args:
        energies: iterable of state energies (J)
        T: temperature (K), must be > 0

    Returns:
        Z (dimensionless, ≥ 1)
    """
    if T <= 0:
        raise ValueError(f"T={T} K ≤ 0: temperature must be positive")
    beta = 1.0 / (K_B * T)
    return sum(math.exp(-e * beta) for e in energies)


def mean_energy(energies, T):
    """Mean energy ⟨E⟩ = (1/Z) Σ_i ε_i e^(−ε_i / kT).

    Equivalent to −∂lnZ/∂β where β = 1/(kT).

    Args:
        energies: iterable of state energies (J)
        T: temperature (K), must be > 0

    Returns:
        mean energy (J)
    """
    if T <= 0:
        raise ValueError(f"T={T} K ≤ 0: temperature must be positive")
    beta = 1.0 / (K_B * T)
    weights = [math.exp(-e * beta) for e in energies]
    Z = sum(weights)
    energies_list = list(energies) if not isinstance(energies, list) else energies
    # Re-compute with list for indexing
    energies_list = list(energies) if hasattr(energies, '__iter__') else energies
    weights2 = [math.exp(-e * beta) for e in energies_list]
    Z2 = sum(weights2)
    return sum(e * w for e, w in zip(energies_list, weights2)) / Z2


def entropy_from_partition(Z, T, mean_E):
    """Thermodynamic entropy from partition function.

    S = k_B (ln Z + β ⟨E⟩)

    Args:
        Z: partition function (dimensionless)
        T: temperature (K), must be > 0
        mean_E: mean energy ⟨E⟩ (J)

    Returns:
        entropy (J/K)
    """
    if T <= 0:
        raise ValueError(f"T={T} K ≤ 0: temperature must be positive")
    beta = 1.0 / (K_B * T)
    return K_B * (math.log(Z) + beta * mean_E)


def boltzmann_entropy(W):
    """Boltzmann entropy S = k_B ln W.

    W is the number of microstates (statistical weight).

    Args:
        W: number of microstates (dimensionless, positive integer)

    Returns:
        entropy (J/K)
    """
    if W <= 0:
        raise ValueError(f"W={W} ≤ 0: number of microstates must be positive")
    return K_B * math.log(W)


# ── Quantum Distributions ──────────────────────────────────────────────

def fermi_dirac(E, E_fermi, T):
    """Fermi-Dirac distribution f(E) = 1 / (e^((E−μ)/kT) + 1).

    Occupation probability for a fermionic state at energy E.

    At T = 0: step function — 1 below E_fermi, 0 above.
    At E = E_fermi: always 0.5 for T > 0.

    Args:
        E: energy of the state (J)
        E_fermi: Fermi energy / chemical potential μ (J)
        T: temperature (K).  Use a small positive value instead of 0
           (exact T=0 is handled numerically as a step function).

    Returns:
        occupation probability (0 ≤ f ≤ 1)
    """
    if T <= 0:
        # T=0 limit: exact step function
        if E < E_fermi:
            return 1.0
        elif E > E_fermi:
            return 0.0
        else:
            return 0.5
    x = (E - E_fermi) / (K_B * T)
    # Guard against overflow in exp(x) for large positive x
    if x > 700:
        return 0.0
    if x < -700:
        return 1.0
    return 1.0 / (math.exp(x) + 1.0)


def bose_einstein(E, mu, T):
    """Bose-Einstein distribution f(E) = 1 / (e^((E−μ)/kT) − 1).

    Occupation number for a bosonic mode at energy E.
    Diverges when E → μ (Bose-Einstein condensate onset).

    Args:
        E: energy of the mode (J)
        mu: chemical potential (J), must satisfy mu < E for bosons
        T: temperature (K), must be > 0

    Returns:
        occupation number (≥ 0)

    Raises:
        ValueError: if E ≤ mu (unphysical for bosons at T > 0)
    """
    if T <= 0:
        raise ValueError(f"T={T} K ≤ 0: temperature must be positive")
    if E <= mu:
        raise ValueError(f"E={E} ≤ mu={mu}: Bose-Einstein requires E > μ")
    x = (E - mu) / (K_B * T)
    return 1.0 / (math.exp(x) - 1.0)


# ── Maxwell-Boltzmann Speed Distribution ──────────────────────────────

def maxwell_speed_dist(m, v, T):
    """Maxwell-Boltzmann speed distribution f(v).

    f(v) = 4π n (m/2πkT)^(3/2) v² e^(−mv²/2kT)

    Probability density per unit speed (per m/s), normalised to integrate
    to n (number density). Pass n=1 for the normalised probability density.

    Args:
        m: particle mass (kg)
        v: speed (m/s), must be ≥ 0
        T: temperature (K), must be > 0

    Returns:
        f(v) (m⁻¹ s, probability density per unit speed for n=1)
    """
    if T <= 0:
        raise ValueError(f"T={T} K ≤ 0: temperature must be positive")
    if v < 0:
        raise ValueError(f"v={v} < 0: speed must be non-negative")
    a = m / (2.0 * K_B * T)
    return 4.0 * math.pi * (a / math.pi) ** 1.5 * v**2 * math.exp(-a * v**2)


def rms_speed(m, T):
    """Root-mean-square speed v_rms = √(3kT/m).

    Args:
        m: particle mass (kg)
        T: temperature (K), must be > 0

    Returns:
        v_rms (m/s)
    """
    if T <= 0:
        raise ValueError(f"T={T} K ≤ 0: temperature must be positive")
    return math.sqrt(3.0 * K_B * T / m)


def mean_speed(m, T):
    """Mean speed ⟨v⟩ = √(8kT/πm).

    Args:
        m: particle mass (kg)
        T: temperature (K), must be > 0

    Returns:
        ⟨v⟩ (m/s)
    """
    if T <= 0:
        raise ValueError(f"T={T} K ≤ 0: temperature must be positive")
    return math.sqrt(8.0 * K_B * T / (math.pi * m))


def most_probable_speed(m, T):
    """Most probable speed v_p = √(2kT/m).

    Args:
        m: particle mass (kg)
        T: temperature (K), must be > 0

    Returns:
        v_p (m/s)
    """
    if T <= 0:
        raise ValueError(f"T={T} K ≤ 0: temperature must be positive")
    return math.sqrt(2.0 * K_B * T / m)


# ── Thermodynamic Quantities ───────────────────────────────────────────

def heat_capacity_equipartition(dof, n=1):
    """Classical heat capacity from equipartition theorem.

    C = ½ × dof × n × k_B

    Each degree of freedom contributes ½k_B per particle.

    Args:
        dof: number of degrees of freedom per particle
             (monatomic gas: 3; diatomic gas: 5; solid: 6)
        n: number of particles (default 1 → per-particle capacity)

    Returns:
        heat capacity C (J/K)
    """
    return 0.5 * dof * n * K_B


# ── σ-Connection ───────────────────────────────────────────────────────

def sigma_partition_shift(sigma, T):
    """Effective temperature in a σ-compressed spacetime pocket.

    The QCD energy scale sets the natural energy unit: Λ_eff = Λ_QCD × e^σ.
    Inside a pocket with elevated σ, the effective thermal energy that a
    particle 'experiences' relative to QCD binding energies shifts.

    T_eff(σ) = T × e^σ

    Interpretation: the same physical temperature T corresponds to a
    higher fraction of the binding energy in compressed spacetime, so
    the system is effectively 'hotter' relative to nuclear processes.

    At σ = 0: T_eff = T (standard physics).
    At σ_conv ≈ 1.849: T_eff ≈ 6.35 T.

    Uses scale_ratio(σ) = e^σ from sigma_ground.field.scale.

    Args:
        sigma: σ-field value (dimensionless, ≥ 0)
        T: temperature (K)

    Returns:
        effective temperature (K)
    """
    return T * scale_ratio(sigma)
