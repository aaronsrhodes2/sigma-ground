"""
Electrodynamics — σ-grounded classical EM.

Maxwell electrodynamics with every constant derived from
sigma_ground.field.constants.  Scalar and vector quantities are supported;
for vector operations, 3-tuples (x, y, z) are used to keep this module
free of circular imports.

References:
  Jackson, "Classical Electrodynamics" 3rd ed. (Wiley, 1999)
  Griffiths, "Introduction to Electrodynamics" 4th ed. (Cambridge, 2017)
  NIST CODATA 2018

σ-connection
------------
At the σ-transition threshold, the effective fine structure constant shifts
because the QCD scale (and hence the energy unit) changes.  See
sigma_em_coupling().

Dependency: E_CHARGE, EPS_0, C, HBAR, MU_0, M_ELECTRON_KG, ALPHA, XI
"""

import math

from .constants import (
    E_CHARGE, EPS_0, C, HBAR, MU_0, M_ELECTRON_KG, ALPHA, XI,
)
from .scale import scale_ratio

# ── Scalar Coulomb / Electric ──────────────────────────────────────────

# Coulomb's constant  k_e = 1/(4πε₀)
_K_E = 1.0 / (4.0 * math.pi * EPS_0)


def coulomb_force(q1, q2, r):
    """Coulomb force magnitude F = k_e q₁q₂/r².

    Positive result: repulsive.  Negative: attractive.

    Args:
        q1, q2: charges (C)
        r: separation (m), must be > 0

    Returns:
        force (N), signed
    """
    if r <= 0:
        raise ValueError(f"r={r} ≤ 0: separation must be positive")
    return _K_E * q1 * q2 / r**2


def electric_field_point(q, r):
    """Electric field magnitude from a point charge: E = k_e q/r².

    Positive: field points away from charge (positive charge).
    Negative: field points toward charge (negative charge).

    Args:
        q: charge (C)
        r: distance from charge (m), must be > 0

    Returns:
        E field magnitude (V/m), signed
    """
    if r <= 0:
        raise ValueError(f"r={r} ≤ 0: distance must be positive")
    return _K_E * q / r**2


def electric_potential(q, r):
    """Electric potential V = k_e q/r.

    Args:
        q: charge (C)
        r: distance from charge (m), must be > 0

    Returns:
        potential (V), signed
    """
    if r <= 0:
        raise ValueError(f"r={r} ≤ 0: distance must be positive")
    return _K_E * q / r


# ── Vector Lorentz Force ───────────────────────────────────────────────

def _cross(a, b):
    """3D cross product of tuples (ax, ay, az) × (bx, by, bz)."""
    ax, ay, az = a
    bx, by, bz = b
    return (
        ay * bz - az * by,
        az * bx - ax * bz,
        ax * by - ay * bx,
    )


def _add(a, b):
    """Element-wise sum of two 3-tuples."""
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale_vec(s, a):
    """Scalar × 3-tuple."""
    return (s * a[0], s * a[1], s * a[2])


def magnetic_force(q, v_vec, B_vec):
    """Magnetic force F = q(v × B).

    Args:
        q: charge (C)
        v_vec: velocity 3-tuple (m/s)
        B_vec: magnetic field 3-tuple (T)

    Returns:
        force 3-tuple (N)
    """
    return _scale_vec(q, _cross(v_vec, B_vec))


def lorentz_force(q, E_vec, v_vec, B_vec):
    """Full Lorentz force F = q(E + v × B).

    Args:
        q: charge (C)
        E_vec: electric field 3-tuple (V/m)
        v_vec: velocity 3-tuple (m/s)
        B_vec: magnetic field 3-tuple (T)

    Returns:
        force 3-tuple (N)
    """
    mag = magnetic_force(q, v_vec, B_vec)
    return _scale_vec(q, _add(E_vec, mag))


# ── Radiation ──────────────────────────────────────────────────────────

def radiation_power_larmor(q, a):
    """Larmor formula: power radiated by an accelerating point charge.

    P = q²a² / (6πε₀c³)

    Valid in the non-relativistic limit (v ≪ c).
    For relativistic charges use the Liénard generalisation.

    Reference: Jackson §14.2

    Args:
        q: charge (C)
        a: acceleration magnitude (m/s²)

    Returns:
        radiated power (W)
    """
    return (q**2 * a**2) / (6.0 * math.pi * EPS_0 * C**3)


# ── EM Wave Energetics ─────────────────────────────────────────────────

def em_wave_energy_density(E_amp, time_average=True):
    """Energy density in an EM wave.

    Instantaneous: u = ε₀E²
    Time-averaged:  ⟨u⟩ = ½ε₀E₀²

    (Magnetic contribution equals the electric, so total instantaneous
     u_total = ε₀E², time-average ⟨u_total⟩ = ½ε₀E₀².)

    Args:
        E_amp: electric field amplitude E₀ (V/m)
        time_average: if True, return time-averaged ½ε₀E₀² (default True)

    Returns:
        energy density (J/m³)
    """
    u = EPS_0 * E_amp**2
    return 0.5 * u if time_average else u


def em_wave_intensity(E_amp):
    """Time-averaged intensity (Poynting flux) of a plane EM wave.

    I = ½ε₀c E₀²

    Reference: Griffiths §9.2

    Args:
        E_amp: electric field amplitude E₀ (V/m)

    Returns:
        intensity (W/m²)
    """
    return 0.5 * EPS_0 * C * E_amp**2


# ── Cyclotron / Plasma ─────────────────────────────────────────────────

def cyclotron_frequency(q, m, B):
    """Cyclotron (gyro) frequency ω_c = |q|B/m.

    The angular frequency at which a charged particle spirals in a
    uniform magnetic field.

    Args:
        q: charge magnitude (C)
        m: particle mass (kg)
        B: magnetic field strength (T)

    Returns:
        ω_c (rad/s)
    """
    return abs(q) * B / m


def skin_depth(n_e, omega=None):
    """EM skin depth δ = c/ω_p.

    The depth to which an EM wave penetrates a plasma or conductor.
    Uses the free-electron plasma frequency ω_p = √(n_e e²/(ε₀ m_e)).

    Args:
        n_e: free electron number density (m⁻³)
        omega: angular frequency (rad/s). If None, uses ω_p itself
               (i.e., returns the collisionless skin depth c/ω_p).

    Returns:
        skin depth (m)
    """
    omega_p = math.sqrt(n_e * E_CHARGE**2 / (EPS_0 * M_ELECTRON_KG))
    if omega is None:
        omega = omega_p
    if omega == 0:
        raise ValueError("omega=0: skin depth undefined at DC")
    return C / omega_p  # collisionless skin depth; omega arg reserved for future use


# ── Fundamental EM Constant ────────────────────────────────────────────

def fine_structure_constant():
    """Fine structure constant α = e²/(4πε₀ℏc).

    This function re-derives α from measured constants so tests can
    verify the derivation is internally consistent.  The module-level
    ALPHA constant should match to machine precision.

    Reference: NIST CODATA 2018, α = 7.2973525693×10⁻³ ≈ 1/137.036

    Returns:
        α (dimensionless) ≈ 7.297e-3
    """
    return E_CHARGE**2 / (4.0 * math.pi * EPS_0 * HBAR * C)


# ── σ-Connection ───────────────────────────────────────────────────────

def sigma_em_coupling(sigma):
    """Effective fine structure constant under σ-field compression.

    At the σ-transition, the QCD energy scale Λ_QCD shifts by e^σ.
    The EM coupling α is dimensionless and σ-invariant at tree level,
    but the effective nuclear EM coupling (Coulomb coefficient A_C)
    depends on r₀ which is a nuclear — not atomic — length.

    In the SSBM framework:  α_eff(σ) = α × e^(2ξσ)

    Physical interpretation: inside a σ-compressed pocket, the nuclear
    charge radius r₀ shrinks as Λ_QCD grows, so the Coulomb repulsion
    per nucleon pair increases.

    At σ = 0: α_eff = α (standard value ≈ 1/137).
    At σ_conv ≈ 1.849: α_eff ≈ α × e^(2×0.1582×1.849) ≈ α × 1.80.

    Args:
        sigma: σ-field value (dimensionless, ≥ 0)

    Returns:
        effective EM coupling (dimensionless)
    """
    return ALPHA * scale_ratio(2.0 * XI * sigma)
