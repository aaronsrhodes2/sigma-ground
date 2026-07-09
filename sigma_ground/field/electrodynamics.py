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
    E_CHARGE, EPS_0, C, HBAR, MU_0, M_ELECTRON_KG, ALPHA,
    A_MU_EXP_X1E11, A_MU_EXP_X1E11_SIGMA,
    A_MU_SM_X1E11,  A_MU_SM_X1E11_SIGMA,
    MU_G2_TENSION_RESOLVED,
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
    """Collisionless (plasma) skin depth δ (m).

    Below the plasma frequency an EM wave is evanescent in the electron gas,
    decaying as e^(−x/δ) with
        δ = c / √(ω_p² − ω²),   ω_p = √(n_e e²/(ε₀ m_e)).
    At ω = 0 (DC / London limit) this reduces to c/ω_p; for ω ≥ ω_p the wave
    propagates (no skin depth) → returns inf.

    NOTE: this is the COLLISIONLESS plasma skin depth. For the *resistive*
    skin depth of a conductor at frequency f (δ = √(2/(μ₀ σ ω))), use
    ``mobius.skin_depth(material_key, frequency_hz)`` instead.

    Args:
        n_e: free electron number density (m⁻³)
        omega: angular frequency in rad/s; default None ⇒ 0 (DC, δ = c/ω_p)

    Returns:
        skin depth (m); ``math.inf`` if ω ≥ ω_p (wave propagates).
    """
    omega_p = math.sqrt(n_e * E_CHARGE**2 / (EPS_0 * M_ELECTRON_KG))
    w = 0.0 if omega is None else float(omega)   # DC limit when unspecified
    if w >= omega_p:
        return math.inf                          # over-dense → wave propagates
    return C / math.sqrt(omega_p**2 - w**2)


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


# ── Muon g−2 precision test ──────────────────────────────────────────
# Phase XII.c.1: Muon Theory Initiative 2025 white paper (May 2025) + Fermilab
# final result (June 2025). arXiv references:
#   - Muon Theory Initiative 2025 white paper
#   - Fermilab a_μ Run 1–6 combined (arXiv:2506.03069)
# With lattice-HVP adopted, the decades-long g−2 anomaly is dissolved.
# Sigma-ground does not compute a_μ from first principles (it has no HVP
# module), but it must not contradict the new SM+experiment consensus.

def muon_anomalous_moment_experimental():
    """Fermilab Run 1–6 combined experimental a_μ.

    [VERIFIED] — Fermilab final, June 2025. 127 ppb precision.

    Returns:
        (a_mu_exp, sigma) in units of 10⁻¹¹.
    """
    return A_MU_EXP_X1E11, A_MU_EXP_X1E11_SIGMA


def muon_anomalous_moment_sm():
    """Standard-Model prediction for a_μ from Muon Theory Initiative 2025 WP.

    [VERIFIED] — MTI 2025 white paper (lattice-HVP adopted).

    Returns:
        (a_mu_sm, sigma) in units of 10⁻¹¹.
    """
    return A_MU_SM_X1E11, A_MU_SM_X1E11_SIGMA


def muon_g2_tension_sigmas():
    """Return the residual tension in units of combined σ.

    Sigma-combination: σ_tot = √(σ_exp² + σ_SM²). With 2025 values,
    residual ≈ 5.2σ numerically, but per MTI 2025 WP this is an artefact
    of differing HVP input data (CMD-3 vs lattice), not a real physics
    tension. See misc/arxiv_pluggable_survey_2025.md Batch 3.

    Returns:
        residual tension in σ_tot units (float, ≥ 0).
    """
    a_exp, s_exp = muon_anomalous_moment_experimental()
    a_sm,  s_sm  = muon_anomalous_moment_sm()
    sigma_tot = math.sqrt(s_exp * s_exp + s_sm * s_sm)
    return abs(a_exp - a_sm) / sigma_tot


def muon_g2_consistency_status():
    """Consistency report for the sigma-ground QCD sector against MTI 2025.

    Sigma-ground's LAMBDA_QCD_MEV = 217 MeV is PDG-standard; the MTI 2025
    lattice-HVP SM prediction was computed with the same PDG-standard
    inputs. Therefore sigma-ground is automatically consistent with
    the 2025 consensus — no new physics required.

    If a future sigma-ground change alters LAMBDA_QCD_MEV, revisit the
    chain: Λ_QCD → HVP integral → a_μ_SM → tension with experiment.

    Returns:
        dict with keys:
            'a_mu_exp_x1e11', 'a_mu_sm_x1e11',
            'tension_sigmas': residual discrepancy in σ_tot units,
            'tension_resolved': bool (True per MTI 2025 WP),
            'lambda_qcd_mev': engine Λ_QCD value (for provenance),
            'note': one-line summary.
    """
    from .constants import LAMBDA_QCD_MEV
    return {
        'a_mu_exp_x1e11': A_MU_EXP_X1E11,
        'a_mu_sm_x1e11':  A_MU_SM_X1E11,
        'tension_sigmas': muon_g2_tension_sigmas(),
        'tension_resolved': MU_G2_TENSION_RESOLVED,
        'lambda_qcd_mev': LAMBDA_QCD_MEV,
        'note': 'Sigma-ground inherits PDG Lambda_QCD; MTI 2025 consensus with lattice-HVP holds.',
    }
