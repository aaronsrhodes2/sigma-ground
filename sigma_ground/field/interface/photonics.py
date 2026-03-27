"""
Photonics — waveguides, photonic bandgap, and nonlinear optics.

This module extends optics.py (Drude model, Fresnel, Beer-Lambert) with
wave-guiding phenomena, photonic crystal bandgaps, and nonlinear optical
effects (χ², χ³).

optics.py provides: n, k, reflectance for metals and dielectrics.
electrodynamics.py provides: Coulomb, EM waves, fine structure constant.
This module provides: guided-mode optics, bandgap engineering, and intensity-
dependent phenomena.

Derivation chains:

  1. Slab Waveguide — Total Internal Reflection (FIRST_PRINCIPLES)
     Critical angle: θ_c = arcsin(n_clad / n_core)
     Number of guided modes (ray optics):
       V = (2π d / λ) × √(n_core² − n_clad²)   (V-number)
       m_max ≈ floor(2V / π)   for TE modes in a symmetric slab

     Cutoff condition: V > mπ/2 for mode m.
     Single-mode when V < π/2.

     FIRST_PRINCIPLES: Snell's law + phase-matching at boundaries.

  2. Optical Fiber — Step-Index (FIRST_PRINCIPLES)
     V = (2π a / λ) × NA    where NA = √(n_core² − n_clad²)
     Single-mode cutoff: V < 2.405 (first zero of J₀ Bessel function)
     Number of modes ≈ V²/2 (for large V)

     FIRST_PRINCIPLES: cylindrical waveguide eigenvalue problem.

  3. Photonic Bandgap — 1D Bragg Stack (FIRST_PRINCIPLES)
     Center wavelength: λ_Bragg = 2(n₁d₁ + n₂d₂)   (quarter-wave stack)
     Bandwidth: Δλ/λ = (4/π) arcsin(|n₁−n₂| / (n₁+n₂))

     FIRST_PRINCIPLES: transfer matrix method for periodic dielectric.
     The stop band arises from destructive interference — no material
     absorption needed. Same physics as X-ray Bragg diffraction but
     at optical wavelengths.

  4. Nonlinear Optics — χ² (Second Harmonic Generation)
     P_NL = ε₀ χ² E²

     Requires non-centrosymmetric crystal (no inversion symmetry).
     Conversion efficiency ∝ d_eff² × L² × I / (n³ λ²)
     where d_eff = effective nonlinear coefficient (MEASURED).

     Phase matching: Δk = k(2ω) − 2k(ω) = 0 required for efficiency.

     FIRST_PRINCIPLES: perturbation theory on Maxwell's equations.
     MEASURED: d_eff values for specific crystals.

  5. Nonlinear Optics — χ³ (Kerr Effect, Self-Phase Modulation)
     n(I) = n₀ + n₂ × I

     Intensity-dependent refractive index. All materials exhibit χ³
     (unlike χ², which requires broken inversion symmetry).

     Critical power for self-focusing:
       P_cr = 3.77 λ² / (8π n₀ n₂)

     FIRST_PRINCIPLES: third-order perturbation of polarization.
     MEASURED: n₂ values.

  6. Optical Absorption Edge (FIRST_PRINCIPLES)
     For semiconductors: α(ω) ∝ (ℏω − E_g)^(1/2)  (direct gap)
                         α(ω) ∝ (ℏω − E_g)²        (indirect gap)

     Connects to semiconductor_optics.py bandgap data.

σ-dependence:
  Refractive index is electromagnetic → σ-INVARIANT.
  Bandgap is electronic → σ-INVARIANT.
  Nonlinear coefficients are electronic → σ-INVARIANT.

  The σ-bridge: lattice spacing changes with nuclear mass (phonon hardening
  → slightly different equilibrium positions). This shifts photonic crystal
  period d(σ), moving the Bragg wavelength. The effect is tiny at Earth σ
  but measurable at neutron star σ.

Origin tags:
  - Waveguide modes: FIRST_PRINCIPLES (Maxwell eigenvalue problem)
  - V-number: FIRST_PRINCIPLES (normalized frequency)
  - Bragg bandgap: FIRST_PRINCIPLES (transfer matrix / Floquet)
  - χ² SHG: FIRST_PRINCIPLES (perturbation) + MEASURED (d_eff)
  - χ³ Kerr: FIRST_PRINCIPLES (perturbation) + MEASURED (n₂)
  - σ-dependence: CORE (through lattice period shift)
"""

import math
from ..constants import EPS_0, C, HBAR


# ── Waveguide Fundamentals ───────────────────────────────────────

def numerical_aperture(n_core, n_clad):
    """Numerical aperture of a step-index waveguide.

    NA = √(n_core² − n_clad²)

    FIRST_PRINCIPLES: maximum acceptance angle from Snell's law.

    Args:
        n_core: refractive index of core
        n_clad: refractive index of cladding

    Returns:
        NA (dimensionless)
    """
    if n_core <= n_clad:
        raise ValueError(f"n_core={n_core} must exceed n_clad={n_clad}")
    return math.sqrt(n_core ** 2 - n_clad ** 2)


def critical_angle(n_core, n_clad):
    """Critical angle for total internal reflection (radians).

    θ_c = arcsin(n_clad / n_core)

    FIRST_PRINCIPLES: Snell's law at the TIR boundary.

    Args:
        n_core: refractive index of denser medium
        n_clad: refractive index of rarer medium

    Returns:
        Critical angle in radians
    """
    if n_core <= n_clad:
        raise ValueError(f"n_core={n_core} must exceed n_clad={n_clad}")
    return math.asin(n_clad / n_core)


def v_number_slab(thickness, wavelength, n_core, n_clad):
    """V-number for a symmetric slab waveguide.

    V = (2π d / λ) × √(n_core² − n_clad²)

    FIRST_PRINCIPLES: normalized frequency parameter.
    Determines number of guided modes.

    Args:
        thickness: slab thickness d (m)
        wavelength: free-space wavelength λ (m)
        n_core: core refractive index
        n_clad: cladding refractive index

    Returns:
        V-number (dimensionless)
    """
    NA = numerical_aperture(n_core, n_clad)
    return 2.0 * math.pi * thickness * NA / wavelength


def v_number_fiber(core_radius, wavelength, n_core, n_clad):
    """V-number for a step-index optical fiber.

    V = (2π a / λ) × NA

    FIRST_PRINCIPLES: normalized frequency for cylindrical waveguide.

    Args:
        core_radius: fiber core radius a (m)
        wavelength: free-space wavelength λ (m)
        n_core: core refractive index
        n_clad: cladding refractive index

    Returns:
        V-number (dimensionless)
    """
    NA = numerical_aperture(n_core, n_clad)
    return 2.0 * math.pi * core_radius * NA / wavelength


def is_single_mode_fiber(core_radius, wavelength, n_core, n_clad):
    """Check if fiber supports only a single mode.

    Single-mode cutoff: V < 2.405 (first zero of J₀ Bessel function).

    FIRST_PRINCIPLES: cylindrical waveguide eigenvalue problem.

    Args:
        core_radius: fiber core radius (m)
        wavelength: wavelength (m)
        n_core, n_clad: refractive indices

    Returns:
        True if single-mode
    """
    V = v_number_fiber(core_radius, wavelength, n_core, n_clad)
    return V < 2.405


def number_of_modes_fiber(core_radius, wavelength, n_core, n_clad):
    """Approximate number of guided modes in a multimode fiber.

    N ≈ V²/2   (for V >> 1)
    N = 1      (for V < 2.405)

    FIRST_PRINCIPLES: mode counting from V-number.
    APPROXIMATION: continuous approximation, exact only for large V.

    Args:
        core_radius: fiber core radius (m)
        wavelength: wavelength (m)
        n_core, n_clad: refractive indices

    Returns:
        Approximate number of guided modes
    """
    V = v_number_fiber(core_radius, wavelength, n_core, n_clad)
    if V < 2.405:
        return 1
    return max(1, int(V ** 2 / 2.0))


def slab_modes_count(thickness, wavelength, n_core, n_clad):
    """Number of guided TE modes in a symmetric slab waveguide.

    m_max = floor(2V/π)   where V = (2πd/λ)√(n²_core − n²_clad)

    Always at least 1 mode (fundamental) for any V > 0.

    FIRST_PRINCIPLES: transcendental eigenvalue equation solutions.

    Args:
        thickness: slab thickness (m)
        wavelength: wavelength (m)
        n_core, n_clad: refractive indices

    Returns:
        Number of guided TE modes
    """
    V = v_number_slab(thickness, wavelength, n_core, n_clad)
    return max(1, int(2.0 * V / math.pi))


# ── Photonic Bandgap ─────────────────────────────────────────────

def bragg_wavelength(n1, d1, n2, d2):
    """Center wavelength of a 1D photonic bandgap (Bragg stack) (m).

    λ_Bragg = 2(n₁d₁ + n₂d₂)

    FIRST_PRINCIPLES: constructive interference condition for
    quarter-wave dielectric stack. At this wavelength, reflections
    from all interfaces add in phase → maximum reflectance.

    Args:
        n1: refractive index of layer 1
        d1: thickness of layer 1 (m)
        n2: refractive index of layer 2
        d2: thickness of layer 2 (m)

    Returns:
        Bragg wavelength in metres
    """
    return 2.0 * (n1 * d1 + n2 * d2)


def bragg_bandwidth_fraction(n1, n2):
    """Fractional bandwidth of 1D photonic bandgap.

    Δλ/λ = (4/π) × arcsin(|n₁−n₂| / (n₁+n₂))

    FIRST_PRINCIPLES: transfer matrix analysis of infinite Bragg stack.
    Larger index contrast → wider bandgap.

    Args:
        n1, n2: refractive indices of alternating layers

    Returns:
        Δλ/λ (dimensionless fraction)
    """
    contrast = abs(n1 - n2) / (n1 + n2)
    return (4.0 / math.pi) * math.asin(contrast)


def bragg_reflectance(n1, n2, N_pairs, n_substrate=1.0):
    """Peak reflectance of a Bragg stack with N layer pairs.

    R = ((n₂/n₁)^(2N) − n_sub)² / ((n₂/n₁)^(2N) + n_sub)²

    Assumes n₂ > n₁ (high-index layer at interfaces).
    FIRST_PRINCIPLES: transfer matrix result for quarter-wave stack.

    Args:
        n1: low-index layer
        n2: high-index layer
        N_pairs: number of layer pairs
        n_substrate: substrate refractive index

    Returns:
        Peak reflectance R (0 to 1)
    """
    if n2 < n1:
        n1, n2 = n2, n1  # ensure n2 > n1

    ratio = (n2 / n1) ** (2 * N_pairs)
    return ((ratio - n_substrate) / (ratio + n_substrate)) ** 2


# ── Nonlinear Optics — χ² ────────────────────────────────────────

# Effective nonlinear coefficients (pm/V = 10⁻¹² m/V)
# MEASURED: from Boyd "Nonlinear Optics" (2008), Dmitriev et al.
NONLINEAR_CRYSTALS = {
    'KDP': {
        'name': 'Potassium Dihydrogen Phosphate',
        'd_eff_pm_V': 0.39,        # Type I SHG
        'n_omega': 1.507,          # at 1064 nm
        'n_2omega': 1.471,         # at 532 nm
        'transparency_nm': (200, 1500),
    },
    'BBO': {
        'name': 'Beta-Barium Borate',
        'd_eff_pm_V': 2.01,        # Type I SHG
        'n_omega': 1.655,
        'n_2omega': 1.674,
        'transparency_nm': (190, 3300),
    },
    'LiNbO3': {
        'name': 'Lithium Niobate',
        'd_eff_pm_V': 4.35,        # PPLN quasi-phase-matched
        'n_omega': 2.156,
        'n_2omega': 2.233,
        'transparency_nm': (350, 5000),
    },
    'KTP': {
        'name': 'Potassium Titanyl Phosphate',
        'd_eff_pm_V': 3.18,        # Type II SHG
        'n_omega': 1.740,
        'n_2omega': 1.779,
        'transparency_nm': (350, 4500),
    },
}


def shg_phase_mismatch(n_omega, n_2omega, wavelength):
    """Phase mismatch for second harmonic generation (1/m).

    Δk = 2k(ω) − k(2ω) = (4π/λ)(n_ω − n_2ω)

    FIRST_PRINCIPLES: momentum conservation in nonlinear process.
    Efficient SHG requires Δk → 0 (phase matching).

    Args:
        n_omega: refractive index at fundamental frequency
        n_2omega: refractive index at second harmonic
        wavelength: fundamental wavelength λ (m)

    Returns:
        Phase mismatch Δk in 1/m
    """
    return 4.0 * math.pi * (n_omega - n_2omega) / wavelength


def shg_efficiency_factor(d_eff_pm_V, length, n_omega, n_2omega, wavelength):
    """SHG conversion efficiency factor (normalized).

    η ∝ d_eff² × L² / (n_ω² × n_2ω × λ²)

    FIRST_PRINCIPLES: coupled-wave equations for perfect phase matching.
    Returns relative efficiency — multiply by intensity for actual conversion.

    MEASURED: d_eff values from nonlinear crystal databases.

    Args:
        d_eff_pm_V: effective nonlinear coefficient (pm/V)
        length: crystal length L (m)
        n_omega: refractive index at ω
        n_2omega: refractive index at 2ω
        wavelength: fundamental wavelength (m)

    Returns:
        Efficiency factor (arbitrary units, for comparison)
    """
    d_eff_m_V = d_eff_pm_V * 1e-12  # pm/V → m/V
    return (d_eff_m_V ** 2 * length ** 2 /
            (n_omega ** 2 * n_2omega * wavelength ** 2))


# ── Nonlinear Optics — χ³ (Kerr Effect) ─────────────────────────

# Nonlinear refractive indices n₂ (m²/W)
# MEASURED: from Boyd (2008), Adair et al. (1989)
KERR_MATERIALS = {
    'silica': {
        'name': 'Fused Silica (SiO₂)',
        'n0': 1.45,
        'n2_m2_W': 2.7e-20,        # at 1064 nm; Milam (1998)
    },
    'BK7': {
        'name': 'Borosilicate Crown Glass',
        'n0': 1.52,
        'n2_m2_W': 3.2e-20,        # DeSalvo et al. (1996)
    },
    'sapphire': {
        'name': 'Sapphire (Al₂O₃)',
        'n0': 1.76,
        'n2_m2_W': 3.0e-20,        # Major et al. (2004)
    },
    'silicon': {
        'name': 'Silicon',
        'n0': 3.48,
        'n2_m2_W': 4.5e-18,        # at 1550 nm; Bristow et al. (2007)
    },
    'CS2': {
        'name': 'Carbon Disulfide',
        'n0': 1.63,
        'n2_m2_W': 3.2e-18,        # molecular reorientation contribution
    },
}


def kerr_refractive_index(n0, n2, intensity):
    """Intensity-dependent refractive index (Kerr effect).

    n(I) = n₀ + n₂ × I

    FIRST_PRINCIPLES: χ³ contribution to polarization.
    All materials exhibit this — no symmetry requirement.

    Args:
        n0: linear refractive index
        n2: nonlinear index (m²/W)
        intensity: optical intensity I (W/m²)

    Returns:
        Total refractive index
    """
    return n0 + n2 * intensity


def self_focusing_critical_power(wavelength, n0, n2):
    """Critical power for self-focusing (W).

    P_cr = 3.77 λ² / (8π n₀ n₂)

    FIRST_PRINCIPLES: balance of diffraction divergence and
    Kerr lens convergence. Above P_cr, the beam collapses.

    Marburger (1975), Fibich & Gaeta (2000).

    Args:
        wavelength: wavelength λ (m)
        n0: linear refractive index
        n2: nonlinear index (m²/W)

    Returns:
        Critical power in Watts
    """
    return 3.77 * wavelength ** 2 / (8.0 * math.pi * n0 * n2)


def nonlinear_phase_shift(n2, intensity, length, wavelength):
    """B-integral: accumulated nonlinear phase shift (radians).

    φ_NL = (2π/λ) × n₂ × I × L

    FIRST_PRINCIPLES: phase accumulation from intensity-dependent index.
    B-integral > π indicates significant nonlinear distortion.

    Args:
        n2: nonlinear index (m²/W)
        intensity: optical intensity (W/m²)
        length: propagation length (m)
        wavelength: wavelength (m)

    Returns:
        Phase shift in radians
    """
    return 2.0 * math.pi * n2 * intensity * length / wavelength


# ── Absorption Edge ──────────────────────────────────────────────

def absorption_coefficient_direct(photon_energy_eV, bandgap_eV,
                                  A_coeff=1e5):
    """Absorption coefficient near direct bandgap (1/m).

    α(E) = A × √(E − E_g)   for E > E_g
    α(E) = 0                 for E ≤ E_g

    FIRST_PRINCIPLES: joint density of states for parabolic bands.
    A_coeff is material-specific (MEASURED, typically 10⁴-10⁶ /m per √eV).

    Args:
        photon_energy_eV: photon energy (eV)
        bandgap_eV: bandgap energy (eV)
        A_coeff: absorption strength coefficient (1/m/√eV)

    Returns:
        Absorption coefficient in 1/m
    """
    if photon_energy_eV <= bandgap_eV:
        return 0.0
    return A_coeff * math.sqrt(photon_energy_eV - bandgap_eV)


def absorption_coefficient_indirect(photon_energy_eV, bandgap_eV,
                                    B_coeff=1e3):
    """Absorption coefficient near indirect bandgap (1/m).

    α(E) = B × (E − E_g)²   for E > E_g
    α(E) = 0                 for E ≤ E_g

    FIRST_PRINCIPLES: phonon-assisted transitions require momentum
    conservation → quadratic onset instead of square-root.
    B_coeff is material-specific (MEASURED).

    Args:
        photon_energy_eV: photon energy (eV)
        bandgap_eV: bandgap energy (eV)
        B_coeff: absorption strength (1/m/eV²)

    Returns:
        Absorption coefficient in 1/m
    """
    if photon_energy_eV <= bandgap_eV:
        return 0.0
    return B_coeff * (photon_energy_eV - bandgap_eV) ** 2


# ── σ-Dependence ─────────────────────────────────────────────────

def sigma_bragg_shift(n1, d1, n2, d2, sigma):
    """Bragg wavelength shift under σ-field.

    The lattice spacing shifts through nuclear mass:
      d(σ) ≈ d(0) × (1 + δ)  where δ is the ZPE lattice expansion

    Refractive indices are EM → invariant.
    Net: λ_Bragg shifts proportionally to lattice expansion.

    CORE: σ-dependence through nuclear mass → lattice spacing.

    Args:
        n1, d1, n2, d2: Bragg stack parameters
        sigma: σ-field value

    Returns:
        (lambda_bragg_0, lambda_bragg_sigma) tuple in metres
    """
    from ..scale import scale_ratio
    from ..constants import PROTON_QCD_FRACTION

    lam_0 = bragg_wavelength(n1, d1, n2, d2)

    if sigma == 0.0:
        return (lam_0, lam_0)

    # Lattice expansion from ZPE shift (same model as mechanical.py)
    f_qcd = PROTON_QCD_FRACTION
    mass_ratio = (1.0 - f_qcd) + f_qcd * scale_ratio(sigma)
    # ZPE contributes ~1% to lattice parameter, scales as 1/√m
    f_zpe = 0.01
    expansion = f_zpe * (1.0 - 1.0 / math.sqrt(mass_ratio))

    d1_sigma = d1 * (1.0 + expansion)
    d2_sigma = d2 * (1.0 + expansion)

    lam_sigma = bragg_wavelength(n1, d1_sigma, n2, d2_sigma)
    return (lam_0, lam_sigma)


# ── Nagatha Integration ──────────────────────────────────────────

def waveguide_properties(core_radius, wavelength, n_core, n_clad):
    """Export waveguide properties in Nagatha-compatible format."""
    NA = numerical_aperture(n_core, n_clad)
    V = v_number_fiber(core_radius, wavelength, n_core, n_clad)
    N_modes = number_of_modes_fiber(core_radius, wavelength, n_core, n_clad)
    theta_c = critical_angle(n_core, n_clad)

    return {
        'numerical_aperture': NA,
        'v_number': V,
        'n_modes': N_modes,
        'single_mode': V < 2.405,
        'critical_angle_rad': theta_c,
        'critical_angle_deg': math.degrees(theta_c),
        'core_radius_m': core_radius,
        'wavelength_m': wavelength,
        'n_core': n_core,
        'n_clad': n_clad,
        'origin_tag': (
            "FIRST_PRINCIPLES: Maxwell eigenvalue problem (waveguide modes). "
            "FIRST_PRINCIPLES: V-number (normalized frequency). "
            "FIRST_PRINCIPLES: Snell's law (critical angle, TIR). "
            "σ-INVARIANT: refractive indices are electromagnetic."
        ),
    }
