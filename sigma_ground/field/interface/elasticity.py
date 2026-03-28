"""
Elasticity — stress-strain relationships, elastic energy, and wave propagation.

This module extends mechanical.py with the continuum mechanics framework:
stress tensors, strain tensors, Hooke's law, elastic energy density, and
elastic constants relationships beyond what mechanical.py provides.

mechanical.py provides: K, E, G, ν, τ_th (material-specific, from cohesive energy).
This module provides: the RELATIONSHIPS between elastic quantities, strain energy,
stress-strain mappings, Lamé parameters, elastic wave speeds, and failure criteria.

Derivation chains:

  1. Lamé Parameters (FIRST_PRINCIPLES: isotropic elasticity)
     λ = E×ν / ((1+ν)(1−2ν))
     μ = G = E / (2(1+ν))

     These are the two independent elastic constants for an isotropic solid.
     All other moduli (K, E, G, ν) are functions of (λ, μ).

  2. Stress-Strain (Hooke's Law, FIRST_PRINCIPLES)
     σ_ij = λ δ_ij ε_kk + 2μ ε_ij   (generalized Hooke's law)

     For uniaxial tension: σ = E × ε  (definition of Young's modulus)
     For pure shear:       τ = G × γ  (definition of shear modulus)
     For hydrostatic:      P = K × ΔV/V  (definition of bulk modulus)

  3. Elastic Strain Energy Density (FIRST_PRINCIPLES)
     u = ½ σ_ij ε_ij = ½ λ (ε_kk)² + μ ε_ij ε_ij
     For uniaxial: u = ½ E ε² = σ²/(2E)

  4. Elastic Wave Speeds (FIRST_PRINCIPLES: Newton-Laplace)
     Already in acoustics.py — we import and re-export for completeness.

  5. Poisson Effect (FIRST_PRINCIPLES)
     ε_transverse = −ν × ε_axial
     ΔV/V = (1 − 2ν) × ε_axial   (volume change under uniaxial stress)

  6. Von Mises Yield Criterion (FIRST_PRINCIPLES: J2 plasticity)
     σ_vm = √(½[(σ₁−σ₂)² + (σ₂−σ₃)² + (σ₃−σ₁)²])
     Yielding when σ_vm = σ_yield

     Von Mises (1913): yielding occurs when distortional energy reaches
     a critical value. Equivalent to octahedral shear stress criterion.

σ-dependence:
  All elastic moduli inherit σ-shifts from mechanical.py (through cohesive
  energy → bulk modulus → E, G). Lamé parameters, strain energy, and wave
  speeds shift accordingly. The RELATIONSHIPS (Hooke's law, identities)
  are exact continuum mechanics — σ enters only through the moduli.

Origin tags:
  - Lamé parameters: FIRST_PRINCIPLES (isotropic elasticity identities)
  - Hooke's law: FIRST_PRINCIPLES (linear elasticity, exact for small strain)
  - Strain energy: FIRST_PRINCIPLES (work integral)
  - Von Mises: FIRST_PRINCIPLES (J2 flow theory)
  - Elastic moduli: via mechanical.py (FIRST_PRINCIPLES + MEASURED ν)
  - σ-dependence: CORE (through □σ = −ξR via mechanical.py)
"""

import math
from .mechanical import (
    bulk_modulus, youngs_modulus, shear_modulus,
    theoretical_shear_strength, MECHANICAL_DATA,
)
from .surface import MATERIALS
from ..scale import scale_ratio
from ..constants import PROTON_QCD_FRACTION, SIGMA_HERE


# ── Lamé Parameters ──────────────────────────────────────────────

def lame_lambda(material_key, sigma=SIGMA_HERE):
    """First Lamé parameter λ (Pa).

    λ = E×ν / ((1+ν)(1−2ν))

    FIRST_PRINCIPLES: isotropic elasticity identity.
    λ relates volumetric strain to normal stress in Hooke's tensor.

    Args:
        material_key: key into MATERIALS dict
        sigma: σ-field value

    Returns:
        λ in Pascals
    """
    E = youngs_modulus(material_key, sigma)
    nu = MECHANICAL_DATA[material_key]['poisson_ratio']
    return E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))


def lame_mu(material_key, sigma=SIGMA_HERE):
    """Second Lamé parameter μ = G (Pa).

    μ = E / (2(1+ν)) = G (shear modulus)

    FIRST_PRINCIPLES: μ is identically the shear modulus.

    Args:
        material_key: key into MATERIALS dict
        sigma: σ-field value

    Returns:
        μ in Pascals
    """
    return shear_modulus(material_key, sigma)


# ── Stress-Strain Relations ─────────────────────────────────────

def uniaxial_stress(material_key, strain, sigma=SIGMA_HERE):
    """Axial stress from uniaxial strain (Hooke's law).

    σ = E × ε

    FIRST_PRINCIPLES: definition of Young's modulus (linear regime).
    Valid for |ε| << 1 (small strain approximation).

    Args:
        material_key: key into MATERIALS dict
        strain: engineering strain ε (dimensionless)
        sigma: σ-field value

    Returns:
        Stress in Pascals
    """
    E = youngs_modulus(material_key, sigma)
    return E * strain


def shear_stress(material_key, shear_strain, sigma=SIGMA_HERE):
    """Shear stress from shear strain.

    τ = G × γ

    FIRST_PRINCIPLES: definition of shear modulus.

    Args:
        material_key: key into MATERIALS dict
        shear_strain: engineering shear strain γ (dimensionless)
        sigma: σ-field value

    Returns:
        Shear stress in Pascals
    """
    G = shear_modulus(material_key, sigma)
    return G * shear_strain


def hydrostatic_stress(material_key, volume_strain, sigma=SIGMA_HERE):
    """Hydrostatic pressure from volumetric strain.

    P = −K × (ΔV/V)

    FIRST_PRINCIPLES: definition of bulk modulus.
    Negative sign: compression (ΔV/V < 0) gives positive pressure.

    Args:
        material_key: key into MATERIALS dict
        volume_strain: ΔV/V (dimensionless, negative for compression)
        sigma: σ-field value

    Returns:
        Pressure in Pascals (positive for compression)
    """
    K = bulk_modulus(material_key, sigma)
    return -K * volume_strain


# ── Transverse Strain (Poisson Effect) ─────────────────────────

def transverse_strain(material_key, axial_strain):
    """Transverse strain from axial strain via Poisson's ratio.

    ε_transverse = −ν × ε_axial

    FIRST_PRINCIPLES: definition of Poisson's ratio.

    Args:
        material_key: key into MATERIALS dict
        axial_strain: axial engineering strain

    Returns:
        Transverse strain (dimensionless)
    """
    nu = MECHANICAL_DATA[material_key]['poisson_ratio']
    return -nu * axial_strain


def volume_change_uniaxial(material_key, axial_strain):
    """Volumetric strain under uniaxial stress.

    ΔV/V = (1 − 2ν) × ε_axial

    FIRST_PRINCIPLES: two transverse contractions + one axial extension.
    For ν = 0.5 (incompressible): ΔV/V = 0 (rubber-like).
    For ν = 0 (cork-like): ΔV/V = ε (no lateral contraction).

    Args:
        material_key: key into MATERIALS dict
        axial_strain: axial engineering strain

    Returns:
        Volumetric strain ΔV/V (dimensionless)
    """
    nu = MECHANICAL_DATA[material_key]['poisson_ratio']
    return (1.0 - 2.0 * nu) * axial_strain


# ── Elastic Strain Energy ──────────────────────────────────────

def strain_energy_density_uniaxial(material_key, strain, sigma=SIGMA_HERE):
    """Elastic strain energy density under uniaxial stress (J/m³).

    u = ½ E ε²

    FIRST_PRINCIPLES: work integral ∫σ dε = ∫Eε dε = ½Eε².

    Args:
        material_key: key into MATERIALS dict
        strain: engineering strain ε
        sigma: σ-field value

    Returns:
        Energy density in J/m³
    """
    E = youngs_modulus(material_key, sigma)
    return 0.5 * E * strain ** 2


def strain_energy_density_shear(material_key, shear_strain, sigma=SIGMA_HERE):
    """Elastic strain energy density under pure shear (J/m³).

    u = ½ G γ²

    FIRST_PRINCIPLES: work integral.

    Args:
        material_key: key into MATERIALS dict
        shear_strain: engineering shear strain γ
        sigma: σ-field value

    Returns:
        Energy density in J/m³
    """
    G = shear_modulus(material_key, sigma)
    return 0.5 * G * shear_strain ** 2


def strain_energy_density_hydrostatic(material_key, volume_strain, sigma=SIGMA_HERE):
    """Elastic strain energy density under hydrostatic stress (J/m³).

    u = ½ K (ΔV/V)²

    FIRST_PRINCIPLES: work integral for volumetric deformation.

    Args:
        material_key: key into MATERIALS dict
        volume_strain: ΔV/V
        sigma: σ-field value

    Returns:
        Energy density in J/m³
    """
    K = bulk_modulus(material_key, sigma)
    return 0.5 * K * volume_strain ** 2


# ── Von Mises Yield Criterion ──────────────────────────────────

def von_mises_stress(sigma_1, sigma_2, sigma_3):
    """Von Mises equivalent stress from principal stresses.

    σ_vm = √(½[(σ₁−σ₂)² + (σ₂−σ₃)² + (σ₃−σ₁)²])

    FIRST_PRINCIPLES: von Mises (1913), based on distortional energy.
    Yielding when σ_vm reaches the uniaxial yield stress.

    This is an exact result from J2 plasticity theory.

    Args:
        sigma_1, sigma_2, sigma_3: principal stresses (Pa)

    Returns:
        Von Mises stress in Pascals
    """
    return math.sqrt(0.5 * (
        (sigma_1 - sigma_2) ** 2 +
        (sigma_2 - sigma_3) ** 2 +
        (sigma_3 - sigma_1) ** 2
    ))


def is_yielded(sigma_1, sigma_2, sigma_3, yield_stress):
    """Check if principal stress state exceeds yield (von Mises criterion).

    Args:
        sigma_1, sigma_2, sigma_3: principal stresses (Pa)
        yield_stress: uniaxial yield stress (Pa)

    Returns:
        True if material has yielded
    """
    return von_mises_stress(sigma_1, sigma_2, sigma_3) >= yield_stress


# ── Elastic Moduli Relationships ────────────────────────────────

def moduli_from_lame(lam, mu):
    """Compute all isotropic elastic moduli from Lamé parameters.

    FIRST_PRINCIPLES: exact identities of isotropic elasticity.
    Given (λ, μ), all other moduli follow uniquely.

    Args:
        lam: first Lamé parameter λ (Pa)
        mu: second Lamé parameter μ = G (Pa)

    Returns:
        dict with K, E, G, nu (all in SI)
    """
    K = lam + 2.0 * mu / 3.0
    E = mu * (3.0 * lam + 2.0 * mu) / (lam + mu)
    G = mu
    nu = lam / (2.0 * (lam + mu))
    return {'K_pa': K, 'E_pa': E, 'G_pa': G, 'poisson_ratio': nu}


def p_wave_modulus(material_key, sigma=SIGMA_HERE):
    """P-wave modulus M = K + 4G/3 = λ + 2μ (Pa).

    FIRST_PRINCIPLES: the modulus that governs longitudinal wave speed.
    v_L = √(M/ρ)

    Args:
        material_key: key into MATERIALS dict
        sigma: σ-field value

    Returns:
        M in Pascals
    """
    K = bulk_modulus(material_key, sigma)
    G = shear_modulus(material_key, sigma)
    return K + 4.0 * G / 3.0


# ── σ-Shifted Elastic Properties ────────────────────────────────

def sigma_elastic_shift(material_key, sigma):
    """Fractional change in elastic moduli at given σ.

    Returns E(σ)/E(0) — the ratio of shifted to unshifted Young's modulus.
    Same ratio applies to K, G, and λ since they all scale through E_coh.

    CORE: σ-dependence through nuclear mass → cohesive energy → moduli.

    Args:
        material_key: key into MATERIALS dict
        sigma: σ-field value

    Returns:
        Ratio E(σ)/E(0) (dimensionless, ≥1 for σ>0)
    """
    if sigma == SIGMA_HERE:
        return 1.0
    E_0 = youngs_modulus(material_key, SIGMA_HERE)
    E_s = youngs_modulus(material_key, sigma)
    return E_s / E_0


# ── Nagatha Integration ──────────────────────────────────────────

def material_elastic_properties(material_key, sigma=SIGMA_HERE):
    """Export elastic properties in Nagatha-compatible format.

    Returns a dict that can be merged into Nagatha's material database.
    """
    E = youngs_modulus(material_key, sigma)
    K = bulk_modulus(material_key, sigma)
    G = shear_modulus(material_key, sigma)
    nu = MECHANICAL_DATA[material_key]['poisson_ratio']
    lam = lame_lambda(material_key, sigma)
    M = p_wave_modulus(material_key, sigma)
    tau = theoretical_shear_strength(material_key, sigma)

    return {
        'youngs_modulus_pa': E,
        'bulk_modulus_pa': K,
        'shear_modulus_pa': G,
        'poisson_ratio': nu,
        'lame_lambda_pa': lam,
        'lame_mu_pa': G,
        'p_wave_modulus_pa': M,
        'theoretical_shear_strength_pa': tau,
        'sigma': sigma,
        'elastic_shift_ratio': sigma_elastic_shift(material_key, sigma),
        'origin_tag': (
            "FIRST_PRINCIPLES: Lamé parameters from isotropic elasticity identities. "
            "FIRST_PRINCIPLES: Hooke's law (linear, small strain). "
            "FIRST_PRINCIPLES: strain energy from work integral. "
            "FIRST_PRINCIPLES: von Mises from J2 plasticity. "
            "MEASURED: Poisson's ratio. "
            "CORE: σ-dependence through nuclear mass → cohesive energy → moduli."
        ),
    }
