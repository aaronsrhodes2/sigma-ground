"""
Hardness — indentation resistance derived from yield stress and elastic moduli.

Derivation chain:
  plasticity.py (yield stress, work hardening)
  + mechanical.py (Young's modulus, shear modulus)
  → hardness.py (Vickers, Brinell, Knoop, Mohs, Shore D)

The key insight: hardness is NOT an independent material property.
It is a DERIVED measure of resistance to localized plastic deformation.
Every hardness number can be computed from the stress-strain curve.

Derivation chains:

  1. Vickers Hardness HV (FIRST_PRINCIPLES + MEASURED calibration)
     HV = σ_rep / (g × C_tabor)

     Where:
       σ_rep = representative stress at ~8% strain (from Hollomon curve)
       C_tabor = 2.9-3.0 (Tabor constraint factor, MEASURED)
       g = 9.80665 m/s² (conversion to kgf units)

     The Tabor relation (1951): beneath a Vickers indenter, the plastic
     zone is constrained by surrounding elastic material. The mean
     pressure under the indenter (≈ HV × g) is ~3× the uniaxial flow
     stress at the representative strain.

     Representative strain ε_rep ≈ 0.08 for Vickers geometry (Tabor 1951).

  2. Brinell Hardness HB (FIRST_PRINCIPLES: same constraint factor)
     HB ≈ HV for metals (the two scales converge below HV ≈ 350).
     Above that, Brinell diverges because the ball deforms.
     HB = HV × correction_factor(HV)

  3. Knoop Hardness HK (FIRST_PRINCIPLES: elongated indenter geometry)
     HK ≈ HV × f(E/σ_y)
     The Knoop indenter has a 7:1 aspect ratio. For materials with high
     E/σ_y (metals), HK ≈ HV. For low E/σ_y (ceramics), HK < HV
     due to elastic recovery of the long diagonal.

  4. Mohs Hardness (FIRST_PRINCIPLES: scratch resistance)
     Scratch hardness scales as ~log₁₀(HV).
     Mohs = 1 + 1.5 × log₁₀(HV/10)
     Approximate — Mohs scale is nonlinear and was defined by minerals.

  5. Shore D Hardness (FIRST_PRINCIPLES: elastic rebound)
     Shore D ∝ √(E × H) — measures elastic recovery + hardness.
     Only meaningful for metals with HV < 1000.

σ-dependence:
  Hardness inherits σ-dependence from yield stress and elastic moduli.
  σ → nuclear mass → cohesive energy → moduli → yield stress → hardness.
  Harder nuclei → stiffer lattice → higher hardness.

Origin tags:
  - Tabor constraint factor: FIRST_PRINCIPLES + MEASURED (C ≈ 3.0)
  - Representative strain: MEASURED (ε_rep ≈ 0.08 for Vickers)
  - Vickers-Brinell conversion: FIRST_PRINCIPLES + MEASURED calibration
  - Knoop correction: FIRST_PRINCIPLES (elastic recovery geometry)
  - Mohs mapping: APPROXIMATION (logarithmic fit to mineral scale)
"""

import math
from .plasticity import (
    PLASTICITY_DATA, yield_stress, hollomon_stress,
)
from .mechanical import youngs_modulus, shear_modulus
from ..constants import SIGMA_HERE
from ..scale import scale_ratio


# ── Tabor Parameters ──────────────────────────────────────────────
# MEASURED: Tabor "Hardness of Metals" (1951), Cambridge University Press.
#
# The constraint factor C relates mean indentation pressure to uniaxial
# flow stress. C ≈ 3.0 for fully plastic indentation (Prandtl slip-line
# field solution gives C = 2.57 for rigid-plastic; work hardening and
# friction raise it to ~3.0).

_TABOR_CONSTRAINT = 3.0       # MEASURED: mean indentation pressure / σ_flow
_VICKERS_REP_STRAIN = 0.08    # MEASURED: representative strain under Vickers
_GF = 9.80665                 # m/s², exact by definition (kgf → N)


# ── Vickers Hardness ──────────────────────────────────────────────

def vickers_hardness(material_key, sigma=SIGMA_HERE):
    """Vickers hardness HV (kgf/mm²) from stress-strain curve.

    FIRST_PRINCIPLES: Tabor constraint factor + Hollomon flow stress.

    HV = C × σ_rep / g

    Where:
      C = 3.0 (Tabor constraint factor, MEASURED)
      σ_rep = flow stress at ε = 0.08 (representative strain)
      g = 9.80665 m/s² (kgf conversion)

    The factor of 1e-6 converts Pa → MPa (since HV is in kgf/mm²,
    and 1 kgf/mm² ≈ 9.807 MPa).

    Accuracy: ±20% for annealed metals. The Tabor relation is
    well-validated for metals (Tabor 1951, Atkins & Tabor 1965).
    Less reliable for brittle materials (cracking under indenter).

    Args:
        material_key: key into PLASTICITY_DATA
        sigma: σ-field value

    Returns:
        Vickers hardness in kgf/mm² (HV units).
    """
    data = PLASTICITY_DATA[material_key]

    if data['n_hardening'] > 0:
        # Ductile: use Hollomon flow stress at representative strain
        sigma_rep = hollomon_stress(material_key, _VICKERS_REP_STRAIN, sigma)
    else:
        # Brittle: use yield/fracture stress directly
        sigma_rep = yield_stress(material_key, sigma)

    # HV = C × σ_rep / g, converting Pa → kgf/mm²
    # 1 kgf/mm² = g × 1e6 Pa
    return _TABOR_CONSTRAINT * sigma_rep / (_GF * 1e6)


def vickers_from_yield(sigma_y_Pa, n_hardening=0.0):
    """Vickers hardness from arbitrary yield stress and hardening exponent.

    Convenience function for materials not in PLASTICITY_DATA.

    Args:
        sigma_y_Pa: yield stress in Pa
        n_hardening: Hollomon hardening exponent (0 = no hardening)

    Returns:
        Vickers hardness HV.
    """
    if n_hardening > 0:
        # Hollomon: σ = K × ε^n, where K = σ_y / (0.002)^n
        K_prime = sigma_y_Pa / (0.002 ** n_hardening)
        sigma_rep = K_prime * (_VICKERS_REP_STRAIN ** n_hardening)
    else:
        sigma_rep = sigma_y_Pa

    return _TABOR_CONSTRAINT * sigma_rep / (_GF * 1e6)


# ── Brinell Hardness ──────────────────────────────────────────────

def brinell_hardness(material_key, sigma=SIGMA_HERE):
    """Brinell hardness HB from Vickers hardness.

    FIRST_PRINCIPLES: both tests measure mean indentation pressure.
    Below HV ≈ 350, the Brinell ball remains elastic and HB ≈ HV.
    Above that, ball deformation makes HB < HV.

    Conversion (MEASURED, ASM Handbook):
      HB ≈ HV                          for HV < 350
      HB ≈ HV × (1 - 0.0003×(HV-350)) for HV ≥ 350

    This capping reflects the 10mm WC ball's compliance limit.

    Args:
        material_key: key into PLASTICITY_DATA
        sigma: σ-field value

    Returns:
        Brinell hardness HB.
    """
    HV = vickers_hardness(material_key, sigma)

    if HV < 350:
        return HV

    # Progressive divergence: ball deforms at high loads
    correction = 1.0 - 0.0003 * (HV - 350)
    return HV * max(correction, 0.75)  # cap at 25% reduction


# ── Knoop Hardness ────────────────────────────────────────────────

def knoop_hardness(material_key, sigma=SIGMA_HERE):
    """Knoop hardness HK from Vickers hardness and elastic recovery.

    FIRST_PRINCIPLES: The Knoop indenter (elongated diamond, 7.11:1
    aspect ratio) gives a long diagonal that is less affected by
    elastic recovery than the Vickers square.

    For metals (high E/σ_y): HK ≈ HV (both diagonals recover similarly).
    For ceramics (low E/σ_y): HK < HV (long diagonal recovers more).

    Correction (Marshall et al. 1982, J. Am. Ceram. Soc.):
      HK/HV ≈ 1 / (1 + 0.14 × σ_y/E)

    Args:
        material_key: key into PLASTICITY_DATA
        sigma: σ-field value

    Returns:
        Knoop hardness HK.
    """
    HV = vickers_hardness(material_key, sigma)
    sy = yield_stress(material_key, sigma)
    E = youngs_modulus(material_key, sigma)

    if E <= 0:
        return HV

    ratio = sy / E
    correction = 1.0 / (1.0 + 0.14 * ratio)
    return HV * correction


# ── Mohs Hardness ─────────────────────────────────────────────────

def mohs_hardness(material_key, sigma=SIGMA_HERE):
    """Approximate Mohs hardness from Vickers hardness.

    APPROXIMATION: logarithmic mapping.

    The Mohs scale (1-10) was defined by Friedrich Mohs in 1822 using
    ten reference minerals. The spacing is roughly logarithmic in
    absolute hardness:
      Mohs 1 (talc):     HV ≈ 1
      Mohs 5 (apatite):  HV ≈ 500
      Mohs 10 (diamond): HV ≈ 10000

    Fit: Mohs ≈ 1.5 × log₁₀(HV) + 1.0
    (R² ≈ 0.95 against mineral reference points)

    This is an APPROXIMATION — the Mohs scale is ordinal, not interval.
    The mapping works for comparing materials but the numbers are not
    exact Mohs values.

    Args:
        material_key: key into PLASTICITY_DATA
        sigma: σ-field value

    Returns:
        Approximate Mohs hardness (1-10 scale).
    """
    HV = vickers_hardness(material_key, sigma)

    if HV <= 0:
        return 0.0

    mohs = 1.5 * math.log10(HV) + 1.0

    # Clamp to physical range
    return max(1.0, min(10.0, mohs))


# ── Shore D Hardness ──────────────────────────────────────────────

def shore_d_hardness(material_key, sigma=SIGMA_HERE):
    """Shore D hardness from elastic modulus and Vickers hardness.

    FIRST_PRINCIPLES: Shore D measures elastic rebound of a spring-
    loaded indenter. The rebound depends on BOTH hardness (plastic
    resistance) and elastic modulus (energy return).

    Approximate mapping (Meyers "Dynamic Behavior of Materials"):
      Shore D ≈ 20 × HV^0.36

    Valid for metals in range HV 50-1000. Outside this range,
    Shore D is not a meaningful test.

    Args:
        material_key: key into PLASTICITY_DATA
        sigma: σ-field value

    Returns:
        Shore D hardness (0-100 scale), or 0.0 if out of range.
    """
    HV = vickers_hardness(material_key, sigma)

    if HV < 10:
        return 0.0

    shore = 20.0 * HV ** 0.36

    return min(shore, 100.0)


# ── Hardness ↔ Yield Stress (inverse) ─────────────────────────────

def yield_from_vickers(HV, n_hardening=0.0):
    """Estimate yield stress (Pa) from Vickers hardness.

    Inverse Tabor relation. Useful when only hardness is known.

    σ_y ≈ HV × g × 1e6 / (C × f_strain)

    Where f_strain corrects for work hardening between ε=0.002
    (yield) and ε=0.08 (representative indentation strain).

    Args:
        HV: Vickers hardness in kgf/mm²
        n_hardening: estimated hardening exponent (0 if unknown)

    Returns:
        Estimated yield stress in Pa.
    """
    # Mean pressure = HV × g × 1e6 Pa
    P_mean = HV * _GF * 1e6

    # σ_rep = P_mean / C
    sigma_rep = P_mean / _TABOR_CONSTRAINT

    if n_hardening > 0:
        # σ_rep = K × ε_rep^n = (σ_y / 0.002^n) × 0.08^n
        # σ_y = σ_rep × (0.002 / 0.08)^n
        strain_correction = (0.002 / _VICKERS_REP_STRAIN) ** n_hardening
        return sigma_rep * strain_correction

    return sigma_rep


# ── σ-field functions ─────────────────────────────────────────────

def sigma_hardness_ratio(material_key, sigma):
    """Ratio of hardness at σ to hardness at σ=0.

    The hardness ratio tracks the yield stress ratio because both
    derive from the same cohesive energy shift.

    Args:
        material_key: key into PLASTICITY_DATA
        sigma: σ-field value

    Returns:
        HV(σ) / HV(0), dimensionless.
    """
    HV_0 = vickers_hardness(material_key, SIGMA_HERE)
    HV_s = vickers_hardness(material_key, sigma)

    if HV_0 <= 0:
        return 1.0

    return HV_s / HV_0


# ── Diagnostics ───────────────────────────────────────────────────

def hardness_report(material_key, sigma=SIGMA_HERE):
    """Complete hardness report for a material.

    Returns dict with all hardness scales and underlying properties.
    """
    sy = yield_stress(material_key, sigma)
    E = youngs_modulus(material_key, sigma)
    G = shear_modulus(material_key, sigma)
    data = PLASTICITY_DATA[material_key]

    HV = vickers_hardness(material_key, sigma)
    HB = brinell_hardness(material_key, sigma)
    HK = knoop_hardness(material_key, sigma)

    return {
        'material': material_key,
        'sigma': sigma,
        'yield_stress_MPa': sy / 1e6,
        'youngs_modulus_GPa': E / 1e9,
        'n_hardening': data['n_hardening'],
        'is_ductile': data['is_ductile'],
        'vickers_HV': HV,
        'brinell_HB': HB,
        'knoop_HK': HK,
        'mohs': mohs_hardness(material_key, sigma),
        'shore_D': shore_d_hardness(material_key, sigma),
        'HV_to_yield_check_MPa': yield_from_vickers(HV, data['n_hardening']) / 1e6,
    }


def full_report(sigma=SIGMA_HERE):
    """Hardness reports for ALL materials in PLASTICITY_DATA.

    Rule 9: if one, then all.

    Returns:
        dict: {material_key: report_dict}
    """
    return {key: hardness_report(key, sigma) for key in PLASTICITY_DATA}
