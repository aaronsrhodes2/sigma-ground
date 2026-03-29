"""
Impact mechanics — coefficient of restitution from elastic-plastic contact.

Derivation chain:
  mechanical.py (Young's modulus)
  + plasticity.py (yield stress)
  → impact.py (coefficient of restitution, impact energy partition)

The coefficient of restitution (COR) is NOT an independent material property.
It is DERIVED from the stress-strain response of the contacting bodies.
It depends on material (yield stress, modulus) and impact velocity.

Derivation chains:

  1. Coefficient of Restitution (Johnson 1985, FIRST_PRINCIPLES)
     e = (v_rebound / v_impact)

     For elastic-perfectly-plastic spheres:
       e = 1                           if v < v_y (fully elastic)
       e ≈ (v_y / v)^(1/4)           if v > v_y (elastic-plastic)

     Where v_y is the yield onset velocity:
       v_y = C × (σ_y^(5/2)) / (E*^2 × √(ρ))

     E* = reduced modulus = E / (2(1-ν²)) for self-contact
     C = geometric constant from Hertz contact theory

     Johnson "Contact Mechanics" (1985), Cambridge.
     Thornton (1997), ASME J. Applied Mechanics.

  2. Energy Partition (FIRST_PRINCIPLES: energy conservation)
     E_rebound = e² × E_impact
     E_dissipated = (1 - e²) × E_impact

     The fraction (1-e²) is converted to plastic deformation, heat,
     sound, and surface damage. This is exact energy conservation.

  3. Contact Duration (Hertz 1882, FIRST_PRINCIPLES)
     t_contact ≈ 2.94 × (m_eff / (R_eff^(1/2) × E*))^(2/5) × v^(-1/5)

     The impact duration determines the frequency content of the
     collision sound and the peak deceleration force.

σ-dependence:
  COR inherits σ-dependence from yield stress and elastic modulus.
  σ → cohesive energy → moduli → σ_y → COR.
  Heavier nuclei → stiffer → higher σ_y → higher COR (more elastic).

Origin tags:
  - Hertz contact: FIRST_PRINCIPLES (elasticity theory, 1882)
  - Johnson COR model: FIRST_PRINCIPLES (contact mechanics, 1985)
  - Thornton elastic-plastic: FIRST_PRINCIPLES + MEASURED validation
  - Energy partition: FIRST_PRINCIPLES (conservation of energy)
"""

import math
from .mechanical import youngs_modulus
from .plasticity import PLASTICITY_DATA, yield_stress
from .surface import MATERIALS
from ..constants import SIGMA_HERE
from ..scale import scale_ratio


# ── Reduced Modulus ───────────────────────────────────────────────

def _reduced_modulus(material_key, sigma=SIGMA_HERE):
    """Reduced (contact) modulus E* for self-contact.

    E* = E / (2(1-ν²))

    For contact between two identical bodies. For dissimilar bodies,
    use: 1/E* = (1-ν₁²)/E₁ + (1-ν₂²)/E₂
    """
    from .mechanical import MECHANICAL_DATA
    E = youngs_modulus(material_key, sigma)
    nu = MECHANICAL_DATA[material_key]['poisson_ratio']
    return E / (2.0 * (1.0 - nu ** 2))


def reduced_modulus_pair(E1, nu1, E2, nu2):
    """Reduced modulus for contact between two dissimilar bodies.

    1/E* = (1-ν₁²)/E₁ + (1-ν₂²)/E₂

    FIRST_PRINCIPLES: Hertz contact theory (1882).

    Args:
        E1, nu1: Young's modulus (Pa) and Poisson's ratio of body 1
        E2, nu2: same for body 2

    Returns:
        Reduced modulus E* in Pa.
    """
    inv = (1.0 - nu1**2) / E1 + (1.0 - nu2**2) / E2
    if inv <= 0:
        return 0.0
    return 1.0 / inv


# ── Yield Onset Velocity ──────────────────────────────────────────

def yield_onset_velocity(material_key, radius_m=0.01, sigma=SIGMA_HERE):
    """Impact velocity at which plastic deformation begins (m/s).

    v_y = (π/2)^2 × (σ_y / E*)^(5/2) × √(E* / ρ) × C_geom

    Below v_y: fully elastic impact (e = 1.0).
    Above v_y: plastic deformation begins, e < 1.

    FIRST_PRINCIPLES: Hertz contact stress equals yield stress.
    Johnson "Contact Mechanics" (1985), §11.4.

    Args:
        material_key: key into PLASTICITY_DATA
        radius_m: sphere radius in metres (affects contact area)
        sigma: σ-field value

    Returns:
        Yield onset velocity in m/s.
    """
    sy = yield_stress(material_key, sigma)
    E_star = _reduced_modulus(material_key, sigma)
    rho = MATERIALS[material_key]['density_kg_m3']

    if E_star <= 0 or rho <= 0:
        return 0.0

    # Johnson's yield onset: when max Hertz pressure = 1.6 σ_y
    # v_y ≈ (σ_y / E*)^(5/2) × √(E*/ρ) × geometric factor
    # The geometric factor depends on the constraint factor (1.6 for
    # Hertz contact, where max pressure = 1.6 × mean pressure at yield)
    ratio = sy / E_star
    v_y = 1.56 * (ratio ** 2.5) * math.sqrt(E_star / rho)

    return v_y


# ── Coefficient of Restitution ────────────────────────────────────

def coefficient_of_restitution(material_key, velocity=1.0,
                                radius_m=0.01, sigma=SIGMA_HERE):
    """Coefficient of restitution e for sphere self-impact.

    FIRST_PRINCIPLES: Johnson-Thornton elastic-plastic contact model.

    e = 1.0                      if v ≤ v_y (fully elastic)
    e = (v_y / v)^(1/4)         if v > v_y (elastic-plastic)

    The 1/4 exponent comes from Hertz contact geometry:
    the elastic energy stored scales as v^(5/2) while total
    energy scales as v², giving restitution ∝ v^(1/4) decay.

    Accuracy: ±15% for metals (Johnson 1985, Thornton 1997).
    Less reliable for brittle materials (fracture, not plasticity).

    Args:
        material_key: key into PLASTICITY_DATA
        velocity: impact velocity in m/s
        radius_m: sphere radius in metres
        sigma: σ-field value

    Returns:
        Coefficient of restitution e ∈ [0, 1].
    """
    if velocity <= 0:
        return 1.0

    v_y = yield_onset_velocity(material_key, radius_m, sigma)

    if v_y <= 0:
        return 0.0

    if velocity <= v_y:
        return 1.0  # fully elastic

    # Johnson-Thornton: e = (v_y/v)^(1/4)
    e = (v_y / velocity) ** 0.25

    return max(0.0, min(1.0, e))


def cor_pair(E1, nu1, sy1, rho1, E2, nu2, sy2, rho2,
             velocity=1.0, radius_m=0.01):
    """COR for impact between two DISSIMILAR materials.

    Uses the weaker material's yield stress (first to yield)
    and the combined reduced modulus.

    Args:
        E1, nu1, sy1, rho1: properties of body 1
        E2, nu2, sy2, rho2: properties of body 2
        velocity: impact velocity (m/s)
        radius_m: effective contact radius (m)

    Returns:
        Coefficient of restitution e ∈ [0, 1].
    """
    E_star = reduced_modulus_pair(E1, nu1, E2, nu2)

    # Use the LOWER yield stress (this body yields first)
    sy_eff = min(sy1, sy2)

    # Effective density (from effective mass = m1*m2/(m1+m2))
    # For equal-size spheres: ρ_eff ≈ 2 * ρ1*ρ2/(ρ1+ρ2)
    rho_eff = 2.0 * rho1 * rho2 / (rho1 + rho2) if (rho1 + rho2) > 0 else 0

    if E_star <= 0 or rho_eff <= 0:
        return 0.0

    ratio = sy_eff / E_star
    v_y = 1.56 * (ratio ** 2.5) * math.sqrt(E_star / rho_eff)

    if velocity <= 0 or velocity <= v_y:
        return 1.0

    e = (v_y / velocity) ** 0.25
    return max(0.0, min(1.0, e))


# ── Energy Partition ──────────────────────────────────────────────

def impact_energy_partition(material_key, velocity=1.0, mass_kg=0.01,
                             radius_m=0.01, sigma=SIGMA_HERE):
    """Energy partition during impact.

    E_total = ½mv²
    E_rebound = e² × E_total
    E_dissipated = (1 - e²) × E_total

    FIRST_PRINCIPLES: conservation of energy.

    Args:
        material_key: key into PLASTICITY_DATA
        velocity: impact velocity (m/s)
        mass_kg: impactor mass (kg)
        radius_m: sphere radius (m)
        sigma: σ-field value

    Returns:
        dict with energy partition in Joules.
    """
    e = coefficient_of_restitution(material_key, velocity, radius_m, sigma)
    E_total = 0.5 * mass_kg * velocity ** 2

    return {
        'E_total_J': E_total,
        'E_rebound_J': e ** 2 * E_total,
        'E_dissipated_J': (1.0 - e ** 2) * E_total,
        'cor': e,
        'v_rebound_m_s': e * velocity,
    }


# ── Contact Duration ─────────────────────────────────────────────

def hertz_contact_duration(material_key, velocity=1.0, radius_m=0.01,
                            mass_kg=0.01, sigma=SIGMA_HERE):
    """Hertz elastic contact duration (seconds).

    t_c ≈ 2.94 × (m / (R^(1/2) × E*))^(2/5) × v^(-1/5)

    FIRST_PRINCIPLES: Hertz (1882) elastic contact theory.

    This determines:
    - Peak impact force: F_max ≈ m × v / t_c
    - Sound frequency on impact: f ≈ 1 / (2 × t_c)

    Args:
        material_key: key into PLASTICITY_DATA
        velocity: impact velocity (m/s)
        radius_m: sphere radius (m)
        mass_kg: mass of impactor (kg)
        sigma: σ-field value

    Returns:
        Contact duration in seconds.
    """
    E_star = _reduced_modulus(material_key, sigma)

    if E_star <= 0 or velocity <= 0 or radius_m <= 0:
        return 0.0

    # Hertz: t_c = 2.94 × (m² / (R × E*²))^(1/5) × v^(-1/5)
    term = (mass_kg ** 2 / (radius_m * E_star ** 2)) ** 0.2
    return 2.94 * term * velocity ** (-0.2)


def impact_sound_frequency(material_key, velocity=1.0, radius_m=0.01,
                            mass_kg=0.01, sigma=SIGMA_HERE):
    """Dominant frequency of impact sound (Hz).

    f ≈ 1 / (2 × t_contact)

    The impact pulse is approximately half-sinusoidal with duration
    t_contact. The dominant spectral frequency is ~1/(2t_c).

    Args:
        material_key: key into PLASTICITY_DATA
        velocity, radius_m, mass_kg, sigma: as in hertz_contact_duration

    Returns:
        Frequency in Hz.
    """
    t_c = hertz_contact_duration(material_key, velocity, radius_m,
                                  mass_kg, sigma)
    if t_c <= 0:
        return 0.0

    return 1.0 / (2.0 * t_c)


# ── σ-field function ──────────────────────────────────────────────

def sigma_cor_ratio(material_key, velocity=1.0, sigma=SIGMA_HERE):
    """Ratio of COR at σ to COR at σ=0.

    Heavier nuclei → stiffer lattice → higher yield stress → more
    elastic impacts → higher COR.

    Args:
        material_key: key into PLASTICITY_DATA
        velocity: impact velocity (m/s)
        sigma: σ-field value

    Returns:
        e(σ) / e(0), dimensionless.
    """
    e_0 = coefficient_of_restitution(material_key, velocity, sigma=SIGMA_HERE)
    e_s = coefficient_of_restitution(material_key, velocity, sigma=sigma)

    if e_0 <= 0:
        return 1.0

    return e_s / e_0


# ── Diagnostics ───────────────────────────────────────────────────

def impact_report(material_key, velocity=1.0, radius_m=0.01,
                   mass_kg=0.01, sigma=SIGMA_HERE):
    """Complete impact report for a material."""
    e = coefficient_of_restitution(material_key, velocity, radius_m, sigma)
    v_y = yield_onset_velocity(material_key, radius_m, sigma)
    t_c = hertz_contact_duration(material_key, velocity, radius_m,
                                  mass_kg, sigma)
    partition = impact_energy_partition(material_key, velocity, mass_kg,
                                        radius_m, sigma)

    return {
        'material': material_key,
        'velocity_m_s': velocity,
        'radius_m': radius_m,
        'mass_kg': mass_kg,
        'yield_onset_velocity_m_s': v_y,
        'regime': 'elastic' if velocity <= v_y else 'elastic-plastic',
        'cor': e,
        'contact_duration_s': t_c,
        'impact_sound_Hz': impact_sound_frequency(
            material_key, velocity, radius_m, mass_kg, sigma),
        **partition,
    }


def full_report(velocity=1.0, sigma=SIGMA_HERE):
    """Impact reports for ALL materials. Rule 9."""
    return {key: impact_report(key, velocity, sigma=sigma)
            for key in PLASTICITY_DATA}
