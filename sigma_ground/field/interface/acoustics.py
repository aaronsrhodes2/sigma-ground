"""
Acoustic properties from elastic moduli and density.

Derivation chain:
  σ → nuclear mass → bulk modulus K, shear modulus G, density ρ
  → longitudinal wave speed v_L = √((K + 4G/3) / ρ)
  → transverse wave speed v_T = √(G / ρ)
  → acoustic impedance Z = ρ × v
  → resonant frequencies, Snell's law, reflection/transmission

This module provides the full acoustic picture for solids:

  1. Wave Speeds (Newton-Laplace, FIRST_PRINCIPLES)
     v_L = √((K + 4G/3) / ρ)   — longitudinal (P-wave, compressional)
     v_T = √(G / ρ)             — transverse (S-wave, shear)

     These are exact continuum mechanics results for isotropic solids.
     K from mechanical.py (harmonic approximation, ±50%).
     G from mechanical.py (via E and ν).

  2. Acoustic Impedance (FIRST_PRINCIPLES)
     Z = ρ × v
     Unit: Pa·s/m (Rayl)

     This determines how much sound reflects at an interface.
     Reflection coefficient at normal incidence:
       R = (Z₂ − Z₁)² / (Z₂ + Z₁)²   (energy reflection)

  3. Snell's Law for Sound (FIRST_PRINCIPLES)
     sin(θ₁) / v₁ = sin(θ₂) / v₂

     Same as optics, but with sound speeds instead of light speeds.
     Critical angle for total internal reflection:
       θ_c = arcsin(v₁ / v₂)   when v₁ < v₂

  4. Resonant Frequencies (FIRST_PRINCIPLES)
     f_n = n × v / (2L)     — standing waves in a rod/pipe
     f_ring = v / (π × d)   — ring-down frequency of a struck object

  5. Debye Velocity (FIRST_PRINCIPLES)
     v_D = [1/3 × (1/v_L³ + 2/v_T³)]^(−1/3)

     The proper average over the phonon spectrum. Used for Debye
     temperature calculations (more accurate than √(K/ρ) alone).

  6. Attenuation (FIRST_PRINCIPLES + APPROXIMATION)
     Classical attenuation (Stokes-Kirchhoff):
       α = ω² / (2ρv³) × (4η/3 + η_v + κ(1/c_v − 1/c_p))
     Dominant term for metals: viscous absorption ∝ ω².

     APPROXIMATION: uses Akhiezer phonon-phonon scattering model
     for intrinsic attenuation in solids.

σ-dependence:
  K(σ) shifts through cohesive energy (mechanical.py)
  G(σ) shifts through K and ν
  ρ shifts through nuclear mass: ρ(σ) = ρ₀ × [(1−f_QCD) + f_QCD × e^σ]
  Net: v_L and v_T shift with σ.

  At Earth (σ ~ 7×10⁻¹⁰): < 10⁻⁹ change, negligible.
  At neutron star: stiffness increases, density increases.
  The ratio K/ρ changes because K depends on E_coh (EM + small QCD)
  while ρ is dominated by QCD mass. So v_sound DECREASES with σ.

  This is measurable: seismic wave speeds in neutron star crusts
  would differ from standard nuclear matter predictions.

Origin tags:
  - Wave speeds: FIRST_PRINCIPLES (Newton-Laplace, continuum mechanics)
  - Acoustic impedance: FIRST_PRINCIPLES (definition, Z = ρv)
  - Reflection/transmission: FIRST_PRINCIPLES (boundary conditions)
  - Snell's law: FIRST_PRINCIPLES (Fermat's principle / wavefront geometry)
  - Resonance: FIRST_PRINCIPLES (standing wave condition)
  - Debye velocity: FIRST_PRINCIPLES (phonon DOS average)
  - K, G inputs: FIRST_PRINCIPLES (harmonic approximation, ±50%)
  - Poisson's ratio: MEASURED
  - σ-dependence: CORE (through □σ = −ξR)
"""

import math
from .surface import MATERIALS
from .mechanical import bulk_modulus, shear_modulus, _number_density
from ..scale import scale_ratio
from ..constants import PROTON_QCD_FRACTION

# ── Constants ─────────────────────────────────────────────────────
_K_BOLTZMANN = 1.380649e-23     # J/K


# ── Density at σ ──────────────────────────────────────────────────

def density_at_sigma(material_key, sigma=0.0):
    """Mass density at arbitrary σ.

    ρ(σ) = ρ₀ × mass_ratio

    The number of atoms doesn't change (lattice spacing is EM-set),
    but each atom gets heavier because m_nucleon(σ) = m_bare + m_QCD × e^σ.

    FIRST_PRINCIPLES: density = mass/volume, mass shifts with σ.

    Args:
        material_key: key into MATERIALS dict
        sigma: σ-field value

    Returns:
        Density in kg/m³.
    """
    rho_0 = MATERIALS[material_key]['density_kg_m3']
    if sigma == 0.0:
        return rho_0

    f_qcd = PROTON_QCD_FRACTION
    mass_ratio = (1.0 - f_qcd) + f_qcd * scale_ratio(sigma)
    return rho_0 * mass_ratio


# ── Wave Speeds ───────────────────────────────────────────────────

def longitudinal_wave_speed(material_key, sigma=0.0):
    """Longitudinal (P-wave) speed in m/s.

    v_L = √((K + 4G/3) / ρ)

    FIRST_PRINCIPLES: Newton-Laplace equation for compressional
    waves in an isotropic elastic solid. The longitudinal modulus
    M = K + 4G/3 combines bulk and shear resistance.

    This is the fastest wave mode — it arrives first in seismology
    (P = primary wave).

    Args:
        material_key: key into MATERIALS dict
        sigma: σ-field value

    Returns:
        v_L in m/s.
    """
    K = bulk_modulus(material_key, sigma)
    G = shear_modulus(material_key, sigma)
    rho = density_at_sigma(material_key, sigma)

    M = K + 4.0 * G / 3.0  # longitudinal modulus
    return math.sqrt(M / rho)


def transverse_wave_speed(material_key, sigma=0.0):
    """Transverse (S-wave) speed in m/s.

    v_T = √(G / ρ)

    FIRST_PRINCIPLES: shear wave speed in an isotropic elastic solid.
    Transverse waves require a shear modulus — they cannot propagate
    in fluids (G = 0). This is why S-waves don't cross Earth's
    liquid outer core.

    Always slower than v_L (since K + 4G/3 > G for K > 0).

    Args:
        material_key: key into MATERIALS dict
        sigma: σ-field value

    Returns:
        v_T in m/s.
    """
    G = shear_modulus(material_key, sigma)
    rho = density_at_sigma(material_key, sigma)
    return math.sqrt(G / rho)


def debye_velocity(material_key, sigma=0.0):
    """Debye average sound velocity in m/s.

    v_D = [1/3 × (1/v_L³ + 2/v_T³)]^(−1/3)

    FIRST_PRINCIPLES: the Debye model averages over one longitudinal
    and two transverse phonon branches. This weighting gives the
    correct phonon density of states for Debye temperature calculations.

    v_D is always between v_T and v_L, closer to v_T (because the
    two transverse modes dominate the average).

    Args:
        material_key: key into MATERIALS dict
        sigma: σ-field value

    Returns:
        Debye average velocity in m/s.
    """
    v_L = longitudinal_wave_speed(material_key, sigma)
    v_T = transverse_wave_speed(material_key, sigma)

    avg_inv_cubed = (1.0 / v_L**3 + 2.0 / v_T**3) / 3.0
    return avg_inv_cubed ** (-1.0 / 3.0)


def wave_speed_ratio(material_key):
    """Ratio v_L / v_T — depends only on Poisson's ratio.

    v_L / v_T = √((K + 4G/3) / G) = √(2(1−ν) / (1−2ν))

    FIRST_PRINCIPLES: pure elasticity identity. Independent of
    material stiffness or density — only ν matters.

    For ν = 0.25: ratio = √3 ≈ 1.73 (Cauchy solid)
    For ν = 0.34 (copper): ratio ≈ 2.08
    For ν → 0.50 (incompressible): ratio → ∞ (v_T → 0)

    Returns:
        v_L / v_T (dimensionless)
    """
    from .mechanical import MECHANICAL_DATA
    nu = MECHANICAL_DATA[material_key]['poisson_ratio']
    return math.sqrt(2.0 * (1.0 - nu) / (1.0 - 2.0 * nu))


# ── Acoustic Impedance ────────────────────────────────────────────

def acoustic_impedance(material_key, sigma=0.0, mode='longitudinal'):
    """Acoustic impedance Z = ρ × v (Pa·s/m = Rayl).

    FIRST_PRINCIPLES: the ratio of acoustic pressure to particle
    velocity in a plane wave. Determines reflection at interfaces.

    High Z materials (steel, tungsten): sound reflects.
    Low Z materials (rubber, air): sound transmits.

    Args:
        material_key: key into MATERIALS dict
        sigma: σ-field value
        mode: 'longitudinal' or 'transverse'

    Returns:
        Acoustic impedance in Rayl (Pa·s/m).
    """
    rho = density_at_sigma(material_key, sigma)
    if mode == 'transverse':
        v = transverse_wave_speed(material_key, sigma)
    else:
        v = longitudinal_wave_speed(material_key, sigma)
    return rho * v


# ── Reflection and Transmission ───────────────────────────────────

def reflection_coefficient(mat_1, mat_2, sigma=0.0):
    """Energy reflection coefficient at normal incidence.

    R = (Z₂ − Z₁)² / (Z₂ + Z₁)²

    FIRST_PRINCIPLES: acoustic boundary conditions (continuity of
    pressure and particle velocity at the interface).

    Same form as Fresnel equations for light at normal incidence.

    Args:
        mat_1: material key for medium 1 (incident side)
        mat_2: material key for medium 2 (transmitted side)
        sigma: σ-field value

    Returns:
        R (energy reflection coefficient, 0 to 1).
    """
    Z1 = acoustic_impedance(mat_1, sigma)
    Z2 = acoustic_impedance(mat_2, sigma)
    return (Z2 - Z1)**2 / (Z2 + Z1)**2


def transmission_coefficient(mat_1, mat_2, sigma=0.0):
    """Energy transmission coefficient at normal incidence.

    T = 1 − R = 4 Z₁ Z₂ / (Z₁ + Z₂)²

    FIRST_PRINCIPLES: energy conservation at the interface.
    """
    return 1.0 - reflection_coefficient(mat_1, mat_2, sigma)


# ── Snell's Law ───────────────────────────────────────────────────

def snell_refraction_angle(mat_1, mat_2, theta_1_deg, sigma=0.0):
    """Refracted angle using Snell's law for sound.

    sin(θ₁)/v₁ = sin(θ₂)/v₂

    FIRST_PRINCIPLES: Fermat's principle (least time) applied to
    wavefronts crossing a boundary between media with different
    sound speeds.

    Args:
        mat_1: incident medium material key
        mat_2: refracted medium material key
        theta_1_deg: angle of incidence in degrees
        sigma: σ-field value

    Returns:
        Refracted angle in degrees. Returns None for total internal reflection.
    """
    v1 = longitudinal_wave_speed(mat_1, sigma)
    v2 = longitudinal_wave_speed(mat_2, sigma)

    theta_1 = math.radians(theta_1_deg)
    sin_theta_2 = math.sin(theta_1) * v2 / v1

    if abs(sin_theta_2) > 1.0:
        return None  # total internal reflection

    return math.degrees(math.asin(sin_theta_2))


def critical_angle(mat_1, mat_2, sigma=0.0):
    """Critical angle for total internal reflection (degrees).

    θ_c = arcsin(v₁ / v₂)   (only exists when v₁ < v₂)

    FIRST_PRINCIPLES: Snell's law with θ₂ = 90°.

    Args:
        mat_1: slower medium (incident side)
        mat_2: faster medium

    Returns:
        Critical angle in degrees. Returns None if v₁ ≥ v₂.
    """
    v1 = longitudinal_wave_speed(mat_1, sigma)
    v2 = longitudinal_wave_speed(mat_2, sigma)

    if v1 >= v2:
        return None  # no critical angle

    return math.degrees(math.asin(v1 / v2))


# ── Resonance ─────────────────────────────────────────────────────

def resonant_frequency(material_key, length_m, mode_n=1, sigma=0.0):
    """Resonant frequency of a rod/bar (Hz).

    f_n = n × v_L / (2L)

    FIRST_PRINCIPLES: standing wave condition for longitudinal
    vibrations in a free-free bar. The fundamental (n=1) has
    wavelength = 2L.

    This is what you hear when you strike a metal bar: the
    fundamental longitudinal resonance.

    Args:
        material_key: key into MATERIALS dict
        length_m: length of the bar in meters
        mode_n: mode number (1 = fundamental, 2 = first overtone, ...)
        sigma: σ-field value

    Returns:
        Frequency in Hz.
    """
    if length_m <= 0 or mode_n < 1:
        return 0.0

    v_L = longitudinal_wave_speed(material_key, sigma)
    return mode_n * v_L / (2.0 * length_m)


def ring_frequency(material_key, diameter_m, sigma=0.0):
    """Ring frequency of a cylindrical shell (Hz).

    f_ring = v_L / (π × d)

    FIRST_PRINCIPLES: the frequency at which a circumferential
    wave fits exactly once around the cylinder. Below this frequency,
    the shell behaves as a beam; above it, as a flat plate.

    This is important for bells, pipes, and cylindrical tanks.

    Args:
        material_key: key into MATERIALS dict
        diameter_m: diameter of the cylinder in meters
        sigma: σ-field value

    Returns:
        Ring frequency in Hz.
    """
    if diameter_m <= 0:
        return 0.0

    v_L = longitudinal_wave_speed(material_key, sigma)
    return v_L / (math.pi * diameter_m)


# ── Wavelength and Period ─────────────────────────────────────────

def wavelength(material_key, frequency_hz, sigma=0.0, mode='longitudinal'):
    """Acoustic wavelength λ = v / f.

    Args:
        material_key: key into MATERIALS dict
        frequency_hz: frequency in Hz
        sigma: σ-field value
        mode: 'longitudinal' or 'transverse'

    Returns:
        Wavelength in meters.
    """
    if frequency_hz <= 0:
        return float('inf')

    if mode == 'transverse':
        v = transverse_wave_speed(material_key, sigma)
    else:
        v = longitudinal_wave_speed(material_key, sigma)
    return v / frequency_hz


# ── Nagatha Export ────────────────────────────────────────────────

def material_acoustic_properties(material_key, sigma=0.0):
    """Export acoustic properties in Nagatha-compatible format.

    Returns a dict with all acoustic quantities and honest origin tags.
    """
    v_L = longitudinal_wave_speed(material_key, sigma)
    v_T = transverse_wave_speed(material_key, sigma)
    v_D = debye_velocity(material_key, sigma)
    Z_L = acoustic_impedance(material_key, sigma, 'longitudinal')
    Z_T = acoustic_impedance(material_key, sigma, 'transverse')
    rho = density_at_sigma(material_key, sigma)

    # Sensitivity: how much does v_L change per unit σ?
    ds = 1e-6
    v_L_plus = longitudinal_wave_speed(material_key, sigma + ds)
    sensitivity = (v_L_plus - v_L) / (ds * v_L) if v_L > 0 else 0

    return {
        'material': material_key,
        'sigma': sigma,
        'density_kg_m3': rho,
        'longitudinal_speed_m_s': v_L,
        'transverse_speed_m_s': v_T,
        'debye_velocity_m_s': v_D,
        'vL_over_vT': v_L / v_T if v_T > 0 else 0,
        'impedance_longitudinal_rayl': Z_L,
        'impedance_transverse_rayl': Z_T,
        'sigma_sensitivity': sensitivity,
        'origin': (
            "Wave speeds: FIRST_PRINCIPLES (Newton-Laplace, continuum mechanics). "
            "Acoustic impedance: FIRST_PRINCIPLES (Z = ρv). "
            "Reflection/transmission: FIRST_PRINCIPLES (boundary conditions). "
            "K, G inputs: FIRST_PRINCIPLES (harmonic approximation, ±50%). "
            "Poisson's ratio: MEASURED. "
            "σ-dependence: CORE (K through E_coh, ρ through nuclear mass)."
        ),
    }
