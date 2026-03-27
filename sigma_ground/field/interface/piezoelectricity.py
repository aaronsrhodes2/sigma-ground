"""
Piezoelectricity — stress-electric field coupling in non-centrosymmetric crystals.

Derivation chains:

  1. Direct Piezoelectric Effect (Curie brothers 1880, FIRST_PRINCIPLES)
     P_i = d_ijk σ_jk

     Polarization (C/m²) proportional to applied stress (Pa).
     d_ijk = piezoelectric strain coefficient (C/N = m/V).

     FIRST_PRINCIPLES: in a crystal lacking inversion symmetry, mechanical
     deformation displaces the charge centroid of each unit cell, creating
     a macroscopic polarization.

  2. Converse Piezoelectric Effect (Lippmann 1881, FIRST_PRINCIPLES)
     ε_jk = d_ijk E_i

     Strain proportional to applied electric field.
     Same d_ijk tensor (thermodynamic Maxwell relation).

     This is the basis of piezoelectric actuators, ultrasonic transducers,
     and quartz oscillators.

  3. Piezoelectric Coupling Coefficient k (FIRST_PRINCIPLES)
     k² = d² / (ε^T × s^E)

     Where:
       d = piezoelectric coefficient
       ε^T = permittivity at constant stress
       s^E = compliance at constant field

     k² represents the fraction of input energy converted between
     mechanical and electrical forms. FIRST_PRINCIPLES: energy balance.

     k² < 1 always (thermodynamic requirement).
     For quartz: k ≈ 0.1 (weak piezoelectric).
     For PZT: k ≈ 0.7 (strong piezoelectric).

  4. Resonant Frequency of Piezoelectric Element (FIRST_PRINCIPLES)
     f_r = v_sound / (2t)    (thickness mode)

     Where v_sound = √(c^D / ρ) is the stiffened sound velocity
     (c^D = elastic stiffness at constant displacement).

     Quartz crystal oscillators: stability ~10⁻⁶ to 10⁻¹² depending
     on the cut (AT-cut: temperature-compensated near 25°C).

  5. Pyroelectric Coefficient (FIRST_PRINCIPLES)
     p = dP/dT   (change of spontaneous polarization with temperature)

     All pyroelectrics are piezoelectric (but not all piezo are pyro).
     Requires a polar crystal class (10 of the 32 point groups).

  6. Electromechanical Energy Density (FIRST_PRINCIPLES)
     u = ½ d × E × σ = ½ k² × ε^T × E² = ½ k² × s^E × σ²

     Energy harvested per unit volume per stress cycle.

σ-dependence:
  Piezoelectric coefficients d_ijk are electromagnetic (set by charge
  distribution in the unit cell) → σ-INVARIANT to first order.

  However, elastic compliance s^E shifts with σ through nuclear mass
  (same mechanism as mechanical.py). This shifts:
    - Resonant frequency: f_r ∝ 1/√(ρ × s^E)
    - Coupling coefficient: k² = d²/(ε × s^E)

  The σ-bridge: quartz oscillator frequency shifts through lattice
  stiffening under σ. This is a testable prediction — a quartz clock
  near a black hole would tick at a measurably different rate.

Origin tags:
  - Direct/converse effects: FIRST_PRINCIPLES (crystal symmetry + Hooke)
  - Coupling coefficient: FIRST_PRINCIPLES (energy balance)
  - Resonant frequency: FIRST_PRINCIPLES (standing wave)
  - Piezoelectric coefficients: MEASURED
  - σ-dependence: CORE (through elastic compliance shift)
"""

import math
from ..constants import EPS_0


# ── Piezoelectric Material Database ──────────────────────────────
# d: piezoelectric charge coefficient (C/N = m/V = pC/N × 10⁻¹²)
# epsilon_r: relative permittivity (at constant stress)
# s_E: elastic compliance at constant E-field (m²/N = 1/Pa)
# density: kg/m³
# v_sound: stiffened sound velocity (m/s)
#
# MEASURED: IEEE Standard on Piezoelectricity (1987), manufacturer data.
# Sources: Jaffe, Cook & Jaffe "Piezoelectric Ceramics" (1971),
#          Ikeda "Fundamentals of Piezoelectricity" (1990),
#          APC International "Piezoelectric Ceramics" (2011).

PIEZO_MATERIALS = {
    'quartz': {
        'name': 'α-Quartz (SiO₂)',
        'd_pC_N': 2.31,            # d₁₁ coefficient; very small
        'epsilon_r': 4.5,          # ε₁₁
        's_E_m2_N': 12.77e-12,     # s₁₁
        'density_kg_m3': 2650,
        'v_sound_m_s': 5740,       # AT-cut thickness mode
        'k': 0.10,                 # coupling coefficient
        'symmetry': '32',          # point group
    },
    'PZT4': {
        'name': 'Lead Zirconate Titanate (PZT-4, hard)',
        'd_pC_N': 289,             # d₃₃ (strong axis)
        'epsilon_r': 1300,
        's_E_m2_N': 12.3e-12,      # s₃₃
        'density_kg_m3': 7500,
        'v_sound_m_s': 4600,
        'k': 0.70,                 # k₃₃
        'symmetry': '4mm',
    },
    'PZT5A': {
        'name': 'Lead Zirconate Titanate (PZT-5A, soft)',
        'd_pC_N': 374,             # d₃₃
        'epsilon_r': 1700,
        's_E_m2_N': 16.4e-12,
        'density_kg_m3': 7750,
        'v_sound_m_s': 4350,
        'k': 0.71,
        'symmetry': '4mm',
    },
    'BaTiO3': {
        'name': 'Barium Titanate',
        'd_pC_N': 190,             # d₃₃
        'epsilon_r': 1700,
        's_E_m2_N': 8.55e-12,
        'density_kg_m3': 5700,
        'v_sound_m_s': 5200,
        'k': 0.49,
        'symmetry': '4mm',
    },
    'LiNbO3': {
        'name': 'Lithium Niobate',
        'd_pC_N': 6.0,             # d₃₃
        'epsilon_r': 28.7,
        's_E_m2_N': 5.78e-12,
        'density_kg_m3': 4640,
        'v_sound_m_s': 7340,       # Z-cut
        'k': 0.17,
        'symmetry': '3m',
    },
    'AlN': {
        'name': 'Aluminum Nitride',
        'd_pC_N': 5.5,             # d₃₃
        'epsilon_r': 10.0,
        's_E_m2_N': 3.53e-12,
        'density_kg_m3': 3260,
        'v_sound_m_s': 11000,      # very high — MEMS applications
        'k': 0.065,
        'symmetry': '6mm',
    },
    'PVDF': {
        'name': 'Polyvinylidene Fluoride (polymer)',
        'd_pC_N': 33,              # d₃₁ (in-plane)
        'epsilon_r': 12.0,
        's_E_m2_N': 365e-12,       # very compliant — polymer
        'density_kg_m3': 1780,
        'v_sound_m_s': 2260,
        'k': 0.12,
        'symmetry': 'mm2',
    },
}


# ── Direct Piezoelectric Effect ──────────────────────────────────

def piezoelectric_polarization(material_key, stress_Pa):
    """Polarization from applied stress (C/m²).

    P = d × σ

    FIRST_PRINCIPLES: direct piezoelectric effect.
    Linear response — valid for stresses well below mechanical failure.

    Args:
        material_key: key into PIEZO_MATERIALS
        stress_Pa: applied stress (Pa)

    Returns:
        Polarization in C/m²
    """
    d = PIEZO_MATERIALS[material_key]['d_pC_N'] * 1e-12  # pC/N → C/N
    return d * stress_Pa


def piezoelectric_voltage(material_key, stress_Pa, thickness_m):
    """Open-circuit voltage from applied stress (V).

    V = P × t / (ε₀ ε_r) = d × σ × t / (ε₀ ε_r)

    FIRST_PRINCIPLES: E = P/ε, V = E × t.

    Args:
        material_key: key into PIEZO_MATERIALS
        stress_Pa: applied stress (Pa)
        thickness_m: element thickness (m)

    Returns:
        Voltage in Volts
    """
    data = PIEZO_MATERIALS[material_key]
    d = data['d_pC_N'] * 1e-12
    eps_r = data['epsilon_r']

    P = d * stress_Pa
    return P * thickness_m / (EPS_0 * eps_r)


# ── Converse Piezoelectric Effect ────────────────────────────────

def piezoelectric_strain(material_key, E_field):
    """Strain from applied electric field (dimensionless).

    ε = d × E

    FIRST_PRINCIPLES: converse piezoelectric effect.
    Same d coefficient as direct effect (thermodynamic identity).

    Args:
        material_key: key into PIEZO_MATERIALS
        E_field: electric field (V/m)

    Returns:
        Strain (dimensionless)
    """
    d = PIEZO_MATERIALS[material_key]['d_pC_N'] * 1e-12  # m/V
    return d * E_field


def piezoelectric_displacement(material_key, E_field, length_m):
    """Displacement from applied field (m).

    δ = ε × L = d × E × L

    Args:
        material_key: key into PIEZO_MATERIALS
        E_field: electric field (V/m)
        length_m: element length in field direction (m)

    Returns:
        Displacement in metres
    """
    strain = piezoelectric_strain(material_key, E_field)
    return strain * length_m


# ── Coupling and Energy ─────────────────────────────────────────

def coupling_coefficient(material_key):
    """Electromechanical coupling coefficient k.

    k² = d² / (ε₀ ε_r × s_E)

    FIRST_PRINCIPLES: fraction of energy converted between
    mechanical and electrical domains.

    We return the MEASURED k value from the database (more accurate
    than computing from d, ε, s individually).

    Args:
        material_key: key into PIEZO_MATERIALS

    Returns:
        k (dimensionless, 0 to 1)
    """
    return PIEZO_MATERIALS[material_key]['k']


def coupling_coefficient_computed(material_key):
    """Coupling coefficient computed from d, ε, s.

    k² = d² / (ε₀ ε_r × s_E)

    FIRST_PRINCIPLES: energy balance. Useful for checking consistency
    of the material data.

    Args:
        material_key: key into PIEZO_MATERIALS

    Returns:
        k_computed (dimensionless)
    """
    data = PIEZO_MATERIALS[material_key]
    d = data['d_pC_N'] * 1e-12  # C/N
    eps = EPS_0 * data['epsilon_r']
    s = data['s_E_m2_N']  # m²/N

    k_sq = d ** 2 / (eps * s)
    return math.sqrt(k_sq)


def energy_density_harvested(material_key, stress_Pa):
    """Energy density harvested per stress cycle (J/m³).

    u = ½ k² × s_E × σ²

    FIRST_PRINCIPLES: electromechanical energy conversion.
    This is the electrical energy extracted per unit volume per cycle,
    assuming optimal impedance matching.

    Args:
        material_key: key into PIEZO_MATERIALS
        stress_Pa: peak stress in cycle (Pa)

    Returns:
        Energy density in J/m³
    """
    data = PIEZO_MATERIALS[material_key]
    k = data['k']
    s = data['s_E_m2_N']

    return 0.5 * k ** 2 * s * stress_Pa ** 2


# ── Resonant Frequency ──────────────────────────────────────────

def resonant_frequency_thickness(material_key, thickness_m):
    """Fundamental thickness-mode resonant frequency (Hz).

    f_r = v_sound / (2t)

    FIRST_PRINCIPLES: standing wave in thickness direction.
    Uses stiffened sound velocity (at constant D, not constant E).

    Args:
        material_key: key into PIEZO_MATERIALS
        thickness_m: element thickness (m)

    Returns:
        Resonant frequency in Hz
    """
    v = PIEZO_MATERIALS[material_key]['v_sound_m_s']
    return v / (2.0 * thickness_m)


def quartz_frequency(thickness_m):
    """Quartz crystal oscillator frequency (Hz).

    AT-cut quartz: f × t = N_t = 1.661 MHz·mm (frequency constant)

    This is the industry standard. AT-cut has near-zero temperature
    coefficient at 25°C (cubic dependence ~±1 ppm over -40 to +85°C).

    MEASURED: frequency constant for AT-cut α-quartz.

    Args:
        thickness_m: quartz blank thickness (m)

    Returns:
        Frequency in Hz
    """
    # N_t = 1661 Hz·m (= 1.661 MHz·mm)
    N_t = 1661.0  # Hz·m
    return N_t / thickness_m


# ── σ-Dependence ─────────────────────────────────────────────────

def sigma_resonant_frequency_shift(material_key, thickness_m, sigma):
    """Resonant frequency shift under σ-field.

    f_r ∝ v_sound / t = √(c^D / ρ) / t

    Under σ:
      - Elastic stiffness c^D shifts through nuclear mass (heavier lattice)
      - Density ρ shifts through nuclear mass
      - Net: v_sound(σ) = v_sound(0) × √(c_ratio / mass_ratio)

    The elastic stiffness increases slightly (stiffer bonds from ZPE shift)
    but density increases more (linear in mass). Net: frequency decreases.

    A quartz clock near a neutron star runs slower — not from time dilation
    (that's GR), but from the lattice being physically heavier.

    CORE: σ-dependence through nuclear mass → elastic + density shifts.

    Args:
        material_key: key into PIEZO_MATERIALS
        thickness_m: element thickness (m)
        sigma: σ-field value

    Returns:
        (f_0, f_sigma) tuple in Hz
    """
    from ..scale import scale_ratio
    from ..constants import PROTON_QCD_FRACTION

    f_0 = resonant_frequency_thickness(material_key, thickness_m)

    if sigma == 0.0:
        return (f_0, f_0)

    f_qcd = PROTON_QCD_FRACTION
    mass_ratio = (1.0 - f_qcd) + f_qcd * scale_ratio(sigma)

    # Stiffness shifts through ZPE correction (~1% of E_coh)
    f_zpe = 0.01
    stiffness_ratio = 1.0 + f_zpe * (1.0 - 1.0 / math.sqrt(mass_ratio))

    # v_sound ∝ √(c/ρ): stiffness in numerator, density (∝ mass) in denominator
    # f_r(σ)/f_r(0) = √(stiffness_ratio / mass_ratio)
    f_sigma = f_0 * math.sqrt(stiffness_ratio / mass_ratio)

    return (f_0, f_sigma)


# ── Nagatha Integration ──────────────────────────────────────────

def material_piezoelectric_properties(material_key, sigma=0.0):
    """Export piezoelectric properties in Nagatha-compatible format."""
    data = PIEZO_MATERIALS[material_key]
    ref_stress = 1e6  # 1 MPa reference stress
    ref_thickness = 1e-3  # 1 mm reference thickness

    P = piezoelectric_polarization(material_key, ref_stress)
    V = piezoelectric_voltage(material_key, ref_stress, ref_thickness)
    f_r = resonant_frequency_thickness(material_key, ref_thickness)
    u = energy_density_harvested(material_key, ref_stress)

    result = {
        'name': data['name'],
        'symmetry': data['symmetry'],
        'd_pC_N': data['d_pC_N'],
        'epsilon_r': data['epsilon_r'],
        'coupling_k': data['k'],
        'polarization_at_1MPa_C_m2': P,
        'voltage_at_1MPa_1mm_V': V,
        'resonant_freq_1mm_Hz': f_r,
        'energy_density_at_1MPa_J_m3': u,
        'v_sound_m_s': data['v_sound_m_s'],
        'sigma': sigma,
    }

    if sigma != 0.0:
        f_0, f_s = sigma_resonant_frequency_shift(
            material_key, ref_thickness, sigma)
        result['resonant_freq_sigma_Hz'] = f_s
        result['frequency_shift_ratio'] = f_s / f_0

    result['origin_tag'] = (
        "FIRST_PRINCIPLES: direct/converse piezoelectric effect (crystal symmetry). "
        "FIRST_PRINCIPLES: coupling coefficient (energy balance). "
        "FIRST_PRINCIPLES: thickness-mode resonance (standing wave). "
        "MEASURED: d, ε, s coefficients (IEEE Std 176). "
        "CORE: σ-dependence through elastic compliance + density shift."
    )
    return result
