"""
Magnetic properties from atomic-scale physics.

Derivation chain:
  σ → nuclear mass → Larmor precession → nuclear magnetic moment
  Z → electron config → unpaired electrons → atomic magnetic moment
  Crystal structure → exchange coupling → Curie/Néel temperature

Three categories of magnetic response, all from first principles:

  1. Diamagnetism (ALL materials)
     χ_dia = −(μ₀ e² / 6m_e) × n_atoms × ⟨r²⟩
     FIRST_PRINCIPLES: Lenz's law at the atomic level. An applied field
     induces orbital currents that oppose the field. Langevin (1905).

     ⟨r²⟩ = mean-square orbital radius, estimated from atomic number:
       ⟨r²⟩ ≈ (a₀ × Z^(−1/3))²  (Thomas-Fermi scaling)
     where a₀ = Bohr radius = ℏ/(m_e × c × α)

     APPROXIMATION: Thomas-Fermi model for ⟨r²⟩. Accurate to ±50% for
     most elements; good enough to get the right order of magnitude and
     the correct Z-dependence (heavier atoms → smaller orbits → weaker
     diamagnetism per electron, but more electrons).

  2. Paramagnetism (materials with unpaired electrons)
     χ_para = μ₀ × n × μ_eff² / (3 k_B T)
     FIRST_PRINCIPLES: Curie's law (1895). Thermal competition between
     alignment (magnetic energy μB) and randomization (k_BT).

     μ_eff = g_J × √(J(J+1)) × μ_B   (effective magnetic moment)
     where g_J = Landé g-factor, J = total angular momentum quantum number
     μ_B = eℏ/(2m_e) = Bohr magneton

     We use MEASURED unpaired electron counts and Hund's rules for g_J.

  3. Ferromagnetism (Fe, Co, Ni — exchange-coupled)
     M_sat = n_atoms × n_unpaired × μ_B
     FIRST_PRINCIPLES: each unpaired electron contributes ~1 μ_B.

     Curie temperature T_C: MEASURED (exchange coupling strength).
     Above T_C → paramagnetic. Below T_C → spontaneous magnetization.

     Temperature dependence (Bloch's T^(3/2) law):
       M(T) = M_sat × (1 − (T/T_C)^(3/2))   for T < T_C
     FIRST_PRINCIPLES: spin-wave (magnon) excitations reduce M.
     Bloch (1930), verified experimentally for all ferromagnets.

σ-dependence:
  Nuclear magnetic moments scale with 1/m_nucleon (gyromagnetic ratio):
    μ_nuclear(σ) = μ_nuclear(0) × m_p(0)/m_p(σ)
  This shifts NMR frequencies and neutron magnetic scattering.

  Electronic magnetism (μ_B, χ) is electromagnetic → σ-INVARIANT.
  But exchange coupling J_ex depends on orbital overlap, which shifts
  slightly through lattice dynamics (phonon hardening from heavier nuclei).

  At Earth (σ ~ 7×10⁻¹⁰): all corrections < 10⁻⁹, negligible.
  At neutron star surface (σ ~ 0.1): nuclear moments shift ~10%.
  Curie temperature shifts ~0.5% through lattice stiffening.

Origin tags:
  - Diamagnetic susceptibility: FIRST_PRINCIPLES (Langevin/Lenz) +
    APPROXIMATION (Thomas-Fermi ⟨r²⟩)
  - Paramagnetic susceptibility: FIRST_PRINCIPLES (Curie's law)
  - Saturation magnetization: FIRST_PRINCIPLES (μ_B counting)
  - Curie temperature: MEASURED
  - Bloch T^(3/2) law: FIRST_PRINCIPLES (spin-wave theory)
  - Nuclear moments: FIRST_PRINCIPLES (gyromagnetic ratio) +
    CORE (σ-dependence through QCD mass)
  - Unpaired electrons: MEASURED (Hund's rules + crystal field)
"""

import math
from .surface import MATERIALS
from ..scale import scale_ratio
from ..constants import (
    PROTON_QCD_FRACTION, HBAR, C, E_CHARGE, K_B, M_ELECTRON_KG,
    BOHR_RADIUS, SIGMA_HERE,
)

# ── Fundamental Constants ─────────────────────────────────────────
_K_BOLTZMANN = K_B
_ELECTRON_MASS = M_ELECTRON_KG
_MU_0 = 1.25663706212e-6          # N/A² (vacuum permeability)
_BOHR_RADIUS = BOHR_RADIUS

# Bohr magneton: μ_B = eℏ/(2m_e)
MU_BOHR = E_CHARGE * HBAR / (2.0 * _ELECTRON_MASS)  # ≈ 9.274e-24 J/T

# Nuclear magneton: μ_N = eℏ/(2m_p)
_PROTON_MASS = 1.67262192369e-27  # kg
MU_NUCLEAR = E_CHARGE * HBAR / (2.0 * _PROTON_MASS)  # ≈ 5.051e-27 J/T


# ── Magnetic Material Database ────────────────────────────────────
# All values are MEASURED from experiment.
# Sources: Kittel "Intro to Solid State Physics", CRC Handbook,
#          Coey "Magnetism and Magnetic Materials"
#
# n_unpaired: number of unpaired d-electrons per atom (Hund's rules)
# curie_temp_K: Curie temperature (ferromagnets only)
# magnetic_type: 'ferromagnetic', 'paramagnetic', or 'diamagnetic'
# nuclear_moment_mu_N: nuclear magnetic moment in nuclear magnetons
#   (MEASURED from NMR; sign gives spin orientation)

MAGNETIC_DATA = {
    'iron': {
        'n_unpaired': 4,           # [Ar] 3d⁶ 4s² → 4 unpaired in 3d
        'curie_temp_K': 1043.0,    # Curie temperature (MEASURED)
        'magnetic_type': 'ferromagnetic',
        'measured_moment_mu_B': 2.22,  # per atom (reduced from free-ion 4 by band effects)
        'nuclear_moment_mu_N': 0.0906,  # ⁵⁶Fe ground state
    },
    'nickel': {
        'n_unpaired': 2,           # [Ar] 3d⁸ 4s² → 2 unpaired in 3d
        'curie_temp_K': 627.0,
        'magnetic_type': 'ferromagnetic',
        'measured_moment_mu_B': 0.616,  # per atom (strongly reduced by band effects)
        'nuclear_moment_mu_N': -0.750,  # ⁶¹Ni
    },
    'copper': {
        'n_unpaired': 0,           # [Ar] 3d¹⁰ 4s¹ → all paired
        'curie_temp_K': 0.0,
        'magnetic_type': 'diamagnetic',
        'measured_moment_mu_B': 0.0,
        'nuclear_moment_mu_N': 2.2233,  # ⁶³Cu
    },
    'aluminum': {
        'n_unpaired': 1,           # [Ne] 3s² 3p¹ → 1 unpaired (weak para)
        'curie_temp_K': 0.0,
        'magnetic_type': 'paramagnetic',
        'measured_moment_mu_B': 0.0,  # too weak to measure bulk moment
        'nuclear_moment_mu_N': 3.6415,  # ²⁷Al
    },
    'gold': {
        'n_unpaired': 0,           # [Xe] 4f¹⁴ 5d¹⁰ 6s¹ → all paired
        'curie_temp_K': 0.0,
        'magnetic_type': 'diamagnetic',
        'measured_moment_mu_B': 0.0,
        'nuclear_moment_mu_N': 0.1458,  # ¹⁹⁷Au
    },
    'silicon': {
        'n_unpaired': 0,           # diamond cubic: all sp³ bonds paired
        'curie_temp_K': 0.0,
        'magnetic_type': 'diamagnetic',
        'measured_moment_mu_B': 0.0,
        'nuclear_moment_mu_N': -0.5553,  # ²⁹Si
    },
    'tungsten': {
        'n_unpaired': 4,           # [Xe] 4f¹⁴ 5d⁴ 6s² → 4 unpaired in 5d
        'curie_temp_K': 0.0,
        'magnetic_type': 'paramagnetic',
        'measured_moment_mu_B': 0.0,  # paramagnetic, no bulk moment
        'nuclear_moment_mu_N': 0.1178,  # ¹⁸³W
    },
    'titanium': {
        'n_unpaired': 2,           # [Ar] 3d² 4s² → 2 unpaired in 3d
        'curie_temp_K': 0.0,
        'magnetic_type': 'paramagnetic',
        'measured_moment_mu_B': 0.0,  # paramagnetic
        'nuclear_moment_mu_N': -0.7885,  # ⁴⁷Ti
    },
}


# ── Diamagnetic Susceptibility ────────────────────────────────────

def mean_square_orbital_radius(Z):
    """Mean-square orbital radius ⟨r²⟩ from Thomas-Fermi scaling.

    ⟨r²⟩ ≈ Z_eff × (a₀)² × Z^(−2/3)

    APPROXIMATION: Thomas-Fermi model. The total ⟨r²⟩ summed over
    all Z electrons scales as Z × r_atom², where r_atom ∝ Z^(−1/3).
    So total ⟨r²⟩ ∝ Z × Z^(−2/3) = Z^(1/3).

    More precisely: ⟨r²⟩_total = Z × ⟨r²⟩_per_electron
    and ⟨r²⟩_per_electron ≈ a₀² × Z^(−2/3)

    Returns:
        ⟨r²⟩_total in m² (summed over all electrons in the atom)
    """
    r_per_electron_sq = _BOHR_RADIUS**2 * Z**(-2.0 / 3.0)
    return Z * r_per_electron_sq


def diamagnetic_susceptibility(material_key):
    """Volume diamagnetic susceptibility χ_dia (dimensionless, SI).

    χ_dia = −(μ₀ e² / 6m_e) × n_atoms × ⟨r²⟩_total

    FIRST_PRINCIPLES: Langevin diamagnetism (1905). Every material
    has this contribution. It's the atomic-scale Lenz's law: applied
    field → induced orbital currents → opposing field.

    Always negative (opposes applied field).
    Typical magnitude: ~10⁻⁶ to 10⁻⁵.

    APPROXIMATION: Thomas-Fermi ⟨r²⟩. Gets the right order of magnitude
    and Z-dependence. Real values need Hartree-Fock wavefunctions.

    Args:
        material_key: key into MATERIALS dict

    Returns:
        χ_dia (dimensionless, negative)
    """
    mat = MATERIALS[material_key]
    Z = mat['Z']
    density = mat['density_kg_m3']
    A = mat['A']

    # Number density (atoms/m³)
    _AMU_KG = 1.66053906660e-27
    n_atoms = density / (A * _AMU_KG)

    r2_total = mean_square_orbital_radius(Z)

    # Langevin formula
    chi = -(_MU_0 * E_CHARGE**2 / (6.0 * _ELECTRON_MASS)) * n_atoms * r2_total

    return chi


# ── Paramagnetic Susceptibility ───────────────────────────────────

def paramagnetic_susceptibility(material_key, T=300.0):
    """Volume paramagnetic susceptibility χ_para from Curie's law.

    χ_para = μ₀ × n × μ_eff² / (3 k_B T)

    FIRST_PRINCIPLES: Curie's law (1895). Magnetic dipoles in thermal
    equilibrium. The competition between alignment energy (μB) and
    thermal randomization (k_BT) gives χ ∝ 1/T.

    μ_eff = √(n_unpaired × (n_unpaired + 2)) × μ_B
    This is the spin-only formula: μ_eff = g_S × √(S(S+1)) × μ_B
    where S = n_unpaired/2 and g_S = 2.

    APPROXIMATION: spin-only (ignores orbital contribution). Good for
    3d transition metals where crystal field quenches L. Less accurate
    for 4f rare earths (we don't have those in our database).

    Only applies to materials with unpaired electrons that are NOT
    ferromagnetically ordered (i.e., paramagnets, or ferromagnets above T_C).

    Args:
        material_key: key into MATERIALS dict
        T: temperature in Kelvin

    Returns:
        χ_para (dimensionless, positive). Returns 0 for diamagnets.
    """
    if T <= 0:
        return 0.0

    mag = MAGNETIC_DATA.get(material_key)
    if mag is None:
        return 0.0

    n_unpaired = mag['n_unpaired']
    if n_unpaired == 0:
        return 0.0

    # For ferromagnets below T_C, Curie's law doesn't apply
    # (spontaneous magnetization dominates)
    if mag['magnetic_type'] == 'ferromagnetic' and T < mag['curie_temp_K']:
        return 0.0

    mat = MATERIALS[material_key]
    density = mat['density_kg_m3']
    A = mat['A']
    _AMU_KG = 1.66053906660e-27
    n_atoms = density / (A * _AMU_KG)

    # Spin-only effective moment: μ_eff = √(n(n+2)) × μ_B
    mu_eff = math.sqrt(n_unpaired * (n_unpaired + 2)) * MU_BOHR

    # Curie's law
    chi = _MU_0 * n_atoms * mu_eff**2 / (3.0 * _K_BOLTZMANN * T)

    return chi


# ── Total Susceptibility ──────────────────────────────────────────

def magnetic_susceptibility(material_key, T=300.0):
    """Total volume magnetic susceptibility χ (dimensionless, SI).

    χ = χ_dia + χ_para

    For diamagnets: χ < 0 (diamagnetic term dominates).
    For paramagnets: χ > 0 (paramagnetic term overwhelms diamagnetic).
    For ferromagnets below T_C: not captured by linear susceptibility
    (use saturation_magnetization instead).

    Args:
        material_key: key into MATERIALS dict
        T: temperature in Kelvin

    Returns:
        χ (dimensionless). Negative for diamagnets, positive for paramagnets.
    """
    chi_dia = diamagnetic_susceptibility(material_key)
    chi_para = paramagnetic_susceptibility(material_key, T)
    return chi_dia + chi_para


# ── Ferromagnetic Properties ──────────────────────────────────────

def saturation_magnetization(material_key):
    """Saturation magnetization M_sat (A/m) at T = 0 K.

    M_sat = n_atoms × n_unpaired × μ_B

    FIRST_PRINCIPLES: every unpaired electron contributes one Bohr
    magneton when fully aligned. This is the maximum possible
    magnetization.

    APPROXIMATION: uses free-ion unpaired electron count. In metals,
    band structure reduces the effective moment. Iron: free-ion gives
    4 μ_B/atom, measured is 2.22 μ_B/atom. We compute the theoretical
    maximum and note the gap.

    For non-ferromagnetic materials, returns 0.

    Args:
        material_key: key into MATERIALS dict

    Returns:
        M_sat in A/m. Returns 0 for non-ferromagnets.
    """
    mag = MAGNETIC_DATA.get(material_key)
    if mag is None or mag['magnetic_type'] != 'ferromagnetic':
        return 0.0

    mat = MATERIALS[material_key]
    density = mat['density_kg_m3']
    A = mat['A']
    _AMU_KG = 1.66053906660e-27
    n_atoms = density / (A * _AMU_KG)

    n_unpaired = mag['n_unpaired']
    return n_atoms * n_unpaired * MU_BOHR


def saturation_magnetization_measured(material_key):
    """Saturation magnetization using MEASURED moment per atom.

    M_sat = n_atoms × μ_measured

    Uses the experimentally measured moment per atom, which accounts
    for band structure effects that reduce the moment below the
    free-ion value.

    Args:
        material_key: key into MATERIALS dict

    Returns:
        M_sat in A/m. Returns 0 for non-ferromagnets.
    """
    mag = MAGNETIC_DATA.get(material_key)
    if mag is None or mag['magnetic_type'] != 'ferromagnetic':
        return 0.0

    mat = MATERIALS[material_key]
    density = mat['density_kg_m3']
    A = mat['A']
    _AMU_KG = 1.66053906660e-27
    n_atoms = density / (A * _AMU_KG)

    mu_measured = mag['measured_moment_mu_B'] * MU_BOHR
    return n_atoms * mu_measured


def magnetization_at_temperature(material_key, T):
    """Magnetization M(T) for a ferromagnet using Bloch's T^(3/2) law.

    M(T) = M_sat × (1 − (T/T_C)^(3/2))   for T < T_C
    M(T) = 0                                for T ≥ T_C

    FIRST_PRINCIPLES: Bloch (1930). At finite temperature, spin waves
    (magnons) are thermally excited. Each magnon flips one spin,
    reducing M. The density of magnon states gives the T^(3/2) law.

    This is the low-temperature approximation. Near T_C, mean-field
    theory gives M ∝ (1 − T/T_C)^β with β ≈ 0.34 (3D Heisenberg).
    We use the Bloch law which is more accurate at T < 0.5 T_C.

    Uses measured moment for M_sat (band-corrected).

    Args:
        material_key: key into MATERIALS dict
        T: temperature in Kelvin

    Returns:
        M(T) in A/m. Returns 0 above T_C or for non-ferromagnets.
    """
    mag = MAGNETIC_DATA.get(material_key)
    if mag is None or mag['magnetic_type'] != 'ferromagnetic':
        return 0.0

    T_C = mag['curie_temp_K']
    if T_C <= 0 or T >= T_C:
        return 0.0

    M_sat = saturation_magnetization_measured(material_key)
    return M_sat * (1.0 - (T / T_C) ** 1.5)


def curie_temperature(material_key):
    """Curie temperature T_C in Kelvin.

    MEASURED. The temperature above which a ferromagnet becomes
    paramagnetic. Determined by the strength of exchange coupling
    between neighboring magnetic atoms.

    Returns 0 for non-ferromagnets.
    """
    mag = MAGNETIC_DATA.get(material_key)
    if mag is None:
        return 0.0
    return mag['curie_temp_K']


def is_ferromagnetic(material_key, T=300.0):
    """Whether the material is ferromagnetically ordered at temperature T."""
    mag = MAGNETIC_DATA.get(material_key)
    if mag is None:
        return False
    if mag['magnetic_type'] != 'ferromagnetic':
        return False
    return T < mag['curie_temp_K']


# ── Nuclear Magnetic Moments & σ-dependence ───────────────────────

def nuclear_magnetic_moment(material_key, sigma=SIGMA_HERE):
    """Nuclear magnetic moment at arbitrary σ (in nuclear magnetons).

    The nuclear magnetic moment depends on the gyromagnetic ratio:
      γ = μ / (I × ℏ)
    and the gyromagnetic ratio scales as 1/m_nucleon:
      γ(σ) = γ(0) × m_p(0) / m_p(σ)

    Since m_p(σ) = m_bare + m_QCD × e^σ, and m_QCD ≈ 0.99 × m_p:
      μ(σ) = μ(0) / [(1 − f_QCD) + f_QCD × e^σ]

    This is a REAL testable SSBM prediction: NMR frequencies shift
    in strong gravitational fields. A precision NMR experiment near
    a neutron star would measure this directly.

    At Earth: shift < 10⁻⁹ (undetectable with current tech).
    At neutron star surface (σ ~ 0.1): ~10% shift (detectable!).

    Args:
        material_key: key into MATERIALS / MAGNETIC_DATA
        sigma: σ-field value

    Returns:
        Nuclear moment in nuclear magnetons (μ_N units).
    """
    mag = MAGNETIC_DATA.get(material_key)
    if mag is None:
        return 0.0

    mu_0 = mag['nuclear_moment_mu_N']

    # Mass ratio: m_p(σ) / m_p(0)
    f_qcd = PROTON_QCD_FRACTION
    mass_ratio = (1.0 - f_qcd) + f_qcd * scale_ratio(sigma)

    # μ ∝ 1/m_nucleon → μ(σ) = μ(0) / mass_ratio
    return mu_0 / mass_ratio


def nmr_frequency_shift(material_key, sigma):
    """Fractional NMR frequency shift due to σ-field.

    ω(σ)/ω(0) = μ(σ)/μ(0) = 1 / mass_ratio

    The NMR frequency is proportional to the nuclear magnetic moment.
    Since the moment scales inversely with nucleon mass, and nucleon
    mass scales with e^σ, the NMR frequency DECREASES in strong
    gravitational fields.

    This is an SSBM-specific prediction. Standard GR predicts
    gravitational redshift (which also shifts frequencies), but the
    SSBM shift is ADDITIONAL — it comes from the mass change, not
    from the metric.

    Returns:
        ω(σ)/ω(0) − 1 (fractional shift, negative means lower frequency)
    """
    f_qcd = PROTON_QCD_FRACTION
    mass_ratio = (1.0 - f_qcd) + f_qcd * scale_ratio(sigma)
    return (1.0 / mass_ratio) - 1.0


# ── Nagatha Export ────────────────────────────────────────────────

def material_magnetic_properties(material_key, T=300.0, sigma=SIGMA_HERE):
    """Export magnetic properties in Nagatha-compatible format.

    Returns a dict with all magnetic quantities and honest origin tags.
    """
    mag = MAGNETIC_DATA.get(material_key)
    if mag is None:
        return {
            'material': material_key,
            'magnetic_type': 'unknown',
            'origin': 'No magnetic data available for this material.',
        }

    chi_dia = diamagnetic_susceptibility(material_key)
    chi_para = paramagnetic_susceptibility(material_key, T)
    chi_total = chi_dia + chi_para

    result = {
        'material': material_key,
        'temperature_K': T,
        'sigma': sigma,
        'magnetic_type': mag['magnetic_type'],
        'n_unpaired_electrons': mag['n_unpaired'],
        'diamagnetic_susceptibility': chi_dia,
        'paramagnetic_susceptibility': chi_para,
        'total_susceptibility': chi_total,
        'nuclear_moment_mu_N': nuclear_magnetic_moment(material_key, sigma),
        'nuclear_moment_sigma_0': mag['nuclear_moment_mu_N'],
        'nmr_frequency_shift': nmr_frequency_shift(material_key, sigma),
    }

    if mag['magnetic_type'] == 'ferromagnetic':
        result['curie_temperature_K'] = mag['curie_temp_K']
        result['saturation_magnetization_A_m'] = saturation_magnetization(material_key)
        result['saturation_magnetization_measured_A_m'] = saturation_magnetization_measured(material_key)
        result['magnetization_at_T_A_m'] = magnetization_at_temperature(material_key, T)
        result['is_ferromagnetic_at_T'] = is_ferromagnetic(material_key, T)

    result['origin'] = (
        "Diamagnetic susceptibility: FIRST_PRINCIPLES (Langevin) + "
        "APPROXIMATION (Thomas-Fermi ⟨r²⟩). "
        "Paramagnetic susceptibility: FIRST_PRINCIPLES (Curie's law, spin-only). "
        "Saturation magnetization: FIRST_PRINCIPLES (μ_B counting). "
        "Bloch T^(3/2) law: FIRST_PRINCIPLES (spin-wave theory). "
        "Curie temperature: MEASURED. "
        "Unpaired electrons: MEASURED (Hund's rules). "
        "Nuclear moments: MEASURED + CORE (σ-shift through QCD mass)."
    )

    return result
