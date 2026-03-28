"""
Magnetic hysteresis from atomic-scale physics.

Derivation chain:
  σ → nuclear mass → exchange coupling → Curie temperature
  Crystal structure → domain pinning → coercivity → B-H loop

History-dependent magnetization: while magnetism.py gives equilibrium
susceptibility and Curie temperatures, this module captures the
irreversible, path-dependent behavior of the B-H loop.

Physics overview:

  1. Langevin function (anhysteretic magnetization backbone)
     L(x) = coth(x) - 1/x,  L(x→0) ≈ x/3 - x³/45

     FIRST_PRINCIPLES: thermal-statistical alignment of magnetic moments
     in an applied field. The Boltzmann factor gives the Brillouin/Langevin
     distribution. This is the equilibrium curve — no hysteresis.

     M_anh(H) = M_sat × L(μ₀ m H / k_B T)

     where m = magnetic moment per domain (we use M_sat / n_atoms, order
     of μ_B scale). APPROXIMATION: mean-field; neglects domain structure.

  2. Hysteresis via coercivity shift (Jiles-Atherton inspired)
     The true B-H loop is history-dependent. A simplified but
     physically motivated model shifts the anhysteretic curve by ±H_c:

       Ascending branch:  M(H) = M_anh(H - H_c)
       Descending branch: M(H) = M_anh(H + H_c)

     APPROXIMATION: This captures the essential topology of the loop
     (remanence, coercivity, saturation) without the full Jiles-Atherton
     differential equation. Accurate for qualitative use.

  3. B-H loop and energy loss
     B = μ₀ (H + M)
     Energy loss per cycle ≈ area of loop ≈ 4 μ₀ H_c M_sat

     APPROXIMATION: rectangular loop approximation. Underestimates for
     soft magnets (rounded loops), overestimates for very hard magnets.

  4. Curie-Weiss susceptibility (above T_C)
     χ = C / (T - T_C)

     FIRST_PRINCIPLES: mean-field theory (Weiss 1907). The exchange
     field proportional to M causes divergence at T_C. The Curie
     constant C = μ₀ n μ_eff² / (3 k_B).

  5. Order-parameter temperature dependence (below T_C)
     M(T)/M_sat ≈ (1 - T/T_C)^β,  β ≈ 0.34 (3D Heisenberg universality)

     FIRST_PRINCIPLES: renormalization group gives universal critical
     exponent β ≈ 0.326 (3D Ising) to 0.365 (3D Heisenberg). We use
     β = 0.34 as an intermediate value appropriate for metallic ferromagnets.

  6. σ-dependence through exchange coupling
     Exchange coupling J_ex ∝ orbital overlap ∝ E_coh (cohesive energy).
     T_C(σ) = T_C(0) × E_coh(σ) / E_coh(0)
     E_coh(σ) = E_coh(0) × [1 + f_QCD × (e^σ - 1)]

     CORE: σ → nuclear mass → lattice stiffening → orbital overlap → T_C.
     At Earth (σ ~ 7×10⁻¹⁰): shift < 10⁻⁹, undetectable.
     At neutron star (σ ~ 0.1): T_C shifts ~0.5%.
     This is a real, in-principle testable SSBM prediction.

Origin tags:
  - Langevin function: FIRST_PRINCIPLES (statistical mechanics, Boltzmann)
  - Anhysteretic magnetization: FIRST_PRINCIPLES + APPROXIMATION (mean-field)
  - Hysteresis branches: APPROXIMATION (coercivity shift, JA-inspired)
  - Energy loss: APPROXIMATION (rectangular loop)
  - Curie-Weiss susceptibility: FIRST_PRINCIPLES (mean-field theory)
  - M(T) critical exponent: FIRST_PRINCIPLES (RG universality)
  - σ-shift of T_C: CORE (exchange coupling through E_coh)
  - Material data: MEASURED (Kittel, Coey, CRC Handbook)
"""

import math

from .surface import MATERIALS
from ..scale import scale_ratio
from ..constants import K_B, MU_0, PROTON_QCD_FRACTION, MU_BOHR, SIGMA_HERE

# Bohr magneton: μ_B = eℏ/(2m_e) ≈ 9.274e-24 J/T
_MU_BOHR = MU_BOHR


# ── Hysteresis Material Database ──────────────────────────────────
#
# All values MEASURED from experiment.
# Sources: Kittel "Introduction to Solid State Physics" (8th ed.),
#          Coey "Magnetism and Magnetic Materials" (Cambridge),
#          CRC Handbook of Chemistry and Physics.
#
# Rule 9 compliance: every material carries every field.
#
# M_sat_A_m    : saturation magnetization at ~room T (A/m). MEASURED.
# H_c_A_m      : coercivity (A/m). MEASURED for soft iron; zero for non-FM.
# B_r_T        : remanence (T). MEASURED; zero for non-FM.
# T_Curie_K    : Curie temperature (K). MEASURED; zero for non-FM.
# is_ferromagnetic : bool. True only for Fe and Ni in this dataset.
# magnetic_type    : 'ferromagnetic', 'paramagnetic', or 'diamagnetic'.

HYSTERESIS_DATA = {
    'iron': {
        # Fe: prototypical soft ferromagnet
        # M_sat measured at ~293 K: 1.707×10⁶ A/m (Coey Table 1.1)
        # H_c for annealed high-purity iron: ~80 A/m (very soft).
        # Cold-worked or alloyed specimens can be 10-100× higher.
        # B_r for annealed pure iron: ~0.8 T
        # T_Curie: 1043 K (Kittel Table 15.1)
        'M_sat_A_m': 1.71e6,
        'H_c_A_m': 80.0,
        'B_r_T': 0.8,
        'T_Curie_K': 1043.0,
        'is_ferromagnetic': True,
        'magnetic_type': 'ferromagnetic',
    },
    'nickel': {
        # Ni: soft ferromagnet, lower moment than Fe
        # M_sat at ~293 K: ~4.85×10⁵ A/m (Coey)
        # H_c for pure Ni: ~0.5 A/m (extremely soft; annealed single-crystal).
        #   Polycrystal or cold-worked: 10-100× higher.
        # B_r: ~0.3 T
        # T_Curie: 627 K (Kittel Table 15.1)
        'M_sat_A_m': 4.85e5,
        'H_c_A_m': 0.5,
        'B_r_T': 0.3,
        'T_Curie_K': 627.0,
        'is_ferromagnetic': True,
        'magnetic_type': 'ferromagnetic',
    },
    'copper': {
        # Diamagnet: no hysteresis, no spontaneous magnetization.
        'M_sat_A_m': 0.0,
        'H_c_A_m': 0.0,
        'B_r_T': 0.0,
        'T_Curie_K': 0.0,
        'is_ferromagnetic': False,
        'magnetic_type': 'diamagnetic',
    },
    'aluminum': {
        # Paramagnet: no hysteresis.
        'M_sat_A_m': 0.0,
        'H_c_A_m': 0.0,
        'B_r_T': 0.0,
        'T_Curie_K': 0.0,
        'is_ferromagnetic': False,
        'magnetic_type': 'paramagnetic',
    },
    'gold': {
        # Diamagnet: no hysteresis.
        'M_sat_A_m': 0.0,
        'H_c_A_m': 0.0,
        'B_r_T': 0.0,
        'T_Curie_K': 0.0,
        'is_ferromagnetic': False,
        'magnetic_type': 'diamagnetic',
    },
    'silicon': {
        # Diamagnet: no hysteresis.
        'M_sat_A_m': 0.0,
        'H_c_A_m': 0.0,
        'B_r_T': 0.0,
        'T_Curie_K': 0.0,
        'is_ferromagnetic': False,
        'magnetic_type': 'diamagnetic',
    },
    'tungsten': {
        # Paramagnet: no hysteresis.
        'M_sat_A_m': 0.0,
        'H_c_A_m': 0.0,
        'B_r_T': 0.0,
        'T_Curie_K': 0.0,
        'is_ferromagnetic': False,
        'magnetic_type': 'paramagnetic',
    },
    'titanium': {
        # Paramagnet: no hysteresis.
        'M_sat_A_m': 0.0,
        'H_c_A_m': 0.0,
        'B_r_T': 0.0,
        'T_Curie_K': 0.0,
        'is_ferromagnetic': False,
        'magnetic_type': 'paramagnetic',
    },
}


# ── 1. Langevin Function ───────────────────────────────────────────

def langevin_function(x):
    """Langevin function L(x) = coth(x) - 1/x.

    FIRST_PRINCIPLES: arises from the classical (continuous-moment)
    statistical mechanics of magnetic dipoles in an applied field.
    Derived by Paul Langevin (1905) from the Boltzmann distribution
    for a dipole of magnitude m in a field H:

      Z = ∫₀^π exp(m B cosθ / k_B T) sinθ dθ
      <cosθ> = L(x),  x = m B / k_B T

    Properties:
      L(0)  = 0    (no field → no net alignment)
      L(∞)  = 1    (strong field → full saturation)
      L(-x) = -L(x) (antisymmetric)
      L(x)  ≈ x/3 - x³/45 + ... for small x (Taylor series)

    The small-x Taylor expansion is used for |x| < 1e-4 to avoid
    catastrophic cancellation in coth(x) - 1/x near x = 0.

    Args:
        x: reduced field parameter (dimensionless). Can be any real number.

    Returns:
        L(x) in range (-1, 1).
    """
    if abs(x) < 1e-4:
        # Taylor: L(x) = x/3 - x³/45 + 2x⁵/945 - ...
        # Two terms are sufficient for |x| < 1e-4 (error < 10⁻²⁰)
        return x / 3.0 - x**3 / 45.0
    else:
        # Standard formula: coth(x) = cosh(x)/sinh(x)
        # Clamp to avoid overflow for very large |x|
        if x > 700.0:
            return 1.0
        if x < -700.0:
            return -1.0
        return (math.cosh(x) / math.sinh(x)) - 1.0 / x


# ── 2. Anhysteretic Magnetization ─────────────────────────────────

def anhysteretic_magnetization(material_key, H_field, T=300.0):
    """Anhysteretic (equilibrium) magnetization M_anh(H) in A/m.

    M_anh = M_sat × L(μ₀ m H / k_B T)

    FIRST_PRINCIPLES: the equilibrium M-H curve from statistical
    mechanics. "Anhysteretic" means the curve that would be traced
    if the system always reached thermal equilibrium — no pinning,
    no irreversibility. It is the backbone curve that the hysteresis
    loop orbits around.

    Derivation:
      x = μ₀ × m × H / (k_B T)
      where m is the effective magnetic moment per domain/atom (J/T).

      We use m = M_sat / n_atoms ≈ n_unpaired × μ_B, i.e. the moment
      of a single domain atom. This enters through M_sat directly:
        x = μ₀ × (M_sat / n_atoms) × H / (k_B T)

      Since n_atoms enters both numerator (via m) and denominator
      (via M_sat = n_atoms × m_per_atom), we can write:
        x = μ₀ × m_atom × H / (k_B T)

      where m_atom is estimated from M_sat and the number density from
      the MATERIALS table.

    APPROXIMATION: mean-field treatment. Real materials have domain
    walls, pinning sites, and magnetocrystalline anisotropy — all
    absent from Langevin theory. Correct for paramagnets; only a
    smooth approximation for ferromagnets.

    For non-ferromagnetic materials, returns 0 (no significant M(H)).

    Args:
        material_key: key into HYSTERESIS_DATA and MATERIALS.
        H_field: applied magnetic field H in A/m.
        T: temperature in Kelvin. Defaults to 300 K.

    Returns:
        M_anh in A/m.
    """
    hyst = HYSTERESIS_DATA[material_key]
    if not hyst['is_ferromagnetic']:
        return 0.0

    M_sat = hyst['M_sat_A_m']
    if M_sat == 0.0 or T <= 0.0:
        return 0.0

    # Estimate atomic magnetic moment from number density
    mat = MATERIALS[material_key]
    _AMU_KG = 1.66053906660e-27
    n_atoms = mat['density_kg_m3'] / (mat['A'] * _AMU_KG)
    # m_atom ≈ M_sat / n_atoms  (moment per atom at saturation)
    m_atom = M_sat / n_atoms

    # Reduced field argument: x = μ₀ m H / k_B T
    x = MU_0 * m_atom * H_field / (K_B * T)

    return M_sat * langevin_function(x)


# ── 3. Single Hysteresis Loop Point ───────────────────────────────

def hysteresis_loop_point(material_key, H_field, ascending=True, T=300.0):
    """Magnetization M at a point on the major hysteresis loop (A/m).

    Jiles-Atherton inspired shift model:
      Ascending branch:  M(H) = M_anh(H - H_c)
      Descending branch: M(H) = M_anh(H + H_c)

    APPROXIMATION: The full Jiles-Atherton (1983) model solves a
    differential equation coupling irreversible (pinned) and reversible
    (bowing) components. This simplified version captures the essential
    physics — the loop is displaced left/right by the coercivity —
    without the differential machinery. The result is topologically
    correct: ascending M < descending M at the same H, the loop
    crosses zero at ±H_c, and saturates at M_sat.

    Physical meaning of the shift: the coercive field H_c is the
    field required to nucleate/move domain walls against pinning
    obstacles (defects, grain boundaries). On the ascending branch,
    you need an additional H_c to move the wall forward; on the
    descending branch, H_c helped you in the past and must be
    overcome in reverse.

    For non-ferromagnetic materials, returns 0.

    Args:
        material_key: key into HYSTERESIS_DATA.
        H_field: applied field H in A/m.
        ascending: True for 0→H_max direction, False for H_max→-H_max.
        T: temperature in Kelvin.

    Returns:
        M in A/m.
    """
    hyst = HYSTERESIS_DATA[material_key]
    if not hyst['is_ferromagnetic']:
        return 0.0

    H_c = hyst['H_c_A_m']

    if ascending:
        return anhysteretic_magnetization(material_key, H_field - H_c, T)
    else:
        return anhysteretic_magnetization(material_key, H_field + H_c, T)


# ── 4. Full Hysteresis Loop ────────────────────────────────────────

def hysteresis_loop(material_key, H_max, steps=100, T=300.0):
    """Trace a full major hysteresis loop.

    Traces three segments:
      1. Virgin-like ascending: H = 0 → +H_max  (ascending branch)
      2. Descending:            H = +H_max → -H_max  (descending)
      3. Re-ascending:          H = -H_max → +H_max  (ascending)

    B is computed from H and M via:
      B = μ₀ (H + M)

    This is the constitutive relation in SI units. In vacuum, B = μ₀ H;
    inside matter, M adds (or subtracts for diamagnets) to give B.

    For non-ferromagnetic materials, returns a loop where M = 0
    everywhere and B = μ₀ H (linear, no hysteresis).

    Args:
        material_key: key into HYSTERESIS_DATA.
        H_max: peak applied field in A/m. Should be >> H_c for a
               full loop; typically 3–10 × H_c.
        steps: number of field steps per segment (total points = 3×steps).
        T: temperature in Kelvin.

    Returns:
        List of dicts, each with:
          'H_A_m': applied field in A/m
          'B_T':   magnetic flux density in T
          'M_A_m': magnetization in A/m
          'branch': 'ascending_initial', 'descending', or 'ascending'
    """
    if steps < 2:
        steps = 2

    result = []

    # Segment 1: ascending initial, H = 0 → +H_max
    for i in range(steps):
        H = H_max * i / (steps - 1)
        M = hysteresis_loop_point(material_key, H, ascending=True, T=T)
        B = MU_0 * (H + M)
        result.append({'H_A_m': H, 'B_T': B, 'M_A_m': M,
                       'branch': 'ascending_initial'})

    # Segment 2: descending, H = +H_max → -H_max
    for i in range(steps):
        H = H_max - 2.0 * H_max * i / (steps - 1)
        M = hysteresis_loop_point(material_key, H, ascending=False, T=T)
        B = MU_0 * (H + M)
        result.append({'H_A_m': H, 'B_T': B, 'M_A_m': M,
                       'branch': 'descending'})

    # Segment 3: ascending return, H = -H_max → +H_max
    for i in range(steps):
        H = -H_max + 2.0 * H_max * i / (steps - 1)
        M = hysteresis_loop_point(material_key, H, ascending=True, T=T)
        B = MU_0 * (H + M)
        result.append({'H_A_m': H, 'B_T': B, 'M_A_m': M,
                       'branch': 'ascending'})

    return result


# ── 5. Energy Loss Per Cycle ───────────────────────────────────────

def energy_loss_per_cycle(material_key, H_max=None):
    """Hysteresis energy loss per unit volume per cycle (J/m³).

    Area of loop ≈ 4 × μ₀ × H_c × M_sat

    APPROXIMATION: rectangular-loop approximation. Assumes the B-H
    loop is a rectangle of width 2 H_c and height 2 μ₀ M_sat.
    This overestimates for smooth (Rayleigh) loops and provides an
    upper bound that is correct in order of magnitude.

    Physical origin: every Joule in the loop area is dissipated as
    heat through domain-wall motion and irreversible magnetization
    processes (Barkhausen jumps). This is the fundamental loss
    mechanism in transformer cores and electric motors.

    The more H_max exceeds H_c, the better the rectangular approximation.
    The H_max argument is accepted but not used in the rectangular
    approximation (provided for API consistency with loop-area integration).

    For non-ferromagnetic materials, returns 0 (no hysteresis loss).

    Args:
        material_key: key into HYSTERESIS_DATA.
        H_max: peak applied field (A/m). Not used in approximation;
               kept for signature consistency. Defaults to None.

    Returns:
        Energy loss per cycle in J/m³.
    """
    hyst = HYSTERESIS_DATA[material_key]
    if not hyst['is_ferromagnetic']:
        return 0.0

    H_c = hyst['H_c_A_m']
    M_sat = hyst['M_sat_A_m']

    # Rectangular loop area: W = ∮ H dB ≈ 4 × H_c × (μ₀ M_sat)
    return 4.0 * H_c * MU_0 * M_sat


# ── 6. Curie-Weiss Susceptibility ─────────────────────────────────

def curie_weiss_susceptibility(material_key, T):
    """Curie-Weiss susceptibility χ = C / (T - T_C) for T > T_C.

    FIRST_PRINCIPLES: Weiss molecular field theory (1907). Below T_C,
    the exchange interaction creates a spontaneous internal field
    proportional to M. Above T_C this leads to a divergent
    susceptibility:

      χ = C / (T - T_C)

    The Curie constant:
      C = μ₀ × n × μ_eff² / (3 k_B)

    where n is the number density of magnetic atoms and μ_eff is the
    effective moment.

    This divergence signals the phase transition: as T → T_C from above,
    infinitesimal fields produce large magnetization — the system is on
    the verge of spontaneous ordering.

    APPROXIMATION: mean-field exponent γ = 1. True critical exponent
    (3D Heisenberg) is γ ≈ 1.4. Mean-field is accurate for T >> T_C;
    fails within ~10% of T_C.

    For non-ferromagnetic materials or T ≤ T_C, returns 0.

    Args:
        material_key: key into HYSTERESIS_DATA and MATERIALS.
        T: temperature in Kelvin. Must be > T_C for a valid result.

    Returns:
        χ (dimensionless, SI). Returns 0 if T ≤ T_C or non-ferromagnetic.
    """
    hyst = HYSTERESIS_DATA[material_key]
    if not hyst['is_ferromagnetic']:
        return 0.0

    T_C = hyst['T_Curie_K']
    if T_C <= 0.0 or T <= T_C:
        return 0.0

    M_sat = hyst['M_sat_A_m']
    if M_sat == 0.0:
        return 0.0

    mat = MATERIALS[material_key]
    _AMU_KG = 1.66053906660e-27
    n_atoms = mat['density_kg_m3'] / (mat['A'] * _AMU_KG)
    m_atom = M_sat / n_atoms

    # Curie constant C = μ₀ n μ_eff² / (3 k_B)
    # Using m_atom as μ_eff (order of magnitude)
    C_curie = MU_0 * n_atoms * m_atom**2 / (3.0 * K_B)

    return C_curie / (T - T_C)


# ── 7. Magnetization vs. Temperature ──────────────────────────────

def magnetization_vs_temperature(material_key, T):
    """Reduced magnetization M(T)/M_sat using critical exponent β ≈ 0.34.

    M(T) = M_sat × (1 - T/T_C)^β    for T < T_C
    M(T) = 0                          for T ≥ T_C

    FIRST_PRINCIPLES: renormalization group (Wilson 1972, Fisher 1974).
    Near the critical point the order parameter (magnetization) vanishes
    as a power law with a universal exponent β that depends only on
    dimension and symmetry class:

      β = 0.326  (3D Ising, 1-component order parameter)
      β = 0.365  (3D Heisenberg, 3-component order parameter)

    Real ferromagnetic metals are intermediate. We use β = 0.34, close
    to the measured value for iron (β ≈ 0.34, Kadanoff 1966).

    This power law is more accurate near T_C than Bloch's T^(3/2) law
    (which is the low-temperature spin-wave expansion). Here we use it
    across the full range as a smooth interpolation.

    At T = 0: returns M_sat (fully ordered).
    At T = T_C: returns 0 (transition point).
    Monotonically decreasing in between.

    For non-ferromagnetic materials, returns 0.

    Args:
        material_key: key into HYSTERESIS_DATA.
        T: temperature in Kelvin.

    Returns:
        M(T) in A/m. Returns M_sat at T=0, 0 at T≥T_C.
    """
    hyst = HYSTERESIS_DATA[material_key]
    if not hyst['is_ferromagnetic']:
        return 0.0

    T_C = hyst['T_Curie_K']
    M_sat = hyst['M_sat_A_m']

    if T_C <= 0.0 or M_sat == 0.0:
        return 0.0

    if T <= 0.0:
        return M_sat

    if T >= T_C:
        return 0.0

    # Critical power law: M(T) = M_sat × (1 - T/T_C)^β
    beta = 0.34  # 3D ferromagnet universality class
    return M_sat * (1.0 - T / T_C) ** beta


# ── 8. σ-Field Shift of Curie Temperature ─────────────────────────

def sigma_hysteresis_shift(material_key, sigma):
    """Effective Curie temperature T_C(σ) shifted by the σ-field.

    Derivation chain:
      σ → e^σ → QCD scale shift → nucleon mass → lattice stiffening
      → orbital overlap → exchange coupling J_ex → T_C

    The Curie temperature is set by the exchange coupling constant J_ex:
      T_C ∝ J_ex / k_B

    J_ex depends on orbital overlap between neighboring atoms. Stiffer
    lattices (heavier nuclei, σ > 0) have slightly reduced orbital
    overlap because atoms vibrate less — but the dominant effect at
    moderate σ is via the cohesive energy E_coh:

      E_coh(σ) = E_coh(0) × [1 + f_QCD × (e^σ - 1)]

    The exchange coupling scales approximately with cohesive energy
    (both measure the strength of the metallic bond):

      T_C(σ) = T_C(0) × E_coh(σ) / E_coh(0)
             = T_C(0) × [1 + f_QCD × (e^σ - 1)]

    CORE: σ → nuclear mass is the SSBM-specific coupling. The rest
    (J_ex ∝ E_coh) is a standard condensed-matter approximation.

    At Earth (σ ~ 7×10⁻¹⁰): |ΔT_C / T_C| < 10⁻⁹ — unmeasurable.
    At neutron star surface (σ ~ 0.1): ΔT_C / T_C ≈ +0.5%.

    σ = 0 is the identity: T_C(0) is unchanged.

    For non-ferromagnetic materials, returns 0.

    Args:
        material_key: key into HYSTERESIS_DATA and MATERIALS.
        sigma: σ-field value (dimensionless). 0 = standard physics.

    Returns:
        T_C(σ) in Kelvin.
    """
    hyst = HYSTERESIS_DATA[material_key]
    if not hyst['is_ferromagnetic']:
        return 0.0

    T_C_0 = hyst['T_Curie_K']
    if T_C_0 == 0.0:
        return 0.0

    # E_coh(σ) / E_coh(0) = 1 + f_QCD × (e^σ - 1)
    f_qcd = PROTON_QCD_FRACTION
    sr = scale_ratio(sigma)  # e^σ
    ratio = 1.0 + f_qcd * (sr - 1.0)

    return T_C_0 * ratio


# ── 9. Nagatha Export ─────────────────────────────────────────────

def hysteresis_properties(material_key, H_field=0.0, T=300.0, sigma=SIGMA_HERE):
    """Export hysteresis properties in Nagatha-compatible format.

    Returns a complete dict of all hysteresis-related quantities with
    honest origin tags. Suitable for downstream rendering, logging, and
    cross-module consumption.

    For non-ferromagnetic materials all magnetization and loop quantities
    are zero; the dict still contains all keys for Rule 9 compliance.

    Args:
        material_key: key into HYSTERESIS_DATA.
        H_field: applied field H in A/m for point evaluations.
        T: temperature in Kelvin.
        sigma: σ-field value.

    Returns:
        Dict with all hysteresis properties and origin tag string.
    """
    hyst = HYSTERESIS_DATA[material_key]

    M_anh = anhysteretic_magnetization(material_key, H_field, T)
    M_asc = hysteresis_loop_point(material_key, H_field, ascending=True, T=T)
    M_desc = hysteresis_loop_point(material_key, H_field, ascending=False, T=T)
    W_cyc = energy_loss_per_cycle(material_key)
    M_T = magnetization_vs_temperature(material_key, T)
    T_C_sigma = sigma_hysteresis_shift(material_key, sigma)

    # Curie-Weiss only meaningful above T_C
    T_C = hyst['T_Curie_K']
    if hyst['is_ferromagnetic'] and T > T_C and T_C > 0.0:
        chi_cw = curie_weiss_susceptibility(material_key, T)
    else:
        chi_cw = 0.0

    result = {
        'material': material_key,
        'H_field_A_m': H_field,
        'temperature_K': T,
        'sigma': sigma,
        # Material record
        'M_sat_A_m': hyst['M_sat_A_m'],
        'H_c_A_m': hyst['H_c_A_m'],
        'B_r_T': hyst['B_r_T'],
        'T_Curie_K': T_C,
        'is_ferromagnetic': hyst['is_ferromagnetic'],
        'magnetic_type': hyst['magnetic_type'],
        # Computed quantities
        'M_anhysteretic_A_m': M_anh,
        'M_ascending_A_m': M_asc,
        'M_descending_A_m': M_desc,
        'B_ascending_T': MU_0 * (H_field + M_asc),
        'B_descending_T': MU_0 * (H_field + M_desc),
        'energy_loss_per_cycle_J_m3': W_cyc,
        'magnetization_at_T_A_m': M_T,
        'curie_weiss_susceptibility': chi_cw,
        'T_Curie_sigma_K': T_C_sigma,
        'origin': (
            "M_sat, H_c, B_r, T_Curie: MEASURED (Kittel, Coey, CRC). "
            "Langevin function: FIRST_PRINCIPLES (Boltzmann statistics). "
            "Anhysteretic M(H): FIRST_PRINCIPLES + APPROXIMATION (mean-field). "
            "Hysteresis branches: APPROXIMATION (coercivity-shift, JA-inspired). "
            "Energy loss: APPROXIMATION (rectangular loop, 4 μ₀ H_c M_sat). "
            "Curie-Weiss χ: FIRST_PRINCIPLES (mean-field, γ=1). "
            "M(T) critical exponent: FIRST_PRINCIPLES (RG, β=0.34). "
            "T_C(σ): CORE (exchange coupling through E_coh, σ → QCD mass)."
        ),
    }

    return result
