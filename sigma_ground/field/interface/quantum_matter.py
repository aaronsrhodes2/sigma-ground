"""
Quantum matter predictions — Mott transition, crystal-field spin Hamiltonians,
and material-specific quantum simulations using cascade-derived parameters.

This module bridges the gap between the material database and quantum
Hamiltonians. Instead of using arbitrary parameters (J=1, U=1), every
Hamiltonian coefficient is derived from measured material properties
through the cascade.

Three prediction pipelines:

  1. MOTT TRANSITION from cascade
     E_coh, z, a, n → t (hopping), U (screened Coulomb) → U/t → metal or insulator
     Prediction: which materials are near the Mott boundary

  2. CRYSTAL FIELD → SPIN HAMILTONIAN
     10Dq, Racah B, β → d-electron Hamiltonian → VQE → spin state, χ
     Prediction: high-spin vs low-spin crossover matches Tanabe-Sugano

  3. TANABE-SUGANO AS MOTT PHYSICS
     Key insight: the high-spin/low-spin crossover in crystal field theory
     IS a Mott transition in disguise. 10Dq plays the role of bandwidth
     (∝ t) and B plays the role of Coulomb repulsion (∝ U). The crossover
     ratio 10Dq/B predicts localized vs delocalized d-electrons — exactly
     the Mott criterion.

     This means every entry in our CRYSTAL_FIELD_10DQ_EV table is a data
     point on the Mott phase diagram, and the Tanabe-Sugano crossover
     values are predictions of the Mott boundary for each d^n configuration.

Cascade connections:
  surface.py    → E_coh, a, ρ, Z, crystal_structure
  mechanical.py → K, G (elastic moduli)
  thermal.py    → θ_D, v_D (Debye velocity)
  magnetism.py  → T_C, n_unpaired, measured_moment
  crystal_field.py → 10Dq, B_free, β_neph
  superconductivity.py → λ_ep, T_c, n_e, v_F
  constants.py  → α, ℏ, e, m_e, k_B, a_0

All algorithms use the quantum simulator from quantum_computing.py.
Pure Python, zero dependencies.
"""

import math

from ..constants import (
    HBAR, K_B, E_CHARGE, M_ELECTRON_KG, ALPHA, BOHR_RADIUS,
    EV_TO_J,
)
from .surface import MATERIALS
from .magnetism import MAGNETIC_DATA, curie_temperature
from .crystal_field import (
    d_electron_count,
    FREE_ION_RACAH_B_EV,
    NEPHELAUXETIC_BETA,
    CRYSTAL_FIELD_10DQ_EV,
)
from .quantum_computing import (
    zero_state, basis_state, run_circuit,
    _n_qubits,
)
from .quantum_output import (
    probabilities, expectation_pauli, entanglement_entropy,
    extract_max_probability,
)


# =====================================================================
# 1. HUBBARD PARAMETERS FROM CASCADE
# =====================================================================

# Coordination numbers by crystal structure
_Z_COORD = {'fcc': 12, 'bcc': 8, 'hcp': 12, 'diamond_cubic': 4}

# Valence electron count for our materials
# MEASURED: from electron configuration
_VALENCE_D = {
    'iron':     8,   # [Ar] 3d⁶ 4s² → 8 valence
    'copper':   11,  # [Ar] 3d¹⁰ 4s¹ → 11 valence
    'aluminum': 3,   # [Ne] 3s² 3p¹ → 3 valence (sp metal, no d)
    'gold':     11,  # [Xe] 4f¹⁴ 5d¹⁰ 6s¹ → 11 valence
    'silicon':  4,   # [Ne] 3s² 3p² → 4 valence (covalent)
    'tungsten': 6,   # [Xe] 4f¹⁴ 5d⁴ 6s² → 6 valence
    'nickel':   10,  # [Ar] 3d⁸ 4s² → 10 valence
    'titanium': 4,   # [Ar] 3d² 4s² → 4 valence
}

# Published Hubbard U values for validation (eV)
# MEASURED: from photoemission / inverse photoemission spectroscopy
# Sources: Anisimov, Zaanen, Andersen, PRB 44, 943 (1991)
#          Cococcioni & de Gironcoli, PRB 71, 035105 (2005)
_U_PUBLISHED = {
    'iron':     4.0,   # range 3.5-5.0 eV
    'copper':   7.5,   # range 6.5-8.5 eV (in Cu metal; higher in CuO)
    'nickel':   6.0,   # range 5.0-7.0 eV
    'titanium': 3.5,   # range 3.0-4.5 eV
    'tungsten': 2.0,   # range 1.5-3.0 eV (5d, more delocalized)
}


def wigner_seitz_radius(material_key):
    """Wigner-Seitz radius in meters from density and atomic mass.

    FIRST_PRINCIPLES: r_ws = (3 Ω_atom / 4π)^(1/3)
    where Ω_atom = A / (ρ × N_A) is the volume per atom.

    Returns:
        r_ws in meters.
    """
    mat = MATERIALS[material_key]
    A = mat['A']
    rho = mat['density_kg_m3']
    N_A = 6.02214076e23
    m_atom = A * 1e-3 / N_A  # kg per atom
    omega = m_atom / rho      # m³ per atom
    return (3 * omega / (4 * math.pi)) ** (1.0 / 3.0)


def thomas_fermi_screening_length(material_key):
    """Thomas-Fermi screening length from free-electron density.

    FIRST_PRINCIPLES:
        k_TF² = 4 k_F / (π a₀)
        λ_TF = 1 / k_TF = √(π a₀ / (4 k_F))

    where k_F = (3π² n_e)^(1/3) and a₀ = Bohr radius.

    This determines how effectively the metallic electron gas screens
    the bare Coulomb repulsion. Smaller λ_TF = stronger screening = smaller U.

    Returns:
        λ_TF in meters.
    """
    r_ws = wigner_seitz_radius(material_key)
    # Free-electron density: one atom per Wigner-Seitz sphere
    n_val = _VALENCE_D.get(material_key, 4)
    n_e = n_val / (4.0 / 3.0 * math.pi * r_ws**3)
    k_F = (3 * math.pi**2 * n_e) ** (1.0 / 3.0)
    k_TF = math.sqrt(4 * k_F / (math.pi * BOHR_RADIUS))
    return 1.0 / k_TF


def fermi_energy_eV(material_key):
    """Free-electron Fermi energy from valence electron density.

    FIRST_PRINCIPLES:
        E_F = (ℏ² / 2m_e) × (3π² n_e)^(2/3)

    This sets the TOTAL bandwidth scale (including sp hybridization),
    which is the relevant energy for metallic vs insulating behavior.
    The d-band width (from E_coh) is a subset of this.

    Returns:
        E_F in eV.
    """
    r_ws = wigner_seitz_radius(material_key)
    n_val = _VALENCE_D.get(material_key, 4)
    n_e = n_val / (4.0 / 3.0 * math.pi * r_ws**3)
    k_F = (3 * math.pi**2 * n_e) ** (1.0 / 3.0)
    E_F_J = HBAR**2 * k_F**2 / (2 * M_ELECTRON_KG)
    return E_F_J / EV_TO_J


def hubbard_parameters(material_key):
    """Derive Hubbard U and t from cascade material properties.

    DERIVATION:

    Two bandwidth scales:
        t_d = E_coh / (2z): narrow d-band hopping (for d-metals)
        W_sp = E_F: total sp+d bandwidth from free-electron model

    For Mott physics, the relevant comparison is U vs the TOTAL
    bandwidth W_eff = max(2z × t_d, E_F). A material is metallic
    if ANY band provides enough kinetic energy to delocalize electrons.

    On-site Coulomb U:
        U_bare = e² / (4πε₀ r_ws) — unscreened Coulomb at Wigner-Seitz radius
        U_screened = U_bare × (λ_TF / r_ws) — Thomas-Fermi screening

        ORIGIN: r_ws (from density, MEASURED), λ_TF (from n_e, FIRST_PRINCIPLES)

    Mott criterion (DMFT, Bethe lattice):
        U_c / W ≈ 1.2 (Georges, Kotliar, Krauth, Rozenberg, RMP 1996)
        Material is Mott insulator when U_screened > 1.2 × W_eff

    Returns:
        dict with t_eV, U_eV, U_over_W, is_mott_insulator, etc.
    """
    mat = MATERIALS[material_key]
    E_coh = mat['cohesive_energy_ev']
    struct = mat['crystal_structure']
    z = _Z_COORD.get(struct, 8)

    # Narrow-band (d-electron) hopping
    t_d = E_coh / (2.0 * z)

    # Total bandwidth from free-electron Fermi energy
    E_F = fermi_energy_eV(material_key)
    W_d = 2.0 * z * t_d   # d-band width
    W_eff = max(W_d, E_F)  # effective bandwidth (sp or d, whichever is wider)

    # Wigner-Seitz radius
    r_ws = wigner_seitz_radius(material_key)

    # Bare Coulomb at r_ws
    EPS_0 = 8.854187817e-12
    U_bare_J = E_CHARGE**2 / (4 * math.pi * EPS_0 * r_ws)
    U_bare_eV = U_bare_J / EV_TO_J

    # Thomas-Fermi screening
    lam_TF = thomas_fermi_screening_length(material_key)

    # Screened U: linear screening approximation
    screening_factor = min(1.0, lam_TF / r_ws)
    U_screened_eV = U_bare_eV * screening_factor

    # Mott criterion: U / W_eff > 1.2 (DMFT result)
    U_over_W = U_screened_eV / W_eff if W_eff > 0 else float('inf')
    is_mott = U_over_W > 1.2

    return {
        'material': material_key,
        'E_coh_eV': E_coh,
        'z': z,
        't_d_eV': t_d,
        'W_d_eV': W_d,
        'E_F_eV': E_F,
        'W_eff_eV': W_eff,
        'r_ws_angstrom': r_ws * 1e10,
        'lambda_TF_angstrom': lam_TF * 1e10,
        'U_bare_eV': U_bare_eV,
        'screening_factor': screening_factor,
        'U_screened_eV': U_screened_eV,
        'U_over_W': U_over_W,
        'is_mott_insulator': is_mott,
        'U_published_eV': _U_PUBLISHED.get(material_key),
    }


def mott_phase_diagram():
    """Compute U/t for all materials and classify metal vs Mott insulator.

    PREDICTION: Materials with U/t > z should be Mott insulators.
    All our metals should be in the metallic regime (U/t < z).
    Silicon should be in or near the insulating regime.

    Returns:
        list of dicts sorted by U/t ratio.
    """
    results = []
    for key in MATERIALS:
        r = hubbard_parameters(key)
        results.append(r)
    results.sort(key=lambda x: x['U_over_W'])
    return results


# =====================================================================
# 2. QUANTUM HUBBARD SIMULATION WITH CASCADE PARAMETERS
# =====================================================================

def hubbard_ground_state(material_key):
    """Full 2-site Hubbard model with cascade-derived U and t.

    The key insight: the Hubbard model in the Mott limit (U >> t)
    reduces to a Heisenberg model with exchange coupling:

        J_superexchange = 4t² / U     (Anderson, 1950)

    This is derived entirely from cascade parameters (E_coh → t, screening → U).

    For FERROMAGNETS, we can compare this to the exchange coupling derived
    independently from the Curie temperature:

        J_Curie = k_B T_C / z          (mean-field theory)

    The ratio J_super / J_Curie measures how far the material is from the
    Mott insulator limit:
      - Ratio >> 1: d-electrons are too delocalized for Mott physics;
                     the material is a band metal, not a Mott system
      - Ratio ≈ 1: d-electrons are near the Mott boundary

    This is a PREDICTION: the cascade independently produces two estimates
    of J that should agree for strongly correlated materials.

    Args:
        material_key: key in MATERIALS

    Returns:
        dict with Hubbard parameters, exchange couplings, predictions
    """
    params = hubbard_parameters(material_key)
    t = params['t_d_eV']
    U = params['U_screened_eV']

    # Exact 2-site Hubbard ground state (full 4-state singlet sector)
    E_exact = U / 2.0 - math.sqrt((U / 2.0)**2 + 4 * t**2)

    # Double occupancy (exact)
    denom = 2.0 * math.sqrt((U / 2.0)**2 + 4 * t**2)
    double_occ = 0.5 - (U / 2.0) / denom if denom > 0 else 0.0

    # Superexchange coupling (Mott limit prediction)
    J_super_eV = 4 * t**2 / U if U > 0 else 0.0

    # Compare to Curie-derived J (if ferromagnetic)
    J_curie_eV = 0.0
    J_ratio = None
    T_C = curie_temperature(material_key)
    if T_C > 0:
        struct = MATERIALS.get(material_key, {}).get('crystal_structure', 'bcc')
        z_coord = _Z_COORD.get(struct, 8)
        J_curie_eV = K_B * T_C / (z_coord * EV_TO_J)
        J_ratio = J_super_eV / J_curie_eV if J_curie_eV > 0 else None

    # Run Heisenberg VQE with the superexchange J
    J = J_super_eV
    n_steps = 40
    best_E = float('inf')
    best_state = None

    angles = [2 * math.pi * i / n_steps for i in range(n_steps)]
    for theta in angles:
        for phi in angles:
            circuit = [('x', 1), ('ry', 0, theta), ('cnot', 0, 1), ('ry', 1, phi)]
            state = run_circuit(2, circuit)
            E = 0.0
            for p_char in ['X', 'Y', 'Z']:
                p = ['I', 'I']
                p[0] = p_char
                p[1] = p_char
                E += J * expectation_pauli(state, ''.join(p))
            if E < best_E:
                best_E = E
                best_state = state

    ent = entanglement_entropy(best_state, 0)

    # Phase classification based on U/W
    if params['U_over_W'] < 0.5:
        phase = 'strongly_metallic'
    elif params['U_over_W'] < 1.2:
        phase = 'correlated_metal'
    else:
        phase = 'mott_insulator'

    return {
        'material': material_key,
        't_eV': t,
        'U_eV': U,
        'U_over_W': params['U_over_W'],
        'E_hubbard_exact_eV': E_exact,
        'double_occupancy': double_occ,
        'J_superexchange_meV': J_super_eV * 1000,
        'J_curie_meV': J_curie_eV * 1000 if T_C > 0 else None,
        'J_ratio': J_ratio,
        'T_C_K': T_C if T_C > 0 else None,
        'heisenberg_E_vqe_eV': best_E,
        'heisenberg_E_exact_eV': -3 * J,
        'entanglement_entropy': ent,
        'phase': phase,
        'is_mott': params['is_mott_insulator'],
    }


# =====================================================================
# 3. CRYSTAL FIELD → SPIN HAMILTONIAN → VQE
# =====================================================================

# Tanabe-Sugano crossover ratios: 10Dq/B at the high-spin → low-spin transition
# FIRST_PRINCIPLES: from eigenvalue crossing of the Tanabe-Sugano matrices.
# Standard T-S diagrams plot Dq/B on x-axis; crossover occurs at Dq/B ≈ 2-3.
# In 10Dq/B units (which we compute), crossover = 10 × (Dq/B crossover).
_TS_CROSSOVER = {
    4: 27.0,   # d⁴ (Cr²⁺, Mn³⁺): ⁵E → ³T₁ at Dq/B ≈ 2.7
    5: 28.0,   # d⁵ (Fe³⁺, Mn²⁺): ⁶A₁ → ²T₂ at Dq/B ≈ 2.8
    6: 20.0,   # d⁶ (Fe²⁺, Co³⁺): ⁵T₂ → ¹A₁ at Dq/B ≈ 2.0
    7: 22.0,   # d⁷ (Co²⁺): ⁴T₁ → ²E at Dq/B ≈ 2.2
}


def crystal_field_mott_ratio(Z, oxidation_state, coord_key):
    """Compute 10Dq/B ratio — the crystal-field Mott parameter.

    KEY INSIGHT: The ratio 10Dq/B in crystal field theory plays exactly
    the same role as t/U in the Hubbard model:
      - 10Dq ↔ kinetic energy (bandwidth, hopping)
      - B ↔ electron-electron repulsion (Coulomb, Hubbard U)
      - High-spin (10Dq/B small) ↔ Mott insulator (localized)
      - Low-spin (10Dq/B large) ↔ metal (delocalized)

    The Tanabe-Sugano crossover value is the Mott transition point
    for each d^n configuration.

    Args:
        Z: atomic number
        oxidation_state: ion charge
        coord_key: coordination environment key

    Returns:
        dict with ratio, crossover, is_localized, d_count
    """
    d = d_electron_count(Z, oxidation_state)
    dq = CRYSTAL_FIELD_10DQ_EV.get((Z, oxidation_state, coord_key))
    B_free = FREE_ION_RACAH_B_EV.get((Z, oxidation_state))
    beta = NEPHELAUXETIC_BETA.get(coord_key, 0.82)

    if dq is None or B_free is None or B_free == 0:
        return None

    B_crystal = beta * B_free
    ratio = dq / B_crystal

    crossover = _TS_CROSSOVER.get(d)
    if crossover is not None:
        is_localized = ratio < crossover
    else:
        # d¹, d², d³: always high-spin (only one multiplet below crossover)
        # d⁸: always high-spin in octahedral (³A₂ ground state, no crossing)
        # d⁹: one electron hole, no spin crossover
        is_localized = True  # always high-spin for these configurations

    return {
        'Z': Z,
        'oxidation_state': oxidation_state,
        'coord_key': coord_key,
        'd_count': d,
        '10Dq_eV': dq,
        'B_free_eV': B_free,
        'B_crystal_eV': B_crystal,
        'beta': beta,
        '10Dq_over_B': ratio,
        'TS_crossover': crossover,
        'is_high_spin': is_localized,
        'mott_analogy': 'localized' if is_localized else 'delocalized',
    }


def crystal_field_phase_diagram():
    """Map all crystal field entries onto the Mott phase diagram.

    PREDICTION: For d⁴-d⁷ ions, the 10Dq/B ratio determines whether
    the ion is high-spin (localized, Mott insulator analog) or low-spin
    (delocalized, metal analog). The crossover ratios are:
      d⁴: 10Dq/B ≈ 2.7    d⁵: 10Dq/B ≈ 2.8
      d⁶: 10Dq/B ≈ 2.0    d⁷: 10Dq/B ≈ 2.2

    Returns:
        list of dicts for every entry in CRYSTAL_FIELD_10DQ_EV.
    """
    results = []
    for (Z, ox, coord), dq in CRYSTAL_FIELD_10DQ_EV.items():
        r = crystal_field_mott_ratio(Z, ox, coord)
        if r is not None:
            results.append(r)
    results.sort(key=lambda x: x['10Dq_over_B'])
    return results


def two_site_spin_hamiltonian_from_crystal_field(
    Z, oxidation_state, coord_key, J_exchange=None
):
    """Build a 2-site Heisenberg Hamiltonian with crystal-field-derived J.

    For magnetic ions, the exchange coupling J can be estimated from:
      - Curie temperature: J = k_B T_C / z  (ferromagnets)
      - Or from the Goodenough-Kanamori rules: antiferromagnetic for
        half-filled to half-filled superexchange via oxygen

    The Hamiltonian on 2 qubits (spin-1/2):
      H = J (X₁X₂ + Y₁Y₂ + Z₁Z₂) + D (Z₁ + Z₂)

    where D = single-ion anisotropy (from crystal field, second order).
    D ≈ (10Dq)² × (spin-orbit coupling) / (Racah B) — typically < 1 meV.

    For this demo, D = 0 (isotropic Heisenberg).

    Args:
        Z, oxidation_state, coord_key: crystal field parameters
        J_exchange: exchange coupling in eV (default: from cascade if available)

    Returns:
        dict with Hamiltonian coefficients, VQE result, spin state.
    """
    cf = crystal_field_mott_ratio(Z, oxidation_state, coord_key)
    if cf is None:
        return None

    # Determine J from cascade if possible
    if J_exchange is None:
        # Look up material by Z
        _Z_TO_MAT = {26: 'iron', 28: 'nickel', 22: 'titanium',
                     29: 'copper', 74: 'tungsten'}
        mat_key = _Z_TO_MAT.get(Z)
        if mat_key and mat_key in MAGNETIC_DATA:
            T_C = curie_temperature(mat_key)
            if T_C > 0:
                struct = MATERIALS.get(mat_key, {}).get('crystal_structure', 'bcc')
                z_coord = _Z_COORD.get(struct, 8)
                J_exchange = K_B * T_C / (z_coord * EV_TO_J)
            else:
                # Estimate from crystal field: antiferromagnetic superexchange
                # J_AF ≈ 2t²/U where t ~ 10Dq/10 and U ~ 5B
                t_eff = cf['10Dq_eV'] / 10.0
                U_eff = 5.0 * cf['B_crystal_eV']
                if U_eff > 0:
                    J_exchange = 2 * t_eff**2 / U_eff
                else:
                    J_exchange = 0.01  # fallback
        else:
            # Estimate from superexchange
            t_eff = cf['10Dq_eV'] / 10.0
            U_eff = 5.0 * cf['B_crystal_eV']
            if U_eff > 0:
                J_exchange = 2 * t_eff**2 / U_eff
            else:
                J_exchange = 0.01

    # Run VQE for 2-site Heisenberg with this J
    n_steps = 40
    best_E = float('inf')
    best_state = None

    def heisenberg_energy(state, J):
        E = 0.0
        for pauli_char in ['X', 'Y', 'Z']:
            p = ['I', 'I']
            p[0] = pauli_char
            p[1] = pauli_char
            E += J * expectation_pauli(state, ''.join(p))
        return E

    angles = [2 * math.pi * i / n_steps for i in range(n_steps)]
    for theta in angles:
        for phi in angles:
            circuit = [('x', 1), ('ry', 0, theta), ('cnot', 0, 1), ('ry', 1, phi)]
            state = run_circuit(2, circuit)
            E = heisenberg_energy(state, J_exchange)
            if E < best_E:
                best_E = E
                best_state = state

    # Exact 2-site result
    E_exact = -3.0 * J_exchange

    # Observables
    zz_corr = expectation_pauli(best_state, 'ZZ')
    ent = entanglement_entropy(best_state, 0)

    # Spin state classification
    if zz_corr < -0.5:
        spin_state = 'singlet (antiferromagnetic)'
    elif zz_corr > 0.5:
        spin_state = 'triplet (ferromagnetic)'
    else:
        spin_state = 'mixed'

    return {
        'Z': Z,
        'oxidation_state': oxidation_state,
        'coord_key': coord_key,
        'd_count': cf['d_count'],
        'J_exchange_eV': J_exchange,
        'J_exchange_meV': J_exchange * 1000,
        'E_vqe_eV': best_E,
        'E_exact_eV': E_exact,
        'zz_correlation': zz_corr,
        'entanglement_entropy': ent,
        'spin_state': spin_state,
        '10Dq_over_B': cf['10Dq_over_B'],
        'is_high_spin': cf['is_high_spin'],
    }


# =====================================================================
# 4. NEPHELAUXETIC SERIES AS METALLIC CHARACTER
# =====================================================================

def nephelauxetic_metallicity():
    """Rank coordination environments by metallic character.

    PREDICTION: The nephelauxetic ratio β measures how much the ligand
    environment reduces the free-ion Coulomb repulsion. Lower β means
    stronger covalent overlap → more delocalized electrons → more metallic.

    The nephelauxetic series (β order) should correlate with:
    - Conductivity (lower β → higher conductivity in the compound)
    - Likelihood of metallic behavior in the compound
    - Inverse of the Mott gap

    The series should be: CN⁻ > S²⁻ > silicate > carbonate > oxide > F⁻ > H₂O

    Returns:
        list of (coord_key, beta, metallicity_index) sorted by beta
    """
    results = []
    for coord, beta in sorted(NEPHELAUXETIC_BETA.items(), key=lambda x: x[1]):
        metallicity = 1.0 - beta  # 0 = ionic (β=1), 1 = perfectly covalent (β=0)
        results.append({
            'coordination': coord,
            'beta': beta,
            'metallicity_index': metallicity,
            'character': 'strongly_covalent' if beta < 0.65 else
                         'covalent' if beta < 0.80 else
                         'ionic' if beta > 0.88 else 'intermediate',
        })
    return results


# =====================================================================
# 5. SUPERCONDUCTOR MOTT PROXIMITY
# =====================================================================

def superconductor_correlation_strength():
    """Classify superconductors by proximity to Mott transition.

    PREDICTION: The electron-phonon coupling λ_ep correlates with
    proximity to the Mott transition. Strong-coupling superconductors
    (λ > 1) are materials where electrons are on the verge of
    localization — their large effective mass enhances the coupling.

    The McMillan λ can be viewed as a measure of how strongly
    correlated the electrons are:
    - λ < 0.5: weakly correlated, weak superconductor
    - 0.5 < λ < 1.0: moderately correlated
    - λ > 1.0: strongly correlated, near Mott boundary

    Returns:
        list of (material, lambda_ep, Tc, correlation_class)
    """
    from .superconductivity import SUPERCONDUCTORS

    results = []
    for mat, data in SUPERCONDUCTORS.items():
        lam = data.get('lambda_ep')
        if lam is None:
            continue
        Tc = data.get('Tc_K', 0)

        if lam < 0.5:
            corr_class = 'weakly_correlated'
        elif lam < 1.0:
            corr_class = 'moderately_correlated'
        else:
            corr_class = 'strongly_correlated_near_Mott'

        results.append({
            'material': mat,
            'lambda_ep': lam,
            'Tc_K': Tc,
            'correlation_class': corr_class,
            'suppression': data.get('suppression_reason', 'superconducting'),
        })

    results.sort(key=lambda x: x['lambda_ep'])
    return results


# =====================================================================
# 6. COLLECTED PREDICTIONS
# =====================================================================

def all_predictions():
    """Generate all material-specific quantum predictions.

    Returns a dict of prediction categories, each containing testable
    predictions derived from the cascade.
    """
    predictions = {}

    # Prediction 1: Mott phase diagram from cascade
    mott = mott_phase_diagram()
    predictions['mott_phase_diagram'] = {
        'description': (
            'U/t ratio from cascade parameters (E_coh, density, screening). '
            'All elemental metals should be in the metallic regime. '
            'Silicon should be near or past the Mott boundary.'
        ),
        'materials': mott,
        'testable': (
            'Compare U_screened to published DFT+U values. '
            'Check that all metals give U/t < U_c/t and silicon gives U/t > U_c/t.'
        ),
    }

    # Prediction 2: Crystal field Mott analogy
    cf_diagram = crystal_field_phase_diagram()
    predictions['crystal_field_mott'] = {
        'description': (
            'The 10Dq/B ratio in crystal field theory is the Mott parameter. '
            'High-spin ions (small 10Dq/B) are Mott-localized; low-spin ions '
            '(large 10Dq/B) are delocalized. Crossover values are predictions '
            'of the Mott boundary for each d^n configuration.'
        ),
        'entries': cf_diagram,
        'testable': (
            'Verify that all known high-spin complexes have 10Dq/B below crossover '
            'and all known low-spin complexes have 10Dq/B above crossover.'
        ),
    }

    # Prediction 3: Nephelauxetic series = metallicity ranking
    neph = nephelauxetic_metallicity()
    predictions['nephelauxetic_metallicity'] = {
        'description': (
            'The nephelauxetic series (β order) predicts metallic character: '
            'CN⁻ (β=0.50) is most metallic, F⁻ (β=0.90) is most ionic. '
            'This correlates with: conductivity, band gap, Mott gap size.'
        ),
        'series': neph,
        'testable': (
            'Compare β ranking to measured conductivity of transition metal '
            'compounds with these ligands (e.g., NiO vs NiS vs Ni(CN)₂).'
        ),
    }

    # Prediction 4: Strong-coupling superconductors are near-Mott materials
    sc = superconductor_correlation_strength()
    predictions['superconductor_mott_proximity'] = {
        'description': (
            'λ_ep > 1 marks strongly correlated materials near the Mott boundary. '
            'These have the highest Tc among conventional superconductors because '
            'the large effective mass enhances electron-phonon coupling. '
            'Pb (λ=1.55, Tc=7.2K), Hg (λ=1.60, Tc=4.2K), Nb (λ=1.26, Tc=9.25K).'
        ),
        'materials': sc,
        'testable': (
            'Strong-coupling materials should show larger McMillan gap (deviation '
            'between inverted λ and measured λ) — already verified in session 15. '
            'Predict: non-superconducting metals with high λ_ep (Pd: 0.47) are '
            'suppressed by spin fluctuations, not by weak coupling.'
        ),
    }

    return predictions


# =====================================================================
# MODULE REPORT
# =====================================================================

def quantum_matter_report():
    """Standard module report (Rule 9)."""
    return {
        'module': 'quantum_matter',
        'prediction_pipelines': [
            'Mott phase diagram from cascade (U/t from E_coh, ρ, screening)',
            'Crystal field Mott analogy (10Dq/B = t/U proxy)',
            'Nephelauxetic metallicity ranking (β → conductivity)',
            'Superconductor Mott proximity (λ_ep → correlation strength)',
        ],
        'cascade_connections': [
            'surface.py → E_coh, ρ, a',
            'mechanical.py → K, G',
            'magnetism.py → T_C → J_exchange',
            'crystal_field.py → 10Dq, B, β',
            'superconductivity.py → λ_ep, T_c',
            'constants.py → α, a₀, ℏ',
        ],
        'key_insight': (
            'The Tanabe-Sugano high-spin/low-spin crossover IS a Mott transition. '
            '10Dq/B is the crystal-field analog of t/U. Every entry in our '
            'crystal field database is a data point on the Mott phase diagram.'
        ),
    }


def full_report():
    """Extended report with example results."""
    report = quantum_matter_report()
    report['iron_hubbard'] = hubbard_parameters('iron')
    report['silicon_hubbard'] = hubbard_parameters('silicon')
    return report
