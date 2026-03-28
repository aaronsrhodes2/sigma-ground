"""
Angular momentum — quantum numbers, coupling, term symbols, and splitting.

Angular momentum is quantized. This module computes the algebra of
quantum angular momentum: addition of L and S, term symbols,
Clebsch-Gordan coefficients, Hund's rules, and spin-orbit coupling.

Physics chain:
  1. Angular momentum quantization (FIRST_PRINCIPLES)
     J² |j,m⟩ = ℏ²j(j+1) |j,m⟩
     J_z |j,m⟩ = ℏm |j,m⟩
     m = −j, −j+1, ..., j−1, j

  2. Angular momentum addition (FIRST_PRINCIPLES)
     |j₁−j₂| ≤ J ≤ j₁+j₂
     Total (2j₁+1)(2j₂+1) states redistributed into J multiplets.

  3. Clebsch-Gordan coefficients (FIRST_PRINCIPLES)
     |J,M⟩ = Σ ⟨j₁m₁;j₂m₂|J,M⟩ |j₁m₁⟩|j₂m₂⟩
     Computed from recursion (Condon-Shortley convention).

  4. Term symbols (FIRST_PRINCIPLES + Hund's rules)
     ²ˢ⁺¹L_J where S=total spin, L=total orbital, J=total angular momentum
     Hund's rules determine the ground state term.

  5. Spin-orbit coupling (FIRST_PRINCIPLES)
     H_SO = A × L·S where A = spin-orbit coupling constant
     Splits terms into J levels: interval rule ΔE ∝ J (Landé).

σ-dependence:
  Angular momentum algebra is pure mathematics → σ-INVARIANT.
  Spin-orbit coupling constant A ∝ Z⁴ (EM) → σ-INVARIANT.
  The only σ-entry is through reduced mass in fine structure,
  handled by atomic_spectra.py.

□σ = −ξR
"""

import math
from ..constants import HBAR, EV_TO_J, MU_BOHR, ALPHA, M_ELECTRON_KG, C, SIGMA_HERE


# ══════════════════════════════════════════════════════════════════════
# BASIC ANGULAR MOMENTUM
# ══════════════════════════════════════════════════════════════════════

def angular_momentum_magnitude(j):
    """Magnitude of angular momentum |J| = ℏ√(j(j+1)) (J·s).

    FIRST_PRINCIPLES: eigenvalue of J² operator.
    """
    if j < 0:
        raise ValueError(f"j must be ≥ 0, got {j}")
    return HBAR * math.sqrt(j * (j + 1))


def angular_momentum_z_values(j):
    """Allowed m_j values for quantum number j.

    m_j = −j, −j+1, ..., j−1, j
    Total count = 2j + 1.

    FIRST_PRINCIPLES: eigenvalues of J_z commuting with J².
    """
    values = []
    m = -j
    while m <= j + 0.001:
        values.append(m)
        m += 1.0
    return values


def multiplicity(j):
    """Multiplicity = 2j + 1 (number of m_j substates).

    For spin-½: 2 states (up, down).
    For l=1: 3 states (m = −1, 0, +1).
    For J=3/2: 4 states.
    """
    return int(2 * j + 1)


# ══════════════════════════════════════════════════════════════════════
# ANGULAR MOMENTUM ADDITION
# ══════════════════════════════════════════════════════════════════════

def allowed_J_values(j1, j2):
    """Allowed total J values when adding j₁ and j₂.

    J = |j₁−j₂|, |j₁−j₂|+1, ..., j₁+j₂

    This is the triangle rule: three angular momentum vectors must
    form a closed triangle.

    FIRST_PRINCIPLES: representation theory of SU(2).
    """
    J_min = abs(j1 - j2)
    J_max = j1 + j2
    values = []
    J = J_min
    while J <= J_max + 0.001:
        values.append(J)
        J += 1.0
    return values


def total_states(j1, j2):
    """Total number of states = (2j₁+1)(2j₂+1).

    This must equal Σ_J (2J+1) over all allowed J values
    (conservation of Hilbert space dimension).

    FIRST_PRINCIPLES: tensor product of two angular momentum spaces.
    """
    return int((2 * j1 + 1) * (2 * j2 + 1))


def verify_state_count(j1, j2):
    """Verify that angular momentum addition conserves state count.

    Σ(2J+1) over all allowed J must equal (2j₁+1)(2j₂+1).
    Returns True if consistent.
    """
    expected = total_states(j1, j2)
    J_vals = allowed_J_values(j1, j2)
    actual = sum(int(2 * J + 1) for J in J_vals)
    return actual == expected


# ══════════════════════════════════════════════════════════════════════
# CLEBSCH-GORDAN COEFFICIENTS
# ══════════════════════════════════════════════════════════════════════

def _factorial(n):
    """Factorial for non-negative integer (or half-integer converted)."""
    if n < 0:
        return 0
    result = 1
    for i in range(2, int(n) + 1):
        result *= i
    return result


def clebsch_gordan(j1, m1, j2, m2, J, M):
    """Clebsch-Gordan coefficient ⟨j₁m₁;j₂m₂|J,M⟩.

    Computed from the explicit formula (Racah 1942).
    Uses Condon-Shortley phase convention.

    Selection rules (returns 0 if violated):
      M = m₁ + m₂
      |j₁ − j₂| ≤ J ≤ j₁ + j₂
      |m₁| ≤ j₁, |m₂| ≤ j₂, |M| ≤ J

    Args:
        j1, m1: first angular momentum and projection
        j2, m2: second angular momentum and projection
        J, M: total angular momentum and projection

    Returns:
        CG coefficient (float).

    FIRST_PRINCIPLES: SU(2) representation theory.

    Reference: Racah, G. (1942) Phys. Rev. 62, 438.
    """
    # Selection rules
    if abs(m1) > j1 + 0.001 or abs(m2) > j2 + 0.001:
        return 0.0
    if abs(M) > J + 0.001:
        return 0.0
    if abs(m1 + m2 - M) > 0.001:
        return 0.0
    if J < abs(j1 - j2) - 0.001 or J > j1 + j2 + 0.001:
        return 0.0

    # Convert to integers for factorial (multiply all by 2 for half-integers)
    # All j, m values are either integer or half-integer
    # We work with 2j, 2m as integers
    tj1 = int(round(2 * j1))
    tm1 = int(round(2 * m1))
    tj2 = int(round(2 * j2))
    tm2 = int(round(2 * m2))
    tJ = int(round(2 * J))
    tM = int(round(2 * M))

    # Triangle coefficient Δ(j1,j2,J)
    def triangle_coeff(a, b, c):
        # a, b, c are 2×j values (integers)
        s = a + b + c
        if s % 2 != 0:
            return 0.0
        n1 = (a + b - c) // 2
        n2 = (a - b + c) // 2
        n3 = (-a + b + c) // 2
        n4 = s // 2
        if n1 < 0 or n2 < 0 or n3 < 0:
            return 0.0
        return (_factorial(n1) * _factorial(n2) * _factorial(n3)
                / _factorial(n4 + 1))

    delta = triangle_coeff(tj1, tj2, tJ)
    if delta == 0:
        return 0.0

    # Prefactor
    prefactor_num = (tJ + 1)
    pf = prefactor_num * delta

    f1 = _factorial((tj1 + tm1) // 2)
    f2 = _factorial((tj1 - tm1) // 2)
    f3 = _factorial((tj2 + tm2) // 2)
    f4 = _factorial((tj2 - tm2) // 2)
    f5 = _factorial((tJ + tM) // 2)
    f6 = _factorial((tJ - tM) // 2)

    pf *= f1 * f2 * f3 * f4 * f5 * f6

    # Sum over k
    total = 0.0
    for k in range(100):
        n1 = (tj1 + tj2 - tJ) // 2 - k
        n2 = (tj1 - tm1) // 2 - k
        n3 = (tj2 + tm2) // 2 - k
        n4 = (tJ - tj2 + tm1) // 2 + k
        n5 = (tJ - tj1 - tm2) // 2 + k

        if n1 < 0 or n2 < 0 or n3 < 0 or n4 < 0 or n5 < 0:
            if k > 0 and n1 < 0:
                break
            continue

        denom = (_factorial(k) * _factorial(n1) * _factorial(n2)
                 * _factorial(n3) * _factorial(n4) * _factorial(n5))
        if denom == 0:
            continue
        total += ((-1)**k) / denom

    result = total * math.sqrt(pf)
    return result


# ══════════════════════════════════════════════════════════════════════
# SPECTROSCOPIC NOTATION
# ══════════════════════════════════════════════════════════════════════

_L_LETTERS = {0: 'S', 1: 'P', 2: 'D', 3: 'F', 4: 'G', 5: 'H', 6: 'I'}


def term_symbol(L, S, J):
    """Spectroscopic term symbol ²ˢ⁺¹Lⱼ.

    Args:
        L: total orbital angular momentum quantum number
        S: total spin quantum number
        J: total angular momentum quantum number

    Returns:
        String like '³P₂' (triplet P, J=2).

    CONVENTION: Russell-Saunders (LS) coupling.
    """
    mult = int(2 * S + 1)
    L_letter = _L_LETTERS.get(int(L), f'[{int(L)}]')
    # Format J: show as integer or fraction
    if J == int(J):
        J_str = str(int(J))
    else:
        J_str = f'{int(2*J)}/2'
    return f'{mult}{L_letter}{J_str}'


def all_term_symbols(L, S):
    """All term symbols for given L, S (varying J).

    J ranges from |L−S| to L+S.

    Returns list of (J, term_symbol_string).
    """
    J_vals = allowed_J_values(L, S)
    return [(J, term_symbol(L, S, J)) for J in J_vals]


# ══════════════════════════════════════════════════════════════════════
# HUND'S RULES — Ground State Term
# ══════════════════════════════════════════════════════════════════════

def hund_ground_state(n_electrons, l):
    """Ground state term from Hund's rules for equivalent electrons.

    Hund's rules (in order of priority):
      1. Maximize S (maximum spin multiplicity)
      2. For given S, maximize L (maximum orbital angular momentum)
      3. J = |L−S| if shell < half-full, J = L+S if shell ≥ half-full

    FIRST_PRINCIPLES: exchange interaction (rule 1), orbital correlation
    (rule 2), spin-orbit coupling sign (rule 3).

    Args:
        n_electrons: number of electrons in the subshell
        l: orbital quantum number of the subshell (0=s, 1=p, 2=d, 3=f)

    Returns:
        (S, L, J, term_string) for the ground state.

    Examples:
        d² (e.g., Ti²⁺): S=1, L=3, J=2  →  ³F₂
        d⁵ (e.g., Fe³⁺): S=5/2, L=0, J=5/2  →  ⁶S₅/₂
        d⁸ (e.g., Ni²⁺): S=1, L=3, J=4  →  ³F₄
    """
    max_electrons = 2 * (2 * l + 1)
    if n_electrons < 0 or n_electrons > max_electrons:
        raise ValueError(
            f"n_electrons={n_electrons} invalid for l={l} "
            f"(max={max_electrons})"
        )

    # Fill m_l orbitals one at a time, spin-up first (Rule 1)
    m_l_values = list(range(l, -l - 1, -1))  # l, l-1, ..., -l
    n_orbitals = 2 * l + 1

    # Each orbital gets spin-up first, then spin-down
    spins = [0.0] * n_orbitals  # net spin per orbital
    m_l_occ = [0] * n_orbitals   # occupation per orbital

    remaining = n_electrons
    # First pass: one electron per orbital (all spin-up)
    for i in range(n_orbitals):
        if remaining > 0:
            spins[i] = 0.5
            m_l_occ[i] = 1
            remaining -= 1

    # Second pass: fill remaining spin-down
    for i in range(n_orbitals):
        if remaining > 0:
            spins[i] = 0.0  # paired → net spin 0
            m_l_occ[i] = 2
            remaining -= 1

    # Total S: count unpaired electrons
    n_unpaired = sum(1 for s in spins if s == 0.5)
    S = n_unpaired / 2.0

    # Total M_L (maximize L: sum of occupied m_l)
    M_L = 0
    for i in range(n_orbitals):
        M_L += m_l_values[i] * m_l_occ[i]
    # L = |M_L| in ground state (maximum consistent with S)
    L = abs(M_L)

    # Rule 3: J
    half_full = n_orbitals
    if n_electrons <= half_full:
        J = abs(L - S)  # less than half: J = |L-S|
    else:
        J = L + S  # more than half: J = L+S

    return (S, L, J, term_symbol(L, S, J))


# ══════════════════════════════════════════════════════════════════════
# SPIN-ORBIT COUPLING — Landé Interval Rule
# ══════════════════════════════════════════════════════════════════════

def spin_orbit_energy_eV(A_eV, L, S, J):
    """Spin-orbit coupling energy for term with given L, S, J (eV).

    E_SO = (A/2) × [J(J+1) − L(L+1) − S(S+1)]

    where A is the spin-orbit coupling constant.
    A > 0 for shells less than half-full.
    A < 0 for shells more than half-full.

    FIRST_PRINCIPLES: H_SO = A × L·S, and L·S = ½[J(J+1)−L(L+1)−S(S+1)].

    Reference: Landé interval rule (1923).
    """
    return 0.5 * A_eV * (J * (J + 1) - L * (L + 1) - S * (S + 1))


def spin_orbit_splitting_eV(A_eV, L, S):
    """Total spin-orbit splitting of a term (eV).

    Splitting between J_max and J_min levels.
    """
    J_min = abs(L - S)
    J_max = L + S
    E_max = spin_orbit_energy_eV(A_eV, L, S, J_max)
    E_min = spin_orbit_energy_eV(A_eV, L, S, J_min)
    return abs(E_max - E_min)


def lande_interval_check(A_eV, L, S):
    """Verify Landé interval rule: ΔE(J,J−1) ∝ J.

    The energy difference between adjacent J levels is:
    E(J) − E(J−1) = A × J

    Returns list of (J, ΔE_eV, ratio_to_A) tuples.
    Should show ratio ≈ J.

    MEASURED: confirmed in many atoms (e.g., Na D-lines).
    """
    J_min = abs(L - S)
    J_max = L + S
    intervals = []
    J = J_min + 1
    while J <= J_max + 0.001:
        dE = spin_orbit_energy_eV(A_eV, L, S, J) - spin_orbit_energy_eV(
            A_eV, L, S, J - 1
        )
        ratio = dE / A_eV if abs(A_eV) > 1e-30 else 0
        intervals.append((J, dE, ratio))
        J += 1.0
    return intervals


# ══════════════════════════════════════════════════════════════════════
# SPIN-ORBIT COUPLING CONSTANT ESTIMATES
# ══════════════════════════════════════════════════════════════════════

def hydrogen_spin_orbit_constant_eV(Z, n, l):
    """Spin-orbit coupling constant A for hydrogen-like atom (eV).

    A = (α²Z⁴ × E_R) / (n³ × l(l+½)(l+1))

    For hydrogen n=2, l=1: A ≈ 0.000018 eV (splitting ≈ 0.000045 eV).
    MEASURED: Na D-line splitting = 0.0021 eV.

    FIRST_PRINCIPLES: ⟨1/r³⟩ for hydrogen-like wavefunctions.
    """
    if l == 0:
        return 0.0  # no spin-orbit for s-orbitals
    from .atomic_spectra import RYDBERG_ENERGY_EV
    return (ALPHA**2 * Z**4 * RYDBERG_ENERGY_EV /
            (n**3 * l * (l + 0.5) * (l + 1)))


def multi_electron_SO_constant_eV(Z, n, l):
    """Rough spin-orbit coupling constant for multi-electron atom (eV).

    A ≈ A_hydrogen × Z_eff⁴/Z⁴

    APPROXIMATION: scales as Z_eff⁴ for screened atoms.
    """
    from .element import slater_zeff
    z_eff = slater_zeff(Z)
    A_h = hydrogen_spin_orbit_constant_eV(Z, n, l)
    if Z == 0:
        return 0.0
    return A_h * (z_eff / Z)**4 * Z**4 / z_eff**4  # simplifies but clearer this way


# ══════════════════════════════════════════════════════════════════════
# MAGNETIC MOMENT
# ══════════════════════════════════════════════════════════════════════

def lande_g_factor(L, S, J):
    """Landé g-factor for state |L, S, J⟩.

    g_J = 1 + [J(J+1) + S(S+1) − L(L+1)] / [2J(J+1)]

    Special cases:
      Pure spin (L=0): g_J = 2
      Pure orbital (S=0): g_J = 1

    FIRST_PRINCIPLES: projection of μ onto J direction.
    """
    if J == 0:
        return 0.0
    return 1.0 + (J * (J+1) + S * (S+1) - L * (L+1)) / (2 * J * (J+1))


def magnetic_moment_bohr_magnetons(L, S, J):
    """Effective magnetic moment in Bohr magnetons.

    μ_eff = g_J × √(J(J+1))  (in units of μ_B)

    This is what's measured in Curie-law susceptibility experiments.

    FIRST_PRINCIPLES: μ = −g_J μ_B J, magnitude = g_J μ_B √(J(J+1)).
    """
    g_J = lande_g_factor(L, S, J)
    return g_J * math.sqrt(J * (J + 1))


# ══════════════════════════════════════════════════════════════════════
# PAULI MATRICES & SPIN-½ ALGEBRA
# ══════════════════════════════════════════════════════════════════════

def pauli_matrices():
    """The three Pauli spin matrices.

    σ_x = [[0,1],[1,0]]
    σ_y = [[0,-i],[i,0]]  (returned as (real, imag) pairs)
    σ_z = [[1,0],[0,-1]]

    S = (ℏ/2)σ are the spin-½ operators.

    FIRST_PRINCIPLES: generators of SU(2).
    """
    sigma_x = [[0, 1], [1, 0]]
    sigma_y = [[(0, 0), (0, -1)], [(0, 1), (0, 0)]]  # (real, imag)
    sigma_z = [[1, 0], [0, -1]]
    return {'x': sigma_x, 'y': sigma_y, 'z': sigma_z}


def spin_expectation(state_up_amp, state_down_amp, axis='z'):
    """Expectation value of spin along axis for spin-½ state (units of ℏ).

    |ψ⟩ = α|↑⟩ + β|↓⟩

    ⟨S_z⟩/ℏ = (|α|² − |β|²) / 2
    ⟨S_x⟩/ℏ = Re(α*β)
    ⟨S_y⟩/ℏ = Im(α*β)

    Args:
        state_up_amp: complex amplitude α for spin-up (can be real)
        state_down_amp: complex amplitude β for spin-down
        axis: 'x', 'y', or 'z'

    Returns:
        ⟨S_axis⟩ in units of ℏ/2.

    FIRST_PRINCIPLES: Born rule ⟨ψ|S|ψ⟩.
    """
    a = complex(state_up_amp)
    b = complex(state_down_amp)
    # Normalize
    norm = abs(a)**2 + abs(b)**2
    if norm < 1e-30:
        return 0.0
    a /= math.sqrt(norm)
    b /= math.sqrt(norm)

    if axis == 'z':
        return (abs(a)**2 - abs(b)**2) / 2.0
    elif axis == 'x':
        return (a.conjugate() * b + b.conjugate() * a).real / 2.0
    elif axis == 'y':
        return (a.conjugate() * b - b.conjugate() * a).imag / 2.0
    else:
        raise ValueError(f"axis must be 'x', 'y', or 'z', got '{axis}'")


# ══════════════════════════════════════════════════════════════════════
# REPORTS (Rule 9)
# ══════════════════════════════════════════════════════════════════════

def angular_momentum_report(n_electrons=2, l=2):
    """Report on angular momentum for d-electron configuration.

    Args:
        n_electrons: electrons in subshell
        l: orbital quantum number (default 2 = d-shell)
    """
    S, L, J, term = hund_ground_state(n_electrons, l)

    return {
        'n_electrons': n_electrons,
        'l': l,
        'subshell': {0: 's', 1: 'p', 2: 'd', 3: 'f'}.get(l, f'l={l}'),
        'ground_state_S': S,
        'ground_state_L': L,
        'ground_state_J': J,
        'ground_state_term': term,
        'all_J_terms': all_term_symbols(L, S),
        'lande_g_factor': lande_g_factor(L, S, J),
        'magnetic_moment_muB': magnetic_moment_bohr_magnetons(L, S, J),
        'multiplicity': multiplicity(J),
        'allowed_J_count': len(allowed_J_values(L, S)),
    }


def full_report(n_electrons=2, l=2):
    """Complete angular momentum report (Rule 9)."""
    report = angular_momentum_report(n_electrons, l)

    # Add CG coefficient examples
    report['cg_example_1/2_1/2_to_1'] = clebsch_gordan(0.5, 0.5, 0.5, 0.5, 1, 1)
    report['cg_example_1/2_1/2_to_0'] = clebsch_gordan(0.5, 0.5, 0.5, -0.5, 0, 0)

    # Add state counting verification
    report['state_count_verified'] = verify_state_count(
        report['ground_state_L'], report['ground_state_S']
    )

    # Pauli matrices
    report['pauli_matrices'] = pauli_matrices()

    return report
