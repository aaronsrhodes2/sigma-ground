"""
Quantum confinement — particle in a box, quantum wells, and quantum dots.

When you confine a particle to a small region, its energy becomes quantized.
The smaller the box, the higher the energy. This is why quantum dots glow
different colors depending on their size.

Physics chain:
  1. Infinite square well / particle in a box (FIRST_PRINCIPLES)
     Eₙ = n²π²ℏ² / (2mL²)     1D
     E(n₁,n₂,n₃) = π²ℏ²/(2m) × (n₁²/L₁² + n₂²/L₂² + n₃²/L₃²)   3D

  2. Finite square well (FIRST_PRINCIPLES)
     Bound state condition: κ tan(κL/2) = k   (even parity)
     where κ = √(2mE)/ℏ,  k = √(2m(V₀−E))/ℏ
     Solved numerically (transcendental equation).

  3. Quantum dot (spherical confinement — FIRST_PRINCIPLES)
     E_gap(R) = E_gap_bulk + π²ℏ²/(2μR²) − 1.8e²/(4πε₀εᵣR)
     Brus equation (1984): confinement energy − Coulomb attraction
     μ = reduced mass of electron-hole pair

  4. Density of states (FIRST_PRINCIPLES)
     3D: g(E) ∝ √E                (bulk)
     2D: g(E) = constant per band  (quantum well)
     1D: g(E) ∝ 1/√E              (quantum wire)
     0D: g(E) = δ-functions         (quantum dot)

σ-dependence:
  Confinement energies involve ℏ, m_e, and geometry — all EM → σ-INVARIANT.
  BUT: quantum dot gap depends on the bulk band gap Eg, which comes from
  crystal bonding (partially QCD through nuclear mass → lattice parameter).
  The size-dependent part π²ℏ²/(2μR²) is purely EM and σ-invariant.
  The Coulomb correction −1.8e²/(εR) is also EM → σ-invariant.

□σ = −ξR
"""

import math
from ..constants import (
    HBAR, H_PLANCK, C, E_CHARGE, EPS_0, M_ELECTRON_KG,
    EV_TO_J, BOHR_RADIUS, SIGMA_HERE,
)


# ══════════════════════════════════════════════════════════════════════
# PARTICLE IN A BOX — Infinite Square Well
# ══════════════════════════════════════════════════════════════════════

def box_energy_1d_eV(n, L_m, mass_kg=None):
    """Energy of particle in 1D infinite square well (eV).

    Eₙ = n²π²ℏ² / (2mL²)

    Args:
        n: quantum number (1, 2, 3, ...)
        L_m: box width in meters
        mass_kg: particle mass (default: electron)

    Returns:
        Energy in eV.

    FIRST_PRINCIPLES: Schrödinger equation with V=0 inside, V=∞ outside.
    Boundary conditions ψ(0) = ψ(L) = 0 force standing waves.
    """
    if n < 1:
        raise ValueError(f"n must be ≥ 1, got {n}")
    if L_m <= 0:
        raise ValueError("L must be > 0")
    m = mass_kg if mass_kg is not None else M_ELECTRON_KG
    E_J = n**2 * math.pi**2 * HBAR**2 / (2 * m * L_m**2)
    return E_J / EV_TO_J


def box_energy_3d_eV(n1, n2, n3, L1_m, L2_m=None, L3_m=None, mass_kg=None):
    """Energy in 3D rectangular box (eV).

    E = π²ℏ²/(2m) × (n₁²/L₁² + n₂²/L₂² + n₃²/L₃²)

    If L2 and L3 are not given, assumes cubic box (L1 = L2 = L3).

    Args:
        n1, n2, n3: quantum numbers (≥ 1)
        L1_m: box dimension x (m)
        L2_m: box dimension y (m), defaults to L1
        L3_m: box dimension z (m), defaults to L1

    FIRST_PRINCIPLES: separation of variables in 3D Schrödinger equation.
    """
    if any(ni < 1 for ni in (n1, n2, n3)):
        raise ValueError("All quantum numbers must be ≥ 1")
    L2 = L2_m if L2_m is not None else L1_m
    L3 = L3_m if L3_m is not None else L1_m
    m = mass_kg if mass_kg is not None else M_ELECTRON_KG
    E_J = (math.pi**2 * HBAR**2 / (2 * m)) * (
        n1**2 / L1_m**2 + n2**2 / L2**2 + n3**2 / L3**2
    )
    return E_J / EV_TO_J


def box_ground_state_eV(L_m, mass_kg=None):
    """Ground state energy of 1D box (eV).

    E₁ = π²ℏ²/(2mL²)

    Example: electron in 1 nm box → E₁ ≈ 0.38 eV.
    """
    return box_energy_1d_eV(1, L_m, mass_kg)


def box_transition_wavelength_nm(n_i, n_f, L_m, mass_kg=None):
    """Wavelength of photon absorbed/emitted in box transition (nm).

    λ = hc / |E_i − E_f|

    Args:
        n_i, n_f: initial and final quantum numbers
        L_m: box width (m)
        mass_kg: particle mass (default: electron)
    """
    E_i = box_energy_1d_eV(n_i, L_m, mass_kg)
    E_f = box_energy_1d_eV(n_f, L_m, mass_kg)
    dE = abs(E_i - E_f) * EV_TO_J
    if dE == 0:
        return float('inf')
    return H_PLANCK * C / dE * 1e9


def degeneracy_3d_cubic(n_squared_total):
    """Degeneracy of energy level in cubic box.

    For a cubic box, E ∝ (n₁² + n₂² + n₃²).
    Degeneracy = number of distinct (n₁,n₂,n₃) with same sum of squares.

    Example: n²_total = 6 → (1,1,2), (1,2,1), (2,1,1) = 3-fold degenerate.

    Args:
        n_squared_total: target value of n₁² + n₂² + n₃²

    Returns:
        Degeneracy (integer).
    """
    count = 0
    n_max = int(math.sqrt(n_squared_total)) + 1
    for n1 in range(1, n_max + 1):
        for n2 in range(1, n_max + 1):
            for n3 in range(1, n_max + 1):
                if n1**2 + n2**2 + n3**2 == n_squared_total:
                    count += 1
    return count


# ══════════════════════════════════════════════════════════════════════
# FINITE SQUARE WELL
# ══════════════════════════════════════════════════════════════════════

def finite_well_bound_states(V0_eV, L_m, mass_kg=None):
    """Number of bound states in a 1D finite square well.

    N = ⌈√(2mV₀L²/π²ℏ²)⌉ = ⌈z₀/π⌉ where z₀ = L√(2mV₀)/ℏ

    Actually N = max(1, ⌊z₀/π + 1⌋) for a symmetric well.
    There is always at least 1 bound state in 1D.

    Args:
        V0_eV: well depth (eV, positive)
        L_m: well width (m)
        mass_kg: particle mass (default: electron)

    FIRST_PRINCIPLES: transcendental equation from matching wavefunctions
    at well boundary.
    """
    m = mass_kg if mass_kg is not None else M_ELECTRON_KG
    V0_J = V0_eV * EV_TO_J
    z0 = L_m * math.sqrt(2 * m * V0_J) / HBAR
    # Number of bound states: always at least 1
    return max(1, int(z0 / math.pi) + 1)


def finite_well_ground_state_eV(V0_eV, L_m, mass_kg=None, tol=1e-8):
    """Ground state energy of 1D finite square well (eV).

    Solves the transcendental equation numerically:
      √(E) × tan(√(2mE) × L/(2ℏ)) = √(V₀ − E)    (even parity)

    Uses bisection on the variable z = kL/2 where k = √(2mE)/ℏ.

    Args:
        V0_eV: well depth (eV, positive)
        L_m: well width (m)
        mass_kg: particle mass (default: electron)
        tol: relative tolerance for bisection

    Returns:
        Ground state energy in eV (0 < E < V0).

    FIRST_PRINCIPLES: continuity of ψ and dψ/dx at boundaries.
    """
    m = mass_kg if mass_kg is not None else M_ELECTRON_KG
    V0_J = V0_eV * EV_TO_J

    # z0 = L √(2mV₀) / (2ℏ)  — the dimensionless well depth parameter
    z0 = L_m * math.sqrt(2 * m * V0_J) / (2 * HBAR)

    if z0 < 1e-12:
        return 0.0  # well too shallow

    # Even parity ground state: z tan(z) = √(z0² − z²)
    # where z = kL/2, k = √(2mE)/ℏ
    # z ranges from 0 to min(z0, π/2)
    def f(z):
        if z <= 0:
            return -1.0
        rhs_sq = z0**2 - z**2
        if rhs_sq <= 0:
            return 1.0
        return z * math.tan(z) - math.sqrt(rhs_sq)

    # Bisect in (0, min(z0, π/2 - ε))
    z_max = min(z0, math.pi / 2 - 1e-10)
    z_lo, z_hi = 1e-10, z_max

    for _ in range(200):
        z_mid = (z_lo + z_hi) / 2.0
        if f(z_mid) < 0:
            z_lo = z_mid
        else:
            z_hi = z_mid
        if (z_hi - z_lo) / max(z_hi, 1e-30) < tol:
            break

    z = (z_lo + z_hi) / 2.0
    # E = (2ℏz/L)² / (2m)
    k = 2 * z / L_m
    E_J = HBAR**2 * k**2 / (2 * m)
    return E_J / EV_TO_J


def tunneling_depth_m(V0_eV, E_eV, mass_kg=None):
    """Evanescent penetration depth outside finite well (m).

    δ = ℏ / √(2m(V₀ − E))

    The wavefunction leaks into the classically forbidden region
    as ψ ∝ exp(−x/δ). This is purely quantum — classical particles stop.

    Args:
        V0_eV: barrier height (eV)
        E_eV: particle energy (eV, E < V0)
        mass_kg: particle mass (default: electron)

    FIRST_PRINCIPLES: Schrödinger equation in classically forbidden region.
    """
    if E_eV >= V0_eV:
        return float('inf')  # not bound / no evanescent decay
    m = mass_kg if mass_kg is not None else M_ELECTRON_KG
    dV_J = (V0_eV - E_eV) * EV_TO_J
    return HBAR / math.sqrt(2 * m * dV_J)


# ══════════════════════════════════════════════════════════════════════
# QUANTUM DOTS — Brus Equation
# ══════════════════════════════════════════════════════════════════════

# Effective masses as fractions of m_e (MEASURED)
# Format: (m*_e, m*_h, epsilon_r, Eg_bulk_eV)
_QD_MATERIALS = {
    'CdSe': (0.13, 0.45, 10.6, 1.74),     # MEASURED: Norris & Bawendi 1996
    'CdS':  (0.21, 0.80, 5.7,  2.42),      # MEASURED: Brus 1984
    'CdTe': (0.11, 0.35, 10.2, 1.50),      # MEASURED
    'InP':  (0.077, 0.64, 12.5, 1.34),     # MEASURED: Micic 1994
    'InAs': (0.023, 0.41, 15.1, 0.36),     # MEASURED
    'PbS':  (0.085, 0.085, 17.0, 0.41),    # MEASURED: equal mass
    'PbSe': (0.047, 0.047, 23.0, 0.28),    # MEASURED: equal mass
    'GaAs': (0.067, 0.45, 12.9, 1.42),     # MEASURED
    'ZnS':  (0.34, 0.58, 8.9,  3.68),      # MEASURED
    'ZnSe': (0.17, 0.75, 9.1,  2.70),      # MEASURED
    'Si':   (0.26, 0.36, 11.7, 1.12),      # MEASURED (indirect gap)
}


def brus_energy_eV(R_m, material_key='CdSe'):
    """Quantum dot band gap from Brus equation (eV).

    E_gap(R) = E_gap_bulk + π²ℏ²/(2μR²) − 1.8e²/(4πε₀εᵣR)

    Three terms:
      1. Bulk band gap (MEASURED)
      2. Confinement energy (kinetic, ∝ 1/R²) — always positive
      3. Coulomb attraction (electron-hole, ∝ 1/R) — always negative

    At large R: E → E_bulk (bulk semiconductor).
    At small R: confinement dominates → gap widens → blue shift.

    Args:
        R_m: quantum dot radius (m)
        material_key: semiconductor material

    Returns:
        Size-dependent band gap in eV.

    FIRST_PRINCIPLES: Schrödinger equation for particle in sphere +
    electron-hole Coulomb interaction (Brus 1984).

    Reference: Brus, L.E. (1984) J. Chem. Phys. 80, 4403.
    """
    if material_key not in _QD_MATERIALS:
        raise KeyError(f"Unknown QD material '{material_key}'. "
                       f"Available: {sorted(_QD_MATERIALS.keys())}")

    me_frac, mh_frac, eps_r, Eg_bulk = _QD_MATERIALS[material_key]
    me = me_frac * M_ELECTRON_KG
    mh = mh_frac * M_ELECTRON_KG
    mu = me * mh / (me + mh)  # reduced mass

    # Confinement energy (always positive)
    E_conf = math.pi**2 * HBAR**2 / (2 * mu * R_m**2)

    # Coulomb correction (always negative, Rydberg-like)
    E_coul = -1.8 * E_CHARGE**2 / (4 * math.pi * EPS_0 * eps_r * R_m)

    E_total_J = Eg_bulk * EV_TO_J + E_conf + E_coul
    return E_total_J / EV_TO_J


def qd_emission_wavelength_nm(R_m, material_key='CdSe'):
    """Emission wavelength of quantum dot (nm).

    λ = hc / E_gap(R)

    This is why quantum dots are size-tunable light emitters:
    small dots → blue, large dots → red.

    CdSe examples (MEASURED, Murray et al. 1993):
      R = 1.0 nm → ~450 nm (blue)
      R = 2.0 nm → ~530 nm (green)
      R = 3.5 nm → ~620 nm (red)
    """
    Eg = brus_energy_eV(R_m, material_key)
    if Eg <= 0:
        return float('inf')
    return H_PLANCK * C / (Eg * EV_TO_J) * 1e9


def qd_radius_for_wavelength_nm(target_nm, material_key='CdSe',
                                  R_min_m=0.5e-9, R_max_m=20e-9):
    """Find quantum dot radius that emits at target wavelength (m).

    Inverse of qd_emission_wavelength_nm, solved by bisection.

    Args:
        target_nm: desired emission wavelength (nm)
        material_key: semiconductor material
        R_min_m: minimum search radius (m)
        R_max_m: maximum search radius (m)

    Returns:
        Radius in meters.
    """
    for _ in range(200):
        R_mid = (R_min_m + R_max_m) / 2.0
        lam = qd_emission_wavelength_nm(R_mid, material_key)
        if lam < target_nm:
            R_min_m = R_mid  # too small → too blue → increase R
        else:
            R_max_m = R_mid
        if (R_max_m - R_min_m) / R_mid < 1e-8:
            break
    return (R_min_m + R_max_m) / 2.0


def qd_color_rgb(R_m, material_key='CdSe'):
    """Approximate RGB color of quantum dot emission.

    Uses the wavelength→RGB mapping from atomic_spectra.
    """
    from .atomic_spectra import wavelength_to_rgb
    lam = qd_emission_wavelength_nm(R_m, material_key)
    return wavelength_to_rgb(lam)


def confinement_energy_eV(R_m, mass_kg=None):
    """Pure kinetic confinement energy for particle in sphere (eV).

    E_conf = π²ℏ²/(2mR²)

    This is the minimum kinetic energy a particle must have
    when confined to a sphere of radius R (Heisenberg uncertainty).

    Args:
        R_m: confinement radius (m)
        mass_kg: particle mass (default: electron)

    FIRST_PRINCIPLES: ΔpΔx ~ ℏ → KE ~ ℏ²/(2mR²).
    """
    m = mass_kg if mass_kg is not None else M_ELECTRON_KG
    E_J = math.pi**2 * HBAR**2 / (2 * m * R_m**2)
    return E_J / EV_TO_J


# ══════════════════════════════════════════════════════════════════════
# DENSITY OF STATES
# ══════════════════════════════════════════════════════════════════════

def dos_3d(E_eV, mass_kg=None):
    """3D free-particle density of states (states/eV/m³).

    g(E) = (1/2π²) × (2m/ℏ²)^(3/2) × √E

    This is the bulk density of states — it grows as √E.
    Electrons fill this up to the Fermi energy.

    Args:
        E_eV: energy (eV, must be ≥ 0)
        mass_kg: particle mass (default: electron)

    Returns:
        DOS in states/(eV·m³).

    FIRST_PRINCIPLES: counting plane-wave states in 3D k-space.
    """
    if E_eV < 0:
        return 0.0
    m = mass_kg if mass_kg is not None else M_ELECTRON_KG
    E_J = E_eV * EV_TO_J
    prefactor = (1.0 / (2 * math.pi**2)) * (2 * m / HBAR**2)**1.5
    return prefactor * math.sqrt(E_J) * EV_TO_J  # convert per-J to per-eV


def dos_2d(mass_kg=None):
    """2D density of states per sub-band (states/eV/m²).

    g(E) = m / (πℏ²)   (constant within each sub-band)

    This step-function DOS is the hallmark of quantum wells.
    Each sub-band contributes a flat step at its threshold energy.

    FIRST_PRINCIPLES: counting states in 2D k-space.
    """
    m = mass_kg if mass_kg is not None else M_ELECTRON_KG
    return m / (math.pi * HBAR**2) * EV_TO_J  # states/(eV·m²)


def dos_1d(E_eV, E_subband_eV=0.0, mass_kg=None):
    """1D density of states (states/eV/m).

    g(E) = (1/π) × √(2m) / (ℏ × √(E − E_sub))

    Diverges at sub-band edge (van Hove singularity).
    This is the DOS of a quantum wire.

    Args:
        E_eV: energy (eV)
        E_subband_eV: sub-band threshold energy (eV)
        mass_kg: particle mass (default: electron)

    FIRST_PRINCIPLES: counting states in 1D k-space.
    """
    if E_eV <= E_subband_eV:
        return 0.0
    m = mass_kg if mass_kg is not None else M_ELECTRON_KG
    dE_J = (E_eV - E_subband_eV) * EV_TO_J
    return (1.0 / math.pi) * math.sqrt(2 * m) / (HBAR * math.sqrt(dE_J)) * EV_TO_J


def dos_0d(E_eV, levels_eV, broadening_eV=0.01):
    """0D density of states — quantum dot (states/eV).

    g(E) = Σᵢ δ(E − Eᵢ)  (ideally delta functions)

    Broadened as Lorentzians with width broadening_eV for numerical use.

    Args:
        E_eV: energy (eV)
        levels_eV: list of discrete energy levels (eV)
        broadening_eV: Lorentzian half-width (eV)

    FIRST_PRINCIPLES: discrete spectrum → delta-function DOS.
    """
    g = 0.0
    gamma = broadening_eV
    for E_i in levels_eV:
        # Lorentzian: (1/π) × γ / ((E−Eᵢ)² + γ²)
        g += (1.0 / math.pi) * gamma / ((E_eV - E_i)**2 + gamma**2)
    return g


# ══════════════════════════════════════════════════════════════════════
# QUANTUM WELL (2D confinement)
# ══════════════════════════════════════════════════════════════════════

def quantum_well_subbands_eV(L_m, n_max=5, mass_kg=None):
    """Energy levels of infinite quantum well (eV).

    Eₙ = n²π²ℏ²/(2mL²)  for n = 1, 2, ..., n_max

    These are the sub-band threshold energies.
    Within each sub-band, electrons move freely in 2D.

    Args:
        L_m: well width (m)
        n_max: number of sub-bands to compute
        mass_kg: particle mass (default: electron)

    Returns:
        List of (n, E_eV) tuples.
    """
    m = mass_kg if mass_kg is not None else M_ELECTRON_KG
    levels = []
    for n in range(1, n_max + 1):
        E_J = n**2 * math.pi**2 * HBAR**2 / (2 * m * L_m**2)
        levels.append((n, E_J / EV_TO_J))
    return levels


def quantum_wire_subbands_eV(Ly_m, Lz_m, n_max=3, mass_kg=None):
    """Energy levels of quantum wire (2D confinement, free in x).

    E(ny, nz) = π²ℏ²/(2m) × (ny²/Ly² + nz²/Lz²)

    Returns sorted list of (ny, nz, E_eV).
    """
    m = mass_kg if mass_kg is not None else M_ELECTRON_KG
    levels = []
    for ny in range(1, n_max + 1):
        for nz in range(1, n_max + 1):
            E_J = math.pi**2 * HBAR**2 / (2 * m) * (
                ny**2 / Ly_m**2 + nz**2 / Lz_m**2
            )
            levels.append((ny, nz, E_J / EV_TO_J))
    levels.sort(key=lambda x: x[2])
    return levels


# ══════════════════════════════════════════════════════════════════════
# SIZE-DEPENDENT PROPERTIES
# ══════════════════════════════════════════════════════════════════════

def critical_radius_nm(material_key='CdSe'):
    """Radius below which quantum confinement dominates (nm).

    R_crit ~ exciton Bohr radius = ε_r × a₀ × m_e / μ

    Below this radius, the particle is "quantum confined" and
    the band gap differs significantly from bulk.

    FIRST_PRINCIPLES: when confinement energy ~ Coulomb energy,
    the particle "feels" the walls of its box.
    """
    if material_key not in _QD_MATERIALS:
        raise KeyError(f"Unknown material '{material_key}'")
    me_frac, mh_frac, eps_r, _ = _QD_MATERIALS[material_key]
    mu_frac = me_frac * mh_frac / (me_frac + mh_frac)
    # Exciton Bohr radius
    a_exc = eps_r * BOHR_RADIUS / mu_frac
    return a_exc * 1e9  # m → nm


def size_vs_gap(material_key='CdSe', R_min_nm=1.0, R_max_nm=10.0, n_pts=20):
    """Compute band gap vs size curve for quantum dots.

    Returns list of (R_nm, Eg_eV, lambda_nm).
    """
    points = []
    for i in range(n_pts):
        R_nm = R_min_nm + (R_max_nm - R_min_nm) * i / (n_pts - 1)
        R_m = R_nm * 1e-9
        Eg = brus_energy_eV(R_m, material_key)
        lam = qd_emission_wavelength_nm(R_m, material_key)
        points.append((R_nm, Eg, lam))
    return points


# ══════════════════════════════════════════════════════════════════════
# REPORTS (Rule 9)
# ══════════════════════════════════════════════════════════════════════

def quantum_wells_report(L_nm=5.0, material_key='CdSe', R_nm=2.0):
    """Report on quantum confinement effects.

    Args:
        L_nm: well width (nm) for particle-in-box calculations
        material_key: semiconductor for quantum dot
        R_nm: dot radius (nm)
    """
    L_m = L_nm * 1e-9
    R_m = R_nm * 1e-9

    return {
        'well_width_nm': L_nm,
        'box_ground_state_eV': box_ground_state_eV(L_m),
        'box_1_2_transition_nm': box_transition_wavelength_nm(2, 1, L_m),
        'well_subbands_eV': quantum_well_subbands_eV(L_m, n_max=5),
        'dos_2d_per_subband': dos_2d(),
        'material': material_key,
        'dot_radius_nm': R_nm,
        'brus_gap_eV': brus_energy_eV(R_m, material_key),
        'dot_emission_nm': qd_emission_wavelength_nm(R_m, material_key),
        'dot_color_rgb': qd_color_rgb(R_m, material_key),
        'critical_radius_nm': critical_radius_nm(material_key),
        'bulk_gap_eV': _QD_MATERIALS[material_key][3],
    }


def full_report(L_nm=5.0, material_key='CdSe', R_nm=2.0):
    """Complete quantum confinement report (Rule 9)."""
    report = quantum_wells_report(L_nm, material_key, R_nm)

    # Add finite well example
    report['finite_well_depth_eV'] = 1.0
    report['finite_well_bound_states'] = finite_well_bound_states(1.0, L_nm * 1e-9)
    report['finite_well_ground_eV'] = finite_well_ground_state_eV(1.0, L_nm * 1e-9)

    # Add all available QD materials
    report['available_materials'] = sorted(_QD_MATERIALS.keys())

    return report
