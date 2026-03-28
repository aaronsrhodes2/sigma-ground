"""
Quantum tunneling — transmission through classically forbidden barriers.

A quantum particle can pass through a potential barrier that classical
mechanics says is impenetrable. The transmission probability depends
exponentially on barrier width, height, and particle mass.

This module unifies tunneling physics that was previously scattered:
  - radioactive_decay.py: Gamow tunneling (nuclear Coulomb barrier)
  - nucleosynthesis.py: stellar fusion (Coulomb penetration)
  - NEW: rectangular, triangular, and arbitrary barriers
  - NEW: WKB approximation for smooth barriers
  - NEW: tunnel diode I-V, STM current, field emission

Physics chain:
  1. Rectangular barrier (FIRST_PRINCIPLES — Schrödinger equation)
     T = [1 + V₀²sinh²(κd)/(4E(V₀−E))]⁻¹
     where κ = √(2m(V₀−E))/ℏ

  2. WKB approximation (FIRST_PRINCIPLES — semiclassical)
     T ≈ exp(−2 ∫ κ(x) dx)
     where κ(x) = √(2m(V(x)−E))/ℏ over classically forbidden region

  3. Fowler-Nordheim field emission (FIRST_PRINCIPLES)
     J ∝ E² × exp(−4√(2m)Φ^(3/2) / (3eℏE))
     Triangular barrier from applied electric field.

  4. STM tunneling current (FIRST_PRINCIPLES)
     I ∝ V × exp(−2κd) where κ = √(2mΦ)/ℏ
     Exponential sensitivity to tip-sample distance d.

σ-dependence:
  Tunneling through EM barriers (work function, band gap) uses m_e → σ-INVARIANT.
  Nuclear tunneling (Gamow) uses nucleon mass → σ-DEPENDENT (already in
  radioactive_decay.py and nucleosynthesis.py).
  This module focuses on EM tunneling (electrons).

□σ = −ξR
"""

import math
from ..constants import (
    HBAR, C, E_CHARGE, EPS_0, M_ELECTRON_KG,
    H_PLANCK, EV_TO_J, SIGMA_HERE,
)


# ══════════════════════════════════════════════════════════════════════
# RECTANGULAR BARRIER — Exact Solution
# ══════════════════════════════════════════════════════════════════════

def rectangular_barrier_T(E_eV, V0_eV, d_m, mass_kg=None):
    """Transmission coefficient through rectangular barrier.

    For E < V₀ (tunneling regime):
      T = [1 + V₀² sinh²(κd) / (4E(V₀−E))]⁻¹
      where κ = √(2m(V₀−E)) / ℏ

    For E > V₀ (above-barrier, still quantum — oscillates):
      T = [1 + V₀² sin²(k'd) / (4E(E−V₀))]⁻¹
      where k' = √(2m(E−V₀)) / ℏ

    For E = V₀: T = [1 + mV₀d²/(2ℏ²)]⁻¹

    Args:
        E_eV: particle kinetic energy (eV)
        V0_eV: barrier height (eV)
        d_m: barrier width (m)
        mass_kg: particle mass (default: electron)

    Returns:
        Transmission probability T ∈ [0, 1].

    FIRST_PRINCIPLES: matching ψ and dψ/dx at both barrier edges.

    Reference: Griffiths, Introduction to Quantum Mechanics, §2.6.
    """
    if E_eV <= 0:
        return 0.0
    if d_m <= 0:
        return 1.0

    m = mass_kg if mass_kg is not None else M_ELECTRON_KG
    E_J = E_eV * EV_TO_J
    V0_J = V0_eV * EV_TO_J

    if abs(E_J - V0_J) < V0_J * 1e-10:
        # E ≈ V₀: limiting form
        arg = m * V0_J * d_m**2 / (2 * HBAR**2)
        return 1.0 / (1.0 + arg)

    if E_J < V0_J:
        # Tunneling regime: E < V₀
        kappa = math.sqrt(2 * m * (V0_J - E_J)) / HBAR
        kd = kappa * d_m
        # Guard against overflow in sinh
        if kd > 500:
            return 0.0
        sinh_kd = math.sinh(kd)
        denom = 1.0 + V0_J**2 * sinh_kd**2 / (4 * E_J * (V0_J - E_J))
        return 1.0 / denom
    else:
        # Above-barrier: E > V₀ (still has quantum oscillations)
        k_prime = math.sqrt(2 * m * (E_J - V0_J)) / HBAR
        sin_kd = math.sin(k_prime * d_m)
        denom = 1.0 + V0_J**2 * sin_kd**2 / (4 * E_J * (E_J - V0_J))
        return 1.0 / denom


def rectangular_barrier_R(E_eV, V0_eV, d_m, mass_kg=None):
    """Reflection coefficient for rectangular barrier.

    R = 1 − T (probability conservation).
    """
    return 1.0 - rectangular_barrier_T(E_eV, V0_eV, d_m, mass_kg)


def decay_constant_m(V0_eV, E_eV, mass_kg=None):
    """Exponential decay constant κ in barrier (1/m).

    κ = √(2m(V₀−E)) / ℏ

    The wavefunction amplitude decays as exp(−κx) inside the barrier.
    Larger κ → faster decay → less tunneling.

    Args:
        V0_eV: barrier height (eV)
        E_eV: particle energy (eV, E < V₀)
        mass_kg: particle mass (default: electron)

    FIRST_PRINCIPLES: Schrödinger equation in classically forbidden region.
    """
    if E_eV >= V0_eV:
        return 0.0
    m = mass_kg if mass_kg is not None else M_ELECTRON_KG
    return math.sqrt(2 * m * (V0_eV - E_eV) * EV_TO_J) / HBAR


# ══════════════════════════════════════════════════════════════════════
# WKB APPROXIMATION — General Barriers
# ══════════════════════════════════════════════════════════════════════

def wkb_transmission(V_func, E_eV, x1_m, x2_m, mass_kg=None, n_steps=1000):
    """WKB transmission coefficient for arbitrary barrier shape.

    T ≈ exp(−2 ∫_{x₁}^{x₂} κ(x) dx)

    where κ(x) = √(2m(V(x)−E)) / ℏ in the classically forbidden region
    [x₁, x₂] where V(x) > E.

    The WKB (Wentzel-Kramers-Brillouin) approximation is valid when
    the barrier varies slowly compared to the de Broglie wavelength.
    It's exact for constant barriers and excellent for smooth ones.

    Args:
        V_func: callable V(x) returning potential in eV at position x (m)
        E_eV: particle energy (eV)
        x1_m, x2_m: integration limits (barrier region, m)
        mass_kg: particle mass (default: electron)
        n_steps: number of integration steps

    Returns:
        Transmission probability.

    FIRST_PRINCIPLES: semiclassical limit of Schrödinger equation.
    """
    m = mass_kg if mass_kg is not None else M_ELECTRON_KG
    dx = (x2_m - x1_m) / n_steps
    integral = 0.0

    for i in range(n_steps):
        x = x1_m + (i + 0.5) * dx
        V = V_func(x)
        dV = V - E_eV
        if dV > 0:
            integral += math.sqrt(2 * m * dV * EV_TO_J) * dx

    integral /= HBAR
    # Guard against overflow
    exponent = -2 * integral
    if exponent < -500:
        return 0.0
    return math.exp(exponent)


def wkb_rectangular(E_eV, V0_eV, d_m, mass_kg=None):
    """WKB approximation for rectangular barrier (for comparison).

    T_WKB = exp(−2κd) where κ = √(2m(V₀−E))/ℏ

    This is the leading-order WKB result. Compare with the exact
    rectangular_barrier_T() to see the WKB error (typically
    underestimates T by a factor of ~4E(V₀-E)/V₀² for thin barriers).
    """
    if E_eV >= V0_eV:
        return 1.0
    kappa = decay_constant_m(V0_eV, E_eV, mass_kg)
    exponent = -2 * kappa * d_m
    if exponent < -500:
        return 0.0
    return math.exp(exponent)


# ══════════════════════════════════════════════════════════════════════
# DOUBLE BARRIER — Resonant Tunneling
# ══════════════════════════════════════════════════════════════════════

def double_barrier_resonances_eV(V0_eV, d_barrier_m, d_well_m, mass_kg=None,
                                  n_max=5):
    """Resonant energy levels of a double-barrier structure (eV).

    Two barriers of height V₀ and width d_barrier, separated by a well
    of width d_well. At resonance energies, T → 1 even though each
    individual barrier has T << 1. This is the physics of the resonant
    tunnel diode (RTD).

    Resonance condition: standing wave in well → same as particle-in-box
    with effective well width d_well.

    Eₙ ≈ n²π²ℏ²/(2m × d_well²)  for E < V₀

    FIRST_PRINCIPLES: constructive interference of multiply-reflected waves.
    """
    m = mass_kg if mass_kg is not None else M_ELECTRON_KG
    resonances = []
    for n in range(1, n_max + 1):
        E_J = n**2 * math.pi**2 * HBAR**2 / (2 * m * d_well_m**2)
        E_eV = E_J / EV_TO_J
        if E_eV < V0_eV:
            resonances.append((n, E_eV))
    return resonances


# ══════════════════════════════════════════════════════════════════════
# FIELD EMISSION — Fowler-Nordheim
# ══════════════════════════════════════════════════════════════════════

def fowler_nordheim_current_density(E_field_V_m, phi_eV, mass_kg=None):
    """Fowler-Nordheim field emission current density (A/m²).

    J = (e³E²)/(8πhΦ) × t²(y) × exp(−4√(2m)Φ^(3/2) v(y) / (3eℏE))

    Simplified (t≈1, v≈1 approximation):
    J ≈ (e³/(8πhΦ)) × E² × exp(−4√(2m)Φ^(3/2) / (3eℏ|E|))

    An applied electric field makes the barrier triangular, allowing
    electrons to tunnel out of the metal. This is cold emission —
    no thermal activation needed.

    Args:
        E_field_V_m: applied electric field (V/m), must be > 0
        phi_eV: work function (eV)
        mass_kg: particle mass (default: electron)

    Returns:
        Current density in A/m².

    FIRST_PRINCIPLES: WKB through triangular barrier V(x) = Φ − eEx.

    Reference: Fowler & Nordheim (1928) Proc. R. Soc. A 119, 173.
    """
    if E_field_V_m <= 0:
        return 0.0
    m = mass_kg if mass_kg is not None else M_ELECTRON_KG
    phi_J = phi_eV * EV_TO_J

    # Prefactor
    prefactor = E_CHARGE**3 / (8 * math.pi * H_PLANCK * phi_J)

    # Exponent
    exponent = -4 * math.sqrt(2 * m) * phi_J**1.5 / (
        3 * E_CHARGE * HBAR * E_field_V_m
    )

    if exponent < -500:
        return 0.0

    return prefactor * E_field_V_m**2 * math.exp(exponent)


def field_emission_onset_V_m(phi_eV, J_target=1e4):
    """Electric field needed for field emission onset (V/m).

    Finds E such that J_FN ≈ J_target (default 10⁴ A/m² ≈ measurable).

    Typical values:
      Φ = 4.5 eV (tungsten) → E ≈ 3-5 GV/m (with tip enhancement)

    Uses approximate inversion of Fowler-Nordheim.
    """
    phi_J = phi_eV * EV_TO_J
    # Rough estimate: exp term dominates, need exponent ≈ −20 to −30
    # E ≈ 4√(2m)Φ^(3/2) / (3eℏ × 25)
    E_est = 4 * math.sqrt(2 * M_ELECTRON_KG) * phi_J**1.5 / (
        3 * E_CHARGE * HBAR * 25.0
    )
    return E_est


# Work functions (MEASURED, CRC Handbook)
WORK_FUNCTIONS_EV = {
    'tungsten':  4.55,   # MEASURED: photoelectric (Eastman 1970)
    'iron':      4.50,   # MEASURED
    'copper':    4.65,   # MEASURED
    'gold':      5.10,   # MEASURED
    'aluminum':  4.28,   # MEASURED
    'nickel':    5.15,   # MEASURED
    'titanium':  4.33,   # MEASURED
    'silicon':   4.85,   # MEASURED (intrinsic)
    'platinum':  5.65,   # MEASURED
    'silver':    4.26,   # MEASURED
}


# ══════════════════════════════════════════════════════════════════════
# STM — Scanning Tunneling Microscope
# ══════════════════════════════════════════════════════════════════════

def stm_current(V_bias_V, d_m, phi_eV, area_m2=1e-18):
    """STM tunneling current (A).

    I ∝ V × exp(−2κd)

    where κ = √(2mΦ)/ℏ and Φ is the average work function.

    The exponential sensitivity to distance d is why STM achieves
    atomic resolution: 0.1 nm change in d → 10× change in current.

    Args:
        V_bias_V: tip-sample bias voltage (V)
        d_m: tip-sample distance (m)
        phi_eV: average work function (eV)
        area_m2: effective tunneling area (m², default 1 nm²)

    Returns:
        Tunneling current in Amperes.

    FIRST_PRINCIPLES: planar tunneling junction with barrier = work function.

    Reference: Binnig & Rohrer (1987) Rev. Mod. Phys. 59, 615.
    """
    kappa = math.sqrt(2 * M_ELECTRON_KG * phi_eV * EV_TO_J) / HBAR
    exponent = -2 * kappa * d_m
    if exponent < -500:
        return 0.0

    # Conductance quantum-based prefactor (per unit area)
    G0 = E_CHARGE**2 / (2 * math.pi * HBAR)  # ≈ 7.75e-5 S
    return G0 * area_m2 * abs(V_bias_V) * math.exp(exponent)


def stm_resolution_m(phi_eV):
    """Lateral resolution of STM (m).

    Δx ~ √(d/κ) where d ~ 0.5 nm typical tip-sample distance
    and κ = √(2mΦ)/ℏ.

    For Φ ≈ 4 eV: κ ≈ 10.2 nm⁻¹, d ≈ 0.5 nm → Δx ≈ 0.2 nm.
    This is why STM can image individual atoms.

    Returns resolution in meters.
    """
    kappa = math.sqrt(2 * M_ELECTRON_KG * phi_eV * EV_TO_J) / HBAR
    d_typical = 0.5e-9  # 0.5 nm typical tip-sample distance
    return math.sqrt(d_typical / kappa)


def stm_decay_per_angstrom(phi_eV):
    """Current decay factor per Ångström of tip retraction.

    Factor = exp(−2κ × 1Å)

    For Φ = 4 eV: factor ≈ 0.13 → current drops to 13% per Ångström.
    This extreme sensitivity is the basis of atomic-resolution imaging.
    """
    kappa = math.sqrt(2 * M_ELECTRON_KG * phi_eV * EV_TO_J) / HBAR
    return math.exp(-2 * kappa * 1e-10)


# ══════════════════════════════════════════════════════════════════════
# TUNNEL DIODE (Esaki diode)
# ══════════════════════════════════════════════════════════════════════

def tunnel_diode_peak_current(Eg_eV, d_depletion_m, V_peak_V=None,
                               n_doping_m3=1e25, mass_kg=None):
    """Estimate peak tunnel current density in Esaki diode (A/m²).

    In a heavily doped p-n junction, the depletion region is thin enough
    for electrons to tunnel directly. The I-V curve shows negative
    differential resistance (NDR): current DECREASES with increasing voltage.

    Peak occurs when filled states on one side align with empty states
    on the other.

    J_peak ∝ n × exp(−2κ × d)

    where κ = √(2m* × Eg) / ℏ and d is the depletion width.

    FIRST_PRINCIPLES: interband tunneling through thin p-n junction.

    Reference: Esaki, L. (1958) Phys. Rev. 109, 603.
    """
    m = mass_kg if mass_kg is not None else M_ELECTRON_KG
    kappa = math.sqrt(2 * m * Eg_eV * EV_TO_J) / HBAR
    exponent = -2 * kappa * d_depletion_m
    if exponent < -500:
        return 0.0

    # Current density: J = e × n × v_tunnel × T
    # v_tunnel ~ ℏκ/m (tunneling velocity)
    v_tunnel = HBAR * kappa / m
    J = E_CHARGE * n_doping_m3 * v_tunnel * math.exp(exponent)
    return J


# ══════════════════════════════════════════════════════════════════════
# ALPHA DECAY CONNECTION
# ══════════════════════════════════════════════════════════════════════

def gamow_factor(Z_daughter, E_alpha_MeV, R_nuclear_m, mass_kg=None):
    """Gamow tunneling factor for alpha decay (dimensionless).

    G = exp(−2 ∫ κ(r) dr) from R_nuclear to R_coulomb

    where the Coulomb barrier V(r) = Z_d × Z_α × e² / (4πε₀r)
    and R_coulomb = Z_d × Z_α × e² / (4πε₀ × E_α).

    The alpha particle must tunnel through the Coulomb barrier to escape.
    G determines the half-life: t_{1/2} ∝ 1/G.

    This is the same physics as in radioactive_decay.py but exposed
    as a general tunneling function.

    Args:
        Z_daughter: atomic number of daughter nucleus
        E_alpha_MeV: kinetic energy of alpha particle (MeV)
        R_nuclear_m: nuclear radius (m)
        mass_kg: alpha particle mass (default: 4 AMU)

    Returns:
        Gamow factor (dimensionless, typically 10⁻¹⁵ to 10⁻⁵⁰).

    FIRST_PRINCIPLES: WKB through Coulomb barrier.

    Reference: Gamow, G. (1928) Z. Phys. 51, 204.
    """
    from ..constants import AMU_KG, MEV_TO_J

    Z_alpha = 2
    m_alpha = mass_kg if mass_kg is not None else 4 * AMU_KG
    E_J = E_alpha_MeV * MEV_TO_J

    # Coulomb turning point
    k_e = E_CHARGE**2 / (4 * math.pi * EPS_0)
    R_coulomb = Z_daughter * Z_alpha * k_e / E_J

    if R_coulomb <= R_nuclear_m:
        return 1.0  # no barrier

    # Sommerfeld parameter
    eta = Z_daughter * Z_alpha * E_CHARGE**2 / (
        4 * math.pi * EPS_0 * HBAR * math.sqrt(2 * E_J / m_alpha)
    )

    # Gamow integral (analytic for Coulomb)
    rho = R_nuclear_m / R_coulomb
    if rho >= 1:
        return 1.0
    integral = math.acos(math.sqrt(rho)) - math.sqrt(rho * (1 - rho))
    G = math.exp(-2 * eta * integral * 2)

    # Clamp to physical range
    return max(G, 0.0)


# ══════════════════════════════════════════════════════════════════════
# TUNNELING TIME
# ══════════════════════════════════════════════════════════════════════

def phase_time_s(E_eV, V0_eV, d_m, mass_kg=None):
    """Büttiker-Landauer phase tunneling time (s).

    τ_phase = ℏ × d(arg T)/dE

    For rectangular barrier (E < V₀):
    τ ≈ m d / (ℏκ)  where κ = √(2m(V₀−E))/ℏ

    This is one definition of tunneling time. The "tunneling time problem"
    is one of the oldest debates in quantum mechanics (1932-present).

    The phase time can be shorter than d/c (Hartman effect),
    but this does NOT violate causality — it's a reshaping of the
    wave packet, not superluminal signal propagation.

    Args:
        E_eV: particle energy (eV)
        V0_eV: barrier height (eV)
        d_m: barrier width (m)
        mass_kg: particle mass (default: electron)

    Returns:
        Phase time in seconds.

    Reference: Büttiker & Landauer (1982) Phys. Rev. Lett. 49, 1739.
    """
    if E_eV >= V0_eV:
        # Above barrier: just transit time
        m = mass_kg if mass_kg is not None else M_ELECTRON_KG
        v = math.sqrt(2 * E_eV * EV_TO_J / m)
        return d_m / v

    m = mass_kg if mass_kg is not None else M_ELECTRON_KG
    kappa = math.sqrt(2 * m * (V0_eV - E_eV) * EV_TO_J) / HBAR
    if kappa * d_m > 200:
        # Opaque barrier limit: τ → m/(ℏκ²) (saturates — Hartman effect)
        return m / (HBAR * kappa**2)
    return m * d_m / (HBAR * kappa)


# ══════════════════════════════════════════════════════════════════════
# REPORTS (Rule 9)
# ══════════════════════════════════════════════════════════════════════

def tunneling_report(E_eV=1.0, V0_eV=2.0, d_nm=1.0, phi_eV=4.5):
    """Report on tunneling physics.

    Args:
        E_eV: particle energy (eV)
        V0_eV: barrier height (eV)
        d_nm: barrier width (nm)
        phi_eV: work function for STM/field emission (eV)
    """
    d_m = d_nm * 1e-9

    return {
        'E_eV': E_eV,
        'V0_eV': V0_eV,
        'barrier_width_nm': d_nm,
        'transmission_exact': rectangular_barrier_T(E_eV, V0_eV, d_m),
        'transmission_WKB': wkb_rectangular(E_eV, V0_eV, d_m),
        'reflection': rectangular_barrier_R(E_eV, V0_eV, d_m),
        'decay_constant_per_nm': decay_constant_m(V0_eV, E_eV) * 1e-9,
        'phase_time_s': phase_time_s(E_eV, V0_eV, d_m),
        'work_function_eV': phi_eV,
        'stm_current_1V_0.5nm': stm_current(1.0, 0.5e-9, phi_eV),
        'stm_resolution_nm': stm_resolution_m(phi_eV) * 1e9,
        'stm_decay_per_angstrom': stm_decay_per_angstrom(phi_eV),
        'fn_current_5GVm': fowler_nordheim_current_density(5e9, phi_eV),
    }


def full_report(E_eV=1.0, V0_eV=2.0, d_nm=1.0, phi_eV=4.5):
    """Complete tunneling report (Rule 9)."""
    report = tunneling_report(E_eV, V0_eV, d_nm, phi_eV)

    # Add double-barrier resonances
    d_barrier = d_nm * 1e-9
    d_well = 2 * d_nm * 1e-9
    report['double_barrier_resonances'] = double_barrier_resonances_eV(
        V0_eV, d_barrier, d_well
    )

    # Work functions table
    report['work_functions_eV'] = dict(WORK_FUNCTIONS_EV)

    return report
