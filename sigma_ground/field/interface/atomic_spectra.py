"""
Atomic spectra — energy levels, spectral lines, and selection rules.

Every element has a spectral fingerprint. This module computes it from Z.

Physics chain:
  1. Hydrogen atom (Bohr 1913, Schrödinger 1926 — FIRST_PRINCIPLES)
     Eₙ = −13.6 eV / n²  (exact for hydrogen)
     Eₙ = −13.6 eV × Z_eff² / n²  (multi-electron, Slater screening)

  2. Rydberg constant (DERIVED from measured constants)
     R∞ = m_e e⁴ / (8 ε₀² h³ c) = α² m_e c / (2h)

  3. Spectral series (Rydberg formula — FIRST_PRINCIPLES)
     1/λ = R∞ Z² (1/n_f² − 1/n_i²)
     Lyman:   n_f=1  (UV)
     Balmer:  n_f=2  (visible)
     Paschen: n_f=3  (near-IR)
     Brackett: n_f=4 (IR)
     Pfund:   n_f=5  (far-IR)

  4. Multi-electron atoms (Slater screening — APPROXIMATION)
     E_nl = −13.6 eV × (Z_eff / n*)²
     Z_eff from Slater's rules (element.py)

  5. Fine structure (spin-orbit coupling — FIRST_PRINCIPLES)
     ΔE_fs = −E_n × α² Z² / n × [1/j − 1/(j+1)] / 2
     where j = l ± ½

  6. Zeeman effect (external magnetic field — FIRST_PRINCIPLES)
     ΔE = m_j g_J μ_B B
     where g_J = Landé g-factor

  7. Selection rules (FIRST_PRINCIPLES — dipole matrix elements)
     Δl = ±1, Δm_l = 0, ±1, Δm_s = 0

  8. Quantum harmonic oscillator (FIRST_PRINCIPLES)
     Eₙ = ℏω(n + ½)
     Zero-point energy E₀ = ℏω/2

σ-dependence:
  Hydrogen energy levels are electromagnetic → σ-INVARIANT.
  Fine structure depends on α (EM) → σ-INVARIANT.
  BUT: reduced mass corrections for atoms with σ-dependent nuclear mass
  shift the Rydberg constant slightly: R_atom = R∞ × μ/(m_e).
  For hydrogen μ = m_e × m_p/(m_e + m_p) and m_p shifts with σ via QCD.
  Effect: ~5×10⁻⁴ relative shift at σ=0, decreasing lines spacing at high σ.

□σ = −ξR
"""

import math
from ..constants import (
    HBAR, C, E_CHARGE, EPS_0, M_ELECTRON_KG, H_PLANCK,
    ALPHA, BOHR_RADIUS, MU_BOHR, K_B, EV_TO_J,
    PROTON_TOTAL_MEV, PROTON_BARE_MEV, PROTON_QCD_MEV,
    MEV_TO_J, SIGMA_HERE,
)


# ══════════════════════════════════════════════════════════════════════
# FUNDAMENTAL CONSTANTS — ALL DERIVED
# ══════════════════════════════════════════════════════════════════════

# Rydberg energy (ground-state ionization of hydrogen)
# DERIVED: E_R = m_e e⁴ / (2(4πε₀)²ℏ²) = α² m_e c² / 2
# = 13.6057 eV (MEASURED: 13.5984 eV — the 0.05% difference is
# reduced-mass correction for finite proton mass)
RYDBERG_ENERGY_J = 0.5 * ALPHA**2 * M_ELECTRON_KG * C**2
RYDBERG_ENERGY_EV = RYDBERG_ENERGY_J / EV_TO_J  # ≈ 13.606 eV

# Rydberg constant (spectroscopy)
# DERIVED: R∞ = E_R / (hc) = α² m_e c / (2h)
RYDBERG_CONSTANT = RYDBERG_ENERGY_J / (H_PLANCK * C)  # m⁻¹ ≈ 1.0974e7

# Proton mass in kg (for reduced mass)
_M_PROTON_KG = PROTON_TOTAL_MEV * MEV_TO_J / C**2


# ══════════════════════════════════════════════════════════════════════
# HYDROGEN-LIKE ENERGY LEVELS
# ══════════════════════════════════════════════════════════════════════

def _reduced_mass(m1_kg, m2_kg):
    """Two-body reduced mass μ = m₁m₂/(m₁+m₂).

    FIRST_PRINCIPLES: Classical mechanics → quantum two-body problem.
    """
    return m1_kg * m2_kg / (m1_kg + m2_kg)


def hydrogen_reduced_mass(sigma=0.0):
    """Reduced mass of electron-proton system (kg).

    μ = m_e × m_p / (m_e + m_p)

    At σ=0: μ ≈ 0.99946 m_e (proton is ~1836× heavier).
    At σ>0: proton QCD mass increases → μ → m_e (ratio improves).

    σ-dependence: proton mass shifts via QCD binding.
    """
    m_p = _proton_mass_kg(sigma)
    return _reduced_mass(M_ELECTRON_KG, m_p)


def _proton_mass_kg(sigma=0.0):
    """Proton mass with σ-dependent QCD scaling.

    m_p(σ) = m_bare + m_QCD × e^σ

    m_bare is Higgs-origin (σ-invariant).
    m_QCD is QCD binding energy (scales with Λ_QCD).
    """
    m_total_mev = PROTON_BARE_MEV + PROTON_QCD_MEV * math.exp(sigma)
    return m_total_mev * MEV_TO_J / C**2


def hydrogen_energy_eV(n, sigma=0.0):
    """Energy of hydrogen level n (eV, negative = bound).

    Eₙ = −μ/m_e × E_R / n²

    Includes reduced-mass correction for finite proton mass.
    This gives 13.598 eV for n=1 (MEASURED: 13.598 eV).

    Args:
        n: principal quantum number (1, 2, 3, ...)
        sigma: σ-field value

    Returns:
        Energy in eV (negative for bound states).

    FIRST_PRINCIPLES: Schrödinger equation for Coulomb potential.
    """
    if n < 1:
        raise ValueError(f"n must be ≥ 1, got {n}")
    mu = hydrogen_reduced_mass(sigma)
    # E_n = -(μ/m_e) × E_R / n²
    return -(mu / M_ELECTRON_KG) * RYDBERG_ENERGY_EV / n**2


def hydrogen_like_energy_eV(Z, n, sigma=0.0):
    """Energy of hydrogen-like ion with nuclear charge Z.

    Eₙ = −μ/m_e × E_R × Z² / n²

    Valid for single-electron ions: H, He⁺, Li²⁺, etc.

    Args:
        Z: nuclear charge
        n: principal quantum number
        sigma: σ-field value

    FIRST_PRINCIPLES: Coulomb potential with charge Ze.
    """
    if n < 1:
        raise ValueError(f"n must be ≥ 1, got {n}")
    mu = hydrogen_reduced_mass(sigma)
    return -(mu / M_ELECTRON_KG) * RYDBERG_ENERGY_EV * Z**2 / n**2


def ionization_energy_hydrogen_eV(sigma=0.0):
    """Ionization energy of hydrogen from ground state (eV).

    IE = −E₁ = μ/m_e × E_R ≈ 13.598 eV
    MEASURED: 13.5984 eV (NIST)
    """
    return -hydrogen_energy_eV(1, sigma)


# ══════════════════════════════════════════════════════════════════════
# SPECTRAL LINES — Rydberg Formula
# ══════════════════════════════════════════════════════════════════════

def transition_energy_eV(Z, n_upper, n_lower, sigma=0.0):
    """Energy of photon emitted in transition n_upper → n_lower (eV).

    ΔE = E_R × Z² × (1/n_lower² − 1/n_upper²) × μ/m_e

    Args:
        Z: nuclear charge (1 for hydrogen)
        n_upper: initial level
        n_lower: final level (n_lower < n_upper)
        sigma: σ-field value

    Returns:
        Photon energy in eV (positive for emission).

    FIRST_PRINCIPLES: energy conservation E_photon = E_upper − E_lower.
    """
    if n_upper <= n_lower:
        raise ValueError(f"n_upper ({n_upper}) must be > n_lower ({n_lower})")
    E_up = hydrogen_like_energy_eV(Z, n_upper, sigma)
    E_lo = hydrogen_like_energy_eV(Z, n_lower, sigma)
    return E_up - E_lo  # positive (emission): upper is less negative


def transition_wavelength_nm(Z, n_upper, n_lower, sigma=0.0):
    """Wavelength of spectral line from transition n_upper → n_lower (nm).

    λ = hc / ΔE

    Args:
        Z: nuclear charge
        n_upper: initial level
        n_lower: final level
        sigma: σ-field value

    Returns:
        Wavelength in nanometers.

    FIRST_PRINCIPLES: E = hc/λ.
    """
    dE = transition_energy_eV(Z, n_upper, n_lower, sigma)
    dE_J = dE * EV_TO_J
    wavelength_m = H_PLANCK * C / dE_J
    return wavelength_m * 1e9  # m → nm


def transition_frequency_Hz(Z, n_upper, n_lower, sigma=0.0):
    """Frequency of spectral line (Hz).

    ν = ΔE / h

    FIRST_PRINCIPLES: Planck-Einstein relation E = hν.
    """
    dE = transition_energy_eV(Z, n_upper, n_lower, sigma)
    return dE * EV_TO_J / H_PLANCK


def transition_wavenumber(Z, n_upper, n_lower, sigma=0.0):
    """Wavenumber of spectral line (cm⁻¹, spectroscopic convention).

    ν̃ = 1/λ = R∞ Z² (1/n_lower² − 1/n_upper²)

    This is how spectroscopists measure: cm⁻¹.
    """
    lam_nm = transition_wavelength_nm(Z, n_upper, n_lower, sigma)
    lam_cm = lam_nm * 1e-7  # nm → cm
    return 1.0 / lam_cm


# ── Named Series ─────────────────────────────────────────────────────

def lyman_series(Z=1, n_max=7, sigma=0.0):
    """Lyman series: transitions to n=1 (UV).

    Returns list of (n_upper, wavelength_nm, energy_eV).
    """
    lines = []
    for n in range(2, n_max + 1):
        lam = transition_wavelength_nm(Z, n, 1, sigma)
        E = transition_energy_eV(Z, n, 1, sigma)
        lines.append((n, lam, E))
    return lines


def balmer_series(Z=1, n_max=7, sigma=0.0):
    """Balmer series: transitions to n=2 (visible).

    Hα (656 nm, red), Hβ (486 nm, cyan), Hγ (434 nm, violet), Hδ (410 nm)
    MEASURED: Balmer 1885.
    """
    lines = []
    for n in range(3, n_max + 1):
        lam = transition_wavelength_nm(Z, n, 2, sigma)
        E = transition_energy_eV(Z, n, 2, sigma)
        lines.append((n, lam, E))
    return lines


def paschen_series(Z=1, n_max=8, sigma=0.0):
    """Paschen series: transitions to n=3 (near-IR)."""
    lines = []
    for n in range(4, n_max + 1):
        lam = transition_wavelength_nm(Z, n, 3, sigma)
        E = transition_energy_eV(Z, n, 3, sigma)
        lines.append((n, lam, E))
    return lines


def series_limit_nm(Z, n_final, sigma=0.0):
    """Wavelength of the series limit (ionization from level n_final).

    As n_upper → ∞, the lines converge to λ_limit = hc / |E_n_final|.
    """
    E = abs(hydrogen_like_energy_eV(Z, n_final, sigma))
    return H_PLANCK * C / (E * EV_TO_J) * 1e9


# ══════════════════════════════════════════════════════════════════════
# MULTI-ELECTRON ATOMS — Slater-screened levels
# ══════════════════════════════════════════════════════════════════════

def multi_electron_energy_eV(Z, n, l, sigma=0.0):
    """Approximate energy of orbital (n, l) in multi-electron atom.

    Uses Slater's screening rules to estimate Z_eff.
    E_nl ≈ −E_R × (Z_eff / n*)²

    For exact single-electron ions, use hydrogen_like_energy_eV instead.

    APPROXIMATION: Slater screening (1930).
    Uses element.py for Z_eff computation.

    Args:
        Z: atomic number
        n: principal quantum number
        l: angular momentum quantum number
        sigma: σ-field value (enters through reduced mass only)

    Returns:
        Approximate orbital energy in eV (negative for bound).
    """
    from .element import slater_zeff, aufbau_configuration

    # Get effective nuclear charge for outermost electrons
    # This is a simplification — ideally we'd compute Z_eff per orbital
    z_eff = slater_zeff(Z)

    # Effective principal quantum number (Slater)
    n_star_map = {1: 1.0, 2: 2.0, 3: 3.0, 4: 3.7, 5: 4.0, 6: 4.2, 7: 4.4}
    n_star = n_star_map.get(n, float(n))

    mu_ratio = hydrogen_reduced_mass(sigma) / M_ELECTRON_KG
    return -mu_ratio * RYDBERG_ENERGY_EV * z_eff**2 / n_star**2


# ══════════════════════════════════════════════════════════════════════
# FINE STRUCTURE — Spin-Orbit Coupling
# ══════════════════════════════════════════════════════════════════════

def fine_structure_shift_eV(Z, n, l, j, sigma=0.0):
    """Fine structure energy shift for hydrogen-like atom.

    ΔE_fs = −Eₙ × (Zα)² / n × [n/(j+½) − 3/4]

    More precisely (Dirac theory, first-order in α²):
    ΔE_fs = Eₙ × α²Z² / n² × [1/(j+½) − 3/(4n)]

    This splits each l>0 level into j=l+½ and j=l−½ components.

    Args:
        Z: nuclear charge
        n: principal quantum number
        l: orbital angular momentum (0,1,2,...)
        j: total angular momentum (l-½ or l+½)
        sigma: σ-field value

    Returns:
        Fine structure shift in eV.

    FIRST_PRINCIPLES: relativistic correction + spin-orbit from Dirac equation.
    """
    if j < abs(l - 0.5) or j > l + 0.5:
        raise ValueError(f"j={j} not valid for l={l}")

    E_n = hydrogen_like_energy_eV(Z, n, sigma)
    shift = E_n * (ALPHA * Z)**2 / n * (1.0 / (j + 0.5) - 3.0 / (4.0 * n))
    return shift


def fine_structure_splitting_eV(Z, n, l, sigma=0.0):
    """Energy splitting between j=l+½ and j=l−½ levels (eV).

    For l=0: no splitting (only j=½ exists).
    For l≥1: ΔE_fs = |E(j=l+½) − E(j=l−½)|

    FIRST_PRINCIPLES: spin-orbit interaction V_SO = (1/2m²c²)(1/r)(dV/dr)L·S.
    """
    if l == 0:
        return 0.0

    j_plus = l + 0.5
    j_minus = l - 0.5
    shift_plus = fine_structure_shift_eV(Z, n, l, j_plus, sigma)
    shift_minus = fine_structure_shift_eV(Z, n, l, j_minus, sigma)
    return abs(shift_plus - shift_minus)


# ══════════════════════════════════════════════════════════════════════
# ZEEMAN EFFECT — Magnetic Field Splitting
# ══════════════════════════════════════════════════════════════════════

def lande_g_factor(l, s, j):
    """Landé g-factor for state |l, s, j⟩.

    g_J = 1 + [j(j+1) + s(s+1) − l(l+1)] / [2j(j+1)]

    Special cases:
      l=0 → g_J = 2 (pure spin)
      s=0 → g_J = 1 (pure orbital)

    FIRST_PRINCIPLES: vector model of angular momentum addition.
    """
    if j == 0:
        return 0.0
    return 1.0 + (j*(j+1) + s*(s+1) - l*(l+1)) / (2.0 * j * (j+1))


def zeeman_shift_eV(m_j, g_J, B_tesla):
    """Energy shift in external magnetic field (eV).

    ΔE = m_J × g_J × μ_B × B

    Args:
        m_j: magnetic quantum number (−j to +j)
        g_J: Landé g-factor
        B_tesla: magnetic field strength (T)

    Returns:
        Zeeman shift in eV.

    FIRST_PRINCIPLES: magnetic dipole in external field V = −μ·B.
    """
    return m_j * g_J * MU_BOHR * B_tesla / EV_TO_J


def zeeman_splitting_count(j):
    """Number of Zeeman sub-levels for state with quantum number j.

    Count = 2j + 1 (from m_j = −j to +j in integer steps).

    FIRST_PRINCIPLES: angular momentum quantization.
    """
    return int(2 * j + 1)


def zeeman_pattern(l, s, j, B_tesla):
    """Full Zeeman splitting pattern for a state.

    Returns list of (m_j, energy_shift_eV) tuples.

    Args:
        l: orbital quantum number
        s: spin quantum number
        j: total angular momentum quantum number
        B_tesla: magnetic field (T)
    """
    g_J = lande_g_factor(l, s, j)
    pattern = []
    # m_j goes from -j to +j in steps of 1
    m_j = -j
    while m_j <= j + 0.01:
        dE = zeeman_shift_eV(m_j, g_J, B_tesla)
        pattern.append((m_j, dE))
        m_j += 1.0
    return pattern


# ══════════════════════════════════════════════════════════════════════
# SELECTION RULES
# ══════════════════════════════════════════════════════════════════════

def is_allowed_transition(l_i, l_f, m_l_i=0, m_l_f=0, delta_s=0):
    """Check if electric dipole transition is allowed.

    Selection rules (FIRST_PRINCIPLES — dipole matrix element ⟨f|r|i⟩ ≠ 0):
      Δl = ±1  (parity change required)
      Δm_l = 0, ±1  (angular momentum conservation)
      Δm_s = 0  (photon doesn't flip spin)

    Args:
        l_i, l_f: initial and final orbital quantum numbers
        m_l_i, m_l_f: initial and final magnetic quantum numbers
        delta_s: change in spin quantum number

    Returns:
        True if transition is allowed by electric dipole selection rules.
    """
    delta_l = abs(l_f - l_i)
    delta_m = abs(m_l_f - m_l_i)
    return delta_l == 1 and delta_m <= 1 and delta_s == 0


def allowed_transitions(n_max=5):
    """List all allowed hydrogen transitions up to n_max.

    Returns list of (n_i, l_i, n_f, l_f, wavelength_nm).
    Only includes emission (n_i > n_f) and allowed Δl=±1.
    """
    transitions = []
    for n_i in range(2, n_max + 1):
        for l_i in range(0, n_i):
            for n_f in range(1, n_i):
                for l_f in range(0, n_f):
                    if is_allowed_transition(l_i, l_f):
                        lam = transition_wavelength_nm(1, n_i, n_f)
                        transitions.append((n_i, l_i, n_f, l_f, lam))
    return transitions


# ══════════════════════════════════════════════════════════════════════
# QUANTUM HARMONIC OSCILLATOR
# ══════════════════════════════════════════════════════════════════════

def qho_energy_eV(omega_rad_s, n):
    """Energy eigenvalue of quantum harmonic oscillator (eV).

    Eₙ = ℏω(n + ½)

    The ½ is zero-point energy — a purely quantum effect.
    Even at n=0 (ground state), the oscillator has E₀ = ℏω/2.

    Args:
        omega_rad_s: angular frequency ω (rad/s)
        n: quantum number (0, 1, 2, ...)

    Returns:
        Energy in eV.

    FIRST_PRINCIPLES: Schrödinger equation for V = ½mω²x².
    """
    if n < 0:
        raise ValueError(f"n must be ≥ 0, got {n}")
    return HBAR * omega_rad_s * (n + 0.5) / EV_TO_J


def qho_zero_point_energy_eV(omega_rad_s):
    """Zero-point energy of harmonic oscillator (eV).

    E₀ = ℏω/2

    This is the energy that cannot be removed even at T=0.
    It's why helium doesn't freeze at atmospheric pressure.
    """
    return qho_energy_eV(omega_rad_s, 0)


def qho_transition_energy_eV(omega_rad_s, n_i, n_f):
    """Energy of transition between oscillator levels (eV).

    ΔE = ℏω × |n_i − n_f|

    Selection rule for electric dipole: Δn = ±1.
    All transitions have the same energy ℏω (equally spaced levels).
    """
    return abs(qho_energy_eV(omega_rad_s, n_i) - qho_energy_eV(omega_rad_s, n_f))


def qho_classical_amplitude(omega_rad_s, mass_kg, n):
    """Classical turning point of quantum harmonic oscillator (m).

    x_max = √(2Eₙ / (mω²)) = √((2n+1)ℏ / (mω))

    At n=0: x₀ = √(ℏ/(mω)) — the zero-point amplitude.

    FIRST_PRINCIPLES: equate E_n to ½mω²x² and solve for x.
    """
    if n < 0:
        raise ValueError(f"n must be ≥ 0, got {n}")
    return math.sqrt((2 * n + 1) * HBAR / (mass_kg * omega_rad_s))


def qho_level_spacing_eV(omega_rad_s):
    """Energy spacing between adjacent levels (eV).

    ΔE = ℏω (constant for all n).

    This is the hallmark of the harmonic oscillator:
    equally spaced energy levels.
    """
    return HBAR * omega_rad_s / EV_TO_J


# ══════════════════════════════════════════════════════════════════════
# σ-FIELD EFFECTS ON SPECTRA
# ══════════════════════════════════════════════════════════════════════

def rydberg_constant_at_sigma(sigma=0.0):
    """Rydberg constant corrected for σ-dependent nuclear mass.

    R_atom = R∞ × μ/m_e

    At σ=0: R_H = R∞ × 0.99946 (hydrogen).
    At σ>0: proton heavier → μ→m_e → R_H→R∞.

    The shift is tiny (~5×10⁻⁴) but measurable in precision spectroscopy.
    """
    mu = hydrogen_reduced_mass(sigma)
    return RYDBERG_CONSTANT * mu / M_ELECTRON_KG


def sigma_spectral_shift(Z, n_upper, n_lower, sigma):
    """Fractional wavelength shift of spectral line due to σ-field.

    Δλ/λ = (R(0) − R(σ)) / R(0)

    Positive means red-shifted (lines move to longer wavelength).

    This is a MEASURABLE prediction of SSBM: hydrogen spectral lines
    near a neutron star (high σ) would shift by ~10⁻⁴.
    """
    lam_0 = transition_wavelength_nm(Z, n_upper, n_lower, 0.0)
    lam_s = transition_wavelength_nm(Z, n_upper, n_lower, sigma)
    return (lam_s - lam_0) / lam_0


# ══════════════════════════════════════════════════════════════════════
# EMISSION / ABSORPTION SPECTRUM
# ══════════════════════════════════════════════════════════════════════

def is_visible(wavelength_nm):
    """Check if wavelength falls in visible range (380-750 nm).

    MEASURED: human eye sensitivity range.
    """
    return 380.0 <= wavelength_nm <= 750.0


def visible_lines(Z=1, n_max=10, sigma=0.0):
    """All visible spectral lines for element with nuclear charge Z.

    Scans all transitions up to n_max and filters to 380-750 nm.

    Returns list of (n_upper, n_lower, wavelength_nm, energy_eV).
    """
    lines = []
    for n_lower in range(1, n_max):
        for n_upper in range(n_lower + 1, n_max + 1):
            lam = transition_wavelength_nm(Z, n_upper, n_lower, sigma)
            if is_visible(lam):
                E = transition_energy_eV(Z, n_upper, n_lower, sigma)
                lines.append((n_upper, n_lower, lam, E))
    return lines


def wavelength_to_rgb(wavelength_nm):
    """Approximate RGB color for a spectral wavelength.

    Uses the CIE-like approximation from Dan Bruton (1996).
    Returns (R, G, B) with values in [0, 1].

    APPROXIMATION: simplified spectral → sRGB mapping.
    """
    lam = wavelength_nm
    if lam < 380 or lam > 750:
        return (0.0, 0.0, 0.0)

    if lam < 440:
        r = -(lam - 440) / (440 - 380)
        g = 0.0
        b = 1.0
    elif lam < 490:
        r = 0.0
        g = (lam - 440) / (490 - 440)
        b = 1.0
    elif lam < 510:
        r = 0.0
        g = 1.0
        b = -(lam - 510) / (510 - 490)
    elif lam < 580:
        r = (lam - 510) / (580 - 510)
        g = 1.0
        b = 0.0
    elif lam < 645:
        r = 1.0
        g = -(lam - 645) / (645 - 580)
        b = 0.0
    else:
        r = 1.0
        g = 0.0
        b = 0.0

    # Intensity rolloff at edges
    if lam < 420:
        factor = 0.3 + 0.7 * (lam - 380) / (420 - 380)
    elif lam > 700:
        factor = 0.3 + 0.7 * (750 - lam) / (750 - 700)
    else:
        factor = 1.0

    return (r * factor, g * factor, b * factor)


def emission_spectrum(Z=1, n_max=10, sigma=0.0):
    """Complete emission spectrum for hydrogen-like ion.

    Returns list of dicts with:
        n_upper, n_lower, wavelength_nm, energy_eV, frequency_Hz,
        wavenumber_cm, series_name, visible, rgb

    Covers all named series (Lyman through Pfund) and beyond.
    """
    series_names = {1: 'Lyman', 2: 'Balmer', 3: 'Paschen',
                    4: 'Brackett', 5: 'Pfund'}
    lines = []
    for n_lower in range(1, n_max):
        for n_upper in range(n_lower + 1, n_max + 1):
            lam = transition_wavelength_nm(Z, n_upper, n_lower, sigma)
            E = transition_energy_eV(Z, n_upper, n_lower, sigma)
            freq = transition_frequency_Hz(Z, n_upper, n_lower, sigma)
            wn = transition_wavenumber(Z, n_upper, n_lower, sigma)
            vis = is_visible(lam)
            rgb = wavelength_to_rgb(lam) if vis else (0, 0, 0)
            series = series_names.get(n_lower, f'n={n_lower}')
            lines.append({
                'n_upper': n_upper,
                'n_lower': n_lower,
                'wavelength_nm': lam,
                'energy_eV': E,
                'frequency_Hz': freq,
                'wavenumber_cm': wn,
                'series': series,
                'visible': vis,
                'rgb': rgb,
            })
    return lines


# ══════════════════════════════════════════════════════════════════════
# REPORTS (Rule 9: full coverage)
# ══════════════════════════════════════════════════════════════════════

def atomic_spectra_report(Z=1, sigma=0.0):
    """Spectral properties report for hydrogen-like ion.

    Returns dict with energy levels, spectral series, fine structure,
    and σ-field effects.
    """
    # Energy levels
    levels = {}
    for n in range(1, 8):
        levels[n] = hydrogen_like_energy_eV(Z, n, sigma)

    # Balmer series (most commonly observed)
    balmer = balmer_series(Z, n_max=7, sigma=sigma)

    # Visible lines
    vis = visible_lines(Z, n_max=10, sigma=sigma)

    # Fine structure of n=2
    fs_2p = fine_structure_splitting_eV(Z, 2, 1, sigma)

    # Rydberg constant
    R = rydberg_constant_at_sigma(sigma)

    return {
        'Z': Z,
        'sigma': sigma,
        'rydberg_energy_eV': RYDBERG_ENERGY_EV,
        'rydberg_constant_m_inv': R,
        'ionization_energy_eV': -hydrogen_like_energy_eV(Z, 1, sigma),
        'energy_levels_eV': levels,
        'balmer_series': [(n, lam, E) for n, lam, E in balmer],
        'visible_line_count': len(vis),
        'fine_structure_2p_eV': fs_2p,
    }


def full_report(Z=1, sigma=0.0):
    """Complete atomic spectra report (Rule 9).

    Calls atomic_spectra_report and adds QHO and Zeeman info.
    """
    report = atomic_spectra_report(Z, sigma)

    # Add harmonic oscillator example (typical molecular vibration ~10¹⁴ rad/s)
    omega_example = 1e14  # rad/s
    report['qho_example_omega_rad_s'] = omega_example
    report['qho_zero_point_eV'] = qho_zero_point_energy_eV(omega_example)
    report['qho_level_spacing_eV'] = qho_level_spacing_eV(omega_example)

    # Zeeman example (1 T field, 2p state)
    report['zeeman_2p_pattern_1T'] = zeeman_pattern(1, 0.5, 1.5, 1.0)

    return report
