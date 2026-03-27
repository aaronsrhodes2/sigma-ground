"""
Electronics — electrical transport, semiconductor physics, and junctions.

Builds on thermoelectric.py (room-temp ρ), thermal.py (Θ_D), optics.py (Drude),
and semiconductor_optics.py (band gap) to provide temperature-dependent transport,
carrier dynamics, and device physics.

Derivation chains:

  1. Bloch-Grüneisen Resistivity (FIRST_PRINCIPLES)
     ρ(T) = ρ_0 + ρ_BG × (T/Θ_D)^5 × J₅(Θ_D/T)

     Where:
       ρ_0 = residual resistivity (impurity scattering, MEASURED)
       Θ_D = Debye temperature (from thermal.py or superconductivity.py)
       J₅(x) = ∫₀ˣ t⁵/((eᵗ−1)(1−e⁻ᵗ)) dt  (Bloch-Grüneisen integral)

     The coefficient ρ_BG is calibrated from measured ρ(300K):
       ρ_BG = (ρ_300 − ρ_0) / ((300/Θ_D)⁵ × J₅(Θ_D/300))

     Grüneisen (1933), Bloch (1930). Exact for phonon-limited scattering.

  2. Carrier Mobility (FIRST_PRINCIPLES: Drude)
     μ = 1 / (n_e × e × ρ)

     Where n_e = free electron density, e = elementary charge.
     Drude (1900): carriers drift under electric field, scattered by phonons.

  3. Hall Coefficient (FIRST_PRINCIPLES)
     R_H = −1 / (n_e × e)    (free-electron, single band)

     Hall (1879): Lorentz force deflects carriers in magnetic field.
     Sign gives carrier type (negative for electrons).

  4. Intrinsic Carrier Concentration (FIRST_PRINCIPLES)
     n_i = √(N_c × N_v) × exp(−E_g / (2k_BT))

     Where:
       N_c = 2(2π m_e* k_BT / h²)^{3/2}  (conduction band DOS)
       N_v = 2(2π m_h* k_BT / h²)^{3/2}  (valence band DOS)
       m_e*, m_h* = effective masses (MEASURED)

  5. p-n Junction (FIRST_PRINCIPLES: Shockley)
     V_bi = (k_BT/e) × ln(N_A × N_D / n_i²)     (built-in voltage)
     W = √(2ε(V_bi−V)/e × (1/N_A + 1/N_D))      (depletion width)
     I = I_0 × (exp(eV/k_BT) − 1)                (Shockley diode)

  6. Geometric Capacitance (FIRST_PRINCIPLES: Gauss's law)
     C = ε₀ε_r A/d     (parallel plate)
     C = 2πε₀ε_r L / ln(b/a)   (coaxial)

σ-dependence:
  Resistivity ρ itself is electromagnetic (σ-INVARIANT at fixed T).
  BUT ρ(T) depends on Θ_D which shifts with σ through nuclear mass:
    Θ_D(σ) → different phonon scattering → different ρ(T) curve.
  Band gap is electromagnetic → σ-INVARIANT.
  Junction physics: V_bi, W shift only through Θ_D effects on transport.

Origin tags:
  - Bloch-Grüneisen: FIRST_PRINCIPLES (electron-phonon scattering theory)
  - Drude mobility: FIRST_PRINCIPLES (kinetic theory of electrons)
  - Hall coefficient: FIRST_PRINCIPLES (Lorentz force)
  - n_i: FIRST_PRINCIPLES (Fermi-Dirac statistics + band theory)
  - Shockley diode: FIRST_PRINCIPLES (drift-diffusion)
  - Capacitance: FIRST_PRINCIPLES (Gauss's law, exact)
  - σ-dependence: CORE (through Θ_D → phonon spectrum)
"""

import math
from ..constants import HBAR, K_B, E_CHARGE, M_ELECTRON_KG, EPS_0


# ── Physical Constants (local) ────────────────────────────────────
_H_PLANCK = 2.0 * math.pi * HBAR
_N_A = 6.02214076e23


# ── Bloch-Grüneisen Integral ─────────────────────────────────────

def _bg_integrand(t):
    """Integrand for Bloch-Grüneisen: t⁵ / ((eᵗ−1)(1−e⁻ᵗ)) = t⁵ eᵗ / (eᵗ−1)²."""
    if t <= 0:
        return 0.0
    if t > 300:
        return 0.0  # Exponentially suppressed — integrand ~ t⁵ × e⁻ᵗ
    et = math.exp(t)
    denom = (et - 1.0) ** 2
    if denom == 0:
        return 0.0
    return t ** 5 * et / denom


def _bg_integral(x_max, n_steps=200):
    """Evaluate J₅(x) = ∫₀ˣ t⁵eᵗ/(eᵗ−1)² dt using Simpson's rule.

    No external dependencies — pure numerical integration.
    """
    if x_max <= 0:
        return 0.0

    h = x_max / n_steps
    result = _bg_integrand(0.0) + _bg_integrand(x_max)
    for i in range(1, n_steps):
        t = i * h
        weight = 4.0 if i % 2 == 1 else 2.0
        result += weight * _bg_integrand(t)
    return result * h / 3.0


# ── Metal Transport Data ─────────────────────────────────────────
# Rule 9 — every metal with resistivity gets every field.
#
# ρ_300: MEASURED resistivity at 300 K (Ω·m), CRC Handbook
# ρ_0: MEASURED residual resistivity at T→0 (Ω·m), high-purity single crystal
# theta_D: MEASURED Debye temperature (K), from thermal.py / CRC
# n_e: DERIVED or MEASURED free electron density (m⁻³)
# Z_val: valence electrons per atom
# rho_kg_m3: density (kg/m³)
# M_g: molar mass (g/mol)

METAL_TRANSPORT = {
    'copper': {
        'rho_300': 1.68e-8,
        'rho_0': 1.0e-10,      # Very pure Cu
        'theta_D': 343,
        'Z_val': 1,
        'rho_kg_m3': 8960,
        'M_g': 63.546,
    },
    'aluminum': {
        'rho_300': 2.65e-8,
        'rho_0': 1.0e-10,
        'theta_D': 428,
        'Z_val': 3,
        'rho_kg_m3': 2700,
        'M_g': 26.982,
    },
    'gold': {
        'rho_300': 2.24e-8,
        'rho_0': 5.0e-11,
        'theta_D': 165,
        'Z_val': 1,
        'rho_kg_m3': 19300,
        'M_g': 196.967,
    },
    'iron': {
        'rho_300': 9.70e-8,
        'rho_0': 5.0e-9,       # α-Fe, higher residual (magnetic scattering)
        'theta_D': 470,
        'Z_val': 2,
        'rho_kg_m3': 7874,
        'M_g': 55.845,
    },
    'nickel': {
        'rho_300': 6.99e-8,
        'rho_0': 1.0e-9,
        'theta_D': 450,
        'Z_val': 2,
        'rho_kg_m3': 8908,
        'M_g': 58.693,
    },
    'tungsten': {
        'rho_300': 5.28e-8,
        'rho_0': 5.0e-10,
        'theta_D': 400,
        'Z_val': 2,
        'rho_kg_m3': 19250,
        'M_g': 183.84,
    },
    'titanium': {
        'rho_300': 4.20e-7,
        'rho_0': 5.0e-9,
        'theta_D': 420,
        'Z_val': 2,
        'rho_kg_m3': 4507,
        'M_g': 47.867,
    },
    'silicon': {
        'rho_300': 6.4e2,       # Intrinsic semiconductor — very high
        'rho_0': 6.4e2,         # Not metallic — BG not applicable
        'theta_D': 645,
        'Z_val': 4,
        'rho_kg_m3': 2330,
        'M_g': 28.086,
    },
}

# Pre-compute BG calibration coefficients for metals
_BG_COEFF = {}


def _calibrate_bg():
    """Calibrate Bloch-Grüneisen coefficient from ρ(300K) for each metal."""
    for key, data in METAL_TRANSPORT.items():
        if data['rho_300'] > 1.0:
            # Semiconductor — BG not applicable
            _BG_COEFF[key] = 0.0
            continue
        theta = data['theta_D']
        rho_300 = data['rho_300']
        rho_0 = data['rho_0']
        x = theta / 300.0
        j5 = _bg_integral(x)
        bg_term = (300.0 / theta) ** 5 * j5
        if bg_term > 0:
            _BG_COEFF[key] = (rho_300 - rho_0) / bg_term
        else:
            _BG_COEFF[key] = 0.0


_calibrate_bg()


# ── Free Electron Density ────────────────────────────────────────

def free_electron_density(material_key):
    """Free electron density n_e (m⁻³).

    n_e = Z_val × N_A × ρ / M

    FIRST_PRINCIPLES: each atom contributes Z_val conduction electrons.

    Args:
        material_key: key into METAL_TRANSPORT

    Returns:
        Electron density in m⁻³
    """
    data = METAL_TRANSPORT[material_key]
    return (data['Z_val'] * _N_A * data['rho_kg_m3']
            / (data['M_g'] * 1e-3))


# ── Bloch-Grüneisen Resistivity ──────────────────────────────────

def resistivity(material_key, T):
    """Temperature-dependent resistivity ρ(T) (Ω·m).

    ρ(T) = ρ_0 + ρ_BG × (T/Θ_D)⁵ × J₅(Θ_D/T)

    FIRST_PRINCIPLES: Bloch-Grüneisen electron-phonon scattering.
    Calibrated to reproduce measured ρ(300K) exactly.

    High-T limit (T >> Θ_D): ρ ∝ T (linear, phonon population ∝ T)
    Low-T limit (T << Θ_D): ρ ∝ T⁵ (phonon freeze-out)

    Args:
        material_key: key into METAL_TRANSPORT
        T: temperature (K)

    Returns:
        Resistivity in Ω·m
    """
    data = METAL_TRANSPORT[material_key]

    if T <= 0:
        return data['rho_0']

    # Semiconductors: BG not applicable
    if data['rho_300'] > 1.0:
        return data['rho_300']

    theta = data['theta_D']
    x = theta / T
    j5 = _bg_integral(x)
    bg_term = (T / theta) ** 5 * j5

    return data['rho_0'] + _BG_COEFF[material_key] * bg_term


def resistivity_sigma(material_key, T, sigma):
    """Resistivity under σ-field.

    σ shifts Θ_D through nuclear mass → different phonon spectrum
    → different ρ(T) curve. The residual ρ_0 is σ-invariant (impurities).

    CORE: through Θ_D(σ) = Θ_D(0) / √(mass_ratio(σ)).

    Args:
        material_key: key into METAL_TRANSPORT
        T: temperature (K)
        sigma: σ-field value

    Returns:
        Resistivity in Ω·m
    """
    if sigma == 0.0:
        return resistivity(material_key, T)

    from ..scale import scale_ratio
    from ..constants import PROTON_QCD_FRACTION

    data = METAL_TRANSPORT[material_key]

    if T <= 0:
        return data['rho_0']
    if data['rho_300'] > 1.0:
        return data['rho_300']

    f_qcd = PROTON_QCD_FRACTION
    mass_ratio = (1.0 - f_qcd) + f_qcd * scale_ratio(sigma)
    theta_sigma = data['theta_D'] / math.sqrt(mass_ratio)

    x = theta_sigma / T
    j5 = _bg_integral(x)
    bg_term = (T / theta_sigma) ** 5 * j5

    # Re-calibrate coefficient for shifted Θ_D
    x_300 = theta_sigma / 300.0
    j5_300 = _bg_integral(x_300)
    bg_300 = (300.0 / theta_sigma) ** 5 * j5_300

    rho_300 = data['rho_300']
    rho_0 = data['rho_0']
    if bg_300 > 0:
        coeff = (rho_300 - rho_0) / bg_300
    else:
        coeff = 0.0

    return rho_0 + coeff * bg_term


# ── Carrier Mobility ─────────────────────────────────────────────

def carrier_mobility(material_key, T=300.0):
    """Carrier mobility μ (m²/(V·s)).

    μ = 1 / (n_e × e × ρ(T))

    FIRST_PRINCIPLES: Drude model — mobility is the drift velocity
    per unit electric field. Equals e×τ/m_e where τ is the mean
    scattering time.

    Args:
        material_key: key into METAL_TRANSPORT
        T: temperature (K)

    Returns:
        Mobility in m²/(V·s)
    """
    n_e = free_electron_density(material_key)
    rho = resistivity(material_key, T)
    if rho <= 0 or n_e <= 0:
        return float('inf')
    return 1.0 / (n_e * E_CHARGE * rho)


def mean_free_time(material_key, T=300.0):
    """Drude mean scattering time τ (s).

    τ = m_e × μ / e = m_e / (n_e × e² × ρ)

    FIRST_PRINCIPLES: time between collisions.

    Args:
        material_key: key into METAL_TRANSPORT
        T: temperature (K)

    Returns:
        τ in seconds
    """
    mu = carrier_mobility(material_key, T)
    return M_ELECTRON_KG * mu / E_CHARGE


def mean_free_path(material_key, T=300.0):
    """Electron mean free path ℓ (m).

    ℓ = v_F × τ

    FIRST_PRINCIPLES: Fermi velocity × scattering time.

    Args:
        material_key: key into METAL_TRANSPORT
        T: temperature (K)

    Returns:
        Mean free path in metres
    """
    n_e = free_electron_density(material_key)
    v_F = HBAR * (3.0 * math.pi ** 2 * n_e) ** (1.0 / 3.0) / M_ELECTRON_KG
    tau = mean_free_time(material_key, T)
    return v_F * tau


# ── Hall Effect ──────────────────────────────────────────────────

def hall_coefficient(material_key):
    """Hall coefficient R_H (m³/C).

    R_H = −1 / (n_e × e)

    FIRST_PRINCIPLES: Lorentz force deflection of free electrons.
    Negative sign indicates electron carriers.
    Single-band free-electron model — metals only.

    Args:
        material_key: key into METAL_TRANSPORT

    Returns:
        R_H in m³/C (negative for electrons)
    """
    n_e = free_electron_density(material_key)
    return -1.0 / (n_e * E_CHARGE)


def hall_voltage(material_key, current, B_field, thickness):
    """Hall voltage V_H (V).

    V_H = R_H × I × B / t

    FIRST_PRINCIPLES: Lorentz force balance in steady state.

    Args:
        material_key: key into METAL_TRANSPORT
        current: current through sample (A)
        B_field: magnetic field (T)
        thickness: sample thickness in field direction (m)

    Returns:
        Hall voltage in Volts (negative for electron carriers)
    """
    R_H = hall_coefficient(material_key)
    return R_H * current * B_field / thickness


# ── Semiconductor Data ───────────────────────────────────────────
# Rule 9: every semiconductor gets every field.
#
# E_g_eV: band gap at 0 K (eV), MEASURED
# m_e_eff: electron effective mass (units of m_e), MEASURED
# m_h_eff: hole effective mass (units of m_e), MEASURED
# epsilon_r: relative permittivity, MEASURED
# E_donor_eV: typical donor ionization energy (eV), MEASURED
# E_acceptor_eV: typical acceptor ionization energy (eV), MEASURED
#
# Sources: Sze "Physics of Semiconductor Devices" (2007),
#          Pierret "Semiconductor Device Fundamentals" (1996)

SEMICONDUCTORS = {
    'silicon': {
        'E_g_eV': 1.17,             # Indirect gap at 0 K
        'm_e_eff': 1.08,            # DOS effective mass (6 valleys)
        'm_h_eff': 0.56,            # DOS effective mass (light+heavy)
        'epsilon_r': 11.7,
        'E_donor_eV': 0.045,        # Phosphorus in Si
        'E_acceptor_eV': 0.045,     # Boron in Si
        'varshni_alpha': 4.73e-4,   # eV/K
        'varshni_beta': 636,        # K
    },
    'germanium': {
        'E_g_eV': 0.74,             # Indirect gap at 0 K
        'm_e_eff': 0.55,
        'm_h_eff': 0.37,
        'epsilon_r': 16.0,
        'E_donor_eV': 0.012,        # Arsenic in Ge
        'E_acceptor_eV': 0.011,     # Gallium in Ge
        'varshni_alpha': 4.77e-4,
        'varshni_beta': 235,
    },
    'gallium_arsenide': {
        'E_g_eV': 1.52,             # Direct gap at 0 K
        'm_e_eff': 0.067,           # Very light electrons — high mobility
        'm_h_eff': 0.45,
        'epsilon_r': 12.9,
        'E_donor_eV': 0.006,        # Si donor in GaAs
        'E_acceptor_eV': 0.028,     # Zn acceptor in GaAs
        'varshni_alpha': 5.41e-4,
        'varshni_beta': 204,
    },
    'silicon_carbide': {
        'E_g_eV': 3.26,             # 4H-SiC, indirect
        'm_e_eff': 0.42,
        'm_h_eff': 1.0,
        'epsilon_r': 9.7,
        'E_donor_eV': 0.065,        # Nitrogen in 4H-SiC
        'E_acceptor_eV': 0.20,      # Aluminum in SiC
        'varshni_alpha': 3.3e-4,
        'varshni_beta': 700,
    },
    'gallium_nitride': {
        'E_g_eV': 3.50,             # Direct gap, wurtzite
        'm_e_eff': 0.20,
        'm_h_eff': 1.4,
        'epsilon_r': 8.9,
        'E_donor_eV': 0.025,        # Si donor in GaN
        'E_acceptor_eV': 0.17,      # Mg acceptor in GaN
        'varshni_alpha': 7.7e-4,
        'varshni_beta': 600,
    },
    'indium_phosphide': {
        'E_g_eV': 1.42,             # Direct gap
        'm_e_eff': 0.077,
        'm_h_eff': 0.60,
        'epsilon_r': 12.4,
        'E_donor_eV': 0.007,        # Sn donor in InP
        'E_acceptor_eV': 0.028,     # Zn acceptor in InP
        'varshni_alpha': 4.5e-4,
        'varshni_beta': 327,
    },
}


# ── Semiconductor Band Gap ───────────────────────────────────────

def band_gap(sc_key, T=300.0):
    """Temperature-dependent band gap E_g(T) (eV).

    E_g(T) = E_g(0) − α T² / (T + β)    (Varshni 1967)

    FIRST_PRINCIPLES form, MEASURED coefficients.
    The gap shrinks with temperature due to lattice expansion
    and electron-phonon interaction.

    Args:
        sc_key: key into SEMICONDUCTORS
        T: temperature (K)

    Returns:
        Band gap in eV
    """
    data = SEMICONDUCTORS[sc_key]
    E0 = data['E_g_eV']
    alpha = data['varshni_alpha']
    beta = data['varshni_beta']
    return E0 - alpha * T ** 2 / (T + beta)


# ── Intrinsic Carrier Concentration ──────────────────────────────

def effective_dos_conduction(sc_key, T=300.0):
    """Effective density of states in conduction band N_c (m⁻³).

    N_c = 2 × (2π m_e* k_B T / h²)^{3/2}

    FIRST_PRINCIPLES: parabolic band approximation.

    Args:
        sc_key: key into SEMICONDUCTORS
        T: temperature (K)

    Returns:
        N_c in m⁻³
    """
    m_eff = SEMICONDUCTORS[sc_key]['m_e_eff'] * M_ELECTRON_KG
    return 2.0 * (2.0 * math.pi * m_eff * K_B * T / _H_PLANCK ** 2) ** 1.5


def effective_dos_valence(sc_key, T=300.0):
    """Effective density of states in valence band N_v (m⁻³).

    N_v = 2 × (2π m_h* k_B T / h²)^{3/2}

    Args:
        sc_key: key into SEMICONDUCTORS
        T: temperature (K)

    Returns:
        N_v in m⁻³
    """
    m_eff = SEMICONDUCTORS[sc_key]['m_h_eff'] * M_ELECTRON_KG
    return 2.0 * (2.0 * math.pi * m_eff * K_B * T / _H_PLANCK ** 2) ** 1.5


def intrinsic_carrier_concentration(sc_key, T=300.0):
    """Intrinsic carrier concentration n_i (m⁻³).

    n_i = √(N_c × N_v) × exp(−E_g / (2 k_B T))

    FIRST_PRINCIPLES: Fermi-Dirac statistics in thermal equilibrium.
    At T=300 K, Si n_i ≈ 1.5×10¹⁶ m⁻³ (1.5×10¹⁰ cm⁻³).

    Args:
        sc_key: key into SEMICONDUCTORS
        T: temperature (K)

    Returns:
        n_i in m⁻³
    """
    if T <= 0:
        return 0.0

    N_c = effective_dos_conduction(sc_key, T)
    N_v = effective_dos_valence(sc_key, T)
    E_g = band_gap(sc_key, T)
    E_g_J = E_g * E_CHARGE  # Convert eV to Joules

    exponent = -E_g_J / (2.0 * K_B * T)
    if exponent < -700:
        return 0.0

    return math.sqrt(N_c * N_v) * math.exp(exponent)


# ── Doped Carrier Concentration ──────────────────────────────────

def carrier_concentration(sc_key, T=300.0, N_D=0.0, N_A=0.0):
    """Carrier concentrations in doped semiconductor (m⁻³).

    n-type (N_D > N_A): n ≈ N_D − N_A, p = n_i²/n
    p-type (N_A > N_D): p ≈ N_A − N_D, n = n_i²/p
    Intrinsic: n = p = n_i

    FIRST_PRINCIPLES: charge neutrality + mass action law n×p = n_i².
    Full ionization approximation (valid for T > ~100 K in Si).

    Args:
        sc_key: key into SEMICONDUCTORS
        T: temperature (K)
        N_D: donor concentration (m⁻³)
        N_A: acceptor concentration (m⁻³)

    Returns:
        (n, p) tuple — electron and hole concentrations in m⁻³
    """
    n_i = intrinsic_carrier_concentration(sc_key, T)
    if n_i <= 0:
        return (0.0, 0.0)

    net = N_D - N_A  # Positive for n-type, negative for p-type

    if abs(net) < n_i * 0.01:
        # Near-intrinsic
        return (n_i, n_i)

    # Quadratic: n² − net×n − n_i² = 0
    # n = (net + √(net² + 4n_i²)) / 2
    discriminant = net ** 2 + 4.0 * n_i ** 2
    n = (net + math.sqrt(discriminant)) / 2.0
    p = n_i ** 2 / n if n > 0 else 0.0
    return (n, p)


def fermi_level_from_intrinsic(sc_key, T=300.0, N_D=0.0, N_A=0.0):
    """Fermi level position relative to intrinsic level E_Fi (eV).

    E_F − E_Fi = k_BT × ln(n/n_i)   for n-type
    E_F − E_Fi = −k_BT × ln(p/n_i)  for p-type

    FIRST_PRINCIPLES: Boltzmann approximation of Fermi-Dirac.

    Args:
        sc_key: key into SEMICONDUCTORS
        T: temperature (K)
        N_D: donor concentration (m⁻³)
        N_A: acceptor concentration (m⁻³)

    Returns:
        E_F − E_Fi in eV (positive for n-type, negative for p-type)
    """
    n_i = intrinsic_carrier_concentration(sc_key, T)
    if n_i <= 0 or T <= 0:
        return 0.0

    n, p = carrier_concentration(sc_key, T, N_D, N_A)
    if n <= 0:
        return 0.0

    kT_eV = K_B * T / E_CHARGE
    return kT_eV * math.log(n / n_i)


# ── p-n Junction ─────────────────────────────────────────────────

def built_in_voltage(sc_key, N_D, N_A, T=300.0):
    """Built-in voltage V_bi of a p-n junction (V).

    V_bi = (k_BT / e) × ln(N_A × N_D / n_i²)

    FIRST_PRINCIPLES: thermal equilibrium requires E_F constant
    across the junction → contact potential.

    Args:
        sc_key: key into SEMICONDUCTORS
        N_D: donor concentration on n-side (m⁻³)
        N_A: acceptor concentration on p-side (m⁻³)
        T: temperature (K)

    Returns:
        V_bi in Volts
    """
    n_i = intrinsic_carrier_concentration(sc_key, T)
    if n_i <= 0 or N_D <= 0 or N_A <= 0:
        return 0.0

    kT = K_B * T / E_CHARGE  # in Volts
    return kT * math.log(N_A * N_D / n_i ** 2)


def depletion_width(sc_key, N_D, N_A, V_applied=0.0, T=300.0):
    """Depletion region width W (m).

    W = √(2 ε₀ ε_r (V_bi − V) / e × (1/N_A + 1/N_D))

    FIRST_PRINCIPLES: Poisson's equation in the abrupt junction
    approximation.

    Args:
        sc_key: key into SEMICONDUCTORS
        N_D, N_A: doping concentrations (m⁻³)
        V_applied: applied voltage (V), positive = forward bias
        T: temperature (K)

    Returns:
        Depletion width in metres
    """
    V_bi = built_in_voltage(sc_key, N_D, N_A, T)
    V_eff = V_bi - V_applied
    if V_eff <= 0:
        return 0.0  # Forward bias exceeds V_bi — no depletion

    eps = EPS_0 * SEMICONDUCTORS[sc_key]['epsilon_r']
    return math.sqrt(2.0 * eps * V_eff / E_CHARGE * (1.0 / N_A + 1.0 / N_D))


def junction_capacitance(sc_key, N_D, N_A, area, V_applied=0.0, T=300.0):
    """Junction capacitance C_j (F).

    C_j = ε₀ ε_r A / W

    FIRST_PRINCIPLES: depletion region acts as a parallel-plate capacitor.
    Capacitance varies with voltage (varactor effect).

    Args:
        sc_key: key into SEMICONDUCTORS
        N_D, N_A: doping concentrations (m⁻³)
        area: junction area (m²)
        V_applied: applied voltage (V)
        T: temperature (K)

    Returns:
        Capacitance in Farads
    """
    W = depletion_width(sc_key, N_D, N_A, V_applied, T)
    if W <= 0:
        return float('inf')  # Forward bias → diffusion capacitance dominates
    eps = EPS_0 * SEMICONDUCTORS[sc_key]['epsilon_r']
    return eps * area / W


def diode_saturation_current(sc_key, N_D, N_A, area,
                              D_n=0.0035, D_p=0.0012,
                              L_n=100e-6, L_p=50e-6, T=300.0):
    """Reverse saturation current I₀ (A).

    I₀ = e A n_i² (D_n/(L_n N_A) + D_p/(L_p N_D))

    FIRST_PRINCIPLES: minority carrier diffusion at junction edges.

    Default diffusion constants for silicon at 300K:
      D_n = 35 cm²/s = 0.0035 m²/s (electrons)
      D_p = 12 cm²/s = 0.0012 m²/s (holes)
      L_n = 100 μm, L_p = 50 μm (diffusion lengths)

    Args:
        sc_key: key into SEMICONDUCTORS
        N_D, N_A: doping concentrations (m⁻³)
        area: junction area (m²)
        D_n: electron diffusion coefficient (m²/s)
        D_p: hole diffusion coefficient (m²/s)
        L_n: electron diffusion length (m)
        L_p: hole diffusion length (m)
        T: temperature (K)

    Returns:
        I₀ in Amperes
    """
    n_i = intrinsic_carrier_concentration(sc_key, T)
    return (E_CHARGE * area * n_i ** 2
            * (D_n / (L_n * N_A) + D_p / (L_p * N_D)))


def diode_current(I_0, V, T=300.0):
    """Shockley diode equation I(V) (A).

    I = I₀ × (exp(eV / k_BT) − 1)

    FIRST_PRINCIPLES: Shockley (1949). Drift-diffusion of minority
    carriers across the junction. Valid below breakdown and for
    V >> k_BT/e in forward bias.

    Args:
        I_0: saturation current (A)
        V: applied voltage (V)
        T: temperature (K)

    Returns:
        Current in Amperes
    """
    if T <= 0:
        return 0.0
    V_T = K_B * T / E_CHARGE  # Thermal voltage
    exponent = V / V_T
    if exponent > 500:
        return I_0 * math.exp(500)  # Guard overflow
    if exponent < -500:
        return -I_0  # Reverse saturation
    return I_0 * (math.exp(exponent) - 1.0)


# ── Geometric Capacitance ────────────────────────────────────────

def parallel_plate_capacitance(area, separation, epsilon_r=1.0):
    """Parallel-plate capacitor C (F).

    C = ε₀ ε_r A / d

    FIRST_PRINCIPLES: Gauss's law, exact for infinite plates.
    Fringe fields add ~5% for finite plates (not included).

    Args:
        area: plate area (m²)
        separation: plate spacing (m)
        epsilon_r: relative permittivity of dielectric

    Returns:
        Capacitance in Farads
    """
    return EPS_0 * epsilon_r * area / separation


def coaxial_capacitance(length, r_inner, r_outer, epsilon_r=1.0):
    """Coaxial capacitor C (F).

    C = 2π ε₀ ε_r L / ln(b/a)

    FIRST_PRINCIPLES: Gauss's law in cylindrical geometry.

    Args:
        length: cable length (m)
        r_inner: inner conductor radius (m)
        r_outer: outer conductor radius (m)
        epsilon_r: relative permittivity

    Returns:
        Capacitance in Farads
    """
    return 2.0 * math.pi * EPS_0 * epsilon_r * length / math.log(r_outer / r_inner)


def spherical_capacitance(r_inner, r_outer, epsilon_r=1.0):
    """Spherical capacitor C (F).

    C = 4π ε₀ ε_r r_a r_b / (r_b − r_a)

    FIRST_PRINCIPLES: Gauss's law in spherical geometry.

    Args:
        r_inner: inner sphere radius (m)
        r_outer: outer sphere radius (m)
        epsilon_r: relative permittivity

    Returns:
        Capacitance in Farads
    """
    return (4.0 * math.pi * EPS_0 * epsilon_r
            * r_inner * r_outer / (r_outer - r_inner))


def energy_stored(capacitance, voltage):
    """Energy stored in a capacitor (J).

    U = ½ C V²

    FIRST_PRINCIPLES: work integral to charge a capacitor.

    Args:
        capacitance: capacitance (F)
        voltage: voltage across capacitor (V)

    Returns:
        Energy in Joules
    """
    return 0.5 * capacitance * voltage ** 2


# ── Nagatha Integration ──────────────────────────────────────────

def metal_transport_properties(material_key, T=300.0, sigma=0.0):
    """Export metal transport properties in Nagatha format.

    Args:
        material_key: key into METAL_TRANSPORT
        T: temperature (K)
        sigma: σ-field value

    Returns:
        Dict of transport properties
    """
    rho = resistivity_sigma(material_key, T, sigma) if sigma != 0.0 \
        else resistivity(material_key, T)
    n_e = free_electron_density(material_key)
    mu = 1.0 / (n_e * E_CHARGE * rho) if rho > 0 and n_e > 0 else 0.0

    return {
        'material': material_key,
        'T_K': T,
        'resistivity_ohm_m': rho,
        'conductivity_S_m': 1.0 / rho if rho > 0 else float('inf'),
        'mobility_m2_V_s': mu,
        'n_e_m3': n_e,
        'hall_coefficient_m3_C': hall_coefficient(material_key),
        'mean_free_path_m': mean_free_path(material_key, T),
        'sigma': sigma,
        'origin_tag': (
            "FIRST_PRINCIPLES: Bloch-Grüneisen ρ(T) (electron-phonon). "
            "FIRST_PRINCIPLES: Drude mobility μ = 1/(n_e e ρ). "
            "FIRST_PRINCIPLES: Hall coefficient R_H = −1/(n_e e). "
            "MEASURED: ρ(300K), ρ₀, Θ_D. "
            "CORE: σ-dependence through Θ_D shift."
        ),
    }


def semiconductor_properties(sc_key, T=300.0, N_D=0.0, N_A=0.0):
    """Export semiconductor properties in Nagatha format.

    Args:
        sc_key: key into SEMICONDUCTORS
        T: temperature (K)
        N_D: donor concentration (m⁻³)
        N_A: acceptor concentration (m⁻³)

    Returns:
        Dict of semiconductor properties
    """
    n_i = intrinsic_carrier_concentration(sc_key, T)
    n, p = carrier_concentration(sc_key, T, N_D, N_A)
    E_g = band_gap(sc_key, T)

    result = {
        'material': sc_key,
        'T_K': T,
        'band_gap_eV': E_g,
        'n_i_m3': n_i,
        'n_m3': n,
        'p_m3': p,
        'N_D_m3': N_D,
        'N_A_m3': N_A,
        'carrier_type': 'n' if n > p else ('p' if p > n else 'intrinsic'),
        'E_F_minus_E_Fi_eV': fermi_level_from_intrinsic(sc_key, T, N_D, N_A),
        'epsilon_r': SEMICONDUCTORS[sc_key]['epsilon_r'],
        'origin_tag': (
            "FIRST_PRINCIPLES: Varshni band gap E_g(T). "
            "FIRST_PRINCIPLES: Fermi-Dirac n_i = √(N_c N_v) exp(−E_g/2kT). "
            "FIRST_PRINCIPLES: charge neutrality + mass action. "
            "MEASURED: E_g, m_e*, m_h*, ε_r."
        ),
    }

    if N_D > 0 and N_A > 0:
        V_bi = built_in_voltage(sc_key, N_D, N_A, T)
        W = depletion_width(sc_key, N_D, N_A, 0.0, T)
        result['V_bi_V'] = V_bi
        result['depletion_width_m'] = W

    return result
