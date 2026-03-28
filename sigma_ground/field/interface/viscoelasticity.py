"""
Viscoelasticity — time-dependent mechanical response of materials.

Derivation chain:
  mechanical.py (elastic moduli)
  + thermal.py (Debye temperature → attempt frequency)
  + diffusion.py (activation energy → relaxation time)
  → viscoelasticity.py (Maxwell, Kelvin-Voigt, SLS models)

Real materials are neither perfectly elastic (springs) nor perfectly
viscous (dashpots). They are viscoelastic: they creep under sustained
load and relax under sustained strain. The time-dependent response is
fully determined by the elastic moduli and the thermal relaxation time.

Derivation chains:

  1. Relaxation Time τ (FIRST_PRINCIPLES: Arrhenius)
     τ = (1/ν_D) × exp(E_a / k_BT)

     Where:
       ν_D = Debye frequency (attempt frequency for atomic jumps)
       E_a = activation energy for diffusive rearrangement (MEASURED)
       k_B = Boltzmann constant
       T = temperature

     ν_D is DERIVED from Debye temperature: ν_D = k_B Θ_D / h
     E_a comes from diffusion.py (same barrier governs viscous flow).

  2. Maxwell Model (FIRST_PRINCIPLES: series spring-dashpot)
     Stress relaxation:   σ(t) = σ₀ × exp(−t/τ)
     Creep:               ε(t) = ε₀ + (σ/E)×(t/τ)
     Complex modulus:     E*(ω) = E × iωτ / (1 + iωτ)

     The Maxwell model captures: stress relaxation (correct),
     unbounded creep at constant stress (incorrect for solids).

  3. Kelvin-Voigt Model (FIRST_PRINCIPLES: parallel spring-dashpot)
     Creep:               ε(t) = (σ/E) × (1 − exp(−t/τ))
     No stress relaxation (immediate elastic response).

     The KV model captures: bounded creep (correct for solids),
     no stress relaxation (incorrect — real materials relax).

  4. Standard Linear Solid (Zener Model) — combines both
     ε(t) = (σ/E_R) × [1 − (1 − E_R/E_U) × exp(−t/τ_ε)]
     σ(t) = σ₀ × [E_R/E_U + (1 − E_R/E_U) × exp(−t/τ_σ)]

     Where:
       E_U = unrelaxed (instantaneous) modulus ≈ Young's modulus
       E_R = relaxed (long-time) modulus ≈ 0.3-0.9 × E_U
       τ_σ = stress relaxation time
       τ_ε = creep retardation time (τ_σ × E_U/E_R)

     The SLS captures BOTH bounded creep and stress relaxation.

  5. Dynamic Mechanical Properties (FIRST_PRINCIPLES: harmonic response)
     Storage modulus:    E'(ω) = E × (ωτ)² / (1 + (ωτ)²)     [Maxwell]
     Loss modulus:       E"(ω) = E × ωτ / (1 + (ωτ)²)         [Maxwell]
     Loss tangent:       tan δ = E"/E' = 1/(ωτ)                [Maxwell]

     The loss tangent peaks where ωτ = 1 — the material absorbs
     maximum energy when the loading frequency matches the natural
     relaxation rate. This is measurable via DMA (Dynamic Mechanical
     Analysis).

  6. Viscosity from Elastic Modulus (FIRST_PRINCIPLES: Maxwell)
     η = E × τ

     The Maxwell viscosity is the product of stiffness and relaxation
     time. For metals at room temperature, τ is astronomical (years)
     and η ≈ 10²⁰ Pa·s. Near the melting point, τ drops to ~10⁻¹² s
     and η ≈ 10⁻³ Pa·s (liquid-like).

σ-dependence:
  Elastic moduli: shift via cohesive energy (documented in mechanical.py).
  Relaxation time: τ(σ) = (1/ν_D(σ)) × exp(E_a(σ)/k_BT).
    - ν_D shifts through Debye temperature (heavier nuclei → lower ν)
    - E_a shifts with cohesive energy (stiffer lattice → higher barrier)
  Net effect: heavier nuclei → stiffer + slower relaxation.

Origin tags:
  - Maxwell model: FIRST_PRINCIPLES (constitutive equation)
  - Kelvin-Voigt: FIRST_PRINCIPLES (constitutive equation)
  - Standard Linear Solid: FIRST_PRINCIPLES (Zener 1948)
  - Relaxation time: FIRST_PRINCIPLES (Arrhenius) + MEASURED (E_a)
  - Dynamic moduli: FIRST_PRINCIPLES (Fourier transform of constitutive)
"""

import math
from .mechanical import youngs_modulus, shear_modulus
from .thermal import debye_temperature
from .diffusion import activation_energy_ev, DIFFUSION_DATA
from ..constants import K_B, HBAR, H_PLANCK, EV_TO_J, SIGMA_HERE
from ..scale import scale_ratio


# ── Relaxation Time ───────────────────────────────────────────────

def relaxation_time(material_key, T, sigma=SIGMA_HERE):
    """Primary relaxation time τ (seconds).

    FIRST_PRINCIPLES: Arrhenius attempt-frequency model.

    τ = (1/ν_D) × exp(E_a / k_BT)

    Where:
      ν_D = k_B × Θ_D / h  (Debye attempt frequency)
      E_a = diffusion activation energy (eV)

    This τ governs the rate of atomic rearrangement under stress.
    It's the same physics as diffusion and creep — all controlled
    by the same thermal activation barrier.

    Args:
        material_key: key into DIFFUSION_DATA (must also be in MATERIALS)
        T: temperature in Kelvin (must be > 0)
        sigma: σ-field value

    Returns:
        Relaxation time in seconds.
    """
    if T <= 0:
        return float('inf')

    theta_D = debye_temperature(material_key, sigma)
    nu_D = K_B * theta_D / H_PLANCK  # attempt frequency (Hz)

    E_a = activation_energy_ev(material_key, sigma)
    E_a_J = E_a * EV_TO_J

    exponent = E_a_J / (K_B * T)
    # Clamp to prevent overflow
    exponent = min(exponent, 700)

    return (1.0 / nu_D) * math.exp(exponent)


# ── Relaxation Ratio ──────────────────────────────────────────────
# The ratio E_R / E_U (relaxed to unrelaxed modulus) depends on
# how much of the elastic strain can relax via atomic diffusion.
#
# At T << T_melt: almost no relaxation, E_R/E_U → 1.0
# At T → T_melt: significant relaxation, E_R/E_U → 0.3-0.5
#
# We model this as:
#   E_R/E_U = 1 - 0.7 × (T/T_melt)²
#
# where the T² dependence captures the accelerating onset of
# relaxation near melting. The 0.7 cap means E_R ≥ 0.3 × E_U
# even at the melting point (the lattice doesn't fully liquefy).

_RELAX_RATIO_MAX = 0.7  # Maximum fractional relaxation (at T_melt)


def _relaxation_ratio(material_key, T):
    """Ratio E_R/E_U from temperature and melting point."""
    from .surface import MATERIALS
    T_melt = MATERIALS[material_key].get('melting_point_K', 1800)
    if T_melt <= 0:
        return 1.0

    homologous = min(T / T_melt, 1.0)
    return 1.0 - _RELAX_RATIO_MAX * homologous ** 2


# ── Maxwell Model ─────────────────────────────────────────────────

def maxwell_stress_relaxation(material_key, t, sigma_0, T,
                               sigma_field=SIGMA_HERE):
    """Stress relaxation under constant strain (Maxwell model).

    σ(t) = σ₀ × exp(−t/τ)

    FIRST_PRINCIPLES: series spring-dashpot constitutive equation.

    This gives the stress decay when a material is suddenly strained
    and held. The stress decays exponentially with time constant τ.

    Args:
        material_key: key into DIFFUSION_DATA
        t: time in seconds
        sigma_0: initial stress in Pa
        T: temperature in K
        sigma_field: σ-field value

    Returns:
        Stress at time t in Pa.
    """
    tau = relaxation_time(material_key, T, sigma_field)

    if tau <= 0 or tau == float('inf'):
        return sigma_0  # no relaxation

    return sigma_0 * math.exp(-t / tau)


def maxwell_creep_strain(material_key, t, applied_stress, T,
                          sigma=SIGMA_HERE):
    """Creep strain under constant stress (Maxwell model).

    ε(t) = σ/E + σ/(E×τ) × t = σ/E × (1 + t/τ)

    FIRST_PRINCIPLES: Maxwell constitutive equation.

    WARNING: Maxwell creep is unbounded (ε → ∞ as t → ∞).
    For bounded creep, use kelvin_voigt_creep or sls_creep.

    Args:
        material_key: key into DIFFUSION_DATA
        t: time in seconds
        applied_stress: constant applied stress in Pa
        T: temperature in K
        sigma: σ-field value

    Returns:
        Total strain (dimensionless).
    """
    E = youngs_modulus(material_key, sigma)
    tau = relaxation_time(material_key, T, sigma)

    if E <= 0:
        return 0.0

    elastic_strain = applied_stress / E

    if tau <= 0 or tau == float('inf'):
        return elastic_strain

    return elastic_strain * (1.0 + t / tau)


# ── Kelvin-Voigt Model ───────────────────────────────────────────

def kelvin_voigt_creep(material_key, t, applied_stress, T,
                        sigma=SIGMA_HERE):
    """Creep strain under constant stress (Kelvin-Voigt model).

    ε(t) = (σ/E) × (1 − exp(−t/τ))

    FIRST_PRINCIPLES: parallel spring-dashpot constitutive equation.

    KV creep is bounded: strain asymptotes to σ/E. This is the
    correct behavior for solids (unlike Maxwell's unbounded creep).
    However, KV has no instantaneous elastic response.

    Args:
        material_key: key into DIFFUSION_DATA
        t: time in seconds
        applied_stress: constant applied stress in Pa
        T: temperature in K
        sigma: σ-field value

    Returns:
        Creep strain (dimensionless).
    """
    E = youngs_modulus(material_key, sigma)
    tau = relaxation_time(material_key, T, sigma)

    if E <= 0:
        return 0.0

    equilibrium_strain = applied_stress / E

    if tau <= 0 or tau == float('inf'):
        return equilibrium_strain

    return equilibrium_strain * (1.0 - math.exp(-t / tau))


# ── Standard Linear Solid (Zener Model) ──────────────────────────

def sls_creep(material_key, t, applied_stress, T, sigma=SIGMA_HERE):
    """Creep strain under constant stress (Standard Linear Solid).

    ε(t) = (σ/E_R) × [1 − (1 − E_R/E_U) × exp(−t/τ_ε)]

    FIRST_PRINCIPLES: Zener (1948) three-element model.

    The SLS combines the best of Maxwell and Kelvin-Voigt:
    - Instantaneous elastic response (like Maxwell)
    - Bounded creep (like Kelvin-Voigt)
    - Stress relaxation (like Maxwell)

    Args:
        material_key: key into DIFFUSION_DATA
        t: time in seconds
        applied_stress: constant applied stress in Pa
        T: temperature in K
        sigma: σ-field value

    Returns:
        Creep strain (dimensionless).
    """
    E_U = youngs_modulus(material_key, sigma)  # unrelaxed modulus
    ratio = _relaxation_ratio(material_key, T)
    E_R = E_U * ratio  # relaxed modulus

    if E_R <= 0:
        return 0.0

    tau_sigma = relaxation_time(material_key, T, sigma)

    # Retardation time τ_ε = τ_σ × E_U/E_R
    if tau_sigma <= 0 or tau_sigma == float('inf') or ratio <= 0:
        return applied_stress / E_U

    tau_epsilon = tau_sigma * (E_U / E_R) if E_R > 0 else float('inf')

    if tau_epsilon == float('inf'):
        return applied_stress / E_U

    delta = 1.0 - ratio  # fractional relaxation
    return (applied_stress / E_R) * (1.0 - delta * math.exp(-t / tau_epsilon))


def sls_stress_relaxation(material_key, t, initial_strain, T,
                           sigma=SIGMA_HERE):
    """Stress relaxation under constant strain (Standard Linear Solid).

    σ(t) = ε₀ × E_U × [E_R/E_U + (1 − E_R/E_U) × exp(−t/τ_σ)]

    Stress relaxes from σ₀ = ε₀ × E_U down to σ_∞ = ε₀ × E_R.

    Args:
        material_key: key into DIFFUSION_DATA
        t: time in seconds
        initial_strain: applied constant strain (dimensionless)
        T: temperature in K
        sigma: σ-field value

    Returns:
        Stress at time t in Pa.
    """
    E_U = youngs_modulus(material_key, sigma)
    ratio = _relaxation_ratio(material_key, T)

    tau = relaxation_time(material_key, T, sigma)

    if tau <= 0 or tau == float('inf'):
        return initial_strain * E_U

    sigma_0 = initial_strain * E_U
    delta = 1.0 - ratio
    return sigma_0 * (ratio + delta * math.exp(-t / tau))


# ── Dynamic Mechanical Properties ─────────────────────────────────

def storage_modulus(material_key, omega, T, sigma=SIGMA_HERE):
    """Storage modulus E'(ω) — elastic energy stored per cycle.

    E'(ω) = E_R + (E_U − E_R) × (ωτ)² / (1 + (ωτ)²)

    FIRST_PRINCIPLES: real part of complex modulus (SLS model).

    E' → E_R at low frequency (relaxed) and E' → E_U at high
    frequency (unrelaxed, glassy).

    Args:
        material_key: key into DIFFUSION_DATA
        omega: angular frequency (rad/s)
        T: temperature in K
        sigma: σ-field value

    Returns:
        Storage modulus in Pa.
    """
    E_U = youngs_modulus(material_key, sigma)
    ratio = _relaxation_ratio(material_key, T)
    E_R = E_U * ratio

    tau = relaxation_time(material_key, T, sigma)

    if tau <= 0 or tau == float('inf'):
        return E_U

    wt = omega * tau
    wt2 = wt * wt
    return E_R + (E_U - E_R) * wt2 / (1.0 + wt2)


def loss_modulus(material_key, omega, T, sigma=SIGMA_HERE):
    """Loss modulus E"(ω) — energy dissipated per cycle.

    E"(ω) = (E_U − E_R) × ωτ / (1 + (ωτ)²)

    FIRST_PRINCIPLES: imaginary part of complex modulus (SLS model).

    Peaks at ωτ = 1 (the glass transition frequency for that T).

    Args:
        material_key: key into DIFFUSION_DATA
        omega: angular frequency (rad/s)
        T: temperature in K
        sigma: σ-field value

    Returns:
        Loss modulus in Pa.
    """
    E_U = youngs_modulus(material_key, sigma)
    ratio = _relaxation_ratio(material_key, T)
    E_R = E_U * ratio

    tau = relaxation_time(material_key, T, sigma)

    if tau <= 0 or tau == float('inf'):
        return 0.0

    wt = omega * tau
    return (E_U - E_R) * wt / (1.0 + wt * wt)


def loss_tangent(material_key, omega, T, sigma=SIGMA_HERE):
    """Loss tangent tan δ = E"/E' — damping ratio.

    FIRST_PRINCIPLES: ratio of dissipated to stored energy per cycle.

    tan δ peaks at the glass transition frequency. Higher tan δ
    means more damping (good for vibration absorption, bad for
    structural stiffness).

    Typical values:
      Steel at room temp:   tan δ ≈ 10⁻⁴ (very low damping)
      Rubber at room temp:  tan δ ≈ 0.1-1.0 (high damping)
      Metal near T_melt:    tan δ ≈ 0.01-0.1

    Args:
        material_key: key into DIFFUSION_DATA
        omega: angular frequency (rad/s)
        T: temperature in K
        sigma: σ-field value

    Returns:
        Loss tangent (dimensionless).
    """
    Ep = storage_modulus(material_key, omega, T, sigma)
    Epp = loss_modulus(material_key, omega, T, sigma)

    if Ep <= 0:
        return 0.0

    return Epp / Ep


def peak_damping_frequency(material_key, T, sigma=SIGMA_HERE):
    """Frequency at which loss tangent peaks (rad/s).

    ω_peak = 1/τ (for single-relaxation-time models).

    FIRST_PRINCIPLES: the maximum of E"(ω)/E'(ω) occurs when ωτ = 1
    for the SLS model. This is the "glass transition" frequency at
    temperature T.

    Args:
        material_key: key into DIFFUSION_DATA
        T: temperature in K
        sigma: σ-field value

    Returns:
        Angular frequency in rad/s.
    """
    tau = relaxation_time(material_key, T, sigma)

    if tau <= 0 or tau == float('inf'):
        return 0.0

    return 1.0 / tau


# ── Maxwell Viscosity ─────────────────────────────────────────────

def maxwell_viscosity(material_key, T, sigma=SIGMA_HERE):
    """Effective viscosity from Maxwell model (Pa·s).

    η = E × τ

    FIRST_PRINCIPLES: Maxwell constitutive equation.

    For metals: η ≈ 10²⁰ Pa·s at room T (solid behavior).
    Near melting: η drops to ~10⁻² Pa·s (liquid-like).

    Args:
        material_key: key into DIFFUSION_DATA
        T: temperature in K
        sigma: σ-field value

    Returns:
        Viscosity in Pa·s.
    """
    E = youngs_modulus(material_key, sigma)
    tau = relaxation_time(material_key, T, sigma)

    if tau == float('inf'):
        return float('inf')

    return E * tau


# ── σ-field functions ─────────────────────────────────────────────

def sigma_relaxation_ratio(material_key, T, sigma):
    """Ratio of relaxation time at σ to relaxation time at σ=0.

    τ(σ)/τ(0) = (ν_D(0)/ν_D(σ)) × exp((E_a(σ) − E_a(0)) / k_BT)

    For σ > 0: heavier nuclei → lower ν_D AND higher E_a → τ increases.
    Materials become STIFFER and SLOWER.

    Args:
        material_key: key into DIFFUSION_DATA
        T: temperature in K
        sigma: σ-field value

    Returns:
        τ(σ)/τ(SIGMA_HERE), dimensionless.
    """
    tau_0 = relaxation_time(material_key, T, SIGMA_HERE)
    tau_s = relaxation_time(material_key, T, sigma)

    if tau_0 <= 0 or tau_0 == float('inf'):
        return 1.0

    return tau_s / tau_0


# ── Diagnostics ───────────────────────────────────────────────────

def viscoelastic_report(material_key, T=300.0, omega=1.0,
                         sigma=SIGMA_HERE):
    """Complete viscoelastic report for a material at given T and ω.

    Args:
        material_key: key into DIFFUSION_DATA
        T: temperature in K
        omega: angular frequency in rad/s
        sigma: σ-field value

    Returns:
        dict with all derived viscoelastic properties.
    """
    tau = relaxation_time(material_key, T, sigma)
    E_U = youngs_modulus(material_key, sigma)
    ratio = _relaxation_ratio(material_key, T)

    return {
        'material': material_key,
        'T_K': T,
        'omega_rad_s': omega,
        'sigma': sigma,
        'relaxation_time_s': tau,
        'youngs_modulus_GPa': E_U / 1e9,
        'relaxed_modulus_GPa': E_U * ratio / 1e9,
        'relaxation_ratio_ER_EU': ratio,
        'storage_modulus_GPa': storage_modulus(material_key, omega, T, sigma) / 1e9,
        'loss_modulus_GPa': loss_modulus(material_key, omega, T, sigma) / 1e9,
        'loss_tangent': loss_tangent(material_key, omega, T, sigma),
        'peak_damping_freq_rad_s': peak_damping_frequency(material_key, T, sigma),
        'maxwell_viscosity_Pa_s': maxwell_viscosity(material_key, T, sigma),
    }


def full_report(T=300.0, omega=1.0, sigma=SIGMA_HERE):
    """Viscoelastic reports for ALL materials in DIFFUSION_DATA.

    Rule 9: if one, then all.
    """
    return {key: viscoelastic_report(key, T, omega, sigma)
            for key in DIFFUSION_DATA}
