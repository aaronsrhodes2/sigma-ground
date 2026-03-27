"""
Diffusion — mass and heat transport in solids, liquids, and gases.

Derivation chains:

  1. Solid-State Diffusion (Arrhenius, FIRST_PRINCIPLES)
     D = D₀ × exp(−E_a / k_BT)

     Where:
       D₀ = pre-exponential factor ≈ ν₀ × a² (attempt frequency × jump distance²)
       E_a = activation energy for vacancy migration
       ν₀ ≈ k_B Θ_D / h  (Debye frequency, from thermal.py)
       a = lattice parameter (from surface.py MATERIALS)

     FIRST_PRINCIPLES: transition state theory (Eyring 1935).
     The atom hops to an adjacent vacancy by thermally overcoming a potential
     barrier. The rate is set by the Boltzmann factor exp(−E_a/k_BT).

     σ-dependence: E_a scales with cohesive energy (stiffer lattice → higher
     barrier). Debye frequency shifts through nuclear mass → attempt frequency
     changes. Net: D(σ) < D(0) at σ > 0 (heavier nuclei diffuse slower).

  2. Fick's First Law (FIRST_PRINCIPLES)
     J = −D × ∂C/∂x   (flux = −diffusivity × concentration gradient)

     The defining relation. A concentration gradient drives a flux
     proportional to the gradient, in the direction of decreasing concentration.

  3. Fick's Second Law (FIRST_PRINCIPLES)
     ∂C/∂t = D × ∂²C/∂x²   (diffusion equation)

     Solution for semi-infinite solid, constant surface concentration:
       C(x,t) = C_s − (C_s − C_0) × erf(x / (2√(Dt)))

     Error function solution — exact for constant D.

  4. Diffusion Length (FIRST_PRINCIPLES)
     L = √(D × t)   — characteristic penetration distance

     After time t, diffusing species has penetrated ~L into the medium.
     Used for estimating case hardening depth, doping profiles, etc.

  5. Thermal Diffusivity (FIRST_PRINCIPLES)
     α = κ / (ρ × c_p)

     Where κ = thermal conductivity, ρ = density, c_p = specific heat.
     This is the thermal analogue of mass diffusivity — how fast temperature
     disturbances propagate. All three quantities come from our cascade.

  6. Einstein-Stokes Diffusion in Liquids (FIRST_PRINCIPLES)
     D = k_BT / (6π η r)

     Where η = viscosity, r = solute radius.
     Einstein (1905), Stokes (1851). Valid for spherical particles in
     continuum fluid (particle >> solvent molecule).

     σ-dependence: through η(σ) from fluid.py.

  7. Gas-Phase Diffusion (Chapman-Enskog, FIRST_PRINCIPLES)
     Already in gas.py as gas_diffusivity(). We import for completeness.

σ-dependence summary:
  Solid D: E_a scales with E_coh, ν₀ scales with Θ_D → D shifts
  Liquid D: through η(σ) from fluid.py
  Thermal α: through κ(σ), ρ(σ), c_p(σ) from thermal.py
  Gas D: through molecular mass from gas.py

Origin tags:
  - Arrhenius diffusion: FIRST_PRINCIPLES (transition state theory)
  - Fick's laws: FIRST_PRINCIPLES (conservation + constitutive relation)
  - Error function solution: FIRST_PRINCIPLES (exact PDE solution)
  - Einstein-Stokes: FIRST_PRINCIPLES (fluctuation-dissipation)
  - Activation energies: MEASURED (tabulated per system)
  - σ-dependence: CORE (through □σ = −ξR via thermal.py, mechanical.py)
"""

import math
from .surface import MATERIALS
from .mechanical import _number_density, MECHANICAL_DATA
from .thermal import (
    debye_temperature, thermal_conductivity, specific_heat_j_kg_K,
    sound_velocity,
)
from ..scale import scale_ratio
from ..constants import HBAR, C, K_B, PROTON_QCD_FRACTION

# ── Conversion ────────────────────────────────────────────────────
_EV_TO_JOULE = 1.602176634e-19   # exact (2019 SI)
_AMU_KG = 1.66053906660e-27      # atomic mass unit in kg
_ANGSTROM_M = 1e-10

# Planck constant (not reduced) for Debye frequency
_H_PLANCK = 2.0 * math.pi * HBAR


# ── Activation Energy Database ───────────────────────────────────
# Activation energy for self-diffusion (vacancy mechanism).
# MEASURED from tracer diffusion experiments.
# Sources: Shewmon "Diffusion in Solids" (1989), Mehrer (2007).
#
# f_coh: fraction of cohesive energy that approximates E_a.
# This is an empirical correlation: E_a ≈ f_coh × E_coh.
# For vacancy diffusion in FCC metals: f_coh ≈ 0.55-0.65
# For BCC metals: f_coh ≈ 0.50-0.60
# For diamond cubic: f_coh ≈ 0.70-0.80 (covalent, higher barrier)
#
# D0_measured: pre-exponential factor (m²/s), MEASURED.
# E_a_measured_ev: activation energy (eV), MEASURED.

DIFFUSION_DATA = {
    'iron': {
        'E_a_measured_ev': 2.87,    # Mehrer (2007), α-Fe self-diffusion
        'D0_measured': 2.0e-4,      # m²/s (Mehrer 2007)
        'f_coh': 0.67,              # 2.87/4.28 — calibration check
    },
    'copper': {
        'E_a_measured_ev': 2.19,    # Rothman & Peterson (1969)
        'D0_measured': 7.8e-5,      # m²/s
        'f_coh': 0.63,              # 2.19/3.49
    },
    'aluminum': {
        'E_a_measured_ev': 1.28,    # Lundy & Murdock (1962)
        'D0_measured': 1.7e-4,      # m²/s
        'f_coh': 0.38,              # 1.28/3.39
    },
    'gold': {
        'E_a_measured_ev': 1.81,    # Makin et al. (1957)
        'D0_measured': 1.07e-5,     # m²/s
        'f_coh': 0.48,              # 1.81/3.81
    },
    'silicon': {
        'E_a_measured_ev': 4.75,    # Tang et al. (1997), Si self-diffusion
        'D0_measured': 5.3e-4,      # m²/s
        'f_coh': 1.03,              # covalent: E_a can exceed E_coh
    },
    'tungsten': {
        'E_a_measured_ev': 5.45,    # Mundy et al. (1978)
        'D0_measured': 5.4e-5,      # m²/s
        'f_coh': 0.61,              # 5.45/8.90
    },
    'nickel': {
        'E_a_measured_ev': 2.88,    # Bakker (1968)
        'D0_measured': 1.9e-4,      # m²/s
        'f_coh': 0.65,              # 2.88/4.44
    },
    'titanium': {
        'E_a_measured_ev': 3.14,    # Herzig & Köhler (1987), β-Ti
        'D0_measured': 8.6e-6,      # m²/s
        'f_coh': 0.64,              # 3.14/4.85
    },
}


# ── Solid-State Self-Diffusion ───────────────────────────────────

def activation_energy_ev(material_key, sigma=0.0):
    """Activation energy for self-diffusion (eV).

    At σ=0: uses MEASURED value from tracer diffusion.
    At σ>0: scales E_a with cohesive energy shift (stiffer lattice
    → higher barrier).

    E_a(σ) = E_a(0) × E_coh(σ)/E_coh(0)

    FIRST_PRINCIPLES scaling: the vacancy migration barrier scales
    with the depth of the interatomic potential well.

    Args:
        material_key: key into DIFFUSION_DATA
        sigma: σ-field value

    Returns:
        Activation energy in eV
    """
    E_a_0 = DIFFUSION_DATA[material_key]['E_a_measured_ev']
    if sigma == 0.0:
        return E_a_0

    # Scale with cohesive energy shift
    mat = MATERIALS[material_key]
    e_coh_0 = mat['cohesive_energy_ev']
    f_qcd = PROTON_QCD_FRACTION
    mass_ratio = (1.0 - f_qcd) + f_qcd * scale_ratio(sigma)
    f_zpe = 0.01
    zpe_correction = f_zpe * (1.0 - 1.0 / math.sqrt(mass_ratio))
    e_coh_sigma = e_coh_0 * (1.0 + zpe_correction)

    return E_a_0 * (e_coh_sigma / e_coh_0)


def solid_diffusivity(material_key, T=1000.0, sigma=0.0):
    """Self-diffusion coefficient in solids (m²/s).

    D = D₀ × exp(−E_a / k_BT)

    FIRST_PRINCIPLES: Arrhenius-type activated process.
    MEASURED: D₀ and E_a from tracer diffusion experiments.

    At σ>0: E_a shifts with cohesive energy → D decreases
    (heavier nuclei → stiffer lattice → higher barrier).

    Args:
        material_key: key into DIFFUSION_DATA
        T: temperature in Kelvin (must be > 0)
        sigma: σ-field value

    Returns:
        Diffusion coefficient in m²/s
    """
    if T <= 0:
        raise ValueError(f"T={T} K: temperature must be positive")

    data = DIFFUSION_DATA[material_key]
    D0 = data['D0_measured']
    E_a = activation_energy_ev(material_key, sigma)

    # E_a in eV → Joules for Boltzmann factor
    return D0 * math.exp(-E_a * _EV_TO_JOULE / (K_B * T))


# ── Fick's Laws ──────────────────────────────────────────────────

def ficks_first_law(D, dC_dx):
    """Diffusion flux from Fick's first law (particles/m²/s or mol/m²/s).

    J = −D × dC/dx

    FIRST_PRINCIPLES: constitutive relation for diffusive transport.
    Flux is proportional to and in the direction of decreasing concentration.

    Args:
        D: diffusion coefficient (m²/s)
        dC_dx: concentration gradient (units/m⁴ or mol/m⁴)

    Returns:
        Flux J (units/m²/s or mol/m²/s), positive in +x direction
    """
    return -D * dC_dx


def ficks_second_law_erf(C_surface, C_initial, x, D, t):
    """Concentration profile from Fick's second law (error function solution).

    C(x,t) = C_s − (C_s − C_0) × erf(x / (2√(Dt)))

    FIRST_PRINCIPLES: exact solution of the diffusion equation
    ∂C/∂t = D ∂²C/∂x² for semi-infinite solid, constant surface
    concentration C_s, initial uniform concentration C_0.

    Args:
        C_surface: surface concentration (constant boundary condition)
        C_initial: initial bulk concentration
        x: depth from surface (m)
        D: diffusion coefficient (m²/s)
        t: time (s)

    Returns:
        Concentration at depth x after time t
    """
    if t <= 0:
        raise ValueError(f"t={t} s: time must be positive")
    if D <= 0:
        raise ValueError(f"D={D} m²/s: diffusivity must be positive")

    eta = x / (2.0 * math.sqrt(D * t))
    return C_surface - (C_surface - C_initial) * math.erf(eta)


# ── Diffusion Length ─────────────────────────────────────────────

def diffusion_length(D, t):
    """Characteristic diffusion length (m).

    L = √(D × t)

    FIRST_PRINCIPLES: dimensional analysis of the diffusion equation.
    Species penetrates distance ~L in time t.

    Args:
        D: diffusion coefficient (m²/s)
        t: time (s)

    Returns:
        Diffusion length in metres
    """
    if D < 0 or t < 0:
        raise ValueError("D and t must be non-negative")
    return math.sqrt(D * t)


def time_to_penetrate(D, depth):
    """Time for diffusion front to reach given depth (s).

    t = depth² / D

    FIRST_PRINCIPLES: inversion of L = √(Dt).

    Args:
        D: diffusion coefficient (m²/s)
        depth: target penetration depth (m)

    Returns:
        Time in seconds
    """
    if D <= 0:
        raise ValueError(f"D={D}: diffusivity must be positive")
    return depth ** 2 / D


# ── Thermal Diffusivity ─────────────────────────────────────────

def thermal_diffusivity(material_key, T=300.0, sigma=0.0):
    """Thermal diffusivity α = κ/(ρ c_p) (m²/s).

    FIRST_PRINCIPLES: the thermal analogue of mass diffusivity.
    How fast temperature disturbances propagate through the material.

    All inputs from our cascade:
      κ from thermal.py (phonon transport)
      ρ from surface.py MATERIALS
      c_p from thermal.py (Debye model)

    σ-dependence: through κ(σ), ρ(σ), and c_p(σ).

    Args:
        material_key: key into MATERIALS dict
        T: temperature in Kelvin
        sigma: σ-field value

    Returns:
        Thermal diffusivity in m²/s
    """
    kappa = thermal_conductivity(material_key, T, sigma)
    rho = MATERIALS[material_key]['density_kg_m3']
    cp = specific_heat_j_kg_K(material_key, T, sigma)

    # σ correction to density (QCD mass shift)
    if sigma != 0.0:
        f_qcd = PROTON_QCD_FRACTION
        mass_factor = (1.0 - f_qcd) + f_qcd * scale_ratio(sigma)
        rho = rho * mass_factor

    return kappa / (rho * cp)


# ── Einstein-Stokes Diffusion (Liquids) ─────────────────────────

def einstein_stokes_diffusivity(T, viscosity, particle_radius):
    """Diffusion coefficient for a spherical particle in a viscous fluid.

    D = k_BT / (6π η r)

    FIRST_PRINCIPLES: Einstein (1905) + Stokes drag.
    The fluctuation-dissipation theorem relates thermal energy to
    viscous drag. Valid when particle radius >> solvent molecular size.

    Args:
        T: temperature in Kelvin
        viscosity: dynamic viscosity η (Pa·s)
        particle_radius: particle radius r (m)

    Returns:
        Diffusion coefficient in m²/s
    """
    if viscosity <= 0:
        raise ValueError(f"η={viscosity}: viscosity must be positive")
    if particle_radius <= 0:
        raise ValueError(f"r={particle_radius}: radius must be positive")

    return K_B * T / (6.0 * math.pi * viscosity * particle_radius)


# ── Interdiffusion (Darken Relation) ────────────────────────────

def darken_interdiffusion(D_A, D_B, x_A, x_B):
    """Interdiffusion coefficient from Darken relation.

    D̃ = x_B × D_A + x_A × D_B

    FIRST_PRINCIPLES: Darken (1948). In a binary alloy, the
    interdiffusion coefficient is the composition-weighted average
    of the individual tracer diffusivities.

    Args:
        D_A: tracer diffusivity of species A (m²/s)
        D_B: tracer diffusivity of species B (m²/s)
        x_A: mole fraction of A
        x_B: mole fraction of B

    Returns:
        Interdiffusion coefficient in m²/s
    """
    return x_B * D_A + x_A * D_B


# ── σ-Shifted Diffusion ─────────────────────────────────────────

def sigma_diffusion_shift(material_key, T=1000.0, sigma=0.0):
    """Fractional change in self-diffusion at given σ.

    Returns D(σ)/D(0) — the ratio of shifted to unshifted diffusivity.

    CORE: σ-dependence through E_a(σ) and attempt frequency.
    At σ > 0: D decreases (higher barrier, heavier atoms).

    Args:
        material_key: key into DIFFUSION_DATA
        T: temperature in Kelvin
        sigma: σ-field value

    Returns:
        Ratio D(σ)/D(0) (dimensionless, ≤1 for σ>0)
    """
    if sigma == 0.0:
        return 1.0
    D_0 = solid_diffusivity(material_key, T, 0.0)
    D_s = solid_diffusivity(material_key, T, sigma)
    if D_0 == 0.0:
        return 0.0
    return D_s / D_0


# ── Nagatha Integration ──────────────────────────────────────────

def material_diffusion_properties(material_key, T=1000.0, sigma=0.0):
    """Export diffusion properties in Nagatha-compatible format.

    Returns a dict that can be merged into Nagatha's material database.
    """
    D = solid_diffusivity(material_key, T, sigma)
    E_a = activation_energy_ev(material_key, sigma)
    alpha = thermal_diffusivity(material_key, min(T, 300.0), sigma)
    L_1hr = diffusion_length(D, 3600.0)

    return {
        'self_diffusivity_m2_s': D,
        'activation_energy_ev': E_a,
        'thermal_diffusivity_m2_s': alpha,
        'diffusion_length_1hr_m': L_1hr,
        'temperature_K': T,
        'sigma': sigma,
        'diffusion_shift_ratio': sigma_diffusion_shift(material_key, T, sigma),
        'origin_tag': (
            "FIRST_PRINCIPLES: Arrhenius diffusion (transition state theory). "
            "FIRST_PRINCIPLES: Fick's laws (conservation + constitutive). "
            "MEASURED: D₀ and E_a from tracer diffusion experiments. "
            "FIRST_PRINCIPLES: thermal diffusivity κ/(ρc_p). "
            "CORE: σ-dependence through nuclear mass → E_a shift."
        ),
    }
