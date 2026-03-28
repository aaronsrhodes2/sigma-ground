"""
Viscosity — drag forces, terminal velocity, and viscous flow phenomena.

This module extends fluid.py (which provides Eyring viscosity for liquid metals
and measured viscosity for known liquids) with:
  - Stokes drag and terminal velocity
  - Drag coefficient models
  - Viscous dissipation (energy loss in flow)
  - Nabarro-Herring creep (solid-state viscous deformation)
  - Viscous boundary layer thickness

fluid.py provides: η(T,σ) for liquids via Eyring or measured data.
gas.py provides: η(T,σ) for gases via Chapman-Enskog kinetic theory.
This module provides: the CONSEQUENCES of viscosity — forces, velocities,
energy dissipation, and creep.

Derivation chains:

  1. Stokes Drag (Stokes 1851, FIRST_PRINCIPLES)
     F_drag = 6π η r v

     Drag force on a sphere of radius r moving at velocity v through
     fluid of viscosity η. Exact solution of Navier-Stokes for Re << 1
     (creeping flow). Valid for small particles in viscous fluids.

  2. Terminal Velocity (FIRST_PRINCIPLES: force balance)
     v_t = 2r²(ρ_p − ρ_f)g / (9η)

     Stokes drag balances gravity minus buoyancy.
     Valid for Re_p < 1 (Stokes regime).

  3. Drag Coefficient (FIRST_PRINCIPLES: dimensional analysis)
     F = ½ C_D ρ A v²

     General drag: C_D depends on Reynolds number.
     - Re << 1: C_D = 24/Re (Stokes regime, exact)
     - 1 < Re < 1000: C_D ≈ 24/Re × (1 + 0.15 Re^0.687) (Schiller-Naumann)
     - Re ~ 10⁵: C_D ≈ 0.44 (Newton regime, turbulent)

  4. Viscous Dissipation Rate (FIRST_PRINCIPLES)
     Φ = τ_ij × ∂v_i/∂x_j

     For simple shear: Φ = η × (dv/dy)²
     Power dissipated per unit volume in viscous flow.

  5. Poiseuille Flow (Hagen-Poiseuille, FIRST_PRINCIPLES)
     Q = π R⁴ ΔP / (8 η L)

     Volume flow rate through a pipe of radius R, length L,
     pressure drop ΔP. Exact solution of Navier-Stokes for
     steady, laminar, fully-developed pipe flow.

  6. Nabarro-Herring Creep (FIRST_PRINCIPLES: diffusion-controlled)
     ε̇ = A_NH × D × σ_stress × Ω / (k_B T d²)

     High-temperature creep by vacancy diffusion through the grain.
     Nabarro (1948), Herring (1950). Rate depends on:
       D = self-diffusion coefficient (from diffusion.py)
       Ω = atomic volume
       d = grain size
       σ_stress = applied stress

     σ-dependence: through D(σ) from diffusion.py.

  7. Boundary Layer Thickness (Blasius, FIRST_PRINCIPLES)
     δ ≈ 5x / √(Re_x)    where Re_x = ρvx/η

     Thickness of the viscous boundary layer at distance x from
     leading edge of a flat plate. Blasius (1908) solution.

σ-dependence summary:
  Liquid η: via fluid.py (Eyring activation through E_coh)
  Gas η: via gas.py (Chapman-Enskog through molecular mass)
  Stokes drag/terminal velocity: through η(σ)
  Creep rate: through D(σ) from diffusion.py
  All flow relations (Poiseuille, boundary layer): through η(σ)

Origin tags:
  - Stokes drag: FIRST_PRINCIPLES (Navier-Stokes solution, Re << 1)
  - Terminal velocity: FIRST_PRINCIPLES (drag-gravity balance)
  - Drag coefficient: FIRST_PRINCIPLES (Stokes) + APPROXIMATION (Schiller-Naumann)
  - Poiseuille flow: FIRST_PRINCIPLES (Navier-Stokes pipe solution)
  - Nabarro-Herring creep: FIRST_PRINCIPLES (diffusion-controlled deformation)
  - Boundary layer: FIRST_PRINCIPLES (Blasius flat-plate solution)
  - σ-dependence: CORE (through □σ = −ξR via fluid.py, diffusion.py)
"""

import math
from ..constants import K_B, AMU_KG as _AMU_KG


# ── Stokes Drag ──────────────────────────────────────────────────

def stokes_drag(viscosity, radius, velocity):
    """Stokes drag force on a sphere (N).

    F = 6π η r v

    FIRST_PRINCIPLES: Stokes (1851), exact Navier-Stokes solution
    for creeping flow (Re << 1) around a rigid sphere.

    Args:
        viscosity: dynamic viscosity η (Pa·s)
        radius: sphere radius r (m)
        velocity: speed v (m/s)

    Returns:
        Drag force in Newtons
    """
    return 6.0 * math.pi * viscosity * radius * abs(velocity)


def terminal_velocity_stokes(radius, rho_particle, rho_fluid, viscosity,
                             g=9.80665):
    """Terminal velocity in Stokes regime (m/s).

    v_t = 2r²(ρ_p − ρ_f)g / (9η)

    FIRST_PRINCIPLES: balance of Stokes drag, gravity, and buoyancy.
    Valid for Re_p = ρ_f v_t (2r) / η < 1.

    Args:
        radius: particle radius (m)
        rho_particle: particle density (kg/m³)
        rho_fluid: fluid density (kg/m³)
        viscosity: dynamic viscosity η (Pa·s)
        g: gravitational acceleration (m/s²), default Earth standard

    Returns:
        Terminal velocity in m/s (positive = sinking, negative = rising)
    """
    if viscosity <= 0:
        raise ValueError(f"η={viscosity}: viscosity must be positive")
    return 2.0 * radius ** 2 * (rho_particle - rho_fluid) * g / (9.0 * viscosity)


def particle_reynolds_number(rho_fluid, velocity, diameter, viscosity):
    """Particle Reynolds number (dimensionless).

    Re_p = ρ_f × v × d / η

    FIRST_PRINCIPLES: ratio of inertial to viscous forces.
    Determines which drag regime applies.

    Args:
        rho_fluid: fluid density (kg/m³)
        velocity: speed (m/s)
        diameter: particle diameter (m)
        viscosity: dynamic viscosity η (Pa·s)

    Returns:
        Re_p (dimensionless)
    """
    return rho_fluid * abs(velocity) * diameter / viscosity


# ── Drag Coefficient ─────────────────────────────────────────────

def drag_coefficient_sphere(Re):
    """Drag coefficient for a sphere as function of Reynolds number.

    Three regimes:
      Re < 1:    C_D = 24/Re  (Stokes, FIRST_PRINCIPLES: exact)
      1-1000:    C_D = 24/Re × (1 + 0.15 Re^0.687) (Schiller-Naumann 1933)
                 APPROXIMATION: empirical correlation, ±5% accuracy.
      1000-2×10⁵: C_D = 0.44 (Newton regime)
                 APPROXIMATION: roughly constant in turbulent wake regime.

    Args:
        Re: Reynolds number (must be > 0)

    Returns:
        C_D (dimensionless)
    """
    if Re <= 0:
        raise ValueError(f"Re={Re}: must be positive")

    if Re < 1.0:
        return 24.0 / Re
    elif Re < 1000.0:
        return 24.0 / Re * (1.0 + 0.15 * Re ** 0.687)
    else:
        return 0.44


def general_drag_force(C_D, rho_fluid, velocity, cross_section_area):
    """General drag force (N).

    F = ½ C_D ρ A v²

    FIRST_PRINCIPLES: definition of drag coefficient.

    Args:
        C_D: drag coefficient (dimensionless)
        rho_fluid: fluid density (kg/m³)
        velocity: speed (m/s)
        cross_section_area: frontal area A (m²)

    Returns:
        Drag force in Newtons
    """
    return 0.5 * C_D * rho_fluid * cross_section_area * velocity ** 2


# ── Poiseuille Flow ──────────────────────────────────────────────

def poiseuille_flow_rate(radius, delta_P, viscosity, length):
    """Volume flow rate through a circular pipe (m³/s).

    Q = π R⁴ ΔP / (8 η L)

    FIRST_PRINCIPLES: Hagen (1839), Poiseuille (1840).
    Exact solution of Navier-Stokes for steady, laminar,
    fully-developed flow in a straight circular pipe.

    Valid for Re_pipe = ρvD/η < ~2300 (laminar regime).

    Args:
        radius: pipe inner radius R (m)
        delta_P: pressure drop along pipe (Pa)
        viscosity: dynamic viscosity η (Pa·s)
        length: pipe length L (m)

    Returns:
        Volume flow rate in m³/s
    """
    if viscosity <= 0:
        raise ValueError(f"η={viscosity}: viscosity must be positive")
    if length <= 0:
        raise ValueError(f"L={length}: length must be positive")

    return math.pi * radius ** 4 * abs(delta_P) / (8.0 * viscosity * length)


def poiseuille_max_velocity(radius, delta_P, viscosity, length):
    """Maximum (centerline) velocity in Poiseuille flow (m/s).

    v_max = R² ΔP / (4 η L)

    FIRST_PRINCIPLES: parabolic velocity profile, maximum at center.

    Args:
        radius: pipe inner radius R (m)
        delta_P: pressure drop (Pa)
        viscosity: dynamic viscosity η (Pa·s)
        length: pipe length (m)

    Returns:
        Maximum velocity in m/s
    """
    if viscosity <= 0:
        raise ValueError(f"η={viscosity}: viscosity must be positive")
    if length <= 0:
        raise ValueError(f"L={length}: length must be positive")

    return radius ** 2 * abs(delta_P) / (4.0 * viscosity * length)


# ── Viscous Dissipation ─────────────────────────────────────────

def viscous_dissipation_simple_shear(viscosity, shear_rate):
    """Viscous dissipation rate per unit volume in simple shear (W/m³).

    Φ = η × (du/dy)²

    FIRST_PRINCIPLES: energy dissipated by viscous friction.
    The kinetic energy of the flow is irreversibly converted to heat.

    Args:
        viscosity: dynamic viscosity η (Pa·s)
        shear_rate: velocity gradient du/dy (1/s)

    Returns:
        Dissipation rate in W/m³
    """
    return viscosity * shear_rate ** 2


def viscous_heating_temperature_rise(viscosity, shear_rate, time,
                                     density, specific_heat):
    """Temperature rise from viscous heating (K).

    ΔT = Φ × t / (ρ × c_p)  where Φ = η × (du/dy)²

    FIRST_PRINCIPLES: energy balance — all dissipated energy goes to heat.
    Assumes adiabatic conditions (no heat loss). Upper bound estimate.

    Args:
        viscosity: dynamic viscosity η (Pa·s)
        shear_rate: velocity gradient (1/s)
        time: duration (s)
        density: fluid density (kg/m³)
        specific_heat: specific heat c_p (J/kg/K)

    Returns:
        Temperature rise in Kelvin
    """
    Phi = viscous_dissipation_simple_shear(viscosity, shear_rate)
    return Phi * time / (density * specific_heat)


# ── Nabarro-Herring Creep ────────────────────────────────────────

def nabarro_herring_strain_rate(diffusivity, stress, atomic_volume,
                                grain_size, T):
    """Nabarro-Herring creep rate (1/s).

    ε̇ = A_NH × D × σ × Ω / (k_B × T × d²)

    FIRST_PRINCIPLES: Nabarro (1948), Herring (1950).
    Vacancy diffusion through the grain interior, driven by
    applied stress. Dominates at high T, low stress, small grains.

    A_NH = 14 (geometric constant for equiaxed grains).
    Coble creep (grain-boundary diffusion) would give d³ dependence
    and a different A — not included here.

    σ-dependence: through D(σ) from diffusion.py.

    Args:
        diffusivity: self-diffusion coefficient D (m²/s)
        stress: applied stress σ (Pa)
        atomic_volume: Ω (m³/atom)
        grain_size: grain diameter d (m)
        T: temperature (K)

    Returns:
        Creep strain rate in 1/s
    """
    if T <= 0:
        raise ValueError(f"T={T}: temperature must be positive")
    if grain_size <= 0:
        raise ValueError(f"d={grain_size}: grain size must be positive")

    A_NH = 14.0  # geometric constant (equiaxed grains)
    return A_NH * diffusivity * stress * atomic_volume / (K_B * T * grain_size ** 2)


# ── Boundary Layer ───────────────────────────────────────────────

def boundary_layer_thickness(x, rho_fluid, velocity, viscosity):
    """Laminar boundary layer thickness at distance x (m).

    δ ≈ 5x / √(Re_x)    where Re_x = ρvx/η

    FIRST_PRINCIPLES: Blasius (1908) flat-plate boundary layer solution.
    Valid for laminar flow (Re_x < ~5×10⁵).

    Args:
        x: distance from leading edge (m)
        rho_fluid: fluid density (kg/m³)
        velocity: free-stream velocity (m/s)
        viscosity: dynamic viscosity η (Pa·s)

    Returns:
        Boundary layer thickness δ in metres
    """
    if x <= 0 or velocity <= 0 or viscosity <= 0 or rho_fluid <= 0:
        raise ValueError("All parameters must be positive")

    Re_x = rho_fluid * velocity * x / viscosity
    return 5.0 * x / math.sqrt(Re_x)


def wall_shear_stress(rho_fluid, velocity, viscosity, x):
    """Wall shear stress on flat plate at distance x (Pa).

    τ_w = 0.332 × ρ v² / √(Re_x)

    FIRST_PRINCIPLES: Blasius solution derivative at wall.

    Args:
        rho_fluid: fluid density (kg/m³)
        velocity: free-stream velocity (m/s)
        viscosity: dynamic viscosity η (Pa·s)
        x: distance from leading edge (m)

    Returns:
        Wall shear stress in Pascals
    """
    if x <= 0 or velocity <= 0 or viscosity <= 0 or rho_fluid <= 0:
        raise ValueError("All parameters must be positive")

    Re_x = rho_fluid * velocity * x / viscosity
    return 0.332 * rho_fluid * velocity ** 2 / math.sqrt(Re_x)


# ── σ-Shifted Drag ──────────────────────────────────────────────

def sigma_terminal_velocity_shift(radius, rho_particle, rho_fluid,
                                  viscosity_0, sigma, viscosity_sigma):
    """Ratio of terminal velocity at σ vs σ=0.

    v_t(σ)/v_t(0) = η(0)/η(σ) × [ρ_p(σ)−ρ_f(σ)] / [ρ_p(0)−ρ_f(0)]

    CORE: σ-dependence through viscosity and density shifts.
    Heavier nuclei → denser particle + more viscous fluid → complex interplay.

    Args:
        radius: particle radius (m) — unused, cancels in ratio
        rho_particle: particle density at σ=0 (kg/m³)
        rho_fluid: fluid density at σ=0 (kg/m³)
        viscosity_0: fluid viscosity at σ=0 (Pa·s)
        sigma: σ-field value
        viscosity_sigma: fluid viscosity at σ (Pa·s)

    Returns:
        Velocity ratio v_t(σ)/v_t(0) (dimensionless)
    """
    from ..constants import PROTON_QCD_FRACTION
    from ..scale import scale_ratio

    if sigma == 0.0:
        return 1.0

    f_qcd = PROTON_QCD_FRACTION
    mass_factor = (1.0 - f_qcd) + f_qcd * scale_ratio(sigma)

    rho_p_sigma = rho_particle * mass_factor
    rho_f_sigma = rho_fluid * mass_factor

    delta_rho_0 = rho_particle - rho_fluid
    delta_rho_sigma = rho_p_sigma - rho_f_sigma

    if delta_rho_0 == 0.0:
        return 1.0

    # Density difference scales identically → ratio = η(0)/η(σ)
    return (viscosity_0 / viscosity_sigma) * (delta_rho_sigma / delta_rho_0)


# ── Nagatha Integration ──────────────────────────────────────────

def viscous_flow_properties(viscosity, rho_fluid, velocity, pipe_radius=None,
                            pipe_length=None, particle_radius=None,
                            rho_particle=None):
    """Export viscous flow properties in Nagatha-compatible format.

    Returns a dict of computed flow quantities given viscosity and geometry.
    """
    result = {
        'viscosity_pa_s': viscosity,
        'fluid_density_kg_m3': rho_fluid,
        'velocity_m_s': velocity,
    }

    if particle_radius is not None and rho_particle is not None:
        F = stokes_drag(viscosity, particle_radius, velocity)
        vt = terminal_velocity_stokes(
            particle_radius, rho_particle, rho_fluid, viscosity)
        Re_p = particle_reynolds_number(
            rho_fluid, velocity, 2 * particle_radius, viscosity)
        result.update({
            'stokes_drag_N': F,
            'terminal_velocity_m_s': vt,
            'particle_reynolds': Re_p,
        })

    if pipe_radius is not None and pipe_length is not None:
        delta_P = 1000.0  # 1 kPa reference
        Q = poiseuille_flow_rate(pipe_radius, delta_P, viscosity, pipe_length)
        v_max = poiseuille_max_velocity(
            pipe_radius, delta_P, viscosity, pipe_length)
        result.update({
            'poiseuille_flow_rate_m3_s': Q,
            'poiseuille_max_velocity_m_s': v_max,
            'reference_delta_P_pa': delta_P,
        })

    result['origin_tag'] = (
        "FIRST_PRINCIPLES: Stokes drag (Navier-Stokes, Re<<1). "
        "FIRST_PRINCIPLES: Hagen-Poiseuille pipe flow (exact laminar). "
        "FIRST_PRINCIPLES: Blasius boundary layer (flat plate). "
        "APPROXIMATION: Schiller-Naumann drag for 1<Re<1000. "
        "CORE: σ-dependence through viscosity and density shifts."
    )
    return result
