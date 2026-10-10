"""
Continuum bridge — connects the interface physics cascade to SPH dynamics.

The interface layer computes material properties: ρ(P,T), κ(T), η(T), cp(T),
K(T), T_melt(P). The dynamics engine pushes particles around. This module
bridges them: each parcel carries thermodynamic state (T, P, internal energy),
and the stepper pulls properties from the interface cascade at every timestep.

Design:
  ContinuumParcel extends PhysicsParcel with:
    - temperature (K)
    - pressure (Pa)
    - internal_energy (J)
    - material_key (str) — key into MATERIALS dict
    - phase ('solid' or 'liquid')
    - sph_density (kg/m³) — from kernel density sum
    - neighbors (list) — neighbor indices for SPH sums

  continuum_step() replaces the rigid-body stepper with:
    1. SPH density sum (ρ from kernel)
    2. Property lookup (K, η, κ, cp from interface cascade via material_key + T)
    3. Pressure from EOS (Tait or Birch-Murnaghan)
    4. Force accumulation (gravity + buoyancy + pressure gradient + viscosity
       + optional pairwise Newtonian self-gravity, Phase E)
    5. Leapfrog advance (position + velocity)
    6. Heat equation (conduction + compression + sources)
    7. Phase check (T vs T_melt(P))

This is the module that makes "physics out" real. Every force term is
computed from derived material properties, not hardcoded constants.

Physics used:
  SPH momentum equation (Monaghan 1992):
    dv/dt = -Σⱼ mⱼ (Pᵢ/ρᵢ² + Pⱼ/ρⱼ²) ∇W + g + F_visc

  SPH artificial viscosity (Monaghan 1992):
    Πᵢⱼ = -α_visc h cᵢⱼ μᵢⱼ / ρ̄ᵢⱼ    (when particles approach)
    where μᵢⱼ = h vᵢⱼ·rᵢⱼ / (rᵢⱼ² + 0.01h²)

  SPH heat equation:
    dT/dt = (1/ρcp) Σⱼ mⱼ (κᵢ+κⱼ)(Tᵢ-Tⱼ)/(ρᵢρⱼ) (rᵢⱼ·∇W) / rᵢⱼ²
          + (P/ρ²) dρ/dt × (1/cp)     (adiabatic compression)
          + Q_source / (ρ cp)           (internal heat generation)

  Buoyancy (Boussinesq approximation):
    F_buoy = -α ΔT ρ₀ g   where α = thermal expansion coefficient

σ-dependence:
  All σ effects flow through the interface cascade. The bridge doesn't
  add any σ physics — it just connects the existing derivations to
  the dynamics loop. material_properties() calls interface functions
  with the parcel's sigma value, and they handle the rest.

□σ = −ξR
"""

import math
from .vec import Vec3
from .parcel import PhysicsParcel
from .scene import PhysicsScene
from .fluid.kernel import W, grad_W, smoothing_length
from .fluid.eos import pressure_tait


# ══════════════════════════════════════════════════════════════════════
# MATERIAL PROPERTY LOOKUP — the bridge to the interface cascade
# ══════════════════════════════════════════════════════════════════════

def material_properties(material_key, T, P=0.0, sigma=0.0):
    """Pull material properties from the interface cascade.

    This is the single function that connects dynamics to physics.
    Every property is DERIVED by the interface layer from measured
    inputs — nothing is hardcoded here.

    Args:
        material_key: key into MATERIALS ('iron', 'copper', etc.)
        T: temperature in K
        P: pressure in Pa (for phase checks)
        sigma: σ-field value

    Returns:
        dict with:
            density_kg_m3, bulk_modulus_Pa, shear_modulus_Pa,
            thermal_conductivity_W_mK, specific_heat_J_kgK,
            viscosity_Pa_s (if liquid), thermal_expansion_1_K,
            melting_point_K, sound_speed_m_s
    """
    from ..field.interface.surface import MATERIALS
    from ..field.interface.mechanical import (
        bulk_modulus, shear_modulus, youngs_modulus,
    )
    from ..field.interface.thermal import (
        debye_temperature, specific_heat_j_kg_K, thermal_conductivity,
    )
    from ..field.interface.acoustics import (
        longitudinal_wave_speed, density_at_sigma,
    )
    from ..field.interface.thermal_expansion import (
        expansion_coefficient_at_T,
    )
    from ..field.interface.phase_transition import (
        lindemann_melting_estimate, clausius_clapeyron_slope,
    )

    mat = MATERIALS[material_key]

    rho = density_at_sigma(material_key, sigma)
    K = bulk_modulus(material_key, sigma)
    G = shear_modulus(material_key, sigma)
    cp = specific_heat_j_kg_K(material_key, T, sigma)
    kappa = thermal_conductivity(material_key, T, sigma)
    alpha = expansion_coefficient_at_T(material_key, T, sigma)
    v_sound = longitudinal_wave_speed(material_key, sigma)

    # Melting point with pressure correction (Clausius-Clapeyron)
    T_melt_0 = lindemann_melting_estimate(material_key, sigma)
    try:
        dT_dP = clausius_clapeyron_slope(material_key)
        T_melt = T_melt_0 + dT_dP * P
    except (KeyError, ZeroDivisionError):
        T_melt = T_melt_0

    return {
        'density_kg_m3': rho,
        'bulk_modulus_Pa': K,
        'shear_modulus_Pa': G,
        'thermal_conductivity_W_mK': kappa,
        'specific_heat_J_kgK': cp,
        'thermal_expansion_1_K': alpha,
        'melting_point_K': T_melt,
        'sound_speed_m_s': v_sound,
    }


# ══════════════════════════════════════════════════════════════════════
# CONTINUUM PARCEL — SPH particle with thermodynamic state
# ══════════════════════════════════════════════════════════════════════

class ContinuumParcel(PhysicsParcel):
    """A parcel of matter with thermodynamic state for continuum simulation.

    Extends PhysicsParcel with temperature, pressure, internal energy,
    and a material_key that connects to the interface physics cascade.

    The key insight: PhysicsParcel.material is a duck-typed object with
    density_at_sigma(). We create a thin wrapper that provides this
    interface while also storing the material_key for cascade lookups.

    Args:
        material_key: key into MATERIALS dict ('iron', 'copper', etc.)
        radius: parcel radius (m)
        temperature: initial temperature (K)
        position: Vec3 position
        velocity: Vec3 velocity
        sigma: σ-field value
        heat_source_W_kg: internal heat generation rate (W/kg),
            e.g. from radioactive decay
        label: debugging name
    """

    def __init__(self, material_key, radius, temperature=300.0,
                 position=None, velocity=None, sigma=0.0,
                 heat_source_W_kg=0.0, label=''):

        # Create a minimal material object for PhysicsParcel
        mat = _CascadeMaterial(material_key, sigma)

        super().__init__(
            radius=radius,
            material=mat,
            position=position,
            velocity=velocity,
            sigma=sigma,
            label=label or material_key,
        )

        self.material_key = material_key
        self.temperature = float(temperature)
        self.pressure = 0.0
        self.internal_energy = 0.0
        self.heat_source_W_kg = float(heat_source_W_kg)
        self.phase = 'solid'

        # SPH state (set during neighbor search)
        self.sph_density = mat.density_kg_m3
        self.neighbors = []

        # Cached material properties (updated each step)
        self._props = None

    def update_properties(self):
        """Pull current properties from the interface cascade.

        Called once per timestep. Caches results so multiple force
        calculations don't redundantly query the cascade.
        """
        self._props = material_properties(
            self.material_key, self.temperature, self.pressure, self.sigma
        )
        # Check phase
        if self.temperature > self._props['melting_point_K']:
            self.phase = 'liquid'
        else:
            self.phase = 'solid'

    @property
    def props(self):
        """Cached material properties dict."""
        if self._props is None:
            self.update_properties()
        return self._props

    @property
    def sound_speed(self):
        """Speed of sound in this parcel (m/s)."""
        return self.props['sound_speed_m_s']

    @property
    def bulk_modulus(self):
        """Bulk modulus K (Pa)."""
        return self.props['bulk_modulus_Pa']

    @property
    def specific_heat(self):
        """Specific heat cp (J/(kg·K))."""
        return self.props['specific_heat_J_kgK']

    @property
    def thermal_conductivity(self):
        """Thermal conductivity κ (W/(m·K))."""
        return self.props['thermal_conductivity_W_mK']

    @property
    def thermal_expansion(self):
        """Linear thermal expansion α (1/K)."""
        return self.props['thermal_expansion_1_K']

    def __repr__(self):
        return (f"ContinuumParcel('{self.material_key}', "
                f"T={self.temperature:.0f}K, P={self.pressure:.0e}Pa, "
                f"phase='{self.phase}', pos={self.position})")


class _CascadeMaterial:
    """Thin wrapper that provides density_at_sigma() from the cascade.

    PhysicsParcel expects material.density_at_sigma(sigma).
    This wrapper provides that by looking up the MATERIALS dict.
    """

    def __init__(self, material_key, sigma=0.0):
        from ..field.interface.surface import MATERIALS
        self.material_key = material_key
        self.density_kg_m3 = MATERIALS[material_key]['density_kg_m3']
        self.restitution = 0.5

    def density_at_sigma(self, sigma):
        from ..field.interface.acoustics import density_at_sigma
        return density_at_sigma(self.material_key, sigma)


# ══════════════════════════════════════════════════════════════════════
# CONTINUUM SCENE — scene with SPH neighbor infrastructure
# ══════════════════════════════════════════════════════════════════════

class ContinuumScene(PhysicsScene):
    """A scene for continuum (SPH) simulation.

    Extends PhysicsScene with:
      - Smoothing length h
      - Reference density ρ₀ and temperature T₀
      - SPH neighbor search
      - Optional Newtonian self-gravity (Phase E — the fluid side of the
        fluid-vs-gravity comparison needs the fluid to gravitate)

    (An earlier docstring claimed a "periodic boundary option" — no such
    code ever existed; the claim was aspirational and is removed rather
    than relied on.)

    Args:
        parcels: list of ContinuumParcel
        gravity: Vec3 UNIFORM gravitational acceleration (a background
            field, e.g. Earth surface g — independent of self_gravity_G)
        ground: GroundPlane or False
        smoothing_h: SPH smoothing length (m). None → auto from particle spacing.
        reference_density: ρ₀ for buoyancy (kg/m³). None → from first parcel.
        reference_temperature: T₀ for Boussinesq (K). Default 300.
        self_gravity_G: gravitational constant for pairwise MUTUAL
            attraction between parcels. None (default) → off, byte-identical
            behavior to before Phase E. Pass scipy-free CODATA 6.674e-11 for
            real matter, or a scaled value for scaled scenarios (a Choice to
            disclose either way).
        gravity_softening_m: Plummer softening ε (m). None → 0.5·h — close
            pairs stay finite; standard numerical regularization.
        eos: "tait" (default — packed condensed matter, P = K(ρ/ρ₀−1)) or
            "isothermal" (dilute swarm, P = ρ·c_s² — the ring/disk regime;
            see _pressure_isothermal for why Tait cannot serve there).
        sound_speed_m_s: effective sound speed for eos="isothermal" —
            physically the velocity dispersion; REQUIRED for that EOS and
            also replaces the material sound speed in the artificial
            viscosity (a km/s condensed-matter c̄ would be just as wrong
            for a dilute swarm as the Tait pressure was).
    """

    def __init__(self, parcels, gravity=None, ground=None,
                 smoothing_h=None, reference_density=None,
                 reference_temperature=300.0, self_gravity_G=None,
                 gravity_softening_m=None, eos="tait", sound_speed_m_s=None):
        super().__init__(parcels, gravity=gravity, ground=ground)
        if eos not in ("tait", "isothermal"):
            raise ValueError(f"unknown eos {eos!r} — 'tait' or 'isothermal'")
        if eos == "isothermal" and not sound_speed_m_s:
            raise ValueError("eos='isothermal' needs sound_speed_m_s (the "
                             "velocity dispersion — a disclosed Choice)")
        self.eos = eos
        self.sound_speed_m_s = sound_speed_m_s

        if smoothing_h is not None:
            self.h = smoothing_h
        elif len(parcels) > 0:
            # Auto: h from particle volume
            r_avg = sum(p.radius for p in parcels) / len(parcels)
            vol = (4.0 / 3.0) * math.pi * r_avg ** 3
            self.h = smoothing_length(vol)
        else:
            self.h = 0.01

        if reference_density is not None:
            self.rho_0 = reference_density
        elif len(parcels) > 0:
            self.rho_0 = parcels[0].props['density_kg_m3']
        else:
            self.rho_0 = 1000.0

        self.T_0 = reference_temperature
        self.self_gravity_G = self_gravity_G
        self.gravity_softening_m = (gravity_softening_m
                                    if gravity_softening_m is not None
                                    else 0.5 * self.h)

    def continuum_parcels(self):
        """Return only ContinuumParcel instances."""
        return [p for p in self.parcels
                if isinstance(p, ContinuumParcel) and not p.is_static]


# ══════════════════════════════════════════════════════════════════════
# SPH OPERATORS — density, pressure gradient, viscosity, heat
# ══════════════════════════════════════════════════════════════════════

def _sph_density_sum(parcels, h, gravity_G=None, softening=None):
    """Compute SPH density for each parcel.

    ρᵢ = Σⱼ mⱼ W(|xᵢ − xⱼ|, h)

    Also builds neighbor lists for force computation.

    Self-gravity (Phase E): when ``gravity_G`` is set, the same pair loop
    accumulates the Newtonian mutual acceleration

        aᵢ += G mⱼ (xⱼ − xᵢ) / (r² + ε²)^(3/2)

    as a SEPARATE, UNCUTOFF accumulator — gravity has no 2h horizon, so it
    must not reuse the kernel-radius-filtered neighbor list. The distances
    are already computed for every pair here, which is why the accumulator
    piggybacks on this loop instead of adding a second O(N²) pass.
    ``softening`` ε (Plummer) keeps close pairs finite — a standard
    numerical regularization, disclosed by the caller, not physics.

    Returns dict {index: Vec3 acceleration} when gravity_G is set, else None.
    """
    n = len(parcels)
    grav = {} if gravity_G is not None else None
    eps2 = (softening * softening) if softening else 0.0
    for i in range(n):
        pi = parcels[i]
        rho_i = pi.mass * W(0.0, h)  # self-contribution
        pi.neighbors = []
        gx = gy = gz = 0.0

        for j in range(n):
            if i == j:
                continue
            pj = parcels[j]
            dx = pi.position.x - pj.position.x
            dy = pi.position.y - pj.position.y
            dz = pi.position.z - pj.position.z
            r2 = dx * dx + dy * dy + dz * dz
            r = math.sqrt(r2)

            if r < 2.0 * h:
                rho_i += pj.mass * W(r, h)
                pi.neighbors.append(j)

            if grav is not None:
                inv_r3 = (r2 + eps2) ** -1.5
                c = gravity_G * pj.mass * inv_r3
                gx -= c * dx          # (xⱼ − xᵢ) = −(dx, dy, dz)
                gy -= c * dy
                gz -= c * dz

        pi.sph_density = max(rho_i, 1e-10)
        if grav is not None:
            grav[i] = Vec3(gx, gy, gz)
    return grav


def _pressure_from_eos(parcel):
    """Compute pressure from density using Tait EOS.

    P = K × (ρ/ρ₀ − 1)

    K comes from the interface cascade (mechanical.py → bulk_modulus).

    REGIME LIMIT (found live, Phase E 2026-07-30): this EOS assumes parcels
    sample a packed continuum near ρ₀. A DILUTE swarm (ring/cloud particles
    whose kernel density is orders below the material's ρ₀) lands at
    P ≈ −K — e.g. −1.7e11 Pa of spurious tension for iron — which ejects
    any pair that drifts inside the kernel radius. Dilute astrophysical
    scenarios use the isothermal EOS instead (eos="isothermal" on the
    scene — see _pressure_isothermal), never a clamp on this one.
    """
    K = parcel.bulk_modulus
    rho_0 = parcel.props['density_kg_m3']
    rho = parcel.sph_density
    P = pressure_tait(rho, rho_0, K)
    parcel.pressure = P
    return P


def _pressure_isothermal(parcel, sound_speed_m_s):
    """Isothermal / velocity-dispersion EOS:  P = ρ_sph · c_s².

    The standard dilute-regime closure (Phase E3): in ring/disk fluid
    theory the effective sound speed IS the particles' velocity dispersion
    (Goldreich & Tremaine 1978 viscous-disk treatment) — pressure comes
    from random motions, not from compressing condensed matter. Always
    non-negative, so a dilute swarm can never manufacture the spurious
    ~ −K tension the Tait EOS extrapolates to (the failure mode found live
    2026-07-30). c_s is a caller-disclosed Choice, typically set from the
    seeded velocity dispersion.
    """
    P = parcel.sph_density * sound_speed_m_s * sound_speed_m_s
    parcel.pressure = P
    return P


def _sph_pressure_force(parcels, h):
    """SPH pressure gradient force.

    aᵢ = −Σⱼ mⱼ (Pᵢ/ρᵢ² + Pⱼ/ρⱼ²) ∇ᵢWᵢⱼ

    Returns dict of {parcel_index: Vec3 acceleration}.
    """
    accel = {}
    n = len(parcels)

    for i in range(n):
        ax, ay, az = 0.0, 0.0, 0.0
        pi = parcels[i]
        rho_i = pi.sph_density
        P_i = pi.pressure

        for j in pi.neighbors:
            pj = parcels[j]
            rho_j = pj.sph_density
            P_j = pj.pressure

            dx = pi.position.x - pj.position.x
            dy = pi.position.y - pj.position.y
            dz = pi.position.z - pj.position.z

            gx, gy, gz = grad_W(dx, dy, dz, h)

            # Symmetric pressure term (Monaghan 1992)
            coeff = -pj.mass * (P_i / (rho_i ** 2) + P_j / (rho_j ** 2))
            ax += coeff * gx
            ay += coeff * gy
            az += coeff * gz

        accel[i] = Vec3(ax, ay, az)

    return accel


def _sph_viscous_force(parcels, h, alpha_visc=1.0, c_override=None):
    """SPH artificial viscosity (Monaghan 1992).

    Πᵢⱼ = −α h c̄ μᵢⱼ / ρ̄   when vᵢⱼ · rᵢⱼ < 0
    μᵢⱼ = h (vᵢⱼ · rᵢⱼ) / (rᵢⱼ² + 0.01h²)

    ``c_override``: use this sound speed for c̄ instead of the parcels'
    material sound speed — set by the isothermal EOS path, where the
    dispersion c_s is the physical signal speed (iron's ~5 km/s would be
    the condensed-matter regime error all over again).

    Returns dict of {parcel_index: Vec3 acceleration}.
    """
    accel = {}
    n = len(parcels)
    eta2 = 0.01 * h * h

    for i in range(n):
        ax, ay, az = 0.0, 0.0, 0.0
        pi = parcels[i]
        rho_i = pi.sph_density

        for j in pi.neighbors:
            pj = parcels[j]
            rho_j = pj.sph_density

            dx = pi.position.x - pj.position.x
            dy = pi.position.y - pj.position.y
            dz = pi.position.z - pj.position.z
            r2 = dx * dx + dy * dy + dz * dz

            dvx = pi.velocity.x - pj.velocity.x
            dvy = pi.velocity.y - pj.velocity.y
            dvz = pi.velocity.z - pj.velocity.z

            vr = dvx * dx + dvy * dy + dvz * dz

            if vr >= 0:
                continue  # particles separating, no viscosity

            mu_ij = h * vr / (r2 + eta2)
            c_bar = (c_override if c_override is not None
                     else 0.5 * (pi.sound_speed + pj.sound_speed))
            rho_bar = 0.5 * (rho_i + rho_j)

            Pi_ij = -alpha_visc * c_bar * mu_ij / rho_bar

            gx, gy, gz = grad_W(dx, dy, dz, h)
            coeff = -pj.mass * Pi_ij
            ax += coeff * gx
            ay += coeff * gy
            az += coeff * gz

        accel[i] = Vec3(ax, ay, az)

    return accel


def _buoyancy_force(parcel, rho_0, T_0, gravity):
    """Boussinesq buoyancy force.

    F_buoy / m = −α (T − T₀) g

    where α is thermal expansion coefficient from the interface cascade.

    FIRST_PRINCIPLES: density perturbation ρ' = −ρ₀ α ΔT drives buoyancy.
    """
    alpha = parcel.thermal_expansion
    dT = parcel.temperature - T_0
    # Buoyancy acceleration: opposite to gravity, proportional to ΔT
    return Vec3(
        -alpha * dT * gravity.x,
        -alpha * dT * gravity.y,
        -alpha * dT * gravity.z,
    )


def _heat_conduction(parcels, h):
    """SPH heat conduction.

    dTᵢ/dt = (1/ρᵢcpᵢ) Σⱼ mⱼ (κᵢ+κⱼ)(Tᵢ−Tⱼ)/(ρᵢρⱼ)
              × (rᵢⱼ · ∇Wᵢⱼ) / (rᵢⱼ² + 0.01h²)

    FIRST_PRINCIPLES: Fourier's law discretized via SPH (Cleary & Monaghan 1999).

    Returns dict of {parcel_index: dT/dt in K/s}.
    """
    dTdt = {}
    eta2 = 0.01 * h * h

    for i, pi in enumerate(parcels):
        rate = 0.0
        rho_i = pi.sph_density
        T_i = pi.temperature
        kappa_i = pi.thermal_conductivity
        cp_i = pi.specific_heat

        if cp_i <= 0 or rho_i <= 0:
            dTdt[i] = 0.0
            continue

        for j in pi.neighbors:
            pj = parcels[j]
            rho_j = pj.sph_density
            T_j = pj.temperature
            kappa_j = pj.thermal_conductivity

            dx = pi.position.x - pj.position.x
            dy = pi.position.y - pj.position.y
            dz = pi.position.z - pj.position.z
            r2 = dx * dx + dy * dy + dz * dz

            gx, gy, gz = grad_W(dx, dy, dz, h)
            rdotg = dx * gx + dy * gy + dz * gz

            kappa_avg = kappa_i + kappa_j  # factor 2 absorbed into formula
            coeff = pj.mass * kappa_avg * (T_i - T_j) / (rho_i * rho_j)
            rate += coeff * rdotg / (r2 + eta2)

        dTdt[i] = rate / (rho_i * cp_i)

    return dTdt


# ══════════════════════════════════════════════════════════════════════
# CONTINUUM STEPPER — the full loop
# ══════════════════════════════════════════════════════════════════════

def continuum_step(scene, dt, alpha_visc=1.0):
    """Advance a ContinuumScene by one timestep.

    The full physics loop:
      1. Update material properties from interface cascade
      2. SPH density sum
      3. Pressure from EOS
      4. Pressure gradient force
      5. Viscous force (artificial viscosity)
      6. Buoyancy force (Boussinesq)
      7. Gravity
      8. Leapfrog velocity + position advance
      9. Heat conduction + internal sources
      10. Phase check

    Args:
        scene: ContinuumScene
        dt: timestep in seconds
        alpha_visc: artificial viscosity coefficient (default 1.0)

    Returns:
        actual_dt (float): elapsed time
    """
    parcels = scene.continuum_parcels()
    if not parcels:
        scene.time += dt
        return dt

    h = scene.h
    g = scene.gravity

    # ── 1. Update material properties from cascade ──
    for p in parcels:
        p.update_properties()

    def _forces():
        """One full force evaluation at the parcels' CURRENT positions:
        SPH density (+ optional pairwise self-gravity, Phase E), EOS
        pressure, pressure gradient, artificial viscosity, buoyancy,
        uniform g. Returns a list of Vec3 accelerations, parcel-ordered."""
        a_grav = _sph_density_sum(
            parcels, h,
            gravity_G=getattr(scene, "self_gravity_G", None),
            softening=getattr(scene, "gravity_softening_m", None))
        iso_cs = (scene.sound_speed_m_s
                  if getattr(scene, "eos", "tait") == "isothermal" else None)
        for p in parcels:
            if iso_cs is not None:
                _pressure_isothermal(p, iso_cs)
            else:
                _pressure_from_eos(p)
        a_pressure = _sph_pressure_force(parcels, h)
        a_viscous = _sph_viscous_force(parcels, h, alpha_visc,
                                       c_override=iso_cs)
        out = []
        for i, p in enumerate(parcels):
            a_p = a_pressure.get(i, Vec3(0, 0, 0))
            a_v = a_viscous.get(i, Vec3(0, 0, 0))
            a_b = _buoyancy_force(p, scene.rho_0, scene.T_0, g)
            a_g = (a_grav or {}).get(i, Vec3(0, 0, 0))
            out.append(Vec3(
                g.x + a_p.x + a_v.x + a_b.x + a_g.x,
                g.y + a_p.y + a_v.y + a_b.y + a_g.y,
                g.z + a_p.z + a_v.z + a_b.z + a_g.z,
            ))
        return out

    # ── 2-8. Velocity-Verlet advance ──
    # TRUE velocity Verlet: drift with a(x_n), re-evaluate forces at x_{n+1},
    # kick with the average. The previous scheme applied a(x_n) in BOTH
    # half-kicks — first-order Euler in disguise — and drifted a bound
    # two-body orbit's energy ~13% over three periods (caught by
    # test_continuum_self_gravity when Phase E made orbits possible).
    # End-of-step accelerations are cached on the scene so this stays ONE
    # force evaluation per step (viscosity is velocity-dependent, so the
    # scheme is the standard SPH approximation, exact for conservative
    # forces). The cache is invalidated if the parcel list changes.
    acc0 = getattr(scene, "_vv_accel", None)
    if acc0 is None or len(acc0) != len(parcels):
        acc0 = _forces()
    for i, p in enumerate(parcels):
        a0 = acc0[i]
        p.position = Vec3(
            p.position.x + p.velocity.x * dt + 0.5 * a0.x * dt * dt,
            p.position.y + p.velocity.y * dt + 0.5 * a0.y * dt * dt,
            p.position.z + p.velocity.z * dt + 0.5 * a0.z * dt * dt,
        )
    acc1 = _forces()
    for i, p in enumerate(parcels):
        a0, a1 = acc0[i], acc1[i]
        p.velocity = Vec3(
            p.velocity.x + 0.5 * (a0.x + a1.x) * dt,
            p.velocity.y + 0.5 * (a0.y + a1.y) * dt,
            p.velocity.z + 0.5 * (a0.z + a1.z) * dt,
        )
    scene._vv_accel = acc1

    # ── 9. Heat equation ──
    # PHYSICS_GAP: only conduction + the external heat source are booked here.
    # The module docstring's adiabatic-compression term (P/ρ²)(dρ/dt)/cp is NOT
    # implemented, and the artificial-viscosity dissipation (step 5) is never
    # converted to internal energy — mechanical energy lost to Π_ij leaves the
    # ledger instead of heating the parcels. Phase E's cloud run showed this as a
    # −387% 'mechanical energy loss'. Filed 2026-10-10 in KNOWN_GAPS.md (the
    # fluid→thermo bridge of the unification program depends on closing it).
    dTdt = _heat_conduction(parcels, h)
    for i, p in enumerate(parcels):
        # Conduction
        dT_cond = dTdt.get(i, 0.0) * dt

        # Internal heat source (e.g. radioactive decay)
        dT_source = p.heat_source_W_kg * dt / max(p.specific_heat, 1.0)

        p.temperature += dT_cond + dT_source
        p.temperature = max(p.temperature, 1.0)  # floor at 1 K

    # ── 10. Phase check ──
    for p in parcels:
        T_melt = p.props['melting_point_K']
        if p.temperature > T_melt and p.phase == 'solid':
            p.phase = 'liquid'
        elif p.temperature < T_melt and p.phase == 'liquid':
            p.phase = 'solid'

    # ── Collision with ground ──
    if scene.ground is not None:
        from .collision import resolve_sphere_plane
        gnd = scene.ground
        for p in parcels:
            resolve_sphere_plane(p, gnd.point, gnd.normal, gnd.restitution)

    scene.time += dt
    return dt


def continuum_step_to(scene, t_end, dt=0.001, alpha_visc=1.0, callback=None):
    """Advance a ContinuumScene until scene.time >= t_end.

    Args:
        scene: ContinuumScene
        t_end: target time (s)
        dt: timestep (s)
        alpha_visc: viscosity coefficient
        callback: optional function(scene, frame_index)

    Returns:
        history: list of (time, snapshot) tuples
    """
    history = []
    frame = 0

    while scene.time < t_end:
        remaining = t_end - scene.time
        actual_dt = min(dt, remaining)
        continuum_step(scene, actual_dt, alpha_visc)

        snapshot = [(p.label, p.position, p.velocity, p.temperature, p.pressure)
                    for p in scene.parcels if isinstance(p, ContinuumParcel)]
        history.append((scene.time, snapshot))

        if callback:
            callback(scene, frame)
        frame += 1

    return history


# ══════════════════════════════════════════════════════════════════════
# CONVENIENCE: CFL timestep for continuum
# ══════════════════════════════════════════════════════════════════════

def cfl_timestep(scene, safety=0.3):
    """CFL-limited timestep for the continuum solver.

    dt = safety × h / max(c_s, v_max)

    where c_s is the fastest sound speed and v_max is the fastest particle.

    Args:
        scene: ContinuumScene
        safety: CFL safety factor (default 0.3)

    Returns:
        dt in seconds
    """
    h = scene.h
    c_max = 0.0
    v_max = 0.0

    for p in scene.continuum_parcels():
        c_max = max(c_max, p.sound_speed)
        v_max = max(v_max, p.velocity.length())

    signal = max(c_max, v_max, 1e-10)
    return safety * h / signal
