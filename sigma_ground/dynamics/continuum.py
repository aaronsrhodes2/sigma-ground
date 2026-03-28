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
    4. Force accumulation (gravity + buoyancy + pressure gradient + viscosity)
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
      - Periodic boundary option

    Args:
        parcels: list of ContinuumParcel
        gravity: Vec3 gravitational acceleration
        ground: GroundPlane or False
        smoothing_h: SPH smoothing length (m). None → auto from particle spacing.
        reference_density: ρ₀ for buoyancy (kg/m³). None → from first parcel.
        reference_temperature: T₀ for Boussinesq (K). Default 300.
    """

    def __init__(self, parcels, gravity=None, ground=None,
                 smoothing_h=None, reference_density=None,
                 reference_temperature=300.0):
        super().__init__(parcels, gravity=gravity, ground=ground)

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

    def continuum_parcels(self):
        """Return only ContinuumParcel instances."""
        return [p for p in self.parcels
                if isinstance(p, ContinuumParcel) and not p.is_static]


# ══════════════════════════════════════════════════════════════════════
# SPH OPERATORS — density, pressure gradient, viscosity, heat
# ══════════════════════════════════════════════════════════════════════

def _sph_density_sum(parcels, h):
    """Compute SPH density for each parcel.

    ρᵢ = Σⱼ mⱼ W(|xᵢ − xⱼ|, h)

    Also builds neighbor lists for force computation.
    """
    n = len(parcels)
    for i in range(n):
        pi = parcels[i]
        rho_i = pi.mass * W(0.0, h)  # self-contribution
        pi.neighbors = []

        for j in range(n):
            if i == j:
                continue
            pj = parcels[j]
            dx = pi.position.x - pj.position.x
            dy = pi.position.y - pj.position.y
            dz = pi.position.z - pj.position.z
            r = math.sqrt(dx * dx + dy * dy + dz * dz)

            if r < 2.0 * h:
                rho_i += pj.mass * W(r, h)
                pi.neighbors.append(j)

        pi.sph_density = max(rho_i, 1e-10)


def _pressure_from_eos(parcel):
    """Compute pressure from density using Tait EOS.

    P = K × (ρ/ρ₀ − 1)

    K comes from the interface cascade (mechanical.py → bulk_modulus).
    """
    K = parcel.bulk_modulus
    rho_0 = parcel.props['density_kg_m3']
    rho = parcel.sph_density
    P = pressure_tait(rho, rho_0, K)
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


def _sph_viscous_force(parcels, h, alpha_visc=1.0):
    """SPH artificial viscosity (Monaghan 1992).

    Πᵢⱼ = −α h c̄ μᵢⱼ / ρ̄   when vᵢⱼ · rᵢⱼ < 0
    μᵢⱼ = h (vᵢⱼ · rᵢⱼ) / (rᵢⱼ² + 0.01h²)

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
            c_bar = 0.5 * (pi.sound_speed + pj.sound_speed)
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

    # ── 2. SPH density sum ──
    _sph_density_sum(parcels, h)

    # ── 3. Pressure from EOS ──
    for p in parcels:
        _pressure_from_eos(p)

    # ── 4-6. Force accumulation ──
    a_pressure = _sph_pressure_force(parcels, h)
    a_viscous = _sph_viscous_force(parcels, h, alpha_visc)

    # ── 7-8. Leapfrog advance ──
    for i, p in enumerate(parcels):
        # Total acceleration
        a_p = a_pressure.get(i, Vec3(0, 0, 0))
        a_v = a_viscous.get(i, Vec3(0, 0, 0))
        a_b = _buoyancy_force(p, scene.rho_0, scene.T_0, g)

        a_total = Vec3(
            g.x + a_p.x + a_v.x + a_b.x,
            g.y + a_p.y + a_v.y + a_b.y,
            g.z + a_p.z + a_v.z + a_b.z,
        )

        # Leapfrog: kick-drift-kick
        # Half kick
        p.velocity = Vec3(
            p.velocity.x + a_total.x * dt * 0.5,
            p.velocity.y + a_total.y * dt * 0.5,
            p.velocity.z + a_total.z * dt * 0.5,
        )
        # Drift
        p.position = Vec3(
            p.position.x + p.velocity.x * dt,
            p.position.y + p.velocity.y * dt,
            p.position.z + p.velocity.z * dt,
        )
        # Second half kick
        p.velocity = Vec3(
            p.velocity.x + a_total.x * dt * 0.5,
            p.velocity.y + a_total.y * dt * 0.5,
            p.velocity.z + a_total.z * dt * 0.5,
        )

    # ── 9. Heat equation ──
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
