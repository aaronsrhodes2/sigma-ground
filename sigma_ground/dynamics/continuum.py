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

  SPH energy equation (first law; implemented 2026-10-10):
    duᵢ/dt = ½ Σⱼ mⱼ (Pᵢ/ρᵢ² + Pⱼ/ρⱼ² + Πᵢⱼ) vᵢⱼ·∇ᵢWᵢⱼ   (PdV work + viscous heating)
    — the energy-conserving companion of the symmetric momentum equation;
    dT = du/cp for eos="tait"; booked to scene.energy_bath_J for "isothermal".
    Then conduction + sources:
    dT/dt = (1/cp) Σⱼ mⱼ 2κᵢⱼ(Tᵢ-Tⱼ)/(ρᵢρⱼ) (rᵢⱼ·∇W) / (rᵢⱼ² + 0.01h²)
          + q_source / cp               (q_source in W/kg: internal heat generation)
    with κᵢⱼ the harmonic mean (Cleary & Monaghan 1999); see _heat_conduction.
    scene.energy_ledger() reports kinetic + internal + bath + self-gravity + uniform g.

  Buoyancy (Boussinesq approximation; not applied for eos="ideal"):
    F_buoy / m = −β ΔT g   where β = 3α_L is the volumetric expansion coefficient

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
from ..field.constants import R_GAS as _R_GAS   # R = k_B·N_A (derived, CODATA-exact) — authoritative constants


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
        eos="ideal" (2026-10-10): an ideal gas, P = ρ (R/M) T, with the parcel
            temperature evolved by the first law at constant c_v = R/(M(γ−1)).
            Needs ``ideal_molar_mass_kg_mol`` and ``ideal_gamma`` (e.g. from
            field.interface.gas: molecular mass and heat_capacity_ratio).
            The viscosity's signal speed is the hottest parcel's √(γRT/M)
            (one global c̄ — stated approximation). Heat conduction still uses
            the parcel material's cascade κ; set ``scene.heat_conduction =
            False`` for an adiabatic gas experiment.
    """

    def __init__(self, parcels, gravity=None, ground=None,
                 smoothing_h=None, reference_density=None,
                 reference_temperature=300.0, self_gravity_G=None,
                 gravity_softening_m=None, eos="tait", sound_speed_m_s=None,
                 ideal_molar_mass_kg_mol=None, ideal_gamma=None):
        super().__init__(parcels, gravity=gravity, ground=ground)
        if eos not in ("tait", "isothermal", "ideal"):
            raise ValueError(f"unknown eos {eos!r} — 'tait', 'isothermal' or 'ideal'")
        if eos == "isothermal" and not sound_speed_m_s:
            raise ValueError("eos='isothermal' needs sound_speed_m_s (the "
                             "velocity dispersion — a disclosed Choice)")
        if eos == "ideal" and not (ideal_molar_mass_kg_mol and ideal_gamma and ideal_gamma > 1.0):
            raise ValueError("eos='ideal' needs ideal_molar_mass_kg_mol and ideal_gamma > 1")
        self.eos = eos
        self.sound_speed_m_s = sound_speed_m_s
        self.ideal_molar_mass_kg_mol = ideal_molar_mass_kg_mol
        self.ideal_gamma = ideal_gamma
        self.heat_conduction = True

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
        # First law (2026-10-10): PdV work + viscous heating are booked each
        # step (to parcel T for "tait", to this bath for "isothermal").
        self.energy_equation = True
        self.energy_bath_J = 0.0

    def energy_ledger(self):
        """Energy budget of the continuum parcels (J).

        kinetic      Σ ½ m v²
        internal     Σ internal_energy  (accumulated PdV + viscous work; zero at start)
        bath         work handed to the isothermal reservoir (eos="isothermal")
        self_gravity −Σ_{i<j} G mᵢ mⱼ / √(r² + ε²)  — the potential of the SAME
                     Plummer-softened force continuum_step integrates
        uniform_g    −Σ m g·x  (the background field)
        total        their sum — conserved to O(dt²) by continuum_step when the
                     only forces are pressure, viscosity, self-gravity and
                     uniform g (buoyancy, ground contact, heat sources and
                     conduction to nothing are excluded by design)
        """
        ps = self.continuum_parcels()
        ke = sum(0.5 * p.mass * (p.velocity.x ** 2 + p.velocity.y ** 2 + p.velocity.z ** 2)
                 for p in ps)
        u_int = sum(p.internal_energy for p in ps)
        ug = 0.0
        if self.self_gravity_G is not None:
            eps2 = (self.gravity_softening_m or 0.0) ** 2
            for i in range(len(ps)):
                for j in range(i + 1, len(ps)):
                    dx = ps[i].position.x - ps[j].position.x
                    dy = ps[i].position.y - ps[j].position.y
                    dz = ps[i].position.z - ps[j].position.z
                    ug -= (self.self_gravity_G * ps[i].mass * ps[j].mass
                           / math.sqrt(dx * dx + dy * dy + dz * dz + eps2))
        g = self.gravity
        uext = -sum(p.mass * (g.x * p.position.x + g.y * p.position.y + g.z * p.position.z)
                    for p in ps) if g is not None else 0.0
        total = ke + u_int + self.energy_bath_J + ug + uext
        return {"kinetic": ke, "internal": u_int, "bath": self.energy_bath_J,
                "self_gravity": ug, "uniform_g": uext, "total": total}

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


def _sph_pressure_force(parcels, h, pairs=None):
    """SPH pressure gradient force (and, optionally, its pair forces for the energy ledger).

    aᵢ = −Σⱼ gᵢⱼ,   gᵢⱼ = mⱼ (Pᵢ/ρᵢ² + Pⱼ/ρⱼ²) ∇ᵢWᵢⱼ

    If ``pairs`` (a list of per-parcel lists) is given, each pair term is
    recorded as ``pairs[i].append((j, gx, gy, gz))``. continuum_step books the
    first law from them with the step-averaged velocities,
        Δuᵢ = ½ Δt Σⱼ ḡᵢⱼ·(v̄ᵢ − v̄ⱼ),
    which is the energy-conserving SPH energy equation (Monaghan 1992, Ann.
    Rev. Astron. Astrophys. 30:543) made EXACT in discrete time: since
    mᵢgᵢⱼ = −mⱼgⱼᵢ, Σmᵢ Δuᵢ = −Σmᵢ v̄ᵢ·āᵢ Δt = −ΔK for these forces, to roundoff.

    Returns dict of {parcel_index: Vec3 acceleration}.
    """
    accel = {}
    n = len(parcels)

    for i in range(n):
        ax, ay, az = 0.0, 0.0, 0.0
        pi = parcels[i]
        rho_i = pi.sph_density
        P_i = pi.pressure
        rec = pairs[i].append if pairs is not None else None

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
            if rec is not None:
                rec((j, -coeff * gx, -coeff * gy, -coeff * gz))

        accel[i] = Vec3(ax, ay, az)

    return accel


def _sph_viscous_force(parcels, h, alpha_visc=1.0, c_override=None, pairs=None):
    """SPH artificial viscosity (Monaghan 1992).

    Πᵢⱼ = −α h c̄ μᵢⱼ / ρ̄   when vᵢⱼ · rᵢⱼ < 0
    μᵢⱼ = h (vᵢⱼ · rᵢⱼ) / (rᵢⱼ² + 0.01h²)

    ``c_override``: use this sound speed for c̄ instead of the parcels'
    material sound speed — set by the isothermal EOS path, where the
    dispersion c_s is the physical signal speed (iron's ~5 km/s would be
    the condensed-matter regime error all over again).

    If ``pairs`` is given, each pair term gᵢⱼ = mⱼ Πᵢⱼ ∇ᵢWᵢⱼ is recorded as in
    _sph_pressure_force, so the mechanical energy Πᵢⱼ removes is booked as
    internal energy (dissipation ≥ 0 for approaching pairs).

    Returns dict of {parcel_index: Vec3 acceleration}.
    """
    accel = {}
    n = len(parcels)
    eta2 = 0.01 * h * h

    for i in range(n):
        ax, ay, az = 0.0, 0.0, 0.0
        pi = parcels[i]
        rho_i = pi.sph_density
        rec = pairs[i].append if pairs is not None else None

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
            if rec is not None:
                rec((j, -coeff * gx, -coeff * gy, -coeff * gz))

        accel[i] = Vec3(ax, ay, az)

    return accel


def _buoyancy_force(parcel, rho_0, T_0, gravity):
    """Boussinesq buoyancy force.

    F_buoy / m = −β (T − T₀) g,   β = 3 α_L

    FIRST_PRINCIPLES: the density perturbation ρ' = −ρ₀ β ΔT drives buoyancy,
    and β is the VOLUMETRIC coefficient: an isotropic solid's volume strain is
    (1 + α_L ΔT)³ − 1 ≈ 3 α_L ΔT (thermal_expansion.volumetric_expansion_coefficient;
    e.g. iron α_L = 11.8, α_V = 35.4 × 10⁻⁶ K⁻¹, Wikipedia 'Thermal expansion'
    table at 20 °C). `parcel.thermal_expansion` is the cascade's LINEAR α_L(T).
    Anisotropic carriers (carbon_fiber, kevlar) inherit the isotropic 3α_L
    convention of the cascade.

    Not applied under eos="ideal": there P = ρ(R/M)T, so the SPH pressure
    gradient already carries the thermal buoyancy (continuum_step gates it).

    FIXED 2026-10-10: β = 3α_L (was α_L, 1/3 of Archimedes) and off for the ideal EOS (A14).
    """
    alpha = 3.0 * parcel.thermal_expansion        # volumetric β = 3 α_L
    dT = parcel.temperature - T_0
    # Buoyancy acceleration: opposite to gravity, proportional to ΔT
    return Vec3(
        -alpha * dT * gravity.x,
        -alpha * dT * gravity.y,
        -alpha * dT * gravity.z,
    )


def _heat_conduction(parcels, h):
    """SPH heat conduction (Fourier's law).

    duᵢ/dt = Σⱼ mⱼ/(ρᵢρⱼ) · 2κᵢⱼ (Tᵢ−Tⱼ) (rᵢⱼ·∇ᵢWᵢⱼ)/(rᵢⱼ² + 0.01h²)   [W/kg]
    dTᵢ/dt = (duᵢ/dt) / cpᵢ

    The pair sum is already the specific heating rate du/dt: it is the
    Brookshaw (1985) / Monaghan (1992) second-derivative operator, written as
    Jubelgas, Springel & Dolag 2004 (MNRAS 351, 423) eq. 17, whose continuum
    limit is (1/ρ)∇·(κ∇T). The conductivity pair mean is Cleary & Monaghan's
    (1999, J. Comput. Phys. 148, 227) harmonic mean κᵢⱼ = 2κᵢκⱼ/(κᵢ+κⱼ)
    (JSD 2004 eq. 18), chosen deliberately: it is the series conductance of
    two half-gaps, so heat flux stays continuous across a material interface
    (an insulator next to copper limits the flow); it equals κ for like
    neighbours.

    Each pair's heat flow Qᵢⱼ = mᵢmⱼ/(ρᵢρⱼ)·2κᵢⱼ(Tᵢ−Tⱼ)(rᵢⱼ·∇ᵢWᵢⱼ)/(rᵢⱼ²+η²)
    (W) is computed once and booked +Qᵢⱼ to i and −Qᵢⱼ to j, so
    Σ mᵢ cpᵢ dTᵢ/dt = 0 exactly in floating point, whatever the densities.

    FIXED 2026-10-10: dT/dt = rate/cp (was rate/(ρcp), ρ_sph times too slow) and pairwise antisymmetric (A15).

    Returns dict of {parcel_index: dT/dt in K/s}.
    """
    eta2 = 0.01 * h * h
    n = len(parcels)
    heat_W = [0.0] * n            # Σⱼ Qᵢⱼ, heat flowing INTO parcel i (W)

    for i, pi in enumerate(parcels):
        rho_i = pi.sph_density
        T_i = pi.temperature
        kappa_i = pi.thermal_conductivity

        for j in pi.neighbors:
            if j <= i:
                continue                      # each pair once; neighbour lists are symmetric
            pj = parcels[j]
            rho_j = pj.sph_density
            kappa_j = pj.thermal_conductivity
            ksum = kappa_i + kappa_j
            if ksum <= 0.0 or rho_i <= 0.0 or rho_j <= 0.0:
                continue

            dx = pi.position.x - pj.position.x
            dy = pi.position.y - pj.position.y
            dz = pi.position.z - pj.position.z
            r2 = dx * dx + dy * dy + dz * dz

            gx, gy, gz = grad_W(dx, dy, dz, h)
            rdotg = dx * gx + dy * gy + dz * gz       # < 0 inside the kernel

            two_kappa_ij = 4.0 * kappa_i * kappa_j / ksum   # 2 × harmonic mean
            q = (pi.mass * pj.mass / (rho_i * rho_j) * two_kappa_ij
                 * (T_i - pj.temperature) * rdotg / (r2 + eta2))
            heat_W[i] += q
            heat_W[j] -= q

    dTdt = {}
    for i, pi in enumerate(parcels):
        cp_i = pi.specific_heat
        dTdt[i] = heat_W[i] / (pi.mass * cp_i) if (cp_i > 0 and pi.mass > 0) else 0.0
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

    energy_on = getattr(scene, "energy_equation", True)

    def _forces():
        """One full force evaluation at the parcels' CURRENT positions:
        SPH density (+ optional pairwise self-gravity, Phase E), EOS
        pressure, pressure gradient, artificial viscosity, buoyancy,
        uniform g. Returns (accelerations, pairs): a list of Vec3
        accelerations and, when the energy equation is on, the per-parcel
        lists of internal pair forces (j, gx, gy, gz) the first law is
        booked from (None otherwise), both parcel-ordered."""
        a_grav = _sph_density_sum(
            parcels, h,
            gravity_G=getattr(scene, "self_gravity_G", None),
            softening=getattr(scene, "gravity_softening_m", None))
        eos = getattr(scene, "eos", "tait")
        iso_cs = scene.sound_speed_m_s if eos == "isothermal" else None
        if eos == "ideal":
            r_spec = _R_GAS / scene.ideal_molar_mass_kg_mol          # J/(kg·K)
            t_max = 0.0
            for p in parcels:
                p.pressure = p.sph_density * r_spec * p.temperature
                t_max = max(t_max, p.temperature)
            iso_cs = math.sqrt(scene.ideal_gamma * r_spec * t_max)   # viscosity's c̄ (global)
        else:
            for p in parcels:
                if iso_cs is not None:
                    _pressure_isothermal(p, iso_cs)
                else:
                    _pressure_from_eos(p)
        pairs = [[] for _ in parcels] if energy_on else None
        a_pressure = _sph_pressure_force(parcels, h, pairs=pairs)
        a_viscous = _sph_viscous_force(parcels, h, alpha_visc,
                                       c_override=iso_cs, pairs=pairs)
        out = []
        zero = Vec3(0, 0, 0)
        for i, p in enumerate(parcels):
            a_p = a_pressure.get(i, zero)
            a_v = a_viscous.get(i, zero)
            # Boussinesq only where the EOS carries no thermal pressure: the
            # ideal gas's P = ρ(R/M)T already makes hot gas buoyant.
            a_b = zero if eos == "ideal" else _buoyancy_force(p, scene.rho_0, scene.T_0, g)
            a_g = (a_grav or {}).get(i, Vec3(0, 0, 0))
            out.append(Vec3(
                g.x + a_p.x + a_v.x + a_b.x + a_g.x,
                g.y + a_p.y + a_v.y + a_b.y + a_g.y,
                g.z + a_p.z + a_v.z + a_b.z + a_g.z,
            ))
        return out, pairs

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
    cached = getattr(scene, "_vv_accel", None)
    pairs0 = getattr(scene, "_vv_pairs", None)
    if (cached is None or len(cached) != len(parcels)
            or (energy_on and (pairs0 is None or len(pairs0) != len(parcels)))):
        acc0, pairs0 = _forces()
    else:
        acc0 = cached
    for i, p in enumerate(parcels):
        a0 = acc0[i]
        p.position = Vec3(
            p.position.x + p.velocity.x * dt + 0.5 * a0.x * dt * dt,
            p.position.y + p.velocity.y * dt + 0.5 * a0.y * dt * dt,
            p.position.z + p.velocity.z * dt + 0.5 * a0.z * dt * dt,
        )
    acc1, pairs1 = _forces()
    v_old = [(p.velocity.x, p.velocity.y, p.velocity.z) for p in parcels] if energy_on else None
    for i, p in enumerate(parcels):
        a0, a1 = acc0[i], acc1[i]
        p.velocity = Vec3(
            p.velocity.x + 0.5 * (a0.x + a1.x) * dt,
            p.velocity.y + 0.5 * (a0.y + a1.y) * dt,
            p.velocity.z + 0.5 * (a0.z + a1.z) * dt,
        )
    scene._vv_accel = acc1
    scene._vv_pairs = pairs1

    # ── 9. First law: PdV work + viscous dissipation, then conduction ──
    # The work the internal SPH forces (pressure + artificial viscosity) do on
    # the parcels over this step is booked as internal energy, pair by pair:
    #     Δuᵢ = ¼ Δt Σ_{evaluations 0,1} Σⱼ gᵢⱼ·(v̄ᵢ − v̄ⱼ),   v̄ = (vₙ + vₙ₊₁)/2,
    # with the SAME pair forces gᵢⱼ the kick applied. Because mᵢgᵢⱼ = −mⱼgⱼᵢ,
    # Σ mᵢΔuᵢ = −ΔK from these forces exactly (to roundoff): the discrete form of
    # duᵢ/dt = ½Σⱼ mⱼ(Pᵢ/ρᵢ² + Pⱼ/ρⱼ² + Πᵢⱼ) vᵢⱼ·∇ᵢWᵢⱼ (Monaghan 1992). Gravity and
    # uniform g stay in the potential ledger (velocity-Verlet, O(dt²)). Closes the
    # 2026-10-10 KNOWN_GAPS entry "SPH continuum bridge books no viscous heating
    # and no PdV work".
    #   eos="tait"       : the energy heats the parcel, dT = du / c_p
    #                      (c_p ≈ c_v for condensed matter — stated approximation).
    #   eos="isothermal" : T is held by definition (a dispersion closure in
    #                      contact with a bath); the same energy is booked to
    #                      scene.energy_bath_J so the ledger still closes.
    #   eos="ideal"      : u = c_v T with constant c_v = R/(M(γ−1)).
    # scene.energy_equation = False restores the pre-2026-10-10 behaviour.
    eos_now = getattr(scene, "eos", "tait")
    if energy_on:
        c_v_ideal = (_R_GAS / (scene.ideal_molar_mass_kg_mol * (scene.ideal_gamma - 1.0))
                     if eos_now == "ideal" else None)
        vbar = [(0.5 * (vo[0] + p.velocity.x), 0.5 * (vo[1] + p.velocity.y), 0.5 * (vo[2] + p.velocity.z))
                for vo, p in zip(v_old, parcels)]
        for i, p in enumerate(parcels):
            bix, biy, biz = vbar[i]
            w = 0.0
            for lst in (pairs0[i], pairs1[i]):
                for (j, gx, gy, gz) in lst:
                    bj = vbar[j]
                    w += gx * (bix - bj[0]) + gy * (biy - bj[1]) + gz * (biz - bj[2])
            du = 0.25 * dt * w                                # J/kg
            if eos_now == "isothermal":
                scene.energy_bath_J = getattr(scene, "energy_bath_J", 0.0) + p.mass * du
            elif eos_now == "ideal":
                p.internal_energy += p.mass * du
                p.temperature += du / c_v_ideal              # u = c_v T for an ideal gas
            else:
                p.internal_energy += p.mass * du
                p.temperature += du / max(p.specific_heat, 1.0)

    if not getattr(scene, "heat_conduction", True):
        dTdt = {}
    else:
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
