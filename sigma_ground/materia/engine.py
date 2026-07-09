"""Materia engine — assemble matter + energy, evolve in time, extract the answer.

This module owns three things:

  1. A worked-result type (MateriaStep / MateriaResult) — the same
     value/units/formula/source provenance the procedures layer uses, plus a
     self-validation verdict. A Materia answer is its own worked example.

  2. An *altitude-aware* aerodynamic force callback. The labs force layer
     (sigma_ground.labs.forces) computes drag/buoyancy against a CONSTANT
     medium density. Real falls happen through a density gradient. Materia
     closes that gap by feeding the LOCAL density — sigma_ground.field.
     interface.atmosphere.density_at_altitude(z) — into the existing,
     validated drag/buoyancy functions at every step. We reuse their
     Reynolds-regime C_d; we only vary ρ with height.

  3. simulate_fall(...) — drive the existing leapfrog integrator
     (sigma_ground.dynamics.stepper.step) for a single body under
     gravity + atmospheric drag + buoyancy until it reaches the ground,
     and report the trajectory's physically meaningful quantities.

Nothing here re-derives physics. Mass comes from the material cascade,
drag from labs.forces, the atmosphere from field.interface, the integration
from dynamics.stepper. Materia is the wiring and the self-check.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


# ── Worked-result provenance types ──────────────────────────────────────
@dataclass
class MateriaStep:
    """One line of the worked answer (mirrors the procedures-layer Step)."""
    label: str
    value: object
    units: str = ""
    formula: str = ""
    source: str = ""


@dataclass
class MateriaResult:
    name: str
    inputs: dict
    steps: list  # list[MateriaStep]
    summary: str = ""
    validation: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)   # named numeric results, for chaining

    def to_dict(self) -> dict:
        return {
            "scenario": self.name,
            "inputs": self.inputs,
            "steps": [vars(s) for s in self.steps],
            "result": vars(self.steps[-1]) if self.steps else None,
            "summary": self.summary,
            "validation": self.validation,
            "outputs": self.outputs,
        }

    def render(self) -> str:
        """Human-readable worked answer with provenance."""
        lines = [f"━━ Materia · {self.name} ━━", self.summary, ""]
        for s in self.steps:
            val = s.value
            if isinstance(val, float):
                val = f"{val:.6g}"
            unit = f" {s.units}" if s.units else ""
            lines.append(f"  • {s.label}: {val}{unit}")
            if s.formula:
                lines.append(f"        {s.formula}   [{s.source}]")
            elif s.source:                       # provenance even without a formula
                lines.append(f"        [{s.source}]")
        v = self.validation
        if v:
            mark = "✓ PASS" if v.get("passed") else "✗ FAIL"
            lines += ["", f"  self-check {mark}: {v.get('note', '')}"]
        return "\n".join(lines)


# ── Air viscosity (Sutherland) — needed for the Reynolds number ─────────
def _air_viscosity(T: float) -> float:
    """Dynamic viscosity of air (Pa·s) via Sutherland's law.

    MEASURED constants: mu_ref = 1.827e-5 Pa·s at 291.15 K, S = 110.4 K.
    Matches sigma_ground.labs.environment.Medium.air().
    """
    mu_ref, T_ref, S = 1.827e-5, 291.15, 110.4
    return mu_ref * (T / T_ref) ** 1.5 * (T_ref + S) / (T + S)


# ── Matter: density wrapper the dynamics parcel expects ─────────────────
class _DensityMaterial:
    """Minimal material exposing density_at_sigma(), as PhysicsParcel wants."""

    def __init__(self, density_kg_m3: float, restitution: float = 0.5):
        self.density_kg_m3 = float(density_kg_m3)
        self.restitution = restitution

    def density_at_sigma(self, sigma):  # σ≈0 at Earth surface
        return self.density_kg_m3


def _material_density(material_key: str, T: float) -> tuple[float, str]:
    """Look up a material's density (kg/m³) and display name from the library."""
    from ..field.interface.surface import MATERIALS
    if material_key not in MATERIALS:
        raise KeyError(
            f"Unknown material {material_key!r}. Known: {sorted(MATERIALS)[:8]}…")
    m = MATERIALS[material_key]
    return float(m["density_kg_m3"]), m.get("name", material_key)


# ── Energy: altitude-aware drag + buoyancy callback ─────────────────────
def make_atmospheric_drag(gravity_vec, T: float = 288.15,
                          uniform_density: float | None = None):
    """Build an external-force callback: drag + buoyancy against the LOCAL air.

    Reuses sigma_ground.labs.forces.drag_force (Reynolds-regime C_d) and
    buoyancy_force verbatim; the only Materia contribution is varying ρ with
    altitude via density_at_altitude(z). Pass uniform_density to pin a
    constant medium (used for the closed-form self-check).
    """
    from .labs.forces import drag_force, buoyancy_force
    from ..field.interface.atmosphere import density_at_altitude

    mu = _air_viscosity(T)

    def callback(parcel):
        if uniform_density is not None:
            rho = uniform_density
        else:
            z = max(0.0, parcel.position.y)         # altitude = +Y
            rho = density_at_altitude(z, T)
        fd = drag_force(parcel, rho, mu)
        fb = buoyancy_force(parcel, rho, gravity_vec)
        return fd + fb

    return callback


# ── Closed-form reference (the self-validation target) ──────────────────
def analytic_terminal_velocity(mass_kg: float, area_m2: float,
                               rho_kg_m3: float, cd: float = 0.44) -> float:
    """v_t = √(2 m g / (ρ C_d A)) — drag–gravity equilibrium.

    FIRST_PRINCIPLES: at terminal velocity ½ρv²C_dA = mg. C_d=0.44 is the
    high-Reynolds (Re>1000) sphere plateau, the regime any dense falling
    sphere settles into.
    """
    from ..field.constants import G  # not used for g; keep import discipline
    g = 9.80665  # standard gravity (BIPM)
    return math.sqrt(2.0 * mass_kg * g / (rho_kg_m3 * cd * area_m2))


# ── The simulation: one body falling through the atmosphere ─────────────
def simulate_fall(material_key: str, radius_m: float, start_altitude_m: float,
                  T: float = 288.15, uniform: bool = False,
                  dt_max: float = 0.01, t_max: float = 600.0,
                  record_every: float | None = None) -> dict:
    """Drop a solid sphere and integrate gravity + atmospheric drag to the ground.

    Args:
        material_key:    key into the library MATERIALS table (e.g. 'copper').
        radius_m:        sphere radius (m).
        start_altitude_m: release altitude (m, +Y).
        T:               air temperature (K, isothermal column).
        uniform:         if True, use constant sea-level density everywhere
                         (gives the exact closed-form asymptote — the self-check).
        dt_max, t_max:   integration controls.
        record_every:    if set, sample {t, altitude_m, speed_m_s, q_drag_J}
                         every this many sim-seconds into ``history`` — the
                         time-resolved feed for a thermal render (the running
                         q_drag is the SAME trapezoidal integral the scalar
                         answer cites, cut into moments).

    Returns a dict of the trajectory's meaningful quantities (all SI).
    """
    from ..shapes import Sphere
    from ..dynamics.vec import Vec3
    from ..dynamics.parcel import PhysicsParcel
    from ..dynamics.scene import PhysicsScene
    from ..dynamics.stepper import step
    from ..field.interface.atmosphere import density_at_altitude

    density, mat_name = _material_density(material_key, T)
    shape = Sphere(radius_m)
    parcel = PhysicsParcel(shape, _DensityMaterial(density),
                           position=Vec3(0.0, start_altitude_m, 0.0),
                           velocity=Vec3(0.0, 0.0, 0.0), label=material_key)

    scene = PhysicsScene([parcel], gravity=Vec3(0.0, -9.80665, 0.0),
                         ground=False)
    rho0 = density_at_altitude(0.0, T)
    cb = make_atmospheric_drag(scene.gravity, T,
                               uniform_density=(rho0 if uniform else None))

    from .labs.forces import drag_force
    g = 9.80665
    mu = _air_viscosity(T)
    max_speed, max_speed_alt = 0.0, start_altitude_m
    n_steps = 0
    q_drag = 0.0          # ∫ |F_drag|·v dt — drag work integrated from the force law
    prev_power = 0.0      # released from rest → zero drag power at t=0
    history = []
    next_rec = 0.0
    if record_every is not None:                     # the t=0 state (at rest)
        history.append({"t": 0.0, "altitude_m": start_altitude_m,
                        "speed_m_s": 0.0, "q_drag_J": 0.0})
        next_rec = record_every
    while parcel.position.y > 0.0 and scene.time < t_max:
        actual_dt = step(scene, dt=dt_max, sub_steps=4, external_forces=cb)
        spd = parcel.velocity.length()
        z = max(0.0, parcel.position.y)
        rho = rho0 if uniform else density_at_altitude(z, T)
        power = drag_force(parcel, rho, mu).length() * spd     # |F_drag|·|v|
        q_drag += 0.5 * (prev_power + power) * actual_dt        # trapezoidal
        prev_power = power
        if record_every is not None and scene.time >= next_rec:
            history.append({"t": scene.time, "altitude_m": z,
                            "speed_m_s": spd, "q_drag_J": q_drag})
            next_rec += record_every
        if spd > max_speed:
            max_speed, max_speed_alt = spd, parcel.position.y
        n_steps += 1

    impact_speed = parcel.velocity.length()
    if record_every is not None:                     # the impact state, always
        history.append({"t": scene.time, "altitude_m": max(0.0, parcel.position.y),
                        "speed_m_s": impact_speed, "q_drag_J": q_drag})

    # Energy budget — an INDEPENDENT measure of the same dissipation, taken
    # from the trajectory endpoints (not the force law). q_budget and q_drag
    # use different inputs, so their agreement is the self-validation that the
    # integrator faithfully integrated the drag force it applied.
    drop_distance = start_altitude_m - parcel.position.y       # actual Δh (y_final ≲ 0)
    pe_lost = parcel.mass * g * drop_distance
    ke_gained = 0.5 * parcel.mass * impact_speed * impact_speed
    q_budget = pe_lost - ke_gained
    # Normalise the drag-work mismatch by the TOTAL energy dropped (pe_lost), NOT
    # by the drag loss q_budget. On a short, slow drop drag is negligible, so
    # q_budget → 0 and |q_budget − q_drag| / q_budget blows up (two tiny numbers
    # disagreeing) even though the impact speed is exactly free-fall-correct.
    # Per energy-dropped it stays well-conditioned for every fall and matches the
    # old ratio when drag dominates (q_budget ≈ pe_lost at terminal velocity).
    energy_residual = abs(q_budget - q_drag) / pe_lost if pe_lost else 0.0
    # Reynolds number at impact → which drag regime we ended in
    Re = rho0 * impact_speed * (2.0 * radius_m) / _air_viscosity(T)
    regime = ("Newton (Re>1e3, C_d≈0.44)" if Re > 1000 else
              "Schiller-Naumann transition" if Re > 1 else "Stokes (Re<1)")

    return {
        "material": material_key,
        "material_name": mat_name,
        "density_kg_m3": density,
        "mass_kg": parcel.mass,
        "area_m2": shape.cross_section(),
        "start_altitude_m": start_altitude_m,
        "sea_level_density_kg_m3": rho0,
        "impact_speed_m_s": impact_speed,
        "fall_time_s": scene.time,
        "max_speed_m_s": max_speed,
        "max_speed_altitude_m": max_speed_alt,
        "reynolds_at_impact": Re,
        "drag_regime": regime,
        "n_steps": n_steps,
        "drag_dissipation_J": q_drag,
        "energy_budget": {"pe_lost_J": pe_lost, "ke_gained_J": ke_gained,
                          "q_drag_budget_J": q_budget},
        "energy_residual": energy_residual,
        "uniform_atmosphere": uniform,
        "history": history,
    }


# ── Translate dissipated mechanical energy into a temperature rise ───────
def drag_heating_temperature_rise(material_key: str, dissipation_J: float,
                                  mass_kg: float, body_fraction: float = 1.0,
                                  T0: float = 288.15) -> dict:
    """ΔT a body would gain if a fraction of drag dissipation entered it.

    ΔT = f · Q / (m · c_p). The body_fraction f is an EXPLICIT assumption — how
    much of the dissipated energy heats the body vs the surrounding air. f=1 is
    the adiabatic UPPER BOUND (all drag heat into the body); real reentry
    deposits far less in the body and most in the shock-heated air. Materia
    computes the defensible number and flags the assumption rather than hiding
    it. c_p comes from the library, not from memory.
    """
    from ..field.interface.thermal import specific_heat_j_kg_K
    c_p = specific_heat_j_kg_K(material_key, T0)
    dT = body_fraction * dissipation_J / (mass_kg * c_p)
    return {"delta_T_K": dT, "peak_T_K": T0 + dT,
            "specific_heat_J_kgK": c_p, "body_fraction": body_fraction,
            "dissipation_J": dissipation_J}


# ── Mach-dependent drag coefficient (empirical transonic curve) ─────────
def drag_coefficient_mach(mach, cd_subsonic=0.15, cd_peak=0.75,
                          cd_supersonic=0.30, m_rise=0.8, m_peak=1.2):
    """C_d as a function of Mach number — the transonic drag-rise curve.

    EMPIRICAL / FITTED (not first-principles): a subsonic plateau, a sharp
    transonic rise to a peak near M≈1.2 (drag divergence), then a slow
    supersonic decay. The three C_d anchors are INPUTS so each body (streamlined
    vs bluff) supplies its own measured curve. Flagged as fitted — never
    presented as derived.
    """
    m = abs(mach)
    if m < m_rise:
        return cd_subsonic
    if m < m_peak:                                   # transonic rise
        frac = (m - m_rise) / (m_peak - m_rise)
        return cd_subsonic + frac * (cd_peak - cd_subsonic)
    return cd_supersonic + (cd_peak - cd_supersonic) * math.exp(-(m - m_peak) / 0.8)


# ── General 1-D drag motion (vertical descent OR horizontal coast) ──────
def simulate_drag_run(mass_kg, area_m2, *, v0_mps=0.0, start_altitude_m=0.0,
                      orientation="vertical", cd_mode="fixed", cd_value=0.47,
                      cd_mach=None, T=288.15, dt=0.02, t_max=3000.0,
                      sample_every=20):
    """Integrate 1-D motion under drag, recording the full velocity history.

    orientation 'vertical': released from start_altitude, gravity pulls it down,
        air density rises as it descends (density_at_altitude). The high-altitude
        descent / parachute case.
    orientation 'horizontal': launched at v0 along X at sea level, no gravity
        along the motion, constant density. The supersonic-projectile case.

    cd_mode 'fixed' → cd_value; 'mach' → drag_coefficient_mach(M, **cd_mach) with
    M = speed / speed_of_sound(T). Drag uses the universal F = ½ρv²C_dA; only the
    C_d source is pluggable. The atmosphere is isothermal, so the sound speed is
    constant (a simplification, flagged).
    """
    from ..shapes import Sphere
    from ..dynamics.vec import Vec3
    from ..dynamics.parcel import PhysicsParcel
    from ..dynamics.scene import PhysicsScene
    from ..dynamics.stepper import step
    from ..field.interface.atmosphere import density_at_altitude, speed_of_sound

    a_sound = speed_of_sound(T)
    rho_sea = density_at_altitude(0.0, T)
    r_eff = math.sqrt(area_m2 / math.pi)
    vertical = orientation == "vertical"

    if vertical:
        pos, vel = Vec3(0.0, start_altitude_m, 0.0), Vec3(0.0, -v0_mps, 0.0)
        gravity = Vec3(0.0, -9.80665, 0.0)
    else:
        pos, vel = Vec3(0.0, 0.0, 0.0), Vec3(v0_mps, 0.0, 0.0)
        gravity = Vec3(0.0, 0.0, 0.0)

    parcel = PhysicsParcel(Sphere(r_eff), _DensityMaterial(1.0), mass=mass_kg,
                           position=pos, velocity=vel, label="body")
    scene = PhysicsScene([parcel], gravity=gravity, ground=False)

    def cd_at(speed):
        if cd_mode == "mach":
            return drag_coefficient_mach(speed / a_sound, **(cd_mach or {}))
        return cd_value

    def cb(p):
        v = p.velocity
        speed = v.length()
        if speed < 1e-12:
            return Vec3(0.0, 0.0, 0.0)
        rho = density_at_altitude(max(0.0, p.position.y), T) if vertical else rho_sea
        F = 0.5 * rho * speed * speed * cd_at(speed) * area_m2
        return v * (-F / speed)

    history, q_force, prev_power = [], 0.0, 0.0
    max_speed, max_speed_alt, max_mach, mach1_alt = 0.0, start_altitude_m, 0.0, None
    n = 0
    while scene.time < t_max:
        if vertical and parcel.position.y <= 0.0:
            break
        if not vertical and parcel.velocity.length() < max(0.5, 0.30 * v0_mps):
            break
        actual_dt = step(scene, dt=dt, sub_steps=4, external_forces=cb)
        speed = parcel.velocity.length()
        alt = parcel.position.y
        rho = density_at_altitude(max(0.0, alt), T) if vertical else rho_sea
        cd = cd_at(speed)
        q_force += 0.5 * (prev_power + 0.5 * rho * speed ** 3 * cd * area_m2) * actual_dt
        prev_power = 0.5 * rho * speed ** 3 * cd * area_m2
        mach = speed / a_sound
        if speed > max_speed:
            max_speed, max_speed_alt, max_mach = speed, alt, mach
        if vertical and mach1_alt is None and max_mach > 1.0 and mach < 1.0:
            mach1_alt = alt
        if n % sample_every == 0:
            history.append({
                "t": scene.time, "altitude_m": alt,
                "distance_m": None if vertical else parcel.position.x,
                "speed_m_s": speed, "mach": mach, "cd": cd,
                "decel_m_s2": 0.5 * rho * speed * speed * cd * area_m2 / mass_kg})
        n += 1

    final_speed = parcel.velocity.length()
    if vertical:
        drop = start_altitude_m - parcel.position.y
        q_budget = mass_kg * 9.80665 * drop - (
            0.5 * mass_kg * final_speed ** 2 - 0.5 * mass_kg * v0_mps ** 2)
    else:
        q_budget = 0.5 * mass_kg * (v0_mps ** 2 - final_speed ** 2)
    residual = abs(q_budget - q_force) / q_budget if q_budget else 0.0

    return {
        "mass_kg": mass_kg, "area_m2": area_m2, "orientation": orientation,
        "start_altitude_m": start_altitude_m, "speed_of_sound_m_s": a_sound,
        "max_speed_m_s": max_speed, "max_mach": max_mach,
        "max_speed_altitude_m": max_speed_alt, "mach1_altitude_m": mach1_alt,
        "final_speed_m_s": final_speed, "final_mach": final_speed / a_sound,
        "distance_m": None if vertical else parcel.position.x,
        "fall_time_s": scene.time, "history": history,
        "drag_dissipation_J": q_force, "energy_budget_J": q_budget,
        "energy_residual": residual, "went_supersonic": max_mach > 1.0,
    }


# ── Vertical launch (throw straight up) → apex ──────────────────────────
def simulate_vertical_launch(mass_kg, area_m2, cd, v0_mps, T=288.15,
                             dt=0.02, t_max=600.0):
    """Launch a body straight up at v0; integrate gravity + drag to the apex.

    Produces the apex altitude — the natural OUTPUT to feed a descent verb,
    so 'throw it up, then how does it come down?' becomes a two-step chain.
    """
    from ..shapes import Sphere
    from ..dynamics.vec import Vec3
    from ..dynamics.parcel import PhysicsParcel
    from ..dynamics.scene import PhysicsScene
    from ..dynamics.stepper import step
    from ..field.interface.atmosphere import density_at_altitude

    parcel = PhysicsParcel(Sphere(math.sqrt(area_m2 / math.pi)),
                           _DensityMaterial(1.0), mass=mass_kg,
                           position=Vec3(0.0, 0.0, 0.0),
                           velocity=Vec3(0.0, v0_mps, 0.0), label="body")
    scene = PhysicsScene([parcel], gravity=Vec3(0.0, -9.80665, 0.0), ground=False)

    def cb(p):
        v = p.velocity
        speed = v.length()
        if speed < 1e-12:
            return Vec3(0.0, 0.0, 0.0)
        rho = density_at_altitude(max(0.0, p.position.y), T)
        F = 0.5 * rho * speed * speed * cd * area_m2
        return v * (-F / speed)

    apex = 0.0
    while scene.time < t_max and parcel.velocity.y > 0.0:
        step(scene, dt=dt, sub_steps=4, external_forces=cb)
        apex = max(apex, parcel.position.y)
    return {"apex_altitude_m": apex, "time_to_apex_s": scene.time,
            "launch_speed_m_s": v0_mps}
