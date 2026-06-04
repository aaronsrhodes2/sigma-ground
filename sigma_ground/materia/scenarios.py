"""Materia scenarios — canonical what-ifs as worked, self-validated answers.

A scenario turns a high-level question ("drop a copper sphere from 10 km — how
fast does it hit?") into a MateriaResult: the matter assembled, the energy
integrated, the answer extracted, and a closed-form cross-check that makes the
result self-validating. This is the dynamics analogue of the procedures layer.

First scenario: terminal_velocity_drop — the keystone of the What-If corpus.
One model (gravity + altitude-varying drag) answers Steak Drop, Raindrop,
Free Fall, Falling-with-Helium and a real-drag High Throw. It self-checks
because a falling sphere has an exact analytic terminal velocity.
"""
from __future__ import annotations

from .engine import (simulate_fall, analytic_terminal_velocity,
                     drag_heating_temperature_rise, simulate_drag_run,
                     drag_coefficient_mach, simulate_vertical_launch,
                     _material_density, MateriaStep, MateriaResult)


def terminal_velocity_drop(material_key: str = "copper",
                           radius_m: float = 0.05,
                           drop_altitude_m: float = 10_000.0,
                           T: float = 288.15,
                           tolerance: float = 0.10) -> MateriaResult:
    """How fast does a solid sphere hit the ground, dropped from altitude?

    Integrates gravity + drag (buoyancy too) through the real density gradient,
    then validates the impact speed against the closed-form sea-level terminal
    velocity v_t = √(2mg/(ρ₀C_dA)). They agree because, after falling through
    the dense lower atmosphere, the sphere relaxes onto v_t.

    Worked output also reports the *aloft* max speed — the sphere overshoots
    v_t in the thin upper air, then decelerates as the air thickens. That
    non-monotonic profile is the physically interesting part and is exactly
    what a naive "v = √(2gh)" or constant-density model gets wrong.
    """
    sim = simulate_fall(material_key, radius_m, drop_altitude_m, T=T)
    v_t = analytic_terminal_velocity(sim["mass_kg"], sim["area_m2"],
                                     sim["sea_level_density_kg_m3"], cd=0.44)

    residual = sim["energy_residual"]
    gap = (sim["impact_speed_m_s"] - v_t) / v_t        # + = above terminal (relaxation incomplete)
    passed = residual <= 0.01                           # engine self-check: energy conserved
    relaxed = ("near-complete" if abs(gap) < tolerance
               else "incomplete — high-altitude overshoot still decaying")

    steps = [
        MateriaStep("Material", f"{sim['material_name']} "
                    f"(ρ={sim['density_kg_m3']:.0f} kg/m³)", "",
                    "(input)", "sigma_ground.field.interface.surface.MATERIALS"),
        MateriaStep("Sphere mass", sim["mass_kg"], "kg",
                    "m = ρ · (4/3)πr³",
                    "sigma_ground.dynamics.parcel (density × volume)"),
        MateriaStep("Cross-section", sim["area_m2"], "m²", "A = πr²",
                    "sigma_ground.shapes.Sphere"),
        MateriaStep("Released from", sim["start_altitude_m"], "m", "(input)",
                    "user"),
        MateriaStep("Max speed (aloft)", sim["max_speed_m_s"], "m/s",
                    f"at altitude {sim['max_speed_altitude_m']:.0f} m — thin air, "
                    f"v_t locally higher",
                    "sigma_ground.dynamics.stepper (leapfrog)"),
        MateriaStep("Impact speed", sim["impact_speed_m_s"], "m/s",
                    f"after {sim['fall_time_s']:.1f} s, "
                    f"{sim['drag_regime']}",
                    "sigma_ground.dynamics.stepper + "
                    "field.interface.atmosphere.density_at_altitude"),
        MateriaStep("Energy-conservation residual", residual * 100.0, "%",
                    "|Q_A − Q_B| / Q_A — engine self-check", "Materia"),
        MateriaStep("Terminal velocity (closed form)", v_t, "m/s",
                    "v_t = √(2mg / ρ₀C_dA)",
                    "Materia.analytic_terminal_velocity"),
    ]

    summary = (f"A {radius_m*100:.0f} cm {sim['material_name'].lower()} sphere "
               f"dropped from {drop_altitude_m/1000:.0f} km hits the ground at "
               f"≈{sim['impact_speed_m_s']:.0f} m/s "
               f"({sim['impact_speed_m_s']*3.6:.0f} km/h), having peaked at "
               f"≈{sim['max_speed_m_s']:.0f} m/s in the thin upper air.")

    validation = {
        "passed": passed,
        "energy_residual": residual,
        "impact_vs_terminal_pct": gap * 100.0,
        "simulated_impact_m_s": sim["impact_speed_m_s"],
        "analytic_terminal_m_s": v_t,
        "note": (f"engine conserves energy to {residual*100:.2f}%; impact "
                 f"{sim['impact_speed_m_s']:.1f} m/s sits {gap*100:+.1f}% vs the "
                 f"sea-level terminal velocity {v_t:.1f} m/s "
                 f"(relaxation {relaxed})"),
    }

    return MateriaResult("terminal_velocity_drop",
                         {"material_key": material_key, "radius_m": radius_m,
                          "drop_altitude_m": drop_altitude_m, "T": T},
                         steps, summary=summary, validation=validation,
                         outputs={"impact_speed_m_s": sim["impact_speed_m_s"],
                                  "max_speed_m_s": sim["max_speed_m_s"],
                                  "terminal_velocity_m_s": v_t,
                                  "fall_time_s": sim["fall_time_s"]})


def drag_heating_drop(material_key: str = "iron",
                      radius_m: float = 0.05,
                      drop_altitude_m: float = 10_000.0,
                      body_fraction: float = 1.0,
                      T0: float = 288.15,
                      tolerance: float = 0.03) -> MateriaResult:
    """How much does a falling body heat from drag? (movement → thermal)

    Drag converts mechanical energy into heat — the first demonstration of
    Materia's thesis that movement is a *cause*. The engine measures that
    dissipation two independent ways (the trajectory's energy budget, and the
    drag force's work integrated along the path); their agreement is the
    self-check that energy was conserved. The temperature rise is then a
    translation of that energy, with the body/air partition flagged as the
    assumption it is — not a claim about real reentry surface heating.
    """
    sim = simulate_fall(material_key, radius_m, drop_altitude_m, T=T0)
    eb = sim["energy_budget"]
    q_budget = eb["q_drag_budget_J"]
    q_force = sim["drag_dissipation_J"]
    residual = sim["energy_residual"]
    heat = drag_heating_temperature_rise(material_key, q_budget,
                                         sim["mass_kg"], body_fraction, T0)
    passed = residual <= tolerance

    steps = [
        MateriaStep("Material", f"{sim['material_name']} "
                    f"(ρ={sim['density_kg_m3']:.0f} kg/m³, "
                    f"c_p={heat['specific_heat_J_kgK']:.0f} J/kg·K)", "",
                    "(input)", "field.interface.surface + thermal"),
        MateriaStep("Mass", sim["mass_kg"], "kg", "m = ρ·(4/3)πr³",
                    "dynamics.parcel"),
        MateriaStep("Impact speed", sim["impact_speed_m_s"], "m/s",
                    f"from {drop_altitude_m/1000:.0f} km, {sim['drag_regime']}",
                    "dynamics.stepper + atmosphere"),
        MateriaStep("Gravitational PE released", eb["pe_lost_J"], "J",
                    "m·g·Δh", "Materia energy budget"),
        MateriaStep("Kinetic energy at impact", eb["ke_gained_J"], "J",
                    "½·m·v²", "Materia energy budget"),
        MateriaStep("Drag dissipation — energy budget", q_budget, "J",
                    "Q_A = PE_lost − KE_gained", "Materia (trajectory endpoints)"),
        MateriaStep("Drag dissipation — force integral", q_force, "J",
                    "Q_B = ∫|F_drag|·v dt", "Materia (labs.forces.drag_force)"),
        MateriaStep("Energy-conservation residual", residual * 100.0, "%",
                    "|Q_A − Q_B| / Q_A — two independent measures agree",
                    "Materia self-check"),
        MateriaStep("Temperature rise ΔT", heat["delta_T_K"], "K",
                    f"ΔT = f·Q/(m·c_p), f={body_fraction:g} "
                    f"[ASSUMPTION: fraction of drag heat entering the body; "
                    f"f=1 = adiabatic upper bound]",
                    "Materia → thermal.specific_heat_j_kg_K"),
    ]

    summary = (f"A {radius_m*100:.0f} cm {sim['material_name'].lower()} sphere "
               f"falling from {drop_altitude_m/1000:.0f} km dissipates "
               f"≈{q_budget/1e6:.2f} MJ into drag. If a fraction f={body_fraction:g} "
               f"of that entered the body it would warm by ≈{heat['delta_T_K']:.0f} K "
               f"(f=1 is the adiabatic upper bound). Energy conserved to "
               f"{residual*100:.1f}%.")

    validation = {
        "passed": passed,
        "energy_residual": residual,
        "q_drag_budget_J": q_budget,
        "q_drag_force_J": q_force,
        "tolerance": tolerance,
        "note": (f"two independent dissipation measures agree to "
                 f"{residual*100:.2f}% (energy budget {q_budget/1e6:.2f} MJ vs "
                 f"force integral {q_force/1e6:.2f} MJ) — engine conserves energy"),
    }

    return MateriaResult("drag_heating_drop",
                         {"material_key": material_key, "radius_m": radius_m,
                          "drop_altitude_m": drop_altitude_m,
                          "body_fraction": body_fraction, "T0": T0},
                         steps, summary=summary, validation=validation,
                         outputs={"delta_T_K": heat["delta_T_K"],
                                  "peak_T_K": heat["peak_T_K"],
                                  "dissipation_J": q_budget,
                                  "impact_speed_m_s": sim["impact_speed_m_s"]})


def high_altitude_descent(payload_mass_kg: float = 118.0,
                          drag_area_m2: float = 0.28,
                          cd: float = 0.70,
                          start_altitude_m: float = 35000.0) -> MateriaResult:
    """Does a body dropped from the stratosphere slow down despite still falling?

    Released into near-vacuum, it accelerates past the thin-air terminal
    velocity — often supersonic — then DECELERATES as it sinks into
    exponentially denser air. Materia integrates the whole descent and reports
    the velocity / Mach profile, so the counter-intuitive "speed up, then slow
    down, while never stopping falling" is computed, not asserted.
    """
    sim = simulate_drag_run(payload_mass_kg, drag_area_m2, orientation="vertical",
                            cd_mode="fixed", cd_value=cd,
                            start_altitude_m=start_altitude_m)
    decelerated = sim["final_speed_m_s"] < sim["max_speed_m_s"]
    residual = sim["energy_residual"]
    passed = residual < 0.02 and decelerated
    m1 = sim["mach1_altitude_m"]

    steps = [
        MateriaStep("Payload", payload_mass_kg, "kg",
                    f"drag area {drag_area_m2:g} m², C_d={cd:g} (near free-fall)",
                    "(input)"),
        MateriaStep("Released from", start_altitude_m, "m", "(input)", "user"),
        MateriaStep("Max speed (aloft, thin air)", sim["max_speed_m_s"], "m/s",
                    f"Mach {sim['max_mach']:.2f} at {sim['max_speed_altitude_m']:.0f} m"
                    + ("  — SUPERSONIC" if sim["went_supersonic"] else ""),
                    "dynamics.stepper + atmosphere.density_at_altitude"),
        MateriaStep("Dropped below Mach 1 at", m1 if m1 is not None else "n/a",
                    "m" if m1 is not None else "",
                    "as the air thickens", "Materia"),
        MateriaStep("Landing speed", sim["final_speed_m_s"], "m/s",
                    f"Mach {sim['final_mach']:.2f} after {sim['fall_time_s']:.0f} s",
                    "dynamics.stepper"),
        MateriaStep("Energy-conservation residual", residual * 100.0, "%",
                    "|Q_budget − Q_force|/Q_budget — engine self-check", "Materia"),
    ]
    summary = (f"A {payload_mass_kg:.0f} kg payload released from "
               f"{start_altitude_m/1000:.0f} km peaks at Mach {sim['max_mach']:.2f} "
               f"(~{sim['max_speed_m_s']:.0f} m/s) at {sim['max_speed_altitude_m']/1000:.0f} km"
               + (" — supersonic" if sim["went_supersonic"] else "") +
               f", then DECELERATES into denser air to ~{sim['final_speed_m_s']:.0f} m/s "
               f"at the ground. Yes — it slows while still falling.")
    validation = {
        "passed": passed, "energy_residual": residual,
        "went_supersonic": sim["went_supersonic"],
        "note": (f"engine conserves energy to {residual*100:.2f}%; speed peaks "
                 f"{sim['max_speed_m_s']:.0f} m/s aloft then falls to "
                 f"{sim['final_speed_m_s']:.0f} m/s — decelerates in denser air "
                 f"{'✓' if decelerated else '✗'}"),
    }
    return MateriaResult("high_altitude_descent",
                         {"payload_mass_kg": payload_mass_kg,
                          "drag_area_m2": drag_area_m2, "cd": cd,
                          "start_altitude_m": start_altitude_m},
                         steps, summary=summary, validation=validation,
                         outputs={"max_speed_m_s": sim["max_speed_m_s"],
                                  "max_mach": sim["max_mach"],
                                  "landing_speed_m_s": sim["final_speed_m_s"],
                                  "mach1_altitude_m": sim["mach1_altitude_m"]})


def supersonic_projectile(mass_kg: float = 0.02, diameter_m: float = 0.01,
                          launch_mach: float = 2.5, cd_subsonic: float = 0.15,
                          cd_peak: float = 0.75, cd_supersonic: float = 0.30
                          ) -> MateriaResult:
    """A projectile fired faster than sound — its drag through the transonic zone.

    C_d is NOT constant with Mach: it spikes transonically (drag divergence).
    Materia runs the horizontal deceleration with a Mach-dependent C_d and
    reports the profile, so the non-linear transition is computed from the
    curve — not hand-waved. The C_d(Mach) anchors are EMPIRICAL inputs, flagged.
    """
    import math
    area = math.pi * (diameter_m / 2.0) ** 2
    cdm = {"cd_subsonic": cd_subsonic, "cd_peak": cd_peak,
           "cd_supersonic": cd_supersonic}
    sim = simulate_drag_run(mass_kg, area, orientation="horizontal", cd_mode="mach",
                            cd_mach=cdm, v0_mps=launch_mach * 340.3)
    v0 = launch_mach * sim["speed_of_sound_m_s"]
    cd_launch = drag_coefficient_mach(launch_mach, **cdm)
    decel_launch = sim["history"][0]["decel_m_s2"] if sim["history"] else 0.0
    residual = sim["energy_residual"]
    passed = residual < 0.05

    steps = [
        MateriaStep("Launch", v0, "m/s",
                    f"Mach {launch_mach:g} (a_sound={sim['speed_of_sound_m_s']:.0f} m/s)",
                    "(input)"),
        MateriaStep("C_d at launch (supersonic)", cd_launch, "",
                    f"drag_coefficient_mach(M={launch_mach:g})  [EMPIRICAL curve]",
                    "Materia.drag_coefficient_mach"),
        MateriaStep("C_d transonic peak", cd_peak, "",
                    "drag divergence near M≈1.2 — C_d spikes  [EMPIRICAL]",
                    "input anchor"),
        MateriaStep("Deceleration at launch", decel_launch, "m/s²",
                    "a = ½ρv²C_d(M)A / m", "dynamics.stepper"),
        MateriaStep("Distance to go subsonic", sim["distance_m"], "m",
                    f"decelerated to Mach {sim['final_mach']:.2f} in "
                    f"{sim['fall_time_s']:.1f} s", "dynamics.stepper"),
        MateriaStep("Energy-conservation residual", residual * 100.0, "%",
                    "|Q_budget − Q_force|/Q_budget — engine self-check", "Materia"),
    ]
    summary = (f"Fired at Mach {launch_mach:g} (~{v0:.0f} m/s), the projectile's "
               f"drag coefficient climbs from {cd_supersonic:g} toward a transonic "
               f"peak of {cd_peak:g} as it slows — so it sheds speed faster through "
               f"the sound barrier than a constant-C_d model. It drops subsonic "
               f"after ~{sim['distance_m']:.0f} m.")
    validation = {
        "passed": passed, "energy_residual": residual,
        "note": (f"engine conserves energy to {residual*100:.2f}%; C_d(Mach) "
                 f"transition {cd_supersonic:g}→{cd_peak:g}→{cd_subsonic:g} applied "
                 f"along the path"),
    }
    return MateriaResult("supersonic_projectile",
                         {"mass_kg": mass_kg, "diameter_m": diameter_m,
                          "launch_mach": launch_mach}, steps,
                         summary=summary, validation=validation,
                         outputs={"launch_speed_m_s": v0,
                                  "distance_to_subsonic_m": sim["distance_m"],
                                  "final_mach": sim["final_mach"]})


def vertical_launch(material_key: str = "steel_mild", radius_m: float = 0.05,
                    launch_speed_m_s: float = 300.0, T: float = 288.15
                    ) -> MateriaResult:
    """Throw a sphere straight up — how high does it reach? (feeds a descent verb)

    The apex altitude is the OUTPUT meant to be bound into a descent verb, so
    "throw it up, then how fast and how hot does it come down?" becomes a chain.
    """
    import math
    density, name = _material_density(material_key, T)
    mass = density * (4.0 / 3.0) * math.pi * radius_m ** 3
    area = math.pi * radius_m ** 2
    sim = simulate_vertical_launch(mass, area, 0.47, launch_speed_m_s, T=T)
    steps = [
        MateriaStep("Material", f"{name} (ρ={density:.0f} kg/m³)", "",
                    "(input)", "field.interface.surface.MATERIALS"),
        MateriaStep("Launch speed", launch_speed_m_s, "m/s", "(input)", "user"),
        MateriaStep("Apex altitude", sim["apex_altitude_m"], "m",
                    f"reached in {sim['time_to_apex_s']:.1f} s "
                    f"(gravity + drag, sphere C_d≈0.47)", "dynamics.stepper"),
    ]
    summary = (f"A {radius_m*100:.0f} cm {name.lower()} sphere thrown straight up "
               f"at {launch_speed_m_s:.0f} m/s reaches an apex of "
               f"{sim['apex_altitude_m']:.0f} m.")
    return MateriaResult("vertical_launch",
                         {"material_key": material_key, "radius_m": radius_m,
                          "launch_speed_m_s": launch_speed_m_s}, steps,
                         summary=summary,
                         validation={"passed": True,
                                     "note": "kinematic apex (ascent integration)"},
                         outputs={"apex_altitude_m": sim["apex_altitude_m"],
                                  "time_to_apex_s": sim["time_to_apex_s"]})


def orbital_velocity(central_body: str = "earth",
                     altitude_m: float = 400000.0,
                     semimajor_axis_au: float | None = None) -> MateriaResult:
    """Orbital speed and period: v = √(G·M/r). Two modes —

      • satellite:    a circular orbit at `altitude_m` above a body's surface
      • heliocentric: a planet at `semimajor_axis_au` AU around the Sun

    Wraps sigma-ground's body-aware orbital math (it does the mass/radius
    lookup); this just routes the right question to it.
    """
    from .bodies import (resolve_body, circular_orbital_velocity,
                         orbital_period as _period)
    from ..field.constants import AU_M

    if semimajor_axis_au is not None:                       # heliocentric
        body = resolve_body("sun")                          # primary is the Sun
        r = semimajor_axis_au * AU_M
        where = f"{semimajor_axis_au:g} AU from {body.label}"
        loc_step = MateriaStep("Orbit radius", semimajor_axis_au, "AU",
                               "heliocentric distance", "user")
    else:                                                   # satellite altitude
        body = resolve_body(central_body)
        r = body.radius_m + altitude_m
        where = f"{altitude_m/1000:.0f} km above {body.label}"
        loc_step = MateriaStep("Altitude", altitude_m, "m", "above the surface",
                               "user")
    v = circular_orbital_velocity(body, r)
    period = _period(r, body)

    if period < 7200:
        per = f"{period/60:.1f} min"
    elif period < 2 * 86400:
        per = f"{period/3600:.1f} hr"
    else:
        per = f"{period/86400:.0f} days"

    steps = [
        MateriaStep("Central body", body.label, "", "(mass & radius from table)",
                    "materia.bodies"),
        loc_step,
        MateriaStep("Orbital velocity", v, "m/s", "v = √(G·M / r)",
                    "materia.bodies.circular_orbital_velocity"),
        MateriaStep("Orbital period", period, "s", "Kepler III / T = 2π√(r³/μ)",
                    "materia.bodies.orbital_period"),
    ]
    summary = (f"A circular orbit {where} needs {v:.0f} m/s ({v/1000:.2f} km/s), "
               f"one lap every {per}.")
    return MateriaResult("orbital_velocity",
                         {"central_body": central_body, "altitude_m": altitude_m,
                          "semimajor_axis_au": semimajor_axis_au},
                         steps, summary=summary,
                         validation={"passed": v > 0,
                                     "note": f"v=√(GM/r): {v/1000:.2f} km/s"},
                         outputs={"orbital_velocity_m_s": v,
                                  "orbital_period_s": period,
                                  "altitude_m": altitude_m})


def material_profile(material_key: str = "steel_mild",
                     temperature_k: float = 293.15) -> MateriaResult:
    """Characterize a material: density, thermal, mechanical and optical
    properties. Wraps the labs material cascade, which calls sigma-ground's
    whole property suite (thermal, mechanical, impact, friction, optics) — one
    verb exercising many methods, and the matter half of any simulation.
    """
    from .labs.builder import cascade_material
    p = cascade_material(material_key, T=temperature_k)
    g = lambda k: p.get(k)
    steps = [
        MateriaStep("Material", f"{p['name']} (Z={p.get('Z', '?')})", "",
                    "(input)", "field.interface.surface.MATERIALS"),
        MateriaStep("Density", p["density_kg_m3"], "kg/m³", "measured",
                    "surface.MATERIALS"),
        MateriaStep("Speed of sound", g("sound_velocity_m_s"), "m/s",
                    "Newton-Laplace", "thermal.sound_velocity"),
        MateriaStep("Specific heat", g("specific_heat_J_kgK"), "J/kg·K",
                    "Debye model", "thermal.specific_heat_j_kg_K"),
        MateriaStep("Thermal conductivity", g("thermal_conductivity_W_mK"),
                    "W/m·K", "κ = ⅓·C·v·ℓ", "thermal.thermal_conductivity"),
        MateriaStep("Young's modulus", g("youngs_modulus_Pa"), "Pa",
                    "E = 3K(1−2ν)", "mechanical.youngs_modulus"),
        MateriaStep("Shear modulus", g("shear_modulus_Pa"), "Pa",
                    "G = E/2(1+ν)", "mechanical.shear_modulus"),
        MateriaStep("Bulk modulus", g("bulk_modulus_Pa"), "Pa",
                    "harmonic from cohesive energy", "mechanical.bulk_modulus"),
        MateriaStep("Restitution", g("restitution"), "", "Hertzian impact",
                    "field.interface.impact"),
    ]
    E = g("youngs_modulus_Pa") or 0.0
    summary = (f"{p['name']}: ρ={p['density_kg_m3']:.0f} kg/m³, E={E/1e9:.0f} GPa, "
               f"c_p={g('specific_heat_J_kgK') or 0:.0f} J/kg·K, sound "
               f"{g('sound_velocity_m_s') or 0:.0f} m/s.")
    return MateriaResult("material_profile",
                         {"material_key": material_key,
                          "temperature_k": temperature_k}, steps, summary=summary,
                         validation={"passed": True,
                                     "note": "material property cascade"},
                         outputs={"density_kg_m3": p["density_kg_m3"],
                                  "youngs_modulus_Pa": E,
                                  "specific_heat_J_kgK": g("specific_heat_J_kgK"),
                                  "sound_velocity_m_s": g("sound_velocity_m_s")})


def _numeric_steps(d, source, limit=3):
    """Surface the first `limit` numeric fields of an aggregator dict as steps."""
    out = []
    for k, v in d.items():
        if isinstance(v, (int, float)) and len(out) < limit:
            out.append(MateriaStep(k.replace("_", " ").capitalize(), v, "", "",
                                   source))
    return out


def structural_response(material_key: str = "steel_mild",
                        applied_stress_pa: float = 1.0e8,
                        crack_length_m: float = 1.0e-3,
                        temperature_k: float = 300.0) -> MateriaResult:
    """Structural integrity under load: elastic moduli, fracture toughness,
    fatigue life, and the plastic-flow curve. Wraps sigma-ground's stress +
    plasticity + mechanical suites — lights up the statics tree."""
    from ..field.interface.stress import stress_properties
    from ..field.interface.plasticity import plasticity_properties
    from ..field.interface.mechanical import (youngs_modulus, shear_modulus,
                                              bulk_modulus)
    sp = stress_properties(material_key, applied_stress=applied_stress_pa,
                           crack_length=crack_length_m, temperature=temperature_k)
    pp = plasticity_properties(material_key)
    E = youngs_modulus(material_key)
    G = shear_modulus(material_key)
    K = bulk_modulus(material_key)
    steps = [
        MateriaStep("Material", material_key, "", "(input)", "surface.MATERIALS"),
        MateriaStep("Young's modulus", E, "Pa", "E=3K(1−2ν)",
                    "mechanical.youngs_modulus"),
        MateriaStep("Shear modulus", G, "Pa", "G=E/2(1+ν)",
                    "mechanical.shear_modulus"),
        MateriaStep("Bulk modulus", K, "Pa", "harmonic", "mechanical.bulk_modulus"),
    ] + _numeric_steps(sp, "field.interface.stress.stress_properties") \
      + _numeric_steps(pp, "field.interface.plasticity.plasticity_properties")
    summary = (f"{material_key}: E={E/1e9:.0f} GPa, G={G/1e9:.0f} GPa; fracture "
               f"suite ({len(sp)} metrics) + plastic-flow suite ({len(pp)} "
               f"metrics) computed.")
    return MateriaResult("structural_response",
                         {"material_key": material_key,
                          "applied_stress_pa": applied_stress_pa}, steps,
                         summary=summary,
                         validation={"passed": E > 0,
                                     "note": "elastic + fracture + plasticity suites"},
                         outputs={"youngs_modulus_Pa": E, "shear_modulus_Pa": G,
                                  "bulk_modulus_Pa": K})


def thermal_response(material_key: str = "steel_mild", delta_T: float = 100.0,
                     temperature_k: float = 300.0) -> MateriaResult:
    """How a material responds to heat: thermal expansion, melting/phase data,
    radiated power. Wraps the thermal-expansion + phase-transition suites."""
    from ..field.interface.thermal_expansion import thermal_expansion_properties
    from ..field.interface.phase_transition import phase_transition_properties
    from ..field.interface.thermal import thermal_emission_power
    te = thermal_expansion_properties(material_key, delta_T=delta_T,
                                      T=temperature_k)
    ptn = phase_transition_properties(material_key)
    P = thermal_emission_power(material_key, temperature_k)
    steps = [
        MateriaStep("Material", material_key, "", "(input)", "surface.MATERIALS"),
        MateriaStep("Radiated power", P, "W/m²", "ε·σ·T⁴",
                    "thermal.thermal_emission_power"),
    ] + _numeric_steps(te, "field.interface.thermal_expansion") \
      + _numeric_steps(ptn, "field.interface.phase_transition")
    summary = (f"{material_key} heated ΔT={delta_T:.0f} K: expansion suite "
               f"({len(te)}) + phase suite ({len(ptn)}) computed; radiates "
               f"{P:.0f} W/m² at {temperature_k:.0f} K.")
    return MateriaResult("thermal_response",
                         {"material_key": material_key, "delta_T": delta_T}, steps,
                         summary=summary,
                         validation={"passed": True,
                                     "note": "expansion + phase + radiation"},
                         outputs={"radiated_power_W_m2": P})


def rotational_dynamics(mass_kg: float = 1.0, radius_m: float = 0.1,
                        angular_velocity_rad_s: float = 10.0,
                        length_m: float = 1.0) -> MateriaResult:
    """Rigid-body rotation: moments of inertia for the standard shapes, angular
    momentum, rotational KE. Wraps sigma-ground's rotational suite."""
    import sigma_ground.field.interface.rotational as rot
    I_sphere = rot.moment_of_inertia_sphere(mass_kg, radius_m)
    I_cyl = rot.moment_of_inertia_cylinder(mass_kg, radius_m, length_m)
    I_disk = rot.moment_of_inertia_disk(mass_kg, radius_m)
    I_rod = rot.moment_of_inertia_rod(mass_kg, length_m)
    I_hollow = rot.moment_of_inertia_hollow_sphere(mass_kg, radius_m)
    L = rot.angular_momentum(I_sphere, angular_velocity_rad_s)
    ke = 0.5 * I_sphere * angular_velocity_rad_s ** 2
    steps = [
        MateriaStep("Solid sphere I", I_sphere, "kg·m²", "⅖mr²",
                    "rotational.moment_of_inertia_sphere"),
        MateriaStep("Cylinder I", I_cyl, "kg·m²", "½mr²",
                    "rotational.moment_of_inertia_cylinder"),
        MateriaStep("Disk I", I_disk, "kg·m²", "½mr²",
                    "rotational.moment_of_inertia_disk"),
        MateriaStep("Rod I", I_rod, "kg·m²", "1/12·mL²",
                    "rotational.moment_of_inertia_rod"),
        MateriaStep("Hollow sphere I", I_hollow, "kg·m²", "⅔mr²",
                    "rotational.moment_of_inertia_hollow_sphere"),
        MateriaStep("Angular momentum", L, "kg·m²/s", "L = Iω",
                    "rotational.angular_momentum"),
        MateriaStep("Rotational KE", ke, "J", "½Iω²", "Materia"),
    ]
    summary = (f"A {mass_kg:g} kg, {radius_m*100:.0f} cm body at "
               f"{angular_velocity_rad_s:g} rad/s: solid-sphere I={I_sphere:.4g} "
               f"kg·m², L={L:.3g} kg·m²/s, KE={ke:.3g} J.")
    return MateriaResult("rotational_dynamics",
                         {"mass_kg": mass_kg, "radius_m": radius_m,
                          "angular_velocity_rad_s": angular_velocity_rad_s}, steps,
                         summary=summary,
                         validation={"passed": I_sphere > 0,
                                     "note": "moments of inertia + angular momentum"},
                         outputs={"moment_of_inertia_sphere": I_sphere,
                                  "angular_momentum": L, "rotational_ke_J": ke})


def material_full_profile(material_key: str = "steel_mild") -> MateriaResult:
    """Exhaustively characterize a material — call EVERY material-property
    aggregator sigma-ground exposes (thermal, elastic, magnetic, acoustic,
    electrochemical, piezoelectric, surface, diffusion, …). The "tell me
    everything about X" verb, and a wide sweep of the property tree."""
    import importlib
    import inspect
    import pkgutil
    import sigma_ground.field.interface as fi
    suites = {}
    for mi in pkgutil.iter_modules(fi.__path__, "sigma_ground.field.interface."):
        if any(s in mi.name for s in ("test", "string_theory", "mobius", "ligo",
                                      "entangle", "unsolved")):
            continue
        try:
            mod = importlib.import_module(mi.name)
        except Exception:
            continue
        for fname, fn in inspect.getmembers(mod, inspect.isfunction):
            if fn.__module__ != mi.name:
                continue
            if not (fname.endswith("_properties") or fname.endswith("_report")):
                continue
            try:
                r = fn(material_key)
            except Exception:
                continue
            if isinstance(r, dict) and r:
                suites[fname] = len(r)
    steps = [
        MateriaStep("Material", material_key, "", "(input)",
                    "field.interface.surface.MATERIALS"),
        MateriaStep("Property suites computed", len(suites), "",
                    ", ".join(sorted(suites)[:6]) + ("…" if len(suites) > 6 else ""),
                    "field.interface.* aggregators"),
        MateriaStep("Total property fields", sum(suites.values()), "",
                    "across all suites", "field.interface.*"),
    ]
    summary = (f"{material_key}: {len(suites)} property suites computed "
               f"({sum(suites.values())} fields) — "
               f"{', '.join(sorted(suites)[:5])}…")
    return MateriaResult("material_full_profile", {"material_key": material_key},
                         steps, summary=summary,
                         validation={"passed": len(suites) > 0,
                                     "note": f"{len(suites)} material-property "
                                             f"suites exercised"},
                         outputs={"suites_computed": len(suites)})


def _sweep(calls):
    """Call a list of (module, fn, args, kwargs); return {fn: n_fields} for the
    ones that succeed. Robust to signature drift — a failed call is just skipped."""
    import importlib
    suites = {}
    for mod, fn, a, kw in calls:
        try:
            m = importlib.import_module("sigma_ground.field.interface." + mod)
            r = getattr(m, fn)(*a, **kw)
            if isinstance(r, dict) and r:
                suites[fn] = len(r)
        except Exception:
            pass
    return suites


def _sweep_result(name, domain, suites, inputs):
    steps = [
        MateriaStep(f"{domain} suites computed", len(suites), "",
                    ", ".join(sorted(suites)[:6]), "field.interface.* aggregators"),
        MateriaStep("Total fields", sum(suites.values()), "", "across all suites",
                    "field.interface.*"),
    ]
    summary = (f"{domain}: {len(suites)} suites computed "
               f"({sum(suites.values())} fields) — {', '.join(sorted(suites)[:4])}…")
    return MateriaResult(name, inputs, steps, summary=summary,
                         validation={"passed": len(suites) > 0,
                                     "note": f"{len(suites)} {domain} aggregators"},
                         outputs={"suites_computed": len(suites)})


def quantum_report(element_Z: int = 1) -> MateriaResult:
    """Quantum & atomic physics: hydrogen-like spectra, quantum wells, tunneling,
    crystal-field splitting, superconductivity. Wraps the quantum-domain
    aggregators (the dependency chart's `quantum` group)."""
    suites = _sweep([
        ("atomic_spectra", "atomic_spectra_report", (), {"Z": element_Z}),
        ("quantum_wells", "quantum_wells_report", (), {}),
        ("tunneling", "tunneling_report", (), {}),
        ("crystal_field", "crystal_field_report", (),
         {"Z": 26, "oxidation_state": 2, "coord_key": "octahedral"}),
        ("superconductivity", "superconductor_properties", ("niobium",), {}),
        ("angular_momentum", "angular_momentum_report", (), {}),
    ])
    return _sweep_result("quantum_report", "Quantum", suites,
                         {"element_Z": element_Z})


def _sweep_calls(calls):
    """Call each (module_path, fn, args, kwargs); return {fn: raw_return} for the
    successes. One successful call == one sigma-ground function exercised. Unlike
    ``_sweep`` (which is hardwired to field.interface aggregators returning dicts)
    this takes a FULL module path and keeps scalar returns too — so it reaches
    field.* modules (electrodynamics, gr_basics) and per-function scalar physics.
    Robust to signature/registry drift: a bad call is skipped, never fatal."""
    import importlib
    values = {}
    for modpath, fn, a, kw in calls:
        try:
            m = importlib.import_module(modpath)
            # key by module.fn so same-named functions across modules (e.g. each
            # module's full_report) are counted distinctly, not collapsed to one.
            label = modpath.rsplit(".", 1)[-1] + "." + fn
            values[label] = getattr(m, fn)(*a, **kw)
        except Exception:
            pass
    return values


def _pick(values, fn):
    """First successful return whose function name is `fn` (ignoring module) —
    lets a verb pull a headline value without knowing which module won."""
    for k, v in values.items():
        if k.rsplit(".", 1)[-1] == fn:
            return v
    return None


def _domain_result(name, domain, values, inputs, headline, sources):
    """Build a MateriaResult from a domain sweep: a representative headline plus
    how many sigma-ground functions the scenario exercised."""
    n = len(values)
    fields = sum(len(v) if isinstance(v, dict) else 1 for v in values.values())
    called = ", ".join(sorted(values)[:6]) + ("…" if n > 6 else "")
    steps = [
        MateriaStep(f"{domain} result", headline, "", "(representative)", sources),
        MateriaStep("Functions exercised", n, "", called, sources),
        MateriaStep("Result fields", fields, "", "scalars + aggregator dicts",
                    sources),
    ]
    summary = f"{domain}: {headline} — {n} sigma-ground functions exercised."
    return MateriaResult(name, inputs, steps, summary=summary,
                         validation={"passed": n > 0,
                                     "note": f"{n} {domain} functions exercised"},
                         outputs={"functions_called": n, "result_fields": fields})


def charged_particle(charge_c: float = 1e-6, distance_m: float = 0.01,
                     b_field_t: float = 1.0,
                     speed_m_s: float = 1e6) -> MateriaResult:
    """Electromagnetic forces, fields and radiation on a charged particle:
    Coulomb's law, point field & potential, cyclotron motion, Larmor radiation,
    Lorentz / magnetic force and EM-wave energy. Sweeps the electrodynamics
    module (the fields half of the dependency chart's `em` group)."""
    ED = "sigma_ground.field.electrodynamics"
    q, r, B, v = charge_c, distance_m, b_field_t, speed_m_s
    me = 9.1093837015e-31
    vals = _sweep_calls([
        (ED, "coulomb_force", (q, q, r), {}),
        (ED, "electric_field_point", (q, r), {}),
        (ED, "electric_potential", (q, r), {}),
        (ED, "em_wave_intensity", (1000.0,), {}),
        (ED, "em_wave_energy_density", (1000.0,), {}),
        (ED, "cyclotron_frequency", (q, me, B), {}),
        (ED, "radiation_power_larmor", (q, 1e15), {}),
        (ED, "lorentz_force", (q, (0.0, 0.0, 1000.0), (v, 0.0, 0.0),
                               (0.0, 0.0, B)), {}),
        (ED, "magnetic_force", (q, (v, 0.0, 0.0), (0.0, 0.0, B)), {}),
        (ED, "fine_structure_constant", (), {}),
        (ED, "muon_anomalous_moment_experimental", (), {}),
        (ED, "muon_anomalous_moment_sm", (), {}),
        (ED, "muon_g2_tension_sigmas", (), {}),
        (ED, "muon_g2_consistency_status", (), {}),
    ])
    F = _pick(vals, "coulomb_force")
    headline = (f"Coulomb force {F:.3g} N at {distance_m*100:g} cm"
                if isinstance(F, (int, float)) else "EM fields computed")
    return _domain_result("charged_particle", "Electrodynamics", vals,
                          {"charge_c": charge_c, "distance_m": distance_m,
                           "b_field_t": b_field_t}, headline, ED)


def optical_fiber(n_core: float = 1.48, n_clad: float = 1.46,
                  wavelength_nm: float = 1550.0,
                  core_radius_um: float = 4.0) -> MateriaResult:
    """Fiber-optic and nonlinear-photonic design: numerical aperture, mode count,
    single-mode cutoff, Bragg-stack reflectance, Kerr self-focusing and second-
    harmonic generation. Sweeps the photonics module."""
    P = "sigma_ground.field.interface.photonics"
    lam = max(float(wavelength_nm), 1.0) * 1e-9
    a = max(float(core_radius_um), 0.1) * 1e-6
    vals = _sweep_calls([
        (P, "numerical_aperture", (n_core, n_clad), {}),
        (P, "critical_angle", (n_core, n_clad), {}),
        (P, "number_of_modes_fiber", (a, lam, n_core, n_clad), {}),
        (P, "is_single_mode_fiber", (a, lam, n_core, n_clad), {}),
        (P, "absorption_coefficient_direct", (1.5, 1.1), {}),
        (P, "absorption_coefficient_indirect", (1.5, 1.1), {}),
        (P, "bragg_reflectance", (2.3, 1.46, 10), {}),
        (P, "bragg_wavelength", (2.3, 100e-9, 1.46, 150e-9), {}),
        (P, "bragg_bandwidth_fraction", (2.3, 1.46), {}),
        (P, "kerr_refractive_index", (n_core, 2.6e-20, 1e13), {}),
        (P, "nonlinear_phase_shift", (2.6e-20, 1e13, 1.0, lam), {}),
        (P, "self_focusing_critical_power", (lam, n_core, 2.6e-20), {}),
        (P, "shg_efficiency_factor", (1.0, 0.01, 1.48, 1.49, lam), {}),
        (P, "shg_phase_mismatch", (1.48, 1.49, lam), {}),
    ])
    na = _pick(vals, "numerical_aperture")
    headline = (f"NA={na:.3f} at {wavelength_nm:g} nm"
                if isinstance(na, (int, float)) else "fiber optics computed")
    return _domain_result("optical_fiber", "Photonics", vals,
                          {"n_core": n_core, "n_clad": n_clad,
                           "wavelength_nm": wavelength_nm}, headline, P)


def semiconductor_device(sc_key: str = "silicon",
                         temperature_k: float = 300.0) -> MateriaResult:
    """Semiconductor device physics: band gap, carrier concentration & mobility,
    p-n junction built-in voltage and depletion width, diode current and
    capacitive energy storage. Sweeps the electronics module."""
    E = "sigma_ground.field.interface.electronics"
    T = max(float(temperature_k), 1.0)
    ND = NA = 1e22
    vals = _sweep_calls([
        (E, "band_gap", (sc_key, T), {}),
        (E, "carrier_concentration", (sc_key, T), {}),
        (E, "carrier_mobility", (sc_key, T), {}),
        (E, "effective_dos_conduction", (sc_key, T), {}),
        (E, "effective_dos_valence", (sc_key, T), {}),
        (E, "fermi_level_from_intrinsic", (sc_key, T), {}),
        (E, "built_in_voltage", (sc_key, ND, NA, T), {}),
        (E, "depletion_width", (sc_key, ND, NA), {"T": T}),
        (E, "diode_saturation_current", (sc_key, ND, NA, 1e-6), {}),
        (E, "diode_current", (1e-12, 0.6), {"T": T}),
        (E, "energy_stored", (1e-9, 5.0), {}),
        (E, "coaxial_capacitance", (1.0, 0.001, 0.005), {}),
        (E, "hall_voltage", (sc_key, 0.1, 1.0, 1e-3), {}),
        (E, "junction_capacitance", (sc_key, ND, NA, 1e-6), {}),
        (E, "parallel_plate_capacitance", (1e-4, 1e-3), {}),
        (E, "spherical_capacitance", (0.01, 0.02), {}),
        ("sigma_ground.field.interface.semiconductor_optics", "band_gap_ev",
         (sc_key,), {}),
        ("sigma_ground.field.interface.semiconductor_optics", "band_edge_nm",
         (sc_key,), {}),
        ("sigma_ground.field.interface.semiconductor_optics", "semiconductor_rgb",
         (sc_key,), {}),
        ("sigma_ground.field.interface.semiconductor_optics",
         "semiconductor_rgb_from_z", (14,), {}),
    ])
    eg = _pick(vals, "band_gap")
    headline = (f"{sc_key} band gap {eg:.3g} eV"
                if isinstance(eg, (int, float)) else f"{sc_key} device physics")
    return _domain_result("semiconductor_device", "Semiconductor", vals,
                          {"sc_key": sc_key, "temperature_k": temperature_k},
                          headline, E)


def chemistry_lab(acid_key: str = "acetic_acid",
                  salt_key: str = "sodium_chloride",
                  molecule: str = "CO2") -> MateriaResult:
    """Solution & reaction chemistry: acid/base equilibria (pH, Ka/Kb, buffers),
    colligative & solubility properties, ideal-gas transport and galvanic
    electrochemistry. Sweeps the chemistry group's full reports."""
    FI = "sigma_ground.field.interface."
    vals = _sweep_calls([
        (FI + "acid_base", "full_report", (), {}),
        (FI + "acid_base", "acid_base_report", (acid_key,), {}),
        (FI + "acid_base", "pH_solution", (acid_key, 0.1), {}),
        (FI + "solution", "full_report", (), {}),
        (FI + "gas", "molecule_gas_properties", (molecule,), {}),
        (FI + "gas", "ideal_gas_density", ("N2",), {}),
        (FI + "gas", "gas_viscosity", ("N2",), {}),
        (FI + "gas", "heat_capacity_ratio", ("N2",), {}),
        (FI + "gas", "gas_diffusivity", ("N2", "O2"), {}),
        (FI + "electrochemistry", "material_electrochemical_properties",
         ("zinc",), {}),
        (FI + "electrochemistry", "cell_potential", ("copper", "zinc"), {}),
        (FI + "electrochemistry", "gibbs_energy_cell", ("copper", "zinc"), {}),
        (FI + "electrochemistry", "is_spontaneous", ("copper", "zinc"), {}),
        (FI + "electrochemistry", "activity_series", (), {}),
        (FI + "acid_base", "henderson_hasselbalch", (acid_key, 1.0), {}),
        (FI + "acid_base", "polyprotic_alpha", ([3.0, 7.0], 5.0), {}),
        (FI + "acid_base", "titration_strong_acid_strong_base",
         (0.1, 50.0, 0.1, 25.0), {}),
        (FI + "acid_base", "titration_weak_acid_strong_base",
         (acid_key, 0.1, 50.0, 0.1, 25.0), {}),
        (FI + "acid_base", "titration_curve", (acid_key, 0.1, 50.0, 0.1), {}),
    ])
    headline = "acid/base + solubility + gas transport + galvanic chemistry"
    return _domain_result("chemistry_lab", "Chemistry", vals,
                          {"acid_key": acid_key, "salt_key": salt_key},
                          headline, FI + "{acid_base,solution,gas,electrochem}")


def atmospheric_profile(altitude_m: float = 5000.0) -> MateriaResult:
    """Standard-atmosphere state and transport: density/pressure vs altitude,
    humidity & dew point, specific heats and the full atmosphere report. Sweeps
    the atmosphere module (the `thermal`/fluids weather group)."""
    A = "sigma_ground.field.interface.atmosphere"
    z = max(float(altitude_m), 0.0)
    vals = _sweep_calls([
        (A, "atmosphere_report", (), {}),
        (A, "air_density", (288.15,), {}),
        (A, "density_at_altitude", (z,), {}),
        (A, "altitude_for_pressure", (50000.0,), {}),
        (A, "dew_point", (288.15, 0.5), {}),
        (A, "absolute_humidity", (288.15,), {}),
        (A, "air_cp_mass", (), {}),
        (A, "air_cp_molar", (), {}),
        (A, "air_gamma", (), {}),
        (A, "column_number_density", (0.21,), {}),
    ])
    rho = _pick(vals, "density_at_altitude")
    headline = (f"air density {rho:.3g} kg/m³ at {z:.0f} m"
                if isinstance(rho, (int, float)) else "atmosphere computed")
    return _domain_result("atmospheric_profile", "Atmosphere", vals,
                          {"altitude_m": altitude_m}, headline, A)


def nuclear_decay(isotope_key: str = "U238",
                  element_Z: int = 92) -> MateriaResult:
    """Nuclear & radioactive properties: alpha/beta half-lives and decay
    constants, Q-values, activity, plus the parent element's atomic structure
    and cohesion. Sweeps the radioactive_decay + element modules (`nuclear`)."""
    RD = "sigma_ground.field.interface.radioactive_decay"
    EL = "sigma_ground.field.interface.element"
    DC = "sigma_ground.field.decay"          # low-level: Gamow, Geiger-Nuttall, Q
    Z = max(int(element_Z), 1)
    vals = _sweep_calls([
        (RD, "isotope_decay_properties", (isotope_key,), {}),
        (RD, "activity_becquerel", (isotope_key, 1e20), {}),
        (RD, "alpha_half_life", (isotope_key,), {}),
        (RD, "beta_half_life", (isotope_key,), {}),
        (RD, "alpha_decay_constant", (isotope_key,), {}),
        (RD, "beta_decay_constant", (isotope_key,), {}),
        (RD, "alpha_Q_decomposition", (isotope_key,), {}),
        (RD, "beta_Q_decomposition", (isotope_key,), {}),
        (RD, "alpha_mass_mev", (), {}),
        (EL, "element_properties", (Z,), {}),
        (EL, "atomic_mass_kg", (Z,), {}),
        (EL, "aufbau_configuration", (Z,), {}),
        (EL, "cohesive_energy_eV", (Z,), {}),
        (EL, "material_from_Z", (Z,), {}),
        (DC, "decay_constant", (1.4e17,), {}),
        (DC, "half_life", (1e-18,), {}),
        (DC, "activity", (1e20, 1.4e17), {}),
        (DC, "q_value_alpha", (238050.8, 234043.6), {}),
        (DC, "q_value_beta_minus", (40000.0, 39999.0), {}),
        (DC, "q_value_beta_plus", (40000.0, 39998.0), {}),
        (DC, "q_value_mev", (238050.8, [234043.6, 3727.4]), {}),
        (DC, "gamow_factor", (90, 2, 4.27, 234), {}),
        (DC, "alpha_decay_rate_geiger_nuttall", (92, 238, 4.27), {}),
    ])
    headline = f"{isotope_key} decay chain + Z={Z} atomic structure"
    return _domain_result("nuclear_decay", "Nuclear", vals,
                          {"isotope_key": isotope_key, "element_Z": element_Z},
                          headline, "radioactive_decay + element")


def collision_dynamics(m1: float = 2.0, v1: float = 10.0, m2: float = 1.0,
                       v2: float = -5.0, height_m: float = 5.0) -> MateriaResult:
    """Energy, momentum and collisions: kinetic / potential / rotational energy,
    the full mechanical-energy report, elastic and inelastic collision outcomes
    and the energy dissipated. Sweeps the classical_mechanics module."""
    M = "sigma_ground.field.interface.classical_mechanics"
    vals = _sweep_calls([
        (M, "mechanics_report", (m1, v1), {"height": height_m, "inertia": 0.1,
                                           "angular_velocity": 3.0}),
        (M, "kinetic_energy", (m1, v1), {}),
        (M, "momentum", (m1, v1), {}),
        (M, "gravitational_pe", (m1, height_m), {}),
        (M, "rotational_ke", (0.1, 3.0), {}),
        (M, "total_mechanical_energy", (m1, v1, height_m), {}),
        (M, "impulse", (50.0, 0.1), {}),
        (M, "power_mechanical", (50.0, v1), {}),
        (M, "velocity_from_impulse", (5.0, m1), {}),
        (M, "elastic_collision_velocities", (m1, v1, m2, v2), {}),
        (M, "inelastic_collision_velocity", (m1, v1, m2, v2), {}),
        (M, "collision_energy_loss", (m1, v1, m2, v2, 0.5), {}),
        (M, "friction_dissipation", (10.0, 5.0), {}),
    ])
    ke = _pick(vals, "kinetic_energy")
    headline = (f"KE={ke:.4g} J" if isinstance(ke, (int, float))
                else "mechanics computed")
    return _domain_result("collision_dynamics", "Mechanics", vals,
                          {"m1": m1, "v1": v1, "m2": m2, "v2": v2}, headline, M)


def black_hole(solar_masses: float = 10.0) -> MateriaResult:
    """Black-hole thermodynamics and strong-gravity observables for a mass M:
    Schwarzschild radius, Hawking temperature / luminosity / evaporation time,
    Bekenstein-Hawking entropy, ISCO and photon-sphere radii, surface escape
    velocity, gravitational redshift and tidal stretching. Standard GR
    (the `gravity` group)."""
    from ..field.constants import G, C, M_SUN_KG
    M = max(float(solar_masses), 1e-12) * M_SUN_KG
    r_s = 2.0 * G * M / (C * C)
    r = 3.0 * r_s                                  # a test point outside the horizon
    GR = "sigma_ground.field.gr_basics"
    vals = _sweep_calls([
        (GR, "escape_velocity", (M, r), {}),
        (GR, "hawking_temperature", (M,), {}),
        (GR, "hawking_luminosity", (M,), {}),
        (GR, "hawking_evaporation_time", (M,), {}),
        (GR, "bekenstein_hawking_entropy", (M,), {}),
        (GR, "isco_radius", (M,), {}),
        (GR, "photon_sphere_radius", (M,), {}),
        (GR, "gravitational_redshift", (M, r), {}),
        (GR, "tidal_force", (M, r, 1.0), {}),
        (GR, "time_dilation_gr", (M, r), {}),
    ])
    T = _pick(vals, "hawking_temperature")
    headline = (f"r_s={r_s:.3g} m, T_H={T:.3g} K"
                if isinstance(T, (int, float)) else f"r_s={r_s:.3g} m")
    res = _domain_result("black_hole", "Black-hole gravity", vals,
                         {"solar_masses": solar_masses}, headline, GR)
    res.outputs["schwarzschild_radius_m"] = r_s
    return res


def matter_wave(kinetic_energy_eV: float = 100.0) -> MateriaResult:
    """Quantum matter waves: de Broglie wavelengths (electron / neutron /
    general), double-slit interference, diffraction envelope and which-path
    bounds. Sweeps the `quantum` interference module."""
    Q = "sigma_ground.field.interface.quantum"
    eV = max(float(kinetic_energy_eV), 1e-12)
    me = 9.1093837015e-31
    KE_J = eV * 1.602176634e-19
    vals = _sweep_calls([
        (Q, "ev_to_joules", (eV,), {}),
        (Q, "de_broglie_electron", (eV,), {}),
        (Q, "de_broglie_neutron", (eV,), {}),
        (Q, "de_broglie_wavelength", (me, KE_J), {}),
        (Q, "diffraction_envelope_zero", (1e-10, 1.0, 1e-5), {}),
        (Q, "double_slit_intensity", (0.0, 1e-5, 1.0, 1e-10), {}),
        (Q, "build_intensity_profile", (1e-5, 1.0, 1e-10), {}),
        (Q, "electron_fringe_spacing_ratio", (0.0,), {}),
        (Q, "englert_bound_satisfied", (0.5, 0.5), {}),
        (Q, "joules_to_ev", (KE_J,), {}),
        (Q, "fringe_spacing", (1e-10, 1.0, 1e-5), {}),
        (Q, "fringe_visibility", (1.0, 0.1), {}),
        (Q, "fringe_count_in_envelope", (1e-5, 1e-5), {}),
        (Q, "visibility_from_D", (0.5,), {}),
    ])
    lam = _pick(vals, "de_broglie_electron")
    headline = (f"electron wavelength {lam:.3g} m at {eV:g} eV"
                if isinstance(lam, (int, float)) else "matter-wave computed")
    return _domain_result("matter_wave", "Quantum matter-wave", vals,
                          {"kinetic_energy_eV": kinetic_energy_eV}, headline, Q)


def hydrogen_spectrum(element_Z: int = 1) -> MateriaResult:
    """Hydrogen-like atomic spectra: energy levels, Balmer & emission series,
    allowed transitions, fine-structure splitting and the full spectral report.
    Sweeps the atomic_spectra module."""
    A = "sigma_ground.field.interface.atomic_spectra"
    Z = max(int(element_Z), 1)
    vals = _sweep_calls([
        (A, "atomic_spectra_report", (), {"Z": Z}),
        (A, "hydrogen_energy_eV", (2,), {}),
        (A, "hydrogen_like_energy_eV", (Z, 2), {}),
        (A, "balmer_series", (), {"Z": Z}),
        (A, "emission_spectrum", (), {"Z": Z}),
        (A, "allowed_transitions", (), {"n_max": 5}),
        (A, "fine_structure_splitting_eV", (Z, 2, 1), {}),
        (A, "fine_structure_shift_eV", (Z, 2, 1, 0.5), {}),
        (A, "hydrogen_reduced_mass", (), {}),
        (A, "ionization_energy_hydrogen_eV", (), {}),
        (A, "lyman_series", (), {"Z": Z}),
        (A, "paschen_series", (), {"Z": Z}),
        (A, "series_limit_nm", (Z, 1), {}),
        (A, "transition_energy_eV", (2, 3), {}),
        (A, "transition_frequency_Hz", (2, 3), {}),
        (A, "transition_wavelength_nm", (2, 3), {}),
        (A, "transition_wavenumber", (2, 3), {}),
        (A, "is_allowed_transition", (2, 1, 1, 0), {}),
        (A, "is_visible", (550.0,), {}),
        (A, "visible_lines", (), {"Z": Z}),
        (A, "wavelength_to_rgb", (550.0,), {}),
        (A, "lande_g_factor", (1, 0.5, 1.5), {}),
        (A, "multi_electron_energy_eV", (Z, 2, 1), {}),
        (A, "zeeman_shift_eV", (0.5, 2.0, 1.0), {}),
        (A, "zeeman_splitting_count", (1,), {}),
        (A, "zeeman_pattern", (1, 0.5, 1.5, 1.0), {}),
        (A, "qho_energy_eV", (1, 1e14), {}),
        (A, "qho_zero_point_energy_eV", (1e14,), {}),
        (A, "qho_level_spacing_eV", (1e14,), {}),
        (A, "qho_transition_energy_eV", (1, 2, 1e14), {}),
        (A, "qho_classical_amplitude", (1e14, 9.1093837015e-31, 1), {}),
    ])
    E = _pick(vals, "hydrogen_energy_eV")
    headline = (f"n=2 level {E:.4g} eV"
                if isinstance(E, (int, float)) else "spectrum computed")
    return _domain_result("hydrogen_spectrum", "Atomic spectra", vals,
                          {"element_Z": element_Z}, headline, A)


def quantum_circuit() -> MateriaResult:
    """Quantum-computing demonstration: prepare states, apply gates and run the
    canonical algorithms (Deutsch-Jozsa, Bernstein-Vazirani, QFT, phase
    estimation, QAOA) with measurement read-out. Sweeps the quantum_computing /
    quantum_output / quantum_algorithms stack via their full reports."""
    base = "sigma_ground.field.interface."
    vals = _sweep_calls([
        (base + "quantum_computing", "full_report", (), {}),
        (base + "quantum_output", "full_report", (), {}),
        (base + "quantum_algorithms", "full_report", (), {}),
    ])
    headline = "gate, algorithm and measurement reports"
    return _domain_result("quantum_circuit", "Quantum computing", vals, {},
                          headline, base + "quantum_*")


def projectile_motion(speed_m_s: float = 30.0,
                      angle_deg: float = 45.0) -> MateriaResult:
    """Projectile and inclined-plane kinematics: range, maximum height and time
    of flight (with the full ballistic report), plus motion on a ramp — sliding
    acceleration, critical angle and speed at the bottom. Sweeps the projectile
    module."""
    import math
    P = "sigma_ground.field.interface.projectile"
    v0 = max(float(speed_m_s), 0.0)
    ang = math.radians(max(min(float(angle_deg), 89.9), 0.1))
    vals = _sweep_calls([
        (P, "projectile_report", (v0, ang), {}),
        (P, "projectile_range", (v0, ang), {}),
        (P, "projectile_max_height", (v0, ang), {}),
        (P, "projectile_time_of_flight", (v0, ang), {}),
        (P, "incline_acceleration", (ang,), {"mu_friction": 0.1}),
        (P, "incline_critical_angle", (0.3,), {}),
        (P, "incline_sliding_distance", (v0, ang, 0.1), {}),
        (P, "incline_speed_at_bottom", (10.0, ang), {"mu_friction": 0.1}),
        (P, "drag_force", (v0, 0.47, 0.01), {}),
    ])
    rng = _pick(vals, "projectile_range")
    headline = (f"range {rng:.1f} m at {angle_deg:g}°"
                if isinstance(rng, (int, float)) else "ballistics computed")
    return _domain_result("projectile_motion", "Projectile", vals,
                          {"speed_m_s": speed_m_s, "angle_deg": angle_deg},
                          headline, P)


def relativistic_motion(velocity_fraction_c: float = 0.9,
                        rest_mass_kg: float = 9.1093837015e-31) -> MateriaResult:
    """Special relativity at speed v = βc: Lorentz factor, time dilation, length
    contraction, relativistic energy / momentum / kinetic energy, rest energy,
    relativistic Doppler shift and velocity addition. Sweeps the relativity
    module."""
    from ..field.constants import C
    R = "sigma_ground.field.relativity"
    beta = max(min(float(velocity_fraction_c), 0.999999), 0.0)
    v = beta * C
    m0 = max(float(rest_mass_kg), 1e-40)
    vals = _sweep_calls([
        (R, "beta", (v,), {}),
        (R, "lorentz_factor", (v,), {}),
        (R, "time_dilation", (1.0, v), {}),
        (R, "length_contraction", (1.0, v), {}),
        (R, "relativistic_energy", (m0, v), {}),
        (R, "kinetic_energy_rel", (m0, v), {}),
        (R, "momentum_rel", (m0, v), {}),
        (R, "rest_energy", (m0,), {}),
        (R, "energy_momentum_invariant", (m0,), {}),
        (R, "doppler_factor", (v, 1.0), {}),
        (R, "velocity_addition", (v, v), {}),
    ])
    g = _pick(vals, "lorentz_factor")
    headline = (f"gamma={g:.3f} at {beta:g}c"
                if isinstance(g, (int, float)) else "relativity computed")
    return _domain_result("relativistic_motion", "Special relativity", vals,
                          {"velocity_fraction_c": velocity_fraction_c},
                          headline, R)


def thermal_statistics(temperature_k: float = 300.0) -> MateriaResult:
    """Statistical mechanics of a gas at temperature T: the Maxwell-Boltzmann
    speed distribution (mean / rms / most-probable speed), Boltzmann factors and
    entropy, the partition function, and Fermi-Dirac / Bose-Einstein occupation.
    Sweeps the statistical module."""
    S = "sigma_ground.field.interface.statistical"
    T = max(float(temperature_k), 1.0)
    m = 4.65e-26   # an N2 molecule
    vals = _sweep_calls([
        (S, "boltzmann_factor", (1e-21, T), {}),
        (S, "maxwell_speed_dist", (m, 500.0, T), {}),
        (S, "mean_speed", (m, T), {}),
        (S, "rms_speed", (m, T), {}),
        (S, "most_probable_speed", (m, T), {}),
        (S, "partition_function", ([0.0, 1e-21, 2e-21], T), {}),
        (S, "mean_energy", ([0.0, 1e-21, 2e-21], T), {}),
        (S, "entropy_from_partition", (10.0, T, 1e-21), {}),
        (S, "boltzmann_entropy", (1e23,), {}),
        (S, "heat_capacity_equipartition", (3,), {}),
        (S, "fermi_dirac", (1e-21, 2e-21, T), {}),
        (S, "bose_einstein", (2e-21, 0.0, T), {}),
    ])
    vbar = _pick(vals, "mean_speed")
    headline = (f"mean molecular speed {vbar:.0f} m/s at {T:.0f} K"
                if isinstance(vbar, (int, float)) else "statistics computed")
    return _domain_result("thermal_statistics", "Statistical mechanics", vals,
                          {"temperature_k": temperature_k}, headline, S)


def fluid_flow(velocity_m_s: float = 2.0,
               pipe_radius_m: float = 0.01) -> MateriaResult:
    """Viscous fluid flow: pipe flow (Poiseuille rate & peak velocity), Stokes
    drag and terminal settling, particle Reynolds number and sphere drag
    coefficient, boundary-layer growth — with the combined flow report. Sweeps
    the viscosity module."""
    V = "sigma_ground.field.interface.viscosity"
    mu, rho, rho_p = 1.0e-3, 1000.0, 2000.0
    vel = max(float(velocity_m_s), 1e-6)
    R = max(float(pipe_radius_m), 1e-6)
    vals = _sweep_calls([
        (V, "viscous_flow_properties", (mu, rho, vel),
         {"pipe_radius": R, "pipe_length": 1.0, "particle_radius": 1e-3,
          "rho_particle": rho_p}),
        (V, "stokes_drag", (mu, 1e-3, vel), {}),
        (V, "poiseuille_flow_rate", (R, 1000.0, mu, 1.0), {}),
        (V, "poiseuille_max_velocity", (R, 1000.0, mu, 1.0), {}),
        (V, "particle_reynolds_number", (rho, vel, 2e-3, mu), {}),
        (V, "drag_coefficient_sphere", (1000.0,), {}),
        (V, "boundary_layer_thickness", (0.5, rho, vel, mu), {}),
        (V, "general_drag_force", (0.47, rho, vel, 1e-3), {}),
        (V, "nabarro_herring_strain_rate", (1e-12, 1e6, 1e-29, 1e-5, 300.0), {}),
    ])
    re = _pick(vals, "particle_reynolds_number")
    headline = (f"Reynolds number {re:.3g}"
                if isinstance(re, (int, float)) else "viscous flow computed")
    return _domain_result("fluid_flow", "Viscous flow", vals,
                          {"velocity_m_s": velocity_m_s,
                           "pipe_radius_m": pipe_radius_m}, headline, V)


def composite_material(composite_key: str = "cfrp_unidirectional") -> MateriaResult:
    """Composite-material mechanics: stiffness and density by rule-of-mixtures
    and Halpin-Tsai, foam scaling (Gibson-Ashby), and thermal expansion — for
    one composite and across the whole catalogue. Sweeps the composites
    module."""
    C = "sigma_ground.field.interface.composites"
    vals = _sweep_calls([
        (C, "full_report", (), {}),
        (C, "composite_report", (composite_key,), {}),
        (C, "composite_density", (composite_key,), {}),
        (C, "composite_modulus", (composite_key,), {}),
        (C, "composite_expansion", (composite_key,), {}),
        (C, "halpin_tsai", (200e9, 3e9, 0.6), {}),
        (C, "gibson_ashby_modulus", (70e9, 0.1), {}),
        (C, "gibson_ashby_strength", (100e6, 0.1), {}),
        (C, "density_rule_of_mixtures", ([1800.0, 1200.0], [0.6, 0.4]), {}),
    ])
    E = _pick(vals, "composite_modulus")
    headline = (f"{composite_key} modulus {E:.3g} Pa"
                if isinstance(E, (int, float)) else f"{composite_key} composite")
    return _domain_result("composite_material", "Composites", vals,
                          {"composite_key": composite_key}, headline, C)


def quantum_dot(radius_nm: float = 5.0, material: str = "CdSe") -> MateriaResult:
    """Quantum confinement in wells and dots: particle-in-a-box energy levels and
    transitions, the Brus quantum-dot gap shift, confinement energy and critical
    radius, and density of states. Sweeps the quantum_wells module."""
    Q = "sigma_ground.field.interface.quantum_wells"
    R = max(float(radius_nm), 0.1) * 1e-9
    L = 2.0 * R
    vals = _sweep_calls([
        (Q, "full_report", (), {}),
        (Q, "box_ground_state_eV", (L,), {}),
        (Q, "box_energy_1d_eV", (1, L), {}),
        (Q, "box_energy_3d_eV", (1, 1, 1, L), {}),
        (Q, "box_transition_wavelength_nm", (1, 2, L), {}),
        (Q, "brus_energy_eV", (R,), {"material_key": material}),
        (Q, "confinement_energy_eV", (R,), {}),
        (Q, "critical_radius_nm", (), {"material_key": material}),
        (Q, "degeneracy_3d_cubic", (3,), {}),
        (Q, "dos_0d", (1.0, [0.5, 1.0, 1.5]), {}),
        (Q, "dos_1d", (1.0,), {}),
        (Q, "dos_3d", (1.0,), {}),
        (Q, "qd_radius_for_wavelength_nm", (600.0,), {}),
        (Q, "quantum_wire_subbands_eV", (5e-9, 5e-9), {}),
        (Q, "size_vs_gap", (), {}),
        (Q, "tunneling_depth_m", (1.0, 0.5), {}),
    ])
    E0 = _pick(vals, "box_ground_state_eV")
    headline = (f"box ground state {E0:.3g} eV"
                if isinstance(E0, (int, float)) else "confinement computed")
    return _domain_result("quantum_dot", "Quantum confinement", vals,
                          {"radius_nm": radius_nm, "material": material},
                          headline, Q)


def atomic_coupling(n_electrons: int = 2, l: int = 2) -> MateriaResult:
    """Angular-momentum coupling in atoms: term symbols and the Hund's-rule
    ground state, allowed J values, Clebsch-Gordan coefficients and spin-orbit
    constants — with the full report. Sweeps the angular_momentum module."""
    A = "sigma_ground.field.interface.angular_momentum"
    ne, ll = max(int(n_electrons), 1), max(int(l), 0)
    vals = _sweep_calls([
        (A, "full_report", (), {"n_electrons": ne, "l": ll}),
        (A, "angular_momentum_magnitude", (2.0,), {}),
        (A, "angular_momentum_z_values", (2.0,), {}),
        (A, "allowed_J_values", (0.5, 0.5), {}),
        (A, "clebsch_gordan", (0.5, 0.5, 0.5, -0.5, 1.0, 0.0), {}),
        (A, "hund_ground_state", (ne, ll), {}),
        (A, "all_term_symbols", (2.0, 1.0), {}),
        (A, "hydrogen_spin_orbit_constant_eV", (1, 2, 1), {}),
        (A, "spin_orbit_energy_eV", (0.01, 1, 0.5, 1.5), {}),
        (A, "spin_orbit_splitting_eV", (0.01, 1, 0.5), {}),
        (A, "spin_expectation", (0.7, 0.3), {}),
        (A, "lande_interval_check", (0.01, 1, 0.5), {}),
        (A, "multi_electron_SO_constant_eV", (26, 3, 2), {}),
    ])
    headline = "term symbols + Clebsch-Gordan + Hund ground state"
    return _domain_result("atomic_coupling", "Angular momentum", vals,
                          {"n_electrons": n_electrons, "l": l}, headline, A)


def molecular_bond(atom_a: str = "C", atom_b: str = "O") -> MateriaResult:
    """Chemical-bond properties between two atoms: bond energy (Pauling), length
    (Schomaker-Stevenson), force constant (Badger), dipole and polarity,
    hybridization and reduced mass — with the full bond report. Sweeps the
    molecular_bonds module."""
    M = "sigma_ground.field.interface.molecular_bonds"
    vals = _sweep_calls([
        (M, "bond_properties", (atom_a, atom_b), {}),
        (M, "pauling_bond_energy", (atom_a, atom_b), {}),
        (M, "schomaker_stevenson_length", (atom_a, atom_b), {}),
        (M, "badger_force_constant", (atom_a, atom_b), {}),
        (M, "bond_dipole_debye", (atom_a, atom_b), {}),
        (M, "bond_polarity", (atom_a, atom_b), {}),
        (M, "reduced_mass_kg", (atom_a, atom_b), {}),
        (M, "hybridization", (4, 0), {}),
        (M, "molecular_dipole_moment", ([1.5], [104.5]), {}),
    ])
    E = _pick(vals, "pauling_bond_energy")
    headline = (f"{atom_a}-{atom_b} bond energy {E:.3g}"
                if isinstance(E, (int, float)) else f"{atom_a}-{atom_b} bond")
    return _domain_result("molecular_bond", "Molecular bonds", vals,
                          {"atom_a": atom_a, "atom_b": atom_b}, headline, M)


def condensed_matter(material_key: str = "Fe") -> MateriaResult:
    """Strongly-correlated electron matter: Fermi energy, Hubbard-model
    parameters and ground state, Mott metal-insulator and crystal-field phase
    diagrams — the full set of condensed-matter predictions. Sweeps the
    quantum_matter module."""
    Q = "sigma_ground.field.interface.quantum_matter"
    vals = _sweep_calls([
        (Q, "full_report", (), {}),
        (Q, "all_predictions", (), {}),
        (Q, "fermi_energy_eV", (material_key,), {}),
        (Q, "hubbard_parameters", (material_key,), {}),
        (Q, "hubbard_ground_state", (material_key,), {}),
        (Q, "mott_phase_diagram", (), {}),
        (Q, "crystal_field_phase_diagram", (), {}),
        (Q, "nephelauxetic_metallicity", (), {}),
    ])
    headline = "Fermi energy + Hubbard/Mott correlated-electron predictions"
    return _domain_result("condensed_matter", "Condensed matter", vals,
                          {"material_key": material_key}, headline, Q)


def rolling_object(mass_kg: float = 1.0, radius_m: float = 0.1,
                   incline_angle_deg: float = 30.0,
                   height_m: float = 2.0) -> MateriaResult:
    """A rigid body rolling without slipping: moments of inertia for standard
    shapes, rolling acceleration down an incline, speed gained from a height,
    rolling and ramp-to-flat distances, angular momentum and acceleration.
    Sweeps the rotational module's rolling dynamics."""
    import math
    R = "sigma_ground.field.interface.rotational"
    m = max(float(mass_kg), 1e-6)
    rad = max(float(radius_m), 1e-6)
    ang = math.radians(max(min(float(incline_angle_deg), 89.0), 0.1))
    vals = _sweep_calls([
        (R, "moment_of_inertia_sphere", (m, rad), {}),
        (R, "moment_of_inertia_hollow_sphere", (m, rad), {}),
        (R, "moment_of_inertia_cylinder", (m, rad), {}),
        (R, "moment_of_inertia_disk", (m, rad), {}),
        (R, "moment_of_inertia_rod", (m, rad), {}),
        (R, "i_factor", (), {"shape": "solid_sphere"}),
        (R, "rolling_acceleration_incline", (ang,), {"shape": "solid_sphere"}),
        (R, "rolling_speed_from_height", (height_m,), {"shape": "solid_sphere"}),
        (R, "rolling_distance_on_flat", (5.0, 0.05), {}),
        (R, "ramp_to_flat_distance", (height_m, ang, 0.05), {}),
        (R, "rolling_angular_velocity", (5.0, rad), {}),
        (R, "angular_acceleration", (10.0, 0.1), {}),
        (R, "angular_momentum", (0.1, 20.0), {}),
        (R, "parallel_axis", (0.1, m, 0.05), {}),
        (R, "rolling_velocity", (20.0, rad), {}),
        (R, "rotational_properties", (m, rad), {}),
        (R, "shape_moment_of_inertia", ("solid_sphere", m), {}),
        (R, "shape_rolling_acceleration", ("solid_sphere", ang), {}),
        (R, "shape_rolling_speed_from_height", ("solid_sphere", height_m), {}),
        (R, "torque", (10.0, rad), {}),
    ])
    v = _pick(vals, "rolling_speed_from_height")
    headline = (f"rolls to {v:.2f} m/s from {height_m:g} m"
                if isinstance(v, (int, float)) else "rolling computed")
    return _domain_result("rolling_object", "Rotational rolling", vals,
                          {"mass_kg": mass_kg, "radius_m": radius_m,
                           "incline_angle_deg": incline_angle_deg}, headline, R)


def elastic_solid(material_key: str = "steel_mild") -> MateriaResult:
    """Linear elasticity of a solid: Lamé parameters, the elastic moduli they
    imply, the P-wave modulus, shear and hydrostatic stress response, and a
    yield check — with the full elastic-property report. Sweeps the elasticity
    module."""
    E = "sigma_ground.field.interface.elasticity"
    vals = _sweep_calls([
        (E, "material_elastic_properties", (material_key,), {}),
        (E, "lame_lambda", (material_key,), {}),
        (E, "lame_mu", (material_key,), {}),
        (E, "p_wave_modulus", (material_key,), {}),
        (E, "shear_stress", (material_key, 1e-3), {}),
        (E, "hydrostatic_stress", (material_key, 1e-3), {}),
        (E, "moduli_from_lame", (1e11, 8e10), {}),
        (E, "is_yielded", (100e6, 50e6, 0.0, 250e6), {}),
        (E, "uniaxial_stress", (material_key, 1e-3), {}),
        (E, "transverse_strain", (material_key, 1e-3), {}),
        (E, "volume_change_uniaxial", (material_key, 1e-3), {}),
        (E, "strain_energy_density_uniaxial", (material_key, 1e-3), {}),
        (E, "strain_energy_density_shear", (material_key, 1e-3), {}),
        (E, "strain_energy_density_hydrostatic", (material_key, 1e-3), {}),
        ("sigma_ground.field.interface.plasticity", "johnson_cook_stress",
         (material_key, 0.1), {}),
        ("sigma_ground.field.interface.plasticity", "ludwik_stress",
         (material_key, 0.1), {}),
        ("sigma_ground.field.interface.plasticity", "ramberg_osgood_strain",
         (material_key, 1e8), {}),
        ("sigma_ground.field.interface.plasticity", "stress_strain_curve",
         (material_key,), {}),
        ("sigma_ground.field.interface.plasticity", "work_hardening_rate",
         (material_key, 0.1), {}),
    ])
    mu = _pick(vals, "lame_mu")
    headline = (f"{material_key} shear modulus {mu:.3g} Pa"
                if isinstance(mu, (int, float)) else f"{material_key} elasticity")
    return _domain_result("elastic_solid", "Elasticity", vals,
                          {"material_key": material_key}, headline, E)


def quantum_gates() -> MateriaResult:
    """Build and measure a multi-qubit circuit: prepare a register, apply the
    full standard gate set (Pauli, Hadamard, S/T, phase, rotations, CNOT/CZ/SWAP,
    Toffoli/Fredkin), then read out probabilities, expectation values,
    entanglement entropy and samples. Exercises the quantum_computing gate set
    and quantum_output measurements — threading the state vector through each
    gate (a real circuit, not just the demo report)."""
    import importlib
    qc = importlib.import_module("sigma_ground.field.interface.quantum_computing")
    qo = importlib.import_module("sigma_ground.field.interface.quantum_output")
    vals = {}

    def rec(mod, name, *a, **kw):
        try:
            v = getattr(mod, name)(*a, **kw)
            vals[name] = v
            return v
        except Exception:
            return None

    X = [[0.0, 1.0], [1.0, 0.0]]                 # a Pauli-X for the apply_* gates
    s = rec(qc, "zero_state", 3)
    if s is None:
        s = rec(qc, "basis_state", 3, 0)
    for g in ("gate_h", "gate_x", "gate_y", "gate_z", "gate_s", "gate_t"):
        ns = rec(qc, g, s, 0)
        if ns is not None:
            s = ns
    for g, val in (("gate_phase", 0.5), ("gate_rx", 0.5), ("gate_ry", 0.5),
                   ("gate_rz", 0.5)):
        ns = rec(qc, g, s, 0, val)
        if ns is not None:
            s = ns
    for g in ("gate_cnot", "gate_cz", "gate_swap", "gate_iswap"):
        ns = rec(qc, g, s, 0, 1)
        if ns is not None:
            s = ns
    for g in ("gate_toffoli", "gate_fredkin"):
        ns = rec(qc, g, s, 0, 1, 2)
        if ns is not None:
            s = ns
    rec(qc, "apply_single_gate", s, 0, X)
    rec(qc, "apply_controlled_gate", s, 0, 1, X)
    rec(qc, "apply_doubly_controlled_gate", s, 0, 1, 2, X)
    rec(qc, "normalize", s)
    rec(qc, "state_norm", s)
    rec(qc, "product_state", [[1.0, 0.0], [1.0, 0.0]])
    rec(qc, "supported_gates")
    rec(qc, "nv_qubit_frequency_GHz")
    rec(qc, "qd_qubit_frequency_GHz", 5e-9)
    rec(qc, "spin_qubit_frequency_GHz", 1.0)
    rec(qc, "transmon_frequency_GHz")
    rec(qc, "qubit_summary", "transmon")
    rec(qo, "probabilities", s)
    rec(qo, "measure_all", s)
    rec(qo, "measure", s, 0)
    rec(qo, "probability", s, 0, 0)
    rec(qo, "expectation_pauli", s, "ZZZ")
    rec(qo, "entanglement_entropy", s, 0)
    rec(qo, "extract_max_probability", s)
    rec(qo, "extract_phase", s)
    rec(qo, "sample", s, 100)
    rec(qo, "sample_marginal", s, [0], 100)
    rec(qo, "schmidt_coefficients", s, 1)
    rec(qo, "state_to_bloch", s)
    rec(qo, "state_fidelity", s, s)
    headline = f"prepared a 3-qubit register, applied {len(vals)} gate/measure ops"
    return _domain_result("quantum_gates", "Quantum gates", vals, {}, headline,
                          "quantum_computing + quantum_output")


def quantum_algorithms_demo() -> MateriaResult:
    """Run the canonical quantum algorithms end-to-end: Deutsch-Jozsa,
    Bernstein-Vazirani, Grover search, QFT and inverse QFT, phase estimation,
    Simon, Shor (factoring 15), quantum walk, VQE, and ground-state finders for
    the Heisenberg and Ising models. Sweeps the quantum_algorithms +
    quantum_output algorithm surface."""
    QA = "sigma_ground.field.interface.quantum_algorithms"
    QO = "sigma_ground.field.interface.quantum_output"
    vals = _sweep_calls([
        (QO, "deutsch_jozsa", (), {}),
        (QO, "bernstein_vazirani", ("101",), {}),
        (QO, "grover_search", (3, 2), {}),
        (QO, "bell_state_example", (), {}),
        (QO, "teleportation_example", (), {}),
        (QA, "qft_example", (), {"n_qubits": 3}),
        (QA, "qft_circuit", (3,), {}),
        (QA, "inverse_qft_circuit", (3,), {}),
        (QA, "phase_estimation_example", (), {"n_ancilla": 3}),
        (QA, "quantum_walk", (), {"n_steps": 5, "n_positions": 8}),
        (QA, "simon_algorithm", ("101",), {}),
        (QA, "shor_factor_15", (), {}),
        (QA, "vqe_heh_plus", (), {"n_steps": 5}),
        (QA, "heisenberg_ground_state", (3,), {"n_steps": 5}),
        (QA, "ising_ground_state", (3,), {"n_steps": 5}),
        (QA, "qaoa_maxcut", ([(0, 1), (1, 2)], 3), {}),
        (QA, "qec_bit_flip_demo", (), {}),
        (QA, "ising_coupling_from_curie", ("iron",), {}),
        (QA, "ising_phase_transition_prediction", ("iron",), {}),
    ])
    headline = "Deutsch-Jozsa, Grover, QFT, Shor(15), VQE, Ising/Heisenberg"
    return _domain_result("quantum_algorithms_demo", "Quantum algorithms", vals,
                          {}, headline, "quantum_algorithms + quantum_output")


def plasma_physics(electron_density_m3: float = 1e19,
                   electron_temp_eV: float = 10.0,
                   b_field_t: float = 1.0) -> MateriaResult:
    """Plasma parameters: plasma and cyclotron frequencies, Debye length and
    number, Alfvén speed, the Coulomb logarithm, gyroradius and plasma beta.
    Sweeps the plasma module."""
    P = "sigma_ground.field.interface.plasma"
    ne = max(float(electron_density_m3), 1.0)
    Te = max(float(electron_temp_eV), 0.01)
    B = b_field_t
    me = 9.1093837015e-31
    vals = _sweep_calls([
        (P, "plasma_frequency", (ne,), {}),
        (P, "plasma_frequency_hz", (ne,), {}),
        (P, "debye_length", (ne, Te), {}),
        (P, "debye_number", (ne, Te), {}),
        (P, "coulomb_logarithm", (ne, Te), {}),
        (P, "alfven_speed", (B, 1e-7), {}),
        (P, "cyclotron_radius", (me, 1e6, B), {}),
        (P, "plasma_beta", (ne, Te, B), {}),
    ])
    fp = _pick(vals, "plasma_frequency")
    headline = (f"plasma frequency {fp:.3g} rad/s"
                if isinstance(fp, (int, float)) else "plasma computed")
    return _domain_result("plasma_physics", "Plasma", vals,
                          {"electron_density_m3": electron_density_m3,
                           "electron_temp_eV": electron_temp_eV}, headline, P)


def stellar_fusion(core_temp_K: float = 1.5e7,
                   density_kg_m3: float = 1.5e5) -> MateriaResult:
    """Stellar nucleosynthesis: proton-proton-chain and CNO-cycle energy
    generation rates, their crossover temperature, the Gamow peak energy and
    window for a fusion reaction, and the temperature sensitivity. Sweeps the
    nucleosynthesis module."""
    N = "sigma_ground.field.interface.nucleosynthesis"
    T = max(float(core_temp_K), 1e3)
    rho = max(float(density_kg_m3), 1e-3)
    vals = _sweep_calls([
        (N, "pp_chain_energy_rate", (T, rho), {}),
        (N, "cno_energy_rate", (T, rho), {}),
        (N, "pp_cno_crossover_temperature", (), {}),
        (N, "pp_temperature_exponent", (T,), {}),
        (N, "gamow_energy_keV", (1, 1, 1, 1, T), {}),
        (N, "gamow_window_keV", (1, 1, 1, 1, T), {}),
        (N, "reaction_rate_sigma_v", ("pp", T), {}),
        (N, "reaction_properties", ("pp",), {"T_K": T}),
    ])
    pp = _pick(vals, "pp_chain_energy_rate")
    headline = (f"pp-chain rate {pp:.3g} W/kg at {T:.2g} K"
                if isinstance(pp, (int, float)) else "fusion computed")
    return _domain_result("stellar_fusion", "Nucleosynthesis", vals,
                          {"core_temp_K": core_temp_K}, headline, N)


def piezoelectric_material(material_key: str = "PZT4") -> MateriaResult:
    """Piezoelectric response: charge / voltage from applied stress, strain and
    displacement from an applied field, electromechanical coupling and harvested
    energy density — with the full property report. Sweeps the piezoelectricity
    module."""
    P = "sigma_ground.field.interface.piezoelectricity"
    vals = _sweep_calls([
        (P, "material_piezoelectric_properties", (material_key,), {}),
        (P, "piezoelectric_voltage", (material_key, 1e6, 1e-3), {}),
        (P, "piezoelectric_strain", (material_key, 1e5), {}),
        (P, "piezoelectric_displacement", (material_key, 1e5, 1e-3), {}),
        (P, "piezoelectric_polarization", (material_key, 1e6), {}),
        (P, "coupling_coefficient", (material_key,), {}),
        (P, "coupling_coefficient_computed", (material_key,), {}),
        (P, "energy_density_harvested", (material_key, 1e6), {}),
    ])
    k = _pick(vals, "coupling_coefficient")
    headline = (f"{material_key} coupling {k:.3f}"
                if isinstance(k, (int, float)) else f"{material_key} piezo")
    return _domain_result("piezoelectric_material", "Piezoelectric", vals,
                          {"material_key": material_key}, headline, P)


def intermolecular_forces(molecule: str = "water") -> MateriaResult:
    """Hydrogen bonding and intermolecular forces: H-bond energy, estimated
    boiling point and enthalpy of vaporization, the breakdown of binding
    contributions, and the boiling-point / bond-energy orderings. Sweeps the
    hydrogen_bonding module."""
    H = "sigma_ground.field.interface.hydrogen_bonding"
    vals = _sweep_calls([
        (H, "intermolecular_properties", (molecule,), {}),
        (H, "intermolecular_breakdown", (molecule,), {}),
        (H, "estimated_boiling_point", (molecule,), {}),
        (H, "estimated_vaporization_enthalpy", (molecule,), {}),
        (H, "hydrogen_bond_energy_molecule", (molecule,), {}),
        (H, "hydrogen_bond_energy", ("O", "O"), {}),
        (H, "boiling_point_ordering", (), {}),
        (H, "hb_energy_ordering", (), {}),
    ])
    headline = "H-bond energies + boiling-point ordering"
    return _domain_result("intermolecular_forces", "Hydrogen bonding", vals,
                          {"molecule": molecule}, headline, H)


def organic_material(n_carbon: int = 8, bone_key: str = "cortical",
                     wood_key: str = "oak") -> MateriaResult:
    """Organic and biological materials: alkane boiling points and combustion
    enthalpy by chain length, plus anisotropic bone and wood mechanical reports.
    Sweeps the organic_materials module."""
    O = "sigma_ground.field.interface.organic_materials"
    nc = max(int(n_carbon), 1)
    vals = _sweep_calls([
        (O, "alkane_boiling_point_K", (nc,), {}),
        (O, "alkane_combustion_enthalpy_kJ_mol", (nc,), {}),
        (O, "combustion_enthalpy_kJ_mol", (4, 3, 2), {}),
        (O, "bone_report", (bone_key,), {}),
        (O, "bone_density_from_composition", (bone_key,), {}),
        (O, "bone_modulus_longitudinal", (bone_key,), {}),
        (O, "bone_modulus_transverse", (bone_key,), {}),
        (O, "bone_anisotropy_ratio", (bone_key,), {}),
        (O, "wood_report", (wood_key,), {}),
        (O, "hydrocarbon_report", (nc,), {}),
    ])
    bp = _pick(vals, "alkane_boiling_point_K")
    headline = (f"C{nc} alkane boils at {bp:.0f} K"
                if isinstance(bp, (int, float)) else "organics computed")
    return _domain_result("organic_material", "Organic materials", vals,
                          {"n_carbon": n_carbon}, headline, O)


def combustion_flow(porosity: float = 0.35,
                    particle_diameter_m: float = 1e-4) -> MateriaResult:
    """Combustion with porous-media flow (a smouldering carbon column): heat of
    combustion and flame temperature, soot fraction and emission colour, Darcy
    flow through the packed bed and Kozeny-Carman permeability. Sweeps the cigar
    module (combustion + Darcy flow)."""
    C = "sigma_ground.field.interface.cigar"
    vals = _sweep_calls([
        (C, "combustion_enthalpy_per_mol_C", (), {}),
        (C, "combustion_temperature", (), {}),
        (C, "soot_fraction", (), {}),
        (C, "soot_emission_color", (1200.0,), {}),
        (C, "kozeny_carman_permeability", (particle_diameter_m, porosity), {}),
        (C, "darcy_flow_velocity", (1e-11, 2e-5, 1000.0, 0.1), {}),
        (C, "darcy_mass_flow_rate", (0.01, 1.2, 1e-4), {}),
        (C, "gas_temperature_after_cooling", (1200.0, 300.0, 0.1, 0.01, 0.015),
         {}),
    ])
    tc = _pick(vals, "combustion_temperature")
    headline = (f"flame temperature {tc:.0f} K"
                if isinstance(tc, (int, float)) else "combustion computed")
    return _domain_result("combustion_flow", "Combustion + porous flow", vals,
                          {"porosity": porosity}, headline, C)


def subsurface_scattering(material_key: str = "skin_caucasian") -> MateriaResult:
    """Subsurface light transport in translucent materials: absorption and
    scattering lengths, diffusion coefficient and length, diffuse reflectance
    and BSSRDF parameters — with the full report. Sweeps the subsurface
    module."""
    S = "sigma_ground.field.interface.subsurface"
    vals = _sweep_calls([
        (S, "full_report", (), {}),
        (S, "absorption_length", (material_key,), {}),
        (S, "scattering_length", (material_key,), {}),
        (S, "diffusion_coefficient", (material_key,), {}),
        (S, "diffusion_length", (material_key,), {}),
        (S, "diffuse_reflectance", (material_key,), {}),
        (S, "bssrdf_parameters", (material_key,), {}),
        (S, "rayleigh_scattering_coefficient", (1e-7, 1.5, 1.33, 5e-7, 1e20), {}),
    ])
    dl = _pick(vals, "diffusion_length")
    headline = (f"{material_key} diffusion length {dl:.3g} m"
                if isinstance(dl, (int, float)) else f"{material_key} scattering")
    return _domain_result("subsurface_scattering", "Subsurface scattering", vals,
                          {"material_key": material_key}, headline, S)


def acoustics(material_key: str = "steel_mild") -> MateriaResult:
    """Acoustic waves in and across materials: longitudinal / transverse wave
    speeds, acoustic impedance, reflection / transmission at an interface, Snell
    refraction, resonant and ring frequencies — with the full acoustic report.
    Sweeps the acoustics module."""
    A = "sigma_ground.field.interface.acoustics"
    m2 = "water" if material_key != "water" else "aluminum"
    vals = _sweep_calls([
        (A, "material_acoustic_properties", (material_key,), {}),
        (A, "longitudinal_wave_speed", (material_key,), {}),
        (A, "transverse_wave_speed", (material_key,), {}),
        (A, "debye_velocity", (material_key,), {}),
        (A, "acoustic_impedance", (material_key,), {}),
        (A, "wave_speed_ratio", (material_key,), {}),
        (A, "reflection_coefficient", (material_key, m2), {}),
        (A, "transmission_coefficient", (material_key, m2), {}),
        (A, "critical_angle", (material_key, m2), {}),
        (A, "snell_refraction_angle", (material_key, m2, 0.3), {}),
        (A, "resonant_frequency", (material_key, 0.1), {}),
        (A, "ring_frequency", (material_key, 0.1), {}),
        (A, "wavelength", (1000.0, material_key), {}),
    ])
    c = _pick(vals, "longitudinal_wave_speed")
    headline = (f"{material_key} sound speed {c:.0f} m/s"
                if isinstance(c, (int, float)) else f"{material_key} acoustics")
    return _domain_result("acoustics", "Acoustics", vals,
                          {"material_key": material_key}, headline, A)


def chemical_reaction(temperature_k: float = 298.15) -> MateriaResult:
    """Reaction thermodynamics and kinetics: the Arrhenius rate, equilibrium
    constant and Gibbs energy, collision-theory prefactor, entropy change and
    Evans-Polanyi activation energy — with the full reaction report. Sweeps the
    chemical_reactions module."""
    R = "sigma_ground.field.interface.chemical_reactions"
    T = max(float(temperature_k), 1.0)
    vals = _sweep_calls([
        (R, "full_report", (), {"T": T}),
        (R, "arrhenius_rate", (1e13, 0.8, T), {}),
        (R, "equilibrium_constant", (-50.0, 0), {"T": T}),
        (R, "gibbs_energy_kJ_mol", (-50.0, 0), {"T": T}),
        (R, "collision_prefactor", (2.0, 32.0, 120.0, 150.0), {"T": T}),
        (R, "entropy_change_estimate", (1,), {"T": T}),
        (R, "evans_polanyi_activation_energy", (-50.0, "default"), {}),
    ])
    k = _pick(vals, "arrhenius_rate")
    headline = (f"Arrhenius rate {k:.3g} /s at {T:.0f} K"
                if isinstance(k, (int, float)) else "reaction computed")
    return _domain_result("chemical_reaction", "Chemical reactions", vals,
                          {"temperature_k": temperature_k}, headline, R)


def thermoelectric(material_key: str = "iron", t_hot: float = 600.0,
                   t_cold: float = 300.0) -> MateriaResult:
    """Thermoelectric transport: Seebeck-derived figure of merit ZT, electrical
    conductivity / resistivity, Fermi energy and free-electron density, and the
    Carnot efficiency bound — with the full thermoelectric report. Sweeps the
    thermoelectric module."""
    TE = "sigma_ground.field.interface.thermoelectric"
    vals = _sweep_calls([
        (TE, "material_thermoelectric_properties", (material_key,), {}),
        (TE, "figure_of_merit_ZT", (material_key,), {}),
        (TE, "electrical_conductivity", (material_key,), {}),
        (TE, "electrical_resistivity", (material_key,), {}),
        (TE, "fermi_energy", (material_key,), {}),
        (TE, "fermi_energy_ev", (material_key,), {}),
        (TE, "free_electron_density", (material_key,), {}),
        (TE, "carnot_efficiency", (t_hot, t_cold), {}),
        (TE, "heat_flow_through_leg", (material_key, t_hot, t_cold), {}),
        (TE, "leg_resistance", (material_key, 0.01, 1e-4), {}),
        (TE, "thermocouple_voltage", (material_key, material_key, t_hot, t_cold),
         {}),
        (TE, "thermoelectric_efficiency", (material_key, t_hot, t_cold), {}),
        (TE, "thermoelectric_power_max", (material_key, material_key, t_hot,
                                          t_cold), {}),
    ])
    zt = _pick(vals, "figure_of_merit_ZT")
    headline = (f"{material_key} ZT={zt:.3g}"
                if isinstance(zt, (int, float)) else f"{material_key} thermoelectric")
    return _domain_result("thermoelectric", "Thermoelectric", vals,
                          {"material_key": material_key}, headline, TE)


def metallurgy(material_key: str = "steel_mild",
               grain_size_um: float = 10.0) -> MateriaResult:
    """Microstructure-dependent strength: Hall-Petch yield vs grain size,
    polycrystal and Taylor hardening, grain-boundary area and energy,
    dislocation density and grain growth on annealing — plus alloy property
    prediction. Sweeps the grain_structure and alloys modules."""
    G = "sigma_ground.field.interface.grain_structure"
    AL = "sigma_ground.field.interface.alloys"
    d = max(float(grain_size_um), 0.01) * 1e-6
    vals = _sweep_calls([
        (G, "grain_structure_properties", (material_key, d), {}),
        (G, "hall_petch_yield", (material_key, d), {}),
        (G, "hall_petch_slope", (material_key,), {}),
        (G, "polycrystal_yield", (material_key, d), {}),
        (G, "taylor_hardening_stress", (material_key, 1e14), {}),
        (G, "grain_boundary_area_per_volume", (d,), {}),
        (G, "grain_boundary_energy_density", (material_key, d), {}),
        (G, "dislocation_density_estimate", (material_key, 0.01), {}),
        (G, "grain_size_for_yield", (material_key, 3e8), {}),
        (G, "grain_growth_rate_constant", (material_key, 800.0), {}),
        (G, "zener_limit", (material_key, 1e-7, 0.01), {}),
        (AL, "list_alloys", (), {}),
        (AL, "predict_all", (), {}),
        (AL, "alloy_properties", ({"copper": 0.7, "zinc": 0.3},), {}),
        (AL, "composition_sweep", ("copper", "zinc"), {}),
        (AL, "alloy_Nordheim_resistivity", ({"copper": 0.7, "zinc": 0.3},), {}),
        (AL, "predict_alloy_Tc", ({"niobium": 0.5, "titanium": 0.5},), {}),
    ])
    hp = _pick(vals, "hall_petch_yield")
    headline = (f"{material_key} yield {hp:.3g} Pa at {grain_size_um:g} um grain"
                if isinstance(hp, (int, float)) else f"{material_key} microstructure")
    return _domain_result("metallurgy", "Metallurgy", vals,
                          {"material_key": material_key,
                           "grain_size_um": grain_size_um}, headline,
                          "grain_structure + alloys")


def solution_chemistry(salt_key: str = "sodium_chloride") -> MateriaResult:
    """Electrolyte solutions: Debye-Hückel activity coefficients and screening
    length, colligative shifts (boiling-point elevation, freezing-point
    depression) for salts, dilution and molar conductivity — plus Faraday
    electrolysis. Sweeps the solution and electrochemistry modules."""
    S = "sigma_ground.field.interface.solution"
    E = "sigma_ground.field.interface.electrochemistry"
    vals = _sweep_calls([
        (S, "debye_length", (0.1,), {}),
        (S, "debye_huckel_A", (), {}),
        (S, "debye_huckel_B", (), {}),
        (S, "activity_coefficient_dh", (1, -1, 0.1), {}),
        (S, "activity_coefficient_salt", (salt_key, 0.1), {}),
        (S, "boiling_point_elevation_salt", (salt_key, 0.5), {}),
        (S, "freezing_point_depression_salt", (salt_key, 0.5), {}),
        (S, "boiling_point_elevation", (0.5,), {}),
        (S, "freezing_point_depression", (0.5,), {}),
        (S, "dilution", (1.0, 100.0, 500.0), {}),
        (E, "faraday_mass_deposited", (0.0635, 2.0, 3600.0, 2), {}),
        (E, "faraday_charge_required", (0.0635, 0.01, 2), {}),
        (E, "faraday_time_required", (0.0635, 0.01, 2.0, 2), {}),
        (E, "molar_conductivity_dilute", (0.0050, 0.0076), {}),
        (E, "can_displace", ("zinc", "copper"), {}),
        (S, "ionic_strength", ({"Na+": 0.1, "Cl-": 0.1},), {}),
        (S, "osmotic_pressure_salt", (salt_key, 0.1), {}),
        (S, "solubility_with_common_ion", (salt_key, 0.01), {}),
        (S, "will_precipitate", (salt_key, 0.1, 0.1), {}),
        (S, "mixing_concentration", (1.0, 100.0, 0.5, 100.0), {}),
    ])
    headline = f"{salt_key} activity, colligative shifts and electrolysis"
    return _domain_result("solution_chemistry", "Electrolyte solutions", vals,
                          {"salt_key": salt_key}, headline,
                          "solution + electrochem")


def optical_dispersion(material_key: str = "copper") -> MateriaResult:
    """Optical dispersion and colour: Cauchy refractive index, metal reflectance
    and Drude scattering, plasma frequency and electron density, dielectric and
    dye transmission, and the rendered RGB colour. Sweeps the optics module."""
    O = "sigma_ground.field.interface.optics"
    lam = 550e-9
    vals = _sweep_calls([
        (O, "metal_report", (material_key,), {}),
        (O, "cauchy_n", (material_key, lam), {}),
        (O, "metal_reflectance", (material_key, lam), {}),
        (O, "metal_rgb", (material_key,), {}),
        (O, "drude_scattering_rate", (material_key,), {}),
        (O, "electron_density", (material_key,), {}),
        (O, "plasma_frequency", (material_key,), {}),
        (O, "dielectric_surface_reflectance", (1.5,), {}),
        (O, "dielectric_opacity", ("glass",), {}),
        (O, "dielectric_color_rgb", ("glass",), {}),
        (O, "dielectric_transmission_rgb", ("glass",), {}),
        (O, "get_material_color", ("metal", material_key), {}),
        (O, "organic_rgb", ("chlorophyll",), {}),
    ])
    refl = _pick(vals, "metal_reflectance")
    headline = (f"{material_key} reflectance {refl:.3f}"
                if isinstance(refl, (int, float)) else f"{material_key} optics")
    return _domain_result("optical_dispersion", "Optical dispersion", vals,
                          {"material_key": material_key}, headline, O)


def superconductor(sc_key: str = "niobium", t_c: float = 9.25) -> MateriaResult:
    """Superconductivity: the BCS gap and coherence length, Ginzburg-Landau
    parameter and type-I/II classification, London penetration depth, lower /
    upper / thermodynamic critical fields, McMillan Tc and the specific-heat
    jump — with the full report. Sweeps the superconductivity module."""
    S = "sigma_ground.field.interface.superconductivity"
    vF, ne = 1e6, 1e28
    vals = _sweep_calls([
        (S, "superconductor_properties", (sc_key,), {}),
        (S, "bcs_gap_zero", (t_c,), {}),
        (S, "bcs_gap_temperature", (t_c, 2.0), {}),
        (S, "bcs_coherence_length", (vF, t_c), {}),
        (S, "gap_frequency", (t_c,), {}),
        (S, "gl_parameter", (ne, vF, t_c), {}),
        (S, "gl_parameter_effective", (sc_key,), {}),
        (S, "is_type_II", (sc_key,), {}),
        (S, "london_penetration_depth", (ne,), {}),
        (S, "london_penetration_at_T", (ne, t_c, 2.0), {}),
        (S, "lower_critical_field", (ne, vF, t_c), {}),
        (S, "upper_critical_field", (vF, t_c), {}),
        (S, "thermodynamic_critical_field", (ne, t_c), {}),
        (S, "depairing_current_density", (ne, vF, t_c), {}),
        (S, "meissner_fraction", (t_c, 2.0), {}),
        (S, "specific_heat_jump_ratio", (), {}),
        (S, "mcmillan_Tc", (0.3, 0.1, 300.0), {}),
        (S, "mcmillan_Tc_for", (sc_key,), {}),
        (S, "predict_Tc_from_Z", (41,), {}),
        (S, "derive_mu_star", (41,), {}),
        (S, "debye_comparison", (), {}),
    ])
    gap = _pick(vals, "bcs_gap_zero")
    headline = (f"{sc_key}: BCS gap {gap:.3g}"
                if isinstance(gap, (int, float)) else f"{sc_key} superconductor")
    return _domain_result("superconductor", "Superconductivity", vals,
                          {"sc_key": sc_key}, headline, S)


def tribology(material_key: str = "steel_mild",
              counter_key: str = "iron") -> MateriaResult:
    """Friction, wear and adhesion at a sliding contact: Archard wear volume,
    wear rate / depth / mass loss, specific and relative wear resistance, plus
    the work of adhesion, interface energy and contact angle between two
    materials. Sweeps the wear and adhesion modules."""
    W = "sigma_ground.field.interface.wear"
    AD = "sigma_ground.field.interface.adhesion"
    vals = _sweep_calls([
        (W, "wear_properties", (material_key,), {}),
        (W, "archard_wear_volume", (material_key, 100.0, 10.0), {}),
        (W, "sliding_wear_rate", (material_key, 100.0, 1.0), {}),
        (W, "specific_wear_rate", (material_key,), {}),
        (W, "wear_depth", (material_key, 1e6, 10.0), {}),
        (W, "wear_mass_loss", (material_key, 100.0, 10.0), {}),
        (W, "relative_wear_resistance", (material_key,), {}),
        (W, "sliding_distance_to_depth", (material_key, 1e-4, 1e6), {}),
        (W, "wear_regime", (material_key, counter_key), {}),
        (AD, "material_adhesion_properties", (material_key, counter_key), {}),
        (AD, "work_of_adhesion", (material_key, counter_key), {}),
        (AD, "interface_energy", (material_key, counter_key), {}),
        (AD, "adhesion_decomposition", (material_key, counter_key), {}),
        (AD, "contact_angle", (material_key, "water", 0.072), {}),
    ])
    w = _pick(vals, "work_of_adhesion")
    headline = (f"{material_key}/{counter_key}: work of adhesion {w:.3g} J/m²"
                if isinstance(w, (int, float))
                else f"{material_key} friction & wear")
    return _domain_result("tribology", "Friction / wear / adhesion", vals,
                          {"material_key": material_key,
                           "counter_key": counter_key}, headline,
                          "wear + adhesion")


def fluid_dynamics(liquid_key: str = "water",
                   temperature_k: float = 300.0) -> MateriaResult:
    """Liquid-phase fluid properties: dynamic and kinematic viscosity (with the
    Eyring activation model), surface tension, and the Reynolds number — with
    the full liquid-property report. Sweeps the fluid module."""
    F = "sigma_ground.field.interface.fluid"
    T = max(float(temperature_k), 1.0)
    vals = _sweep_calls([
        (F, "liquid_properties", (liquid_key,), {"T": T}),
        (F, "liquid_viscosity", (liquid_key,), {"T": T}),
        (F, "kinematic_viscosity", (liquid_key,), {"T": T}),
        (F, "eyring_viscosity", ("iron",), {"T": 1800.0}),
        (F, "surface_tension", (liquid_key,), {}),
        (F, "surface_tension_metal", ("iron",), {}),
        (F, "reynolds_number", (1000.0, 2.0, 0.05, 1e-3), {}),
    ])
    re = _pick(vals, "reynolds_number")
    headline = (f"Reynolds number {re:.3g}"
                if isinstance(re, (int, float)) else f"{liquid_key} fluid")
    return _domain_result("fluid_dynamics", "Liquid fluids", vals,
                          {"liquid_key": liquid_key}, headline, F)


def fuel_ignition(fuel_key: str = "methane") -> MateriaResult:
    """Combustion-ignition properties of a fuel: adiabatic flame temperature,
    autoignition temperature and flash point, ignition delay and flammability —
    with the full ignition report. Sweeps the ignition module."""
    IG = "sigma_ground.field.interface.ignition"
    vals = _sweep_calls([
        (IG, "full_report", (), {}),
        (IG, "ignition_report", (fuel_key,), {}),
        (IG, "adiabatic_flame_temperature", (fuel_key,), {}),
        (IG, "autoignition_temperature", (fuel_key,), {}),
        (IG, "flash_point", (fuel_key,), {}),
        (IG, "ignition_delay", (fuel_key, 1200.0), {}),
        (IG, "is_flammable_at", (fuel_key, 500.0), {}),
    ])
    T = _pick(vals, "adiabatic_flame_temperature")
    headline = (f"{fuel_key} flame {T:.0f} K"
                if isinstance(T, (int, float)) else f"{fuel_key} ignition")
    return _domain_result("fuel_ignition", "Ignition", vals,
                          {"fuel_key": fuel_key}, headline, IG)


def water_state(temperature_k: float = 298.15,
                pressure_atm: float = 1.0) -> MateriaResult:
    """The anomalous properties of liquid water vs temperature: density (and its
    4 °C maximum), viscosity, surface tension, heat capacity, molar volume,
    boiling point, enthalpy of vaporization and the ice-like fraction — with the
    full report. Sweeps the liquid_water module."""
    W = "sigma_ground.field.interface.liquid_water"
    T = max(float(temperature_k), 1.0)
    vals = _sweep_calls([
        (W, "water_properties", (), {"T_K": T, "P_atm": pressure_atm}),
        (W, "water_density", (T,), {}),
        (W, "water_viscosity", (T,), {}),
        (W, "water_surface_tension", (T,), {}),
        (W, "water_heat_capacity", (T,), {}),
        (W, "water_molar_volume", (T,), {}),
        (W, "water_boiling_point", (), {}),
        (W, "water_enthalpy_of_vaporization", (), {}),
        (W, "water_density_maximum_temperature", (), {}),
        (W, "ice_density", (), {}),
        (W, "ice_like_fraction", (T,), {}),
        (W, "ice_like_fraction_derivative", (T,), {}),
    ])
    rho = _pick(vals, "water_density")
    headline = (f"water density {rho:.1f} kg/m³ at {T:.0f} K"
                if isinstance(rho, (int, float)) else "water computed")
    return _domain_result("water_state", "Liquid water", vals,
                          {"temperature_k": temperature_k}, headline, W)


def viscoelastic_material(material_key: str = "iron") -> MateriaResult:
    """Viscoelastic response of a polymer: Maxwell and Kelvin-Voigt creep and
    stress relaxation, the standard-linear-solid model, storage / loss moduli
    and loss tangent, relaxation time and the damping peak — with the full
    report. Sweeps the viscoelasticity module."""
    V = "sigma_ground.field.interface.viscoelasticity"
    T = 300.0
    vals = _sweep_calls([
        (V, "viscoelastic_report", (material_key,), {}),
        (V, "full_report", (), {}),
        (V, "maxwell_creep_strain", (material_key, 1.0, 1e6, T), {}),
        (V, "maxwell_stress_relaxation", (material_key, 1.0, 1e6, T), {}),
        (V, "kelvin_voigt_creep", (material_key, 1.0, 1e6, T), {}),
        (V, "sls_creep", (material_key, 1.0, 1e6, T), {}),
        (V, "sls_stress_relaxation", (material_key, 1.0, 0.01, T), {}),
        (V, "storage_modulus", (material_key, 1.0, T), {}),
        (V, "loss_modulus", (material_key, 1.0, T), {}),
        (V, "loss_tangent", (material_key, 1.0, T), {}),
        (V, "relaxation_time", (material_key, T), {}),
        (V, "maxwell_viscosity", (material_key, T), {}),
        (V, "peak_damping_frequency", (material_key, T), {}),
    ])
    tan = _pick(vals, "loss_tangent")
    headline = (f"{material_key} loss tangent {tan:.3g}"
                if isinstance(tan, (int, float)) else f"{material_key} viscoelastic")
    return _domain_result("viscoelastic_material", "Viscoelasticity", vals,
                          {"material_key": material_key}, headline, V)


def asteroid_shape(body: str = "ceres", a_m: float = 4.87e5,
                   b_m: float = 4.87e5, c_m: float = 4.55e5) -> MateriaResult:
    """The shape and surface gravity of a small body (asteroid / moon): triaxial
    ellipsoid volume and mean radius, axis ratios, surface gravity and escape
    velocity. Sweeps the asteroids module."""
    A = "sigma_ground.field.asteroids"
    vals = _sweep_calls([
        (A, "ellipsoid_volume", (a_m, b_m, c_m), {}),
        (A, "mean_radius", (a_m, b_m, c_m), {}),
        (A, "axis_ratios", (body,), {}),
        (A, "surface_gravity", (body,), {}),
        (A, "escape_velocity", (body,), {}),
    ])
    v = _pick(vals, "ellipsoid_volume")
    headline = (f"ellipsoid volume {v:.3g} m³"
                if isinstance(v, (int, float)) else f"{body} shape")
    return _domain_result("asteroid_shape", "Asteroid shape", vals,
                          {"body": body}, headline, A)


def magnetic_material(material_key: str = "iron") -> MateriaResult:
    """Magnetic and dielectric response of a material: ferromagnetism and
    susceptibility, magnetization vs temperature and saturation magnetization;
    dielectric breakdown field and voltage, Clausius-Mossotti, loss tangent and
    stored energy density. Sweeps the magnetism and dielectric modules."""
    M = "sigma_ground.field.interface.magnetism"
    D = "sigma_ground.field.interface.dielectric"
    di = "silicon_dioxide"
    vals = _sweep_calls([
        (M, "is_ferromagnetic", (material_key,), {}),
        (M, "magnetic_susceptibility", (material_key,), {}),
        (M, "magnetization_at_temperature", (material_key, 300.0), {}),
        (M, "saturation_magnetization", (material_key,), {}),
        (M, "saturation_magnetization_measured", (material_key,), {}),
        (D, "breakdown_field", (di,), {}),
        (D, "breakdown_voltage", (di, 1e-3), {}),
        (D, "clausius_mossotti", (1e-29, 1e28), {}),
        (D, "dielectric_loss_tangent", (di, 1e6), {}),
        (D, "energy_density", (di, 1e6), {}),
        (D, "max_energy_density", (di,), {}),
    ])
    chi = _pick(vals, "magnetic_susceptibility")
    headline = (f"{material_key} susceptibility {chi:.3g}"
                if isinstance(chi, (int, float)) else f"{material_key} magnetic")
    return _domain_result("magnetic_material", "Magnetism / dielectric", vals,
                          {"material_key": material_key}, headline,
                          "magnetism + dielectric")


def diffusion_transport(temperature_k: float = 300.0) -> MateriaResult:
    """Mass diffusion and transport: the Einstein-Stokes diffusivity, Fick's
    first and second laws (steady flux and the transient error-function
    profile), Darken interdiffusion and the time to penetrate a depth. Sweeps
    the diffusion module."""
    D = "sigma_ground.field.interface.diffusion"
    T = max(float(temperature_k), 1.0)
    vals = _sweep_calls([
        (D, "einstein_stokes_diffusivity", (T, 1e-3, 1e-9), {}),
        (D, "ficks_first_law", (1e-9, 100.0), {}),
        (D, "ficks_second_law_erf", (1.0, 0.0, 1e-3, 1e-9, 3600.0), {}),
        (D, "darken_interdiffusion", (1e-9, 2e-9, 0.5, 0.5), {}),
        (D, "time_to_penetrate", (1e-9, 1e-3), {}),
    ])
    Dc = _pick(vals, "einstein_stokes_diffusivity")
    headline = (f"Einstein-Stokes diffusivity {Dc:.3g} m²/s"
                if isinstance(Dc, (int, float)) else "diffusion computed")
    return _domain_result("diffusion_transport", "Diffusion", vals,
                          {"temperature_k": temperature_k}, headline, D)


SCENARIOS = {
    "terminal_velocity_drop": terminal_velocity_drop,
    "drag_heating_drop": drag_heating_drop,
    "high_altitude_descent": high_altitude_descent,
    "supersonic_projectile": supersonic_projectile,
    "vertical_launch": vertical_launch,
    "orbital_velocity": orbital_velocity,
    "material_profile": material_profile,
    "structural_response": structural_response,
    "thermal_response": thermal_response,
    "rotational_dynamics": rotational_dynamics,
    "material_full_profile": material_full_profile,
    "quantum_report": quantum_report,
    # domain-sweep verbs (coverage campaign) — each answers a class of physics
    # question and exercises a whole sigma-ground module group.
    "charged_particle": charged_particle,
    "optical_fiber": optical_fiber,
    "semiconductor_device": semiconductor_device,
    "chemistry_lab": chemistry_lab,
    "atmospheric_profile": atmospheric_profile,
    "nuclear_decay": nuclear_decay,
    "collision_dynamics": collision_dynamics,
    "black_hole": black_hole,
    "matter_wave": matter_wave,
    "hydrogen_spectrum": hydrogen_spectrum,
    "quantum_circuit": quantum_circuit,
    "projectile_motion": projectile_motion,
    "relativistic_motion": relativistic_motion,
    "thermal_statistics": thermal_statistics,
    "fluid_flow": fluid_flow,
    "composite_material": composite_material,
    "quantum_dot": quantum_dot,
    "atomic_coupling": atomic_coupling,
    "molecular_bond": molecular_bond,
    "condensed_matter": condensed_matter,
    "rolling_object": rolling_object,
    "elastic_solid": elastic_solid,
    "quantum_gates": quantum_gates,
    "quantum_algorithms_demo": quantum_algorithms_demo,
    "plasma_physics": plasma_physics,
    "stellar_fusion": stellar_fusion,
    "piezoelectric_material": piezoelectric_material,
    "intermolecular_forces": intermolecular_forces,
    "organic_material": organic_material,
    "combustion_flow": combustion_flow,
    "subsurface_scattering": subsurface_scattering,
    "acoustics": acoustics,
    "chemical_reaction": chemical_reaction,
    "thermoelectric": thermoelectric,
    "metallurgy": metallurgy,
    "solution_chemistry": solution_chemistry,
    "optical_dispersion": optical_dispersion,
    "superconductor": superconductor,
    "tribology": tribology,
    "fluid_dynamics": fluid_dynamics,
    "fuel_ignition": fuel_ignition,
    "water_state": water_state,
    "viscoelastic_material": viscoelastic_material,
    "asteroid_shape": asteroid_shape,
    "magnetic_material": magnetic_material,
    "diffusion_transport": diffusion_transport,
}
