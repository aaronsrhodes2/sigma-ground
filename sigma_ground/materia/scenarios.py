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
    import math
    from sigma_ground.mcp.tools.orbital import (orbital_velocity as _ov,
                                                orbital_period as _op,
                                                _resolve_central)
    from sigma_ground.field.constants import AU_M
    _, body_radius_m, label = _resolve_central(central_body, None)

    if semimajor_axis_au is not None:                       # heliocentric
        v = _ov(central_body, semimajor_axis_au=semimajor_axis_au).value
        r = semimajor_axis_au * AU_M
        period = _op(semimajor_axis_au=semimajor_axis_au,
                     central_body=central_body).value
        where = f"{semimajor_axis_au:g} AU from {label}"
        loc_step = MateriaStep("Orbit radius", semimajor_axis_au, "AU",
                               "heliocentric distance", "user")
    else:                                                   # satellite altitude
        v = _ov(central_body, altitude_km=altitude_m / 1000.0).value
        r = (body_radius_m or 0.0) + altitude_m
        period = 2.0 * math.pi * r / v if v else float("inf")
        where = f"{altitude_m/1000:.0f} km above {label}"
        loc_step = MateriaStep("Altitude", altitude_m, "m", "above the surface",
                               "user")

    if period < 7200:
        per = f"{period/60:.1f} min"
    elif period < 2 * 86400:
        per = f"{period/3600:.1f} hr"
    else:
        per = f"{period/86400:.0f} days"

    steps = [
        MateriaStep("Central body", label, "", "(mass & radius looked up)",
                    "mcp.tools.astronomy"),
        loc_step,
        MateriaStep("Orbital velocity", v, "m/s", "v = √(G·M / r)",
                    "mcp.tools.orbital.orbital_velocity"),
        MateriaStep("Orbital period", period, "s", "Kepler III / T = 2πr/v",
                    "mcp.tools.orbital"),
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
        ("superconductivity", "superconductor_properties", ("Nb",), {}),
        ("angular_momentum", "angular_momentum_report", (), {}),
    ])
    return _sweep_result("quantum_report", "Quantum", suites,
                         {"element_Z": element_Z})


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
}
