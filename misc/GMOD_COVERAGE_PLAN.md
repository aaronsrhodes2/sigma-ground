# Testing Mentat's toolbelt in Garry's Mod: scenarios, limits, and what a physics mod could add

**Status:** 2026-10-10. The ledger (`stage/gmod/coverage/`) is machine-checked. Engine scenarios **S01 to S06 are
built and pass** against Mentat on the dedicated server (`stage/gmod/experiments/`; results and findings in
`stage/gmod/README.md`). S05's gear train and hand tool are deferred. **S07 cannot be an engine test** (no installed
map has water) and is reclassified PORT. Nothing past S07 is built.

## 1. The answer in numbers

Mentat's toolbelt is two layers: **221 MCP tools** and **68 simulation verbs** behind
`run_simulation`. That is **289 entries**. Every one has a class in the ledger.

| Class | Entries | Meaning |
|---|---|---|
| ENGINE | 21 (7 %) | Garry's Mod's own solver independently produces the observable. The only class that is a test of Mentat by an outside oracle |
| PORT | 67 | GMod has no such physics; the scenario implements the law in Lua at 1:1. Verified by parity against Mentat, not by an independent engine |
| SHOW | 45 | A bench instrument displaying a value Mentat computes |
| PREDICT | 35 | Math, unit or parameter tool that builds the prediction another scenario checks |
| LOOKUP, GENERATOR | 18 | Reference data and the verbs that generate the scenario data |
| BLOCKED | 3 | Room scale in principle, but we lack a metric fluid solver |
| OUT | 100 (35 %) | Cannot be observed at 1:1 room scale (section 4) |

**It takes 16 experiment scenarios plus a console, 17 to build, plus 1 that is blocked.**
Estimated effort is about 33 working days for all 17, dominated by the Lua ports. Six engine scenarios (S01 to
S06) are built and graded; two replay scenarios (`fall_skillet`, `motor_spin`) also run through the real game.

Read the table honestly: **21 of 289 entries (7 %) can be validated against an independent physics
engine.** The 168 buildable entries are mostly demonstrations. That is still worth building, because a
demonstration with a parity test catches regressions and makes the physics operable and visible, but it
is not 168 independent checks.

## 2. What "test" can mean here

Three things limit what a scenario proves:

1. **Havok is fed Mentat's numbers.** Friction, restitution and densities are surface properties we set
   from Mentat. Havok then cannot validate those numbers, only the formulas that use them (the incline
   distance, the collision outcome). Entries that supply such a parameter are classed PREDICT, not ENGINE:
   `friction_analysis`, `hertzian_impact_analysis`, `tribology`, `water_state`.
2. **The engine has its own error.** Measured in the real game: with its air drag and its 0.1 per second
   linear damping removed, Havok integrates a fall exactly like semi-implicit Euler at its 15 ms tick
   (agreement 5e-5 HU over a 125 HU drop). That differs from the textbook formula by g·t·τ/2, about
   1.4 % over a 0.75 s fall. Every ENGINE test must carry that term or raise the tick rate.
3. **Havok's drag is not physics.** It cannot judge Mentat's Stokes drag. `viscous_flow_analysis`,
   `fluid_flow`, `terminal_velocity_drop` and `drop_object` are PORT: a Lua drag force replaces the engine's.

## 3. The 17 scenarios

Each names its oracle. Tool counts are entries where the scenario is listed (primary or secondary).

**Engine cross-checks (Havok is the oracle)**

| ID | Scenario | What you see and measure | Covers (examples) |
|---|---|---|---|
| S01 | Drop tower | Spheres of several metals, a coin and a feather fall from known heights; fall time, impact speed, energy and momentum; a second run adds a Lua drag port | `free_fall_time`, `free_fall_velocity`, `kinetic_energy`, `gravitational_potential_energy`, `work_energy_analysis`, `viscous_flow_analysis`, `atmospheric_profile`, `joules_to_TNT` (15) |
| S02 | Cannon range | A launcher at set angle and speed; apex, flight time, range by trigger zones; a straight-up throw | `projectile_analysis`, `projectile_range`, `procedure_projectile_trajectory`, `differentiate_expr`, `solve_equation`, verbs `projectile_motion`, `vertical_launch` (11) |
| S03 | Ramp and brake | Blocks of different material pairs on an adjustable incline; critical angle, distance slid, stopping distance | `incline_analysis`, `friction_stopping_distance`, `friction_analysis`, `integrate_expr`, `tribology` (8) |
| S04 | Collision track | Carts and spheres colliding head-on; elastic and inelastic outcomes, momentum and energy bookkeeping, bounce height | `collision_analysis`, `hertzian_impact_analysis`, `momentum`, `solve_linear_system` (11) |
| S05 | Rolling race and gear train | A disc, ring and sphere race down a ramp; a hand tool and gear train run under torque | `rotational_dynamics`, `work_energy_analysis`, `matrix_eigenvalues` (inertia tensor), `matrix_determinant`, `matrix_multiply`, verbs `rolling_object`, `actuate_hand_tool` (12) |
| S06 | Pendulum and spring | Pendulum and mass-on-spring at several amplitudes; period and decay vs the ODE solution | `solve_ode`, `series_expansion`, `compute_limit`, `simplify_expr`, `expand_expression`, `factor_expression` (6) |
| S07 | Pool | Floating and sinking blocks, an ice cube, falling balls; draft depth, flotation, settling | `buoyancy_analysis`, `density`, `viscous_flow_analysis`, verbs `water_state`, `fluid_flow` (7) |

**Lua ports (GMod lacks the physics; parity against Mentat, not an independent solver)**

| ID | Scenario | What you see | Covers (examples) |
|---|---|---|---|
| S08 | Furnace | A heated iron ball glows by blackbody colour; a hot block on a cold slab relaxes; a block melts; a thermocouple reads a voltage | `blackbody_peak_wavelength`, `blackbody_total_power`, `procedure_stellar_blackbody`, `thermal_contact_analysis`, `thermoelectric_generator_analysis`, `melting_point`, `boiling_point`, verbs `contact_conduction`, `thermal_response` (18) |
| S09 | Gas lab | A clear cylinder with a weighted piston (pressure, volume, temperature), a Carnot cycle, a convection plume, a drifting scent | `ideal_gas_pressure`, `ideal_gas_volume`, `carnot_efficiency`, `natural_convection_analysis`, `diffusion_analysis` (8) |
| S10 | Echo hall | Clap, bell and stone-strike sounds delayed by distance at the true speed of sound; echoes off walls; the bell spectrum | `speed_of_sound_in_ideal_gas`, `acoustic_interface_analysis`, `fourier_transform`, verbs `acoustics`, `strike_bell` (9) |
| S11 | Ray-optics bench | Lasers through glass, a prism, lenses and a fibre; refraction, total internal reflection, image distance, dispersion | `snells_law_refraction_angle`, `critical_angle_for_tir`, `thin_lens_*`, `lens_magnification`, `optical_waveguide_analysis`, `refractive_index` (9) |
| S12 | Wave optics, spectra and decay bench | Slit and grating patterns, a Bragg mirror, a hydrogen discharge tube through a spectroscope, an afterglow, a Geiger counter | `double_slit_fringe_spacing`, `diffraction_grating_angle`, `photonic_bandgap_analysis`, `hydrogen_emission_wavelength`, `phosphor_decay_analysis`, `radioactivity_analysis`, `summation` (22) |
| S13 | Circuit bench | Resistor, capacitor, inductor and diode circuits with lamps and a scope | `ohms_law_current`, `rc_time_constant`, `rlc_resonant_frequency`, `capacitor_analysis`, `diode_current`, `laplace_transform`, device readouts (23) |
| S14 | Charge and field bench | Charged pith balls on strings, a charge in a magnetic field, a Hall probe, a B-H loop, field-line maps | `coulomb_force`, `electromagnetic_force_analysis`, `gradient`, `divergence`, `curl`, `hall_effect_analysis` (10) |
| S15 | Material test rig | A rod pulled until it stretches, yields and breaks; a creeping polymer; a worn pin. **Constraint proxy only: no geometry changes shape** | `elastic_analysis`, `plasticity_analysis`, `stress_failure_analysis`, `viscoelastic_creep_analysis`, `wear_analysis`, `composite_bounds_analysis` (15) |
| S16 | Wet-chemistry bench | Beakers whose colour and a pH meter follow titration, buffers, solubility, kinetics, calorimetry | `titration_analysis`, `buffer_ph`, `molar_solubility`, `osmotic_pressure`, `cell_potential`, `combustion_enthalpy_analysis` (20) |

**Infrastructure and blocked**

| ID | Scenario | Notes |
|---|---|---|
| S00 | Bench console | An in-world terminal that calls any tool and shows its answer with its source. Calls every tool, tests none. Hosts the constants, materials, units and bodies lookups. Also cross-checks Source's 1 HU = 0.01905 m against Mentat's `convert_units` |
| S17 | Surface and wave tank | `wetting_analysis`, `wind_wave_analysis`. **Blocked**: GWater2 is position-based and not metric |

**Build order.** By raw entry count the greedy order is S13, S12, S16, S08, S15, then S01 (13 % to 63 %),
but those are the weakest-verified scenarios. By validation value the order is the ENGINE set first:
**S01, S02, S03, S04, S05, S06, S07**, which together are about 9 working days and give the only real
cross-checks. Recommended: ENGINE set, then S08 (it reuses existing bundles and the thermal channel),
then the rest by appetite. S00 and S16 need the live bridge to Mentat (Phase 5 of the stage plan).

## 4. What cannot be tested at 1:1 room scale

**100 of 289 entries (35 %).** These cannot be observed in a 6 m room, a 600 m map or a 66 Hz solver.

| Kind | Entries | Why | Examples |
|---|---|---|---|
| ATOMIC_QUANTUM | 45 | Atoms, nuclei, molecules, microscopic solid-state parameters, qubits, cryogenic quantum matter | `quantum_tunneling_analysis`, `bond_energy`, `de_broglie_wavelength`, `carrier_mobility`, `superconducting_gap_analysis`, `nuclear_binding_energy` |
| ASTRO | 29 | Planets, stars, black holes, cosmology | `orbital_period`, `escape_velocity`, `schwarzschild_radius`, `hawking_temperature`, `hubble_radius`, `age_of_universe` |
| RELATIVISTIC | 13 | Needs v near c | `lorentz_factor`, `relativistic_time_dilation`, `mass_to_energy` |
| THEORY | 9 | The SSBM / holographic extended layer, no laboratory observable | `hde_dark_energy_density`, `bekenstein_hawking_entropy` |
| LARGE_SCALE | 2 | Kilometre-scale Earth phenomena | verbs `high_altitude_descent`, `supersonic_projectile` |
| EXTREME_REGIME | 1 | Intensities a room never reaches | `nonlinear_optics_analysis` (Mentat's default is 1e13 W/m²) |
| TIMESCALE | 1 | Too fast to see | `light_travel_time` (Mentat: 3 m takes 10.0 ns) |

Two edge cases worth knowing:

- **Newtonian gravity between room-scale masses is real and invisible.** Mentat's `gravitational_force`
  gives 2.67e-6 N between two 100 kg masses 0.5 m apart, which is 2.7e-9 of one mass's 980.7 N weight.
  Havok cannot resolve it. You waived n-body at room scale, and this is why that is safe.
- **Nine exportable replay bundles are kilometre-scale falls** (`falling_copper_ball`, `heating_iron_ball`
  and others, spans of 1,000 m to 40 km). They are out of reach at 1:1 inside Source's map and wait for
  Phase 4 scaling. Scaling a scene is a *different experiment*, not a test of the 1:1 claim.

For the astronomical entries the right oracle is not a game. It is **JPL Horizons** (positions, via its REST
API) with **REBOUND** as an integrator cross-check. Sigma-ground already has a JPL test: ARCHITECTURE.md
lists `test_jpl_ephemeris` as network-gated. REBOUND is not used anywhere in the repo yet.

## 5. "Solar System Simulator" and Universe Sandbox

You asked whether a universe sandbox could host the stellar scenarios.

- **Solar System Simulator** (Steam app 2355330, publisher WPC, $7.99, released 2024-03-22) is a casual
  gravity-and-exploration sandbox. Its store page lists "recalculate orbits in line with Newton's laws" and
  the ability to add bodies. It lists **no Workshop, no mod support, and no data export**, and the community
  hub shows no scripting threads. It cannot be driven or read programmatically, so it cannot be a test
  oracle or a host.
- **Universe Sandbox** is the better-known one. Its saves are zip files holding editable JSON, and it can
  graph a property and export the graph to CSV. It has **no scripting API**, and a user reports it recomputes
  some orbital and climate values on import. So it can display a scenario we generate, and give an
  eyeball cross-check, but it is not authoritative: its own model is the thing you would be comparing against.

Verdict: neither is a test oracle. Use them, if at all, to *look at* a stellar scene. They do not belong in
the test chain, and they are not needed for the GMod work.

## 6. A GMod mod that adds the missing physics

You asked whether we could build a GMod mod that integrates the physics GMod lacks, with uniform downward
gravity as an acceptable simplification. **Yes for the room-scale subset, no for "all", and "seamless" needs
care.** The 21 + 67 + 45 ENGINE, PORT and SHOW entries are the realistic ceiling: 133 of 289.

The mod would be a layer on top of Havok, not a replacement. Proposed layers, each with its own scenario
as acceptance test:

| Layer | What it adds | Status of the engine hooks |
|---|---|---|
| L0 Materials | Real density, friction, restitution per material from Mentat `material_profile`, registered as Source surface data | `physenv.AddSurfaceData` is referenced by the wiki, field format not yet verified here |
| L1 Air and fluids | A Lua drag force from Mentat's drag laws, true buoyancy, with Source's own drag off | Drag and the 0.1/s damping switches are **verified**; buoyancy ratio exists |
| L2 Thermal | Per-object temperature, contact conduction, radiative glow, phase change at the melting point | Lumped model is cheap in Lua; fields need the Mentat bridge |
| L3 Failure | Break and crack events via break-force constraints and pre-fractured hulls | **No continuous deformation**: sigma-ground has none either (section 14 of the stage plan) |
| L4 Instruments | Optics, acoustics, circuits, EM bench props | Ports, verified by offline parity |

How each layer stays honest follows what the stage already does: pure-Lua ports are loaded under LuaJIT in
the test suite and compared with independent Python oracles, and anything solver-heavy calls Mentat over
the Phase 5 bridge. The in-game autotest and the independent grader exist and have already caught real
engine facts (angular velocity is local-frame, Source applies two hidden slowdowns, `ceramic` is not a
surface name).

Where "seamless" is a stretch:

- **It is a second physics engine on top of Havok.** Forces from Lua and forces from Havok interact (drag
  plus engine damping, energy accounting). Each added force needs the engine's matching term disabled, as
  already done for drag and linear damping.
- **Deformation and real fluids are not available** (S15 is a proxy, S17 is blocked). A dent or a splash
  cannot be shown by any layer until sigma-ground has a displacement channel and a metric fluid.
- **Cost and scope.** Lua at 66 Hz is fine for hundreds of objects, not thousands. Multiplayer needs the
  state networked. Other addons that add their own forces will fight the layer.

## 7. Risks and open questions

- The ledger's classification is my own judgment. An independent audit was started and cancelled mid-run;
  the circularity points in section 2 came from my own second pass, and any ENGINE claim deserves a
  challenge before a scenario is built on it.
- Effort estimates are rough. The S15 and S12/S13 ports are the large unknowns.
- Phase 5 (live bridge) gates S00 and S16 and everything that wants a generated scenario.
- The shapes gap still limits replays: only Sphere, Box, Cylinder export, and 38 of the 86 trajectory bundles are refused for
  other shapes (4 more exceed the client file limit). The ENGINE scenarios mostly use those three, which is another reason to build them first.

## 8. Keeping coverage honest

`python stage/gmod/coverage/check_coverage.py` validates the ledger against the frozen toolbelt
snapshot and prints these tables. `--refresh` re-freezes the snapshot from the live Mentat container.
`test_gmod_coverage.py` fails the suite if Mentat gains or loses a tool without a ledger decision, if an
entry is internally inconsistent, or if a scenario cites a bundle that does not exist.
