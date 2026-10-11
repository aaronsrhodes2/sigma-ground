# SkippyVerse physics: the reuse map

*2026-10-11. Read-only survey of the master tree (`.claude/worktrees/` ignored). Nothing was run: Mentat is down, no game, no servers.
Marks: **V** = verified by reading code (function bodies, not docstrings); **D** = docstring or document only; **U** = unverified guess.
Companion to `misc/SKIPPYVERSE_PHYSICS_IMPORT_PLAN.md` (integration-mode plan) and `misc/GMOD_COVERAGE_PLAN.md`; this file is the inventory underneath them.*

## 0. Headline

1. Aaron is right about **reuse**. A new "spec / validator / runner" would duplicate three things that exist: `materia/spec.py` (compile-then-run `SimulationSpec` with step-to-step bindings, V), `engine/` (`WorldHost`, closed `AppliedForce` taxonomy, replayable command log, V), and `radiance/trajectory.py` + `export_gmod.py` (a solver run recorded into a frame bundle the game replays, V).
2. He is only partly right about the **data**. The "static libraries" are small closed tables (counts by import, V): 25 surface materials, 14 metals with transport data, 9 reactions, 17 electrode couples, 19 solubilities, 24 ions, 21 acids, 19 catalogued Deckard shapes. A railgun or a chemistry bench runs on those; anything off-table is refused or missing.
3. Nothing in the repo models a railgun, coil gun or launcher (searched `sigma_ground`, `misc`, `stage`, `docs`, `tests`: zero hits). The only "railgun" strings are two **weapon presets** in `games/weapon_vs_armor/presets.py` (a muzzle velocity and a slug, no launcher physics, V).
4. The real gap is not physics data and not a spec language. It is (a) **coupled process models** for the three domains, (b) a **game-side read-back** (solver state to colour, glow, sound, readouts), (c) a **live path** from the game server to the code (none exists; "Phase 5" of the stage plan).

## 1. Materia and the simulation front door

Path of a sentence today (V): `mcp/tools/simulation.py:simulate` -> `scenario_intent.build_manifest` (keyword intent) -> `materia.translate` (`materia/translator.py`: keyword router with regex slot extraction `_extract_*`; `_qwen_translate` / `_qwen_plan` via Ollama as residual) -> `SimulationSpec` (`materia/spec.py`) -> `run_spec` (deterministic; `bindings = {slot: (prior_step, output_field)}` threads outputs forward, a missing hand-off raises) -> `_wrap` (provenance, self-check, `synthesize_chain`). Structured path: `run_simulation(verb, params)` (no LLM) and `list_simulation_scenarios()` (reads `materia/manifest.py:VERB_MANIFEST`).

Notes (V): `SCENARIOS` has 70 entries, `VERB_MANIFEST` 68. `scenario_intent` is explicitly a provisional emphasis table that nothing downstream consumes yet (`read_emphasis` returns None until the spine emits one). The LLM only emits a spec; it never runs a step (spec.py docstring and code agree).

| Family | Verbs | What it computes | Inputs -> outputs | Kind |
|---|---|---|---|---|
| Fall / flight with drag | `terminal_velocity_drop`, `drag_heating_drop`, `high_altitude_descent`, `supersonic_projectile`, `vertical_launch` | Leapfrog (`dynamics.stepper`) through `field.interface.atmosphere` density, Reynolds/Mach-dependent Cd (`engine.simulate_fall`, `simulate_drag_run`, `simulate_vertical_launch`); drag heating via `drag_heating_temperature_rise`; `drag_heating_drop` can add a per-cell windward field (`materia/thermal_field.py` -> `dynamics.fields.heat.diffuse`) | material_key/radius/altitude or mass/diameter/Mach -> impact_speed_m_s, fall_time_s, delta_T_K, peak_T_K, max_mach, distance_to_subsonic_m (named `outputs`, chainable) | **Time-stepped** (V) |
| Contact heat | `contact_conduction` | Hot block on cold slab on a voxel grid, harmonic-mean face conductivity (`dynamics.fields.heat.diffuse_fvm`) | two materials, T_hot, environment -> heat_crossed_J, interface_k_W_mK, peak_T_end_K, glows | **Time-stepped** (V) |
| Named object | `drop_object`, `actuate_hand_tool`, `strike_bell` | `drop_object`: Deckard Construct (mass, section) then **analytic** terminal velocity only; `actuate_hand_tool`: discovers a revolute pivot from geometry, no stepping in the verb (render deferred); `strike_bell`: `acoustics.ring_frequency` + speed of sound | name/material -> terminal speed; pivot_type, mass; f_ring, arrival time | Closed form (V) |
| Orbit, planet, relativity, nuclear | `orbital_velocity`, `planetary_surface`, `mass_energy`, `black_hole`, `asteroid_shape`, `relativistic_motion`, `stellar_fusion`, `nucleon_masses`, ... | Library calls | body/mass -> speeds, radii, energies | Closed form (`orbital_velocity` D, rest V by the shared pattern) |
| Material profiles | `material_profile`, `structural_response`, `thermal_response`, `rotational_dynamics`, `material_full_profile`, `elastic_solid`, `composite_material`, `metallurgy`, `viscoelastic_material`, `tribology`, `acoustics`, `magnetic_material`, `magnetic_hysteresis`, `piezoelectric_material`, `thermoelectric`, `optical_dispersion`, `transmission_line`, ... | Property bundles from `field.interface.*` | material_key -> named property values | Closed form (V) |
| Domain sweeps (about 45) | `charged_particle`, `semiconductor_device`, `chemistry_lab`, `solution_chemistry`, `chemical_reaction`, `molecular_bond`, `corrosion_attack`, `combustion_flow`, `fuel_ignition`, `quantum_gates`, `quantum_algorithms_demo`, `quantum_tunneling`, `plasma_physics`, `superconductor`, `diffusion_transport`, ... | `_sweep_calls` / `_domain_result` call a fixed list of library functions at **default arguments** (`_AUTO_ARGS`) and report values | few or no slots; manifest `outputs` is just `functions_called` -> **not parameterisable, not chainable** | Closed-form survey (V) |

**Reuse:** `run_spec` bindings are the chain runner. Adding a scenario = adding a function to `SCENARIOS` and an entry to `VERB_MANIFEST` (triggers, slots, `bind_from`); the translator then routes and chains it unchanged (contract in the manifest header, V).
**Gap:** the 45 sweep verbs cannot take railgun/chemistry parameters; only the 6 stepped verbs have real chainable outputs; no verb couples two domains except `drag_heating_drop` (drag to heat).

## 2. Playground (`mcp/tools/playground.py`)

| Question | Answer (all V) |
|---|---|
| World model | **Not the engine.** A module-level dict `_SCENES[handle]` holding an `inventory` `Structure` (atoms/molecules/quarks via `playground_load`) or a "bulk" spec (`playground_make` via `object_library.resolve`). No ECS, no Region, no command log, no AppliedForce. |
| What you can build | One object per handle: a named material structure or a standard object ("brick"; "ball of plutonium" asks for size). Not assemblies, circuits or chemistry. |
| Mutation | `playground_apply` -> `inventory.behaviors.apply_env`, 5 knobs (temperature_k, pressure_pa, electric_field_vm, magnetic_field_t, energy_ev); returns a diff of bond lengths / charge states / spin. Atomic scale only; bulk objects refuse. |
| Solvers | `playground_simulate` forwards the **material name** and the sentence to `simulation.simulate` (Materia). The mutated scene state is not read. `playground_render` -> radiance. |
| Read-back | `ToolResult.to_dict()` JSON (value, units, source, provenance_tag, inputs). |
| Deterministic / serialisable | In-process only, live objects, no replay. **Excluded from the public HTTP Mentat**: `SkippyTel/mentat/app.py` `EXCLUDED_PREFIXES=("playground_",)`, `EXCLUDED={"simulate","request_clarification"}`; reachable via stdio MCP (`docker exec -i skippytel-mentat python -m sigma_ground.mcp.server`) or `/ask`. |
| Verdict | Not consumable by a game. The thing that is: `engine/` (section 3). |

## 3. Solvers and libraries

| Package | Domain / what it steps | State | Coupling hooks | Mark |
|---|---|---|---|---|
| `dynamics/stepper.py` (`step`, `step_to`), `collision.py`, `solver.py` | Leapfrog rigid parcels (spheres), sequential-impulse contacts/friction | `PhysicsScene` of `PhysicsParcel` | **`external_forces=f(parcel)->Vec3`** callback (used for drag/buoyancy by `materia/engine.py`); this is where an EM force goes | V |
| `dynamics/joints.py`, `mechanisms/*` | Weld / Revolute / Prismatic / GearCoupling joints, spring, bearing, escapement, wind, pump, oscillating actuator | joint state | torque-capped motors | V (defs/docstrings) |
| `dynamics/continuum.py`, `fluid/` | SPH continuum: density sum, EOS (Tait / Birch-Murnaghan), pressure, viscosity, buoyancy, gravity, heat conduction, phase check per `ContinuumParcel` | parcels with T, P, U, material_key, phase | material cascade each step | D |
| `dynamics/fields/heat.py` | FVM heat on a voxel grid (`diffuse`, `diffuse_fvm`, `stable_dt`, `thermal_energy`) | numpy grid | used by `contact_conduction`, `thermal_field` | V |
| `dynamics/gravity/barnes_hut.py`, `field/interface/nbody.py`, `phase_e.py` | Gravity (2D quadtree; N-body FR4 symplectic), n-body vs SPH comparison | bodies | `field/interface/adapters/integrator.py`: **`sg_solve_ivp` / `sg_odeint`, fixed step, `method='rk4'` for non-separable systems** | D / V (integrator) |
| `engine/` (`ecs.py`, `region.py`, `host.py`, `intents.py`, `audio_events.py`) | **WorldHost**: single-writer stepper, `SIM_DT=1/960`, bounded observer queues, `pace` 0.01-10, `save()`/`load()` as `{scenario, dt, step_count, command log}` replayed bit-exactly, `state_hash`; `Region` protocol (`step/apply/extract/spawn`); ECS `Matter` (mass, CoM, composition, T, energy ledger, mandatory provenance) | The only region is `StackSimRegion` = `radiance.push_off.StackSim`, a **planar (x,z) stack of boxes** | Closed `AppliedForce` kinds (`dynamics/mechanisms/applied.py`): impulse, force, drive, hold, source, tilt. Region route table: impulse/force/hold/tilt live; **source (heat) and drive (joints) REFUSED**. `propulsion.impulse_burn_applied_force` / `sustained_burn_applied_force` already emit these. `audio_events` annotates impacts server-side. | V |
| `mcp/cockpit.py` | HTTP + websocket around WorldHost: `GET /world` attach / apply / intent / pace / save / hash; `python -m sigma_ground.mcp.cockpit` (port 8765); frames carry `t`, bodies, support, settled, `events` | named worlds persist in process | the one place a game could attach to a deterministic world | V (handlers + docstring); shipped in the Mentat container: U |
| `kernel/` (`shapes`, `csg`, `parts`, `gear`, `voxel`, `outline`); `sigma_ground/{shapes,csg,parts}.py` are shims | Analytic SDF primitives, CSG, volume/inertia; parts: ANSI pipes, ISO bolts, AISC W-beams, `convert_to_primitives`; involute gear | geometry | consumed by `deckard.Construct` | V (shims) / D |
| `deckard/` | Name -> Construct (layers: material per part, mass, CoM, inertia, bbox, SDF); 19 catalogued shapes; LLM research path (~60 s then cached); `scene.build_stack`, `hand_tools`, `articulate`, `dodoxel*` | Construct | feeds Materia (`deckard_bridge.request_shape`) and ECS (`matter_from_construct`) | V |
| `blueprint/` | Cited `MechanismSpec` (gear, spring, escapement) + `validate` arithmetic checks | data | mechanisms only, no electrical parts | D |
| `inventory/` | Atom / molecule / quark structures (16), `behaviors`, `apply_env` | Structure | playground only | V |
| `field/` | about 140 modules of closed-form physics plus `electrodynamics.py`; time stepping only in `nbody`, `rolling_shootout`, `adapters.integrator` | | | V (defs) |
| `labs/` | **A shim**: `sigma_ground/labs` re-exports `materia/labs` (scene / environment / runner / forces). Not a solver package. | | | V |

## 4. Radiance's physics-in-scripts and how they reach the game

| Script | Physics folded in | Output | In the game today |
|---|---|---|---|
| `trajectory.py` (`record_fall`, `record_object_fall`, `record_descent`, `record_horizontal_run`, `record_motor_spin`, `record_gear_train_spin`, `record_windmill_*`, `record_escapement_clock`, `record_bell_strike`, `record_hand_tool_actuation`) | Reuses `materia.engine.make_atmospheric_drag`, `dynamics.stepper`, joints / mechanisms; runs the solver and **records frames** `{t_sim, bodies:[{pos, quat, temperature_k?}]}` + SceneSpec + auto `suggested_rate` (playback rate from the phenomenon's timescale) | `radiance/web/data/<name>.json` | via `export_gmod` |
| `thermal_record.py` (`record_fall_thermal`, `record_thermal_field`) | Drag work integral to body T per frame; **re-integrates and refuses if it disagrees by more than 2 %** with Materia's scalar (a cross-check pattern worth copying) | bundle with `temperature_k` | exported, **ignored** (no thermal colouring, "Phase 3"; `export_gmod.py` ~lines 388-396, V) |
| `push_off.py` (`StackSim`, `record_push_off`) | Planar stack / friction (doughnut, plate, table arc; push-speed sweep) | bundles `plate_pushed_off_*`; also the WorldHost region | replay |
| `water_waves.py`, `bell_acoustics.py` | Gravity-capillary dispersion height field; decaying single mode + inverse-square propagation | functions / WAV | web viewer only (S10 / S17 planned) |
| `theaters.py`, `scene.py`, `scene_export.py` | Staging presets (room / plain / void / deep_space) stamping `physics_env`; Construct to SceneSpec with colour **baked server-side** from the optics library | JSON | colour baked in the bundle |
| `phase_e_export.py` | n-body / SPH bundle pairs | 4 `phase_e_*` bundles | exceed the 64 KB client limit unless `--decimate` |
| `entangler/`, `raymarch.py`, `render.py`, `camera.py` | SDF raymarch rendering, Planck x Kirchhoff incandescence | images | n/a |
| `export_gmod.py` (`export_scenario`, `export_file`, `detect_up`) | Bundle to one Lua file, format `mentat-scenario/1`; Sphere / Box / Cylinder only; mass clamp 0.1-50,000 kg; at most 60 KB compressed; 44 of 86 bundles export (README, D) | `stage/gmod/addon/lua/mentat/scenarios/*.lua` | `mentat_load / play / rate / scrub / release`: Lua replay by kinematic teleport, optional hand-off to Havok |

`radiance/web/data` holds about 120 JSON files (V by listing): falls of 8 metals, drag / heating, motors, gears, escapement clock, windmills, bell strike, pliers, doughnut arcs, phase E, leaf-stress sweeps, voxel scenes, theaters. **None contains EM, chemistry or quantum content.**

## 5. What the game stage already does with physics

| Item | Fact | Mark |
|---|---|---|
| Ledger | 289 entries (221 MCP tools + 68 verbs) classified in `stage/gmod/coverage/` (ENGINE 21, PORT 67, SHOW 45, PREDICT 35, LOOKUP / GEN 18, BLOCKED 3, OUT 100). An accounting, not an execution. | D |
| Executed | `stage/gmod/experiments/mentat_cache.json`: 659 cached answers over **34 distinct tools** (my count; the plan says 36), S01-S06 only: falls, cannon range, ramp / friction, collisions, rolling, pendulum / spring | V (counted) |
| Pattern | Mentat answer (cached with `source`) -> `experiments/specs.py` spec JSON -> Lua builds, releases, records -> independent Python grader. **Mentat is the oracle; Havok is the realisation.** A live call only on a cache miss, through `tools/mentat_client.py` = `docker exec skippytel-mentat` (urllib to `localhost:8080`) | V |
| Ignored | `temperature_k`; material / composition on holodeck props (pose + mass only); deformation (contract has none) | V / D |
| Replays in the real game | `fall_skillet`, `motor_spin` | D |
| Live channel to code | `stage/gmod/holodeck/skippy_bridge.py` on the PC: `GET /health`, `POST /plan /reply /speak /resolve /mesh` (fixed actions, token, budgets, PLATINUM 19); Lua calls it with `HTTP({...})` (`holodeck.lua:236`). `addon/lua/mentat/tool.lua` runs headless Lua jobs from `data/mentat/tool.txt`. **No physics endpoint; README: "no live bridge (Phase 5)".** | V |
| Run-time payload to clients | `meshnet.lua` / `meshclient.lua` relay a bridge payload (<= 256 KB, `svm1-lzma`) to clients for found-object meshes | D / V (bridge docstring) |

## 6. Domain coverage for the three target scenarios

### 6a. Railgun

| Need | What exists (file:function) | Mark | Gap |
|---|---|---|---|
| EM force on an armature | `field/electrodynamics.py:lorentz_force`, `magnetic_force` (point charge q v x B); `mcp/tools/plasma_em.py:electromagnetic_force_analysis` (fixed vectors) | V | No force on a current-carrying conductor (F = 1/2 L' I^2), no B field of a rail geometry (no Biot-Savart function found), no magnetic pressure on rails |
| Rail inductance | `field/interface/mobius.py:parallel_pair_inductance(_per_m)`, `coaxial_*`, `twisted_pair_*` (round wires; formula not read: U) | V (names) | Flat rails with armature position; skin / proximity effects |
| Capacitor bank, discharge | `field/interface/electronics.py:parallel_plate_capacitance`, `energy_stored`; `mcp/tools/circuits.py:rc_time_constant`, `rl_time_constant`, `rlc_resonant_frequency`, `power_dissipation_resistor`; `devices.py:capacitor_analysis` | V | Closed-form time constants only. **No RLC transient with a moving back-EMF load**; no bank (ESR / ESL), switch or diode |
| Generic ODE stepping | `field/interface/adapters/integrator.py:sg_solve_ivp(method='rk4')` | V (docstring + defs) | Not an MCP tool; `mathx.solve_ode` is symbolic `sympy.dsolve` only (V) |
| Resistive heating | `electronics.py:resistivity(material, T)` (Bloch-Gruneisen, 14 metals, V), `thermal.py:specific_heat_j_kg_K`, `phase_transition.py` (melting), `dynamics/fields/heat.py` | V | Not coupled: nothing integrates I^2 R into T(t) and feeds R(T) back; melt / phase change of the armature as a process is missing |
| Rail erosion, plasma | `wear.py:archard_wear_volume`, `plasma.py:spitzer_resistivity`, `plasma_frequency`, `debye_length` | V (names) | No sliding-contact to arc transition, no ablation, no rail damage model |
| Flight | `engine.simulate_drag_run` (Mach Cd, altitude density, V), `field/interface/projectile.py:projectile_with_drag`, verb `supersonic_projectile` | V | adequate |
| Impact, penetration | `games/weapon_vs_armor/physics/kinetic.py:interact` (Poncelet + Birkhoff, empirical ceramic / fibre multipliers, V); `hertzian_impact_analysis`, `field/interface/impact.py`, `stress.py` | V | Lives under `games/` (served by SkippyTel `mentat/wva.py` `/wva/solve`); no deformation / crater state |
| Strength | `stress_failure_analysis`, `plasticity_analysis`, `field/interface/plasticity.py` | M | no transient structural loading |
| Parts | `kernel/parts.py` (pipes, bolts, beams), `deckard` (19 shapes), `kernel/shapes` + `csg` | V | No capacitor / rail / busbar / insulator parts; no electrical connectivity in a Construct |

Chain today: closed-form calls (C, V, R(T), cp, melt T, density) -> **a new ODE** (circuit + armature motion + heating) stepped with `sg_solve_ivp` or `stepper.step(external_forces=...)` -> muzzle state -> `simulate_drag_run` -> `kinetic.interact`. Every arrow except the middle one exists.

### 6b. Quantum computer

| Need | What exists | Mark | Gap |
|---|---|---|---|
| State-vector simulator | `field/interface/quantum_computing.py`: `zero_state`, `basis_state`, `product_state`, `apply_single_gate`, `apply_controlled_gate`, `apply_doubly_controlled_gate`, gates X Y Z H S T Rx Ry Rz Phase CNOT CZ SWAP iSWAP Toffoli Fredkin, `run_circuit(n, [('h',0),('cnot',0,1)])`, `supported_gates()` | V | Pure Python lists of `complex`, no numpy; hard guard `n_qubits <= 25` in `zero_state` (V); practical speed well below that (not timed: U) |
| Measurement, analysis | `quantum_output.py`: `probabilities`, `measure`, `measure_all`, `sample`, `expectation_pauli`, `state_fidelity`, `entanglement_entropy`, `state_to_bloch`, `schmidt_coefficients`, Deutsch-Jozsa, Grover, Bernstein-Vazirani, teleportation | V (defs) | **No density-matrix or noisy simulation** (module states "decoherence-free"; T1 / T2 / fidelity in `qubit_summary` are quoted typical values) |
| Algorithms | `quantum_algorithms.py`: QFT, phase estimation, Shor(15), Simon, QAOA maxcut, Ising / Heisenberg ground state, VQE (HeH+), bit-flip QEC, quantum walk | V (defs) | fixed demos |
| Hardware physics | `qubit_summary(type)` for transmon / spin / quantum dot / NV; frequencies from `superconductivity`, `quantum_wells` | V | typical measured values only |
| MCP exposure | `mcp/tools/qcomputing.py`: `quantum_algorithm_analysis` (Grover n, QAOA edges, Simon string), `quantum_state_analysis` (|+> demo), `qubit_hardware_analysis`; verbs `quantum_gates`, `quantum_algorithms_demo` (fixed circuits) | V | **`run_circuit` is not exposed as a tool** (no hit in `mcp/tools`). An agent cannot submit its own circuit. |
| Visual | none | | nothing maps amplitudes or Bloch vectors to anything visible |

### 6c. Chemistry with visible reactions

| Need | What exists | Mark | Gap |
|---|---|---|---|
| Reaction thermochemistry | `chemical_reactions.py`: `REACTIONS` (9: methane and ethanol combustion, two hydrogenations, Haber, HCl, HF, water formation, methane chlorination), `reaction_enthalpy_kJ_mol` (bond-energy sums), `gibbs_energy_kJ_mol`, `equilibrium_constant`, `entropy_change_estimate`; MCP `reaction_enthalpy`, `combustion_enthalpy_analysis` | V | Only 9 reactions; bond-energy accuracy not assessed (U); no stoichiometry balancer or species database |
| Kinetics | `arrhenius_rate`, `evans_polanyi_activation_energy`, `half_life`, `temperature_for_rate`; MCP `reaction_kinetics_analysis` | V | Rate constants only. **No concentration-vs-time integration**, no multi-step mechanisms |
| Acid-base, titration | `acid_base.py`: `pH_*`, `henderson_hasselbalch`, `titration_curve`, `polyprotic_alpha` (21 acids); MCP `weak_acid_ph`, `buffer_ph`, `titration_analysis`, `acid_speciation_analysis` | V | no indicator / colour model found (U) |
| Electrochemistry | `electrochemistry.py`: Nernst, cell potential, Faraday, Tafel (17 couples); MCP `cell_potential`, `electrolysis_mass`, `electrochemistry_analysis` | V | not time-stepped |
| Solutions | `solution.py`: solubility (19 salts), `will_precipitate`, activity (24 ions), colligative; MCP `molar_solubility`, `osmotic_pressure` | V | no precipitate or gas as a visible object |
| Combustion, ignition | `ignition.py` (autoignition, flash point, adiabatic flame T, delay), `organic_materials.py` | V (defs) | no flame propagation |
| Transport | `diffusion.py`: Fick 1st and 2nd (1D erf solution), Einstein-Stokes; heat FVM in `dynamics/fields/heat.py` | V (defs) | **No spatial reaction-diffusion, no species in the SPH fluid** |
| Appearance | `thermal_emission.py:thermal_emission_rgb`, `thermal.py:blackbody_color`, `atomic_spectra.py` (flame / discharge lines), `quantum_wells:qd_color_rgb`; MCP `material_color_analysis` | V (names) | hot-body glow and spectra yes; colour of solutions and precipitates no |
| Gas | `gas.py` (ideal gas, Cv / Cp, viscosity), `ideal_gas_*` tools | V | no gas volume evolving from a reaction |

## 7. The binding gap

What exists between "Skippy says a sentence" and "the result is visible":

```
sentence -> skippy_bridge /plan (Ollama qwen3:8b; fixed holodeck vocabulary only)
         -> game parser -> props (kind, pose, mass)      [material, T, composition ignored]
Mentat:  sentence -> /ask or simulate() -> SimulationSpec -> run_spec -> ToolResult   [text/JSON, no game path]
Solver run -> trajectory.record_* -> bundle -> export_gmod -> .lua file -> mentat_load  [offline, read at map load]
```

| Missing link | Detail | Reusable pieces |
|---|---|---|
| 1. Intent to solver inputs | No mapping from a holodeck object / card (kind, material, mass, dims) to solver slots. The bridge vocabulary is a closed catalog of HL2 props (no "capacitor", "rail", "reagent"). | `deckard.request_shape`, `materia/deckard_bridge.py`, `engine.ecs.matter_from_construct`; the verb **slot schema** in `VERB_MANIFEST` as the parameter contract; the bridge's catalog-validation style |
| 2. Scenario is not a verb yet | Railgun / chemistry / quantum have no verb; sweep verbs take no parameters | add verbs (section 8); manifest triggers make the translator route them |
| 3. Read-back to the world | The scenario contract carries pose and `temperature_k` only; no colour / emissive, particles, sound events, readouts, scalar series. The game ignores temperature. Bodies move by teleport. | `thermal_emission_rgb` (bake colour server-side as `scene_export` does for materials); the `audio_events` event shape; `suggested_rate` and `mentat_rate` for time scale |
| 4. Delivery at run time | Scenarios are `.lua` files read at map load (README: a regenerated file reaches clients only at map load); 64 KB client limit | `meshnet.lua` / `meshclient.lua` relay of a bridge payload; `tool.lua` server-side job runner |
| 5. Where Mentat's "agent" sits | Two kinds exist: (a) Materia compile-then-run (LLM emits a spec, deterministic runner); (b) tool-calling loops in `mcp/benchmark/run_gemini.py` and the portal chat (`ask_mentat`). Neither is an open planner / executor with state. Mentat runs in container `skippytel-mentat` (`SkippyTel/mentat/app.py`: `/tools/<name>`, `/ask`, `/mcp`, `/wva/*`) that **publishes no port** (reach it by `docker exec` or the compose name `mentat-home:8080`), installs sigma-ground at a **pinned GitHub commit** (new verbs need a push and an image rebuild), and **excludes `simulate` and `playground_*`** from `/tools`. | `tools/mentat_client.py:mentat_raw` (docker exec), `/mcp`, or importing `sigma_ground` in the bridge process on the PC (likely possible: U) |
| 6. Determinism, time scale | Materia verbs are deterministic; the qwen translation is not (`_route_consistent` exists). WorldHost replay is bit-exact (V), sim dt 1/960 s, `pace` 0.01-10. A railgun shot lasts milliseconds, chemistry minutes; bundles carry `suggested_rate`. | record then replay (not live co-simulation): matches the frozen-output doctrine |
| 7. Honesty gate | PLATINUM 21: every number from Mentat with `source`. In-process import bypasses that unless the recorder gets its constants through the tool layer. | `ToolResult` provenance; `thermal_record`'s cross-check-or-refuse |

## 8. Recommended thin bridge (reuse everything above)

Principle: **a scenario = a Materia verb (python composing existing library calls) + a recorder that turns it into a frame bundle + a generic player.** No new spec language: the verb's slots in `VERB_MANIFEST` are the spec, `run_spec` is the runner, the bundle is the result.

1. **Verb layer** (`materia/scenarios.py`, listed in `SCENARIOS` + `VERB_MANIFEST` with triggers, slots, `outputs`): `rail_shot`, `rlc_discharge`, `reaction_batch`, `quantum_circuit_run` (circuit as a slot, wrapping `quantum_computing.run_circuit`). Named `outputs` make `bind_from` chaining work (`rail_shot` -> `supersonic_projectile` -> impact).
2. **Recorder layer** (siblings of `thermal_record.record_fall_thermal`): run the verb's solver with recording on, emit frames `{t_sim, bodies:[{pos, quat, temperature_k, rgb?, emit?}], events?, readouts?}`, refuse on cross-check failure.
3. **Contract extension** (additive, as `temperature_k` was): optional `rgb` / `emit` per body, `events` (muzzle flash, impact, precipitate), `readouts` (name, unit, value, source).
4. **Player layer**: extend `export_gmod` and one Lua module to apply `rgb` / `emit` (replacing the Phase 3 stub), play events, show readouts on a bench prop; deliver the bundle at run time through the existing bridge-to-client relay instead of a map-load `.lua`.
5. **Service layer**: one new **fixed action** on `skippy_bridge` (for example `POST /scenario {"verb","params"}`), allow-listed to manifest verbs, under the existing token and budget machinery; it calls Mentat (`/mcp` or `mentat_raw`) and returns the bundle. The sentence reaches it through the existing `/plan` path with "scenario" as an action whose `verb` must be in the manifest (validated like catalog kinds today).
6. Mentat stays calculator and oracle; the S01-S06 pattern (Mentat prediction, game realisation, grader) is the acceptance test per scenario.

### First slice: railgun

| Step | Work | Reuses |
|---|---|---|
| a | Verb `rail_shot(cap_F, volts, rail_len_m, rail_L_per_m, armature_material, armature_mass_kg, circuit_R)` with state (I, x, v, T), F = 1/2 L' I^2, Joule heating to T, R(T) feedback, stop at melt | `electronics.energy_stored`, `resistivity`, `thermal.specific_heat_j_kg_K`, `phase_transition`, `mobius.parallel_pair_inductance_per_m` (or a flagged flat-rail Choice), `sg_solve_ivp(rk4)` |
| b | Chain `supersonic_projectile` (`bind_from=muzzle_speed`) and `kinetic.interact` for impact | existing |
| c | Self-checks: energy ledger closes (capacitor energy = KE + Joule + magnetic + residual), v below sqrt(2E/m), agreement with `electromagnetic_force_analysis`; refuse on failure | `MateriaResult.validation` |
| d | Recorder: armature / projectile pos and `temperature_k` to `rgb` via `thermal_emission_rgb`; readouts I(t), V_cap(t), v(t); `suggested_rate` about 1e-3 for the millisecond shot | `trajectory.py`, `thermal_record.py` |
| e | In game: bench prop (rails from Box / Cylinder leaves, which `export_gmod` supports), projectile replay with glow, readout board | `export_gmod`, `mentat_play / rate` |
| f | Tests: same params through `run_simulation` and through the bundle agree; Lua replay equals bundle (as `test_gmod_stage.py` does for the others) | existing test style |

It does **not** need a new law registry, a new spec / validator, or the WorldHost (planar boxes only; heat and drive refused). The WorldHost matters for interactive, command-logged worlds (import plan P2); a railgun shot is a recorded run.

### Biggest real gaps, ranked

1. Coupled process models (EM circuit + motion + heating; reaction-network ODE; an arbitrary-circuit quantum tool): the library has the terms, nobody integrates them together.
2. Game read-back channels (colour / emissive, events, readouts, run-time payload): the game displays pose only and ignores temperature.
3. A live, allow-listed path game -> bridge -> Mentat/verb, with a deploy story (container pinned to a GitHub commit, no published port, `simulate` excluded from `/tools`).

## 9. Open points

- The playground and the WorldHost are different things; do not assume "playground + WorldHost" is one world model.
- A new verb needs a push, a new `SIGMA_GROUND_REF` in the SkippyTel image and a rebuild.
- Data scarcity will bite before solvers do (9 reactions, round-wire inductance, 14 metals with transport data). Missing numbers must be owner-entered with a cited handbook, not invented.
- Not checked: speed of the pure-Python state vector above about 12 qubits; whether the Mentat container ships `mcp/cockpit`; accuracy of the bond-energy reaction enthalpies; `orbital_velocity` internals; how many of the 289 ledger classes are still accurate.
