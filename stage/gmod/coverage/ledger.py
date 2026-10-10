"""Coverage ledger: every Mentat tool and simulation verb -> the GMod scenario that exercises it.

Mentat's toolbelt is two layers, both accounted for here:
  * 221 MCP tools (GET /tools on the Mentat HTTP API)
  * 68 simulation verbs behind the `run_simulation` tool (list_simulation_scenarios)

Every entry gets exactly one CLASS, and every class states what a scenario can honestly claim:

  ENGINE     Garry's Mod's own physics (Havok) independently produces the observable, so the
             scenario CROSS-VALIDATES Mentat against a mature solver. The only class that is a
             test of Mentat's physics by an outside oracle. Where Havok is crude the note says so
             and the test measures the gap instead.
  PORT       GMod has no such physics. The scenario implements the law in Lua and shows its effect
             at 1:1 scale. Verified by offline parity of the Lua port against the Mentat tool
             (as already done for pose interpolation), NOT by an independent engine: a
             demonstration plus a regression test, not a validation.
  SHOW       A bench instrument: a value Mentat computes is displayed in-world (live bridge). No
             check beyond provenance.
  PREDICT    A math/unit/conversion tool used to compute the prediction another scenario checks.
             Exercised, but with no observable of its own.
  LOOKUP     Reference data (constants, materials, bodies, units, manifest): the bench console.
  GENERATOR  Produces the scenario data itself (run_simulation builds the trajectory bundles).
  BLOCKED    Room-scale in principle, but we lack a capability to render it right: needs a fluid
             solver or real deformation. Named so the gap is visible, not hidden.
  OUT        Cannot be observed in a 1:1 room-scale sandbox. `kind` says why:
               ASTRO           planets, stars, black holes, cosmology
               LARGE_SCALE     km-scale Earth phenomena (stratosphere drops, supersonic range)
               RELATIVISTIC    effects that need v ~ c
               ATOMIC_QUANTUM  atoms, nuclei, molecules, solid-state microscopic parameters,
                               qubits, cryogenic quantum matter
               THEORY          the SSBM / holographic extended layer (no laboratory observable)
               TIMESCALE       too fast or too slow to see (light crossing a room in 10 ns)
               EXTREME_REGIME  intensity or plasma regimes a room does not reach

The ledger is data. check_coverage.py proves it is complete against the frozen toolbelt snapshot
and test_gmod_coverage.py runs that check on every test run.
"""
from __future__ import annotations

# --------------------------------------------------------------------------------------
# Scenarios. `kind` is the dominant coverage class. `bundles` are existing Radiance bundles
# that already hold (part of) the data. `needs` lists the work that gates the scenario.
# --------------------------------------------------------------------------------------
SCENARIOS = {
    "S00": dict(
        name="Bench console", kind="LOOKUP", effort_days=1,
        what="An in-world terminal: type any Mentat tool and parameters, see the answer with its source. "
             "Also hosts constants, materials, units and bodies. Calls every tool; tests none against physics.",
        oracle="none (provenance only)", bundles=[], needs=["live bridge"]),
    "S01": dict(
        name="Drop tower", kind="ENGINE", effort_days=1,
        what="Spheres of different materials, a coin and a feather fall from known heights; a second run adds "
             "air resistance. Measures fall time, impact speed, energy and momentum, and terminal speed.",
        oracle="Havok free fall with its air resistance turned off, so it is exactly checkable; drag is a Lua port",
        bundles=["drop", "fall_sphere_*", "fall_cup", "fall_glass", "falling_*"], needs=["Cone/Ellipsoid for the feather"]),
    "S02": dict(
        name="Cannon range", kind="ENGINE", effort_days=1,
        what="A launcher fires a ball at set angles and speeds on a flat floor; apex, flight time and range are "
             "measured with trigger zones. Includes a straight-up throw.",
        oracle="Havok ballistic flight",
        bundles=["iron_ball_thrown_up", "sim_copper_ball"], needs=[]),
    "S03": dict(
        name="Ramp and brake", kind="ENGINE", effort_days=1,
        what="Blocks of different material pairs on an adjustable incline: critical sliding angle, distance slid, "
             "and stopping distance on the flat after a launch.",
        oracle="Havok friction with Mentat's coefficient fed in via surface data: validates the incline/stopping formulas, not the coefficient",
        bundles=[], needs=["physenv.AddSurfaceData from Mentat material tables"]),
    "S04": dict(
        name="Collision track", kind="ENGINE", effort_days=1,
        what="Carts and spheres colliding head-on on a frictionless-as-possible rail: elastic and inelastic "
             "outcomes, momentum and energy bookkeeping, and bounce height from restitution.",
        oracle="Havok rigid-body collisions (restitution fed from Mentat): validates the collision outcomes, not the restitution value",
        bundles=["clatter"], needs=["restitution fed into surface data"]),
    "S05": dict(
        name="Rolling race and gear train", kind="ENGINE", effort_days=2,
        what="A solid disc, a ring and a sphere race down a ramp; a hinged hand tool and a gear train run under "
             "torque. Measures acceleration ratios, rotational energy and mechanical power.",
        oracle="Havok rigid-body inertia and hinge/gear constraints",
        bundles=["gear_train", "gear_mesh", "motor_spin", "pliers_opening_and_closing", "clock"],
        needs=["Gear/Rotated/Voxel shapes for some bundles"]),
    "S06": dict(
        name="Pendulum and spring", kind="ENGINE", effort_days=1,
        what="A pendulum and a mass on a spring at several amplitudes. Period and amplitude decay measured against "
             "the closed-form and ODE solutions Mentat's math tools produce.",
        oracle="Havok rope/spring constraints; the prediction chain runs through solve_ode and series_expansion",
        bundles=[], needs=[]),
    "S07": dict(
        name="Pool", kind="PORT", effort_days=2,
        what="A tank with floating and sinking blocks of known material, an ice cube, and falling balls: draft "
             "depth, flotation, and viscous settling speed.",
        oracle="Lua port of buoyancy and Stokes drag against the Mentat tools (offline parity). NOT an engine test: neither installed map "
               "(gm_construct, gm_flatgrass) has water, Lua cannot create a water volume, and Source buoyancy would be a constant ratio anyway",
        bundles=[], needs=["a Lua buoyancy and drag force on bodies below a plane", "real densities (Mentat density)"]),
    "S08": dict(
        name="Furnace", kind="PORT", effort_days=3,
        what="A heated iron ball glows by blackbody colour as it heats; a hot block is set on a cold slab and the "
             "temperature field relaxes; a block melts at its melting point; a thermocouple reads a voltage.",
        oracle="Lua port of the heat laws against the Mentat tools (offline parity); not an independent solver",
        bundles=["heating_iron_ball", "heating_mild_steel_ball", "windward_iron_ball",
                 "hot_iron_cube_on_a_copper_slab_irt"], needs=["thermal colour channel (Phase 3)", "cell-field display"]),
    "S09": dict(
        name="Gas lab", kind="PORT", effort_days=2,
        what="A clear cylinder with a weighted piston: pressure-volume-temperature, a Carnot cycle, a convection "
             "plume above a hot plate, and a drifting scent cloud.",
        oracle="Lua ideal-gas port; piston equilibrium height is settled by Havok statics against the prediction",
        bundles=[], needs=["pure-Lua gas law port", "particle effect for convection"]),
    "S10": dict(
        name="Echo hall", kind="PORT", effort_days=2,
        what="Clap, bell and stone-strike sounds delayed by distance at the true speed of sound, with echoes off "
             "walls of different material, and a spectrum of the bell ring.",
        oracle="Lua timer port of the speed of sound and reflection coefficients; not an independent solver",
        bundles=["bell_struck_by_a_stone"], needs=["Cone shape for the bell", "sound delay port"]),
    "S11": dict(
        name="Ray-optics bench", kind="PORT", effort_days=2,
        what="Laser beams through a glass block, a prism, lenses and a fibre: refraction angles, total internal "
             "reflection, image distance and magnification, dispersion.",
        oracle="Lua refraction port over engine line traces; checked against the Mentat tools offline",
        bundles=[], needs=["beam renderer", "refraction port"]),
    "S12": dict(
        name="Wave-optics, spectra and decay bench", kind="PORT", effort_days=3,
        what="Slit and grating patterns on a screen, a Bragg mirror, a hydrogen discharge tube seen through a "
             "spectroscope, a glow-in-the-dark afterglow and a Geiger counter on a sample.",
        oracle="Lua ports of the pattern, line and decay laws against the Mentat tools offline",
        bundles=[], needs=["screen/pattern renderer", "Poisson click source"]),
    "S13": dict(
        name="Circuit bench", kind="PORT", effort_days=3,
        what="Resistor, capacitor, inductor and diode circuits with lamps and a scope: Ohm's law, RC/RL charging, "
             "RLC resonance, capacitor energy, a Seebeck pair, a piezo tip.",
        oracle="Lua circuit port stepped in time, compared with the Mentat tools offline",
        bundles=[], needs=["scope prop", "circuit port"]),
    "S14": dict(
        name="Charge and field bench", kind="PORT", effort_days=2,
        what="Charged pith balls on strings (Coulomb), a charge in a magnetic field (Lorentz), a Hall probe, a "
             "B-H loop, and field-line maps from gradient, divergence and curl.",
        oracle="Lua applies the Mentat force; Havok finds the equilibrium angle the statics predict",
        bundles=[], needs=["force-application port", "field-line renderer"]),
    "S15": dict(
        name="Material test rig", kind="PORT", effort_days=4,
        what="A rod pulled until it stretches, yields and breaks; a creeping polymer; a worn pin; a composite beam. "
             "Done with spring and break-force constraints as a PROXY: no geometry ever changes shape.",
        oracle="Lua stress-strain port vs the Mentat tools offline; a true deformation test is blocked",
        bundles=[], needs=["constraint proxy", "real deformation (sigma-ground gap)"]),
    "S16": dict(
        name="Wet-chemistry bench", kind="SHOW", effort_days=2,
        what="Beakers whose liquid colour and a pH meter follow titration, buffers, solubility, kinetics and "
             "calorimetry computed by Mentat; a thermometer shows combustion enthalpy and colligative shifts.",
        oracle="none beyond provenance (Mentat computes, the bench displays)",
        bundles=[], needs=["live bridge", "liquid colour props"]),
    "S17": dict(
        name="Surface and wave tank", kind="BLOCKED", effort_days=None,
        what="A droplet contact angle and wind ripples on water. Needs a fluid solver in physical units; "
             "GWater2 (FleX) is position-based and not metric, so no quantitative test is possible yet.",
        oracle="n/a until a metric fluid exists", bundles=[], needs=["metric fluid solver"]),
}

# Build status, as of 2026-10-10. "BUILT" means: spec builder, Lua scene, independent grader, recorded run and offline
# tests exist under stage/gmod/experiments, and the grader passes on the dedicated server.
_STATUS = {
    "S01": "BUILT (stage/gmod/experiments, run.py S01): four metal spheres agree with Mentat free fall to 1e-6. The coin, the feather and the drag run are not built.",
    "S02": "BUILT: three launch angles; range, apex and flight time agree with Mentat to 5e-6. The straight-up throw is not built.",
    "S03": "BUILT: friction handed over from Mentat is realized to 3e-4 on the steady lanes; critical angle bracketed 44-46 deg around 45.",
    "S04": "BUILT: momentum conserved to 6e-4, Mentat collision outcomes agree to 1e-4. Sphere bounce height is not built.",
    "S05": "BUILT IN PART: the rolling race (solid sphere, solid disc, hollow sphere) and the inertia audit pass; the gear train and the hand tool are deferred.",
    "S06": "BUILT: pendulum periods within 0.3 % of Mentat's exact period, springs within 6e-4, damped spring within 0.4 %.",
    "S07": "NOT BUILT, AND NOT AN ENGINE TEST: no installed map has water; reclassified PORT.",
}
for _sid, _text in _STATUS.items():
    SCENARIOS[_sid]["status"] = _text


CLASSES = ("ENGINE", "PORT", "SHOW", "PREDICT", "LOOKUP", "GENERATOR", "BLOCKED", "OUT")
OUT_KINDS = ("ASTRO", "LARGE_SCALE", "RELATIVISTIC", "ATOMIC_QUANTUM", "THEORY", "TIMESCALE", "EXTREME_REGIME")

TOOLS: dict = {}
VERBS: dict = {}


def _add(table, names, cls, scn=(), kind=None, note=""):
    for n in names.split():
        if n in table:
            raise ValueError(f"duplicate ledger entry: {n}")
        table[n] = dict(cls=cls, scenarios=list(scn), kind=kind, note=note)


def T(names, cls, scn=(), kind=None, note=""):
    _add(TOOLS, names, cls, scn, kind, note)


def V(names, cls, scn=(), kind=None, note=""):
    _add(VERBS, names, cls, scn, kind, note)


# ======================================================================================
# 221 MCP tools
# ======================================================================================

# ---- ENGINE: Havok is an independent oracle ----------------------------------------
T("free_fall_time free_fall_velocity gravitational_potential_energy kinetic_energy",
  "ENGINE", ["S01", "S04"], note="vacuum formulas; Havok free fall with drag disabled is the oracle")
T("work_energy_analysis", "ENGINE", ["S01", "S03", "S05"],
  note="energy bookkeeping across a fall, a ramp and a rolling body")
T("viscous_flow_analysis", "PORT", ["S07", "S01"],
  note="Havok's drag is a game model, not physics, so it cannot judge Mentat's Stokes drag. The stage "
       "turns Havok's drag off (measured: ~9 % over a 3 m fall) and a Lua port applies Mentat's drag instead")
T("projectile_analysis projectile_flight_time projectile_max_height projectile_range "
  "procedure_projectile_trajectory", "ENGINE", ["S02"])
T("incline_analysis friction_stopping_distance", "ENGINE", ["S03"],
  note="validates the incline and stopping-distance FORMULAS against Havok, with Mentat's friction coefficient "
       "fed in through surface data")
T("friction_analysis", "PREDICT", ["S03"],
  note="supplies the coefficient Havok is handed, so Havok cannot validate it: the engine only shows what follows from it")
T("momentum", "ENGINE", ["S04", "S01"])
T("collision_analysis", "ENGINE", ["S04"],
  note="momentum and energy bookkeeping of the outcomes is the real engine test")
T("hertzian_impact_analysis", "PREDICT", ["S04"],
  note="supplies the restitution Havok is handed (a surface property); Havok cannot validate the number itself")
T("rotational_dynamics", "ENGINE", ["S05"], note="moments of inertia: Havok derives inertia from the hull")
T("buoyancy_analysis", "PORT", ["S07"],
  note="no water volume exists on the installed maps, so the engine offers no buoyancy to compare with: a Lua force port stands in")

# ---- PREDICT: math / conversion tools that build another scenario's prediction -----
T("solve_ode series_expansion compute_limit simplify_expr expand_expression factor_expression",
  "PREDICT", ["S06"], note="pendulum/spring: ODE, small-angle series, limit sin(x)/x, algebra on the period")
T("differentiate_expr solve_equation", "PREDICT", ["S02"], note="apex = zero of dy/dt; landing time = root of y(t)")
T("integrate_expr", "PREDICT", ["S03"], note="work against friction as an integral of force over distance")
T("solve_linear_system", "PREDICT", ["S04"], note="momentum + restitution as a 2x2 system for the outgoing speeds")
T("matrix_determinant matrix_inverse matrix_multiply matrix_eigenvalues", "PREDICT", ["S05"],
  note="inertia tensor eigenvalues = principal moments; rotation matrices: det=+1, inverse=transpose, composition")
T("gradient divergence curl", "PREDICT", ["S14"], note="field operators on the point-charge field the bench draws")
T("fourier_transform", "PREDICT", ["S10"], note="spectrum of the bell strike")
T("laplace_transform", "PREDICT", ["S13"], note="RC step response")
T("summation", "PREDICT", ["S12"], note="N-slit / grating phasor sum")
T("temperature_celsius_to_kelvin temperature_kelvin_to_celsius", "PREDICT", ["S08"])
T("eV_to_joules joules_to_eV hydrogen_like_energy_level", "PREDICT", ["S12"],
  note="energy-level and unit arithmetic behind the hydrogen emission lines")
T("joules_to_TNT tnt_to_joules", "PREDICT", ["S01"], note="impact energy of the falling bodies expressed in TNT")
T("energy_power_time", "PREDICT", ["S05", "S13"], note="E = P t for a motor and for a resistor")

# ---- LOOKUP / GENERATOR -------------------------------------------------------------
T("lookup_constant list_constants get_manifest list_materials convert_units parse_quantity percent_of "
  "element_atomic_data list_bodies solar_system_body named_star", "LOOKUP", ["S00"],
  note="reference data; Mentat convert_units(0.75 inch -> m) = 0.01905 also cross-checks Source's 1 HU")
T("density", "LOOKUP", ["S07", "S00"], note="material densities that decide float/sink in the pool")
T("youngs_modulus", "LOOKUP", ["S15", "S00"])
T("refractive_index", "LOOKUP", ["S11", "S00"])
T("melting_point boiling_point", "LOOKUP", ["S08", "S00"], note="phase thresholds for the furnace")
T("run_simulation list_simulation_scenarios", "GENERATOR", ["S01", "S02", "S05", "S08", "S10"],
  note="the 68 verbs build the trajectory bundles every replay scenario consumes")

# ---- PORT: GMod lacks the physics; Lua implements it at 1:1 -------------------------
T("blackbody_peak_wavelength blackbody_total_power procedure_stellar_blackbody thermal_contact_analysis "
  "thermoelectric_generator_analysis", "PORT", ["S08"],
  note="stellar inputs are out of scale; the same laws are tested on a hot iron ball")
T("carnot_efficiency", "PORT", ["S09", "S08"])
T("ideal_gas_pressure ideal_gas_volume natural_convection_analysis", "PORT", ["S09"])
T("speed_of_sound_in_ideal_gas acoustic_interface_analysis", "PORT", ["S10", "S09"],
  note="Source sound has no propagation delay: the port adds one")
T("critical_angle_for_tir thin_lens_focal_length thin_lens_image_distance snells_law_refraction_angle "
  "lens_magnification optical_waveguide_analysis", "PORT", ["S11"])
T("double_slit_fringe_spacing single_slit_first_minimum_angle diffraction_grating_angle "
  "interference_visibility_analysis photonic_bandgap_analysis em_wave_frequency em_wave_wavelength "
  "photon_energy_from_wavelength photon_energy_from_frequency procedure_photon_spectrum "
  "hydrogen_emission_wavelength rydberg_hydrogen_wavelength phosphor_decay_analysis radioactivity_analysis",
  "PORT", ["S12"], note="slit sources are microscopic but the pattern and the light are room-scale")
T("ohms_law_current ohms_law_voltage electrical_power power_dissipation_resistor rc_time_constant "
  "rl_time_constant rlc_resonant_frequency parallel_plate_capacitance capacitor_analysis diode_current "
  "electrical_resistivity", "PORT", ["S13"])
T("coulomb_force electromagnetic_force_analysis", "PORT", ["S14"],
  note="Lua applies the Mentat force to charged balls; Havok settles the equilibrium")
T("elastic_analysis plasticity_analysis stress_failure_analysis viscoelastic_creep_analysis", "PORT", ["S15"],
  note="constraint proxy only (spring, permanent set, break force, creep): no geometry changes shape")

# ---- SHOW: bench instruments --------------------------------------------------------
T("material_color_analysis", "SHOW", ["S12"], note="physical sRGB colour of a material rendered on a prop")
T("alloy_resistivity_analysis semiconductor_junction_analysis piezoelectric_actuator_analysis "
  "dielectric_polarization_analysis mobius_bimetallic_analysis pn_built_in_voltage depletion_width",
  "SHOW", ["S13"], note="device and material readouts on the circuit bench")
T("hall_effect_analysis hall_coefficient", "SHOW", ["S14"])
T("composite_bounds_analysis dislocation_strengthening_analysis wear_analysis", "SHOW", ["S15"],
  note="values only: microstructure inputs or a timescale a room cannot wait for")
T("diffusion_analysis", "SHOW", ["S09"])
T("acid_speciation_analysis titration_analysis buffer_ph weak_acid_ph molar_solubility solution_analysis "
  "boiling_point_elevation freezing_point_depression osmotic_pressure electrolysis_mass cell_potential "
  "electrochemistry_analysis reaction_kinetics_analysis reaction_enthalpy combustion_enthalpy_analysis",
  "SHOW", ["S16"], note="bulk solution chemistry at bench scale; no molecular physics is observed")

# ---- BLOCKED ------------------------------------------------------------------------
T("wetting_analysis wind_wave_analysis", "BLOCKED", ["S17"],
  note="needs a fluid solver in physical units; GWater2 is position-based and not metric")

# ---- OUT OF SCALE -------------------------------------------------------------------
T("escape_velocity orbital_period orbital_raise_energy orbital_velocity circular_orbit_velocity "
  "gravitational_force gravitational_binding_energy luminosity_to_mass_conversion_rate asteroid_analysis "
  "planet_moment_of_inertia_analysis mond_a0_constant mond_regime_classifier age_of_universe critical_density "
  "hubble_radius gravitational_redshift gravitational_time_dilation hawking_evaporation_time hawking_temperature "
  "isco_radius photon_sphere_radius schwarzschild_radius procedure_black_hole_profile unruh_temperature",
  "OUT", kind="ASTRO",
  note="gravitational_force works at room scale but is invisible: Mentat gives 2.67e-6 N between two 100 kg "
       "masses 0.5 m apart, 2.7e-9 of one mass's 980.7 N weight, far below anything Havok resolves")
T("lorentz_factor relativistic_energy relativistic_momentum relativistic_time_dilation "
  "relativistic_length_contraction relativistic_velocity_addition relativistic_energy_analysis "
  "procedure_relativistic_particle doppler_shift_factor mass_to_energy energy_to_mass",
  "OUT", kind="RELATIVISTIC")
T("baryon_vs_disc bekenstein_hawking_entropy entanglement_channel entanglements_to_pop_bubble "
  "eta_desi_band_check eta_value_report hde_dark_energy_density holographic_matching_mass "
  "holographic_dark_energy_analysis", "OUT", kind="THEORY")
T("atomic_angular_momentum de_broglie_from_kinetic_energy de_broglie_wavelength first_ionization_energy "
  "quantum_algorithm_analysis quantum_box_energy_analysis quantum_state_analysis quantum_tunneling_analysis "
  "qubit_hardware_analysis zeeman_effect_analysis magnetic_exchange_analysis constituent_behaviors_analysis "
  "material_inventory_analysis superconducting_gap_analysis superconductor_critical_field_analysis "
  "thermal_energy_per_molecule maxwell_boltzmann_most_probable_speed statistical_distribution carrier_mobility "
  "free_electron_density electron_mean_free_path intrinsic_carrier_density band_gap_ev semiconductor_band_gap "
  "band_dos_shape_analysis bond_angle bond_energy molecular_dipole_analysis fission_energy_release "
  "nuclear_binding_energy plasma_parameters_analysis", "OUT", kind="ATOMIC_QUANTUM")
T("nonlinear_optics_analysis", "OUT", kind="EXTREME_REGIME",
  note="Mentat's own default intensity is 1e13 W/m^2, a pulsed-laser regime a room never reaches")
T("light_travel_time", "OUT", kind="TIMESCALE", note="Mentat: 3 m takes 10.0 ns; nothing to watch")

# ======================================================================================
# 68 simulation verbs behind run_simulation
# ======================================================================================
V("collision_dynamics", "ENGINE", ["S04"])
V("projectile_motion vertical_launch", "ENGINE", ["S02"])
V("rolling_object rotational_dynamics", "ENGINE", ["S05"])
V("actuate_hand_tool", "ENGINE", ["S05"], note="DEFERRED: needs the hinge and gear constraint scenes (the pivot is discovered from Deckard contact geometry)")
V("tribology", "PREDICT", ["S03"], note="supplies the friction coefficient Havok is handed; the Archard wear part is value-only")
V("fluid_flow", "PORT", ["S07", "S01"], note="Stokes settling needs Mentat's drag applied by a Lua port; Havok's own drag is not physical")
V("water_state", "PREDICT", ["S07"], note="density(T) is fed to Source buoyancy; the 4 C maximum decides whether ice floats")
V("drop_object terminal_velocity_drop", "PORT", ["S01"],
  note="the fall includes air drag, so it matches Havok only once a Lua drag port stands in for the engine's")
V("atmospheric_profile", "PREDICT", ["S01"], note="air density that sets the drag in the drop")
V("material_profile material_full_profile", "PREDICT", ["S03", "S04", "S07", "S08", "S10"],
  note="the property sheet (density, friction, restitution, sound speed, conductivity) that configures every prop")
V("contact_conduction thermal_response drag_heating_drop thermoelectric", "PORT", ["S08"])
V("thermal_statistics", "OUT", kind="ATOMIC_QUANTUM", note="molecular speed distributions")
V("diffusion_transport", "SHOW", ["S09"])
V("acoustics strike_bell", "PORT", ["S10"])
V("optical_dispersion optical_fiber", "PORT", ["S11"])
V("hydrogen_spectrum nuclear_decay", "PORT", ["S12"], note="emission lines and Geiger clicks")
V("subsurface_scattering", "SHOW", ["S12"])
V("transmission_line piezoelectric_material semiconductor_device", "SHOW", ["S13"])
V("charged_particle", "PORT", ["S14"])
V("magnetic_hysteresis magnetic_material", "SHOW", ["S14"])
V("elastic_solid structural_response viscoelastic_material", "PORT", ["S15"], note="constraint proxy only")
V("composite_material metallurgy corrosion_attack organic_material", "SHOW", ["S15"])
V("chemistry_lab chemical_reaction solution_chemistry combustion_flow fuel_ignition", "SHOW", ["S16"])
V("fluid_dynamics", "BLOCKED", ["S17"], note="viscosity and surface tension of liquids: display needs a metric fluid")
V("high_altitude_descent supersonic_projectile", "OUT", kind="LARGE_SCALE",
  note="km altitudes and Mach 2 ranges of kilometres; the Source map is about 600 m across")
V("orbital_velocity planetary_surface asteroid_shape black_hole stellar_fusion", "OUT", kind="ASTRO")
V("relativistic_motion mass_energy", "OUT", kind="RELATIVISTIC")
V("atomic_coupling condensed_matter intermolecular_forces matter_wave molecular_bond nucleon_masses "
  "plasma_physics quantum_algorithms_demo quantum_dot quantum_gates quantum_report quantum_tunneling "
  "superconductor", "OUT", kind="ATOMIC_QUANTUM")
