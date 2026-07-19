# Known Gaps — the greppable ledger

Per GOLDEN_RULES: every `# PHYSICS_GAP:` / `# GEOMETRY_GAP:` tag worth
tracking above file level lands here, so gaps get reviewed instead of
quietly accumulating.

## PHYSICS_GAP: electrolyte/soil corrosion kinetics not modeled (2026-07-15)

`sigma_ground/field/interface/corrosion.py` computes DRY-AIR Wagner oxidation
only. Corrosion in an electrolyte (soil, seawater, acid/caustic service) is an
electrochemical problem — pH-dependent rate, O2 mass transport, soil
resistivity cell — with no citable *general* model small enough to wire in
honestly. What exists instead: `environment_assessment()` places a metal in
its CITED regime (pH low-corrosion windows — Roetheli 1932 Zn, Whitman 1924
Fe, Pourbaix 1966 Al; ASTM G57 soil-resistivity corrosivity scale) and the
`corrosion_attack` verb states "dry-air kinetics" explicitly instead of
silently ignoring the environment words.

- Future quantitative layer: digitize Romanoff, "Underground Corrosion",
  NBS Circular 579 (1957) field data (zinc/steel mass-loss per soil type),
  and/or the Roetheli 1932 rate-vs-pH curve for zinc.
- Regression sentinels: `sigma_ground/field/interface/test_corrosion.py::
  TestEnvironment`, `sigma_ground/materia/tests/test_translator.py`
  (duration + environment extraction).

## PHYSICS_GAP: RevoluteJoint swing loop unstable under orientation-coupled external torque (2026-07-15)

A wind-loaded rotor mounted on a `RevoluteJoint` tumbles: the joint's
SHAKE/RATTLE swing-correction loop is exponentially unstable when an
EXTERNAL torque feeds back on axis-tilt error (the wind torque responds to
orientation; the correction lags one substep; loop gain > 1 in a mid-omega
band). Transverse error grows ~x2.5/step from float noise (measured from
1e-117 up to tumble in ~280 steps), at every dt tried (1/960..1/3840) and
solver iteration count (10..30 — more iterations made it WORSE, suggesting
stale-Jacobian over-solving).

Isolation is decisive: the identical aero model
(`dynamics/mechanisms/wind.py`) on an ideal hand-projection spins up
monotonically to the closed-form terminal tip speed with zero drops.
All prior joint loads were solver-internal rows (motors, limits,
couplings) — orientation-coupled external torque is a genuinely new load
regime the RATTLE scheme hasn't been validated against.

- Workaround in use: `dynamics/mechanisms/bearing.py` (`RigidBearing`,
  ideal projection, energy-ledgered) — also the physically better
  abstraction for a stiff nacelle bearing assembly.
- Regression sentinel: `tests/test_mechanism_wind.py::
  test_wind_rotor_on_revolute_joint_stays_swing_stable` (strict xfail —
  fixing the solver loop makes it XPASS and forces this entry's review).

## GEOMETRY_GAP: gear module size uncited across all gear-train demos (2026-07-15)

Every `InvoluteGear`-based demo (`record_gear_train_spin`, `record_gear_mesh_spin`,
`record_escapement_clock`, `record_clock`) renders real, cited TOOTH COUNTS
(Kelly, *A Practical Course in Horology*, 1944 — center wheel 80/12, third
75/10, fourth 80/8, escape 15, per `sigma_ground/blueprint/catalog/
kelly_1944_watch_going_train_18000bph.md`) but an ESTIMATED module (physical
tooth size in mm/m). Kelly's text states tooth counts and train ratios only
— period horology sources of this kind describe a movement's timekeeping,
not its manufacturing dimensions, so no module is available to cite from the
same source. `sigma_ground/blueprint/validate.py:55-57` already reports this
honestly as a `# GEOMETRY_GAP:` (`ValidationReport.add_gap`) rather than
silently defaulting — center-distance/mesh cross-checks are skipped
wherever a module is missing (`validate.py:73-76`), never faked. Every
affected gear leaf's `"source"` string says `"module [estimated]"` explicitly
(`sigma_ground/radiance/trajectory.py:633,640,1302,1395,1416`) so the
render-facing citation stays honest even though the underlying spec's
`validate()` gap is Python-internal and doesn't itself reach the scene JSON.

- Ratios/timekeeping are unaffected — every derived rate (gear ratios,
  minute-hand rate, tick cadence) depends only on tooth COUNTS, which are
  fully cited; module only affects rendered tooth SIZE, not any simulated
  physics.
- Future fix: extract a real module from a source that states physical
  dimensions (a manufacturer's spec sheet, an expired patent's claim
  drawing with dimensions, or KHK Gears' technical reference for a period-
  correct module series) and re-run `validate()` — the center-distance
  cross-check activates automatically once every meshing gear's
  `module_mm` is populated, no code change needed.
- Regression sentinel: none needed — this is a data-completeness gap, not
  a behavior a test could regress. `blueprint/validate.py`'s existing gap-
  reporting tests (`tests/test_blueprint_validate.py` if present, else
  covered implicitly by every `record_*` test's `"estimated" in src.lower()`
  assertion, e.g. `tests/test_trajectory_clock.py::
  test_every_gear_leaf_carries_citation_and_estimated_module_flag`) already
  hold the CURRENT honest state; review this entry once a real module is
  ever sourced and those assertions need to flip.

## GEOMETRY_GAP: hand-tool actuation dims are representative, not cited (2026-07-19)

`record_hand_tool_actuation` / `deckard/hand_tools.py`'s catalog (pliers,
tongs, scissors, tweezers, nutcracker, shears, wire cutters) has NO real
blueprint on file for any specific product — every jaw/handle length, pivot
half-angle, swing amplitude, and actuation speed is a `Choice`-flagged
demo value, same tier as `record_motor_spin`'s "placeholder disc" (not the
clock's cited Kelly-1944 going train). What IS real: the pivot geometry
(narrow in-plane, deep along the rotation axis) is empirically tuned so
`infer_dodoxel_joints` genuinely DISCOVERS a revolute from contact
geometry — the joint TYPE and AXIS are earned, not declared; only the
tool's overall dimensions are representative.

- Physics is unaffected in kind: mass/inertia/joint dynamics are exactly
  computed from whatever dimensions are given (dodoxel moment sums, the
  same closed-form machinery gated in the dodoxel arc) — only the absolute
  scale/proportions of a specific catalog entry are uncited.
- Future fix: source real dimensions for one named product (a patent
  drawing or manufacturer spec, same sourcing tier as the clock's Kelly
  1944 citation) and add it as a `blueprint/catalog/` entry; the builder
  already accepts `pitch_m`/`rest_half_angle_deg` overrides, so a cited
  entry slots in without a code change to `build_hand_tool_field` itself.
- Regression sentinel: `tests/test_hand_tools.py::
  test_geometry_is_a_choice_with_a_stated_reason` holds the current honest
  state (asserts the Choice's description explicitly says "no blueprint
  mechanism pins a real product"); review once a real catalog entry lands.

## PHYSICS_GAP: bell-strike acoustics is a single-mode, uncited-decay model (2026-07-19, revised 2026-07-19)

`record_bell_strike` computes real, cited numbers on TWO separate physics
paths, not one: `ring_frequency` (`field.interface.acoustics.
ring_frequency`, the same formula the `acoustics()` Materia verb already
cites for bells, f_ring = v_L/(π·d)) for the SUSTAINED tone, and the
Johnson-Thornton/Hertz impact model (`field.interface.impact` —
`impact_energy_partition`, `impact_sound_frequency`, `hertz_contact_
duration`) for the strike itself. **Revision note**: this second path was
ALREADY a real, existing part of the physics library (`record_fall`'s own
bounce restitution already uses the same `coefficient_of_restitution`
machinery) but the first cut of this demo did not use it — it computed
loudness from raw kinetic energy via an invented Choice-scaled curve
instead. Caught when the Captain asked "I don't even know if our
simulations are considering sound already" — prompted an audit that found
the unused-but-real `impact_energy_partition`/`impact_sound_frequency`
functions. Fixed: the WAV's amplitude now scales from
`impact_energy_partition`'s real `E_dissipated_J` (the Johnson-Thornton
energy-conservation fraction that becomes deformation/heat/sound — NOT the
full impact KE, part of which stays as the stone's rebound motion and
never becomes sound at all), and the Hertzian impact "click" frequency/
contact duration are now reported as their own real, distinct numbers
alongside the sustained ring. What's still an explicitly flagged
SIMPLIFIED_MODEL:

- Real bells ring with several inharmonic partials (hum/prime/tierce/
  quint/nominal), each its own decay rate; this synthesizes exactly ONE
  sustained mode (the click is reported as a number, not a second audio
  layer). `ring_frequency` itself is the correct circumferential-wave
  frequency for a thin cylindrical shell, not necessarily the bell's
  dominant PERCEIVED pitch (real bells are dominated by lower bending
  modes) — consumed exactly as the existing library documents it, not
  re-derived or corrected here.
- The decay time constant (`tau`) is still a `Choice`-flagged guess (no
  material Q-factor table on file). The amplitude-vs-energy CURVE's
  absolute scale is also still a `Choice` — the ENERGY it scales is now
  real (`E_dissipated_J`), but no first-principles energy-to-SPL/
  radiation-efficiency model exists on file to fix the scale itself.
- The bell's collision envelope for the stone strike is its bounding
  SPHERE (`PhysicsParcel.radius` from `Cone.bounding_radius()`), not the
  true conical SDF — an honest, already-established engine-wide
  approximation (every non-trivial shape's pairwise contact uses its
  bounding sphere), not something invented for this demo.
- `coefficient_of_restitution`/`impact_energy_partition`/
  `impact_sound_frequency` are SELF-contact formulas (one `material_key`)
  — used here with the bell's material only, the same simplification
  `record_fall`'s own bounce already makes (ignoring the falling body vs.
  floor material split); a real two-material contact would need
  `reduced_modulus_pair` (exists, unused) threaded through a new
  dissimilar-body wrapper — not built here.
- Atmospheric propagation (`bell_acoustics.propagate_in_air`) models only
  inverse-square (spherical spreading) amplitude loss; frequency-dependent
  atmospheric absorption is NOT modeled.
- Every bell/stone/observer parameter left at its default (which EVERY
  current text request does — Materia's translator has no slot extraction
  yet for size/mass/distance) is now an exposed `Choice` too
  (`scenario_choice`), not a silent default — the same gap class as the
  clock-dial aluminum incident, caught by the same Captain-raised concern
  this round.
- A real bug WAS found and fixed here, not just flagged: a small/stiff
  bell can ring above 22050 Hz's default 11025 Hz Nyquist limit (a 150 mm
  iron bell rings at ~12.6 kHz) — undersampling would ALIAS to the wrong
  audible pitch, not just lose fidelity. `record_bell_strike` now sizes
  the WAV's sample rate to the actual `ring_frequency` with margin.
- Future fix: a real per-material Q-factor / decay-time table, a real
  energy-to-SPL calibration, a two-material contact model, and (much
  bigger) a multi-partial modal model, would each need their own gated
  phases — not attempted here per this project's "closed form or don't
  build it" discipline; every remaining simplification is stated as a
  limit, not solved around.
- Regression sentinels: `tests/test_bell_strike.py::
  test_audio_artifact_is_a_valid_wav_above_the_nyquist_safe_margin` (the
  aliasing bug), `::test_dissipated_energy_matches_the_real_impact_
  partition_model` (amplitude tracks the REAL Johnson-Thornton energy, not
  raw KE), `::test_impact_click_frequency_is_real_and_distinct_from_ring_
  frequency`, `::test_every_defaulted_scenario_parameter_is_a_recorded_
  choice`, and `::test_provenance_cites_the_solver_not_a_scripted_impact`
  (asserts "SIMPLIFIED_MODEL" stays in the citation string).

## PHYSICS_GAP: windmill's bearing-to-gearset coupling is load-blind (2026-07-16)

`dynamics/mechanisms/bearing_gear_coupling.py` (`BearingGearCoupling`)
drives a downstream `RevoluteJoint`'s motor from a `RigidBearing`-mounted
rotor's own spin rate each step (`follower.motor_speed = -ratio*
bearing.omega()`, the orchestration idiom already used by `MainspringState`/
`Escapement`). This is kinematically correct — the follower tracks the
commanded ratio — but the rotor pays NOTHING for it: `RigidBearing.project()`
has no path to accept a reaction torque (it only ever discards transverse
KE, ledgered as `absorbed_energy_j`), and the follower's own motor
manufactures whatever torque it needs (up to its cap) out of nothing. The
rotor spins up to the exact same closed-form terminal speed whether or not
anything downstream is "loaded" — measured directly in
`tests/test_mechanism_bearing_gear_coupling.py::
test_coupling_is_load_blind_matching_the_flagged_gap` (a bearing+follower
run and an isolated bearing-alone run produce identical `omega()` traces to
1e-6 relative).

- Why not fixed now: an energy-honest version needs a reaction torque fed
  back through `RigidBearing.project()` (which keeps the axial component of
  `angular_velocity` and only strips the transverse part, so an externally-
  applied AXIAL reaction would survive `project()`'s cleanup like the aero
  torque does today) — but computing the follower's actual delivered torque
  that substep needs it BEFORE the joint solve runs, one substep ahead of
  when it's known. Real new solver-composition work, not justified to
  unblock a first drivetrain slice.
- Future fix: a lagged (one-substep) or analytically-estimated reaction
  torque applied via the same `external_forces` `(force, torque)` callback
  slot `RotorWind` already uses on the rotor.
- Regression sentinel: `tests/test_mechanism_bearing_gear_coupling.py`
  (all three tests hold the CURRENT load-blind behavior; an energy-honest
  fix should make `test_coupling_is_load_blind_matching_the_flagged_gap`
  fail, which is the intended trigger to review/rewrite this entry).
