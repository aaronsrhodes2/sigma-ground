# Known Gaps — the greppable ledger

Per GOLDEN_RULES: every `# PHYSICS_GAP:` / `# GEOMETRY_GAP:` tag worth
tracking above file level lands here, so gaps get reviewed instead of
quietly accumulating.

Cross-index: the unification program's full gap census (engine defects, missing capabilities such as the
nuclear-matter EOS, and open decisions) is [misc/MENTAT_GAPS_UNIFICATION.md](misc/MENTAT_GAPS_UNIFICATION.md).

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

## ✅ CLOSED 2026-10-10 — PHYSICS_GAP: SPH continuum bridge books no viscous heating and no PdV work (2026-10-10)

**Closed the same day** (unification Stage 8): `continuum_step` books the work of the internal SPH forces
(pressure + artificial viscosity) as internal energy pair by pair — Δuᵢ = ¼Δt Σ_{both evaluations} Σⱼ gᵢⱼ·(v̄ᵢ − v̄ⱼ),
using the same pair forces gᵢⱼ the kick applied and the step-averaged velocities. Since mᵢgᵢⱼ = −mⱼgⱼᵢ this conserves
K + U_int **exactly** (to roundoff) for those forces; gravity and uniform g stay in the velocity-Verlet potential ledger
(O(dt²)). Pair forces are recorded inside the existing loops (no extra sweep). eos="tait" heats the parcel
(dT = du/c_p); eos="isothermal" holds T and books the work to `scene.energy_bath_J`; new eos="ideal" (P = ρRT/M,
u = c_vT) makes gas expansion testable. `scene.energy_ledger()` reports kinetic + internal + bath + softened
self-gravity + uniform g. Sentinels: `dynamics/test_continuum_energy.py` (10 tests). A first, simpler version used the
end-of-step rate with the pre-kick velocity; a free-expanding gas then gained 24 % more K than it lost in heat — caught
by Stage 8 before release. `scene.energy_equation = False` restores the old behaviour.

Original entry:

`sigma_ground/dynamics/continuum.py::continuum_step` step 9 updates parcel temperature
from conduction (`_heat_conduction`) and `heat_source_W_kg` only. Two terms the
module docstring promises are missing: the adiabatic compression term
`(P/ρ²)(dρ/dt)/cp`, and the conversion of artificial-viscosity dissipation
(`_sph_viscous_force`, Monaghan Π_ij) into internal energy. Consequence: the
SPH side has no first law — mechanical energy removed by viscosity vanishes from
the ledger (Phase E cloud report: SPH mechanical energy −387% with nothing
booked as heat), and compression never warms a parcel. Found while building the
fluid→thermo bridge of the unification program (misc/UNIFICATION_PROGRAM.md):
the entropy-production test σ_s = (Φ_visc + q²/κT²)/T ≥ 0 cannot be run until
dissipation is booked.

- Fix direction: per-pair viscous heating `du_i/dt = ½ Σ_j m_j Π_ij v_ij·∇W_ij`
  (Monaghan 1992 eq. 2.30-style) and `du/dt = −(P/ρ²) dρ/dt` with
  `dρ/dt` from the SPH continuity sum; then `dT = du/cp`. Add an energy ledger
  (K + U_grav + U_int) to `phase_e.diagnostics` and gate on conservation.
- Regression sentinel to add: an isolated viscous collision of two parcels must
  conserve K + U_int to integrator tolerance.

## PHYSICS_GAP: NBodySystem.total_energy() ignores Plummer softening (2026-10-10)

`sigma_ground/field/interface/nbody.py::NBodySystem.total_energy` sums the
unsoftened pair potential −G m_i m_j / r while `compute_accelerations` applies
Plummer softening r² → r² + ε² whenever `softening_m > 0`. The integrator then
conserves K + U_soft, not the reported quantity: in unification pilot P5
(N = 64, ε = 0.02R, 4 t_ff) the reported "energy drift" was 0.96 % while the
softened Hamiltonian drifted 2.4e-5. Every N-body energy-conservation claim in
the repo that used softening is a statement about a non-conserved bookkeeping
number. Found by the adversarial verification of the unification pilots.

- Fix direction: give `total_energy()` a `softened=True` path that uses
  `sqrt(r² + softening_m²)`, and make `phase_e.diagnostics` report both.
- Regression sentinel to add: a softened two-body orbit must conserve the
  softened energy to integrator tolerance over many periods.
- Decision for the Captain: which energy the public API reports by default.

## PHYSICS_GAP: viscoelastic relaxation ratio uses a silent 1800 K melting point for every metal (2026-10-10)

`sigma_ground/field/interface/viscoelasticity.py::_relaxation_ratio` computes
E_R/E_U = 1 − 0.7·(T/T_melt)² with `T_melt = MATERIALS[m].get('melting_point_K', 1800)`,
but `surface.MATERIALS` has no `melting_point_K` key, so every material silently gets
1800 K. Mentat's own `melting_point` tool (CRC) knows the real values (Al 933.47 K).
Effect: `storage_modulus`, `loss_modulus`, `loss_tangent` and everything downstream
are too stiff for low-melting metals (Al at 900 K: E_R/E_U = 0.825 vs ≈ 0.35 with
its real T_melt). Two further items in the same module: the relaxation-ratio cap
`_RELAX_RATIO_MAX = 0.7` is an uncited heuristic, and the module docstring's §5
gives Maxwell-element moduli while the functions implement the standard linear solid.
Found by unification Stage 7 (`scripts/unification/stage7_maxwell_crossover.py`).

- Fix direction: read T_melt from the same CRC table the `melting_point` tool uses;
  cite or derive the 0.7 cap; align the docstring with the SLS code.
- Regression sentinel to add: `_relaxation_ratio('aluminum', 900)` must use 933.47 K.

## PHYSICS_GAP: Eyring liquid-metal viscosity is 10³–10⁶× too high (2026-10-10)

`sigma_ground/field/interface/fluid.py::eyring_viscosity` (η = (h N_A/V_m)·exp(0.408·E_coh/RT)) returns, at each
metal's melting point, aluminum 1170 Pa·s, copper 10.8 Pa·s, iron 4.06 Pa·s. Measured liquid-metal viscosities near
melting are a few mPa·s (order of magnitude; Assael et al. 2006, J. Phys. Chem. Ref. Data 35, 285 give the Al and Fe
reference correlations — exact values not re-fetched in this session). The docstring claims ±30 %. Any caller using
liquid-metal viscosity (Reynolds numbers, drag in melts, η/s comparisons) inherits the error. Found by unification
Stage 9b while placing matter on the η/s axis.

- Fix direction: re-derive the activation energy for viscous flow of metals (it is a small fraction of E_coh, not
  0.408 of it) or fall back to the measured reference correlations with a `PHYSICS_GAP` tag; add a regression test
  against Assael 2006 at T_melt.
- Decision for the Captain: derived-with-error vs measured-with-citation.


## PHYSICS_GAP: the N-body engine's 1PN GR corrections have reversed signs — Mercury precesses backwards (2026-10-10)

`sigma_ground/field/interface/nbody.py`, `compute_accelerations`:

- **`gr_1pn`** (the layer behind the legacy `include_gr=True`) builds the textbook bracket
  (4GM/r − v²)n + 4(n·v)v on `r_hat = r_ij/r`, which points from the body *toward* the attractor
  (`r_ij = x_j − x_i`). The textbook n points outward (PPN test-particle Lagrangian; Iorio,
  arXiv:gr-qc/0405053, eq. 1). The whole correction is therefore minus GR: fitted coefficients
  (−4, −1, −4) against (4, 1, 4), and **Mercury's perihelion moves −42.997″/cy against +42.98″/cy**
  (Will 2014, Living Rev. Relativ. 17, 4).
- **`eih_cross`** (live in the `jpl_de440` predictor, `rolling_shootout.py`) has a correct radial
  bracket but dots `r_ij = x_j − x_i` into (4vᵢ − 3vⱼ), where EIH uses n_ij = (xᵢ − xⱼ)/r. The velocity
  cross-term is reversed: Mercury −71.66″/cy, and the precession changes under a boost (+119 % for
  30 km/s along the perihelion velocity; secular de = −2.45 u/orbit along x). With only that sign
  flipped back, the same code precesses +6πGM/(c²p) in every frame.
- `gr_1pn` uses absolute vᵢ, so even with the sign fixed it holds only in the attractor's rest frame
  (documented as "single-body", but not the frame dependence).
- `forest_ruth_step` is 4th-order and symplectic only for position-only forces; with the
  velocity-dependent layers on it converges at first order in dt (measured ratio 1.997).
- **Why no test caught it:** `test_physics_stability.py` asserts only that GR moves Mercury by more
  than zero; `test_nbody.py::test_eih_two_body_reduces_to_schwarzschild_at_leading_order` compares
  magnitudes at n·v = 0 with a 1e-4 tolerance, which hides the 1.5e-7 disagreement and cannot see
  the cross-term at all. No test asserts the sign or the size of the 43″/cy.

Found by unification round 2, experiment U9 (`scripts/unification/round2/U9/`); confirmed by three
independent skeptics with their own integrators (DOP853, classical RK4), all reproducing −6.000 u and
−10.000 u where u = πGM/(c²p).

- Fix direction (not applied — the Captain's call): `gr_1pn` bracket on `n = −r_ij/r`; `eih_cross`
  cross-term on `−r_ij`; document `gr_1pn`'s rest-frame validity; note the integrator order.
- Regression sentinels to add: Mercury two-body perihelion advance = +42.98″/cy within 1 % for both
  layers; `eih_cross` precession unchanged within 1 % under a 30 km/s boost along x, y and z.
- Downstream: every `include_gr=True` and `eih_cross=True` result, including the DE440 shoot-out
  scores, carries the reversed 1PN term. The orbit adapters `adapters/base.py::sg_propagate_orbit` and
  `adapters/integrator.py::sg_nbody` default to `include_gr=True`, so callers get it without asking.

## PHYSICS_GAP: SPH buoyancy is 1/3 of physical, and SPH heat conduction divides by density twice (2026-10-10)

`sigma_ground/dynamics/continuum.py`:

- **`_buoyancy_force`** uses `parcel.thermal_expansion`, the cascade's *linear* α_L, where the
  Boussinesq density perturbation needs the volumetric β = 3α_L. Measured through `continuum_step`
  on a lone parcel: 0.322–0.350 of the correct buoyancy for 10 cubic crystals (0.966–1.051 with a
  volumetric control). It is also applied under `eos="ideal"`, where the SPH pressure gradient
  already produces thermal buoyancy, so ideal-gas runs count it twice with the carrier solid's α_L.
- **`_heat_conduction`** sets dT/dt = rate/(ρᵢcpᵢ), but the pair sum is already du/dt in W/kg
  (Brookshaw 1985; Monaghan 1992; Jubelgas, Springel & Dolag 2004 eq. 17). Conduction therefore runs
  ρ_sph[SI] times too slow (iron 1.16e-4 of the Fourier rate, copper 1.02e-4; the formula without the
  extra factor gives 0.91), scales as ρ⁻² instead of ρ⁻¹, and is not antisymmetric between
  neighbours of unequal density: two iron parcels at ρ = 2524 and 3183 lose or gain 0.724 W of
  3.50 W depending on which is hotter. The docstring credits Cleary & Monaghan 1999, whose harmonic
  mean of κ the code does not use.
- Consequence for the energy ledger added this session: `energy_ledger()` closes exactly for the
  mechanical work (the 10 sentinels hold), but with `heat_conduction=True` and unequal neighbour
  densities the conduction step itself does not conserve heat.

Found by unification round 2, item U14G, which replaced the blocked Rayleigh–Bénard onset test
(`scripts/unification/round2/U14G/`); confirmed by three skeptics and a recheck.

- Fix direction (not applied — the Captain's call): β = 3α_L (or a measured β for liquids), gated
  off for `eos="ideal"`; dT/dt = rate/cpᵢ; docstrings to match.
- Regression sentinels to add: lone-parcel buoyancy = 3α_L ΔT g; two-parcel conduction rate against
  Fourier's law and exactly antisymmetric for unequal densities.

## DATA_GAP: Mentat's Sun mass pairs an IAU GM with an obsolete G (2026-10-10)

`sigma_ground/mcp/tools/astronomy.py`, `solar_system_body('sun')` returns 1.98892e30 kg, labelled "IAU
2015 nominal". It equals the IAU 2015 nominal GM☉ = 1.3271244e20 m³/s² (IAU 2015 Resolution B3,
arXiv:1510.07674) divided by the obsolete G = 6.67259e-11. With Mentat's own G = 6.6743e-11 the product
GM is 2.57e-4 high (Mentat `simplify_expr`), which is why the GR perihelion formula gives 42.997″/cy
instead of 42.98″/cy. Flagged in round 2 (U9 physics skeptic) and confirmed through Mentat.

- Fix direction (not applied): store GM☉ as the primary quantity and derive M☉ = GM☉/G.

## PHYSICS_GAP: N-body gravitational-wave damping uses r⁻⁴ and rounds small losses to zero (2026-10-10)

`sigma_ground/field/interface/nbody.py::_apply_gw_damping` computes the Peters (1964) power with r⁴ in the denominator; Peters
has r⁵. As coded dE/dt has units of W·m and the loss is r[m] times too strong: PSR J1738+0333 decays 1.73 × 10⁹ times faster
than Peters, and Earth–Sun radiates 2.9 × 10¹³ W instead of 196 W. Second, the loss is applied as a per-step velocity factor
`damp = 1 + dE/dt·dt/KE_rel`; any fractional loss below the half-ulp of 1.0 (5.55 × 10⁻¹⁷) rounds to exactly 1, and the
centre-of-mass recompute that follows then injects a systematic spurious decay (8.6–9.2× Peters at dt = Pb/2000, scaling with
step count). `test_physics_stability.py` encodes the same r⁻⁴ law (docstrings "~3e13 W") and asserts only inequalities;
`test_nbody.py` asserts only that energy decreases. Present since the initial release. Found by unification round 2, U1.

- Fix direction (not applied): r⁵; accumulate the GW loss and apply it when it exceeds ~10⁻¹³ of KE_rel (or evolve the orbital
  elements); skip the recompute when `damp == 1`.
- Regression sentinel to add: J1738+0333 Ṗ_b within 2 % of Peters (−2.75 × 10⁻¹⁴ s/s with Mentat leaves).

## PHYSICS_GAP: SEMF asymmetry penalty is 4× too small (2026-10-10)

`sigma_ground/field/interface/element.py` (binding energy) writes the asymmetry term as a_a(A − 2Z)²/(4A) with a_a = 23.29 MeV.
That coefficient belongs to the (A − 2Z)²/A form, which `stable_mass_number` in the same file assumes. Over the 136 nuclides with
A ≥ 12 in `inventory/data/isotopes.json` the rms binding-energy error is 6.82 % as coded and 0.78 % with /A (Pb-208 +8.8 % →
−1.1 %, U-238 +11.6 % → −0.3 %). A second, different SEMF coefficient set is hard-coded inside the Mentat tool
`mcp/tools/nuclear.py` (rms 1.14 %, He-4 −19 %), so the repo has two sources of truth. Found by round 2, N1 (nuclear-matter plan
survey); the plan's module M3 replaces both with one fitted SEMF.

## PHYSICS_GAP: Mentat's symbolic tools misread variables named E, I, N, S, O or Q (2026-10-10)

`sigma_ground/mcp/tools/symbolic.py`: `differentiate_expr`, `solve_equation` and `integrate_expr` parse the expression before
creating the variable symbol, so E (Euler's number), I, N, S, O and Q become sympy objects and the answer is silently wrong.
Live: `differentiate_expr('E**2','E')` → `0`; `solve_equation('E**2 - 4','E')` → `[]`; `integrate_expr('E','E')` → `E*E`.
This matters under PLATINUM §21: physicists name energy E. Found by round 2, U1; reproduced live. Fix direction (not applied):
`sp.sympify(expr, locals={variable: sp.Symbol(variable)})`. Sentinel: the three calls above with E give `2*E`, `[-2, 2]`, `E**2/2`.
`mcp/tools/mathx.py` has the same defect in `compute_limit`, `series_expansion` and `summation`
(live: `compute_limit('S**2','S','2')` fails to parse, while the same call with x returns 4).

## PHYSICS_GAP: Mentat `elastic_analysis` reports fixed moduli for every material (2026-10-10)

`sigma_ground/mcp/tools/materials_strength.py`: the `moduli_from_lame` entry is computed from fixed Lamé parameters (50, 30 GPa)
and `is_yielded_at_200MPa` from a fixed 250 MPa yield, regardless of `material_key`. Every material reports K = 70 GPa,
E = 78.75 GPa, G = 30 GPa inside a per-material analysis, while the same call's `hydrostatic_stress` gives iron K = 168.4 GPa.
Found by round 2, U8; confirmed by three skeptics.

## PHYSICS_GAP: SPH engine — timestep, ideal-gas temperature lag, floor contact, coincident parcels (2026-10-10)

`sigma_ground/dynamics/continuum.py` and helpers (round 2, U2, U7, U11, U13):

- `cfl_timestep` takes the signal speed from each parcel's cascade sound speed (iron 5942 m/s; the longitudinal c_L including
  shear) even for `eos="isothermal"`/`"ideal"`, where the step uses `scene.sound_speed_m_s` or √(γRT/M), and has no
  acceleration criterion. A c_s = 1 m/s gas gets dt 4354× too small; "N steps at cfl_timestep" protocols cover negligible
  physical time (U11: 2000 steps = 0.014 H/c_s; U13: 3 × 10⁻⁴ t_ff).
- The same c_L feeds the artificial-viscosity c̄, so Monaghan viscosity in Tait runs is 23 % (ice) to 28 % (iron) stronger than
  prescribed (the Tait operator carries long waves at 0.99 √(K/ρ₀)).
- With `eos="ideal"` both force evaluations of a step use the pre-update temperature (T is updated after the kick and the
  end-of-step force is cached), so the coupled (x, v, u) scheme is first order and not time-reversible (Loschmidt echo
  5.35 × 10⁻⁴ R → 2.68 × 10⁻⁴ R under dt halving, against 2 × 10⁻¹⁴ R isothermal). The viscous booking is first order in dt
  for the same reason (documented as the standard SPH approximation).
- Floor contact uses the rigid-body `resolve_sphere_plane` (restitution 0.5 plus a position pull), so fluid resting on the
  floor never comes to rest, and in a planar column corrected parcels become bitwise coincident; `fluid/kernel.py::grad_W`
  returns zero below the Planck length, so coincident parcels never separate.
- `seeding.to_continuum` cannot pass `ideal_molar_mass_kg_mol`/`ideal_gamma`, so `eos="ideal"` is unreachable through the adapter.
- Documentation: the `_sph_viscous_force` docstring carries a spurious factor h (the code is Monaghan 1992's form).

Fix direction (not applied): EOS signal speed plus a force criterion in `cfl_timestep`; temperature at n + ½ for the ideal EOS;
an SPH wall boundary instead of the rigid-body resolver; adapter pass-through.

## PHYSICS_GAP: σ couples to matter through five incompatible laws; the thermal chain freezes density (2026-10-10)

Round 2, U12 (σ propagated through the material chain): e^σ reaches matter mainly through nuclear mass (d ln ρ/dσ = 0.990),
but `thermal.sound_velocity` and `debye_temperature` keep the σ-free density, so Θ_D(σ) has the wrong sign and
`specific_heat_j_kg_K` misses the 1/m_nucleon(σ) factor; `phase_transition.lindemann_melting_estimate` is right only by a
cancelling σ-invariant atomic mass; `atmosphere._AR_MASS_KG` is σ-invariant while N₂/O₂/CO₂ follow the QCD law. Cohesive
energy E_coh(σ) is coded three incompatible ways (ZPE law with derived f_zpe; ZPE with fixed 0.01; full QCD fraction), and
plasticity/grain modules scale by e^σ directly. Overall the repo carries at least five σ-mass conventions (uniform QCD fraction
0.990; inventory atom law, composition-dependent, 0.986 for Fe-56; `nbody` GM·e^σ; clock t₀e^σ; T e^σ). Law (1) gives zero
free-fall violation, law (2) gives S9a's Δf = 2.08 × 10⁻³. This is decision C2's wider form: one σ-mass law, applied everywhere.

## PHYSICS_GAP: smaller data and numerics defects found in round 2 (2026-10-10)

- `field/gr_basics.py::gravitational_redshift` computes 1/√(1 − r_s/r) − 1, which cancels catastrophically in a weak field
  (Earth-surface z relative error 8.9 × 10⁻⁸; z(R) − z(R + 1 m) = 0.0 against 1.09 × 10⁻¹⁶). Use expm1(−½·log1p(−r_s/r)). (U4)
- `phase_transition.lindemann_melting_estimate` gives water ice T_melt = 92.16 K (real 273.15 K), so `material_properties`
  flags every water_ice parcel above 92 K as liquid; at 200 K the cascade's ice k is ~10× and c_p ~2× literature values
  [LIT, not re-verified]. `surface.MATERIALS` has no liquid water. (U2, U7)
- `inventory/data/isotopes.json`: 55 of 103 stable isotopes have `abundance_percent = 0` although `atomic_weights.json` has
  nonzero NIST abundances; the same rows carry placeholder spin-parity ('1/2 +' for every odd-A) and null magnetic moments. (T5)
- `inventory/core/sigma.py`: nuclear Coulomb energy is declared σ-invariant while nuclear scales follow Λ·e^σ; with α fixed and
  R ∝ 1/Λ it would scale as e^σ (Damour–Donoghue's treatment). This one choice carries 93–106 % of Δf in the torsion-balance
  test bodies. A model decision, not a code bug. (T5)
- `inventory/core/sigma.py` docstrings: "σ ~ 7 × 10⁻¹⁰ at Earth's surface" (the function returns 1.10 × 10⁻¹⁰; 7 × 10⁻¹⁰ has no ξ);
  "σ_horizon = ξ/2" is the weak-field formula evaluated outside its domain (the only static, horizon-regular solution of □σ = −ξR
  is σ = σ_∞); ξ = 0.1582 is attributed to Planck 2018 values that give 0.1571; G hard-coded. (U4, U5)
- Two solar masses: `constants.M_SUN_KG = 1.989e30` vs `solar_system_body('sun') = 1.98892e30` (see the DATA_GAP above). (J6)
- `test_physics_stability.py`, `test_nbody.py`: GR and GW tests assert only "nonzero" or "decreasing", so neither sign nor size is
  tested (see the GR and GW entries above).
- Outside this repo (SkippyTel `mentat/app.py`, recorded only): the gateway's `_UNSAFE` regex refuses the apostrophe, so
  `solve_ode`'s documented prime syntax (y″) can never reach the tool, and the `_HUGE` power-tower guard rejects ordinary
  expressions such as `3*x^2*(1+x^2)^(-5/2)`. (N1, U3)

## DATA_GAP: cascade material outliers and stale provenance found in round 2 (2026-10-10)

Recorded for the census (A31–A33 in `misc/MENTAT_GAPS_UNIFICATION.md`); none affects a round-2 verdict.

- Cascade data: iron thermal conductivity 108 W/m/K against ~80, with dκ/dT of the wrong sign; glass κ 78 W/m/K; Lindemann
  T_melt 1.9 × 10⁻⁴ K for rubber and 2.6 K for ABS (the same estimator gives ice 92 K). Under `eos="ideal"` conduction uses the
  carrier solid's c_p rather than the gas's c_v.
- Provenance and documentation: `field/audit.py:145` claims σ = ξGM/rc² is derived from □σ = −ξR (the derivation gives twice
  that; decision C2) and `audit.py:266` lists a TOV integrator as live (it exists only in `ssbm-theory`); `relativity.py`'s
  t₀e^σ and `gr_basics.time_dilation_gr` are two clock laws nothing reconciles; `gr_basics` dates Comm. Math. Phys. 43, 199 as
  1974 (it is 1975); `mcp/tools/gr.py` re-implements `hawking_temperature` while labelling the source `gr_basics`;
  `mcp/tools/frontier.py` hard-codes c; `solar_system_body('earth')` returns standard gravity 9.80665 labelled NASA; docstring
  numbers in `hysteresis.py` (+0.5 % vs +10.4 % coded), `corrosion.py` (f_QCD ≈ 0.08 vs 0.990) and `phase_transition.py`
  contradict their code; `planets.NEUTRON_STAR` is uncited; the nuclear-matter anchors in `constants.py` carry no citation and
  no L_sym; `GOLDEN_RULES.md` Rule 2 points to the deleted `field/bounds.py`.
- Research scripts: `pilot_p6_sigma_bridge.py`'s neutron-star row is excluded by GW170817 (1.61× the 90 % bound at 2 n_sat) and
  its white-dwarf row is 3.8× the Chandrasekhar pressure; `pilot_p7`'s `poisson_fvm` carries a +0.63 % split-kick bias.
