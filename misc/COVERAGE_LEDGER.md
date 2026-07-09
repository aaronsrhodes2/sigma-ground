# Transitive-Coverage Ledger (honest denominator)

**Goal:** every standard-physics leaf function in `sigma_ground/field/**` and
`sigma_ground/inventory/**` is reached by at least one *realistic* MCP question
whose answer validates against a textbook formula or external source — OR is
explicitly listed below as a non-capability and excluded from the denominator.

**Measurement:** `misc/trace_coverage.py` installs a call tracer, runs every
`sigma_ground/mcp/tools/*` function plus 5 procedures with auto-inferred
realistic args (`misc/coverage_harness.py`), and takes the union of every
`sigma_ground` function that executed. A function is *covered* iff it executes
as a byproduct of a question — direct or transitive.

**Denominator honesty rule:** we NEVER fake-cover. A function is either
(a) WIRED — reached by a real, validated question, or
(b) EXCLUDED — listed here with a reason (render/plumbing/private/dead), or
(c) REMOVED — deleted from the library (was dead/speculative).
The final 100% target = covered == (total − EXCLUDED). The regression test
(added at the end) asserts the tracer reaches 100 % of the honest denominator.

---

## Progress

| Baseline | Covered | Total | % | Note |
|----------|---------|-------|---|------|
| C        | 909     | 1093  | 83% | pre-marathon |
| batch 1-3 (mechanics/transport/rotation) | — | 1093 | — | → 909 |
| batch 4 (materials strength) | 931 | 1093 | 85% | elasticity/stress/plasticity/composites |
| batch 5 (photonics/electroceramics) | 946 | 1093 | 87% | waveguide/bandgap/nonlinear/color/phosphor/piezo/dielectric |
| batch 6 (thermal-systems/mech-response) | 964 | 1093 | 88% | TEG/convection/viscoelastic-creep/acoustic-interface |
| batch 7 (devices/quantum-solids) | 976 | 1093 | 89% | capacitor/Hall/junction/BCS-gap/tunneling/box/DOS/exchange |
| batch 8 (plasma-em/relativity-spectra) | 988 | 1093 | 90% | plasma-params/EM-forces/relativistic-energy/Zeeman |
| batch 9 (tribology/materials-micro) | 1001 | 1093 | 92% | friction/wear/Taylor/Nordheim/dipole/combustion |
| batch 10 (chemistry-extended) | 1014 | 1093 | 93% | titration/speciation/solution/echem/kinetics/radioactivity |
| batch 11 (qcomputing) | 1025 | 1093 | 94% | Grover/QAOA/Simon/qubit-state/qubit-HW/visibility |
| batch 12 (misc-physics) | 1033 | 1093 | 95% | asteroid/mobius/Hertzian-impact/HDE |
| contact-conductance (un-deferred) | 1052 | 1097 | 96% | contact_conductance rebuilt on Cooper-Mikic-Yovanovich + thermal_contact_analysis wired (validated vs textbook closed form). Denominator +4 as the library grew (incl. new cmy_contact_conductance helper); tracer measured post-change. |
| batch 13 (quarksum inventory) | 1052 | 1097 | 96% | material-inventory / constituent-behaviors / planet-MoI (water p/n/e=10/8/10, up-quark 2.16/336 MeV, Earth C/MR^2=0.325) |
| **honest 100%** | **1053** | **1097** | **100%** | covered == total - 44 documented non-capabilities; all 4 DEFER bugs fixed + wired. Enforced by `sigma_ground/mcp/test_coverage_ledger.py`. |

---

## EXCLUDED functions (non-capabilities — out of the denominator)

These are presentation, plumbing, private-by-intent, or dead code. They are not
physics capabilities a user would ask Mentat to compute, so they do not count
against coverage. Listed by the batch in which they were ruled out.

### batch 5
- `field.interface.phosphor.build_ascii_histogram` — ASCII-art terminal render
  helper (`█` bars). Presentation, not physics. EXCLUDE.

### batch 6
- `field.interface.viscosity.viscous_flow_properties` — explicitly a
  "Nagatha-compatible format" export aggregator; it only re-bundles physics
  already covered directly (stokes_drag, poiseuille_*, terminal_velocity_stokes,
  particle_reynolds_number). Serialization helper, not a new capability. EXCLUDE.

### batch 7
- `field.interface.superconductivity.block_cooling_profile` — multi-step
  *simulation* returning a per-temperature-step series (resistivity/gap/Meissner/
  London/H_c). It only re-bundles already-covered physics (bcs_gap_temperature,
  meissner_fraction, london_penetration_at_T, thermodynamic_critical_field). A
  simulation pipeline (Materia's lane), not a scalar Q&A. EXCLUDE.

### batch 9
- `field.interface.friction.material_friction_properties` — "Export friction
  properties in Nagatha-compatible format"; re-bundles already-covered
  interfacial_shear_strength / friction_coefficient / ploughing_friction.
  Serialization helper. EXCLUDE.
- `field.interface.wear.wear_profile` — "Simulate wear depth vs time", returns a
  per-time-step list; re-bundles the Archard physics covered via wear_analysis.
  Simulation pipeline, not Q&A. EXCLUDE.
- `field.interface.hysteresis.hysteresis_loop` — traces a full B-H major loop and
  returns a per-step list of points (re-bundles the already-covered
  hysteresis_loop_point). Simulation, not a scalar Q&A. EXCLUDE.

### batch 10
- `field.interface.acid_base.titration_curve` — generates a full titration curve
  (list of (V_base, pH) from 0 to 2x equivalence); re-bundles the scalar
  titration point functions covered via titration_analysis. Curve/sim, not a
  scalar Q&A. EXCLUDE.

### batch 11
Per the Phase-0 triage ruling ("wire high-level QC, exclude gate primitives"):
- `field.interface.quantum_computing.{gate_cz, gate_fredkin, gate_iswap,
  gate_phase, gate_rx, gate_rz, gate_s, gate_t, gate_y}` (9) — single/two-qubit
  gate-MATRIX constructors. Building blocks, not standalone Q&A capabilities;
  demonstrated in action by the algorithm tools (Grover/QAOA/Simon). EXCLUDE.
- `field.interface.quantum_output.{extract_phase, extract_function_value,
  histogram_to_answer}` (3) — algorithm-result post-processing plumbing
  (histogram_to_answer takes a callable interpret_fn). Internal to running an
  algorithm, not a standalone question. EXCLUDE.
- `field.interface.quantum.{sample_hit_position, cumulative_probability}` (2) —
  double-slit Monte-Carlo CDF-sampling internals. sample_hit_position is
  non-deterministic (takes a rand_val); cumulative_probability builds the CDF
  array that feeds it. EXCLUDE.

### batch 12
- `field.interface.orbital.{fit_orbit, predict_ssb_position}` (2) — the JPL/DE440
  ephemeris least-squares FIT pipeline. fit_orbit fits Keplerian elements from
  yearly DE440 fixture snapshots (rv-to-elements + LSQ residual refine);
  predict_ssb_position takes a FittedOrbit object it produces. Needs internal
  ephemeris fixtures; an internal fit pipeline, not a scalar Q&A. EXCLUDE.
- `field.interface.projectile.projectile_report` — "Export projectile analysis in
  Nagatha-compatible format"; re-bundles already-covered projectile_range /
  projectile_max_height. Serialization helper. EXCLUDE.

### batch 13 (quarksum inventory + rolling)
- `field.interface.rolling_analysis.{ablation_table, add_rtn_components,
  cross_predictor_correlation, fingerprint_diff, orbital_correlation,
  per_predictor_body_summary, print_report, rtn_summary, save_plots}` (9) +
  `field.interface.rolling_shootout.run_rolling_shootout` (1) — a numpy-based
  ephemeris-prediction BENCHMARK + analysis/reporting/plotting harness (RTN
  residuals, ablation tables, print_report, save_plots). A research experiment
  runner, not a physics Q&A capability. EXCLUDE.
- `inventory.behaviors.apply_env` + `inventory.behaviors.{atom,molecule,particle,
  quark}_behaviors.resolve_*_env` (5) — the universal SETTER cascade that mutates
  entity state under an applied environment. State mutation is Materia's lane
  (the simulation playground), not the Q&A switchboard. The corresponding
  GETTERS (compute_*_behaviors via `behaviors()`) ARE wired
  (constituent_behaviors_analysis). EXCLUDE.
- `inventory.builder.load_structure` — redundant convenience loader; the
  canonical load_structure_spec -> build_structure_from_spec path is the one
  exercised by material_inventory_analysis. EXCLUDE.
- `inventory.checksum.{particle_count.count_particles_in_structure,
  quark_chain.compute_quark_chain_checksum, quark_chain.walk_quark_chain,
  stoq_checksum.compute_stoq_checksum}` (4) — the low-level quark-chain checksum
  *verification* walkers. The user-facing inventory result (particle counts +
  mass closure) is already reported by compute_particle_inventory via
  material_inventory_analysis; these are internal verification plumbing. EXCLUDE.
- `field.interface.adhesion.contact_angle` — SUPERSEDED. The Owens-Wendt
  `wetting_contact_angle` (with the WETTING_LIQUIDS/WETTING_SOLIDS DBs) is the
  wired, physically-correct wetting capability; the old contact_angle draws both
  phases from the solids broken-bond DB and reports 0 deg for metal pairs that
  really bead. Kept in the library, unwired. EXCLUDE (was DEFER).
- `field.interface.superconductivity.gl_parameter` — SUPERSEDED by
  `gl_parameter_effective` (the measured / dirty-limit kappa), which is what
  superconductor_critical_field_analysis uses. gl_parameter returns only the
  clean-limit kappa (Nb 0.11 vs measured 1.05). Kept, unwired. EXCLUDE (was DEFER).
- `field.interface.resolve.material_profile` — DECKARD-INTERNAL. The
  voxelizer's default-density resolver (material name → profile dict), consumed
  by `deckard.voxelize` when stamping real meshes. Not a user-facing Q&A
  capability — the material data itself is already wired via the
  materials/material-profile tools. EXCLUDE. (Arrived with the voxel arc on the
  master lane; ledgered at the mentat-lane merge.)

The machine-checked list lives in `sigma_ground/mcp/test_coverage_ledger.py`
(EXCLUDE_SET); that test asserts `uncovered ⊆ EXCLUDE_SET` so no future
capability is silently left unwired.

---

## DEFERRED (real capability, but current model fails the accuracy bar)

NOT exposed and NOT counted as covered — a known gap to fix, then wire. Distinct
from EXCLUDE (which is "not a capability at all"). Listed so the regression test's
honest denominator can subtract them while keeping them visible as TODO.

- ~~`field.interface.thermal.contact_conductance`~~ — **RESOLVED 2026-06-04.**
  The atomic-gap model (h ~ 1.9e9 W/(m^2.K) for Cu-Al at 1 MPa, linear in
  pressure, ~4-5 orders too high) was replaced with the Cooper-Mikic-Yovanovich
  plastic joint-conductance correlation h_c = 1.25 k_s (m/sigma)(P/H_c)^0.95.
  Roughness and asperity slope are now explicit engineering surface-finish
  inputs (not lattice-derived). Validated against the textbook closed form
  (stainless-like pair -> ~7.06e3 W/(m^2.K)) and the CMY dimensionless invariant;
  Cu-Al at 1 MPa now returns ~6e4 W/(m^2.K), inside the 1e3-1e5 band. Wired as
  the `thermal_contact_analysis` MCP tool. No longer deferred.
- ~~`field.interface.superconductivity.{lower_critical_field,
  upper_critical_field}`~~ — **RESOLVED.** `thermodynamic_critical_field` was
  ~12 orders too small (Al implied ~2.5e-9 A/m) so Hc1/Hc2 were ~1e-9 and
  inverted (Hc1 > Hc2). The chain was rebuilt (Hc from condensation energy with a
  free-electron DOS; Hc1/Hc2 from the measured/dirty-limit kappa via
  gl_parameter_effective). Now physical and correctly ordered: Nb Hc1=16.4 < Hc2=66
  kA/m; NbTi Bc2~6.5 T; type-I metals (Al, Pb) return a single Hc and no Hc1/Hc2.
  Wired as the `superconductor_critical_field_analysis` MCP tool. The clean-limit
  `gl_parameter` is now superseded by `gl_parameter_effective` (see EXCLUDE).

### RESOLVED (was deferred, now fixed + WIRED)
- `field.interface.plasma.spitzer_resistivity` — fixed: dropped the spurious
  `1/(4 pi eps0)` factor AND corrected the prefactor (the old `numerator/
  denominator` was itself pi^2 high vs the canonical (4 pi eps0)^2 form).
  Now returns the Spitzer parallel resistivity eta_par = 0.51*(4 sqrt(2 pi)/3)
  sqrt(m_e) Z e^2 lnLambda / ((4 pi eps0)^2 (k_B T_e)^1.5), validated within
  ~1.2% of NRL (5.26e-7 vs 5.2e-7 ohm.m at 100 eV; 9e-7 at 10^6 K). Wired into
  `plasma_parameters_analysis` as `spitzer_resistivity_ohm_m` (with `z_eff`).
- `field.interface.adhesion.{wetting_contact_angle, work_of_solid_liquid_adhesion,
  spreading_coefficient}` — NEW wetting capability backed by a liquids
  surface-tension DB (`WETTING_LIQUIDS`: water, mercury, ethanol, glycerol,
  ethylene glycol, diiodomethane, molten solder, gallium) and a dispersive/polar
  substrate DB (`WETTING_SOLIDS`: PTFE, paraffin, PE, PMMA, glass, silicon, steel,
  gold), combined with the Owens-Wendt geometric-mean rule. Metallic liquids
  (mercury, molten metals) use the dispersive-only Fowkes term on dielectrics,
  so mercury beads instead of wetting. Reproduces textbook angles: water/clean-
  glass ~0 deg, water/PTFE ~108 deg, water/paraffin ~111 deg, mercury/glass
  ~133 deg (textbook ~140), water/gold ~55 deg. WIRED as `wetting_analysis`.
  The legacy `field.interface.adhesion.contact_angle` (solids-only — the function
  that returned 0 deg for metal pairs) is SUPERSEDED by these. It is retained for
  backward-compat + its sigma-scaling unit tests, and is EXCLUDED from the honest
  denominator (the user-facing wetting capability is now the Owens-Wendt path
  above; the old signature is not a separate capability a user would ask for).

---

## Pending EXCLUDE candidates (to confirm against uncovered.txt in later batches)

Categories flagged during Phase-0 triage, to be enumerated function-by-function
when their batch is reached (do NOT pre-list unverified):
- Nagatha-format exporters / serialization helpers
- ASCII / render / color-swatch display helpers
- CDF-sampling internals (private Monte-Carlo plumbing)
- `rolling_analysis` / `rolling_shootout` simulation pipeline (sim, not Q&A)
- orbital ephemeris least-squares fit internals
- QC gate/state primitives (covered indirectly via high-level QC tools)
- inventory env-resolver / builder / checksum plumbing (internal-only)
