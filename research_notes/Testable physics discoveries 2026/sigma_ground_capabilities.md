# sigma-ground capability map: which code entry points can test a new physics result (codebase investigation, 2026-10-10)

Scope note: this was a read-only investigation of `/home/user/sigma-ground` at commit `ce8a453` (2026-10-10). Sources below are repo-relative file paths, with line numbers where useful. Numbers marked **[run]** were computed in this session by importing the repo's own functions (`PYTHONPATH=. python3 ...`). They are the code's outputs, not literature values. The Mentat MCP server failed to connect in this session (CONNECTION_CLOSED), so I could not call it live. Its tools are documented from source. `pytest` is not installed in this environment, so I counted tests statically and did not run them.

**The most important finding for the report writer:** the speculative SSBM/σ-theory machinery (ξ as a field coupling, σ_conv, BH formation, bond-failure layers, space cavitation, BH-merger echo predictions, TOV/Chandrasekhar) has mostly been **moved out of this repo into a separate `ssbm-theory` repo**, which is not on disk here. The live `sigma_ground.field` package now describes itself as "standard, observation-anchored". σ is an inert hook, e^σ = 1 at σ = 0. What remains σ-specific is:
- `sigma_ground/inventory/core/sigma.py`, which still has ξ = 0.1582 and the σ-shifted nucleon masses
- `field/nucleon.py` and `field/binding.py`
- about 30 `sigma_*_shift` helpers in `field/interface/`
- documentation and verdict files in `misc/`

The repo's own newest result (S9a, 2026-10-10) is that a long-range σ with ξ ≈ 0.16 is excluded by atomic clocks and by the MICROSCOPE test by factors of 10⁵ to 10¹¹.

---

## Q1. What do the top-level docs say (tier map, testing baseline, known gaps, physics rules)?

### Takeaway
The repo ("Mentat") is a pure-Python physics and rendering stack in five dependency tiers. The tiers run from `kernel` (tier 0) up to `mcp` (tier 4), and every test run enforces the layering. `field/constants.py` is the single authoritative source for constants. The project's stated goal is a Q&A "physics professor assistant" benchmarked against Wolfram Alpha, not a cosmology research code. The physics rules require derivation and citation, explicit domain bounds, and `# PHYSICS_GAP:` tags that are logged in KNOWN_GAPS.md.

### Cited Findings
- **Tier map.** Each tier may import only its own tier or below; `tests/test_layering.py` is the AST guard that enforces this. [ARCHITECTURE.md](ARCHITECTURE.md)
  - tier 0: `kernel/` (geometry, `Vec3`, csg, gear)
  - tier 1: `field/` (σ-field physics plus authoritative constants), `inventory/` (Quarksum particle inventory and mass closure), `dynamics/` (N-body, SPH, Barnes-Hut, integrators, mechanisms)
  - tier 2: `deckard/` (name → validated SDF `Construct`), `blueprint/` (mechanism compiler)
  - tier 3: `materia/` (physics/movement engine plus labs), `radiance/` (renderer)
  - tier 4: `mcp/` (`FastMCP("mentat")`)
- **Constants.** `sigma_ground/field/constants.py` is authoritative. `sigma_ground/constants.py` and `sigma_ground.kernel` re-export it: "Never define a physical constant twice." [ARCHITECTURE.md](ARCHITECTURE.md)
- **Testing baseline in ARCHITECTURE.md.** "4429 passed, 6 failed, 6 skipped, 2 xfailed plus ~2440 subtests". The 6 failures are known-red benchmark QA-threshold tests in `mcp/benchmark/test_targets.py`. `test_jpl_ephemeris` (network-gated) and `test_position_precision` are permanently red. [ARCHITECTURE.md](ARCHITECTURE.md)
- **Newer baseline from the session log.** Session 34 (2026-08-05) reported a full suite of "5023 passed / 6 skipped / 3 xfailed / 2508 subtests, 1 failed". The failure was a benchmark Wolfram-parity test. [misc/SESSION_LOG.md](misc/SESSION_LOG.md) (Session 34)
- **README is stale.** It says "1198 tests passing". It also lists `field.bounds`, `field.entanglement`, `field.universe` and `field.sandbox` modules, which no longer exist in `sigma_ground/field/` (directory listing). [README.md](README.md)
- **Project goal.** "Build a physics professor assistant". The targets are ≥98% of Wolfram Alpha's accuracy on a 150-question physics-major corpus, every tool returning `(value, units, source, uncertainty, provenance_tag)`, and a conversation mode that is "not started". As of 2026-05-17 the Q&A baseline was sigma-ground 28% vs Wolfram 13.3%. [PROJECT_GOAL.md](PROJECT_GOAL.md)
- **Golden Rules.** [GOLDEN_RULES.md](GOLDEN_RULES.md)
  1. Derive, don't assume.
  2. Explicit domain bounds (Design by Contract; the SAFE/EDGE/WALL/BEYOND classes originally came from `field/bounds.py`).
  3. Prefer exact analytic solutions and cite PDG/NIST.
  4. No silent approximations: tag them `# PHYSICS_GAP:` and log them in KNOWN_GAPS.md.
  5. Test against a volume of matter.
  6. Fix the physics, never patch the picture.
  7. Only artificial forces are adjustable during a simulation.
- **CLAUDE.md on Mentat.** For any simulation, test or experiment, use the Mentat MCP tools rather than reasoning about the physics directly. Mentat "flags suspected bugs (bad sigma defaults, earth-sigma violations) but NEVER fixes them". [CLAUDE.md](CLAUDE.md)
- **CLAUDE.md on ssbm-theory.** It "Stays separate (own repo now). Unverified SSBM black-hole-merger research, extracted from `field/` but never merged back and currently broken in isolation." [CLAUDE.md](CLAUDE.md)
- **docs/ folder.** It holds only superconductivity material: `alloy_Tc_predictions.html`, `alloy_predictions.json`, `beyond_mcmillan.html`, `mcmillan_validation.html`, `dependency_chart.html`, `DATA_SOURCES.md` (directory listing). [docs/](docs/)

### Inferences
- The live repo is a broad standard-physics calculator and simulation engine with an SSBM heritage. Most new 2026 results will map onto standard-physics entry points (constants, GR basics, nuclear binding, cosmology helpers), not onto σ-specific code.
- README's API table is stale. The report writer should rely on ARCHITECTURE.md and the actual file tree.

### Gaps
- I did not read `../PLATINUM_RULES.md` or `../LOGGING_STANDARD.md`, which sit outside the repo.

---

## Q2. What does each subpackage cover (key public functions/classes)?

### Takeaway
About 518 non-test Python files. Most of the physics is in `sigma_ground/field/interface/`: about 85 modules and ~49k lines covering materials, thermo, quantum, nuclear, plasma, optics, N-body and cosmology. `field/` top-level holds constants, SR, Schwarzschild GR and decay. `inventory/` holds the particle and mass ledger, plus the σ-scaling core. `dynamics/` holds SPH, N-body and mechanisms. `mcp/` exposes 232 tools.

### Cited Findings
- **`field/` top-level.**
  - `constants.py` (458 lines), `scale.py` (`scale_ratio(σ)=e^σ`, `schwarzschild_radius`), `nucleon.py` (`proton_mass_mev`, `neutron_mass_mev`, `nucleon_decomposition`), `binding.py` (`coulomb_energy_mev`, `binding_energy_mev`, `binding_decomposition`).
  - `relativity.py`: `lorentz_factor`:25, `velocity_addition`:131, `doppler_factor`:179, `sigma_time_dilation`:204, `effective_liv_scale_gev`:229.
  - `gr_basics.py`: `gravitational_redshift`:45, `time_dilation_gr`:64, `escape_velocity`:83, `isco_radius`:99, `photon_sphere_radius`:108, `hawking_temperature`:118, `hawking_luminosity`:132, `hawking_evaporation_time`:145, `page_time_sm`:158, `bekenstein_hawking_entropy`:173, `tidal_force`:188.
  - `decay.py`: `decay_constant`, `half_life`, `q_value_alpha/beta_minus/beta_plus`, `gamow_factor`:168, `alpha_decay_rate_geiger_nuttall`:199.
  - `electrodynamics.py`: `fine_structure_constant`:250, `muon_anomalous_moment_experimental`:274, `muon_anomalous_moment_sm`:285, `muon_g2_tension_sigmas`:296, `muon_g2_consistency_status`:313.
  - `planets.py`: `EARTH`, `MARS`, `JUPITER`, `NEUTRON_STAR` layered-body dicts.
  - `asteroids.py`: Bennu, Ryugu, Itokawa, Eros, Vesta, Ceres.
  - `audit.py`: SSBM formula audit via `build_audit`:41 and `confidence_summary`:505.
  - `verify.py`: `three_measures`:16, `verify_all`:73.
  - [sigma_ground/field/](sigma_ground/field/)
- **`field/interface/` highlights (non-exhaustive).**
  - `cosmology.py` (HDE, η band check, MOND).
  - `nbody.py`: `NBodySystem`:255, `CelestialBody`:153, `PhysicsToggles`:77. It offers velocity-Verlet and 4th-order Forest-Ruth integration, 1PN Schwarzschild correction, J₂, tidal Love k₂, Peters-1964 GW energy loss, and solar radiation pressure.
  - `orbital.py`: `fit_orbit`:365, `predict_ssb_position`:477.
  - `nucleosynthesis.py`: `gamow_energy_keV`:213, `reaction_rate_sigma_v`:279, `pp_chain_energy_rate`:460, `cno_energy_rate`:525, `pp_cno_crossover_temperature`:577, `stellar_burning_summary`:694.
  - `radioactive_decay.py`: `alpha_Q_at_sigma`:370, `alpha_half_life`:460, `beta_half_life`:607, `geiger_nuttall_check`:726, `isotope_decay_properties`:757.
  - `plasma.py`: `plasma_frequency`, `debye_length`, `alfven_speed`, `coulomb_logarithm`, `spitzer_resistivity`, `sigma_plasma_transition`:241.
  - `statistical.py`: partition function, Fermi-Dirac, Bose-Einstein, Maxwell speeds, `sigma_partition_shift`:278.
  - `gas.py`: `gas_viscosity`:617, `gas_thermal_conductivity`:662, `gas_diffusivity`:702, `heat_capacity_ratio`:592, `grashof_number`:784.
  - `fluid.py`: `eyring_viscosity`:196, `liquid_viscosity`:249, `reynolds_number`:358, `surface_tension`:342.
  - `viscosity.py`: `stokes_drag`:94, `terminal_velocity_stokes`:113, `particle_reynolds_number`:137, `drag_coefficient_sphere`:159, `poiseuille_flow_rate`:207, `boundary_layer_thickness`:341.
  - Also: `atomic_spectra.py`, `superconductivity.py` (1385 lines), `quantum_*` modules, `thermal_emission.py`, `atmosphere.py`, and the materials modules (elasticity, plasticity, stress, wear, corrosion, and others).
  - [sigma_ground/field/interface/](sigma_ground/field/interface/)
- **`inventory/` (Quarksum).** It resolves material → molecule → atom → particle → quark. [sigma_ground/inventory/](sigma_ground/inventory/), [README.md](README.md)
  - `stoq`, `quark_chain`, `inventory`, `resolve`, `build_quick_structure`, `load_structure`, `physics`, `tangle`
  - `core/sigma.py`: σ scaling
  - `checksum/sigma_chain.py`: `sigma_checksum_nucleus`:34, `sigma_sweep`:119
  - `data/particle_masses.json`: the PDG 2026 snapshot
- **`dynamics/`.** [sigma_ground/dynamics/](sigma_ground/dynamics/)
  - `continuum.py::continuum_step`: the SPH continuum bridge, with eos options "tait", "isothermal" and "ideal", plus `scene.energy_ledger()`
  - `fluid/eos.py`, `fluid/kernel.py` (cubic spline)
  - `gravity/barnes_hut.py` (2-D quadtree only, per gap B8)
  - `stepper.py` (leapfrog with CFL), `collision.py`, `joints.py`, `fields/heat.py::diffuse_fvm`, `phase_e.py`, `mechanisms/` (gears, escapement, wind, spring)
- **`materia/`.** `scenarios.py` (3222 lines) holds the simulation "verbs" behind the MCP `simulate`/`run_simulation` tools. They include `terminal_velocity_drop`:21, `drag_heating_drop`:129, `supersonic_projectile`:321, `orbital_velocity`:436, `nuclear_decay`:1213, `black_hole`:1282, `hydrogen_spectrum`:1345, `relativistic_motion`:1436, `fluid_flow`:1497, `plasma_physics`:1842, `stellar_fusion`:1871, `superconductor`:2198 and `fluid_dynamics`:2269. [sigma_ground/materia/scenarios.py](sigma_ground/materia/scenarios.py)
- **Other packages.**
  - `kernel/`, `deckard/`, `blueprint/`, `radiance/` and `engine/` are geometry, object compilation, rendering and game-engine code, with no fundamental-physics predictions.
  - `labs/` is a 9-line compatibility shim to `materia.labs`.
  - [ARCHITECTURE.md](ARCHITECTURE.md)
- **`mcp/`.** `server.py` (1884 lines) registers 232 `@server.tool()` functions (counted via AST in this session). `benchmark/` holds the Wolfram/Gemini Q&A evaluation harness with domain classifiers: cosmology, GR, astro, frontier, modern and others. [sigma_ground/mcp/server.py](sigma_ground/mcp/server.py)

### Inferences
- For a new measurement, the most direct "drop-in" points are:
  - constants in `field/constants.py`
  - the PDG snapshot `inventory/data/particle_masses.json`, through the drift gate
  - fixture JSONs under `field/interface/fixtures/`

### Gaps
- I did not inventory every function in the ~85 interface modules. Only the physics-relevant ones are listed.

---

## Q3. What does the σ field do exactly, and what numerical predictions differ from standard physics?

### Takeaway
In the live code, σ scales only the QCD portion of hadronic masses and nuclear binding by e^σ. The Higgs (bare-quark) and electromagnetic parts are held fixed. σ is set from the gravitational potential as σ = ξGM/(rc²), which gives ≈1.1×10⁻¹⁰ at Earth's surface and ξ/2 = 0.0791 at a horizon. At our location σ = SIGMA_HERE ≈ 1.18×10⁻⁶¹, so all standard physics is unchanged. The BH-formation, cavitation, bond-failure and merger-echo machinery is not in this repo; it survives only as `misc/` documents. The repo's own 2026-10-10 analysis (S9a) finds that the long-range field-equation reading is excluded by clocks and the equivalence principle by 3.2×10⁵ to 9.4×10¹⁰.

### Cited Findings

#### Defining equations and defaults (live code)
- **Core equation, as recorded in the audit.**
  - □σ = −ξR, with static solution σ(R) = −ξR.
  - σ = ξ·G·M/(r·c²), giving σ_horizon = ξ/2.
  - Λ_eff = Λ_QCD·e^σ.
  - σ_conv = −ln ξ, described as the "conversion threshold … at confinement limit".
  - m_p(σ) = m_bare + m_QCD·e^σ.
  - BE(σ) = BE_strong·e^σ − BE_Coulomb.
  - Relocated "nesting" formulas: M_N = M_Hubble·ξ^N, N_max = 76; "dark energy from η" ρ_DE = η·ρ_matter·f_QCD·(e^σ_conv − 1).
  - [sigma_ground/field/audit.py:41-360](sigma_ground/field/audit.py) (printed via `build_audit()` [run])
- **Parameters.** [sigma_ground/inventory/core/sigma.py](sigma_ground/inventory/core/sigma.py)
  - `XI = 0.1582`, described as Ω_b/(Ω_b+Ω_c) from Planck 2018 and "the one new free parameter" (line 39).
  - `ETA = 0.642² = 0.412164` (line 43).
  - `SIGMA_FLOOR = l_P·H0/c ≈ 1.18e-61` (line 66).
  - `LAMBDA_QCD_MEV = 217.0` (line 76).
- **ξ removed from `field/constants.py`.** The comments say "(XI … relocated to ssbm-theory.)" and "(SIGMA_CONV … relocated to ssbm-theory.)". [sigma_ground/field/constants.py:26,158](sigma_ground/field/constants.py)
- **σ-dependent and σ-invariant parts.** [sigma_ground/inventory/core/sigma.py:1-29](sigma_ground/inventory/core/sigma.py)
  - Scale with e^σ: constituent quark dressing, nucleon QCD mass, strong nuclear binding, gluon field energy.
  - Invariant: bare quark masses (u = 2.16, d = 4.70 MeV), electron mass, Coulomb energy in nuclei, chemical bond energies, α_EM.
- **Mass decomposition constants.** M_UP_MEV = 2.16 and M_DOWN_MEV = 4.7 (PDG 2026). PROTON_BARE = 9.02 and PROTON_QCD = 929.252 MeV, so the QCD fraction is 0.9904. NEUTRON_BARE = 11.56 and NEUTRON_QCD = 928.005 MeV, fraction 0.9878. DELTA_NP_QCD = 1.29333 − 2.54 = −1.247 MeV. [sigma_ground/field/constants.py:172-196](sigma_ground/field/constants.py)
- **σ from the gravitational potential.** `sigma_from_potential(r, M) = XI·GM/(rc²)`, with compactness capped at 0.5. The docstring says "At event horizon: σ = ξ/2 = 0.0791. At Earth's surface: σ ~ 7×10⁻¹⁰". [sigma_ground/inventory/core/sigma.py:290-304](sigma_ground/inventory/core/sigma.py)
- **Observer frame.** `SIGMA_HERE = SIGMA_FLOOR` = l_P/R_H ≈ 1.18×10⁻⁶¹; exp of that is exactly 1.0 in double precision. The comments say "Inside a black hole accretion disk: σ > 0", "At the Big Bang / inside cavitation: σ → σ_conv ≈ 1.85", and "Earth surface: ~7×10⁻¹⁰". [sigma_ground/field/constants.py:371-457](sigma_ground/field/constants.py)
- **The "r_s/R_H identity" in code.** `M_HUBBLE_KG = c³/(2GH0)` is the mass whose Schwarzschild radius equals the Hubble radius c/H0. It evaluates to 9.24×10⁵² kg [run]. `SIGMA_FLOOR` itself is defined as l_P/R_H. [sigma_ground/field/constants.py:421,454](sigma_ground/field/constants.py)
- **Time dilation.** `sigma_time_dilation(σ, t0) = t0·e^σ`. The docstring quotes a factor ≈6.35 at σ_conv ≈ 1.849 and ≈1.08 at the horizon. [sigma_ground/field/relativity.py:204-226](sigma_ground/field/relativity.py)
- **No Lorentz-invariance violation.** `effective_liv_scale_gev(σ)` always returns ∞. The docstring states "There is NO σ-dependent modification of the dispersion relation", which trivially satisfies the KM3-230213A bound Λ₂ > 5.0×10¹⁹ GeV. [sigma_ground/field/relativity.py:229-252](sigma_ground/field/relativity.py)
- **σ-shift helpers.** Roughly 30 `sigma_*_shift` functions in `field/interface/` propagate σ through nuclear mass into materials properties. Examples: `rydberg_constant_at_sigma` (atomic_spectra:570), `sigma_spectral_shift` (:584), `sigma_Tc_shift` (superconductivity:885, through Θ_D ∝ 1/√mass), `alpha_Q_at_sigma` (radioactive_decay:370: Q(σ) = Q_strong·e^σ + Q_coulomb), `sigma_melting_shift`, `sigma_nernst_shift`, `sigma_pKa_shift` and `sigma_plasma_transition`. (grep of `^def sigma_|_at_sigma`) [sigma_ground/field/interface/](sigma_ground/field/interface/)
- **"Earth sigma" constraint as tested.** About 20 interface test files contain `test_earth_sigma_negligible` with `sigma_earth = 7e-10`; they assert negligible property shifts at Earth's surface. Examples: `test_thermal.py:377`, `test_friction.py:297`, `test_molecular_bonds.py:499`, `test_radioactive_decay.py:485`. [sigma_ground/field/interface/test_thermal.py](sigma_ground/field/interface/test_thermal.py)
- **Space cavitation and the 8 bond-failure layers.** No code implements either. A grep for "cavitation" or "bond_failure" in `.py` files hits only the constants comment above. The bond-failure layer appears only as a range in a doc: "σ ∈ [ξ/2, σ_conv)" (UP-006). [misc/unpublished_predictions.md](misc/unpublished_predictions.md)

#### σ predictions as computed by the code [run]

| quantity | σ = SIGMA_HERE (standard) | σ = ξ/2 = 0.0791 (horizon) | σ = σ_conv = −ln 0.1582 = 1.8439 |
|---|---|---|---|
| e^σ, which is also the `sigma_time_dilation` factor | 1 | 1.0823 | 6.3211 (= 1/ξ) |
| Λ_eff = 217·e^σ (MeV) | 217 | 234.86 | 1371.7 |
| m_p (MeV), `field/nucleon.py` | 938.272 | 1014.761 | 5882.93 |
| m_n (MeV) | 939.565 | 1015.951 | 5877.58 |
| m_n − m_p (MeV) | 1.2933 | 1.1904 | **−5.342** (neutron lighter than proton) |
| ⁵⁶Fe binding (MeV), `binding_energy_mev(492.254,26,56,σ)` | 492.254 | 542.72 | 3754.4 |
| ⁴He binding (MeV) | 28.296 | 30.699 | 183.63 |
| Rydberg for hydrogen, `rydberg_constant_at_sigma` (m⁻¹) | 10,967,758.35 | 10,968,208.36 | 10,972,778.47 |

- The m_n − m_p sign flip follows from the code's formula: Δ(σ) = 2.54 − 1.247·e^σ. It crosses zero at e^σ = 2.037, i.e. σ = 0.711 [run]. [sigma_ground/field/nucleon.py](sigma_ground/field/nucleon.py), [sigma_ground/field/constants.py:192-196](sigma_ground/field/constants.py)
- `sigma_from_potential` values [run]. [sigma_ground/inventory/core/sigma.py:290](sigma_ground/inventory/core/sigma.py), [sigma_ground/field/planets.py](sigma_ground/field/planets.py)

  | location | σ |
  |---|---|
  | Earth surface | 1.10×10⁻¹⁰ |
  | Mars | 2.22×10⁻¹¹ |
  | Jupiter | 3.19×10⁻⁹ |
  | Sun surface | 3.36×10⁻⁷ |
  | 1.4 M☉ neutron star (R = 11 km, from `planets.NEUTRON_STAR`) | 0.0297 |
  | any horizon | 0.0791 |

#### The repo's own falsification of long-range σ (S9a, 2026-10-10)
- **Formulation.** The field is written as (□ − λ⁻²)σ = −(8πGξ/c⁴)(ε_b − 3P). For λ → ∞, σ = −2ξΦ/c²; for λ → 0, σ = 8πGξλ²ρ/c². Because σ moves μ = m_p/m_e (f_μ = 0.9904) and leaves α fixed, μ is the observable. [misc/UNIFICATION_PROGRAM.md:215-231](misc/UNIFICATION_PROGRAM.md)
- **Three measurements cut the long-range (field-equation) reading.** [misc/UNIFICATION_PROGRAM.md:233-239](misc/UNIFICATION_PROGRAM.md); [misc/unification/sigma_exclusion.json](misc/unification/sigma_exclusion.json); script [scripts/unification/sigma_exclusion.py](scripts/unification/sigma_exclusion.py)
  - **Yb⁺/Cs clocks** (Lange 2021). Predicted (c²/μ)dμ/dΦ = −0.313 vs measured 7(45)×10⁻⁸. Excluded by ×3.2×10⁵; long-range ξ < 4.9×10⁻⁷.
  - **MICROSCOPE** (Touboul 2022). Predicted η = 6.6×10⁻⁴, from Δf = 2.08×10⁻³ between Ti and Pt alloys, vs (−1.5±2.3±1.5)×10⁻¹⁵. Excluded by ×9.4×10¹⁰; ξ < 1.7×10⁻¹².
  - **White-dwarf H₂** (Bagdonaite 2014). Predicted Δμ/μ = +3.8 and +6.0×10⁻⁵ vs (−2.7±4.7) and (−5.8±3.8)×10⁻⁵. Tension of 1.4σ and 3.1σ; ξ < 0.36.
- **Surviving corner.** Only short range survives. The paper value m_σ ≈ Λ_QCD gives λ = 0.909 fm, and then σ at nuclear density is 6.5×10⁻⁴⁰. Reaching σ = 0.34–0.41 needs ρ ≈ 1.4–1.7×10⁵⁶ kg/m³. If ξ = 0.13–0.17, the range must be λ < ~34.5 km (MICROSCOPE) and < 9.7×10⁹ m (clocks). The suggested next cut is Eöt-Wash torsion balances over 1 cm–35 km. [misc/UNIFICATION_PROGRAM.md:241-266](misc/UNIFICATION_PROGRAM.md)
- **Unresolved theory decisions.** [misc/MENTAT_GAPS_UNIFICATION.md](misc/MENTAT_GAPS_UNIFICATION.md) (section C)
  - C2: the σ normalisation, 2ξGM/rc² vs ξGM/rc², is a factor 2 on every prediction.
  - C3: ξ = 0.1582 vs 0.1571 (0.7%).
  - C4: covariant vs flat □.
  - C5: the range λ.
  - B5: no defined Ricci-scalar sign convention.

#### Relocated SSBM black-hole research (documents in misc/, code modules NOT in this repo)
- **QNM ringdown amplitude test.** Predicted Γ = A_obs/A_GR is 1 (GR), 0.8395 (exp), 0.7924 (1−η/2 with the old η = 0.4153) or 0.7461 (η^(1/3)). The data were five GWOSC O1–O3 events. Status: "Corrected null — no evidence; decisive test requires pycbc on top-5 GWTC-3 events". [misc/bh_sigma_conv_synthesis.md](misc/bh_sigma_conv_synthesis.md)
- **ξ-shell echoes.** Shells sit at r_n = r_s·ξⁿ, with echo delay Δt_n = 2·r_s·n·σ_conv/c. For GW150914, Δt_1 = 2.26 ms. [misc/bh_merger_predictions.md](misc/bh_merger_predictions.md), [misc/bh_echo_search_refined.md](misc/bh_echo_search_refined.md)
- **Mass-loss hypotheses.** [misc/bh_imr_verdict.md](misc/bh_imr_verdict.md), [misc/bh_mass_function_verdict.md](misc/bh_mass_function_verdict.md), [misc/bh_o4_o5_forecast_b2.md](misc/bh_o4_o5_forecast_b2.md), [misc/bh_b3_sgr_a_star_verdict.md](misc/bh_b3_sgr_a_star_verdict.md), [misc/bh_collision_phenomenology.md](misc/bh_collision_phenomenology.md)
  - **B1** (a 15.8% mass loss at merger) was "falsified at ≥6σ" (6.3σ) by GWTC-3 IMR consistency.
  - **B2** (a critical-mass threshold) requires M_crit ≥ 71 M☉ (from GW190521). Its clean survival window is [135, 200] M☉, with O4/O5 as the discriminator.
  - **B3** (spontaneous conversion) is untested: Sgr A* astrometry is ~10⁹× too weak.
  - The surviving "RODM-A" merger picture is a GR-standard waveform with ringdown corrections, and no EM or mass-loss signal.
- **Code modules not on disk.** The docs name `field/interface/bh_merger.py`, `imr_consistency.py`, `bh_mass_function.py` and `sgr_a_star_rate.py` (29/19/24/19 tests). None exist in the current tree, and they have no git history in this clone. [sigma_ground/field/interface/](sigma_ground/field/interface/) (listing)
- **"Unpublished predictions" log.** [misc/unpublished_predictions.md](misc/unpublished_predictions.md)
  - UP-001: DESI DR3 HDE c² lands within 1% of 0.4153. Falsifier: a central value outside [0.410, 0.420].
  - UP-002: Page time shifted by (1−η/2) ≈ 0.792.
  - UP-003: H₀ tension correlates with BH density along sightlines, at parts in 10⁴.
  - UP-004: S₈ anti-correlates with η.
  - UP-005: ξ matches an asymptotic-safety fixed point.
  - UP-006: intra-horizon Lorentz-invariance violation in the bond-failure layer.
  - UP-007: η as a bit-thread fraction.
  - All are `[SPECULATIVE]`.
- **Moduli-stabilisation paper.** It claims □σ = −ξR with m_σ ~ Λ_QCD, σ ∈ [0, σ_conv), Weak Gravity Conjecture safety ("conversion wall at sigma_conv ~ 1.849 intervenes 22x before saturation"), and compactification radii R_c = l_P·(1/ξ)^(1/n). [misc/PAPER_sigma_moduli_stabilization.txt](misc/PAPER_sigma_moduli_stabilization.txt)

### Inferences
- **Every σ-driven deviation from standard physics vanishes at the observer's frame by construction.** The only predictions that can be confronted with data are the following, and the repo has already done the first:
  - (a) composition-dependent fifth-force and μ-variation signals (clocks, MICROSCOPE, white-dwarf H₂, quasar H₂, torsion balances, μ from molecular spectroscopy). New 2026 μ-variation or equivalence-principle bounds plug straight into `scripts/unification/sigma_exclusion.py`.
  - (b) strong-field signals near compact objects: neutron-star structure, BH ringdown or echoes, Page time.
  - (c) the HDE c² ≡ η identification against new DESI results.
- **The m_n − m_p sign flip at σ ≈ 0.71 is a sharp structural consequence of the code's decomposition.** Above it, hydrogen would be unstable to electron capture. It only matters where σ ≈ 0.7, i.e. inside horizons under the field reading.
- **The code's Earth σ is about 6× smaller than the value the tests use.** `sigma_from_potential` gives 1.1×10⁻¹⁰, while the tests and docstrings use 7×10⁻¹⁰. This is the "third value" referred to in gap C2.

### Gaps
- `ssbm-theory` is not checked out at `/home/user`. BH-formation thresholds, the 8 bond-failure layers, space-cavitation equations, the TOV/Chandrasekhar code with m_n(σ), the nesting chain and the bridge equations (an alternative σ_conv = 1.086 is cited) could not be read or run.
- The code has no "cosmic age" or H₀ prediction from σ. H₀ is an input (67.4).

---

## Q4. Which cosmology, particle/nuclear, fluid/gas, astrophysics and GR/BH quantities can the code compute? (lookup table)

### Takeaway
Coverage is strong for: Schwarzschild BH thermodynamics, special relativity, nuclear binding and decay, stellar burning rates, plasma parameters, transport and viscosity, and N-body solar-system ephemerides (1PN GR). Coverage is thin for cosmology (fixed Planck-2018 ΛCDM numbers, an HDE density, MOND helpers). Absent entirely: H(z), w0/wa, distance measures, BBN, CMB, Kerr/QNM/GW strain, TOV/Chandrasekhar/polytropes, shocks, and turbulence spectra.

### Cited Findings

#### Lookup table: "new result X → testable via function Y in file Z"

| domain / new-result type | entry point (file:line) | what it computes | status |
|---|---|---|---|
| H₀ measurement (any value) | `field/constants.py:414` `H0 = 67.4e3/3.086e22`; also duplicated at `inventory/core/sigma.py:63` | input constant (Planck 2018); feeds R_H, ρ_crit, age, M_Hubble, SIGMA_FLOOR | implemented; hard-coded single value, no H₀-tension handling |
| Ω_m, Ω_Λ | `field/constants.py:419-420` | 0.315 / 0.685 (Planck 2018, flat) | implemented |
| cosmic age | `mcp/tools/cosmology.py:164` `age_of_universe(mode)` | flat ΛCDM t₀ = (2/(3H₀√Ω_Λ))·asinh(√(Ω_Λ/Ω_m)) = **13.798 Gyr** [run]; Hubble time 14.51 Gyr | implemented (MCP wrapper only; no test file found specifically) |
| Hubble radius | `field/interface/cosmology.py:185` `hubble_radius()` | c/H₀ = 1.3726×10²⁶ m [run] | implemented and tested (`test_cosmology.py`) |
| critical density | `mcp/tools/cosmology.py:147` | 3H₀²/8πG = 8.531×10⁻²⁷ kg/m³ [run] | implemented |
| dark energy (HDE) | `field/interface/cosmology.py:196` `hde_rho_de(c_squared, L)`; `:231` `hde_c_squared_given_rho_de` | ρ_DE = 3c²·c⁴/(8πG L²); at c² = η = 0.4122 and L = R_H it gives 3.16×10⁻¹⁰ J/m³ vs observed Ω_Λρ_c c² = 5.25×10⁻¹⁰ J/m³ (implied c² = 0.685 = Ω_Λ) [run] | implemented and tested (`test_hde_rho_de_at_eta_matches_observed_rho_de_to_factor_eta_over_omega_de`) |
| DESI HDE c² (new DR3 or other release) | `field/constants.py:41-56` (`C_HDE_UNION3 = 0.642`, `C_HDE_DESY5 = 0.701`, `ETA_HDE_UNION3_DR3 = None`, `ETA_HDE_DESY5_DR3 = None`); `cosmology.py:54,82,127` (`eta_candidates`, `eta_coincidence_report`, `eta_in_desi_union3_band('dr3')`) | band check: Union3 1σ band on c² = [0.3770, 0.4489]; DESY5 deviates 19.2% [run] | DR3 slots empty ("CITATION NEEDED"); the dr3 path is a placeholder |
| EDE / interacting DE (H₀, S₈) | `cosmology.py:274` `ede_ide_h0_s8_prediction`; `constants.py:65-70` (`F_EDE_UPPER_95CL = 0.113`, `XI_IDE_UPPER_95CL = 0.071`) | raises `NotImplementedError` ("ΔH₀/ΔS₈ coefficients … not yet lifted") [run] | stub |
| w0/wa, H(z), distances, BBN, CMB, S₈ | none (grep for friedmann / H_of_z / luminosity_distance / w0 found nothing) | — | **absent** |
| MOND a₀ | `cosmology.py:303` `newtonian_regime`; `:326` `a0_mond_from_cosmological_scale`; `constants.py:113` `A0_MOND_MS2 = 1.2e-10` | cH₀/2π = 1.042×10⁻¹⁰ m/s² vs 1.2×10⁻¹⁰ (ratio 0.868) [run] | implemented and tested; a SPARC radial-acceleration-relation comparison is "unrun" (`misc/UNIFICATION_PROGRAM.md` §3.2) |
| neutrino masses and 0νββ | `constants.py:75-106` | Δm²₂₁ = 7.42e-5, Δm²₃₁ = 2.514e-3 (PDG 2022); Σm_ν < 0.12 / 0.072 eV; LEGEND-200 T½ > 0.5e26 yr, combined > 1.9e26 yr | constants only, no computation |
| Lorentz-invariance violation bound | `relativity.py:229` `effective_liv_scale_gev`; `constants.py:137` (Λ₂ > 5.0e19 GeV) | always ∞ | implemented and tested |
| muon g−2 | `electrodynamics.py:274-313`; `constants.py:147-153` | exp 116592061(22), SM 116591810(43) (×10⁻¹¹); `muon_g2_tension_sigmas()` = **5.20σ** [run], yet `MU_G2_TENSION_RESOLVED = True` | implemented and tested (`tests/test_electrodynamics.py:214-243`; asserts 4.5 < t < 6) |
| α, fundamental constants | `constants.py:161-233`; `electrodynamics.py:250` | α derived from e, ε₀, ħ, c gives 1/137.035999065 [run] | implemented |
| particle masses (quarks, leptons, W, Z, H, p, n) | `inventory/data/particle_masses.json` (edition "2026"); `inventory/core/constants.py` `CONSTANTS` | e.g. top 172603.56, W 80362.5, Z 91187.87, Higgs 125130.94 MeV | implemented; drift-gated (`tests/test_pdg_drift_gate.py`) |
| QCD vs Higgs mass fraction | `field/nucleon.py::nucleon_decomposition`; `inventory/core/sigma.py:309` `nucleon_qcd_fraction` | proton QCD fraction 0.9904, neutron 0.9878 (bare masses from PDG MS-bar u and d) | implemented |
| nuclear binding energies | `mcp/tools/nuclear.py:27` `nuclear_binding_energy(protons, neutrons, measured_mass_u)` (exact with measured mass); `field/binding.py` (σ-decomposition, a_C = 0.7111 MeV from r₀ = 1.215 fm) | BE and mass defect | implemented; AME2020 validation in `tests/test_reference_validation.py` |
| nuclear-matter parameters | `constants.py:399-411` | n₀ = 0.16 fm⁻³, E_sat = −16, K = 230, J_sym = 32 MeV | constants only; no EOS solver (gap B1) |
| α/β decay rates, half-lives | `field/decay.py:168,199` (Gamow, Geiger-Nuttall); `field/interface/radioactive_decay.py:392-757` (`alpha_half_life`, `beta_half_life`, `geiger_nuttall_check`, `isotope_decay_properties`); MCP `radioactivity_analysis` | λ, T½, activity, Q-values | implemented and tested (58 + 25 tests) |
| fission energy | `mcp/tools/nuclear.py:132` `fission_energy_release` | J per kg of isotope | implemented |
| stellar fusion (pp/CNO) | `field/interface/nucleosynthesis.py:213-694` | Gamow peak and window, ⟨σv⟩, ε_pp, ε_CNO, pp/CNO crossover temperature, temperature exponents | implemented and tested (44 tests) |
| stellar structure, TOV, Chandrasekhar, white dwarfs, polytropes | none in tree. `field/audit.py` lists "Chandrasekhar Fermi gas (exact)" and "TOV integration (full)" under a relocated `unsolved` module. `test_unsolved_frontiers.py:562` tests only neutron pairing gaps (it says "The existing TOV solver …", but that solver is absent) | — | **absent** (gap B1: "separate plan"; data anchors listed: Chandrasekhar ≈ 1.4 M☉, NICER radii, ≥2 M☉ pulsars) |
| Schwarzschild BH geometry and thermodynamics | `field/gr_basics.py:45-188` | 1 M☉: r_s = 2954 m, r_ISCO = 8862 m, T_H = 6.17×10⁻⁸ K, t_evap = 6.62×10⁷⁴ s, Page time = 3.56×10⁷⁴ s (0.5384·t_evap), S_BH = 1.449×10⁵⁴ J/K [run]; also redshift, time dilation, tidal force, Hawking luminosity | implemented and tested (`tests/test_gr_basics.py`, 35 tests) |
| holographic / "frontier" BH | `mcp/tools/frontier.py:29-181` (`bekenstein_hawking_entropy`, `holographic_matching_mass`, `baryon_vs_disc`, `entanglements_to_pop_bubble`, `unruh_temperature`) | matching mass M = ħc/(4πG m_p) = 2.25×10¹⁰ kg [run] | implemented (SSBM-framed) |
| Kerr spin, QNMs, ringdown, GW strain/chirp | none (no kerr / quasinormal / chirp in physics code). `nbody.py` has only Peters-1964 GW energy loss | — | **absent** in this repo (the QNM work is in relocated ssbm-theory) |
| GR orbital dynamics / ephemerides | `field/interface/nbody.py:255` `NBodySystem` (1PN, J₂, Love k₂, SRP, Forest-Ruth); `orbital.py:365,477`; fixtures `de440_state_vectors.json` and `ssbm_*_predictions.json` | 30-year planetary positions | implemented and tested (93 nbody tests + prediction tests) |
| special relativity | `field/relativity.py:25-179` | γ, E, p, velocity addition, Doppler, contraction | implemented and tested (29 tests) |
| gas EOS and transport | `field/interface/gas.py:454-829`; `dynamics/fluid/eos.py` (Tait P(ρ, ρ₀, K)); `continuum.py` eos ∈ {tait, isothermal, ideal} | ideal-gas density, c_v, c_p, γ, kinetic viscosity, κ, D, Grashof | implemented; real-gas EOS (van der Waals), Joule-Thomson, nozzles, shocks and Mach relations **absent** (gap B2) |
| liquid viscosity, Reynolds number | `fluid.py:196` `eyring_viscosity` (**10³–10⁶× too high for liquid metals**, gap A5); `fluid.py:358` `reynolds_number`; `viscosity.py:94-393` (Stokes, C_D(Re), Poiseuille, boundary layer) | — | implemented; Eyring flagged as wrong |
| turbulence | only drag-coefficient/Reynolds regimes and wind-wave shear (`viscosity.py:159`, MCP `wind_wave_analysis`) | — | no Kolmogorov spectrum or turbulence model |
| SPH hydrodynamics | `dynamics/continuum.py::continuum_step` (exact first law since S8, `energy_ledger()`); 2-D Barnes-Hut | — | implemented; adaptive h, 3-D BH and 1-D transonic flow absent (B6–B8) |
| η/s (viscosity over entropy, KSS) | `scripts/unification/stage9b_eta_over_s.py`; `misc/unification/stage9b_eta_over_s.json` | KSS ħ/4πk_B = 6.08×10⁻¹³ K·s; water η/s = 378× KSS; QGP band 1–3× KSS (literature) | script only |
| plasma | `field/interface/plasma.py:31-241`; MCP `plasma_parameters_analysis` | ω_p, λ_D, N_D, v_A, Coulomb log, Spitzer resistivity, β | implemented and tested |
| superconductivity Tc | `field/interface/superconductivity.py` (McMillan, BCS, GL); `docs/mcmillan_validation.html` | Tc, gap, critical fields | implemented and tested (101 tests) |

- Sources for the table rows: [sigma_ground/field/constants.py](sigma_ground/field/constants.py), [sigma_ground/field/interface/cosmology.py](sigma_ground/field/interface/cosmology.py), [sigma_ground/mcp/tools/cosmology.py](sigma_ground/mcp/tools/cosmology.py), [sigma_ground/field/gr_basics.py](sigma_ground/field/gr_basics.py), [sigma_ground/field/electrodynamics.py](sigma_ground/field/electrodynamics.py), [sigma_ground/field/interface/nbody.py](sigma_ground/field/interface/nbody.py), [sigma_ground/field/interface/nucleosynthesis.py](sigma_ground/field/interface/nucleosynthesis.py), [sigma_ground/field/interface/radioactive_decay.py](sigma_ground/field/interface/radioactive_decay.py), [sigma_ground/mcp/tools/nuclear.py](sigma_ground/mcp/tools/nuclear.py), [sigma_ground/mcp/tools/frontier.py](sigma_ground/mcp/tools/frontier.py), [sigma_ground/inventory/data/particle_masses.json](sigma_ground/inventory/data/particle_masses.json), [misc/MENTAT_GAPS_UNIFICATION.md](misc/MENTAT_GAPS_UNIFICATION.md), [KNOWN_GAPS.md](KNOWN_GAPS.md), [sigma_ground/field/audit.py](sigma_ground/field/audit.py), [sigma_ground/field/interface/test_unsolved_frontiers.py](sigma_ground/field/interface/test_unsolved_frontiers.py).
- **The repo's own HDE test documents a deficit.** HDE with c² = η and L = Hubble radius under-predicts the observed dark-energy density by η/Ω_DE ≈ 0.60. [sigma_ground/field/interface/test_cosmology.py:171](sigma_ground/field/interface/test_cosmology.py)

### Inferences
- **New DESI (DR2/DR3/2026) results.** HDE c² values can be lifted into `ETA_HDE_UNION3_DR3` and `ETA_HDE_DESY5_DR3`. They are then testable with `eta_in_desi_union3_band('dr3')` and `eta_coincidence_report()`, and against the UP-001 falsifier window [0.410, 0.420]. Results stated as w0/wa or H(z) have **no** entry point; the module would need a Friedmann/CPL function.
- **Caveat on the HDE identification.** Standard HDE fits (Li 2004, and the DESI-era fits) normally use the future event horizon as the IR cutoff L. The code's default is the Hubble radius, which makes c² ≡ Ω_Λ trivially. A new HDE c² therefore cannot be compared one-to-one with `hde_rho_de()` at default L. This is based on general knowledge of HDE and was not verified against the papers in this session.
- **New neutron-star mass or radius results** (NICER, heavy pulsars), white-dwarf and Chandrasekhar tests, or BH spin, QNM and echo results (O4/O5) have no live code path. They can be tested only against the `misc/` verdict documents, or after a TOV module (gap B1) is built or ssbm-theory is re-attached.
- **muon g−2.** The stored SM value 116591810(43)×10⁻¹¹ looks like the 2020 Theory Initiative White Paper value, not the 2025 one. The 2025 WP central value is, to my knowledge, ≈116592033(62)×10⁻¹¹. The stored "Fermilab final" 116592061 also differs from the 2025 final world average, ≈116592071.5(14.5). That would explain why the code computes 5.2σ while asserting the tension is resolved. The report writer should verify both numbers against other researchers' sources before stating this. It is a prime "update with 2026 data" target: `constants.py:147-150` plus `tests/test_electrodynamics.py:214-232`.

### Gaps
- I did not find a test file dedicated to `mcp/tools/cosmology.py::age_of_universe`. Benchmark corpora may cover it, but I did not check.
- I did not verify whether any Materia verb (e.g. `black_hole`, `stellar_fusion`) adds physics beyond the field functions listed above.

---

## Q5. What is the Mentat MCP toolset?

### Takeaway
`sigma_ground/mcp/server.py` registers 232 tools on `FastMCP("mentat")`. They are mostly closed-form "intro-to-graduate" calculators plus Materia simulation and "playground" stateful tools. **No MCP tool exposes σ ≠ 0 physics.** The cosmology tools are explicitly "standard, observation-anchored". The server normally runs over stdio inside a Docker container (`skippytel-mentat`) and was unreachable this session.

### Cited Findings
- **Wiring.** Tools are registered with `@server.tool()` inside `server.py`, 232 functions (AST count [run]). `get_manifest` lists all tools by domain and tier. [sigma_ground/mcp/server.py:130-1884](sigma_ground/mcp/server.py)
- **Deployment.** `.mcp.json` runs `docker exec -i … python -m sigma_ground.mcp.server` inside the `skippytel-mentat` container. `tools/mentat_call.sh` offers the same tools over the container's HTTP JSON API (`list`, `<tool> '<json>'`, `ask '<question>'`). [git commit 164b9bb](.mcp.json), [tools/mentat_call.sh](tools/mentat_call.sh)
- **Cosmology tools.** `hubble_radius`, `hde_dark_energy_density(c_squared, L_meters)`, `eta_desi_band_check(dataset)`, `mond_regime_classifier`, `mond_a0_constant`, `critical_density`, `age_of_universe(mode)`, `eta_value_report`, and `holographic_dark_energy_analysis(rho_de_J_m3)` (inverts ρ_DE to c²). The module docstring says: "Mentat reports standard, observation-anchored cosmology … No speculative interpretation is layered on top." [sigma_ground/mcp/server.py:510-556,1750](sigma_ground/mcp/server.py); [sigma_ground/mcp/tools/cosmology.py:1-9](sigma_ground/mcp/tools/cosmology.py)
- **GR/BH tools.** `schwarzschild_radius`, `isco_radius`, `photon_sphere_radius`, `hawking_temperature`, `hawking_evaporation_time`, `gravitational_redshift`, `gravitational_time_dilation`, `procedure_black_hole_profile` (a cascade from r_s through T_H and entropy), `bekenstein_hawking_entropy`, `holographic_matching_mass`, `baryon_vs_disc`, `entanglements_to_pop_bubble`, `unruh_temperature`, and `gravitational_binding_energy`. [sigma_ground/mcp/server.py:315-346,955-1003](sigma_ground/mcp/server.py)
- **Special-relativity tools.** `lorentz_factor`, `relativistic_time_dilation`, `relativistic_length_contraction`, `relativistic_energy`, `relativistic_momentum`, `relativistic_velocity_addition`, `doppler_shift_factor`, `relativistic_energy_analysis`, `procedure_relativistic_particle`. [sigma_ground/mcp/server.py:468-503,1015,1558](sigma_ground/mcp/server.py)
- **Nuclear and atomic tools.** `nuclear_binding_energy(protons, neutrons, measured_mass_u)`, `fission_energy_release`, `radioactivity_analysis`, `first_ionization_energy`, `hydrogen_like_energy_level`, `hydrogen_emission_wavelength`, `rydberg_hydrogen_wavelength`, `de_broglie_*`, `photon_energy_*`, `zeeman_effect_analysis`, `atomic_angular_momentum`, `element_atomic_data`. [sigma_ground/mcp/server.py:661,798-946,1681](sigma_ground/mcp/server.py)
- **Astronomy and orbital tools.** `solar_system_body`, `named_star`, `list_bodies`, `light_travel_time`, `orbital_velocity`, `orbital_period`, `gravitational_force`, `orbital_raise_energy`, `circular_orbit_velocity`, `escape_velocity`, `asteroid_analysis`, `planet_moment_of_inertia_analysis`, `procedure_stellar_blackbody`. [sigma_ground/mcp/server.py:417-423,833-906,1722,1771](sigma_ground/mcp/server.py)
- **Thermodynamics, fluids and plasma tools.** `ideal_gas_pressure/volume`, `blackbody_peak_wavelength`, `blackbody_total_power`, `carnot_efficiency`, `speed_of_sound_in_ideal_gas`, `maxwell_boltzmann_most_probable_speed`, `statistical_distribution`, `viscous_flow_analysis` (Reynolds number, Stokes drag, terminal velocity), `diffusion_analysis`, `natural_convection_analysis`, `buoyancy_analysis`, `wind_wave_analysis`, `plasma_parameters_analysis`. [sigma_ground/mcp/server.py:556-615,1263-1280,1412,1534,1858-1867](sigma_ground/mcp/server.py)
- **Simulation tools.** `simulate(scenario)` (natural-language what-if through Materia), `run_simulation(verb, params)`, `list_simulation_scenarios`, plus nine stateful `playground_*` tools (load, inspect, apply, simulate, render, reset, status, clear, make). [sigma_ground/mcp/server.py:1041-1066,1779-1839](sigma_ground/mcp/server.py)
- **Remaining tool groups.** About 20 symbolic-math tools (sympy solve, integrate, ODE, transforms, vector calculus); constants and units (`lookup_constant`, `list_constants`, `convert_units`, `parse_quantity`); and materials, solid-state, electronics, chemistry, optics, quantum-computing and tribology analyzers (lines 688–1736). [sigma_ground/mcp/server.py](sigma_ground/mcp/server.py)
- **Known defects.** Mentat's `lookup_constant` cannot resolve the Planck length even though `L_PLANCK` exists (gap B9), and `melting_point` lacks titanium (B10). [misc/MENTAT_GAPS_UNIFICATION.md](misc/MENTAT_GAPS_UNIFICATION.md)

### Inferences
- If Mentat is restored, the cosmology and GR tools give quick checks: R_H, ρ_crit, t₀, HDE c² from a new ρ_DE, BH temperature and entropy. Any σ-specific test still has to go through Python imports such as `inventory.core.sigma` or `field.nucleon`, not MCP.

### Gaps
- I could not verify tool behaviour live because of the MCP connection failure. The descriptions come from source docstrings.

---

## Q6. What do the tests look like, which validate against literature, and which constants editions are used?

### Takeaway
There are about 5,155 `def test_` functions in the repo (static count). Of these, 1,136 are in `tests/`, about 2,900 are co-located in `sigma_ground/field/interface/`, and the rest are in dynamics, MCP, radiance and deckard. The literature-anchored tests are a CODATA/PDG/AME/NIST reference suite, a PDG-2026 drift gate, cosmology/DESI tests, and SHA-256-hashed 30-year ephemeris predictions. Constants are a mix: `field/constants.py` is mostly CODATA 2018 with a pre-2019 ε₀, while the distilled `codata_constants.json` is CODATA 2022.

### Cited Findings
- **Counts** (grep of `def test_` in `test_*.py` [run]):
  - total 5,155; `tests/` 1,136
  - largest physics files: `test_superconductivity.py` 101, `test_nbody.py` 93, `test_element.py` 78, `test_molecular_bonds.py` 75, `test_atomic_spectra.py` 63, `test_radioactive_decay.py` 58, `test_gas.py` 49, `test_nucleosynthesis.py` 44, `test_unsolved_frontiers.py` 44, `test_viscosity.py` 33, `test_cosmology.py` 29
  - `tests/test_gr_basics.py` 35, `tests/test_relativity.py` 29, `tests/test_decay.py` 25, `tests/test_plasma.py` 34, `tests/test_statistical.py` 34, `tests/test_electrodynamics.py` 28
  - `dynamics/test_continuum*.py` 59
  - [sigma_ground/field/interface/](sigma_ground/field/interface/), [tests/](tests/)
- **Reference validation suite.** `tests/test_reference_validation.py` has 77 tests against "CODATA 2018 / 2022 (NIST), PDG 2024, IAEA AME2020, NIST ASD (ionization energies)". Tolerance is the measurement uncertainty or 0.1%, whichever is larger. [tests/test_reference_validation.py:1-40](tests/test_reference_validation.py)
- **PDG drift gate.** `tests/test_pdg_drift_gate.py` holds `inventory/core/constants.py` and `field/constants.py` to the vendored snapshot `inventory/data/particle_masses.json` (`_edition: "2026"`). Adopting a new PDG edition is a "DELIBERATE, loud, single-commit event": run `tools/distill_pdg.py`, the gate fails on each drifted value, then a human updates the constants. [tests/test_pdg_drift_gate.py:1-46](tests/test_pdg_drift_gate.py); [sigma_ground/inventory/data/particle_masses.json](sigma_ground/inventory/data/particle_masses.json)
- **Cosmology tests.** `test_cosmology.py` covers the DESI Union3 band, DESY5 disagreement, DR3 pending, HDE round-trip and scaling, HDE-vs-observed factor η/Ω_DE, EDE/IDE upper bounds matching arXiv:2505.23382 Table 2, "refuses to fabricate", MOND regimes, and a₀ vs SPARC. [sigma_ground/field/interface/test_cosmology.py:28-268](sigma_ground/field/interface/test_cosmology.py)
- **Pre-registered predictions with hashes.**
  - `fixtures/ssbm_future_predictions.json` (created 2026-03-24, 360 months from J2025, Forest-Ruth 4th order + GR, 9 bodies, `predictions_sha256`)
  - `ssbm_minor_bodies_predictions.json` (17 moons plus Pluto/Charon, Keplerian fit to DE440)
  - `ssbm_over_physics_predictions.json` (all 26 DE440 bodies, with solar radiation pressure)
  - `de440_state_vectors.json`
  - Tests: `test_future_predictions.py` (11), `test_minor_bodies_predictions.py` (10), `test_shootout.py`
  - Network-gated `test_jpl_ephemeris` is permanently red.
  - [sigma_ground/field/interface/fixtures/](sigma_ground/field/interface/fixtures/); [ARCHITECTURE.md](ARCHITECTURE.md)
- **Constants editions.** [sigma_ground/field/constants.py:161-233](sigma_ground/field/constants.py); [sigma_ground/inventory/data/codata_constants.json](sigma_ground/inventory/data/codata_constants.json); [sigma_ground/field/PROVENANCE.md](sigma_ground/field/PROVENANCE.md)
  - `field/constants.py`: G = 6.67430e-11, ħ = 1.054571817e-34, m_e = 9.1093837015e-31 kg ("CODATA 2018"), μ₀ = 1.25663706212e-6 (CODATA 2018), AMU = 1.66053906660e-27 (CODATA 2018). EPS_0 = 8.854187817e-12 is the pre-2019 exact value. α is derived from these and gives α⁻¹ = 137.035999065 [run].
  - `codata_constants.json` holds CODATA 2022 values (α = 0.0072973525643(11), ε₀ = 8.8541878188e-12).
  - `PROVENANCE.md` classifies 89 quantities: 52 measured, 30 derived, 1 free input, 4 speculative-pending, 1 rejected.
- **Earth-σ tests.** About 20 `test_earth_sigma_negligible` tests use σ = 7×10⁻¹⁰ (see Q3). [sigma_ground/field/interface/test_friction.py:297](sigma_ground/field/interface/test_friction.py)

### Inferences
- **Most valuable update hooks for 2026 measurements.**
  - New CODATA or PDG editions go through the drift gate and the `test_reference_validation.py` expected values.
  - DESI HDE c² goes into the `constants.py` DR3 slots and the `test_cosmology.py` band tests.
  - Muon g−2 goes into `constants.py:147-153` and `tests/test_electrodynamics.py:214-243`.
  - Neutrino and 0νββ bounds go into `constants.py:75-106`.
  - New solar-system ephemerides can score the hashed 30-year predictions (an honest pre-registration).
- `field/constants.py` lags the CODATA 2022 JSON by about 10⁻⁹ relative on α. That is negligible for physics but would fail a strict CODATA-2022 equality test.

### Gaps
- I could not run pytest, so I have no fresh pass/fail state. The latest recorded state is ARCHITECTURE.md's baseline and Session 34's full run.

---

## Q7. Where are the known gaps where a new external result would be especially valuable?

### Takeaway
The highest-value gaps for new astro/cosmo results are:
- **B1**, nuclear and degenerate matter (TOV, Chandrasekhar, neutron-star EOS)
- the **missing cosmology functions** (H(z), w0/wa, distances)
- the **empty DESI DR3 slots and the EDE/IDE stub**
- **B4/B5**, a σ-wave solver and the Ricci sign convention
- the open σ-theory decisions **C2–C5**

Engine defects (Eyring viscosity, softened N-body energy, viscoelastic T_melt) are the main measured-data correction targets.

### Cited Findings
- **KNOWN_GAPS.md open entries.** [KNOWN_GAPS.md](KNOWN_GAPS.md)
  - electrolyte/soil corrosion kinetics (dry-air Wagner only)
  - RevoluteJoint instability under orientation-coupled torque
  - uncited gear module and hand-tool dimensions
  - single-mode bell acoustics
  - load-blind bearing-gear coupling
  - `NBodySystem.total_energy()` ignores Plummer softening (0.96% reported drift vs 2.4×10⁻⁵ true)
  - viscoelastic relaxation uses a silent 1800 K T_melt
  - **Eyring liquid-metal viscosity 10³–10⁶× too high**: Al 1170 Pa·s, Cu 10.8, Fe 4.06 at T_melt vs a few mPa·s measured (Assael 2006 suggested as the reference)
  - The SPH first-law gap was CLOSED on 2026-10-10.
- **Missing capabilities.** [misc/MENTAT_GAPS_UNIFICATION.md](misc/MENTAT_GAPS_UNIFICATION.md)
  - **B1**: electron and neutron Fermi-gas EOS, semi-empirical mass formula, saturation and symmetry energy, TOV integrator, bag-model deconfinement. Anchors: Chandrasekhar mass ≈ 1.4 M☉, NICER radii, ≥2 M☉ pulsars. Status "separate plan (Captain's decision)".
  - **B2**: real-gas EOS, Joule-Thomson, compressible flow (nozzles, shocks, Mach).
  - **B3**: FEM/MPM.
  - **B4**: σ-wave solver.
  - **B5**: Ricci convention.
  - **B6**: 1-D transonic flow (Bondi).
  - **B7**: adaptive h.
  - **B8**: 3-D Barnes-Hut.
  - **B9/B10**: Mentat lookup holes.
  - **A8**: Quarksum mass closure misses by 0.6–1.4% for the gold ring, bronze and seawater.
- **Cosmology stubs.** DR3 HDE slots are `None` with "CITATION NEEDED" (arXiv:2512.07281 was checked and lacks HDE content). `ede_ide_h0_s8_prediction` raises `NotImplementedError` pending ΔH₀/ΔS₈ coefficients. `DARK_AXION_PORTAL_Z_SENSITIVITY = None` and `NEUTRINO_NATURE = None` are speculative-pending. [sigma_ground/field/constants.py:46-56,96,129](sigma_ground/field/constants.py); [sigma_ground/field/interface/cosmology.py:274](sigma_ground/field/interface/cosmology.py)
- **Unrun discriminators.** [misc/UNIFICATION_PROGRAM.md:122-138](misc/UNIFICATION_PROGRAM.md)
  - analogue sonic horizon (does a sound pulse escape?)
  - σ monopole radiation from a pressure pulsation
  - a₀ = cH₀/2π vs the SPARC radial acceleration relation (g† = 1.20×10⁻¹⁰; "download needs the Captain's OK")
  - a Jacobson-derivation sympy reproduction
  - η/s placement

### Inferences
- **Ranked by how directly a 2026 external result plugs in.**
  1. DESI/Euclid dark-energy results: fill the DR3 slots, or add w0/wa and H(z) functions.
  2. New equivalence-principle, clock or μ-variation bounds: rerun `sigma_exclusion.py`.
  3. NICER/LIGO neutron-star mass-radius and heavy-pulsar results: need B1 first.
  4. LIGO O4 ringdown, QNM and echo results: compare against Γ ∈ {0.84, 0.79, 0.75} and Δt_1 = 2.26 ms for GW150914-like remnants in the `misc/` docs, or against B2's [135, 200] M☉ window. These need the relocated ssbm-theory code.
  5. New muon g−2 and PDG values: constants plus drift gate.
  6. Liquid-metal viscosity data: fixes A5.

### Gaps
- None beyond those noted in Q3 and Q4.

---

## Q8. What does recent work (SESSION_LOG and unification program) say about cosmology, dark energy, BH formation or σ predictions?

### Takeaway
The last SESSION_LOG entry is Session 35 (2026-08-06), about rendering and engine recovery, with no cosmology content. The σ and gravity thread was active again on 2026-10-09/10 in the "unification program" (stages S0–S11). Its headline result is S9a: long-range σ is excluded, and only a fm-range or <35 km-range σ survives. The open σ decisions belong to the "Captain". Earlier threads: the η re-anchoring to DESI (2026-05-15) and two open SSBM questions about Bekenstein saturation.

### Cited Findings
- **Last ~150 lines of SESSION_LOG.** Sessions 34–35 (2026-08-05/06) covered the Ellipsoid SDF fix, the browser gauntlet, post-incident recovery and standing directives (no unattended multi-agent runs, live-UI tests paused). There are no cosmology or BH items. [misc/SESSION_LOG.md:2800-2939](misc/SESSION_LOG.md)
- **Earlier σ/η threads in the log.** [misc/SESSION_LOG.md:1657-1803](misc/SESSION_LOG.md)
  - The 2026-05-15 audit found ETA_FORMULA = exp(−φ/σ_conv) to be "numerology". ETA was reassigned as [EMPIRICAL-INPUT] at the DESI 2024 Union3 HDE c², and the SSBM free inputs went from "1 (with hidden numerology)" to 2 (XI, ETA).
  - Three SSBM-framed holographic tools were added (bubble-pop, matching-mass, baryon-vs-disc).
  - Two open forks remain: "does SSBM 'reform' trigger AT or BEFORE Bekenstein saturation", and "which side of M ≈ 2.25e10 kg SSBM cares about".
- **Commit history.** Commits on 2026-10-10 include "Unification S9a: what sigma is not … cut against data", "S9b, S10, S11: eta/s axis, Quarksum as sigma's ledger, Mentat gap census", "Exact first law on the SPH engine", and "pilot P7: three fields of one operator on an n=1 polytrope (σ-bridge)". (`git log` [run])
- **σ-bridge statement and status.** The field equation with a fluid source is (1/c²)∂²σ/∂t² − ∇²σ = (8πGξ/c⁴)(ρc² − 3P). It predicts monopole σ radiation from pressure pulsations only: σ ≈ (2Gξ/c²r)[M − (3/c²)∫P dV](t − r/c). It is "untested … the repo has no σ-wave solver". [misc/UNIFICATION_PROGRAM.md:95-120](misc/UNIFICATION_PROGRAM.md)
- **Open decisions.** C1 asks where the SSBM constants live: `bounds.py` and ξ were excised from `field/` in commit e66615b, and only `inventory/core/sigma.py` keeps ξ. C2–C5 are listed under Q3. [misc/MENTAT_GAPS_UNIFICATION.md](misc/MENTAT_GAPS_UNIFICATION.md) (section C)
- **η/s placement.** KSS = 6.08×10⁻¹³ K·s, water = 378× KSS, horizon = 1× KSS, QGP 1–3× KSS (literature band). [misc/unification/stage9b_eta_over_s.json](misc/unification/stage9b_eta_over_s.json)

### Inferences
- **The project is mid-pivot.** The σ theory is being treated as a falsifiable hypothesis and cut against data, not used as a working assumption. A new 2026 result is most useful to the project if it either tightens the short-range σ window (torsion balances, nuclear-scale μ variation) or gives a dark-energy or HDE number for the η slot.
- The session log has not recorded Session 36 (the unification work). Its write-up lives in `misc/UNIFICATION_PROGRAM.md` and `misc/UNIFICATION_PILOT_RESULTS.md` instead.

### Gaps
- I did not read `misc/UNIFICATION_PILOT_RESULTS.md` or the S10/S11 sections of the program document in full.
