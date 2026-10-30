# Astrophysics review of sigma-ground: what the n-body work lets us fold in (2026-10-30)

Scope the Captain chose: celestial mechanics and ephemerides; gravity, GR tests, tides and figure; stars, compact objects and cosmology; radiation and thermal of bodies.
Deliverable: this review and a ranked list. Nothing below is built yet; each item names the data that grounds it and the test that would disprove it (data-alone doctrine, CLAUDE.md).

**What this review read.** The MCP tool surfaces (`mcp/tools/{astronomy,orbital,gr,relativity,cosmology,rotation,photonics,thermal_systems,misc_physics}.py`: names and what each docstring claims), `field/{gr_basics,planets,asteroids}.py`
and `field/dense/` by listing, the n-body fixtures and ledgers (C1-C20), `misc/COVERAGE_LEDGER.md`. It did NOT re-derive each formula or re-check each tool's numbers; "serves today" below means the tool exists and says it does it.
A formula audit of the older tools is a separate piece of work (item 13).

## 1. What the n-body work gives us to draw on

| Asset | What it is | Where |
|---|---|---|
| The answer key | JPL DE440 state vectors, 56 bodies at annual snapshots (r8), perturbers, header constants, Horizons copies of centres, moons, comets, asteroids, KBOs | `fixtures/de440_*`, `D:/datasets/jpl` |
| The predictor | Forest-Ruth N-body with 1PN/EIH, zonals J2-J4, tides, solar pressure, the Moon's C22, asteroid/ring/KBO perturbers; labels `restated_zon`, `restated_hdr` | `nbody.py`, `rolling_shootout.py` |
| Measured skill | hindcast by lead for 11 bodies; 21-window shoot-out for 55 bodies at +5 and +20 yr; converged moons (C14); step convergence (C19) | `prediction_error_table.json`, `SHOOTOUT_FULL_RESTATED_r8.md`, `SYSTEM_LOCAL_MOONS.md` |
| Layer ablations | the same windows with tides, J3, J4, 2PN switched on/off, and the asteroid ladder | `rolling_shootout_toggle_iteration.json`, `D:/datasets/sigma_runs/asteroid_ladder` |
| Facts that survived testing | order 3.95-4.09 on the real Jupiter system; Moon 0.46x error with header constants (4/4 starts), Mercury 1.34x worse; 14 moons lose phase within a year; measured zonals fix Callisto/Titan | C12-C20 |

## 2. Area by area

### Celestial mechanics and ephemerides
Serves today: `body_position` (known from DE440 to today, labelled prediction after, with the measured error), `body_constants`, `solar_system_body`, `small_body` (343 asteroids, 30 KBOs, 20 comets), `orbital_velocity`, `orbital_period` (Kepler III), `gravitational_force`, `surface_gravity`.
Not served although the engine or data has it: skill for bodies other than the 11 reference points; which bodies cannot be predicted; steps required; orbital elements of a planet or moon at a date; Hill radius, Roche limit (a method on `NBodySystem`, no tool), Lagrange points, synodic period, transfer burns, sphere of influence.
Cannot be served from what we hold: positions of moons other than ours. The r8 snapshots are annual, which cannot be interpolated for Io (1.77 d period). That needs the satellite kernels (JUP365 and so on), a download.

### Gravity, GR tests, tides, figure
Serves today: Schwarzschild radius, ISCO, photon sphere, Hawking temperature and time, gravitational redshift and time dilation (all closed-form, `gr.py`), Lorentz and Doppler (`relativity.py`).
Engine only, no tool: perihelion advance and its budget, geodetic precession, the size of each layer (GR, J2, tides, asteroids) per body and lead, gravitational-wave loss (`test_nbody_gw.py`), the Moon's tidal recession from the Earth's tidal delay (the delay and k2 are in the header file we hold).
Not present anywhere: Shapiro delay, light deflection.

### Stars, compact objects, cosmology
Serves today: `named_star` (table of bright stars, source per star not checked here), blackbody tools, critical density, Hubble radius, age of the universe, MOND classifier, holographic dark-energy density with the DESI band check.
In the library, unwired (the coverage ledger already tracks it as a wiring gap): `field/dense/` white dwarf structure and mass-radius, Chandrasekhar mass, polytropes, TOV (neutron-star maximum mass), nuclear matter, Fermi gas.
Not present: mass-luminosity, Eddington luminosity, Jeans mass, stellar lifetimes.
Flag, not remove: `hde_*`, `eta_*`, `mond_*` carry an adopted free parameter (`EMPIRICAL-INPUT`) and a data check. Under the data-alone rule they stay only as parameters with a stated test; the DESI band check is such a test, the density itself is not a result.

### Radiation and thermal of bodies
Serves today: blackbody peak and total power, light travel time, thermal tools for engineering contacts and convection; the engine has solar radiation pressure (`area_m2`, `reflectivity`).
Not present: equilibrium temperature of a body from its distance and albedo, albedo/phase/brightness of a body, Yarkovsky drift. Data held for a test: asteroid albedos and diameters in the SBDB copies; Earth and Moon radiation constants are not held.

## 3. Ranked fold-in list

Rank = (decisions taken from the solver) x (data we already hold) / effort. "Test" is what would disprove it; every test uses data already on disk unless marked.

| # | Item | Grounded by | Test (disproves it if it fails) | Solver decision removed | Effort |
|---|---|---|---|---|---|
| 1 | `body_skill(body)`: predictability class (predictable to km / loses phase within a year / unscored) and measured error at +1, +5, +20 yr for all 55 bodies | r8 shoot-out, C14 converged moons | tool returns the ledger numbers exactly; a body in the "loses phase" class never gets a position claim | whether a body can be predicted and how well | S |
| 2 | `step_check(primary, moons, dt)`: wire `step_rule.step_adequacy` (committed ae32d22) to a tool | C14 rule, C19 order 3.95 | Io 71 steps/orbit adequate=false; Kepler periods to 0.1 % | which step to use | S |
| 3 | `layer_effect(body, lead)`: size in km of GR, J2/J3/J4, tides, asteroids, solar pressure on that body at that lead | toggle ablations against DE440 | the sum of switching a layer off reproduces the stored paired difference to 1e-6 km | "does X matter here" | M |
| 4 | `perihelion_advance(body)`: closed form 6 pi GM / (c^2 a (1-e^2)) beside the engine's GR-on minus GR-off | `test_gr_precession.py`, DE440 | the two agree for Mercury within the noise of the other planets' Newtonian pull (the textbook value is about 43 arcsec per century; its source to be read before it is quoted); disagreement = a bug in one | formula choice, units | S |
| 5 | Orbit set with body lookup: `hill_radius`, `roche_limit` (engine method exists), `lagrange_points`, `synodic_period`, `hohmann_transfer`, `sphere_of_influence` | GM from `body_constants` (DE440) | independent recomputation from the DE440 state at a date (Earth's Hill radius, the Sun-Earth L1 distance, the Earth-Mars synodic period from the two DE440 periods) | choosing the formula and the GM | S each |
| 6 | `orbital_elements(body, when)` from a DE440 state | SBDB osculating elements (343 asteroids held) | elements from our state vector equal SBDB's at the same epoch to the catalogue's precision | state-to-elements conversion | M |
| 7 | Close the constants audit: Uranus and Neptune J2 R^2 and Mars's radius convention (the rows C15 could not compare) | header file, papers | each row compared with its source and the table moved or the note stands | none; removes a known unknown | S |
| 8 | Wire `white_dwarf_*`, `chandrasekhar_mass`, `tov` as tools | `field/dense` tests | the tool's Chandrasekhar mass at mu_e=2 equals the textbook closed form evaluated independently; Sirius B mass-radius from a published measurement (source to be read: not yet fetched) | EOS and formula choice | S |
| 9 | Lunar tidal recession from the header's Earth tidal delay and k2 | DE440 header | against LLR's measured recession (about 3.8 cm/yr; the number must be read from its paper first, not from memory) | none yet; a new derived claim | M |
| 10 | GW inspiral of a binary from the engine's GW loss | `test_nbody_gw.py` | published binary-pulsar orbital decay (source to be read) | formula choice | M |
| 11 | Equilibrium temperature and phase brightness of a body | SBDB albedo and diameter | predicted against the catalogue's measured values for the 343 asteroids; a median error is reported, not hidden | formula, albedo lookup | M |
| 12 | Moon positions (Io, Titan, ...) as `known` | satellite kernels (a download) | kernel against Horizons copies we hold | "no tool for moons" | M + approval |
| 13 | Formula audit of the older closed-form tools (gr, relativity, orbital, cosmology) against independent recomputation | CODATA constants | each tool equals a second implementation to 1e-12 | none; finds bugs | M |

Order I would build in: 2, 1, 4, 5, 3, then 7, 6, 8; 9-12 need a source read or an approval first. All of 1-8 use data on disk and need no new long run.

## 4. What stays out until it can be disproved
* The Moon's header constants as a default: gain 0.46x, but Mercury 1.34x and Venus 1.08x worse; needs the one-body arms (C20). Not folded in.
* Any statement that the model "explains" the 14 phase-losing moons, Nereid, or Uranus's 5 mm/s: causes unknown.
* `eta`/`hde`/`mond` as results: they are parameters with a check, nothing more.

## 5. Needs the Captain
* Item 12 downloads the satellite kernels (public JPL files, free, large): approval and a licence check first.
* New tools change Mentat's surface; they reach the live service when sigma-ground's pin in SkippyTel is bumped, which the Mentat thread owns.
