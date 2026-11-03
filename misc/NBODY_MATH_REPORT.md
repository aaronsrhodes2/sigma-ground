# The N-body predictor: the mathematics, term by term, with its sources (2026-11-02)

The Captain's standard for this report: we calculate our way with known terms, not guesses; nothing is fitted to make a number come out; and the one quantity that is ever
solved from data, a body's starting state, is shown as such, apart from the physics. Every row below names the equation, where it comes from, the constants it uses and where
they come from, and which of our tests checks it. The code is `sigma_ground/field/interface/` (engine `nbody.py`, model `reduced_model.py`, layers `rolling_shootout.py`,
`figure_forces.py`, `lunar_figure.py`, `prescribed_sources.py`, `perturber_paths.py`); the claims and their tests are in [CANDIDATE_LEDGER.md](CANDIDATE_LEDGER.md).

## 1. What is calculated and what is read

| kind | what | status |
|---|---|---|
| **Known fact, read** | JPL's ephemerides: DE440 (the start state of every run and the paths of the 343 asteroids, the ring and the Kuiper-belt objects), DE442 (reality, the fit to the measurements through about 2024), the past DE403 to DE438 (JPL's predictions at their issue dates) | public NAIF kernels, checksums verified; never edited |
| **Known fact, read** | the constants: DE440's header (GM of every body, the Earth-Moon mass ratio, J2/J3/J4 of the Sun, Earth and Moon, the Moon's C22, the Earth's three-band Love numbers and time delays, the asteroid GMs, GMSDOT), Juno's Jupiter J2, Cassini's Saturn J2/J4, GRAIL's lunar figure, IERS's Earth rotation rate, the IAU pole models | each cited at the line that uses it; `fixtures/de440_header_constants.json` holds the header verbatim, `de440_constants.py` calculates from it (never a stored derivative) |
| **Solved from observations** | a body's starting state when we do not borrow JPL's (C28: Eros from 2015-2019 MPC positions) | least squares of six numbers to measured sky positions, with the covariance reported; the physics is held fixed while it runs |
| **Calculated** | everything else: every force, every step, every position we quote for a future date | the equations of sections 2-4, run at the time of the question (the Captain's rule, 2026-11-02) |
| **Measured** | how far the calculation lands from reality, by body and lead (the error tables, the layer sizes, the contest) | reported with every answer; a measurement of the calculation, not an input to it |

## 2. The equations of motion (every layer, its source, its constants, its test)

The state of each body is its barycentric position and velocity in the ICRF, time in TDB seconds. The acceleration of body i is the sum of the layers below; a layer is on or
off by a named toggle (`PhysicsToggles`), and a predictor label is a fixed list of toggles and constants that is never edited once results are published.

| layer | equation (as implemented) | source | constants and their source | checked by |
|---|---|---|---|---|
| Newtonian gravity | a_i = sum_j GM_j (x_j - x_i) / r^3 | Newton | GM_j: DE440 header (planets, Sun, Earth/Moon split by EMRAT); satellite ephemerides JUP365, SAT441, URA111, NEP095, PLU058, MAR097 for the moons | `test_nbody.py` (two-body orbit against the closed form) |
| relativity, N-body first post-Newtonian | the Einstein-Infeld-Hoffmann equations: velocity, potential and acceleration cross terms between every pair, c = 299,792,458 m/s | Einstein, Infeld & Hoffmann 1938; Will 1993 as written; the form used by DE (Moyer 2003) | c exact (SI); GM as above | `test_nbody_integrator_order.py`: the engine against scipy's DOP853 on the textbook law; Mercury's perihelion advance 42.98"/cy reproduced (ledger C23b: relativity is 1046 km of Mercury's 5-yr error) |
| oblateness J2, J3, J4 | the gradient of the zonal terms of the potential, -GM R^n J_n P_n(cos theta)/r^(n+1), about each body's IAU spin axis, with the equal and opposite reaction on the oblate body | Vallado 2013 §9.4 (J2); the Legendre expansion (J3, J4) | Sun, Earth, Moon: DE440 header on its reference radii (RE 6378.1366, AM 1738.0, ASUN 696000 km); Jupiter: Juno (Folkner et al. 2017); Saturn: Cassini Grand Finale (Iess et al. 2019); others: NASA fact sheets, flagged as such | `test_nbody_zonal_reaction.py` (analytic J2 acceleration to 1.6e-7, momentum conserved); `test_math_levers.py` (the reference radius is its own field: the 2026-11-02 fix) |
| the Moon's triaxial figure C22 | MacCullagh's quadrupole, T = 2 C22 M R^2 (y y^T - x x^T), with the long axis from the IAU rotation model, evaluated at the sub-step time | MacCullagh 1844; Archinal et al. 2011 (IAU WGCCRE) | C22 = 2.2380850528e-5 on 1738 km (DE440 header, from GRAIL) | `test_nbody_sectoral_c22.py`, `test_nbody_substep_time.py` |
| tides, static | the Love-number bulge: induced potential -k2 GM_j R_i^5 P2 / (d^3 r^3), raised on each body by its dominant perturber, acting on every other body with the reaction | Love 1911; Cowling 1938 (apsidal motion check) | k2: Earth 0.299, Moon 0.024, planets per the literature (table in `rolling_shootout._BODY_PARAMS`) | `test_nbody.py` tidal apsidal motion |
| tides with a time lag (lever L3, 2026-11-02) | DE430's three-band model: the tide-raising body's position taken at t - tau_j and turned about the Earth's spin axis by theta_dot tau_j, bands j = 0 (long period), 1 (diurnal), 2 (semi-diurnal); the acceleration of the Moon relative to the Earth, eq. (32) | Folkner, Williams, Boggs, Park & Kuchynka 2014, IPN Progress Report 42-196, eqs. (31)-(32) | k2 = 0.335 / 0.277 / 0.283, tau = 0.077 / -0.01 / -0.2098 d, tau_R = 0 / 0.01193 / -0.00074 d (DE440 header); theta_dot = 7.292115e-5 rad/s (IERS Conventions 2010) | `test_de_tide.py`: with zero delays and one k2 the three bands sum to the isotropic Love result for any pole direction (the transcription check); the barycentre is untouched; `scripts/tide_secular_check.py`: the Moon's mean-longitude acceleration comes out -24.1"/cy^2 against DE430's -25.82 |
| solar radiation pressure | a = L (1 + reflectivity) A / (4 pi c m r^2) r_hat | classical | L_sun = 3.828e26 W (IAU 2015); areas pi R^2 | `test_nbody.py` |
| the 343 asteroids, the Kuiper-belt ring, the identified Kuiper-belt objects | Newtonian pull of prescribed massive sources whose paths are tabulated (daily nodes, cubic Hermite between them), the asteroids and KBOs integrated as test particles in DE440's planet field from their JPL elements, the ring as 36 equal points at 44 au | the DE440 force model (Park et al. 2021 §2.2) | the 343 GMs and the ring's total GM: DE440 header; elements: JPL SBDB | `test_restated_predictors.py` (header masses reproduced); `PRESCRIBED_EQUIVALENCE.md` (prescribed sources against the same bodies integrated, 2e-4 km) |
| the Sun's mass loss (lever L4) | GM_sun(t) = GM_sun (1 + GMSDOT t) | DE440 header | GMSDOT = -9e-14 per year | `test_de_tide.py` (GM rate) |
| not in the model (open, recorded) | integrated lunar librations and the figure-figure torque, Lense-Thirring, the Moon's own time-lagged tide (K2M, TAUM), the ring gravity of Saturn on its ring-edge moons, Earth J5 and the lunar degree-3 field | | | NBODY_HANDOFF.md §7, NBODY_CAMPAIGN.md §6 |

Nothing in this table was adjusted to improve agreement. Where a constant has two published values (the Sun's J2: our table's 2.1106e-7 against the header's 2.19614e-7), both are run as separate labels and the ledger says which agrees with reality and by how much; neither is "tuned".

## 3. The integration (how the equations are stepped)

| piece | what | source | test |
|---|---|---|---|
| Forest-Ruth step | 4th-order symplectic composition, three force evaluations a step; the velocity-dependent (relativistic) kick is the time-symmetric implicit trapezoidal kick solved by fixed-point iteration, so the order is 4 with relativity on | Forest & Ruth 1990; Hairer, Lubich & Wanner 2006 | `test_nbody_integrator_order.py` (measured order 4.0 against DOP853) |
| Yoshida steps (2026-11-02) | 6th- and 8th-order compositions of the Verlet stage (solution A coefficients), both implicit kicks kept per stage with relativity on (merging them breaks the order, measured) | Yoshida 1990, Phys. Lett. A 150, 262 | `test_yoshida_order.py` (orders 6 and 8 measured); ledger C26 Phase 3 (the same answer as fine Forest-Ruth at 60 % of the evaluations) |
| the step | 0.0125 d is the converged step for the inner system (Mercury and the Moon); 0.025 d is the step of the sealed 2026-10-27 forecast; coarser steps are never served | ledger C19, C20 | `MOON_STEP_CONVERGENCE.md`, `FINE_STEP_ALL_BODIES.md` |
| sub-step time | time-dependent terms (the rotating lunar figure, the tabulated sources, the lagged tide, the GM rate) are evaluated at the time of the sub-step, not the step's start | | `test_nbody_substep_time.py` |
| the reduced model | Sun, Mercury, Venus, Earth, Moon, Mars; Jupiter to Pluto as their system barycentres (their moons' masses summed) | validated against the full 56-body run to 1e-6 km without layers, 2e-4 km with | `tests/test_reduced_model.py` |

## 4. The starting state and what "prediction" means here

* **Borrowed start (the shoot-out, the hindcast tables, the contest):** the state is read from a JPL kernel at the start date; what is measured is how far our dynamics drift from
  reality afterwards. This is dynamics parity with a quantified deficit, never a forecast contest against JPL (NBODY_HANDOFF.md §1).
* **Own start (C28, 2026-11-02):** Eros's state at an epoch was solved by least squares from 1,563 MPC optical positions of 2015-2019, with our propagation as the model, finite-difference
  partials, equal per-observation weights and 3-sigma rejection; post-fit RMS 0.369", 33 km and 2.4 mm/s from JPL's state; it then predicted 4,435 later positions as well as JPL
  for about four years. The six solved numbers and their covariance are reported; no constant of section 2 moved.
* **The fair contest:** both we and JPL's own past ephemeris start from that ephemeris's state at its issue date, so the comparison is dynamics against dynamics; reality is DE442 where
  successive ephemerides have converged, with the DE440-DE442 spread shown as how well reality is known there.

## 5. How a number is reported

Every served position carries: the predictor label, the fingerprint of the physics code and constants it was calculated with, the step and integrator, the start (which kernel, what
date), the lead, JPL's value beside it, and the measured error at that lead from the hindcast table. A stored table (the 2026-10-27 sealed forecast) is a record to be judged later,
not the serving path; the serving path is the calculation itself (the live `body_position`, in progress 2026-11-02).

## 6. Where the proofs are

| claim | where |
|---|---|
| each layer's size for each body (what each term earned) | `fixtures/layer_effect_table.json`, Mentat `layer_effect`, ledger C23b |
| the error by body and lead (the deficit) | `fixtures/prediction_error_table.json`, `v2physics_restated/SHOOTOUT_FULL_RESTATED_r8.md`, Mentat `body_skill` |
| the predictions written before the runs, and the misses | ledger C1-C28 (C26: the 2026-11-02 levers; C27: the MPC scoreboard; C28: the Eros fit) |
| ours against JPL's prediction at the time against reality | `fixtures/vintages/FAIR_CONTEST.md` (2026-10-12), contest v2 (running), Mentat `prediction_contest`, the portal's N-body page |
| daily, against this morning's observations | `D:/datasets/mpc_live/scoreboard.jsonl`, `misc/MPC_LIVE_SCOREBOARD.md` |
