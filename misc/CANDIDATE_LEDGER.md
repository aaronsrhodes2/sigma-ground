# Candidate ledger: what we add to the model, the test, and what the data said

Rule (the Captain, 2026-10-11/12): a candidate cause is a falsifiable hypothesis. It is added to the model only if it shrinks the dynamics gap to JPL's own continuation on cases
it was not tuned on and does not make other bodies worse. Failures go to `DISPROOF_LEDGER.md`. The test bench is the fair contest (`scripts/fair_jpl_contest.py`, converged step 0.025 d).

## C1. The seven heaviest asteroids with measured masses (2026-10-12): SUPPORTED on the first test
Ceres, Vesta, Pallas, Hygiea, Interamnia, Psyche, Kalliope (JPL SBDB elements and GM, `asteroid_perturbers.py`), added as massive bodies. 8 matched cases (DE405/DE410/DE421 x start J2006/J2010/J2013, 5 yr each).
Raw runs: `fixtures/vintages/candidate_baseline_dt0025.jsonl` (without) and `candidate_asteroids7_dt0025.jsonl` (with).
Median gap, km, ours vs JPL's own continuation (runs improved of 8):

| body | +1 yr | +3 yr | +5 yr |
|---|---|---|---|
| Mars | 0.76 -> 0.25 (6) | 0.69 -> 0.53 (6) | 2.53 -> 0.87 (6) |
| Jupiter | 0.18 -> 0.03 (8) | 1.32 -> 0.26 (8) | 4.04 -> 0.87 (7) |
| Saturn | 0.24 -> 0.02 (8) | 1.33 -> 0.17 (8) | 1.85 -> 0.35 (7) |
| Uranus | 0.22 -> 0.02 (8) | 1.30 -> 0.15 (8) | 1.56 -> 0.37 (7) |
| Neptune | 0.22 -> 0.02 (8) | 1.32 -> 0.15 (8) | 1.62 -> 0.35 (7) |
| Pluto | 0.23 -> 0.02 (8) | 1.34 -> 0.16 (8) | 1.66 -> 0.36 (7) |
| Mercury | 0.075 -> 0.073 (6) | 0.24 -> 0.23 (5) | 0.46 -> 0.44 (5) |
| Venus | 0.13 -> 0.12 (6) | 0.37 -> 0.35 (6) | 0.61 -> 0.57 (6) |
| Earth | 0.14 -> 0.14 (4) | 0.42 -> 0.43 (4) | 0.67 -> 0.71 (4) |
| Moon | 0.40 -> 0.44 (4) | 0.96 -> 0.96 (2) | 1.82 -> 1.83 (3) |

Outcome: Mars' largest deficit closes by two thirds; every outer planet's gap falls by 75-80 %; nothing is made worse beyond noise (Earth +0.04 km, Moon +0.00). After this candidate every body is within
0.9 km of JPL's own continuation at +5 yr except the Moon (1.8 km) and Earth (0.7 km), which the asteroids do not touch.
Not yet isolated: why all the giants improve together (a plausible mechanism is the asteroids moving the Sun about the barycentre, which shifts every heliocentric position; unproven), and which of the seven matter (Ceres, Vesta, Pallas are the likely core).
Limits: 8 cases, three vintages, three start years; the other ~340 asteroids of the DE440 model and Kuiper-belt objects are still missing; the asteroid states are test-particle integrations from 2026 elements.
Next check: a wider set of starts and all vintages at the converged step (about 8 hours for 24 more cases).

### C1 update (2026-10-13): confirmed on 24 further cases across all 7 vintages
24 propagations (DE403 to DE432, start years 2000 to 2015), converged step, seven asteroids on, against the matching dt = 0.05 no-asteroid baseline (Mars, Venus, Earth and Jupiter to Pluto are step-converged at 0.05; Mercury and the Moon are excluded). 0 failures.
Median gap to the vintage's own continuation at +5 yr, baseline -> with asteroids (runs improved of 23 that reach +5 yr):
Mars 2.43 -> 0.84 km (22), Venus 0.51 -> 0.42 (21), Earth 0.73 -> 0.46 (21), Jupiter 3.61 -> 1.21 (16), Saturn 2.00 -> 0.41 (21), Uranus 1.51 -> 0.22 (23), Neptune 1.48 -> 0.23 (23), Pluto 1.51 -> 0.24 (23).
Mars improves in every vintage (DE405 4.37 -> 2.03, DE421 0.68 -> 0.42, DE432 3.03 -> 0.63). C1 is supported; the mechanism for the giants is still not isolated.

## Open candidates (not yet tested)
- C2 lunar figure and Earth-Moon tides (Moon 1.8 km, Earth 0.7 km): lunar J2 and sectoral C22, Earth tidal phase lag; needs lunar gravity coefficients (published, to be checked against the kernel).
- C3 solar oblateness and mass loss, higher-order relativity (Mercury 0.44 km, Venus 0.57 km).
- C4 the remaining asteroids and a Kuiper-belt ring (all bodies, small).

## C2a. Time-varying spin axes (IAU precession, nutation and lunar libration) for the zonal bodies (2026-10-13): DISPROVED as the cause of the Moon's gap
Hypothesis: the engine holds each body's pole fixed at the start epoch (the code says so), the Earth's pole moves ~100 arcsec and the Moon's up to ~2.3 deg in 5 yr, so refreshing the poles from the IAU
models (Earth, Moon, Mercury, Venus, Mars, Sun; every 0.5 d) should shrink the Moon's 1.8 km gap. Option `--pole-update-days` in `scripts/fair_jpl_contest.py`.
Test (DE421, start J2010, +1 and +3 yr, coarse step so only the DIFFERENCE between runs matters): change in position error with the refresh = 0.000 km for Mercury, Venus, Earth and Mars,
+0.000 km (+1 yr) and +0.005 km (+3 yr) for the Moon, against a 1.8 km gap.
Why (checked by hand): a pole shift tilts the plane about which the orbit precesses; for the Earth, 1.5 J2 (R/a)^2 n x (100 arcsec) x 5 yr ~ 0.03 km, and the Moon's own J2 with a 2 deg offset gives the same order.
My first estimate (~10 km, treating the offset as a net acceleration) was wrong for that reason, and the measurement caught it. Not adopted; the option stays in the runner.
Remaining ideas for the Moon, in the order they should be tried: (1) confirm the numerical floor with a third step size (0.0125 d; running), because the 4th-order assumption rested on two points; (2) Earth-Moon GM and other constants
that differ between each vintage and DE440; (3) lunar sectoral and tesseral harmonics and the figure-figure coupling; (4) tidal dissipation (estimated 0.06 km at +5 yr: too small alone).

## C3. Shape of the Moon's deficit (residual atlas, 2026-10-13): measured; cause open
Four cases at the converged step (0.025 d), the Moon's gap (ours minus the vintage's own continuation) sampled every 5 days for 5 yr and split about the Earth. Data: `fixtures/vintages/moon_atlas_runs.json`, report `MOON_ATLAS.md`.
- **Along-track: a straight line.** Drift -0.06 (DE405 from J2010), -0.29 (DE410 J2010), -0.56 (DE421 J2010), -0.40 km/yr (DE421 J2013); the quadratic term is 0.001-0.002 km/yr^2, i.e. none. Our Moon lags a vintage's by a CONSTANT rate
  (about 1e-8 of the mean motion for 0.4 km/yr), which also rules out tidal dissipation (that is quadratic; its estimated size is 0.06 km at +5 yr anyway).
- **A 27.6-day oscillation** (the anomalistic month) in the radial and along-track components, 60-130 m, explaining 70-99 % of what remains: an eccentricity/apsidal mismatch of small size.
- Cross-track: about 30 m at the draconic month; the inclination and node are fine. Nothing significant at the variation (14.8 d) or evection (31.8 d).
- **The drift rate differs by vintage at the same start epoch** (J2010: DE405 -0.06, DE410 -0.29, DE421 -0.56), so part of it depends on each vintage's constants or initial osculating state, not on our dynamics alone.
Open hypotheses, each testable: (a) constants (Earth+Moon GM, mass ratio) differ between each vintage and DE440 (a GM offset of about 2e-8 gives this drift); (b) a lunar-model term we lack; the DE440-start control (identical constants and state) separates them: if the drift persists
with DE440 itself it is our model, if it vanishes it is the vintages' constants. Control running (3 cases).

### C3 update (2026-10-13): the DE440-start control: the deficit is OUR model
Starting from DE440 itself (identical constants and initial state; data in `fixtures/vintages/moon_atlas_control_de440start.json`, 3 cases, converged step): the Moon still drifts along-track by -0.29 to -0.44 km/yr (linear, quadratic term <= 0.003 km/yr^2)
and the 27.55-day term grows from 14-22 m in year 1 to 156-201 m in year 5, the same pattern and size as against the vintages. So the vintage constants are NOT the cause; a lunar-model term is missing from the engine.
The growth of the once-per-orbit term is linear in time: a mismatch of ~0.4 arcsec/yr (~3 ppm) in the rate of the Moon's perigee advance, with a mean-motion offset of ~1e-8. The Sun-phase terms (synodic 29.53 d, variation 14.77 d) explain only 0-3 % of the variance: not a solar-geometry effect.

## C3a. The Moon's triaxial figure, sectoral C22 with the long axis locked to the Earth (2026-10-13): coarse test SUPPORTIVE, converged test running
`sigma_ground/field/interface/lunar_figure.py` (MacCullagh quadrupole force; 9 tests: engine J2 limit, spherical-harmonic derivative on both axes, Newton III, zero-C22 identity). C22 = 2.2382e-5 [LIT, GRAIL, reference radius 1738 km].
Coarse check (same step with and without, so only the difference counts; DE440 start J2010): the Moon moves forward by +0.291 km at +1 yr and +0.882 km at +3 yr, a steady +0.29 km/yr, against a deficit of -0.29 to -0.44 km/yr.
Next: the converged control with C22 (3 cases, 5-day sampling) and then vintage cases; success = the drift and the growing 27.55-day term shrink without harming other bodies.

### C3a result (2026-10-13): SUPPORTED. The Moon's gap at +5 yr falls from 2.15 to 0.44 km
The DE440-start control, 3 cases, converged step (0.025 d), Moon vs DE440's own continuation, medians; data files `moon_atlas_control_*.json` in `fixtures/vintages/`:

| variant | along-track drift (km/yr) | along-track acceleration (km/yr^2) | 27.55-d amplitude, yr 1/3/5 (m) | gap at +5 yr (km) |
|---|---|---|---|---|
| no C22 | -0.416 | +0.0019 | 22 / 108 / 194 | 2.15 |
| C22, long axis locked to the Earth | -0.121 | +0.0017 | 23 / 107 / 192 | 0.73 |
| C22, IAU rotation model, engine clock frozen inside a step | -0.102 | +0.0662 | 3 / 28 / 84 | 1.25 |
| **C22, IAU rotation model, sub-step clock (`_t_eval`)** | **-0.101** | **+0.0022** | **2 / 5 / 11** | **0.44** |

How it was found: the locked-axis version removed the mean-motion error (a bulge that never averages away) but not the growing once-per-orbit term, which pointed at the bulge orientation: the real bulge follows the uniform rotation while the Earth direction librates by 2e = 6.3 deg.
The IAU rotation model (prime meridian validated against the DE440 Earth direction: mean libration +0.017 deg, 27.55-d amplitude 6.30 deg) cut that term by 94 %, and exposed an ENGINE BUG: the engine advanced its clock only at the end of a Forest-Ruth step, so
time-dependent forces were evaluated up to a step late (0.17 deg lag, a constant tangential force, a spurious +0.066 km/yr^2). Fixed in `nbody.py` (`_t_eval`, 8 lines, bit-identical trajectory, test `test_nbody_substep_time.py`).
Other bodies unchanged (Mercury, Venus, Mars, Jupiter, Saturn < 0.001 km; Earth +0.011 km). Remaining: a steady along-track lag of about -0.10 km/yr (a mean-motion offset of ~3e-9; linear, so not tidal dissipation).
Candidates for the remainder, not yet tested: the C22 coefficient and the reference radius convention, the tesseral and degree-3 lunar harmonics, the Earth's tidal response, and the Moon's J2 value.

### C3a engine layer + combined run (2026-10-13): C22 is now `PhysicsToggles.sectoral_c22`; the 24-case test against the vintages is MIXED for the Moon
Engine: `sectoral_c22` toggle, per-body `c22` / `figure_ref_radius_m`, prescribed rotation models evaluated at the Forest-Ruth sub-step times (`figure_forces.py`, `nbody.py`; 7 layer tests). Equivalence with the earlier
subclass path on a coarse case: worst difference over every body and lead = 0.0 (bit-identical). Merged to master; branch and worktree removed.

Combined run: seven asteroids + C22 (IAU rotation), converged step 0.025 d, the same 24 cases as the C1 wide check (7 vintages, starts 2000 to 2015, 0 failures). Data: `fixtures/vintages/candidate_asteroids7_c22_wide24_dt0025.jsonl`.
Median gap to the vintage's own continuation, km (runs where the combined run beats asteroids-only, of n):

| body | +1 yr | +3 yr | +5 yr |
|---|---|---|---|
| Moon: asteroids -> + C22 | 0.38 -> 0.20 (13/24) | 1.07 -> 0.66 (13/24) | 1.68 -> 1.11 (15/23) |
| Earth: asteroids -> + C22 | 0.09 -> 0.09 | 0.28 -> 0.29 | 0.46 -> 0.45 |
| Mercury, Venus, Mars, Jupiter to Pluto | unchanged to the printed 0.01 km | | |

Against the no-asteroid baseline (dt 0.05) the combined model improves Venus, Earth, Mars and every outer planet as in C1 (outer planets 23 of 23 at +5 yr; Mars 22 of 23). Mercury and the Moon have no converged no-asteroid baseline for all 24, so their reference is asteroids-only.
The Moon, by vintage (gap at +5 yr, asteroids-only -> combined): DE418, DE421 and DE432 (the vintages closest to DE440's lunar model) improve in every case (e.g. DE432 j2015 1.37 -> 0.43, DE418 j2014 2.92 -> 1.22);
DE413 gets WORSE in every case (e.g. j2011 0.31 -> 1.20 km; its Moon was already within 0.05 to 0.12 km of our no-C22 run at +1 yr); DE403, DE405 and DE410 are mixed.
Reading (not yet tested): the C22 term closes the gap to DE440-like lunar physics (the DE440-start control: 2.15 -> 0.44 km), while each older vintage carries its own lunar figure and initial state, so a vintage whose Moon is already close to ours without C22 moves away from it.
The test that would separate "our model got better" from "the vintage differs": compare the same runs with DE440 as the reference (the DE440-start control already does, and shows the improvement). So C3a stays SUPPORTED against DE440 and is NOT a uniform improvement against the old vintages.
Remaining Moon items unchanged: the steady -0.10 km/yr along-track drift, C22 coefficient and reference-radius convention, tesseral and degree-3 harmonics.

### C1 + C3a in the shoot-out (2026-10-14): restated at step 0.025 d from DE440 starts, 27 propagations, 0 failures
Data and tables: `fixtures/v2physics_restated/` (README has the attribution and the wording). Medians over the six windows that have all three versions, +5 yr, km, r2 (0.05 d) -> control (0.025 d, no asteroids/C22) -> candidate:
Mercury 1.41 -> 0.02 -> 0.03; Venus 0.19 -> 0.21 -> 0.12; Earth 0.45 -> 0.45 -> 0.17; Moon 3.70 -> 1.45 -> 0.48; Mars 1.03 -> 1.03 -> 0.57. The step accounts for Mercury and half of the Moon; the asteroids + C22 for the rest of the Moon and for Earth, Mars and Venus.
Over all 16 windows the candidate beats r2 in every window for Mercury and the Moon (16 of 16 at +1, +3, +5 yr). At +20 yr (7 windows) Moon 14.7 -> 1.0 km, Earth 1.89 -> 0.79; Mars 3.92 -> 3.66 (barely moved).
Remaining, untested: Mars beyond +5 yr (the other asteroids?), Uranus/Neptune/Pluto at +20 yr (14, 30, 35 km), Jupiter 9.9 km at +20 yr.

## C5. DE440's Kuiper-belt ring (36 equal point masses at 44 au, total GM 8917 km3/s2) and its other 343 asteroids (2026-10-15): SUPPORTED, the long-range outer-planet deficit closes
Source of the model: Park et al. 2021 (AJ 161:105, sec. 2.2) and the DE440 header constants MA0001-MA1467 (343 asteroids), MA8201-MA8236 (ring); masses read off `header.440`. Prediction made BEFORE the run (2026-10-14): the missing ring is what lets Uranus, Neptune and Pluto grow to
14, 30 and 35 km by +20 yr (order-of-magnitude estimate ~34 km for Neptune from the ring's tidal pull). Test: reduced model, step 0.025 d, DE440 starts, two 20-yr windows (J2000, J2003), the Moon's C22 on in all arms; asteroids and ring as prescribed sources
(proved to replay live membership to 0.13 % of the asteroids' effect, `v2physics_restated/PRESCRIBED_EQUIVALENCE.md`). Data: `v2physics_restated/ring_reduced_runs.jsonl`. Median of the two windows, km against DE440:

| body | +20 yr, 7 asteroids (C1) | + all 343 | + 343 and the ring |
|---|---|---|---|
| Uranus | 14.5 | 16.1 | **1.8** |
| Neptune | 29.5 | 30.2 | **1.4** |
| Pluto | 34.2 | 34.9 | **4.1** |
| Saturn | 2.8 | 4.8 | **0.7** |
| Jupiter | 11.0 | 0.9 | **0.4** |
| Mars | 3.4 | 0.22 | **0.18** |
| Earth | 0.52 | 0.05 | **0.04** |
| Venus | 0.51 | 0.28 | **0.28** |
| Mercury | 0.20 | 0.16 | 0.17 |
| Moon | 1.0 | 0.94 | 0.94 |

* The ring removes 88 % (Uranus), 95 % (Neptune) and 88 % (Pluto) of the +20 yr error, as predicted, and without it the 343 asteroids leave them unchanged (16, 30, 35 km). Same pattern at +5 and +10 yr.
* The other 336 asteroids close Mars (3.4 -> 0.22 km at +20 yr), Earth (0.52 -> 0.05) and Jupiter (11 -> 0.9), which the seven heaviest alone did not.
* Wording: this reproduces DE440's own model (DE440's ring is a modelling device for the belt's mass, equally spaced points at one radius). It shows that DE440's ring mass and radius carry the long-range outer-planet dynamics; it does not show that the belt's mass is distributed as the ring says.
* Remaining at +20 yr: Pluto 4 km, Uranus/Neptune 1.2-2 km, Venus 0.28, Moon 0.94 (unchanged by any of this), Mercury 0.17. Candidates, untested: the 30 individual Kuiper-belt objects (Pluto), the Moon's degree-3 gravity terms and tides (Moon).
Two windows only; the full-system run (21 windows) will confirm or contradict.

### C1 + C3a + C5 in the full 33-body system at a converged step (2026-10-15): confirmed; the moons are untouched by them
Fine-step arm: the headline stack with and without DE440's asteroids, ring and the Moon's C22, step 0.0125 d, windows J2008/J2014/J2020, all 33 bodies, 6 of 6 jobs clean. Data `v2physics_restated/rolling_shootout_fine_0.0125.json`, table `FINE_STEP_ALL_BODIES.md`
(`scripts/shootout_fine_report.py`). Median km at +5 yr, r2 (0.05 d) / control (0.0125 d) / candidate:
Mercury 1.48 / 0.08 / 0.07; Venus 0.29 / 0.31 / **0.07**; Earth 0.46 / 0.44 / **0.05**; Moon 3.51 / 1.33 / **0.31**; Mars 0.93 / 0.93 / **0.10**; Saturn 1.51 / 2.14 / **0.19**.
* The layers improve Venus, Earth, Moon, Mars and Saturn in 3 of 3 windows. The Moon's +5 yr error in the full system is 0.31 km (11-fold below r2).
* The 24 moons change by less than 0.1 % under the layers, so none of their errors are caused by missing perturbers. Their remaining errors are step and model floors: Io 1,030, Europa 821, Ganymede 1,087, Mimas 108,725, Enceladus 20,458, Tethys 11,300, Dione 5,326, Miranda 8,136 km at +5 yr; Titan 217, Callisto 214, Oberon 1,120, Titania 963.
* Planet-centre floors that nothing here touches: Jupiter 3.6-4.1 km, Uranus 796 and Neptune 478 km (fixture planet-centre data, known since the DE440 fixture review), Pluto 30 km. The reduced-model system barycentres of the same bodies are 0.1-4 km at +20 yr, so these are not dynamics.

## C6. Comets as tracked test particles against JPL's own solutions (2026-10-16): four additions, all supported
19 periodic comets x 3 windows (J2005, J2010, J2015), 57 runs, our propagation (`comet_orbits.py`: Sun + nine planet systems from DE440s, adaptive Dormand-Prince at rtol 1e-13) from JPL's state at the window start against JPL's states +1..+5 yr of the
same solution. Data and tables: `fixtures/horizons_comets/` (`COMET_LADDER.md`, `ladder_full.json`). Median position error at +5 yr, km:

| rung | all 57 | far (>10 au, 13) | near, no delay (27) | near, with delay (17) |
|---|---|---|---|---|
| R0 gravity only | 1,641 | 0.13 | 1,967 | 17,148 |
| R2 + Marsden push (JPL's A1, A2, A3) | 5.7 | 0.17 | 5.0 | 10,638 |
| R3 + the push's time delay DT | 4.3 | 0.17 | 5.0 | **5.9** |
| R3 without the Sun's 1PN | 50.5 | 0.13 | 63.3 | 73.9 |
| **R4 = R3 + DE440's 343 asteroids** | **0.82** | 0.07 | **1.1** | **1.3** |

* **Non-gravitational push (R2):** brings the 27 comets without a delay from ~2,000 km to ~5 km (24 of 27 improved); 21P goes 221,227 -> 6 km, 17P 17,944 -> 10 km. It is the dominant physics for every comet inside ~10 au.
* **Time delay (R3):** required for the 17 runs that carry one: 10,638 -> 5.9 km, better in 17 of 17 (45P is worse than gravity alone, 446,778 km, without it).
* **Sun's 1PN term:** judged where it can be (once the push is in): removing it raises the median from 4.3 to 50.5 km, and every near comet is worse without it (2P 5 -> 308 km). Supported. On its own, before the push, it is invisible under the push-sized errors (R1 vs R0).
* **Asteroids (R4):** 54 of 57 runs improve, median 4.25 -> 0.82 km; far comets 0.13 -> 0.07 km. Largest residuals left at +5 yr: 9P/Tempel 1 (J2010) 62 km, 45P 10 km, 67P 5 km, 10P 4 km, 22P 3.8 km.
* Far comets (Halley 1P, 12P, 55P, 109P; 15-35 au): gravity alone matches JPL to 0.1-0.5 km: the planets, frames, units and integrator agree.
* Wording: the truth is JPL's solution as propagated by Horizons (DE441 planets, 16 asteroids, the same Marsden model). This is dynamics parity with JPL's own treatment of each comet, NOT a measure of how well JPL's solution predicts the real comet.
* Open: 9P Tempel 1 (an encounter-sensitive case?), 73P (data pending), the 16 asteroids Horizons treats individually vs our 343 prescribed ones, DE441 vs DE440 planets.

### C1 + C3a + C5 in the full 33-body system, 21 windows at step 0.025 d (2026-10-16): confirmed
Data and tables `fixtures/v2physics_restated/` (`SHOOTOUT_FULL_RESTATED.md`, README section "Full 33-body restatement"). 23 candidate and 6 control jobs, all clean (32 h wall). Median km, r2 (0.05 d) -> candidate (16 five-year / 7 twenty-year windows; every cell better than r2 in 16/16 or 7/7 unless noted):
* +5 yr: Mercury 1.35 -> 0.03; Venus 0.18 -> 0.07 (14/16); Earth 0.45 -> **0.06**; Moon 3.90 -> **0.40**; Mars 1.26 -> **0.07**.
* +20 yr: Mercury 5.84 -> 0.10; Venus 0.59 -> 0.28 (6/7); Earth 1.89 -> **0.16**; Moon 14.67 -> **0.95**; Mars 3.92 -> **0.19**; Saturn 12.3 -> 2.2.
* The ring, read from the prediction difference (ring minus no ring, J2000 and J2003): Uranus 14.6, Neptune 30.2, Pluto 32.4, Saturn 5.3, Jupiter 1.1 km at +20 yr: the size of the deficits the reduced-model test closed. In the direct error against the fixture truth the outer planets' planet-centre
  floors (Uranus 3,100 km, Neptune 1,650 km at +20 yr) hide it, so that comparison cannot show the ring; the barycentre comparison in C5 does.
* Honest limits: at 0.025 d Jupiter's planet-centre error (10.3 km at +5 yr) and every fast moon are step-limited (read `FINE_STEP_ALL_BODIES.md` for the converged values: Jupiter 4.1 km); the layers do not change any moon (< 0.1 % at the converged step). Callisto looks 2-2.5x worse than r2
  (86 -> 200 km at +5 yr) in all 16 windows: the converged value is 214 km at 0.0125 d, so r2's 86 km was integration error that happened to cancel, not a regression. Titania moves between steps (phase noise at 0.025 d; equal at 0.0125 d).
