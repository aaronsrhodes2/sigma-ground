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
