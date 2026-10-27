# Shoot-out restated (2026-10-14): step 0.025 d, seven asteroids, the Moon's C22 figure

Label `restated_c1_c3a_2026-10-14`. An ADDED set: the r2 baseline in `../v2physics/` is untouched and is the "before".

| file | what |
|---|---|
| `shootout_restated_runs.jsonl` | the 27 raw propagations (one per line): label `cand_long` (7 windows J2000-J2006, 20 yr), `cand_short` (14 windows J2007-J2020, 5 yr), `base_dt0025` (6 windows J2005/08/11/14/17/20, 5 yr, no asteroids, no C22); per body and lead year `eO` = heliocentric position error against DE440 in km |
| `SHOOTOUT_RESTATED.md` / `.json` | the tables: r2 vs control vs candidate, per body and lead, with paired window counts |

Run: `python scripts/fair_jpl_contest.py --out DIR --plan misc/shootout_restate_plan.json --workers 10 --dt-days 0.025 --asteroids sigma_ground/field/interface/fixtures/asteroid_perturber_states_v2physics.json --lunar-c22 2.2382e-5 --checkpoint`
(engine on master at `42c310d`: Forest-Ruth 4th order, full EIH, zonal harmonics, tides, solar pressure, sectoral C22 with the IAU rotation model). Report: `python scripts/shootout_restate_report.py DIR --out THIS_DIR`. 27 of 27 propagations, no failures or retries.

## What this is and is not (wording)

* Every window STARTS from DE440's own state. This tests whether our dynamics reproduce DE440's dynamics over the next 1 to 20 years. It is not a forecast contest and not a claim to beat JPL: DE440 is the answer key and the start state comes from it.
* The model is the REDUCED system (Sun, Mercury, Venus, Earth, Moon, Mars; Jupiter, Saturn, Uranus, Neptune, Pluto as system barycentres), validated against the full shoot-out (Venus, Earth, Mars agree with r2 to three digits, below). The giants and Pluto are therefore compared with DE440's barycentres and not with r2's planet centres.
* Seven asteroids with measured GM (Ceres, Vesta, Pallas, Hygiea, Interamnia, Psyche, Kalliope) are in, the other ~340 asteroids and the Kuiper belt are not.

## Result, median km over the six windows that have all three versions (J2005, 08, 11, 14, 17, 20)

| body | lead | r2 (0.05 d, full model) | control (0.025 d, no asteroids, no C22) | candidate (0.025 d, asteroids + C22) |
|---|---|---|---|---|
| Mercury | +5 yr | 1.41 | 0.023 | 0.033 |
| Venus | +5 yr | 0.189 | 0.205 | 0.115 |
| Earth | +5 yr | 0.451 | 0.453 | 0.170 |
| Moon | +5 yr | 3.70 | 1.45 | 0.48 |
| Mars | +5 yr | 1.03 | 1.03 | 0.57 |

Attribution (which change did what):
* **Step size** (r2 -> control): Mercury 1.41 -> 0.02 km and the Moon 3.70 -> 1.45 km. These two were numerical error at 0.05 d (the earlier convergence study), not physics. Venus, Earth and Mars are identical to r2 to three digits, which is also the check that the reduced model is faithful.
* **Asteroids + C22** (control -> candidate): Moon 1.45 -> 0.48, Earth 0.45 -> 0.17, Mars 1.03 -> 0.57, Venus 0.205 -> 0.115 km; Mercury unchanged (0.02 to 0.03 km, differences below the noise).
* Whole 16 windows, candidate: Mercury 0.00 / 0.02 / 0.03 km at +1 / +3 / +5 yr, Moon 0.10 / 0.30 / 0.43, Earth 0.04 / 0.12 / 0.19; at +20 yr (7 windows) Mercury 0.13, Venus 0.49, Earth 0.79, Moon 1.01, Mars 3.66 km (r2: 5.84, 0.59, 1.89, 14.67, 3.92).

## What is still wrong (remaining deficits, not explained)

* **Mars** does not improve at long range: +10 yr 1.18 km, +20 yr 3.66 km (r2 1.49 and 3.92); only 3 of 7 windows better at +10 yr. The missing ~340 asteroids are the leading suspect (untested).
* **Uranus, Neptune, Pluto** grow fast at long range: +20 yr 14.5, 29.5 and 34.9 km (barycentres). Missing Kuiper-belt mass, the Sun's reflex through unmodelled bodies, or the barycentre treatment are candidates, none tested.
* **Jupiter** 2.07 km at +5 yr and 9.92 km at +20 yr; Saturn 2.11 km at +20 yr.
* Venus +20 yr 0.49 km, Earth +20 yr 0.79 km: unexplained growth, linear in the lead.

# Full 33-body restatement at step 0.025 d (2026-10-14/16)
`rolling_shootout_full_short_0.025.json`, `rolling_shootout_full_long_0.025.json`, report `SHOOTOUT_FULL_RESTATED.md/.json` (`scripts/shootout_full_restate_report.py`). 23 candidate jobs + 6 control jobs (`de440_stack`, same step), all clean.
Candidate = headline stack + DE440's 343 asteroids + its 36-point Kuiper-belt ring + the Moon's C22 (DE440's own value 2.23808505e-5), as prescribed sources / engine layers; `restated_ast_c22` (no ring) on J2000 and J2003 gives the ring pair.
Read it with `MOON_STEP_CONVERGENCE.md` and `FINE_STEP_ALL_BODIES.md`: at 0.025 d the giants' planet-centre errors (Jupiter above all) and every fast moon are step-limited, so the converged numbers for those are in the 0.0125 d fine arm.
The planet-centre truth of Uranus, Neptune (and, less, Pluto and Jupiter) carries a fixture floor of hundreds of km, which hides the ring's 15-30 km in a direct error comparison; the ring's effect is read from the
prediction difference (ring minus no ring): Uranus 14.6, Neptune 30.2, Pluto 32.4, Saturn 5.3, Jupiter 1.1 km at +20 yr (median of J2000, J2003), the same size as the deficits the reduced-model barycentre test closed.

# Figures (2026-10-19): `figures/F*.png` with the data behind each in `figures/F*.json`
Made by `scripts/make_shootout_figures.py` from the stored runs (palette: the validated reference slots 1-3 and blue ramp; aqua is below 3:1 on the light surface, so every series is also direct-labelled or in a legend row and the numbers are in the JSON).
* F1 headline before/after dumbbell, every body, +5 yr, converged step (0.0125 d), same three windows (r2 vs restated on the r4 fixture)
* F2 inner-system error against lead time (r2, the same stack at the finer step only, restated), medians over windows
* F3 what each addition bought for the outer planets at +20 yr (7 asteroids, all 343, + ring, + 7 Kuiper-belt objects)
* F4 the fixture repair: giants' planet-centre error at +5 yr, old fixture against r4
* F5 moon step-convergence heatmap (23 moons x 4 steps)
* F6 comets: the ladder at +5 yr by group
* F7 the 30 heaviest asteroids scored as bodies (gravity only -> + relativity -> + the other asteroids), per asteroid
* F8 the three slow moons added in r5 (Himalia, Nereid, Helene), error against lead time, two windows
* F9 fast moons of Jupiter and Saturn in system-local models, table zonals against measured zonals
F2 now reads the full r4 run (below); regenerate with `python scripts/make_shootout_figures.py --full-short rolling_shootout_r4_full_short_0.025.json --full-long rolling_shootout_r4_full_long_0.025.json --candidate restated_ast_ring_kbo_c22`.

# Slow moons on the r5 fixture (2026-10-20)
Run `D:/datasets/sigma_runs/r5_moons` (2 windows, J2010 and J2015, 5 yr, step 0.0125 d, candidate stack; 52,000 s wall each), raw data `rolling_shootout_r5_moons_0.0125.json`, table `MOON_SCORES_R5.md` (`scripts/moon_scores_r5.py`), figure F8.
Median of the two windows at +5 yr: Himalia 12.6 km (1.0 and 24.2), Nereid 5,496 km, Helene 5,353 km; for scale the same run gives the Moon 0.3, Pluto 10.4, Jupiter 0.6, Titan 222, Io 1,047, Triton 1,376 km.
* Himalia is scored. Nereid and Helene grow almost linearly with lead time (Helene about 1,000 km per year in both windows), which looks like a period error; the cause is NOT tested. Candidates, none run at the time of writing: Helene's co-orbital partner Dione's mass, Nereid's eccentricity (0.75) against the step. (Correction 2026-10-23: an earlier version of this line named planetary J2 as a candidate and said the restated predictors do not set it; that was wrong, the giants' J2, J3 and J4 are in `_BODY_PARAMS` and `j2_zonal` is on in the headline stack, so oblateness of the giants is already modelled. Diagnosis in progress: `scripts/slow_moon_diagnosis.py`.)
* The other 20 of the 23 moons fetched from Horizons have periods under about 4 days and are not scoreable at 0.0125 d (planet-centred sub-integrator needed; separate work, not started).

# Full 33-body shoot-out on the r4 fixture (finished 2026-10-21)
29 jobs, 0 failed, 134,380 s wall, step 0.025 d. Candidate = DE440's 343 asteroids + 36-point ring + seven identified KBOs + the Moon's C22, on the r4 fixture. Raw runs `rolling_shootout_r4_full_short_0.025.json` / `_long_0.025.json`, table `SHOOTOUT_FULL_RESTATED_r4.md` (the earlier `SHOOTOUT_FULL_RESTATED.md` is unchanged). Median km, candidate against the r2 baseline:
| body | lead | r2 | candidate |
|---|---|---|---|
| Moon | +5 yr | 3.90 | 0.40 |
| Moon | +20 yr | 14.67 | 0.95 |
| Earth | +5 yr | 0.45 | 0.06 |
| Uranus | +5 yr | 778 | 0.21 |
| Uranus | +20 yr | 3,924 | 1.17 |
| Neptune | +5 yr | 441 | 0.21 |
| Neptune | +20 yr | 1,673 | 1.33 |
| Pluto | +5 yr | 43.4 | 11.4 |
| Pluto | +20 yr | 279 | 44.5 |
| Jupiter | +20 yr | 67.6 | 20.6 |
* Giants' planet-centre errors fall by 100-3,000x (Uranus +20 yr 3,924 -> 1.17 km), the C8 repair seen over 21 windows. Pluto (11.4 km at +5 yr) and Jupiter (5.3 km at +5 yr, step-limited at 0.025 d; 2.2 km at 0.0125 d) remain the largest planet-centre residuals.
* Moons not in the integrator-limited table improve against r2 in nearly every case (Triton +20 yr 18,918 -> 2,319 km, Charon 957 -> 368, Ariel 377,214 -> 42,401); not better than r2: Callisto (0 of 16 windows: 86.5 vs 203 km at +5 yr), Oberon at +5 yr (0 of 16), Titania, Iapetus, Phoebe (about even).
* Ring and KBOs (windows J2000, J2003, vs the no-ring/no-KBO stack on r4): Neptune +20 yr 30.1 -> 1.19, Uranus 16.3 -> 1.03, Saturn 6.7 -> 1.31 km; Pluto +20 yr 51.5 -> 44.7 km.

# Measured zonals, the missing-moon offset, and the fast moons (2026-10-24)
Diagnosis and tables: `MOON_DIAGNOSIS.md`, `diag/`. Predictor `restated_zon` (Jupiter J2 from Juno, Saturn J2/J4 from Cassini; a new label, the stored ones are untouched).
* Paired full-system run (r4 fixture, 0.025 d, 6 windows J2008/10/12/14/15/17, `rolling_shootout_zon_short_0.025.json`, table `ZONAL_PAIRED.md`): +5 yr Callisto 203 -> 36.9 km, Hyperion 220 -> 54.8, Titan 229 -> 60.2, Rhea 3,082 -> 1,691, Iapetus 43.3 -> 32.7, better in 6 of 6 windows; Dione, Ganymede, Mimas, Tethys, Enceladus 5-11 % better; everything else (inner planets, Moon, Uranus, Neptune, Pluto, Triton, Charon, the Uranian moons) unchanged within 0.1 %. Jupiter (4.89 -> 5.22), Io and Europa (+8 %) came out slightly worse at this step, which cannot integrate them; the converged check below settles it.
* System-local models (`scripts/system_local_moons.py`: Sun + planet + all its moons, step = shortest period / 300; r7 fixture; windows J2010, J2015; raw runs in `system_local/`, controls in `system_local_control/`, table `SYSTEM_LOCAL_MOONS.md`). +5 yr, km, measured zonals against the table's: Io 159 vs 8,588, Europa 100 vs 5,179, Ganymede 23.1 vs 644, Callisto 39.6 vs 193, Amalthea 3,021 vs 73,887, Thebe 4,107 vs 47,104; Mimas 1,963 vs 22,667, Enceladus 358 vs 9,075, Tethys 672 vs 6,826, Dione 167 vs 3,657, Helene 182 vs 3,682, Rhea 93.6 vs 1,487, Titan 52.9 vs 221, Hyperion 43.1 vs 203 (better in 2 of 2 windows); Himalia, Iapetus, Phoebe unchanged. Other systems with the measured zonals (no control run): Miranda 165, Ariel 106, Umbriel 596, Titania 3,431, Oberon 1,329, Triton 145, Nereid 6,258, Charon 91.6, Nix 127, Kerberos 72.3, Hydra 76.4, Styx 109 km. Figure F9.
* The 14 small close-in moons (Saturn: Pan, Atlas, Prometheus, Pandora, Janus, Epimetheus; Uranus: Juliet, Portia, Belinda, Puck; Neptune: Despina, Galatea, Larissa, Proteus) lose their orbital phase within a year (errors 28,000-270,000 km, larger than their orbits) with either set of zonals. Proteus is unchanged by +-2 % in Neptune's J2 or by switching the zonals off. Not explained; several have no published mass in the data held (Horizons gives none for Janus, Epimetheus, Atlas, Prometheus, Pandora, Pan, Puck, Portia, Juliet, Belinda, Nereid).
* The fixture's planet-centre velocity "defect" at Neptune (2.9 mm/s) was mostly missing mass: with Proteus, Larissa, Galatea and Despina (GM 3.14 km^3/s^2) in the barycentre it is 0.52 mm/s (Uranus keeps 5 mm/s: its small moons have no masses). Fixture r7 (`scripts/build_fixture_r7.py`): original + 23 Horizons moons in their own frame, each system shifted rigidly over ALL its members; Neptune's shift is 0.22 mm/s. With the small moons in, Triton in the Sun+Neptune system goes from 302/246 km to 39/11 km at +1 yr. The full shoot-out above was run on r4, not r7.

# Full 56-body shoot-out on the r8 fixture (finished 2026-10-26)
21 jobs, 0 failed, 160,229 s wall (44.5 h), step 0.025 d, fixture r8 (original 33 bodies + the 23 Horizons moons, Neptune and Triton from the current Horizons vintage, each planet system shifted rigidly onto DE440's barycentre), predictor `restated_zon`
(DE440's 343 asteroids, the ring, the 7 identified Kuiper-belt objects, the Moon's C22, measured Jupiter and Saturn zonals). 14 five-year windows J2007-J2020 and 7 twenty-year windows J2000-J2006; the five-year numbers use all 21 (the long jobs carry +1 to +5 too).
Raw runs `rolling_shootout_r8_full_short_0.025.json` / `_long_0.025.json`; table `SHOOTOUT_FULL_RESTATED_r8.md` (all 56 bodies; the 22 moons whose period is under about 2 days are in the step-limited table, not the headline); paired comparison with the earlier candidate on the same windows `PAIRED_R4_TO_R8.md`. Figures F1-F9 regenerated.

Median heliocentric position error, km, earlier candidate (r4 fixture) -> r8 run, same windows; windows better in brackets:
| body | +5 yr | +20 yr |
|---|---|---|
| Mercury, Venus, Earth, Mars, Uranus | unchanged within 1 % | unchanged within 2 % |
| Moon | 0.40 -> 0.40 (13/21) | 0.95 -> 1.00 (2/7) |
| Saturn | 0.17 -> 0.16 | 1.54 -> 1.58 |
| Neptune | 0.20 -> 0.13 (14/21) | 1.33 -> 1.02 (7/7) |
| Triton (new truth track) | 930 -> 261 (17/21) | 2,319 -> 1,054 (6/7) |
| Callisto | 210 -> 31.9 (21/21) | 976 -> 106 (7/7) |
| Titan | 227 -> 59.8 (21/21) | 930 -> 242 (7/7) |
| Ganymede | 7,019 -> 6,355 (21/21) | 25,651 -> 22,818 (7/7) |
| Jupiter | 5.54 -> 5.79 (0/21) | 20.55 -> 22.18 (0/7) |
| Io, Europa | +7 to +8 % (0/21) | +7 to +9 % (0/7) |

* Adding 23 moons did not disturb the planets: the inner system and the Moon are unchanged to within the noise of the windows (the Moon at +20 yr is 5 % higher, better in 2 of 7 windows; not explained, not distinguishable from window noise with 7 windows).
* New moons scored in the full system, median km at +5 / +20 yr: Himalia 7.4 / 20.6, Nereid 26.6 / 126.5, Nix 84 / 324, Hydra 90 / 382, Kerberos 75 / 274, Styx 81 / 215. The fast inner moons (Amalthea, Thebe, Janus, Epimetheus, Prometheus, Pandora, Atlas, Pan, Puck, Portia, Juliet, Belinda, Proteus,
  Larissa, Galatea, Despina) are step-limited at 0.025 d (3e4 to 2.4e5 km) and are scored in the system-local models instead (`SYSTEM_LOCAL_MOONS.md`); Helene follows Dione's global-step error (29,069 km at +5 yr; 182 km in the system-local model).
* NOT resolved: Jupiter's planet-centre error is 4 to 8 % worse in every window with the measured zonals, as are Io and Europa. These are the step-limited bodies, and the converged check (system-local, Io 8,588 -> 159 km) says the zonals help them; but Jupiter's own error at a converged step with the
  measured zonals has not been measured. The test that would settle it is the 0.0125 d arm on three windows (about 3 days).
