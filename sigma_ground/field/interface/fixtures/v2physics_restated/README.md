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
