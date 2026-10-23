# Why do Nereid, Helene, Callisto, Charon and Triton drift? (2026-10-23)

Method: `scripts/slow_moon_diagnosis.py` runs a SMALL system (Sun, the planet, the moon, and the neighbours that act on it) with the shoot-out's own integrator on the r5 fixture, and scores the moon's position relative to its planet against the fixture's (Horizons) at yearly epochs. One switch is changed per run; the switch that moves the error is the cause. The small system reproduces the full run (Nereid +1 yr 402 km small vs 412 full; Callisto +5 yr 229 vs 203; Charon +5 yr 96 vs 92). The Helene small system needed the whole Saturn system (Sun, Saturn and Dione alone gave 5,323 km at +1 yr against 872 in the full run). Raw outputs are in `diag/`.

Position error, km (window J2010 unless stated):

| moon | test | +1 yr | +2 yr | +3 yr |
|---|---|---|---|---|
| Callisto | table values (Jupiter J2 = 14736e-6) | 46.5 | 92.7 | 138.4 |
| Callisto | step 0.025 d instead of 0.0125 | 44.0 | 90.5 | 132.8 |
| Callisto | giants' zonals off | 16,721 | 33,289 | 49,925 |
| Callisto | Jupiter J2 = 14696e-6 | **1.0** | **2.1** | - |
| Callisto | Jupiter J2 = 14716e-6 | 23.8 | 47.4 | - |
| Callisto | measured zonals (`restated_zon`) | **1.6** | **3.2** | **4.3** |
| Helene | table values | 872 | 1,764 | - |
| Helene | Saturn J2 16290.71e-6 on 60330 km | 235 | 479 | - |
| Helene | measured zonals (J2 and J4 of Cassini) | **222** | 453 | 846 |
| Helene | measured zonals, Dione GM 73.5 / 72.7 (table 73.113) | 280 / 758 | - | - |
| Nereid | table values (step 0.0125 / 0.025 / 0.05 identical) | 402 | 845 | 1,333 |
| Nereid | giants' zonals off / tides off / radiation pressure off | 389 / 402 / 402 | 818 / 845 / 845 | - |
| Nereid | Triton removed | 37,998 | 79,568 | - |
| Nereid | Neptune GM less 1.4 / on the DE440 total | 398 / 401 | 835 / 841 | - |
| Nereid | starting velocity corrected by the 2.9 mm/s system offset | 478 | 1,004 | - |
| Nereid | best fit of the three initial velocity components (0.99 mm/s) | 382 | 802 | 1,266 |
| Triton | table values, J2010 / J2015 | 302 / 246 | 605 / 493 | - |
| Triton | giants' zonals off | 5,366 / 6,497 | 10,961 / 12,830 | - |
| Triton | Neptune GM less 1.4 / plus 1.3 | 358 / 248 | 716 / 497 | - |
| Charon | table values (step 0.0125 / 0.00625) | 19.1 / 19.1 | 38.3 / 38.3 | - |
| Charon | Charon GM 106.0 / 106.2 (table 106.1) | 1,459 / 1,420 | 2,920 / 2,843 | - |

What the data say:
* **Callisto: the cause is Jupiter's J2.** The code's 14736e-6 is an older value; Juno's measurement is 14696.514 +- 0.272 e-6 on R = 71492 km (Folkner et al. 2017, tabulated in arXiv:1708.06177 Table 1). With it Callisto is 1.6 km at +1 yr instead of 46.5. The earlier r2 run looked better only because its coarse step (0.05 d) cancelled this error.
* **Helene: Saturn's zonals explain three quarters** (872 to 222 km), with Cassini's J2 = 16290.573e-6 and J4 = -935.314e-6 (Iess et al. 2019, arXiv:2304.09215 Table IV.1; the reference radius is not printed on the page read, 60330 km is the one the J2 R^2 requires). The remaining 222 km a year is not explained: it is steeply sensitive to Dione's mass (about 1,300 km per km^3/s^2 per year; the drift would vanish at a GM of about 73.28, 0.23 % above the table's 73.113). That is a statement of sensitivity, not a correction: no independent source for a different Dione mass has been found, so none is used.
* **Charon: Pluto's 11 km is Charon's orbit.** Pluto's planet-centre error is the Charon relative-orbit error times Charon's mass fraction (0.12). Charon drifts 19.1 km a year at either step; the drift is a pure mean-motion error of 4e-7 and it is V-shaped in Charon's GM with its zero at 106.1013 against the tabulated 106.1 (4 digits). The table's rounding is therefore enough to explain it; nothing was changed (a fitted mass would be fitting the answer key).
* **Nereid: not explained.** The error is entirely in the orbital plane (normal part under 1 km; radial and along-track both hundreds of km, opposite in sign), the signature of an energy error, but it does not move with zonals, tides, radiation pressure, step, Neptune's mass, or the velocity correction, and no initial-velocity change removes it (fit 402 to 382 km). Either a force in the Neptune system is missing from the model or the Horizons Nereid track is not a clean solution of these dynamics.
* **Triton shows the same size of drift (300 km a year), also not explained by Neptune's mass** (zero near GM + 7.5, 2e-6 off the table, too far for rounding).

Falsified here: "planetary J2 is not modelled" (it is, for all four giants), and "the 2.9 mm/s fixture velocity offset explains Nereid" (correcting it made Nereid slightly worse).

Next tests, none run: Neptune's pole direction (Triton and Nereid are inclined to the equator, so the J2 term depends on it), the missing small Neptune moons (Proteus, Larissa, Galatea, Despina), and an independent source for Dione's and Charon's GM.
