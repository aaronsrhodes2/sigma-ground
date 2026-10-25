# Constants: ours against DE440 (and why they differ)

Ours = the predictor table `rolling_shootout._BODY_PARAMS` (and, for radius, Mentat's `solar_system_body` before 2026-10-24, which reads the Fact Sheet's mean radii). Theirs = `fixtures/de440_header_constants.json` (DE440 header; Jupiter, Saturn from the Juno and Cassini papers). Both Mentat and the predictor now read that file through `de440_constants.py`, so they cannot disagree.

| body | quantity | ours | DE440 / paper | difference | class | why |
|---|---|---|---|---|---|---|
| Sun | radius_km | 695700 | 696000 | -0.0431 % | convention | ours is the IAU 2015 nominal solar radius (695,700 km, the Mentat table has the same); DE440's J2 is defined on its own reference radius ASUN = 696,000 km |
| Sun | j2 | 2.1106e-07 | 2.19613915e-07 | -3.89 % | unresolved | the code comment calls 2.1106e-7 'the DE440 fitted value (Park et al. 2021 Table 3)'; the header that JPL's integration read has 2.19614e-7. Table 3 was not re-read, so the origin of 2.1106e-7 is not established; the header is what was integrated |
| Sun | j4 | -9e-07 | 0 | n/a (DE440 = 0) | vintage | ours (-9e-7) is not in DE440, whose header sets J3 = J4 = 0 for the Sun; no source for -9e-7 was found in the code |
| Earth | radius_km | 6371 | 6378.1366 | -0.112 % | convention | 6371.0 is the Fact Sheet's volumetric MEAN radius; Earth's J2, J3, J4 are defined on the EQUATORIAL radius 6378.1366 km. Using the mean radius in the force makes J2 R^2 0.22 % low |
| Earth | j2 | 0.0010826 | 0.00108262539 | -0.00235 % | rounding | 1.0826e-3 is 1.08262539e-3 cut to five digits |
| Earth | j3 | -2.5e-06 | -2.53241e-06 | -1.28 % | rounding | -2.5e-6 is -2.53241e-6 cut to two digits |
| Earth | j4 | -1.6e-06 | -1.619898e-06 | -1.23 % | rounding | -1.6e-6 is -1.619898e-6 cut to two digits |
| Moon | radius_km | 1737.4 | 1738 | -0.0345 % | convention | 1737.4 is the Fact Sheet's mean radius; DE440's lunar harmonics (J2M ... C66M, C22M) are defined on AM = 1738.0 km |
| Moon | j2 | 0.0002034 | 0.00020321436 | +0.0914 % | vintage | 2.034e-4 against the header's 2.03214e-4 (0.09 %); the same size as the difference between lunar gravity solutions of different eras, source of ours not found in the code |
| Moon | j3 | 8.5e-06 | 8.45964582e-06 | +0.477 % | rounding | 8.5e-6 is 8.45965e-6 cut to two digits |
| Moon | j4 | -1.2e-05 | -9.70439002e-06 | +23.7 % | vintage | -1.2e-5 against the header's -9.704e-6 (24 %): an older lunar gravity field; DE440 uses the GRAIL-era value. Source of ours not found in the code |
| Jupiter | j2 | 0.014736 | 0.014696514 | +0.269 % | vintage | 1.4736e-2 is the pre-Juno value; Juno measured 14696.514e-6 +- 0.272e-6 (Folkner et al. 2017, arXiv:1708.06177 Table 1). Already replaced in restated_zon |
| Saturn | radius_km | 60268 | 60330 | -0.103 % | convention | 60268 km is the Fact Sheet's equatorial (1 bar) radius; the Cassini field is normalised on 60330 km (arithmetic from a second source, not printed on the page read) |
| Saturn | j2 | 0.016298 | 0.016290573 | +0.0456 % | vintage | 1.6298e-2 is the pre-Cassini-Grand-Finale value; Cassini gives 16290.573e-6 (Iess et al. 2019, arXiv:2304.09215 Table IV.1). Already replaced in restated_zon |
| Saturn | j4 | -0.000915 | -0.000935314 | -2.17 % | vintage | -9.15e-4 on 60268 km against Cassini's -935.314e-6 on 60330 km (-939.5e-6 on 60268 km): 2.6 % after converting the radius |
| Sun | gm_km3_s2 | 1.3271244e+11 | 1.3271244e+11 | -3.11e-08 % | rounding | field.constants uses the IAU 2015 nominal GM (1.3271244e20 m^3/s^2, exact by definition of the nominal); DE440's fitted value has more digits. Irrelevant to the shoot-out, which takes GM from the fixture |

## Not compared

* Mars (R 3389.5 km mean, J2 1.9606e-3) has no DE440-header or cited-paper value in the local data; the radius pattern is the same as Earth's and the Moon's (mean radius with an equatorial coefficient) but that is NOT verified here
* Uranus (R 25559, J2 3.5107e-3) and Neptune (R 24764, J2 3.539e-3): J2 R^2 agrees with the satellite-ephemeris papers' values to 1e-3 (checked by arithmetic against remembered values: Uranus 3341.29e-6 on 26200 km, Neptune 3408.43e-6 on 25225 km; not re-read)

## Effect

None of these changes any stored result. The predictor label `restated_hdr` applies the Sun, Earth and Moon header values (plus the Jupiter and Saturn ones of `restated_zon`); its effect on the answer key is a separate run.
