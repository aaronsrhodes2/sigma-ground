# SPARC a0 test (round 3, 2026-10-11)

Question: is the radial-acceleration-relation scale g† consistent with the cosmological coincidence a0 = cH0/2π?

Data: SPARC (Lelli et al. 2016/2017), `RAR.mrt`, 2693 points, from astroweb.case.edu/SPARC, kept at `D:/datasets/sparc/` (outside the repo, 0.47 MB, sizes and hash prefixes in `SHA256_prefix.txt` there).
Prediction: 1.042e-10 m/s^2 from Mentat `mond_a0_constant` (H0 = 67.4 km/s/Mpc inside Mentat).

Result (`sparc_a0_results.json`, our own fit of the McGaugh-Lelli-Schombert function with errors on both axes and a fitted intrinsic scatter):
- g† = 1.148e-10 +/- 0.021e-10 m/s^2 (statistical, bootstrap over points); published 1.20e-10 +/- 0.02 (random) +/- 0.24 (systematic) [LIT]. Our value is 4% below the published one, a weighting difference, inside the published systematic.
- Prediction / fit = 0.907, i.e. 0.042 dex low, which is 0.32 of the 0.133 dex point-to-point scatter. Pre-registered criterion (FAIL if outside by more than the scatter): **passes**.
- Statistically (points taken as independent) the difference is 5 sigma, but the dominant uncertainty is systematic (stellar mass-to-light ratios, ~20%), and the H0 tension alone moves the prediction by ~8%. So: **consistent within systematics, not a precision test**.

What this does not show: it cannot separate MOND, emergent gravity, or a dark-matter halo explanation, because all three reproduce the RAR; a coincidence of scale is not a mechanism. The fit uses the tabulated RAR, not the 175 individual galaxy mass models; fitting those with free mass-to-light ratios is the natural next step.

## Per-galaxy fit (2026-10-11, `sparc_galaxy_fit.py`, `sparc_galaxy_results.json`)

149 galaxies (SPARC quality 1-2, inclination >= 30 deg), rotation curves fitted directly with the same RAR function, bootstrap over galaxies.

| analysis | g_dagger (1e-10 m/s^2) | prediction / fit |
|---|---|---|
| tabulated RAR, 2693 points (above) | 1.148 +/- 0.021 | 0.907 |
| 149 galaxies, mass-to-light fixed (0.5, 0.7) | 1.046 +/- 0.077 | 0.996 |
| 149 galaxies, mass-to-light free (0.1 dex priors) | 0.836 +/- 0.019 | 1.246 |
| published (Lelli et al. 2017) | 1.20 +/- 0.02 +/- 0.24 | 0.87 |

**Reading:** the fitted scale moves from 0.84 to 1.20 (about +/- 18 %) with the analysis choice alone (data table vs per-galaxy curves, fixed vs free
mass-to-light ratio). The prediction cH0/2pi = 1.042 lies inside that spread, so the data cannot discriminate it: it neither confirms nor refutes the
coincidence. Freeing the mass-to-light ratios gives median disc ratio 0.57 (16-84 %: 0.36-1.06) and lowers g_dagger by 20 %. Statistical "sigmas" in either
direction (0.05 sigma fixed, 10.7 sigma free) are not meaningful against an analysis systematic of this size.
Not modelled: distance and inclination uncertainties. To say anything sharper the mass-to-light ratios need independent constraints (stellar population models), not more rotation curves.
