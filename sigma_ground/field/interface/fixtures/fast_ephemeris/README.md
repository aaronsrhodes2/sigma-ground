# Fast ephemeris: where is it, instantly (milestone 1)

`from sigma_ground.field.interface.fast_ephemeris import where`; `where("Mars", days_ahead=120)` (Earth's view) or `where("Mars", 120, view="body")`
(the body's view: direction and distance to Earth and the Sun, and their separation in its sky). About 1.4 ms a call, nothing
integrated or looked up at query time.

**Model.** Per body: one Keplerian ellipse whose elements drift in time, plus a few sinusoids at integer combinations of the
bodies' own mean motions. Jupiter to Neptune orbit the system barycentre and the model subtracts the Sun's own wobble
(`SunBary`), which is itself a fixed combination of the four giants' orbits fitted on the true positions: the barycentre condition M_sun r_sun + sum m_i r_i = 0. The fitted coefficients come out as -9.5478e-4, -2.8588e-4, -4.366e-5, -5.152e-5, i.e. minus the textbook planet/Sun mass ratios (1/1047.3, 1/3498, 1/22900, 1/19410), which is the check that the structure is right. Fitted offline to JPL DE440s by `scripts/fit_fast_ephemeris.py` on data before 1 Jan 2020 (1850 for the outer planets,
1950 for the inner ones, 1975 for the Moon); the term count per body is chosen on the last 15 years *inside* the training span.

**Held-out error** (angular, from Earth, against DE440s; median arcsec, after the training end). `in-sample` is the fit span itself.

| body | in-sample | 0-10 yr | 10-30 yr | 30-80 yr | 30-80 yr, km |
|---|---|---|---|---|---|
| Mercury | 0.4 | 2.0 | 2.8 | 4.0 | 4,491 |
| Venus | 0.5 | 2.3 | 4.2 | 4.9 | 6,765 |
| Mars | 3.0 | 6.9 | 5.1 | 9.9 | 19,793 |
| Jupiter | 82.2 | 216.7 | 293.1 | 333.3 | 1,483,873 |
| Saturn | 115.3 | 934.5 | 230.3 | 726.1 | 5,109,786 |
| Uranus | 2.0 | 18.6 | 116.2 | 198.9 | 3,313,426 |
| Neptune | 0.4 | 8.7 | 82.4 | 585.8 | 12,772,977 |
| Moon | 109.6 | 169.8 | 226.6 | 336.5 | 670 |
| Sun | 0.4 | 1.9 | 2.6 | 3.9 | 4,177 |

Terms chosen: {'Mercury': 128, 'Venus': 32, 'EMB': 32, 'Mars': 16, 'Jupiter': 0, 'Saturn': 0, 'Uranus': 8, 'Neptune': 4, 'Moon': 20, 'SunBary': 0}.

## What this does and does not show
- Sun, Mercury, Venus and Mars: a few arcseconds out to 80 years, from a few dozen numbers. That is the headline so far.
- Moon: 3 to 6 arcminutes. A lunar theory needs thousands of terms for arcsecond accuracy; this form saturates near 400 km.
- Jupiter and Saturn: hundreds of arcseconds. Their dominant perturbation, the 880-year Great Inequality, cannot be learned from
  170 years of data by free sinusoids; it needs to be built in as a known near-resonant term. This is the first lever to pull.
- Uranus and Neptune: tens to hundreds of arcseconds; training is less than one orbit for Neptune, so its long horizon is unvalidated.
- The uncertainty reported by `where()` is these measured numbers, by distance past the training span; beyond 80 years it says so.
- DE440 is the answer key, not a real measurement. This is an agreement test with JPL's model, not yet a test against the sky.
- Limits: Jupiter to Neptune are system barycentres (as in the DE440 planetary segments), not planet centres. Geometric positions
  plus one-way light time; no stellar aberration, nutation or precession of the date frame. Pole and rotation of the body are not modelled.
