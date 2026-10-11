# Fast ephemeris: where is it, instantly (milestone 1)

`from sigma_ground.field.interface.fast_ephemeris import where`; `where("Mars", days_ahead=120)` (Earth's view) or `where("Mars", 120, view="body")`
(the body's view: direction and distance to Earth and the Sun, and their separation in its sky). About 1.4 ms a call, nothing
integrated or looked up at query time.

**Model.** Per body: one Keplerian ellipse whose elements drift in time, plus a few sinusoids at integer combinations of the
bodies' own mean motions. Jupiter to Neptune orbit the system barycentre and the model subtracts the Sun's own wobble
(`SunBary`). Fitted offline to JPL DE440s by `scripts/fit_fast_ephemeris.py` on data before 1 Jan 2020 (1850 for the outer planets,
1950 for the inner ones, 1975 for the Moon); the term count per body is chosen on the last 15 years *inside* the training span.

**Held-out error** (angular, from Earth, against DE440s; median arcsec, after the training end). `in-sample` is the fit span itself.

| body | in-sample | 0-10 yr | 10-30 yr | 30-80 yr | 30-80 yr, km |
|---|---|---|---|---|---|
| Mercury | 0.4 | 2.0 | 2.8 | 4.0 | 4,491 |
| Venus | 0.5 | 2.3 | 4.2 | 4.9 | 6,765 |
| Mars | 3.0 | 6.9 | 5.1 | 9.9 | 19,793 |
| Jupiter | 69.3 | 215.3 | 292.9 | 288.2 | 1,339,418 |
| Saturn | 98.4 | 867.2 | 319.9 | 712.2 | 4,972,601 |
| Uranus | 2.2 | 13.5 | 83.1 | 203.8 | 3,035,493 |
| Neptune | 4.8 | 13.3 | 57.4 | 597.7 | 13,016,064 |
| Moon | 109.6 | 169.8 | 226.6 | 336.5 | 670 |
| Sun | 0.4 | 1.9 | 2.6 | 3.9 | 4,177 |

Terms chosen: {'Mercury': 128, 'Venus': 32, 'EMB': 32, 'Mars': 16, 'Jupiter': 0, 'Saturn': 0, 'Uranus': 8, 'Neptune': 4, 'Moon': 20, 'SunBary': 4}.

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
