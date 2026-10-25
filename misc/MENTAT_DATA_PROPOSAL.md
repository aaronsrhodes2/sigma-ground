# Putting the n-body data in Mentat: findings and proposal (2026-10-25)

Investigated read-only; nothing in Mentat has been changed except the DE440 field constants already committed in `1ccadba`.

## What is wrong today (checked, not assumed)
* `lookup_constant("GM_earth")`, `("earth_mass")`, `("GM_sun")` all return "not found". Only `solar_system_body("earth")` gives a mass, as a rounded Fact Sheet number with no uncertainty.
* Mentat knows 11 bodies (Sun, 8 planets, Pluto, Moon). Our data holds 56.
* The Moon's mass in Mentat is 7.342e22 kg; DE440's GM divided by G is 7.34579e22 kg (Mentat is 0.05 % low: an older value).
* "Exact" has to be said carefully. DE440 gives GM to about nine digits (Earth 398600.435507 km^3/s^2). A mass in kg needs G, which is known to 2.2e-5, so Earth is 5.9721684e24 kg +- 1.3e20 kg (G's uncertainty). The right answer to "the mass of Earth" states the GM first and the kg with the G uncertainty, and says which is which.

## What we hold (all local, all from JPL or the cited papers)
| asset | content | where | quality |
|---|---|---|---|
| DE440 header constants (new) | 645 constants; GM of Sun, planets, Earth, Moon, systems; Earth/Moon/Sun fields; tide lags | `fixtures/de440_header_constants.json` | exact JPL values |
| body table of the n-body fixture | 56 bodies, each with GM and the ephemeris it came from (DE440 4, GRAIL 1, MAR097 3, JUP365 5, SAT441 10, URA111 6, NEP095 2, PLU058 2, Horizons header 12, massless 11) | `de440_state_vectors_v2physics_r8.json` | published GMs; 11 moons have NO published mass |
| asteroid and Kuiper-belt catalogue | 343 asteroids with GM and orbital elements, 30 Kuiper-belt objects (7 identified), the 36-point ring (GM 8,917) | `de440_perturbers_v2physics.json` | masses from the DE440 header |
| asteroid physical data | 30 heaviest: GM, radius, rotation period, H, G, albedo, spectral type, orbit solution | `horizons_asteroids/` | JPL Horizons headers |
| moon data | 23 moons: radius, GM where given, albedo, mean semi-major axis, period, inclination, 27 yearly states | `horizons_moons/` | JPL Horizons |
| comets | 60 orbit solutions of 20 comets with non-gravitational parameters | `horizons_comets/` | JPL Horizons |
| planet-centre tracks | Neptune, Triton, Saturn, Titan, Uranus, Titania from the current Horizons vintage | `horizons_centres/` | JPL Horizons |
| measured prediction skill | median position error against DE440 by body and lead (1, 3, 5, 10, 20 yr), all 33 bodies, 16 to 21 windows; asteroid, comet and moon ladders | `v2physics_restated/*.md`, `.json` | measured, with the windows used |
| sealed forecast | 30-year monthly positions, hash-bound to the date it was made (2026-03-24) | `ssbm_future_predictions.json` | a true prediction; months now elapsed can be checked |
| fast ephemeris | closed-form fit to DE440 with an error table | `fixtures/fast_ephemeris/` | scored on dates it never saw |
| the kernel | DE440s (positions 1849 to 2150) | `D:/datasets/jpl/de440s.bsp`, 32 MB | JPL |

## Proposal, in priority order
1. **`body_constants(name)`** for any of the 56 bodies, the 343 asteroids and the 30 Kuiper-belt objects: GM (km^3/s^2), mass (kg, with the G uncertainty), mean radius and the field reference radius, J2..J4, the ephemeris the GM came from, and "mass unknown" for the 11 massless moons. Route `lookup_constant` aliases (`GM_earth`, `earth_mass`, `M_moon`, ...) to it. This fixes the example you gave, today.
2. **`body_position(body, when)`** (already started): known value up to today from DE440, prediction after, each labelled, with the measured error at that lead.
3. **`small_body(name)`**: orbital elements, physical parameters, and for comets the non-gravitational parameters, from the files above.
4. **`prediction_skill(body, lead)`**: the measured median error at that lead from the shoot-out tables, so Mentat can say how far to trust a prediction, and a check of the sealed forecast against the months that have since passed.
5. **`mass_budget()`**: the asteroid belt (343 sum 170.5 km^3/s^2; Pitjeva and Pitjev's whole-belt estimate about 160), the Kuiper belt (11,232 against about 7,850), planet systems. Answers "how heavy is the belt?" with the range and the sources.
6. **One source for Mentat's old table.** Generate the Fact Sheet table's masses from the DE data (so the Moon cannot drift again) and add a consistency test that fails if the two disagree beyond the stated uncertainty.

## Rules for what Mentat may say
* A GM from JPL is "exact" to its digits; a derived mass says it depends on G.
* A prediction is labelled a prediction and carries its measured error. Bodies we cannot score (the 14 close-in moons, Pan, Puck and the like) are returned as "not validated", not as numbers with a straight face.
* Values marked unresolved in `CONSTANTS_AUDIT.md` (the Sun's J2) carry that flag.

## Shipping the data to skippy-core
The live Mentat is built from a git ref (`SIGMA_GROUND_REF`; `tools/check_mentat_deploy.sh` flags drift) and the data lives under `fixtures/` in the repo, so items 1, 3, 4 and 5 need no extra files. Items 2 and the kernel need the 32 MB DE440s file; `tools/data_bundle.py` already packs and verifies a data set for another machine with sha256 per file, and is the model to follow. I have no way to see the skippy-core machine from here, so what is installed there is for you to confirm.
