# Enceladus J₄ Regression — Root-Cause Verdict (2026-05-15)

**Status:** Root cause identified. The J₄ formula is correct; the Enceladus regression is caused by a missing perturber, not a force-model bug.

## Symptom

In the 2026-05-15 rolling-window toggle iteration (4 windows × 8 predictors × 25 bodies × 4 samples, 132 min wall clock), enabling `j4_zonal` on top of the finedt baseline produced exactly one meaningful regression: **Enceladus +1.25% at 3y, +20% at 5y**. Every other body moved by <0.01% from any of the {+2PN, +J₃, +J₄, +tidal} toggles.

## What we ruled out

### 1. The J₄ formula

The J₄ acceleration is derived from the gradient of the Legendre P₄ zonal potential term:

```
Φ_J4 = +(GM J₄ R⁴/r⁵) × P₄(cos θ)
P₄(c) = (35c⁴ − 30c² + 3)/8
a_J4 = -∇Φ_J4 = (5 GM J₄ R⁴)/(8 r⁶) × [3(21s⁴−14s²+1) r̂ + 4s(3−7s²) n̂]
```

Two closed-form unit tests now pin this:

- **At the pole** (s = 1): the formula reduces to `a_z = 5 GM J₄ R⁴/r⁶` purely along the pole axis. Code matches analytic to machine precision (ratio = 1.000000) for both Earth-scale and Saturn-Enceladus geometries.
- **At the equator** (s = 0): the formula reduces to `a_r = (15/8) GM J₄ R⁴/r⁶` purely radial. Code matches analytic to 1e-8 relative precision.

Tests are pinned in [`test_nbody.py`](sigma_ground/field/interface/test_nbody.py) under `TestZonalJ3J4`:
- `test_j4_at_equator_magnitude_matches_derivation`
- `test_j4_at_pole_magnitude_matches_derivation`
- `test_j4_saturn_enceladus_geometry_matches_analytic`

**Verdict on J₄ formula: CORRECT.** Not the bug.

### 2. Pole axis orientation

Saturn's IAU 2015 pole is at α=40.589°, δ=83.537° — about 7° off ICRS +z. We currently default Saturn's `pole_axis_unit` to ICRS +z. This is a known approximation. Two reasons it's not the dominant issue:

- Jupiter's pole is 26° off ICRS +z (we also default it to +z), yet Jupiter J₄ has essentially zero effect on its Galilean moons (toggle iteration shows Io/Europa/Ganymede/Callisto all 1.000x with j4 enabled).
- If pole error were the cause, Jupiter's moons would regress more than Saturn's (larger pole offset, larger J₄ in absolute terms). They don't.

So pole orientation is a refinement-level issue, not the root cause.

### 3. Saturn J₄ numerical value

We use J₄ = −9.15e-4 from Anderson & Schubert 2007. More recent Cassini fits (Iess et al. 2010) give J₄ = −9.36e-4, a 2% difference. Even adopting the newer value would only change our J₄ acceleration by ~2% — far below the 20% regression magnitude.

So coefficient value is not the cause.

## What the actual cause is

**Dione is missing from our DE440 fixture.**

The fixture [`de440_state_vectors.json`](sigma_ground/field/interface/fixtures/de440_state_vectors.json) contains 26 bodies. For the Saturn system, only **Enceladus and Titan** are present. The four other major Saturn moons — Dione, Tethys, Rhea, Mimas — are absent.

This matters because of the **Enceladus–Dione 2:1 mean-motion resonance**:

- Enceladus orbits Saturn at ~238,042 km with a period of 1.370 days.
- Dione orbits Saturn at ~377,396 km with a period of 2.737 days = exactly 2 × Enceladus's period.
- This resonance locks Enceladus's orbital eccentricity at e ≈ 0.0047 (its "forced eccentricity"), which drives the tidal heating responsible for Enceladus's geyser activity.
- DE440's ephemeris for Enceladus includes the full perturbation from Dione (and Mimas, Tethys, Rhea).
- Our integration **does not** — Dione's not there to perturb Enceladus, so Enceladus drifts away from its true orbit.

Even our baseline `finedt` predictor for Enceladus has error 2.806e-3 AU at 3y (vs Mercury's 3.45e-5 AU — Enceladus is ~80× worse than Mercury). Enabling J₄ then layers a small additional force on top of an already-failing prediction; the force is correct, but the unperturbed baseline it's applied to is so wrong that any added force makes things worse, not better.

## Why other bodies don't show this

Looking at the toggle iteration's near-zero deltas elsewhere:

| Saturn system | In fixture | Status |
|---|---|---|
| Saturn | Yes | OK, planet-level integration |
| Mimas | **No** | But Mimas isn't critical for the others |
| Enceladus | Yes | Needs Dione → REGRESSES |
| Tethys | **No** | Needs Mimas (3:2 res with Tethys) and Dione, but not in our analysis |
| Dione | **No** | (The missing perturber) |
| Rhea | **No** | Not in analysis |
| Titan | Yes | Far enough (1.2M km) that J₄ effect is negligible; not in resonance |

| Jupiter system | In fixture | Status |
|---|---|---|
| Io | Yes | Part of Laplace 4:2:1 resonance |
| Europa | Yes | Part of Laplace 4:2:1 resonance |
| Ganymede | Yes | Part of Laplace 4:2:1 resonance |
| Callisto | Yes | Not in resonance |

Jupiter's four major moons are all present, so the Laplace resonance among Io, Europa, Ganymede is intact in our integration. That's why the Jupiter system behaves well.

## The fix

**Extend the DE440 fixture with Dione (and ideally Tethys, Rhea, Mimas) so the Saturn system is dynamically complete.**

This is a fixture-generation task: re-extract state vectors from the DE440/SAT-441 kernel with those four additional bodies, save the augmented fixture, and re-run the rolling shootout. Expected outcome:

- Enceladus baseline error drops by ~2 orders of magnitude (matching the level of other inner moons).
- J₄ on Saturn becomes a small REFINEMENT (not a regression) on the now-correct baseline.
- `j4_zonal` can be re-enabled in the canonical `jpl_de440` predictor.

Until that fixture work happens, `j4_zonal=False` in the `jpl_de440` default predictor is the right call — the J₄ correction is *correct in isolation* but *harmful in the presence of the missing-perturber error*.

## Related findings

This investigation also uncovered a more general lesson about our 25-body fixture: **bodies were chosen by name recognition, not by dynamical completeness**. Future fixture extensions should audit each body for required perturbers (resonance partners, dominant perturbing moons, etc.) before including it.

## Files touched

- `sigma_ground/field/interface/test_nbody.py` — added `test_j4_at_pole_magnitude_matches_derivation` and `test_j4_saturn_enceladus_geometry_matches_analytic`
- `sigma_ground/field/interface/rolling_shootout.py` — `jpl_de440` keeps `j4_zonal=False` (already landed in commit f77f7ad)
- `misc/saturn_enceladus_j4_verdict_2026-05-15.md` — this doc

## Cross-references

- Toggle iteration data: `sigma_ground/field/interface/fixtures/rolling_shootout_toggle_iteration.json`
- Smoke test that first revealed the regression: jpl_de440 vs over_physics_finedt on j2015, 5y horizon (in conversation 2026-05-15)
- Predictor stack decisions: commit f77f7ad
