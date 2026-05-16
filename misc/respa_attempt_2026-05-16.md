# RESPA Per-Body dt: Attempt Verdict — 2026-05-16

**Status:** Implementation committed as EXPERIMENTAL — works for 2-body cases where the parent-pair Newton force genuinely dominates, but fails for the general solar-system case where the parent-pair force and the "slow" forces are comparable in magnitude. Not added to the canonical `PREDICTORS` list. The path to a correct general solution is Wisdom-Holman with an analytic Kepler inner solver, deferred to a future session.

## Background

After the body-split hierarchical scheme failed (see `misc/hierarchical_dt_known_bug_2026-05-16.md`), the next attempt was a force-split scheme: separate the Hamiltonian into `H_fast` (Newtonian gravity for declared parent-child pairs) and `H_slow` (everything else), and use the Tuckerman-Berne-Martyna 1992 RESPA scheme:

    exp(dt L) ≈ exp(dt/2 L_slow) · [exp(dt/N L_fast)]^N · exp(dt/2 L_slow)

with the inner block run as N steps of velocity-Verlet leapfrog over `H_fast`.

This is symplectic (a real Strang splitting on the Hamiltonian), so it doesn't have the same operator-split correctness bug as the body-split scheme. Implementation in [`sigma_ground/field/interface/nbody.py`](sigma_ground/field/interface/nbody.py):

- `compute_fast_accelerations()` — Newton-pair acceleration for each body's declared parent, with the reciprocal back-reaction on the parent.
- `compute_slow_accelerations()` — `compute_accelerations() − compute_fast_accelerations()`.
- `respa_step(dt, n_substeps)` — the RESPA scheme above.
- `NBodySystem` now accepts `parent_attractor_indices: list[int | None]` to declare which body is each fast body's parent.

## Validation

Two controlled tests using known-good reference (uniform dt=0.1d Forest-Ruth over 30 simulated days):

### 2-body Earth-Moon (no Sun)

```
Reference (dt=0.1d FR):     Moon at (3.019e8, 2.678e8, 0)
Coarse (dt=1d FR):          err = 5.063e6 m
RESPA (1d / 0.1d × 10):     err = 4.041e5 m
```

**RESPA is 12.5× better than uniform-coarse.** The scheme works correctly here.

The reason it works: in the 2-body system, the entire dynamics is the Earth-Moon Newton pair. With Moon's parent set to Earth, `compute_fast` captures the entire force, `compute_slow` is identically zero, and the inner velocity-Verlet at sub_dt=0.1d gives full-system accuracy at 0.1d resolution.

### 3-body Sun-Earth-Moon

```
Reference (dt=0.1d FR):     Earth and Moon precise
Coarse (dt=1d FR):          Earth err 6.6e4 m, Moon err 5.5e6 m
RESPA (1d / 0.1d × 10):     Earth err 7.4e5 m, Moon err 6.5e7 m
```

**RESPA is 11× WORSE than uniform-coarse** for both Earth and Moon.

The reason it fails: with Moon's parent set to Earth, the Sun's gravitational pull on the Moon (≈5.9×10⁻³ m/s²) lands in `H_slow`, while Earth's pull on Moon (≈2.7×10⁻³ m/s²) lands in `H_fast`. The "slow" force is *larger* than the "fast" force. The Strang splitting only works well when slow forces are genuinely slow-varying and small compared to fast — in this case neither holds.

The Sun's pull on the Moon varies on the Moon's *orbital* timescale (because the Moon's heliocentric position changes as it orbits Earth), so it isn't slow in any meaningful sense. Sampling it only at the macro_dt boundaries (1 day) loses the within-orbit variation.

## Why the attempt fails in general

The body-pair parent-attractor decomposition assumes a clean hierarchy where each body has *one* dominant force from a single attractor and all other forces are perturbations. This is the assumption in:

- Wisdom-Holman 1991: Moon orbits Earth, perturbed weakly by Sun's tidal pull (Sun's *direct* force on Moon would be in the "fast" Kepler part, not slow).
- N-body codes like SyMBA / rebound: bodies orbit the central star, perturbed by each other.

The Wisdom-Holman trick is that the inner block isn't velocity-Verlet — it's an **analytical Kepler propagator**. The fast block reduces to "advance each body's two-body Kepler orbit by dt analytically (no integration error)" plus interaction kicks for perturbations. This sidesteps the order-of-magnitude question entirely: even if the Kepler force is comparable to perturbations, the Kepler step is *exact*, so the Strang split only carries error from the perturbations (which ARE small).

My implementation uses velocity-Verlet for the inner block. That's only sufficient when the fast force genuinely dominates AND varies on a much shorter timescale than the slow force. For solar-system moons, the natural split (Kepler around parent + perturbations) doesn't work with leapfrog inner because Sun's pull on a moon is comparable to the moon's parent-pull at the orbital timescale.

## What was retained

The methods `compute_fast_accelerations`, `compute_slow_accelerations`, and `respa_step` remain in `nbody.py` with a **⚠ EXPERIMENTAL** banner in the docstring pointing to this verdict doc. They are functionally correct for the specific case where the parent-pair force genuinely dominates (validated in 2-body). They are NOT plumbed into any canonical predictor in `PREDICTORS` — the rolling shootout uses `jpl_de440` (Forest-Ruth at uniform dt=0.1d) and its variants without RESPA.

The implementation has value as scaffolding for the eventual Wisdom-Holman implementation: the `parent_attractor_indices` machinery, the fast/slow accelerator split, and the symplectic outer structure are all reusable. What needs to change is the inner block: replace velocity-Verlet with an analytic Kepler propagator.

## Path forward

A proper Wisdom-Holman implementation would need:

1. **Analytic Kepler propagator.** Given Moon's position, velocity, and Earth's GM, propagate the Moon's Kepler orbit around Earth forward by dt analytically. Solve Kepler's equation iteratively; convert mean anomaly to true anomaly to position+velocity. This is standard textbook material (e.g. Murray & Dermott §2.4) but involves careful handling of eccentric / parabolic / hyperbolic orbits.

2. **Frame transformation.** Wisdom-Holman uses Jacobi coordinates: each body's position is referenced to its parent, not to the inertial frame. The Kepler step advances each body in its parent-relative coordinates. Conversions between inertial and Jacobi happen at the symplectic boundary.

3. **Perturbation kicks.** Between Kepler steps, apply interaction kicks for all non-Kepler forces (Sun's *direct* force on moons via Sun-moon Newton, J₂/J₃/J₄, GR, etc.). These are the "slow" forces in this scheme — and they're genuinely small relative to the parent-pair Kepler force, so the Strang splitting works.

This is roughly the rebound code's `whfast` integrator. Implementing it from scratch is a substantial undertaking — probably a full session focused on just the Kepler-solve + Jacobi transformations + testing.

## Tests added

- `test_respa_two_body_earth_moon_beats_coarse` — pins the 2-body win (RESPA error < 0.2 × coarse error).
- `test_respa_three_body_KNOWN_LIMITATION_sun_earth_moon` — pins the 3-body failure (RESPA error > coarse error) so we'll notice if a future fix improves it.
- `test_respa_n_substeps_zero_raises` — argument validation.
- `test_respa_with_no_parents_degenerates_to_slow_kick_only` — sanity check.

## Lessons

1. **Symplecticity is necessary but not sufficient.** My broken hierarchical scheme wasn't symplectic and broke for that reason. The RESPA scheme IS symplectic and still breaks for our specific use case. Operator splitting requires the slow forces to genuinely be small/slow relative to fast; otherwise the Strang error from inserting delta-function kicks at the macro boundaries dominates.

2. **Pre-baked solver inside an integrator beats numerical inner integration.** Wisdom-Holman's Kepler-solve gives exact two-body propagation; my velocity-Verlet inner gives 2nd-order error per step. For high-eccentricity or fast orbits, the difference matters.

3. **Verify decomposition magnitudes before coding.** I should have computed the relative magnitudes of `|F_parent|` and `|F_other|` for Mimas (Saturn ~3e-2 m/s² vs Sun ~5e-5 m/s², ratio ~600) before writing the 3-body Earth-Moon test. The Saturn-Mimas case actually does have a clean hierarchy (Saturn's pull is 600× Sun's pull on Mimas), so RESPA would have worked there. The Earth-Moon-Sun system was a poor first validation target precisely *because* it doesn't have a clean hierarchy.

## References

- Tuckerman, Berne, Martyna 1992: "Reversible multiple time scale molecular dynamics", J. Chem. Phys. 97:1990
- Wisdom & Holman 1991: "Symplectic maps for the n-body problem", AJ 102:1528 — the right tool for our problem
- The implementation: commit [this one], methods `compute_fast_accelerations`, `compute_slow_accelerations`, `respa_step` in nbody.py
- The body-split predecessor that failed: `misc/hierarchical_dt_known_bug_2026-05-16.md`
- The dt-tradeoff context: `misc/dt_tradeoff_verdict_2026-05-15.md`
