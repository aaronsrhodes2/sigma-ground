# Hierarchical Per-Body dt: Known Correctness Bug — 2026-05-16

**Status:** Implementation committed in [1cc116e] is *broken* in the general case. The methods are kept in the codebase for future repair, but the `jpl_de440_hier` predictor has been removed from `PREDICTORS` and the canonical force-model lineup is unchanged.

## What was tried

After the [c9310a7]/[fd5d631] finding that globally-finer dt helps most bodies but regresses Mimas (28×), Phobos/Deimos (4–6×), and Jupiter (1.7×) via shadow-Hamiltonian phase drift, the next architectural target was **per-body dt**: slow bodies stay at dt=0.1d while fast moons substep at dt/N.

The committed implementation in [`sigma_ground/field/interface/nbody.py`](sigma_ground/field/interface/nbody.py:825) adds two methods:

- `_selective_fr_step(dt, body_indices)` — Forest-Ruth update on a subset of bodies; other bodies are frozen positionally but contribute to force calculations.
- `forest_ruth_step_hierarchical(dt, fast_indices, n_substeps)` — operator-split macro step: slow bodies advance at dt with fast bodies frozen, then fast bodies substep through dt at dt/N with slow bodies at linearly-interpolated positions.

This is a **body-split operator scheme**: H = H_slow + H_fast where the split is by *body*. It's the most intuitive approach but it has a correctness bug.

## The bug

Tested on two controlled systems where uniform-dt reference results are known:

### Test 1 — 2-body Earth-Moon (no Sun)

Reference: uniform dt=0.1d, 300 steps → 30 simulated days.
Compare against:
- Uniform dt=1.0d, 30 steps (deliberately coarse)
- Hierarchical macro=1.0d, Moon substepped 10× (effective Moon dt=0.1d)

```
Reference uniform dt=0.1d:  Moon at (3.019e8, 2.678e8, 0)
Uniform dt=1.0d:            Moon at (3.051e8, 2.638e8, 0)  |Δ|=5.06e6 m
Hierarchical 1.0d / 0.1d:   Moon at (3.345e8, 2.392e8, 0)  |Δ|=4.34e7 m
```

Hierarchical is **8.5× worse** than uniform-coarse. The Moon ends up 43,400 km off the reference trajectory.

### Test 2 — 3-body Sun-Earth-Moon

Same comparison; Moon position after 30 simulated days:

```
Reference uniform dt=0.1d:  Moon at (1.304e11, 7.409e10, 0)
Uniform dt=1.0d:            Moon at (1.304e11, 7.409e10, 0)  |Δ|=5.47e6 m
Hierarchical 1.0d / 0.1d:   Moon at (1.303e11, 7.335e10, 0)  |Δ|=7.38e8 m
```

Hierarchical is **135× worse** than uniform-coarse. Moon drifts 738,000 km off.

### Root cause

The bug is in the slow-body advancement step. When we step slow bodies at dt with fast bodies frozen at their start positions, the slow bodies feel a **constant** gravitational pull from the fast body, instead of a **time-averaged** pull as the fast body orbits.

For the 2-body Earth-Moon test, Earth's entire motion is driven by Moon's gravity. With Moon frozen, Earth gets a constant +x pull, producing a parabolic trajectory that diverges from the true Earth-Moon barycenter orbit.

For the 3-body case, Sun dominates Earth's motion, but the Moon's pull on Earth is still wrong (frozen instead of time-averaged), and the Moon's substeps then run against an incorrect Earth trajectory. Errors compound: wrong slow trajectory → wrong fast trajectory → wrong final state.

This affects ANY case where the fast body produces a non-negligible perturbation on slow bodies, which includes most solar-system moon systems. The Earth-Moon barycenter motion alone produces wobbles of ~tens of km in Earth's heliocentric path; missing this entirely shifts the Moon's heliocentric trajectory dramatically.

## What the fix is

The standard correct approach for multi-timescale Hamiltonian integration is **symplectic multi-timestep** (Tuckerman, Berne, Martyna 1992; commonly called "RESPA" in molecular dynamics, "Wisdom-Holman" in astrodynamics). The decomposition is by **force type**, not body:

```
H = H_slow + H_fast
exp(dt · L) ≈ exp(dt/2 · L_slow) · [exp(dt/N · L_fast)]^N · exp(dt/2 · L_slow)
```

For our problem:
- `H_fast` = dominant Newtonian force from the parent body (e.g. Saturn on Mimas)
- `H_slow` = everything else (Sun on Mimas, J₂/J₃/J₄ corrections, EIH cross-terms, perturbations from other moons)

The structure:
1. Half-kick all bodies using slow forces (gravity from non-parent attractors + zonal corrections + EIH + tides + SRP)
2. Run N substeps where each step is a Forest-Ruth step on just the parent-Kepler force for fast bodies (and an analytic Kepler step would even better — exact, no integration error)
3. Half-kick all bodies using slow forces again

This is symplectic for fixed dt and preserves long-term stability. Implementation cost is significantly higher than the current attempted scheme — needs per-body identification of "parent attractor" and per-force-type acceleration computation.

## What landed today

- Methods `_selective_fr_step` and `forest_ruth_step_hierarchical` remain in `nbody.py` with **prominent ⚠ KNOWN BROKEN ⚠ banners** in their docstrings. They can be repaired later when the proper algorithm is implemented; deleting them and re-writing later is unnecessary churn.
- `jpl_de440_hier` was removed from `PREDICTORS` — the canonical predictor lineup is unchanged from [c9310a7]. Use `jpl_de440` (dt=0.1d uniform) as the canonical for now.
- 6 unit tests for the hierarchical method still pass (they test the *mechanics* of the operator split: time advances correctly, n_substeps=1 reduces to uniform, empty fast_indices reduces to uniform). The mechanics are right; the algorithm is wrong.
- 1 new unit test `test_hierarchical_KNOWN_BROKEN_two_body_earth_moon` documents the bug as a regression test. When the fix lands, the assertion direction needs to be flipped.

## Lessons

1. **Always validate against a known-good reference before trusting a new integration scheme.** The 6 unit tests for the hierarchical method validated its mechanics but not its accuracy — they passed even though the algorithm is wrong. The 2-body Earth-Moon test is the simplest possible accuracy benchmark and would have caught this in seconds. Run that test FIRST next time.

2. **Operator splitting by body is not the same as operator splitting by force type.** The first is intuitive but loses too much physics (the fast body's gravity on slow bodies). The second is the classical correct approach.

3. **Symplectic integrators have subtleties** that aren't apparent from "I followed the recipe in Wikipedia." Forest-Ruth at fixed dt is symplectic; my operator-split scheme isn't, and the secular drift quickly dominates.

4. **Commit broken code with a banner, not silently.** [1cc116e] committed the broken hierarchical with a clean test suite. The follow-up [this commit] adds the failing accuracy test and the verdict doc — that's the honest record. Squashing or reverting would lose the audit trail.

## References

- Tuckerman, Berne, Martyna 1992: "Reversible multiple time scale molecular dynamics", J. Chem. Phys. 97:1990 — the original RESPA paper.
- Wisdom & Holman 1991: "Symplectic maps for the n-body problem", AJ 102:1528 — the astrodynamics analog using Kepler splitting.
- Hairer, Lubich, Wanner 2006: "Geometric Numerical Integration" — chapter VIII on multiple timestep methods.
- The committed broken implementation: commit [1cc116e]
- The failing validation test: `TestHierarchicalForestRuth::test_hierarchical_KNOWN_BROKEN_two_body_earth_moon`
- The dt-tradeoff context that motivated this work: `misc/dt_tradeoff_verdict_2026-05-15.md`
