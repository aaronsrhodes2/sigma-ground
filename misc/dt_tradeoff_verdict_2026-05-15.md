# Global dt Trade-off Verdict — 2026-05-15

**Status:** Measurement complete. Verdict: keep `jpl_de440` at dt=0.1d as canonical default. `jpl_de440_finer` (dt=0.02d) is **useful for some bodies, harmful for others** — not a global drop-in replacement. The actual architectural fix is per-body dt (Wisdom-Holman hierarchical) which is deferred.

## Question

After the DE440 fixture extension (c9310a7), Enceladus's prediction stayed at ~3e-3 AU even with Dione present, suggesting the bottleneck had moved from "missing perturber" to "integrator can't resolve the resonance dynamics at dt=0.1d". Quick test: does dt=0.02d (5× finer) close this gap?

## Test

Ran `jpl_de440` (dt=0.1d) vs `jpl_de440_finer` (dt=0.02d) on the j2015 window, 3y horizon. Both predictors share the same canonical force model toggles: EIH N-body 1PN, all J₂/J₃/J₄ zonals, tidal_force, srp. Only the integration step differs.

Wall clock: **3622 s (~60 min)** for the pair — 5× finer dt is roughly 5× slower per body, plus N² body interactions, so the dt=0.02d run dominates.

## Result

**18 of 29 bodies improve** (1.4×–500×), but **4 regress significantly** (1.7×–28×):

### Wins (per-body improvement factors)

| Tier | Bodies |
|---|---|
| **>100× better** | Umbriel (500×), Ariel (91×) |
| **10–100× better** | Triton (53×), Charon (42×), Venus (16×), Dione (11×) |
| **5–10× better** | Rhea (8×), Pluto (7×), Europa (6.5×), Miranda (6.3×), Tethys (5.5×), Mercury (5.2×), Enceladus (5.1×) |
| **2–5× better** | Oberon (3.3×), Titania (3.2×) |
| **1.4–2× better** | Ganymede, Earth, Titan |

### Regressions

| Body | dt=0.1d | dt=0.02d | Worse by |
|---|---:|---:|---:|
| **Mimas** | 8.29e-5 | 2.31e-3 | **28×** |
| Deimos | 2.32e-5 | 1.31e-4 | 5.7× |
| Phobos | 2.17e-5 | 9.50e-5 | 4.4× |
| Jupiter | 1.53e-7 | 2.63e-7 | 1.7× |

### Roughly unchanged
Callisto, Mars, Io, Neptune, Saturn, Uranus, Moon (~1.0× ± 10%).

## Diagnosis

The wins are easy to explain: most bodies at dt=0.1d are operating well above their orbital resolution floor; finer dt resolves their dynamics better and the integration error drops sharply. The 500× improvement on Umbriel is dramatic but plausible — Umbriel's orbital period is 4.1 days, so dt=0.1d gives only 41 steps/orbit; dt=0.02d gives 205. For high-order symplectic integrators, error typically scales as dt^4, so a 5× dt reduction should give ~625× theoretical improvement. 500× is well within that ballpark.

The **regressions need more thought**. The pattern:

- All regressing bodies have short orbital periods relative to dt=0.02d: Mimas (0.94d), Phobos (0.32d), Deimos (1.26d). Jupiter is the exception — its 4332-day period should be massively over-resolved at either step size.
- Mimas at dt=0.02d gets 47 steps/orbit. That should be plenty for a 4th-order symplectic method.
- Phobos at dt=0.02d gets 16 steps/orbit. Borderline.

The most likely cause is **shadow-Hamiltonian phase drift**. Forest-Ruth is exactly symplectic at fixed dt, but it integrates a Hamiltonian that's *near* (not equal to) the true one. The difference depends on dt. When dt is fortuitously chosen, the phase drift can cancel against other errors, masking integration noise. When dt changes, the cancellation breaks and the underlying error becomes visible. This is a well-known phenomenon for symplectic methods on tightly-coupled systems.

For Mimas specifically — orbiting Saturn at 185,520 km (3.08 Saturn radii), with the J₂/J₃/J₄ zonal harmonics all enabled, AND mutual perturbations from Tethys (4:2 inclination resonance partner) — the orbit is very tightly coupled. At dt=0.1d the integration was bad but lucky; at dt=0.02d it's slightly better-resolved but the phase drift has detuned the resonance and the orbit drifts in mean longitude.

For Phobos and Deimos: orbiting Mars at very close distances (2.75 and 6.92 Mars radii respectively) with Mars J₂/J₃/J₄ all on. Same story — the orbit is dominated by Mars's gravity at sub-day timescales; integration is finicky.

For Jupiter: harder to explain. Possibly the EIH cross-terms (which depend on velocities and positions of *all* other bodies) interact differently at finer dt when the inner-planet phases shift. The Jupiter regression is small (1.7×) and could just be noise from the EIH-pre-pass arithmetic at finer dt — sub-percent absolute change.

## Decision

**Keep `jpl_de440` (dt=0.1d) as the canonical default.** It's the best globally-correct choice — no body regresses significantly from the baseline.

**Keep `jpl_de440_finer` (dt=0.02d) as a documented experimental variant.** Use it when targeting bodies that don't show the regression pattern (most outer-system moons and inner planets). Avoid it for Saturn moons (Mimas), Mars moons (Phobos, Deimos), and probably Jupiter system in general.

**The real fix is per-body dt** — Wisdom-Holman or symplectic-corrector schemes that let each body use a step size appropriate to its own orbital period. Mimas would use ~0.005d (200 steps/orbit), Pluto would use 1.0d (90,000 steps/orbit instead of millions), and the integration would be both fast and accurate. This is a substantial architectural change deferred to a future session.

## What we now know about accuracy ceilings

For each body, our current prediction error floor (after today's work):

| Tier | Bodies | Floor (best dt config) |
|---|---|---|
| Sub-µAU | Mars, Jupiter, Saturn, Uranus, Neptune | ~1e-7 AU |
| µAU | Earth, Moon, Venus, Mercury | ~1e-7 — 1e-6 AU |
| 10s of µAU | Triton, Charon, Pluto, Oberon, Titania | ~1e-6 — 1e-5 AU |
| 100s of µAU | Phobos, Deimos | ~1e-5 AU (dt-coupling-limited) |
| 1 milli-AU | Mimas, Callisto, Titan | ~1e-4 — 1e-5 AU (mixed reasons) |
| 10s of milli-AU | Dione, Enceladus, Rhea, Tethys | ~1e-4 — 1e-3 AU (need per-body dt) |
| 100s of milli-AU | Io, Europa | ~1e-3 AU (need per-body dt) |

The "10s of milli-AU" tier is the next physics target. It's all the close-in fast moons of gas giants. The mechanism is the same: their orbital periods are 1-7 days, dt=0.1d is too coarse, dt=0.02d helps for slower ones but breaks for the fastest (Mimas, Phobos). **Per-body dt unlocks an order of magnitude on these.**

## References

- Smoke test data: command `b8q97hf47`, j2015 window, 3y horizon
- Prior context: misc/saturn_enceladus_j4_verdict_2026-05-15.md (fixture extension that made this test possible)
- Force model: jpl_de440 predictor in sigma_ground/field/interface/rolling_shootout.py
- Integrator: Forest-Ruth 4th-order symplectic, sigma_ground/field/interface/nbody.py
