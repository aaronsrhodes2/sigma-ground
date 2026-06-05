# PINNACLE TEST — "the chair that falls like a chair"

**Status: EXPECTED RED.** This is the north-star acceptance test for the whole
Materia + shapes + render stack. It is red *by design* until the system is
genuinely good — a goal, never a current claim. (Same spirit as the suite's
permanently-red EXTREME tests.)

## The gold standard

Generate a chair from a plain-English request, tip it, and play the fall in
real time. It passes when **both** hold:

- **A — Perception:** a naive observer watching the real-time playback says
  "yes, that's how a chair falls" — no uncanny floatiness, wrong spin, or
  sinking through the floor.
- **B — Video:** the simulated topple matches real footage of a comparable
  chair tipping, within tolerance.

## Why human perception is a valid oracle (and where it isn't)

Humans are exquisite intuitive physicists for *familiar, everyday* gravity — a
falling chair is maximally familiar, and we flag violations instantly. So
perception is a strong **gate**. But intuitive physics has documented *biases*
in unfamiliar regimes (tubes, microgravity, "heavier falls faster"). So:

- **Perception is the gate** (catches gross unnaturalness, free and immediate).
- **Video is the anchor** (objective, where intuition's biases can't mislead).
- Where they would disagree, trust the math + video.

## Test A — perceptual rubric (decompose "looks wrong" into catchable misses)

Each is a yes/no a layperson can call:

1. **Timescale** — falls in the right *amount of time* (floaty = too slow; frantic = too fast).
2. **Pivot** — tips about the correct edge (the two downhill legs), not a wrong axis.
3. **Rotation** — smooth single-axis rotation, no unphysical pirouette or wobble.
4. **Contact** — legs meet the floor and do **not** sink through (no interpenetration).
5. **Energy** — never bounces higher than it started; loses energy on impact.
6. **Settle** — comes to rest in a plausible pose, not jittering forever.

## Test B — video comparison (input-uncertainty-tolerant)

Film a known chair (measured leg length L, mass, controlled push); mark tilt
angle θ(t) per frame. Compare the sim's θ(t) — robustly, because a real chair's
foot-friction (slide vs. pivot), flex, exact mass distribution and push impulse
are all uncertain:

- **Time-to-topple in units of √(L/g)** — the topple's characteristic timescale.
  Dimensionless, so it normalizes away most size/mass uncertainty. (This is the
  single best objective number.)
- **Angular velocity at first impact.**
- **Qualitative trajectory shape.**

Match the *dimensionless trajectory*, not an exact frame overlay.

## The fidelity ladder — rungs toward the pinnacle

Each rung carries a self-check that is **GREEN when the engine is correct**, so
the math can be *provably right* long before the perceptual pinnacle passes.
This is the whole point: we are never confidently wrong about what's validated.

| Rung | Capability | Self-check (green = engine correct) | Status |
|---|---|---|---|
| 0 | Point-mass drop | terminal velocity = closed form | **GREEN (done, 0.05%)** |
| 1 | Rigid body about a fixed pivot | matches analytic inverted-pendulum; energy + angular momentum conserved | to build |
| 2 | Topple trigger | CoM crosses the support edge at the right tilt (closed form) | to build |
| 3 | Free rotation to first impact | matches analytic rigid rotation | to build |
| 4 | First impact | restitution energy partition; zero interpenetration | to build |
| 5 | Multi-contact settle | bounces and comes to rest on its legs | **RED frontier** |
| 6 | **PINNACLE** | real-time playback passes Test A **and** Test B | **EXPECTED RED** |

## Three properties that keep this honest and tractable

- **Green self-checks under a red pinnacle.** Rungs 0–3 prove the dynamics with
  conservation laws and closed forms while 5–6 are still red. Progress is
  measured by which rungs are green, not by a single binary.
- **Determinism makes the pinnacle debuggable.** When perception flags "wrong,"
  the deterministic trace points to the exact rung's physics — not a vague "tune
  it." A black-box animator can't do that.
- **Real-time playback ≠ real-time solver.** Because the engine is
  deterministic, we pre-compute the trajectory and replay frames at wall-clock.
  Test A runs on the *playback*; the solver may be slow.

## Next concrete step (once the rigid-body verb exists)

Wire rung 6 into the suite as a `pytest.mark.xfail(reason="pinnacle — not yet
built")` test so it is tracked as expected-red from day one — and turn rungs 1–4
green one at a time, each with its closed-form self-check.
