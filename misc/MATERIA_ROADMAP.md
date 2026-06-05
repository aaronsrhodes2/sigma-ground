# Materia Improvement Roadmap — multi-verb decomposition

## North star

Handle complex (3–4 verb) scenarios by **decomposing them into sub-solves**:
the translator *plans* the decomposition, each grounded verb *does* its physics,
and the engine *combines* the results deterministically. Complexity becomes
*more steps*, never a harder single step — which is what keeps a 7b sufficient.

Render / shape is **delegated to MatterShaper**. The physics library stays
scalar/primitive-parameterized (mass, area, length, dimensions). What a body
*looks like* never affects how it behaves, so shape is a rendering concern, not
a physics one. This is a hard boundary: **no shape generation in the physics
library.**

## The honest boundary (truth first)

Decomposition conquers **width**, not **depth**.

- **Separable** sub-problems (solve A, then B with A's result) → chain them. This
  is where "break it apart and it won't matter how complex" genuinely holds.
- **Coupled** feedback (A and B affect each other every instant — drag heats a
  body, heat thins it, thinning changes drag…) → CANNOT be decomposed into
  independent steps. It must be solved *together* inside one coupled verb.

So: width (many parts) is free; depth (tight coupling) needs a dedicated coupled
solver. The planner must route coupled scenarios to one verb, not a chain.

## Phase 1 — Make decomposition real (the keystone)

1. **Chained data flow.** A `SpecStep` can bind an input slot to a prior step's
   output field (e.g. step 2's `v0_mps` ← step 1's `apex_speed_m_s`). `run_spec`
   threads results forward. This is the one missing piece — today's steps are
   independent.
2. **Multi-step planning by qwen.** The translator emits a SHORT (≤4) ordered
   list of verb calls + bindings; every step validated against the manifest,
   every step grounded. Cap the length so the 7b stays reliable.
3. **Synthesis pass.** A `combine` step reads the sub-results and writes ONE
   unified answer. The chain passes its self-check iff *every* sub-step's
   self-check passes (weakest-link).
4. **Prove it** on one genuinely composable example — e.g. vertical launch →
   high-altitude descent → drag heating — so data flow is demonstrated, not
   asserted.

## Phase 2 — Robustness & honesty in chains

5. **Partial answers.** A chain containing an out-of-scope step answers the
   solvable sub-parts and flags the gap `[not yet modeled: X]` — never all-or-
   nothing, never faked.
6. **Coupled-vs-separable guard.** Coupled scenarios route to a dedicated
   coupled verb; the planner is prevented from decomposing a feedback loop into
   independent steps.

## Phase 3 — Render handoff (delegated to MatterShaper)

7. **Trajectory / render data contract.** Every verb result carries the
   `history` (positions + velocities over time, already recorded by
   `simulate_drag_run`) plus a primitive body descriptor (shape type, scalar
   dimensions, material). MatterShaper consumes that and draws it. The physics
   side ships numbers; MatterShaper owns the picture.

## Phase 4 — Grow the building blocks (parallel, ongoing)

8. **More verbs per family** — statics → rigid-body → continuum. Each new verb
   shrinks the honest-decline list AND gives the planner more pieces to compose.
   Statics next: **catenary sag** and **Euler buckling** are exact closed forms,
   so they convert three current declines (cantilever, tripod, catenary) into
   answered, self-checking procedures.

## Sequencing

Phase 1 (decomposition keystone) → Phase 2 (honesty in chains) → Phase 3 (render
contract; can run in parallel) → Phase 4 (coverage; ongoing). Phase 4 also feeds
Phase 1 — the more verbs exist, the more the planner can decompose into.
