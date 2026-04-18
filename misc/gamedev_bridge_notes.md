# Game-Dev Bridge — Live MCP Oracle (direction doc)

**Date:** 2026-04-17
**Status:** Design notes. No implementation this phase. Phase XIII will build.
**User direction (captured 2026-04-17):**

> Linking sigma-ground to recent Claude-Code-driven game-dev work. The chosen approach
> is **Nagatha MCP as a live physics oracle inside a game loop** — the game calls
> Nagatha at runtime for real physics answers instead of faking them.

**North-star reminder (captured 2026-04-17):**

> "Of course I secretly hope that someday I have something concrete to contribute to
> science, so that is the side-goal of the project is to bring all of physics together
> in one place and hope obvious patterns emerge."

The game-dev bridge is compatible with the side-goal in an important way: every live
MCP query from a game is **another place where disparate physics gets compared against
itself**. If a game queries `lookup_material` for a crystal and later queries `optics`
for the same crystal, sigma-ground has to answer consistently — and the consistency
check *is* a pattern-emergence opportunity. The game becomes another vantage point on
the same library.

## Grounding — what already exists

Nagatha MCP (v5.0.0) exposes sigma-ground as callable MCP tools. From this conversation
I can see the surface:

| Tool | Kind | Good for |
|------|------|----------|
| `search` | discovery | find a physics function by keyword |
| `list_categories` / `list_functions` | discovery | browse sigma-ground domains |
| `describe` | introspection | get a function's docstring + signature |
| `lookup_material` / `lookup_shape` | read-only lookup | material/shape by name |
| `material` / `shape` / `element` | full object | build a material/shape/element |
| `run` | compute | call an arbitrary sigma-ground function |
| `simulate` | compute | step a dynamics scene |
| `harvest` | aggregation | pull results out of a scene/run |
| `history` | state | recall prior runs |
| `generate_test` | auxiliary | produce a pytest for a function |
| `bg3_logs` | adjacent | existing game-log scaffolding |

Plus 78 `sigma_ground.field.interface.*` modules (optics, thermal, mechanical, magnetism,
fluid, crystal_field, etc.) all reachable through `run`.

## The five design questions, answered

### 1. Client side — what does the game engine need?

**Preferred path:** game engine speaks MCP directly via an MCP client library. Options:
- **Python game loop** (pygame, arcade, or custom) — direct `mcp` Python package,
  lowest friction, matches sigma-ground's own language.
- **Godot** — GDScript can call out to an HTTP shim over MCP, or use GDExtension with
  a Rust/Python MCP client. Medium friction.
- **Unity / Unreal** — would need an HTTP/WebSocket shim fronting Nagatha; native
  MCP client libraries for C# / C++ are less mature. Higher friction.

**Recommendation:** start with a **Python game loop + pygame**. Same language as
sigma-ground, no shim, fastest path to a working demo. If the game concept grows past
what pygame can handle, the second step is a thin HTTP/JSON wrapper around Nagatha
that non-Python engines can call.

### 2. Latency budget — what can run at each cadence?

| Cadence | Budget | Safe MCP calls |
|---------|--------|----------------|
| **Load-time** (one-shot) | seconds | `simulate` (scene init), full inventory build, `generate_test`, anything on `sigma_ground.inventory.builder` |
| **Tick-rate** (10–30 Hz) | ~30–100 ms | `run` for σ-effect recomputation at scale-change events, `material` for newly-spawned objects, `describe` for NPC dialogue lookups |
| **Frame-rate** (60+ Hz) | <16 ms | **Cached** `lookup_material` / `lookup_shape` / `element` results only — never a raw MCP call |

Rule of thumb: MCP round-trip latency on localhost is typically 5–30 ms. That is fine
for tick-rate and load-time but **unsafe for frame-rate without a cache layer**. The
game needs a client-side cache keyed by `(tool_name, args_hash)` with a TTL long enough
to span a frame but short enough that σ-dependent values refresh when σ changes.

### 3. Determinism — save/load reproducibility

Sigma-ground is largely deterministic (pure Python, no RNG in `constants.py` or `field/*`).
But several modules are **NOT** deterministic without explicit seeding:

- `sigma_ground.dynamics.fluid.*` — SPH steps call `random` in parcel perturbation
  (verify). Needs `seed=` parameter audit.
- `sigma_ground.inventory.generator.*` — procedural material generation likely uses
  randomness. Needs `seed=` audit.
- Any function that uses a Monte-Carlo integration path.

**Action before Phase XIII:** grep for `random.` and `numpy.random.` across sigma_ground,
list every function that calls them, confirm each accepts a `seed=` parameter, and have
Nagatha surface `seed` in its MCP signatures for those functions. Functions that cannot
be made deterministic (e.g., wall-clock-dependent) are blacklisted from the game's
save-affecting queries.

### 4. Failure mode — what if Nagatha is down?

Three tiers of degradation:

1. **Cached last-known-good response** — for lookups (materials, shapes, elements).
   Game continues with stale but valid physics; user sees no difference.
2. **Fake-physics fallback** — for runtime `run`/`simulate` calls that have no cache.
   Game falls back to a hand-authored approximation (linear interpolation, hard-coded
   values) and flags the session as "non-canonical" so telemetry knows the physics
   wasn't live.
3. **Hard fail** — for content-pipeline steps that *must* be canonical. The game
   surfaces an error, the user retries.

Recommendation: implement (1) always; (2) for frame-rate paths only; (3) for load-time
content generation.

### 5. MVP surface — which 5–10 tools?

For the first game demo, the minimum viable Nagatha surface is:

1. `lookup_material(name)` — resolve a named material to its physics bundle
2. `element(Z)` — build an element by atomic number
3. `shape(kind, params)` — build a geometric shape with σ-aware properties
4. `run("optics.refractive_index", args)` — a single `run` target that covers the
   demo's optical queries
5. `run("mechanical.hardness", args)` — hardness for collision / damage calcs
6. `describe(fn_name)` — so NPC dialogue or tooltips can pull the real docstring
7. `search(keyword)` — for a "physics encyclopedia" in-game UI
8. `history(session_id)` — for save/load debugging

Everything else can wait. Keep the game tiny; expand surface area only after demo
works end-to-end.

## Concrete first-demo proposal

A **"σ-sandbox" demo game** scoped to one scene:

- Player has a slider from σ=0 to σ=σ_conv ≈ 1.85.
- Three demo objects on screen: a crystal (queries `optics`), a steel beam (queries
  `mechanical.hardness`), a flame (queries `thermal.combustion`).
- As σ moves, each object's visible behavior updates from live sigma-ground queries.
  The crystal's refractive index shifts, the beam softens and eventually fails at
  σ_conv, the flame changes color per thermal-emission spectrum.
- No complex game loop, no goal — pure physics exploration.

Deliverable: ~500 lines of pygame code + a ~50-line Nagatha client wrapper + a screenshot
recording the scene as a short GIF. This is the shippable proof-of-concept for Phase XIII.

## Dependency list for Phase XIII

1. **Nagatha MCP server running locally** (v5.0.0+)
2. **Python MCP client library** (pip install)
3. **pygame** or equivalent
4. **Determinism audit** (grep + `seed=` additions) — blocks save/load features, can
   ship without for the first demo
5. **Cache layer** (client-side dict with TTL) — 30 lines of code
6. **Fallback-physics table** (hand-authored) — only needed if Nagatha-down degradation
   is in scope for the demo

## Why this matters to the side-goal

Every game query is another place where disparate physics is compared against itself.
If Phase XIII ships a demo where three independent sigma-ground modules (optics,
mechanical, thermal) all agree on the same σ-slider, that is direct evidence that the
library's σ-threading is internally consistent. If they *disagree*, the game exposes a
bug more vividly than any unit test would — because the player *sees* the disagreement.

Either way, the game is a pattern-detector. Ship it.
