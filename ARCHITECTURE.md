# Mentat — Architecture

**Mentat** is the umbrella for a pure-Python physics + rendering stack. The
Python import root stays `sigma_ground`; "Mentat" is the product/brand, and the
**MCP server is its public face** — exposing every service's tools to LLM clients.

## The role layering (the import contract)

Six service roles, in dependency tiers. **A module may import only its own tier
or below** — enforced on every test run by `tests/test_layering.py`, a
zero-dependency AST guard:

```
tier 0   kernel/       geometry + math primitives: shapes, csg, parts, vec (the single Vec3)
             ▲
tier 1   field/        σ-field physics + authoritative constants     (Sigma Ground)
         inventory/    particle inventory & mass closure             (Quarksum)
         dynamics/     N-body, SPH, Barnes-Hut, integrators (shared sim engine)
             ▲
tier 2   deckard/      matter compiler: a name → a validated Construct
             ▲
tier 3   materia/      physics / movement engine (+ materia.labs)
         radiance/     renderer: SDF ray-march + entangler push renderer
             ▲
tier 4   mcp/          the Mentat face — MCP tools over every service
```

`field.constants` is the authoritative constants source and is importable from
any tier (a universal foundation, exempt from the tier rule).

## Package map (`sigma_ground/`)

| Role | Location | Notes |
|---|---|---|
| **Sigma Ground** | `kernel/` + `field/` | `kernel/` = geometry/math primitives; `field/` = σ-physics (`field/constants.py` authoritative) |
| **Quarksum** | `inventory/` | materials → molecules → atoms → particles → quarks; CLI `mentat` |
| **Materia** | `materia/` (+ `materia/labs/`) | drag / orbital / scenario engine; `dynamics/` is the shared kernel-tier sim engine |
| **Deckard** | `deckard/` | fits primitives → SDF `Construct` with a mass/CoM/inertia self-check |
| **Radiance** | `radiance/` (+ `radiance/entangler/`, `radiance/materials/`) | SDF renderer plus the MatterShaper push/entangler renderer, folded in |
| **Mentat MCP** | `mcp/` | `FastMCP("mentat")` server + tools; `mcp/benchmark/` is a dev-only eval harness |

### Compatibility shims
The geometry kernel moved into `kernel/`, but the old top-level paths remain as
thin re-export shims, so existing imports keep working with **class identity
preserved** (`isinstance` / dataclasses unaffected):

| Old path | Canonical home |
|---|---|
| `sigma_ground.shapes` / `.csg` / `.parts` | `sigma_ground.kernel.*` |
| `sigma_ground.dynamics.vec` | `sigma_ground.kernel.vec` |
| `sigma_ground.labs` | `sigma_ground.materia.labs` |

New code should import from the canonical location.

## Constants — single source of truth
`sigma_ground/field/constants.py` is authoritative; `sigma_ground/constants.py`
and `sigma_ground.kernel` re-export it. Never define a physical constant twice.

## The Deckard → Materia / Radiance contract
Deckard compiles a *name* into a `Construct` (an SDF plus per-point material and
density, with mass / centre-of-mass / inertia). **Materia** consumes it to move
it; **Radiance** consumes it to render it. Deckard depends only on the kernel —
it never imports its consumers, which keeps the pipeline one-directional.

## Testing
- `pytest` from the repo root. Baseline on this branch: **4429 passed, 6 failed,
  6 skipped, 2 xfailed** plus ~2440 subtests.
- The 6 failures are all in `mcp/benchmark/test_targets.py` (QA-threshold tests
  that depend on benchmark result data) — known-red, not a regression signal.
- `tests/test_layering.py` enforces the tier contract on every run.
- The folded-in entangler ships its own tests under `radiance/entangler/`
  (run explicitly: `pytest sigma_ground/radiance/entangler/`).
- Permanently red (expected, EXTREME): `test_jpl_ephemeris` (network-gated),
  `test_position_precision` (simulation-gated).
