# Mentat audit, 2026-10-13 — quarksum, sigma-ground, Mentat, against the Wolfram-contender goal

Read-only audit of **master** (main checkout) by scripted census, not spot reading. The scripts call every tool in
process (FastMCP, no stdio) and replay one known-good call per tool from the router's verified training seeds.
Plan that answers this audit: [CONTENDER_PLAN.md](CONTENDER_PLAN.md). Severity: **S1** a wrong answer can reach a
user, **S2** a right answer is refused or unreachable, **S3** hygiene that will cost us later.

## What is already strong (keep it)

* **The result record.** 211 of the 215 tools that ran return the same record: `value`, `units`, `source`,
  `uncertainty`, `provenance_tag`, `formula`, `notes`, `inputs`. That is Wolfram-grade provenance on almost every
  answer, and the `formula` field is a ready "show the formula" pod.
* **Missing-parameter errors name the parameter** (151 of 151 tools with required inputs).
* **Lookups explain a miss**: `density("unobtainium")` says the key is unknown and lists keys that exist; aliases
  such as `H2O` and `aluminium` resolve.
* **The material resolver** (`field/interface/resolve.py::material_profile`) returns `MaterialFact`s with value,
  source, licence and confidence. That is the right shape for an empirical database.
* **The n-body engine** reaches km-level parity with JPL DE440 over years (Mars 0.84 km, Moon 0.44 km at +5 yr),
  and `fast_ephemeris.where()` answers "where is Mars" in under a millisecond with a measured uncertainty.
* **Tests**: `sigma_ground/mcp` + `sigma_ground/inventory` + layering: 400 passed, 2 failed (below), 3 min.

## Findings

### S1 — wrong answers that look right

1. **Math notation is read silently wrong.** `differentiate_expr` on 20 common ways of writing an expression:
   9 right, **6 wrong with a confident answer**, 5 failed. Wrong: `x²` → 0, `x³ + 1` → 0, `√x` → 0, `πx` → 0
   (Python folds `x²` into a variable named `x2`), `e^x` and `e**x` → `e**x*log(e)` (e read as a variable, not
   Euler's number). Failed with `value: null`: `2x`, `sin x`, `3x^2 + 2x`, `(x+1)(x-1)`, `|x|`.
   `compute_limit` at `point="infinity"` returns `1/infinity`; at `"∞"` it returns `1/∞`; only `"oo"` works.
   Cause: one `sympy.sympify(..., convert_xor=True)` call (`mcp/tools/symbolic.py`, `mathx.py`) with no implicit
   multiplication, no unicode handling, no constant conventions, and no check that the result uses only the
   declared variables. Every expression-taking tool shares it.
2. **The same material has two values inside Mentat.** Mentat's lookup tools read a 35-entry table in
   `mcp/tools/materials.py`; Materia, the resolver and the physics modules read other tables. Same key, different
   numbers: concrete 2400 vs 2300 kg/m³, rubber 1100 vs 920, oak 770 vs 700, iron 7874 (CRC, 20 °C) vs 7870
   (NASA TPSX, condition "unknown"). The answer depends on which path a question takes.
3. **The live Mentat is not master.** The `skippytel-mentat` container (what Claude's `mcp__mentat__*` tools and
   SkippyTel call) is a stale copy: `mcp/tools/symbolic.py` differs from master (it lacks the 2026-10-10
   reserved-name fix), `materia/scenarios.py` differs, `field/interface/fast_ephemeris.py` is missing. There is
   no version stamp in any result, so nobody can tell.

### S2 — right answers we refuse or cannot reach

4. **No quantity with units is accepted.** 0 of 92 unit-bearing numeric parameters accept `"54 km/h"`,
   `"150 cm"`, `"2.4 GHz"` (pydantic rejects the string). The caller must convert, which is where the router's
   unit slips came from. `pint` is already a dependency (`convert_units`, `parse_quantity`).
5. **Required parameters that need not be.** `integrate_expr("x**2")` is refused for want of `variable` when
   the expression has one free symbol. 231 of 545 parameters are required; the defaults design
   ([DEFAULTS_AND_RANGES_DESIGN.md](DEFAULTS_AND_RANGES_DESIGN.md)) needs most physics ones optional with a
   disclosed default.
6. **Common materials are unknown to the lookup tools.** `density("steel")`, `"stainless steel"`, `"copper wire"`
   → `null`. The resolver knows steel; quarksum's `build_quick_structure` does not (KeyError for 35 of 61 common
   names: steel, stainless steel, rubber, plastic, wood, paper, sand, salt, PTFE, nylon, olive oil, milk ...). Three
   vocabularies, none covers the others: lookup table 35, resolver 42 of 61, quarksum 26 of 61.
7. **Physics we built but Mentat cannot reach.** Mentat's own coverage test fails on 13 unwired functions: dense
   matter (`white_dwarf_mass_radius`, `white_dwarf_structure`, `polytrope_structure`, `tov.is_stable_branch`,
   `eos.first_law_error`) and the shoot-out's models (`lunar_figure.*`, `asteroid_perturbers.propagate_to`). Nor is
   any ephemeris exposed: no tool answers "where is Mars", "how far is the Moon tonight", "when is the next
   opposition", although `fast_ephemeris.where()` and the DE440-parity N-body engine (1PN/2PN/EIH relativity,
   J2–J4, tides, radiation pressure, GW damping, the Moon's C22 figure, asteroid perturbers) exist.
8. **Two engines answer physics.** 232 MCP tools and 71 Materia scenarios (behind the front door) overlap
   (orbital velocity, material profile, ...) with different data sources (finding 2).

### S3 — contract hygiene

9. **Parameters are undocumented.** 477 of 545 parameters are never named in their tool's description; 217 of
   232 tools have no `Args` section; 200 have no example. The model sees a name and a type, nothing about units,
   ranges or format.
10. **Names disagree.** 34 different names for "a key into a table" (`material`, `material_key`, `material_1`,
    `material_key_1`, `mat_a`, `mat_p`, `metal_a`, `metal_key`, `sc_key`, `semiconductor_key`, ...); temperatures as
    `t_hot` and `t_hot_k`; `molar_mass_kg` and `molar_mass_kg_per_mol`. 107 numeric parameters carry no unit in the
    name (some correctly dimensionless: `order`, `n1`).
11. **Errors are not uniform.** Validation errors arrive as pydantic text with a URL and no hint of the expected
    unit or format; tool-level failures arrive as an ordinary result with `value: null` and the reason in `notes`.
    A client cannot tell "no answer" from "an answer of null" without reading prose.
12. **Overlapping tools to review** (not yet proven duplicates): `rydberg_hydrogen_wavelength` /
    `hydrogen_emission_wavelength`; `de_broglie_wavelength` / `de_broglie_from_kinetic_energy`; `relativistic_energy`
    / `relativistic_energy_analysis`; `orbital_velocity` / `circular_orbit_velocity`; four `projectile_*` tools /
    `projectile_analysis` / `procedure_projectile_trajectory`. Similar tools are a known cause of wrong-tool errors
    in small models (ToolScan).
13. **Constants mostly single-sourced, standard gravity is not.** `field/constants.py` has no standard gravity;
    9.80665 is defined separately in about 15 modules. `G` is copied in `inventory/core/sigma.py` (twice) and
    `dynamics/seeding.py`. All copies agree (exact values), so this is hygiene, not a wrong number.
14. **Layering violation on master.** `materia/railgun.py` (tier 3) imports `mcp.tools.plasma_em` (tier 4);
    `tests/test_layering.py` fails (introduced 2026-10-11, ff2de65).
15. **Stripped theory still on the public face.** The manifest's EXTENDED tier is still labelled "SSBM
    (Scale-Shifted Baryonic Matter) theoretical layer" and serves `entanglements_to_pop_bubble`,
    `holographic_matching_mass`, `baryon_vs_disc`; "SSBM" appears in 32 code files (21 lines in `mcp/`); the n-body
    toggles call textbook GR "pending SSBM σ-field re-derivation"; PROJECT_GOAL.md lists "SSBM scale dynamics".
    Under the 2026-10-11 data-alone rule these need the Captain's call (remove, or reword as plain arithmetic).
16. **The mission statement predates the goal.** PROJECT_GOAL.md: "Not a Wolfram Alpha replacement on
    symbolic-math breadth". The Captain's goal (2026-10-13): a local Wolfram contender in physics **and math**.

## How it was measured (rerun it)

Scripts in the session scratchpad (`audit/mentat_census.py`, `audit/probe2.py`), to be moved into
`tools/mentat_contract_census.py` as the first contract test: schema census of all 232 tools, replay of one
verified call per tool (215 ran, 5 had no example, 11 stateful or long-running were schema-only, 1 replay failed:
`procedure_black_hole_profile` with a malformed v1 example), probes P0 (no arguments), P1 (a non-number), P2 (a
quantity string), the 20-way notation table, and the material cross-checks. Tests:
`pytest sigma_ground/mcp sigma_ground/inventory tests/test_layering.py --ignore=sigma_ground/mcp/benchmark`.

## Follow-up 2026-10-13/14 — W1, the math kernel (finding F1 closed)

One parser, `sigma_ground/mcp/tools/mathparse.py`, now reads the expression of every math tool (`symbolic.py`, `mathx.py`). Its reading rules and
safety rules are in its docstring; 240 tests in `sigma_ground/mcp/tools/test_mathparse.py` (a 150-row notation table checked against plain
`sympify` of a canonical spelling, the audit's cases, refusals, declared-variable overrides, and a property test over the generator's four
renderings).

| | before (audit) | after |
|---|---|---|
| `differentiate_expr` on 20 ordinary writings | 9 right, 6 silently wrong, 5 null | 20 right (`|x|` gives sign(x), the real derivative) |
| notation table through the server (100 x-only rows) | not measured | 100 right, 0 wrong, 0 refused |
| `compute_limit` at "infinity" / "∞" | `1/infinity`, `1/∞` | 0 |
| `variable` required by the math tools | yes (`integrate_expr("x**2")` refused) | optional when the expression has one free symbol; the choice is disclosed |
| result says how it read the input | no | `interpreted_as` (e.g. `d/dx [3*x**2 + 2*x]`); disclosures in `notes` (e read as Euler's number, a treated as a constant, ax read as a·x) |
| refusals | silent `null` or a wrong answer | `value: null` + the reason + a corrected example; unknown functions, quotes, attribute access, runaway powers refused |
| regression check | | all 3,340 stored seed results of the 16 math tools replay identically |

The ratchet that keeps it so: `tools/mentat_contract_census.py` (the audit's census as a tool) and `sigma_ground/mcp/test_contract_census.py`, which holds
the numbers to `misc/contract_budget.json` (a number may only go down). Still open from the audit: F2–F16 (W2 contract, W3 materials, W4 sky tools).
