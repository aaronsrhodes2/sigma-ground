# Static intents: what the 30 suite misses say should be code, not model (2026-10-29)

The Captain's thought (2026-10-29): fold the common combinations and orders of tool calls that answer the suites into sigma-ground as static
code, so the 4B router has less "magic" to do. This plan checks that thought against the replay rows of 2026-10-27
(`misc/audit_2026-10-27/cascade/replay_*_CBm.json`, the deployed code path: cascade ahead of the router, maths stage off) and the live
maths rows (`math329_live_final_rows.json`).

## What the rows say

- **No chains exist.** Every plan on both suites is one tool call: physics `{1: 150}`, maths `{1: 329}`. The router's job today is to pick one
  tool of 224 and fill its arguments. There are no orders of calls to fold.
- **The folding already happened once.** 75 of the 132 physics answers come from the deterministic cascade (regex and semantic classifiers), 71
  from the router, 1 from the position rule. The cascade IS static code ahead of the model, and it is where the last +24 came from.
- **Of the 30 misses, 7 are a right tool call whose answer is a whole record** (a body or an element) when one field was asked for; 9 are scorer
  or suite defects (right answer, no credit); only 2 are genuine two-step chains. The lever is intent and field selection, not sequencing.

## The 30 misses, grouped by cause

| group | ids | what happened | static fix | where | gain |
|---|---|---|---|---|---|
| A. one field of a record | mech_adv_005 (surface gravity of Mars), astro_002 (Sun's surface temperature), astro_011 (Jupiter's mass), atom_001/006 (atomic number), atom_002/008 (atomic mass) | router right (`solar_system_body('mars')`, `element_atomic_data('Au')`), answer is the dict, `extracted_value` None | **field pick**: after a record tool, a cue table (`surface gravity` → `surface_gravity_m_s2`, `temperature` → `surface_temperature_k`, `atomic number` → `atomic_number`, `atomic mass` → `atomic_mass_u`, `mass` → `mass_kg`, `radius`, `density`, `escape velocity`, ...) selects the field, units from the record's own units; the whole record stays in `inputs`. Pure code, no model; applies to every record tool the same way (If One Then All) | sigma-ground `mcp/front_door.py` (`cascade_answer`'s post-step, so harness and live share it) | **+7** |
| B. right answer, scorer blind | math_001 ([-2, 2]), math_002 (`1/3`), math_004 ([1, 2, 3]), math_005 (`pi`) | the physics scorer's `_values_match` reads neither a list nor a sympy string; the maths oracle does both | `/ask` returns `value_float` beside a symbolic `value` when it evaluates to a number (`1/3` → 0.333…, `pi` → 3.14159); the physics scorer compares lists as sets like the maths oracle | sigma-ground `mcp/provenance.py` (the record) + `benchmark/score.py` | **+4** (honest label: scorer/record fix, not a physics gain) |
| C. a real two-step chain | mech_adv_004 (light Sun→Earth), astro_007 (light across 1 parsec) | distance ÷ c; the router converted parsec to metres and stopped | one tool `light_travel_time(distance, units)`: `distance / C`, answer in s and in years, source IAU + CODATA c. One tool is honest here: a composite of two tools is still one formula | sigma-ground `mcp/tools/astronomy.py` + a cascade cue (`how long does light take`, `light travel time`) | **+2** |
| D. inverse solve | em_intro_012, em_adv_004 (plate area from C, d, εr), gr_008 (GPS altitude from 38 µs/day) | the tool wants `area_m2`; the question gives C and asks for area | **solve for the missing parameter**: a formula tool whose one missing required parameter is the asked quantity solves its formula with sympy for it (`A = C d / (ε0 εr)`); the contract already knows the required set and the formula string. gr_008 is not a single formula (GR + SR terms); leave it | sigma-ground `mcp/contract.py` (`wrap_function`) | **+2** (+1 hard, not planned) |
| E. extremum over a group | atom_004 (highest noble-gas first ionisation energy: He 24.587, router picked Ne) | needs argmax over a set, not a lookup | `extreme_in_group(property, group, max|min)` over the element table; cue `highest|lowest|largest|smallest <property> <group>` | sigma-ground `mcp/tools/atomic.py` + cascade cue | **+1** |
| F. gradient dimension | msuite_0211, 0212 | 2-component gradient; suite expects 3 with ∂/∂z = 0 | `gradient` returns ∇ in R³ whenever the variables are a subset of {x, y, z} (the suite's convention); say the convention in `notes` | sigma-ground `mcp/tools/mathx.py` | **+2** |
| G. primes in ODEs | msuite_0283 (`x'' - 4*x' + 3*x`), 0289/0291 (`f' + 2f`), 0285 (`y(t)` solved in x) | `f'` is read by Python as an f-string; `x''` as an unterminated string; the independent variable named in `y(t)` is dropped | the maths kernel rewrites `f'`, `f''` → `Derivative(f(t), t)` before any parse, reads `<name>(<var>)` as the function and its variable, and `2f` as `2*f` (implicit multiplication is already a kernel rule); nothing reaches Python's tokenizer | sigma-ground `mcp/mathparse.py` + `solve_ode` | **+4** |
| H. no ground truth | msuite_0304–0308 | `expected: None`; the five answers are right (cos²−sin², 3x², 1, 26/3, 2x³+2x²) | write the five expected values into the suite | the suite file | **+5** (count only) |
| I. a system, not an equation | msuite_0324 (`solve for x and y: 2x + y = 5, x - y = 1`) | router sent two equations to `solve_equation` | cue: two or more equations joined by `,`/`and` with `solve for <a> and <b>` → `solve_linear_system`; the alias also accepted by `solve_equation` (a list of equations hands over) | sigma-ground cascade cue + `mathx.py` | **+1** |
| left to the model | mech_intro_015 (ground-relative speed at GEO = 0: knowledge, a correct-looking 3075 m/s is the trap), gr_008 | not one formula, not one field | — | — | 0 |

If every row lands: physics 132 → **148 of 150** (A+B+C+D+E = 16; B is the scorer seeing what is already right); maths 317 → **329 of 329**
(F+G+I = 7 by code, H = 5 by ground truth). None of it needs the router retrained, and none of it adds a second call per question.

## Done

- **A, field pick** (2026-10-29): sigma-ground c70cea6 (`pick_field`, `extracted_field`/`extracted_by` on the harness record, tests) +
  SkippyTel ad71d88 (the router stage presents a cued field, pin bumped). The harness had been picking the field for scoring all along; the
  live stage was discarding it. Paired replay, 150 physics, deployed code path: **132 -> 139, +7 -0**, exactly the seven rows of group A.
  Live confirmation: the next daily live rows after the deploy.

- **F, gradient dimension** (2026-10-29): NOT a tool change. The kernel infers the field's own variables since 489ec0c (2026-10-14, a
  deliberate decision, with a test saying "not the old default x,y,z"; Wolfram returns two components for a two-variable function too). The
  suite's two expected 3-vectors were generated with `variables='x,y,z'` before that change. msuite_0211/0212 move to group H (ground truth).
- **G, primes** (2026-10-29, sigma-ground): `solve_ode` reads the function the primes name (`f' + 2*f` is about f), a function named like the
  variable is a function of t (`x'' - 4*x' + 3*x` -> x(t)), and `func="y(t)"` names the variable. Paired replay on the box: **317 -> 320,
  +3 -0** (cda74e4). msuite_0285 stays: the question says y(t), the router passes no variable, the tool cannot know t.
- **I, a system of equations** (2026-10-29): `solve_equation` reads `2*x + y = 5, x - y = 1` (commas, `;` or `and` at bracket depth 0 with
  `=` on both sides) as a system and answers one value per unknown; `param_aliases` renames the router's `equations`/`variables` to the
  singular parameters, so the vocabulary budget (291) is unchanged.
- **C, light travel time** (2026-10-29): the astro classifier answers "how long does light take to travel 1 parsec / reach Earth from the
  Sun" with `light_travel_time` over the stated distance (years for parsec and light-year distances, seconds otherwise); a named star keeps
  its own rule; negative tests keep it narrow.
- **H, ground truth** (2026-10-29): seven suite rows amended in `local-cache/router_math_suite.json` (now tracked: the suite's ground truth
  was an untracked file), each with the reason in its `oracle` field and an `amended` record at the top; the five wild rows had `result:
  None` because the tool refused those arguments when the suite was written; the two gradient rows per group F. The generator (`router_math.py
  p_grad`) no longer writes `variables="x,y,z"` into a held-out call whose question names no variables.
- **Paired, one engine** (2026-10-29, local Ollama with mentat-router-v31-4b, because the box's shim stopped serving the trained router that
  evening: `mentat-router-v31-4b` 404, `mentat-router` answering as a chat model, the live interpreter reading `router qwen3-8b-stock`):
  maths **320 -> 328 of 329, +8 -0** (F, H, I); physics **137 -> 139, +2 -0** (C). `audit_replay.py` now honours `OLLAMA_URL`. The box
  and the local engine differ by two physics rows (139 vs 137 before) on the same code, so a pair is only a pair on one engine.

- **D, solve for the missing parameter** (2026-10-30, `mcp/contract.py solve_for_missing`): when a call has exactly one required parameter
  missing and exactly one argument the tool has no parameter for, and that argument's unit has the dimension of the tool's result, the
  tool is inverted numerically for the missing parameter (positive quantities, bracketed over decades, bisection) and the forward check
  is disclosed in the notes. No inverse formula is written by hand: the tool is the function. Works for every formula tool at once
  ("what height gives a 3 s fall" -> 44.1 m). Found on the way: the em_adv_004 ground truth was off by 1000 in its own arithmetic
  (A = 1e-4 * 1e-4 / (8.854e-12 * 100) = 11.29 m^2, not 0.0113); corrected with the reason in its note. Paired on the local engine:
  physics **139 -> 143, +4 -0**: em_intro_012 and em_adv_004 are D; mech_intro_010 and atom_003 flipped on router nondeterminism
  (mech_intro_010 was planned with an unasked `joules_to_eV` tail: 4.28e23 eV, scored right only because the scorer converts units;
  the unit-conversion tails are in misc/FOLDED_CHAINS.md "not folded"). Maths 328 unchanged. gr_008 stays (not one formula).

## Order of work

1. **A, field pick** (+7, a day). The largest gain, pure code, and it fixes the live dashboard too: the daily live rows count a dict as "not
   answered". Also the 2026-10-27 audit's item 4.
2. **G, primes** (+4, half a day) and **F, gradient convention** (+2, an hour): kernel rules, each with a notation test in `test_mathparse.py`.
3. **B, value_float + list scoring** (+4, half a day): the record carries the number the symbolic value evaluates to; the scorer reads it.
4. **H, the five ground truths** (+5, an hour, with the Wolfram trickle as the second opinion where it answers).
5. **C, light_travel_time** (+2, an hour) and **I, system cue** (+1, an hour).
6. **D, solve for the missing parameter** (+2, a day; a contract feature every formula tool inherits, so it is the one item here that
   generalises beyond the suite: "what height gives a 3 s fall", "what mass has a 2 km Schwarzschild radius").
7. **E, extremum** (+1, half a day).

## How each lands (the rule from the cascade work)

Paired replay before and after, per question, with `misc/audit_2026-10-27/cascade/replay_router.py` on this PC: a change ships only when it
gains and loses nothing (the maths regex stage lost 19 to "over the interval [a, b]" and was switched off for exactly this reason). Every new
cue gets a negative test: a question that must NOT match (for A: "how massive is Jupiter compared to Saturn" must not pick one field). Then the
pin bump in SkippyTel, then the daily live rows confirm it on the deployed `/ask`.

## What this does not touch

The adversarial set (8 of 18) is a refusal problem, not a selection one: the audit's item 5 (a router DECLINE becomes the answer). Conversation
mode is where chains will exist (apply a force, evolve, read a value); the suites do not exercise it, so nothing here speaks for it.
