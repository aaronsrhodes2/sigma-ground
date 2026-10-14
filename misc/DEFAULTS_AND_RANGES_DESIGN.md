# Defaults and ranges — answer first, from a table, never ask without answering

Design, 2026-10-13. Written from the Captain's rule and the evidence below; **not yet built** beyond the
pieces marked done. Review the decisions at the end before any of it touches the live server.

## The rule (the Captain, 2026-10-13)

1. Every parameter of every tool has **bounds**.
2. Every value the prompt does not give is **filled from a defaults-and-ranges table, relative to the
   experiment**, and the system answers with it. Example: "throw a ball" with no size, material or speed
   uses a ball's typical size, mass, material and throw speed, and gives the answer. It does not ask
   questions without giving an answer.

## Why the evidence says this must be general, not a fix for six questions

* Router-path census, all 232 tools, one gold call each (`tools/tool_census.py shapes`): 113 return a
  number and are read; 73 of 80 dicts are read; **every symbolic string (14 tools), numeric string (3),
  list (7) and procedure result (5) is unreadable.** That is the whole non-numeric math family: any
  derivative, integral, factoring, matrix inverse or product, limit, sum, determinant, root list, for
  any question. The benchmark's six math questions are the visible tip. Fix: `router_results.present`
  (done, tested, the cascade's own conversion).
* Parameter coverage (`tool_census.py coverage`): 99 of 545 parameters have under 10 training examples
  and 98 have none. The data only ever showed the required ones. Every optional math parameter is blank:
  `integrate_expr.lower/upper`, `compute_limit.point/direction/variable`, `gradient/divergence/curl.variables`,
  `laplace/fourier` variables, and many material keys. A different question that needs one of them fails
  exactly as the benchmark's did.
* The training labels themselves are clean (`tool_census.py lint`: 41,224 calls, no unknown tool, unknown
  parameter, missing required parameter or wrong type). The 4B's own plans are not (11 invented
  parameters, 9 missing required arguments, 3 nonexistent tools in 150 questions). The gap is coverage.
* 231 of 545 parameters, in 153 of 232 tools, are **required with no default**. "Field required" is what
  a question that leaves one out gets. The Captain's rule removes that class of failure at its source.

## Design

**One table, in Mentat, never in the model.** PLATINUM §21: a physical number comes from Mentat. The model
never writes a default from memory. It either leaves the parameter out (the tool fills it) or looks it up
and passes the field on (`"$1.mass_kg"`, the plan format v3 already has).

1. `param_specs`: for each tool parameter, `lo`, `hi`, `unit`, and a `default` that is either a constant or a
   function of the other arguments ("relative to the experiment"), with a one-line derivation (Golden Rule 1:
   derive, don't assume). Enforced at the server boundary with a precise out-of-range message (Rule 2:
   Design by Contract). Applied generically, by wrapping the 232 `@server.tool()` functions, so no tool body
   changes: a parameter with a spec becomes optional in the schema and is filled before the call.
2. `typical_values(kind)`: a scenario table (generalising `object_library.py`, which already holds canonical
   sizes and densities for everyday objects) giving `lo / typical / hi` per property for an object or
   experiment kind. "A baseball" -> mass 0.145 kg (0.142-0.149), throw speed 28 m/s (10-45), and so on.
   Ranges can be relative: a person's throw speed falls with the mass thrown (a roughly constant energy
   budget), so a bowling ball is not thrown at a baseball's speed.
3. **Disclosure** (Golden Rule 4: no hidden assumptions). Every result lists what was filled in:
   `assumed: [{param, value, range, why}]`, and the answer text says so ("assuming a 0.145 kg baseball thrown
   at 28 m/s; across the plausible range the answer spans A to B"). The Captain's wording was that the
   system should not ask without answering; it can still invite a correction after the answer.
4. **Sensitivity.** Where cheap, report the answer at `lo`, `typical` and `hi` of each assumed value. That
   tells the user how much the guess mattered, and it handles the next point.
5. **Carve-out kept from `object_library.py`.** Where a missing value is *decisive* (a fissile sphere's size
   decides inert versus supercritical), a single guess would silently imply one regime. The answer is then the
   regime split across the range ("below about 10 kg bare it cannot go critical; above it, it can"), which
   is still an answer, not a question.

## Derive before you default: select a mass and a material (the Captain, 2026-10-13)

Most of the "missing" values are not independent. Pick a mass and a material and nearly everything about a
body follows, so the table of priors shrinks to what cannot be derived (a throw speed, a launch angle, an
ambient temperature) and everything else comes from a closure (Golden Rule 1: derive, don't assume). Checked
against the code, for a 0.145 kg iron object:

* **quarksum** (`sigma_ground.inventory`, `build_quick_structure(material, mass_kg)`) fills the composition:
  4.07e25 protons, 4.07e25 electrons, 8.76e25 baryons, quark counts, `GM` = 9.68e-12 m^3/s^2, the mass defect,
  mass closure to 1e-5. It does **not** give geometry: `radius_m`, `v_escape_m_s` and `sigma_surface` come
  back `None` (a radius exists only for the nine default loads in `defaults.py`).
* **the materials tools** give density (7874 kg/m^3), Young's modulus (211 GPa), melting point (1811 K),
  refractive index, and the rest of the per-material tables.
* **the missing link is geometry**: V = m/rho, then for a sphere r = (3V/4pi)^(1/3) = 1.64 cm and
  cross-section 8.43e-4 m^2 for this object. Nothing in Mentat composes these today, so a question about "a
  0.145 kg iron ball" cannot get its radius, area, moment of inertia (2/5 m r^2), heat capacity (m c_p), or drag
  area from one place.

So the first thing to build is **`object_properties(material, mass_kg, shape="sphere")`**: geometry from the
density, the material's own tables, quarksum's census, and (setting the derived radius) quarksum's own
`v_escape_m_s` and `GM`. The plan looks the object up once and passes fields on (`"$1.radius_m"`), which the
v3 plan format already supports, and the server fills a tool's missing area, radius, volume, specific heat,
modulus and the like from the same object. The scenario prior table (`typical_values`) then only has to say
what object a scenario *is*: "a thrown ball" -> a material, a mass range, a speed range, an angle.

What this does not cover, so it is not over-claimed: shapes other than the sphere, box and cylinder need a
declared shape factor; composites and layered bodies need the layered structures quarksum already has; and
a question that mentions neither a mass nor a material still needs a scenario prior (a typical object) before
anything can be derived.

## The materials database is the empirical reference (the Captain, 2026-10-13)

"We have a materials database as well; I want to make sure that is referenceable as empirical properties."
A property of a material that was measured (a refractive index, a melting or boiling point, a density, a
Young's modulus, a band gap, a resistivity) lives in that database and **nowhere else**: not in the model's
memory, not in a training label, not in a default table. A plan gets the value by calling the lookup tool and
passing the result on (`"$1"`, or `"$1.field"` for a record), so the number that reaches the physics tool
is the table's number and the answer can name the table.

* **Precedence.** (1) a value the user gave; (2) a measured value from the database, by lookup; (3) a value
  derived from (2) (V = m/rho, then r); (4) a scenario prior, marked `assumed` in the answer. Only (4) is a
  judgement call, and only (4) needs the `needs review` mark.
* **Provenance** gains a kind: `db` (table and key, for example `refractive_index: glass`). The answer says
  "using the database's refractive index for glass" the way it says "assuming 0.145 kg".
* **v3.1 trains this for seven chains** (`router_recipes_v31.py`, family `db`, 255 training records): critical
  angle and Snell's law between two media (two `refractive_index` lookups), wavelength in a medium, peak
  blackbody wavelength at a melting point, thermal energy per molecule at a boiling point, Carnot efficiency
  between two boiling points, and floating in a fluid (the fluid's `density` into `buoyancy_analysis`). In
  every one the question names materials and no number, and the plan contains no measured value. The
  recipes run against the real server: a material key the database does not know yields an empty record
  and the example is dropped (the generator will not label a plan the server cannot answer).
* **Still to do.** The same rule for the rest of the property tables (moduli, resistivities, band gaps) and
  a lint that fails any training plan with a literal in a slot a lookup tool fills (a refractive index
  typed as `1.33`), so a measured value from memory cannot be taught by accident.
* **Why it matters for grounding.** The model cannot suggest a far-out value for a property the table
  holds: it has no number to write, only a lookup to call.

## Grounded in reality (the Captain's reading of why this matters)

Today the groundedness gate says an *answer* came from a tool. This rule lets it say the same of the
*inputs*, which is what keeps a model from "suggesting far-out values" or from asking a string of questions
instead of answering. Every numeric argument in a plan gets a provenance, checked after the model writes it:

| provenance | meaning | allowed |
|---|---|---|
| `user` | the number is in the question (after unit conversion) | yes |
| `table` | the parameter was left out, or looked up, and Mentat filled it from the defaults-and-ranges table (with its source and range) | yes, disclosed |
| `step` | it is a `"$n"` / `"$n.field"` reference to an earlier result | yes |
| `model` | a literal the model wrote that is neither in the question nor from the table | **flagged**; rejected under the strict profile |

Plus the bounds: a value outside its range is refused with the range in the message, whoever supplied it
(so a typo, a unit slip or a made-up number never silently produces a confident wrong answer). The two
together are the grounded-in-reality property: nothing the answer rests on came from the model's memory
or imagination. The `model` check is testable now, against the plans in the existing runs and training
data (`tools/tool_census.py` can count them), and it is the same comparison `check_rewrite` already makes
when it builds the training set.

**Measured, on the 4B's cascade-off benchmark plans (150 plans):** 19 of 162 numeric arguments (12%), in 16
plans, are not traceable to the question text. They are mostly things written from memory: the electron mass
(9.109e-31), 273.15 K and 298.15 K for "0 / 25 Celsius", 101325 Pa for "atmospheric", g = 9.81 and a copper
density (8960). One is plainly wrong: `modern_001`, "0.9 times the speed of light", was written as
`velocity_m_s = 2700000.0`, a factor of 100 below 0.9c. (The 1.7B: 15 of 161.) That is the "far-out value" the
provenance check exists to catch, and it is a unit/constant slip, not a physics one.

**Quote, don't compute.** The most direct cure is to remove the model from the arithmetic altogether. A
quantity parameter may carry the user's own words as a string ("0.9 c", "20 degrees Celsius", "1 atm",
"10 years", "an electron") and Mentat converts it into the parameter's unit with its own constants
(`parse_quantity` and `convert_units` already exist, on pint). The model copies the quantity from the
question; it never converts a unit or recalls a constant. The `unit` in each spec says what to convert to.
This also makes the training data simpler (copy the quantity text) and the provenance check exact (the
string appears in the question, or it is `table`).

## What it changes for the router

* The system prompt says "when a required input is missing, ask (`request_clarification`). Never invent it."
  The 773 clarification examples in the training set teach exactly that. Under the new rule they must be
  re-labelled: leave the parameter out, or look the typical value up. `request_clarification` shrinks to the
  cases with no answerable core (no quantity, no tool).
* New training data: each tool called with its optional parameters (so `lower`/`upper`, `point`, `direction`
  are shown), questions that leave values out and plans that omit or look them up, and the assumption shown
  in the answer. The lint (`tool_census.py lint`) gains a bounds check once specs exist.
* Retraining is required: the labels change. The 4B is about 2.6 h, the 8B about 5 h.

## Done now / next / decisions

Done and tested (no production code touched): `tools/tool_census.py` (lint, coverage, signatures, shapes),
`router_results.py` (reads every string/list result; names a rejection by its cause).
Next, after the v3 sweep finishes: the `structured_results` switch in the harness (opt-in, so every earlier
run stays reproducible) and a cascade-off re-run of all three sizes; then `param_specs` with the math family
complete first (its content parameters, the expression or matrix, are intrinsic; its auxiliary ones
`variable`, `lower`, `upper`, `point`, `direction`, `order` get specs), then the 20 most-used physics tools,
then the rest as a reviewed table; then `typical_values`.

Decisions for the Captain:
1. **Retire clarification for missing inputs** (answer with defaults and say what was assumed)? This also
   changes the router prompt and the 773 examples.
2. **Server-side fill changes the schema every MCP client sees** (formerly required parameters become
   optional). Acceptable for the live Mentat?
3. **Defaults are physics judgement calls.** Each needs a derivation. I will draft the table and mark every
   entry `needs review` until you have seen it; the answer text will carry `assumed:` until then.
4. **The decisive-value carve-out** (answer as a regime split instead of one guess). Agreed?
5. **Retrain scope** once the data is ready: 4B only first (about 2.6 h), or all three sizes.
