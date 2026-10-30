# Folded chains: a body's name straight into the formula tool (2026-10-29)

The Captain's ask: common combinations of tool calls become static code in sigma-ground, so Mentat's 4B router (mentat-router-v31-4b) picks ONE tool
with plain arguments instead of planning a chain. Companion to misc/STATIC_INTENTS_PLAN.md (that one is about the suite misses; this one is about
chains and lookups the model does in its head, which mostly answer right today but take two steps or a remembered number).

## What was measured

**Mentat's research log** (skippy-core `/var/lib/skippy-core/mentat-research/2026-10.jsonl`, 9,860 `/ask` rows, read 2026-10-29; deduplicated to the last
answer per question). The cascade rows record no steps, so chains are counted over router answers.

| set | distinct questions | with tool steps | multi-step chains |
|---|---|---|---|
| not tagged as a benchmark run (no `via`, portal, claude-check, mentat_call.sh) | 521 | 422 | **12** |
| everything | 534 | 433 | 15 |

The 12 chains in the untagged set:

| chain | count |
|---|---|
| `solar_system_body > escape_velocity` | **6** |
| `solar_system_body > schwarzschild_radius` | 1 |
| `em_wave_wavelength > em_wave_wavelength` | 1 |
| `ohms_law_voltage > electrical_power` | 1 |
| `gravitational_potential_energy > joules_to_eV` | 1 |
| `fission_energy_release > joules_to_TNT` | 1 |
| `tnt_to_joules > energy_to_mass` | 1 |

**Benchmark runs** (`sigma_ground/mcp/benchmark/results/*_run.json`): the live suite (`live_mentat_run.json`) has 8 chains in 71 tool answers, 3 of them
`solar_system_body > escape_velocity`, all 8 scored correct. The model-only router runs show 16 of 143, adding `solar_system_body > orbital_velocity` (2),
`named_star > convert_units > light_travel_time` (2), `solar_system_body > schwarzschild_radius` (1). **Lookups done in the head:** 15 single calls across
the runs typed a named body's mass in from memory (escape_velocity 11, schwarzschild_radius 4; e.g. `escape_velocity(mass_kg=1.898e27, radius_m=6.995e7)`
for "escape velocity from Jupiter's surface"), and 9 more rows on gravitational_time_dilation for Earth (all the GPS altitude question, one with a radius of 4,000 km: inside the Earth).
Materials: no single call typed a material property in from memory (the material tools already take `material`).

**Where the chain came from:** the v3 recipes `body_escape_velocity`, `body_schwarzschild` and `body_hawking` (69 seeds in local-cache/router_recipes_v3)
teach exactly `solar_system_body > <formula>` with `$1.mass_kg`. The router does what it was taught.

## What was folded

`body_name` (an existing parameter name; with `altitude_km`, also existing, so the vocabulary budget in test_param_names.py is unchanged at 291) on:

- `escape_velocity(body_name="mars")`: the surface; `altitude_km` for a height above it; `radius_m` (from the centre) still wins when given.
- `schwarzschild_radius`, `isco_radius`, `photon_sphere_radius`, `hawking_temperature`, `hawking_evaporation_time` (If One Then All: every GR tool that takes only a mass).
- `gravitational_time_dilation`, `gravitational_redshift` (mass and radius, like escape_velocity).

The lookup is `mcp/tools/body_lookup.py`: the body catalogue body_constants already reads (`field/interface/body_data.py`: Sun, planets, 40+ moons, DE440's
asteroids), JPL's GM exact to its published digits, the radius with its own source. The answer's `source` names the GM's source (DE440 header, MAR097,
JUP365, ...) and the radius's; `provenance_tag` is DERIVED; `inputs` echo the body, the mass (GM / G) and r. A mass typed alongside a body is not used,
and the notes say so. Without `body_name` every tool is exactly what it was.

Checked against the suite's ground truth (the tool's own numbers): Jupiter 59,532 m/s against 59,530 expected (the chain gave 60,202: Fact-Sheet mean
radius, still inside the 2% tolerance), Earth 11,186.1 against 11,186, the Sun 617,675 against 617,700.

**Not folded, and why:**
- The unit-conversion tails (`> joules_to_TNT`, `> energy_to_mass`, `> joules_to_eV`, `ohms_law_voltage > electrical_power`): once each, all answered right,
  and an `output_unit` argument would be a new parameter name on many tools. Revisit if the log shows them more often.
- `solar_system_body > orbital_velocity` ("how fast does Mars go around the Sun", v3 recipe `planet_orbit_speed`, 2 router rows): needs a name for the
  ORBITING body on orbital_velocity, a new concept in the vocabulary. Next candidate.
- `named_star > convert_units > light_travel_time` (2 rows): the cascade's astro classifier already answers most of these; light_travel_time taking a
  star or body name is the fold (STATIC_INTENTS_PLAN.md item C is the same tool).

## Router training data

- New recipes: `sigma_ground/mcp/benchmark/router_recipes_fold.py` (six recipes, one call each, registered in test_router_recipes.py). Seeds written:
  **`local-cache/router_fold`** (226 seeds, all one call: escape 44 + 50 at altitude, Schwarzschild 45, photon sphere 30, Hawking 25, time dilation 32).
  The next router run adds it to `SEED_DIRS` AFTER the old seed folders (the rewrite cache is indexed by seed).
- `refold()` relabels the 69 old v3 chain seeds in place as the one folded call, with the value recomputed by the folded tool; `tools/build_router_v2.py`
  applies it as it reads the seeds, so the next dataset never teaches both forms. Question, group and index are unchanged, so rewrites.jsonl still lines up.

## Before / after

Not measurable until the next router training: mentat-router-v31-4b has never seen `body_name` on these tools and keeps planning the chain, which still
works (both forms are served). Expected after training: the 6 + 1 chains in the log and the 3 on the live suite become one call, and the 15 memorised
masses become a name. The score on the live suite should not move (all 8 chains were already right); what changes is one fewer step per question and
no remembered number. Deployment: Mentat on skippy-core serves these once SkippyTel's sigma-ground pin moves past this commit.
