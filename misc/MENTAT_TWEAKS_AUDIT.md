# Mentat tweaks audit — what stands in for training, and how the two are paired

2026-10-12. Scope: the benchmark harness and the Mentat MCP layer around it. There is no SKILL.md in
this repo; the "skill" is the hand-written code listed below. Every number here was measured from
the finished 150-question runs in `sigma_ground/mcp/benchmark/results/` or from the code; nothing is
estimated.

## The question

Before the router was fine-tuned, a modest model needed help, so rules were written to stand in
for it. Training was put off until the physics settled. Which rules should now be training data,
and how do we test a trained model **without** the rule that used to cover for it, so a result is
never "the model, plus a regex that hid its mistakes" and no rule is removed that nothing replaced?

## What is in the way (inventory)

| tweak (registry name) | where | what it does | measured |
|---|---|---|---|
| `cascade.*` (13 classifiers) + `cascade.semantic` | `*_classifier.py`, `interpreter_demo.py`, called in `run_sigma_ground._run_one_question` | answers a question by regex (or embedding match) **before the model is asked**; tool calls happen in-process, so the record shows `tool_calls: []` | 86 of 150 benchmark answers: thermo 11, modern 12, astro 10, new_tools 9, gr 9, cosmology 8, conversion 7, math 6, classical_intro 5, semantic 5, optics 4; **frontier, refusal, clarification fired 0 times**. 84 of the 86 were right |
| `hidden_tools` | `new_tools_classifier.HIDDEN_FROM_LLM` | 15 tools are left off the model's tool list | the v2 training data has **0 examples of any of the 15**; `orbital_raise_energy` is unreachable from the router (no classifier emits it either) |
| `param_aliases` | `param_aliases.PARAM_ALIASES` | renames a made-up parameter (`gravity_ms2` → `g_m_s2`) before the call | **fired 0 times** in the five router/chain runs |
| `body_name_chain` | `param_aliases.resolve_body_name_chain` | `body_name='sun'` → `solar_system_body` → `mass_kg` | **fired 0 times** |
| `pattern_hints` | `run_sigma_ground._PATTERN_HINTS` | question phrases → a domain, at the top of the chain-mode prompt | chain mode only; the router prompt has no index |
| `answer_key_hint` | `_extract_value_from_tool_calls(primary_tool_expected=…)` | chain-mode fallback prefers the tool the benchmark says is expected | chain mode only; a measurement-integrity issue (the answer key reaches the runtime path) |
| nudges, fallback extraction | chain loop | up to 2 "answer now" nudges; salvage a number from tool results | chain run: 8 nudged, 2 fallback; **the router has no loop, so it needs neither** |
| headline-field picker | `_FIELD_PRIORITY_BY_QUESTION_KEYWORD` | for a dict-valued lookup, the *question's wording* picks the field returned | decides what a "$1" is for a lookup; see "field references" below |
| groundedness gate | `groundedness.apply_gate` | every answer is tool-grounded or labelled `[Fitted due to incompetence]` | a trust feature, not a tweak: **kept, always on** |

### Findings worth a separate look

1. **Four classifiers do physics in the harness** (Platinum §21: Mentat does the physics).
   `astro_classifier` computes Wien's law, `sqrt(GM/r)`, `L·t` and pc→ly itself and returns a
   `result_override`; `optics_classifier` does `asin`, a diffraction angle and `331.4 + 0.6·T`;
   `classical_intro` does `sqrt(2KE/m)`; `modern` does `n·4.184e15`. Run alone over the benchmark they
   match 12 + 3 + 1 + 1 = **17 of the 150 questions** (the live cascade order hands a couple to earlier
   classifiers). Their constants are not Mentat's: the classifier uses a solar temperature of
   **5778 K; the library's Sun is 5772 K** (`solar_system_body('sun')`), so the harness and the
   library disagree about the same star.
2. **Param aliasing and the body-name chain never fired.** Removing them costs nothing on this
   benchmark; they only matter if a model invents names, which is a training question.
3. **A missing parameter is silent.** The MCP layer drops an unknown parameter and runs on the
   default (the free-fall-on-Moon question became free-fall-on-Earth that way). With aliasing off that
   would turn a model's slip into a plausible wrong number, so the new `strict_params` check makes it
   an error and every call records `unknown_params`.
4. **Five tools cannot be taught without training on a benchmark instance.** `age_of_universe`,
   `critical_density`, `eta_value_report`, `mond_a0_constant` and `holographic_matching_mass` take no inputs
   (bar `age_of_universe`'s mode switch), so a question for one of them has the same answer as the
   benchmark's question for it (the benchmark guard holds 4, 2, 2, 2 and 1 such instances; 30 phrasings of
   the last all came back as overlaps). Teaching a rephrasing would be training on
   the benchmark, which the ground rules forbid, so no recipe was written for them, `cascade.cosmology`
   stays for those four, and the pairing check lists them as *guard-blocked*, not as gaps.

## Decisions

| tweak | decision | what replaces it (training data, v3) |
|---|---|---|
| `hidden_tools` | **train, then remove** | 20 `hidden` recipes: one or more per hidden tool (14 of the 15; `holographic_matching_mass` is guard-blocked) (orbital speed/period by body or AU, `orbital_raise_energy`, gravitational and Coulomb force, nuclear binding energy, de Broglie from kinetic energy, `E = P·t` three ways, black-hole entropy, binding energy of a sphere, Unruh, bubble, entanglement) |
| `param_aliases` | **train, then remove** (+ `strict_params`) | 7 `drill` recipes in the units a model slips on (Earth/Jupiter masses, km/h, tonnes, GHz, pF…); the pairing check requires ≥ 20 examples using each of the 27 canonical parameter names the alias table maps to, and **no** training target that uses an alias spelling |
| `body_name_chain` | **train, then remove** | 8 `lookup` recipes using the new `"$1.field"` reference (below) |
| `cascade.astro`, `.optics`, `.classical_intro`, `.modern` | **retire first**: they break §21 | `lookup` chains (escape velocity of Mars, Schwarzschild radius of a planet, orbital speed of a planet, light-travel time of a star, energy of the Sun over N years), hidden tools, `contrast` and `drill` recipes. Where a question can only be taught through a benchmark instance (the Sun's temperature), the classifier stays and the pairing check says so |
| `cascade.refusal`, `.clarification` | **train, then retire if the model matches** | 391 hand-written declines (was 76 of 26,491): 281 plain, 110 hard negatives that wear a physics word (a poem about a black hole); clarifications stay at v1's 773 (a required input removed) |
| other `cascade.*` groups, `.semantic` | **keep by default, measure for removal** | `tools/tweak_ablation.py` compares, per group, the cascade's right answers with the model's on the same questions in a cascade-off run. RETIRE when the model alone is at least as good (≥ 3 questions to judge from) |
| spurious second steps (not a tweak, but v2's worst failure) | **train** | 17 `contrast` recipes: the same slots as a chain, but the question asks only the first quantity (a distractor such as the satellite's mass is stated and unused), so the plan stops at one call |
| `pattern_hints`, nudges, fallback | leave; chain mode only | — (switch exists for `pattern_hints`) |
| `answer_key_hint` | **leave on in old runs, off in any new chain run** | — (switch exists; default unchanged so past chain numbers stay reproducible) |
| groundedness gate | keep | — |

### Field references (plan format v3)

`solar_system_body` and `named_star` return several fields. In v2 the harness decided which field a
later call got, from the wording of the question — a rule standing in for the model. v3 adds
`"$n.field"`: `{"steps":[{"tool":"solar_system_body","tool_args":{"body_name":"mars"}},
{"tool":"escape_velocity","tool_args":{"mass_kg":"$1.mass_kg","radius_m":"$1.radius_m"}}]}`.
A missing field or a non-number is an error, never a guess. The v3 system prompt is v2's plus one
sentence, so the v2 model stays reproducible. (The **final** answer of a single lookup is still picked
by the harness from the question's wording; letting the plan name it is the next format change.)

## How removal is paired with training (no double-ignores)

`sigma_ground/mcp/benchmark/tweaks.py` is the one place a run's tweaks are decided.

* **One switch per tweak.** Call sites ask `tweaks.on("cascade.astro")`; a test asserts every one of the
  13 classifier calls in the harness sits behind its own switch, so none can escape.
* **Profiles.** `legacy` (everything on: every result so far), `trained` (15 hidden tools shown, no
  parameter renaming, strict parameters; cascade stays as the safety net), `model-only` (trained + the whole
  cascade off: the model faces all 150 questions alone). `--tweaks`, `--tweaks-off`, `--tweaks-on` or
  `MENTAT_TWEAKS*` environment variables; a misspelled name is an error, never a silent no-op.
* **Every answer carries its profile.** `rec["tweaks"]` is written for every question, and
  `compare_router_runs.py` prints a `tweaks` column and flags a run that mixes settings.
* **A profile may only run if the data teaches what it removes.** `tools/tweak_pairs.py --data DIR
  --profile P` lists, for every tweak the profile switches off, the tools its replacement must teach
  and how many training examples teach each (minimum 20). A gap makes the exit status 1, and
  `run_router_v3.sh` then **skips that profile and says so**. Measured on the v2 data: profile
  `trained` was **not paired (15 gaps: every hidden tool had 0 examples)**; `model-only` 47 gaps.
* **Cascade-off is a measurement, not a switch-flip.** `tools/tweak_ablation.py` compares, group by group,
  the legacy run (cascade answers) with the model-only run on the same questions, and says RETIRE or KEEP
  per group. The first result, for the v2 model (before retraining), is produced by the pipeline's
  `ablate_v2` stage.

Comparisons stay honest because the ground rules still hold: compared runs use the same profile
(the headline stock-vs-tuned comparison is the `legacy` profile, where the cascade is identical for every
model); the other two profiles are separate, labelled runs with their own stock baselines.

## Not done, on purpose

* Nothing in the cascade was deleted. Every classifier still works exactly as before under `legacy`.
* No benchmark file, ground truth, scorer or tolerance was touched.
* The `astro`/`optics`/`classical_intro`/`modern` arithmetic is not "fixed" in place: the right fix is
  the model calling Mentat's tools, which is what the v3 data teaches; patching 5778 → 5772 in the
  classifier would only move the harness's physics around.

## A server fault found on the way

While generating the v3 data, a fresh Mentat server's first `convert_units` call hung (120 s+ in a probe,
15 min in the v2 recipes stage, which was put down to CPU contention with the n-body job). A thread dump
showed the worker thread stuck inside `import numpy` (`numpy/_core/multiarray.py <- pint/_typing.py <-
units._get_registry`), loading its native library; the same call in a plain process takes 0.7 s, and with
numpy imported first in the main thread the server answered in 0.1 s. `sigma_ground/mcp/server.py` now
imports numpy once at start-up, and `sigma_ground/mcp/test_server_first_call.py` pins it (1.9 s). Any
Mentat server on this machine had this exposure: the first pint-backed tool call could stall for minutes.
It is in this branch only; the Mentat that serves live traffic picks it up when it is redeployed.

## State of the data (CPU-only preview, before the teacher's rewrites)

`tools/tweak_pairs.py` on 25,037 training records: profile `trained` PAIRED (hidden tools 36–204 examples
each, 27 parameter names all covered, no alias spelling in any target); profile `model-only` PAIRED across all
13 classifiers and the semantic router, with the five guard-blocked tools listed as such. On the v2 data
the same check found 15 and 47 gaps. A pairing is the minimum bar (≥ 20 examples per tool), not proof of skill:
the model-only run and `tweak_ablation.py` are the proof.
