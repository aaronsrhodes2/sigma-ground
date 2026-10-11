# Router v2: plan

Branch `claude/router-v2` (worktree `.claude/worktrees/router-v2`), stacked on `claude/beautiful-einstein-9oiut1` (PR #1).
Written 2026-10-11, before the work, from the v1 results in `misc/SESSION_LOG.md` ("Session 38").

## What v1 taught us

| finding (measured) | consequence for v2 |
|---|---|
| Ollama sorts a JSON schema's keys, so the harness grammar forced `args` before `tool`: 73% → 99.7% right tool on 300 validation records once the format was dropped | name the keys so alphabetical order is the training order (`tool` < `tool_args`); keep the tool-name enum |
| 8 of the 15 benchmark failures are wrong for both models; several need two calls (satellite kinetic energy, TNT equivalent of a fission) | multi-step plans (`steps`, with `"$1"` references) in the data, the harness and the evaluator |
| validation 95% exact, benchmark model path 80%: validation shares phrasing families with training | a hand-written validation set in a different voice, kept out of training |
| free format let the model name a tool that does not exist (`thin_lens_magnification`) | constrained decoding with the tool-name enum, now possible because the keys are in order |
| "2 solar masses" came back as 4.0e30 (the model rounds the solar mass to 2e30) | solar-mass drills: many non-round masses across the black-hole tools |
| weak families: particle/energy conversions, `percent_of`, ionization energy vs energy level | hand-written recipe families for each, written from the tool descriptions |

Not attempted: `solve_equation` set-ups (the model would have to write equations with constants in them; that is the model doing physics).

## Plan format v2

One JSON object, nothing else:

- `{"tool": T, "tool_args": {...}}`: one call.
- `{"steps": [{"tool": T1, "tool_args": {...}}, {"tool": T2, "tool_args": {"velocity_m_s": "$1"}}]}`: two or three calls; `"$n"` is the headline value of call n.
- `{"tool": "request_clarification", "tool_args": {"variable": ..., "question": ...}}` and `{"decline": "..."}` as before.

The JSON schema for Ollama's `format` has three alternatives whose first keys are `decline`, `steps`, `tool`, all in alphabetical order, so the grammar and the training agree.

## Work, in order (CPU or light inference)

1. `router_plan.py`: the format, the v2 system prompt, schema, canonical form, `$n` resolution. Tests.
2. Harness: `--router-plan {v1,v2}` (prompt) and `--router-format {schema,free,toolfirst}`; a multi-step executor. Tests.
3. `eval_router_val.py`: compare canonical plans (v1 and v2 spellings, one step or several), per-source and per-family tables.
4. `router_recipes.py`: about 25 chain recipes and 12 single-call families (including solar-mass drills), hand-written templates, every plan executed against the real server and bounds-checked, benchmark guard on every step.
5. `build_router_v2.py`: v1 records converted (key renamed, new prompt, rewrites reused: no new teacher pass for them) plus the recipe records; optional teacher rewrites of the recipe seeds; group-aware split.
6. Hard validation set: about 100 hand-written questions in a different voice, plans verified by execution, similarity to all training questions measured.
7. Experiment on the v1 model, no retraining (about 5 GB, fits beside Skippy): does the `tool`/`tool_args` schema recover the constrained-decoding accuracy? Measured on validation and the hard set, then the benchmark.
8. `run_router_study.sh` parametrised for v2 (own cache, data, model name `mentat-router-v2`).

## GPU phase (waits for the Captain: it needs the whole GPU and Skippy's Ollama stopped)

| stage | what | estimate |
|---|---|---|
| teacher | rewrites of the recipe seeds only, `qwen2.5:14b` (9.5 GB), 3 variants | from the recipe seed count, at 0.34-0.45 s per attempt |
| train_pilot | 50 steps, Unsloth | 10 min |
| train | QLoRA from the base, 2 epochs | about 2.2 h for 21k examples; scales with the new total |
| export | merge, GGUF, `ollama create mentat-router-v2` | 6 min |
| eval | val, hard val, benchmark in `toolfirst` and `free`; stock Qwen on the hard set | about 45 min (stock needs 10 GB) |

Training from the base rather than continuing the v1 adapter, so that the v1/v2 comparison is not confounded.

## Decisions I made so the work could proceed (change before the GPU phase if you disagree)

- Key name `tool_args` (not a restructure of the plan) because it is the smallest change that puts the keys in order.
- Chains are limited to three calls and to references to a call's headline value; field references (`"$1.mass_kg"`) are not supported.
- No `solve_equation` family.
- Recipes follow tool I/O compatibility and physics, not benchmark items. Some chains do share their tool pair with a benchmark question's solution (satellite kinetic energy, TNT equivalent of a fission, mass from TNT); the guard drops text and number look-alikes, and the report lists which pairs overlap so a v2 gain there is not read as pure generalisation.
- The hard validation set is written by me, so my own phrasing habits could leak into both it and the recipe templates; its similarity to every training question is measured and reported.

## Findings from the prep work (measured, 2026-10-11)

1. **The key-order fix works on the v1 model, unchanged.** Same 300 validation records, v1 model, v1 prompt, a schema
   with `tool` / `tool_args` (alphabetical = training order): **99.7% right tool** (299/300), against 73.0% under the v1
   schema and 99.7% free. Exact plan 92.7% (free 96.3%).
2. **Ollama's constrained decoding cuts exponents that have a leading zero.** In every constrained run (plain JSON mode,
   v1 schema, tool-first schema) `2.42e-06` came out as `2.42e-0`: 46 of 46 cases with `e-05`, `e-06`, `e-07`, `e-09`
   were cut, and 135 of 135 with `e-10`, `e-15`, `e-26`, `e+22` .. `e+31` survived (181 of 181 consistent with "no leading zero
   in an exponent"). Free generation never cut one (0 of 53). A truncated exponent is a plausible number wrong by a
   factor of a million, so this is a correctness bug in the request format, not a style point. It also affected the v1
   `tuned_schema` and `stock_schema` benchmark rows. The v1 training data wrote `e-06` (Python's spelling) and so taught
   exactly what the grammar cannot produce.
   **Fix in v2:** every training plan is written with `router_plan.dumps_plan` (`2.42e-6`), validated by the assembler; the
   tool-first schema then keeps the tool-name enum without corrupting numbers. Whether it does is measured in the eval
   stage (tool-first vs free); until then free generation stays the safe default.
3. **The CPU is shared.** n-body work holds 18 of 20 threads, so a fresh Mentat server's first tool call (it imports pint
   and friends) takes minutes, the dataset's template phase swung between 172 s and 461 s, and my early probes timed out
   on it. Warm-up timeouts are 900 s, Windows-side jobs run below normal priority, and WSL steps cap torch at 2 threads
   and `llama-quantize` at 4.
