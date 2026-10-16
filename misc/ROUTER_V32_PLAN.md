# Router v3.2 (plan v4) — the data, how it was built, how to train it and how to judge it

Written 2026-10-16. The data work the Captain asked for on 2026-10-15 ("do the data work now to execute there for training"), after the six decisions he made the same
day. Nothing here has been trained: the model is trained on the Bosgame M5 (`misc/M5_TRAINING_PLAN.md`); this is the data, the tools around it, and the evidence that they work.

## The Captain's decisions and what became of each

| decision (2026-10-15) | what was built | where |
|---|---|---|
| 1. Answers in the measurement system the question used, SI in parentheses ("1 yard (0.914 m)") | **Output:** every result of a tool called with a non-SI quantity carries `display`, e.g. `30 feet` in → `29.96 mph (13.39 m/s)`; `value` stays SI. **Input:** the router no longer converts units in its head; it writes the quantity as the question gave it (`"5.09 km"`) and the contract converts. | `mcp/unit_display.py`, `mcp/contract.py`, plan **v4** (`benchmark/router_plan.py`, `router_quantities.py`) |
| 2. Conventions, yes if the physics tests still pass | `fourier_transform(convention="ordinary"\|"angular"\|"unitary")`, `series_expansion(convention="exclusive"\|"inclusive")`; defaults are the old behaviour; the result states the convention; ODE and indefinite integrals state their constants. Condition checked: see "Physics tests" below. | `mcp/tools/mathx.py`, `symbolic.py`, `server.py` |
| 3. Renames, sure | 47 parameters of 33 tools renamed to one vocabulary (311 → 290 distinct names); old names still accepted. | `mcp/param_names.py`, `misc/CONTRACT_RENAMES.md` |
| 4. Sizes: "you know better" | 4B in both `nf4` and `bf16` (so the recipe change is measured), 8B in `bf16`; none larger as the deployed router. | `misc/M5_TRAINING_PLAN.md` |
| 5. Linux | `tools/run_router_linux.sh`, rehearsed under Linux (WSL, Python 3.12). | `tools/run_router_linux.sh` |
| 6. The 4070 PC: "the fast-small-model runner, as well as dev and gaming" | Recorded; nothing to build yet. The router (4B) and small models are served from it; the M5 hosts the big models and SkippyTel. | memory, `M5_TRAINING_PLAN.md` |

## The data

`local-cache/router_dataset_v32` (plan v4): the v3.1 data relabelled, plus 400 convention seeds.

| | v3.1 (plan v3) | v3.2 (plan v4) |
|---|---|---|
| train / val records | 39,153 / 4,204 | **39,512 / 4,242** |
| unit-bearing arguments written as the question wrote them | 0 | **41,887** (37,640 train + 4,247 val), in 23,103 + 2,583 records |
| parameter names | old | canonical (the lint refuses an old one) |
| convention examples | 0 | 250 of 361 Fourier calls, 150 of 454 series calls name a convention |
| system prompt | "Convert every input to the unit its parameter name states" | "Write every quantity exactly as the question gives it ... the tool converts" |

Held-out: the math suite (329 questions, unchanged), **a new convention suite** (58 questions in wordings the training never uses; `local-cache/router_conventions_suite.json`),
the hard validation (the same 90 questions, 66 of their gold plans relabelled) and the validation split. Build: `tools/build_router_v2.py --plan v4 ...` from the seeds, the v1 data
and the cached teacher rewrites; **no teacher is needed** (the convention seeds, like the other math seeds, take their wording from templates).

35% of the unit-bearing arguments in the v3.1 data (13,868 of 39,605) were in a unit other than the parameter's, so the router was doing unit arithmetic in its head;
10% were quantities with no unit-word pint understood ("7.35 solar masses" needed a recalled constant: the contract now reads "solar masses", "light years", "earth radii"...).

## What was verified

* **The relabel preserves meaning.** 1,500 plans whose labels changed, run through the real server old versus new: 1,442 identical answers, 0 different; the 58 value-less
  procedure results identical except a `summary` string that echoes the input with more digits (the quoted value is the exact one). The 90 hard-validation plans: 57 single-step
  plans identical, 0 different.
* **Lint** (`tools/tool_census.py lint --canonical-only`, which now validates a quoted quantity with the server's own reader): 48,976 calls in train and val, no issues.
* **The convention seeds are right.** 400 training and 58 suite items, each checked by an independent method (a quadrature of the transform with the convention's own kernel; the series
  check). The oracle accepts the right convention and rejects the other two (a test pins it).
* **The data rebuilds identically on another operating system.** `tools/data_bundle.py rebuild-check` regenerates the dataset on the CPU with no teacher and gets the same bytes on
  Windows (Python 3.13) and Linux (Python 3.12): `train.jsonl 676538b6…`, `val.jsonl 1d2d1fcc…`, `rewrites.jsonl 4444f2a6…`.
* **Physics tests.** Clean-room full run (a plain export of the commit, `pytest` with the project's own testpaths), baseline f62d5e7: 6,848 passed, 24 failing or erroring. At e3b3a53:
  6,999 passed, 5 failing. 20 failures went away (test pollution, below); 4 are older and untouched by this work (`test_restated_predictors`, `test_router_recipes::real_server`,
  `test_targets` QA thresholds, `test_coverage_ledger` for another session's comet functions); **1 was new and mine** (a bare numeric string was scored equal to a number), fixed in 85664c6.
  Final run: see the end of this file.

## What rehearsing it found (each fixed, each with a test)

1. **Test pollution (pre-existing).** Seven `field/interface` tests did `sys.path.insert(0, <sigma_ground>)`, so after them `import mcp` found `sigma_ground/mcp` instead of the MCP SDK:
   20 tests in `sigma_ground/mcp` failed or errored in every full run, among them the contract tests. Now `append`. (38e3c26)
2. **A silent environment dependency.** `classifier_route` swallowed every exception, including a missing package: without `periodictable` (declared in the `mcp` extra) the astro classifier went
   silent and a rebuilt dataset differed. A missing package now raises, and the Linux runner's preflight checks every package the project declares. (99156d2)
3. **Line endings.** Python's text mode wrote CRLF on Windows and LF on Linux: 39,512 identical lines, two different hashes. The writers force LF; the bundle's manifest also carries a
   line-ending-independent hash. (99156d2) My earlier claim that the rebuild was "byte-identical" had only been proven on one operating system.
4. **Three label bugs in my own seeds**, caught by checking that every variable a label names appears in its question: a rename helper that cannot see the `x` in `4x`, kernels written
   fused ("2πiνt"), and a held-out wording that contradicted its own label. The same helper bug is in 3 of the 3,291 older math seeds; those get no record (their slot is kept: the rewrite cache is indexed by position).
5. **My comparator was too loose** (the regression above) and **my runner swallowed failures** (`set -e` is suppressed in a function called with `||`; a stage marked done after a failing check).

## Open

* **56 of 547 parameters appear in no training example** (4 required, e.g. `parse_quantity.quantity_string`; lookup keys such as `pka_list`). `tools/tool_census.py coverage`.
* **The deployed Mentat container still runs the code from before 2026-10-15**: `display`, the conventions, the renamed parameters and the two-word units are not live until it is rebuilt
  from a newer ref. Old clients keep working either way (old names are accepted).
* **Not trained, not evaluated.** The numbers to beat are the v3.1 4B's (math suite 309 → 315 of 329 answers; narrative benchmark unchanged from v3) and the no-tool baselines (`misc/BASELINES.md`).
  The new things to measure: the convention suite, and the hard validation's quantity arguments.
* The remaining coverage and tool-name gaps of v3.1 (an invented `solve_system_of_equations`, `solve_ode` without `func`) are not addressed by this data.

## Run it

```bash
python tools/data_bundle.py create router_v32 --out local-cache/bundles/router_v32_2026-10-15.zip     # already made; this PC
python tools/data_bundle.py rebuild-check local-cache/bundles/router_v32_2026-10-15.zip              # on the new machine: expect three IDENTICAL
SIZE=4b PRECISION=bf16 tools/run_router_linux.sh pilot                                              # then without "pilot" for the whole run
```
