# Router v3.1 (4B) — math and parameter coverage, trained overnight 2026-10-14

Prepared 2026-10-13 for the 00:00 run. Follows [ROUTER_V3_PLAN.md](ROUTER_V3_PLAN.md); the design it starts on
is [DEFAULTS_AND_RANGES_DESIGN.md](DEFAULTS_AND_RANGES_DESIGN.md). Only the **4B** is trained (v3 showed a trained 4B
ties the trained 8B on the benchmark, and a 4B can stay loaded all the time).

## Why v3.1 exists

The v3 models read symbolic results and call the right tool, but the *arguments* of the math tools are wrong
too often. On a held-out math suite (329 questions the model was never trained on: wordings the training
templates do not use, plus 25 hand-written ones; each plan is run on the real Mentat server and checked
against an independent oracle):

| v3 on the held-out math suite | tool | arguments | answer |
|---|---|---|---|
| 4B | 310/329 | 160/329 | 194/329 |
| 8B | 323/329 | 185/329 | 211/329 |

The weak spots are the same for both: `compute_limit` (arguments 0/26), `summation` (1/26), `series_expansion`
(1/19), `gradient` (3/13), `integrate_expr`, and `differentiate_expr` (the 4B picks `integrate_expr` for "give the
derivative" and writes `e^x`). Details: `local-cache/eval_math_v3_4b.json`, `eval_math_v3_8b.json`.

## What v3.1 adds to the v3 data (all seeds verified by running them on the real server)

| seeds | count | what |
|---|---|---|
| `router_math_v31` | 3,291 (no teacher rewrites: the generator already writes 4 renderings of every expression: spoken, caret, Python, unicode) | 22 recipes over the math tools: differentiate, integrate (indefinite, definite, improper), solve, simplify, factor, expand, limits, series, sums, Laplace, Fourier, gradient, divergence, curl, determinant, eigenvalues, inverse, multiply, linear systems, ODEs. Every answer was checked against an **independent** oracle (scipy, mpmath, numpy, sympy by another route; a generator bug that fails the oracle is dropped, not labelled). Conventions the server needs are in every plan: `**`, explicit `*`, bounds as strings, infinity as `"oo"` |
| `router_params_v31` | 1,156 (teacher rewrites, 3 variants each) | parameter-coverage recipes: material-key pairs (friction, wear, thermal contact, acoustic, wetting, alloys, convection), isotopes, the van 't Hoff factor (from the salt, or stated), binding energy from a stated measured mass, lens image side, central-mass orbits |
| (same file) `db` family | 255 training records | **the materials database as the empirical reference**: critical angle / Snell's law between two media, wavelength in a medium, peak wavelength at a melting point, thermal energy at a boiling point, Carnot efficiency between boiling points, floating in a fluid. The plan looks the property up and passes it on (`"$1"`); the model writes no measured value |

Everything from v3 stays (33,387 training records), including the teacher-rewrite cache (indexed by seed: old seeds
first, checked to line up — rehearsed on 2026-10-13: the cache lines up and only the 3,468 new teacher requests (1,156 seeds × 3) need the GPU).

Preview data (before the new rewrites): 37,409 train / 4,021 validation. Checks run on it:

* `tool_census lint`: 33,368 tool calls against 232 signatures, **no issues**.
* `tweak_pairs`, profiles `trained` and `model-only`: **PAIRED, 0 gaps**.
* Parameters of the 232 tools with fewer than 10 examples: 99 → 57 (56 have none; 4 of the thin ones are required).
  The rest are mostly `playground_*` handles, `request_clarification`, and parameters a question rarely states.
* The benchmark guard stays on (no `--no-guard`): 214 candidate parameter seeds were dropped for resembling a
  benchmark question, 293 for a classifier conflict.

## What v3.1 does **not** do (so it is not over-claimed)

* No defaults-and-ranges table, no `object_properties` (mass + material → geometry), no "quote, don't compute"
  checks, and **`request_clarification` is still taught**. The design decisions in the doc are still open.
* The 57 thin parameters stay thin.
* The `db` rule covers seven chains; the lint that forbids a literal where a lookup fills the slot is not built.

## Launch (scheduled on the evening of 2026-10-13 for 2026-10-14 00:00)

```bash
cd /d/Aaron/development/sigma-ground/.claude/worktrees/router-v2
R=$PWD/local-cache
tools/run_at.sh 00:00 env FAMILY=v31 SIZE=4b DOCKER_OLLAMA=skippytel-ollama \
  PREV_DATA=$R/router_dataset_v3 \
  SEED_DIRS="$R/router_recipes $R/router_recipes_v3 $R/router_declines $R/router_math_v31 $R/router_params_v31" \
  SKIP_BASELINE_STOCK=1 ABLATE_V2=0 BENCH_PROFILES="legacy trained model-only model-only+sr" \
  COMPARE_EXTRA="v3_4b_legacy=$PWD/sigma_ground/mcp/benchmark/results/router_mentat-router-v3-4b_free_legacy_run.json v3_4b_trained=$PWD/sigma_ground/mcp/benchmark/results/router_mentat-router-v3-4b_free_trained_run.json v3_4b_model-only=$PWD/sigma_ground/mcp/benchmark/results/router_mentat-router-v3-4b_free_model-only_run.json v3_4b_model-only-structured=$PWD/sigma_ground/mcp/benchmark/results/router_mentat-router-v3-4b_free_model-only-structured_run.json" \
  tools/run_router_v3.sh
```

Order of seed folders matters (old first). A dry run (`--dry-run`) passed on 2026-10-13.
The model is `mentat-router-v31-4b`; everything is under `local-cache/router_v31/` (stage state, logs,
`4b/comparison.md`). The pipeline stops the `skippytel-ollama` container while it needs the GPU and starts it again
when it ends (as for the earlier overnight runs).

## Timeline (estimates from the v3 4B run's recorded stage times)

| stage | estimate |
|---|---|
| dataset (teacher `qwen2.5:14b`, 3,468 requests) | ≈ 25 min |
| train pilot + train (≈ 39–40k examples; v3 4B: 161 min at 33.4k) | ≈ 3 h 15 min |
| merge, GGUF, create, smoke | ≈ 5 min |
| eval_val (31 min measured for v3 4B), eval_hard, **eval_math** | ≈ 1 h |
| benchmark: `legacy`, `trained` (free + tool-first), `model-only`, `model-only+sr` | ≈ 45 min |
| **total** | **≈ 5½ h → done about 05:30–06:00** |

## What to read in the morning

1. `local-cache/router_v31/4b/comparison.md`: the benchmark profiles beside v3 4B and the earlier runs.
2. `local-cache/router_v31/4b/logs/eval_math.log`: held-out math suite, **same suite and scorer as the v3 baseline
   above** (tool / arguments / answer, by tool). The question this run answers: did training on the generated math
   move the held-out arguments (160/329) and answers (194/329), and did it cost anything on the benchmark?
3. `pairs_*.txt`: the pairing check (both profiles are expected to run).
4. Anything that is not an improvement is reported as it is; the scorer, the corpora and the tolerances are not touched.
