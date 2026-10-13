# Router v3 — retrain tonight, and test whether a smaller model is enough

Prepared 2026-10-12; **nothing in this plan has run on the GPU.** What changed and why:
[MENTAT_TWEAKS_AUDIT.md](MENTAT_TWEAKS_AUDIT.md). The v2 results it follows: `misc/ROUTER_V2_PLAN.md`
and `local-cache/router_v2/comparison.md` (v2 tuned 134/150 against stock 138/150 and the chain
baseline 134/150, because of spurious second steps, a unit slip, and tools it could not reach).

## What v3 is

| | v2 | v3 |
|---|---|---|
| plan format | `tool_args`, `steps`, `"$n"` | + `"$n.field"` (a lookup's field named by the plan, not guessed by the harness) |
| catalogue tools with a training example (232 tools) | 209 (201 with ≥ 20); the 15 `HIDDEN_FROM_LLM` tools: none | 224 (216 with ≥ 20). The 8 without: 5 guard-blocked (their only question is a benchmark's) and `parse_quantity`, `playground_apply`, `playground_simulate`, `simulate`, which no tweak covers |
| "stop after one call" | none | 17 `contrast` recipes (880 training records): same slots as a chain, the question asks only the first quantity |
| declines in the training set | 76 | 434 (391 new seeds: 281 plain, 110 hard negatives that wear a physics word) |
| unit slips | — | 7 `drill` recipes (Earth/Jupiter masses, AU, km/h, tonnes, GHz, pF) |
| tweaks in the benchmark | all on | three profiles, each run only if the data teaches what it removes |
| base models | Qwen3-8B | Qwen3-1.7B, 4B, 8B — same data, same hyperparameters |

Preview data (CPU only, before any teacher rewrites, so smaller than v2's 26,491): 25,037 train, 2,679
validation; `tools/tweak_pairs.py` reports **PAIRED, 0 gaps for both `trained` and `model-only`**
(v2 data: 15 and 47 gaps). Per hidden tool: 36–204 examples before rewrites.

## Stages and time

`tools/run_router_sweep.sh` (sizes smallest first, so the pipeline is proven on a run under an hour
before the 8b trains for about three). Measured = from the v2 run's recorded stage times; estimate =
scaled from them (parameter count, 33k vs 26.5k examples) — the 50-step pilot of each size replaces the
estimate with a measurement and the pipeline stops if the projection exceeds `MAX_TRAIN_H` (6 h).

| stage | uses | 1.7b | 4b | 8b |
|---|---|---|---|---|
| recipes_v3, declines, hardval, pairs | CPU, done or minutes | done | — | — |
| dataset (teacher `qwen2.5:14b` rewrites 2,856 new seeds; v2's rewrites reused) | whole GPU | ≈ 70 min, once | — | — |
| ablate_v2 (v2 model, cascade off) | shared GPU | ≈ 10 min, once | — | — |
| pilot + train | whole GPU | ≈ 45 min (est.) | ≈ 95 min (est.) | ≈ 185 min (est.; v2 measured 148 at 26.5k) |
| merge + GGUF + create + smoke | CPU/disk, then GPU | ≈ 8 min | ≈ 8 min | 8 min (measured) |
| eval_val, eval_hard (tool-first and free) | shared GPU | ≈ 10 min | ≈ 12 min | ≈ 20 min (measured) |
| baseline_stock + benchmark (up to 7 runs, 3 at a time) | shared GPU + CPU | ≈ 25 min | ≈ 25 min | ≈ 25 min |
| **total** | | **≈ 1.6 h** | **≈ 2.6 h** | **≈ 5.6 h** → ≈ **10 h** with the data stage |

The whole sweep needs the GPU for most of that, so `skippytel-ollama` is stopped for the duration
(`DOCKER_OLLAMA=skippytel-ollama`; restarted by the script on exit, also on failure) — unless a second
card takes Skippy. With two cards (`TWO_CARDS=1`, or `TRAIN_GPU`/`SERVE_GPU`) the sizes run in pairs and
Skippy keeps its card.

## What will be measured, and compared like for like

For every size: validation (98.0 %/93.8 % for v2) and the 90-question hard set (both output formats), the
150-question benchmark in three tweak profiles, each against the **stock** model of the same size under
the same profile:

* `legacy` — the cascade answers its 86 questions, the model the other 64 (the comparison of record;
  the cascade is identical for every model).
* `trained` — hidden tools shown, no parameter renaming, strict parameters; cascade still on.
* `model-only` — cascade off: the model faces all 150 alone. `tools/tweak_ablation.py` then says, per
  cascade group, RETIRE or KEEP.

Skipped, loudly, if `tweak_pairs.py` finds a gap. The answer to "is a smaller model enough" is the
sweep table (`local-cache/router_v3/sweep.md`): per size, legacy / trained / model-only correct counts, router
errors, unit-level misses, latency. The 8b v3 is the reference; a smaller size is a candidate if it is equal
or better on those, on the same data.

## Ground rules, unchanged

Only measured numbers are reported. The scorer, ground truth, corpora and tolerances are untouched, and
the benchmark guard still keeps training data away from the benchmark and the hard set (five no-input
tools are blocked by it and keep their classifier). No `--no-guard` run is reported. Datasets,
checkpoints, GGUFs and run files stay out of git (`local-cache/`). A surprise (validation high, benchmark
worse) stops the pipeline for a decision, as in v2.

## Run

```bash
# preflight only, no GPU
SIZE=8b tools/run_router_v3.sh --dry-run
# everything, smallest first (stops skippytel-ollama; restarts it at the end)
DOCKER_OLLAMA=skippytel-ollama tools/run_router_sweep.sh
# one size
SIZE=4b DOCKER_OLLAMA=skippytel-ollama tools/run_router_v3.sh
```

## Known limits

* The 0.6B size is not wired in (a single safetensors file, no index, which the merge step expects).
* The final answer of a single lookup (`solar_system_body('mars')`: which field?) is still chosen by the
  harness from the question's wording; a plan field for it is the next format change.
* Cascade-off runs depend on the model emitting exact parameter names; that is the point, and what
  `strict_params` makes visible.
