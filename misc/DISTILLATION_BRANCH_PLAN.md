# Improvement distillation: a daily branch of the Wolfram contender experiment (plan, 2026-10-30)

The Captain's ask (2026-10-30): on the experiments page, a new branch. Every day purple Skippy researches a NEW set of science questions,
of the kind Wolfram|Alpha's free tier accepts; we put them to Wolfram and to Mentat; when Mentat fails, we find out why.

**Two rules, his words:** we only COMPARE against Wolfram; Wolfram's data never improves Mentat. We DO use our own data and outputs
(the questions, Mentat's answers, its miss reasons, our oracle) to improve Mentat.

## What exists

- `services/wa-trickle` (SkippyTel, port 3010): a queue of questions, **one Wolfram call per question ever** (frozen), the same question to
  Mentat at every new ref, `/export` as JSON lines, budget 60/day of the free tier's ~2,000/month. Today: 497 questions queued (math329,
  physics150, adversarial18), **all asked, 0 waiting**: the daily budget is idle.
- `sigma_ground/mcp/benchmark/daily_job.py` + `live_ask.py`: the live suites every day, scored with `score.py`, rows on the dashboard.
- `portal/experiments.py`: the experiments registry; the "Mentat vs Wolfram Alpha vs Gemini" benchmark tile reads committed result files.
- `misc/STATIC_INTENTS_PLAN.md`: the hand-made version of this loop (30 misses -> causes -> static fixes, paired replays). The branch
  automates the first two steps and queues the third for a person (or a purple goal, when goal mode exists).

## The daily loop (one worker, the house's own machines, no tokens)

1. **Question maker** (purple Skippy, Core's chat slot). N new questions a day (proposed N = 40, leaving 20 of the budget for re-asks and the
   old suites' rare additions). Each question is rendered from a **spec** (tool + arguments, like `router_math.py`'s recipes) so OUR oracle can
   compute the expected answer where one exists (maths: sympy, exact; physics: the tool itself at the spec's arguments, checked for
   self-consistency), plus a quota of free-written "wild" questions with no oracle (marked `unverified`). Style constraints: terse,
   standard notation, SI or common units, no Python syntax, one quantity asked: what Wolfram's free tier parses. Domain rota per
   PROJECT_GOAL: maths first (half), then physics, chemistry, a little astronomy. **New** means not in the trickle's database (hash of the
   normalised question) and numbers drawn fresh; the generator's seed, prompt and model are logged with the set.
2. **Ask both.** The set is enqueued as suite `distill-YYYY-MM-DD`; the trickle asks Wolfram (once) and Mentat (`/ask`, at the live ref,
   re-asked when the ref changes). Nothing else about the trickle changes.
3. **Diagnose Mentat, from our own record.** For every row where Mentat's answer disagrees with OUR oracle, or Mentat gave none, the cause is
   read from Mentat's own result (`inputs.route`, `router_miss`, `error.code`, the plan): `no_route` (decline/clarify), `wrong_tool`,
   `bad_args` (a contract error), `tool_error`, `wrong_value` (a value, but not the oracle's), `unverified` (no oracle: for a person to read).
   Wolfram's answer is recorded beside each row as `wa_status` (answered / no result) and `wa_agrees_with_us` (when we have an oracle), as a
   yardstick. **It is never a cause, never an expected value, never a prompt or an alias.** A row where Wolfram answers and Mentat does not
   is the most interesting kind, and still the fix is found from Mentat's miss reason.
4. **The ledger.** `sigma_ground/mcp/benchmark/results/distill/YYYY-MM-DD.json` (rows, causes, the generator's record) and an aggregate
   `misc/DISTILLATION_LEDGER.md`: causes grouped into candidate fixes (the shape STATIC_INTENTS_PLAN has), each with its rows, a status
   (open / fixed in <commit> / not a computation) and the paired-replay rule for landing it. The three old suites keep running as the
   regression guard: a fix that costs them a row does not land.
5. **The page.** On the experiments page, under the benchmark tile, a branch "Improvement distillation": a day-by-day table (questions,
   Wolfram answered / no result, Mentat answered, agreement with our oracle, Mentat misses by cause) and the open fix list with status.
   Public like the rest of the page, so it is honest about misses; the data is the committed ledger, as the benchmark tile's is.

## What I need decided

| # | decision | my proposal |
|---|---|---|
| 1 | questions per day | 40 (20 kept back) |
| 2 | domain rota | maths 20, physics 12, chemistry 5, astronomy 3; shift as a domain's miss rate falls |
| 3 | where the maker runs | purple Skippy as a scheduled job on skippy-core (the bridge's scheduled-task hook), in the quiet window; until goal mode exists, the sigma-ground side has a `tools/distill_day.py` that can also drive Core's chat model directly, so day one does not wait on the bridge |
| 4 | unverified rows | counted separately, never as misses; the Captain (or a later oracle) turns them into ground truth |
| 5 | page | a branch section inside the benchmark experiment (not a new tile) |

## Build order (sigma-ground first, then the portal)

1. `tools/distill_day.py`: spec recipes + oracle, the wild-question prompt, the novelty check, the enqueue; dry run writes the set without
   asking anyone. ~half a day.
2. `benchmark/distill_score.py`: read the trickle export for `distill-*`, diagnose causes, write the day's JSON and the ledger. ~half a day.
3. Portal: the branch section reading the committed ledger; the trickle's `/status` by suite. ~half a day.
4. Scheduling on skippy-core and the first week of data; then the first fixes through the usual paired replay. From then the loop runs.

Honest limits: a question Wolfram's free tier cannot parse teaches nothing about Mentat (it is still asked, once; the no-result rate is
itself a result). An oracle we do not have is not invented: physics questions outside the tool's own forward check stay `unverified`.
