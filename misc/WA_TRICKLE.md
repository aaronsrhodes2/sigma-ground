# Wolfram|Alpha free-tier trickle — the first 61 answers (2026-10-14)

SkippyTel's `services/wa-trickle` sends each question of the held-out math suite to Wolfram|Alpha **once, verbatim**, on a free-tier schedule (60 a day, one
call per question ever), and freezes the reply. `tools/wa_trickle_score.py` reads those replies and scores them with the router's own scorer
(`eval_math_suite.same_answer`, unchanged) against the suite's oracle, and pairs them with the router + Mentat route on the same questions.

```bash
python tools/wa_trickle_score.py --router local-cache/eval_math_v31_4b_on_master_tools3.json \
    --adjudications misc/wa_trickle_adjudication.json --out local-cache/wa_trickle_score.json --show-misses
```

The scored snapshot is `local-cache/wa_trickle_snapshots/2026-10-14_61_math329.jsonl`. It is **not committed**: it holds Wolfram's replies. The service's own copy is
in `SkippyTel/state/wa-trickle`. Pass `--export <file>` to score a snapshot, or leave it off to read the live `/export`.

## What came back

| | count |
|---|---|
| questions asked (all `math329`: 25 hand-written "wild", then templated by tool) | 61 |
| Wolfram returned **no result** (no pods; a "did you mean" or a tip at most) | 30 |
| Wolfram answered | 31 |
| of the 61, with no oracle answer in the suite (`result: null`, `msuite_0304`–`0308`) | 5 (not scored) |
| **scorable** | **56** |

## Scores on the 56 (strict = the shared scorer; lenient = strict + the cases I read as the same answer in another convention, `misc/wa_trickle_adjudication.json`)

| system | strict | lenient | notes |
|---|---|---|---|
| **v3.1 4B router + Mentat** (master tools) | **54 / 56** | 54 | misses: invented a tool `solve_system_of_equations` (0324); omitted `func`/`variable` for `solve_ode` (0283) |
| **Wolfram|Alpha**, the question as written | **21 / 56** | 23 | 29 no result |
| Wolfram, only the 27 it answered | 21 / 27 | 23 / 27 | the router scores 25 / 27 on the same 27 |
| deployed Mentat `/ask` | — | — | returned a value for 1 of 61: `26` from the falling-object profile for "how fast is x cubed changing" — not an answer. The deployed text front door is still Materia's (W5) |

By wording (scorable):

| wording | n | Wolfram no result | Wolfram strict | router |
|---|---|---|---|---|
| hand-written terse ("det of [[3,1],[2,5]]", "factor x^2 + 7x + 12") | 20 | 0 | 18 (19 lenient) | 19 |
| templated, Python syntax (`**`, `[[ ]]`) | 11 | 11 | 0 | 11 |
| templated, plain English | 25 | 18 | 3 (4 lenient) | 24 |

Paired on the same questions, a two-sided exact sign test on the questions where the two disagree:

| set | both | only Wolfram | only router | neither | p |
|---|---|---|---|---|---|
| all 56, strict | 20 | 1 | 34 | 1 | < 0.001 |
| all 56, lenient | 22 | 1 | 32 | 1 | < 0.001 |
| Wolfram's 27, strict | 20 | 1 | 5 | 1 | 0.22 |
| Wolfram's 27, lenient | 22 | 1 | 3 | 1 | 0.63 |

## Wolfram's answered-but-not-right cases, read by hand

| id | what happened | verdict (counted?) |
|---|---|---|
| 0025 | antiderivative of (4x+1)^3 differs from the oracle's by the constant 1/16 | equivalent (lenient) |
| 0328 | `y'' = -4y`: the same two-parameter family, constants named c_1/c_2 with sin and cos swapped | equivalent (lenient) |
| 0318 | "taylor series for cos x up to order 4": Wolfram keeps the x^4 term, the oracle stops before it | convention (not counted) |
| 0113, 0114 | "break … into factors": Wolfram took "factor" for a function name and answered whether it is continuous | misread (not counted) |
| 0283 | "every function x(t) with x'' - 4x' + 3x = 0": Wolfram read x(t)·x''(t) and called it nonlinear | misread (not counted) |

The five questions with no oracle answer were checked with sympy: Wolfram is right on all four it answered (0304, 0306, 0307, 0308); 0305 got no result.

## Changes to the extractor after the first look

The first run scored Wolfram 17 / 56 strict. Four of the misses were mine, not Wolfram's, and I fixed them in the extractor (a pod-selection and a type rule, both from the
pod structure, none from the oracle): gradient, divergence and curl read a "Del operator form" restatement pod instead of the "Result in 2D/3D Cartesian coordinates" pod
(a pod titled "…result…" is now preferred), and `x = e^2` was read as a number where the oracle has the one-element solution set (a bare `name = value` pod is a solution
set). Strict became 21. The scorer, the corpus and the tolerances were not touched. Nothing was changed to move the router's number.

## What this does and does not say

* **It is not a measurement of Wolfram's mathematics.** Wolfram never saw a clean query: it saw the held-out wordings verbatim ("Multiply out the brackets in (3x + 2)^2." gets
  no result, with Wolfram's own "did you mean" being `(3x + 2)^2.`). 30 of the 61 replies were empty (29 of the 56 scorable), and it is not only Python syntax: 18 of the 25 plain-English
  templated questions failed too. On the 27 questions it did parse it is close to the router (21–23 against 25; not distinguishable at this size).
* **The router has two advantages that are not about mathematics.** The suite's wordings are held out but its question families are the ones the router trained on, and the
  oracle is sympy, the engine under Mentat, so conventions (series order, ODE constants, antiderivative constants) agree by construction. Wolfram has neither.
* **The 61 are not a random sample.** The first 25 asked are the hand-written ones; the rest go round-robin over the tools. The 268 still waiting are templated, and the trickle
  will finish in about 4.5 days at the budget; the per-wording table is the thing to watch as they arrive.
* The stored pods carry no `primary` flag (the answer pod is chosen by title), because the service kept only id, title and text. Capturing `primary` for the 268 still to come
  would remove that guess. The replies already frozen cannot be re-fetched (one call per question ever).
* To separate "Wolfram did not understand the sentence" from "Wolfram cannot do it", the 29 scored no-results could be re-sent in Wolfram's own style (`expand (3x+2)^2`);
  that is about half a day of the free budget and a change to the "one call per question" rule, so it is the Captain's call.

## Update 2026-10-15 — 95 asked (34 more; the extractor was not touched)

The 34 new answers are an out-of-sample test of the extractor, which was designed on the first 61. It read all six answered ones without refusing and made no extraction fault; the five that are not
right are Wolfram's misreadings, visible in its own input pods (`misc/wa_trickle_adjudication.json`): two ODE questions where "y(x) with y'" became the product y(x)·y'(x), two "where does … cross zero"
where it read the point (5, 0), and one series "to order 4" with the x^4 kept. The other 28 got no result.

| set (scorable) | Wolfram strict | Wolfram lenient | router | notes |
|---|---|---|---|---|
| all 90 | 22 | 24 | 87 | 57 no result (all 20 with Python syntax; 37 of 50 plain-English templated) |
| only the 33 Wolfram answered | 22 | 24 | 30 | sign test p = 0.02 strict, 0.07 lenient |

So the earlier "not distinguishable on the 27" was a small-sample reading: with six more answered questions (one right for Wolfram, five for the router) the gap on the answered set is larger. The new rows are
the round-robin's templated questions, the chattiest wordings, so this is not a random draw either. Twenty tool kinds now have 3–6 templated questions each; the rest arrive at about 60 a day.
