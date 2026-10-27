# Mentat audit, 2026-10-27 — the live Mentat on skippy-core, against the Wolfram-contender goal

The Captain asked (2026-10-27): put Mentat's new home through its paces, look at the router code, and audit whether Mentat has veered off its goal.
Is it the best physics professor / Wolfram|Alpha contender we can provide, and how do we improve it?

Severity as in [MENTAT_AUDIT_2026-10-13.md](MENTAT_AUDIT_2026-10-13.md): **S1** a wrong answer can reach a user, **S2** a right answer is refused or
unreachable, **S3** hygiene that will cost us later. Every number below was measured on 2026-10-27 against the live service unless it says otherwise.

## What runs, and what was checked

* `skippycore-mentat-1` on skippy-core, sigma-ground **ef2d4c0** (= origin/master; `/version` read before and after every run, unchanged), SkippyTel
  929f85c, 224 public tools. `MENTAT_ROUTER_MODEL=mentat-router-v31-4b`, `MENTAT_USE_LLM=1`.
* The models sit behind `skippycore-ollama-1`, which is **not Ollama**: a gunicorn shim (`SkippyTel/core/ollama-shim/shim.py`) that translates
  Ollama's API to llama-swap / llama.cpp (Vulkan). Router slot: the v3.1 4B GGUF, **byte-identical** to the one trained on skippy-pc
  (sha256 c929e636…, 2,497,280,832 bytes), `--temp 0`, thinking off, `--parallel 2`, 16K context. A JSON schema becomes a llama.cpp grammar.
  The chat slot (`skippy-chat`, alias `qwen3:8b`) is Qwen3.6-35B-A3B at temperature 0.7: it is the keyword translator's "Qwen".
* Reached from skippy-pc through an SSH tunnel (127.0.0.1:7350 and :11434 on the box are not published to the LAN).
* The stack was redeployed twice during the audit (07:14 and 07:20 UTC, pushes from another session); one run hit it and was repeated. The runner now
  retries transport errors and records how often it did (0 in every reported run).

Suites and scorers, all unchanged: the held-out math suite (329, `local-cache/router_math_suite.json`, `tools/eval_math_suite.same_answer`), the
150-question narrative physics benchmark (`benchmark/questions.json`, `score.py`'s `_score_one`), and the Wolfram free-tier trickle (`tools/wa_trickle_score.py`).
Scripts: `tools/audit/` was not created; the one-off runners are recorded at the end of this file.

## Results

| path | math (329) | physics (150) | median time |
|---|---|---|---|
| **deployed `/ask` (what users get)** | **0** (329 ended in `clarify`) | **105** (router 104 of 124 it answered; fallback 1 of 26) | 1.1 s / 0.7 s |
| same code replayed on this PC, as deployed (D) | 0 of 60 | 105 (agrees with the deployed row on all 150) | 1.0 s |
| same code, harness defaults + `structured_results` (S) | **317** | **109** (paired against D: +4, −0) | 0.8 s |
| **deployed `/ask` after the fix (SkippyTel 0d62ade, 08:05 UTC, stack unchanged through the run)** | **317** | **109** | 0.5 s / 0.6 s |
| v3.1 router in the harness, 2026-10-14 (4070, Ollama): `model-only` / `trained` profile | 309 (315 on master tools) | 109 / 140 | |
| Wolfram free API, trickle (math: 324 with an oracle) | 52 strict, 54 lenient (225 no result) | 56 (merged run) | |
| public cloud Mentat (keywords only), 5 probes | 0 of 1 | 1 of 4 | 0.1 s |
| **programme step 2: the box's chat slot (Qwen3.6-35B-A3B, "vanilla Qwen") as the router, same code path, same prompt and grammar** | not measurable: the shim dropped 261 of 329 connections (RemoteProtocolError), 79 again on the retry; the 68 that answered all ended in `clarify` | **39** (router 28 of 56 it answered; 44 `clarify`; 24 to the translator) | 1.5 s |

Step 2's answer for the chat slot: the trained 4B router stays first (109 vs 39 on physics; 317 vs nothing on maths). A 35B that writes
grammar-constrained plans it was never trained to write is not a baseline, it is a different system; the honest "vanilla Qwen" baseline is the
chat model with Mentat's tools in its own tool-calling loop (the harness's chain mode), which is a separate run and not yet made.

Row S, and the live service after the fix, equal the October `model-only` measurement of this router (109) exactly. The harness's 140 came from its `trained` profile, where the
benchmark's own cascade (regex classifiers and the semantic interpreter, `benchmark/interpreter_demo.py`) answers ahead of the model; `/ask` has no such
stage, only Materia's keyword translator behind the router (1 of 26 right). The 12 maths misses in row S: 3 ODEs whose primes the parser refuses
(`x'' - 4*x'`), 2 gradients returned as a list the oracle writes differently, 5 in the hand-written "wild" set, 2 wrong tool choices (the same
weak spots as in October).

## Findings

### S2 — the deployed front door answers no maths at all

1. **Every maths answer the router gets right is thrown away.** `SkippyTel/mentat/router_stage.py` runs the benchmark harness's `_route_once` as it is,
   but never sets the harness's tweak switches. `structured_results` defaults to off, and with it off `_route_once` treats a symbolic or list result
   (a derivative, an integral, roots, a matrix) as "returned no value". `router_stage.answer` then returns None, and `/ask` falls through to the keyword
   + Qwen translator, which answers "differentiate x^2 sin x" with *"I can simulate a falling object's impact speed…"* (route `clarify`, tagged
   `SPECULATIVE-PENDING`). Measured: **0 of 329** through the live `/ask`; every one ended in `clarify`. Replayed locally with the same code, the plan for
   the first question is right (`differentiate_expr(x**2*sin(x), x)`) and the tool's value is right (`x**2*cos(x) + 2*x*sin(x)`); the stage drops it.
   The v3.1 scores (309/315) were measured with `trained` + `structured_results` on, so the live router is not running the configuration it was
   measured in: **contender gate 6 ("deployed is what we measured") fails**, although `/version` equals master.
2. **Why nothing caught it.**
   * `router_stage` swallows every exception and every miss reason (by design, so the router can never stop `/ask`), and logs none of them.
   * `mentat/tests/test_mentat_router.py` replaces `_run` with a hand-made record, so the harness's value reading never runs in a test; one test
     asserts that `extracted_value: None` falls through.
   * The commit that put the router first (SkippyTel 2c0cce2, 2026-10-25) measured it on the 150 physics questions only.
   * The alarm already exists and was ignored: `tools/wa_trickle_score.py` prints "The deployed Mentat front door (/ask) returned a value for N of
     329" (N = 0 today), and [WA_TRICKLE.md](WA_TRICKLE.md) recorded 1 of 61 (a wrong one) on 2026-10-14. The daily job does not run that scorer.

### S1 — answers that look right and are not the answer

3. **"Where is the Moon right now?" / "How far is Mars from the Sun right now?"** return the body's static record (fact-sheet mass, radius, mean
   semimajor axis) tagged `VERIFIED`, not a position. `body_position` exists and is served, but the router has never seen it.
4. **A record instead of the asked quantity.** "What is the surface temperature of the Sun?" returns the whole solar record (5772 K is one field of
   ten). Correct data, not an answer: a reader has to find it, and the benchmark scorer counts it wrong.
5. **Nonsense clarifications from the fallback.** "A car going 60 mph brakes to a stop on dry asphalt. How far does it skid?" → the translator reads a
   slide down an incline and asks for `angle_deg`, offering 45°.

### S2 — reachability and the slim MCP

6. **The router has never seen 8 of the 224 tools it may pick**, including all three new data tools: `body_position`, `body_constants`, `small_body`
   (also `age_of_universe`, `mond_a0_constant`, `parse_quantity`, `eta_value_report`, `holographic_matching_mass`). The ephemeris work of
   2026-10-20…27 is unreachable from a plain question.
7. **The public cloud copy runs keywords only** (no model): 1 of 5 basic probes answered ("escape velocity of Earth", "density of copper", a Carnot
   efficiency and a derivative all end in `clarify`). This is the copy the Captain's Claude Desktop science thread reaches through the connector.
8. **The 3-tool MCP (929f85c) rests on a number that is mostly a benchmark bot.** Of 1,095 `/ask` calls in October's research log (my audit traffic
   excluded), **1,074 (98%) are the Wolfram trickle replaying the suites**; 21 are anything else (mostly projectile tests). The "97% of real calls are
   plain questions" figure measures the trickle, not people; Claude's own MCP use goes to the cloud copy and is not in this log. `call_tool` tells a
   client to "prefer `ask`", which steers the best front end we have (a frontier model) to the weaker router (home) or to keywords (cloud). Not
   measured yet: whether Claude answers better with the full list, a compact catalogue, or the 3 tools.
9. **`find_tools` is a substring match.** "integral" finds only `nonlinear_optics_analysis` (not `integrate_expr`); "skid" and any phrase ("temperature
   of the sun") find nothing.

### S1 — refusals: the right decision is thrown away

14. **Live adversarial set (2026-10-27, first daily live run): 1 of 6 refusal questions right.** The router declines "What is the kinetic energy of
    love?" with `{"decline": "not a question about a physical system"}`, which is the right answer; `router_stage` counts a decline as a miss, and
    the keyword translator then answers with its falling-object offer. "Is the Earth flat?" is routed to `body_constants(earth)` and comes back as
    the Earth's constants record, tagged VERIFIED. The stored (harness) adversarial run scores 6 of 6 because the harness's cascade has a refusal
    classifier that /ask does not have (finding 3b).

### S2 — a Mentat that cannot boot passes the deploy

15. **skippy-core's deploy smoke test never asks Mentat anything.** `core/smoke.py` checks the model slots, the Flask health, intent, matter and
    the car; a Mentat container that crash-loops at import passes. Seen 2026-10-27 09:21 UTC: another session's push carried my local commit
    (app.py importing `cascade_stage`, not yet in the image's COPY list) to the trunk, CI passed (the tests import by path), the deployer
    reported "deployed", and the live Mentat was down for six minutes until a131f67. Proposed (core/ is the Captain's): a `check_mentat` in
    smoke.py that waits for `GET 127.0.0.1:7350/health` to say `ready` and asks one known question. On the sigma-ground side,
    `mentat/tests/test_mentat_image_files.py` now fails before the push when app.py imports a module the Dockerfile does not copy.

### S3 — the dashboard measures the wrong system

13. **The daily job's TARGET DASHBOARD says PASS (91.3%, "sg matches WA-correct 30/30") while the live Mentat answers 105 of 150 and 0 of 329.**
    It re-scores a stored run (`results/sigma_ground_run.json`, dated 2026-07-15, one row re-run on 2026-10-16) of the benchmark harness's cascade on
    this PC: a different front end from the deployed `/ask`, and a run nobody has repeated against the live service. A dashboard that cannot go red
    when the product breaks is the reason finding 1 lived for 13 days.

### S3 — hygiene

10. Answers never carry the question's own units (`display` is null on every probe: "fall 100 feet" answers in seconds after converting to 30.48 m
    silently). Planned in the v3.2 data, not trained.
11. `uncertainty` is null on every answer probed, including constants that have a CODATA uncertainty.
12. Duplicate `MENTAT_ROUTER_MODEL` line in `core/compose.core.yml` (from the handoff; not checked further).

## Has Mentat veered off its goal?

The goal ([PROJECT_GOAL.md](../PROJECT_GOAL.md)): a local engine that contends with Wolfram|Alpha, **mathematics first**, then physics; every answer with
value, units, source, uncertainty, formula and the way the input was read; never silently wrong; the deployed system is what was measured; training last.

* **The parts are on goal; the product is not.** The tools, the provenance record, the math kernel and the router are strong: on the math suite the
  router path (measured in October) is right on 315 of 324 where Wolfram's free API is right on 52, and replayed with one switch on, the live code
  does 317 of 329. But a user who asks Mentat a maths question today gets nothing, and has since the front door was first measured (1 of 61 on
  2026-10-14).
* **Effort has gone where it does not reach the user.** Since 2026-10-20: the N-body forecast tables and shoot-outs (excellent physics, unreachable from
  `/ask`), readiness contracts, deploy pipelines, the slim MCP, status cards. Each is sensible alone; none was checked against "does a plain question get
  a right answer". The one measurement that would have shown it (the trickle's front-door line) exists and is not in the daily job.
* **The comparison with Wolfram is fair only on our own suites.** Wolfram returns no result on 225 of the 329 math questions, mostly wordy or
  Python-syntax templates it was never built for. The gate asks for a terse suite (≥ 500 maths, ≥ 300 physics one-liners: Wolfram's home turf);
  neither exists yet.

## Is it the best physics professor / Wolfram contender we can provide?

Not yet, and the gap is mostly wiring, not science. Measured order of the paths we already have, best first: a frontier model driving Mentat's tools
(the gate's ceiling, not yet measured), the v3.1 router with structured results, the deployed router (no maths), the keyword translator (cloud).
A professor also explains: Mentat returns a value, a formula and a source, but no worked line ("t = √(2h/g) with h = 100 ft = 30.48 m, g = 9.81 m/s²
→ 2.49 s"), which Wolfram keeps behind its paywall and which every result record already has the parts for.

## How to improve it, in order

| # | change | cost | what it fixes | how we know it worked |
|---|---|---|---|---|
| 1 | `router_stage`: run the harness in the measured profile (`trained` + `structured_results`) | one line + a real (unfaked) test | finding 1 | 317 of 329 in the replay; rerun the 329 through `/ask` after the deploy |
| 2 | log the router's miss reason in the research log (`inputs.router_miss`) and count misses in `/health` | small | finding 2: a silent fall-through becomes visible | the log names every miss |
| 3 | daily job: ask the LIVE `/ask` the 150, the 329 and the adversarial set (`tools/ask_suites.py`), score them with the unchanged scorers, and put those rows on the dashboard beside the stored run; red when a count drops | half a day | findings 2 and 13 | the dashboard goes red on a regression like this one |
| 3b | behind the router, the benchmark's own cascade (regex classifiers + semantic interpreter) instead of, or ahead of, the keyword translator | a day | the 31-answer gap between 109 and 140 on physics; the Sun-temperature fix of 1c0ff2f reaches users | the 150 through `/ask`, paired |
| 4 | deterministic intents ahead of the router for "where is X / right now / on <date>" → `body_position`, and "<field> of <body>" → that field | a day | 3, 4, 6 until the router is retrained | probes 3/4 answer with a position / a single value |
| 5 | a router DECLINE becomes /ask's answer (value null, the decline's reason as the refusal) instead of a miss that falls to the translator; the fallback says "I could not read this as a calculation" for anything it cannot place, instead of a falling-object or incline clarification | small | 5, 14 | the live adversarial refusal row; no `clarify` with an unrelated scenario |
| 6 | cloud copy: answer `ask` through the home router (tailnet) or say in the connector's instructions that `call_tool` is the way | small–medium | 7 | the 5 probes |
| 7 | measure Claude with the full list vs a compact catalogue (~5 K tokens: name + one line) vs 3 tools, 50 questions, before keeping the slim MCP | half a day | 8, 9 | paired counts |
| 8 | `find_tools`: word stems and synonyms (integral → integrate), ranking | small | 9 | the probe words |
| 9 | "show the work" line built from `formula` + `inputs` + unit conversions, no model | 1–2 days | professor mode | every router answer carries one |
| 10 | v3.2 training on the M5 (built; adds conventions and units-as-asked) + the 8 unseen tools in the data | days, GPU | 6, 10 | math suite, convention suite, hard validation vs v3.1 |
| 11 | the terse suites the gate asks for (≥ 500 maths, ≥ 300 physics), through the trickle | a week of Wolfram budget | the fair comparison | gate items 1–2 |
| 12 | frontier ceiling run (needs the Captain's spending cap) | API cost | the ceiling | gate item 4 |

Items 1 and 2 are DONE (SkippyTel f9fb119 then 0d62ade, deployed by skippy-core's CI + deployer, verified live: maths 0 → 317 of 329, physics 105 → 109 of 150).
Two lessons from deploying them:
* **The first version used the harness's `trained` profile** ("the profile the router was scored in") and lost 2 physics answers: `strict_params`
  refuses a plan naming a parameter the schema no longer lists, and the v3.1 router writes pre-rename names the server accepts as aliases
  (`joules_to_eV(energy_joules)`) and invented extras the tool ignores (`free_fall_time(initial_velocity_m_s)`). The configuration that was
  actually replayed (defaults + `structured_results`) is what shipped. The new `router_miss` field found this in one re-ask.
* **The live measurement was cut three times by stack recreations** (07:14, 07:20, 07:52, 08:03 UTC: `docker compose up --recreate` of every
  container, ~1 min of 502s from the model shim each time; two were deploys of other sessions' pushes, 07:52 was a `repair()` that the doctor's
  state file does not claim: the portal's Recovery or a hand-run). A live suite run must bracket itself with the process start (`/health.uptime_s`; NOT `/health.since`, which is the readiness
  contract's "last made progress" clock and moves on its own) or it measures the outage. Item 3 (the dashboard on the live service) needs the same guard.

Item 3b (the cascade) was measured as a replay of `/ask` on this PC (same code path as the box, the live router and chat slot, scorers unchanged,
rows paired per question; `misc/audit_2026-10-27/` holds the runners). `before` = the cascade runs ahead of the router, `after` = only when the router missed:

| variant | physics (150) | maths (329) | adversarial (18) |
|---|---|---|---|
| router only (C0) | 108 | 317 | 1 |
| cascade after the router | 121 (+13, −0) | 317 (+0, −0) | 7 (+6, −0) |
| cascade before the router, every stage | 133 (+26, −1) | **298 (+0, −19)** | 8 (+7, −0) |
| **cascade before, `math` stage off (shipped)** | **132 (+24, −0)** | **317 (+0, −0)** | **8** |

The harness's maths regexes ("integral of … from a to b", "factor …", "gradient of …") answered 27 of the 329 and got 7 right, against the router's
317; they took 19 answers the router had right: 18 of them are "Over the interval [a, b], what is the integral of …", which the stage reads as
an indefinite integral (it only knows "from a to b"), so it returns an antiderivative where a number was asked for. They also read
"are its emission wavelengths redshifted" as `factor_expression` (the one physics loss with every stage on, modern_012). So the deployed
configuration is `MENTAT_CASCADE=before`, `MENTAT_CASCADE_OFF=math` (SkippyTel fa4ffe0, `core/compose.core.yml`); every other stage only gained.
The ahead-of-router order wins by 11 on physics and the same on adversarial, and costs nothing once `math` is off. The maths stage's regexes are a
bug list for the router-free path, not part of this deploy. Live check: pending the deploy (the box's CI defers while the private Skippy is working).

## Reproduce

`misc/audit_2026-10-27/`: `ask_math_suite.py` (329 through a deployed `/ask`), `audit_replay.py` (deployed physics; local replay of `app._ask` in
variants D and S against the box's models), `replay_router.py`, `results.json` (every run's count), and the final live rows for both suites.
