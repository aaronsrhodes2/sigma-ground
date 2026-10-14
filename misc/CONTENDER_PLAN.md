# Mentat, the local Wolfram contender — plan

The Captain, 2026-10-13: a **local** system that contends with Wolfram|Alpha in **physics and math**. Work out every
kink in the engine and its contract first; **training is the last step, after we are sure we have a contender.**
Evidence for every item below: [MENTAT_AUDIT_2026-10-13.md](MENTAT_AUDIT_2026-10-13.md) (findings F1–F16).
Earlier design notes this plan absorbs: [DEFAULTS_AND_RANGES_DESIGN.md](DEFAULTS_AND_RANGES_DESIGN.md),
[MENTAT_TWEAKS_AUDIT.md](MENTAT_TWEAKS_AUDIT.md), [ROUTER_V31_PLAN.md](ROUTER_V31_PLAN.md).

## The shape of the system (why this can win)

Wolfram|Alpha is a language front end that turns a question into a precise query, plus a deterministic engine and
curated data. Mentat has the same shape: a small local model translates, Mentat computes, the data carries its
sources. The model never does arithmetic and never supplies a measured value. Everything that can be made exact,
testable and shared by every client (Claude over MCP, SkippyTel, the router) lives in Mentat, not in the weights.
Where we can beat Wolfram: conversational word problems (its parser gives up; ours is a language model) and
multi-step chains, without giving up its precision.

## What the research says, and what we take from it

| finding | source | what we do |
|---|---|---|
| Tools for agents need unambiguous parameter names, strict input models, errors that say how to fix the call with a correct example, descriptions written "as for a new hire", consolidated overlapping tools, and evaluation on held-out tasks | [Anthropic, Writing effective tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents) | W2: parameter specs, uniform errors, generated descriptions, overlap review |
| Wolfram's LLM API returns an **input interpretation** and **assumptions**; the caller re-sends with a chosen assumption instead of rephrasing; named constants by name; one property per call | [Wolfram\|Alpha LLM API docs](https://products.wolframalpha.com/llm-api/documentation) | every result echoes how the input was read; defaults come back as `assumed`; ambiguity becomes listed alternatives, not a question |
| When an argument is missing, models fill it "from their internal model of the world" | [HammerBench](https://arxiv.org/pdf/2412.16516) | W2: Mentat fills missing values from its own tables and says so; the model is trained to leave them out |
| Similar APIs are a leading cause of tool errors; insufficient calls are the most common error | [ToolScan](https://arxiv.org/pdf/2411.13547) | W2: review and merge overlapping tools (F12) |
| A small model shown irrelevant tool descriptions gets confused; retrieving only the relevant tools (ToolRAG) raised accuracy enough for 1.1B/7B models to rival GPT-4-Turbo on their task | [TinyAgent](https://arxiv.org/abs/2409.00608) | W5: retrieve a short tool list per question (we already run `nomic-embed-text`) |
| Function-name surface cues mislead; masking names and training on irrelevance makes small models robust | [Hammer](https://arxiv.org/abs/2410.04587) | W7 (training): mask names in part of the data; keep the decline set |
| Strict JSON-mode decoding can lower reasoning accuracy; reason first, format after, repairs work | [Let Me Speak Freely?](https://arxiv.org/abs/2408.02442) | W5: measure free vs constrained plan decoding; validate plans after generation |
| Execution-verified voting: run several candidate solutions, drop those whose execution disagrees, vote on the rest (the authors report up to +18 points on GSM8K for 0.5B–13B models) | [PROVE](https://arxiv.org/abs/2410.12608) | W5: sample N plans, execute, vote on results; disagreement lowers confidence and surfaces alternatives |
| Tool-integrated reasoning (the model reasons, a solver computes) is what lifted open 7B–34B models on MATH | [ToRA](https://arxiv.org/abs/2309.17452) | confirms the split; W6 tests a reasoning front end against our translator |
| Dimension and unit checks, conservation and limit tests in a propose–check–revise loop | [CMPhysBench](https://arxiv.org/pdf/2508.18124), [PhysicsEval](https://arxiv.org/abs/2508.00079) | W5: a dimensional check on every result before it is shown |
| An open physics model exists: **P1** (RL on olympiad physics; 235B gold at IPhO 2025, 30B-A3B silver); small reasoning models are strong at math (Phi-4-mini-reasoning, 3.8B) | [P1](https://arxiv.org/abs/2511.13612), [Phi-4-mini-reasoning](https://arxiv.org/abs/2504.21233) | W6: they are baselines we must measure, alone and with Mentat as their tools |

## The contender gate (all must hold before any training)

1. **Math.** On a terse-math suite of at least 500 questions (the held-out suite plus Wolfram-style one-liners),
   final-answer accuracy at least Wolfram's on the same questions, measured through the Wolfram API; and **zero
   silently wrong answers** on the notation suite (a parse Mentat is unsure of is refused or shown as an assumption,
   never answered as something else).
2. **Physics.** At least Wolfram on (a) the 150-question narrative benchmark, (b) a new terse physics suite of at
   least 300 (Wolfram's home turf), (c) the adversarial set; compared question by question with a paired test, not
   by totals.
3. **Data.** Every number cites a source and conditions; no question gets two different answers by two paths
   (the conflict report is empty).
4. **Ceiling and baselines measured.** A frontier model driving Mentat (the ceiling), P1-30B-A3B,
   Phi-4-mini-reasoning and stock Qwen3 (thinking) with and without Mentat. If a reasoning front end plus Mentat
   clearly beats translator plus Mentat, the front end changes before we train.
5. **Local and fast.** Median answer time on this PC within a budget we set from the baselines (target ≤ 5 s for a
   one-tool question).
6. **Deployed is what we measured.** The live Mentat reports the commit it runs, and it is master.

## Workstreams, in order

**W0 — prerequisites (day 1, short).**
Merge `claude/router-v2` into master (41 commits: structured results, the tweak registry, the numpy-hang fix in
`server.py`; master is 138 ahead) — *needs the Captain's go*. Fix the layering violation (F14) and wire or ledger
the 13 uncovered functions (F7) so the suite is green. Version stamp: every result and `get_manifest` carries the
git commit; a deploy check compares the container to master (F3).

**W1 — the math kernel (day 1–2, first real work).**
One parser for every expression input (`symbolic.py`, `mathx.py`, and any other tool taking an expression):
implicit multiplication (`2x`, `(x+1)(x-1)`), `^` and unicode powers (`x²`), `√`, `π`, `∞`/"infinity", `|x|`,
`sin x`; `e` is Euler's number and `i` the imaginary unit unless declared as a variable; the variable is inferred
when there is one free symbol; any free symbol the caller did not declare is reported. Every math result carries
`interpreted_as` (Wolfram's "input interpretation") so a misreading is visible. A notation suite (each notation ×
each tool, expected meaning) plus property tests (random expressions rendered several ways must parse equal). Then
breadth against Wolfram's math: what one-liners it answers that we cannot (step-by-step, plots, number theory,
statistics) — a measured list, built only where the gate needs it.

**W2 — the tool contract (day 2–5).**
A spec per parameter: unit, bounds (Golden Rule 2's SAFE/EDGE/WALL/BEYOND from `field/bounds.py`), meaning, example,
and for physics inputs a default or a derivation (`DEFAULTS_AND_RANGES_DESIGN.md`). From the specs: quantities with
units accepted everywhere (`"54 km/h"` → 15 m/s through pint, the conversion echoed in `inputs`); descriptions
generated with Args, units and an example (F9); one naming convention (`material`, `material_a`/`material_b`,
SI suffixes; old names kept as aliases for a deprecation period) (F10); one error contract — `ok: false`, a code,
the parameter, what was expected, a corrected example (F11); an `assumed` block for every default used. Overlaps
reviewed and merged where they are true duplicates (F12). A contract test runs the census on every commit.

**W3 — one materials database (day 4–8).**
One canonical vocabulary and alias table shared by the lookup tools, the resolver, Materia and quarksum (F2, F6).
Each value carries source, conditions (temperature, pressure, wavelength) and uncertainty; a precedence rule
(handbook at stated conditions > TPSX ranges > derived); a conflict report that must be empty. Class names get a
disclosed default ("steel" → carbon steel, with stainless as the listed alternative). Tools: `material_property`
and `material_profile` over the one database. Then `object_properties(material, mass, shape)` — geometry from
density, quarksum's census, the material's own tables — so "a 0.145 kg iron ball" is fully specified.

**W4 — the sky, from the shoot-out (day 6–10).** (The Captain: open the n-body physics to Mentat.)
`body_position(body, when, view)` from `fast_ephemeris` (RA/Dec, distance, light time, its measured uncertainty;
planets and the Moon); a `precise` mode through the DE440-parity N-body engine; `moon_phase`, Moon distance,
separations, next opposition / conjunction / greatest elongation by search over the fast ephemeris; named-asteroid
positions from the SBDB fixtures. Formula tools the engine already implements: perihelion precession split into
relativistic and planetary parts, J2 nodal precession (sun-synchronous orbits), tidal and GW-inspiral times. Dense
matter from the 10-11 work: white-dwarf mass–radius (Chandrasekhar limit), polytropes, neutron-star structure. Each
validated against DE440 / JPL Horizons or a textbook value, with the error it reports.

**W5 — the precision layer, no training (day 8–12).**
Tool retrieval (a short list per question); plan validation against the specs before execution; execution-verified
voting over N sampled plans; a dimensional check on every result; the answer shows the interpretation and the
assumptions; low agreement shows the alternatives instead of one guess. Each piece is measured on the suites and
kept only if it helps.

**W6 — the yardstick (in parallel from day 2).**
The Wolfram head-to-head on the math suite and a new terse physics suite (Wolfram API, captured once and frozen
like the parity cases); the baselines and the frontier ceiling (gate 4); one scoreboard with paired statistics;
latency per answer. The 150-question benchmark stays, but it is home turf for us and not enough on its own.

**W7 — training, last.**
Only when gates 1–6 hold for the untrained system plus the best front end. Relabel the data to the final contract
(no clarification for missing inputs, quantities quoted with units, lookups for every measured value), retrain once,
re-run the gate.

## Tonight and day 1

* Tonight (scheduled before this plan): router v3.1 4B trains at 00:00 as a **measurement** of the generated math
  data. It does not change this plan; its model is not the contender.
* Day 1: W0 (after the Captain's go on the merge), then W1 — the parser, the notation suite, `interpreted_as` —
  with the census script moved into `tools/` as the first contract test.

## Decisions for the Captain

1. **Merge `claude/router-v2` into master** so the contract work starts from one tree?
2. **The stripped theory on the public face (F15)**: remove `entanglements_to_pop_bubble`,
   `holographic_matching_mass`, `baryon_vs_disc` and the "SSBM" EXTENDED tier, or reword them as plain arithmetic?
   Reword "pending SSBM re-derivation" in the n-body docs?
3. **PROJECT_GOAL.md** says "not a Wolfram replacement on symbolic-math breadth". Update it to the contender goal
   (physics and math, local)?
4. **Wolfram API for the head-to-head**: the parity cases used the Wolfram MCP; the suites need a few thousand
   queries captured once. Free tier or your key?
5. **Baselines download**: P1-30B-A3B (a 30B MoE; about 18 GB at 4-bit, partly on the CPU) and
   Phi-4-mini-reasoning (3.8B). Licence check before download, as usual.
6. **Frontier ceiling**: running Sonnet or Opus as the front end on the suites costs API calls. Yes?
