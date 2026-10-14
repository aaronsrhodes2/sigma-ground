# sigma-ground / quarksum / Mentat — Project Goal

## Mission (the Captain, 2026-10-13)

**Build a local computational-knowledge engine that contends with Wolfram|Alpha, in mathematics and physics first.**
It runs on this PC, a small local model translates, Mentat computes, and every value carries its source.

Order of conquest: **mathematics first** (the most immutable ground), then **physics**, then **chemistry**, then **biology**.
A domain is not "done" until it passes the contender gate below. Training the front-end model is the **last** step, after the gate
holds for the untrained system (`misc/CONTENDER_PLAN.md`).

It must do two things well:

### 1. One-liner Q&A (Q&A mode — the benchmark)

Answer a standalone question — terse (`integrate x^2 sin x`, `density of copper`) or conversational ("how long does a dropped ball take to fall
10 meters?") — with an authoritative, cited answer and the way the input was read.

Wolfram is the reference for accuracy on short, well-formed queries; a language-model front end is where we can beat it, on conversational
phrasing and multi-step chains, without giving up its precision. What we owe on every answer:
  - the value, its units, its source, its uncertainty, and the formula used;
  - how the input was interpreted (and, where an input was missing, what was assumed and the listed alternatives);
  - never a silently wrong parse: a reading Mentat is unsure of is refused or shown, not answered as something else.

### 2. Conversational physics (conversation mode — roadmap)

Hold a casual back-and-forth physics conversation in natural language, the way one would talk with a knowledgeable colleague.
Persisted state across turns, with the sigma-ground + quarksum simulation engine as the playground.

What this requires (partly built — see the playground and front-door tools):
  - Persisted conversation state across turns.
  - Stateful tools that mutate a simulation (apply force, evolve a system, change a boundary condition).
  - A conversation-mode system prompt distinct from the Q&A switchboard prompt.
  - A way for the user to switch modes ("let's simulate this" vs "what's the answer to ...").

## What backs the assistant

  - **sigma-ground** — the curated physics library. Constants, formulas, named bodies, materials, with provenance on every value. Every tool
    returns `(value, units, source, uncertainty, provenance_tag)`; a result also says how its input was read.
  - **quarksum** — the particle inventory and mass-closure engine (materials → molecules → atoms → particles → quarks), plus the N-body engine
    (relativistic, tidal, zonal, radiation) validated against JPL DE440. Used by conversation mode as the playground.
  - **Mentat** — the MCP server, the public face: every tool, one contract (quantities with units, bounds, uniform errors, assumptions disclosed).
  - **A local LLM front end** (a small model, Qwen3 today) — the translation layer between user language and tool calls.
    NOT the source of physics or arithmetic: Mentat computes every number (PLATINUM §21).

## What this assistant is NOT

  - Not an LLM tutor: it describes what it computed; it does not derive new physics.
  - Not an interactive proof assistant.
  - Not a source of speculative physics: hypotheses that have not survived a test are filed in `misc/DISPROOF_LEDGER.md`, served only on explicit
    request, and always labelled.
  - (Step-by-step solutions and plots are not goals today. If the contender gate shows Wolfram's lead there matters, they join the plan.)

## Success metrics — the contender gate (all must hold before any training)

| Metric | Where measured | Target |
|---|---|---|
| Math, terse suite (≥500 questions) | `misc/CONTENDER_PLAN.md` gate 1; Wolfram free-tier trickle + historic data | final-answer accuracy ≥ Wolfram's on the same questions |
| Math, silently wrong answers | notation suite (`test_mathparse.py`, contract census) | **zero** |
| Physics, narrative benchmark (150) + terse suite (≥300) + adversarial set | daily_job; trickle | ≥ Wolfram, compared question by question (paired), not by totals |
| One answer per question | material / constant conflict report | empty |
| Ceiling and baselines | frontier model driving Mentat; P1-30B-A3B; Phi-4-mini-reasoning; stock Qwen3 | measured and recorded |
| Latency | harness | median ≤ 5 s for a one-tool question, on this PC |
| Deployed = measured | `tools/check_mentat_deploy.sh` | the live Mentat runs master |

## Current state (as of 2026-10-13)

  - 232 tools; 211 of 215 replayed tools return the full provenance record. Audit and plan: `misc/MENTAT_AUDIT_2026-10-13.md`, `misc/CONTENDER_PLAN.md`.
  - Narrative benchmark (150): the v3 4B router scores 140 with its rule cascade (109 without it) against Wolfram's 30 as asked / 56 rephrased.
  - Held-out math suite (329): the v3 4B router gets the right answer on 194; the notation audit found ordinary writings (`x²`, `√x`, `e^x`) read wrongly — the
    first thing being fixed (the math kernel).
  - Next: the math kernel, the tool contract, one materials database, the sky tools, the precision layer, the yardstick, then training.

## How this document is used

  - When making a design decision, ask "does this serve the goal?"
  - When adding a tool, ask "does it close a gap the benchmark or the contender gate surfaced, or is it speculative?"
  - Domains are added in order; this document gets an update section when a gate is passed.
