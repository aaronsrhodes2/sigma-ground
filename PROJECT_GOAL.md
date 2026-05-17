# sigma-ground / quarksum — Project Goal

## Mission

**Build a physics professor assistant.**

It must do two things well:

### 1. One-liner Q&A (Q&A mode — current benchmark)

Answer a standalone physics question in plain English with a single
authoritative numerical answer, fully cited.

**Accuracy target: within 98% of Wolfram Alpha's accuracy on the same
question set.** Wolfram is the canonical "right answer" reference for
the topical breadth a physics undergrad would ask. We don't need to
beat it on math; we need to match it on numbers and beat it on
natural-language phrasing.

What "within 98% of WA" means concretely:
  - For any question WA can answer correctly, we want to answer it
    correctly too (within the stated tolerance band of the question's
    ground truth).
  - For questions WA cannot parse (its NLP gives up on conversational
    English), sigma-ground should still answer correctly — that's
    where the differentiation is.
  - On the 150-question physics-major curriculum, target ≥98% × max(WA, 1)
    correct.

### 2. Conversational physics (conversation mode — roadmap)

Hold a casual back-and-forth physics conversation in natural human
language, the way one would talk with a knowledgeable colleague.
Persisted state across turns, with the sigma-ground + quarksum
simulation engine as the playground.

What this requires (not yet built):
  - Persisted conversation state across turns.
  - Stateful tools that mutate a simulation (apply force, evolve a
    system, change a boundary condition) — distinct from the current
    immutable Q&A tool surface.
  - Conversation-mode system prompt distinct from the Q&A switchboard
    prompt.
  - A way for the user to switch modes ("let's simulate this" vs
    "what's the answer to ...").

See `memory/project_mcp_modes.md` for the architectural distinction.

## What backs the assistant

  - **sigma-ground** — the curated physics library. Constants,
    formulas, named bodies, named stars, materials, with full
    provenance on every value. Every tool returns
    `(value, units, source, uncertainty, provenance_tag)`.
  - **quarksum** — the simulation engine. Real-matter physics, σ-field,
    SSBM scale dynamics, n-body with full GR corrections. Used by
    conversation mode as the playground.
  - **An LLM front-end** (currently qwen2.5:14b via Ollama) — the
    translation layer between user language and library/simulation
    calls. NOT the source of physics knowledge.

## What this assistant is NOT

  - Not a Wolfram Alpha replacement on symbolic-math breadth (we don't
    do step-by-step integrals or arbitrary plots).
  - Not an LLM tutor (we don't generate explanations beyond what tools
    return). We can describe what we computed, not derive new physics.
  - Not an interactive proof assistant.

## Success metrics

| Metric | Where measured | Target |
|---|---|---|
| Q&A numerical accuracy | 150-Q corpus + daily_job | ≥98% of Wolfram on same Qs |
| Q&A coverage of physics-major curriculum | corpus domains | 14 domains, all ≥80% |
| Q&A latency (median per question) | run_sigma_ground.py | ≤30 s |
| Library-gap rate ("Fitted due to incompetence") | IMPROVEMENT_PLAN.md | ≤5% of failures |
| Conversation mode (to be built) | future eval | initial: 1-turn handoff |

## Current state (as of 2026-05-17)

  - Q&A baseline: sigma-ground 28% / Wolfram 13.3% / Gemini-flash partial 63%.
  - 5 systemic bugs identified and fixed (param aliases, body chain,
    rad/s unit handling, multi-tool fallback, K↔C conversion). Rerun in flight.
  - Daily job + IMPROVEMENT_PLAN.md established.
  - Conversation mode: not started.

## How this document is used

  - When making a design decision, ask "does this serve the goal?"
  - When adding a tool, ask "does it close a gap that the daily_job
    surfaced, or is it speculative?"
  - When the conversation mode lands, this document gets an update
    section, not a rewrite.
