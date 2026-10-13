# sigma-ground — Project Instructions

**This is the one canonical repo.** If you are looking at quarksum,
sigma-ground-mentat, matter-shaper, sigma-api, or ssbm-theory, stop — you're
in a dead end. See "Sibling directories" below before spending tool calls
exploring any of them.

**First time here?** Read [PLATINUM_RULES.md](../PLATINUM_RULES.md) — this
project inherits all universal rules there. Package/tier map: see
[ARCHITECTURE.md](ARCHITECTURE.md). Physics coding rules: see
[GOLDEN_RULES.md](GOLDEN_RULES.md).
See also: [LOGGING_STANDARD.md](../LOGGING_STANDARD.md) — format for session logs and operatic scene files.

## Sibling directories — what happened to them (read once, don't re-discover)

| Directory | Status | Why it's not canonical |
|---|---|---|
| `quarksum/` | Archived to `../_archive/quarksum-legacy/` | Was a parallel fork of this repo that stalled in 2026-06; its own structure was already folded into `sigma_ground/{inventory,field}` before it stopped. No unique code. |
| `sigma-ground-mentat/` | Removed | Was a stale `git worktree` of *this repo*, not a separate project — frozen a month behind master. Anything unique (`run_gemini.py`, a `GOLDEN_RULES.md` fix) was extracted before removal. |
| `matter-shaper/` | Archived to `../_archive/matter-shaper-legacy/` | Its renderer was folded into `sigma_ground/radiance/entangler/` (2026-06-03); the rest were stale prototypes of roles this repo already rebuilt (`radiance/web/`, `deckard/`). |
| `sigma-api/` | Archived to `../_archive/sigma-api-legacy/` | An early prototype of this repo's own MCP/Mentat layer ("Nagatha") — confirmed fully redundant before archiving. |
| `ssbm-theory/` | Stays separate (own repo now) | Unverified SSBM black-hole-merger research, extracted from `field/` but never merged back and currently broken in isolation. Not ready to fold in. |

## Session Persistence

Maintain `misc/SESSION_LOG.md` (append after each session) and create `misc/OPERATIC_PLAY_SCENE[N]_[TITLE].txt` files (prose narratives of discoveries). Scenes rotate into `../operatic-archive/` once a project's count gets large — see that repo for scenes 1-22; `misc/` here holds only the current, not-yet-rotated ones.

See [LOGGING_STANDARD.md](../LOGGING_STANDARD.md) for templates and formatting guidelines.

## Project Context

Package/tier map, testing baseline, and the compatibility-shim table all
live in [ARCHITECTURE.md](ARCHITECTURE.md) — that's the single source of
truth for repo structure; don't duplicate it here.

**MCP Server — Mentat**
- Mentat is the umbrella brand for this whole stack; its MCP server (`mcp/`)
  is its public face, exposing every service's tools to LLM clients.
  ("Nagatha" was an earlier, unrelated prototype attempt at this same role —
  now archived as `sigma-api`, see the sibling-directories table above. Do
  not use that name for this repo's own MCP server.)
- When asked to produce a **simulation**, **test**, **experiment**, or
  **scene**, use the MCP tools instead of reasoning about the physics
  yourself — Mentat runs the real code; its results are authoritative.
- **Diagnostics:** Mentat flags suspected bugs (bad sigma defaults,
  earth-sigma violations) but NEVER fixes them. Report findings to user for
  review.
- **Scientifically honest:** Mentat reports what the code actually does,
  not what it should do. If a function produces a wrong value, the test
  captures that wrong value and flags it.

## Parallel work policy — subagents first, worktrees rarely

Default to in-session subagent fan-out for parallel work — investigate one
thing while implementing another, or run independent research threads at
once — inside this one worktree. Do not create a new git worktree per
parallel task; that produced quarksum's 5-worktree/9-branch sprawl and
sigma-ground-mentat's stale, 112-commits-behind fork, both since retired.

A new worktree is justified only when the work is genuinely long-running and
needs to survive independently of this session. If you create one, name the
worktree directory to match its branch name — mismatches cost the next
session a double-take every time. **Cleanup trigger:** the moment a
worktree's branch merges into master, remove the worktree and delete the
branch in the same sitting — don't leave it "for later."

**Long jobs get a watchdog (the Captain, 2026-10-12).** Any job expected to run more than ~20 minutes is launched with a driver that retries failed workers and
logs exit codes, and with `scripts/jobwatch.py` attached as a tracked background command (`run_in_background`), so the session is woken the moment the driver
dies, a worker logs a traceback, no progress is made for the stall limit, or the job finishes. Silent deaths have cost hours three times. Presets:
`--preset shootout` and `--preset fair`; state and event log in `<state-dir>/<name>.status.json` / `.jsonl`.

**What counts as physics here (the Captain, 2026-10-11): data alone.**
- A claim stays in the model only if it can be disproved and has survived the test. A disproved claim is not part of the model; at most it is
  speculation about ourselves.
- The σ-field guesses and SSBM theory were put through that test and removed. The record of what was tested and what the data did is
  [misc/DISPROOF_LEDGER.md](misc/DISPROOF_LEDGER.md); the removed documents are in git at tag `pre-strip-2026-10-11`.
- The `sigma=` plumbing in the code stays (Captain, 2026-10-11): at σ = 0 it reproduces standard physics, and the goal is to make σ properly variable.
  It is a parameter, not a claim; nothing in the model asserts a value of σ or a mechanism that sets it.
- Do not add numerology (formula searches over constants). Do not describe an untested idea as a result.
