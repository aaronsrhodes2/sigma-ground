# Code audit 2026-10-27: does sigma-ground serve its purpose?

The purpose is [PROJECT_GOAL.md](../PROJECT_GOAL.md): a local engine that contends with Wolfram|Alpha, maths first, then physics; every answer
carries value, units, source, uncertainty, formula and the way the input was read; never silently wrong; the deployed system is the measured one.
This audit scores the source against that purpose. Method: read the goal, architecture and ledgers; measure the tree (`git ls-files`, line
counts, census files); probe tools in-process and the LIVE `/ask` on skippy-core:7350; run the tier guard and the alias/body suites (938 passed). The
`tests/` directory did not finish in 25 minutes on today's loaded machine (0.5 GB free RAM) and was stopped; the full suite was not run. Numbers for the suites are the daily live rows of 2026-10-27 (physics 134/150, maths 317/329, adversarial 8/18), not re-run.
Companion: [MENTAT_AUDIT_2026-10-27.md](MENTAT_AUDIT_2026-10-27.md) (the product audit, 15 findings, improvement list) and the quarksum merge
(commit f019fda, `misc/quarksum_legacy/README.md`).

## Scorecard (10 = fully serves the goal)

| # | dimension of the goal | score | the evidence |
|---|---|---|---|
| 1 | Right answers on the suites, live | **8** | maths 317/329, physics 134/150 through the deployed `/ask`; the router answers a one-tool question in ~0.5 s; the trained 4B router beats every untrained baseline by a paired test (misc/BASELINES.md). Adversarial 8/18 is the weak row. |
| 2 | Every answer carries value, units, source, uncertainty, formula, interpretation | **5** | formula and `interpreted_as` are on every router answer (probed live). But `lookup_constant("speed_of_light")` returns `units: ""`, `uncertainty: null`, `provenance_tag: null`, live and in-process: `_lookup_scipy` takes the module-attribute branch, which has no units, and never reaches the CODATA dict branch three lines below that has value, unit and uncertainty. The previous audit's finding 11 (uncertainty null everywhere) is still open. |
| 3 | Never silently wrong | **6** | the contract census is enforced by a test with ceilings that may only fall (`notation_wrong 0`, `undescribed_params 1` from 471); the maths kernel reads `x²`, `√x`, `|A|`. Against: 10 of the 18 adversarial questions get an answer instead of a refusal, which is the definition of silently wrong. |
| 4 | Deployed = measured | **8** | `tools/check_mentat_deploy.sh` works and reported the live pin 3 commits behind master today; `benchmark/live_ask.py` measures the live service daily and brackets by process start. The pin must still be bumped by hand in another repo. |
| 5 | Data alone, no speculation served as model | **7** | DISPROOF_LEDGER keeps the record; 50 `PHYSICS_GAP`/`DATA_GAP` tags in code and 25 entries in KNOWN_GAPS, 4 closed with the fix named. Against: `field/constants.py` still opens "SSBM Constants — the three numbers that define Scale-Shifted Baryonic Matter" and line 411 states K(σ) = K₀·e^σ as fact; `inventory/physics.py` calls entanglement "the mechanism"; 41 code files mention SSBM. Docstrings are what a reader (and a model) learns from. |
| 6 | Architecture and code health | **7** | six tiers, enforced by an AST test (it had been red for two days from c967cda; fixed 466187b). Core library is genuinely zero-dependency (`dependencies = []`, the MCP extras carry scipy/pint/sympy/astropy). Tests are large: field 61k code / 39k test lines, radiance 14k / 31k. Against: 46 modules still import through the compatibility shims the architecture says are for old code; four files over 2,000 lines (`materia/scenarios.py` 3,591, `radiance/trajectory.py` 3,203, two benchmark files); 46 library modules print to stdout; `blueprint/` has no tests. |
| 7 | Repository hygiene | **4** | README says "1198 tests passing" (baseline is 5,026; last touched 2026-07-18); CLAUDE.md claimed three siblings were archived when none was (fixed today); 7 of 8 side branches merged and undeleted; a merged worktree with 11 dirty files; 87 MB of tracked data (fixtures 37, misc 27, viewer data 23) and scratch outputs under version control (`misc/_audit2.txt`, `*_output.txt`, `.winpid`); 55 uncommitted files from other sessions in the working tree, including scene JSONs the architecture says tests never write. |
| 8 | Effort on the goal (the contender gate) | **5** | gate items open: the terse suites (≥500 maths, ≥300 physics) do not exist, so the Wolfram comparison is on our own wordy questions where Wolfram returns nothing on 225 of 329; the frontier ceiling is unmeasured; the conflict report is not wired to the gate. Files touched in the last 30 days: `stage/gmod` 461, `misc/unification` + `scripts/unification` 464, `field` 497, `mcp` 235: more than half of the work was on the holodeck and the unification programme, which the goal lists as roadmap or not at all. |

**Overall: 6.5 / 10.** The engine is better than the product. The tools, the maths kernel, the router, the census and the live measurement are
a real contender on the questions we wrote; the record each answer carries is incomplete on the simplest case (a constant), the refusal
behaviour is below half, the fair comparison has not been built, and the repository tells a visitor things that are not true.

## Defects found by this audit (new, not in the 10-27 product audit)

1. **`mcp/tools/constants.py::_lookup_scipy` drops units and uncertainty for every CODATA constant scipy exposes as a module attribute**
   (c, h, k, e, m_e, ...). The branch order is the cause: the attribute branch returns first with `units=""`; the CODATA dict branch
   (`spc.value/unit/precision`) is never reached for those names. Fix: look the name up in `physical_constants` first, fall back to the
   attribute only for names the dict does not have, and map the scipy unit string through the contract's unit normaliser. Verified live.
2. `tests/test_layering.py` was red on master for two days (fixed 466187b): the guard is only a guard if the suite is run before a push.
   The daily job could run this one test (24 s) and go red.
3. README test count and date are stale; ARCHITECTURE's baseline is 2026-10-10.
4. Tracked scratch files in `misc/` (`_audit*.txt`, the `bh_phase_*_output.txt` set) and two `.winpid` files.

## What is sound and should not be touched

The tier contract, `mcp/contract.py` and its census test, the DISPROOF ledger, `KNOWN_GAPS.md` with regression sentinels, the live measurement
in the daily job, `tools/check_mentat_deploy.sh`, and the zero-dependency core. These are the parts that make the goal checkable.

## Order of work, by effect on the score

| # | change | moves | cost |
|---|---|---|---|
| 1 | constants: CODATA dict first (defect 1); then `uncertainty` and `provenance_tag` on every constant answer (finding 11) | row 2 from 5 to 8 | hours |
| 2 | refusals: the adversarial set to ≥ 15/18 (improvement item 5 of the product audit) | row 3 | a day |
| 3 | the terse suites through the trickle, then the paired comparison | row 8, the gate | a week of Wolfram budget |
| 4 | remove SSBM framing from docstrings in `field/constants.py`, `inventory/physics.py` (keep the `sigma=` parameter and its "σ = 0 is standard physics" note) | row 5 | hours |
| 5 | README and ARCHITECTURE baselines; delete merged branches; untrack scratch files; move large fixtures to `local-cache/` or Git LFS-free download | row 7 | hours |
| 6 | retire the shims: 46 import sites to canonical paths, then a layering-style test that fails on a new shim import | row 6 | a day |
