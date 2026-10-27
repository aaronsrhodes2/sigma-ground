# sigma-ground and the N-body experiment: handoff (written 2026-10-27)

Everything done on sigma-ground's N-body proof and the engine around it, as of 2026-10-27, written to be read cold. The Mentat side (the physics MCP) has its own handoff: [MENTAT_HANDOFF.md](MENTAT_HANDOFF.md). Per-claim detail lives in [CANDIDATE_LEDGER.md](CANDIDATE_LEDGER.md) (what survived tests, C1-C17) and [DISPROOF_LEDGER.md](DISPROOF_LEDGER.md) (what did not).

## 1. What the experiment is, and the rules it runs under

**The question.** Does our N-body engine, started from JPL DE440's own state at an epoch, reproduce DE440's trajectories over the next 1 to 20 years, body by body? DE440 is the answer key.

**Wording discipline (the Captain).** "Dynamics parity with a quantified deficit." Every window STARTS from DE440's state, so this is not a forecast contest and not a claim to beat JPL. Failures go to the disproof ledger, successes to the candidate ledger.

**Standing rules that shaped the work**
* Data alone: a claim stays in the model only if it can be disproved and has survived a test (2026-10-11). The sigma-field guesses and SSBM theory were removed (tag `pre-strip-2026-10-11`); `sigma=` plumbing stays as a parameter, at sigma = 0 it is standard physics. No numerology.
* Past data is never changed. A new result gets a new label/fixture beside the old (fixtures r2 ... r8, predictors `restated_*`).
* Mentat computes physical numbers; the agent designs and judges (PLATINUM §21). Bash first (§20). D: not C: for large outputs (§22).
* **Keep calculations as calculations** (the Captain, 2026-10-26): GM, mass = GM/G, periods, perihelion are computed when asked, never stored; a stored table is a cache that must detect staleness (physics fingerprint).
* Long jobs (> ~20 min) get a retrying driver and `scripts/jobwatch.py` as a tracked background command. Show exact requests before any network fetch (Horizons rule). Single-agent flow (no parallel subagent workflows). Do not push/deploy without approval (pushes to the house trunk and the Mentat rollout were approved 2026-10-27).
* Fix the cause of repeated shell/script errors and record it in `.claude/skills/windows-shell/SKILL.md`.

## 2. The model

* **Engine:** `sigma_ground/field/interface/nbody.py`, Forest-Ruth 4th-order symplectic, full EIH relativity, zonal harmonics (J2-J4), tides, solar pressure, sectoral lunar C22 with the IAU rotation model. Step matters: 0.25 d is far too coarse for Mercury and the Moon (a forecast built at 0.25 d was 102 km and 259 km off and was rejected); **0.025 d is the shoot-out step** (the Moon is converged there), 0.0125 d the converged arm.
* **Reduced model** (library `reduced_model.py`, factored out of `scripts/fair_jpl_contest.py`): Sun, Mercury, Venus, Earth, Moon, Mars; Jupiter, Saturn, Uranus, Neptune, Pluto as system barycentres. Validated against the full shoot-out (identical without layers to 1e-6 km, 2e-4 km with layers; the 0.17 m Moon difference is unexplained, the perturber horizon was refuted as the cause).
* **Layers added from predictors** (`rolling_shootout.py`): prescribed sources (DE440's 343 asteroids, a 36-point ring at 44 au carrying 8917 km^3/s^2, 7 identified Kuiper-belt objects), the Moon's C22, measured Jupiter/Saturn zonals.
* **Full shoot-out** (`scripts/run_shootout_parallel.py`, `rolling_shootout.py`): 56 bodies (planets, Moon, 54 other moons, as bodies shifted with their systems), many start windows, leads +1 to +20 yr, per body and lead median heliocentric position error.

### Fixture lineage (`sigma_ground/field/interface/fixtures/de440_state_vectors_v2physics*.json`)
| label | what |
|---|---|
| original / r2 | the baseline; planet-centre velocities from Horizons planet centres (inconsistent with DE440 barycentres) |
| r3 | planet-only shift; SUPERSEDED (broke the moons) |
| r4 | each planet system moved rigidly onto DE440's barycentre, relative states kept (C8) |
| r5, r6 | + Himalia, Nereid, Helene; + more moons |
| r7 | original + 23 Horizons moons in the Horizons frame, systems shifted rigidly over ALL members |
| r8 | r7 + Neptune and Triton from the 2026-10-23 Horizons fetch (Triton's truth track moves ~100 km / 1.3 m/s); 56 bodies |

### Predictor labels (never edited once published)
`restated_ast_ring_kbo_c22` (the control: 343 asteroids + ring + 7 KBOs + C22, table zonals), `restated_zon` (+ measured Jupiter/Saturn zonals: Juno, Cassini), `restated_hdr` (header Sun/Earth/Moon constants; its effect test is RUNNING, see section 6).

## 3. Results, in order (full numbers and file names in the ledger)

| id | finding | headline |
|---|---|---|
| C1 | seven heaviest asteroids with measured masses | supported on the first test (8 matched cases) |
| C2a | time-varying spin axes (IAU precession/nutation/libration) | **DISPROVED** as the cause of the Moon's gap (0.000 km change for the inner planets) |
| C3 / C3a | Moon's deficit shape; triaxial C22 locked to the Earth (GRAIL 2.2382e-5) | supportive; the Moon moves +0.29 km/yr against a deficit of -0.29 to -0.44 km/yr |
| C5 | DE440's Kuiper-belt ring + its 343 other asteroids | long-range outer-planet deficit closes (the prediction of ~34 km for Neptune from the ring's tide was made before the run) |
| C6 | comets as tracked test particles (73P) | 60 runs from 20 comets: +5 yr median 2,097 -> 0.96 km |
| C7 | seven identified KBOs (unique 1 % mass match) | Pluto +20 yr 4.06 -> 1.45 km; Uranus 1.75 -> 0.86; Neptune 1.41 -> 0.88 |
| C8 | planet-centre "floors" (Uranus ~800 km, Neptune ~450, Pluto 30-46) were the fixture's barycentre velocities, not dynamics | rigid-shift fix (r4, +5 yr, converged step): Uranus 796 -> 0.14 km, Neptune 478 -> 0.13, Pluto 31.5 -> 9.8, Jupiter 4.06 -> 2.18; r3's planet-only shift broke the moons |
| C9 | the 30 heaviest asteroids scored as bodies | +5 yr median 61.1 -> 10.0 (relativity) -> 2.56 km (others' pull) |
| C10 | three slow moons | Himalia supported (1.0 / 24.2 km); Nereid and Helene thousands of km, open |
| C11 | full 33-body shoot-out on r4, 29 jobs, 21 windows | Moon 3.90/14.67 -> 0.40/0.95 km (+5/+20 yr); Earth 0.45/1.89 -> 0.06/0.16; Mars 1.26/3.92 -> 0.07/0.19; Uranus 778/3,924 -> 0.21/1.17; Neptune 441/1,673 -> 0.21/1.33; Jupiter +20 yr 67.6 -> 20.6; Pluto 43.4/279 -> 11.4/44.5 |
| C12 | measured Jupiter/Saturn zonals replace the table's stale ones (engine J2-J4 verified analytically to 1.6e-7) | Callisto 203 -> 36.9, Titan 229 -> 60.2, Hyperion 220 -> 54.8 km (+5 yr, 6/6 windows) |
| C13 | Neptune's "2.9 mm/s defect" was missing small-moon mass | -> 0.5 mm/s; Triton +1 yr 302 -> 39 km; Uranus 5 mm/s remains (no masses published); **Nereid's error unexplained** |
| C14 | fast moons scored at a converged step (system-local models) | 29 moons supported (Io 159, Europa 100, Enceladus 358 km ...), 14 not (Pan, Atlas, Prometheus, Pandora, Janus, Epimetheus, Juliet, Portia, Belinda, Puck, Despina, Galatea, Larissa, Proteus lose phase within a year; cause unknown) |
| C15 | DE440's 645 header constants folded into local data; audit of ours vs header (16 rows: 4 convention, 5 rounding, 6 vintage, 1 unresolved = Sun J2 2.1106e-7 vs 2.19614e-7) | data in place; effect test running (section 6) |
| C16 | full 56-body shoot-out on r8, 21 jobs, 0 failed | planets unchanged within 1 % by the added moons; Callisto 210 -> 31.9, Titan 227 -> 59.8 km; **open: Jupiter +4 to +9 % worse (and Io/Europa) at 0.025 d in 0 of 21 windows** |
| C17 | sealed 30-yr forecast table + 4-start hindcast error table | forecast within 0.05 km of DE440 over its first 0.82 yr; hindcast median at +5 yr: Earth 0.10, Moon 0.38, Jupiter 0.054 km |

Also measured: the original r2 vs corrected-step comparison (0.05 d -> 0.025 d: Mercury 1.41 -> 0.02 km, Moon 3.70 -> 1.45 km; those were numerical error, not physics); Charon's 4-digit GM rounding explains Pluto's 11 km (not adopted: that would be fitting).

## 4. Tooling (what each script is for)

* Runners: `scripts/run_shootout_parallel.py` (plan-driven shoot-out, retries, checkpoints), `scripts/fair_jpl_contest.py` (original contest), `scripts/run_hindcast_variants.py` (per-start hindcast driver with retries), `scripts/run_system_local.py`, `run_slow_moon_diag.py`.
* Fixture builders: `build_fixture_r7.py`, `build_fixture_r8.py`, `extend_fixture_with_moons.py`, `fetch_horizons_batch.py` (group `centres`), `make_de440_constants.py`.
* Diagnosis: `slow_moon_diagnosis.py` (variants: measured_zonals, j2_of_, gm_of_, no_gr, newtonian), `slow_moon_fit.py`, `phase_slope.py` (mean-motion phase slope; wrap-around ambiguity spaced 1/orbits-per-year), `system_local_moons.py`, `constants_audit.py`.
* Reports/figures: `shootout_full_restate_report.py`, `zonal_paired_report.py`, `hindcast_paired_report.py`, `moon_scores_r5.py`, `system_local_report.py`, `make_shootout_figures.py` (F1-F9).
* Forecast: `make_forecast_table.py` (refuses step > 0.05 d or elapsed part > 2 km from DE440), `make_prediction_error_table.py`, `stamp_model_fingerprint.py` (stamps only if untouched, or re-run reproduces the table).
* Watchdog: `scripts/jobwatch.py` (presets `shootout`, `fair`; generic mode with `--log`, `--err-glob`, `--success-regex`).
* Tests: `tests/test_reduced_model.py`, `test_restated_predictors.py`, `test_build_fixture_r7/r8.py`, `test_jobwatch.py`, `test_forecast_builder.py`, `test_where_when.py`, `test_hindcast_paired.py` and others.

## 5. Where everything is

* Raw runs and reports: `sigma_ground/field/interface/fixtures/v2physics_restated/` (README.md maps them; `SHOOTOUT_FULL_RESTATED_r8.md/json`, `PAIRED_R4_TO_R8.md`, `ZONAL_PAIRED.md`, `MOON_DIAGNOSIS.md`, `SYSTEM_LOCAL_MOONS.md`, `CONSTANTS_AUDIT.md`, `figures/F1-F9`, raw `rolling_shootout_*.json`). Superseded/earlier versions are kept beside the new ones.
* Big run outputs: `D:/datasets/sigma_runs/` (e.g. `jup_fine/`, `hdr_hindcast/`, `hdr_dry/`), the forecast `D:/datasets/forecast/` (plus `rejected/` with the 0.25 d build and a README), data inputs `D:/datasets/jpl/` (DE440s kernel, Horizons copies, SBDB). Mirrored to skippy-core `/srv/datasets/{jpl,forecast}`.
* Ledgers: `misc/CANDIDATE_LEDGER.md`, `misc/DISPROOF_LEDGER.md`; plans: `misc/IMPROVEMENT_PLAN.md`, `misc/NEXT_FETCH_CENTRES.md`; session log `misc/SESSION_LOG.md`; scenes in `misc/OPERATIC_PLAY_SCENE*.txt`.
* Git: sigma-ground's remote is the local trunk `D:/trunk/git/sigma-ground.git` (not GitHub); master is pushed through `a313083`. The working tree has many untracked/modified run products (`sigma_ground/radiance/web/data/*.json`, `misc/BROWSER_REPORT.json`, `scratch_*.json`, ...) that are NOT part of this work and were left alone.

## 6. Running now (check before starting anything heavy)

| job | what | where | status/ETA |
|---|---|---|---|
| Jupiter convergence check (C16's falsifiable test) | `restated_ast_ring_kbo_c22` and `restated_zon` on r8 at 0.0125 d, windows j2008/j2014/j2020, +5 yr, 6 jobs | driver PID 11484, `D:/datasets/sigma_runs/jup_fine`, watchdog `jupfine` | started 2026-10-26 18:00, ETA 14-24 h. **Criterion: Jupiter +5 yr with measured zonals must be at or below the table zonals' ~2.2 km, else the step is not the cause and measured zonals worsen Jupiter** |
| C15 header-constants test | `restated_hdr` hindcast, starts 2000/2005/2010/2015, reduced model, 0.025 d | driver PID 21500, `D:/datasets/sigma_runs/hdr_hindcast`, watchdog `hdrhind` | started 2026-10-27 00:30, ETA ~2-3 h. **Prediction written before the run: Moon and Mercury do not get worse; Mercury moves 0.01-0.06 km at +5 to +20 yr.** Compare: `python scripts/hindcast_paired_report.py D:/datasets/sigma_runs/hdr_hindcast --out misc/ --label HINDCAST_HDR_PAIRED`, then write the ledger entry (supported or not) |

Machine: RAM was ~2.7 GB free with the Jupiter job running; do not stack more large jobs.

## 7. Open items (nothing here is explained yet)

1. Jupiter planet-centre error 4-9 % worse (and Io/Europa) with measured zonals at 0.025 d; the Jupiter 0.0125 d check decides whether the step is the cause.
2. Close-in moons lose phase within a year: Pan, Atlas, Prometheus, Pandora, Janus, Epimetheus, Juliet, Portia, Belinda, Puck, Despina, Galatea, Larissa, Proteus (not step, not zonals; several are massless in the fixture because no mass is published). Relative start state suspect; truth track clean.
3. Nereid (402-507 km at +1 yr, in-plane, insensitive to zonals/tides/step/Neptune's mass/start velocity), Helene, Phoebe 1,527 km, Phobos 1,088 / Deimos 176 km after 3 yr (Phobos probably tidal).
4. Uranus 5 mm/s barycentre velocity offset (small-moon masses unpublished).
5. Sun J2 provenance: ours 2.1106e-7 vs header 2.19614e-7 (unresolved audit row).
6. The Moon's +20 yr value 0.95 -> 1.00 km (2 of 7 windows): unexplained, possibly noise; the 1.7e-4 km Moon difference between the reduced model and the full run.
7. Terms not yet integrated: Earth tidal time lag, Moon tides, libration (full), solar mass loss (GMSDOT), lunar degree-3+ gravity, Earth J5.
8. Hindcast error table has 4 starts; leads past 15 yr rest on 1-2 starts (the Moon's +20 yr is one start).
9. Mentat-side roadmap: `prediction_skill`, `mass_budget`, router retrain (see the Mentat handoff).

## 8. Decisions waiting on the Captain

* If `restated_hdr` or the Jupiter check changes the best predictor, rebuilding the sealed forecast table is a separate decision (the table becomes stale by design; about 3.3 h).
* Whether to take the cloud Mentat to data parity (kernel 32 MB + table 21 MB into its image).
* Whether to scope any of open items 2-7 as the next experiment.

## 9. Gotchas that cost time (all recorded in `.claude/skills/windows-shell/SKILL.md`)

* Bash heredocs collapse backslashes: use Edit/Write for any text with backslashes, run `py_compile` in the same command.
* `/d/...` vs `D:/...` paths differ between Git Bash and Windows Python; `git apply` against a CRLF tree needs `--ignore-whitespace`.
* jobwatch: `--log` is repeated per file; a launcher that redirects to `driver.out` hides success markers unless the preset reads it (fixed); read the driver's own log before assuming a death.
* A timed-out foreground tool call is moved to the background and keeps running (it can hold a repo-wide grep or a long job).
* Fixture changes: never overwrite; add a vintage. A rigid shift must move ALL system members or the moons' relative states break (r3).
* Phase errors of close-in moons are ambiguous by whole orbits per year; use the phase-slope method before reading a position error as dynamics.
