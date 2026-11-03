# The N-body improvement campaign: the standing brief (written 2026-11-02, for the private Skippy or any session that picks it up)

**Goal (the Captain, 2026-11-02):** the best model-free, token-free, low-computation celestial N-body predictor; at least half of our error removed by straight maths; its data and
abilities in sigma-ground and in Mentat's toolset (MCP descriptor and routing); the public experiment page showing *our prediction / JPL's prediction at the time / reality*.
Ledger entries C26 onward in [CANDIDATE_LEDGER.md](CANDIDATE_LEDGER.md) are the record; this file is the map.

## 1. Rules (not optional)
* **Data alone.** A claim stays only if it can be disproved and survived the test. Write the prediction (size and direction, computed by Mentat from sourced inputs) in the ledger
  BEFORE the run; a miss is recorded as a miss. No fitting a free parameter to the target; a constant comes from a source (the DE440 header, a paper) or it is not used.
* **Never edit a published label or fixture.** A change is a new predictor label beside the old (`restated_*` in `rolling_shootout.restated_predictors`); the stored forecast
  must still reproduce bit for bit (`scripts/stamp_model_fingerprint.py --forecast --verify-years 1` after any engine edit; it re-stamps only if the first year reproduces to 0 km).
* **Mentat does the physics** (PLATINUM §21): any physical number comes from a Mentat tool with its source; when Mentat misroutes or lacks a tool, record the defect in the ledger
  entry and use `evaluate_quantity` on sourced inputs. Bash first (§20); D: not C: (§22); long jobs with the retrying driver and `scripts/jobwatch.py` attached; no cloud sessions;
  single-agent flow; ask before any network fetch beyond NAIF/JPL public files and before any push or deploy.
* **Reality is DE442** (`datasets.truth_kernel()`); DE440 is the start state and the perturber-path source; JPL's past ephemerides (DE403 ... DE440) are "JPL's prediction at the time".
* **Converged step.** Score at 0.0125 d (or the Yoshida step once its table says it is equivalent); the 0.025 d Mercury numbers are cancelling errors (C19).
* **RAM.** The machine runs out below ~1 GB free; four 0.0125 d hindcast workers take ~0.4 GB; never stack more than three arms (12 workers) plus one script.

## 2. The two metrics (pre-registered in C26)
* **M-A, design:** reduced model, DE440 start at 2000/2005/2010/2015, 0.0125 d, median heliocentric error at +5 yr against DE442. Baseline `restated_zon`: Mercury 0.085, Venus 0.069,
  Earth 0.103, Moon 0.247, Mars 0.108 km. Target: each at or below half. Secondary: Moon +20 yr 1.22 -> <= 0.6 km.
* **M-B, public proof:** the fair contest redone (`scripts/fair_jpl_contest.py --predictor <label> --dt-days 0.0125`, `scripts/fair_contest_report.py`), every JPL vintage's own state as
  the start, ours against the vintage's continuation, both against DE442. Target: dynamics gap at +5 yr at or below half of Moon 3.40, Mars 2.11, Earth 0.90, Mercury 1.20 km;
  Venus to Pluto indistinguishable from JPL-at-the-time (sign test p > 0.05).
* A lever counts if it beats the control in 3 of 4 starts at +5 yr.

## 3. The levers and their state (2026-11-02, 23:00; the ledger C26 has the tables)
| id | label | what | predicted | result (4 starts, +5 yr, against DE442) |
|---|---|---|---|---|
| L1 | `restated_hdr_moon` | the Moon's header J2/J3/J4 + radius alone | Moon gain | **null** (0.983, 2/4): a recorded miss |
| L1 | `restated_hdr_sun`, `restated_hdr_earth` | the Sun's, the Earth's header constants alone | Mercury cost; Earth nothing | not run (the Earth part is answered by L2's attribution) |
| L2 | `restated_geo` | zonal reference radii + poles at the start date | Moon -0.08 km | Moon 0.245 -> 0.113 (0.463, 4/4), Mercury +1.9 % |
| L2 | `restated_fieldr_earth` | the Earth's J2-J4 on RE 6378.1366 km, the one change | carries most | **Moon 0.245 -> 0.113 (0.460, 4/4), all else within 0.7 %: the lever** |
| L2 | `restated_fieldr_moon`, `restated_pole` | the Moon's radius alone; the pole epoch alone | under 0.03 / 0.02 km | both null on four starts (0.985, 0.997) |
| L3 | `restated_tide` | DE430's time-delayed Earth tide, on restated_zon | Moon -0.06 km | **miss alone: Moon 22 % worse (0/4)** on the wrong Earth radius |
| L3 | `restated_math_tide` | the same tide on `restated_math` | | **Moon 0.116 -> 0.097 (4/4); +10 yr 0.283 -> 0.222**: the tide helps on the right radius |
| L4 | `restated_gmdot` | the Sun's GMSDOT | below floor | not run (sized 0.0016 km at +5 yr) |
| L5 | | Earth J5, lunar degree-3 | below floor | not built |
| L6 | | the Mars zonal flag | the Sun's J2 | not built |
| P3 | `y6`/`y8` | Yoshida steps | Y8 0.2 d at 1/3 the evaluations | **miss on cost** (0.62); what stands: Y6 0.1 d = fine Forest-Ruth to 1.3 m/yr at 60 % of the evaluations; it is the live tool's integrator |
| M-A | | the design metric | each inner body halved | **Moon met** (`restated_math_tide` 0.097 against 0.122); Mercury, Venus, Earth, Mars unchanged by every lever (0.08-0.12 km): OPEN |
| M-B | | the public proof (contest v2, `restated_zon`, 0.025 d, 147 propagations) | each gap halved | **all four met**: Moon 3.40 -> 0.97, Mars 2.11 -> 0.27, Earth 0.90 -> 0.33, Mercury 1.20 -> 0.36 km; parity Venus/Uranus/Pluto yes, Jupiter/Saturn/Neptune no by sign test (only Saturn's 0.34 km gap is above reality's spread) |
`restated_math_tide` is the candidate served model; the ledger-grade contest with it at 0.0125 d is running on skippy-core (`~/sigma-nbody/runs/contest_v3_math_tide`).

## 4. The commands
```bash
# one arm (4 starts, 10 yr, 0.0125 d, scored against DE442), with the watchdog
python -X utf8 scripts/run_hindcast_variants.py --predictor <label> --starts 2000 2005 2010 2015 --dt-days 0.0125 --max-lead 10 --outdir D:/datasets/sigma_runs/<name>
python -X utf8 scripts/jobwatch.py --name <name> --pid <driver pid> --log D:/datasets/sigma_runs/<name>/driver.log --success-regex "finished:" --interval 180 --expected-job-minutes 240
# the paired report against the control (same step, same truth)
python scripts/hindcast_paired_report.py D:/datasets/sigma_runs/<name> --new <label> --control-dir D:/datasets/sigma_runs/base_de442 --control restated_zon --out misc/ --label HINDCAST_<NAME>
# the contest and its report
python -X utf8 scripts/fair_jpl_contest.py --predictor <label> --dt-days 0.0125 --out D:/datasets/sigma_runs/contest_v2 --workers 10 --checkpoint
python -X utf8 scripts/fair_contest_report.py D:/datasets/sigma_runs/contest_v2/fair_contest_contest.jsonl
# the portal page's data (SkippyTel), then commit and push main to the house git: Cloud Run CD deploys
python scripts/nbody_blankets_build.py D:/datasets/sigma_runs/contest_v2/fair_contest_contest.jsonl
# the live arena (C27): ours and JPL's against the newest MPC sky positions of 32 asteroids; runs daily at 04:30 (Windows task 'sigma-ground mpc live daily'), by hand:
python -X utf8 scripts/mpc_live_driver.py --date YYYY-MM-DD          # scoreboard D:/datasets/mpc_live/scoreboard.jsonl, day mirror fixtures/mpc_live/<date>.json
python -X utf8 scripts/mpc_live_report.py                            # misc/MPC_LIVE_SCOREBOARD.md and fixtures/mpc_live/scoreboard_portal.json (the portal reads this; not built yet)
```

## 5. The fold-in (after `restated_math` is measured)
1. Rebuild the sealed forecast and the error table with the winning label at the converged step (`scripts/make_forecast_table.py`, `scripts/make_prediction_error_table.py`), new manifest
   entry beside the old, stamp; `where_when.py` `_STEP_CAVEATS` updated to the measured numbers.
2. Re-measure `body_skill` and `layer_effect` tables; `prediction_contest` reads the v2 report (already wired).
3. Mentat: `body_position` for asteroids and comets (integrate on demand from `small_body` elements through the reduced model); keywords; add rows to the live suites and measure
   before/after; commit + push bumps the Mentat pin (deploy.json AND Dockerfile), asked first.
4. Docs: NBODY_HANDOFF.md §6-8, ARCHITECTURE.md (truth kernel), session log, scene file.

## 6. Open items that are leads, not promises
* The ring-region moons lose phase within a year (C14): Saturn's ring gravity is a missing term for Pan, Atlas, Prometheus, Pandora, Janus, Epimetheus; add the moons to the body
  catalogue (SAT415 masses from JPL's satellite pages; Daphnis from Thomas & Helfenstein 2020) and a ring potential (a flat annulus of Cassini's 1.54e19 kg), then re-score.
* Daphnis and the Keeler Gap waves: [DAPHNIS_WAVE.md](DAPHNIS_WAVE.md); a `ring_edge_wave` tool needs Weiss et al. 2009's fitted amplitude relation (full text not read yet).
* The Mercury deficit (0.085 km at +5 yr) and the Sun's J2: L1's Sun arm and L6 decide whether any single J2 value fits; if not, the EIH frame or the Sun's spin axis is next.
* Nereid, Helene, Phoebe, Phobos/Deimos, the Uranus 5 mm/s offset: unchanged from NBODY_HANDOFF.md §7.
* Mentat defects met in this campaign (fix the tool, not the question): `ask` misroutes compound tide questions; `evaluate_quantity` returns kg^2 for kg/kg, refuses nested powers, has
  no `sin`, and parses `a / b c` as a/b*c.
* Our own orbit fit to observations (the real "beat JPL" step): built and measured on Eros (C28, `scripts/orbit_fit.py`, `misc/ORBIT_FIT_EROS.md`): from 2015-2019 optical data alone, post-fit RMS 0.369 arcsec, the fitted state 33 km / 2.4 mm/s from JPL's, and the forward prediction matches JPL's for about four years, 1.6 times worse by year six. Next levers: per-station weights, a longer arc, radar, Yarkovsky; Ceres not yet fitted.
