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

## 3. The levers and their state (fill the result column from the ledger as arms finish)
| id | label | what | predicted (C26) | result |
|---|---|---|---|---|
| L1 | `restated_hdr_sun` / `_earth` / `_moon` | DE440 header J2/J3/J4 + reference radius, one body at a time, zonal radius only | Moon <= 0.15 km from hdr_moon; hdr_sun costs Mercury 0.02-0.03 km; Earth nothing | hdr_moon RUN 2026-11-02 (`D:/datasets/sigma_runs/l1_hdr_moon`); sun, earth not run |
| L2 | `restated_geo` | zonal radii per DE440 + IAU poles at the start date | Moon -0.08 km; else nothing | RUN (`l2_geo`) |
| L1+L2 | `restated_geo_hdr`, `restated_geo_hdr_em` | both, with / without the Sun's header J2 | geo_hdr_em reaches the Moon target without the Mercury cost | not run |
| L3 | `restated_tide` | DE430 eq. 31-32 time-delayed Earth tide (header k2/tau), static Earth tide off | Moon +5 yr -0.06 km, +20 yr -0.96 km (the +20 yr residual) | RUN (`l3_tide`); falsifier `scripts/tide_secular_check.py` (target -25.82"/cy^2) |
| L4 | `restated_gmdot` | Sun GM(t) with GMSDOT | below floor | not run |
| L5 | (to build) | Earth J5, Moon J3M/degree-3 | below floor | not built |
| L6 | ablations Sun-J2-only / Mars-J2-only | the Mars zonal flag (C23b) | the Sun's J2 alone reproduces it | not built |
| L7 | policy | served tables at the converged step | honest, not improved | with the fold-in |
| P3 | `integrator="y6"/"y8"` | Yoshida 6th/8th-order steps (`NBodySystem.yoshida_step`) | Y8 at 0.2 d within 1 % of FR 0.0125 d at a third of the evaluations | `scripts/integrator_convergence.py` RUNNING (`D:/datasets/sigma_runs/integ_conv`) |
Then `restated_math` = `restated_zon` + every arm that won, measured on M-A, then M-B, then the fold-in (section 5).

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
* Our own orbit fit to observations (the real "beat JPL" step): not started; needs the MPC/raw-sky path of C24 generalised.
