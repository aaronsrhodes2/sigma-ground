# The deviance mesh: JPL's own past ephemerides against DE440 (2026-10-11)

Seven JPL planetary ephemerides, each a snapshot of what JPL believed at its release, compared with DE440 (2020) by target date:
DE403 (14 Nov 1995), DE405 (26 Aug 1998), DE410 (24 Apr 2003), DE413 (4 Nov 2004), DE418 (3 Aug 2007), DE421 (13 Feb 2008), DE432 (30 Apr 2014).
Release dates are the ones written in each kernel's own comments. Kernels are official NAIF files (naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets,
77 MB total) kept at `D:/datasets/jpl/vintages/`, outside the repo. Targets run 1996 to 2019, because DE440's own fit data ends about 2020 and later dates
are its forecast, not truth. Code: `scripts/vintage_deviance_mesh.py`. Output: `DEVIANCE_MESH.json`, `DEVIANCE_MESH_lines.png`.

## What the mesh shows
- A vintage's deviation does NOT grow visibly after its release: it is about the same size before and after the dotted release line. Inner planets show a
  constant, orbit-sized offset (a fixed difference in the orbit); Neptune shows a steady linear drift. So most of JPL's historical "error" is a standing
  difference in the fitted orbit, not forecast growth with lead time.
- Median deviation (km, heliocentric) of the vintages at 1 to 5 years past their release is flat: Mercury 1.1-1.3, Venus 0.6-0.7, Mars 0.3-0.4, Earth 0.3,
  Jupiter 27-53, Saturn 205-271, Uranus 404-577, Neptune 1856-2390.
- Stripe test (a date where several vintages are above their own level together): after normalising each vintage by its own level and stopping at 2019, no body
  has a stripe beyond what chance gives over ~190 tests (one 3.1 sigma for Venus 2003 with 4 vintages). Earlier flags in 2023-2026 were DE440's own forecast, and
  flags in 1997-2005 were old vintages being worse overall; both were removed as artifacts of the test, not findings.

## Comparison with our forecasts (`comparison_with_ours` in the JSON, r1 = before the zonal fix) and the caveat that matters
At 1 to 5 years of lead our shoot-out error (heliocentric, km) is below the vintages' median for Mercury (to 4 yr), Venus, Saturn, Uranus (to 3 yr) and Neptune, and above it for Mars,
Earth and Jupiter at long lead (Jupiter's 630 km at 5 yr is the zonal defect fixed afterwards; the r2 numbers replace it).
**This is NOT a statement that we beat JPL.** Our shoot-out starts every forecast from the DE440 state at the issue date (perfect initial conditions), so it measures
integration error alone, while a JPL vintage also carried the orbit-determination error of its day. The honest reading is: for the outer planets the limit is the data (initial
conditions), not the dynamics; for the inner planets our integration error and JPL's vintage offsets are both about a kilometre. A fair contest needs one of:
(a) both start from the SAME initial state (propagate each vintage's own state at its release with our engine and compare with that vintage and with DE440), or
(b) our own orbit fit from the data available at the issue date.

## Update 2026-10-12: comparison with the corrected (r2) forecasts
Median error (km) at 1 to 5 years of lead, JPL vintages | ours (r2, zonal-fixed), same like-for-unlike caveat as above:
Mercury 1.1-1.3 | 0.3-1.3; Venus 0.6-0.7 | 0.0-0.2; Mars 0.3-0.4 | 0.4-1.5; Earth 0.3 | 0.1-0.5; Jupiter 27-53 | 18-42; Saturn 205-271 | 0.3-1.6; Uranus 404-577 | 153-778; Neptune 1856-2390 | 88-441.
With the zonal defect fixed, Jupiter now sits inside the vintages' band (it was 630 km at 5 yr in r1). Saturn's 0.3-1.6 km against the vintages' ~240 km says the vintages' Saturn orbit was uncertain by hundreds of km at their
release, an initial-condition (data) limit, not a dynamics limit. Mars and Earth at the longest lead are where our integration is slightly worse than a vintage's standing offset.

## The fair contest (2026-10-12): same initial state, different dynamics
`scripts/fair_jpl_contest.py`, report `FAIR_CONTEST.md/json`, scatter `FAIR_CONTEST_scatter.png`, raw per-run results `fair_contest_runs.jsonl`.
67 propagations (7 JPL vintages x every start year from 2000 to 2015 at or after the release; the 4 DE403 starts after 2011 are impossible, that kernel ends in 2011).
Each starts from the vintage's OWN state, runs 5 years with our engine (headline stack, 0.05 d, reduced model validated against the full shoot-out to ~20 % on
Mercury, Venus, Earth, Moon), and is compared with the vintage's own continuation and with DE440. Median km at +5 yr (ours vs DE440 | JPL's old forecast vs DE440 | the gap between the two dynamics):
Mercury 1.97|1.52|1.20, Venus 1.14|0.75|0.52, Earth 0.76|0.40|0.90, Moon 3.61|0.38|3.40, Mars 2.33|0.67|2.11, Jupiter 53.4|52.3|3.8, Saturn 276|277|1.9, Uranus 956|956|1.5, Neptune 3707|3707|1.5, Pluto 7950|7951|1.5.

**What it supports.** (1) Given the same initial state, our engine reproduces JPL's own old forecasts to **no worse than 4 km over 5 years for every body** (the gap column: 0.2-4 km),
and for Jupiter through Pluto our error and JPL's are indistinguishable (sign test mostly p > 0.05): what separated those old forecasts from the truth was the initial state, the data of the day, not the physics.
(2) For the inner solar system and the Moon the two dynamics are NOT at parity at the 1 km level: JPL's old continuation stays flat at its initial offset (about 0.4-1.5 km from lead 1 to 5 yr), so JPL's old dynamics
were already close to DE440's, while ours drifts away: Moon 0.9 -> 3.6 km (about +0.7 km/yr), Mars 0.9 -> 2.3, Earth 0.3 -> 0.8, Mercury 1.6 -> 2.0 (JPL better in nearly every run: sign test p ~ 0).
So our remaining model error over 5 years is of order 1-4 km for those bodies, and it is physics we lack (the candidates: asteroid perturbations, especially for Mars; lunar figure and Earth-Moon tides; solar oblateness and mass loss), not initial conditions.
**Wording to use:** "km-level dynamics parity with JPL's historical model over 5 years (gap <= 4 km), with a known, quantified deficit of 1-3 km on the Moon, Mars, Earth and Mercury." Not "mathematical parity".
Caveats: DE440 is the truth reference and is itself a fit (data end about 2020); our model has no asteroids; the reduced model carries the giants as system barycentres.

## Step-size convergence (2026-10-12): separating numerics from physics
Eight of the contest cases (DE405/DE410/DE421 x J2006/J2010/J2013) were rerun at half the step (0.025 d against 0.05 d). Median of the "dynamics gap" (ours vs the vintage's own continuation, km):

| body | +1 yr (0.05 -> 0.025) | +3 yr | +5 yr |
|---|---|---|---|
| Moon | 0.80 -> 0.40 | 1.90 -> 0.96 | 3.17 -> 1.82 |
| Mercury | 0.27 -> 0.08 | 0.66 -> 0.24 | 1.25 -> 0.46 |
| Venus, Earth, Mars, Jupiter | unchanged to 0.01 km | unchanged | unchanged |

Reading: Venus, Earth, Mars and Jupiter are converged at 0.05 d, so their gaps (Mars 2.5 km, Earth 0.7, Venus 0.6 at +5 yr) are PHYSICS we lack. The Moon and Mercury were not converged:
fitting gap(dt) = floor + c dt^4 (a fourth-order scheme) gives numerical errors of about 1.4 km (Moon) and 0.8 km (Mercury) at 0.05 d and 0.09 and 0.05 km at 0.025 d, leaving physics floors of about
**1.7 km (Moon), 0.4 km (Mercury) at +5 yr**. The scheme keeps its order; the standard step was too coarse for the fast inner pair. (The engine's hierarchical stepper is documented broken, so the fix is the global step.)
