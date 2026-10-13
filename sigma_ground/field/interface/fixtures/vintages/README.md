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
