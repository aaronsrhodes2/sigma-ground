# Comet truth data (JPL Horizons) and the candidate ladder

`horizons_<designation>_<window>.json`: 60 Horizons API responses (20 periodic comets x windows J2005, J2010, J2015), fetched 2026-10-15/16 with the Captain's approval from the requests listed in
`misc/COMET_FETCH_REQUESTS.md` (public domain, no account). Each holds JPL's orbital solution (with its non-gravitational parameters A1, A2, A3, DT and the Marsden-law constants) and the
barycentric ICRF state (km, km/day) at the window start and +1..+5 yr, at the DE440 fixture's snapshot epochs. 73P (Schwassmann-Wachmann 3) was added 2026-10-16 with the corrected record request (`COMMAND='90000740'`, JPL#K222/31): 60 responses, 20 comets x 3 windows.
`ladder_full.json` / `COMET_LADDER.md`: our propagation from JPL's state at the window start against JPL's later states of the same solution (`scripts/comet_ladder.py`, `scripts/comet_ladder_report.py`).

Wording: the truth is JPL's own SOLUTION propagated by Horizons (DE441 planets, 16 massive asteroids, the Marsden model). This tests whether our dynamics reproduce JPL's dynamics for these bodies; it does not
measure how well JPL's solution predicts the comet's real position.
