# Truth for the 30 heaviest of DE440's 343 asteroids (JPL Horizons), fetched 2026-10-16 with the Captain's approval
30 responses, one per asteroid (`horizons_ast_<number>.json`): the orbit solution header and the barycentric ICRF state (km, km/day) at the 27 yearly fixture epochs J2000 ... J2026 (TDB). Requests: `misc/NEXT_SHOOTOUT_FETCH_REQUESTS.md`
(item 1), sent by `scripts/fetch_horizons_batch.py asteroids`; every response passed its checks (HTTP 200, ephemeris block, expected body name, 27 rows). Raw copies also in D:/datasets/jpl/horizons_asteroids/.
Wording: the truth is JPL's solution as propagated by Horizons (DE441 planets, 16 massive asteroids); scoring against it measures agreement with JPL's dynamics, not the asteroid's real position.

`asteroid_ladder.json` / `asteroid_ladder.log` (2026-10-19): the scoring of these 30 as bodies, 90 runs (`scripts/asteroid_ladder.py`); summary in `misc/CANDIDATE_LEDGER.md` C9 and figure F7.
