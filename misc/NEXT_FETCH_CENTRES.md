# Fetch request for approval: six planet-centre and big-moon tracks (nothing has been fetched)

Prepared 2026-10-23. JPL Horizons API (public domain, no account), one request at a time, 0.5 s apart, one file per request in a NEW directory `D:/datasets/jpl/horizons_centres/`, never overwriting; the first is sent alone and checked before the rest. No data of yours is sent. Run with `python scripts/fetch_horizons_batch.py centres` (add `--dry-run` to print the first URL only).

**Why.** The 23 moons fetched on 2026-10-16 are Horizons states in the SSB frame from satellite ephemerides (nep098_merged etc.) with DE441. The close-in ones (Proteus, Pan, Puck, ...) and Nereid, Helene drift by a constant orbital-period error that no force switch removes: the truth is clean (its own mean period matches the literature to 2e-7) and the engine's oblateness force matches the analytic orbit (1.6e-7), but the starting RELATIVE state I built, moon minus the fixture's planet centre, behaves as if it were wrong by about 140 km or 5 m/s. The fixture's planet centres came from an earlier Horizons vintage. These six requests give the planet centre and one big moon of each system from the SAME vintage as the moons, so the offset can be measured directly instead of guessed.

Epochs: the 27 yearly fixture snapshots (TDB JD): `2451545.00 2451910.25 2452275.50 2452640.75 2453006.00 2453371.25 2453736.50 2454101.75 2454467.00 2454832.25 2455197.50 2455562.75 2455928.00 2456293.25 2456658.50 2457023.75 2457389.00 2457754.25 2458119.50 2458484.75 2458850.00 2459215.25 2459580.50 2459945.75 2460311.00 2460676.25 2461041.50`

Parameters of every request: `format=json`, `OBJ_DATA='YES'`, `MAKE_EPHEM='YES'`, `EPHEM_TYPE='VECTORS'`, `CENTER='500@0'`, `TLIST_TYPE='JD'`, `TIME_TYPE='TDB'`, `VEC_TABLE='2'`, `REF_PLANE='FRAME'`, `REF_SYSTEM='ICRF'`, `OUT_UNITS='KM-D'`, `CSV_FORMAT='YES'`; only `COMMAND` differs.

| # | COMMAND | body | expected name check | file |
|---|---|---|---|---|
| 1 | `899` | Neptune | `^Neptune\b` | `horizons_centre_899_Neptune.json` |
| 2 | `801` | Triton | `^Triton\b` | `horizons_centre_801_Triton.json` |
| 3 | `699` | Saturn | `^Saturn\b` | `horizons_centre_699_Saturn.json` |
| 4 | `606` | Titan | `^Titan\b` | `horizons_centre_606_Titan.json` |
| 5 | `799` | Uranus | `^Uranus\b` | `horizons_centre_799_Uranus.json` |
| 6 | `703` | Titania | `^Titania\b` | `horizons_centre_703_Titania.json` |

Exact first URL (the others differ only in `COMMAND`):

https://ssd.jpl.nasa.gov/api/horizons.api?format=json&
  COMMAND=%27899%27&
  OBJ_DATA=%27YES%27&
  MAKE_EPHEM=%27YES%27&
  EPHEM_TYPE=%27VECTORS%27&
  CENTER=%27500%400%27&
  TLIST=%272451545.00%202451910.25%202452275.50%202452640.75%202453006.00%202453371.25%202453736.50%202454101.75%202454467.00%202454832.25%202455197.50%202455562.75%202455928.00%202456293.25%202456658.50%202457023.75%202457389.00%202457754.25%202458119.50%202458484.75%202458850.00%202459215.25%202459580.50%202459945.75%202460311.00%202460676.25%202461041.50%27&
  TLIST_TYPE=%27JD%27&
  TIME_TYPE=%27TDB%27&
  VEC_TABLE=%272%27&
  REF_PLANE=%27FRAME%27&
  REF_SYSTEM=%27ICRF%27&
  OUT_UNITS=%27KM-D%27&
  CSV_FORMAT=%27YES%27

**What I will do with the answer.** Compare each centre with the fixture's (Neptune, Saturn, Uranus) at the 27 epochs; if they differ by tens of km, rebuild the moons' relative states and the scoring truth against the new centres as a NEW fixture (r8), leaving r4-r7 untouched, and run the shoot-out on r8. If they agree, the offset hypothesis is dead and the close-in moons stay open.
