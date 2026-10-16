# Next shoot-out: the exact data requests, for the Captain's approval (nothing has been fetched)

Prepared 2026-10-16. All requests go to the JPL Horizons API or the JPL SBDB API (public domain, no account), one at a time, 0.5 s apart, one file per request in a NEW directory, never overwriting. The first request of each group is sent alone and checked before the rest.

Epochs: the 27 yearly DE440-fixture snapshots J2000 ... J2026 as a Horizons TLIST (TDB JD), so truth lands on exactly the fixture's epochs:

`2451545.00 2451910.25 2452275.50 2452640.75 2453006.00 2453371.25 2453736.50 2454101.75 2454467.00 2454832.25 2455197.50 2455562.75 2455928.00 2456293.25 2456658.50 2457023.75 2457389.00 2457754.25 2458119.50 2458484.75 2458850.00 2459215.25 2459580.50 2459945.75 2460311.00 2460676.25 2461041.50`

## Item 1. The 30 heaviest of DE440's 343 asteroids as SCORED bodies (30 requests, about 12 KB each = 360 KB, to D:/datasets/jpl/horizons_asteroids/)

Chosen by DE440's own GM (header constants): these 30 carry 78 % of the 343's mass. Barycentric ICRF state (km, km/day) at the 27 epochs, plus the orbit solution header. `COMMAND='<number>;'` selects the numbered asteroid; a response that is not the intended body (name check) is logged and skipped, never guessed.

| # | asteroid | DE440 GM km3/s2 |
|---|---|---|
| 1 | Ceres | 62.629 |
| 4 | Vesta | 17.288 |
| 2 | Pallas | 13.666 |
| 10 | Hygiea | 5.625 |
| 511 | Davida | 3.894 |
| 704 | Interamnia | 2.830 |
| 52 | Europa | 2.683 |
| 87 | 87 Sylvia (A866 KA) | 2.168 |
| 15 | Eunomia | 2.023 |
| 3 | Juno | 1.921 |
| 16 | Psyche | 1.590 |
| 107 | 107 Camilla (A868 WA) | 1.444 |
| 88 | Thisbe | 1.190 |
| 7 | Iris | 1.140 |
| 31 | Euphrosyne | 1.079 |
| 65 | 65 Cybele (A861 EB) | 0.938 |
| 209 | Dido | 0.875 |
| 94 | Aurora | 0.873 |
| 48 | Doris | 0.856 |
| 154 | Bertha | 0.855 |
| 29 | Amphitrite | 0.798 |
| 375 | Ursula | 0.795 |
| 532 | Herculina | 0.792 |
| 566 | 566 Stereoskopia (A905 KA) | 0.763 |
| 386 | Siegena | 0.751 |
| 702 | Alauda | 0.750 |
| 39 | Laetitia | 0.728 |
| 9 | Metis | 0.650 |
| 6 | Hebe | 0.647 |
| 324 | Bamberga | 0.619 |

First request, exactly as it would be sent (1 Ceres):

```
https://ssd.jpl.nasa.gov/api/horizons.api?format=json&COMMAND=%271%3B%27&OBJ_DATA=%27YES%27&MAKE_EPHEM=%27YES%27&EPHEM_TYPE=%27VECTORS%27&CENTER=%27500%400%27&TLIST=%272451545.00%202451910.25%202452275.50%202452640.75%202453006.00%202453371.25%202453736.50%202454101.75%202454467.00%202454832.25%202455197.50%202455562.75%202455928.00%202456293.25%202456658.50%202457023.75%202457389.00%202457754.25%202458119.50%202458484.75%202458850.00%202459215.25%202459580.50%202459945.75%202460311.00%202460676.25%202461041.50%27&TLIST_TYPE=%27JD%27&TIME_TYPE=%27TDB%27&VEC_TABLE=%272%27&REF_PLANE=%27FRAME%27&REF_SYSTEM=%27ICRF%27&OUT_UNITS=%27KM-D%27&CSV_FORMAT=%27YES%27
```

The other 29 differ only in `COMMAND`: `'4;'`, `'2;'`, `'10;'`, `'511;'`, `'704;'`, `'52;'`, `'87;'`, `'15;'`, `'3;'`, `'16;'`, `'107;'`, `'88;'`, `'7;'`, `'31;'`, `'65;'`, `'209;'`, `'94;'`, `'48;'`, `'154;'`, `'29;'`, `'375;'`, `'532;'`, `'566;'`, `'386;'`, `'702;'`, `'39;'`, `'9;'`, `'6;'`, `'324;'`.

## Item 2. Moons missing from the fixture (23 requests, about 12 KB each = 280 KB, to D:/datasets/jpl/horizons_moons/)

Barycentric ICRF state at the 27 epochs plus the body's GM where Horizons gives it. Horizons body ids are checked against the returned `Target body name`; a mismatch is logged and skipped. Moons chosen: those with a well-determined ephemeris and a dynamical role (inner moons that shepherd rings, co-orbitals, the small Pluto and Neptune moons, one outer Jovian). The very small ones can be added as massless test particles if their GM is n.a.

- Jupiter: Amalthea `505`, Thebe `514`, Himalia `506`
- Saturn: Janus `610`, Epimetheus `611`, Prometheus `616`, Pandora `617`, Atlas `615`, Pan `618`, Helene `612`
- Uranus: Puck `715`, Portia `712`, Juliet `711`, Belinda `714`
- Neptune: Proteus `808`, Nereid `802`, Larissa `807`, Galatea `806`, Despina `805`
- Pluto: Nix `902`, Hydra `903`, Kerberos `904`, Styx `905`

First request (Proteus, Neptune):

```
https://ssd.jpl.nasa.gov/api/horizons.api?format=json&COMMAND=%27808%27&OBJ_DATA=%27YES%27&MAKE_EPHEM=%27YES%27&EPHEM_TYPE=%27VECTORS%27&CENTER=%27500%400%27&TLIST=%272451545.00%202451910.25%202452275.50%202452640.75%202453006.00%202453371.25%202453736.50%202454101.75%202454467.00%202454832.25%202455197.50%202455562.75%202455928.00%202456293.25%202456658.50%202457023.75%202457389.00%202457754.25%202458119.50%202458484.75%202458850.00%202459215.25%202459580.50%202459945.75%202460311.00%202460676.25%202461041.50%27&TLIST_TYPE=%27JD%27&TIME_TYPE=%27TDB%27&VEC_TABLE=%272%27&REF_PLANE=%27FRAME%27&REF_SYSTEM=%27ICRF%27&OUT_UNITS=%27KM-D%27&CSV_FORMAT=%27YES%27
```

Others differ only in `COMMAND`.

## Item 3. Planet-centre floors: NO fetch needed; found and fixed by derivation (2026-10-16)
The floors (Uranus ~800 km, Neptune ~450 km, Pluto 30-46 km, Jupiter ~4 km at +5 yr) are the fixture's planet-centre VELOCITIES: against DE440's system barycentre velocity and the fixture's own moon states they are off by
5.3, 2.9, 0.29 and 0.021 mm/s (0.003 m/s over 5 yr is 474 km; I wrongly called it negligible earlier today). Started from the kernel's barycentres the same dynamics ends within 0.6-5 km for every giant; started from the fixture's
states it ends 1,205 / 453 / 40 km off for Uranus / Neptune / Pluto. `scripts/rederive_giant_states.py` writes a NEW fixture (`de440_state_vectors_v2physics_r3.json`, the old one untouched) with the planet centres re-derived from
the kernel barycentres and the fixture's moon states, at every snapshot (initial states and later truth alike). A two-window verification run is going.

## Item 5. The 30 Kuiper-belt objects (two steps)
A. Identify MA8001-MA8030: DE440's header gives only their GM; the list is Pitjeva & Pitjev (2018). I would read that paper's table (a web search/read, no data download) and match by name, never by guessing from the GM values.
B. Then one SBDB request per object for its orbital elements: `https://ssd-api.jpl.nasa.gov/sbdb.api?sstr=<name>&full-prec=1`, 30 requests of about 2 KB. The exact list will be shown to you after step A.

