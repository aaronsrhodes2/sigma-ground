# Distillation ledger: what Mentat missed on the daily new questions, and why

Written by `sigma_ground/mcp/benchmark/distill_score.py` from the trickle's export; do not edit the tables by hand (the fix status column comes from `misc/distill_fixes.json`: cause|tool -> {status, commit}). Wolfram's column is a yardstick: the causes come from Mentat's own record.

## Days

| day | questions | judged | Mentat right | misses by cause | unverified | Wolfram answered / no result | Wolfram agrees / disagrees with our oracle | Mentat ref |
|---|---|---|---|---|---|---|---|---|
| 2026-10-30 | 40 | 35 | 33 | wrong_value 2 | 5 | 1 / 0 | 0 / 1 | 8145ed8 |

## Causes, grouped: the candidate fixes

| cause | expected tool | rows | examples | status |
|---|---|---|---|---|
| wrong_value | solve_ode | 1 | `solve y'' + y = 0` (got Eq(y(x), C1*sin(x) + via solve_ode) | set wording f0a0e56 |
| wrong_value | photon_energy_from_wavelength | 1 | `Photon energy at the Wien peak for 3.5e3 K?` (got 8.279348442857143e-0 via None) | fixed f0a0e56 |

## Unverified (5): wild questions with no oracle, for a person to read

- `molar mass of water in g/mol` -> Mentat {'atomic_mass_amu': 1.008, 'at  (router); Wolfram unasked
- `density of gold in kg/m³` -> Mentat 19300.0 kg/m^3 (router); Wolfram unasked
- `boiling point of mercury in degrees Celsius` -> Mentat 629.88 K (cascade); Wolfram unasked
- `mass of the Sun in kilograms` -> Mentat 1.988409870698051e+30 kg (router); Wolfram unasked
- `age of the Universe in years` -> Mentat 4.3542154628946586e+17 s (cascade); Wolfram unasked
