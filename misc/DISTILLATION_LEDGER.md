# Distillation ledger: what Mentat missed on the daily new questions, and why

Written by `sigma_ground/mcp/benchmark/distill_score.py` from the trickle's export; do not edit the tables by hand (the fix status column comes from `misc/distill_fixes.json`: cause|tool -> {status, commit}). Wolfram's column is a yardstick: the causes come from Mentat's own record.

## Days

| day | questions | judged | Mentat right | misses by cause | unverified | Wolfram answered / no result | Wolfram agrees / disagrees with our oracle | Mentat ref |
|---|---|---|---|---|---|---|---|---|
| 2026-10-30 | 40 | 35 | 34 | wrong_value 1 | 5 | 29 / 11 | 16 / 6 | 883ea00 |
| 2026-10-31 | 40 | 35 | 35 | none | 5 | 29 / 11 | 16 / 5 | 883ea00 |
| 2026-11-01 | 40 | 35 | 34 | wrong_value 1 | 5 | 34 / 6 | 13 / 11 | 883ea00 |

## Causes, grouped: the candidate fixes

| cause | expected tool | rows | examples | status |
|---|---|---|---|---|
| wrong_value | solve_ode | 1 | `solve y'' + y = 0` (got Eq(y(x), C1*sin(x) + via solve_ode) | set wording f0a0e56 |
| wrong_value | fourier_transform | 1 | `Fourier transform of exp(-2*t^2)` (got 0 via fourier_transform) | open |

## Unverified (15): wild questions with no oracle, for a person to read

- `molar mass of water in g/mol` -> Mentat {'atomic_mass_amu': 1.008, 'at  (router); Wolfram answered
- `density of gold in kg/m³` -> Mentat 19300.0 kg/m^3 (router); Wolfram answered
- `boiling point of mercury in degrees Celsius` -> Mentat 629.88 K (cascade); Wolfram answered
- `mass of the Sun in kilograms` -> Mentat 1.988409870698051e+30 kg (router); Wolfram answered
- `age of the Universe in years` -> Mentat 4.3542154628946586e+17 s (cascade); Wolfram answered
- `molar mass of sulfuric acid` -> Mentat {'atomic_mass_amu': 15.999, 'a  (router); Wolfram answered
- `density of mercury at 25 C` -> Mentat 13534.0 kg/m^3 (router); Wolfram answered
- `pH of pure water at 25 C` -> Mentat None  (deterministic); Wolfram no_result
- `surface gravity of Jupiter in m/s^2` -> Mentat {'field_constants_source': 'Fo  (router); Wolfram answered
- `age of the universe in billion years` -> Mentat 4.3542154628946586e+17 s (cascade); Wolfram answered
- `molar mass of H2SO4` -> Mentat {'atomic_mass_amu': 1.008, 'at  (router); Wolfram answered
- `density of gold g/cm3` -> Mentat 19300.0 kg/m^3 (router); Wolfram answered
- `speed of light m/s` -> Mentat 299792458.0 m / s (router); Wolfram answered
- `mass of the Sun in kg` -> Mentat 1.988409870698051e+30 kg (router); Wolfram answered
- `radius of Earth in meters` -> Mentat 6378100.0 m (router); Wolfram answered
