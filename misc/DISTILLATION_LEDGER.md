# Distillation ledger: what Mentat missed on the daily new questions, and why

Written by `sigma_ground/mcp/benchmark/distill_score.py` from the trickle's export; do not edit the tables by hand (the fix status column comes from `misc/distill_fixes.json`: cause|tool -> {status, commit}). Wolfram's column is a yardstick: the causes come from Mentat's own record.

## Days

| day | questions | judged | Mentat right | misses by cause | unverified | Wolfram answered / no result | Wolfram agrees / disagrees with our oracle | Mentat ref |
|---|---|---|---|---|---|---|---|---|
| 2026-10-30 | 40 | 35 | 34 | wrong_value 1 | 5 | 29 / 11 | 16 / 6 | 00afc1e |
| 2026-10-31 | 40 | 35 | 35 | none | 5 | 29 / 11 | 16 / 5 | 00afc1e |
| 2026-11-01 | 40 | 35 | 34 | wrong_value 1 | 5 | 34 / 6 | 13 / 11 | 00afc1e |
| 2026-11-02 | 40 | 35 | 35 | none | 5 | 32 / 8 | 15 / 7 | 00afc1e |
| 2026-11-03 | 40 | 35 | 34 | wrong_value 1 | 5 | 22 / 18 | 15 / 4 | 00afc1e |
| 2026-11-04 | 40 | 35 | 34 | wrong_value 1 | 5 | 28 / 12 | 15 / 6 | 00afc1e |
| 2026-11-05 | 39 | 35 | 34 | wrong_value 1 | 4 | 29 / 10 | 15 / 5 | 00afc1e |
| 2026-11-06 | 39 | 35 | 35 | none | 4 | 31 / 8 | 14 / 7 | 00afc1e |
| 2026-11-07 | 40 | 35 | 33 | wrong_value 2 | 5 | 29 / 11 | 17 / 7 | 00afc1e |
| 2026-11-09 | 37 | 35 | 34 | wrong_value 1 | 2 | 27 / 10 | 14 / 8 | 00afc1e |

## Causes, grouped: the candidate fixes

| cause | expected tool | rows | examples | status |
|---|---|---|---|---|
| wrong_value | joules_to_TNT | 4 | `Fully splitting 1.74 g of Pu-239 releases how many megatons of TNT?` (got 0.03737652989966689 via None); `Fully splitting 6.01 kg of Pu-239 releases how many megatons of TNT?` (got 129.09939350402183 via None) | open |
| wrong_value | fourier_transform | 2 | `Fourier transform of exp(-2*t^2)` (got 0 via fourier_transform); `Fourier transform of exp(-2*x^2)` (got sqrt(2)*sqrt(pi)*exp via fourier_transform) | open |
| wrong_value | solve_ode | 1 | `solve y'' + y = 0` (got Eq(y(x), C1*sin(x) + via solve_ode) | set wording f0a0e56 |
| wrong_value | curl | 1 | `curl of (0, t + s, r*s)` (got ['s - 1', '0', '1'] via curl) | fixed 2cdb61d |

## Unverified (45): wild questions with no oracle, for a person to read

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
- `density of gold in g/cm3` -> Mentat 19300.0 kg/m^3 (router); Wolfram answered
- `speed of light in vacuum m/s` -> Mentat 299792458.0 m / s (router); Wolfram answered
- `atomic number of Uranium` -> Mentat 92.0  (router); Wolfram answered
- `mass of Jupiter in kg` -> Mentat 1.89813e+27 kg (router); Wolfram answered
- `luminosity of the Sun in Watts` -> Mentat 3.828e+26 W (router); Wolfram answered
- `Avogadro constant value` -> Mentat 6.02214076e+23 1/mol (router); Wolfram no_result
- `boiling point of water in Kelvin` -> Mentat 373.15 K (cascade); Wolfram answered
- `atomic radius of hydrogen in pm` -> Mentat 13.598 eV (router); Wolfram no_result
- `surface gravity of the Sun in m/s^2` -> Mentat {'field_constants_source': 'JP  (router); Wolfram no_result
- `radius of the Moon in km` -> Mentat 1737400.0 m (router); Wolfram answered
- `atomic radius of hydrogen in picometers` -> Mentat 13.598 eV (router); Wolfram answered
- `density of mercury at 25 degrees Celsius` -> Mentat 13534.0 kg/m^3 (router); Wolfram answered
- `specific heat capacity of aluminum in J/gK` -> Mentat None  (qwen); Wolfram no_result
- `effective temperature of the Sun` -> Mentat {'field_constants_source': 'JP  (router); Wolfram answered
- `orbital period of Earth in days` -> Mentat 365.256 day (cascade); Wolfram answered
- `atomic radius of carbon in picometers` -> Mentat 11.26 eV (router); Wolfram answered
- `standard enthalpy of formation for CO2 in kJ/mol` -> Mentat None  (research); Wolfram answered
- `escape velocity of Jupiter in km/s` -> Mentat 59532.140511916 m/s (fold); Wolfram answered
- `surface gravity of the Moon in m/s^2` -> Mentat 1.625 m/s^2 (router); Wolfram answered
- `density of gold at 20 celsius` -> Mentat 19300.0 kg/m^3 (router); Wolfram answered
- `standard reduction potential of copper` -> Mentat {'GM_m3_s2': 1.996082900999999 counts, kg, m^3/s^2 (router); Wolfram no_result
- `wavelength of visible light peak in nm` -> Mentat 5.020394932432432e-07 m (router); Wolfram no_result
- `speed of light in vacuum in m/s` -> Mentat 299792458.0 m / s (router); Wolfram answered
- `molar mass of water` -> Mentat {'atomic_mass_amu': 1.008, 'at  (router); Wolfram answered
- `density of gold at 20 C` -> Mentat 19300.0 kg/m^3 (router); Wolfram answered
