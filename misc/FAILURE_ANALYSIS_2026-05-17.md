# Failure Analysis — sigma-ground main run, 2026-05-17

Detailed breakdown of the 108 failures in `sigma_ground_run.json`
(42/150 correct = 28.0%). Companion to `IMPROVEMENT_PLAN.md` (which
the daily_job appends to automatically).

## Failure-mode distribution

| Mode                              | Count | % of corpus | What it means                                          |
|-----------------------------------|------:|------------:|--------------------------------------------------------|
| CORRECT                           |    42 | 28.0%       | Tool was found, called correctly, answer matched.      |
| WRONG_TOOL_USED                   |    42 | 28.0%       | Qwen picked a tool that doesn't answer the question.   |
| EXCEEDED_MAX_TURNS                |    33 | 22.0%       | Looped without converging on an answer.                |
| WRONG_VALUE_FROM_CORRECT_TOOL     |    15 | 10.0%       | Right tool called; wrong inputs or wrong extraction.   |
| NO_TOOL_CALLED                    |    14 |  9.3%       | Qwen reasoned in prose, never tried a tool.            |
| ERROR                             |     3 |  2.0%       | MCP/network error.                                     |
| FITTED_DUE_TO_INCOMPETENCE        |  **1**|  0.7%       | True library gap. **Library coverage is solid.**       |

**Key insight: the library is not the bottleneck.** Only 1 of 150
failures was an actual missing physics tool. The other 99% of misses
are tool-selection or input-format problems.

## Top "swiss-army-knife" misuses

Tools Qwen grabs when it doesn't know what else to try:

| Tool                         | Wrongly called |
|------------------------------|---------------:|
| `light_travel_time`          |             10 |
| `hydrogen_like_energy_level` |              7 |
| `solar_system_body`          |              6 |
| `photon_energy_from_frequency` |            5 |
| `power_dissipation_resistor` |              4 |

`light_travel_time` is the most-abused — Qwen routes any "how long /
how far / how much time" question through it. Pattern hints help but
not enough.

## Tools that should have been called but weren't

| Tool                         | Times needed | Notes |
|------------------------------|-------------:|-------|
| `solve_equation`             |            4 | Qwen never thinks "use symbolic solver for inverse problems." |
| `eV_to_joules`               |            3 | Qwen quotes MeV instead of converting. |
| `circular_orbit_velocity`    |            2 | Orbital scenarios mishandled. |
| `blackbody_peak_wavelength`  |            2 | Wien's-law questions mistargeted. |
| `mass_to_energy`             |            2 | E=mc² questions go to other tools. |
| `mond_regime_classifier`     |            2 | Cosmology regime questions. |
| `age_of_universe`            |            2 | |
| `gravitational_time_dilation`|            2 | |
| `joules_to_TNT`              |            2 | TNT-equivalent conversions. |

## Domain-level breakdown

| Domain                       | Correct | Wrong tool | No tool | Max turns | Fitted |
|------------------------------|--------:|-----------:|--------:|----------:|-------:|
| `electromagnetism_intro`     |  10/15  |          0 |       0 |         4 |      0 |
| `quantum_mechanics`          |  10/12  |          0 |       0 |         1 |      0 |
| `waves_optics`               |   6/12  |          3 |       2 |         0 |      0 |
| `electrodynamics_advanced`   |   4/10  |          3 |       0 |         2 |      0 |
| `atomic_molecular`           |   3/ 8  |          0 |       0 |         4 |      0 |
| `classical_mechanics_intro`  |   3/15  |          3 |       4 |         1 |      1 |
| `thermodynamics_statmech`    |   2/12  |          6 |       1 |         1 |      0 |
| `modern_physics`             |   2/12  |          5 |       0 |         1 |      0 |
| `astrophysics`               |   1/12  |          0 |       0 |        10 |      0 |
| `classical_mechanics_advanced`|  1/10  |          1 |       1 |         6 |      0 |
| **`cosmology`**              | **0/8** |          7 |       0 |         1 |      0 |
| **`general_relativity`**     | **0/10**|          7 |       1 |         1 |      0 |
| **`nuclear_physics`**        | **0/ 7**|          7 |       0 |         0 |      0 |
| **`mathematical_methods`**   | **0/ 7**|          0 |       5 |         1 |      0 |

The four 0%-domains have **different** failure modes:
- **Cosmology / GR / Nuclear**: wrong tool used. Tools exist; Qwen
  doesn't connect "Sun collapsed into BH" → `schwarzschild_radius`.
  The newly-shipped keyword hints (commit 6128d5d) should help here.
- **Mathematical methods**: no tool called. Qwen reasons in prose
  through algebra/calculus instead of invoking `solve_equation` /
  `integrate_expr` / `differentiate_expr`.
- **Astrophysics**: max turns. Qwen calls `named_star` or
  `solar_system_body` repeatedly and never chains to the formula tool.

## Specific bugs surfaced

### Bug 1 — `free_fall_time` silently accepts unknown gravity param

`mech_intro_011` ("drop ball on Moon", g=1.625):

  - Qwen called `free_fall_time(height_m=10, gravity_ms2=1.625)`
  - The function signature is `free_fall_time(height_m, g_m_s2=...)`
  - Pydantic ignored the unknown `gravity_ms2` and used the default `g_m_s2=9.80665`
  - Result: 1.428s (Earth answer) instead of 3.508s (Moon answer)

**Fix options:**
- Add `extra="forbid"` to pydantic config so unknown kwargs error out
- Add parameter aliases (`gravity_ms2`, `g_m_s2`, `g` all accepted)
- Rename to match the most-common Qwen usage

### Bug 2 — Parameter-name mismatches (Qwen invents synonyms)

| Question        | Tool                          | Qwen passed                  | Tool expects               |
|-----------------|-------------------------------|------------------------------|----------------------------|
| `mech_intro_003`| `projectile_range`            | `initial_velocity_m_s`       | `initial_speed_m_s`        |
| `mech_adv_003`  | `escape_velocity`             | `body_name='sun'`            | `mass_kg` (no body_name)   |
| `mech_adv_007`  | `escape_velocity`             | `planet_name='jupiter'`      | `mass_kg`                  |
| `modern_004`    | `mass_to_energy`              | `{}` (empty!)                | `mass_kg`                  |
| `modern_007`    | `relativistic_velocity_addition`| `v1, v2`                   | `u_m_s, v_m_s`             |
| `mech_adv_010`  | `gravitational_potential_energy`| `G, M, m, r`               | `mass_planet_kg, mass_object_kg, height_m` |

The `body_name='sun'` cases reveal a **chaining failure**, not a
naming issue: Qwen wanted `escape_velocity` to look up the Sun's mass
internally instead of chaining `solar_system_body(sun)` →
`escape_velocity(mass_kg=...)`.

### Bug 3 — Scorer doesn't fully handle radian-vs-Hertz

| Question        | Tool returned       | Expected     | Issue              |
|-----------------|---------------------|--------------|--------------------|
| `em_intro_005`  | 31622 rad/s         | 5032.9 Hz    | Need /(2π) factor  |
| `em_adv_006`    | 1e9 rad/s           | 1.59e8 Hz    | Same               |

Pint can convert these in principle but only via explicit context. The
scorer's `_try_unit_convert` should be extended with a Hz↔rad/s alias.

### Bug 4 — Unit conversions the scorer should already do but didn't

| Question        | Tool returned  | Expected     | Why didn't scorer fix it? |
|-----------------|----------------|--------------|---------------------------|
| `astro_001`     | 133859 s       | 4.24 year    | Should auto-convert       |
| `astro_004`     | 17345332 s     | 547 year     | Same                      |
| `thermo_008`    | 1811.15 K      | 1538 C       | K→C is offset, not scale  |

The scorer probably tried `pint.Quantity(1811.15, 'K').to('C')` and
got 1538 C (correct!). Need to verify the scorer is invoking this
path. If `expected_units` is empty (`""`) then the conversion path
short-circuits — check that the corpus has proper expected_units.

### Bug 5 — Multi-tool result confusion (fallback grabs wrong tool's value)

| Question      | Last good tool result | Qwen reported  |
|---------------|-----------------------|----------------|
| `mech_intro_009` | 37500 kg·m/s (momentum, correct) | 8.0 V (from earlier Ohm's law in trace) |
| `atom_004`    | 24.587 eV (He IE, correct)           | 15.76 eV (Ar IE, from sibling lookup) |
| `em_adv_002`  | 60 W (correct draw)                  | 60 W — but Q wanted 60×0.02 = 1.2 W |

`_extract_value_from_tool_calls` grabs the LAST tool call's value. If
Qwen made multiple unrelated calls, the last one wins regardless of
relevance. Should prefer the call matching the expected primary tool,
or the one with the most semantic overlap with the question.

## Recommendations (in rough priority order)

1. **Add pydantic `extra="forbid"` to all tool argument models.** Forces
   Qwen to use exact param names — bug 1 and bug 2 disappear (with
   visible error messages Qwen can correct on).

2. **Add param-name aliases for the top 5 misused params.**
   `g_m_s2`/`gravity_ms2`/`g`/`gravity`, `initial_speed_m_s` vs `velocity`,
   `mass_kg` accepting `m`/`mass`. Surgical, high leverage.

3. **Fix `escape_velocity` to optionally accept `body_name`.** Currently
   requires `mass_kg`. Add `body_name` as alt input; internally call
   `solar_system_body(body_name).value['mass_kg']`. Saves the chaining
   step Qwen keeps failing on. Same for `gravitational_potential_energy`,
   `circular_orbit_velocity`.

4. **Strengthen the scorer's unit-conversion path.** Verify pint handles
   K↔C, s↔year, rad/s↔Hz (Hz needs angular-frequency context).

5. **Improve the multi-tool fallback extractor.** Prefer the call whose
   tool name matches `q['primary_tool_expected']`, or whose units match
   `expected_units`. Fall back to "last call" only if no match.

6. **Tighten the prompt for math_methods domain.** Add a rule:
   "If the question asks to solve, integrate, or differentiate, you
   MUST call `solve_equation` / `integrate_expr` / `differentiate_expr`.
   Do not work the algebra in prose."

7. **For astrophysics**: questions like "how long does light from X
   take to reach us" need chaining `named_star(X) → light_travel_time`.
   Add a chaining hint in the system prompt, or build a composite
   `light_travel_time_from_star(name)` tool.

8. **Run the enlightened-prompt test against the full corpus** to
   quantify how much keywords + aliases moved the 28% baseline.
   Initial weak-domains result: cosmology 60% (was 0%), GR 40% (was 0%).
