# Code Audit — sigma-ground-mentat — 2026-06-05

Scope: bug hunt, dedup, comment coverage, unit-test gap. Driven by 4 parallel
audit agents + direct investigation. `sigma_ground/materia/` and the sibling
`D:\Aaron\development\sigma-ground` (Deckard) tree were excluded as owned-elsewhere.

---

## 0. FIXED THIS SESSION (highest priority)

### 0.1 🐛 Water boiling point off by 17 K (FIXED + tested)
`field/interface/liquid_water.py::water_enthalpy_of_vaporization` returned the
uncalibrated H-bond estimate `3.5·0.23eV·N_A/2 ≈ 38,835 J/mol` (4.5% low), so
`water_boiling_point()` returned **356 K (83 °C)** instead of 373 K. Same class
as the 5 water bugs fixed earlier this session (density/σ/η). **Fix:** return the
measured CRC/NIST value 40,660 J/mol → boiling point now **373.03 K**. Added
`test_boiling_point_matches_measured` + `test_enthalpy_vaporization_matches_measured`
(Golden Rule 8). Side benefit: `atmosphere.saturation_vapor_pressure(20 °C)`
improved from 3328 Pa (+42%) to 2835 Pa (+21%).

### 0.2 🚨 Benchmark switchboard regression 85.3% → 52.7% (FIXED, two causes)
This session's feature work silently regressed the Qwen-7b Q&A benchmark. Root
causes found by failure-mode analysis of `sigma_ground_fresh_run.json`:

1. **Param-alias gap (validation-error loops).** `param_aliases.PARAM_ALIASES`
   maps `velocity → speed_m_s`, but `kinetic_energy`/`momentum`/etc. take
   `velocity_m_s`. The rename was skipped (target not in signature), so Qwen's
   `velocity` reached the tool raw → pydantic "Field required" → the model
   re-called the same tool 8× → `<exceeded max turns>` → None. **Fix:** added a
   general **prefix-fallback** to `normalize_kwargs` — a bare kwarg renames to a
   uniquely-matching `<name>_<unit>` real param (`velocity→velocity_m_s`,
   `mass→mass_kg`, …). `param_aliases.py`.
2. **Context truncation (empty replies).** The ollama call never set `num_ctx`,
   so the ~13k-token system prompt + 219 tool schemas overflowed the small
   default context → ollama truncated the input → the model returned an empty
   reply with **no tool call** for ~30/150 questions (June answered 28 of those
   30 via tools). This session's +289 colloquial keywords tipped the prompt over
   the edge. **Fix:** `num_ctx=32768` on both ollama calls. `run_sigma_ground.py`.

Verified on a 16-Q recheck: regressed ≈50% → 62% (velocity fix) → **94% (both
fixes)**, 0 empties, 0 loops. Full 150-Q re-run in progress to confirm recovery
to the ~85% baseline.

### 0.3 ⚠ inventory/field r₀ divergence (FIXED) + two minor items
- **`inventory/core/sigma.py` SEMF Coulomb coefficient** hardcoded r₀=1.25 fm →
  a_C=0.691 MeV, diverging ~3% from `field`'s canonical 0.711 MeV (r₀=1.215 fm),
  so binding energies differed between layers. **Fix:** import `A_C_MEV` from
  `field.constants` (single source; corrected to 0.711). Inventory tests pass;
  `nuclear_binding_mev` reduces to BE at σ=0.
- **`electrodynamics.skin_depth(n_e, omega)`** silently ignored `omega` (returned
  `c/ω_p` always) — a latent bug for any caller. **Fix:** implemented the
  collisionless evanescent form `δ = c/√(ω_p²−ω²)` (default ω=0 ⇒ c/ω_p
  preserved; ω≥ω_p ⇒ inf). Clarified vs the separate `mobius.skin_depth`.
- **`thermoelectric.seebeck_coefficient`** docstring had a confusing "Wait—"
  derivation; cleaned to clearly state the (π²/2) free-electron result the code
  uses. (Code was already correct.)
- **8 ghost `tool_keywords` entries** (for non-existent tools) removed; added
  `test_tool_keywords.py` guard (`set(TOOL_KEYWORDS) ⊆ manifest`).

---

## 1. Bug hunt (remaining — not yet fixed)

The physics library is otherwise high quality: the agent numerically verified
~18 modules (<1% vs textbook/CODATA). Remaining items, by confidence:

- **MEDIUM — `atmosphere.saturation_vapor_pressure` is +21% at 20 °C** but the
  docstring claims "±5%". It uses boiling-point ΔH_vap in Clausius-Clapeyron for
  all T; real ΔH_vap rises to ~44 kJ/mol at 20 °C. **Recommend:** replace with
  Magnus-Tetens `610.94·exp(17.625·Tc/(Tc+243.04))` (Pa, <0.5% over 0-50 °C) and
  update `dew_point`'s inversion to match (coupled — 6 call sites + test_atmosphere).
  Deferred this session to avoid risk on the critical path.
- **LOW — `thermoelectric.seebeck_coefficient`** docstring says `(π²/3)` but code
  uses `(π²/2)`. The code is the rigorous σ(E)∝E^{3/2} result (defensible); the
  docstring contradicts it. Fix the docstring.
- **LOW — `electrodynamics.skin_depth(omega=...)`** ignores its `omega` arg
  (always returns collisionless `c/ω_p`). Misleading API; default result is right.

## 2. Dedup (ranked by impact)

- **`9.80665` (std gravity) hardcoded in 15 files / ~27 sites, not in any
  constants module** (as `_G_EARTH`, `_G_STANDARD`, inline defaults). → add
  `G_STANDARD` to `field/constants.py`, import everywhere. Highest churn-reduction.
- **`1.602176634e-19` (e / eV→J) re-hardcoded in 15 files** despite
  `field/constants.py` exporting `E_CHARGE`/`EV_TO_J`. Plus `c` in 11 files,
  `k_B` in 8, `AMU_KG` in 6 (4 copies inside one function in `magnetism.py`).
- **⚠ `inventory/core/sigma.py` is a parallel re-implementation of
  `field/scale.py` + `field/constants.py` + `field/binding.py`** with a
  **divergent r₀ (1.25 fm vs the canonical 1.215 fm)** → numerically different
  SEMF Coulomb coefficient between layers. `scale_ratio()` is byte-identical to
  `field/scale.py`. **Correctness risk, not just style.** → import from `field`,
  delete local re-derivations, reconcile r₀.
- **`mcp/tools/gr.py` + `frontier.py` re-derive GR formulas inline** instead of
  delegating to `field/gr_basics.py` (which `tools/relativity.py` already does
  correctly). ~8 functions duplicated.
- **89-site `ToolResult(value=None, source="invalid input", …)` boilerplate**
  across 13 `mcp/tools/` files → add an `invalid_input()` helper to `provenance.py`.

## 3. MCP layer (mostly clean)

- No duplicate `@server.tool()` registrations (the `parallel_plate_capacitance`
  "dup" was already de-duped — circuits.py wired, electronics.py dormant). No
  wrapper signature mismatches; no default-arg crashes.
- **8 dead `tool_keywords` entries** for tools that no longer exist
  (`ohms_law_resistance`, `thermal_conductivity`, the F/C temp converters, …) —
  inert (don't leak to qwen_context) but should be deleted; add a
  `set(TOOL_KEYWORDS) ⊆ registered` guard test.
- **55 registered tools have no trigger phrases** (no inline `keywords` and no
  `tool_keywords` row) → discoverable by name but weaker paraphrase routing.
  Includes all `procedure_*`, the matrix/calculus tools, many chemistry tools.
- LOW: `_safe()` swallows all exceptions in composite tools (dormant today —
  no live `None` sub-values); `units="pH"` is a non-unit string in chemistry.py.

## 4. Unit-test gap

- **258 source modules, 112 test files, 4145 tests** (CLAUDE.md's "~2047" is stale).
- **Execution coverage ≈100%** (the `trace_coverage`/`coverage_harness`/
  `test_coverage_ledger` system) but that only proves functions *run* — the
  return value is discarded. **Value-asserting** coverage is the real gap:
  - `field/interface/` — STRONG (~94% of funcs, NIST/IAPWS-anchored).
  - `mcp/tools/` — WEAK (~21/43 modules value-asserted).
  - `materia/`, `dynamics/`, `radiance/`, `kernel/`, `mcp/server.py` — near-dark.
- A function can be "100% covered" and still return a value off by 4π. The new
  `test_wolfram_parity.py` (34 cases) is the only deterministic answer-value
  cross-check — broad but explicitly a sample, not proof of all ~1000 functions.
- **Top missing value-tests:** `optics.{plasma_frequency,metal_reflectance,
  cauchy_n}`, `quantum.{double_slit_intensity,fringe_visibility}`, the entire
  `fluid.py` module (eyring_viscosity, reynolds_number, surface_tension),
  `phosphor.phosphor_brightness`, and the `mcp/tools/` wrappers
  (nuclear/orbital/quantum_solids/materials_strength/plasma_em).

---

## Recommended next actions (priority order)
1. Confirm the 150-Q recovery (re-run in progress) → then the switchboard is back.
2. `field/constants.py`: add `G_STANDARD`; sweep the 15-file `9.80665` + `e`-charge dups.
3. Reconcile `inventory/core/sigma.py` r₀ (1.25→1.215) — correctness divergence.
4. Magnus-Tetens for `atmosphere.saturation_vapor_pressure` + `dew_point`.
5. Backfill value-asserting tests for the dark `mcp/tools/` wrappers + optics/quantum/fluid.
