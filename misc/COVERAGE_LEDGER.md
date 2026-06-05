# Transitive-Coverage Ledger (honest denominator)

**Goal:** every standard-physics leaf function in `sigma_ground/field/**` and
`sigma_ground/inventory/**` is reached by at least one *realistic* MCP question
whose answer validates against a textbook formula or external source — OR is
explicitly listed below as a non-capability and excluded from the denominator.

**Measurement:** `misc/trace_coverage.py` installs a call tracer, runs every
`sigma_ground/mcp/tools/*` function plus 5 procedures with auto-inferred
realistic args (`misc/coverage_harness.py`), and takes the union of every
`sigma_ground` function that executed. A function is *covered* iff it executes
as a byproduct of a question — direct or transitive.

**Denominator honesty rule:** we NEVER fake-cover. A function is either
(a) WIRED — reached by a real, validated question, or
(b) EXCLUDED — listed here with a reason (render/plumbing/private/dead), or
(c) REMOVED — deleted from the library (was dead/speculative).
The final 100% target = covered == (total − EXCLUDED). The regression test
(added at the end) asserts the tracer reaches 100 % of the honest denominator.

---

## Progress

| Baseline | Covered | Total | % | Note |
|----------|---------|-------|---|------|
| C        | 909     | 1093  | 83% | pre-marathon |
| batch 1-3 (mechanics/transport/rotation) | — | 1093 | — | → 909 |
| batch 4 (materials strength) | 931 | 1093 | 85% | elasticity/stress/plasticity/composites |
| batch 5 (photonics/electroceramics) | 946 | 1093 | 87% | waveguide/bandgap/nonlinear/color/phosphor/piezo/dielectric |
| batch 6 (thermal-systems/mech-response) | 964 | 1093 | 88% | TEG/convection/viscoelastic-creep/acoustic-interface |
| batch 7 (devices/quantum-solids) | 976 | 1093 | 89% | capacitor/Hall/junction/BCS-gap/tunneling/box/DOS/exchange |
| batch 8 (plasma-em/relativity-spectra) | 988 | 1093 | 90% | plasma-params/EM-forces/relativistic-energy/Zeeman |

---

## EXCLUDED functions (non-capabilities — out of the denominator)

These are presentation, plumbing, private-by-intent, or dead code. They are not
physics capabilities a user would ask Mentat to compute, so they do not count
against coverage. Listed by the batch in which they were ruled out.

### batch 5
- `field.interface.phosphor.build_ascii_histogram` — ASCII-art terminal render
  helper (`█` bars). Presentation, not physics. EXCLUDE.

### batch 6
- `field.interface.viscosity.viscous_flow_properties` — explicitly a
  "Nagatha-compatible format" export aggregator; it only re-bundles physics
  already covered directly (stokes_drag, poiseuille_*, terminal_velocity_stokes,
  particle_reynolds_number). Serialization helper, not a new capability. EXCLUDE.

### batch 7
- `field.interface.superconductivity.block_cooling_profile` — multi-step
  *simulation* returning a per-temperature-step series (resistivity/gap/Meissner/
  London/H_c). It only re-bundles already-covered physics (bcs_gap_temperature,
  meissner_fraction, london_penetration_at_T, thermodynamic_critical_field). A
  simulation pipeline (Materia's lane), not a scalar Q&A. EXCLUDE.

---

## DEFERRED (real capability, but current model fails the accuracy bar)

NOT exposed and NOT counted as covered — a known gap to fix, then wire. Distinct
from EXCLUDE (which is "not a capability at all"). Listed so the regression test's
honest denominator can subtract them while keeping them visible as TODO.

- `field.interface.thermal.contact_conductance` — returns h ~ 1.9e9 W/(m^2.K)
  at 1 MPa for Cu-Al (scales linearly with pressure). Real engineering joint
  conductance is ~1e3-1e5 W/(m^2.K); this model uses an atomic-scale gap length,
  yielding a near-ballistic value 4-5 orders too high. "Never confidently wrong"
  -> do not expose until the gap-length / asperity model is reviewed. (Review
  task spawned.)
- `field.interface.superconductivity.{gl_parameter, lower_critical_field,
  upper_critical_field}` — the critical-field chain is broken: for every DB
  superconductor `thermodynamic_critical_field` yields H_c ~12 orders too small
  (Al should be ~8000 A/m, model implies ~2.5e-9), so Hc1/Hc2 come out ~1e-9 to
  1e-18 A/m AND inverted (Hc1 > Hc2). `gl_parameter` returns only the clean-limit
  kappa (e.g. Nb 0.11 vs measured ~1.05) and feeds the same machinery. Wired only
  `gap_frequency` (correct, 677 GHz for Nb). Whole Hc chain deferred. (Review
  task spawned.)
- `field.interface.plasma.spitzer_resistivity` — carries a spurious extra
  `1/(4 pi eps0)` factor (line ~220, labelled a "Dimensional prefactor /
  correction") that inflates the result by ~9e9. Returns ~1.7e5 ohm.m for a
  10^6 K hydrogen plasma; the NRL Spitzer value is ~1e-6 ohm.m (hot plasmas are
  near-perfect conductors). Drop the bogus factor and it's right-order. Wired
  the other 4 plasma parameters (Debye length/number, ln Lambda, Larmor radius,
  all validated). Deferred pending the one-line fix. (Review task spawned.)

---

## Pending EXCLUDE candidates (to confirm against uncovered.txt in later batches)

Categories flagged during Phase-0 triage, to be enumerated function-by-function
when their batch is reached (do NOT pre-list unverified):
- Nagatha-format exporters / serialization helpers
- ASCII / render / color-swatch display helpers
- CDF-sampling internals (private Monte-Carlo plumbing)
- `rolling_analysis` / `rolling_shootout` simulation pipeline (sim, not Q&A)
- orbital ephemeris least-squares fit internals
- QC gate/state primitives (covered indirectly via high-level QC tools)
- inventory env-resolver / builder / checksum plumbing (internal-only)
