# Nuclear and degenerate matter in Mentat: implementation plan (gap B1)

**Item:** N1 of the round-2 research run · **Date:** 2026-10-10 · **Status:** PLAN ONLY, as agreed with the Captain. No code changes to the repo.
**Gap closed:** B1 in `D:/Aaron/development/sigma-ground/misc/MENTAT_GAPS_UNIFICATION.md`. That gap covers electron and neutron Fermi-gas EOS in both regimes, a semi-empirical mass formula, saturation and symmetry energy, a TOV integrator, composition layers and the bag-model deconfinement pressure.
**Rules in force:** Golden Rules 1–7, PLATINUM §3 (one source of truth), §10 (if one, then all), §14 (tests land with the code), §21 (Mentat does the physics), and "Mentat flags, never fixes".
**Companion files (this folder):** `n1_survey.py` (the survey, read-only on the repo), `n1_survey.json` (every number below with its Mentat `source`), `n1_references.json` (31 references with how each was verified), `n1_figure.py` → `n1_plan_figure.png`.

Tags used: [MEASURED] = computed this session by Mentat or a repo engine · [DERIVED] = follows from stated inputs by algebra/code read · [LIT] = literature, citation verified by web search this session · [BY CONSTRUCTION] = true by the way the code is built (engine validation, not physics) · [SPECULATIVE] = rests on the σ-bridge postulate.

---

## 0. The plan on one page

**Why.** The σ-bridge sources σ by the trace ε − 3P. Pressure is an order-one share of that source only inside neutron stars. Using P6's own [LIT] rows and Mentat's c, 3P/ρc² is 5.6 × 10⁻⁶ at the Sun's core, 3.3 × 10⁻⁴ at a white-dwarf centre and 0.67 at a neutron-star centre [MEASURED from [LIT] inputs]. Mentat cannot build either kind of degenerate star today. No Fermi-gas EOS, nuclear-matter EOS or TOV integrator exists in `sigma_ground/`, and no numeric ODE tool either: `solve_ode` is symbolic `dsolve`, and it refused the Lane–Emden input. P6 and P7 therefore had to borrow their dense-matter rows.

**What.** One new tier-1 package, `sigma_ground/field/dense/`, plus a pure-Python adaptive ODE integrator in `kernel/`. The modules:

| # | module | delivers |
|---|---|---|
| M0 | `kernel/ode.py` | Dormand–Prince 5(4), dense output, terminal events; the core stays zero-dependency |
| M1 | `field/dense/fermi_gas.py` | exact T = 0 Chandrasekhar functions for e, μ, n, p (NR ↔ UR); NR finite-T Fermi–Dirac (degenerate ↔ classical) |
| M2 | `field/dense/polytrope.py`, `white_dwarf.py` | Lane–Emden, Chandrasekhar mass, white dwarfs with Salpeter Coulomb corrections |
| M3 | `field/dense/semf.py` | ONE semi-empirical mass formula, fitted to AME2020, replacing the two divergent copies |
| M4 | `field/dense/nuclear_matter.py` | saturation and symmetry energy (meta-model on measured parameters, plus Skyrme SLy4), β-equilibrium npeμ |
| M5 | `field/dense/eos.py`, `tov.py` | EOS protocol; TOV with Tolman mass, trace mass, baryon mass, M–R curves, M_max |
| M6 | `field/dense/composition.py` | outer crust (BPS), neutron drip, crust–core edge, layer radii from a TOV profile |
| M7 | `field/dense/quark_matter.py` | MIT bag EOS, Maxwell deconfinement against M4, hybrid and strange stars |
| M8 | `field/dense/trace.py` | GR trace diagnostics (3P/ε, Δ, sign-flip radius, M_tr/M) and an opt-in [SPECULATIVE] σ profile on the TOV metric |
| M9 | `mcp/tools/dense_matter.py` | 17 new Mentat tools (plus `nuclear_binding_energy` re-pointed), manifest and keywords, deploy |

**Order and effort.** P0 foundations → P1 Fermi gas → P2 white dwarfs → P3 TOV → P4 trace diagnostics (P7 rerun beyond the weak field) → P5 SEMF → P6 nuclear matter and neutron stars → P7 layers → P8 quark cores → P9 MCP sweep and deploy. That is about **12.75 working days**, with tests landing with every module.

**Milestones.**
- **MS1** (≈ 4.3 d): Chandrasekhar mass and the free-neutron star computed, not borrowed.
- **MS2** (≈ 5.3 d): P7 rerun on the exact static metric.
- **MS3** (≈ 8.5 d): 3P/ε of order one computed along real neutron-star sequences, validated against NICER, PSR J0740+6620, GW170817 and the Podkowka–Mendes–Poisson trace-flip compactness. The σ-bridge pressure term becomes scoreable.
- **MS5** (≈ 12.75 d): live in Mentat.

**Found on the way (flagged, not fixed):**
- `element.py` writes the SEMF asymmetry term 4× too weak. Pb-208 binding comes out +8.8 % high and U-238 +11.6 %. It falls to −1.1 % and −0.3 % with the convention the same file's `stable_mass_number` assumes [MEASURED].
- Two different SEMF coefficient sets exist.
- The saturation constants in `field/constants.py` are uncited.
- `audit.py` claims a TOV that lives only in `../ssbm-theory`.
- `planets.NEUTRON_STAR` is hand-entered.
- **PSR J0348+0432 is no longer a 2 M☉ pulsar**: it was revised to 1.806 ± 0.037 M☉ (Saffer et al. 2025) [LIT]. The "≥ 2 M☉" anchor now rests on PSR J0740+6620 alone.

![plan figure](n1_plan_figure.png)

---

## 1. What exists today (the survey)

Grep over `sigma_ground/{field,inventory,mcp,dynamics}` and `scripts/` [MEASURED, `n1_survey.json → grep_hits`]: `white_dwarf` 0 files, `equation_of_state` 0, `bag.model` 0, `symmetry_energy` 0, `Lane.Emden` 2 (both comments in scripts), `TOV` 6 (none an integrator), `Chandrasekhar` 3 (an audit entry and two unification scripts).

| where | what is there | verdict for B1 |
|---|---|---|
| `field/constants.py:395-411` | `N0_FM3 = 0.16`, `E_SAT_MEV = −16`, `K_SAT_MEV = 230`, `J_SYM_MEV = 32` | values right in magnitude; **no citation, no uncertainty, no L_sym**; no engine uses them (only `scripts/unification/sigma_exclusion.py`) |
| `field/constants.py:268-276` | `R0_FM = 1.215`, `A_C_MEV` derived from Coulomb's law (0.7111 MeV) | keep: the one SEMF coefficient already derived (Golden Rule 1) |
| `field/binding.py` | Coulomb term; σ-scaling of a *given* binding energy | keep; M3 supplies the binding energy |
| `field/interface/element.py:110-116, 485-528` | SEMF set #1 (15.56 / 17.23 / A_C / 23.29 / 12.0), "FITTED", uncited | **defect** (asymmetry convention, §1.1); superseded by M3 |
| `mcp/tools/nuclear.py:57-72` | SEMF set #2 inline (15.75 / 17.8 / 0.711 / 23.7 / 11.18) + private `u`, `m_H`, `m_n` | PLATINUM §3 duplicate; internals move to M3, tool API kept |
| `field/nucleon.py` | `proton_mass_mev(σ)`, `neutron_mass_mev(σ)` | the σ hook M1/M4 call (σ = `SIGMA_HERE` ⇒ standard physics). A second copy lives in `inventory/core/sigma.py` (existing §3 duplicate, noted) |
| `field/interface/statistical.py:124` | Fermi–Dirac occupation only | no EOS |
| `field/interface/quantum_matter.py:119-160` | metal free-electron E_F, Thomas–Fermi length, keyed by material | not a degenerate-star EOS |
| `field/planets.py:137-176` | `NEUTRON_STAR`: 1.4 M☉, 11 km, four layers with round densities, uncited | replaced by M6 output (PLATINUM §10) |
| `field/audit.py:247-279` | ledger entries "Chandrasekhar Fermi gas (exact)", "TOV integration (full)", "M_TOV = 2.071 M☉" | the code is in `../ssbm-theory`, not here; re-point when M5 lands |
| `inventory/data/isotopes.json` (+ `loader.IsotopeDB`) | 145 nuclides refreshed from AME2020 | the SEMF fit set for M3 and the measured masses for M6 |
| `mcp/tools/mathx.py:209` `solve_ode` | sympy `dsolve` only; the Lane–Emden ODE was refused ("looks like code") [MEASURED] | Mentat has no numeric IVP tool; M0 fills it |
| `pyproject.toml:12` | core `dependencies = []`; scipy only in the `mcp` extra | integrators must be pure Python (M0) |
| `SkippyTel/mentat/Dockerfile:9` | Mentat container pins sigma-ground at a commit (`SIGMA_GROUND_REF`) and exposes every public tool; `MENTAT_CALL_TIMEOUT = 90` s | deploy = push, bump the ref, rebuild `mentat-home`; every tool must answer in ≪ 90 s |

**Prior art outside the repo.** `../ssbm-theory/sigma_ground/field/unsolved.py:148-635` has a Chandrasekhar EOS and a TOV. It is not portable [DERIVED from a code read]:
- It covers pure neutron matter only, with no β-equilibrium.
- An interaction term tuned so that it *alone* saturates is added on top of the full neutron kinetic term. The total E/A(n₀) is therefore not −16 MeV and P(n₀) ≠ 0.
- The fitted K is the interaction part's only.
- Negative pressures are clamped to 0, which is thermodynamically inconsistent.
- Integration is forward Euler at dr = 10 m.
- σ comes from a fixed R = 12 km.

Its "M_TOV = 2.071 M☉" is not reproducible evidence. The parallel-session scratch harness `scratchpad/sigma-bridge-1pn/tov_sigma_pair.py` is a better seed. It has a DOP853 TOV, the Tolman mass, a σ monopole on the TOV metric, and two exact benchmarks. Its tests are reused in M5 and M8.

### 1.1 Engine and document defects found (Mentat flags, never fixes)

| # | file:line | defect | evidence | disposition |
|---|---|---|---|---|
| N1-a | `field/interface/element.py:518` | asymmetry term `a_a (A−2Z)²/(4A)` with `a_a = 23.29`, the coefficient of the `(A−2Z)²/A` form that `stable_mass_number()` (line 563, `coeff = a_c/(2 a_a)`) assumes ⇒ penalty 4× too small | vs AME2020: Pb-208 +8.84 %, U-238 +11.58 %; with the consistent form −1.09 %, −0.30 %; rms (A ≥ 12) 5.93 % → 0.79 % [MEASURED, Fig. b] | M3 lands the fix behind a failing-first regression test (Captain's OK, D6) |
| N1-b | `mcp/tools/nuclear.py:62` | second SEMF set inline; private `_U_MEV`, `_M_H_ATOM_U`, `_M_NEUTRON_U` | rms 1.14 % (A ≥ 12), He-4 −19.3 % [MEASURED]; PLATINUM §3 | M3 replaces it; the tool keeps its signature |
| N1-c | `field/constants.py:399-411` | saturation anchors uncited, no uncertainties, no L_sym | read; `lookup_constant` source is only "sigma_ground.field.constants" | P0 cites them (§8) and adds `L_SYM_MEV` |
| N1-d | `field/audit.py:266` | lists a live "TOV integration (full)" and "M_TOV = 2.071 M☉" | no integrator under `sigma_ground/` | re-point to M5 when it lands |
| N1-e | `field/planets.py:138` | hand-entered neutron star | read | M6 derives it |
| N1-f | `GOLDEN_RULES.md:31` | Rule 2 points at `sigma_ground/field/bounds.py`, removed in e66615b (SSBM excision) | `ls` fails | doc fix; the new modules return a local SAFE/EDGE/WALL/BEYOND label |
| N1-g | `field/constants.py:221` vs `lookup_constant('m_e')` | m_e stored as CODATA 2018 while Mentat's lookup serves scipy CODATA 2022 | relative difference 1.4 × 10⁻⁹ [MEASURED] | harmless; noted for §3 |

---

## 2. Design principles (how B1 obeys the rules)

1. **Placement.** All physics lives in `sigma_ground/field/dense/` (tier 1), next to `field/binding.py` and `field/nucleon.py`. It imports only `field.constants`, `field.nucleon`, `kernel.ode` and the AME table through `inventory.data.loader.IsotopeDB`. `inventory` is tier 1 too, so the layering test passes; `field/interface/ore_abundance.py` already does this. The MCP layer (`mcp/tools/dense_matter.py`) is tier 4. `tests/test_layering.py` guards all of it.
2. **Zero dependencies.** The core is `dependencies = []`, so ODEs, quadrature and root-finding are pure Python: M0, plus Brent and Gauss–Legendre/Laguerre helpers. scipy may appear only in tests, as an independent cross-check.
3. **Golden Rule 1 (derive, don't assume).** Every module's docstring carries a derivation chain: measured seeds → identities → outputs, each step tagged MEASURED / FIRST_PRINCIPLES / FITTED. Fitted numbers are fitted *in the repo* from cited data (the M3 SEMF fit) or copied from a cited fit (the SLy4 parameters, the bag constant range).
4. **Golden Rule 2 (bounds).** Each public function enforces its preconditions (`ValueError`) and returns a `regime` label:
   - SAFE: inside validity.
   - EDGE: approaching the breakdown, e.g. T/T_F > 0.1 for a T = 0 EOS, or c_s² > 0.9.
   - WALL: the breakdown itself, e.g. neutron drip reached in the WD EOS, or causality violated.
   - BEYOND: outside validity.
5. **Golden Rule 4 (no gaps).** Every simplification gets a `# PHYSICS_GAP:` at the point of omission and a `KNOWN_GAPS.md` line (§9 lists them).
6. **Golden Rule 7.** Matter is fact; the bag constant B is not a measurement. It is a *required* argument with its cited range in the docstring and has no silent default. Meta-model parameters the data do not fix (Q_sat, K_sym) default to 0 under a PHYSICS_GAP and can be swept.
7. **The σ hook.** Every EOS takes `sigma=SIGMA_HERE` and gets nucleon masses from `field/nucleon.py`; σ = `SIGMA_HERE` reproduces standard physics exactly. The engine **never imports ξ**, because C1 (where ξ lives), C2 (normalisation), C3 (value), B5/C4 (sign and which □) and C5 (range) are open. The [SPECULATIVE] σ profile in M8 takes all of them as required arguments.
8. **One EOS protocol.** An `EOS` object exposes `pressure(n)`, `energy_density(n)`, `n_of_P(P)`, `eps_of_P(P)`, `sound_speed_sq(n)` and `regime(n)`. It is built from an analytic model or tabulated in (log n, log P, log ε) with monotone cubic interpolation. A self-check enforces the first law dε/dn = (ε + P)/n to 10⁻⁶. M5 and M8 accept any `EOS`. A registry names the built-ins: `free_neutron`, `electron:mu_e`, `polytrope:n,K`, `metamodel`, `SLy4`, `SLy4+BPS`, `bag:B`, `hybrid:SLy4+bag:B`.
9. **Units.** SI at the public boundary (Pa, J/m³, kg/m³, m⁻³). MeV and fm helpers inside nuclear modules use `HBAR*C` and `MEV_TO_J` from `constants.py`, never typed literals. This fixes the `1.78266192e-30` and `0.16` literals in the ssbm-theory prior art.

---

## 3. Package layout

```
sigma_ground/kernel/ode.py                 M0  tier 0  pure-Python DOPRI5(4) + dense output + events
sigma_ground/kernel/test_ode.py
sigma_ground/field/dense/__init__.py           tier 1  public re-exports
sigma_ground/field/dense/eos.py            M5a         EOS protocol, tabulation, registry, first-law check
sigma_ground/field/dense/fermi_gas.py      M1
sigma_ground/field/dense/polytrope.py      M2a
sigma_ground/field/dense/white_dwarf.py    M2b
sigma_ground/field/dense/semf.py           M3
sigma_ground/field/dense/nuclear_matter.py M4
sigma_ground/field/dense/tov.py            M5b
sigma_ground/field/dense/composition.py    M6
sigma_ground/field/dense/quark_matter.py   M7
sigma_ground/field/dense/trace.py          M8  (GR diagnostics; one [SPECULATIVE] function, fenced)
sigma_ground/field/dense/test_*.py             one test file per module (co-located, as field/interface does)
sigma_ground/mcp/tools/dense_matter.py     M9  tier 4
sigma_ground/mcp/test_dense_matter_mcp.py
```

---

## 4. The modules

Each block lists the file and tier, the public functions, the derivation chain (Golden Rule 1), the data anchors, the tests that land with it, the Mentat tools and the effort. In the test tables, **can fail?** says whether the test could come out against the code for a physics reason (validation), or only for a bug (engine validation / [BY CONSTRUCTION]).

### M0 — `kernel/ode.py` (tier 0) · 0.5 d (inside P0)

- **Public:** `dopri5(f, t_span, y0, rtol, atol, events=(), dense=True) -> ODEResult` with FSAL, PI step control, a 4th-order continuous extension, and terminal events located by Brent on the dense output. Also `series_start(...)`, a helper for regular-singular starts (Lane–Emden, TOV centre).
- **Why here:** M2, M5 and M8 need it, and so do B6 (spherical transonic flow) and B4 (σ-wave solver). `kernel` is tier 0 math.
- **Tests:**

| test | target | can fail? |
|---|---|---|
| y' = −y | e^{−t} to 1e-10 | bug only |
| harmonic oscillator, 100 periods | energy drift ≤ rtol-scaled bound | bug only |
| Lane–Emden n = 0, 1, 5 | θ₀ = 1 − ξ²/6 (ξ₁ = √6); θ₁ = sin ξ/ξ (ξ₁ = π); θ₅ = (1 + ξ²/3)^{−1/2} (no surface) to 1e-9 | bug only |
| event location | zero of sin ξ/ξ at π to 1e-12 | bug only |

### M1 — `field/dense/fermi_gas.py` · 1.25 d (P1)

- **Public:**
  - `FermionSpecies(name, mass_mev, g=2)` with built-ins `ELECTRON`, `MUON`, `PROTON`, `NEUTRON` (masses from `constants.py`; neutron and proton from `field/nucleon.py` at `sigma`).
  - `fermi_momentum(n)` and `relativity_parameter(n, species)`, giving x = p_F/(mc).
  - `chandrasekhar_pressure(x)`, `chandrasekhar_energy(x, include_rest=True)` and `trace(x)`.
  - `eos_T0(n, species)`, returning P, ε, μ, x, c_s², `regime` ∈ {NR, transitional, UR}.
  - `electron_gas(rho, mu_e)`.
  - `eos_finite_T_nonrel(n, T, species)`, built on Fermi–Dirac integrals F₁/₂ and F₃/₂ from degenerate to classical, returning T/T_F and the regime.
- **Derivation:** Pauli exclusion → n = g p_F³/(6π²ħ³), then T = 0 phase-space integrals. In units of m⁴c⁵/(π²ħ³), with x = p_F/mc:
  - P = (1/3)∫₀ˣ t⁴/√(1+t²) dt = (1/3)[x³√(1+x²)/4 − 3x√(1+x²)/8 + 3 asinh x/8].
  - ε = ∫₀ˣ t²√(1+t²) dt.
  - ε − 3P = ∫₀ˣ t²/√(1+t²) dt = (x√(1+x²) − asinh x)/2.

  These closed forms come from Mentat's `integrate_expr` / `simplify_expr` [MEASURED] and agree with Chandrasekhar 1935 [LIT].
- **Both regimes:**
  - The exact functions span NR ↔ UR continuously. Mentat's `series_expansion` gives P ≈ x⁵/15 (NR) and `compute_limit` gives P/x⁴ → 1/12 (UR) [MEASURED].
  - Degenerate ↔ classical is covered by the NR finite-T integrals.
  - Fully relativistic finite T is a PHYSICS_GAP. It is needed only for hot white dwarfs and proto-neutron stars.
- **Anchors:** mathematical identities only, Mentat-certified. Species masses: CODATA via Mentat. m_μ = 105.6583755 MeV, m_n = 939.56542194 MeV, m_p = 938.27208943 MeV, all from scipy CODATA [MEASURED source]. P0 adds `M_MUON_MEV` to `constants.py`, because the core cannot import scipy.
- **Tests:**

| test | target | can fail? |
|---|---|---|
| NR limit | P/(x⁵/15) → 1 as x → 0 (Mentat series) | bug only |
| UR limit | P/(x⁴/12) → 1, 3P/ε → 1⁻ (Mentat limit) | bug only |
| free-gas trace positive | ε − 3P > 0 for all x > 0 (integrand t²/√(1+t²) > 0, Mentat-simplified) | [BY CONSTRUCTION] theorem; matters for M8 (§5) |
| Gibbs–Duhem | P = nμ − ε to 1e-12 | bug only |
| causality | 0 < c_s² < 1/3 for a free gas | bug only |
| classical limit | P → n k_B T when e^{μ/kT} ≪ 1 | bug only |
| Sommerfeld | low-T correction matches the analytic π²/… term | bug only |

- **MCP:** `degenerate_gas_eos(species="electron", number_density_m3=None, mass_density_kg_m3=None, mu_e=2.0, temperature_k=0.0)`.

### M2 — `field/dense/polytrope.py` + `white_dwarf.py` · 1.0 d (P2)

- **Public:**
  - `lane_emden(n)` → ξ₁, ω_n = ξ₁²|θ'(ξ₁)|, profile.
  - `polytrope_structure(n, K, rho_c)`.
  - `chandrasekhar_mass(mu_e, baryon_mass="amu")`.
  - `white_dwarf_structure(rho_c | mass, mu_e, gr=False, coulomb=True)`.
  - `mu_e_from_composition(mass_fractions)`, using Z/A from `elements.json`.
  - `white_dwarf_mass_radius(mu_e, ...)`.
- **Derivation:**
  - Newtonian hydrostatic equilibrium plus M1's electron gas (μ_e electrons per baryon) gives Chandrasekhar 1935.
  - Salpeter 1961 adds the Coulomb lattice, Thomas–Fermi and exchange corrections. This is Golden Rule 5, a volume effect.
  - GR comes via M5.
  - M_Ch = (√(3π)/2) ω₃ (ħc/G)^{3/2} / (μ_e m_B)².
- **Anchors:**
  - ω₃ = 2.01824 and ξ₁(3) = 6.89685 [LIT: Chandrasekhar 1939/1967 Table 4; Shapiro & Teukolsky 1983 eq. 3.3.12].
  - **M_Ch(μ_e = 2) = 1.4559 M☉ with m_B = m_u, 1.4333 with m_H, 1.4310 with m_n** [DERIVED from Mentat ħ, c, G, M☉, u and the [LIT] ω₃]. The 1.74 % spread is why the function names its convention.
  - Sirius B M = 1.018 ± 0.011 M☉ [LIT: Bond et al. 2017, arXiv:1703.10625]. Its radius is not in the abstract and must be read from the paper before the test is written.
  - Hamada & Salpeter 1961 [LIT] for the composition dependence.
- **Tests:**

| test | target | can fail? |
|---|---|---|
| ω₃, ξ₁(3) | 2.01824, 6.89685 to 1e-5 | bug only |
| n = 1 | ξ₁ = π exactly; M, R = analytic | bug only |
| Chandrasekhar limit | Newtonian WD with the exact EOS: M(ρ_c → ∞) → M_Ch to 0.2 % | bug only (validates M1 + M0 together) |
| low-mass slope | d ln M / d ln ρ_c → 1/2 (n = 3/2 polytrope) | bug only |
| GR instability | GR white-dwarf sequence peaks **below** M_Ch | physics: yes (if GR did nothing, it would fail) |
| Sirius B | predicted R(1.018 M☉, C/O, Salpeter) inside Bond et al.'s radius interval | **yes**: real validation |
| Coulomb correction sign | lowers R at fixed M and lowers M_max; effect grows with Z (C < O < Fe) | yes |

- **MCP:** `chandrasekhar_mass(mu_e=2.0)`, `polytrope_structure(n, central_density_kg_m3, K=None)`, `white_dwarf_structure(mass_msun=None, central_density_kg_m3=None, mu_e=2.0, general_relativity=True, coulomb_correction=True)`.

### M3 — `field/dense/semf.py` · 0.75 d (P5)

- **Public:**
  - `semf_binding_energy_mev(Z, A)`.
  - `semf_terms(Z, A)` returns the volume, surface, Coulomb, asymmetry and pairing terms separately (Golden Rule 4: every term visible).
  - `semf_coefficients()` returns the values, fit set, N, rms, source and date.
  - `nuclear_mass_kg(Z, A)`: AME2020 when measured, else SEMF; the return carries `provenance` ∈ {MEASURED, SEMF}.
  - `beta_stable_Z(A)`.
- **Derivation:**
  - a_C is DERIVED from Coulomb's law (`A_C_MEV`, unchanged).
  - a_V, a_S, a_A and a_P come from a linear least-squares fit, done in the repo and cached with provenance, to measured AME2020 binding energies (`isotopes.json`, A ≥ 16, measured entries only).
  - One convention throughout: a_A (N − Z)²/A, with pairing ± a_P A^{−1/2}.
- **Anchors:** AME2020 [LIT: Wang et al. 2021, Chinese Phys. C 45, 030003]. The full ≈ 3,400-nuclide table needs the refresh download (D3).
- **Callers updated (PLATINUM §10):**
  - `field/interface/element.py::_binding_energy_MeV` and `stable_mass_number`.
  - `mcp/tools/nuclear.py::nuclear_binding_energy`, whose private constants go to `constants.py`.
  - `inventory.core.sigma.nuclear_binding_mev` callers keep receiving a binding energy.
- **Tests:**

| test | target | can fail? |
|---|---|---|
| regression N1-a (written first, must fail on today's code) | Pb-208 and U-238 within 2 % of AME2020 (today +8.8 %, +11.6 %) | **yes** (it fails now) |
| fit quality | rms ≤ 1 % for A ≥ 16 on the fit set | yes |
| domain | A < 16 returns regime EDGE; He-4 flagged, not silently used | bug only |
| β-stability | `beta_stable_Z(56) = 26`, `beta_stable_Z(208) = 82` within ±1 | yes |
| consistency with M4 | \|a_V − \|E_sat\|\| < 1.5 MeV; a_A < J (surface symmetry lowers the finite-nucleus value) | **yes**: two independent data sets must agree |
| element.py masses | predicted atomic masses change by < 0.1 % for every Z (bounded side-effect of the fix) | yes |

- **MCP:** `nuclear_binding_energy` (unchanged signature; result gains `terms` and `provenance`), `semf_coefficients()`, `beta_stable_nucleus(mass_number)`.

### M4 — `field/dense/nuclear_matter.py` · 2.5 d with neutron-star validation (P6)

- **Public:**
  - `MetaModel(n_sat, E_sat, K_sat, J, L, Q_sat=0, K_sym=0, sigma=SIGMA_HERE)`.
  - `Skyrme(name="SLy4")`.
  - Common interface: `energy_per_baryon_mev(n_fm3, delta)`, `pressure(n, delta)`, `symmetry_energy(n)`, `saturation_properties()` (recovered numerically).
  - `beta_equilibrium(n_b, model, muons=True)` → Y_p, Y_e, Y_μ, chemical potentials.
  - `npe_mu_eos(model)` → `EOS`.
- **Derivation:**
  - The meta-model has a kinetic part from M1 (non-relativistic, isospin-split) and a potential part as a Taylor series in x = (n − n_sat)/(3n_sat) whose coefficients ARE the measured empirical parameters (Margueron et al. 2018). Measured quantities go straight into the EOS with their uncertainties, which supports an error band. It is valid to about 2 n_sat.
  - Skyrme SLy4 is a FITTED energy functional (Chabanat et al. 1998). It extends to the maximum mass and is the same interaction behind the Douchin & Haensel unified EOS. It serves as the external code-correctness anchor.
  - β-equilibrium: μ_n = μ_p + μ_e, μ_μ = μ_e, n_p = n_e + n_μ.
- **Anchors** (P0 cites them in `constants.py`; values in §8):
  - n_sat, E_sat, K_sat, J, L [LIT: Margueron 2018 table; Lattimer 2023 (J = 32.0 ± 1.1, L = 51.9 ± 7.9 MeV); Garg & Colò 2018 for K].
  - P(2 n_sat) ∈ [1.8, 6.2] × 10³³ Pa (90 %) [LIT: Abbott et al. 2018; unit conversion by Mentat `convert_units`].
  - SLy unified EOS M_max = 2.05 M☉ [LIT: Douchin & Haensel 2001].
- **Tests:**

| test | target | can fail? |
|---|---|---|
| saturation recovery (meta-model) | P(n_sat, δ = 0) = 0 to 1e-10; E/A, K, J, L recovered to 0.1 % | bug only |
| SLy4 transcription | its saturation point matches Chabanat's published n_sat, E/A, K, J | **yes** (catches parameter typos) |
| β-equilibrium | charge neutrality to 1e-12; muon onset where μ_e = m_μc² | bug only |
| GW170817 | P_β(2 n_sat) inside [1.8, 6.2] × 10³³ Pa | **yes** |
| maximum mass (with M5) | SLy4 core + crust: M_max = 2.05 M☉ ± 2 % (DH2001) | **yes** |
| ≥ 2 M☉ pulsar | M_max ≥ 2.08 − 2 × 0.07 = 1.94 M☉ (PSR J0740+6620, Fonseca 2021). The free-neutron EOS must **fail** this (shows the test has teeth) | **yes** |
| NICER radii | R(1.34 M☉) inside 12.71 (+1.14/−1.19) km (J0030, Riley 2019) and inside Miller 2019's 13.02 (+1.24/−1.06) km at 1.44 M☉; R(2.072 M☉) inside 12.39 (+1.30/−0.98) km (J0740, Riley 2021) | **yes** |
| causality | c_s² ≤ 1 up to the M_max central density, else WALL | **yes** |
| trace-flip compactness (with M8) | the C = GM/Rc² at which 3P_c = ε_c lies in [0.245, 0.273] for every EOS that reaches it (Podkowka, Mendes & Poisson 2018) | **yes** |

- **MCP:** `nuclear_matter_properties(model="metamodel", density_fm3=None, asymmetry=0.0)`, `beta_equilibrium(baryon_density_fm3, model="SLy4")`.

### M5 — `field/dense/eos.py` + `tov.py` · 1.5 d (P3)

- **Public:**
  - `solve_tov(eos, central_pressure_pa=None, central_density_kg_m3=None, rtol=1e-10)` → `TOVStar`. Fields: R, M, M_baryon, compactness, binding_energy, tolman_mass, trace_mass, and profiles r, P, ε, m, ν, λ, n_b.
  - `mass_radius_curve(eos, n_points=30)`.
  - `maximum_mass(eos)`: golden-section search on log P_c.
  - `is_stable(eos, P_c)`: dM/dP_c > 0.
  - `newtonian_structure(eos, ...)`: the same EOS without GR, so weak-field claims can be checked against the full answer.
- **Derivation:** Einstein's equations for a static perfect fluid (Tolman 1939; Oppenheimer & Volkoff 1939):
  - dP/dr = −G(ε + P)(m + 4πr³P/c²)/[c² r²(1 − 2Gm/rc²)]
  - dm/dr = 4πr²ε/c² (ε energy density in J/m³, m in kg)
  - dν/dr = −2(dP/dr)/(ε + P)

  The centre starts from a series expansion; the surface is a terminal event at the EOS's minimum pressure. ν is normalised so e^{ν(R)} = 1 − r_s/R. The Tolman (1930) mass is ∫(ε + 3P) e^{(ν+λ)/2} dV/c². The enthalpy formulation (Lindblom 1992) is an option if pure-Python step counts are too high.
- **Anchors:** exact interior solutions; OV 1939's free-neutron maximum of about 0.7 M☉ [LIT: their abstract says no static equilibrium above 3/4 M☉; ≈ 0.7 per Chamel et al. 2013, arXiv:1307.3995].
- **Tests:**

| test | target | can fail? |
|---|---|---|
| Schwarzschild interior (uniform ε), C = 0.5 | R, M and P(r) exact to 1e-9 (benchmark from the scratch harness) | bug only |
| Buchdahl | 2GM/Rc² < 8/9 for every star produced | bug only |
| Newtonian limit | n = 1 polytrope at w_c = 1e-5: M/M_LE − 1 and R/πa − 1 < 1e-4 | bug only |
| Tolman identity | tolman_mass = M to 1e-8 | [BY CONSTRUCTION] if the metric normalisation is right; catches a wrong ν |
| first law along the profile | dε/dn = (ε + P)/n on the tabulated EOS to 1e-6 | bug only |
| free neutron gas | M_max ∈ [0.68, 0.75] M☉ | **yes** (against OV 1939) |
| baryon mass | M_baryon > M for bound stars | bug only |
| performance | one star < 0.5 s; a 30-point curve < 30 s, pure Python (Mentat's call limit is 90 s) | yes |

- **MCP:** `tov_star(eos="SLy4+BPS", mass_msun=None, central_density_kg_m3=None)`, `mass_radius_curve(eos, n_points=30)`, `maximum_mass(eos)`.

### M6 — `field/dense/composition.py` · 2.0 d (P7)

- **Public:**
  - `outer_crust(pressure_pa)` → equilibrium (Z, A), n_b, ε, μ_e.
  - `neutron_drip()` → P and ρ at drip.
  - `crust_core_transition(model)`.
  - `unified_eos(core_model)` → `EOS` (outer crust + inner-crust join + core).
  - `layers(star)` → list of {name, r_inner, r_outer, ρ range, composition, mass fraction} from a `TOVStar`.
  - `white_dwarf_layers(core="C/O", envelope=("He", "H"), masses)` → μ_e per layer for M2.
- **Derivation:**
  - BPS 1971: at each pressure, minimise the Gibbs energy per nucleon over (Z, A). The energy counts nuclear mass (M3: AME where measured), the electron gas (M1) and the bcc lattice Coulomb energy (Salpeter 1961).
  - Neutron drip is where the neutron chemical potential reaches m_n c².
  - The crust–core edge comes from the thermodynamic instability of β-equilibrated uniform matter (M4).
  - The inner crust (nuclei in a neutron gas) is a PHYSICS_GAP in round 1: a smooth join between drip and the crust–core edge, flagged. A compressible liquid drop comes later.
- **Anchors:**
  - BPS: ⁵⁶Fe at the lowest pressures, more neutron-rich nuclei with depth, valid to drip at ≈ 4.3 × 10¹¹ g/cm³ = 4.3 × 10¹⁴ kg/m³ [LIT, secondary-verified].
  - DH2001 crust–core edge ≈ 0.08 fm⁻³ for SLy [LIT, secondary].
- **Tests:**

| test | target | can fail? |
|---|---|---|
| first nucleus | ⁵⁶Fe at P → 0 (not ⁶²Ni: the lowest mass per nucleon including electrons) | **yes** |
| monotone neutronisation | Z/A non-increasing with P | yes |
| drip | ρ_drip within 10 % of 4.3 × 10¹⁴ kg/m³ with AME masses (and documented SEMF-only drift) | **yes** |
| crust–core edge (SLy4) | 0.08 fm⁻³ ± 15 % | yes |
| layers | radii nest, mass fractions sum to 1, crust mass fraction < 5 % at 1.4 M☉ | bug only |
| planets.NEUTRON_STAR | the `neutron_star` key now built from `layers(solve_tov(unified_eos('SLy4'), mass=1.4))`; same keys as before, layers cited | bug only (PLATINUM §10) |

- **MCP:** `neutron_star_layers(mass_msun=1.4, eos="SLy4+BPS")`, `outer_crust_composition(mass_density_kg_m3)`.

### M7 — `field/dense/quark_matter.py` · 1.25 d (P8)

- **Public:**
  - `bag_eos(B_mev_fm3, m_s_mev=0.0, alpha_s=0.0)` → `EOS`.
  - `strange_matter_energy_per_baryon(B, m_s)`, evaluated at P = 0.
  - `deconfinement_transition(hadronic_eos, quark_eos)` → P_t, μ_B,t, Δε. This is a Maxwell construction: equal P and μ_B.
  - `hybrid_eos(hadronic, quark)`.
  - `strange_star(B)` (via M5).
- **Derivation:** free u, d, s quark Fermi gases (M1 with colour × spin degeneracy 6 per flavour) plus the vacuum energy B (MIT bag; Chodos et al. 1974). β-equilibrium and charge neutrality apply when m_s > 0. Strange-matter stability is judged by E/A at P = 0 against the energy per nucleon of ⁵⁶Fe (from M3/AME). The strange-quark mass goes into `constants.py` with a PDG citation in P0.
- **Anchors:**
  - Farhi & Jaffe 1984 window B^{1/4} = 145–164 MeV for massless quarks with α_s = 0 [LIT, secondary-verified].
  - Witten 1984 (absolute stability hypothesis) [LIT].
- **Tests:**

| test | target | can fail? |
|---|---|---|
| bag trace | ε − 3P = 4B exactly (massless) | [BY CONSTRUCTION]; it is the σ source in a quark core (§5) |
| window edges | B^{1/4} = 145 MeV ⇒ E/A below the ⁵⁶Fe value (stable); 165 MeV ⇒ above | **yes** (degeneracy and unit errors show) |
| homology | massless bag stars: M_max ∝ B^{−1/2}, R ∝ B^{−1/2} to 1e-6 | bug only |
| Maxwell construction | P and μ_B continuous, ε jumps by Δε > 0 | bug only |
| hybrid branch | stability reported (dM/dP_c); a twin branch is flagged, not hidden | yes |

- **MCP:** `bag_model_eos(bag_constant_mev_fm3, strange_mass_mev=0.0, alpha_s=0.0)`, `deconfinement_transition(hadronic_eos="SLy4", bag_constant_mev_fm3=...)`.

### M8 — `field/dense/trace.py` · 1.0 d (P4; real-EOS rows in P6)

- **Public, pure GR (no σ):**
  - `trace_profile(star)`, giving:
    - w(r) = 3P/ε;
    - the normalised trace anomaly Δ(r) = (ε − 3P)/(3ε) (Fujimoto et al. 2022);
    - the sign-flip radius r₀ where P = ε/3, if any.
  - `trace_mass(star)`: M_tr = ∫(ε − 3P) e^{(ν+λ)/2} dV/c² and M_tr/M.
  - `pressure_virial_fraction(star)`: 3∫P e^{(ν+λ)/2} dV/(Mc²), the GR form of P2b's pressure term.
  - `weak_field_comparison(star)`: P7's flat-space ratio 1 − 3∫P dV/(c²∫ρ dV) side by side with the exact M_tr/M.
- **Public, [SPECULATIVE] and fenced:** `sigma_static_profile(star, xi, normalisation, sign, range_m=math.inf)`. It integrates the static σ equation on the TOV metric:

      d/dr[e^{(ν−λ)/2} r² σ'] − e^{(ν+λ)/2} r² σ/λ_σ² = ∓(8πGξ/c⁴)(ε − 3P) e^{(ν+λ)/2} r²

  The exterior is matched to Schwarzschild (massless) or to a decaying solution (Yukawa range λ_σ). The function returns σ(r), σ_c and the exterior monopole. ξ, normalisation (C2), sign (B5/C4) and range (C5) are required, with no defaults.
- **Derivation:** the trace and Tolman mass are GR (Tolman 1930). The σ equation is the program's §3.1 postulate, labelled.
- **Anchors:**
  - Podkowka, Mendes & Poisson 2018: the central trace flips at C = 0.262 (+0.011/−0.017), roughly independent of the EOS [LIT].
  - Fujimoto et al. 2022: observation-inferred EOSs give Δ near 0 by about 5 n_sat [LIT].
  - Mendes 2015 and Damour & Esposito-Farèse 1993: strong-field scalar–tensor behaviour of neutron stars [LIT].
- **Tests:**

| test | target | can fail? |
|---|---|---|
| Tolman = M | 1e-8 on every star | [BY CONSTRUCTION] |
| small-w_c slope | n = 1 polytrope, w_c = 1e-3 … 1e-2: (1 − M_tr/M)/w_c → 1.00 ± 0.01. The scratch harness recorded 0.99900 at w_c = 1e-3, against P7's flat 1 − w_c/2 = 0.9995 | yes: it decides the factor P7 got |
| `weak_field_comparison` reproduces P7 | flat ratio = 1 − w_c/2 exactly for n = 1 | bug only |
| free Fermi-gas stars | Δ > 0 everywhere (M1 theorem) | [BY CONSTRUCTION] |
| bag stars | Δ = 4B/(3ε) pointwise | [BY CONSTRUCTION] |
| trace-flip compactness | shared with M4: C at 3P_c = ε_c inside [0.245, 0.273] | **yes** |
| Yukawa limits | λ_σ → ∞ recovers massless; λ_σ ≪ R recovers the local §6 formula σ ∝ λ_σ² ρ inside | bug only |

- **MCP:** `stellar_trace_profile(eos, mass_msun)` (GR, no tag), `sigma_bridge_profile(eos, mass_msun, xi, normalisation, sign, range_m)`, whose description opens with "[SPECULATIVE] — rests on the unadopted □σ = −ξR postulate".

### M9 — Mentat integration · 1.0 d (P9; each module's tool already landed with it)

- **Code:** `mcp/tools/dense_matter.py` (the 17 new tools listed under M1–M8; `nuclear_binding_energy` keeps its name and signature).
  - Registration in `mcp/server.py` next to `t_nuc`.
  - `manifest.py` entries in a domain `dense_matter`.
  - `tool_keywords.py` phrasings ("how heavy can a white dwarf get", "neutron star radius", "what is inside a neutron star", "Chandrasekhar limit").
  - `mcp/test_dense_matter_mcp.py` asserts each tool returns `source` and `formula`.
  - `test_tool_keywords.py` gets the routing cases.
- **Docs:**
  - `ARCHITECTURE.md` package-map row.
  - `KNOWN_GAPS.md` lines (§9).
  - Re-point `field/audit.py`'s TOV entries.
  - Strike B1 in `MENTAT_GAPS_UNIFICATION.md`.
  - `misc/SESSION_LOG.md` per session.
- **Deploy (CLAUDE.md "deploy and report"):**
  1. Commit and push sigma-ground (solo-repo rule §17, after checking contributors).
  2. Bump `SIGMA_GROUND_REF` in `SkippyTel/mentat/Dockerfile:9` and `SkippyTel/.github/workflows/deploy.yml:40`.
  3. Rebuild: `cd /d/Aaron/development/SkippyTel && docker compose stop mentat-home && docker compose rm -f mentat-home && docker compose up --build -d mentat-home`.
  4. Verify: `bash tools/mentat_call.sh list | grep -E "tov_star|chandrasekhar_mass"` and `bash tools/mentat_call.sh chandrasekhar_mass '{"mu_e": 2}'` returns 1.456 M☉ (m_u convention).
  5. Optional: add a "Compact stars" group to `domain_of()` in `SkippyTel/mentat/app.py`.

---

## 5. How this unlocks the σ-bridge pressure term and P7 beyond the weak field

**Today.** P6/P7 took dense-matter rows from the literature. The 3P/ρc² magnitudes in §0 come from P6's own [LIT] rows. P7 solved the weak-field σ equation on a Lane–Emden n = 1 polytrope with a flat-space source ρ − 3P/c². By its own verification note, this drops a (1 + 2Φ/c²) metric factor of the same post-Newtonian order as the term it keeps (statement j). Its w_c = 0.5 row was outside validity.

**After MS2 (P4: M0 + M1 + M5 + M8, polytropes and free gases).**
- The static σ equation is solved on the exact TOV metric. The metric factors are automatic, so the (1 + 2Φ/c²) omission disappears rather than being estimated.
- The parallel-session scratch harness (untracked `tov_sigma_pair_results.json`, not rerun here) recorded M_σ/M_Tolman for an n = 1 polytrope: 0.99900, 0.99006, 0.9054, 0.7439 and 0.6120 at w_c = 0.001, 0.01, 0.1, 0.3 and 0.5. The 1PN prediction 1 − w_c gives 0.999, 0.99, 0.9, 0.7, 0.5; P7's flat ratio 1 − w_c/2 gives 0.9995, 0.995, 0.95, 0.85, 0.75.
- Two readings follow, both to be pinned as M8 tests:
  - At small w_c the observable σ-charge per unit *gravitational* mass is 1 − w_c, twice P7's pressure deficit. The gravitational mass is itself sourced by ε + 3P (Tolman), so pressure enters twice.
  - By w_c ≈ 0.3 neither expansion holds.
- A free Fermi gas can never flip the σ source sign: ε − 3P > 0 is a theorem for any free gas (M1), approached only as 3P/ε → 1. **A sign flip in a real star is therefore purely an interaction effect**, which is why M4 is needed.

**After MS3 (P6: realistic neutron stars).**
- 3P/ε, Δ and M_tr/M become computed functions along each EOS's M–R sequence, replacing borrowed numbers.
- The trace-flip compactness is tested against Podkowka–Mendes–Poisson.
- **The observed pulsars sit right at the edge.** With Mentat's Schwarzschild radius and the NICER (M, R) [MEASURED from [LIT] inputs]:
  - PSR J0030+0451 has C = 0.156 (Riley) or 0.163 (Miller).
  - PSR J0740+6620 has C = 0.247, at the lower edge of the flip interval 0.245–0.273.
- Whether J0740's central σ source is positive, zero or negative is therefore EOS-dependent. M8 answers it per EOS: the one place in nature where the σ-bridge's pressure term is not a small correction but decides the sign.

**The surviving short-range corner [SPECULATIVE, DERIVED under §6's stated reading].** §6 of the program leaves σ alive only at range λ_σ ≲ 35 km (for ξ = 0.13–0.17). Using §6's local formula σ = 8πGξλ_σ²ρ/c² with Mentat G, c, m_n and the repo's ξ = 0.1582, at nuclear saturation density (ρ = 2.68 × 10¹⁷ kg/m³), σ comes out at:

| λ_σ | σ |
|---|---|
| 0.3 km | 7.1 × 10⁻⁵ |
| 1 km | 7.9 × 10⁻⁴ |
| 3 km | 7.1 × 10⁻³ |

Higher values follow in the core. The local formula holds only for λ_σ ≪ R, so km ranges need M8's Yukawa solve on the TOV profile. A neutron star is the only object where the allowed corner is not invisible. A σ-modified TOV (masses ∝ e^{fσ} inside the star) would make NICER radii and the ≥ 1.94 M☉ floor the first data that could constrain λ_σ between ~1 and 35 km. That follow-on is out of B1's scope but becomes possible once M5 and M8 exist; it stays [SPECULATIVE] until decisions C1–C5 and an action (next-experiment 2) exist.

**What this cannot do.** None of it tests "gravity is a fluid". The TOV, Tolman mass and trace are general relativity; passing NICER or 2 M☉ validates the EOS, not the thesis. Every σ number stays conditional on the [SPECULATIVE] postulate, and the long-range σ is already excluded (§6, ×3 × 10⁵ to ×10¹¹).

---

## 6. Order of work, milestones and effort

| phase | content | depends on | days | ends at |
|---|---|---|---|---|
| P0 | M0 `kernel/ode.py`; cite saturation constants (§8), add `L_SYM_MEV`, `M_MUON_MEV`, `M_STRANGE_MEV`; fix the GOLDEN_RULES bounds pointer (N1-f) | — | 0.5 | 0.5 |
| P1 | M1 Fermi gas (T = 0 exact; NR finite T) | P0 | 1.25 | 1.75 |
| P2 | M2 Lane–Emden + white dwarfs, Chandrasekhar mass, Salpeter | P1 | 1.0 | 2.75 |
| P3 | M5 EOS protocol + TOV; GR white dwarfs; free-neutron star | P1, P2 | 1.5 | **4.25 (MS1)** |
| P4 | M8 trace diagnostics; P7 rerun on the TOV metric (polytropes, free gases) | P3 | 1.0 | **5.25 (MS2)** |
| P5 | M3 one SEMF fitted to AME2020; element.py + nuclear.py re-pointed; N1-a regression | P0 | 0.75 | 6.0 |
| P6 | M4 meta-model + SLy4 + β-equilibrium; NS sequences; NICER, J0740, GW170817, PMP tests | P1, P3, P4, P5 | 2.5 | **8.5 (MS3)** |
| P7 | M6 outer crust, drip, crust–core edge, layers; `planets.NEUTRON_STAR` derived; DH2001 2.05 M☉ test | P5, P6 | 2.0 | 10.5 |
| P8 | M7 bag EOS, deconfinement, hybrid and strange stars | P1, P6 | 1.25 | **11.75 (MS4)** |
| P9 | MCP sweep, docs, KNOWN_GAPS, push, ref bump, rebuild Mentat, verify | all | 1.0 | **12.75 (MS5)** |

- P5 has no physics dependency on P1–P4 and can move earlier if the Captain wants the N1-a fix first.
- Each phase is one commit series with tests in the same change (PLATINUM §14). The MCP tool for a module lands with the module; P9 only integrates.
- Effort is rough, for one agent with tests. The two largest risks to the estimate are SLy4 transcription plus the 2.05 M☉ reproduction, and the inner-crust join.

---

## 7. Decisions for the Captain (before P0)

| # | decision | recommendation |
|---|---|---|
| D1 | package home: `field/dense/` (or `field/compact/`) | `field/dense/` |
| D2 | ODE: pure-Python DOPRI5 in `kernel/ode.py` (core stays zero-dependency) vs allow scipy in the core | pure Python |
| D3 | SEMF fit set: the 145 nuclides in `isotopes.json` now; the full AME2020 table (download from IAEA AMDC, run `refresh_isotopes.py`) needs your OK | 145 now, full table when approved |
| D4 | reference functional SLy4 (code-correctness anchor via DH2001) + meta-model on measured parameters (error bands); unmeasured Q_sat, K_sym default 0 under PHYSICS_GAP with a sweep | yes |
| D5 | σ hook: EOS at `SIGMA_HERE` by default; the σ-bridge profile takes ξ, normalisation, sign and range as required arguments, with no import of ξ, until C1–C5 are decided | yes |
| D6 | land the N1-a fix (element.py asymmetry convention) inside M3 behind a failing-first test; it shifts predicted atomic masses by < 0.1 % | yes, as with S8 |
| D7 | `planets.NEUTRON_STAR` becomes TOV-derived (key kept) | yes |

---

## 8. Data anchors (all verified by web search this session; details and URLs in `n1_references.json`)

| quantity | value | reference | used by |
|---|---|---|---|
| Chandrasekhar mass, μ_e = 2 | 1.4559 M☉ (m_u) / 1.4333 (m_H) / 1.4310 (m_n) [DERIVED]; "1.44 M☉" in the modern statement | Chandrasekhar 1931, ApJ 74, 81; 1935, MNRAS 95, 207; ω₃ = 2.01824 (Chandrasekhar 1939/1967; Shapiro & Teukolsky 1983) | M2 |
| PSR J0030+0451 | M = 1.34 (+0.15/−0.16) M☉, R = 12.71 (+1.14/−1.19) km | Riley et al. 2019, ApJL 887, L21 (arXiv:1912.05702) | M4/M5 |
| PSR J0030+0451 | M = 1.44 (+0.15/−0.14) M☉, R = 13.02 (+1.24/−1.06) km | Miller et al. 2019, ApJL 887, L24 (arXiv:1912.05705) | M4/M5 |
| PSR J0740+6620 radius | R = 12.39 (+1.30/−0.98) km at M = 2.072 (+0.067/−0.066) M☉ | Riley et al. 2021, ApJL 918, L27 (arXiv:2105.06980) | M4/M5 |
| PSR J0740+6620 mass | 2.08 ± 0.07 M☉ (68.3 %) | Fonseca et al. 2021, ApJL 915, L12 (arXiv:2104.00880) | M4 (≥ 1.94 M☉ floor) |
| PSR J0348+0432 | 2.01 ± 0.04 M☉ (2013) **revised to 1.806 ± 0.037 M☉** | Antoniadis et al. 2013, Science 340, 1233232 → Saffer et al. 2025, ApJL 983, L20 (arXiv:2412.02850) | not a 2 M☉ anchor any more |
| nuclear saturation (empirical parameters, 50-functional average) | n_sat 0.1543 ± 0.0054 fm⁻³, E_sat −16.03 ± 0.20 MeV, K_sat 251 ± 29 MeV, E_sym 33.30 ± 2.65 MeV, L_sym 76.6 ± 29.2 MeV | Margueron, Hoffmann Casali & Gulminelli 2018, PRC 97, 025805 (ar5iv text partly garbled: re-read the table at implementation) | M4, `constants.py` |
| symmetry energy | J = 32.0 ± 1.1 MeV, L = 51.9 ± 7.9 MeV (neutron-matter theory, 68 %); PREX+CREX jointly L = 53 ± 13 MeV | Lattimer 2023, Particles 6, 30 | M4 |
| incompressibility | review of K∞ from giant monopole resonances (value not in the abstract; ~240 MeV per a Colò abstract; Sn suggests lower) | Garg & Colò 2018, PPNP 101, 55 (arXiv:1801.03672) | M4 |
| pressure at 2 n_sat | 3.5 (+2.7/−1.7) × 10³⁴ dyn/cm² = [1.8, 6.2] × 10³³ Pa (90 %; Mentat `convert_units`) | Abbott et al. 2018, PRL 121, 161101 | M4 |
| SLy unified EOS | M_max = 2.05 M☉, M_min = 0.094 M☉ | Douchin & Haensel 2001, A&A 380, 151 | M4/M5/M6 |
| SLy4 parameters | (copied from the paper's table at implementation) | Chabanat et al. 1998, NPA 635, 231 | M4 |
| nuclide masses | AME2020 | Wang et al. 2021, Chinese Phys. C 45, 030003 | M3/M6 |
| outer crust | ⁵⁶Fe bcc lattice → neutron-rich nuclei; drip ≈ 4.3 × 10¹¹ g/cm³ | Baym, Pethick & Sutherland 1971, ApJ 170, 299 | M6 |
| TOV | — | Tolman 1939, Phys. Rev. 55, 364; Oppenheimer & Volkoff 1939, Phys. Rev. 55, 374 (free neutrons: no equilibrium above 3/4 M☉; ≈ 0.7 M☉) | M5 |
| Tolman mass | ∫(ε + 3P)√g₀₀ dV | Tolman 1930, Phys. Rev. 35, 875 | M5/M8 |
| trace sign flip | C = 0.262 (+0.011/−0.017), 90 %, ~EOS-independent | Podkowka, Mendes & Poisson 2018, PRD 98, 064057 | M8 |
| trace anomaly | Δ = (ε − 3P)/(3ε) → ~0 by ~5 n_sat | Fujimoto, Fukushima, McLerran & Praszałowicz 2022, PRL 129, 252702 | M8 |
| scalar–tensor strong field | spontaneous scalarization; positive-trace instability | Damour & Esposito-Farèse 1993, PRL 70, 2220; Mendes 2015, PRD 91, 064024 | §5 |
| WD Coulomb corrections | — | Salpeter 1961, ApJ 134, 669; Hamada & Salpeter 1961, ApJ 134, 683 | M2/M6 |
| Sirius B | 1.018 ± 0.011 M☉ | Bond et al. 2017, ApJ (arXiv:1703.10625) | M2 |
| bag model | B = vacuum energy difference; stability window B^{1/4} = 145–164 MeV (massless, α_s = 0) | Chodos et al. 1974, PRD 9, 3471; Witten 1984, PRD 30, 272; Farhi & Jaffe 1984, PRD 30, 2379 | M7 |
| piecewise polytrope (optional EOS family) | 3 parameters fit candidate EOSs to ~4 % rms | Read et al. 2009, PRD 79, 124032 | M5 registry |

---

## 9. PHYSICS_GAP lines this work will add to KNOWN_GAPS.md

- Fully relativistic finite-T Fermi gas (hot WDs, proto-NS): T = 0 plus NR finite-T only.
- Inner crust as a smooth join between drip and the crust–core edge (no compressible liquid drop, no pasta).
- No hyperons, no superfluidity or superconductivity, no magnetic fields, no rotation (Hartle slow rotation and the tidal Love number are optional extensions).
- Meta-model higher-order parameters Q_sat, K_sym default 0 (swept, not measured).
- Bag model without colour superconductivity; B is a declared model parameter.
- White-dwarf crystallisation and cooling not modelled (zero-T sequences only).

---

## 10. Risks

| risk | effect | mitigation |
|---|---|---|
| pure-Python speed | `mass_radius_curve` exceeds Mentat's 90 s call limit | tabulate each EOS once and cache; enthalpy form; cap n_points; performance test in M5 |
| SLy4 transcription error | wrong M_max | two independent checks: saturation recovery vs Chabanat, M_max vs DH2001 |
| inner-crust scope creep | P7 overruns | PHYSICS_GAP join first; compressible liquid drop as a later item |
| tests that pass by construction mistaken for evidence | overclaiming | the "can fail?" column. Only NICER, J0740, GW170817, PMP, BPS drip, Sirius B, DH2001 and the N1-a regression are validation |
| σ decisions still open | ambiguous σ numbers | required arguments, [SPECULATIVE] tag on the one tool, the engine never imports ξ |
| literature drift (as with J0348) | stale anchors | anchors carry citation and date in `constants.py`; the references file records how each was verified |

---

## 11. Corrections to the gap census and the program (design_corrections)

1. **J0348+0432 is no longer a ≥ 2 M☉ pulsar.** Saffer et al. 2025 measure 1.806 ± 0.037 M☉ [LIT]. B1's "≥ 2 M☉ pulsars" anchor rests on J0740+6620 alone (2.08 ± 0.07, Fonseca 2021). The program's §6 table also cites J0348 for dipole radiation; that use is unaffected, because the orbital-decay rate is reported consistent within 1.2σ.
2. **"Chandrasekhar mass ≈ 1.4 M☉" needs a convention.** For μ_e = 2 it is 1.456 M☉ with m_u, 1.433 with m_H and 1.431 with m_n [DERIVED; 1.74 % spread]. The test target names the convention.
3. **"Both regimes" is two axes.** NR ↔ UR at T = 0 (exact) and degenerate ↔ classical at finite T (NR integrals). Fully relativistic finite T is a declared gap.
4. **P7's pressure deficit, as an observable.** Normalised to the gravitational (Tolman) mass, the σ-charge deficit is ≈ w_c at small w_c, not w_c/2 (parallel-session scratch, not rerun; M8 pins it).
5. **The repo's K_SAT_MEV = 230 vs the 2018 meta-model average 251 ± 29.** Consistent within 1σ; the citation and uncertainty go in.
6. **Golden Rule 2's pointer to `field/bounds.py`** is stale since e66615b.
