# Nuclear and degenerate matter in Mentat: implementation plan (gap B1), v2

**Item:** N1 of the round-2 research run · **Date:** 2026-10-10 · **Status:** PLAN ONLY, as agreed with the Captain. No code changes to the repo.
**Version:** v2, repaired after three independent skeptic reviews (reproduce; physics and literature; could-it-fail). The v1 files are kept unchanged next to this one. Every repair is listed in §12.
**Gap closed:** B1 in `D:/Aaron/development/sigma-ground/misc/MENTAT_GAPS_UNIFICATION.md`. B1 covers electron and neutron Fermi-gas EOS in both regimes, a semi-empirical mass formula, saturation and symmetry energy, a TOV integrator, composition layers and the bag-model deconfinement pressure.
**Rules in force:** Golden Rules 1–7; PLATINUM §3 (one source of truth), §10 (if one, then all), §14 (tests land with the code), §21 (Mentat does the physics); "Mentat flags, never fixes".
**Companion files (this folder):**
- `n1_survey_v2.py` → `n1_survey_v2.json`. The survey, read-only on the repo. Every number below carries its Mentat `source` there.
- `n1_references_v2_build.py` → `n1_references_v2.json`. 35 references, each with how it was verified.
- `n1_figure_v2.py` → `n1_plan_figure_v2.png`.

**Tags:**
- [MEASURED]: computed this session by Mentat or a repo engine.
- [DERIVED]: follows from stated inputs by algebra, with Mentat constants and Mentat symbolic tools.
- [VERIFIED numerics]: scratch numerics in this folder. This means scipy, mainly the TOV on a [LIT] EOS fit. Mentat has no TOV, which is the gap, so this is not a Mentat result.
- [LIT]: literature. The citation was verified this session by fetching the arXiv/ar5iv record or querying Crossref.
- [BY CONSTRUCTION]: engine validation, not physics.
- [SPECULATIVE]: rests on the σ-bridge postulate.

---

## 0. The plan on one page

**Why.** The σ-bridge sources σ by the trace ε − 3P. Pressure is an order-one share of that source only in neutron-star cores:

| object | 3P/ε | basis |
|---|---|---|
| Sun core | 5.6 × 10⁻⁶ | P6's [LIT] row (Bahcall SSM), Mentat c |
| white-dwarf centres | 2.2 × 10⁻⁵ to 3.5 × 10⁻³ (ρ_c = 10⁸–10¹³ kg/m³) | Chandrasekhar EOS, μ_e = 2, Mentat ħ, c, m_e, u, Mentat closed form [DERIVED] |
| matter at 2 ρ_nuc | **0.21 (90 %: 0.11–0.37)** | GW170817 P(2 ρ_nuc) [LIT, Abbott 2018], Mentat `convert_units`, c [DERIVED] |
| SLy neutron-star centres | 0.28 (1.0 M☉), **0.46 (1.4 M☉)**, 0.78 (1.8 M☉), 1.18 (2.0 M☉), 1.62 (M_max) | scratch TOV on the Haensel–Potekhin 2004 SLy fit [VERIFIED numerics on LIT] |

- v1 quoted **0.67 "at a neutron-star centre"**. That number is withdrawn. It is [DERIVED from LIT] P6's row (5 × 10¹⁷ kg/m³, 10³⁴ Pa, "~2 n₀, very stiff, order of magnitude"). That row lies below 2 n_sat, yet its pressure is **1.6× above GW170817's 90 % upper bound at 2 n_sat**, so the plan's own anchor excludes it.
- P6's white-dwarf row (10⁹ kg/m³, 10²² Pa) is likewise **3.8× the Chandrasekhar pressure** at that density (2.63 × 10²¹ Pa). Both rows are flagged (§1.1, N1-h).
- Order one is reached only toward the central densities of massive stars. Those are what M5 and M8 will compute.

Mentat cannot build either kind of degenerate star today:
- `sigma_ground/` has no Fermi-gas EOS, nuclear-matter EOS or TOV integrator.
- Mentat has no **numeric** ODE tool. `solve_ode` is symbolic (`sympy.dsolve`). With `Derivative()` syntax it solves Lane–Emden n = 1 in closed form [MEASURED]. Its documented prime syntax is refused by the SkippyTel front door (§1.1, N1-c).
- The repo does hold fixed-step integrators (`sg_solve_ivp`: RK4, Forest–Ruth, Verlet) and an SPH pressure layer (`dynamics/fluid/eos.py`). Neither offers adaptive steps, terminal events or a barotropic EOS object, and the plan must reconcile with both (D2).
- P6 and P7 therefore had to borrow their dense-matter rows.

**What.** The plan has three parts:
- one adaptive ODE integrator, `sigma_ground/kernel/ode.py` (tier 0), which `sg_solve_ivp` delegates to;
- eight modules in a new tier-1 package `sigma_ground/field/dense/` (the recommended home, **pending D1**);
- an integration phase.

| # | module | delivers |
|---|---|---|
| M0 | `kernel/ode.py` (tier 0) | Dormand–Prince 5(4), dense output, terminal events; the one adaptive integrator (D2) |
| M1 | `field/dense/fermi_gas.py` | exact T = 0 Chandrasekhar functions for e, μ, n, p (NR ↔ UR); NR finite-T Fermi–Dirac (degenerate ↔ classical) |
| M2 | `field/dense/polytrope.py`, `white_dwarf.py` | Lane–Emden, Chandrasekhar mass, white dwarfs with Salpeter Coulomb corrections |
| M3 | `field/dense/semf.py` | ONE semi-empirical mass formula, fitted to AME2020, replacing the two divergent copies |
| M4 | `field/dense/nuclear_matter.py` | meta-model on Margueron 2018's experiment-based parameters, Skyrme SLy4, β-equilibrium npeμ |
| M5 | `field/dense/eos.py`, `tov.py` | EOS protocol; TOV with Tolman mass, trace mass, baryon mass, M–R curves, M_max |
| M6 | `field/dense/composition.py` | outer crust (BPS), neutron drip, crust–core edge, layers from a TOV profile |
| M7 | `field/dense/quark_matter.py` | MIT bag EOS, Maxwell deconfinement against M4, hybrid and strange stars |
| M8 | `field/dense/trace.py` | GR trace diagnostics (3P/ε, Δ, sign-flip radius, M_tr/M) and one opt-in [SPECULATIVE] σ profile on the TOV metric |
| M9 | `mcp/tools/dense_matter.py` | integration: 17 new Mentat tools, `nuclear_binding_energy` re-pointed, manifest, keywords, deploy |

**Order and effort.** P0 foundations → P1 Fermi gas → P2 white dwarfs → P3 TOV → P4 trace diagnostics (P7 beyond the weak field) → P5 SEMF → P6 nuclear matter and neutron stars (provisional crust) → P7 layers and the unified EOS (NICER scored here) → P8 quark cores → P9 MCP sweep and deploy.
- This is about **13.0 working days**: v1's 12.75 plus 0.25 d in P0 for the D2 reconciliation.
- Tests land with every module.

**Milestones:**
- **MS1** (≈ 4.5 d): Chandrasekhar mass and the free-neutron star computed. The free-neutron star should come out at M_max ≈ 0.710 M☉ [VERIFIED numerics].
- **MS2** (≈ 5.5 d): P7 rerun on the exact static metric.
- **MS3** (≈ 8.75 d): 3P/ε computed along real neutron-star sequences, with a provisional crust. The code-correctness checks are DH2001 M_max and OV. The EOS-vs-data scores are GW170817 and the J0740 mass floor.
- **MS4** (≈ 12.0 d): unified EOS, with the NICER radius scores, plus quark cores.
- **MS5** (≈ 13.0 d): live in Mentat.

**Found on the way (flagged, not fixed):**
- **`element.py:515`** writes the SEMF asymmetry term 4× too weak. Line 560's `stable_mass_number` assumes the other convention.
  - On the 10 named nuclides, Pb-208 comes out +8.84 % high and U-238 +11.58 %. With the consistent form these become −1.09 % and −0.30 %.
  - On all 136 AME2020 nuclides with A ≥ 12, the rms falls from 6.82 % to 0.78 % [MEASURED].
- Two different SEMF coefficient sets exist.
- The saturation constants are uncited.
- `audit.py` claims a TOV that lives only in `../ssbm-theory`.
- `planets.NEUTRON_STAR` is hand-entered.
- `solve_ode`'s documented syntax cannot pass the Mentat front door.
- P6's two dense-matter rows fail a check.
- **PSR J0348+0432 was revised to 1.806 ± 0.037 M☉** by one 2025 timing analysis (Saffer et al. 2025, ApJL 983, L20) [LIT]. The ≥ 2 M☉ anchor now rests on PSR J0740+6620 alone.
- **The 2019 NICER J0030 solutions have been revisited** (Vinciguerra et al. 2024), and J0740's radius updated (Salmi et al. 2024).
- **The plan's own reference EOS, SLy, sits below two radius windows the v1 plan pre-registered** [VERIFIED numerics on LIT]. Those tests are rewritten in v2 as EOS-vs-data scores with their expected SLy outcomes recorded in advance (§4, M4).

![plan figure v2](n1_plan_figure_v2.png)

---

## 1. What exists today (the survey)

Grep over `sigma_ground/{field,inventory,mcp,dynamics,kernel}` and `scripts/` [MEASURED, `n1_survey_v2.json → grep_hits`]:
- `white_dwarf` 0, `equation_of_state` 0, `bag.model` 0, `symmetry_energy` 0.
- `TOV`: none an integrator.
- `solve_ivp|odeint`, `rk4|runge-kutta|dopri`, `eos`: see the JSON.

| where | what is there | verdict for B1 |
|---|---|---|
| `field/constants.py:395-411` | `N0_FM3 = 0.16`, `E_SAT_MEV = −16`, `K_SAT_MEV = 230`, `J_SYM_MEV = 32` | **no citation, no uncertainty, no L_sym**. Against Margueron 2018 Table 1 the pulls are +1.0σ (n_sat 0.155 ± 0.005), −0.67σ (E_sat −15.8 ± 0.3), 0 (K 230 ± 20) and 0 (J 32 ± 2) [DERIVED]. No engine uses them |
| `field/constants.py:268-276` | `R0_FM = 1.215`, `A_C_MEV` derived from Coulomb's law (0.7111 MeV) | keep: the one SEMF coefficient already derived (Golden Rule 1) |
| `field/binding.py` | Coulomb term; σ-scaling of a *given* binding energy | keep; M3 supplies the binding energy |
| `field/interface/element.py:110-116, 485-528` | SEMF set #1 (15.56 / 17.23 / A_C / 23.29 / 12.0), "FITTED", uncited | **defect N1-a** (asymmetry convention); superseded by M3 |
| `mcp/tools/nuclear.py:57-72` | SEMF set #2 inline (15.75 / 17.8 / 0.711 / 23.7 / 11.18) + private `u`, `m_H`, `m_n` | PLATINUM §3 duplicate; internals move to M3, tool API kept |
| `field/nucleon.py` | `proton_mass_mev(σ)`, `neutron_mass_mev(σ)` | the σ hook M1/M4 call |
| `field/interface/statistical.py:124` | Fermi–Dirac occupation only | no EOS |
| `field/interface/quantum_matter.py:119-160` | metal free-electron E_F, Thomas–Fermi length | not a degenerate-star EOS |
| **`field/interface/adapters/integrator.py`** | `sg_solve_ivp`, `sg_odeint`, `sg_nbody`. **Fixed-step** Forest–Ruth, Verlet, RK4; numpy; scipy-compatible signature; aimed at Hamiltonian systems | **must be reconciled (D2)**: no adaptive error control, dense output or terminal events, which a TOV surface needs. The adaptive method lands once, and `sg_solve_ivp(method="dopri5")` delegates to it |
| **`dynamics/stepper.py`** | leapfrog sub-stepping with a CFL step for the SPH/continuum scene | a scene stepper, not a general ODE solver; untouched |
| **`dynamics/fluid/eos.py`** + `dynamics/continuum.py` | SPH pressure functions of ρ (Tait, ideal gas, liquid sound speed); `eos ∈ {tait, isothermal, ideal}` | no barotropic EOS object, no energy density, no degenerate matter. M5's protocol is for cold barotropic P(n), ε(n). Wrapping the SPH functions as `EOS` instances is a later option, D2b |
| `field/planets.py:137-176` | `NEUTRON_STAR`: 1.4 M☉, 11 km, four layers, round densities, uncited | replaced by M6 output (PLATINUM §10) |
| `field/audit.py:247-279` | "Chandrasekhar Fermi gas (exact)", "TOV integration (full)", "M_TOV = 2.071 M☉" | code is in `../ssbm-theory`; re-point when M5 lands |
| `inventory/data/isotopes.json` | 145 nuclides from AME2020 (136 with A ≥ 12) | the SEMF fit set for M3, the measured masses for M6 |
| `mcp/tools/mathx.py:209` `solve_ode` | `sympy.dsolve` only. With `Derivative()` syntax Mentat returns Lane–Emden n = 1 as (C1 J₁/₂(x) + C2 Y₁/₂(x))/√x [MEASURED]. Prime syntax is refused at the front door (N1-c) | no numeric IVP tool in Mentat; M0 + M5 fill it |
| `pyproject.toml:12` | `dependencies = []` | true of the metadata only. **8 non-test modules under `field/` and `dynamics/` import numpy at module level** [MEASURED] (list in the JSON) |
| `SkippyTel/mentat/Dockerfile:9` | Mentat pins sigma-ground at a commit (`SIGMA_GROUND_REF`); `MENTAT_CALL_TIMEOUT = 90` s | deploy = push, bump the ref, rebuild `mentat-home`; every tool must answer in ≪ 90 s |

**Prior art outside the repo.** `../ssbm-theory/sigma_ground/field/unsolved.py:148-635` has a Chandrasekhar EOS and a TOV. It is not portable:
- pure neutron matter, no β-equilibrium;
- an interaction term that saturates on its own, added on top of the full kinetic term;
- negative pressures clamped to 0;
- forward Euler at dr = 10 m;
- σ from a fixed R = 12 km.

Its "M_TOV = 2.071 M☉" is not reproducible evidence. The parallel-session scratch harness `scratchpad/sigma-bridge-1pn/tov_sigma_pair.py` is the better seed: DOP853 TOV, Tolman mass, σ monopole on the TOV metric, two exact benchmarks. Its benchmark `benchmark_uniform_C0.5` uses **r_s/R = 0.5**, not GM/(Rc²) = 0.5; see M5.

### 1.1 Engine and document defects found (Mentat flags, never fixes)

| # | file:line | defect | evidence | disposition |
|---|---|---|---|---|
| N1-a | `field/interface/element.py:515` | asymmetry term `a_a (A−2Z)²/(4A)` with `a_a = 23.29`. That coefficient belongs to the `(A−2Z)²/A` form that `stable_mass_number()` assumes (line 560, `coeff = a_c/(2 a_a)`), so the penalty is 4× too small | 10 nuclides: Pb-208 +8.84 % → −1.09 %, U-238 +11.58 % → −0.30 %. All 136 nuclides with A ≥ 12: rms 6.82 % → 0.78 %. Spearman(residual, N − Z) = 0.95. Best divisor k = 1.03 (code: 4) [MEASURED] | M3 lands the fix behind a failing-first regression test (D6) |
| N1-b | `mcp/tools/nuclear.py:62` | second SEMF set inline; private `_U_MEV`, `_M_H_ATOM_U`, `_M_NEUTRON_U` | rms 1.14 % (A ≥ 12), He-4 −19.3 % [MEASURED]; PLATINUM §3 | M3 replaces it; output values change (D9) |
| N1-c | `SkippyTel/mentat/app.py:178` (regex), `:236` (refusal) | the front-door `_UNSAFE` regex refuses any apostrophe. It therefore blocks the prime syntax that `solve_ode` documents (`mathx.py:213`, example `"y'' + y"`). The guard is a deliberate `eval()` defence: the mismatch is between the guard and the tool's documented interface, not a reason to drop the guard. Same family: the `_HUGE` power-tower guard refuses `(4*pi**2)**(3/4)` as a false positive | Mentat refused `"y'' + y"` and the prime-form Lane–Emden. With `x*Derivative(y(x),x,2) + 2*Derivative(y(x),x) + x*y(x)` it returns (C1 J₁/₂ + C2 Y₁/₂)/√x [MEASURED] | flag for SkippyTel; the docstring or a pre-guard translation should change, not the guard |
| N1-d | `field/constants.py:399-411` | saturation anchors uncited, no uncertainties, no L_sym | read; source is only "sigma_ground.field.constants" | P0 cites Margueron 2018 Tables 1–2 and adds `L_SYM_MEV` (§8) |
| N1-e | `field/audit.py:266` | lists a live "TOV integration (full)" and "M_TOV = 2.071 M☉" | no integrator under `sigma_ground/` | re-point to M5 |
| N1-f | `field/planets.py:138` | hand-entered neutron star | read | M6 derives it (D7) |
| N1-g | `GOLDEN_RULES.md:31` | Rule 2 points at `sigma_ground/field/bounds.py`, removed in e66615b | `ls` fails | doc fix, Captain's OK (D8) |
| N1-h | `scripts/unification/pilot_p6_sigma_bridge.py:49-50` | P6's [LIT] dense rows fail checks: NS row P is 1.61× GW170817's 90 % bound at 2 n_sat while sitting at 1.87 n_sat; WD row P is 3.81× Chandrasekhar's at 10⁹ kg/m³ (x = 0.80) | [DERIVED] Mentat c, m_n, N0_FM3, m_e, ħ, u + [LIT] Abbott 2018 | flag; P6's rows are replaced by M5/M8 output when they land |
| N1-i | `field/constants.py:221` vs `lookup_constant('m_e')` | m_e CODATA 2018 vs scipy CODATA 2022 | relative difference 1.4 × 10⁻⁹ [MEASURED] | harmless; noted |

---

## 2. Design principles (how B1 obeys the rules)

1. **Placement (pending D1).** All physics would live in `sigma_ground/field/dense/` (tier 1). It imports only:
   - `field.constants`;
   - `field.nucleon`;
   - `kernel.ode`;
   - the AME table through `inventory.data.loader.IsotopeDB`. `inventory` is tier 1, as `field/interface/ore_abundance.py` already uses it.

   MCP is tier 4. `tests/test_layering.py` guards it.
2. **One integrator (PLATINUM §3, D2).** The adaptive DOPRI5(4) lives once, in `kernel/ode.py`, written in pure Python so tier 0 stays numpy-free. `sg_solve_ivp(method="dopri5", events=...)` delegates to it, so callers keep one entry point. The fixed-step symplectic methods stay where they are. v1 justified pure Python by "the core is zero-dependency". That is true of `pyproject.toml` only, since 8 modules under `field/` and `dynamics/` already import numpy. The real reasons are tier-0 purity and Mentat's container footprint, and the Captain may instead allow numpy in `field/dense/` (D2).
3. **Golden Rule 1.** Every module's docstring carries a derivation chain: measured seeds → identities → outputs, tagged MEASURED / FIRST_PRINCIPLES / FITTED. Fitted numbers are either fitted in the repo from cited data (M3) or copied from a cited fit (SLy4, the bag-constant range).
4. **Golden Rule 2.** Each public function enforces its preconditions and returns a `regime` label:
   - SAFE: inside validity.
   - EDGE: approaching breakdown, e.g. T/T_F > 0.1 for a T = 0 EOS, or c_s² > 0.9.
   - WALL: the breakdown itself, e.g. drip reached in the WD EOS, or causality violated.
   - BEYOND: outside validity.

   It is a local label until D8 decides where the shared classifier lives.
5. **Golden Rule 4.** Every simplification gets a `# PHYSICS_GAP:` and a `KNOWN_GAPS.md` line (§9).
6. **Golden Rule 7.** The bag constant B is a *required* argument with its cited range and no silent default. The meta-model's poorly measured parameters default to **Margueron 2018's central values** (K_sym = −100 ± 100, Q_sat = 300 ± 400 MeV), not 0, and are swept over their quoted ranges.
7. **The σ hook.** Every EOS takes `sigma=SIGMA_HERE`, and at that value standard physics is reproduced exactly. The engine never imports ξ. The [SPECULATIVE] σ profile in M8 takes ξ, normalisation (C2), sign (B5/C4) and range (C5) as required arguments.
8. **One EOS protocol.** The `EOS` object exposes:
   - `pressure(n)`, `energy_density(n)`, `n_of_P(P)`, `eps_of_P(P)`;
   - `sound_speed_sq(n)`, `regime(n)`.

   It is analytic or tabulated in (log n, log P, log ε) with monotone cubic interpolation. A first-law self-check requires dε/dn = (ε + P)/n to 10⁻⁶. The registry holds:
   - `free_neutron`, `electron:mu_e`, `polytrope:n,K`;
   - `metamodel`, `SLy4`, `SLy4+BPS`;
   - `bag:B`, `hybrid:SLy4+bag:B`;
   - `HP2004:SLy` (provisional, P6; §4 M4).
9. **One compactness symbol.** C = GM/(Rc²) everywhere, as Podkowka–Mendes–Poisson define it. Where a harness uses r_s/R = 2C, the plan writes r_s/R explicitly. The Buchdahl bound is C < 4/9.
10. **Units.** SI at the public boundary. MeV/fm helpers use `HBAR*C` and `MEV_TO_J` from `constants.py`, never literals.

---

## 3. Package layout

```
sigma_ground/kernel/ode.py                 M0  tier 0  pure-Python DOPRI5(4) + dense output + events (one adaptive integrator)
sigma_ground/kernel/test_ode.py
sigma_ground/field/interface/adapters/integrator.py   (existing) sg_solve_ivp gains method="dopri5" -> kernel.ode
sigma_ground/field/dense/__init__.py           tier 1  public re-exports     (package home pending D1)
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
sigma_ground/field/dense/test_*.py             one test file per module
sigma_ground/mcp/tools/dense_matter.py     M9  tier 4
sigma_ground/mcp/test_dense_matter_mcp.py
```

That is `kernel/ode.py` plus **eight** modules in `field/dense/` (`eos`+`tov` count as M5, `polytrope`+`white_dwarf` as M2), then M9.

---

## 4. The modules

**How tests are labelled.** Every test carries exactly one label, and the same labels are used in the risk table (§10) and in the structured result:
- **DATA**: compares against a measurement. It can come out against the code for a physics reason, and that is validation.
- **LIT**: reproduces a published *calculation*. A failure means a bug or a transcription error. This is code correctness, not evidence.
- **BUG**: an identity, a limit or a conservation law.
- **BY CONSTRUCTION**: holds by how the code is built.
- **SOFT**: reported and flagged, never failed.

### M0 — `kernel/ode.py` (tier 0) · 0.75 d (P0, including the `sg_solve_ivp` delegation)

- **Public:**
  - `dopri5(f, t_span, y0, rtol, atol, events=(), dense=True) -> ODEResult`: FSAL, PI step control, 4th-order dense output, terminal events located by Brent;
  - `series_start(...)` for regular-singular centres;
  - `sg_solve_ivp(..., method="dopri5", events=...)` forwarding to it.
- **Tests:**

| test | target | label |
|---|---|---|
| y' = −y | e^{−t} to 1e-10 | BUG |
| harmonic oscillator, 100 periods | energy drift ≤ rtol-scaled bound | BUG |
| Lane–Emden n = 0, 1, 5 | θ₀ = 1 − ξ²/6; θ₁ = sin ξ/ξ (ξ₁ = π; scratch DOP853 gives ξ₁ − π = −3.6e-13); θ₅ = (1 + ξ²/3)^{−1/2} | BUG |
| event location | zero of sin ξ/ξ at π to 1e-12 | BUG |
| delegation | `sg_solve_ivp(method="dopri5")` equals `dopri5` bit for bit | BUG |

### M1 — `field/dense/fermi_gas.py` · 1.25 d (P1)

- **Public:**
  - `FermionSpecies` (`ELECTRON`, `MUON`, `PROTON`, `NEUTRON`);
  - `fermi_momentum(n)`, `relativity_parameter(n, species)`;
  - `chandrasekhar_pressure(x)`, `chandrasekhar_energy(x, include_rest=True)`, `trace(x)`;
  - `eos_T0(n, species)`, `electron_gas(rho, mu_e)`, `eos_finite_T_nonrel(n, T, species)`.
- **Derivation.** In units of m⁴c⁵/(π²ħ³), with x = p_F/mc:
  - P = (1/3)∫₀ˣ t⁴/√(1+t²) dt;
  - ε = ∫₀ˣ t²√(1+t²) dt;
  - ε − 3P = ∫₀ˣ t²/√(1+t²) dt = (x√(1+x²) − asinh x)/2.

  These are Mentat `integrate_expr`/`simplify_expr` closed forms [MEASURED], agreeing with Chandrasekhar 1935 [LIT].
- **Both regimes, read as two axes:**
  - T = 0, non-relativistic ↔ ultra-relativistic, exact: P ≈ x⁵/15 by Mentat series, P/x⁴ → 1/12 by Mentat limit.
  - degenerate ↔ classical, through NR Fermi–Dirac integrals.
  - Fully relativistic finite T is a PHYSICS_GAP.
- **Tests:**

| test | target | label |
|---|---|---|
| NR limit | P/(x⁵/15) → 1 | BUG |
| UR limit | P/(x⁴/12) → 1, 3P/ε → 1⁻ | BUG |
| free-gas trace positive | ε − 3P > 0 for all x > 0 | BY CONSTRUCTION (theorem; matters for M8) |
| Gibbs–Duhem | P = nμ − ε to 1e-12 | BUG |
| causality | 0 < c_s² < 1/3 | BUG |
| classical limit / Sommerfeld | P → n k_B T; low-T π² term | BUG |
| WD pressure share | 3P/ρc² = 8.77e-5 at 10⁹ kg/m³, μ_e = 2 (x = 0.8007) [DERIVED] | BUG |

- **MCP:** `degenerate_gas_eos(species, number_density_m3=None, mass_density_kg_m3=None, mu_e=2.0, temperature_k=0.0)`.

### M2 — `field/dense/polytrope.py` + `white_dwarf.py` · 1.0 d (P2)

- **Public:**
  - `lane_emden(n)`, `polytrope_structure(n, K, rho_c)`;
  - `chandrasekhar_mass(mu_e, baryon_mass="amu")`;
  - `white_dwarf_structure(rho_c | mass, mu_e, gr=False, coulomb=True)`, `mu_e_from_composition`, `white_dwarf_mass_radius`.
- **Derivation:** Newtonian hydrostatics + M1 gives Chandrasekhar 1935. Salpeter 1961 adds the Coulomb corrections. GR enters via M5. The mass formula is M_Ch = (√(3π)/2) ω₃ (ħc/G)^{3/2}/(μ_e m_B)².
- **Anchors:**
  - ω₃ = 2.01824, ξ₁(3) = 6.89685 [LIT]. Scratch DOP853 gives 2.018236 and 6.896849 [VERIFIED numerics].
  - **M_Ch(μ_e = 2)** [DERIVED]: 1.4559 M☉ (m_u), 1.4333 (m_H), 1.4310 (m_n). The spread is 1.74 %.
  - Sirius B M = 1.018 ± 0.011 M☉ [LIT: Bond et al. 2017, ApJ 840, 70]. Its radius must be read from the paper before the test is written.
- **Tests:**

| test | target | label |
|---|---|---|
| ω₃, ξ₁(3) | 2.01824, 6.89685 to 1e-5 | LIT |
| n = 1 | ξ₁ = π; M, R analytic | BUG |
| Chandrasekhar limit | exact-EOS WD: M(ρ_c → ∞) → M_Ch (m_u convention) to 0.2 % | BUG (M1 + M0 together) |
| low-mass slope | d ln M/d ln ρ_c → 1/2 | BUG |
| GR instability | GR sequence peaks below M_Ch | BUG (a theorem of GR; fails only if GR is not applied) |
| Coulomb correction sign | lowers R at fixed M and lowers M_max; grows with Z | BUG |
| Sirius B | predicted R(1.018 M☉, C/O, Salpeter) inside Bond et al.'s radius interval | **DATA** |

- **MCP:** `chandrasekhar_mass(mu_e=2.0)`, `polytrope_structure(...)`, `white_dwarf_structure(...)`.

### M3 — `field/dense/semf.py` · 0.75 d (P5)

- **Public:**
  - `semf_binding_energy_mev(Z, A)`, `semf_terms(Z, A)`, `semf_coefficients()`;
  - `nuclear_mass_kg(Z, A)`, with provenance MEASURED or SEMF;
  - `beta_stable_Z(A)`.
- **Derivation:**
  - a_C is DERIVED (`A_C_MEV`).
  - a_V, a_S, a_A and a_P come from a least-squares fit, done in the repo, to AME2020 (`isotopes.json`, A ≥ 16, measured entries).
  - One convention: a_A (N − Z)²/A, pairing ± a_P A^{−1/2}.
  - Scratch refit on the 131 nuclides with A ≥ 16 [VERIFIED numerics]: a_V = 15.69, a_S = 17.79, a_A = 23.43, a_P = 12.67 MeV, with **rms 0.49 %** (3.33 MeV).
- **Anchors:** AME2020 [LIT: Wang et al. 2021, Chinese Phys. C 45, 030003].
- **Callers updated (§10):**
  - `element.py::_binding_energy_MeV` and `stable_mass_number`;
  - `mcp/tools/nuclear.py::nuclear_binding_energy`. Its SEMF output values change; the exact-mass path does not (D9).
- **Tests:**

| test | target | label |
|---|---|---|
| regression N1-a (written first; fails on today's code) | Pb-208 and U-238 within 2 % of AME2020 (today +8.84 %, +11.58 %) | **DATA** |
| fit quality | rms ≤ **0.6 %** for A ≥ 16 on the fit set. The attainable optimum, 0.49 %, is known in advance, so a looser fit means a wrong term or a broken fit | BUG (v1: "≤ 1 %, yes"; too loose to fail on physics) |
| domain | A < 16 returns EDGE; He-4 flagged | BUG |
| β-stability | `beta_stable_Z(56) = 26`, `beta_stable_Z(208) = 82` within ±1 | DATA (weak: the fit set is the same data family) |
| consistency with M4 | \|a_V − \|E_sat\|\| < 1.5 MeV; a_A < J | **DATA** (two independent data sets) |
| element.py masses | predicted atomic masses change by < 0.1 % | BUG (bounded side-effect) |

- **MCP:** `nuclear_binding_energy` (signature unchanged; gains `terms`, `provenance`), `semf_coefficients()`, `beta_stable_nucleus(mass_number)`.

### M4 — `field/dense/nuclear_matter.py` · 2.5 d (P6)

- **Public:**
  - `MetaModel(n_sat, E_sat, K_sat, J, L, Q_sat=300, K_sym=-100, sigma=SIGMA_HERE)`, with defaults from Margueron 2018 Table 2;
  - `Skyrme(name="SLy4")`;
  - `energy_per_baryon_mev(n_fm3, delta)`, `pressure(n, delta)`, `symmetry_energy(n)`, `saturation_properties()`;
  - `beta_equilibrium(n_b, model, muons=True)`, `npe_mu_eos(model)`.
- **Derivation:**
  - The meta-model is M1's kinetic part plus a Taylor potential in x = (n − n_sat)/(3n_sat). Its coefficients are the measured empirical parameters with their uncertainties, which gives an error band. It is valid to about 2 n_sat.
  - Skyrme SLy4 is a FITTED functional (Chabanat 1998). It is the interaction behind DH2001 and serves as the external code-correctness anchor.
  - β-equilibrium: μ_n = μ_p + μ_e, μ_μ = μ_e, n_p = n_e + n_μ.
- **Provisional crust (v2).** Until M6 lands in P7, M–R curves use the `HP2004:SLy` analytic unified EOS [LIT: Haensel & Potekhin 2004, A&A 428, 191] below the crust–core edge. Radius depends on the crust, so the NICER scores are recomputed on M6's unified EOS in P7, and those are the ones recorded.
- **Anchors** (P0 cites them in `constants.py`; §8):
  - **Margueron 2018 Tables 1–2** (experiment-based "present estimation"): n_sat 0.155 ± 0.005 fm⁻³, E_sat −15.8 ± 0.3, K_sat 230 ± 20, E_sym 32 ± 2, L_sym 60 ± 15, K_sym −100 ± 100, Q_sat 300 ± 400 MeV. Table 4's 50-model average (K_sat 251 ± 29, L_sym 76.6 ± 29.2) is kept only as a *model spread*.
  - Lattimer 2023 (J = 32.0 ± 1.1, L = 51.9 ± 7.9 MeV) is the theory cross-check.
  - GW170817: P(2 ρ_nuc) = 3.5 (+2.7/−1.7) × 10³⁴ dyn/cm² = [1.8, 6.2] × 10³³ Pa at 90 %, with **ρ_nuc = 2.8 × 10¹⁴ g/cm³ as Abbott et al. 2018 define it**. So 2 ρ_nuc = 5.6 × 10¹⁷ kg/m³ (Mentat `convert_units`).
  - DH2001 SLy M_max = 2.05 M☉.
  - J0740 mass 2.08 ± 0.07 M☉ (Fonseca 2021).
  - NICER 2024 radii (below).
- **Tests.** The EOS-vs-data tests say which EOS they score and at what credible level. They score R over the measured *mass interval*, not at the point mass. Each pre-states what a miss means. The "expected for SLy" column is the scratch TOV on the HP2004 SLy fit [VERIFIED numerics on LIT]. It is recorded now so that the implementation can be judged against it.

| test | EOS scored | criterion | expected for SLy | label · what a miss means |
|---|---|---|---|---|
| saturation recovery | meta-model | P(n_sat, δ = 0) = 0 to 1e-10; E/A, K, J, L recovered to 0.1 % | n/a | BUG |
| SLy4 transcription | SLy4 | saturation point = Chabanat's published n_sat, E/A, K, J | pass | LIT · a typo |
| β-equilibrium | all | neutrality to 1e-12; muon onset at μ_e = m_μc² | pass | BUG |
| DH2001 reproduction | SLy4 + crust | M_max = 2.05 M☉ ± 2 % | 2.047 M☉ | LIT · a code bug (TOV, EOS or units) |
| GW170817 | every registered EOS | P_β(ρ = 2 × 2.8e14 g/cm³) ∈ [1.8, 6.2] × 10³³ Pa (90 %) | 2.93 × 10³³ Pa: consistent | **DATA** · that EOS is disfavoured at 90 % |
| ≥ 2 M☉ floor | every registered EOS; `free_neutron` must miss (negative control) | M_max ≥ 1.94 M☉ (J0740, Fonseca 2021, 2σ) | 2.047: consistent | **DATA** · that EOS is excluded |
| J0740 radius (Salmi 2024) | every EOS reaching 2.004 M☉ | some stable R(M), M ∈ [2.004, 2.142], inside 12.49 (+1.28/−0.88) km (68 %); also report against the approximate 95 % edge 10.73 km (2σ, Gaussian approximation) | **R = 10.12–10.59 km over M ∈ [2.004, 2.045]: tension at 68 % and below the ≈95 % edge** | **DATA** · that EOS is disfavoured. For SLy this is the expected physics outcome (SLy is soft), not a bug |
| J0437 radius (Choudhury 2024) | every EOS | some R(M), M ∈ [1.381, 1.455], inside 11.36 (+0.95/−0.63) km (68 %) | R = 11.66–11.71 km: consistent | **DATA** · that EOS is disfavoured |
| J0030 (Riley 2019, Miller 2019, Vinciguerra 2024) | every EOS | report R(M) against each solution; never fail | Riley 2019: consistent (11.64–11.80 vs ≥ 11.52); Miller 2019: tension at 68 % (11.55–11.75 vs ≥ 11.96), consistent at ≈95 % (≥ 10.90) | SOFT · the 2019 solutions were revisited and the 2024 posterior is multimodal |
| causality | all | c_s² ≤ 1 up to the M_max central density, else WALL | pass | BUG (regime label) |
| trace-flip compactness (with M8) | every EOS that flips | report C = GM/(Rc²) at 3P_c = ε_c against PMP's 0.262 (+0.011/−0.017); flag outside the band | C = 0.262 at M = 1.94 M☉ | SOFT · PMP's band is a 90 % spread over 20,000 piecewise polytropes with a DH2001 (SLy) low-density part, not a measurement; for SLy it reproduces the literature |

Posterior samples. If the implementation scores against published NICER posterior samples instead of the 68 % intervals, it records the sample source and the likelihood definition before running.

- **MCP:** `nuclear_matter_properties(model="metamodel", density_fm3=None, asymmetry=0.0)`, `beta_equilibrium(baryon_density_fm3, model="SLy4")`.

### M5 — `field/dense/eos.py` + `tov.py` · 1.5 d (P3)

- **Public:**
  - `solve_tov(eos, central_pressure_pa=None, central_density_kg_m3=None, rtol=1e-10)` → `TOVStar`. It returns R, M, M_baryon, compactness C = GM/(Rc²), binding energy, Tolman mass, trace mass, and profiles.
  - `mass_radius_curve`, `maximum_mass`, `is_stable`, `newtonian_structure`.
- **Derivation:** the TOV equations (Tolman 1939; Oppenheimer & Volkoff 1939), with ε in J/m³:
  - dP/dr = −G(ε + P)(m + 4πr³P/c²)/[c²r²(1 − 2Gm/rc²)]
  - dm/dr = 4πr²ε/c²
  - dν/dr = −2(dP/dr)/(ε + P)

  The centre starts by series and the surface is a terminal event. The Tolman (1930) mass is ∫(ε + 3P)e^{(ν+λ)/2}dV/c².
- **Tests:**

| test | target | label |
|---|---|---|
| Schwarzschild interior (uniform ε) | **r_s/R = 2GM/(Rc²) = 0.5, i.e. C = GM/(Rc²) = 0.25**: R, M, P(r) exact to 1e-9 (v1 wrote "C = 0.5", which in this plan's convention violates Buchdahl's C < 4/9) | BUG |
| Buchdahl | C < 4/9 (r_s/R < 8/9) for every star | BUG |
| Newtonian limit | n = 1 polytrope at w_c = 1e-5: M/M_LE − 1, R/πa − 1 < 1e-4 | BUG |
| Tolman identity | tolman_mass = M to 1e-8 | BY CONSTRUCTION (catches a wrong ν) |
| first law | dε/dn = (ε + P)/n to 1e-6 | BUG |
| free neutron gas | M_max = **0.710 ± 0.005 M☉** (scratch TOV on Mentat-closed-form Chandrasekhar functions, m_n: 0.7097 M☉ at R = 8.97 km, central 3P/ε = 0.30) | LIT (OV 1939) · a bug |
| baryon mass | M_baryon > M for bound stars | BUG |
| performance | one star < 0.5 s; 30-point curve < 30 s pure Python | engineering |

- **MCP:** `tov_star(eos="SLy4+BPS", mass_msun=None, central_density_kg_m3=None)`, `mass_radius_curve(eos, n_points=30)`, `maximum_mass(eos)`.

### M6 — `field/dense/composition.py` · 2.0 d (P7)

- **Public:**
  - `outer_crust(pressure_pa)`, `neutron_drip()`, `crust_core_transition(model)`;
  - `unified_eos(core_model)`, `layers(star)`, `white_dwarf_layers(...)`.
- **Derivation:**
  - BPS 1971 minimises the Gibbs energy per nucleon over (Z, A), using M3/AME masses, the M1 electron gas and the Salpeter lattice energy.
  - Drip is where μ_n = m_nc².
  - The crust–core edge comes from M4's thermodynamic instability.
  - The inner crust is a flagged smooth join (PHYSICS_GAP).
- **Tests:**

| test | target | label |
|---|---|---|
| first nucleus | ⁵⁶Fe at P → 0 | LIT (BPS) · a bug in lattice/electron terms |
| monotone neutronisation | Z/A non-increasing with P | BUG |
| drip | ρ_drip within 10 % of 4.3 × 10¹⁴ kg/m³ | LIT (BPS, secondary-verified) |
| crust–core edge (SLy4) | 0.08 fm⁻³ ± 15 % | LIT (DH2001, secondary) |
| NICER scores, final | the M4 J0740/J0437/J0030 rows recomputed on `unified_eos('SLy4')` and the meta-model band; recorded as the official scores | DATA (as M4) |
| layers | radii nest; fractions sum to 1; crust mass fraction < 5 % at 1.4 M☉ | BUG |
| planets.NEUTRON_STAR | built from `layers(solve_tov(unified_eos('SLy4'), mass=1.4))`, keys kept | BUG (§10) |

- **MCP:** `neutron_star_layers(mass_msun=1.4, eos="SLy4+BPS")`, `outer_crust_composition(mass_density_kg_m3)`.

### M7 — `field/dense/quark_matter.py` · 1.25 d (P8)

- **Public:**
  - `bag_eos(B_mev_fm3, m_s_mev=0.0, alpha_s=0.0)`;
  - `strange_matter_energy_per_baryon(B, m_s)`;
  - `deconfinement_transition(hadronic, quark)` (Maxwell construction);
  - `hybrid_eos`, `strange_star(B)`.
- **Derivation:** free u, d, s quark gases (degeneracy 6 per flavour) plus B (Chodos et al. 1974). With massless quarks, at P = 0 one has ε = 4B and E/A = k B^{1/4}:
  - k₃ = √2·3^{3/4}·√π = 5.7139 (three flavours);
  - k₂ = √2·√π·(1 + 2·2^{1/3})^{3/4} = 6.4414 (two flavours, neutral, no electrons).

  Both come from Mentat `simplify_expr` [DERIVED].
- **Anchors:**
  - **stability window B^{1/4} = 144.4–162.8 MeV** (massless, α_s = 0, with E/A(⁵⁶Fe atom) = 930.41 MeV from AME2020 and the Mentat atomic mass unit) [DERIVED]. v1 quoted 145–164 MeV from a secondary source.
  - Witten 1984 and Farhi & Jaffe 1984 [LIT].
- **Tests:**

| test | target | label |
|---|---|---|
| bag trace | ε − 3P = 4B exactly (massless) | BY CONSTRUCTION |
| window edges | B^{1/4} = 144 MeV ⇒ two-flavour matter bound (excluded); 150 MeV ⇒ strange matter stable; 165 MeV ⇒ unstable | LIT/BUG · degeneracy or unit error |
| homology | M_max, R ∝ B^{−1/2} to 1e-6 | BUG |
| Maxwell construction | P, μ_B continuous; Δε > 0 | BUG |
| hybrid branch | stability reported; twin branch flagged | SOFT |

- **MCP:** `bag_model_eos(...)`, `deconfinement_transition(hadronic_eos="SLy4", bag_constant_mev_fm3=...)`.

### M8 — `field/dense/trace.py` · 1.0 d (P4; real-EOS rows in P6)

- **Public, pure GR:**
  - `trace_profile(star)`: w(r) = 3P/ε, Δ(r) = (ε − 3P)/(3ε) (Fujimoto et al. 2022), and the sign-flip radius;
  - `trace_mass(star)`;
  - `pressure_virial_fraction(star)`;
  - `weak_field_comparison(star)`.
- **Public, [SPECULATIVE] and fenced:** `sigma_static_profile(star, xi, normalisation, sign, range_m)`. It integrates

      d/dr[e^{(ν−λ)/2} r² σ'] − e^{(ν+λ)/2} r² σ/λ_σ² = ∓(8πGξ/c⁴)(ε − 3P) e^{(ν+λ)/2} r²

  on the TOV metric. All four arguments are required.
- **Anchors:**
  - **PMP 2018.** C = GM/(Rc²) at central flip is 0.262 (+0.011/−0.017). This is the median and 90 % spread over 20,000 piecewise polytropes whose low-density part is fixed to DH2001's fit: a *model-ensemble* spread.
  - **Fujimoto et al. 2022.** The matter part of the trace anomaly "may be positive definite" when matched to high-density QCD, i.e. **no flip at all**. Whether real stars flip is therefore open.
  - Mendes 2015 and Damour & Esposito-Farèse 1993 [LIT].
- **Tests:**

| test | target | label |
|---|---|---|
| Tolman = M | 1e-8 on every star | BY CONSTRUCTION |
| small-w_c slope | n = 1 polytrope, w_c = 1e-3 … 1e-2: (1 − M_tr/M)/w_c → 1.00 ± 0.01. This follows analytically [DERIVED]: 3∫P dV/∫ρc² dV = w_c (π/2)/π = w_c/2 by Mentat integrals, and the Tolman normalisation doubles it | **BUG** (v1 said "yes: decides the factor P7 got"; it does not) |
| `weak_field_comparison` reproduces P7 | flat ratio = 1 − w_c/2 exactly for n = 1 | BUG |
| free Fermi-gas stars | Δ > 0 everywhere | BY CONSTRUCTION |
| bag stars | Δ = 4B/(3ε) pointwise | BY CONSTRUCTION |
| trace-flip compactness | shared with M4: reported against PMP's band | SOFT |
| Yukawa limits | λ_σ → ∞ massless; λ_σ ≪ R recovers the local §6 formula | BUG |

- **MCP:** `stellar_trace_profile(eos, mass_msun)` (GR) and `sigma_bridge_profile(eos, mass_msun, xi, normalisation, sign, range_m)`. The latter's description opens with "[SPECULATIVE] — rests on the unadopted □σ = −ξR postulate".

### M9 — Mentat integration · 1.0 d (P9)

- **Code:**
  - `mcp/tools/dense_matter.py`: 17 new tools, with `nuclear_binding_energy` keeping its signature;
  - registration in `mcp/server.py`, `manifest.py` domain `dense_matter`, `tool_keywords.py` phrasings;
  - `mcp/test_dense_matter_mcp.py`: every tool returns `source` and `formula`.
- **Docs:** `ARCHITECTURE.md` row, `KNOWN_GAPS.md` lines (§9), re-pointing `audit.py`, striking B1 from the gap list, `misc/SESSION_LOG.md`.
- **Deploy:**
  1. Push sigma-ground (solo-repo rule §17, after checking contributors).
  2. Bump `SIGMA_GROUND_REF` in `SkippyTel/mentat/Dockerfile:9` and `deploy.yml`.
  3. Rebuild with `cd /d/Aaron/development/SkippyTel && docker compose stop mentat-home && docker compose rm -f mentat-home && docker compose up --build -d mentat-home`.
  4. Verify that `bash tools/mentat_call.sh chandrasekhar_mass '{"mu_e": 2}'` returns 1.456 M☉ (m_u convention).
- **Separately, for SkippyTel:** N1-c (the front-door/docstring mismatch) is flagged to the Captain, not fixed here.

---

## 5. How this unlocks the σ-bridge pressure term and P7 beyond the weak field

**Today.** P6/P7 borrowed dense-matter rows, and two of P6's rows fail checks (N1-h). P7 solved the weak-field σ equation on an n = 1 polytrope with a flat-space source. That drops a (1 + 2Φ/c²) factor of the same post-Newtonian order as the term it keeps, and its w_c = 0.5 row was outside validity.

**After MS2** (P4: M0 + M1 + M5 + M8; polytropes and free gases):
- **The metric factors.** The static σ equation is solved on the exact TOV metric, so the metric factors are automatic.
- **What the parallel-session harness recorded.** It is an untracked file (`tov_sigma_pair_results.json`), read here, not rerun. For an n = 1 polytrope it gives M_σ/M_Tolman = 0.99900, 0.99006, 0.9054, 0.7439 and 0.6120 at w_c = 0.001, 0.01, 0.1, 0.3 and 0.5.
- **Normalisation, not a correction.** At small w_c the deficit per unit *Tolman* mass is w_c, while P7's flat ratio, normalised to the rest mass, is w_c/2 [DERIVED]. The factor of 2 is a choice of normalisation, not an error in P7. The observable is the one normalised to the gravitational mass, and M8 reports both. By w_c ≈ 0.3 neither expansion holds, and only the full solve answers.
- **No flip in a free gas.** A free Fermi gas cannot flip the σ source sign, because ε − 3P > 0 is a theorem (M1). A flip in a real star would be an interaction effect, which is why M4 is needed.

**After MS3/MS4** (realistic neutron stars):
- **Computed, not borrowed.** 3P/ε, Δ and M_tr/M become computed along each EOS's M–R sequence. For SLy [VERIFIED numerics on LIT], 3P_c/ε_c = 0.46 at 1.4 M☉ and 1.18 at 2.0 M☉. The centre flips at C = 0.262, M = 1.94 M☉, which reproduces PMP's median, since PMP's ensemble shares SLy's low-density part.
- **Compactness of the observed pulsars**, from Mentat's Schwarzschild radius:

  | pulsar | C = GM/(Rc²) |
  |---|---|
  | J0740+6620, Salmi 2024 | 0.245 (0.222–0.264 over the 68 % radius interval) |
  | J0740+6620, Riley 2021 | 0.247 (0.224–0.268) |
  | J0437–4715 | 0.184 (0.170–0.195) |
  | J0030 | 0.156 to 0.180 across the 2019 and 2024 solutions |

  J0740's interval runs from below PMP's band into its lower half.
- **Whether J0740's centre flips is open on two counts.** Within PMP's ensemble it is EOS-dependent. If Fujimoto et al.'s conjecture of a positive-definite trace anomaly holds, no star flips at all. M8 answers it per EOS. Neither answer is evidence for or against the σ-bridge. It decides only whether the pressure term in the σ source changes sign in the most compact star known.

**The surviving short-range corner [SPECULATIVE, DERIVED under §6's stated reading].** §6's local formula σ = 8πGξλ_σ²ρ/c² uses Mentat G, c, m_n and the repo's ξ = 0.1582. At ρ_sat = 2.68 × 10¹⁷ kg/m³:

| λ_σ | σ |
|---|---|
| 0.3 km | 7.1 × 10⁻⁵ |
| 1 km | 7.9 × 10⁻⁴ |
| 3 km | 7.1 × 10⁻³ |

The formula holds only for λ_σ ≪ R; km ranges need M8's Yukawa solve. A σ-modified TOV is out of B1's scope and stays [SPECULATIVE] until C1–C5 are decided.

**What this cannot do.** None of it tests "gravity is a fluid". The TOV, Tolman mass and trace are general relativity. Passing NICER or the J0740 floor validates an EOS, not the thesis. Every σ number stays conditional on the [SPECULATIVE] postulate.

---

## 6. Order of work, milestones and effort

| phase | content | depends on | days | ends at |
|---|---|---|---|---|
| P0 | M0 `kernel/ode.py` + `sg_solve_ivp(method="dopri5")` delegation (D2); cite the saturation constants (Margueron Tables 1–2), add `L_SYM_MEV`, `K_SYM_MEV`, `Q_SAT_MEV`, `M_MUON_MEV`, `M_STRANGE_MEV`; GOLDEN_RULES pointer if D8 approves | — | 0.75 | 0.75 |
| P1 | M1 Fermi gas | P0 | 1.25 | 2.0 |
| P2 | M2 Lane–Emden, white dwarfs, Chandrasekhar mass, Salpeter | P1 | 1.0 | 3.0 |
| P3 | M5 EOS protocol + TOV; GR white dwarfs; free-neutron star (0.710 M☉) | P1, P2 | 1.5 | **4.5 (MS1)** |
| P4 | M8 trace diagnostics; P7 on the TOV metric | P3 | 1.0 | **5.5 (MS2)** |
| P5 | M3 one SEMF; element.py + nuclear.py re-pointed (D6, D9); N1-a regression | P0 | 0.75 | 6.25 |
| P6 | M4 meta-model + SLy4 + β-equilibrium; NS sequences on the provisional HP2004 crust; DH2001 reproduction, GW170817, J0740 floor; provisional NICER and PMP reports | P1, P3, P4, P5 | 2.5 | **8.75 (MS3)** |
| P7 | M6 outer crust, drip, crust–core edge, unified EOS, layers; `planets.NEUTRON_STAR` derived; **official NICER scores** | P5, P6 | 2.0 | 10.75 |
| P8 | M7 bag EOS, deconfinement, hybrid and strange stars | P1, P6 | 1.25 | **12.0 (MS4)** |
| P9 | MCP sweep, docs, KNOWN_GAPS, push, ref bump, rebuild Mentat, verify | all | 1.0 | **13.0 (MS5)** |

- P5 can move earlier if the Captain wants the N1-a fix first.
- Each phase is one commit series with its tests (§14).
- The estimates are rough, for one agent. The largest risks are the SLy4 transcription plus the DH2001 reproduction, and the inner-crust join.

---

## 7. Decisions for the Captain (before P0)

| # | decision | recommendation |
|---|---|---|
| D1 | package home: `field/dense/` (or `field/compact/`) | `field/dense/` |
| D2 | one adaptive integrator. Option A: pure-Python DOPRI5 in `kernel/ode.py`, with `sg_solve_ivp(method="dopri5")` delegating. Option B: add it to `adapters/integrator.py` and accept numpy in `field/dense/` (8 field/dynamics modules already import numpy). Option C: coexist (rejected, PLATINUM §3) | A |
| D2b | `dynamics/fluid/eos.py` (SPH pressure functions) and M5's `EOS` protocol: wrap the SPH functions as `EOS` instances later, or keep them separate | keep separate now; wrap later if a scene needs a degenerate EOS |
| D3 | SEMF fit set: 145 nuclides now; full AME2020 table later | 145 now, full table when approved |
| D4 | SLy4 (code-correctness anchor) + meta-model on Margueron 2018 Tables 1–2, with K_sym and Q_sat at the cited central values and swept over their ranges | yes |
| D5 | σ hook: `SIGMA_HERE` by default; σ profile needs ξ, normalisation, sign and range; no ξ import | yes |
| D6 | land the N1-a fix inside M3 behind a failing-first test (atomic masses shift < 0.1 %) | yes |
| D7 | `planets.NEUTRON_STAR` becomes TOV-derived (key kept) | yes |
| **D8** | edit `GOLDEN_RULES.md:31` (Rule 2's pointer to the deleted `bounds.py`): restore a shared SAFE/EDGE/WALL/BEYOND helper, or re-point the rule to per-module labels | re-point to per-module labels; a shared helper is a separate item |
| **D9** | `nuclear_binding_energy` (MCP) SEMF *output values* change when M3 replaces its inline set; the measured-mass path is unchanged | yes, noted in the tool's changelog |
| D10 | NICER scoring: 68 % intervals plus a Gaussian-approximate 95 % edge, or published posterior samples (Zenodo releases; source to be confirmed at implementation) | intervals now; samples if the Captain wants a likelihood |

---

## 8. Data anchors (verified this session; how each was verified is in `n1_references_v2.json`)

The shared web-search budget ran out during v2, so v2 verification used direct fetches of arXiv abstract pages and ar5iv full text, plus Crossref metadata queries.

| quantity | value | reference | used by |
|---|---|---|---|
| Chandrasekhar mass, μ_e = 2 | 1.4559 (m_u) / 1.4333 (m_H) / 1.4310 (m_n) M☉ [DERIVED] | Chandrasekhar 1931, ApJ 74, 81; 1935, MNRAS 95, 207; ω₃ = 2.01824 | M2 |
| PSR J0740+6620 radius | **R = 12.49 (+1.28/−0.88) km, M = 2.073 ± 0.069 M☉ (68 %)** | **Salmi et al. 2024, ApJ 974, 294 (arXiv:2406.14466)** | M4/M6 (DATA) |
| PSR J0740+6620 radius (superseded) | 12.39 (+1.30/−0.98) km at 2.072 M☉ | Riley et al. 2021, ApJL 918, L27 | comparison |
| PSR J0740+6620 mass | 2.08 ± 0.07 M☉ (68.3 %) | Fonseca et al. 2021, ApJL 915, L12 | M4 (1.94 M☉ floor) |
| PSR J0437–4715 | **M = 1.418 ± 0.037 M☉, R = 11.36 (+0.95/−0.63) km (68 %)** | **Choudhury et al. 2024, ApJL 971, L20 (arXiv:2407.06789)** | M4/M6 (DATA) |
| PSR J0030+0451 (2019) | 1.34 / 12.71 km (Riley); 1.44 / 13.02 km (Miller) | Riley et al. 2019, ApJL 887, L21; Miller et al. 2019, ApJL 887, L24 | SOFT |
| PSR J0030+0451 (2024) | 2019 solutions disfavoured with XMM-Newton; ~[1.4 M☉, 11.5 km] or ~[1.7 M☉, 14.5 km], model-dependent | **Vinciguerra et al. 2024, ApJ 961, 62 (arXiv:2308.09469)** | SOFT |
| PSR J0348+0432 | 2.01 ± 0.04 M☉ (2013), **revised to 1.806 ± 0.037 M☉ by one 2025 timing analysis** | Antoniadis 2013, Science 340, 1233232 → Saffer et al. 2025, **ApJL 983, L20, doi:10.3847/2041-8213/adc25e** (verified by Crossref) | not a 2 M☉ anchor |
| nuclear saturation (experiment-based) | n_sat 0.155 ± 0.005 fm⁻³, E_sat −15.8 ± 0.3, K_sat 230 ± 20, E_sym 32 ± 2, L_sym 60 ± 15, K_sym −100 ± 100, Q_sat 300 ± 400 MeV | **Margueron, Hoffmann Casali & Gulminelli 2018, PRC 97, 025805, Tables 1–2** | M4, `constants.py` |
| model spread (not data) | 50-model average K_sat 251 ± 29, L_sym 76.6 ± 29.2 MeV | Margueron 2018, Table 4 | M4 sweep context |
| symmetry energy (theory) | J = 32.0 ± 1.1, L = 51.9 ± 7.9 MeV | Lattimer 2023, Particles 6, 30 | M4 cross-check |
| incompressibility | review; no value in the abstract | Garg & Colò 2018, PPNP 101, 55 | context only |
| pressure at 2 ρ_nuc | 3.5 (+2.7/−1.7) × 10³⁴ dyn/cm² = [1.8, 6.2] × 10³³ Pa at 90 %; ρ_nuc = 2.8 × 10¹⁴ g/cm³ | Abbott et al. 2018, PRL 121, 161101 | M4 (DATA) |
| SLy unified EOS | M_max = 2.05 M☉, M_min = 0.094 M☉ | Douchin & Haensel 2001, A&A 380, 151 | M4/M5/M6 (LIT) |
| SLy analytic fit | fit of the unified SLy EOS; provisional crust in P6 | Haensel & Potekhin 2004, A&A 428, 191 | M4 (P6) |
| SLy4 parameters | read from the paper at implementation | Chabanat et al. 1998, NPA 635, 231 | M4 |
| nuclide masses | AME2020 | Wang et al. 2021, Chinese Phys. C 45, 030003 | M3/M6 |
| outer crust | ⁵⁶Fe → neutron-rich nuclei; drip ≈ 4.3 × 10¹¹ g/cm³ | Baym, Pethick & Sutherland 1971, ApJ 170, 299 (secondary-verified) | M6 |
| TOV | — | Tolman 1939; Oppenheimer & Volkoff 1939 (free-neutron M_max recomputed: 0.710 M☉; the OV abstract could not be fetched) | M5 |
| Tolman mass | ∫(ε + 3P)√g₀₀ dV | Tolman 1930, Phys. Rev. 35, 875 | M5/M8 |
| trace sign flip | C = GM/(Rc²) = 0.262 (+0.011/−0.017), 90 % spread over 20,000 EOS | Podkowka, Mendes & Poisson 2018, PRD 98, 064057 | M8 (SOFT) |
| trace anomaly | matter part "may be positive definite"; steep approach to the conformal limit | Fujimoto, Fukushima, McLerran & Praszałowicz 2022, PRL 129, 252702 | M8, §5 |
| scalar–tensor strong field | spontaneous scalarization | Damour & Esposito-Farèse 1993; Mendes 2015 | §5 |
| WD Coulomb corrections | — | Salpeter 1961, ApJ 134, 669; Hamada & Salpeter 1961, ApJ 134, 683 | M2/M6 |
| Sirius B | 1.018 ± 0.011 M☉ | Bond et al. 2017, **ApJ 840, 70** | M2 (DATA) |
| bag model | B^{1/4} window 144.4–162.8 MeV [DERIVED] | Chodos 1974; Witten 1984; Farhi & Jaffe 1984 | M7 |
| piecewise polytrope (optional family) | — | Read et al. 2009, PRD 79, 124032 | M5 registry |

---

## 9. PHYSICS_GAP lines this work will add to KNOWN_GAPS.md

- Fully relativistic finite-T Fermi gas (hot WDs, proto-NS).
- Inner crust as a smooth join (no compressible liquid drop, no pasta).
- No hyperons, superfluidity, magnetic fields or rotation.
- The meta-model's Q_sat and K_sym are poorly measured (central values used, swept).
- Bag model without colour superconductivity; B is a declared model parameter.
- No white-dwarf crystallisation or cooling.

---

## 10. Risks

| risk | effect | mitigation |
|---|---|---|
| pure-Python speed | `mass_radius_curve` exceeds the 90 s call limit | cache tabulated EOS; enthalpy form; cap n_points; performance test |
| SLy4 transcription error | wrong M_max | two LIT checks: saturation vs Chabanat, M_max vs DH2001 (expected 2.047) |
| inner-crust scope creep | P7 overruns | PHYSICS_GAP join first |
| tests that pass by construction, or reproduce literature, mistaken for evidence | overclaiming | one label per test (§4). **DATA tests, the only validation, are:** the N1-a regression, the M3–M4 cross-check, β-stable Z (weak), Sirius B, GW170817, the J0740 mass floor, and the J0740 (Salmi 2024) and J0437 (Choudhury 2024) radius scores. Everything else is LIT, BUG, BY CONSTRUCTION or SOFT |
| a DATA miss read as a code bug, or the reverse | wrong fix | code correctness is settled by the LIT reproductions (DH2001, OV, Chabanat). A DATA miss then means that EOS is disfavoured. SLy's expected J0740 radius tension is recorded in advance |
| literature drift (as with J0348 and the NICER reanalyses) | stale anchors | anchors carry citation and date in `constants.py`; the references file records how each was verified; re-check NICER/pulsar anchors at P6 and P7 |
| σ decisions still open | ambiguous σ numbers | required arguments, the [SPECULATIVE] tag, no ξ import |

---

## 11. Corrections to the gap census and the program (design corrections)

1. **J0348+0432 is not a ≥ 2 M☉ anchor any more.** One 2025 timing analysis (Saffer et al. 2025, ApJL 983, L20) gives 1.806 ± 0.037 M☉ [LIT], so B1's "2 M☉ pulsars" anchor rests on J0740+6620 alone. Against the brief's original anchor set the plan is incomplete; against the corrected one it is satisfied.
2. **"Chandrasekhar mass ≈ 1.4 M☉" needs a convention.** The value is 1.456 / 1.433 / 1.431 M☉ for m_u / m_H / m_n [DERIVED].
3. **"Both regimes" is two axes.** NR ↔ UR at T = 0 is exact, degenerate ↔ classical is NR at finite T, and fully relativistic finite T is a declared gap.
4. **P7's factor is a normalisation choice, not an error.** Normalised to the rest mass, the pressure deficit is w_c/2. Normalised to the Tolman mass it is w_c at first order [DERIVED, Mentat integrals]. M8 reports both, and the slope test is bug-only.
5. **Saturation anchors.** v1 used Margueron 2018's Table 4 *model average* as "empirical parameters" and offered "230 vs 251 ± 29" as a correction. That correction is withdrawn: the repo's 230 MeV is Margueron's own experiment-based Table 1 value, and Tables 1–2 are now the anchors.
6. **NICER anchors drifted.** v1's anchor check reported only J0348. It missed that the J0030 2019 solutions were revisited (Vinciguerra 2024) and that J0740's radius was updated (Salmi 2024), and it did not use J0437 (Choudhury 2024). The v1 pre-registered check (2) is therefore **partly failed**; v2 adds all three.
7. **The PMP band is a model spread, not a measurement.** C is GM/(Rc²), as PMP define it. The trace-flip test is SOFT and is off the validation list.
8. **The 0.67 figure is withdrawn.** "3P/ρc² = 0.67 at a neutron-star centre" came from a P6 row that GW170817 excludes. At 2 ρ_nuc the value is 0.21 (0.11–0.37), and order one is reached only toward the centres of massive stars (SLy: 0.46 at 1.4 M☉, 1.18 at 2.0 M☉).
9. **v1's NICER tests could not be met by v1's own reference EOS.** SLy misses Miller's R(1.44) window, cannot reach 2.072 M☉, and lies below J0740's radius window. The tests are rewritten as EOS-vs-data scores with the EOS named, the credible level stated, mass intervals instead of point masses, the meaning of a miss pre-stated, and SLy's expected outcome recorded.
10. **M5's uniform-density benchmark** is r_s/R = 0.5 (C = 0.25). v1's "C = 0.5" violates Buchdahl in this plan's convention.
11. **The repo has integrators and an EOS layer.** `sg_solve_ivp` and `dynamics/fluid/eos.py` exist, and numpy is already imported by 8 field/dynamics modules. M0 is reconciled with them (D2, D2b). "No numeric ODE" holds only at the level of Mentat's MCP tools.
12. **The `solve_ode` refusal was mis-attributed.** It comes from the SkippyTel front door (`app.py:178`), not from `mathx.py`. Mentat does solve Lane–Emden n = 1 symbolically with `Derivative()` syntax.
13. **Whether real neutron stars flip the trace sign is open.** Fujimoto et al. 2022 conjecture a positive-definite trace anomaly, which means no flip.
14. **The bag window is 144.4–162.8 MeV** [DERIVED], not 145–164.
15. **GW170817 is evaluated at Abbott's 2 ρ_nuc** (ρ_nuc = 2.8 × 10¹⁴ g/cm³), not at a model's own 2 n_sat.
16. **Golden Rule 2's pointer** to `field/bounds.py` is stale since e66615b (D8).

---

## 12. What changed from v1 (repairs)

- **element.py citations:** :518/:563 → **:515/:560**. The lines are now computed by grep in the survey.
- **`solve_ode` defect re-attributed:**
  - it now points at `SkippyTel/mentat/app.py:178` (regex) and `:236` (refusal), against the documented syntax at `mathx.py:213`;
  - the `Derivative()` Lane–Emden solution is recorded;
  - "no numeric IVP tool" is kept;
  - the `_HUGE` false positive is noted.
- **The pressure-share number:** 0.67 withdrawn. Replaced by GW170817 at 2 ρ_nuc (0.21, 0.11–0.37), Chandrasekhar WD values and SLy central values. P6's NS and WD rows are flagged (N1-h). The figure panel is redrawn with "3P = ε" as the sign-change line.
- **M5 benchmark:** r_s/R = 0.5 (C = 0.25). One compactness symbol throughout.
- **Headline and module count:** `kernel/ode.py` (tier 0) + 8 modules in `field/dense/` (tier 1, pending D1), then M9.
- **NICER radius scores** move to P7 on the unified EOS; P6 uses a provisional HP2004 crust.
- **Saturation anchors:** Margueron Tables 1–2. K_sym and Q_sat default to the cited central values. Design correction 5 is replaced.
- **PMP:** C defined as GM/(Rc²); the band described as a model spread; the test made SOFT and taken off the validation list.
- **NICER anchors:** Vinciguerra 2024, Salmi 2024 and Choudhury 2024 added, and the J0030 2019 solutions made SOFT. Literature drift added to the risks.
- **Fujimoto 2022:** its no-flip conjecture added to §5 and M8. The unverified "Δ ≈ 0 by 5 n_sat" phrase is withdrawn.
- **Bag window:** 144.4–162.8 MeV [DERIVED].
- **References:**
  - Saffer 2025 verified as ApJL 983, L20 by Crossref, and worded as "one 2025 timing analysis".
  - Bond 2017 is ApJ 840, 70.
  - Abbott's ρ_nuc convention was read and is used.
  - The Chamel 2013 "~0.7 M☉" attribution is dropped in favour of a computed 0.710 M☉.
- **Check (2):** reported as partly failed in v1.
- **NICER/J0740 tests rewritten:** they name the EOS, state the credible level, use mass intervals and pre-state what a miss means, with SLy's expected outcomes recorded (§4 M4).
- **Survey:** `adapters/integrator.py`, `dynamics/stepper.py` and `dynamics/fluid/eos.py` added; numpy imports counted (8); D2 restated; D2b added.
- **M8 slope test:** relabelled BUG, with an analytic derivation. Design correction 4 is reworded as a normalisation difference.
- **J0740 compactness** is given as an interval (0.222–0.264 Salmi; 0.224–0.268 Riley). For SLy, the PMP flip reproduces the literature.
- **Test labels:** one label scheme (DATA / LIT / BUG / BY CONSTRUCTION / SOFT) used everywhere. The M3 fit threshold is tightened to 0.6 % and labelled BUG.
- **Captain decisions:** the headline presents `field/dense/` as pending D1. D8 (GOLDEN_RULES edit), D9 (`nuclear_binding_energy` output change) and D10 (NICER scoring level) added.
