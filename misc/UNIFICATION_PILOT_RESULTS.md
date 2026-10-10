# Unification pilots — fluid / thermodynamics / gravity in sigma-ground

**Date:** 2026-10-09/10 · **Status:** 🔄 PILOT RESULTS, pending adversarial verification (workflow 2) ·
**Program:** [UNIFICATION_PROGRAM.md](UNIFICATION_PROGRAM.md) · **Scripts:** `scripts/unification/`
**Leaf quantities:** from Mentat (PLATINUM §21) via `tools/mentat_call.sh` / `tools/mentat_client.py`; the engines driven
directly are Mentat's own (`dynamics.fields.heat`, `dynamics.continuum`, `field.interface.nbody`).

The Captain's question: are fluid dynamics, thermodynamics and gravitational dynamics one phenomenon? The pilots below
test the sharpest *mathematical* statements of "same" and "different" that the repo's engines can reach today.
Each reports what the code actually did. Tags: [VERIFIED] established physics reproduced; [DERIVED] follows from
repo constants; [RESIDUAL] a measured gap that is itself a finding; [SPECULATIVE] fenced off, never a pass/fail input.

---

## P1 — Same operator: steady heat ≡ Newtonian gravity (Green's-function identity) [VERIFIED]

**Claim tested.** The steady Fourier equation `k∇²T = −q` and Poisson's equation `∇²φ = 4πGρ` are the same
elliptic operator with the same Green's function (−1/4πr). For a uniform sphere of heat source `q₀` that is also a
uniform sphere of mass density `ρ_m`, the exact map is `T − T₀ = −φ · q₀ / (4πG ρ_m k)`.

**Method.** `dynamics.fields.heat.diffuse_fvm` (the repo's conservative face-flux law) on an n³ grid; volumetric
heating by operator splitting (`T[inside] += q₀Δt/ρc_p` between calls — its fixed point *is* the discrete
Poisson equation); walls Dirichlet-pinned to the exact Newtonian values (a harmonic function is fixed by its boundary
data); compare in the exterior `r > R + 1.5dx`. Script: `pilot_p1_greens_function.py`.

| grid | dx (m) | sphere R | exterior max rel. error | median | wall |
|---|---|---|---|---|---|
| 16³ | 0.0175 | 3 dx | 4.15e-3 | 9.6e-4 | 0.1 s |
| 32³ | 0.00875 | 6 dx | 1.74e-3 | 2.9e-4 | 3.3 s |

Error falls with resolution (ratio 2.4; the staircase sphere boundary limits it below the clean 2nd-order 4.0).
**Verdict:** the repo's heat law relaxes to the repo's gravitational potential. Same operator. [VERIFIED]

**The difference, measured in the same run [RESIDUAL]:** the heat field takes time to *reach* the Poisson
solution — max exterior deviation vs t/τ (τ = L²/3π²α), 32³ grid: 1τ → 42 %, 2τ → 14.5 %, 3τ → 5.1 %,
5τ → 0.72 %, 7τ → 0.22 %. The Newtonian potential is the Poisson solution *instantly* (elliptic), and sound
would carry the change at c_s (hyperbolic). One operator in space, three time structures. *This is the first
concrete "same/different" split: identical spatial law, different causal law.*

**Pitfall found and fixed:** a single pinned hot cell is not a point mass on a lattice (the discrete lattice Green's
function at the origin is not 1/r at an off-centre sample point): a first attempt showed 20 K errors on a 33 K
field. Extended smooth sources only.

---

## P2 — The locus where they are one equation: Jeans / virial threshold in SPH [RESIDUAL]

**Claim tested.** In a self-gravitating isothermal gas the virial identity `½ d²I/dt² = 2K + 3∫P dV + U` makes
pressure (thermo), motion (fluid) and gravity literally three terms of one equation. For a cold uniform sphere
(K = 0, P = ρc_s², U₀ = −(3/5)GM²/R) the initial motion is set by the sign of `3Mc_s² − |U₀|`:
collapse below `c_s* = √(GM/5R)` (f* = c_s/√(GM/R) = 1/√5 = 0.447), expansion above.

**Method.** `dynamics.seeding.seed_rotating_cloud` (ω = 0, 2 % jitter, seed 7) → `to_continuum` with
`eos="isothermal"`, `self_gravity_G` = CODATA G, N = 64, M = 10¹² kg, R = 5000 m, Plummer softening 0.02R,
dt = t_ff/400, run 2 t_ff; diagnostics from `dynamics.phase_e.diagnostics`. Script: `pilot_p2_jeans_virial.py`.

| f = c_s/√(GM/R) | 3Mc_s²/|U₀| (analytic) | r_median(2t_ff)/r_median(0) | r_max ratio | 2K/|U| at end |
|---|---|---|---|---|
| 0.15 | 0.11 | 0.21 | 1.40 | 0.89 |
| 0.30 | 0.45 | 0.17 | 1.23 | 0.85 |
| 0.40 | 0.80 | 0.17 | 0.90 | 0.79 |
| 0.45 | 1.01 | 0.26 | 1.18 | 0.62 |
| 0.50 | 1.25 | 0.78 | 1.42 | 0.50 |
| 0.55 | 1.51 | **1.31** | 1.74 | 0.49 |
| 0.60 | 1.80 | 1.81 | 2.08 | 0.77 |
| 0.65 | 2.11 | 2.40 | 2.41 | 1.32 |
| 0.70 | 2.45 | 2.84 | 2.77 | 2.13 |
| 1.00 | 5.00 | 4.71 | 4.72 | 10.6 |

**Verdict:** a sharp collapse→support transition exists and sits at **f ≈ 0.52** vs the continuum prediction
**0.447** (+16 % in c_s, +35 % in thermal energy). The crossover is the physics; the offset needs explaining.

**Convergence study (discrete |U₀| measured from the realisation, not assumed):**

| N | |U₀|_discrete / (3/5)GM²/R | f = 0.45: 3Mc_s²/|U₀|, r_med ratio | f = 0.50 | f = 0.55 | neighbours |
|---|---|---|---|---|---|
| 64 | 1.018 | 0.995, 0.26 | 1.228, 0.78 | 1.486, 1.31 | 43 |
| 128 | 1.038 | 0.976, 0.19 | 1.205, 0.74 | 1.458, 1.23 | 61 |

The r_median crossover stays at 3Mc_s²/|U₀| ≈ 1.33–1.35 from N = 64 to 128 — it is **not** moving toward 1.0 with
resolution. The reason is the observable, not the physics: at f = 0.45 the core collapses (r_med 0.26) while the
edge expands (r_max 1.18–1.28) — a uniform sphere at exact virial balance is not in *local* equilibrium (its
surface has pressure gradient but little enclosed mass). The virial identity is a statement about
`d²I/dt²` of the *whole* cloud at t = 0, and that is what must be measured. The r_median version stays in the
record as the lesson.

### P2b — the virial identity measured directly [VERIFIED + RESIDUAL]

`½ d²I/dt² = 2K + 3∫P dV + U` at t → 0 for a cold cloud (K = 0): I(t) = Σ m r² over the first 0.05 t_ff
(100 steps, dt = t_ff/2000), quadratic fit (residual ≤ 2e-3 of the I range except at the exact threshold),
|U₀| measured from the realisation. Script: `pilot_p2b_virial_curvature.py`. Define the **SPH pressure-virial
efficiency** x = (1 + I''/2|U₀|) / (3Mc_s²/|U₀|), i.e. the fraction of the continuum pressure virial 3∫P dV
the discrete engine actually delivers (x = 1 is the continuum identity).

| N | kernel neighbours | x at f = 0.40 | x at f = 0.45 | x at f = 0.50 | x at f = 0.55 | 1 − x | N^(−1/3) |
|---|---|---|---|---|---|---|---|
| 64 | 43 | 0.731 | 0.731 | 0.728 | 0.726 | 0.27 | 0.25 |
| 128 | 61 | 0.781 | — | 0.776 | — | 0.22 | 0.20 |
| 256 | ~85 | 0.804 | — | 0.800 (I'' ≈ 0: at threshold) | — | 0.20 | 0.16 |
| 512 | — | 0.822 (60 steps, fit resid 8e-5) | — | — | — | 0.18 | 0.13 |

**Verdict:** the gravity term of the virial identity is delivered exactly (the cold limit I'' → 2U holds), and
the pressure term is delivered at a fraction x that is independent of c_s to 0.5 % and rises monotonically
with resolution (0.73 → 0.78 → 0.80 → 0.82). The deficit 1 − x falls as ≈ N^(−0.20) (log-log slope over
N = 64…512), slower than the naive free-surface estimate N^(−1/3) — the kernel-smoothed density under-counts
in a surface layer ~2h thick, and the SPH pressure-gradient estimator has its own O(h) edge error on top.
With x in hand every earlier number falls into place: the r_median threshold f* = 0.447/√x = 0.52 (N = 64),
and at N = 256 the measured I'' crosses zero at exactly 3Mc_s²·x = |U₀|.
**Same equation, confirmed; the 18–27 % residual is the discrete closure's surface term, converging toward
the continuum, not a new physics term.** Open point: the exponent (−0.20 measured) — a Wendland kernel or a
density-corrected free-surface treatment would be the next lever, not more particles.

**Limit found [RESIDUAL]:** with the adapter's default smoothing length (derived from the material packing radius)
every run collapsed regardless of c_s — the parcels never saw each other's pressure until already fallen in. A
dilute self-gravitating cloud needs `smoothing_h ≈ 1.5·(V/N)^{1/3}` (~43 neighbours here). This is the Phase E
"KNOWN LIMIT" biting; it should become a guard or a documented parameter in `to_continuum`.

---

## P3 — A fluid that makes a horizon: Bondi accretion as an acoustic black hole [DERIVED]

**Claim tested.** Bondi (1952) isothermal accretion onto mass M is one exact solution containing gravity
(GM/r), fluid flow (continuity + Euler) and thermodynamics (P = ρc_s²). Its sonic point is an acoustic horizon
(Unruh 1981; Visser 1998): sound inside cannot escape. The analog surface gravity and Hawking temperature should
mirror the Schwarzschild ones with c → c_s.

**Derivation (sympy, in the script).** Dimensionless Bondi relation `u²/2 − ln u − 2 ln x − 2/x = −3/2`
(u = v/c_s, x = r/r_sonic, r_sonic = GM/2c_s²) is 0/0 at the sonic point; L'Hôpital gives the two branches
`du/dx = ±1`; the inflow branch is −1. Hence `κ = |d(c_s − v)/dr| = c_s/r_sonic = 2c_s³/GM` and
`T_analog = ħκ/2πk_B`. With Schwarzschild `κ = c³/4GM`, `T_H = ħc³/8πGMk_B` (Mentat `hawking_temperature`):

    T_analog / T_Hawking = 8 (c_s / c)³    — exactly, for the same M.

| c_s (m/s) | r_sonic (m), M = 1 M☉ | κ (1/s) | T_analog (K) | T_analog/T_H | 8(c_s/c)³ |
|---|---|---|---|---|---|
| 10³ | 6.64e13 | 1.51e-11 | 1.83e-23 | 2.969121e-16 | 2.969121e-16 |
| 10⁴ | 6.64e11 | 1.51e-8 | 1.83e-20 | 2.969121e-13 | 2.969121e-13 |
| 10⁵ | 6.64e9 | 1.51e-5 | 1.83e-17 | 2.969121e-10 | 2.969121e-10 |

Mentat leaf values (M = 1.989e30 kg): r_s = 2954.13 m, T_H = 6.168e-8 K — `sigma_ground.field.gr_basics (Hawking 1974)`.

**Verdict:** the dictionary {c → c_s, Schwarzschild horizon → sonic surface} is exact at the level of the wave
equation (kinematics). **Honest limit:** analog gravity reproduces the *kinematics* of fields on a curved metric,
not Einstein's *dynamics* — the acoustic metric is not sourced by the fluid's stress-energy through G.
That is precisely the gap a "same force" claim has to cross, and this pilot measures where the identity stops.

---

## P4 — Gravity's own thermodynamics: first law, Smarr, entropic chain [VERIFIED / consistency]

| check | relation | result (repo constants; T_H, S_BH from Mentat) |
|---|---|---|
| first law of BH mechanics | `dE = T_H dS_BH`, E = Mc² | T from numerical dS/dM ÷ T_H = 1 − 6.9e-11 |
| Smarr (Schwarzschild) | `Mc² = 2 T_H S_BH` | ratio 1 + 9e-16 |
| Verlinde 2011 entropic force | `F = T ΔS/Δx`, Unruh T, holographic N = A/l_P² | F/F_Newton = 1 − 4e-16 |
| Unruh vs screen temperature | `ħa/2πck_B` vs equipartition `2Mc²/Nk_B` | equal to 4e-16 |

**Verdict:** gravity carries a complete thermodynamic structure (S ∝ area, T ∝ surface gravity, a first law, a
Smarr relation) and Newton's law follows from entropy + equipartition. **Honest limit:** these are identities
*by construction* (`l_P² = ħG/c³`); they prove the dictionary is self-consistent, not that gravity *is*
entropic — Verlinde's proposal has live critiques (Kobakhidze 2011 neutron-bounce argument; Visser 2011).
They belong in the program as the consistency floor, not as evidence.

---

## P5 — Where gravity and heat part ways: the sign of the heat capacity [VERIFIED]

**Claim tested.** A virialised self-gravitating system has `E = −K`, so its kinetic "temperature" *rises* when
energy is removed: negative heat capacity (Lynden-Bell & Wood 1968; Thirring 1970). An ideal gas has
`C_V = (3/2)Nk_B > 0` (`statistical.heat_capacity_equipartition`). Same magnitude, opposite sign, is the
prediction: **C_grav = −(3/2)Nk_B.**

**Method.** `field.interface.nbody.NBodySystem` (velocity-Verlet), N = 64, M = 10¹² kg, two uniform spheres
R = 5000 m and 10 000 m with isotropic velocities scaled to 2K = |U|, softening 0.02R, dt = t_ff/300, 4 t_ff,
⟨K⟩ over the last 2 t_ff; T_eff = (2/3)⟨K⟩/Nk_B. Script: `pilot_p5_negative_heat_capacity.py`.

| R (m) | E (J) | 2⟨K⟩/|U| | energy drift | T_eff (K) |
|---|---|---|---|---|
| 5 000 | −3.857e9 | 1.016 | 0.96 % | 3.00e30 |
| 10 000 | −1.929e9 | 1.016 | 0.96 % | 1.50e30 |

    dT/dE = −7.79e20 K/J  →  C_grav = −1.284e-21 J/K ;  ideal gas (3/2)·64·k_B = +1.325e-21 J/K ;  ratio −0.969

**Verdict:** the self-gravitating system is the ideal gas with its heat capacity sign flipped, to 3 % (the
residual is the 1.6 % virial excess from softening, squared). This is the cleanest *difference*: gravity's
thermodynamics is non-extensive, has no stable maximum-entropy state (Antonov), and runs "backwards". Any
"one phenomenon" theory must *produce* this sign flip, not explain it away.

---

## P6 — The σ-bridge: the repo's own field equation is sourced by the fluid's thermodynamic state [DERIVED]

**Claim tested.** The repo's □σ = −ξR, with a perfect fluid on the right-hand side, ties σ directly to the
fluid's (ρ, P). Trace in (−,+,+,+): `T = −ρc² + 3P` ⇒ `R = (8πG/c⁴)(ρc² − 3P)` ⇒
`□σ = −(8πGξ/c⁴)(ρc² − 3P)`. Static weak field: `∇²σ = −(8πGξ/c²)(ρ − 3P/c²)` — **the same elliptic
operator as P1's heat/gravity pair**, with the fluid's trace as the source; σ = −(2ξ/c²)Φ + σ_P, and for a point
mass σ = ξ·r_s/r. (Derivation checked symbolically by the math agent, sympy checks (b), (e).) Script:
`pilot_p6_sigma_bridge.py`; leaf values from Mentat; ξ = 0.1582 from `inventory.core.sigma` (the only surviving copy).

| state | ρ (kg/m³) | P (Pa) | 3P/ρc² | σ-source sign | provenance |
|---|---|---|---|---|---|
| air, 288 K, 1 atm | 1.225 | 1.013e5 | 2.8e-12 | attractive (ρ-like) | Mentat ideal_gas_pressure (CODATA R) |
| water, 1 atm | 999.97 | 1.013e5 | 3.4e-15 | attractive | Mentat density (CRC) |
| iron, 1 atm | 7874 | 1.013e5 | 4.3e-16 | attractive | Mentat density (CRC) |
| Sun core | 1.5e5 | 2.5e16 | 5.6e-6 | attractive | [LIT] Bahcall 2005, not a Mentat tool |
| white-dwarf centre | 1e9 | 1e22 | 3.3e-4 | attractive | [LIT] order of magnitude |
| neutron-star centre | 5e17 | 1e34 | 0.67 | attractive, weakening | [LIT] ~3n₀ stiff EOS, order of magnitude |
| radiation | — | ρc²/3 | 1 | **zero** (trace-free) | exact |
| cosmic critical density, dust | 8.53e-27 | 0 | 0 | attractive | Mentat critical_density (Planck 2018) |

| σ | field equation (ξ r_s/r) | `inventory.sigma_from_potential` | its docstring |
|---|---|---|---|
| Earth surface | 2.20e-10 | 1.10e-10 | 7e-10 |
| Sun surface | 6.7e-7 | — | — |
| at the horizon | ξ = 0.1582 | ξ/2 = 0.0791 (clamped) | — |

**Verdict:** in the repo's own theory the σ field and the Newtonian potential and the steady temperature field
are three solutions of one operator with three sources (ρ − 3P/c², ρ, q/k). The Captain's "pressure layers"
enter the σ equation only through 3P/ρc², which is below 10⁻⁵ for everything short of degenerate matter and
reaches O(1) only in neutron-star cores; for radiation the source vanishes exactly. **Diagnostic:** the
field-equation σ is exactly 2× the inventory ansatz everywhere (and 7× below the ansatz's own docstring).

## P7 — Three fields of one operator on a Lane–Emden n = 1 polytrope [VERIFIED + DERIVED]

**Claim tested.** The σ-bridge says Φ, σ and T_steady are three solutions of one operator with three sources.
Make the pressure term *matter*: an n = 1 polytrope (ρ = ρ_c sin(r/a)/(r/a), P = Kρ², R = πa) with central
w_c = 3P_c/ρ_c c² chosen at 10⁻³, 0.1, 0.5. Exact n = 1 predictions: ∫P dV = ½Kρ_c M ⇒ **Q_Ψ/Q_Φ = 1 − w_c/2**
(monopole of the σ-source vs the mass), and compactness r_s/R = (4/3)w_c. The repo's FVM heat law is the
Poisson solver (P1); ξ from `inventory.core.sigma`. Script: `pilot_p7_sigma_three_fields.py` (40³ grid, a = 5 km).

| w_c | ρ_c (kg/m³) | M | r_s/R (pred. 4w_c/3) | Q_Ψ/Q_Φ (pred. 1 − w_c/2) | interior reshaping by P | σ_centre | σ_centre, ρ only | σ_surface = ξ r_s/R |
|---|---|---|---|---|---|---|---|---|
| 10⁻³ | 2.9e15 | 7 M_Jupiter·10³ (1.4e28 kg) | 1.333e-3 (1.333e-3) | 0.99950 (0.9995) | 1.1e-4 | 4.24e-4 | 4.24e-4 | 2.1e-4 |
| 0.1 | 2.9e17 | 0.71 M☉ | 0.1333 (0.1333) | 0.94999 (0.95) | 1.2 % | 0.0398 | 0.0424 | 0.0211 |
| 0.5 | 1.4e18 | 3.5 M☉ | 0.667 (0.667) | 0.74998 (0.75) | 7.4 % | 0.147 | 0.212 | 0.105 |

**Verdict:** the operator identity holds with the pressure term switched on (monopole to 5e-5, shape to
discretization), and the thermodynamic state reshapes σ — the P term is more centrally concentrated than ρ, so a
stiff core lowers central σ by 6 % (w_c = 0.1) to 31 % (w_c = 0.5) relative to the ρ-only field. The w_c = 0.5 row
is beyond the weak field (r_s/R = 0.67) and stands as an operator test only; w_c = 0.1 is marginal. **This is the
σ-bridge framework in one picture: gravity (G, Φ), thermodynamics (P(ρ) through the trace) and the field all sit
in ∇²Ψ = 4πG(ρ − 3P/c²); fluid dynamics enters when the static limit is lifted (□ instead of ∇²).** What it does
NOT claim: that e^σ ≈ 1.04–1.16 inside a neutron star is observed — that is the next, data-anchored question
(nuclear masses, Debye temperatures, cooling curves through `field/interface`'s `sigma=` hooks).

## Diagnostics flagged to the Captain (Mentat flags, never fixes)

1. **SPH has no first law** — `dynamics/continuum.py::continuum_step` step 9 books conduction and the external
   heat source only; the docstring's adiabatic `(P/ρ²)(dρ/dt)/cp` term is not implemented and viscous dissipation
   never becomes heat (Phase E cloud: −387 % "mechanical energy loss"). Tagged `# PHYSICS_GAP` in the code and
   filed in `KNOWN_GAPS.md` (2026-10-10). The fluid→thermo entropy-production test waits on it.
2. **σ-from-potential is off by 2 from the field equation.** `inventory/core/sigma.py::sigma_from_potential`
   uses σ = ξ·GM/(rc²) (and its docstring quotes Earth-surface σ ≈ 7e-10, which is the compactness *without* ξ;
   the function returns 1.1e-10). Inserting the perfect-fluid trace into the repo's own □σ = −ξR gives, in the
   static weak field, ∇²σ = −(8πGξ/c²)(ρ − 3P/c²) ⇒ σ = 2ξGM/(rc²) = ξ r_s/r for a point mass (2.2e-10 at
   Earth's surface). Three numbers for one quantity; the field equation's is the derived one. Decision needed.
3. **SSBM constants are quarantined.** `XI`, `SIGMA_CONV`, `field/bounds.py` were excised from sigma-ground in
   commit e66615b (2026-06-03); only `inventory/core/sigma.py` keeps ξ = 0.1582. Any σ ≠ 0 experiment in this
   program imports from there and says so; where SSBM theory should live is the Captain's call, not the program's.
4. **Dilute-cloud smoothing length.** `seeding.to_continuum` derives h from the material packing radius; a
   self-gravitating cloud needs h ≈ 1.5·(V/N)^{1/3} or it is pressureless until collapsed (P2). Candidate guard.

## Mentat leaf values used (PLATINUM §21)

| quantity | value | Mentat source |
|---|---|---|
| Hubble radius | 1.3726e26 m | sigma-ground (Planck 2018 H₀) |
| MOND a₀ | 1.2e-10 m/s² | sigma-ground (Milgrom 1983; SPARC 2016) |
| c_s, air, 288.15 K | 340.09 m/s | sigma-ground (ideal-gas adiabatic sound speed) |
| Grashof, 350 K plate in air, 1 cm | 4710.6 | sigma_ground.field.interface.gas |
| S_BH / k_B, 1 M☉ | 1.0495e77 | sigma-ground (Bekenstein 1973 / Hawking 1974) |

## Scoreboard so far

| pilot | bridge | same? | different? |
|---|---|---|---|
| P1 | heat ↔ gravity | **same spatial operator** (Green's function identity, 0.17 %) | causal structure: parabolic vs elliptic |
| P2 / P2b | fluid ↔ thermo ↔ gravity | **one equation** (virial identity: gravity term exact, pressure term delivered at x = 0.73→0.80 for N = 64→256) | residual is the discrete closure's free-surface term, converging ~N^(−1/3); not new physics |
| P3 | fluid → gravity | **exact kinematic dictionary** (8(c_s/c)³) | no dynamics: acoustic metric is not sourced by G |
| P4 | thermo → gravity | **complete thermodynamic structure** | identities by construction, not evidence |
| P5 | gravity ↔ thermo | same magnitude (3/2)Nk_B | **opposite sign** — the decisive difference |
| P6 | σ ↔ fluid state | **same operator again** (∇²σ with source ρ − 3P/c²) | pressure matters only where 3P/ρc² ~ 1 (neutron stars); inventory σ ansatz is 2× off the field equation |
| P7 | σ ↔ Φ ↔ T, polytrope | **three fields, one operator**, monopole 1 − w_c/2 to 5e-5 | P reshapes σ 1–7 % only in degenerate cores |

> **[SPECULATIVE — the Captain's wider thesis, filed at his request, never a pass/fail input]:** that this one
> force also underlies black holes, dark energy, dark matter and holography; that light's speed is not
> fundamentally finite but medium-set; that decoherence/coherence of matter are governed by the same force;
> that it lives in a 4-dimensional fabric string theory struggles to express. The program's data-anchored
> follow-ups (a₀ = cH₀/2π vs SPARC; η/s; analog-Hawking spectra) are where any of this could first touch data.
