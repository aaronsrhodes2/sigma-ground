# Unification pilots — fluid / thermodynamics / gravity in sigma-ground

**Date:** 2026-10-09/10 · **Status:** ✅ v2 — CORRECTED after adversarial verification (21 verdicts, 3 lenses × 7 pilots, 2026-10-10) ·
**Program:** [UNIFICATION_PROGRAM.md](UNIFICATION_PROGRAM.md) — Stages 1–2 written 2026-10-10; later stages pending · **Scripts:** `scripts/unification/`
**Leaf quantities:** Mentat (PLATINUM §21) via `tools/mentat_call.sh` / `tools/mentat_client.py` / `mcp__mentat__*`; the engines
driven directly are Mentat's own (`dynamics.fields.heat`, `dynamics.continuum`, `field.interface.nbody`). Where a script still
imports a constant directly, the verification confirmed it equals Mentat's (`lookup_constant`, source `sigma_ground.field.constants`).

The Captain's question: are fluid dynamics, thermodynamics and gravitational dynamics one phenomenon? Version 1 of this file
scored seven pilots as "same". The verification kept every number and rejected most of the labels, for one reason stated
seven ways: **a statement that is true for any linear field, any force law, or any choice of constants is not evidence
about gravity.** What follows is the corrected record. Tags: [VERIFIED] established physics reproduced by an engine;
[DERIVED] follows algebraically from stated inputs; [CONSISTENCY] true by construction (not evidence); [RESIDUAL] a measured
gap that is itself a finding; [INCONCLUSIVE] the record holds conflicting measurements or an untested alternative; [LIT] imported from the literature, not tested here; [SPECULATIVE] fenced, never a pass/fail input.

**Could any of these tests have failed?** No. Any static linear field is Poisson's equation (P1, P6, P7); the virial identity holds for any force law (P2b); P3 and P4 are identities evaluated from one constants module; P5 reduces to Q ≈ 1, which the initial conditions set. A test that cannot fail is engine validation, not a test of the thesis — the referee's one-line summary of this round, adopted here as the file's own verdict.

---

## What the verification changed (read this first)

| pilot | v1 claim | what survives | what fell |
|---|---|---|---|
| P1 | steady heat ≡ Newtonian gravity [VERIFIED] | the repo's FVM heat law is a correct Poisson solver (exterior 0.17 % with exact wall data) [VERIFIED numerics] | "same operator" is a property of all static linear fields; with flat walls the error is 20–29 %; interior carries a 1.8–12 % splitting offset; τ is the box mode; "three clocks" is a definition, not a result |
| P2/P2b | one equation confirmed [VERIFIED] | the engine satisfies the Lagrange–Jacobi virial identity: gravity term to 0.4 %, pressure term at x = 0.72–0.81 [RESIDUAL] | the identity holds for ANY forces (a spring and a magnet share it); "converging to the continuum" is an extrapolation a persistent-deficit model fits equally; "surface layer" is wrong — the whole cloud is sub-resolved |
| P3 | exact dictionary c → c_s [DERIVED] | κ = 2c_s³/GM and the Bondi inflow slope −1 [DERIVED]; Hawking's form T = ħκ/2πk_B is universal [LIT] | it is NOT the c → c_s dictionary (r_sonic = r_s(c→c_s)/4); the 8(c_s/c)³ column is an identity by construction; nothing was simulated |
| P4 | complete thermodynamic structure [VERIFIED] | first law + Smarr hold for Mentat's T_H, S_BH formulas [CONSISTENCY] | four rows are two identities; Verlinde closes by convention (N = A/4l_P² gives 4); "thermo" here is horizon thermodynamics, not gas thermodynamics; nothing fluid was tested |
| P5 | C_grav = −C_gas to 3 % [VERIFIED] | the SIGN: E = −K for a bound system [DERIVED]; dT/dE < 0 in every fixed-seed, fixed-R pair; cloud at Q = 0.93 ± 0.06 | the magnitude is the identity C/C_gas = 1 − 2/Q with Q ≈ 1 by construction; the two rows were one run rescaled; "3 %" is seed-specific (−0.96/−1.30/−1.12) |
| P6 | σ sourced by the fluid's thermodynamic state [DERIVED] | the trace R ∝ ρc² − 3P and its sign flip at 3P = ρc² [DERIVED]; the 3P/ρc² magnitude table | "same operator" is automatic for any static scalar; Φ is sourced by ρ + 3P/c² at the same order (Tolman), so the "three sources" sentence was inconsistent; the horizon row is outside validity; □σ = −ξR is a [SPECULATIVE] postulate ("Candidate A") with R undefined in the live repo |
| P7 | three fields of one operator, monopole to 5e-5 [VERIFIED] | the n = 1 polytrope identities and the two-source linear solve [DERIVED]; central σ reduced by 0.609·w_c [DERIVED] | the 5e-5 columns are source quadrature, never touched the solver; no T field was solved; the weak-field truncation drops a metric term (8/3)w_c that is 4.4× the retained pressure term; the w_c = 0.5 row is invalid |

**The lesson for the framework.** "Same operator" (L1), "same equation" (L2) and "same dictionary" (L3) are each true
for reasons that have nothing to do with gravity in particular. Only L4 — whether the fluid's stress-energy *sources* the
geometry the way Einstein's equation says — distinguishes a unification from a shared mathematics, and no repo engine
reaches L4. The differences (causal structure; the sign of E = −K) are real but were also derivations. What the pilots
genuinely established is engine validation: Mentat's heat, SPH and N-body laws reproduce textbook potential theory and
mechanics to stated precision, and seven Mentat defects were found on the way (end of file).

---

## P1 — The FVM heat law is a Poisson solver (steady heat and Newtonian gravity share one elliptic operator) [VERIFIED numerics / CONSISTENCY as a "same" claim]

**Claim tested.** With homogeneous conductivity, steady `k∇²T = −q` and `∇²φ = 4πGρ` are one operator, so for a uniform
sphere of heat source q₀ that is also a uniform sphere of density ρ_m, `T − T₀ = −φ · q₀/(4πGρ_m k)`. The gravitational side
is the **analytic** Newtonian potential (G = 6.6743e-11 m³ kg⁻¹ s⁻², Mentat `lookup_constant`, source
`sigma_ground.field.constants`); no gravity engine is involved. ρ_m, k, ρc_p, q₀ are free scale parameters that cancel.

**Method.** `dynamics.fields.heat.diffuse_fvm` on an n³ grid; volumetric heating by operator splitting at dt_split = 20×
stable_dt; walls Dirichlet-pinned to the exact Newtonian values; comparison in the exterior. Script: `pilot_p1_greens_function.py`.

| grid | exterior max rel. err (as run) | fixed-region ratio | interior max rel. err (splitting) | note |
|---|---|---|---|---|
| 16³ | 4.15e-3 (median 9.6e-4) | — | 12.4 % | 4.5 % at 1× stable_dt |
| 32³ | 1.74e-3 (median 2.9e-4); 1.62e-3 at 1× stable_dt and 12 τ_eff (Stage 3 rerun) | 2.75 vs 16³ (fixed region) | 4.1 % | median falls to 2.2e-4 at 16τ (not time-converged at 8τ) |
| 64³ | 1.51e-3 as run (ratio 1.15 vs 32³ as run: the staircase floor); 6.06e-4 on the fixed region | 2.49 vs 32³ (fixed region) | 1.1 % | fixed-region order ≈ 1.3–1.5; 8τ is 7–25 % short of time convergence |

**Boundary caveat (the number that was missing).** With walls pinned to a constant instead of the exact exterior values the
same solver gives 20–29 % exterior error; with q₀ = 0 and exact walls alone, 47–58 %. The 0.17 % figure therefore measures
the source-driven remainder given exact boundary data on six walls, not a free-space 1/r field found on its own.
**Method correction.** The split scheme's fixed point is the discrete Poisson solution plus an interior offset
((m−1)/2)·dt_sub·q₀/ρc_p (measured 0.488 K vs 0.485 K derived); the exterior is protected by spherical symmetry.

**Transient [DERIVED, textbook].** The approach to the steady field decays as e^(−t/τ_eff) with τ_eff = ((n−1)/n)²·L²/(3π²α),
the box's fundamental Dirichlet mode: 42 % / 14.5 % / 5.1 % / 0.72 % / 0.22 % at t/τ = 1…7 (32³). These percentages depend on
the artificial box, not on the sphere; "parabolic vs elliptic" is the definition of the two equation classes, not a finding.
The hyperbolic case was not run (see the three-clocks experiment in the program).

**Scope line for the ledger.** The identity holds for homogeneous k only: `div(k∇T)` with varying k has no gravitational
counterpart (G is universal; there is no gravitational medium).

**Pitfall, corrected wording.** A *pinned* cell is a Dirichlet condition, not a source; a one-cell *source* on the 7-point
lattice is ~8 % off 1/(4πr) at r = dx (lattice Green's function G(0) = 0.2527/dx). Extended smooth sources only.

---

## P2 / P2b — The virial identity in SPH: gravity term to 0.4 %, pressure term at 72–86 % [DERIVED + RESIDUAL]

**Claim tested.** The Lagrange–Jacobi identity `½ d²I/dt² = 2K + Σ r·F` holds for *any* particle system with *any* forces;
for an isothermal self-gravitating gas its continuum form is `½Ï = 2K + 3∫P dV + U`, and a cold uniform sphere starts to
collapse or expand according to the sign of `3Mc_s² − |U₀|` (f* = 1/√5 = 0.447 with U₀ = −(3/5)GM²/R; sympy-checked).
Measuring it in SPH tests the engine's closure, not a unification.

**Method.** `seed_rotating_cloud` (ω = 0 — jitter_frac is a no-op on velocities at ω = 0, so the start is exactly cold;
P2 and P2b are different position realisations of seed 7), `to_continuum` with `eos="isothermal"`, CODATA G, M = 10¹² kg,
R = 5000 m, Plummer softening 0.02R, `smoothing_h = 1.5·(V/N)^{1/3}`. Mentat leaves: G (`lookup_constant`),
√(GM/R) = 0.115536 m/s (`circular_orbit_velocity`), (3/5)GM²/R = 8.00916e9 J (`gravitational_binding_energy`).

**P2 (r_median after 2 t_ff, N = 64):** a sharp collapse→expansion crossover exists at f ≈ 0.521 (3Mc_s²/|U₀| ≈ 1.33–1.34),
and it does **not** move from N = 64 to 128. The reason is the observable: ∇P = 0 in the uniform interior (free fall) with a
pressure step only at the free surface (expansion), and (R/c_s)/t_ff = 2√2/(πf) ≈ 2 at f*, so r_median at 2 t_ff conflates
the two. The crossover's existence is the only P2 claim.

**P2b (direct measurement of Ï at t → 0).** Define the pressure-virial efficiency x = Σ m r·a_press / (3Mc_s²). Exact t = 0
sums from the engine's own accelerations (verification), and the pilot's quadratic-fit values for comparison:

| N | 2h/R | kernel neighbours | gravity term −Σ m r·a_grav / |U₀| | x exact (seed 7) | x ensemble (6/6/4/2 seeds) | x fit (pilot) |
|---|---|---|---|---|---|---|
| 64 | 1.21 | 43 | 0.9966 | 0.720 | 0.711 ± 0.021 | 0.734 |
| 128 | 0.96 | 56 | 0.9967 | 0.770 | 0.760 ± 0.010 | 0.781 |
| 256 | 0.76 | 67 | 0.9956 | 0.793 | 0.789 ± 0.013 | 0.804 |
| 512 | 0.60 | 76 | 0.9953 | 0.814 | 0.822 (2 seeds) | 0.822 |
| 1024 / 2048 | — | — | — | 0.849 / 0.863 (verification, direct sums) | — | — |

- Gravity term: 0.995–0.997 of the **unsoftened** |U₀| — the Plummer factor r²/(r²+ε²), 0.1 % at ε = 0.0025R. Not "exact".
- x is c_s-independent **by construction** (the isothermal SPH force is linear in c_s² at fixed positions; exact sums agree
  to 1e-15); the 1.2 % drift across f in the fit values is finite-window evolution.
- The deficit 1 − x falls with N at a log-log slope between −0.20 and −0.33 depending on realisation; over N = 64…2048 a
  pure power law with x → 1 (rms 0.0054) and a persistent-deficit model with x_∞ ≈ 0.92 (rms 0.0055) fit equally.
  **"Converging to the continuum" is not shown.** There is no "surface layer": 2h/R = 0.6–1.2, no particle has full kernel
  support at N ≤ 128, and x is non-monotonic in h at fixed N (0.64 / 0.73 / 0.69 / 0.60 at 1.0× / 1.5× / 2.0× / 2.5×).
  The deficit is most plausibly a kernel-resolution artefact of a sub-resolved cloud [INCONCLUSIVE — a persistent closure
  deficit x_∞ ≈ 0.92 fits the same data]; the decisive discriminator is a periodic uniform box with no free surface.
- f* = 0.447/√x matches the r_median crossover at N = 64 only (0.523 vs 0.521); at N = 128 it predicts 0.506 vs 0.527
  observed. "I'' crosses zero at exactly 3Mc_s²·x = |U₀|" was a tautology (it is the definition of x).

**Verdict.** [VERIFIED] the engine's virial sign structure and gravity term; [RESIDUAL] the pressure term at 72–86 % (N = 64–2048), not yet
shown to reach 100 %. As evidence for the thesis: none — the identity is a theorem of mechanics for any force law.
As engine validation: the SPH closure under-delivers pressure in dilute, sub-resolved clouds; a Wendland kernel (not in the
repo — `dynamics/fluid/kernel.py` implements the cubic spline only) or a density-corrected free surface is the lever, and the default packing-radius h in `to_continuum` makes such clouds
pressureless (every run collapsed until h was set by hand).

---

## P3 — Bondi accretion's sonic point as an acoustic horizon [DERIVED; wave-equation statement is LIT]

**What was derived.** From continuity + isothermal Euler, dv/dr = (GM − 2c_s²r)v / [r²(c_s² − v²)], so the sonic point is
r_sonic = GM/2c_s²; Bernoulli is an exact first integral; the dimensionless relation `u²/2 − ln u − 2 ln x − 2/x = −3/2`
reproduces Bondi 1952's γ = 1 eigenvalue λ_c = e^{3/2}/4 = 1.120; the second-order expansion of the saddle (quadratic in
du/dx, not L'Hôpital) gives du/dx = ±1 and the inflow branch is −1 (confirmed independently by a brentq root trace).
Hence κ = |dv/dr| = c_s/r_sonic = 2c_s³/GM (a Killing-vector computation shows κ² = v'² for any conformal factor, which is
what licenses the (t, r) wave-equation form) and T_analog = ħκ/2πk_B — the universal Hawking form [LIT: Unruh 1981;
Visser 1998]. Mentat leaves in the script: T_H = 6.1684e-8 K, r_s = 2954.13 m, S_BH (`gr_basics`); r_sonic, κ and T_analog
are computed in-script from `sigma_ground.field.constants` (equal to Mentat `lookup_constant`). In verification, not in the
script, T_analog was cross-checked via Mentat `unruh_temperature` at a = κc = 4.5166e-3 m/s² → 1.83149e-23 K
(source "sigma-ground (Unruh 1976)").

| c_s (m/s) | r_sonic (m), 1 M☉ | κ (1/s) | T_analog (K) | T_analog/T_H |
|---|---|---|---|---|
| 10³ | 6.64e13 | 1.51e-11 | 1.83e-23 | 2.969e-16 |
| 10⁴ | 6.64e11 | 1.51e-8 | 1.83e-20 | 2.969e-13 |
| 10⁵ | 6.64e9 | 1.51e-5 | 1.83e-17 | 2.969e-10 |

**Correction.** T_analog/T_H = 8(c_s/c)³ is an algebraic identity (same constants file on both sides) — a consistency check,
not evidence. And it says the Bondi horizon is **not** the Schwarzschild horizon under c → c_s: r_sonic = r_s(c→c_s)/4 and
κ·r_h/c_s = 1 for Bondi vs ½ for Schwarzschild. Nothing was simulated: no transonic flow was driven in
`dynamics.continuum`, no pulse launched, no horizon demonstrated. To carry evidential weight this needs the run (program).
Regime assumption, stated: the hydrodynamic limit must hold over ~c_s/κ = r_sonic; ρ_∞ was never fixed.

---

## P4 — Black-hole first law, Smarr, Verlinde chain [CONSISTENCY by construction]

| check | relation | result (script, 2026-10-10 revision) | what it actually tests |
|---|---|---|---|
| first law | dE = T_H dS, E = Mc² | 1 − 9e-16 with the analytic dS/dM = 2S/M (the v1 "1 − 6.9e-11" was central-difference roundoff at h = 1e-6 M) | Mentat's `hawking_temperature` and `bekenstein_hawking_entropy` share one constants set |
| Smarr | Mc² = 2T_H S | 1 + 9e-16 | follows from the first law + S ∝ M² (Euler homogeneity); G, ħ, k_B cancel |
| Verlinde 2011 | F = T·dS/dx with equipartition screen T = 2Mc²/Nk_B, N = A/l_P² | F/F_Newton = 1 − 4e-16 | G enters through l_P² = ħG/c³; with N = A/4l_P² the same chain gives 4 — convention, not precision |
| Unruh vs screen T | Mentat `unruh_temperature` (Unruh 1976) at a = GM/r² vs 2Mc²/Nk_B | 1 + 4e-16 | the same identity as the Verlinde row: F_v/F_N = T_screen/T_Unruh |

**Verdict.** Two independent algebraic facts, both by construction. "Thermo" here is horizon thermodynamics (QFT on a fixed
background), not the gas thermodynamics of P1/P2/P5 — the scoreboard must not conflate them. No engine, no fluid.

---

## P5 — The sign of the heat capacity [DERIVED sign; engine virial check VERIFIED; magnitude CONSISTENCY]

**What is true.** For a bound virialised system E = −K, so with T := 2K/3Nk_B the heat capacity is −(3/2)Nk_B: negative,
with the ideal gas's magnitude. This is the virial theorem plus a definition, [DERIVED]; Lynden-Bell & Wood 1968, Thirring
1970, Antonov are [LIT].

**What the run showed.** `NBodySystem` (N = 64, seed 11, softening 0.02R, 4 t_ff): the virial ratio Q = 2⟨K⟩/|U| = 1.016,
hence C_grav/C_gas = 1 − 2/Q = −0.969 **identically** — the "measurement" reduces to Q ≈ 1. The R = 10 000 m row was the
R = 5000 m run under exact Newtonian similarity (E, T halved to 13 digits), not a second energy. Seeds 12 and 13 give
−1.30 and −1.12 (Q = 0.868, 0.944); the window alone moves it by ±4 %; the cloud is still expanding at 2–4 t_ff
(I grows 1.4–1.8×), so "virialised" was not demonstrated. The v1 attribution "virial excess from softening, squared" was
wrong twice: the deviation is 2(Q−1), first order, and softening lowers Q rather than raising it.
Genuine fixed-R energy sweeps run in verification (initial Q₀ = 0.5–1.5, seed 11) give C/C_gas = −1.24 (all particles) /
−1.11 (bound only) / −1.03 (linear fit over Q₀ = 0.6–1.4), with pairwise finite differences from −0.8 to −4.2 and one
cross-seed pair at +0.55: the sign holds in every non-homologous pair at fixed seed; the magnitude is [INCONCLUSIVE].
Leaf constants k_B = 1.380649e-23 J/K and G = 6.6743e-11 m³ kg⁻¹ s⁻² are imported from `sigma_ground.field.constants` in
the script (verification confirmed them equal to Mentat `lookup_constant`); t_ff is computed in-script (no Mentat tool).

**Verdict.** The engine's 64-body cloud sits at Q = 0.93 ± 0.06 across seeds 11–14 (1.016 / 0.868 / 0.944 / 0.894), with
±5–11 % oscillation inside one run, so its bound systems have negative kinetic heat capacity of magnitude ~(3/2)Nk_B
within that scatter. A measured C needs relaxed equilibria (Plummer/King), ≥ 10 t_ff,
several seeds, energies differing by more than the scatter, and a fit — program item.

---

## P6 — The σ-bridge: the trace ρc² − 3P as the source of σ [DERIVED under a stated reading of a SPECULATIVE postulate]

**The equation and its status.** □σ = −ξR is "Candidate A" in `ssbm-theory/theory/bridge_equations.md`, a postulate not
derived from a Lagrangian; the live repo nowhere defines R or a Riemann sign convention (SSBM theory was excised in
commit e66615b; only `inventory/core/sigma.py` keeps ξ = 0.1582 — whose docstring's Planck numbers actually give 0.1571;
0.1582 is the +BAO column). **Reading adopted here:** R is the Einstein-sourced Ricci scalar, MTW sign (R > 0 for dust),
(−,+,+,+); under the opposite Riemann sign σ → −σ.

**Derivation (sympy block now in the script; all three checks pass).** T = −ρc² + 3P ⇒ R = (8πG/c⁴)(ρc² − 3P) ⇒
□σ = −(8πGξ/c⁴)(ρc² − 3P); static weak field (GM/rc² ≪ 1): ∇²σ = −(8πGξ/c²)(ρ − 3P/c²); point mass σ = ξr_s/r.
**Consistency at the same order:** Φ is sourced by ρ + 3P/c² (Tolman active mass), and the covariant □ on the weak-field
metric carries a (1 + 2Φ/c²) factor — see P7 for why that matters.

| state | ρ (kg/m³) | P (Pa) | 3P/ρc² | source sign | provenance |
|---|---|---|---|---|---|
| air, 20 °C | 1.2041 | 101 325 | 2.8e-12 | attractive | Mentat `density('air')` (CRC), `lookup_constant('atm')` |
| water | 999.97 | 101 325 | 3.4e-15 | attractive | Mentat `density` (CRC) |
| iron | 7874 | 101 325 | 4.3e-16 | attractive | Mentat `density` (CRC) |
| Sun core | 1.5e5 | 2.5e16 | 5.6e-6 | attractive | [LIT, order of magnitude] |
| white-dwarf centre | 1e9 | 1e22 | 3.3e-4 | attractive | [LIT, order of magnitude] |
| neutron-star centre (~2 n₀) | 5e17 | 1e34 | 0.67 | attractive, weakening | [LIT, order of magnitude; trace only — the weak-field operator does not hold here] |
| radiation | — | ρc²/3 | 1 | **zero** (trace-free) | exact |
| critical density (dust) | 8.53e-27 | 0 | 0 | attractive | Mentat `critical_density`; trace/sign entry only — no static solution in FRW |

σ at Earth's surface: 2.20e-10 (field-equation reading) vs `inventory.sigma_from_potential` 1.10e-10 (its docstring quotes
7e-10, which is the bare compactness). The horizon row of v1 is withdrawn: the weak-field solution is invalid at r = r_s
and the exterior vacuum has R = 0. The factor 2 is a **ξ-normalisation ambiguity in an underdefined theory**:
`field/audit.py:143-146` asserts the inventory formula follows from the same equation (equivalent to □σ = −(ξ/2)R, or
R₀₀ mistaken for R). Decision for the Captain, not for this file.

**Verdict.** "Same operator" is automatic for any static linear scalar (TAUTOLOGICAL in the repo's own audit vocabulary).
The genuine content: the source is the stress tensor's trace, vanishes for radiation, flips sign for 3P > ρc², and is
below 10⁻⁵ of ρ for everything short of degenerate matter.

---

## P7 — Two sources through the P1 Poisson solver on an n = 1 polytrope [DERIVED; incomplete at its own order]

**What was computed.** ρ = ρ_c sin(r/a)/(r/a), P = Kρ², w_c = 3P_c/ρ_c c² ∈ {1e-3, 0.1, 0.5}; Φ (source ρ) and Ψ (source
ρ − 3P/c²) solved with the FVM Poisson solver (40³, a = 5 km); no T field was solved. The columns "Q_Ψ/Q_Φ = 1 − w_c/2 to
5e-5" and "r_s/R = 4w_c/3" are quadrature of the source arrays before any solve — consistency by construction (n = 1
identities, sympy-confirmed). The solver-dependent result: Ψ_c/Φ_c = 1 − (I/2)w_c with I = ∫₀^π sin²x/x dx = 1.21883
(Mentat sympy), i.e. central σ reduced by 0.609 w_c (6.1 % at w_c = 0.1), and interior reshaping 1.1e-4 / 1.2 % / 7.4 %
matching the analytic 0.109 w_c; the solver's own error against the exact n = 1 potential is 0.70 % max at the centre,
0.16 % median, 0.04 % exterior. The cause of the +0.45 % centre bias is [INCONCLUSIVE] between lenses (splitting at 20×
stable_dt, measured +2.0 % on a 24³ grid, vs discretisation). Stage 3 rerun at 1× stable_dt on 28³: Ψ_c/Φ_c =
0.99939 / 0.93868 / 0.69339 for w_c = 1e-3 / 0.1 / 0.5 against the closed form 1 − 0.609·w_c = 0.99939 / 0.93906 / 0.6953.

| w_c | ρ_c (kg/m³) | M | r_s/R | σ_c | σ_c, ρ only | status |
|---|---|---|---|---|---|---|
| 1e-3 | 2.9e15 | 7.4 M_Jupiter (1.41e28 kg) | 1.3e-3 | 4.24e-4 | 4.24e-4 | weak field |
| 0.1 | 2.9e17 | 0.71 M☉ | 0.133 | 0.0398 | 0.0424 | marginal |
| 0.5 | 1.4e18 | 3.5 M☉ | 0.667 | 0.147 | 0.212 | **outside validity** of both the weak-field σ equation and the Newtonian polytrope |

**The order problem (found in verification).** The covariant □σ on the weak-field metric reduces to
(1 + 2Φ/c²)∇²σ = −(8πGξ/c²)(ρ − 3P/c²) + …; for any hydrostatic body 3P/ρc² ~ Φ/c² (virial), and on this polytrope the
dropped metric term is (8/3)w_c at the centre — **4.4× the retained pressure term and opposite in sign**. The 6 % (w_c = 0.1)
reshaping figure is therefore the leading term of an expansion whose other same-order terms were dropped; the 31 %
(w_c = 0.5) figure is withdrawn with its row. Either the metric factor is carried
(the same solver can do it by rescaling the source with Φ from the same run) or the quantitative P-term claims are
confined to the exactly trace-free statements.

---

## Diagnostics flagged to the Captain (Mentat flags, never fixes)

1. **SPH has no first law** — `continuum_step` books conduction and the external source only; no PdV, no viscous heating
   (Phase E cloud −387 % "mechanical energy loss"). Tagged `# PHYSICS_GAP`, filed in `KNOWN_GAPS.md`. Fix in the program (S8).
2. **`NBodySystem.total_energy()` ignores softening** — it sums the unsoftened 1/r potential while `step()` integrates
   Plummer-softened forces (`nbody.py` `total_energy` vs `compute_accelerations`); the "0.96 % drift" is the drift of a
   non-conserved quantity (the softened Hamiltonian drifts 2.4e-5). Filed in `KNOWN_GAPS.md` (2026-10-10) with a
   `PHYSICS_GAP` note in the method's docstring.
3. **σ normalisation is ambiguous by a factor 2** — field-equation reading σ = 2ξGM/rc² vs `inventory.sigma_from_potential`
   and `field/audit.py:143-161` (ξGM/rc²), plus a docstring value (7e-10) that is the bare compactness.
4. **ξ = 0.1582 does not match its own docstring** — Ω_b h² = 0.02237, Ω_c h² = 0.1200 give 0.1571; 0.1582 is the Planck +BAO column.
5. **`lookup_constant` cannot resolve the Planck length** ('l_P', 'planck_length' → not found) although `L_PLANCK` exists.
6. **R and the Riemann sign convention are undefined in the live repo**; □σ = −ξR survives only as docstring assertions and as
   "Candidate A" in ssbm-theory (which also proposes a Kretschmann coupling).
7. **Dilute-cloud smoothing length** — `to_continuum`'s packing-radius h leaves a self-gravitating cloud pressureless.

## Scoreboard (corrected)

| pilot | bridge | evidential weight for "one phenomenon" | engine validation value | could it have failed? |
|---|---|---|---|---|
| P1 | heat ↔ gravity | none (any static linear field is Poisson) | FVM is a correct Poisson solver; splitting and boundary caveats quantified | no — any static linear field is Poisson |
| P2/P2b | fluid ↔ thermo ↔ gravity | none (virial identity holds for any forces) | gravity term 0.3–0.5 % (Plummer); pressure closure 72–86 % (N = 64–2048), sub-resolved; free-surface vs estimator cause untested | no — the identity holds for any force law |
| P3 | fluid → gravity | analogy [LIT]; nothing simulated | Bondi slope and surface gravity derived; a transonic run is the real test | no — algebra in one constants module |
| P4 | horizon thermo → gravity | none (identities by construction, convention-dependent) | Mentat BH tools mutually consistent; l_P lookup gap | no — identities by construction |
| P5 | gravity ↔ gas thermo | the SIGN is a real difference [DERIVED; holds in every fixed-R pair]; magnitude [INCONCLUSIVE] | N-body cloud at Q = 0.93 ± 0.06; `total_energy` softening gap filed | no at t = 0 (Q ≈ 1 set by the ICs); the fixed-R sweeps could, and gave the sign |
| P6 | σ ↔ fluid state | conditional on a [SPECULATIVE] postulate; the trace's sign structure is real | factor-2 and ξ inconsistencies found | no — automatic for any static linear scalar |
| P7 | σ polytrope | incomplete at its own order; weak-field rows only | solver error 0.7 % (centre-bias cause [INCONCLUSIVE]); metric term must be carried | no — quadrature columns; the solver rows are superposition |

> **[SPECULATIVE — the Captain's wider thesis, filed at his request, never a pass/fail input]:** that this one force also
> underlies black holes, dark energy, dark matter and holography; that light's speed is not fundamentally finite but
> medium-set (note: `relativity.effective_liv_scale_gev` is a [VERIFIED] falsification gate against variable c in this
> repo); that decoherence/coherence of matter are governed by the same force; that it lives in a 4-dimensional fabric
> string theory struggles to express.
