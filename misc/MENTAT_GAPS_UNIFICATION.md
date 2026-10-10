# Mentat gap ledger — found by the unification program (2026-10-09/10)

Every defect, missing capability, or open decision the program met, with the evidence and the fix direction.
Rule in force: **Mentat flags, never fixes** — except where the Captain approved the fix in the plan (S8, the SPH first law).
Code-level gaps carry a `PHYSICS_GAP` note at the point of omission and an entry in [KNOWN_GAPS.md](../KNOWN_GAPS.md).
Program: [UNIFICATION_PROGRAM.md](UNIFICATION_PROGRAM.md).

## A. Engine defects

| # | where | defect | evidence | status / fix direction |
|---|---|---|---|---|
| A1 | `dynamics/continuum.py::continuum_step` | SPH bridge had no first law: no PdV work, no viscous heating | Phase E cloud "−387 % mechanical energy" | **CLOSED** (S8): exact pairwise energy booking, `eos="ideal"`, `energy_ledger()`; 10 sentinel tests |
| A2 | `field/interface/nbody.py::NBodySystem.total_energy` | sums the unsoftened potential while the integrator uses Plummer-softened forces | P5: "0.96 % drift" vs 2.4 × 10⁻⁵ for the softened Hamiltonian | filed; add a `softened=True` path; Captain picks the API default |
| A3 | `field/interface/viscoelasticity.py::_relaxation_ratio` | reads `MATERIALS[m]['melting_point_K']`, a key that does not exist → silent 1800 K for every metal | S7: Al at 900 K gives E_R/E_U 0.825 vs ≈ 0.35 with its CRC T_melt 933 K | filed; source T_melt from the CRC table the `melting_point` tool uses |
| A4 | same module | relaxation cap `_RELAX_RATIO_MAX = 0.7` is an uncited heuristic; docstring §5 gives Maxwell moduli while the code is the standard linear solid | S7 code read | filed with A3 |
| A5 | `field/interface/fluid.py::eyring_viscosity` | liquid-metal viscosity 10³–10⁶× too high (Al 1170 Pa·s, Cu 10.8, Fe 4.06 at T_melt vs a few mPa·s measured); docstring claims ±30 % | S9b | filed; re-derive the flow activation energy or use the Assael 2006 correlations with a citation |
| A6 | `dynamics/seeding.py::to_continuum` | default smoothing length comes from the material packing radius → a dilute self-gravitating cloud is pressureless until it has collapsed | P2: every run collapsed regardless of c_s until h was set by hand | candidate guard: derive h from the inter-particle spacing when the cloud is dilute |
| A7 | SPH density estimator (cubic spline, self-term) | pressure virial delivered at x → 1/(1 + 32/(3N_nb)) = 0.914 at h = 1.5 × spacing; the self-contribution m W(0) is in ρ but not in the gradient | S4: exact identity x = Σ m D/ρ / 3M to 10⁻¹⁶; prediction confirmed to 1.4 % | engine fact, documented; Wendland kernel at larger N_nb or the Dehnen & Aly 2012 self-contribution correction — Captain's call |
| A8 | `inventory` (Quarksum) `quark_chain` | mass closure misses by 0.6–1.4 % for the gold ring, bronze cube and seawater (others ≤ 0.15 %) | S10 | open: inspect alloy/solution composition resolution |

### A, round 2 (2026-10-10): found by the verified kill-test battery and its critic (program §15)

All flagged, none fixed. Full text in [KNOWN_GAPS.md](../KNOWN_GAPS.md); evidence in `misc/unification/round2/<item>/`.

| # | where | defect | found by | severity |
|---|---|---|---|---|
| A9 | `nbody.py` `gr_1pn` | whole 1PN correction has the wrong sign: Mercury −42.997″/cy | U9, U5; three independent integrators | ✅ **FIXED** 2026-10-10 (sign; +42.995″/cy; `test_gr_precession.py`) |
| A10 | `nbody.py` `eih_cross` | velocity cross-term sign reversed: Mercury −71.66″/cy, frame-dependent precession | U9, U5 | ✅ **FIXED** 2026-10-10 (cross-term; frame-invariant +42.995″/cy) |
| A11 | `nbody.py` `_apply_gw_damping` | Peters power with r⁻⁴ (W·m), r[m] times too strong; sub-ulp losses round to zero; recompute adds spurious decay | U1 | **high** for compact binaries |
| A12 | `nbody.py` `forest_ruth_step` | 1st order, not 4th, with velocity-dependent forces | U9 | medium |
| A13 | GR/GW tests | assert only "nonzero" or "decreasing"; neither sign nor size tested | U9, U1 | **high** (why A9–A11 survived) |
| A14 | `continuum.py` `_buoyancy_force` | linear α_L where 3α_L is needed (1/3 of Archimedes); double-counted under `eos="ideal"` | U14G | medium |
| A15 | `continuum.py` `_heat_conduction` | divides by ρ twice: ρ_sph[SI]× too slow, ρ⁻² scaling, not antisymmetric (creates/destroys heat) | U14G | **high** for any conduction run |
| A16 | `continuum.py` `cfl_timestep` | uses the cascade c_L (with shear) even for isothermal/ideal EOS; no force criterion | U11, U13, U7 | medium |
| A17 | `continuum.py` ideal EOS | pre-update T in both force evaluations: first order, not reversible | U2 | medium |
| A18 | `continuum.py` floor contact; `fluid/kernel.py` | rigid-body resolver on fluid (never settles, coincident parcels); grad_W = 0 below L_Planck | U11 | medium |
| A19 | `element.py` SEMF | asymmetry 4× too small (rms 6.8 % → 0.8 %); second SEMF copy in `mcp/tools/nuclear.py` | N1 | **high** for nuclear work |
| A20 | `mcp/tools/symbolic.py` | variables named E, I, N, S, O, Q parsed as sympy constants: silent wrong answers | U1 | **high** under §21 |
| A21 | `mcp/tools/materials_strength.py` | `elastic_analysis` returns fixed moduli for every material | U8 | medium |
| A22 | `mcp/tools/astronomy.py` | Sun mass = IAU GM / obsolete G: GM 2.57 × 10⁻⁴ high; a second M☉ in `constants.py` | U9, J6 | low |
| A23 | σ-mass laws (many modules) | five incompatible conventions; thermal chain freezes ρ, so Θ_D(σ) has the wrong sign | U12 | decision C2, widened |
| A24 | `gr_basics.gravitational_redshift` | catastrophic cancellation in a weak field | U4 | low |
| A25 | `lindemann_melting_estimate` | water ice melts at 92 K; no liquid water in `MATERIALS` | U2, U7 | medium |
| A26 | `inventory/data/isotopes.json` | 55 stable isotopes with zero abundance and placeholder spins | T5 | low |
| A27 | `inventory/core/sigma.py` | Coulomb energy σ-invariant (a model choice carrying ~all of Δf); docstring σ values and ξ provenance | T5, U4, U5 | decision |
| A28 | SkippyTel `mentat/app.py` (other repo) | gateway refuses `y''` and some ordinary power expressions | N1, U3 | recorded only |
| A29 | `adapters/base.py`, `adapters/integrator.py` | `include_gr=True` by default: orbit propagation silently uses the reversed 1PN (A9) | U5 recheck, critic | ✅ **FIXED** 2026-10-10 (defaults now `include_gr=False`) |
| A30 | `mcp/tools/mathx.py` | `compute_limit`, `series_expansion`, `summation`: same E/I/N/S/O/Q parsing defect as A20 | critic; reproduced live | **high** under §21 |
| A31 | cascade material data | iron κ 108 vs ~80 W/m/K (dκ/dT wrong sign); glass κ 78; T_melt rubber 1.9 × 10⁻⁴ K, ABS 2.6 K; ideal-EOS conduction uses the solid's c_p | U14G, U2 | medium |
| A32 | provenance and docstrings | audit.py claims a derivation C2 contradicts and a TOV that is not here; two unreconciled clock laws; Hawking year; label and hard-coded-c items; contradicting docstring numbers; dead GOLDEN_RULES pointer | U4, J6, U12, N1 | low |
| A33 | research scripts | P6 dense-matter rows fail GW170817 and Chandrasekhar; P7 `poisson_fvm` +0.63 % split bias | N1, U10 | low (scripts) |

## B. Missing capabilities (the program needed them and Mentat does not have them)

| # | capability | why the program needs it | data anchors when built |
|---|---|---|---|
| B1 | **Nuclear / degenerate matter**: electron and neutron Fermi-gas EOS (both regimes), semi-empirical mass formula, saturation and symmetry energy, Tolman–Oppenheimer–Volkoff integrator, composition layers, bag-model deconfinement pressure | the σ-bridge's pressure term 3P/ρc² is only O(1) in degenerate matter; P6/P7 had to take white-dwarf and neutron-star rows from the literature | Chandrasekhar mass ≈ 1.4 M☉; NICER radii; ≥ 2 M☉ pulsars — **separate plan (Captain's decision)** |
| B2 | real-gas EOS (van der Waals is named in a note, never implemented), Joule–Thomson, free-expansion entropy, compressible-flow relations (nozzles, shocks, Mach) | gas expansion beyond the ideal gas | NIST real-gas data |
| B3 | continuum solid solver (FEM / MPM), deviatoric stress in SPH, crack propagation | solids as a dynamical fourth column, not only constitutive laws | — |
| B4 | σ-wave solver (1-D, sourced by a pulsating polytrope) | the program's one candidate L4 test: monopole σ radiation from pressure, none from mass redistribution | — (theory test first) |
| B5 | a defined Ricci scalar R and Riemann sign convention in the live repo | □σ = −ξR is used without either; the sign of σ near matter is a convention | decision, then `field/` |
| B6 | spherical / 1-D transonic flow in the engine | P3/S5 needed a new finite-volume utility because SPH cannot resolve a transonic profile at affordable N | Bondi 1952 (exact) |
| B7 | adaptive smoothing length in SPH | free expansion and collapse leave fixed-h kernels empty or overfull | — |
| B8 | 3-D Barnes–Hut (only a 2-D quadtree exists) | N-body beyond a few thousand bodies | — |
| B9 | `lookup_constant` cannot resolve the Planck length (`l_P`, `planck_length`) although `L_PLANCK` exists | P4 | — |
| B10 | Mentat `melting_point` lacks titanium | S7 | CRC |

## C. Open decisions (the Captain's)

| # | decision | options | consequence |
|---|---|---|---|
| C1 | where SSBM constants live (ξ, σ_conv, `bounds.py` excised from `field/` in e66615b; only `inventory/core/sigma.py` keeps ξ) | back into `field/` / stay in inventory / stay in ssbm-theory | every σ ≠ 0 computation imports from one of these |
| C2 | the σ normalisation: σ = 2ξGM/rc² (field-equation reading) vs ξGM/rc² (`inventory.sigma_from_potential`, `field/audit.py:143-161`); docstring quotes a third value | — | factor 2 on every σ prediction |
| C3 | ξ = 0.1582 (Planck + BAO) vs 0.1571 (the docstring's own Ω values) | — | 0.7 % on every σ |
| C4 | which □: covariant (carry (1 + 2Φ/c²) and the Tolman source ρ + 3P/c² for Φ) or flat-background (state the theory as non-covariant) | — | P7's reshaping numbers depend on it |
| C5 | σ's range λ (the S9a result): long range is excluded for ξ ≈ 0.16; the PAPER's m_σ ≈ Λ_QCD (λ = 0.91 fm) is allowed and invisible | pick λ, or make it a parameter | sets whether any experiment can ever see σ |
| C6 | `total_energy()` default (A2), Eyring fix (A5), SPH kernel / self-term correction (A7) | — | engine accuracy |
