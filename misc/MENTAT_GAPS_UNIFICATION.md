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
