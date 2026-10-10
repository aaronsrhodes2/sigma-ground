# Mentat engine checks against June–October 2026 physics results (run 2026-10-10)

The six topic notes in this folder plus `sigma_ground_capabilities.md` supplied the hooks. I picked 41 of them and ran each through sigma-ground's own engine: the Mentat MCP server, started locally over stdio. Every number below is the engine's output, not mine, except where a row says otherwise. Engine outputs are stated as computed. Where a value looks wrong, I recorded it and flagged it; no repository file was modified.

## How the runs were done (read before using the numbers)

- **Server.** `sigma_ground.mcp.server` (232 tools), called through the scratchpad helper `mentat.py` with `call` and `batch`. Batch files are in the scratchpad under `b/`, and the driver scripts are under `drv/`.
- **Missing dependencies.** On the first run, `lookup_constant` (and every tool that uses scipy, sympy, pint or periodictable) failed with `No module named 'periodictable'`. These four libraries are not in the base Python here. I installed them with `pip --target` into the scratchpad PYTHONPATH directory (`mcp1`). The repo itself is untouched. This is a deployment gap, not a code bug.
- **Row labels.**
  - **MCP** = a Mentat tool call.
  - **DIRECT** = a direct function call, not an MCP tool. Mentat exposes no σ ≠ 0 physics, no muon-g−2 tool, and no H0 argument, so some checks had to call the code directly.
  - **DIRECT+override** = the engine's own functions, called after setting `H0`, `OMEGA_M` and `OMEGA_LAMBDA` (or the g−2 constants) in memory only. This is the only way to evaluate the engine's cosmology at a non-Planck H0.
  - **CALC** = Mentat's sympy `simplify_expr` evaluating a textbook formula. That is a calculator check, not an engine capability, and those rows are verdict ENGINE GAP.
- **σ-exclusion rerun.** `scripts/unification/sigma_exclusion.py` was imported as a module, so its `__main__` block, which writes files, did not run. Its `mentat_client` calls (normally made via docker) were routed to the local server, and the 2026 bounds were added on top.
- **Systematic offsets.** The engine's Mpc is 3.086e22 m (`constants.py:414`), which puts ρ_c 0.02% low and R_H and ages 0.01% high. The engine's M☉ is 1.989e30 kg, which is +0.03% versus the IAU-consistent 1.98841e30.

Verdict key:
- **REPRODUCES**: agrees with the reference within its stated uncertainty, or within 1% for derived hooks.
- **CONTRADICTS**: disagrees.
- **ENGINE GAP**: no tool or code path can compute it.
- **SUSPECTED BUG**: the engine's value or behaviour looks wrong.

---

## 1. Cosmology (source: dark_energy_expansion.md, holography_quantum_gravity.md, astrophysics_other_physics.md)

The MCP cosmology tools (`hubble_radius`, `critical_density`, `age_of_universe`, `mond_a0_constant`) take **no H0 or Ω_m argument**. They always use Planck 2018 values (H0 = 67.4, Ω_m = 0.315). Rows at other H0 values are DIRECT+override.

| # | Hook | Tool / args | Engine output | Reference | Diff | Verdict |
|---|---|---|---|---|---|---|
| C1 | Hubble radius c/H0 | MCP `hubble_radius` {} (H0 = 67.4); DIRECT+override for the others | 67.4: 1.37264e26 m = 4447.96 Mpc. 66.5: 4508.16 Mpc. 67.24: 4458.54 Mpc. 68.17: 4397.72 Mpc. 73.50: 4078.81 Mpc | Note Q3: 4458.5 / 4397.7 / 4078.8 Mpc (c/H0 by definition for 66.5) | ≤ 0.01% | REPRODUCES. The tool's own note prints "4447.96 **Gpc**" and its description says "~14 Gpc"; see bug B3 |
| C2 | Critical density 3H²/8πG | MCP `critical_density` {}; DIRECT+override | 67.4: 8.5311e-27. 66.5: 8.3048e-27. 67.24: 8.4906e-27. 73.50: 1.01452e-26 kg/m³ | 1.87834e-26 h² kg/m³ (PDG) | −0.02% throughout (engine Mpc rounding) | REPRODUCES |
| C3 | ΛCDM age, DESI+CMB (68.17, Ω_m 0.3027) | DIRECT+override → `mcp.tools.cosmology.age_of_universe('lcdm')` | 13.795 Gyr | 13.79 Gyr (note Q6) | +0.04% | REPRODUCES |
| C4 | ΛCDM age, DR2-IV DESI+CMB (68.11, 0.3042) | same | 13.788 Gyr | 13.78 Gyr | +0.06% | REPRODUCES |
| C5 | Age for the new parameter sets | same; MCP default for 67.4 | 67.4/0.315: 13.798 (MCP). Lyα FS+BBN 66.5/0.325: 13.862. CMB-SPA 67.24/0.3165: 13.812. H0DN 73.50/0.315: **12.653 Gyr** | Oldest stars A★ = 13.80 (+0.20/−0.16) ± 0.4 Gyr, plus about 0.2 Gyr formation delay. Note: "high-H0 ΛCDM ≈ 12.7 Gyr", and t0 < ~13.3 Gyr should be flagged | 73.5 case is ~0.65 Gyr below the 13.3 floor | REPRODUCES the 12.7 Gyr figure. With ΛCDM and H0DN's H0, the engine's age falls **below the oldest-star floor**, which matches the tension the notes describe |
| C6 | Deceleration parameter q0 and transition redshift z_t | No engine function. CALC `simplify_expr` "Ω_m/2 − Ω_Λ", "(2Ω_Λ/Ω_m)^(1/3) − 1" | Ω_m 0.3027: q0 = −0.5460, z_t = 0.6640. Ω_m 0.325: −0.5125, 0.6075. Ω_m 0.315: −0.5275 | −0.546, 0.663 (note Q6) | < 0.2% | ENGINE GAP (calculator only) |
| C7 | w0wa DESI+CMB+Dovekie (67.37, 0.3134, w0 = −0.821, wa = −0.65): t0 and q0 | No w0/wa anywhere. DIRECT+override runs the same parameters as ΛCDM | ΛCDM-only engine: t0 = 13.824 Gyr. CALC q0 = ½(1 + 3w0Ω_DE) = −0.3455 | 13.75 Gyr, q0 = −0.345, phantom crossing z ≈ 0.38 | ΛCDM engine is +0.5% in t0 and q0 −0.53 vs −0.345 | ENGINE GAP |
| C8 | r_s = R_H identity, M = c³/(2GH) | MCP `schwarzschild_radius`(M_HUBBLE_KG = 9.24192e52) = 1.372640e26 m = `hubble_radius`; DIRECT+override for other H0 | r_s/R_H = 1.000 and M_crit/M = 1.000 at every H0. M = 4.6465e22 (67.4), 4.6576e22 (67.24), 4.5940e22 (68.17), 4.2609e22 (73.50) M☉ | 4.65 / 4.658 / 4.595 / 4.262 ×10²² M☉ (notes) | ≤ 0.03% | REPRODUCES. **It is a tautology:** the identity holds for any H0, so it cannot tell cosmologies apart. `constants.py:421` M_HUBBLE_KG uses the **Hubble** horizon c/H0 |
| C9 | Gaztañaga variant r_S = c/(H0√Ω_Λ) | MCP `schwarzschild_radius`(M_HUBBLE/√0.685) = 1.65848e26 m; DIRECT+override | 67.4/0.685: 5.614e22 M☉, 5374 Mpc. 67.24/0.6835: 5.634e22 M☉, 5393 Mpc. 68.17/0.6973: 5.502e22 M☉, 5266 Mpc | 5.63e22 / 5388 Mpc (Ω_Λ 0.6847); 5.50e22 / 5266 Mpc | ≤ 0.1% | REPRODUCES when fed Ω_Λ. The two identities differ by 1/√Ω_Λ ≈ 1.21; the engine's constant uses the Hubble one |
| C10 | HDE density at c² = η, L = R_H: the "0.60×" deficit | MCP `hde_dark_energy_density` {} and {c_squared: 0.685}; DIRECT+override | ρ_HDE = 3.1602e-10 J/m³ vs Ω_Λρ_c c² = 5.2521e-10 → **ratio 0.6017** (67.4). 0.6106 (Ω_m 0.325). 0.5911 (DESI+CMB) | Repo test: η/Ω_DE ≈ 0.603 | ≤ 0.2% | REPRODUCES the repo's own documented deficit. That means the c² ≡ η identification with L = R_H **CONTRADICTS** the observed ρ_DE by about 40% |
| C11 | HDE c² implied by observed ρ_DE (Hubble cut-off) | MCP `holographic_dark_energy_analysis` {rho_de_J_m3: 5.25e-10}; DIRECT+override | c² = 0.6847 (67.4). 0.675 (Ω_m 0.325). 0.697 (DESI+CMB). Always c² ≡ Ω_Λ | Notes: Hubble-cutoff HDE gives c² = Ω_de trivially, and this cut-off class is **ruled out once CMB priors are added** (arXiv:2608.09379). HDE/RDE are "ruled out" (2509.02945) | — | CONTRADICTS (the engine's HDE model is in the observationally disfavoured class). The engine has no future-event-horizon cut-off (needs H(z)) |
| C12 | UP-001: DESI DR3 HDE c² inside [0.410, 0.420] | MCP `eta_desi_band_check` {dataset: dr3}; `eta_value_report` | dr3 → `in_band: false`, `low/high: null`, "DR3 numerics not yet lifted". ETA = 0.412164 ± 0.036 | No DR3 HDE result exists; DR3 BAO and Y5 results are expected in 2027 | — | ENGINE GAP / untestable now. ETA sits inside [0.410, 0.420] by construction. Returning `false` (not "pending") is misleading; see bug B8 |
| C13 | DESI band check (dataset 'dr2') | MCP `eta_desi_band_check` {dataset: dr2} | in_band = true, band [0.3770, 0.4489] | No 2026 HDE c² number in the notes | — | REPRODUCES its own (2024) input only. Provenance: labelled "DESI DR2 Union3" but cites arXiv:2411.08639, a Nov-2024 paper that predates DR2 (Mar 2025); see bug B13 |
| C14 | Σm_ν upper limit | MCP `lookup_constant` SUM_NEUTRINO_MASS_DESI_UPPER_EV | 0.072 eV ("DESI 2024") | DESI DR2: < 0.064. DR2-IV DESI+CMB: < 0.0592. SPT-3G+CMB-SPA+DESI: < 0.072. ACT+DESI: < 0.052 eV | engine bound is 12–22% looser than DESI DR2 | REPRODUCES only the SPT-3G combination. Stale relative to DESI DR2 (easy win) |
| C15 | Oscillation splittings | MCP `lookup_constant` DELTA_M2_21_EV2, DELTA_M2_31_EV2 | 7.42e-5, 2.514e-3 eV² (PDG 2022) | JUNO 2026: 7.388(78)e-5, 2.509(+27/−25)e-3 | +0.43% (0.4σ), +0.2% (0.2σ) | REPRODUCES within errors. PDG 2022 edition (easy win) |

## 2. Black holes, GR and compact stars (source: black_holes_gravitational_waves.md, holography_quantum_gravity.md)

Masses use the engine's M☉ = 1.989e30 kg: m1 = 33.6, m2 = 32.2, M_f = 62.7 M☉. All metrics are **Schwarzschild only**: no spin input exists.

| # | Hook | Tool / args | Engine output | Reference | Diff | Verdict |
|---|---|---|---|---|---|---|
| G1 | r_s, T_H, horizon area for GW250114 | MCP `bekenstein_hawking_entropy` {mass_kg: 6.68304e31 / 6.40458e31 / 1.247103e32}; `run_simulation` black_hole {solar_masses: 62.7} | r_s = 99.26 / 95.12 / **185.22 km**. T_H = 1.836e-9 / 1.916e-9 / **9.838e-10 K**. A = 1.2381e11 / 1.1371e11 / 4.3112e11 m² | T_H(62.7 M☉) ≈ 1.0e-9 K (note, Schwarzschild formula) | −1.6% vs the rounded note value | REPRODUCES (Schwarzschild). The remnant is Kerr with χ = 0.68: the Kerr T_H and area are both about 0.85× this. Kerr: ENGINE GAP |
| G2 | Area theorem A_f > A_i, ΔA/A ≈ 0.57 | From G1 areas | A_i = 2.375e5 km² (matches the note's 2.37e5). A_f = 4.311e5 km² (Schwarzschild). **A_f/A_i = 1.815** | Kerr χ_f = 0.68: A_f = 3.73e5 km², ratio 1.57 | +15.6% on the ratio | Qualitative pass (area grows), but the magnitude is wrong without spin. ENGINE GAP (Kerr) |
| G3 | Radiated energy 3.1 M☉c², ≈ 4.7% | MCP `mass_to_energy` {mass_kg: 6.1659e30} | 5.542e47 J = 4.71% of 65.8 M☉ | 3.1 M☉c², 4.7% | — | REPRODUCES |
| G4 | Kerr QNM f₂₂₀ = 249.0 Hz, τ = 4.1 ms (χ = 0.68, M_z = 68.4 M☉) | none (grep: no kerr / quasinormal / qnm in the code) | — | 249.0 (+5.4/−5.7) Hz, 4.1 (+0.4/−0.3) ms | — | ENGINE GAP |
| G5 | Compactness of PSR J1614−2230 (1.937 M☉, 10.06 km) | MCP `schwarzschild_radius` {mass_kg: 3.852693e30}; `gravitational_redshift`; `gravitational_time_dilation` | r_s = 5722.1 m → GM/Rc² = **0.2844**, R/r_s = 1.758. Surface z = 0.5229, clock rate 0.6567 | 0.284, R/r_s = 1.76 | < 0.2% | REPRODUCES |
| G6 | 2.35 M☉ / 12 km star (J0952 mass) | same, {mass_kg: 4.67415e30} | r_s = 6942.2 m → **0.2893**, z = 0.5403 | 0.289 | < 0.1% | REPRODUCES |
| G7 | Buchdahl limit GM/Rc² ≤ 4/9 | no function; ratio taken from G5/G6 outputs | J1614 is at 64.0% of 4/9 and 56.9% of the horizon value 0.5; R = 1.56 × (9/8) r_s | about 65% of Buchdahl, 57–58% of horizon | — | REPRODUCES via r_s, but there is no Buchdahl or causal-EoS function (partial ENGINE GAP) |
| G8 | NS maximum mass M_TOV > 2.27 M☉ (1σ), ≈ 2.2–2.3 M☉ | none: no TOV, EoS, Chandrasekhar or polytrope code (gap B1; `test_unsolved_frontiers.py` cites an absent "TOV solver"). `solar_system_body`/`named_star` 'neutron_star' → not found | — | M_TOV > 2.27 M☉ | — | ENGINE GAP |
| G9 | T_H at r_s = R_H vs the de Sitter temperature (factor 2) | MCP `hawking_temperature` {mass_kg: 9.24192e52}; `unruh_temperature` {acceleration_m_s2: cH0 = 6.54764e-10} (that is ħH/2πk_B) | T_H = 1.32754e-30 K. T_dS = 2.65508e-30 K. **Ratio 0.50000** at every H0 tested (DIRECT+override) | Exactly ½. T_dS ≈ 2.7e-30 K, T_H ≈ 1.3e-30 K | exact | REPRODUCES |
| G10 | Bekenstein / BH entropy at the Hubble radius | MCP `bekenstein_hawking_entropy` {mass_kg: 9.24192e52}; cross-check `entanglements_to_pop_bubble` {radius_m: R_H} | S = 2.26591e122 k_B (H0 = 67.4). The bubble-thread count gives the identical 2.26591e122 | ≈ 2.2e122 k_B (note, H0 = 67.7; scaling to 67.4 gives 2.246e122) | ≤ 1% | REPRODUCES |
| G11 | Full BH thermodynamics cascade | MCP `procedure_black_hole_profile` {mass_kg: 1.247103e32} | **Error**: "Input should be a valid dictionary … input_value=None" | — | — | SUSPECTED BUG B1 (tool body missing) |
| G12 | (fringe) Haug–Tatum T_CMB = √(T_H(M_Hubble)·T_H(m_P)) | MCP `hawking_temperature` {9.24192e52} and {2.17643e-8} | 1.32754e-30 K × 5.63721e30 K → √ = 2.7356 K (H0 = 67.4) | Note formula: 2.7325 K at H0 = 67.24 (scales to 2.7357 at 67.4). T_CMB = 2.72548 K | +0.37% vs T_CMB | REPRODUCES the fringe formula. Low value: the source is not peer-reviewed |

## 3. The σ field (source: sigma_ground_capabilities.md, dark_matter.md, holography_quantum_gravity.md, particle_nuclear_physics.md, astrophysics_other_physics.md)

No MCP tool exposes σ. `run_simulation` nucleon_masses with {sigma: 0.0791} returns "unexpected keyword argument 'sigma'". All σ rows are DIRECT calls to `inventory.core.sigma.sigma_from_potential`, `field.nucleon.proton_mass_mev`/`neutron_mass_mev` and `inventory.core.sigma.*`.

**S1. σ values and nucleon masses (DIRECT).** The engine's ξ = 0.1582.

| location | GM/Rc² | σ (code, ξGM/Rc²) | σ (S9a field reading, 2ξGM/Rc²) | m_p(σ) MeV | m_n(σ) MeV | m_n − m_p MeV |
|---|---|---|---|---|---|---|
| Earth surface (planets.EARTH, R = 6371 km) | 6.961e-10 | **1.101e-10** | 2.202e-10 | 938.2720 | 939.5650 | 1.29300 |
| Sun surface | 2.123e-6 | 3.359e-7 | 6.72e-7 | 938.2723 | 939.5653 | 1.29300 |
| 1.4 M☉ NS (planets.NEUTRON_STAR, R = 11 km) | 0.18799 | **0.02974** | 0.0595 | 966.323 | 967.578 | 1.2554 |
| PSR J1614−2230 (1.937 M☉, 10.06 km) | 0.28440 | 0.04499 | 0.0900 | 981.036 | 982.272 | 1.2356 |
| 2.35 M☉ / 12 km | 0.28926 | 0.04576 | 0.0915 | 981.783 | 983.018 | 1.2346 |
| any horizon (r = r_s, capped at 0.5) | 0.5 | **0.0791** (= ξ/2) | 0.1582 | 1014.761 | 1015.952 | 1.1904 |

- The m_n − m_p sign flip sits at e^σ = 2.0369, i.e. σ = 0.7114, unchanged from the capability notes.
- `field.nucleon` and `inventory.core.sigma` disagree at the 1e-4 MeV level. At σ = 0, `field.nucleon` gives m_p = 938.272 and m_n − m_p = 1.29300. `inventory` gives 938.2720881 and the CODATA-like difference. See bug B6.

**S2. The σ_Earth inconsistency: CONFIRMED.**
- The code's `sigma_from_potential` gives **1.10e-10**.
- 21 test files (e.g. `test_thermal.py:379`, `test_friction.py:299`, `test_molecular_bonds.py:501`, `test_radioactive_decay.py:488`) use `sigma=7e-10`. So do the docstrings at `inventory/core/sigma.py:294` and `field/constants.py:453`.
- Mentat itself shows where 7e-10 comes from. MCP `schwarzschild_radius`(5.97219e24) = 8.8701e-3 m, so GM/(Rc²) = r_s/2R = 6.954e-10. MCP `gravitational_redshift`(Earth) = 6.954e-10.
- So **7e-10 is the bare potential Φ/c², with ξ left out**. The repo carries three Earth values: 7.0e-10 (ξ omitted), 2.2e-10 (S9a field reading, 2ξ) and 1.1e-10 (code, ξ).
- The tests are conservative, since they use a larger σ, so nothing passes wrongly. But the number is inconsistent with the code.

**S3. σ-exclusion rerun.** `scripts/unification/sigma_exclusion.py` was imported as a module, with its 14 Mentat leaf calls served by the local server (`lookup_constant` G/c/ħ/eV/au, `solar_system_body` sun/earth, `nuclear_binding_energy` ×8). The long-range field reading is λ → ∞, ξ = 0.1582, with 2σ two-sided bounds as in the script.

| bound | engine prediction | bound used | exclusion factor | ξ_max (long range) | Verdict |
|---|---|---|---|---|---|
| Yb⁺/Cs clocks, (c²/μ)dμ/dΦ (Lange 2021, repo baseline) | −0.3134 (f_μ = 0.99039) | 9.7e-7 | **3.23e5** | 4.90e-7 | REPRODUCES the repo's S9a number exactly |
| MICROSCOPE η(Ti,Pt), (−1.5 ± 2.3 ± 1.5)e-15 (repo baseline) | 6.593e-4 (Δf = 2.0838e-3) | 6.99e-15 | **9.43e10** | 1.68e-12 | REPRODUCES S9a exactly |
| MICROSCOPE "no violation at 2.7e-15" (holography note) | 6.593e-4 | 2.7e-15 | **2.44e11** | 6.48e-13 | long-range σ CONTRADICTED |
| MICROSCOPE range limit | ξ_max(λ = 34.5 km) = 0.161 | — | — | — | REPRODUCES the repo's "λ < ~34.5 km" |
| White-dwarf H₂ (repo baseline) | Δμ/μ = 3.76e-5, 5.95e-5 | — | — | 0.357 | REPRODUCES S9a |
| **NEW: China Space Station ⁸⁵Rb/⁸⁷Rb, η = (−2.7 ± 4.7)e-7** (Sci. Adv. 28 Aug 2026) | f(⁸⁵Rb) = 0.9858025 (BE 739.28 MeV), f(⁸⁷Rb) = 0.9858637 (BE 757.86 MeV) → \|Δf\| = 6.125e-5 → η = **1.94e-5** (orbit R⊕ + 390 km) | 1.21e-6 | **×16** | **9.9e-3** | Long-range σ CONTRADICTED. Weaker than MICROSCOPE, but this is the first quantum-matter test, and it alone excludes ξ = 0.158 for λ ≳ 10³ km (ξ_max: 0.060 at λ = 10⁶ m, 19.6 at 10⁵ m). ⁸⁷Rb is **missing from `inventory/data/isotopes.json`**; I supplied its AME2020 mass 86.909180531 u |
| **NEW: PTB (1/α)dα/dt = 1.8(2.5)e-19/yr; (c²/α)dα/dΦ = −2.4(3.0)e-9** (dark_matter.md) | 0 for both: the repo's σ leaves α fixed by construction (`inventory/core/sigma.py:1-29`) | — | — | no constraint | Consistent (null prediction). α-only optical-clock bounds cannot touch this σ. The relevant 2026 clock tests are the μ- and Λ_QCD-sensitive ones |
| **NEW: ²²⁹Th nuclear clock** (Nature 7 Oct 2026) | not computable | no numeric d_g bound in the notes (it appears only in a figure) | — | — | ENGINE GAP. Th-229 is not in `isotopes.json` and there is no isomer or Coulomb-vs-strong split model. The repo's BE(σ) = BE_strong·e^σ − BE_C decomposition is exactly the near-cancellation that gives Th-229 its K ≈ 5900 enhancement. A Th-229 module would probably be the most sensitive σ probe available (my inference, not computed) |
| Paper reading (m_σ ≈ Λ_QCD, λ = 0.909 fm) | σ at nuclear density = 6.54e-40 | — | — | ∞ (unconstrained) | REPRODUCES S9a ("allowed and invisible") |

## 4. Particle and nuclear physics, constants (source: particle_nuclear_physics.md, astrophysics_other_physics.md)

| # | Hook | Tool / args | Engine output | Reference | Diff | Verdict |
|---|---|---|---|---|---|---|
| P1 | Muon g−2 status | No MCP tool. DIRECT `field.electrodynamics.muon_g2_tension_sigmas()`, `muon_g2_consistency_status()`; MCP `lookup_constant` A_MU_* | a_μ(exp) = 116592061(22), a_μ(SM) = 116591810(43) ×10⁻¹¹ → **5.20σ**, while `MU_G2_TENSION_RESOLVED = True` | WA 2025 = 116592071.5(14.5); WP2025 SM = 116592033(62) → 0.6σ | DIRECT+override with WA25/WP25 gives **0.605σ** | CONTRADICTS: the inputs are stale and mislabelled. `constants.py:149` labels the 2020-WP SM value "MTI 2025 WP", and `:147` labels a pre-final experimental average "Fermilab final". Once the constants are updated, the engine's formula reproduces the 0.6σ hook. Bug B5 |
| P2 | Fine-structure constant | MCP `lookup_constant` {name: alpha / fine_structure_constant} | α = 0.007297352570287747 → α⁻¹ = **137.035999065** (derived from pre-2019 ε₀ 8.854187817e-12 and CODATA-2018 e, ħ) | CODATA 2022: 137.035999177(21). Cs 2018: …046(27). Rb 2020: …206(11) | −0.82 ppb (−5.3σ of CODATA-2022 uncertainty); +0.7σ from Cs; −13σ from Rb | Agrees to < 1 ppb, but CONTRADICTS CODATA 2022 at its stated precision. Stale edition (easy win). The engine cannot represent the Cs/Rb split |
| P3 | G (NIST 2026: 6.67387(38)e-11) | MCP `lookup_constant` {name: G} | 6.67430e-11 (CODATA) | 6.67387(38) | +6.4e-5 relative (+1.1σ of NIST) | REPRODUCES (consistent). Keep CODATA. The NIST-page variant 6.67366(20) would be a 2.6σ conflict (unresolved in the notes) |
| P4 | Particle masses vs 2026 values | MCP `lookup_constant` {W_mass, top_quark_mass, higgs_mass, Z_mass, muon_mass} → **all "not found"**. DIRECT read of `inventory/data/particle_masses.json` (PDG 2026 snapshot) | W 80362.5 ± 7.7. Z 91187.87 ± 1.97. t 172603.6 ± 272. H 125130.9 ± 112. p 938.27208943. μ 105.6583755 MeV | m_W = 80362.5 ± 7.7 (PDG 2026). m_Z = 91188.0 ± 2.0. m_t = 172.52 ± 0.33 (Run-1 combination). m_H within 125.09–125.38 | W exact. Z −0.07σ. t +0.25σ. H inside | REPRODUCES (the snapshot is PDG 2026). Mentat cannot return any boson, top or muon mass; that is an MCP gap |
| P5 | ²²⁹Th clock line ν_b = 2.02040729879617e15 Hz | MCP `photon_energy_from_frequency`; `em_wave_wavelength` | E = 1.338736e-18 J = **8.35573 eV**; λ = **148.3822 nm** | 8.35573 eV, 148.382 nm | < 1e-5 | REPRODUCES |
| P6 | Nucleon QCD fraction and bare masses | MCP `run_simulation` nucleon_masses; `lookup_constant` PROTON/NEUTRON_QCD_FRACTION | m_p 938.3, m_n 939.6 MeV, Λ_eff 217 MeV. QCD fraction 0.99039 / 0.98770. m_u = 2.16, m_d = 4.70 (PDG 2026) | PDG 2026 m_u, m_d unchanged | — | REPRODUCES (inputs current) |

## 5. Fluids (source: fluid_gas_dynamics.md)

| # | Hook | Tool / args | Engine output | Reference | Diff | Verdict |
|---|---|---|---|---|---|---|
| F1 | Pipe Re at the new thresholds | MCP `viscous_flow_analysis` {velocity_m_s: 0.175 / 0.204 / 0.29, radius_m: 0.005} (water defaults; Re uses D = 2r) | Re = 1750 / **2040** / **2900** | Re_c ≈ 2040 (directed-percolation onset); puffs 1750–2300; uniform turbulence > 2900 | exact | REPRODUCES the number. **No regime classifier.** `fluid.py:363-365` documents laminar < 2300, transitional 2300–4000, turbulent > 4000. That is the textbook map, not the 2026 map (2040 / 2300 / 2900). ENGINE GAP plus a documentation update |
| F2 | Finite-amplitude threshold E₀^crit = 6.09e5 Re⁻³ | none | — | — | — | ENGINE GAP |
| F3 | Sedov–Taylor R = S(γ)(E t²/ρ)^(1/5), S(1.4) = 1.033 | none (grep "sedov": 0 hits; KNOWN_GAPS B2: no shocks or Mach relations) | — | — | — | ENGINE GAP |
| F4 | Rayleigh collapse time 0.915 R√(ρ/Δp); Rayleigh–Plesset rebound criterion k > 1/3 | none | — | — | — | ENGINE GAP |
| F5 | Weber / Ohnesorge splash thresholds (K_c ≈ 2100; We ≈ 100/200/2000) | none. Surface tension exists in `fluid.py`, but no We/Oh tool | — | — | — | ENGINE GAP |
| F6 | η/s vs KSS 1/4π | No MCP tool. MCP `lookup_constant` ħ, k_B plus CALC → ħ/(4πk_B) = **6.0783e-13 K·s** (= 1/4π = 0.0796 in ħ/k_B units). DIRECT `liquid_water` gives water at 25 °C = 378.4× KSS. The repo's QGP band [1, 3]× is a literature value, not code | — | 2026 QGP η/s ≈ 0.08–0.2, i.e. 1–2.5× KSS | inside the band | REPRODUCES (band consistent). The engine has no QGP or η/s computation |
| F7 | Side findings from F1 | MCP `viscous_flow_analysis` | Poiseuille Q = 2.454e-4 m³/s and v_max = 6.25 m/s **identical for every input velocity**. Stokes terminal velocity 92.6 m/s returned at Re ≈ 10⁶ with no validity warning | — | — | SUSPECTED BUG B10 (hidden hard-coded inputs) |

## 6. Dark matter and MOND (source: dark_matter.md, holography_quantum_gravity.md)

| # | Hook | Tool / args | Engine output | Reference | Diff | Verdict |
|---|---|---|---|---|---|---|
| D1 | a0 vs cH0/2π | MCP `mond_a0_constant` {}; DIRECT+override for other H0 | a0 = 1.2e-10 (constant). cH0/2π = **1.042e-10** (67.4, ratio 0.868). 1.028e-10 (66.5). 1.040e-10 (67.24). 1.136e-10 (73.5) | Note: cH0/2π ≈ 1.05e-10 at H0 = 67.7; SPARC 1.20 ± 0.02 ± 0.24 | 1.042 × 67.7/67.4 = 1.047e-10 | REPRODUCES. Also, MUSE-DARK III's z = 0 intercept a0(0) = 1.00 ± 0.04 is ~1σ from cH0/2π but **5σ from the engine's stored A0_MOND = 1.2e-10** (a cautious flag: it is an extrapolated intercept) |
| D2 | a0 tracking H(z): a0(1)/a0(0) = 1.77 vs MUSE-DARK 1.72–2.6 | none (no H(z)) | — | γ = 0.78 ± 0.15 | — | ENGINE GAP |
| D3 | MOND regime of a wide binary (1 M☉ at 10 kAU, g_N = 5.9e-11) | MCP `mond_regime_classifier` {5.93e-11}; {1.2e-10} | "transition"; "transition" | Chae: low-acceleration regime is g_N < 1e-9 | — | REPRODUCES (classification only; no ṽ or boost prediction) |
| D4 | LZ 248 keV event: q = √(2 m_N E_R) | MCP `convert_units` 130.9050841 Da → 2.17373e-25 kg; `kinetic_energy` / `momentum` at v_N = 6.04635e5 m/s; `convert_units` kg·m/s → MeV/c | E_R = 3.97340e-14 J (= 248.0 keV); **q = 245.93 MeV/c** | q ≃ 246 MeV | −0.03% | REPRODUCES |
| D5 | Elastic m_χ threshold (v_lab,max = 800 km/s, ¹³¹Xe) | MCP `collision_analysis` {m_χ = 74 / 79 / 1000 GeV, v1 = 8e5, m_Xe, v2 = 0, restitution = 1}; `kinetic_energy` | Head-on recoil: 74 GeV → v_N = 6.0427e5 m/s, **E_R,max = 247.7 keV** (threshold). 79 GeV → 268.4 keV | Note: m_χ ≳ 74 GeV (v = 800 km/s); paper ≳ 79 GeV (different v_max) | — | REPRODUCES via the 1D elastic-collision tool. There is no dedicated recoil, form-factor or halo tool (ENGINE GAP for rates) |
| D6 | Haloscope map: m_a [μeV] = 4.1357 f[GHz] | MCP `photon_energy_from_frequency` {1.036e9} | 4.285e-6 eV | 4.28 μeV | — | REPRODUCES |
| D7 | mm-scale Cavendish: two 81 mg gold spheres at 3 mm | MCP `gravitational_force` {8.1e-5, 8.1e-5, 0.003}; `density` gold = 19300 | 4.8656e-14 N (161× the 3.02e-16 N sensor floor) | 4.9e-14 N, ~160× | −0.7% | REPRODUCES |
| D8 | Unruh: about 2.47e20 m/s² per kelvin | MCP `unruh_temperature` {2.47e20} | 1.0016 K | 1 K | +0.16% | REPRODUCES |
| D9 | Lu⁺ clock redshift gΔh/c² ≈ 1.09e-16 per metre | MCP `gravitational_redshift` {Earth, R} vs {R + 1 m} and {R + 1000 m} | 1 m: Δz = **2.220e-16** (exactly one double-precision ULP). 1 km: Δz = 1.0902e-13 | 1.09e-16 /m; 1.09e-13 /km | **+104% at 1 m**; exact at 1 km | SUSPECTED BUG B2 (catastrophic cancellation in `gr_basics.py:61`). The engine cannot resolve sub-km heights at the 1e-16–1e-19 level the 2026 clocks reach |

---

## (a) Easy wins: new data the engine could take in now, ranked

These are recommendations only; nothing was edited.

1. **Muon g−2 inputs.** `sigma_ground/field/constants.py:147-153`: set exp = 116592071.5 ± 14.5 (WA 2025) and SM = 116592033 ± 62 (WP2025). The engine's own formula then gives 0.605σ, which matches `MU_G2_TENSION_RESOLVED = True`. The test `tests/test_electrodynamics.py:228-232` (`assert 4.5 < t < 6.0`) would need to change to about 0.6, and the docstring at `field/electrodynamics.py:296-305` ("≈5.2σ … artefact") would need to change too.
2. **Let the MCP cosmology tools take H0 and Ω_m, and carry early and late H0.**
   - Where: `mcp/tools/cosmology.py:16-31` (hubble_radius), `:141-156` (critical_density), `:164-213` (age); `field/interface/cosmology.py:185-193, 326-344`; `field/constants.py:414-420`.
   - With an in-memory override the existing formulas already reproduce every ΛCDM regression target in the notes to ≤ 0.06% (C1–C5, C8, C9). Only the plumbing is missing.
   - Suggested slots: CMB-SPA 67.24, DESI+CMB 68.17 / Ω_m 0.3027, H0DN 73.50, Lyα FS 66.5 / 0.325.
3. **Neutrino constants.**
   - `constants.py:83`: SUM_NEUTRINO_MASS_DESI_UPPER_EV 0.072 → 0.064 (DESI DR2) or 0.0592 (DR2-IV DESI+CMB).
   - `constants.py:75-77`: Δm²₂₁ → 7.388e-5 and Δm²₃₁ → 2.509e-3 (JUNO 2026, or PDG 2026).
4. **CODATA 2022 constants.**
   - `constants.py:210` EPS_0 → 8.8541878188e-12; `:221` M_ELECTRON_KG; `:229` AMU_KG. This moves α⁻¹ from 137.035999065 to about 137.035999177.
   - Also `:177/:182` PROTON/NEUTRON_TOTAL_MEV 938.272 / 939.565 → 938.27208943 / 939.56542194, already in the PDG-2026 JSON. That fixes m_n − m_p = 1.29300 vs 1.29333.
   - Also `:199` M_ELECTRON_MEV 0.51100 → 0.51099895.
5. **σ_Earth wording.**
   - `inventory/core/sigma.py:294` and `field/constants.py:453` say "σ ~ 7×10⁻¹⁰". The code gives 1.1e-10; 7e-10 is Φ/c².
   - The 21 test files that hard-code `sigma=7e-10` should state that they use Φ/c² as a conservative bound, or switch to `sigma_from_potential`.
6. **Pipe-flow regime map.** `field/interface/fluid.py:363-365`: add Re_c ≈ 2040 (sustained turbulence), puffs 1750–2300, slugs and anti-puffs 2300–2900, uniform turbulence > 2900, citing the 2026 notes.
7. **Isotope table.** `inventory/data/isotopes.json` (145 entries): add ⁸⁷Rb (86.909180531 u; needed for the CSS WEP test) and ²²⁹Th / ²²⁹ᵐTh (needed for any nuclear-clock σ test).
8. **Add the CSS ⁸⁵Rb/⁸⁷Rb bound to `scripts/unification/sigma_exclusion.py`** as a fourth measurement: η = (−2.7 ± 4.7)e-7, giving ξ_max(long range) = 9.9e-3 and ×16 exclusion. Optionally also record the PTB α-dot and α–Φ bounds as "null by construction".
9. **Label and provenance fixes (cheap).**
   - `mcp/tools/cosmology.py:28` "Gpc" → "Mpc"; `mcp/server.py:511` "~14 Gpc" → "~4.4 Gpc / 14.5 Gly".
   - `field/interface/cosmology.py:72-73, 152` and `constants.py:41-44`: the "DR2" label sits on a 2024 paper.
   - `cosmology.py:161`: the DR3 note says "not yet lifted from arXiv:2512.07281", but `constants.py:55` says that paper has no HDE content.
   - `mcp/server.py:1753` and `mcp/tools/misc_physics.py:148`: "c² ~ 0.78 (DESI-consistent)".
10. **Do NOT fill the DR3 HDE slots** (`constants.py:55-56`). No DR3 HDE c² was published in the window; DR3 BAO and Y5 results are due in 2027. UP-001 stays untestable.
11. **One solar mass.** `constants.py:427` uses 1.989e30, while `materia/bodies.py:46` and the benchmarks use 1.98892e30 (labelled "IAU 2015 nominal"). The IAU-consistent value is GM☉/G = 1.98841e30.

## (b) Capability gaps that block testing the biggest new results

1. **No background-expansion module**: no H(z), w0/wa (CPL), D_M, D_H, r_d, growth or S8.
   - It blocks the headline DESI DR2 Lyα AP result (D_H/r_d = 8.600, D_M/r_d = 39.32 at z = 2.33), the w0wa best fit, age and q0 (C7), and phantom crossing.
   - It also blocks the future-event-horizon HDE cut-off (the only HDE cut-off not excluded), a0(z) ∝ H(z) (D2), Σm_ν or CMB-lensing hooks, and S8.
2. **MCP cosmology is pinned to Planck 2018** (no H0, Ω_m or w arguments, and no early/late H0 choice), so Mentat silently takes a side in the 7σ H0 dispute.
3. **No Kerr physics**: no spin, Kerr area, Kerr T_H, Kerr ISCO, QNMs or ringdown. This blocks the GW250114 area-theorem magnitude (G2), f₂₂₀ / τ₂₂₀ / 221 / quadratic modes (G4), and the EHT shadow size.
4. **No dense-matter structure**: no TOV, EoS, Chandrasekhar, polytropes or Buchdahl (gap B1). This blocks M_TOV > 2.27 M☉, NICER M–R (J1614 at 10.06 km, J0030 at 12.68 km), sound-speed stiffening, and any σ-modified NS structure (m_n(σ) is available but unused).
5. **σ is unreachable through MCP**. Every σ ≠ 0 result needs direct imports. There is also no σ-wave solver (B4) and no Th-229 or nuclear-clock sensitivity model (and no ²²⁹Th or ⁸⁷Rb isotope data).
6. **Particle-physics lookups**: `lookup_constant` cannot return W, Z, H, top or muon masses even though the PDG-2026 snapshot has them. There is no a_μ, HVP, W-mass (Δr) or toponium tool.
7. **Fluids**: no pipe-flow regime classifier, finite-amplitude threshold, Sedov–Taylor or shock relations (B2), Rayleigh–Plesset or collapse time, Weber/Ohnesorge splash physics, or η/s/QGP tool. RT/RM growth and Kelvin-wave dispersion are also absent.
8. **Dark matter**: no nuclear-recoil or halo-rate tool (D5 used the 1D collision tool as a stand-in), no rotation-curve or RAR fitter, no SIDM σ/m(v), no FDM soliton, no PBH mass-function bounds.
9. **Numerical precision**: weak-field redshift and time dilation cannot resolve below about 2e-16 (B2), which is above the 2026 clock frontier (1e-18 to 1e-19).

## (c) Suspected bugs (reported, not fixed)

| ID | Where | Symptom (as observed) | Severity |
|---|---|---|---|
| B1 | `mcp/server.py:955-958` `procedure_black_hole_profile` | The function has a docstring and no body, so it returns None and every call fails pydantic validation. The advertised "full BH cascade" tool is always broken | high (dead tool) |
| B2 | `field/gr_basics.py:61` `gravitational_redshift` (and likely `time_dilation_gr`) | 1/√(1 − r_s/r) − 1 cancels catastrophically in weak fields: Earth 1 m height gives Δz = 2.22e-16 (one ULP) vs 1.09e-16 true (+104%). Fine at the km scale. A log1p/expm1 form would fix it | medium (blocks clock hooks) |
| B3 | `mcp/tools/cosmology.py:28`; `mcp/server.py:511` | `hubble_radius` notes print "~4447.96 Gpc" (it is Mpc); the description says "~14 Gpc" (it is about 4.4 Gpc, or 14.5 Gly) | low (wrong units in text) |
| B4 | `mcp/tools/constants.py:254-260` `_guess_units` | Any name ending in "_M" gets units "m", so `OMEGA_M` comes back as "0.315 m". Also, `list_constants` returns "sg:"-prefixed names that `lookup_constant` cannot resolve. And `lookup_constant('planck_length')` fails although L_PLANCK exists (gap B9 confirmed) | low |
| B5 | `field/constants.py:147-153`; `field/electrodynamics.py:285-320` | The engine computes a 5.20σ muon g−2 tension while `MU_G2_TENSION_RESOLVED = True` is tagged VERIFIED. The SM value labelled "MTI 2025 WP (lattice-HVP)" is the 2020 WP number (WP2025 = 116592033(62)). The test `tests/test_electrodynamics.py:228-232` locks in the stale 5σ | medium (internal contradiction) |
| B6 | `field/constants.py:177,182` vs `inventory/core/sigma.py` | `field.nucleon` gives m_n − m_p = 1.29300 MeV at σ = 0, while the same file's DELTA_NP_TOTAL_MEV = 1.29333236. The field and inventory proton masses differ (938.272 vs 938.2720881). SIGMA_HERE also differs slightly between the two modules (1.17747897118e-61 vs 1.17747897154e-61) | low |
| B7 | `inventory/core/sigma.py:294`, `field/constants.py:453`, 21 test files | σ_Earth is quoted and tested as 7e-10, but the code gives 1.10e-10. Mentat shows 7e-10 = GM/(Rc²), i.e. ξ is missing | low (documentation and test inconsistency; the tests are conservative) |
| B8 | `field/interface/cosmology.py:155-162`; `:107-117` | `eta_in_desi_union3_band('dr3')` returns `in_band: False` (reads as "outside the band") when the data are absent. `eta_coincidence_report()` sets `coincidence_live: True` by comparing ETA with ETA_HDE_UNION3, which is ETA by definition (circular) | low |
| B9 | `mcp/server.py:1753`; `mcp/tools/misc_physics.py:148` | The `holographic_dark_energy_analysis` description says "6e-10 J/m³ → c² ~ 0.78 (DESI-consistent)", while the engine's own DESI band is [0.377, 0.449] | low (misleading text) |
| B10 | `mcp/tools/transport.py:82-90` `viscous_flow_analysis` | Hidden fixed inputs: particle density 2700 kg/m³ for terminal velocity, ΔP = 1 kPa and L = 1 m for Poiseuille, so the Poiseuille outputs ignore velocity. Stokes terminal velocity (92.6 m/s) is reported at Re ~ 10⁶ with no validity warning | low |
| B11 | `materia/bodies.py:46`, `mcp/benchmark/*`; `field/constants.py:427` | Two solar masses (1.98892e30 vs 1.989e30), and neither equals the IAU-consistent GM☉/G = 1.98841e30. `solar_system_body` labels 1.98892e30 "IAU 2015 nominal" | low |
| B12 | `materia` `black_hole` verb | Entropy (5.696e57, which is J/K) and evaporation time (1.631e80, which is s) are reported with empty units. The primary output is "functions_called = 10" | cosmetic |
| B13 | `field/constants.py:41-44`, `cosmology.py:72-73,152` | HDE c² 0.642 is labelled "DESI DR2 Union3" but cited to arXiv:2411.08639, a Nov 2024 paper that predates DR2 (Mar 2025). The constants file itself says "(DESI 2024)" | low (provenance) |

**Environment note (not a code bug).** Without scipy, sympy, pint and periodictable installed, `lookup_constant`, `convert_units`, `element_atomic_data` and every symbolic tool fail. They worked after these were installed into the scratchpad PYTHONPATH. The MCP deployment should declare these dependencies.

## Bottom line

- **Reproduces:** the engine reproduces every closed-form hook it has code for. That covers c/H0, ρ_c, ΛCDM ages to ≤ 0.06%, the r_s = R_H masses, the T_H/T_dS = ½ factor, the Hubble-horizon entropy, NS compactness, LZ recoil kinematics, the Th-229 photon numbers and Re.
- **Contradicts:**
  - The c² ≡ η HDE identification with L = R_H, which misses ρ_DE by 40% and sits in the cut-off class 2026 fits rule out.
  - The long-range σ reading, now also excluded by the 2026 China Space Station WEP test (×16) on top of MICROSCOPE (×10¹¹).
  - The stale muon g−2 inputs (5.2σ vs the current 0.6σ).
- **Can't test:** the biggest 2026 results (DESI w0wa and BAO distances, GW250114's Kerr ringdown and area magnitude, M_TOV/NICER, the Th-229 σ sensitivity) all hit missing modules: H(z), Kerr, TOV and nuclear-clock physics.
