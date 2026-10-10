# Decision memo A2 (v2, after review): options for a σ action (Captain's decisions C1–C6)

**For:** the Captain. **Scope:** decisions C1–C6 in `D:/Aaron/development/sigma-ground/misc/MENTAT_GAPS_UNIFICATION.md` §C, and the program's
next step 2 ("Give σ an action") in `D:/Aaron/development/sigma-ground/misc/UNIFICATION_PROGRAM.md` §14.
**This memo does not choose.** It lists every defensible action and gives the price of each.
**Checks:** `a2_sigma_action_v2.py` runs 44 checks, all passing; output in `a2_results_v2.json`. What each kind of check can and cannot show is in §7.
The predictions were fixed in `prereg.json` before the first run and have not been moved. **Figure:** `a2_sigma_action_options_v2.png`.
**Tags:** [DERIVED] means sympy-certified (check ID in brackets). [BY CONSTRUCTION] means it follows from a definition and could not have failed.
[VERIFIED numerics] means a number computed here from Mentat leaves and the repo engine. [LIT] marks a cited measurement, each citation verified on 2026-10-10.
[SPECULATIVE] marks content outside the derivations.

**What changed from v1** (three independent reviews; the runner's v1 files are kept unchanged beside this one):

- The range claim was wrong. v1 said "λ < 34.5 km escapes all §6 bounds". That limit comes from MICROSCOPE alone. The §6 Eöt-Wash bound, evaluated at finite range, still excludes λ = 34.5 km (indicator ×10⁷, §4).
- The headline now says "options that change masses". Pure σR (O3) passes every §6 bound, but only because σ then changes no mass.
- The composition invariance (B5, B6) is retagged [BY CONSTRUCTION]. It holds for any σ amplitude and restates program §6.
- O4's "ghost-free repo sign" holds only near σ = 0. New sympy checks C3 locate the ghost onset (§2, Fig. d).
- New in v2:
  - which G normalises ξ (§3b);
  - O5 with each test's own source;
  - the lunar-ranging row split into its Nordtvedt and composition parts;
  - the pulsar row shown as a lower bound;
  - per-bound values for O7;
  - a list of which checks could fail.

---

## 0. Summary

1. **No choice of action rescues a long-range σ that changes masses (f ≠ 0) with |ξ| ≈ 0.16.**
   - The composition tests (clocks, MICROSCOPE, Eöt-Wash, white dwarfs) see only σ's amplitude ξ_eff and the composition differences Δf. This is **[BY CONSTRUCTION]**: B5 and B6 hold for an arbitrary σ(k), so it restates program §6 and adds no evidence.
   - The numbers: at |ξ| = 0.1582 every option with a mass response is excluded by **2.5–4.7 × 10⁵ (clocks)** and **0.7–1.4 × 10¹¹ (MICROSCOPE)**. The spread comes from the option and from which G normalises ξ (§3b). The bare-G values are 3.2 × 10⁵ and 9.4 × 10¹⁰ for O1, O2 and O4 alike.
   - The C2 alternative (ξ/2) is excluded by 1.4–1.6 × 10⁵ and 4.1–4.7 × 10¹⁰.
   - The cancellation corner O5 is excluded by **≥ 6.1 × 10² (clocks, Sun-sourced)** and 6.6 × 10⁷ (MICROSCOPE, Earth-sourced).
2. **The one escape at long range is to change no mass.** Pure σR (O3) keeps □σ = −ξR literally, with ξ_eff = 0.1582. It passes every §6 bound for ζ ≤ 1.06 × 10⁻⁴ (Brans–Dicke ω ≥ 1.5 × 10⁴), because σ then moves no clock, mass or force beyond a weak Brans–Dicke field. The "scale of matter" content is gone.
3. **The action decides what remains:** whether σ is a ghost, the sign of the fifth force, and the universal tests (Cassini γ, ω, lunar ranging, pulsars).
4. **The program's "repulsive" finding is confirmed for a pure mass coupling, and corrected only locally for a mixed one.**
   - Pure σT at the repo's sign (σ rises toward matter, masses rise with σ) needs κ < 0. That is a ghost with a repulsive force, α² = −0.313 [DERIVED, B7a, B7b].
   - With a σR coupling ζ > f, the same sign is **ghost-free and attractive at σ ≈ 0** [DERIVED, B7d, C1].
   - That holds only near σ = 0. Around a background σ₀, σ's kinetic coefficient is K = κ + 6ζ²/F(σ₀) for the memo's linear F = 1 + 2ζσ [DERIVED, C3, with negative controls].
   - With κ < 0, K turns negative (ghost) above σ = 0.091 at ζ = 2f. That is below the horizon value 0.158. At the tuned corner ζ = f + 10⁻⁴ the threshold is σ = 5.4 × 10⁻⁵, which is already reached at the G29-38 white-dwarf photosphere (σ = 6.0 × 10⁻⁵) [VERIFIED numerics; root by Mentat `solve_equation`].
   - With F = e^{2ζσ}, K = κ + 6ζ²e^{2ζσ₀}. That stays healthy for σ > 0 and turns ghost for σ < −0.078 (ζ = 2f) or σ < −5.4 × 10⁻⁵ (tuned corner).
   - So the answer depends on the nonlinear completion of F, which is a further choice.
5. **The postulate as written cannot come from any action** [DERIVED, F1]. It has a universal source and a non-universal response, so the scalar forces fail Newton's third law: F_A + F_B = 2ξG m_A m_B (f_B − f_A)/r² ≠ 0. Every action makes the source charge equal the response charge, β_i = f_i − ζ [C1].
6. **A mass coupling produces the postulate's trace source by itself** [DERIVED, E1, E2; E2 uses the T = 0 relation P = n∂ε/∂n − ε]. σT counts only the QCD share, so electron degeneracy pressure does not de-source σ. σR counts the full trace.
7. **Covariant and flat □ agree at linear order only** [BY CONSTRUCTION, D2]. That covers clocks, MICROSCOPE, Eöt-Wash, white dwarfs and Cassini γ. Lunar ranging needs PPN β, which is a 1PN quantity at cubic order, and the pulsar rows are strong-field; D2 says nothing about either. The flat box has no consistent 1PN completion to compute them (an argument, not a sympy check).
8. **Range (C5).**
   - MICROSCOPE requires λ < 34.5 km. That is **necessary, not sufficient**.
   - Below 34.5 km the signal at the ground falls only algebraically, as ~1.5λ/R⊕ (8.1 × 10⁻³ at 34.5 km). It does not fall exponentially, as it does at MICROSCOPE's 710 km altitude.
   - A smooth-Earth indicator puts the §6 Eöt-Wash Be–Ti signal at **1.0 × 10⁷ times its bound at λ = 34.5 km**. That is still 1.0 × 10⁶ with a 10× lower near-surface density, and the indicator reaches 1 only near λ ≈ 3 mm to 3 cm [VERIFIED numerics, an indicator, not a bound].
   - Separately, every option with a mass response predicts a near-universal Yukawa of strength α² = 0.16–0.46. [LIT] Inverse-square-law tests limit a gravitational-strength (|α| = 1) Yukawa to λ < 38.6 μm (Lee et al. 2020).
   - **The range that survives is therefore far shorter than 34.5 km and is not mapped here.** The repo's paper point, λ = 0.909 fm, is invisible to every test considered.
9. **The one bound where σ's sign matters is the white-dwarf H₂ test.** The healthy σT sign agrees with both stars (−0.04σ for G29-38, −0.23σ for GD133). The repo's sign is in 3.1σ tension with G29-38. This decides nothing, because the clocks exclude both signs by 10⁵.

![Fig. A2 v2](a2_sigma_action_options_v2.png)

---

## 1. Conventions and the one Lagrangian with four dials

- **Conventions**, fixed in B0 by recovering Newton's attraction and γ = 1:
  - MTW signature (−,+,+,+) with MTW's Riemann sign, so R = +8πG(ε − 3P)/c⁴ for ordinary matter in GR.
  - Weak field ds² = −(1 + 2Φ/c²)c²dt² + (1 − 2Ψ/c²)dx².
- **The family** (Jordan frame for leptons and photons, M = c⁴/8πG):

      L = √−g [ (M/2) F(σ) R  −  (M/2) κ g^{μν}∂_μσ ∂_νσ  −  (M/2) m² σ² ]  +  L_m[g; m_i(σ) = m_i⁰(1 + f_i σ)],   F = 1 + 2ζσ (or e^{2ζσ})

  - **σT** (charge f_i) is the mass response; for dust it is exactly a σ·T term [BY CONSTRUCTION, E3]. f_μ = f_p = 0.9904, and f ranges from 0.9844 (Pt-195) to 0.9898 (H-1) [repo `inventory/core/sigma.py`; binding energies from Mentat `nuclear_binding_energy`].
  - **σR**: the coupling ζ.
  - **κ**: the kinetic sign and normalisation.
  - **m²**: the range.
- **Field equation:**

      κ_E □σ − m²σ = −(8πG/c⁴)[ ζ (ε − 3P)_all − Σ_i f_i (ε_i − 3P_i) ],   κ_E ≡ κ + 6ζ²   (at σ ≈ 0)

  In the static weak field, σ = 2ξ_eff G M/(r c²) with **ξ_eff = (ζ − f)/κ_E** [DERIVED, B1]. Here G is the action's bare G; §3b covers the observed one.
- **Results for every option:**

  | result | formula | check |
  |---|---|---|
  | ghost at σ = 0 iff | κ_E = κ + 6ζ² < 0 (the naive κ fails the same test) | C1, C0 |
  | ghost around a background σ₀ iff | K(σ₀) < 0, with K = κ + 6ζ²/F₀ (linear F) or κ + 6ζ²F₀ (exponential F); the σ₀-independent coefficient fails the test | C3, C3 controls |
  | Einstein-frame charge | β_i = f_i − ζ; α_i = β_i √(2/κ_E) | C1, G3 |
  | fifth force | G_AB/G = 1 + α_Aα_B with α_Aα_B = 2(f_A − ζ)(f_B − ζ)/κ_E; like bodies attract iff κ_E > 0 | B2 |
  | PPN γ | γ − 1 = −2α_Aα_B/(1 + α_Aα_B). Light sees exactly GR's Φ + Ψ, with or without a mass term | B3, B4, B4m |
  | PPN β | β − 1 = ½β₀α²/(1 + α²)² (DEF 1992), η_N = 4(β − 1) − (γ − 1) | G4, G5 |
  | Brans–Dicke | pure σR: ω = κ/(4ζ²) | G1, G2 |
  | massive σ | Newton plus a Yukawa term of the massless amplitude; λ² = κ_E/m²; Yukawa iff m²/κ_E > 0, otherwise cos(r/λ)/r | B8, B9 |
  | composition tests | η(B,C) = −2ξ_eff Δf and (c²/μ)dμ/dΦ_N = −2ξ_eff f_μ | B5, B6: **[BY CONSTRUCTION]**, true for any σ(k) |

---

## 2. The options

Numbers use ξ = 0.1582 in the field reading (|ξ_eff| = ξ, bare G) unless noted.

| ID | Lagrangian (beyond EH and matter) | static field equation | frame | ghost? | σ near matter | fifth force on matter | what it keeps / loses |
|---|---|---|---|---|---|---|---|
| **O1** σT, healthy | κ = +f/ξ = 6.26 | ∇²σ = +(8πGξ/c²)ρ (the postulate with the source sign flipped) | Einstein; non-universal QCD-scale dilaton (Damour–Donoghue d_g-type) | no, for any σ (K = κ > 0) | **σ < 0** | attractive, α² = +0.313 | keeps an action, health, the QCD-share thesis and the trace source. Loses the repo's sign: Λ_eff falls near matter, and σ_conv = +1.84 is not reached by collapse |
| **O2** σT, ghost | κ = −f/ξ = −6.26 | ∇²σ = −(8πGξ/c²)ρ (the postulate exactly) | Einstein; non-universal | **yes**, for any σ (ω = −3.10) | σ > 0 (repo sign) | **repulsive**, α² = −0.313 | keeps the literal repo equation and sign. Loses stability: negative-energy quanta and negative-energy σ radiation |
| **O1n** σT, healthy, ξ/2 | κ = 2f/ξ = 12.5 | σ = −ξGM/rc² | Einstein | no | σ < 0 | attractive, α² = 0.157 | the C2 alternative: every composition prediction halves |
| **O3** σR only | F(σ)R − κ(∂σ)²; matter minimal | κ□σ = −ζR literally; ξ_eff = ζ/(κ + 6ζ²) | Jordan; universal (Brans–Dicke) | no iff K > 0 (κ > 0 at both points shown) | sign of ζ | attractive, α² = 2ζξ_eff | keeps □σ = −ξR literally and health. **Loses the mass response.** Cassini forces ζ ≤ 1.06 × 10⁻⁴ (ω ≥ 1.5 × 10⁴) |
| **O4** σR + σT, ζ > f | ζ = 2f: κ = −17.3, κ_E = 6.26, ω_J = −1.10 | κ_E∇²σ = −(8πG/c²)(ζ − f)ρ | Jordan for leptons; mixed | **no at σ ≈ 0; yes for σ > 0.091 with linear F** (σ > 5.4 × 10⁻⁵ at ζ = f + 10⁻⁴). With e^{2ζσ}: healthy for σ > 0, ghost for σ < −0.078 [C3] | **σ > 0 (repo sign)** | attractive near σ = 0, α² = 2(ζ − f)ξ | keeps the repo's sign, the mass response, and health in the weak field. A horizon (σ = 0.158) lies inside the ghost region for the linear F. Tuning ζ → f⁺ gives κ_E → 0: strong coupling, ω_J → −3/2 |
| **O5** cancellation corner | σR + σT, ζ = f(Fe-56) = 0.9863, ζ/κ = 0.1582 | ξ_eff,i = (ζ − f_i)/κ_E: −3.0 × 10⁻⁴ (H), −1.1 × 10⁻⁴ (O), −1.3 × 10⁻⁴ (C) | mixed | no (κ > 0) | composition-dependent | attractive, α² ≈ 10⁻⁶ | keeps 0.158 as the literal coefficient ζ/κ. Fine-tuned, and still excluded (§3) |
| **O6** flat box (any parent) | √−g g^{μν} → η^{μν} in the kinetic term | identical to the parent at linear order [D2, by construction] | none (non-covariant) | as its parent | as its parent | as its parent | simplest code. Equals the parent for linear-order bounds only. β (lunar ranging) and the strong-field pulsar terms are **not determined**: σ's energy does not gravitate, so ∇_μT^{μν} ≠ 0 against the Bianchi identity (an argument) |
| **O7** massive (healthy parent) | + −(M/2)m²σ², m²/κ_E > 0 | κ_E(∇² − λ⁻²)σ = source; λ = √(κ_E/m²) | as its parent | as its parent | as its parent | the parent's force × e^{−r/λ}(1 + r/λ) | C5. Per-bound values in §3. MICROSCOPE needs λ < 34.5 km; the finite-range Eöt-Wash indicator still excludes 34.5 km by 10⁷ |
| **O8** massive ghost | κ_E < 0 with m² < 0 (Yukawa), or m² > 0 (oscillating) | (∇² − λ⁻²)σ or (∇² + λ⁻²)σ | as its parent | yes | as its parent | repulsive Yukawa, or an unscreened cos(r/λ)/r | Yukawa branch: magnitudes as O7 on O2. Oscillating branch: O2's massless values as an envelope only (not computed per test) |

---

## 3. Consequences for every §6 bound

The table gives prediction ÷ 2σ bound, in the bare-G reading; above 1 means excluded. The bound convention is the program's: |central| + 2σ, two-sided.
- Leaf values (G, c, ħ, eV, au, M☉, R☉, R⊕, binding energies) come from Mentat; sources are in `a2_results_v2.json` → `mentat_sources`.
- **O5** uses each test's own source: the Sun (H-1) for clocks, Cassini and lunar ranging; the Earth (O-16 proxy) for MICROSCOPE and Eöt-Wash; a white-dwarf C/O core (C-12 proxy) for the H₂ test.
- **O7** is evaluated at λ = 34.5 km on an O1 parent. O2 and O4 parents give the same magnitudes.

| bound [LIT] | O1 σT healthy | O2 σT ghost | O1n ξ/2 | O3 ζ = 1 | O3 ζ = 1.06e−4 | O4 ζ = 2f | O4 ζ = f + 10⁻⁴ | O5 (per-test source) | O7 at λ = 34.5 km |
|---|---|---|---|---|---|---|---|---|---|
| clocks (c²/μ)dμ/dΦ < 9.7e−7 (Lange 2021) | **3.2e5** | **3.2e5** | **1.6e5** | 0 (no mass response) | 0 | **3.2e5** | **3.2e5** | **6.1e2** | 0 (e^{−AU/λ}) |
| MICROSCOPE η(Ti,Pt) < 7.0e−15 (Touboul 2022) | **9.4e10** | **9.4e10** | **4.7e10** | 0 | 0 | **9.4e10** | **9.4e10** | **6.6e7** | 1.0 (the edge, by construction) |
| Eöt-Wash η(Be,Ti) < 3.9e−13 (Schlamminger 2008), λ = ∞ | **1.2e9** | **1.2e9** | **6.2e8** | 0 | 0 | **1.2e9** | **1.2e9** | **8.8e5** | finite-range indicator **1.0e7** (not a bound, §4) |
| white-dwarf H₂ G29-38, \|Δμ/μ\| < 1.34e−4 (Bagdonaite 2014) | 0.44 | 0.44 | 0.22 | 0 | 0 | 0.44 | 0.44 | 4e−4 | ≤ 0.44 |
| same, **signed** tension, G29-38 / GD133 | −0.04σ / −0.23σ | **3.1σ** / 1.4σ | 0.7σ / 0.2σ | 1.5σ / 0.6σ (no σ) | same | **3.1σ** / 1.4σ | **3.1σ** / 1.4σ | 1.5σ / 0.6σ | ≤ massless |
| Cassini γ − 1 < 6.7e−5 (Bertotti 2003) | **7.1e3** (γ−1 = −0.477) | **1.4e4** (γ−1 = +0.913) | **4.0e3** | **7.2e3** | 1.0 (edge, by construction) | **7.1e3** | 0.94 | 0.06 | 0 [B4m] |
| ω (Einstein-frame equivalent, (1/α² − 3)/2) | 0.096 | −3.10 | 1.69 | 0.080 | 1.5e4 | 0.096 | 1.6e4 | 2.4e5 | n/a |
| lunar ranging, **Nordtvedt part only**, η_N < 2.4e−4 (Hofmann & Müller 2018) | **2.0e3** | **3.8e3** | **1.1e3** | **2.0e3** | 0.28 | **1.3e3** | **8.8e2** (β₀ = 3.3e3) | 0.02 | 0 |
| lunar ranging, **composition part**, Δ(m_g/m_i)_EM < 1.3e−13 (same paper) | excluded unless the Earth–Moon iron-fraction difference is < 3.8e−10 | same | < 7.6e−10 | 0 | 0 | < 3.8e−10 | < 3.8e−10 | < 2.0e−7 | 0 |
| pulsar dipole \|Δα\| < 2e−3 (Freire 2012, as summarised in Wex 2014) | **not computed**: the composition part alone gives 0.04, a lower bound. Program §6: ≥ 10³ with neutron-star sensitivities | same (0.04) | same (0.03) | 0 | 0 | same (0.04) | **≥ 4.2** | lower bound 0.03 | 0 (no radiation below c/λ) |
| range λ for MICROSCOPE at this ξ_eff | < 34.5 km | < 34.5 km | < 35.7 km | n/a | n/a | < 34.5 km | < 34.5 km | not computed | — |

**Reading the table.**

- **The top three rows do not change across O1, O2 and O4 in the bare-G reading.** That is by construction (B5, B6). Against the observed potential they move by 1/(1 + α²): see §3b.
- **The universal rows are what the action moves.**
  - The ghost flips γ − 1 to +0.91.
  - Tuning ζ → f in O4 passes Cassini (0.94) but fails lunar ranging (×8.8 × 10², or ×7.8 × 10² with F = e^{2ζσ}, because β₀ ∝ 1/κ_E). It also fails the pulsar composition dipole (×4.2).
- **O3 at ζ ≤ 1.06 × 10⁻⁴ passes every row.** It has no mass response, so it has nothing to show.
- **O5 cannot cancel.**
  - The reason: |f_H − f_Fe| = 3.6 × 10⁻³, so no single ζ cancels both an H-rich Sun and an O/Fe-rich Earth.
  - The Sun-sourced clocks see it ×6.1 × 10². The Earth-sourced MICROSCOPE sees it ×6.6 × 10⁷.
  - v1 quoted 1.8 × 10⁸ for MICROSCOPE, using the wrong (solar) source.
- **The composition part of lunar ranging is far stronger than the Nordtvedt part** for every option with a mass response.
  - The prediction is η_EM = 2|ξ_eff| Δx_Fe |f_Fe − f_silicate|. Here f_Fe − f_MgSiO₃ = −1.09 × 10⁻³, using stoichiometric MgSiO₃ as the silicate proxy.
  - So the option is excluded unless the Earth's and Moon's iron mass fractions agree to 4 × 10⁻¹⁰.
  - The Earth has a large metallic core and the Moon a small one. No citation for the core fractions was verified in this run, so no exclusion factor is quoted.
- **The pulsar row is a lower bound, not a pass.** Neutron-star strong-field sensitivities are not computed here. Program §6 puts the full dipole at ≥ 10³.

### 3b. Which G normalises ξ (all options, `a2_results_v2.json` → `v2.normalisation_table`)

The table above defines ξ_eff against the action's bare G. Every composition test, however, is normalised to an observed potential or acceleration: GM☉ from ephemerides, g at MICROSCOPE and Eöt-Wash, the white dwarf's observed mass. The observed value is (1 + α²) times the bare one. Two consistent readings follow:

| option | (a) bare ξ_eff = 0.1582: clocks / MICROSCOPE vs the observed potential | (b) ξ = 0.1582 as the **observed** amplitude (the repo's σ = 2ξGM/rc² with CODATA G): α², γ − 1, Cassini, lunar ranging (Nordtvedt) | (b) composition |
|---|---|---|---|
| O1 σT healthy | ×2.5e5 / ×7.2e10 (factor 0.76) | α² = 0.456, γ − 1 = −0.627, ×9.4e3, ×2.6e3 | ×3.2e5 / ×9.4e10 |
| O2 σT ghost | ×4.7e5 / ×1.4e11 (factor 1.46) | α² = −0.239, γ − 1 = +0.627, ×9.4e3, ×2.6e3 | ×3.2e5 / ×9.4e10 |
| O1n ξ/2 | ×1.4e5 / ×4.1e10 | α² = 0.186, γ − 1 = −0.313 | ×1.6e5 / ×4.7e10 |
| O4 ζ = 2f | ×2.5e5 / ×7.2e10 | α² = 0.456, γ − 1 = −0.627, ×9.4e3, ×1.4e3 | ×3.2e5 / ×9.4e10 |
| O4 ζ = f + 10⁻⁴ | ×3.2e5 / ×9.4e10 (factor ≈ 1) | unchanged (α² ≈ 3 × 10⁻⁵) | ×3.2e5 / ×9.4e10 |

**No verdict changes in either reading.** In (b), O1 has an Einstein-frame-equivalent ω = −0.40; it is still healthy, because κ_E = 4.30 > 0.

---

## 4. The range axis (C5) and the normalisation axes (C2, C3)

| ξ_eff | 0.13 | 0.0791 (inventory reading) | 0.1571 | 0.1582 | 0.17 | 0.3164 |
|---|---|---|---|---|---|---|
| MICROSCOPE λ_max (km) | 34.85 | 35.68 | 34.54 | 34.53 | 34.42 | 33.45 |

- **C3 (0.1582 vs 0.1571)** moves λ_max by 11 m. **C2 (the factor 2)** moves it by 1.1 km.
- **This limit is necessary, not sufficient.** At 710 km altitude the Yukawa signal is suppressed as e^{−710 km/λ}. At the ground it is suppressed only as ~1.5λ/R⊕ [VERIFIED numerics: the uniform-ball form factor at r = R⊕ gives 8.12 × 10⁻³ at λ = 34.5 km, equal to 1.5λ/R⊕].
- The **§6 Eöt-Wash Be–Ti bound toward Earth**, with that same form factor, gives prediction ÷ bound at λ = 34.5 km of:
  - **1.0 × 10⁷** for a uniform Earth;
  - 5.0 × 10⁶ for half the near-surface density;
  - 1.0 × 10⁶ for a tenth.

  The indicator falls to 1 only at λ ≈ 3.4 mm (uniform) or 3.4 cm (a tenth of the density).
- **This is an indicator, not a bound.** A torsion balance senses the horizontal part of the field, which local topography sets when λ ≪ R⊕. [LIT] Schlamminger et al. 2008 report equivalence-principle limits for ranges from 1 m to ∞, so below 1 m their experiment says nothing. The indicator's margin (10⁶–10⁷) is large enough that λ = 34.5 km is very unlikely to escape. Where exactly the escaping range begins is **not mapped**.
- **Inverse-square-law tests [LIT]:** Lee et al. 2020 limit any gravitational-strength (|α| = 1) Yukawa to λ < 38.6 μm (95 %). Every option with a mass response carries a near-universal Yukawa of strength α² = 0.16–0.46. The limit for that strength is not read off the published curve here; it is presumably of the same order [SPECULATIVE inference].
- **The paper point** λ = ħc/Λ_QCD = 0.909 fm (Λ_QCD = 217 MeV from `field.constants`) is invisible to every test considered.
- The mass term must have the same sign as κ_E [B9]; otherwise σ does not screen at all.

---

## 5. Mapping the options to decisions C1–C6

| decision | what the action does to it | options bearing on it | trade-off |
|---|---|---|---|
| **C1** where the SSBM constants live | The parameters are (κ, ζ, m², completion of F) plus the derived f_i. ξ becomes derived: ξ_eff = (ζ − f)/κ_E, or the literal ζ/κ in O3 and O5 | O1/O2: inventory f_i and κ. O3: ζ and κ only. O4/O5: both | σT ties the field equation to Quarksum's mass decomposition. σR keeps it in `field/` but drops the mass response. A mixed action needs one module both import |
| **C2** σ = 2ξGM/rc² vs ξGM/rc² | two values of κ (f/ξ vs 2f/ξ). Also fixes which G normalises ξ (§3b) | O1 vs O1n | factor 2 on every composition prediction, both still excluded. α² 0.313 vs 0.157. λ_max 34.5 vs 35.7 km |
| **C3** ξ = 0.1582 vs 0.1571 | rescales ξ_eff by 0.7 % | all | no verdict changes |
| **C4** covariant or flat □ | identical at linear order [D2, by construction]; they differ at 1PN [D1] | O6 vs the rest | flat is enough for clocks, MICROSCOPE, Eöt-Wash, white dwarfs and Cassini. It cannot give β (lunar ranging), the pulsar terms, compact objects (P7) or the background-σ ghost test (C3). Covariant carries (1 + 2γΦ/c²) and the Tolman source |
| **C5** σ's range λ | needs m²/κ_E > 0 for a Yukawa [B8, B9] | O7 (healthy), O8 (ghost) | MICROSCOPE requires λ < 34.5 km. Eöt-Wash (1 m to ∞) still excludes 34.5 km on the indicator, and inverse-square-law tests bound the universal part. The surviving range is unmapped; 0.91 fm is invisible |
| **C6** engine defaults (A2 `total_energy`, A5 Eyring, A7 SPH kernel) | none | none | independent of this memo |
| **B5** (Ricci sign, filed as a capability) | settled by any action: the sign convention becomes the sign of ζ. The physical statement is sign(ξ_eff·f) | all | O1: masses fall near matter. O2, O4: masses rise |

**Decisions this memo surfaces that C1–C6 do not list** (proposed additions, not choices):

- **C7: the kinetic sign** (ghost or not). It is tied to σ's sign near matter [B7].
- **C8: σT, σR or both.** This decides whether the mass response survives and whether the repo's sign can be ghost-free.
- **C9 (new): the nonlinear completion of F(σ).** With σR + σT at the repo's sign, the linear F gives a ghost at σ = 0.09 or below. F = e^{2ζσ} stays healthy for σ > 0 [C3].

---

## 6. The trade-offs in one place

| if the Captain wants… | then | price |
|---|---|---|
| the repo's equation literally (□σ = −ξR_GR with mass response) | O2 | a ghost and a repulsive fifth force; excluded at long range |
| the repo's sign without a ghost | O4 (ζ > f) plus a completion choice (C9) | a second coupling ζ. With linear F the ghost returns at σ = 0.09 (before a horizon); with e^{2ζσ} it does not for σ > 0. Excluded at long range either way |
| no ghost and the simplest action | O1 | σ's sign flips (σ < 0 near matter); excluded at long range |
| □σ = −ξR literally with no ghost | O3 | the mass response disappears; σ is unobservable once Cassini is met |
| ξ ≈ 0.16 kept in some form | O5 (as ζ/κ) or O7 (short range) | O5 is fine-tuned and still excluded ×6 × 10². O7 needs λ well below 34.5 km (unmapped; Eöt-Wash indicator and inverse-square-law tests). Its natural paper point is invisible |
| any long-range σ that changes masses | \|ξ_eff\| < 1.7 × 10⁻¹² (MICROSCOPE) | σ stops being "the scale of matter" at any observable level |

**Effect on the program's other items.**

- **B1 (nuclear matter):** σT and σR source σ differently in degenerate matter. They agree for nucleon matter [E1, E2] and disagree on electrons.
- **B4 (σ-wave solver):** every covariant option keeps monopole σ radiation sourced by pressure.
- **P7 (compact objects):** needs C4 = covariant and a C9 choice. Neutron-star σ values, of order ξ r_s/R ≈ a few hundredths, are not computed here because no radius was verified in this run [SPECULATIVE estimate]. They would lie above the tuned corner's onset (5.4 × 10⁻⁵) and below the onset for ζ = 2f (0.091).

---

## 7. Honesty

- **Which checks could have failed?** Of the 44 checks:
  - **26 could fail on physics grounds:**
    - B0a, B0b, B1–B4, B4m, B8a, B8b: the variational solve;
    - C1 ×3 and C3 ×6: the Weyl-shift decoupling, at σ = 0 and around a background;
    - D1: the covariant box;
    - E1, E2: the trace identities;
    - F1: reciprocity;
    - G2–G5: cross-route consistency.
  - **3 are negative controls** that must come out nonzero, and do: C0 rejects the naive κ, and the C3 controls reject a σ₀-independent coefficient.
  - **8 are algebraic substitutions or textbook identities** that could fail only through a typo: B7a–B7e, G1, B9a, B9b.
  - **6 are tautologies:** B5, B6 and their arbitrary-σ demonstration, D2, E3, F2.
  - **1 is a summary record:** C2.

  None of the 11 pre-registered predictions failed, and none was moved. The listing per check is in `a2_results_v2.json` → `checks[].can_fail`.
- **Evidential weight.**
  - **What this is.** These are general facts of scalar–tensor theory (DEF, Damour–Donoghue), re-derived on the repo's family [DERIVED]. They hold for any values of the constants, so they are engine-grade structure, not evidence for or against the unification thesis.
  - **Decision-relevant [DERIVED] content:**
    - the postulate's third-law failure (F1);
    - the ghost-free repo sign at ζ > f near σ = 0 (C1), and its loss at a finite σ that depends on the completion (C3);
    - the σT trace-source identity (E1, E2);
    - the failure of the O5 corner;
    - the correction of the range claim.
  - **Not new:** the composition invariance is [BY CONSTRUCTION] and restates program §6.
  - **Exclusion numbers.** They are [DERIVED] from [LIT] bounds and reproduce §6 in the bare-G reading: clocks 3.23 × 10⁵, MICROSCOPE 9.43 × 10¹⁰, α² 0.313, γ − 1 −0.477 / +0.913, ω 0.096, η_N 0.478.
- **Approximations.**
  - Weak field plus the DEF 1PN β; no neutron-star sensitivities, so the pulsar row is a lower bound.
  - One isotope per element. The alloy fractions, the MICROSCOPE altitude (710 km) and the white-dwarf φ are [LIT approx], as encoded in `sigma_exclusion.py`.
  - O5 sources are single-isotope proxies. The silicate proxy is stoichiometric MgSiO₃.
  - The Eöt-Wash finite-range number is a smooth-Earth indicator, not a bound.
  - The flat-box inconsistency is an argument.
- **Engine defects.** None found. Nothing in `sigma_ground` was modified.

## 8. Design corrections

1. **Program §6 ("a canonical scalar … cannot do [this]; either σ is a ghost or the field equation has no action") is incomplete, but only locally.**
   - With σR and ζ > f, the repo's sign is ghost-free at σ ≈ 0. The criterion is κ + 6ζ², not κ [B7d, C1].
   - Around a background σ₀ the criterion becomes K(σ₀) > 0 [C3]. With the linear F that fails above σ = 0.091 (ζ = 2f), so the ghost returns before a horizon. With F = e^{2ζσ} it does not, for σ > 0.
   - The long-range exclusion is unaffected either way.
2. **The item's pairing "□σ = −ξR with a perfect-fluid source" + "m_i ∝ 1 + f_iσ" has no action** [F1].
3. **"Normalisation of ξ" is not an independent axis once there is an action.** It is a value of κ, plus a choice of which G normalises ξ (§3b).
4. **"Frame" has no answer for a flat □** (O6).
5. **v1's range claim was wrong.** "λ < 34.5 km escapes all §6 bounds" came from MICROSCOPE alone. The finite-range Eöt-Wash indicator and inverse-square-law tests leave the surviving range far shorter and unmapped (§4).
6. **The pre-registered P5 printed +0.911** for the ghost γ − 1, from α² rounded to 0.313. The exact α² = 0.31336 gives +0.9127. P5's criterion is the symbolic identity B3, which passes; the printed expectation was a rounding, recorded here and not moved.
7. **Pre-registered P6 is true by construction** at fixed bare ξ_eff (B5B6 demonstration). Against observed potentials the composition coefficients carry 1/(1 + α²), which depends on κ and ζ. "No action choice moves them" holds exactly only at fixed observed amplitude (§3b).
8. **P1 (ghost iff κ + 6ζ² < 0) is the criterion at σ = 0.** P8's linear-order equality (D2) is a tautology covering linear-order bounds only.
9. **Against the pre-registered DELIVERED criterion** ("every option row has … a value for every §6 bound"), v2 fills O7 (at λ = 34.5 km) and O5 (per-test source). **Four gaps remain, stated rather than hidden:**
   - the pulsar row is a lower bound for every option (no neutron-star sensitivities);
   - O6's lunar-ranging and pulsar cells are undetermined;
   - O8's oscillating branch has only an envelope;
   - the finite-range Eöt-Wash cell is an indicator, not a bound.

   The outcome is reported as DELIVERED with these gaps.

## 9. References (verified 2026-10-10)

- C. Brans & R. H. Dicke 1961, Phys. Rev. 124, 925, doi:10.1103/PhysRev.124.925
- T. Damour & G. Esposito-Farèse 1992, Class. Quantum Grav. 9, 2093, doi:10.1088/0264-9381/9/9/015
- T. Damour & J. F. Donoghue 2010, Phys. Rev. D 82, 084033, arXiv:1007.2792
- Y. Fujii & K. Maeda 2003, *The Scalar-Tensor Theory of Gravitation*, CUP, doi:10.1017/CBO9780511535093
- R. Lange et al. 2021, PRL 126, 011102, arXiv:2010.06620. (c²/μ)dμ/dΦ = 7(45) × 10⁻⁸
- P. Touboul et al. 2022, PRL 129, 121102, arXiv:2209.15487. η(Ti,Pt) = [−1.5 ± 2.3 ± 1.5] × 10⁻¹⁵
- J. Bagdonaite et al. 2014, PRL 113, 123002, arXiv:1409.1000. GD133 (−2.7 ± 4.7 ± 0.2) × 10⁻⁵; G29-38 (−5.8 ± 3.8 ± 0.3) × 10⁻⁵
- B. Bertotti, L. Iess & P. Tortora 2003, Nature 425, 374, doi:10.1038/nature01997. γ − 1 = (2.1 ± 2.3) × 10⁻⁵
- F. Hofmann & J. Müller 2018, CQG 35, 035015, doi:10.1088/1361-6382/aa8f7a. η_N = (−0.2 ± 1.1) × 10⁻⁴; Δ(m_g/m_i)_EM = (−3 ± 5) × 10⁻¹⁴ (abstract, via Crossref)
- S. Schlamminger et al. 2008, PRL 100, 041101, arXiv:0712.0607. η(Be,Ti) = (0.3 ± 1.8) × 10⁻¹³; equivalence-principle limits for ranges from 1 m to ∞
- J. G. Lee, E. G. Adelberger, T. S. Cook, S. M. Fleischer & B. R. Heckel 2020, PRL 124, 101101, arXiv:2002.11761. Gravitational-strength Yukawa limited to λ < 38.6 μm (95 %)
- P. C. C. Freire et al. 2012, MNRAS 423, 3328, arXiv:1205.1450. |α_p − α_c| < 2 × 10⁻³ (95 %), as summarised in N. Wex 2014, arXiv:1402.5594

## 10. Reproduce

    cd /d/Aaron/development/sigma-ground
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -X utf8 -B <A2>/v2_build/patch_v2.py        # rebuilds a2_sigma_action_v2.py from v1 + part_I.py
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -X utf8 -B <A2>/a2_sigma_action_v2.py        # ~17 s, 44 checks, writes a2_results_v2.json
    python -X utf8 -B <A2>/a2_figure_v2.py                                                                          # writes a2_sigma_action_options_v2.png

`<A2>` = `C:/Users/aaron/AppData/Local/Temp/claude/D--Aaron-development-sigma-ground/c6652910-47a9-478d-9c83-a8a8c8aaa457/scratchpad/round2/A2`.
