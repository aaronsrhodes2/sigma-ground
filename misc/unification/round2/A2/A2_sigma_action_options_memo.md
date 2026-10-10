# Decision memo A2: options for a σ action (Captain's decisions C1–C6)

**For:** the Captain. **Scope:** decisions C1–C6 in `D:/Aaron/development/sigma-ground/misc/MENTAT_GAPS_UNIFICATION.md` §C, and the program's
next step 2 ("Give σ an action") in `D:/Aaron/development/sigma-ground/misc/UNIFICATION_PROGRAM.md` §14.
**This memo does not choose.** It lists every defensible action and gives the price of each.
**Checks:** `a2_sigma_action.py` runs 34 checks: 32 sympy identities, 1 negative control (C0) and 1 summary record (C2) that follows from C1. All 34 pass; output is in `a2_results.json`.
The predictions were fixed in `prereg.json` before the script ran. **Figure:** `a2_sigma_action_options.png`.
**Tags:** [DERIVED] means sympy-certified, with the check ID in brackets. [MEASURED] or [LIT] marks a cited measurement, each citation verified by web search on 2026-10-10.
[BY CONSTRUCTION] means the result follows from a definition. [SPECULATIVE] marks content outside the derivations.

---

## 0. Summary

1. **No choice of action rescues a long-range σ with |ξ| ≈ 0.16.** [DERIVED, B5 and B6] For every value of the kinetic sign, the frame, and the σR/σT mix, the clock, MICROSCOPE, Eöt-Wash and white-dwarf predictions depend on only two things:
   - the amplitude of σ near matter, ξ_eff, defined by σ = 2ξ_eff GM/(rc²);
   - the composition differences Δf.

   (c²/μ)dμ/dΦ = −2ξ_eff f_μ and η = −2ξ_eff Δf, with no κ or ζ left in either expression. At |ξ_eff| = 0.1582 the clocks exclude every massless option by 3.2 × 10⁵, and MICROSCOPE by 9.4 × 10¹⁰. Only two decisions move these bounds:
   - the range λ (C5): λ < 34.5 km escapes all §6 bounds;
   - giving up |ξ_eff| ≈ 0.16 (MICROSCOPE requires |ξ_eff| < 1.7 × 10⁻¹²).
2. **The action decides what remains:** whether σ is a ghost, the sign of the fifth force, and the universal tests (Cassini γ, Brans–Dicke ω, the lunar-ranging Nordtvedt η_N, and pulsar dipole radiation).
3. **The program's "repulsive" finding is confirmed for a pure mass coupling and corrected for a mixed one.** The repo's sign means σ rises toward matter while masses rise with σ.
   - With only the mass coupling (σT), that sign needs κ < 0. The result is a ghost with a repulsive fifth force (α² = −2ξf = −0.313) [DERIVED, B7a and B7b]. This confirms the program.
   - If σ also couples to curvature (σR) with ζ > f, the same sign is **ghost-free and attractive** [DERIVED, B7d and C1]. The Jordan-frame kinetic coefficient is then negative, but the physical (Einstein-frame) coefficient κ_E = κ + 6ζ² is positive. This is Brans–Dicke's −3/2 < ω < 0 window: ω_J = −1.10 at ζ = 2f.
   - So the program's "ghost or no action" leaves out a third route. That route is still excluded at long range by item 1.
4. **The postulate as written cannot come from any action** [DERIVED, F1]. It has a universal source (−ξR, the trace of all matter) and a non-universal response (f_i differs between bodies). The scalar forces then fail Newton's third law: F_A + F_B = 2ξG m_A m_B (f_B − f_A)/r² ≠ 0. Every action makes the source charge equal the response charge, β_i = f_i − ζ [DERIVED, F2 and C1].
5. **A mass coupling produces the postulate's trace source by itself** [DERIVED, E1 and E2]. For free particles with any momentum distribution, m ∂ε/∂m = ε − 3P. If Λ_QCD is the only scale, Λ ∂ε/∂Λ = ε − 3P for any equation of state. The two couplings differ in what they count:
   - σT counts only the QCD share, so electron degeneracy pressure (white dwarfs) does not de-source σ.
   - σR counts the full trace, electrons included.
6. **Covariant and flat □ are identical at linear order, so no §6 bound can tell them apart** [DERIVED, D2]. At first post-Newtonian (1PN) order the covariant one carries (1 + 2γΦ/c²)∇²σ + (1 − γ)∇Φ·∇σ/c² [DERIVED, D1]. Only the covariant one has an action consistent with Einstein's equations (argument in §3, not a sympy check).
7. **The normalisation question (C2) becomes a value of κ.** The factor 2 halves every composition prediction, which leaves them still excluded by 1.6 × 10⁵ and 4.7 × 10¹⁰. It moves the range limit from 34.5 km to 35.7 km. The ξ value (C3) moves everything by 0.7 %.
8. **The one bound where the sign of σ matters is the white-dwarf H₂ test.**
   - The healthy σT sign (σ < 0 near matter) agrees with both measurements, at −0.23σ for GD133 and −0.04σ for G29-38.
   - The repo's sign is in 3.1σ tension with G29-38.

   This does not decide anything, because the clocks exclude both signs by 10⁵.

![Fig. A2](a2_sigma_action_options.png)

---

## 1. Conventions and the one Lagrangian with four dials

- **Conventions** (fixed in B0 by recovering Newton's attraction and γ = 1): MTW signature (−,+,+,+) with MTW's Riemann sign, so R = +8πG(ε − 3P)/c⁴ for ordinary matter in GR. The weak field is ds² = −(1 + 2Φ/c²)c²dt² + (1 − 2Ψ/c²)dx².
- **The family**, written in the Jordan frame for leptons and photons, with M = c⁴/8πG:

      L = √−g [ (M/2)(1 + 2ζσ) R  −  (M/2) κ g^{μν}∂_μσ ∂_νσ  −  (M/2) m² σ² ]  +  L_m[g; m_i(σ) = m_i⁰(1 + f_i σ)]

  - **σT coupling** (charge f_i): the mass response. For dust it is exactly a σ·T term with charge f [DERIVED, E3]. The repo fixes f_i through Λ_eff = Λ_QCD e^σ. f_μ = f_p = 0.9904, and f ranges from 0.9844 (Pt-195) to 0.9898 (H-1) atom by atom [repo `inventory/core/sigma.py`; binding energies from Mentat `nuclear_binding_energy`].
  - **σR coupling** (ζ): non-minimal coupling to curvature.
  - **κ**: the sign and normalisation of the kinetic term.
  - **m²**: the mass of σ, which sets its range.
- **Field equation** (varying σ):

      κ(□ − m²/κ)σ = −ζ R + (8πG/c⁴) Σ_i f_i (ε_i − 3P_i)

  The trace of the metric equation gives R = R_GR + 6ζ□σ at linear order. Eliminating R:

      κ_E □σ − m² σ = −(8πG/c⁴) [ ζ (ε − 3P)_all matter − Σ_i f_i (ε_i − 3P_i) ],   κ_E ≡ κ + 6ζ²

  In the static weak field, σ = 2ξ_eff GM/(rc²) with **ξ_eff = (ζ − f)/(κ + 6ζ²)** [DERIVED, B1].
- **The postulate.** The program reads □σ = −ξR with R from GR. That is the case ξ_eff = ξ. In this family, ξ is not a parameter. It is derived from (κ, ζ, f).
- **Results that hold for every option** [DERIVED]:

  | result | formula | check |
  |---|---|---|
  | ghost iff | κ_E = κ + 6ζ² < 0. The linearised Weyl shift Φ = Φ̃ − ζσ, Ψ = Ψ̃ + ζσ decouples σ with kinetic energy (M/2)κ_E σ̇². The naive κ fails the same test (negative control) | C1, C0 |
  | Einstein-frame charge | β_i = f_i − ζ, and α_i = β_i √(2/κ_E) | C1, G3 |
  | fifth force | G_AB/G = 1 + α_Aα_B with α_Aα_B = 2(f_A − ζ)(f_B − ζ)/κ_E. Like bodies attract iff κ_E > 0 | B2 |
  | PPN γ | γ − 1 = −2α_Aα_B/(1 + α_Aα_B). Light sees exactly GR's Φ + Ψ (conformal invariance) | B3, B4 |
  | PPN β | β − 1 = ½β₀α²/(1 + α²)², with β₀ from the nonlinear map [DEF 1992 formula; β₀ certified in G4]. η_N = 4(β − 1) − (γ − 1) | G4 |
  | Brans–Dicke | pure σR is Brans–Dicke with ω = κ/(4ζ²). γ_BD = (1 + ω)/(2 + ω) is the same formula | G1, G2 |
  | massive σ | σ(k) = (ζ − f)ρ/[M(κ_E k² + m²)]: Newton plus a Yukawa term of the massless amplitude, with range λ² = κ_E/m². Yukawa e^{−r/λ}/r iff m²/κ_E > 0, otherwise cos(r/λ)/r | B8, B9 |
  | composition tests | η(B,C) = −2ξ_eff(f_B − f_C) and (c²/μ)dμ/dΦ = −2ξ_eff f_μ, for every κ and ζ | B5, B6 |

---

## 2. The options

Each row is a defensible point in the family. Numbers use the repo's ξ = 0.1582 and the field reading |ξ_eff| = ξ, unless noted.

| ID | Lagrangian (beyond EH and matter) | static field equation | frame | ghost? | σ near matter | fifth force on matter | what it keeps / loses |
|---|---|---|---|---|---|---|---|
| **O1** σT, healthy | −(M/2)κ(∂σ)², κ = +f/ξ = 6.26; masses m_i(1 + f_iσ) | ∇²σ = +(8πGξ/c²)ρ: the postulate with the source sign flipped | Einstein frame. Non-universal QCD-scale dilaton (Damour–Donoghue's d_g-type coupling) | no | **σ < 0** (σ = −2ξGM/rc²; at a horizon −ξ) | attractive, α² = +2ξf = 0.313 | keeps an action, health, the QCD-share thesis and the trace source. Loses the repo's sign (Λ_eff falls near matter; σ_conv = +1.84 is not reached by collapse) |
| **O2** σT, ghost | same, κ = −f/ξ = −6.26 | ∇²σ = −(8πGξ/c²)ρ: the postulate exactly (QCD-share trace) | Einstein; non-universal | **yes** (κ_E < 0; Brans–Dicke ω = −3.10 < −3/2) | σ > 0 (the repo's sign) | **repulsive**, α² = −0.313 | keeps the literal repo equation and sign. Loses stability: negative-energy quanta, negative-energy σ radiation (a ghost's dipole flux has the opposite sign) |
| **O1n** σT healthy, inventory normalisation | κ = 2f/ξ = 12.5 | σ = −ξGM/rc² | Einstein | no | σ < 0 | attractive, α² = 0.157 | the C2 alternative: every composition prediction halves |
| **O3** σR only | (M/2)(1 + 2ζσ)R − (M/2)κ(∂σ)²; matter minimal | κ□σ = −ζR literally; ξ_eff = ζ/(κ + 6ζ²) | Jordan; universal (Brans–Dicke) | no iff κ + 6ζ² > 0 | sign of ζ (the repo's sign for ζ > 0) | attractive, α² = 2ζξ_eff | keeps □σ = −ξR literally and health. **Loses the mass response**, so Λ_eff = Λ_QCD e^σ is gone and σ is seen only through the metric. Cassini forces ζ ≤ 1.06 × 10⁻⁴ (ω ≥ 1.5 × 10⁴ at this memo's 2σ convention) |
| **O4** σR + σT, ζ > f | both couplings; e.g. ζ = 2f, κ = −17.3, κ_E = 6.26 | κ_E∇²σ = −(8πG/c²)(ζ − f)ρ | Jordan for leptons; mixed | **no** (κ < 0 but κ_E > 0; ω_J = −1.10) | **σ > 0 (the repo's sign)** | **attractive**, α² = 2(ζ − f)ξ | keeps the repo's sign, the mass response and health. Composition bounds unchanged. The universal force can be tuned toward 0 as ζ → f⁺, but then κ_E → 0 (strong coupling, ω_J → −3/2) |
| **O5** cancellation corner | σR + σT with ζ = f̄ (f of Fe-56, 0.9863) and ζ/κ = 0.1582 | ξ_eff,i = (f̄ − f_i)/κ_E ≈ 10⁻⁴ | mixed | no (κ_E = 12.1) | composition-dependent sign | attractive, α² ≈ 10⁻⁶ | keeps the Captain's 0.158 as the literal coefficient ζ/κ. Gives up an observable σ near matter. Fine-tuned, and still excluded (§3) |
| **O6** flat box (any of the above) | √−g g^{μν} → η^{μν} in the kinetic term | identical at linear order [D2] | none (non-covariant; preferred frame) | as its parent | as its parent | as its parent | simplest code. No consistent action beyond linear order: σ's energy does not gravitate while σ does work on matter, so ∇_μT^{μν} ≠ 0 against the Bianchi identity [DERIVED, argument, not sympy] |
| **O7** massive (any healthy option) | + −(M/2)m²σ², m²/κ_E > 0 | κ_E(∇² − λ⁻²)σ = source; λ = √(κ_E/m²) | as its parent | as its parent | as its parent | its parent's force times e^{−r/λ}(1 + r/λ) | C5. λ < 34.5 km escapes §6. The repo's own paper point λ = ħc/Λ_QCD = 0.909 fm is allowed and invisible (σ = 6.5 × 10⁻⁴⁰ at nuclear density, program §6) |
| **O8** massive ghost | κ_E < 0 and m² < 0 (Yukawa), or m² > 0 | (∇² − λ⁻²)σ, or (∇² + λ⁻²)σ | as its parent | yes | as its parent | repulsive Yukawa, or an oscillating cos(r/λ)/r force | for completeness. A ghost Yukawa needs a negative-energy mass term. The other sign gives no screening at all |

---

## 3. Consequences for every §6 bound

The table gives prediction ÷ 2σ bound. A value above 1 means excluded. The bound convention is the program's: |central value| + 2σ, two-sided.
Leaf values (G, c, ħ, eV, au, M☉, R☉, R⊕) come from Mentat; sources are listed in `a2_results.json` under `mentat_sources`. f_i come from the repo's mass decomposition.

| bound [LIT] | O1 σT healthy | O2 σT ghost | O1n ξ/2 | O3 σR ζ = 1 | O3 σR ζ = 1.06e−4 | O4 ζ = 2f | O4 ζ = f + 10⁻⁴ | O5 (H-rich source) |
|---|---|---|---|---|---|---|---|---|
| clocks (c²/μ)dμ/dΦ < 9.7e−7 (Lange 2021) | **3.2e5** | **3.2e5** | **1.6e5** | 0 (no mass response) | 0 | **3.2e5** | **3.2e5** | **6.1e2** |
| MICROSCOPE η(Ti,Pt) < 7.0e−15 (Touboul 2022) | **9.4e10** | **9.4e10** | **4.7e10** | 0 | 0 | **9.4e10** | **9.4e10** | **1.8e8** (O-rich proxy 6.6e7) |
| Eöt-Wash η(Be,Ti) < 3.9e−13 (Schlamminger 2008) | **1.2e9** | **1.2e9** | **6.2e8** | 0 | 0 | **1.2e9** | **1.2e9** | **2.3e6** |
| white-dwarf H₂, G29-38, |Δμ/μ| < 1.34e−4 (Bagdonaite 2014) | 0.44 | 0.44 | 0.22 | 0 | 0 | 0.44 | 0.44 | 8e−4 |
| same, **signed**: tension with G29-38 / GD133 | −0.04σ / −0.23σ | **3.1σ** / 1.4σ | 0.7σ / 0.2σ | 1.5σ / 0.6σ (no σ) | same | **3.1σ** / 1.4σ | **3.1σ** / 1.4σ | 1.5σ / 0.6σ |
| Cassini γ − 1 < 6.7e−5 (Bertotti 2003) | **7.1e3** (γ−1 = −0.477) | **1.4e4** (γ−1 = +0.913) | **4.0e3** | **7.2e3** | 1.0 (edge, by construction) | **7.1e3** | 0.94 | 0.06 |
| Brans–Dicke ω (Einstein-frame equivalent, ω = (1/α² − 3)/2) | 0.096 | −3.10 (ghost side) | 1.69 | 0.080 | 1.5e4 | 0.096 | 1.6e4 | 2.4e5 |
| lunar ranging η_N < 2.4e−4 (Hofmann & Müller 2018) | **2.0e3** | **3.8e3** | **1.1e3** | **2.0e3** | 0.28 | **1.3e3** | **8.8e2** (β₀ = 3.3e3) | 0.02 |
| pulsar dipole, composition part only, |Δα| < 2e−3 (Freire 2012, as summarised by Wex 2014) | 0.04 | 0.04 (sign of flux reversed) | 0.03 | 0 | 0 | 0.04 | **4.2** | 0.03 |
| range λ for MICROSCOPE at this ξ_eff | < 34.5 km | < 34.5 km | < 35.7 km | n/a | n/a | < 34.5 km | < 34.5 km | not computed |

**Reading the table.**

- **The top three rows do not change across O1, O2 and O4.** That is the content of B5 and B6. Only O1n (half the amplitude) and O5 (a tiny amplitude) move them, and both remain excluded by 10² to 10¹⁰.
- **The middle rows are what the action moves.**
  - The ghost flips γ − 1 to +0.91.
  - Tuning ζ → f in O4 passes Cassini but fails lunar ranging, because β₀ ∝ 1/κ_E blows up. It also fails the pulsar composition dipole, because Δα ∝ 1/√κ_E. These two failures in the tuned corner are [DERIVED] inside the Damour–Esposito-Farèse (DEF) 1PN framework, with the linear F = 1 + 2ζσ. A different nonlinear F changes β₀ but not its 1/κ_E growth: with F = e^{2ζσ}, β₀ = 2.9e3 instead of 3.3e3, and lunar ranging still excludes the corner ×7.8e2 instead of ×8.8e2 (`beta0_completion_check`; α₀ is completion-independent [G5]).
- **O3 at ζ ≤ 1.06 × 10⁻⁴ passes every §6 bound** because it has nothing to show. Without the mass response, σ = 2ξGM/rc² changes no clock, mass or force beyond a Brans–Dicke field with ω ≥ 1.5 × 10⁴.
- **O5 shows the cancellation trick cannot work.** No single ζ cancels both an H-rich Sun and an O/Fe-rich Earth: |f_H − f_Fe| = 3.6 × 10⁻³. The clocks see the Sun at ×2–6 × 10² and MICROSCOPE sees the Earth at ×10⁸.
- **The pulsar row is incomplete.** Neutron-star strong-field sensitivities, the BD-like term Δα ≈ 2s_NS α₀, are not computed here. The program's §6 rows (Δα up to 0.11, orbit decay 10² to 10³ times too fast) rest on them, and they apply to every option with |α₀| ≈ 0.56.

---

## 4. The range axis (C5) and the normalisation axes (C2, C3)

| ξ_eff | 0.13 | 0.0791 (inventory reading of 0.1582) | 0.1571 | 0.1582 | 0.17 | 0.3164 |
|---|---|---|---|---|---|---|
| MICROSCOPE λ_max (km) | 34.85 | 35.68 | 34.54 | 34.53 | 34.42 | 33.45 |

- **C3 (0.1582 vs 0.1571)** moves λ_max by 11 m.
- **C2 (the factor 2)** moves it by 1.1 km. The bound depends on ξ only logarithmically, through the e^{−r/λ} tail.
- **The paper point** λ = 0.909 fm (Λ_QCD = 217 MeV from `field.constants`) lies 19 orders of magnitude below these limits.

The mass term must have the same sign as κ_E [B9]. Otherwise σ does not screen at all.

---

## 5. Mapping the options to decisions C1–C6

| decision | what the action does to it | options bearing on it | trade-off |
|---|---|---|---|
| **C1** where the SSBM constants live | The action's parameters are (κ, ζ, m²) plus the derived f_i. "ξ" becomes a derived quantity: ξ_eff = (ζ − f)/κ_E, or the literal ζ/κ in O3 and O5 | O1, O2: f_i from Quarksum (`inventory`) and κ. O3: ζ and κ only (field-sector). O4, O5: both | σT ties the field equation to the inventory's mass decomposition. σR keeps it in `field/` but drops the mass response. A mixed action needs one module that both import |
| **C2** σ = 2ξGM/rc² vs ξGM/rc² | In an action σ's normalisation is fixed by the mass response (m ∝ e^{fσ}). The two readings are two values of κ: f/ξ vs 2f/ξ | O1 vs O1n (likewise for O2 and O4) | factor 2 on every composition prediction (still excluded ×1.6e5 and ×4.7e10). α² goes from 0.313 to 0.157. λ_max goes from 34.5 to 35.7 km. σ at a horizon is ξ vs ξ/2 |
| **C3** ξ = 0.1582 vs 0.1571 | rescales ξ_eff by 0.7 % | all | no bound changes its verdict. λ_max moves by 11 m |
| **C4** covariant or flat □ | identical at linear order [D2]; differ at 1PN [D1]. Only the covariant one is consistent with Einstein's equations | O6 vs the rest | flat: simpler and enough for every §6 bound, but non-covariant and inconsistent at 1PN (P7's reshaping and neutron-star sensitivities live there). Covariant: carry (1 + 2γΦ/c²) and the Tolman source |
| **C5** σ's range λ | needs m²/κ_E > 0 for a Yukawa [B8, B9] | O7 (healthy) and O8 (ghost) | λ < 34.5 km evades all §6 bounds; the 1 cm to 35 km torsion window is the next cut, not yet mapped. λ = 0.91 fm is invisible to every experiment |
| **C6** engine defaults (A2 `total_energy`, A5 Eyring, A7 SPH kernel) | none: no σ option bears on them | none | independent of this memo |
| **B5** (Ricci sign, filed as a capability) | settled by any action: the sign convention becomes the sign of ζ. The physical statement is sign(ξ_eff·f), i.e. whether masses rise or fall near matter | all | O1: masses fall near matter. O2, O4: masses rise |

**Two decisions this memo surfaces that C1–C6 do not list** (proposed additions, not choices):

- **C7: the kinetic sign** (ghost or not). It is tied to the sign of σ near matter through B7.
- **C8: σT, σR, or both.** This decides whether the "scale of matter" content (the mass response) survives, and whether the repo's sign can be ghost-free.

---

## 6. The trade-offs in one place

| if the Captain wants… | then | price |
|---|---|---|
| the repo's equation literally (□σ = −ξR_GR with mass response) | O2 | a ghost and a repulsive fifth force; excluded at long range |
| the repo's sign without a ghost | O4 (ζ > f) | a second coupling ζ and a negative Jordan-frame kinetic term (healthy in the Einstein frame); excluded at long range |
| no ghost and the simplest action | O1 | σ's sign flips (σ < 0 near matter; σ_conv > 0 is not reached by collapse); excluded at long range |
| □σ = −ξR literally with no ghost | O3 | the mass response disappears; σ is unobservable once Cassini is satisfied |
| ξ ≈ 0.16 kept in some form | O5 (ξ as ζ/κ) or O7 (λ < 34.5 km) | O5 is fine-tuned and still excluded. O7 is allowed, and its natural paper point is invisible |
| any long-range σ that couples to masses | ξ_eff < 1.7 × 10⁻¹² | the σ-bridge stops being "the scale of matter" at any observable level |

**Effect on the program's other items.**

- **B1 (nuclear matter):** σT and σR give different sources in degenerate matter. They agree for nucleon (QCD) matter [E1, E2] and disagree on electrons.
- **B4 (σ-wave solver):** every covariant option keeps monopole σ radiation sourced by the pressure term. In σT only the QCD-share pressure counts.
- **P7 (compact objects):** needs C4 = covariant to be meaningful.

---

## 7. Honesty

- **What could have failed.** Each of the 32 identities could have left a nonzero residual. The negative control C0 shows the decoupling test rejects the naive coefficient κ. The pre-registered predictions P1–P11 were all confirmed. Nothing was moved after the run.
- **Evidential weight.**
  - **Derivations about the theory class, not evidence about the thesis.** These are general facts of scalar–tensor theory (DEF, Damour–Donoghue), re-derived on the repo's own family [DERIVED]. They hold for any values of the constants.
  - **Exclusion numbers.** They are [DERIVED] from [LIT] bounds, the same ones the program used, re-verified. They reproduce §6: clocks 3.23e5, MICROSCOPE 9.43e10, α² = 0.313, γ − 1 = −0.477 / +0.913, ω = 0.096, η_N = 0.478.
  - **The new results:** composition invariance; the third-law failure of the postulate; the ghost-free repo sign at ζ > f; the trace-source identity for σT; the white-dwarf sign; and the failure of the O5 corner.
- **Approximations.**
  - Linear weak field plus the DEF 1PN formula for β; no strong-field neutron-star sensitivities.
  - One isotope per element. Alloy mass fractions are [LIT approx] as in `sigma_exclusion.py`.
  - The MICROSCOPE altitude (710 km) and the white-dwarf potentials φ are as encoded in the program's script.
  - O5's source compositions are single-isotope proxies (H-1, O-16).
  - The flat-box inconsistency is an argument, not a sympy check.
- **Engine defects.** None were found. Nothing in `sigma_ground` was modified.

## 8. Design corrections

1. **The program's §6 sentence "which a canonical scalar with a positive kinetic term cannot do … Either σ is a ghost or the field equation has no action" is incomplete.**
   - It is true for a pure mass coupling.
   - With a σR term and ζ > f, the repo's sign is ghost-free. The ghost criterion is κ + 6ζ², not κ [DERIVED, B7d and C1].
   - The long-range exclusion is unaffected.
2. **The item's note pairs "□σ = −ξR with a perfect-fluid source" with "m_i ∝ 1 + f_iσ".** That pairing is not an option: no action produces it [DERIVED, F1]. Every option above has the source charge equal to the response charge.
3. **"Normalisation of ξ" is not an independent axis once there is an action.** ξ_eff is derived, and the two readings are two values of κ (C2 row).
4. **"Frame" has no answer for a flat □.** O6 is non-covariant.

These corrections do not change the original pass criterion. The memo is DELIVERED against it: every option has a Lagrangian, a field equation, a frame, a ghost status, a fifth-force sign and its value for every §6 bound.

## 9. References (verified by web search, 2026-10-10)

- C. Brans & R. H. Dicke 1961, Phys. Rev. 124, 925, doi:10.1103/PhysRev.124.925
- T. Damour & G. Esposito-Farèse 1992, Class. Quantum Grav. 9, 2093, doi:10.1088/0264-9381/9/9/015
- T. Damour & J. F. Donoghue 2010, Phys. Rev. D 82, 084033, arXiv:1007.2792, doi:10.1103/PhysRevD.82.084033
- Y. Fujii & K. Maeda 2003, *The Scalar-Tensor Theory of Gravitation*, Cambridge University Press, doi:10.1017/CBO9780511535093
- R. Lange et al. 2021, PRL 126, 011102, arXiv:2010.06620. (c²/μ)dμ/dΦ = 7(45) × 10⁻⁸
- P. Touboul et al. 2022, PRL 129, 121102, arXiv:2209.15487. η(Ti,Pt) = [−1.5 ± 2.3 ± 1.5] × 10⁻¹⁵
- J. Bagdonaite et al. 2014, PRL 113, 123002, arXiv:1409.1000. GD133 (−2.7 ± 4.7 ± 0.2) × 10⁻⁵; G29-38 (−5.8 ± 3.8 ± 0.3) × 10⁻⁵
- B. Bertotti, L. Iess & P. Tortora 2003, Nature 425, 374, doi:10.1038/nature01997. γ − 1 = (2.1 ± 2.3) × 10⁻⁵
- F. Hofmann & J. Müller 2018, CQG 35, 035015, doi:10.1088/1361-6382/aa8f7a. η_N = (−0.2 ± 1.1) × 10⁻⁴
- S. Schlamminger et al. 2008, PRL 100, 041101, arXiv:0712.0607. η(Be,Ti) = (0.3 ± 1.8) × 10⁻¹³
- P. C. C. Freire et al. 2012, MNRAS 423, 3328, arXiv:1205.1450. |α_p − α_c| < 2 × 10⁻³ (95 %), as summarised in N. Wex 2014, arXiv:1402.5594

## 10. Reproduce

    cd /d/Aaron/development/sigma-ground
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -X utf8 -B <A2>/a2_sigma_action.py   # ~17 s, 33 checks, writes a2_results.json
    python -X utf8 -B <A2>/a2_figure.py                                                                      # writes a2_sigma_action_options.png

`<A2>` = `C:/Users/aaron/AppData/Local/Temp/claude/D--Aaron-development-sigma-ground/c6652910-47a9-478d-9c83-a8a8c8aaa457/scratchpad/round2/A2`.
