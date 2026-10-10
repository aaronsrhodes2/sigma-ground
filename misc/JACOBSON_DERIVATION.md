# J6 v2: Jacobson 1995 as a checked derivation (after three skeptic reviews)

**Outcome: DELIVERED, at leading order in the patch size.** All 13 steps close in sympy with residual exactly 0. Each
step's status is now computed from its residual. The tags say which parts are live checks and which hold by
construction. Two mutated reruns show which checks could have failed. All five pre-registered Mentat closing numbers are
inside tolerance; the largest deviation is 5.9 × 10⁻¹³, against a tolerance of 10⁻⁸. They test Mentat's code paths,
not the derivation: they come out identical, to the last digit, when the derivation is broken.

| file | what |
|---|---|
| `jacobson_derivation_v2.py` | the derivation (S1 to S10), diagnostic D1, closing numbers N1 to N6, the mutation harness (`J6_MUTATION=none / unruh_x2 / chi_sign`), the pre-registration (verbatim from v1) and the assumptions |
| `combine_v2.py` | merges the two mutation runs into the main JSON (`mutation_controls`) |
| `jacobson_results_v2.json` | every residual, tag and live check; D1; the Mentat numbers; 65 Mentat records (64 MCP tool calls plus 1 direct-import record), each with its `source`; engine flags; mutation controls |
| `jacobson_results_v2_mut_unruh_x2.json`, `..._chi_sign.json` | the two control runs |
| `make_figure_v2.py` → `J6_jacobson_derivation_v2.png` | the figure |
| `run_v2.log`, `run_v2_mut_*.log` | console output. Baseline run: 120.0 s total, 39.7 s of it symbolic, on a shared machine |

The v1 files are kept unchanged beside these. No repository file was modified, and the GMod session was not touched.

## 1. Pre-registration (unchanged from v1)

The prediction and the pass/fail rule are in `PREREG`, word for word as in v1. Jacobson 1995 is a leading-order argument
(A8), so the prediction is about the leading-order steps. Diagnostic D1 looks at the next order. It is not a step of the
paper, and it sits outside the verdict.

## 2. Assumptions (v2 wording)

| id | assumption | kind |
|---|---|---|
| A1 | Lorentzian 4-manifold, with a local inertial frame at every p | geometry input |
| A2 | Local Rindler horizon from a small 2-surface P through p; the null normals are affine and twist-free | construction |
| A3 | **P is chosen so that θ(p) = σ(p) = 0.** Jacobson: "It is always possible to choose P through p so that the expansion and shear vanish in a first order neighborhood of p." This restricts the choice of surface, not the spacetime. Jacobson calls such a horizon "instantaneously stationary (in local equilibrium)". EGJ 2006 remark 4 later dropped equilibrium as the *reason* for σ(p) = 0. | **construction** |
| A4 | dS = η dA with η universal. With an f(R) entropy density, θ(p) ≠ 0 is forced instead (S3b) | physical assumption |
| A5 | Unruh temperature T = ħκ/2π | QFT input |
| A6 | Heat is the boost-energy flux, with χ = −κλk. EGJ remark 5: χ is ambiguous at O(x³), which blurs the heat flux at relative order ε²/L_c² | physical assumption |
| A7 | **The Clausius equality** (no internal entropy production) holds for every null k at every p, on horizons with θ(p) = σ(p) = 0 | **the postulate; carries the physics** |
| A8 | Leading order in the patch size ε only. The relation does *not* hold at the next order (D1) | approximation |
| A9 | ∇^a T_ab = 0 | physical assumption |
| A10 | Spacetime is connected | topology |
| A11 | 2π/(ħη) is **named** 8πG. That identification gives η = 1/(4ħG); nothing is predicted, since η is not measured independently | identification |
| GEO1, GEO2 | The Raychaudhuri equation; the contracted Bianchi identity | identities |

The physics is carried by A4, A5, A6 and A7, together with A9. A3 is a construction (v1 had listed it as a physical
assumption).

## 3. The steps, with tags and mutation response

Under "A5 broken" the Unruh temperature is replaced by ħκ/π. Under "A6 broken" the boost field becomes +κλk. Either
change propagates through every step that uses that input.

| step | content | tag | live checks | by construction | A5 broken | A6 broken |
|---|---|---|---|---|---|---|
| S1 | θ = −λR_kk + O(λ²). General a₁ = −R₀ − s₀² − θ₀²/2 + ω₀² | [VERIFIED numerics] | order-by-order solution | – | closes | closes |
| S1b | exact θ = −√(2R) tan(√(R/2)λ); series agrees with Mentat `series_expansion` | [VERIFIED numerics] | ODE residual; sympy vs Mentat | – | closes | closes |
| S2 | δQ = κε²T_kk A/2 and δA = ε²R_kk A/2 (Mentat `integrate_expr` agrees) | [VERIFIED numerics] | both integrals | – | closes | **FAILS** |
| S3 | T_kk = (ħη/2π)R_kk; κ, ε and A cancel | [VERIFIED numerics] | solved coefficient | – | **FAILS** | **FAILS** |
| S3b | θ(p) = 0 is forced for constant η. With density η(1 + f₁λ), θ₀ = −f₁ instead (EGJ eq. 10). The ε² coefficient is now complete: −Aκ(2R_kk ηħ − 4πT_kk + ηħθ₀²)/(8π) | [LIT reproduced] EGJ remark 4 | O(ε) solve, two variants | – | closes | closes |
| S3c | With σ(p) = s₀: T_kk = (ħη/2π)(R_kk + s₀²). Two surfaces P with the same k but different s₀ differ by ħη(s_a² − s_b²)/2π, so they cannot both satisfy it | [LIT reproduced] EGJ remark 4 | solved coefficient; two-P difference | – | **FAILS** | **FAILS** |
| S4 | null-tensor lemma: 13 conditions, rank 9, null space of dimension 1, basis g_ab. Control: null k confined to one plane gives dimension 5 | [VERIFIED numerics] | exact null space; negative control | – | closes | closes |
| S5 | ∇^a(G_ab + Λg_ab) = 0 on 3 metric families built from arbitrary functions (3.5 s, 5.2 s, 16.8 s) | [VERIFIED numerics] | the divergence on each family | – | closes | closes |
| S6 | ∇^a(R_ab + F g_ab) = ∂_b(F + R/2), with F an arbitrary function. **α is solved for** in ∇^a(R_ab − αR g_ab) = 0 and comes out ½ in both the t and r components | [VERIFIED numerics] + [BY CONSTRUCTION] | divergence identity; α solve | F + R/2 = Λ by A10 | closes | closes |
| S7 | Coupling **taken from S3** as 2π/(ħη); naming it 8πG gives η = 1/(4Għ). Mentat `simplify_expr` turns 2π/(ħη) at the solved η into 8πG | [VERIFIED numerics] + [BY CONSTRUCTION] | η solve; Mentat | Einstein form given α; g_kk = 0 round trip; A11 | **FAILS** | **FAILS** |
| S8 | Exponents (−1, −1, 3) are unique; η′ = k_B c³/(4Għ), solved from the SI Clausius relation | [VERIFIED numerics] + [BY CONSTRUCTION] | exponents; η′ solve; T_SI | η′ = k_B/(4 l_P²) by definition | **FAILS** | **FAILS** |
| S9 | η_s = ħη/4π = 1/(16πG); η_s/s = ħ/4π, whose only free symbol is ħ | [LIT reproduced] + [BY CONSTRUCTION] | entropy production; η_s | η_s/s does not depend on η or G | **FAILS** | **FAILS** |
| S10 | κ = c²/2r_s and T_H equals Hawking's formula. First law using η′ from S8 | [VERIFIED numerics] | κ; T_H vs Hawking | first law closes for any consistent Unruh factor | **FAILS** (vs Hawking) | **FAILS** |

Totals: 13/13 steps close in the baseline, 7/13 with A5 broken and 6/13 with A6 broken. In v1, S3c's "CLOSES" was a
hard-coded string, S6 substituted its own answer, and S7 hard-coded the coupling. All three are now computed. S1, S1b,
S4, S5 and S6 do not use A5 or A6. Their live checks are the closed-form ODE, the planar negative control in S4, and the
α solve in S6. S3b passes under both mutations because θ(p) = 0 is forced for any Unruh factor and either χ sign; it is a
structural statement. With A5 broken, S10's first law still closes, because the first law is internal consistency. Only
the comparison with Hawking's formula fails.

## 4. D1: the next order (diagnostic, outside the verdict) [DERIVED]

Keep R_kk = R₀ + R₁λ and T_kk = T₀ + T₁λ along the generator, take θ from the Raychaudhuri equation, and impose
Einstein's equation along the whole generator, T_i = ħηR_i/2π. Then:

- The O(ε²) gap is 0; this is S3.
- The O(ε³) gap is **−A R₁ηħκε³/(12π)**. This matches the could-it-fail skeptic exactly.
- Closing the relation exactly at O(ε³) would need T₁ = ħηR₁/(4π), which contradicts Einstein's equation.
- Matching integrands instead, as EGJ do, gives a gap of −R₁ηħκλ²/(4π) at O(λ²).

The largest neglected term is the gradient along the generator, of relative size εR₁/(3R₀), which is first order in ε.
v1 named θ²/2 instead; that term is only of relative size ε²R_kk/12, and the fuzz in the Killing vector is
O(ε²/L_c²) [LIT, EGJ remark 5]. Jacobson and EGJ claim the relation only "to lowest order" (EGJ eq. 6), so D1 limits the
scope of the result rather than refuting it. Panel b of the figure shows the scaling.

## 5. Mentat closing numbers (PLATINUM §21)

**Constants**, from `lookup_constant` (source `sigma_ground.field.constants`): G, c, ħ, k_B, L_PLANCK and M_SUN = 1.989 × 10³⁰ kg.

**Bodies**, from `solar_system_body`:

- Sun: 1.98892 × 10³⁰ kg (source "sigma-ground via IAU 2015 nominal solar values").
- Earth: 5.97219 × 10²⁴ kg and g = 9.80665 m s⁻² (source "sigma-ground via NASA Planetary Fact Sheet 2024").

**Sample masses.** Every sample mass is one of the following:

- a Mentat value;
- a multiple of Mentat's M_SUN (10, 10⁶ or 10¹⁰);
- an arbitrary round input, labelled `arbitrary round sample input (no physical claim)`: 10¹⁰, 10¹², 10¹⁵, 10²⁰ or
  10⁴⁰ kg.

| check | result | max \|rel\| | tol | baseline | A5 broken | A6 broken |
|---|---|---|---|---|---|---|
| N1 | η/k_B = c³/(4Għ) = **9.570182792894672 × 10⁶⁸ m⁻²**; η′ = 1.3213 × 10⁴⁶ J K⁻¹ m⁻² | 2.0e-16 | 1e-9 | pass | identical | identical |
| N2 | S_BH/(k_B A) at 9 masses. The second code path, `gr_basics`, gives 6.7e-16; it shares the same constants | 4.0e-16 | 1e-9 | pass | identical | identical |
| N3 | Unruh temperature at Earth's g: 3.9766 × 10⁻²⁰ K | 0 | 1e-9 | pass | identical | identical |
| N4 | Hawking T vs Unruh T at κ. T_H(1 M☉) = 6.1684e-8 K with M_SUN, or 6.1687e-8 K with the `solar_system_body` mass; ≈ 6.17e-8 K either way | 3.4e-16 | 1e-9 | pass | identical | identical |
| N5 | first law at 4 masses | 5.9e-13 | 1e-8 | pass | identical | identical |
| N6 | η_s/s = ħ/(4πk_B) = 6.0783 × 10⁻¹³ K s; c³/(16πG) = 8.0313 × 10³³ kg s⁻¹ (reported only) | – | – | – | – | – |

These are code-path consistency checks, true by construction:

- `unruh_temperature` is literally the formula ħa/(2πck_B) (`frontier.py:171`).
- L_PLANCK is built from G, ħ and c (`constants.py:422-423`).
- The two mutated runs leave every deviation unchanged.

## 6. What this shows and does not show

**It shows** [VERIFIED numerics, LIT reproduced]: take horizons with θ(p) = σ(p) = 0, which can always be arranged by
the choice of P. Given entropy ∝ area (A4), the Unruh temperature (A5), heat as boost-energy flux (A6), conserved matter
(A9) and the Clausius equality for every null k (A7), the leading order leaves only G_ab + Λg_ab = (2π/ħη)T_ab. The
null-tensor lemma (S4, with its negative control) and the α solve (S6) are the non-trivial uniqueness steps. Identifying
2π/(ħη) with 8πG gives η = 1/(4Għ) = 1/(4 l_P²).

**It does not show:**

- **Gravity derived from fluid dynamics.** Geometry is assumed.
- **The area law.** A4 is assumed, and the argument runs for any η.
- **The Unruh effect.** A5 is imported.
- **A value for Λ.** Λ is an integration constant.
- **Anything beyond leading order** (D1).
- **Anything quantum.**
- **A test of the Captain's thesis.**

**Shear and viscosity.** EGJ remark 4 already makes the θ/σ point (credit to EGJ, and to Chirco and Liberati 2010 for
the tidal-heating reading).

- θ(p) = 0 is forced: "We cannot allow for expansion at p".
- σ(p) = 0 is the condition for no internal entropy production. EGJ reject equilibrium as its justification: "This is
  not valid however, since the shear defined with respect to Killing parameter would in any case vanish."
- If σ(p) ≠ 0 is kept, the shear term is booked as a shear viscosity, η_s = 1/(16πG).

**The KSS number.** η_s/s = ħ/4π does not depend on η or G. It is fixed by T = ħκ/2π and the factor 2 in the de
Groot–Mazur dissipation convention. Its equality with the KSS number is an **identity**, because KSS's number is itself
a horizon value. It is not a bridge to measured matter. Only the horizon entry in the program's η/s row gains a
derivation; the matter entries (QGP, water) are untouched.

## 7. Design corrections

1. **θ and σ at p.** θ(p) = 0 is forced by Clausius for constant η (S3b). σ(p) = 0 is a choice of P that removes
   internal entropy production (S3c, S9). Both points reproduce EGJ 2006 remark 4 [LIT]; they are not new findings.
   v1 called σ(p) = 0 an "equilibrium assumption", which is the rationale EGJ withdrew. v1 also said the local limit
   drops θ²/2 (ε²R/12); the leading dropped term is in fact the gradient one, εR₁/(3R₀) (D1).
2. **What the Mentat checks can test.** They validate code paths, not physics, and the mutation runs now show it
   directly.
3. **Leading order.** "On every local Rindler horizon" must be read as "at leading order in ε" (D1).

Against the original criterion and the corrected reading alike, the outcome is DELIVERED.

## 8. Engine flags (reported, not fixed)

- `sigma_ground/field/gr_basics.py:122` (and line 10): "Hawking 1974, Comm. Math. Phys. 43, 199". CMP 43, 199 is the
  1975 paper; EGJ's reference [3] reads "Commun. Math. Phys. 43, 199 (1975)". The MCP source strings "Hawking 1974"
  (`gr.py:100`, `frontier.py:56`) carry the same pairing forward. This is citation hygiene, not a numerical defect.
- `sigma_ground/mcp/tools/gr.py:89-106`: `hawking_temperature` gives its source as `gr_basics` but re-implements the
  formula inline (line 96). The value is identical; the label is wrong.
- **Two solar masses.** `constants.py:427` has M_SUN_KG = 1.989e30, while `materia/bodies.py:46` gives the sun
  1.98892e30, with source label "IAU 2015 nominal solar values". They differ by a relative 4.02e-5.
- `sigma_ground/mcp/tools/frontier.py:24` hard-codes c = 299792458.0 instead of importing it. The value is identical.
- `constants.py:322`: ETA = 0.412, the DESI entanglement fraction. The name collides with Jacobson's η.
