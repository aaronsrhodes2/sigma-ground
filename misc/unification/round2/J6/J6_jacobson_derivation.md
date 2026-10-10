# J6: Jacobson 1995 as a checked derivation

**Outcome: DELIVERED.** All 13 symbolic steps close in sympy with residual exactly 0. All five pre-registered Mentat
closing numbers fall inside tolerance, the largest deviation being 5.9 × 10⁻¹³ against a tolerance of 10⁻⁸.
Three inputs are named below as not checkable symbolically. They are assumptions or theorems the derivation takes in;
none of them is a gap in the algebra.

| file | what |
|---|---|
| `jacobson_derivation.py` | the sympy derivation (S1 to S10), the Mentat closing numbers (N1 to N6), and the pre-registration and assumption list, all as data |
| `jacobson_results.json` | every step's residual and detail, every Mentat call (55 calls, each with its `source`) and the constants used |
| `make_figure.py` → `J6_jacobson_derivation.png` | the one figure |
| `run.log` | console output of the final run (52.9 s, of which the symbolic part takes 24.8 s) |

Folder: `C:/Users/aaron/AppData/Local/Temp/claude/D--Aaron-development-sigma-ground/c6652910-47a9-478d-9c83-a8a8c8aaa457/scratchpad/round2/J6/`.
No repository file was modified.

---

## 0. Sources (each verified by web search on 2026-10-10)

- **[LIT]** T. Jacobson, *Thermodynamics of spacetime: the Einstein equation of state*, Phys. Rev. Lett. **75**, 1260 (1995).
  arXiv gr-qc/9504004, doi:10.1103/PhysRevLett.75.1260.
- **[LIT]** C. Eling, R. Guedens, T. Jacobson, *Non-equilibrium thermodynamics of spacetime*, Phys. Rev. Lett. **96**,
  121301 (2006). arXiv gr-qc/0602001, doi:10.1103/PhysRevLett.96.121301. This is the "EGJ" paper below.
- **[LIT]** W. G. Unruh, *Notes on black-hole evaporation*, Phys. Rev. D **14**, 870 (1976).
- **[LIT]** J. J. Bisognano, E. H. Wichmann, *On the duality condition for quantum fields*, J. Math. Phys. **17**, 303
  (1976). This is the wedge modular theorem behind A5.
- **[LIT]** P. Kovtun, D. T. Son, A. O. Starinets, *Viscosity in strongly interacting quantum field theories from black
  hole physics*, Phys. Rev. Lett. **94**, 111601 (2005). arXiv hep-th/0405231.
- **[LIT]** G. Chirco, S. Liberati, *Non-equilibrium thermodynamics of spacetime: the role of gravitational dissipation*,
  Phys. Rev. D **81**, 024016 (2010). arXiv:0909.4194.

## 1. Pre-registration (fixed before the first run, not moved afterwards)

**Prediction.** Every algebraic step of Jacobson 1995 closes in sympy:

- (S1) Raychaudhuri with θ(p) = σ(p) = 0 gives θ = −λ R_kk + O(λ²).
- (S2) δQ = κ ε² T_kk A/2 and δA = ε² R_kk A/2.
- (S3) δQ = T η δA with T = ħκ/2π gives T_kk = (ħη/2π) R_kk, and κ, ε and A cancel.
- (S4) A symmetric M_ab with M_kk = 0 for every null k equals f g_ab; the solution space has dimension exactly 1.
- (S5) The contracted Bianchi identity holds identically on test metric families built from arbitrary functions.
- (S6) Conservation fixes f = −R/2 + Λ.
- (S7) The result is G_ab + Λ g_ab = (2π/ħη) T_ab, and 8πG = 2π/(ħη) gives η = 1/(4ħG).
- (S8) The SI restoration is unique: η = k_B c³/(4Għ) = k_B/(4 l_P²).
- (S9) The EGJ shear extension gives η_s = ħη/4π = 1/(16πG), so η_s/s = ħ/4π.
- (S10) In Schwarzschild, κ = c²/2r_s and T_H = ħκ/(2πck_B), and the first law holds.

**Pass/fail.**

- DELIVERED if every residual simplifies to exactly 0 and the S4 null space has dimension 1.
- FAIL if any residual is non-zero.
- INCONCLUSIVE for any named step that cannot be checked symbolically.

Mentat tolerances:

| check | comparison | tolerance (relative) |
|---|---|---|
| N1 | η/k_B from Mentat G, ħ, c vs 1/(4 L_PLANCK²) | ≤ 10⁻⁹ |
| N2 | `bekenstein_hawking_entropy` entropy per area vs N1, for masses 10¹⁰ to 10⁴⁰ kg | ≤ 10⁻⁹ |
| N3 | `unruh_temperature` vs ħa/2πck_B | ≤ 10⁻⁹ |
| N4 | `hawking_temperature` vs `unruh_temperature`(c²/2r_s) | ≤ 10⁻⁹ |
| N5 | the first law from Mentat entropies | ≤ 10⁻⁸ |

## 2. Conventions

The paper's units are used: c = k_B = 1, with ħ and G explicit. Step S8 restores SI. The signature is (−,+,+,+).

- k^a is the tangent of the horizon generators. λ is their affine parameter, with λ = 0 at p and λ < 0 to the past.
- R_kk ≡ R_ab k^a k^b, and R_kk > 0 for matter that satisfies the null energy condition (focusing).
- The symbol **η is Jacobson's entropy per unit area**. It is *not* Mentat's `lookup_constant("ETA")`, which is 0.412, the
  repo's DESI "cosmic entanglement fraction" (`sigma_ground/field/constants.py:322`). The names collide; the meanings
  do not.

## 3. Every assumption, as an explicit line

These lines also appear in `ASSUMPTIONS` in the script, and each step cites the ones it uses.

| id | assumption | kind |
|---|---|---|
| A1 | Spacetime is a smooth 4-D Lorentzian manifold. Every point has a local inertial frame (the equivalence principle), so g_ab = diag(−1,1,1,1) at p. | geometry input |
| A2 | Local Rindler horizon: a small spacelike 2-surface P through p. Its past-directed null normals form an affinely parametrised, hypersurface-orthogonal congruence, so the twist ω is 0. | construction |
| A3 | **Local equilibrium / instantaneous stationarity:** the expansion θ(p) = 0 and the shear σ(p) = 0. P is chosen so both vanish to first order, and this choice is what defines the equilibrium state. | **physical assumption** |
| A4 | **Entropy ∝ area:** dS = η dA, with η universal. The motivation is vacuum entanglement across the horizon with a UV cutoff. The argument does *not* fix η. | **physical assumption** |
| A5 | **Unruh temperature:** T = ħκ/2π for the boost orbit with \|χ\| = 1 (Unruh 1976; Bisognano–Wichmann). | **QFT input** |
| A6 | **Heat = boost-energy flux:** δQ = ∫_H T_ab χ^a dΣ^b, with χ^a = −κλ k^a and dΣ^b = k^b dλ dA. T_ab is classical and symmetric, and its quantum fluctuations are neglected. | **physical assumption** |
| A7 | **The Clausius relation δQ = T dS holds for every local Rindler horizon through every point**, that is, for every null k. | **the postulate** |
| A8 | Locality: the horizon patch is small compared with the curvature radius. R_kk and T_kk are evaluated at p, and only the leading power of the patch size ε is kept. | approximation |
| A9 | ∇^a T_ab = 0 | physical assumption |
| A10 | Spacetime is connected, so a scalar whose gradient vanishes is constant. | topology |
| A11 | 2π/(ħη) is *named* 8πG. G is taken from measurement; the derivation then *predicts η*, not G. | identification |
| GEO1 | The Raychaudhuri equation dθ/dλ = −θ²/2 − σ_abσ^ab + ω_abω^ab − R_kk. | identity |
| GEO2 | The contracted Bianchi identity ∇^a G_ab = 0 and ∇g = 0. | identity (spot-checked in S5) |

## 4. The derivation, step by step (all [VERIFIED numerics] = exact sympy residual 0 unless tagged otherwise)

**S1. Focusing.** Uses GEO1, A2, A3, A8.

- Write θ = Σ aₙλⁿ, R_kk(λ) = R₀ + R₁λ + R₂λ² and σ = s₀ + s₁λ. Solving the Raychaudhuri equation order by order gives
  the general O(λ) coefficient a₁ = −R₀ − s₀² − θ₀²/2 + ω₀².
- Under A2 and A3 (θ₀ = s₀ = ω₀ = 0) this becomes θ = −λR₀ − R₁λ²/2 − (R₀²/6 + R₂/3 + s₁²/3)λ³.
- So θ = −λ R_kk + O(λ²). The θ²/2 term and a shear growing from zero both enter only at O(λ³).

**S1b. Exact cross-check.** For constant R_kk with σ = 0, θ = −√(2R) tan(√(R/2) λ) solves the full nonlinear equation
(ODE residual 0). Its series −Rλ − R²λ³/6 agrees with Mentat `series_expansion` (source `sympy.series`).

**S2. Heat and area.** Uses A6, A8 and S1.

- δQ = ∫₋ε⁰ T_kk(−κλ) dλ A = κ ε² T_kk A/2.
- δA = ∫₋ε⁰ (−λR_kk) dλ A = ε² R_kk A/2.

Sympy and Mentat `integrate_expr` (source `sympy.integrate`) agree term by term.

**S3. Clausius.** δQ = (ħκ/2π) η δA gives **T_kk = (ħη/2π) R_kk**. κ, ε and A all cancel; the free symbols left are
{R_kk, η, ħ}. Uses A4, A5 and A7.

**S3b. θ(p) = 0 is forced, not optional.** Keep θ(p) = θ₀. The Clausius gap then has an O(ε) term −ħκηθ₀A/(2π). δQ
starts at O(ε²), so nothing can cancel that term, and Clausius can hold for every small patch only if θ₀ = 0. [DERIVED]

**S3c. Shear cannot be removed by shrinking the patch.** With σ(p) = s₀ ≠ 0 the same step gives
T_kk = (ħη/2π)(R_kk + s₀²). The shear term sits at the **same order** as the curvature. It is not a small correction
that disappears as ε → 0, which is panel a of the figure. s₀² also depends on the choice of P, not only on k, so the
lemma of S4 cannot be applied to it. The shear must vanish at p by assumption (A3), or it must be booked as entropy
production (S9). [DERIVED]

**S4. The null-tensor lemma.** Uses A1, A7 and S3.

- Set M_ab ≡ (2π/ħη)T_ab − R_ab. S3 says M_kk = 0 for all null k.
- Parametrise every null direction as k = (1+u²+v², 2u, 2v, u²+v²−1), the stereographic map of the celestial sphere.
- Then M_kk = 0 is an exact polynomial identity in (u, v): 13 coefficient conditions on 10 unknowns, of rank 9.
- **The null space has dimension 1, with basis diag(−1,1,1,1) = g_ab.** Hence (2π/ħη)T_ab = R_ab + f g_ab.

This is exact, not sampled.

**S5. The Bianchi identity.** ∇^a(G_ab + Λg_ab) vanishes identically on three metric families built from arbitrary
functions:

| family | time |
|---|---|
| spherical and time-dependent, diag(−e^{2A(t,r)}, e^{2B(t,r)}, r², r² sin²θ) | 2.1 s |
| diagonal, every component an arbitrary function of (t, x) | 4.2 s |
| off-diagonal: g_tt = −F, g_tx = H, g_xx = K, g_yy = g_zz = P, all functions of (t, x) | 14.5 s |

The identity is a theorem. These are spot checks that the code's conventions satisfy it, not a general-metric proof.

**S6. Conservation fixes f.** Uses A9, A10, GEO2 and S4. ∇^a T_ab = 0, ∇^a R_ab = ½∂_b R and ∇g = 0 give
∂_b(f + R/2) = 0, so **f = −R/2 + Λ**. Λ is an integration constant the argument does not fix.

**S7. The equation of state.**

- **R_ab − ½R g_ab + Λ g_ab = (2π/ħη) T_ab.**
- Setting 2π/(ħη) = 8πG gives **η = 1/(4ħG)**, so η^(−1/2) = 2√(ħG) = 2 l_P, as the paper states.
- Mentat `simplify_expr` turns 2π/(ħ·(1/4Għ)) into 8πG.
- Contracting with a null k returns S3, since g_kk = 0. The round trip closes.

**S8. Restoring SI units.**

- Dimensional analysis: G^p ħ^q c^s = m⁻² has the **unique** solution (p, q, s) = (−1, −1, 3).
- Direct SI derivation: take x⁰ = ct, κ_geo = a/c² and k_B T = ħcκ_geo/2π. Then T = ħa/(2πck_B).
- The Clausius coefficient becomes R_kk = [2πk_B/(ħcη′)] T_kk.
- Setting that coefficient to 8πG/c⁴ gives **η′ = k_B c³/(4Għ) = k_B/(4 l_P²)**. [DERIVED]

**S9. The Eling–Guedens–Jacobson shear extension.** Relax A3 to allow σ(p) ≠ 0.

- The extra area term becomes internal entropy production, d_iS = η∫(−λ)σ² dλ dA.
- In Killing time v (χ = ∂_v, dλ/dv = −κλ, σ̂ = σ dλ/dv), this is d_iS = (η/κ)∫σ̂² dv dA.
- Match T d_iS to a membrane's viscous dissipation 2η_s σ̂²:
  - **η_s = ħη/4π = 1/(16πG)**
  - **η_s/s = ħ/4π**

These agree with EGJ's remark: their shear viscosity is Tα/2 = ħα/4π = 1/16πG [LIT]. The η_s/s value is the Kovtun–Son–Starinets
number [LIT]. The dissipation convention 2η_s σ_abσ^ab is an input.

EGJ's other result is cited only, not reproduced. An entropy density with an f(R) curvature correction cannot close in
equilibrium. It needs a **bulk-viscosity** entropy-production term, which energy–momentum conservation fixes. [LIT]

**S10. Schwarzschild consistency.**

- From f = 1 − r_s/r, κ = c² f′(r_s)/2 = c²/(2r_s).
- ħκ/(2πck_B) = ħc³/(8πGMk_B) = T_H.
- d(Mc²)/d(η′·4πr_s²) = T_H.

The 1/4 in S_BH was itself obtained *inside* GR (Hawking's temperature plus the first law). This agreement is therefore
consistency, not independent evidence for η.

## 5. The closing numbers from Mentat (PLATINUM §21)

**Constants**, all with source `sigma_ground.field.constants`: G = 6.6743 × 10⁻¹¹, c = 299 792 458, ħ = 1.054571817 × 10⁻³⁴
and k_B = 1.380649 × 10⁻²³ (all SI), and L_PLANCK = 1.6162550239285497 × 10⁻³⁵ m.

| check | result | max \|relative deviation\| | tolerance | pass |
|---|---|---|---|---|
| N1 | η/k_B = c³/(4Għ) = **9.570182792894672 × 10⁶⁸ m⁻²** (η′ = 1.3213 × 10⁴⁶ J K⁻¹ m⁻²), compared with 1/(4 L_PLANCK²) | 2.0 × 10⁻¹⁶ | 10⁻⁹ | ✅ |
| N2 | S_BH/(k_B A) from `bekenstein_hawking_entropy` (source `sigma-ground (Bekenstein 1973 / Hawking 1974)`) at 8 masses from 10¹⁰ to 10⁴⁰ kg, compared with N1; second code path `field.gr_basics.bekenstein_hawking_entropy`, 6.7 × 10⁻¹⁶ | 4.0 × 10⁻¹⁶ | 10⁻⁹ | ✅ |
| N3 | `unruh_temperature` (source `sigma-ground (Unruh 1976)`) compared with ħa/(2πck_B) at 7 accelerations. Earth's g = 9.80665 m s⁻² (source `solar_system_body`, `sigma-ground via NASA Planetary Fact Sheet 2024`) gives **T = 3.98 × 10⁻²⁰ K**; a = 10²⁰ m s⁻² gives 0.4055 K | 0 (bit-identical) | 10⁻⁹ | ✅ |
| N4 | `hawking_temperature` (source `sigma_ground.field.gr_basics (Hawking 1974)`) compared with `unruh_temperature`(c²/2r_s), with r_s from `schwarzschild_radius`, at 7 masses. 1 M☉ gives 6.168 × 10⁻⁸ K | 3.4 × 10⁻¹⁶ | 10⁻⁹ | ✅ |
| N5 | first law (M₊−M₋)c²/(k_B(S₊−S₋)), with both S from Mentat, compared with T_H at 3 masses | 5.9 × 10⁻¹³ | 10⁻⁸ | ✅ |
| N6 | η_s/s = ħ/(4πk_B) = **6.078 × 10⁻¹³ K s**; membrane η_s = c³/(16πG) = 8.03 × 10³³ kg s⁻¹ | (reported) | — | — |

**What the closing numbers can and cannot show.** Mentat's `unruh_temperature` *is* the formula ħa/(2πck_B)
(`sigma_ground/mcp/tools/frontier.py:165`). L_PLANCK is built from G, ħ and c (`constants.py:422-423`). The checks
therefore test code-path consistency: the tools agree with each other and with the derivation's formulas, to rounding.
They cannot test physics. A factor-of-2 or unit bug in any one tool would have shown up as an O(1) deviation, and none
did. [MEASURED]

## 6. What this derivation does and does not show

**It does show** [DERIVED, VERIFIED numerics]:

- Take a Lorentzian geometry (A1, A2), entropy ∝ area with a universal η (A4), the Unruh temperature (A5), heat equal to
  boost-energy flux (A6), conserved matter (A9), and local equilibrium at p (A3). Then requiring δQ = TdS on every local
  Rindler horizon (A7) **leaves no freedom except Λ**. The field equation must be G_ab + Λg_ab = (2π/ħη)T_ab.
- With the measured G it fixes η = 1/(4ħG) = 1/(4 l_P²) per unit area, which is the Bekenstein–Hawking value.
- The non-trivial content is the uniqueness. The null-tensor lemma (S4) and the Bianchi identity (S6) leave only one
  scalar free. That is why the result is called an *equation of state*.

**It does not show:**

1. **It does not derive gravity from fluid dynamics,** nor from thermodynamics alone. Geometry is assumed: a metric,
   curvature, horizons and the Raychaudhuri identity are inputs. The result is a statement about which *dynamics* of an
   assumed geometry is consistent with an assumed horizon thermodynamics.
2. **It does not derive the entropy–area law.** A4 is an assumption. The argument runs for *any* constant η; only the
   name "8πG" (A11) connects η to measured gravity. The microscopic origin (entanglement with a Planck-scale cutoff) is
   motivation, not something computed here or in the paper.
3. **It does not derive the Unruh temperature.** A5 is a QFT theorem the derivation imports.
4. **It does not fix Λ.** Λ is an integration constant (S6).
5. **It needs local equilibrium.** θ(p) = 0 is forced (S3b). σ(p) = 0 cannot be justified by the local limit (S3c, panel
   a). Without it, the honest bookkeeping is non-equilibrium: entropy production gives the horizon a shear viscosity
   η_s = 1/(16πG) (S9). For curvature-corrected entropies a bulk viscosity is needed as well (EGJ 2006). Chirco and
   Liberati (2010) relate the GR shear term to Hartle–Hawking tidal heating.
6. **It is classical.** T_ab fluctuations are neglected (A6), and nothing here says whether or how to quantize.
7. **It is not a test of the Captain's thesis.** Every step is a mathematical consequence of the stated inputs, so the
   derivation could not have "disagreed with nature". It shows the thesis's strongest literature form is internally
   consistent, and it marks exactly which inputs carry the physics (A3, A4, A5, A6, A7).

**Link to the program** (`misc/UNIFICATION_PROGRAM.md` §14):

- This delivers ranked next experiment #6.
- L4 ("same dynamics") stays where §14 put it: a literature route that holds inside a framework, now reproduced and
  checked, still not something the repo's engines produce.
- The one bridge to measured matter is S9/N6. The horizon's η_s/s = ħ/(4πk_B) is exactly the KSS value in the program's
  η/s row, "horizon = KSS bound". That row's provenance can now cite a derivation, not only the literature.

## 7. Design corrections

1. The item lists "shear neglected" next to "vanishing expansion at p", as if both were approximations. They are two
   different kinds of statement:
   - **θ(p) = 0 is forced by the Clausius relation itself** (S3b, an O(ε) obstruction).
   - **σ(p) = 0 is a genuine extra assumption.** Shrinking the patch does not remove shear: it enters at the same order
     as R_kk (S3c; panel a shows the inferred coupling plateauing at 1 + σ²/R_kk).
   - What *is* neglected by the local limit is the θ²/2 term and the shear that grows from zero away from p. Both are
     O(λ³) (S1), with a relative effect of ε²R_kk/12 on δA.

   The original criterion is unaffected: both readings give DELIVERED.
2. The item asks that η = 1/(4 l_P²) be "checked with Mentat". In the repo, L_PLANCK and the Unruh and BH tools are
   computed from the same G, ħ, c and k_B. These checks therefore validate the code paths (tag: engine validation), not
   the physics. A second, independently written code path (`field.gr_basics`, S in J/K, with no Planck length) was
   added to N2 to make the check slightly stronger. It agrees to 6.7 × 10⁻¹⁶.

## 8. Not checkable symbolically (named, as the item requires)

- **A5, the Unruh effect.** It is a QFT theorem (Bisognano–Wichmann). Sympy cannot derive it from this algebra, and
  Mentat's tool encodes the formula; it does not derive it.
- **A4, entropy ∝ area.** This is microphysics. The derivation works for any constant η.
- **S5 in full generality.** It was verified on three families built from arbitrary functions, not on the fully general
  4-D metric. It is a theorem, so it is not in doubt.

None of these is a step *of the algebra*. Every algebraic step closes, so the outcome is DELIVERED, not INCONCLUSIVE.

## 9. Engine notes (flagged, not fixed)

There are no engine defects. Two hygiene notes:

- `sigma_ground/mcp/tools/frontier.py:24`: `_consts()` hard-codes `c = 299792458.0` instead of importing
  `constants.C`. The value is identical today, but it is a second source of truth.
- The constant name `ETA` (`constants.py:322`, the DESI cosmic entanglement fraction, 0.412) collides with the standard
  symbol for Jacobson's entropy density. A reader asking Mentat for "η" in a horizon-thermodynamics context would get an
  unrelated number.
