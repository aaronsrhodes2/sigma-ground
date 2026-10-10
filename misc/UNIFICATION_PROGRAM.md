# Unification program — are fluid dynamics, thermodynamics and gravity one phenomenon?

**Status:** 🔄 being written stage by stage (Session 36, 2026-10-10) · **Results:** [UNIFICATION_PILOT_RESULTS.md](UNIFICATION_PILOT_RESULTS.md) ·
**Figures:** `misc/unification/` · **Scripts:** `scripts/unification/` · **Plan:** the four-level framework below.
Rules in force: PLATINUM §21 (Mentat computes every physical number), Golden Rules 1–7, [SPECULATIVE] fence.

## 1. The question, made testable

The Captain's thesis: fluid dynamics, thermodynamics and gravitational dynamics are one phenomenon that we have
misread as three. "Same phenomenon" is scored at four levels, each a yes/no with a number, plus a differences ledger
the verdict must *explain*:

| level | statement | what it would take | status (after verification) |
|---|---|---|---|
| L1 same operator | the governing PDEs share an operator and Green's function | a static linear field — true of *all* of them | true, and therefore not evidence (P1, P6, P7) |
| L2 same equation | one identity contains all domains as terms | the virial identity — true for *any* force law | true, and therefore not evidence (P2b) |
| L3 same dictionary | an exact map fluid-object ↔ gravity-object | analogue gravity: the wave equation only | analogy [LIT]; not yet simulated here (P3) |
| L4 same dynamics | the fluid's stress-energy *sources* the geometry (Einstein's equation), or gravity's dynamics *is* a fluid's | fluid/gravity duality (AdS, near-horizon), Jacobson's thermodynamic derivation | out of reach of the repo's engines; literature status only |
| differences | causal structure (elliptic / parabolic / hyperbolic); C_grav = −C_gas; non-extensivity; attractive-only, unscreened | derivations and one engine measurement (P5) | real; any one-phenomenon theory must *produce* them |

The verification of the pilots (Stage 0) established the governing rule for the rest of the program: **a statement that
holds for any linear field, any force law, or any choice of constants is not evidence about gravity.** Evidence lives at
L4 and in the measured differences.

## 2. Stage 1 — the literature map (every reference verified externally, 2026-10-10)

Figure 1 (`misc/unification/fig01_bridge_map.png`): the bridges between the domains, coloured by status.

| bridge | direction | status | what it establishes | reference (verified locator) |
|---|---|---|---|---|
| Bekenstein–Hawking | gravity ↔ horizon thermo | theorem within GR + QFT on curved background | S = A/4l_P², T = ħκ/2πk_B, first law, Smarr | Bekenstein 1973 PRD 7, 2333; Hawking 1974 Nature 248, 30 [standard] |
| Einstein equation of state | horizon thermo → gravity | derivation inside a framework (local Rindler horizons + δQ = TdS + Unruh T) | Einstein's equation as an equation of state | Jacobson 1995, PRL 75, 1260; gr-qc/9504004 |
| thermodynamic gravity | horizon thermo → gravity | review / programme | entropy-maximisation derivations of field equations | Padmanabhan 2010, Rep. Prog. Phys. 73, 046901; arXiv:0911.5004 |
| entropic gravity | horizon thermo → gravity | conjecture, critiqued | F = TΔS/Δx with holographic N recovers Newton | Verlinde 2011, JHEP 04(2011)029; arXiv:1001.0785 |
| emergent gravity + dark universe | horizon thermo → gravity | conjecture, critiqued | a₀ ~ cH₀ from de Sitter entropy; lensing test consistent | Verlinde 2017, SciPost Phys. 2, 016; Brouwer et al. 2017, MNRAS 466, 2547 |
| critiques | — | — | ultracold-neutron bound states contradict entropic gravity; conservative forces constrain entropic reformulations | Kobakhidze 2011, PRD 83, 021502(R); Visser 2011, JHEP 10(2011)140 |
| membrane paradigm | gravity → fluid | derivation inside GR | horizon as a viscous, conducting membrane; η = 1/16πG, negative bulk viscosity | Damour 1978, PRD 18, 3598; Price & Thorne 1986, PRD 33, 915 |
| fluid/gravity duality | gravity ↔ fluid | exact duality (AdS; near-horizon) | Einstein in AdS₅ ⇒ boundary Navier–Stokes; every incompressible NS solution has a vacuum-Einstein dual | Bhattacharyya–Hubeny–Minwalla–Rangamani 2008, JHEP 02(2008)045; Bredberg–Keeler–Lysov–Strominger 2012, JHEP 07(2012)146 |
| KSS bound | gravity ↔ fluid | conjecture, supported by data | η/s ≥ ħ/4πk_B; QGP comes closest | Kovtun–Son–Starinets 2005, PRL 94, 111601 |
| analogue gravity | fluid → gravity | analogy (kinematics of a fixed background) | sound in transonic flow obeys □_g φ = 0 on an acoustic metric; sonic horizons radiate thermally | Unruh 1981, PRL 46, 1351; Visser 1998, CQG 15, 1767; Barceló–Liberati–Visser 2011, Living Rev. Relativ. 14, 3 |
| analogue experiments | fluid → gravity | measured data | stimulated (water) and spontaneous/thermal (BEC) Hawking emission | Weinfurtner et al. 2011, PRL 106, 021302; Steinhauer 2016, Nat. Phys. 12, 959 (contested: arXiv:1609.03803); Muñoz de Nova et al. 2019, Nature 569, 688 |
| kinetic theory → hydrodynamics | gas thermo → fluid | theorem (asymptotic) | Boltzmann → Chapman–Enskog → Navier–Stokes–Fourier; σ_s ≥ 0 | Chapman & Cowling 1970; de Groot & Mazur 1962 [standard texts] |
| self-gravitating thermodynamics | gravity ↔ gas thermo | theorem | virial theorem; Jeans instability; negative specific heat; no maximum-entropy state (Antonov) | Lynden-Bell & Wood 1968, MNRAS 138, 495; Thirring 1970, Z. Phys. 235, 339; Padmanabhan 1990, Phys. Rep. 188, 285 |
| spherical accretion | nuclear/degenerate ↔ gravity ↔ fluid | exact solution | transonic Bondi flow, λ_c(γ=1) = e^{3/2}/4 | Bondi 1952, MNRAS 112, 195 |
| induced gravity | vacuum thermo → gravity | conjecture | Einstein–Hilbert action as vacuum-fluctuation response | Sakharov 1967, Dokl. Akad. Nauk SSSR 177, 70 (Sov. Phys. Dokl. 12, 1040) |
| composite-graviton obstruction | — | theorem | no composite massless spin-2 in a Lorentz-covariant QFT with a covariant T^{μν} (emergent models evade it by breaking covariance) | Weinberg & Witten 1980, Phys. Lett. B 96, 59 |
| radial acceleration relation | gravity ↔ data | measured data | g_obs = g_bar/(1 − e^{−√(g_bar/g†)}), g† = 1.20e-10 m/s² over 2693 points, 153 galaxies | McGaugh, Lelli & Schombert 2016, PRL 117, 201101 |

**Reading of the map.** The only theorems are *within* a domain or between gas thermodynamics and fluids (kinetic
theory) and between gravity and gas thermodynamics (virial, negative C_V — a difference, not a unification). Every route
that would make gravity *itself* thermodynamic or fluid is either a derivation inside an assumed framework (Jacobson,
membrane paradigm), an exact duality in a setting that is not our universe (AdS, near-horizon limits), an analogy of
kinematics (Unruh), or a conjecture under live critique (Verlinde). That is the honest state of the art the Captain's
thesis sits inside: suggestive, unestablished, and testable only at L4.

## 3. Stage 2 — the matrix: five domains, fifteen properties

Figure 2 (`misc/unification/fig02_matrix.png`). Symbolic checks behind it: `scripts/unification/stage2_matrix_checks.py`
(8 genuine sympy certifications: Laplacian Green's function; perfect-fluid trace and the radiation zero; Jeans dispersion
from the linearised system; acoustic surface gravity κ = |v′| at the sonic point; virial ⇒ C_V = −(3/2)Nk_B; Navier–Cauchy
as a vector Poisson problem; Maxwell element moduli and the De = 1 crossover; mass-monopole conservation by continuity.
Three further rows are recorded *statements*, not certified: the Tolman ρ + 3P/c² source, the post-Newtonian order of the
dropped metric factor, and the scaling-group exponents.)

| property | fluid dynamics | gas thermodynamics | gravity (Newton / GR) | solid mechanics | nuclear / degenerate matter | verdict |
|---|---|---|---|---|---|---|
| governing structure | balance laws + EOS & transport closure | S(U,V,N) + local equilibrium | ∇·T = 0 (Bianchi) + Einstein eq.; Poisson in the limit | momentum balance + Hooke / flow rule | hydrostatic (TOV) + nuclear EOS | analogous |
| static / steady operator | pressure Poisson | steady Fourier | ∇²Φ = 4πG(ρ + 3P/c²) | Navier–Cauchy (vector Poisson) | ∇²Φ with ρ(P) | **identical** (automatic for any static linear field) |
| Green's function | −1/4πr | −1/4πr | −1/4πr | Kelvin tensor ∝ 1/r | −1/4πr | **identical** |
| PDE class of the dynamics | hyperbolic / parabolic | parabolic | elliptic (Newton) / hyperbolic at c (GR) | hyperbolic, two speeds | ODE (hydrostatic) | different |
| time structure | reversible ideal + dissipative | irreversible | reversible Hamiltonian; horizons the only entropy | elastic + plastic | equilibrium | different |
| variational principle | Hamilton (ideal); Rayleigh dissipation | entropy maximum | Hamilton; Einstein–Hilbert | elastic-energy minimum | energy minimum at fixed baryon number | analogous |
| symmetry group | Galilean; scale-free Euler, broken by viscosity | none spatial; extensivity | Galilean / diffeomorphism; Kepler t² ∝ r³ | Galilean + material symmetry | — | different |
| conserved charges | mass, p, E, circulation | E | E, p, L, Runge–Lenz | mass, p, E | baryon number | analogous |
| entropy & 2nd law | local σ_s ≥ 0 | extensive S | no maximum-S state (Antonov); BH S ∝ area | configurational / plastic | S → 0 | different |
| heat capacity, extensivity | positive, extensive | positive, extensive | **negative, non-extensive** | positive | positive (Fermi gas) | different |
| range & sign | contact | — | 1/r², attractive only, unscreened | short-range elastic | strong force + degeneracy repulsion | different |
| characteristic speed | c_s | ∞ (parabolic) | ∞ (Newton) / c (GR) | c_P, c_S | c_s ≤ c/√3 | different |
| dimensionless crossing | Re, Ma, Fr | Pe, Ra | λ_J, 2K/|U|, r_s/R | De, Wi | 3P/ρc², r_s/R | analogous |
| equilibrium | hydrostatic | thermal | virial; Lane–Emden; Bonnor–Ebert | static | TOV | analogous |
| linear response | ω = c_s k | ω = −iαk² | ω² = c_s²k² − 4πGρ; ω = ck | P / S waves | — | analogous |

**What the matrix says.** The two "identical" rows are the static operator and its Green's function, and both are
properties of *every* static linear field — the verification of P1/P6/P7 made that explicit. The seven "different" rows
are structural: causal class, time arrow, symmetry group, entropy structure, the sign of the heat capacity, range/sign
of the interaction, and signal speed. A one-phenomenon theory must produce those seven from a single underlying
description; a shared Laplacian does not.

**The fourth column earns its place.** Solid mechanics is not a fifth wheel: static elasticity is the same vector
Poisson structure (check g), and viscoelasticity is the one *theorem-grade* bridge between a solid and a fluid — a
Maxwell element is a solid for ωτ ≫ 1 and a fluid for ωτ ≪ 1, with the storage and loss moduli crossing exactly at the
Deborah number De = ωτ = 1 (check h). That is a continuous, measurable dial between two of the Captain's domains,
and Stage 7 runs it on real materials from the repo's own `viscoelasticity.py`.

### 3.1 The σ-bridge with time (the framework statement, corrected)

The repo's postulate □σ = −ξR ("Candidate A"; R and its sign undefined in the live tree; reading adopted: Einstein-
sourced Ricci scalar, MTW sign, (−,+,+,+)) with a perfect fluid inserted is

    (1/c²) ∂²σ/∂t² − ∇²σ = (8πGξ/c⁴)(ρc² − 3P)            [sign under the adopted convention]

— a **wave equation sourced by the trace of the fluid's stress-energy**. Three consequences, two certified:

1. *Static weak field* (GM/rc² ≪ 1): ∇²σ = −(8πGξ/c²)(ρ − 3P/c²), the same operator as Φ and steady T — true, and
   true of any static linear scalar (consistency, not evidence).
2. *Radiation* (check i): mass is conserved by continuity, so the monopole of the σ wave from a pulsating body is driven
   **only by the pressure term**: σ(t, r) ≈ (2Gξ/c²r)·[M − (3/c²)∫P dV](t − r/c). A star whose integrated pressure
   oscillates radiates σ; a star whose mass merely redistributes does not. This is the first statement in the program
   in which the thermodynamic state (P) is the *dynamical* source of a gravitational-sector field — i.e. a candidate
   L4 statement — and it is *untested*: it lives inside a speculative postulate, and the repo has no σ-wave solver.
3. *Order of validity* (statement j, quantified in P7): the covariant □ on the weak-field metric carries a (1 + 2Φ/c²)
   factor of the same post-Newtonian order as 3P/ρc² (virial: 3∫P dV = |U|). Any quantitative P-term claim must carry it;
   P7's did not.

**Where L4 could be tested, and where it cannot.** Inside the repo: the σ-radiation statement (2) could be turned into a
pass/fail once a σ-wave solver exists (one-dimensional, ≤ 60 lines, sourced by a pulsating polytrope) — the prediction
is the *absence* of monopole σ radiation from a mass-only pulsation and its presence from a pressure pulsation, with
amplitude fixed by ξ. Outside the repo: L4 for gravity itself is the fluid/gravity duality and Jacobson's derivation,
neither of which a Newtonian engine can touch; the honest framework statement is that the σ-bridge gives the Captain's
thesis a *concrete, falsifiable* form at the price of resting on a postulate the repo has not derived.

### 3.2 The discriminator table (what could still fail)

The referee's rule, adopted: a test that cannot come out against "same" is engine validation. For every bridge, the
observable where "one phenomenon" and "three phenomena" give *different numbers*, and what result would falsify the thesis.

| bridge | sharpest statement | observable that discriminates | falsifies "one phenomenon" if… | status | evidential weight |
|---|---|---|---|---|---|
| heat ↔ gravity (P1) | shared Laplacian | none in the static limit; the *transient* (parabolic vs elliptic vs hyperbolic) | — (a difference, not a test) | run (static); three-clocks unrun | tautology |
| virial (P2b) | one identity | none (theorem for any forces); the pressure-closure deficit x is engine physics | — | run | solver validation |
| analogue horizon (P3) | Hawking form on a sonic surface | a sound pulse launched inside r_sonic in the repo's own SPH flow: does it escape? | it escapes (no horizon forms) → the analogy fails on this engine | **unrun** | [LIT] until run |
| horizon thermo (P4) | first law / Smarr | none (identities) | — | run | tautology |
| gravity ↔ gas thermo (P5) | sign of C_V | dT/dE across non-homologous energies, relaxed equilibria | dT/dE > 0 for a bound gravitating gas | sign run (fixed-R sweeps); magnitude unrun | derivation + engine-consistent |
| fluid → thermo (first law on the engine) | σ_s = (Φ_visc + q²/κT²)/T ≥ 0 | entropy production in a collapsing SPH cloud once PdV and viscous heating are booked | σ_s < 0 anywhere, or K + U_grav + U_int not conserved | **blocked** by the SPH energy gap (S8 closes it) | the one engine test that can fail |
| σ-bridge (P6/P7) | σ sourced by the trace | monopole σ radiation from a pressure pulsation vs none from a mass redistribution; a consistent (1+2Φ/c²) solve | pressure pulsation radiates no σ, or the metric term reverses the reshaping | **unrun**; gated on decisions below | conditional on a [SPECULATIVE] postulate |
| Jacobson route (L4, [LIT]) | δQ = TdS on local Rindler horizons ⇒ G_μν = 8πG T_μν | a sympy reproduction with assumptions listed (Raychaudhuri, Unruh T, S ∝ A) | — (derivation inside a framework) | unrun | [LIT]; the one thermo ⇒ gravity *dynamics* derivation |
| a₀ = cH₀/2π (data) | emergent-gravity prediction vs RAR | g_obs(g_bar) on SPARC vs g† = 1.20e-10 | a₀_predicted outside the RAR's g† by more than its scatter | unrun (download needs the Captain's OK) | measurement |
| η/s (data) | horizon viscosity on the same axis as matter | KSS bound vs QGP vs water/iron (Mentat) | — (a placement, not a test) | unrun | measurement |

**One thermodynamic object for the whole program** (referee correction, adopted): every bridge from here on uses the same
triple — an internal energy u, an entropy s, and the first law du = T ds − P dv — rather than five different "thermos"
(Fourier conduction in P1, a fixed isothermal c_s in P2, horizon QFT in P3/P4, a kinetic T := 2K/3Nk_B in P5, a barotropic
trace in P6/P7). The SPH engine will carry u once S8 lands; P5's successor measures ⟨K⟩, U_int and U_grav together.

### 3.3 Decisions the σ-bridge is gated on (the Captain's, recorded here when made)

1. **Which □?** Covariant d'Alembertian (then carry the (1 + 2Φ/c²) factor and source Φ by ρ + 3P/c² at the same order) or a
   flat-background operator (then the theory is stated as non-covariant).
2. **Which sign?** The Riemann/Ricci convention. Under MTW, σ > 0 near matter; under the opposite sign σ < 0. The live repo
   fixes neither R nor its sign.
3. **Which ξ, and which normalisation?** σ = 2ξGM/rc² (field-equation reading) vs ξGM/rc² (`inventory/core/sigma.py`,
   `field/audit.py:143-161`); and ξ = 0.1582 (Planck +BAO) vs 0.1571 (the docstring's own numbers).
4. **Close the SPH energy gap now?** Decided yes (plan S8) — the fluid-dynamic leg of the program cannot run without it.

Until 1–3 are decided, P6/P7 numbers are quoted only as "under the stated reading", and no σ-bridge quantitative claim is
reused. Also recorded: the two cheap lens-disagreement reruns (P7 at 1× stable_dt; P1 fixed-region convergence at ≥ 16τ)
are Stage 3 follow-ups; the Stage 3 figure already uses the 1× kick (32³: 1.62e-3 at 12τ_eff vs 1.74e-3 at 20× and 8τ).

## 4. Stage 3 — L1, same operator (Fig. 3)

`scripts/unification/fig03_same_operator.py`, rerun with the verification's corrections (split kick at 1× stable_dt;
the corrected box mode τ_eff = ((n−1)/n)² L²/3π²α):

- **(a)** The repo's FVM heat law, driven to steady state, reproduces the analytic Newtonian potential of a uniform sphere
  (G from Mentat): max exterior relative error 1.62 × 10⁻³ at 32³ and 12 τ_eff; the radial residual peaks at 0.27 % of the
  central value at the staircase sphere edge. [VERIFIED numerics, given exact wall data.]
- **(b)** The same run, plotted in time: the approach to the Poisson solution decays as the box's fundamental Dirichlet
  mode and needs ~8 τ_eff. Gravity's field is there instantly. One spatial operator, two clocks (a third, the hyperbolic
  one, is the three-clocks experiment). [DERIVED — the definition of the equation classes, shown on the engine.]
- **(c)** Two sources through one solver on an n = 1 polytrope (28³, a = 3.5 km): Ψ_c/Φ_c = 0.99939 / 0.93868 / 0.69339 for
  w_c = 10⁻³ / 0.1 / 0.5, against the closed form 1 − 0.609·w_c = 0.99939 / 0.93906 / 0.6953 (0.04 % at w_c = 0.1).
  [DERIVED; conditional on the σ postulate; w_c = 0.5 outside validity; the (1 + 2Φ/c²) metric term still to be carried.]

The honest caption is in the figure title: L1 is true, and true of every static linear field. What it buys the program
is a validated Poisson solver that every later σ-bridge computation can stand on.

## 5. Stage 4 — L2, same equation, and the missing pressure found (Fig. 4)

`scripts/unification/stage4_virial.py` → `misc/unification/stage4_virial.json`; `fig04_same_equation.py`.

**An exact identity on the engine [DERIVED; verified to 10⁻¹⁶].** With the symmetric Monaghan pressure force and pair
antisymmetry of ∇W, Σᵢ mᵢ rᵢ·aᵢ = Σᵢ mᵢ (Pᵢ/ρᵢ²) Dᵢ with Dᵢ := −Σ_{j≠i} mⱼ rᵢⱼ·∇ᵢWᵢⱼ. For the isothermal closure the
pressure-virial efficiency is therefore exactly x = Σᵢ mᵢDᵢ/ρᵢ / 3M — measured x and this sum agree to 1–2 × 10⁻¹⁶ in every
run (23 clouds, N = 64–2048). It also shows why x cannot depend on c_s: c_s cancels identically.

**Two causes of the deficit, separated.** In the continuum D = 3ρ. On particles: (i) the free surface — a particle within
2h of the edge has an incomplete kernel; (ii) the **self-term** — the SPH density sum counts the particle itself,
ρᵢ = mᵢW(0,h) + Σ_{j≠i}…, while Dᵢ has no j = i term. For a uniform interior this predicts, with no free parameter,
x_interior = 1/(1 + 32/(3N_nb)) for the cubic spline (W(0) = 1/πh³), N_nb the full-support neighbour count.

| N (h = 1.5 × spacing) | 2h/R | x, whole cloud (± seed sd) | interior particles | x, interior | gravity / |U_soft| |
|---|---|---|---|---|---|
| 64 | 1.21 | 0.711 ± 0.021 | 0 | — | 0.995 |
| 128 | 0.96 | 0.760 ± 0.010 | 0 | — | 0.997 |
| 256 | 0.76 | 0.789 ± 0.013 | 3 | 0.993 | 0.997 |
| 512 | 0.60 | 0.823 ± 0.001 | 26 | 0.956 | 0.997 |
| 1024 | 0.48 | 0.844 ± 0.004 | 132 | 0.942 | 0.997 |
| 2048 | 0.38 | 0.869 | 467 | 0.930 | 0.997 |
| prediction, interior | | | | **0.914** (N_nb = 113) | |

h-scan at N = 512: where the interior sample is large (h = 1.0×, 85 interior particles) the prediction holds to 1.4 %
(0.769 measured vs 0.759 predicted); with few interior particles the measured value scatters high (21 → 0.99; 3 → 1.10).
The mean-field prediction neglects the correlation between the fluctuations of Dᵢ and ρᵢ (both are sums over the same
neighbours), the plausible source of the 1.4–1.7 % excess [RESIDUAL].

**Verdict.** Gravity's term of the virial identity arrives whole (0.995–0.997 of the softened potential energy). The
pressure term's shortfall is two known SPH estimator effects, a free surface that shrinks with N and a self-term that does
not: at fixed h/spacing = 1.5 the whole-cloud x converges toward 0.914, not 1 — the "persistent deficit x_∞ ≈ 0.92" model
the verification could not rule out is the right one, and it now has a cause. This resolves P2b's [INCONCLUSIVE]. It is an
engine fact, not physics: the self-contribution bias of the SPH density sum is a known topic (Dehnen & Aly 2012, MNRAS 425,
1068, propose subtracting part of the self-contribution [LIT; section not re-read here]). Fix direction for the engine:
a Wendland kernel at larger N_nb, or the self-contribution correction — a Captain's call, filed in the gap ledger (S11).
**For the thesis:** none, as before — the identity holds for any force law.

### 3.4 What σ is not — the Captain's idea made mathematical and cut against measurement (S9a, run 2026-10-10)

**The idea** (Captain): "σ times the density of baryonic energy configures the speed of light and the scale of matter."
**The mathematical form**, using only what the repo already states — script `scripts/unification/sigma_exclusion.py`,
figure `fig09a_sigma_exclusion.png`, numbers `sigma_exclusion.json`:

    scale of matter   m_i(σ) = m_i⁰ [1 + f_i σ + O(σ²)],   f_i = ∂ln m_i/∂σ |₀    (QCD part ∝ e^σ; Higgs + EM invariant)
    field             (□ − λ⁻²) σ = −(8πGξ/c⁴)(ε_b − 3P),   ε_b = ρc²,   λ = ħ/(m_σ c)
    static            σ(x) = (2Gξ/c²) ∫ ρ(x′) e^{−|x−x′|/λ} / |x−x′| d³x′
                        λ → ∞ : σ = −2ξΦ/c²            (σ follows the potential — the field-equation reading)
                        λ → 0 : σ = 8πGξλ²ρ/c²         (σ follows the LOCAL baryonic energy density — the Captain's picture)

One parameter, the range λ, interpolates between "σ is set by where you are in a gravitational well" and "σ is set by the
density of the matter you are in". **The speed-of-light clause**: a σ that rescales c, ħ and every mass together is a change
of units (Duff 2002, hep-th/0208093; disputed by Moffat, hep-th/0208109); only dimensionless ratios are measurable. The
repo's σ moves μ = m_p/m_e (f_μ = 0.9904 from `inventory/core/sigma.py`) and leaves α fixed, so μ is the observable.
A σ that also moved α would face (c²/α)dα/dΦ = 14(11)×10⁻⁹ (Lange et al. 2021).

**Three measurements, three cuts** (2σ, two-sided, conservative; leaf values from Mentat):

| measurement | observable | σ-bridge prediction (λ → ∞, field reading) | measured | verdict |
|---|---|---|---|---|
| Yb⁺/Cs clocks over Earth's orbit — Lange et al. 2021, PRL 126, 011102 | (c²/μ) dμ/dΦ | −2ξ f_μ = −0.313 | 7(45) × 10⁻⁸ | **excluded × 3.2 × 10⁵**; long-range ξ < 4.9 × 10⁻⁷ |
| MICROSCOPE free fall, Ti vs Pt alloys — Touboul et al. 2022, PRL 129, 121102 | η = 2ξ(f_TA6V − f_PtRh10) | 6.6 × 10⁻⁴ (Δf = 2.08 × 10⁻³ from the repo's own mass decomposition) | (−1.5 ± 2.3 ± 1.5) × 10⁻¹⁵ | **excluded × 9.4 × 10¹⁰**; long-range ξ < 1.7 × 10⁻¹² |
| H₂ lines in white-dwarf photospheres — Bagdonaite et al. 2014, PRL 113, 123002 | Δμ/μ at φ = 1.2 / 1.9 × 10⁻⁴ | +3.8 / +6.0 × 10⁻⁵ | (−2.7 ± 4.7) / (−5.8 ± 3.8) × 10⁻⁵ | 1.4σ / 3.1σ tension; long-range ξ < 0.36 |

**What survives.** The only open corner is short range. The repo's own paper puts it there — m_σ ≈ Λ_QCD = 217 MeV, so
λ = ħc/Λ_QCD = 0.909 fm — and in that corner every one of the three tests is blind (σ is exponentially suppressed beyond a
femtometre of the source). The price: σ = 8πGξλ²ρ/c² is 6.5 × 10⁻⁴⁰ at nuclear saturation density, and reaches
σ = 0.34–0.41 (e^σ = 1.4–1.5) only at ρ ≈ 1.4–1.7 × 10⁵⁶ kg/m³ — about 5 × 10³⁸ times nuclear density, 10⁴⁰ below the
Planck density. **So σ is not a long-range field with ξ ≈ 0.16 (excluded by 10⁵–10¹¹), and in the one place it may live it
is invisible to every experiment ever done.** That is the chip-away result; it does not refute σ, it locates it.

**The Captain's bet: ξ = 0.13–0.17** (2026-10-10). That is the coupling, and Planck already measures it as the baryon
fraction (0.1582, or 0.1571 from the docstring's own Ω values). Taking the bet as given, the data fix the *range* instead:

| coupling ξ | MICROSCOPE alone | clocks alone | white dwarfs |
|---|---|---|---|
| 0.13 | λ < 34.9 km | λ < 9.8 × 10⁹ m | no limit |
| 0.1571 / 0.1582 | λ < 34.5 km | λ < 9.7 × 10⁹ m | no limit |
| 0.17 | λ < 34.4 km | λ < 9.6 × 10⁹ m | no limit |

So **if ξ lands at 0.13–0.17, σ's range is shorter than ~35 km** — it cannot reach from Earth to MICROSCOPE's orbit.
Laboratory torsion balances (Eöt-Wash) probe the 1 cm – 35 km window directly; mapping their bounds onto σ's charge
fᵢ (which differs from the baryon-number charge those papers quote) is the next cut, not yet made. The repo's own paper
(λ = 0.909 fm) sits 19 orders of magnitude inside the allowed side.

**On "σ ≈ 1.4–1.5".** No result in the repo's research gives σ in that range. The landmarks are the conversion wall
σ_conv = −ln ξ ≈ 1.84 (`PAPER_sigma_moduli_stabilization.txt`; an alternative candidate in `ssbm-theory/theory/bridge_equations.md`
gives 1.086), the inventory ansatz σ_Earth ≈ 1.1 × 10⁻¹⁰, and `SIGMA_HERE` = 1.2 × 10⁻⁶¹. e^σ = 1.4–1.5 corresponds to
σ = 0.34–0.41; under the field-equation reading (σ = ξ r_s/r) that would require r < r_s — inside a horizon.

*(Stages 3–12 follow.)*
