# Phase J — Horizon as σ = σ_conv phase boundary, and interior-disc horizon flattening

**Date:** 2026-04-17
**Phase:** J — theoretical formalisation of two previously implicit claims
**Scope:** doc-only; no code or data.  Promotes two threads that were
latent in Phases G and H.1–H.7 into first-class sigma-ground commitments.
**Status:** open, flagged for empirical test in Phase I.

## What this doc adds over Phase H

Phase H.1–H.7 used σ_conv = −ln(ξ) ≈ 1.844 as a normalisation scale in
the ξ-echo prediction and Phase G used it as the endpoint of the γ(σ)
candidate curves.  Neither phase ever stated, explicitly, what σ_conv
**is** in terms of spacetime geometry.  Two users intuitions surfaced
this gap:

1. **"The event horizon is a phase threshold, not a trap."**  Light and
   matter crossing the horizon don't stop being able to propagate;
   they phase-decouple from exterior coupling.  Visibility fails
   because interior fields no longer interact with our EM / SM gauge
   sector, not because escape velocity exceeds c.
2. **"If the interior has its own matter and angular momentum, it
   flattens into a disc.  Does that flatten the horizon from outside?"**

Both are consistent with A and with the existing sigma-ground
machinery.  This doc formalises them and lists the specific signatures
that would test each.

## Variable glossary (name[symbol])

| Name | Symbol | Value | Meaning |
|------|--------|-------|---------|
| conversion fraction | ξ | 0.1582 | = Ω_b / (Ω_b + Ω_c), Planck 2018 |
| conversion scale | σ_conv | 1.844 | = −ln(ξ); σ at which matter fully phase-converts |
| entanglement index | η | 0.4153 | two-dim entanglement-per-dim constant |
| horizon-survival amplitude | γ(σ_conv) | 0.7924 | 1 − η/2; Phase G `sigma_coh` mode |
| Schwarzschild radius | r_s | varies | 2GM/c² |
| dimensionless spin | a* | 0 ≤ a* ≤ 1 | J·c / (G·M²) |
| Kerr horizon oblateness | ε_Kerr(a*) | ≥ 0 | polar-equatorial fractional ratio deficit under Kerr |
| observed horizon oblateness | ε_obs | measured | same, inferred from QNM spectroscopy |
| disc-leak oblateness excess | δε | predicted > 0 | ε_obs − ε_Kerr under the interior-disc hypothesis |

## Part 1 — Horizon = σ = σ_conv surface

### The claim

The event horizon of a BH is identified with the surface of constant
σ = σ_conv in sigma-ground's scale field.  Exterior: σ < σ_conv.
Interior: σ > σ_conv.  The horizon itself is the level set σ = σ_conv.

### Why this is the natural identification

- **σ_conv is already the matter-conversion threshold.**  Above it,
  SM matter converts to SSBM (gauge-inert, gravitationally active) —
  see `misc/bh_conversion_mass_hypothesis.md` and the RODM outline.
  Matter crossing the horizon entering σ > σ_conv is exactly the
  process that "happens inside a black hole" under RODM.
- **σ_conv is already the bulletpoint where γ(σ) takes its
  cross-mode special value.**  In all four candidate modes of
  Phase G's `coherence_gamma_from_sigma`, σ = σ_conv is the endpoint
  where coherence drops to its theory-fixed minimum (Θ for `linear`
  / `cbrt`, 1 − η/2 for `sigma_coh`, Θ + (1 − Θ)/e for `exp`).  The
  "coherence drop at the horizon" is already formalised.
- **σ is sigma-ground's gravity proxy.**  σ → ∞ corresponds to infinite
  local redshift; σ = σ_conv is the first isosurface where phase
  coherence with our exterior field drops below unity.

### What this buys us physically

1. **Light crossing the horizon.**  At σ = σ_conv the EM coupling to our
   exterior field goes to its γ(σ_conv) value — below, not zero — and
   the light transitions into a phase-orthogonal propagation mode.
   It is not "sucked in"; it has *moved into a different phase layer*
   and no longer projects onto our photon field.  Externally, the
   horizon is black because σ > σ_conv interior photons have zero
   projection onto our phase.
2. **Matter crossing the horizon.**  SM baryons phase-convert to SSBM
   at σ = σ_conv.  Mass-energy stays (ADM-conserving, hypothesis A),
   but gauge-sector visibility goes to zero.
3. **The no-hair theorem is modestly relaxed.**  Pure GR: only (M, J, Q)
   leak out.  Sigma-ground: (M, J, Q) plus γ(σ_conv) coherence-deficit
   plus (proposed) horizon-shape deficit from interior matter
   distribution.  The leak is quantified and small, not unbounded.

### What this does NOT claim

- The σ = σ_conv identification does **not** contradict GR's null-surface
  characterisation of the horizon.  Both are true simultaneously: the
  horizon is simultaneously a null surface (external causal
  observation) and a σ = σ_conv phase boundary (sigma-ground interior
  characterisation).
- It does **not** imply that light crossing the horizon can re-emerge.
  Phase-decoupled light inside the horizon has no exterior-coupled
  component to re-radiate, by construction.
- It does **not** modify r_s for a given M.  The Schwarzschild /
  Kerr radius is set by M (and J) as usual.  The identification is
  "whatever r_s GR gives, that surface is σ = σ_conv in sigma-ground."

### Testable signature

Not a standalone test — it is a **unifying re-interpretation** that
makes two previously disconnected Phase G / H.1 predictions share a
single underlying mechanism.  γ(σ_conv) ringdown amplitude deficit and
ξ-shell echoes are both phase-decoupling signatures: the first the
small-scale (coherent) version, the second the bulk (reflective)
version.  A confirmed detection of either strengthens the identification;
a confirmed detection of both at the predicted quantitative values
would be strong combined evidence.

## Part 2 — Interior-disc horizon flattening

### The claim

Under RODM, a BH's interior is a nested universe with its own matter
content and its own conserved angular momentum J_int.  Rotating
self-gravitating systems flatten into discs by angular-momentum
redistribution (galaxies, stars, accretion discs, planets).  Therefore
the interior matter flattens.

Under pure GR (no-hair), interior matter shape is invisible externally
beyond (M, J, Q).  But sigma-ground has a known small no-hair violation
(γ(σ_conv) ≈ 0.79 leaks coherence out), so an analogous leak of
interior shape is theoretically possible.

**Predicted leak:** the exterior horizon is more oblate (equatorial bulge
exceeds polar flattening) than pure Kerr would predict for a given
(M, a*), by an amount δε set by the interior disc's mass distribution
and axial alignment with J.

### Caveats — how big can this leak plausibly be?

- **Maximum bound from γ(σ_conv).**  The coherence leak is ~21 % of
  the ringdown amplitude.  If the horizon-shape leak scales
  comparably, it would be a ~20 %-scale oblateness excess on top of
  Kerr.  This is a large effect and should be easy to detect.
- **More conservative estimate.**  The amplitude-coherence leak is a
  pure-phase effect (not mass-shape).  The shape leak, if present at
  all, is plausibly 2nd-order smaller (~η² ≈ 4 %, or ~(η/2)² ≈ 1 %).
  This lower bound is harder to detect but still possible with
  post-O5 data.
- **Conservative null.**  If the sigma-ground leak is strictly confined
  to the coherence channel (γ only) and does not extend to the shape
  channel, δε = 0 exactly.  This is the "Phase J partial-confirm" case:
  horizon = σ_conv identity stands, but interior-disc leak is ruled
  out.

### Testable signature

**Kerr-QNM spectroscopy deviation.**  A ringdown's quasi-normal modes
depend on (M, a*) for pure Kerr.  Extra oblateness shifts QNM
frequencies (mostly ω_R at fixed l, m, n) and decay times (Q-factor)
by amounts computable from the oblateness perturbation.

The LIGO "tests of GR" analyses (GWTC-3 TGR paper) already fit (M, a*)
from the inspiral and test whether the ringdown is consistent with the
(M, a*)-predicted QNM frequencies.  A systematic bias of
ω_R(observed) − ω_R(Kerr at inferred a*) in the direction that extra
oblateness would produce is the Phase J signature.

**Target precision.**  GWTC-3 ringdown bounds on Kerr-QNM frequency
shifts are ~5–10 % per event, ~1–3 % catalog-combined.  The 1 %-leak
estimate is marginal now; the 20 %-leak estimate is ruled out already
(absent a detection).

### Status vs published data

Recent LIGO TGR papers (GW150914 isolated ringdown analyses, Abbott
2019) give Kerr-QNM consistency at the ~10 % level per event.  This
**already** rules out the optimistic 20 %-leak estimate.  It does **not**
rule out the 1–4 % estimates.  With O4a partial catalog + O5 full
catalog, this bound will tighten to ~0.5–1 % and either confirm or
rule out the conservative estimate.

## Part 3 — Unified picture

The two threads combine into a single pictures of the RODM-A horizon:

The horizon is a **σ = σ_conv phase boundary** that separates exterior
SM-matter-and-photon space from an interior SSBM nested-universe.  The
boundary is not perfectly opaque — it leaks at quantitative,
theory-fixed amounts:

- **21 % coherence leak** (γ(σ_conv) amplitude deficit in ringdown) —
  Phase G, already testable.
- **ξ-shell reflective leak** (discrete echo train at n·Δt_1) —
  Phase H.1, testable in Phase I.
- **Horizon-shape leak** (extra oblateness from interior disc
  flattening) — Phase J, measurable in ringdown QNM spectroscopy.

All three leaks share the same source (the σ = σ_conv boundary's
finite coupling across the phase transition) and all three are
quantitatively predicted from the same constants (ξ, η).  A
simultaneous detection of all three at predicted values would be
strong evidence for the entire sigma-ground BH picture.  A null on
any one bounds the corresponding coupling channel.

## Phase J open questions

These deserve follow-up but are not load-bearing for the identification:

1. **What sets the magnitude of the shape-leak δε?**  Need a derivation
   analogous to γ(σ_conv) = 1 − η/2, ideally yielding a closed-form
   δε(a*, interior mass distribution).  The 1 %–20 % range above is a
   placeholder.
2. **Is the interior angular momentum always aligned with exterior J?**
   For a single-formation BH, yes (conservation).  For merger remnants,
   the interior J alignment post-merger is open.  Misalignment would
   produce non-axisymmetric horizon distortion — a different (richer)
   signature.
3. **Does the interior disc have its own ξ-shells?**  If the interior
   universe has its own BH population, sigma-ground applies recursively.
   The "daughter black holes inside the parent" picture is RODM's
   nesting commitment; recursion depth is an open question.

## Cross-references

- Phase G verdict: `misc/duality_ellipse_verdict.md`
- Phase H.1 echo predictions: `misc/bh_merger_predictions.md`
- Phase H.2 hypothesis map: `misc/bh_conversion_mass_hypothesis.md`
- Phase H.3 B1 falsification: `misc/bh_imr_verdict.md`
- Phase H.7 refined echo search: `misc/bh_echo_search_refined.md`
- Collision phenomenology synthesis: `misc/bh_collision_phenomenology.md`
- RODM outline: `matter-shaper/theory/outline.md`
- SSBM gauge commitment: `RODM_hypothesis.md`

## Files

- **New:** `misc/bh_horizon_sigma_conv_identity.md` — this file (doc-only)
- **No code changes.**
- **No test changes.**
