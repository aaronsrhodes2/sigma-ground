# Disproof ledger: what the σ / SSBM material claimed, and what the data did to it

**Rule (the Captain, 2026-10-11):** a claim earns a place in the physics model only if it can be disproved and has survived. A claim that
was disproved is not a useful part of the model; at most it is speculation about ourselves. We win with data alone.

This ledger keeps the *record of the tests*, not the theory. The theory documents and scripts were removed from the working tree in the
commit that added this file; every one of them is recoverable from git (`git show pre-strip-2026-10-11:<path>`). Numbers below are copied
from those documents (UNIFICATION_PROGRAM.md §14–15 and the files named per row), not recomputed here.

## A. Disproved (the data or the mathematics excluded it)

| claim | test | result | where it was recorded |
|---|---|---|---|
| σ is a long-range field that rescales the QCD share of every mass (ξ = 0.13–0.17) | five weak-field gates: Cassini, lunar laser ranging, MICROSCOPE free fall, atomic clocks, pulsar timing | excluded by factors of ~10^2.8–10^4.0 (Cassini), ~10^3.3 (lunar ranging), ~10^11 (free fall), ~10^5.5 (clocks), ≥10^2.8 (pulsar, with a speculative charge) | U1, UNIFICATION_PROGRAM §15.1 |
| the same σ at any range from ~0.1 mm to cosmic | torsion-balance equivalence-principle data | exclude ξ > 0.1375 from 1 cm to 35 km; an inverse-square-law lookup (added after pre-registration) leaves only a range below ~0.1 mm | T5 |
| σ acting through the N-body engine's own hook (GM·e^σ, σ = ξ r_s/r) | Mercury's perihelion | +2.267″/cy, 45× the budget (about 2000× the published β bound) | U5 |
| the postulate □σ = −ξR with a non-universal mass response comes from an action | structural analysis of scalar–tensor actions | impossible: its scalar forces violate Newton's third law; no action rescues a long-range σ that changes masses | A2 |
| ETA = 0.4153 "derived" from a dark-energy constraint | audit 2026-05-15 | not derived (0.4153 is merely inside the DESI 1σ band) | eta_empirical_verdict |
| ETA_FORMULA = exp(−φ/σ_conv) ≈ 0.4158 | audit 2026-05-15 | formula-search numerology, rejected | eta_empirical_verdict |
| σ-sourced pressure term at a neutron-star centre, 3P/ε = 0.67 | cross-check against GW170817's 90 % pressure bound | withdrawn: 1.6× above the bound; the defensible value at 2 ρ_nuc is 0.21 (0.11–0.37) | NUCLEAR_MATTER_PLAN |
| Diósi–Penrose decoherence mode in the self-decoherence purity check | the repo's own purity run | 41 % below the other modes, "firmly excluded" in that model | bh_phase_ix_gamma_purity_results |

## B. Untested (a falsifier was written, no data was ever run)

These are predictions with no supporting evidence. They are out of the model until someone runs the falsifier on real data.

- DESI DR3 central value of the holographic dark-energy c² falls in [0.410, 0.420].
- primordial-black-hole inspiral signatures in LIGO-era analysis.
- direction-dependent H₀ maps (Euclid + LSST) correlating with the σ picture.
- an anti-correlation in a ~100-cosmology simulation grid.
- a renormalisation-group value from arXiv:2505.01422's method.
- photons in a "bond-failure layer" (σ in [ξ/2, σ_conv)).
- black-hole "conversion event" mass shedding (hypotheses A vs B): the LIGO catalogue bounds the inspiral-merger-ringdown mass difference to ≤ 5–10 %, and the weaker variants cannot be tested with current data.
- the duality-ellipse σ>0 predictions (H2, H4): consistent with H1 at σ = 0, which the laboratory test could not exceed.
- the horizon as a σ = σ_conv phase boundary.

## C. Not falsifiable as stated

- ξ = Ω_b/(Ω_b+Ω_c) = 0.158 being "the" baryon fraction of the σ field: a numerical coincidence with Planck's value; Kobakhidze–Liang's derivation is the strongest candidate found but the identification is speculative and was never derived.
- a₀ = cH₀/2π: tested on SPARC (2026-10-11). The fitted scale g† spans 0.84 to 1.20 ×10⁻¹⁰ m/s² depending only on the analysis (tabulated points 1.148; 149 galaxies with fixed mass-to-light 1.046; with free mass-to-light 0.836; published 1.20 ± 0.24 systematic). The prediction 1.042 lies inside that spread, so the data cannot discriminate it, and cannot separate MOND, emergent gravity or a halo. Kept as an observation, not as theory.

## D. Speculative hypotheses still served (fenced; the Captain, 2026-10-13: "file them as speculative physics hypothesis")

Three Mentat tools stay in the public tool list. Their arithmetic is exact and standard (Bekenstein–Hawking pixel counting with the Planck length);
what is speculative is the **interpretation** they were built to illustrate. They are tagged `provenance_tag="SPECULATIVE-PENDING"` in every result, their
notes open with "[SPECULATIVE] physics hypothesis, not part of the model", the manifest lists them in the **SPECULATIVE** tier (dict key `extended`), and
they are never foregrounded. An answer that uses one must say so.

| tool | what it computes (exact) | hypothesis it illustrates | what would test it | status |
|---|---|---|---|---|
| `entanglements_to_pop_bubble(radius_m)` | N = πR²/L_p², the Bekenstein–Hawking pixel count of a sphere of radius R | a "bubble" of radius R saturates ("pops") when its surface pixels are all threaded by entanglement; the smallest bubble pops at N = π | no observable has been identified; a falsifier would have to relate N to a measurable collapse threshold | untested (as section B) |
| `holographic_matching_mass()` | M = ħc/(4πG m_p) ≈ 2.25×10¹⁰ kg, where a black hole's baryon count equals its horizon pixel count | below this mass matter "overflows" the horizon (an information-paradox regime) | an evaporation or remnant signature of ~10¹⁰ kg black holes; none observed | untested |
| `baryon_vs_disc(mass_kg)` | n_disc = A/(4L_p²) against n_baryons = M/m_p | the same crossover, evaluated for a given mass | as above | untested |

Not served: `page_time_sigma_ground`, `eta_as_bit_thread_fraction`, `sigma_time_dilation`, `effective_liv_scale_gev` (σ-specific; removed from the
tool list earlier). Rule: nothing in this table enters an answer as physics; it may be computed and labelled.

## What stays
Everything that is standard physics and was merely labelled with the theory's name: the N-body engine, the integrators, GR, tides, the DE440
comparison, the fast ephemeris, the fluid and thermodynamic results that did not depend on σ (negative heat capacity, three clocks, the
virial identity, Jacobson's derivation at leading order), and the nuclear-matter build.
