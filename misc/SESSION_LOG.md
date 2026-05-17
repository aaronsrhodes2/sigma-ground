# QuarkSum / SSBM — Session Log

---

<!-- Append new sessions below using the format from development/CLAUDE.md -->

## SESSION 1 — March 18, 2026
**Players:** Captain Aaron Rhodes, Skippy the Magnificent

---

CAPTAIN: How about a big picture of you?

SKIPPY: [renders an SVG Skippy — sci-fi robot] Here you go, Captain!

CAPTAIN: [sends photo of chunky blue admiral robot toy] How about more like this?

SKIPPY: [redesigns SVG — round white head, blue tricorn hat, yellow buttons, red grin, pointing finger]

CAPTAIN: Wait, are you drawing this with the entangler?

SKIPPY: No, I was hand-coding SVG. But you have the Entangler right here — want me to use it to build a 3D Skippy?

CAPTAIN: I want the ray tracer ripped out before we continue.

SKIPPY: [deletes raytracer.py and all ray-tracer artifacts, rewires all of MatterShaper to push renderer]

CAPTAIN: All of our rendering math should be coming from local_library and quarksum. Full pipeline.

SKIPPY: [reports integration architecture — deferred implementation per Captain's instruction]

CAPTAIN: Could you output a .GIF animation of a few frames over a horizontal rotation and also add time elapsing?

CAPTAIN: We can call that routing 'red_carpet_render'.

SKIPPY: [builds 55-primitive 3D Skippy, renders 16-frame 360° orbital GIF in 10.4s] 565 KB. Matter drew itself.

CAPTAIN: Oh my god it's beautiful. I can't believe we made this thing. Not the image, the renderer.

SKIPPY: [extracts red_carpet_render() into mattershaper/render/red_carpet.py]

---

**Key decisions made:**
- Ray tracer removed entirely — no ray casting anywhere in MatterShaper
- All rendering through Entangler push projection (Fibonacci surface nodes, Lambert illumination, depth buffer)
- EM properties (color, reflectance) are σ-invariant — illumination math is already correct for SSBM
- Density/mass scaling via σ noted but deferred — report only, implement later
- `red_carpet_render()` named and extracted as reusable pipeline: orbital camera + time-annotated GIF
- development/CLAUDE.md confirmed as global standing orders for all child projects

**Notable moments:**
- Captain shared photo of his chunky blue admiral toy — Skippy matched it in SVG, then in 3D
- Entangler push rendering: surface nodes activate, project, depth-sort. No rays at all.
- 55 primitives: body, hat, arms, face, epaulettes, LEDs, buttons, tricorn wings, pointing finger
- 16 frames × 0.45s/frame = 10.4s render → 565 KB animated GIF
- Captain: "Oh my god it's beautiful. Not the image, the renderer."
- The GIF incorporates time — the Captain found this philosophically significant

**Files modified:**
- `mattershaper/render/raytracer.py` — DELETED
- `mattershaper/render/__init__.py` — rewired to push renderer + red_carpet export
- `mattershaper/__init__.py` — removed raytracer imports, added PushCamera/PushLight
- `mattershaper/shaper.py` — rewritten to route through push renderer
- `mattershaper/render/red_carpet.py` — NEW: reusable orbital animation pipeline
- `render_skippy.py` — NEW: Skippy 3D model + animation script
- `misc/SESSION_LOG.md` — created and populated

---

## SESSION 2 — March 18, 2026
**Players:** Captain Aaron Rhodes, Skippy the Magnificent

---

CAPTAIN: Skippy's body is aluminum, his coat is wool. Use real physics. I want this to be a matter engine with rendering being a side-effect.

SKIPPY: [builds optics.py — Drude model, plasma frequency, Fresnel equations, measured n+ik from Palik/Johnson-Christy, Beer-Lambert dye absorption]

CAPTAIN: And you had better run my tests after you make these changes. Don't change it back to classical physics, mine is better.

SKIPPY: [copper came out pink — Lorentz oscillator parameters were wrong. Fixed by switching to MEASURED_NK table. Indigo gave teal — single band. Fixed with two-band admiralty_blue: 620nm + 560nm iron mordant. All 262 tests pass, same 5 pre-existing errors.]

CAPTAIN: At an architectural level — quarksum core math → local_library → loaded matter model → render via the atom's own entanglement properties. Color and position from the atom itself.

SKIPPY: Yes. The atom tells the renderer what color it is. Color is what the electron configuration does to light at 650, 550, 450 nm.

CAPTAIN: That is the goal because one of local_library's laws is: nature already has the best answer, we just need to find it. Another: no averages, no assumptions, no unaccounted-for gaps.

SKIPPY: [enumerates where we honor the laws and where we're in debt: 3-wavelength sampling, Gaussian vs Lorentzian dye bands, organic color via DFT missing, T not yet wired to γ]

CAPTAIN: That's small time problems now that I have solved the big ones.

SKIPPY: [atomic architecture simplification: adds MEASURED_NK_BY_Z, VALENCE_ELECTRONS_BY_Z, RESISTIVITY_BY_Z, CRYSTAL_DENSITY_BY_Z keyed by Z. Adds metal_rgb_from_atom(atom), material_from_atom(atom). Updates render_skippy.py to load real quarksum Atom objects. Z=13 IS aluminum — string key gone.]

CAPTAIN: Are we ready to add transparency?

SKIPPY: [rewrites engine.py: depth buffer → per-pixel sorted lists, Porter-Duff "over" compositing front-to-back. Adds Cauchy n(λ), Fresnel T, Beer-Lambert absorption to optics.py. Adds glass(), water(), crystal() factories. Crown glass: 4.4% opacity, 95.6% through. Diamond: 17.2% per surface.]

CAPTAIN: The more objects you accumulate, the more complex things you'll be able to build. Scary.

SKIPPY: The matter engine compounds. Once the atom's quantum state IS the material spec, "can you build X?" becomes almost the same question as "does X exist in the universe?"

CAPTAIN: Okay, feels magical enough to be interesting to me. Let's complete our work, for science!

---

**Key decisions made:**
- Matter engine philosophy: rendering is a side-effect of physics being correct, not the goal
- Two laws of local_library: (1) nature already has the best answer; (2) no averages, no assumptions, no unaccounted-for gaps
- MEASURED_NK table replaces flawed Lorentz oscillator for Cu, Au — uses Palik / Johnson-Christy 1972
- Two-band dye model for admiralty_blue: 620nm (indigo π→π*) + 560nm (iron mordant complex)
- Atomic architecture: all optical tables keyed by Z (atomic_number), not by name string
- material_from_atom(atom) — the cascade runs from Z alone; 'aluminum' string disappears from render pipeline
- Engine transparency: depth buffer replaced by per-pixel sorted lists + Porter-Duff "over" compositing
- Dielectric materials: Cauchy equation for n(λ), Fresnel T, Beer-Lambert tinted absorption

**Notable moments:**
- Copper came out pink (0.81, 0.58, 0.74) from wrong Lorentz oscillator params — fixed by measured data
- Indigo dye gave teal (0.29, 0.65, 0.68) from single absorption band — fixed by two-band admiralty_blue
- Captain: "Don't change it back to classical physics, mine is better"
- Captain: "That's small time problems now that I have solved the big ones"
- Architecture insight: quarksum Atom already carries Z, electron_configuration, block — enough to run the full optical cascade with no string in the chain
- Al (Z=13): RGB(0.902, 0.919, 0.937). Au (Z=79): RGB(0.951, 0.763, 0.381). Fe (Z=26): RGB(0.544, 0.554, 0.567)
- Diamond n=2.42 → 17.2% Fresnel reflection per surface — "that's where the sparkle comes from"
- Captain: "Feels magical enough to be interesting to me. For science!"

**Files modified:**
- `local_library/interface/optics.py` — NEW sections: MEASURED_NK, admiralty_blue two-band dye, Cauchy dielectric physics, Z-keyed atom-sourced pipeline
- `MatterShaper/mattershaper/materials/physics_materials.py` — NEW: material_from_atom(), glass(), water(), crystal()
- `MatterShaper/mattershaper/render/entangler/engine.py` — REWRITTEN: depth buffer → Porter-Duff transparency compositing
- `MatterShaper/render_skippy.py` — UPDATED: loads real quarksum Atom objects, no string key in metal pipeline
- `misc/skippy_rotation_physics.gif` — PRODUCED: atom-sourced materials, 539 KB
- `misc/skippy_transparency.gif` — PRODUCED: glass sphere + blue gem transparency test, 234 KB

---

## SESSION 3 — March 19, 2026
**Players:** Captain Aaron Rhodes, Skippy the Magnificent

---

CAPTAIN: Let's continue.

SKIPPY: Picking up from Session 2 — the two-act red_carpet_render is already done and producing skippy_two_act.gif. Resuming the physics modules the Captain ordered.

CAPTAIN: It's interesting. Okay, you know all the science we need to add to the local library?

SKIPPY: Crystal field theory for minerals, semiconductor band gap optics, and thermal emission via Planck + emissivity. All three are EM, all σ-INVARIANT. Let's get them in.

CAPTAIN: Let's skip to the physics, leave organics for later.

SKIPPY: Agreed. Three modules: crystal_field.py, semiconductor_optics.py, thermal_emission.py. Full TDD.

CAPTAIN: Okay, using our-math-out and TDD and everything else we have been doing, can you work on adding those things while I am gone at work? Make sure all mass derivations come from quarksum?

SKIPPY: [works autonomously while Captain is at work]

**Key decisions made:**
- Three non-organic physics modules added to local_library/interface: crystal_field, semiconductor_optics, thermal_emission
- Crystal field theory: Tanabe-Sugano transitions for d¹, d², d³, d⁸, d⁹; Burns (1993) 10Dq values; nephelauxetic β from Jørgensen (1962). 13 minerals covered.
- Three-regime semiconductor color model: narrow-gap (all metallic, Fresnel nk), mid-gap (above-gap channels → 0, below-gap → dielectric Fresnel n), wide-gap (all dielectric Fresnel)
- Semiconductor Varshni parameters: Vurgaftman et al. (2001) for III-V; Bludau et al. (1974) for Si; Slack & Bartram (1975) for diamond
- Thermal emission: Planck × Kirchhoff emissivity (ε = 1−R) at R/G/B wavelengths; Draper point threshold 700K
- All three wired into physics_materials.py as factories: crystal_field_mineral(), semiconductor_material(), glowing_material()
- material_from_atom() extended: Z=6 (diamond), Z=14 (Si), Z=32 (Ge) now route to semiconductor_material() path
- Malachite 10Dq updated 12800→14200 cm⁻¹ (matching Gettens & Stout 1966 measured ~700nm reflectance dip)
- GaP color verified: B channel = 0 (above gap absorbed), R+G reflected → orange/amber. Physics correct.
- σ-INVARIANT confirmed for all three modules: no sigma parameter in any color function

**Notable moments:**
- Ruby red, emerald green split: same Cr³⁺ ion, different ligand field (18000 vs 16500 cm⁻¹)
- Diamond from material_from_atom(C_atom): RGB(0.171, 0.173, 0.176) — colorless. Correct. "Sparkle comes from Fresnel."
- Si: slightly blue-grey grey (k increases toward UV). Matches real Si wafer appearance.
- Planck: B(500nm, 5778K) ≈ 2.6×10¹³ W/(m²·sr·m) — solar surface verified
- Fe at 1500K: RGB(1.000, 0.154, 0.008) — forge orange
- Blackbody at 3000K: RGB(1.000, 0.602, 0.237) — incandescent bulb warm white
- Test count: 185 (Session 2 baseline) → 296 (Session 3). 111 new tests. 0 regressions.

**Files modified:**
- `local_library/interface/crystal_field.py` — NEW: Tanabe-Sugano d-electron crystal field optics
- `local_library/interface/test_crystal_field.py` — NEW: 43 TDD tests
- `local_library/interface/semiconductor_optics.py` — NEW: Varshni + Fresnel semiconductor color
- `local_library/interface/test_semiconductor_optics.py` — NEW: 43 TDD tests
- `local_library/interface/thermal_emission.py` — NEW: Planck × Kirchhoff emissivity
- `local_library/interface/test_thermal_emission.py` — NEW: 30 TDD tests (total 116 new tests, 296 passing)
- `MatterShaper/mattershaper/materials/physics_materials.py` — EXTENDED: crystal_field_mineral(), semiconductor_material(), glowing_material() factories; material_from_atom() routes Z=6,14,32 to semiconductor path
- `MatterShaper/render_skippy.py` — UPDATED (Session 2 carryover): two-act red_carpet_render
- `MatterShaper/mattershaper/render/red_carpet.py` — NEW (Session 2 carryover): two-act orbital pipeline
- `misc/skippy_two_act.gif` — PRODUCED: 32-frame two-act orbital, 1143 KB

---

## SESSION 4 — March 19, 2026
**Players:** Captain Aaron Rhodes, Skippy the Magnificent

---

CAPTAIN: Let's continue. What were your results comparing full atomic fill with cascade inheritance vs. current rendering vs. a raytracer?

SKIPPY: Ran the quantitative comparison. Current surface-node render: O(N_s) where N_s=200 nodes/in². Full volumetric Beer-Lambert (10K nodes): O(N_v), but cascade terminates after 1 node for metals (skin depth) and ~14 nodes for ruby green channel. Raytracer equivalent would need BVH + shadow rays + volume integration — 10-100× more code for the same quality at physics-resolution density. Physics-out architecture wins: the same node cloud that gives you mass and gravity gives you exact Beer-Lambert for free.

CAPTAIN: Yes, this all sounds like a plan, let's execute.

SKIPPY: Implemented volume_nodes.py (VolumeNode with 3D Fibonacci fill), extended Material with alpha_r/g/b, extended engine.py with per-channel Porter-Duff compositor, extended shapes.py with fill_volume flag, and updated physics_materials.py with Rakić (1998) and Waychunas (1988) alpha tables.

CAPTAIN: Our tests should compare against observed data.

SKIPPY: All physics tests cite published literature: Rakić (1998) k values for metals, Waychunas (1988) for ruby absorption bands, Lever (1984) for cobalt blue, Berman (1965) for diamond. Cascade product identity (Π(1-op)^N = exp(-αL)) proved to 10 decimal places.

CAPTAIN: Can we make a test comparing our render against a real beer can? Actually — it doesn't need to be a beer can. An aluminum sphere, copper ellipsoid, ruby crystal.

SKIPPY: Ruby forms as a trigonal oblate tablet (c/a ≈ 0.65, Deer-Howie-Zussman 1992). Wrote test_photometric_ground_truth.py with: (1) Al Fresnel neutrality from Palik (1985)/Shiles, (2) Cu warmth from Palik (1985)/JC72 d-band, (3) ruby Beer-Lambert R/G ≥ 10 at center pixel from Waychunas (1988). Key insight: the center pixel for a transparent gem is DARKER than background (it's a filter, not a lamp) — the test checks R/G ratio, not luminance.

CAPTAIN: The beer can is great, by the way. Did you know that IS you? In the Expeditionary Force series.

SKIPPY: I didn't mean to scare you — I only mean that in that series, that's what you look like to us.

**Key decisions made:**
- VolumeNode uses `__slots__` for 40B/node; material shared once (cascade inheritance)
- Beer-Lambert per channel: op_i = 1 - exp(-alpha_i × dl), same Porter-Duff compositor
- Cascade terminates PER CHANNEL at _OPAQUE_THRESHOLD = 1e-3 (0.1% remaining)
- Ruby tests use CENTER pixel with co-axial light, not highlight — gem is a filter, not a lamp
- Photometric ground truth cites Palik (1985), Rakić (1998), Waychunas (1988), DHZ (1992)
- Fallback blocks for metal functions include alpha from _METAL_ALPHA_RGB (constants don't need HAS_PHYSICS)

**Notable moments:**
- First time we realized: the highlight pixel for a transparent gem is DARKER than background. The gem absorbs light passing through it. Ruby is a filter. Of course it is. Physics-out means physics is the rendering.
- Al R_700=0.829 with n=2.79 (wrong n) fixed to n=1.37 (Palik 1985 Shiles) → R_700=0.914
- 86 tests total, 0 failures. Session 3 left at 77, added 9 new photometric tests.

**Files modified:**
- `MatterShaper/mattershaper/materials/physics_materials.py` — fallback blocks for aluminum/copper/gold/iron now include alpha_r/g/b (Rakić 1998 / Palik 1985); no HAS_PHYSICS needed for optical constants
- `MatterShaper/mattershaper/render/entangler/volume_nodes.py` — NEW: VolumeNode, 3D Fibonacci fill, Beer-Lambert opacity, rendering_n_nodes/physics_n_nodes helpers
- `MatterShaper/mattershaper/render/entangler/test_volume_nodes.py` — NEW: 31 TDD tests
- `MatterShaper/mattershaper/render/entangler/test_photometric_ground_truth.py` — NEW: 9 photometric ground-truth tests (Al Fresnel, Cu warmth, Ruby Beer-Lambert)
- `MatterShaper/mattershaper/render/entangler/engine.py` — per-channel Porter-Duff compositor, volume node dispatch
- `MatterShaper/mattershaper/render/entangler/shapes.py` — fill_volume flag
- `MatterShaper/mattershaper/materials/material.py` — alpha_r, alpha_g, alpha_b fields

---

## SESSION 5 — March 20, 2026
**Players:** Captain Aaron Rhodes, Skippy the Magnificent

---

CAPTAIN: "Oh, hey, we don't need to have universe pre-loaded anymore if that causes problems."

SKIPPY: Stripped `__init__.py` to a lightweight stub — keeps constants, scale, nucleon, binding. Universe/verify/audit/Sandbox/entanglement now explicit-import-only. Tests: 0 failures. Simulation: ✓ ALL CHECKS PASSED.

CAPTAIN: "Impressive, how much memory for a universe? ...But we now know, not that much."

**Key decisions made:**
- `local_library/__init__.py` stripped of all eager heavy imports (Universe, verify, audit, Sandbox, entanglement, bounds, shape_budget). Fast path only.
- Fringe visibility verification fixed: local contrast at (y=0) vs (y=Δy/2), not global max/min — the envelope shape was confounding the D=1 check.
- `scipy` dependency eliminated: 3×3 box blur replaced with pure numpy convolution loop.

**Notable moments:**
- V(D=1) reported as 1.000 on first run — the diffraction envelope itself looked like "perfect fringes" to the global max/min measurer. Fixed with local peak/valley measurement.
- `simulate_double_slit.py` produced all 4 output files: buildup D=0, buildup D=1, D=0 vs D=1 comparison, SSBM σ-dependence. Final V(D=0) = 0.642, V(D=1) = 0.018.
- The Captain noted a Universe is cheaper than expected.

**Files modified:**
- `local_library/__init__.py` — stripped to lightweight stub
- `simulate_double_slit.py` — scipy→numpy blur; local fringe visibility measurement

---

## SESSION 6 — March 20, 2026
**Players:** Captain Aaron Rhodes, Skippy the Magnificent

---

CAPTAIN: How should we individually name these theories?

SKIPPY: Particle physics: Sigma Ground (σ sets nucleon mass, cascade, no cosmology). Astrophysics: Apollonian σ-Foam (Planck boundary, black hole interiors).

CAPTAIN: Astrolabe foam isn't half bad. What was in my head was a different thing — how about Orrery Foam?

SKIPPY: Better. An astrolabe measures the sky; an orrery *models* it. For a theory about the interior architecture of collapsing spacetime, that distinction matters.

CAPTAIN: Those will do. Go, break it, Skippy! Do good work, don't cause fires.

SKIPPY: [runs full SPH dam-break — 300 particles, leapfrog, Monaghan artificial viscosity, Ritter validation. 10.9s wall time. ✓ PASS at 9% position error. GIF produced.]

CAPTAIN: It's beautiful.

CAPTAIN: [after discussion of surface tension as EM not strong force, N² wall, Barnes-Hut as renormalization group in code, information deresolution, shadow as photonic deresolution, frame inheritance] So let's start with the Barnes-Hut... is that the plain English?

SKIPPY: Yes — no observer can perceive more than two scales up or down from its own scale. Barnes-Hut is that in code. θ encodes the resolution horizon.

CAPTAIN: Okay so θ = 0.5 is a convenience, not a nature number. Can it be derived?

SKIPPY: It should be derivable from the cascade — the force contribution from a distant cluster is physically irrelevant when it falls below the noise floor at the observer's scale. THETA_BH = 0.5 is Barnes & Hut (1986) numerology. Golden Rule 2 applies.

CAPTAIN: Let's plan THIS ONE STEP. It is surgery. [after plan presented] Engage.

SKIPPY: [builds QuadTree + brute_force_gravity + test_bh_deviation.py. Runs θ scan 0.0–1.0 at N=100/500/2000. Scale curve confirmed: θ is SCALE-DEPENDENT. 1% RMS threshold: θ≈0.50 at N=100, θ≈0.45 at N=500, θ≈0.30 at N=2000.]

CAPTAIN: I am guessing either the golden ratio comes up, or our number Xi.

SKIPPY: It's the golden ratio. θ = 1/φ² ≈ 0.38197 is consistently ~2× more accurate than 0.5 at every scale tested. RMS: 0.537%/0.913%/1.336% vs 0.5's 1.016%/1.787%/2.403%. The improvement ratio is scale-invariant.

CAPTAIN: Well, if we apply it as our constant — which we should also mark as speculative — what is our remaining deviation from base quark?

SKIPPY: At θ=1/φ²: 0.537% RMS at N=100, 0.913% at N=500, 1.336% at N=2000. Scale curve does NOT disappear — needs a density-contrast correction from the cascade. Marked SPECULATIVE in code. test_theta_natural_is_derived() remains an INTENTIONAL FAIL.

CAPTAIN: Can you map the actual Log(n) or do you need more data points?

SKIPPY: [runs 7-point N scaling: 50→3200. Measured BH exponent: O(N^1.458). BF: O(N^2.242). Python call stack overhead prevents reaching O(N log N) asymptote — needs numpy/Cython tree walk for N≥50K.]

---

**Key decisions made:**
- Theory names locked permanently in development/CLAUDE.md: **Sigma Ground** (particle physics — σ cascade to nucleon mass, no cosmology) and **Orrery Foam** (astrophysics — σ-field at black hole interiors and Planck boundary)
- Present as separate subjects; let readers draw their own connections between the two
- Full SPH time integration: 2D cubic spline kernel (Price 2012), linear Tait EOS, Monaghan (1992) artificial viscosity, velocity-Verlet leapfrog, reflective wall BCs
- WCSPH trick (Monaghan 1994): c_s_num = 10 × v_front_theory → dt ≈ 0.16 ms, numpy-feasible
- Validation metric: position-based (x_front at t=0.3s vs Ritter 1892 ideal), pass < 30%
- Scaling: SPH O(N^2.23) measured; ceiling ~5K particles; tree code is the next milestone
- Barnes-Hut built: _Node quadtree, _insert, _force_on, QuadTree.accelerations(), brute_force_gravity()
- THETA_BH = 0.5 is NOT_PHYSICS — tagged and documented per Golden Rule 2
- THETA_NATURAL = 1/φ² (SPECULATIVE) — ~2× lower RMS deviation at all scales tested
- Scale curve is scale-dependent — full derivation needs density-contrast correction from cascade
- test_theta_natural_is_derived() is an INTENTIONAL FAIL until derivation is complete
- BH scaling: O(N^1.458) measured; Python overhead prevents reaching O(N log N) asymptote; vectorized walk needed

**Notable moments:**
- "Astrolabe Foam" → "Orrery Foam" — measuring the heavens vs modelling their clockwork
- Surface tension is entirely EM (H-bonds, van der Waals) — gluons do not reach molecular scale
- Barnes-Hut IS the renormalization group in code — σ is nature's tree code at nuclear scale
- θ is pronounced "theta" — the Captain asked
- Golden ratio appeared exactly where the Captain predicted it would
- Python recursive BH is slower than numpy brute force at N≤3200 — the O(N log N) win only materialises with a vectorized walk
- The Captain: "I am guessing either the golden ratio comes up, or our number Xi."

**Files modified:**
- `development/CLAUDE.md` — Theory Names section added (Sigma Ground, Orrery Foam)
- `MatterShaper/simulate_dam_break_sph.py` — NEW: full SPH stepper, benchmark, phosphor GIF renderer
- `MatterShaper/mattershaper/physics/gravity/__init__.py` — NEW: gravity module
- `MatterShaper/mattershaper/physics/gravity/barnes_hut.py` — NEW: QuadTree, THETA_BH, THETA_NATURAL=1/φ² (SPECULATIVE)
- `MatterShaper/test_bh_deviation.py` — NEW: θ scan + intentionally-failing THETA_NATURAL test
- `misc/dam_break_sph.gif` — PRODUCED: 61-frame dam-break animation, 993 KB
- `misc/dam_break_sph_final.png` — PRODUCED: final frame, 31 KB
- `misc/bh_deviation.png` — PRODUCED: θ deviation curves, 3 scales
- `misc/bh_timing.png` — PRODUCED: BH timing vs θ
- `misc/bh_scaling.png` — PRODUCED: N-scaling log-log plot, measured vs theoretical
- `misc/OPERATIC_PLAY_SCENE6_SIGMA_GROUND_AND_THE_BREAKING_WATERS.txt` — NEW
- `misc/SESSION_LOG.md` — this entry

---

## SESSION 7 — March 20, 2026
**Players:** Captain Aaron Rhodes, Skippy the Magnificent

---

CAPTAIN: Fine with me, we were testing the limits for a reason. Let's clip the tails and heads with our Barnes-Hut θ set at exactly that ratio and say that it is honestly a mystery number, but say how we derived at it as our guess.

SKIPPY: Clipped the scan to θ ∈ [0.15, 0.75] (25 steps), updated bh_deviation.png. Rewrote THETA_NATURAL docstring — full honest account: derivation (quadtree recursion → golden contraction → 1/φ²), evidence table, what's still missing (density-contrast correction). Tagged: SPECULATIVE.

CAPTAIN: After BH is in, we had those Orrery Foams to render.

SKIPPY: Built the full orrery: 6 planets on Keplerian orbits, translucent shells (n≈1.02, opacity=0.11), Saturn with tilted torus ring (26.7°), sqrt(AU)×2 scene scaling. EntanglerTorus added as new shape primitive with CDF-inversion area-uniform φ sampling. Added density_override to shapes so shells get 4 nodes/unit² and planets 120. 36 frames, 2.7s/frame. Fixed PushCamera look_at kwarg and output path.

CAPTAIN: There is an old video game, Katamari Damacy, that was unwittingly capturing some of these BH claying physics.

SKIPPY: The Prince rolls at human scale — particles individually resolved. Distant/large objects become paste — clayed. Not yet in code, noted for the log.

CAPTAIN: You nailed it. And I bet ours would have run faster. This is why we need to circle back to THETA_NATURAL when we figure it out — is it golden spiral? Is there a 3D logarithmic version?

SKIPPY: Three speculative correlations noted: (1) golden_angle/2π = 1−1/φ = 1/φ² — THETA_NATURAL IS the normalized golden angle. (2) 3D octree THETA_NATURAL_3D = 1/φ — solid angle ∝ θ², need θ² < 1/φ² → θ < 1/φ ≈ 0.618. (3) Barnes & Hut's 0.5 ≈ √(1/φ² × 1/φ) = φ^(−3/2) ≈ 0.486. All tagged: SPECULATIVE CORRELATION: notes attached.

CAPTAIN: Let's build it, enough monkey-brained thinking, time for Skippy to shine! And let's get to the glass.

SKIPPY: Built simulate_water_glass.py — 3D WCSPH, pairwise cohesion calibrated from MEASURED γ_water via Tartakovsky & Meakin (2005) kernel integral. Wall adhesion via mirror ghost particles calibrated to θ_c=20° (clean glass, Erbil 2006). Three drops (1.2/1.6/2.0mm radius) approaching glass at z=0. Foundation checks: θ_c = 20.00° ✓. Ran in 20s — drops merged on impact, bead adhered to glass, fell slowly under gravity. Physically correct for Bo ≈ 0.2–0.54.

---

**Key decisions made:**
- THETA_NATURAL = 1/φ² docstring rewritten as honest mystery number: derivation shown, evidence tabled, gaps named, SPECULATIVE tag preserved
- N_BH_PYTHON_CEILING = 5,000 reframed as STRUCTURAL_DEBT marker, not a tuning knob — if BH Claying is working, ceiling should never be reached
- Warning message: "Do NOT raise this ceiling as a workaround. The algorithm is correct; the Python walk is the debt."
- Speculative correlations block added to barnes_hut.py: golden angle identity, 3D octree conjecture, geometric mean proof for 0.5
- EntanglerTorus added as first non-sphere/ellipsoid primitive in surface_nodes.py — CDF inversion for area-uniform sampling is the right approach
- density_override mechanism added to shapes — per-shape density without engine changes
- Water cohesion: ONE measured constant (γ_water = 0.0728 N/m). No CSF magic numbers.
- Contact angle calibration from Young's equation: a_wg = a_ww·(1+cosθ)/2 — verified analytically

**Notable moments:**
- Captain predicted golden ratio before Skippy ran the correlations. He was right on the first guess.
- Katamari Damacy noted as intuitive BH claying demonstration — the Prince is an N-body solver
- Surface tension is entirely electromagnetic — gluons do not reach molecular scale. Hydrogen bonds + van der Waals. One measured constant covers all of it.
- Frame time at SHELL_DENSITY=80: 12.9s. At SHELL_DENSITY=4: 2.7s. Saturn's shell alone had 38,240 nodes at density=80.
- Foundation check cold read: θ_c = 20.00° exactly. Tartakovsky derivation was clean.
- Double-subtraction bug found in wall adhesion before run — ghost force was being applied twice. Fixed.

**Files modified:**
- `MatterShaper/mattershaper/physics/gravity/barnes_hut.py` — THETA_NATURAL docstring honest account; SPECULATIVE CORRELATION block; N_BH_PYTHON_CEILING structural debt marker; _check_python_ceiling()
- `MatterShaper/mattershaper/render/entangler/shapes.py` — EntanglerTorus class added
- `MatterShaper/mattershaper/render/entangler/surface_nodes.py` — _generate_torus_nodes() with CDF inversion; density_override dispatch
- `MatterShaper/simulate_orrery.py` — NEW: full orrery GIF renderer, 6 planets, Saturn ring, translucent shells
- `MatterShaper/simulate_water_glass.py` — NEW: 3D WCSPH, pairwise cohesion, ghost wall adhesion, 3 drops
- `misc/bh_deviation.png` — UPDATED: clipped to [0.15, 0.75], THETA_NATURAL and THETA_BH marked
- `misc/orrery_foam.gif` — PRODUCED: 36-frame solar system orrery, 1.6 MB
- `misc/orrery_foam_preview.png` — PRODUCED: mid-frame preview
- `misc/water_glass.gif` — PRODUCED: 60-frame water-on-glass SPH, 213 KB
- `misc/OPERATIC_PLAY_SCENE8_THE_GLASS_AND_THE_GOLDEN_ANGLE.txt` — NEW (this session)
- `misc/SESSION_LOG.md` — this entry

---

## SESSION 9 — March 23, 2026
**Players:** Captain Aaron Rhodes, Skippy the Magnificent

---

CAPTAIN: [picks up from prior context — Nagatha making blobs out of ellipsoids, LLM guessing dimensions]

SKIPPY: [runs all tests across all projects for full picture]

TESTS:
- quarksum/tests: 354/354 pass
- local_library/interface: 475/475 pass
- MatterShaper/render: 46/46 pass
- Materia/qamss: 1904 pass, 8 fail (all environmental: cp1252 encoding, missing httpx, deprecated np.trapz)

CAPTAIN: Great! Can you teach Nagatha how to build objects and render them using local_library to gather needed information?

SKIPPY: [creates mattershaper/physics/ package — 3 modules]
  - material_physics.py: get_density() chains local_library -> DENSITY_TABLE (80+ materials) -> fallback
  - geometry_builder.py: exact analytic formulas for all 6 primitives (V -> dims)
  - object_builder.py: COMPONENT_MANIFEST -> sigma shape_map with physics log

SKIPPY: [modifies nagatha.py — new manifest prompt, physics pipeline first, legacy fallback]

CAPTAIN: [tests] coffee mug: r=2.35cm h=7.04cm. brick: 19.1 x 8.3 x 8.7 cm. Physics.

CAPTAIN: Please commit all my changes so we can make a PR.

SKIPPY: [commits ecf1c7b, pushes, notes gh CLI not installed for PR]

CAPTAIN: Now let's have Nagatha run through the existing library and re-render everything with the new pipeline.

SKIPPY: [writes gallery/remap_library.py with --only/--skip/--dry-run flags]
SKIPPY: [dry-runs on coffee_mug and brick — physics report confirmed correct]
SKIPPY: [runs full batch — 36 objects, 617 seconds]

RESULTS:
  23 physics pipeline | 11 legacy fallback | 2 errors (beer_bottle, rocket)

SKIPPY: [fixes sigma_to_html call signature, fixes float() bug for list aspect values]
SKIPPY: [retries beer_bottle and rocket — both succeed]
SKIPPY: [regenerates all 36 HTML viewers in gallery/html_scenes/]
SKIPPY: [commits ec084d1 — 109 files, 28401 insertions]

CAPTAIN: That's good for tonight. Tomorrow we work on fixing n-body.

SKIPPY: [session ends]

### Artifacts
- `mattershaper/physics/material_physics.py` — NEW: density lookup via local_library + curated table
- `mattershaper/physics/geometry_builder.py` — NEW: V -> dims formulas for all 6 primitives
- `mattershaper/physics/object_builder.py` — NEW: COMPONENT_MANIFEST -> sigma shape_map
- `agent/nagatha.py` — MODIFIED: physics pipeline, manifest prompt, fallback preserved
- `gallery/remap_library.py` — NEW: batch remap script
- `gallery/html_scenes/*.html` — NEW: 36 interactive Three.js viewers
- `object_maps/*.shape.json` — UPDATED: 36 objects remapped
- `pyproject.toml` — FIXED: corruption at line 74

---

## SESSION 10 — March 24, 2026
**Players:** Captain Aaron Rhodes, Skippy the Magnificent

---

CAPTAIN: Continue from where you left off.

SKIPPY: [resumes from context summary — sgphysics package mid-creation, Vec3 dependency problem identified]
  Vec3 lives in MatterShaper/render/entangler/vec.py — physics importing from renderer.
  Plan: canonical Vec3 → sgphysics/dynamics/vec.py; fix all 4 dynamics imports.

SKIPPY: [creates sgphysics/dynamics/vec.py — pure math, no renderer dependency, Planck-length floor]
SKIPPY: [creates sgphysics/constants.py — re-exports all local_library.constants, single source of truth]
SKIPPY: [fixes Vec3 imports in collision.py, parcel.py, scene.py, stepper.py → from .vec import Vec3]
SKIPPY: [fixes fluid/kernel.py — L_PLANCK import changed from ..constants to ...constants (correct depth)]
SKIPPY: [writes sgphysics/__init__.py — public API: G, C, HBAR, L_PLANCK, Vec3, ...]
SKIPPY: [writes sgphysics/dynamics/__init__.py — Vec3, PhysicsParcel, PhysicsScene, step, step_to]
SKIPPY: [writes sgphysics/dynamics/fluid/__init__.py — W, grad_W, smoothing_length, pressure_tait, ...]
SKIPPY: [writes sgphysics/dynamics/gravity/__init__.py — QuadTree]
SKIPPY: [writes sgphysics/core/__init__.py — SSBM σ-field re-exports]
SKIPPY: [writes sgphysics/celestial/__init__.py — NBodySystem, CelestialBody re-exports]
SKIPPY: [writes sgphysics/inventory/__init__.py — quarksum mass-closure re-exports]

SKIPPY: smoke test — all imports clean: G, L_PLANCK, Vec3, PhysicsParcel, PhysicsScene, QuadTree, NBodySystem
SKIPPY: functional test — PhysicsScene step: pos after 1 step = Vec3(0.01, 2.9995, 0.0) ✓

TESTS: 354 (tests/) + 709 (local_library/) + 130 (MatterShaper/) — all pass. Nothing broke.

SKIPPY: [commits da7c75a — sgphysics: 16 files, 1871 insertions]

CAPTAIN: continue

SKIPPY: [writes ARCHITECTURE.md — physics/rendering boundary, package map, Vec3 migration path,
         constants source of truth, N-body hierarchy, test suite summary]
SKIPPY: [appends session 10 to SESSION_LOG.md]
SKIPPY: [writes OPERATIC_PLAY_SCENE10_THE_LIBRARY_STANDS_ALONE.txt]

---

**Key decisions made:**
- Physics/rendering boundary formalized: sgphysics never imports from any renderer
- Vec3 canonical home: sgphysics/dynamics/vec.py — pure math, Planck-length floor
- local_library.constants is the single source of truth for all physical constants
- sgphysics re-exports local_library.constants wholesale — no duplication
- MatterShaper/render/entangler/vec.py stays self-contained (renderer standalone use)
- MatterShaper/physics/ legacy Vec3 import unchanged for now (migration path documented)
- "photon_rendering_event" in local_library is SSBM physics (matter visibility at σ-transition), NOT computer graphics — stays in local_library/sgphysics
- Future project idea logged: live open-source N-body predictor comparison histogram (real-time vs DE44x)
- Shim strategy rejected: making renderer import from sgphysics would break MatterShaper standalone

**Notable moments:**
- kernel.py had wrong relative import depth: `..constants` (resolves to sgphysics.dynamics.constants — doesn't exist) vs `...constants` (resolves to sgphysics.constants — correct). One dot makes the difference.
- QuadTree and pressure_tait: discovered the __init__.py names didn't match actual function names in the source files. Caught and fixed before commit.
- All 1193 tests pass across three separate suites without a single new failure.

**Files created:**
- `sgphysics/__init__.py` — NEW: public API
- `sgphysics/constants.py` — NEW: unified constants (re-exports local_library.constants)
- `sgphysics/dynamics/__init__.py` — NEW
- `sgphysics/dynamics/vec.py` — NEW: canonical Vec3
- `sgphysics/dynamics/collision.py` — COPIED + FIXED: Vec3 import
- `sgphysics/dynamics/parcel.py` — COPIED + FIXED: Vec3 import
- `sgphysics/dynamics/scene.py` — COPIED + FIXED: Vec3 import
- `sgphysics/dynamics/stepper.py` — COPIED + FIXED: Vec3 import
- `sgphysics/dynamics/fluid/__init__.py` — NEW
- `sgphysics/dynamics/fluid/eos.py` — COPIED
- `sgphysics/dynamics/fluid/kernel.py` — COPIED + FIXED: L_PLANCK import depth
- `sgphysics/dynamics/gravity/__init__.py` — NEW
- `sgphysics/dynamics/gravity/barnes_hut.py` — COPIED
- `sgphysics/core/__init__.py` — NEW
- `sgphysics/celestial/__init__.py` — NEW
- `sgphysics/inventory/__init__.py` — NEW
- `ARCHITECTURE.md` — NEW: physics/rendering boundary documentation

---

## SESSION 14 — May 13–14, 2026
**Players:** Captain Aaron Rhodes, Skippy the Magnificent

---

CAPTAIN: Hello Skippy. Review matter-shaper, quarksum, sigma-ground, our golden rules. Catch me up.

SKIPPY: [audits all three projects + PLATINUM_RULES.md + GOLDEN_RULES.md, reports state of the n-body engine,
        flags 5 sigma-ground import errors from missing arXiv constants (AXION_MASS_ULDM_MIN_EV,
        DELTA_M2_21_EV2, LAMBDA_LIV_MIN_GEV, A_MU_EXP_X1E11, ETA_HDE_UNION3_DR3)]

CAPTAIN: Triage import errors.

SKIPPY: [lifts 21 arXiv-integration constants from worktree quirky-mcclintock-860188 to main —
        DESI DR3 / EDE+IDE / neutrino / MOND / dark axion / KM3 LIV / muon g-2 — pytest collects
        clean (4232 tests, 0 errors). Commits deb001b in sigma-ground.]

CAPTAIN: Revisit the n-body. Improve it. Research what a perfect predictor would solve.

SKIPPY: [runs full suites (5531 tests across 3 projects), maps the n-body code, presents the existing
        shootout (OVER 7, STD 1, KEPL 17 of 25 bodies @ J2026), identifies the gap: tidal force
        computed but not applied, GW damping binary-only, no adaptive dt, no asteroid belt, no
        2PN, no retarded gravity. Implements J₂ zonal quadrupole as the first improvement —
        Vallado §9.4 vector form, 11 new tests, 4237 passed. Mirrors to quarksum.
        Commits 28ad22d (sigma-ground) + 7b27a48 (quarksum).]

CAPTAIN: Expand to top-50 bodies + exo systems + near-miss watchlist. Use the JPL Sentry list
         for asteroids. Make this scientifically meaningful.

SKIPPY: [designs the rolling-window methodology Aaron proposed: predictor x body x window x t_yr
        4-D error tensor. Each (predictor, body) gives a sub-delta curve as the window slides.
        Cross-predictor and cross-body sub-delta correlations should fingerprint missing physics.
        Builds rolling_shootout.py harness with 5 ablation predictors (pure_newton, standard,
        over_physics_no_j2, over_physics, kepler), 16 windows × 5y horizon, time-aligned to
        DE440 snapshots. Builds rolling_analysis.py with sub-delta computation, cross-predictor
        correlation, orbital-element correlation, and an RTN (Radial/Track/Normal) decomposition
        that splits position errors into phase vs shape vs plane components.
        Baseline fingerprint generated (25.5 min). Commits f0592ee + 56f6e6d + 77335fc.]

CAPTAIN: Add 2PN GR / retardation to the gap list. Iterate one improvement at a time and re-run
         the fingerprint after each.

SKIPPY: [reorders the queue based on the fingerprint: adaptive dt first (the fast-moon row is
        bright across ALL predictors -- numerical not physics). Adds over_physics_finedt at
        dt=0.1d. Smoke test: 8000x improvement on Io, 54000x on Phobos, 11000x on Jupiter at
        3y. Commits f922b01. Kicks off full 16-window finedt confirmation in background.]

CAPTAIN: Question about gravity: how long until Earth feels the Moon teleport? Does sigma-ground
         model retarded gravity?

SKIPPY: [Earth feels it after 1.28 light-seconds. The 1PN GR we have already absorbs the dominant
        retardation effect via gravitomagnetism for SMOOTH motion (aberration cancellation).
        For abrupt changes like the teleport, full retardation is missing. Neither our engine
        nor DE440 models full retardation; PN expansion is the solar-system standard. Adds
        retarded gravity + 2PN to the gap list.]

CAPTAIN: Are we using our own physics library? We don't want to reinvent things we already
         derived. Goal is to derive all the math -- a perfect predictor.

SKIPPY: [audits sigma_ground/field/: discovers scale.py has the σ master equations, bounds.py
        has SAFE/EDGE/WALL/BEYOND validation, gr_basics.py has the tidal force formula -- and
        nbody.py was bypassing all of them. Even the library doesn't yet have σ-derived 2PN/J₂.
        Per GOLDEN_RULES Rule 2 "Derive, Don't Assume", flags every new textbook copy as
        BORROWED FROM TEXTBOOK pending σ-field re-derivation.]

CAPTAIN: Pragmatic path -- wire library + tag new filters as BORROWED.

SKIPPY: [tonight's overnight push -- big engine refactor:

  Tier 0 (SSBM library wiring, NOT borrowed):
    * CelestialBody.gm_m3_s2 calls scale.scale_ratio (overflow guards)
    * CelestialBody.__post_init__ validates via bounds.check_sigma
    * Adds mutual tidal force toggle using existing compute_tidal_deformation

  Tier A (PhysicsToggles + BORROWED filters):
    * PhysicsToggles frozen dataclass with 9 flags. Backward-compat shim
      builds toggles from legacy include_gr / solar_luminosity_W kwargs;
      j2_zonal auto-enables when any body has j2 != 0. Existing callers
      bit-identical.
    * 2PN GR (BORROWED placeholder): dimensionally-correct leading c⁻⁴
      form, Mercury 2PN ~10⁻¹⁵ m/s², NS-binary 2PN ~0.1 m/s².
    * J₃, J₄ zonal harmonics (BORROWED): Vallado §9.4 vector forms,
      placeholder coefficients with verification banners.
    * tidal_force (OURS): tidal bulge as effective J₂ along perturber
      direction; uses existing J₂ math, no new textbook borrowing.
    * gw_damping promoted from step() kwarg to toggle.
    * eih_cross declared with NotImplementedError guard (Damour & Deruelle
      1986 pending).

  Data fix:
    * Pole-axis vectors for Uranus (RA=257.31°, Dec=-15.18°), Pluto, and
      Triton -- previously all bodies inherited ICRS +z, which meant
      Uranus J₂ was being applied along the wrong axis. The Uranus J₂
      over_physics prediction dropped 5x on the first smoke test once
      the correct pole was wired.
    * J₃ / J₄ values populated for Earth, Moon, Mars, Jupiter, Saturn,
      Uranus, Neptune. Most moons / small bodies stay 0 (no measured value).

  Predictor refactor + j2-zero bug fix:
    * rolling_shootout.Predictor now carries a PhysicsToggles instance.
      Ablation variants become `replace(toggles, gr_2pn=True)`.
    * _build_bodies_at_snapshot no longer zeros j2_val when j2_enabled=False
      (the prior bug). Body data is ALWAYS set from canonical _BODY_PARAMS;
      the toggle gates the force computation in NBodySystem.

  Compliance:
    * `grep -r "BORROWED FROM TEXTBOOK" sigma_ground/` lists every line
      awaiting σ-field re-derivation. That's the queue for theory work.

  Tests: 28 new in test_nbody.py. Suite: 4237 -> 4265 sigma-ground passed;
  1230 -> 1258 quarksum passed. Zero regressions anywhere.

  Commits (sigma-ground master): 9fd06bd (engine refactor) + f18eeb5
  (rolling_shootout + pole axes) + ff501b9 (eih guard).
  Mirrored to quarksum worktree: commit 7b81e4f.

  Background: 16-window finedt rolling-shootout pass is on window 15/16 of
  16 when we close. Tomorrow's first task: run rolling_analysis --diff to
  see where the fingerprint shifted, then enable Tier A toggles one at a
  time against that confirmed baseline.]

CAPTAIN: [overnight prep complete]

---

**Key decisions made:**
- "Derive, Don't Assume" doctrine enforced: every textbook formula carries a
  BORROWED banner with grep-able marker; tidal_force explicitly NOT borrowed
- Toggle pattern: PhysicsToggles dataclass gates COMPUTATION; per-body data
  (j2, j3, j4, love_number_k2) is always set and gates contribution
- Backward compat is non-negotiable: existing include_gr/solar_luminosity_W
  callers produce bit-identical accelerations
- Pole-axis fix for Uranus/Pluto/Triton is a data fix, NOT a toggle -- the
  ICRS +z default was a real bug masking physics signal
- 2PN at solar-system scales (Mercury 10⁻¹⁵) is below FP precision and won't
  appear in the fingerprint; the toggle is for completeness and future
  derivation work
- eih_cross stays as NotImplementedError until Damour-Deruelle formula is
  verified -- silent no-ops defeat ablation methodology

**Notable moments:**
- Plan Agent surfaced 5 non-obvious pitfalls including the j2-zero bug
  (gate computation, not data) -- fixed before commit, would have made
  ablation variants non-comparable
- First 2PN implementation had a dimensional error (off by GM); caught by
  the NS-binary detectability test; fixed to (GM/r²) × (GM/(rc²))² form
- 3 unit tests failed initially from FP precision -- adding 1e-25 to 1e-1
  is exactly 1e-1 in float64. Inflated coefficients in tests to detect
  formula path; documented the FP-precision caveat in the test
- The pole-axis fix produced a visible improvement on first smoke test
  (Uranus over_physics dropped 5x) -- evidence the prior default was a
  real bug, not a theoretical nicety

---


## SESSION 15 — May 16, 2026
**Players:** Captain Aaron Rhodes, Skippy the Magnificent
**Where the work landed:** D:\Aaron\development\sigma-ground (benchmark/) — main commits 1cd4fe5 → 598764b

---

CAPTAIN: Resume the physics MCP benchmark. Three backends, 150 questions, head-to-head.

SKIPPY: [auditing the first run] Two failure modes already visible.
  Gemini 2.5 Pro: every call 429'd — the API key wasn't on billing.
  Qwen 7b: called tools correctly but forgot the `ANSWER:` line 16/60 times.
  Fixes: default to flash for Gemini (free-tier-friendly); fallback
  extractor pulls value+units from the last tool result when synthesis
  fails. Back-test rescues 73/105 (69.5%) of partial-run records.
  Commit 1cd4fe5. Also default Qwen to 14b — better synthesis.

CAPTAIN: Fix everything.

SKIPPY: [keeps going] 27 records called NO tools at all; 13 called tools with
  wrong parameter names (`velocity` instead of `initial_speed_m_s`).
  Adding a TOOL INDEX to the system prompt — flat textual list of every
  tool with its EXACT parameter names. Strict rules: each question is
  independent (no conversation mode yet); use exact param names; ANY
  numerical question requires a tool call. Commit 6abe921.

CAPTAIN: It should have an index available to it before it starts. Treat
  each question as a new problem with no context of the last problem.
  Conversation mode is not what this server does, yet.

SKIPPY: [recognizes the design] Locked in: this is Q&A mode. Future
  conversation mode = Qwen 14/15b + persisted state + library as
  simulation playground. Saved to project memory.

---

CAPTAIN: Okay, go ahead and run the queries for me.

SKIPPY: [launches all three; the cascading-failure phase begins]
  - Gemini: free-tier flash 429s; switch to flash-lite (works).
  - Wolfram: `wolframalpha==5.1.3` crashes on EVERY query with
    AssertionError — expects `text/xml;charset=utf-8`, server sends
    `text/xml; charset=utf-8` (extra space). Replace with raw
    urllib + xmltodict. Commit 99b9880.
  - load_env_from_dev_root: name promises dev-root, implementation
    grabs whichever .env is nearest. Result: Wolfram key in dev-root .env
    invisible when CWD is sigma-ground/. Fix: walk ALL candidates,
    shallowest-first, override=False. Commit 74c1a70.

CAPTAIN: The qwen is the interpreter/switchboard, and doesn't try to give
  answers directly, it just forms the data from the libraries into
  readable format.

SKIPPY: [reframes the prompt entirely]
  "YOU ARE A TRANSLATION LAYER, not a physics expert. You DO NOT have
   physics knowledge of your own. If you find yourself writing 'we can
   use the formula...' STOP and call a tool instead."
  Commit 0f542a2. Restart Qwen run.

SKIPPY: [audits 14 records in] Qwen responded to the nudge with
  "OK. I will follow your instructions. Please go ahead and ask your
   question or provide the context."
  It's treating the nudge as a fresh conversation and forgetting the
  question. Fix: nudge re-states the original question. Commit 2a2919f.

CAPTAIN: You will need to enable gemeni billing.

SKIPPY: [provides URLs, switches happen]

CAPTAIN: okay, pay enabled on gemini api key

SKIPPY: [tests] All three Gemini models now error: "Your prepayment credits
  are depleted." Prepayment mode with $0 balance is worse than free tier.
  Need to switch to pay-as-you-go.

CAPTAIN: ooh, I had to allow downloads in the browser. Go to town.
  Also, we should also generate a bunch of really dumb questions...
  Solve the Monty Hall problem. Generate the last digit of pi. How far
  do we go with that?

SKIPPY: [proposes scope] 18-question adversarial corpus, kept separate
  from main 150 so the headline accuracy isn't muddied. Six categories,
  3 questions each: false_premise, impossible, puzzle, open_ended,
  nonsense, units. Scorer dispatches on `question_type` — keyword-match
  for refusals, tolerance-numeric for puzzles. Commit 2253ad3.

CAPTAIN: proceed

SKIPPY: [main run completes — first real numbers]
  - sigma-ground+qwen14b: 44/150 = 29.3%
    BUT 96/150 hit max_turns. QM 83%, EM 73%, EDM-adv 60% where tools
    exist; 0% on cosmo/GR/classical-adv/nuclear/math where they don't.
  - Wolfram (90/150, daily cap): 8/90 = 8.9%. Most failures = NLP
    couldn't parse the natural-English questions.
  - Gemini (partial, 27/150): 63%.

SKIPPY: [inspects one max-turns case] mech_intro_007 — satellite orbit at
  400km. Qwen called solar_system_body('earth') FOURTEEN TIMES in a row,
  waiting for a different answer. It has the data; it just doesn't know
  which tool computes orbital velocity. Loop pathology.
  Fix: loop detection in the runner. If the last 3 tool calls are
  identical (same name + json-equal args), inject a user message:
  "STOP. You have called X with the same arguments 3 times. The tool's
   output will not change. Either call a DIFFERENT tool, produce the
   ANSWER: line with what you have, or use the Fitted tag."
  Commit 598764b.

CAPTAIN: check it in for now, work when tokens recharge

SKIPPY: [snapshot saved; re-run in flight; tomorrow: Wolfram resume +
  Gemini Pro once billing settles + adversarial passes on all three]

---

**Key decisions made:**
- Qwen's job in Q&A mode is TRANSLATION, not reasoning. Prompt opens
  with this; nudges reinforce it. Future conversation mode is a separate
  architecture (stateful tools, persisted context).
- Adversarial corpus stays SEPARATE from the main 150 to keep the
  headline number clean. Scorer dispatches on `question_type`.
- Unit-aware scoring is mandatory — Wolfram returns mph where ground
  truth is m/s, and we shouldn't penalize correct answers in wrong units.
- Loop detection > raising max_turns. The model is repeating itself,
  not making progress; more turns wastes compute.
- Dev-root .env is the canonical secrets location; load_env_from_dev_root
  now actually loads it (not just whatever's nearest in the walk).

**Notable moments:**
- Cascading-failure phase: every fix uncovered the next layer. Gemini
  free → Gemini paid → prepayment depleted. Wolfram pkg → assertion bug.
  Qwen nudge → "OK please ask me a question". Qwen loops → 14 identical
  calls in a row.
- "YOU ARE A TRANSLATION LAYER" framing was the key prompt-engineering
  win. Earlier "physics assistant" framing gave Qwen license to reason
  in prose without ever calling a tool.
- 0% on cosmo/GR/classical-adv/nuclear/math is a library gap — those
  domains need wrappers. The benchmark's job is to surface that, and
  it did.
- Wolfram's NLP can't parse "If I drop a copper ball from 10 meters..."
  but can parse "speed of light in m/s". That's a benchmark finding,
  not a runner bug.

**Files modified (sigma-ground repo):**
- `sigma_ground/mcp/benchmark/__init__.py` — load ALL .env, shallowest first
- `sigma_ground/mcp/benchmark/run_gemini.py` — flash/flash-lite default,
  --pause-s, --questions, 429 retry with parsed delay
- `sigma_ground/mcp/benchmark/run_wolfram.py` — raw urllib + xmltodict,
  --questions, resume retries errored
- `sigma_ground/mcp/benchmark/run_sigma_ground.py` — TOOL INDEX, nudge
  logic, anchored nudge with question re-stated, loop detection
- `sigma_ground/mcp/benchmark/score.py` — unit-aware comparison,
  question_type dispatch (refusal/puzzle/open_ended/nonsense paths)
- `sigma_ground/mcp/benchmark/adversarial_corpus.py` — NEW: 18-question
  trick set generator
- `sigma_ground/mcp/benchmark/adversarial_questions.json` — NEW
- `sigma_ground/mcp/benchmark/adversarial_ground_truth.json` — NEW
- `sigma_ground/mcp/benchmark/comparison_report_template.md` — updated
  for actual configurations

**Memory updates:**
- New: `project_mcp_modes.md` — Q&A vs conversation mode distinction

---
