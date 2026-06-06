# Water-Sim Regeneration — local-models-only, by-the-book

**Scenario:** *"A shallow pond of water at room temperature with a 5 m/s wind
blowing across it, and a copper ball half-sunk in it."* — the beautiful water sim,
re-generated with **no frontier model touching the geometry** (Golden Rule: shapes
are Deckard's, local models only).

**Verdict: achievable.** The pipeline exists (`Deckard Construct →
radiance.scene_export.construct_to_scene → scene_spec_to_sdf → Radiance`). The
gaps are bounded and listed below by owner. Mentat's two enablers are already done.

---

## Division of labor (respects the seam)
- **Deckard (shapes, local models):** the box + water slab + copper sphere, and a
  height-field/displaced-surface primitive for the ripples.
- **Materia (physics):** wind→stress→ripple field + buoyancy. Computes the numbers
  that *parameterize* Deckard's surface — physics, not shapes.
- **Radiance (render):** render the displaced water surface; loop frames for motion.
- **Mentat (me): DONE** — `/simulation /question /render` flags (deterministic
  routing) + colloquial naming cloud ("pond", "wind blowing across", "ripples",
  "half sunk" now route). This removes the model-choosing problem from the test.

---

## Step 0 — FIX THE WATER SEEDS FIRST (blocker)
`sigma_ground/field/interface/liquid_water.py` at 20 °C is off vs NIST (violates
Deckard's Golden Rule 8). The ripple dispersion uses σ and ρ, so wrong seeds →
wrong ripples:

| quantity | model @20 °C | NIST | error |
|---|---|---|---|
| `water_surface_tension` | 0.1006 N/m | 0.0728 N/m | **+38%** |
| `water_viscosity` | 1.01e-4 Pa·s | 1.00e-3 Pa·s | **10× low** |
| `water_density` | 1032.6 kg/m³ | 998.2 kg/m³ | +3.5% |

Fix these (and add NIST-anchored tests) before building the wave verbs.

---

## Step 1 — Materia verbs (3 new; standard analytic physics, Golden Rule 6)
Seeds available: `liquid_water.{water_density, water_surface_tension,
water_viscosity}` (post-fix); air density `gas.ideal_gas_density('N2',T)≈1.16`;
copper density `surface.MATERIALS['copper']['density_kg_m3']=8960`; g = 9.80665.

1. **`buoyancy(material_key, fluid='water', T)`** → floats? + submerged fraction.
   - f_submerged = ρ_body/ρ_fluid (if <1 floats; if ≥1 sinks). Copper 8960/998 ≈ 9
     → **sinks**. So "half-sunk" is *geometry* (shallow pond, ball on the floor,
     half above the waterline), NOT flotation — Deckard places it; this verb states
     the fact and the waterline.
   - validate: pine floats f≈0.5; copper sinks; ice f≈0.92.
2. **`wind_surface_stress(U_wind, T)`** → shear stress on the water.
   - τ = C_d · ρ_air · U²,  C_d ≈ 1.3e-3 (neutral 10 m).  At 5 m/s: τ ≈ 0.038 Pa.
   - friction velocity u* = sqrt(τ/ρ_water).
3. **`capillary_gravity_ripples(U_wind, T, fetch=None)`** → the ripple field.
   - dispersion: ω² = g·k + (σ/ρ)·k³ ; phase speed c = sqrt(g/k + σk/ρ).
   - minimum phase speed ~0.23 m/s at λ ≈ 1.7 cm (the capillary-gravity crossover)
     — the wavelength wind first excites. Dominant wind-ripple λ from u*; amplitude
     A from a bounded energy-input estimate (label as estimate, not full spectrum).
   - **outputs Deckard needs:** wavelength λ (m), amplitude A (m), wave direction
     (= wind direction), phase speed c (m/s). These parameterize the surface.
   - validate: c_min ≈ 0.23 m/s; λ at c_min ≈ 0.017 m (textbook capillary-gravity).

---

## Step 2 — Deckard shape contract (his lane)
- **Construct:** box container ∪ water slab (depth ≈ ball radius, so the ball is
  half-exposed) ∪ copper sphere resting on the floor. Plain CSG (box, sphere).
- **Displaced water surface:** a height-field surface primitive
  `h(x,y,t) = A · cos(k·(x·dx + y·dy) − ω t + φ)` (k = 2π/λ, ω from dispersion,
  (dx,dy) = wind direction). A is Materia's amplitude; λ, ω, direction are
  Materia's outputs. **This is physics-driven geometry, not an invented shape** —
  Deckard builds the primitive; Materia supplies its parameters.

## Step 3 — Radiance
- Render the displaced surface SDF with the existing `WATER` material
  (`radiance/materials/library.py`). Static snapshot first (t fixed); then a phase-
  advancing frame loop (precedent: `render_turntable`) for the rippling motion.

## Step 4 — Drive it locally
`run_sigma_ground.py --mode conversation` (or `--tools simulation`); the colloquial
cloud maps the user's words to the verbs; the `/simulation` flag pins routing so
qwen2.5:7b only fills slots. Target: equal/better than the Claude-shapes original.

---

## Honest caveats
- Linear capillary-gravity wave theory + an empirical wind-input estimate — **not**
  full Navier-Stokes CFD. Label it as such.
- "Half-sunk copper ball" = shallow-pond geometry; copper sinks (buoyancy verb
  confirms). Do not render it floating.
- Seeds must be fixed first (Step 0) or the ripple numbers are wrong.
