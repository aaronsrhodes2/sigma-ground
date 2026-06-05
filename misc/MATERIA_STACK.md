# The Mentat Stack — architecture & vocabulary

**Mentat** is the system — the grounded physics-computer (Quarksum matter +
Sigma Ground energy, exposed as the MCP): it *computes*, it doesn't guess (a
human-computer in the Dune sense — the LLM never does the reasoning). Its
engines are **Deckard** (shape), **Materia** (movement), and **Radiance**
(light), with the **Translator** as the language intake that routes a question
but never computes the answer.

Settled over the 2026-06 design sessions. Intent flows down; matter and light
flow up.

## The layers

| Layer | Name | Home | Owns |
|---|---|---|---|
| Author | **Translator** | `materia/translator.py` | NL → a declarative **Spec** (verbs *and* constructs). 7b qwen + deterministic classifiers. Routes; never computes. |
| Matter | **Deckard** (the Shaper) | `shapes.py`, `csg.py`, `inventory/` (+ web) | **Identifies** a named object → grounded parametric form (knowledge-first, web-fallback, cited/flagged), then **compiles** → SDF/CSG field + layers + mass. |
| Energy | *(sigma-ground field)* | `field/interface/` | Material properties + **interface** physics (drag, friction, thermal, optics) at temperature and relative motion. |
| Movement | **Materia** | `materia/` | Evolves the kinematic state of matter **layers** through time. |
| Render | **Radiance** | *(new; supersedes `matter-shaper`)* | Ray-marches the SDF + first-principles light transport → pixels. Free orbit camera. |

**Deckard** (named for the Diablo NPC who identifies unidentified items) is the
Shaper *and the shape researcher*: it turns a *name* into grounded *matter* by
researching the item's real-world spatial dimensions, fitting the primitive kit
to them, and compiling. Its field has **two consumers** — Materia (simulate)
and Radiance (render) — so it sits *below* both. Shaper and renderer are
separate components: Deckard makes the matter; Radiance draws it. ("MatterShaper"
was always the renderer.)

## Data flow

```
"make me a cup of water and tip it"
  → Translator : construct spec (revolved profile, offset-shells, materials, ratios) + action (tip)
  → Deckard    : researches the cup's real dimensions/structure, fits primitives, compiles → SDF + layers + mass
  → Materia    : integrates the topple (rigid CSG; water hands to SPH once it moves)
  → Radiance   : ray-marches each frame with first-principles material optics
  → save       : the spec itself (deterministic replay)
```

## Settled principles

- **Compile-then-run.** The LLM emits a Spec; deterministic code evaluates it.
  Complexity lives in the verbs/shaper, not the prompt — which is why a 7b
  suffices and a frontier model isn't needed.
- **Identify, then freeze.** Deckard's research resolves a name → concrete
  dimensions, which are *frozen into the spec* — so replay stays deterministic
  and "a different cup next time" = a re-run (seeded) research. Deckard
  researches *form-facts* (measurements, proportions, topology) and **fits
  primitives** to them; it never ingests a foreign mesh or a specific
  copyrighted model.
- **One SDF, five jobs.** Shape, mass/inertia, collision (`f<0`, normal `∇f`,
  depth `−f`), interface location, render — one implicit field, not polygons.
- **Geometry = mass-ratio.** Two readings of one dial; author *one* (thickness
  or ratio), derive the other. Volumes are analytic → exact.
- **Two regimes.** Analytic CSG for rigid/elastic constructs; particle/SPH
  continuum for fluids and large deformation/fracture. (Already split in code.)
- **Color is emergent** — from material (atom → Drude/Fresnel) today, from
  molecular content (chromophores/absorption) on the roadmap. Never painted.
- **Observer effect is quantum-only.** A chair's appearance is
  observer-independent (render it classically); decohering-vs-hidden-observer
  physics lives in a separate quantum-optics mode (which-path, ghost imaging).
- **Determinism buys three asks free:** *not-faked* (render draws engine
  numbers), *save/replay* (the spec is the save file), *reproducible variety*
  (seed the generative author).
- **Groundedness everywhere.** Every number tool-grounded or flagged; clarify
  rather than guess; the pinnacle test is tracked as expected-red, honestly.

## The simulation-layer contract (2026-06)

The renderer should render **the way nature does** — physical light transport from
the laws — not with renderer tricks (baked colours, fake reflections, hand-tuned
materials). So the **simulation layer carries almost nothing**, and the renderer
**derives the rest** from the physics libraries. Per object, the sim layer holds:

| Field | Default | The renderer derives from it |
|---|---|---|
| **Material** (composition + shape) | — (required) | colour (Drude/band-gap/crystal-field/dielectric), mass, density, modulus, sound speed, restitution, refractive index, emissivity, … |
| **Temperature** (K) | **STP (293.15)** | incandescent glow via **Planck × Kirchhoff** (ε=1−R); below the Draper point ~700 K it shows reflectance, no glow. At STP → cold. |
| **Motion over time** (pose stream) | **zero (static)** | rigid rotation + translation playback; the dynamics it implies (collision, bounce…) |

The seed is a few hundred bytes (atoms, counts, bonds, a shape, a temperature, a
motion). Everything you *see* — colour, glow, mass, the bounce, the ring pitch,
the reflection — is recomputed from that seed on demand. **Physics is the codec.**
A conventional engine stores a 50 MB mesh + a material card and still can't tell
you the burst RPM or the glow temperature; this stores the *cause* and derives the
*consequences*, smaller and more complete.

- **Emergent, proven so far:** colour (metals/semiconductors/glazes/dielectrics),
  density (crystallographic, <0.5% vs tabulated), the mechanical signature
  (modulus/sound/restitution/ring-pitch from cohesive energy), Fresnel reflection,
  wind-driven water ripples (gravity-capillary dispersion), and **incandescence**
  (heated matter glows — no colours defined).
- **Still tricks (honest):** the camera headlight, hemispheric ambient, AO, and the
  hard-coded reflection sky all stand in for **global illumination**. The path to
  closing the gap is a **path tracer** — emit light from the real sources
  (including the incandescent objects themselves), bounce it off the BRDFs we
  already derive, and delete the fill tricks.

## The primitive kit (the Shaper's leaves)

`{ plane, sphere, cylinder, cone } + affine transforms + boolean CSG + one sweep`,
plus the **offset-shell** operator `shell(f, t) = max(f, −f − t)` for
uniform-thickness layers. Scale → ellipsoids; sweep → torus / helix / möbius. A
**voxel / scanned-SDF leaf** is the escape hatch for organic forms quadrics
can't reach.

## Deckard's source ladder (most-grounded first)

1. **Standards & dimension databases** — ISO/ANSI for engineered parts (bolts,
   pipes, bricks), furniture/anthropometric references. Exact numbers for
   standardized things.
2. **3D-model repositories** — used as *reference* for proportions and topology;
   Deckard extracts dimensions and fits primitives, it does **not** import the
   mesh (wrong representation, and keeps us clear of licensed assets).
3. **Model knowledge / general web** — the everyday and the long tail; cited, or
   flagged when it's a best guess.

## Status (2026-06)

- **Built & green:** Materia v1 (terminal velocity), v2 (drag heating), the
  Translator (NL → verb specs, deterministic + qwen residual + clarify),
  **Deckard v1** (`sigma_ground/deckard/`: research → layered-SDF compile;
  coffee cup → mass/CoM/inertia, SDF integrator vs closed-form 2.1%; 7 tests).
- **In flight:** Family-D verbs (high-altitude descent, supersonic drag,
  `C_d(Mach)`).
- **Next:** **rigid-body rotation** in Materia (fed by Deckard's CoM + inertia)
  → the **Radiance** viewer → the chair-topple pinnacle.
- **North star:** `misc/PINNACLE_TEST_chair_topple.md` — "the chair that falls
  like a chair" (expected red).
