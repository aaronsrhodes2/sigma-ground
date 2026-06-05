# What-If Simulation Corpus — the simulator's north-star test set

Source of premises: Randall Munroe, *What If?* (2014) — table of contents only.
We use the **scenario premises** as test targets. We do **not** reproduce the
book's analysis, numbers, or artwork; every gloss and every answer here is our
own physics, computed by our own stack.

Purpose: this is to the **Simulator** what `ground_truth.json` is to the Q&A
switchboard — a fixed, external, "known-hard" target set. Each entry is tagged
by physics domain and, critically, by **tractability on our stack**
(quarksum = matter, sigma-ground = energy/closed-form, Simulator = dynamics).

Tractability tiers:

- **A — Closed-form, we can compute it today** (existing tools, exact, self-checking)
- **B — One new dynamics model** (an ODE integrator + one force/heat law)
- **C — Fermi estimation cascade** → belongs in the **Procedures** layer, not the dynamics engine
- **D — Out of scope** (biology / sociology / pure probability — not a physics sim)
- **E — Exotic stress test** (extreme scaling; mostly closed-form, good for breaking things)

---

## TIER A — computable today (closed-form, exact, self-validating)

| Scenario | Physics premise (ours) | Tool / model | Status |
|---|---|---|---|
| Little Planet | Walkable asteroid: can you stand / orbit / escape it? | `surface_gravity`, `escape_velocity`, `orbital_velocity` | **HAVE** |
| High Throw | Highest a human can throw a projectile | `projectile_*` procedure | **HAVE** (no-drag) |
| Orbital Speed | What "orbital velocity" actually means at altitude | `orbital_velocity`, `orbital_period` | **HAVE** |
| Speed Bump | Fastest you can hit a bump and live | impulse → peak g-force | trivial (Δv/Δt) |
| Stirring Tea | Can you heat tea by stirring it? | mechanical work → ΔT energy balance | trivial — great "is it significant?" test |
| Weightless Arrow | Archery recoil in zero-g | conservation of momentum | trivial |
| Everybody Jump | 7e9 people jump at once → seismic energy | Σmv → joules → Richter | arithmetic + 1 conversion |
| Hockey Puck | Hit a puck arbitrarily hard | momentum + material yield threshold | A + quarksum strength |

**Why Tier A matters:** these have analytic answers, so the Simulator can
*check itself* on them on day one. They are the regression suite for the engine.

---

## TIER B — unlocked by ONE new dynamics model each

| Scenario | Premise (ours) | Model needed | Unlocks also |
|---|---|---|---|
| Steak Drop | Drop a steak from the stratosphere — does reentry cook it? | **1-D motion + quadratic drag** + aero heating | Raindrop, Free Fall, Falling-with-Helium, Sparta |
| Raindrop | One single giant raindrop | 1-D drag → terminal velocity | (same model) |
| Free Fall | Highest survivable fall into water/snow/etc. | 1-D drag + impact impulse | (same model) |
| Falling with Helium | Enough balloons to survive a fall | buoyancy + 1-D drag | (same model) |
| Sunless Earth | Sun switches off — how fast do we freeze? | **radiative-cooling ODE** dT/dt = −εσA T⁴/(mc) | Hair Dryer |
| Hair Dryer | Sealed box, ramp the power — equilibrium temp? | heat-balance ODE (Stefan-Boltzmann out) | Sunless Earth |
| Interplanetary Cessna | Fly a Cessna on each planet | **lift/drag vs atmospheric density** | (biplane example) |
| Machine-Gun Jetpack | Fire guns downward for thrust | momentum-flow thrust + rocket eq | Yoda (power) |
| Rising Steadily | Ascend at 1 ft/s into the atmosphere | **standard-atmosphere profile** P(h),T(h) | Interplanetary Cessna |
| Lego Bridge | A Lego bridge spanning a city / sea | **structural yield** (E, σ_yield) | beam-bending example |
| Global Windstorm | Earth's spin stops, the air keeps going | atmospheric KE + drag | — |

---

## TIER C — Fermi cascades → Procedures layer (not the dynamics engine)

A Mole of Moles · Human Computer · FedEx Bandwidth · Twitter · Last Human Light ·
Drain the Oceans (I & II) · Updating a Printed Wikipedia · Laser Pointer (flux) ·
Alien Astronomers (inverse-square) · Lethal Neutrinos (cross-section) · Yoda (power) ·
Loneliest Human · All the Lightning.

These are **multi-step arithmetic with sourced constants** — exactly what the
`procedures.py` layer is for. They never need a time-integration loop.

---

## TIER D — out of scope (not physics simulations)

Soul Mates · Common Cold · No More DNA · Self-Fertilization · Facebook of the Dead ·
Random Sneeze Call · SAT Guessing · Lost Immortals · Flyover States · Everybody Out ·
New York-Style Time Machine.

(Probability, biology, demographics, sociology. The clarification-classifier
should recognize these as "not a physics quantity" and say so.)

---

## TIER E — exotic stress tests (extreme scaling; break-the-engine fun)

Relativistic Baseball (0.9c + air fusion) · Neutron Bullet (neutronium density/gravity) ·
Richter 15 (seismic energy scaling) · Expanding Earth (gravity scaling) ·
Spent Fuel Pool (radiation attenuation) · Periodic Wall of the Elements (reactivity) ·
Glass Half Empty (cavitation) · Orbital Submarine (vacuum/pressure).

---

## The convergence worth noticing

The Captain's own six simulator examples are **Munroe questions in disguise** —
independent arrival at the same target set:

| Captain's example | Munroe twin | Shared model |
|---|---|---|
| Car into a wall — survivable Gs | **Speed Bump** | impulse → peak g-force |
| Biplane vs monoplane L/D | **Interplanetary Cessna** | lift/drag |
| Lead vs steel bullet | **Hockey Puck / Neutron Bullet** | momentum + material yield |
| Moon's-moon orbit | **Orbital Speed** | n-body integrator (`nbody.py` exists) |
| Candle ignites pinewood | **Hair Dryer** | thermal balance / ignition |
| Jug drains in N seconds | **Drain the Oceans** | Torricelli flow |

Two independent sources picking the same scenarios is the signal that these are
the *right* first models to build.

---

## Recommended build order (max questions unlocked per model)

1. **1-D motion + quadratic drag** (the ballistic/terminal-velocity integrator)
   → unlocks Steak Drop, Raindrop, Free Fall, Falling-with-Helium, Sparta, and
   upgrades High Throw from no-drag to real. **6 questions, 1 model, analytic
   terminal-velocity check.** This is the first dynamics model.
2. **Thermal-balance ODE** (radiative cooling / heat-in-a-box) → Sunless Earth,
   Hair Dryer. Stefan-Boltzmann already in the library.
3. **Orbital integrator** (extend existing `nbody.py`) → Orbital Speed,
   Longest Sunset, Loneliest Human, moon's-moon.
4. **Aerodynamic L/D + planetary atmosphere tables** → Interplanetary Cessna;
   also supplies the drag coefficient for model 1.
5. **Structural yield** (needs quarksum matter + tabulated strength) → Lego
   Bridge, Hockey-Puck failure.

Model 1 is the keystone: most questions, exercises the integrator, and is
self-checking against the closed-form terminal velocity.

> **STATUS — Model 1 BUILT (Materia v1, 2026-06).** `sigma_ground/materia/`.
> `terminal_velocity_drop` runs through the real density gradient and
> self-validates: in uniform air the integrator matches the closed-form
> terminal velocity to **0.05%**. Copper sphere from 10 km → 160 m/s impact,
> peaking 201 m/s aloft. 4 regression tests green.

---

## Quantitative stress list (Captain's, 2026-06) — mapped to Materia models

A second, more *numeric* corpus the Captain supplied. The valuable property:
most of these have a **closed form**, so Materia can validate them exactly the
way it validates terminal velocity — the gold standard for "never wrong."

| Scenario | Materia model | Self-check (closed form) | Tier |
|---|---|---|---|
| **Terminal Velocity Pivot** (Cu sphere, gradient) | ballistic+drag | v_t=√(2mg/ρC_dA) | **DONE ✓** |
| Frictionless Sledge (does it move?) | statics force-balance | μ_s N vs applied | A |
| Vertical Column / **Euler Buckling** (P_cr) | structural | P_cr=π²EI/(KL)² | B |
| Torsional Snap (shear failure) | structural | τ=Tr/J | B |
| Horizontal Cantilever (yield) | structural | σ=Mc/I | B |
| Mountain's Limit (crush height) | structural | h=σ_c/(ρg) | A/B |
| **Hydraulic Shock** / Water Hammer | fluid transient | Joukowsky ΔP=ρcΔv | A/B |
| **Radiative Vacuum Cool** (800→400 K) | thermal ODE | dT/dt=−εσA T⁴/(mc), analytic | **B — rec. model #2** |
| Viscous Drag Stop (plate on oil) | fluid ODE | Couette: exp decay τ=m d/(μA) | B |
| Kinetic Orbit Decouple (tether sever) | orbital integrator | energy/momentum split | C |
| Atmosphere Stop (air at 465 m/s) | aero + thermal | dynamic pressure ½ρv² | B/E |
| Flat-Earth Gravity Profile | gravity integral | disk-potential integral | E |
| **Relativistic Impact** (1 g @ 0.9c) | relativistic | KE=(γ−1)mc² | A (reuses our γ tools) |

Recommended **model #2: Radiative Vacuum Cool** — first-principles-exact,
reuses the library's Stefan-Boltzmann, has a closed-form solution to
self-check against (same discipline as model 1), and opens the whole thermal
branch (Sunless Earth, Hair Dryer, candle ignition).
