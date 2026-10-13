# SkippyVerse roadmap: phases (set by Aaron, 2026-10-12)

The rule for every phase: **the game only claims the physics it really implements.** The engine knows rigid bodies, gravity, friction and
buoyancy-by-our-code; it knows nothing about thermodynamics, so we do not pretend. Every physical number comes from Mentat/sigma-ground
(PLATINUM section 21) with its source; the game is the stage, not the calculator.

## Phase A: "Bring the toys" (NOW): form and functionality, Skippy, solid LLM turns

Goal: **a holodeck.** Say what you want and it appears, looks right, behaves believably, and the world around it feels alive. Any playground
the voice can summon. No new physics domains; as much of the construct's own physics as is realistic.

| # | Area | What it means | State |
|---|---|---|---|
| A1 | **World constants** | Real-world physics only (Aaron, 2026-10-12): gravity is the library's Earth value in engine units (the engine's own 600 HU/s2 is about 1.17 x stronger), presets for the Moon, Mars, Jupiter and free fall, and real aerodynamic drag per object (F = 1/2 rho Cd A v2, rho from the library, Cd by shape class, estimated, owner-should-confirm). One table with sources (`stage/gmod/world_constants.json`), applied by `mentat/world.lua`, voice commands, admin-only; the game models gravity and drag only, nothing thermal. See holodeck README "The construct's physics". | built and tested offline; live drop and settling check (`world_drop.py`) still to run |
| A2 | **Water** | Pools, ponds, lakes and seas by voice ("add a pool", "make a lake", "fill it with mercury", "drain it", "what floats"), real-world physics only: Archimedes buoyancy with g from the construct, quadratic + Stokes drag with the liquid's own density and viscosity (water, seawater, ethanol, glycerol, mercury from the library; oil and honey refused, the library lacks them), exact submerged-volume geometry for boxes, spheres and cylinders, swimming, a drawn surface with the real dispersion relation and Fresnel from the library's refractive index, splashes sized by the entry's energy. Caps, bases (mass and volume are only as good as the card's) and what is NOT modelled are in the holodeck README "Water". Particle fluid is Phase C. | built and tested offline (4 test files); live check: see the README |
| A3 | **Stage / environment summoning** | "Make it a beach / a meadow / night": ground, sky and light as summonable things (CC0 textures, a sky dome), not a fixed map. | to build |
| A4 | **Objects on demand** | 4,100 free models streamed by the beam protocol (done); colour and lighting look pass (running: LK2a); caches (client and packed payloads); Draco (done). | done / polishing |
| A5 | **Assemblies** | Multi-part builds as one ledger entry (houses, vehicles, machines): spawn, undo, move as a group; importer for Advanced Duplicator 2 dupes is optional. | to build |
| A6 | **Hands and finger beam** | Hands instead of the gun; finger beam does what the gravity gun did (point, grab, throw, freeze); input behind an event layer so gestures can be added later. | to build |
| A7 | **Skippy alive** | The avatar as a natural part of the world: lots of animation and vocalisation (see below). | **A7a built and tested offline, live-checked hands-off (2026-10-13):** behaviour director (chattiness, cooldowns, never over a human or himself, interruptible), layered body language (hat tip, flourish, conductor, shrug, emphasis, voice-driven head chatter and tongue glow), 183 seed lines + callbacks + model-written lines under a **20 new lines a day** budget with a 14-day no-repeat history and a private clip cache, a per-player notebook with summarised notes, roast levels, commands (shush, be chatty, roast me harder, what do you remember, forget). See holodeck/README "Skippy alive". Still open: the real model route (needs the Anthropic key on the bridge machine), a judgement by ear of the voice and pacing, humming, the persona's live conversation pass, a human second client |
| A8 | **Faces** | Google profile picture becomes the player's head (link code between the portal and the game account). | to build |
| A9 | **Look** | Skippy (done, hat still to check), objects (running), beam / wisp / labels, the stage. | running |

### Solid LLM turns (part of A7)
Skippy decides whether a line was aimed at him (wake word, gaze and proximity, the turn before), helps when asked, comes when called and goes when shooed.
Model output is validated and falls back to the deterministic parser when unsure; resolved objects, packed payloads and common replies are cached;
a corpus of real turns measures reliability.

### Skippy alive (A7)
- **Behaviour director:** hover, drift near the player, look at what you build, react to events (a beam finishing, a failure, a jump, a big pile),
  with cooldowns and a chattiness setting; "shush" and "be chatty" commands.
- **Animation:** hat nod and head turn with speech, jaw and tongue driven by the voice level, the pointing hand, shrugs, a hat tip, bobbing, small
  flourishes when he arrives or something lands.
- **Vocalisation:** remarks on a whim now and then, comments on what is around, humming, and bad opera: bellowed "LA LA LAAA" lines in his own voice.
  Spoken lines come from a pre-rendered bank (cheap, instant, quota-free) plus live lines for conversation; budgeted so the voice quota is never burned.
- **Natural conversation:** the live-Skippy presence pass (speed, awareness of what was just built, interruption).

## Phase B: "Give the toys properties" (AFTER Phase A)

Aaron, 2026-10-12: "we will need to scrutinize each object and give it properties that the game can use." Each object gets a property sheet: material and
makeup, density and mass, conductivity, melting point, hardness and so on, each labelled measured / sourced / estimated / guessed. This is the lazy repair
design (`SKIPPYVERSE_LAZY_REPAIR_DESIGN.md`): real materials make mass derivable. **The property sheet is the contract between Phase B and Phase C**:
decide its fields first (a short design pass), so Phase C can grow without redoing the objects. Packaging and quantifying come from sigma-ground
(quarksum was folded into it).

## Phase C: "Add the physics" (AFTER Phase B)

The game gains systems that READ the properties objects already carry and infer behaviour from them, run by sigma-ground/Mentat and shown in the game:
thermodynamics (heat, fire, phase change), fluid simulation, electromagnetism and electronics with simulated current (the railgun: ablating sabot,
plasma, a steel impact surface; the offline verb and recorder are built, the in-game player is shelved), chemistry, quantum emulation. Each gets a trust label.
The game claims nothing about a domain until it implements it. Needs: coupled solvers, a richer game read-back (colour, glow, events, readouts), and a
live game-to-Mentat path (see `SKIPPYVERSE_PHYSICS_REUSE_MAP.md`).

## Future (explicitly parked)

Viture AR glasses and hand tracking; weather and day-night (StormFox); the backyard 3D scan as the first custom map; two Skippys building a town;
Draco/texture shipping to clients; VRMod. The home-automation track (Leviton lights over HomeKit, audio router, Alexa retirement) runs separately.

## Decisions on record
- Download budget 200 MB per player; objects stay dynamic, Workshop or map content is delivered at join and kept small.
- No mods installed for now (Wiremod, Advanced Duplicator 2 are not needed; our own ledger and assembly cards cover it).
- A game I launch never takes Aaron's mouse; the engine-owned mouse only for a hands-on test he is told about.
