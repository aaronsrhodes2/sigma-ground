# SkippyVerse roadmap: phases (set by Aaron, 2026-10-12)

The rule for every phase: **the game only claims the physics it really implements.** The engine knows rigid bodies, gravity, friction and
buoyancy-by-our-code; it knows nothing about thermodynamics, so we do not pretend. Every physical number comes from Mentat/sigma-ground
(PLATINUM section 21) with its source; the game is the stage, not the calculator.

## Phase A: "Bring the toys" (NOW): form and functionality

Goal: **a holodeck.** Say what you want and it appears, looks right, behaves believably, and the world around it feels alive. Any playground
the voice can summon. No new physics domains; as much of the construct's own physics as is realistic.

| # | Area | What it means | State |
|---|---|---|---|
| A1 | **World constants** | One config file for the construct's physics (gravity, blanket air drag as per-object damping, restitution/friction by material, time scale), each value with its source; Aaron's targets as the starting point (about -30% gravity acceleration, 2% blanket air friction). Values computed by Mentat when it is back, not by hand. | to build |
| A2 | **Water** | Pools, ponds, a sea, summoned by voice: a rendered surface (our own, no game content), buoyancy and drag from each object's real density (cards carry size and mass), splashes, swimming. Real fluid simulation (particles) is Phase B. | to build |
| A3 | **Stage / environment summoning** | "Make it a beach / a meadow / night": ground, sky and light as summonable things (CC0 textures, a sky dome), not a fixed map. | to build |
| A4 | **Objects on demand** | 4,100 free models streamed by the beam protocol (done); colour and lighting look pass (running: LK2a); caches (client and packed payloads); Draco (done). | done / polishing |
| A5 | **Assemblies** | Multi-part builds as one ledger entry (houses, vehicles, machines): spawn, undo, move as a group; importer for Advanced Duplicator 2 dupes is optional. | to build |
| A6 | **Hands and finger beam** | Hands instead of the gun; finger beam does what the gravity gun did (point, grab, throw, freeze); input behind an event layer so gestures can be added later. | to build |
| A7 | **Skippy alive** | The avatar as a natural part of the world: lots of animation and vocalisation (see below). | to build |
| A8 | **Faces** | Google profile picture becomes the player's head (link code between the portal and the game account). | to build |
| A9 | **Look** | Skippy (done, hat still to check), objects (running), beam / wisp / labels, the stage. | running |

### Skippy alive (A7)
- **Behaviour director:** hover, drift near the player, look at what you build, react to events (a beam finishing, a failure, a jump, a big pile),
  with cooldowns and a chattiness setting; "shush" and "be chatty" commands.
- **Animation:** hat nod and head turn with speech, jaw and tongue driven by the voice level, the pointing hand, shrugs, a hat tip, bobbing, small
  flourishes when he arrives or something lands.
- **Vocalisation:** remarks on a whim now and then, comments on what is around, humming, and bad opera: bellowed "LA LA LAAA" lines in his own voice.
  Spoken lines come from a pre-rendered bank (cheap, instant, quota-free) plus live lines for conversation; budgeted so the voice quota is never burned.
- **Natural conversation:** the live-Skippy presence pass (speed, awareness of what was just built, interruption).

## Phase B: "Adding in physics" (AFTER the toys)

Real domains run by sigma-ground/Mentat and shown in the game: thermodynamics (heat, fire, phase change), fluid simulation, electromagnetism
(the railgun: offline verb and recorder already built, in-game player shelved), chemistry, quantum emulation. Each gets a trust label.
Needs: coupled solvers, a richer game read-back (colour, glow, events, readouts), and a live game-to-Mentat path (see `SKIPPYVERSE_PHYSICS_REUSE_MAP.md`).

## Future (explicitly parked)

Viture AR glasses and hand tracking; weather and day-night (StormFox); the backyard 3D scan as the first custom map; two Skippys building a town;
Draco/texture shipping to clients; VRMod. The home-automation track (Leviton lights over HomeKit, audio router, Alexa retirement) runs separately.

## Decisions on record
- Download budget 200 MB per player; objects stay dynamic, Workshop or map content is delivered at join and kept small.
- No mods installed for now (Wiremod, Advanced Duplicator 2 are not needed; our own ledger and assembly cards cover it).
- A game I launch never takes Aaron's mouse; the engine-owned mouse only for a hands-on test he is told about.
