# SkippyCraft Plan — sigma-ground voxel theaters you can fly into, in Minecraft

Named by the Captain, 2026-10-30. Builds on the voxel theater (2026-10-30, `misc/SESSION_LOG.md`): `deckard/voxel_exec.py`
(MineBench code → material blocks), `radiance/voxel_scene.py` (render), `materia/voxel_world.py` (the tick: heat, phase, settling),
`deckard/voxel_resample.py` (resolution, set at build time), `field/interface/block.py` (one block of matter, every property sourced).
Ports the Garry's Mod holodeck (`stage/gmod/holodeck/`, `stage/gmod/AVATAR.md`) and uses the SkippyVerse object cards
(`stage/skippyverse/`).

## 1. Goal and success criterion

A Minecraft server where each **theater** is a region of real space run by sigma-ground. You fly into it (creative or spectator
flight), watch the physics run, and change it from inside by talking to a floating, block-built Skippy.

**Two avatars, no more** (the Captain, 2026-10-30): the Captain's own, and Skippy's virtual one. It is not a multiplayer game.

**Success:** on the home server, from his own Minecraft client over the tailnet, the Captain flies into a theater, says *"Skippy, build a glass tank of mercury and drop a gold ring
and an iron bolt in it"*, watches the ring sink and the bolt float, says *"heat the floor to 1500 kelvin"* and watches what melts,
and every number on the in-game readout traces to sigma-ground with its source. Skippy hovers beside him the whole time, made of
blocks, and answers in his voice.

## 2. Non-goals

- **Minecraft never simulates.** No Minecraft physics (water flow, gravity blocks, fire spread, redstone) inside a theater: the
  plugin turns those off there. sigma-ground is the only authority. This is the difference from GMod, where RELEASE handed a body
  to Havok.
- No new physics in this project. SkippyCraft shows and drives what `VoxelWorld` already does; physics gaps are sigma-ground work.
- No survival gameplay, no mods on the player's side: a stock Minecraft Java client is all a player needs.
- Not a Bedrock-first project (see Risks: Bedrock cannot show display entities).
- **Not multiplayer.** One player (the Captain) and one virtual avatar (Skippy). No other players, so no shared-server safety
  rules, ownership, caps or rationing; the server allows only his account (whitelist) and listens only on the tailnet.

## 3. Why Minecraft

- The theater is already blocks. MineBench showed the local models build well in block code; the voxel theater made those blocks
  matter. Minecraft shows blocks natively, at any count a theater needs, to many players at once.
- Flight and walking through walls are built in (creative / spectator), so "enter as a floating avatar" costs nothing.
- Skippy is not a player: he is display entities driven by the plugin, so he needs no account or client.
- **Block display entities** (Java 1.19.4+) draw blocks off the grid: any position, rotation, scale (smaller than a block too),
  smoothly interpolated. That is how Skippy floats, and how a theater finer than 1 m can still look right.
- Paper (the server) is free and has a large, stable plugin API. Players need a Minecraft Java licence.

## 4. Architecture

```
 the Captain's Minecraft Java client (any of his machines)
          │ game protocol, over the tailnet only
          ▼
 ┌──────────────── home (skippy-core recommended) ───────────┐
 │ Paper server + SkippyCraft plugin (Java)                   │
 │   draws blocks, turns off vanilla physics in a theater,    │
 │   relays his chat/commands, draws Skippy                   │
 │        │ localhost HTTP + WebSocket (block diffs)           │
 │        ▼                                                    │
 │ theater service (Python, sigma-ground)                      │
 │   one VoxelWorld per theater, the tick loop,                │
 │   the holodeck planner, the Skippy bridge  ──► Skippy Core, │
 │                                               Mentat (local)│
 └─────────────────────────────────────────────────────────────┘
```

- **Plugin (thin):** places/clears blocks from diffs; maps each material to a Minecraft block for LOOK only (lead → iron-grey
  block, molten → its glowing variant; the real material, temperature and phase stay in the service); turns off vanilla physics in
  theater regions; sends chat lines starting "Skippy" and `/theater` commands to the service; draws the avatar (display entities);
  shows the readout (a boss bar / sidebar: time, energy ledger, mass, what changed).
- **Theater service (all the physics):** owns each `VoxelWorld`; ticks it on a clock; sends only changed blocks; serves
  `block_properties` for "what is that?"; turns "Skippy, ..." into actions (the holodeck planner, the local model through Skippy
  Core). numpy on the CPU.
- **Brains next door:** Skippy Core (words → actions, MineBench-style build code) and Mentat run on skippy-core, so the service
  reaches them locally: no Funnel, no keys over the internet. (The GMod cloud server never got this bridge:
  `stage/gmod/CLOUD_SERVER.md`, "a bridge container on the VM is the missing piece". At home it is not missing.)
- **Why home, not the games VM:** one player who is always on the tailnet; the models are local; no cloud cost or public game
  port. skippy-core runs it as a deployer-managed service like Mentat; skippy-pc would also do (it has the Minecraft client).

## 5. Scale, resolution and limits

- A theater has a real size (`extent_m`) and a resolution, both fixed when it is built (`field_from_voxel_exec`). Minecraft always
  draws a grid block 1 m wide; the readout shows the true block size ("1 block = 5 cm"). A coarse theater for a planet and a fine
  one for a teacup are both fine: the physics uses the true size.
- **Sizes:** a theater region up to 128³ blocks at first (2 M cells; the tick is numpy and fine at that size, see Phase 1's
  measurement). Minecraft's build height (384) and chunk loading set the hard limits.
- **Tick rate:** Minecraft ticks at 20 Hz; the service ticks the world at its own rate (simulated seconds per real second is a
  theater setting) and sends diffs at most a few times a second. Settling moves one cell per tick: watchable, not fast dynamics.
- **Air is never a block:** label 0 is Minecraft air. A theater's surroundings (the ground plane, the sky) come from its theater
  type (plain / room / void / deep space), as in Radiance.

## 6. What ports from the GMod holodeck, and what does not

| GMod piece | SkippyCraft |
|---|---|
| `parse.lua` (words → request, typo-tolerant, never asks) | **port** to Python in the service (the rules, not the Lua) |
| `plan.lua`, `layout.lua` (rows, circles, U/L, facing, around/beside/on, against a wall) | **port**: layout is engine-independent; poses snap to the block grid |
| `catalog.lua` (127 Half-Life 2 / CS:S models) | **replaced**: things are built as block code by the local model (MineBench), sized from SkippyVerse cards (real sizes, honesty clause) |
| `skippy_bridge.py` (agent turns a wish into a request, checked before the game sees it) | **port** as-is in spirit; Skippy Core is local to the service |
| Safety on a shared server (owners, caps, cooldowns, rationed bridge) | **not needed**: one player |
| Avatar (`AVATAR.md`: call/dismiss words, hover behind the right shoulder, follow, snap past 8 m, voice from where he floats) | **port** the behaviour; the body becomes display blocks (section 7) |
| Replays of frozen trajectory bundles | **not needed**: theaters run live |

## 7. Skippy, made of blocks

- **Body:** the same look the GMod avatar settled on (a beer can, `#00FF88` label band, amber-braided dark bicorn), built in blocks
  by the voxel pipeline from `portal/static/skippy/skippy.glb` (Deckard voxelize, a few hundred cells), shown as block display
  entities, so he floats off the grid at any scale.
- **Behaviour:** `AVATAR.md` where it applies: "Skippy, come here" / "go away"; hovers behind the Captain's right shoulder
  with a small bob; follows; snaps when far; faces him while speaking. The room box's idle float and talking loop (SkippyTel
  `a8bcc4d`) set the motion.
- **Voice:** text in chat always. Spoken audio needs a client-side mod or a resource pack trick; a Phase 5 question, not a promise.

## 8. Commands (what "run and alter from inside" means)

Said to Skippy, or as `/theater …`:

- **Build:** "Skippy, build a copper plate with a lead block on it" → block code → a new theater (or added to this one).
- **Change state:** heat / cool a region to N kelvin, drop an object at a point, remove, undo, reset to the start.
- **Time:** pause, play, step, speed (simulated seconds per real second).
- **Ask:** "what is that?" (the block you look at → `block_properties`: material, phase, temperature, sources); "why did it melt?".
- **Theater:** new theater with a size, a resolution and a type (plain / room / void / deep space); list; go to one.

## 9. Repo layout

- `sigma-ground/stage/skippycraft/` — the theater service (Python), its tests, and `plugin/` (Java, Gradle, Paper API).
- SkippyTel `core/` — a `skippycraft` service on skippy-core (Paper + the service, `core/compose.*.yml`, kept up by the deployer
  and the doctor like Mentat), tailnet only. No games VM, no Game Servers card (it is not a shared game).

## 10. Phases, done criteria, estimates

| Phase | Work | Done when | Est. |
|---|---|---|---|
| **0 Spike** (DONE 2026-10-30: the Captain flew around the tank on Paper 26.3) | Paper server locally; a minimal plugin that reads a `VoxelField` and places its blocks once; material → block look table | the demo tank appears in a local world and a stock client flies through it | 1 day |
| **1 Live theater** (DONE 2026-10-30, over RCON: `stage/skippycraft/theater.py`; no plugin needed yet) | the service with one `VoxelWorld`, a tick loop, block diffs over WebSocket; vanilla physics off in the region; a readout | the gold ring sinks and the bolt floats in front of you; the lead melts on the hot plate; the readout's energy balance matches the service's | 2 to 3 days |
| **2 Commands** | `/theater` new/heat/drop/pause/speed/undo/reset; "what is that?" via `block_properties` | every command in section 8 works from inside, without words | 2 days |
| **3 Talking to Skippy** | port parse/plan/layout; the bridge to Skippy Core; block-code builds via MineBench-style prompts and `voxel_exec` | "Skippy, build ..." and "Skippy, heat the floor ..." work in plain words; failures say why | 3 to 4 days |
| **4 Skippy's body** | voxelize `skippy.glb`; display-entity avatar; AVATAR.md behaviour | he comes when called, hovers, follows, faces the speaker, leaves when sent away | 2 days |
| **5 At home for good** | the `skippycraft` service on skippy-core (deployer, doctor, tailnet only, whitelist of one); theater saves backed up; voice question settled | the success criterion (section 1) passes from the Captain's client over the tailnet | 1 to 2 days |

Phases 0 and 1 answer the big question (does it look and feel right?) before any of the talking is built.

## 10b. The Playground: an empirical physics playground (the Captain, 2026-10-31)

**Goal.** Radiance's goal (matter becomes pixels from the engine's own numbers, never a faked frame) carried into a world you stand
in: one theater where time flows and *every* force sigma-ground knows acts at once, and you watch it play out. Not exhibits that each
run one law (sink, melt, slosh, slit, range), but one world where a hot gold bar melts, the melt flows with momentum, sinks a
wooden block it does not burn yet, cools, freezes where it stops and glows the colour its temperature says the whole time.

**The rule: every value is derived from the block's live state.** A cell is (material, temperature, phase fraction, velocity).
Every property any solver or the renderer uses is sigma-ground's answer for THAT state, never a room-temperature card frozen for
the run: thermal conductivity, heat capacity, density (solid and liquid, thermal expansion), total and spectral emissivity,
colour and transparency (`appearance` at its thickness and temperature), glow (Planck x Kirchhoff), viscosity of a melt, strength
and stiffness, drag and lift in air (from the speed: Reynolds number, then the drag law). For speed a material's properties are
tabulated over temperature once, each entry from sigma-ground; interpolated per cell per tick. A property sigma-ground lacks for a
state is a named gap on the readout, drawn honestly (flagged), never a borrowed number.

**What acts, one clock, operator split each tick:**

| Part | From | State it reads and writes |
|---|---|---|
| Gravity, buoyancy, drag in air | `VoxelFluid` bodies, `atmosphere` | velocity, live density |
| Liquids (water, mercury, any melt) | FLIP (`dynamics.fluid.flip`), variable density per particle | velocity, temperature carried with the particle |
| Rigid bodies with momentum | `voxel_fluid` body coupling, extended to body-body contact and friction | pose, velocity, angular velocity |
| Heat: conduction, carried by moving matter, radiation, latent heat | `diffuse_fvm`, `VoxelWorld._phase`, T-dependent emissivity | temperature, melt fraction |
| Phase: melt to particles, freeze to cells, boil and vent | `VoxelWorld` enthalpy method, FLIP | phase, mass ledger |
| Impacts (the range's weapons) | `weapon_vs_armor` per-layer solve, energy to heat | velocity, temperature, holes |
| Light | `block.appearance` + `block.glow` at the cell's temperature; LabPBR (smoothness, F0, emission) | what you see |

**The empirical part: the ledger is on screen.** Total energy (kinetic, potential, thermal, latent, radiated, vented), momentum and
mass, each summed from the cells every tick; conservation drift shown, not hidden. "What is that?" on any block gives its live
card with sources.

**Phases:**

| Phase | Work | Done when |
|---|---|---|
| P1 Live cards (DONE 2026-10-31: `block(material, T)` at T, `voxel_world.live_table`, enthalpy H(T); the laws it needed fixed, the rest in KNOWN_GAPS) | properties tabulated over T per material from sigma-ground; `VoxelWorld` and `FluidVoxelWorld` read them per cell per tick | tungsten at 1500 K radiates with emissivity 0.113 (it uses 0.023 today); a test pins every card value to a direct sigma-ground call |
| P2 One world (DONE 2026-10-31: `materia/playground.py`, the `playground` theater at 2x, 5 mm blocks, ~0.2-0.35 s a tick; one liquid material, inviscid, flows at its solid's density) | heat + phase + FLIP + bodies in one `PlaygroundWorld`; melt becomes FLIP particles, cooling particles freeze back into cells | molten lead off a hot plate flows, a gold block sinks in it with momentum, the melt freezes where it stops; the energy ledger closes |
| P3 Contact and strength (DONE 2026-10-31: `dynamics/voxel_rigid.py`: bodies rotate; voxmap/point-shell contact solved by sequential impulses with warm starting, measured friction (`friction.contact_friction`), Johnson-Thornton restitution; every cut checked each tick, a ductile one at its cells' yield at their temperature (carbon steel: Eurocode 3's measured fire table) becomes a plastic hinge that sags and tears, a brittle one breaks; contact and bending energy become heat in their cells; the `strength` theater, request 15, 1 cm blocks) | body-body contact with friction, toppling, fracture past strength at the cell's temperature | a stack topples; a heated beam sags and breaks where hot |
| P4 Impacts in the world (DONE 2026-10-31: `PlaygroundWorld.impact`: the path traced through the cells and cut into layers, the game's per-layer solve (`shooting_range.layer_solver`) gives each layer's lost momentum and energy; the bore knocked out (ejecta), the momentum an impulse on the piece hit, the rest heat in the game's own bore volume; bodies can leave the world over its edge, booked out; the `impact` theater, request 16, Weapon button 17) | the range's solve as an event in the playground: momentum and heat into the cells hit | a round into the block wall heats, glows and pushes it as the solve says |
| P5 Looks complete (DONE 2026-10-31: `field/interface/reflectance_spectra.py`: ECOSTRESS spectra for granite, concrete, rubber (files in `fixtures/ecostress/`), oak from a measured L*a*b*, alumina a measured bound (flagged); Kevlar and ABS stay flagged grey; the range's hand-picked colours removed; textures now sRGB-encoded (they were written linear: every non-metal far too dark); glow brightness was already LabPBR emission, now tested) | measured reflectance for rocks, wood, concrete, rubber, plastics (cited spectra); glow into LabPBR emission | no material in the playground draws without a sourced colour, or it is flagged |
| P6 From inside | place any material, heat, drop, shoot, push; pause, step, slow motion (section 8) | the Captain does all of it in game without the console |

**Known gaps going in:** no gas phase (boiled matter vents from the world); no chemistry (wood chars, nothing burns: reactions need
product rules); no electromagnetism in the world (no magnets or currents) yet; FLIP is costly, so the playground is ~64 x 64 x 48
cells at first; air drag is per body, not a simulated air flow.

## 11. Validation (Platinum: test against reality, dry-run first)

- The service's tests run without Minecraft (a fake plugin records diffs). Physics is already tested in sigma-ground; SkippyCraft
  tests that the blocks the player sees equal the world's state after every diff (a full-region checksum each N ticks).
- Each phase ends in the real game with a real client, as GMod's phases did; a screenshot per phase in the session log.
- The readout's numbers are compared with the service's ledger (mass, energy, radiated, vented), not eyeballed.

## 12. Risks and open questions

- **Bedrock players (iPad, phone):** Geyser can bridge Bedrock clients to a Java server, but Bedrock has no display entities, so
  Skippy's floating body (and sub-block detail) would not show there. To verify before promising mobile play.
- **Away from the tailnet:** his devices are on it already; nothing to open to the internet.
- **Diff volume:** a big melt can change thousands of blocks a tick. Batch per chunk, cap the rate, and measure in Phase 1.
- **Where it runs:** skippy-core, tailnet only (one player, local models). If skippy-core's memory is tight while Qwen is resident,
  skippy-pc is the fallback (measure in Phase 0: Paper needs ~2-4 GB).
- **Licences:** Paper is free; the Captain needs Minecraft Java Edition. The EULA must be accepted by him, not by an agent.
- **Spoken voice in game** needs more than a stock client (section 7).
- **Plugin language:** Java (Paper's native) vs Kotlin. Recommendation: Java, small and boring.
- **Inherited physics flags** still open in sigma-ground: air cooling is an upper bound (heat demos honest only in vacuum theaters);
  no thermal data for liquid mercury; reactions need product rules. SkippyCraft shows these as gaps; it does not hide them.

## 13. First command of Phase 0

```bash
# a local Paper server to develop against (skippy-core later)
mkdir -p /d/Aaron/development/sigma-ground/stage/skippycraft/dev-server && cd "$_"
curl -sL -o paper.jar "<the current Paper 1.21 build from https://papermc.io/downloads>"
java -Xmx4G -jar paper.jar --nogui   # first run writes eula.txt; the Captain accepts the EULA by setting eula=true
```
