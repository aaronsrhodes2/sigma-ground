# SkippyCraft Plan — sigma-ground voxel theaters you can fly into, in Minecraft

Named by the Captain, 2026-10-30. Builds on the voxel theater (2026-10-30, `misc/SESSION_LOG.md`): `deckard/voxel_exec.py`
(MineBench code → material blocks), `radiance/voxel_scene.py` (render), `materia/voxel_world.py` (the tick: heat, phase, settling),
`deckard/voxel_resample.py` (resolution, set at build time), `field/interface/block.py` (one block of matter, every property sourced).
Ports the Garry's Mod holodeck (`stage/gmod/holodeck/`, `stage/gmod/AVATAR.md`) and uses the SkippyVerse object cards
(`stage/skippyverse/`).

## 1. Goal and success criterion

A Minecraft server where each **theater** is a region of real space run by sigma-ground. You fly into it (creative or spectator
flight), watch the physics run, and change it from inside by talking to a floating, block-built Skippy.

**Success:** on the cloud server, the Captain flies into a theater, says *"Skippy, build a glass tank of mercury and drop a gold ring
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

## 3. Why Minecraft

- The theater is already blocks. MineBench showed the local models build well in block code; the voxel theater made those blocks
  matter. Minecraft shows blocks natively, at any count a theater needs, to many players at once.
- Flight and walking through walls are built in (creative / spectator), so "enter as a floating avatar" costs nothing.
- **Block display entities** (Java 1.19.4+) draw blocks off the grid: any position, rotation, scale (smaller than a block too),
  smoothly interpolated. That is how Skippy floats, and how a theater finer than 1 m can still look right.
- Paper (the server) is free and has a large, stable plugin API. Players need a Minecraft Java licence.

## 4. Architecture

```
 Minecraft Java client (player)                      skippy-core (home)
          │ game protocol                                ▲  Skippy Core chat (Funnel + SKIPPY_CORE_API_KEY,
          ▼                                              │  the portal's route) for "Skippy, ..." and block code
 ┌──────────────── games VM (GCP) ───────────────────┐   │  Mentat (Funnel :8443) for block_properties, physics answers
 │ Paper server + SkippyCraft plugin (Java)          │   │
 │   draws blocks, turns off vanilla physics in a    │   │
 │   theater, relays chat/commands, draws Skippy     │   │
 │        │ localhost HTTP + WebSocket (block diffs)  │   │
 │        ▼                                           │   │
 │ theater service (Python, sigma-ground)  ──────────┼───┘
 │   one VoxelWorld per theater, the tick loop,      │
 │   the holodeck planner, the Skippy bridge         │
 └───────────────────────────────────────────────────┘
```

- **Plugin (thin):** places/clears blocks from diffs; maps each material to a Minecraft block for LOOK only (lead → iron-grey
  block, molten → its glowing variant; the real material, temperature and phase stay in the service); turns off vanilla physics in
  theater regions; sends chat lines starting "Skippy" and `/theater` commands to the service; draws the avatar (display entities);
  shows the readout (a boss bar / sidebar: time, energy ledger, mass, what changed).
- **Theater service (all the physics):** owns each `VoxelWorld`; ticks it on a clock; sends only changed blocks; serves
  `block_properties` for "what is that?"; turns "Skippy, ..." into actions (the holodeck planner, the local model through Skippy
  Core). Runs on the VM's CPU (numpy); it never needs the home PC, so a theater keeps running when home is down.
- **Brains at home:** Skippy Core (for words → actions and for MineBench-style build code) and Mentat are reached over their
  existing Funnels with the existing keys. This is the bridge the GMod cloud server never got (`stage/gmod/CLOUD_SERVER.md`:
  "a bridge container on the VM is the missing piece"). If home is down, Skippy says so and the parser alone handles commands.

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
| `skippy_bridge.py` (agent turns a wish into a request, checked before the game sees it) | **port** as-is in spirit; now reachable from the cloud through Skippy Core's Funnel |
| Safety on a shared server (owners, caps, cooldowns, rationed bridge) | **port** the rules; Paper permissions replace CPPI |
| Avatar (`AVATAR.md`: call/dismiss words, hover behind the right shoulder, follow, snap past 8 m, voice from where he floats) | **port** the behaviour; the body becomes display blocks (section 7) |
| Replays of frozen trajectory bundles | **not needed**: theaters run live |

## 7. Skippy, made of blocks

- **Body:** the same look the GMod avatar settled on (a beer can, `#00FF88` label band, amber-braided dark bicorn), built in blocks
  by the voxel pipeline from `portal/static/skippy/skippy.glb` (Deckard voxelize, a few hundred cells), shown as block display
  entities, so he floats off the grid at any scale.
- **Behaviour:** `AVATAR.md` verbatim where it applies: "Skippy, come here" / "go away"; hovers behind the caller's right shoulder
  with a small bob; follows; snaps when far; faces whoever is talking. The room box's idle float and talking loop (SkippyTel
  `a8bcc4d`) set the motion.
- **Voice:** text in chat always. Spoken audio needs a client-side mod or a resource pack trick; a Phase 5 question, not a promise.

## 8. Commands (what "run and alter from inside" means)

Said to Skippy, or as `/theater …`:

- **Build:** "Skippy, build a copper plate with a lead block on it" → block code → a new theater (or added to this one).
- **Change state:** heat / cool a region to N kelvin, drop an object at a point, remove, undo, reset to the start.
- **Time:** pause, play, step, speed (simulated seconds per real second).
- **Ask:** "what is that?" (the block you look at → `block_properties`: material, phase, temperature, sources); "why did it melt?".
- **Theater:** new theater with a size, a resolution and a type (plain / room / void / deep space); list; join; leave.

## 9. Repo layout

- `sigma-ground/stage/skippycraft/` — the theater service (Python), its tests, and `plugin/` (Java, Gradle, Paper API).
- SkippyTel `infra/gcp-games/` — a `skippycraft` compose profile (Paper + the service), backup TARGETS for theater saves, the ops
  doors. SkippyTel `portal/` — a SkippyCraft card in Game Servers, like every other game.

## 10. Phases, done criteria, estimates

| Phase | Work | Done when | Est. |
|---|---|---|---|
| **0 Spike** | Paper server locally; a minimal plugin that reads a `VoxelField` and places its blocks once; material → block look table | the demo tank appears in a local world and a stock client flies through it | 1 day |
| **1 Live theater** | the service with one `VoxelWorld`, a tick loop, block diffs over WebSocket; vanilla physics off in the region; a readout | the gold ring sinks and the bolt floats in front of you; the lead melts on the hot plate; the readout's energy balance matches the service's | 2 to 3 days |
| **2 Commands** | `/theater` new/heat/drop/pause/speed/undo/reset; "what is that?" via `block_properties` | every command in section 8 works from inside, without words | 2 days |
| **3 Talking to Skippy** | port parse/plan/layout; the bridge to Skippy Core; block-code builds via MineBench-style prompts and `voxel_exec` | "Skippy, build ..." and "Skippy, heat the floor ..." work in plain words; failures say why | 3 to 4 days |
| **4 Skippy's body** | voxelize `skippy.glb`; display-entity avatar; AVATAR.md behaviour | he comes when called, hovers, follows, faces the speaker, leaves when sent away | 2 days |
| **5 Cloud** | `skippycraft` profile on the games VM, backups, ops doors, the Game Servers card; safety caps; voice question settled | the success criterion (section 1) passes on the cloud server with a second player present | 2 to 3 days |

Phases 0 and 1 answer the big question (does it look and feel right?) before any of the talking is built.

## 11. Validation (Platinum: test against reality, dry-run first)

- The service's tests run without Minecraft (a fake plugin records diffs). Physics is already tested in sigma-ground; SkippyCraft
  tests that the blocks the player sees equal the world's state after every diff (a full-region checksum each N ticks).
- Each phase ends in the real game with a real client, as GMod's phases did; a screenshot per phase in the session log.
- The readout's numbers are compared with the service's ledger (mass, energy, radiated, vented), not eyeballed.

## 12. Risks and open questions

- **Bedrock players (iPad, phone):** Geyser can bridge Bedrock clients to a Java server, but Bedrock has no display entities, so
  Skippy's floating body (and sub-block detail) would not show there. To verify before promising mobile play.
- **Diff volume:** a big melt can change thousands of blocks a tick. Batch per chunk, cap the rate, and measure in Phase 1.
- **Where the service runs:** recommended on the games VM (it keeps running when home is down; one place for ops and backups). The
  alternative, skippy-core at home, is closer to the models but needs a public game port through Tailscale; not recommended.
- **Licences:** Paper is free; every player needs Minecraft Java Edition. Mojang's EULA allows a free server; no paid perks.
- **Spoken voice in game** needs more than a stock client (section 7).
- **Plugin language:** Java (Paper's native) vs Kotlin. Recommendation: Java, small and boring.
- **Inherited physics flags** still open in sigma-ground: air cooling is an upper bound (heat demos honest only in vacuum theaters);
  no thermal data for liquid mercury; reactions need product rules. SkippyCraft shows these as gaps; it does not hide them.

## 13. First command of Phase 0

```bash
# a local Paper server to develop against (games VM later)
mkdir -p /d/Aaron/development/sigma-ground/stage/skippycraft/dev-server && cd "$_"
curl -sL -o paper.jar "<the current Paper 1.21 build from https://papermc.io/downloads>"
java -Xmx4G -jar paper.jar --nogui   # first run writes eula.txt; the Captain accepts the EULA by setting eula=true
```
