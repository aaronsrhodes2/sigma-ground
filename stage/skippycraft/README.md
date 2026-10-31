# SkippyCraft

sigma-ground voxel theaters you can fly into, in Minecraft Java, with a block-built Skippy. Plan: [misc/SKIPPYCRAFT_PLAN.md](../../misc/SKIPPYCRAFT_PLAN.md).
One player (the Captain) and Skippy's virtual avatar; not multiplayer.

## Phase 0 (spike): a theater as a data pack

```bash
# the server (Docker, Java 21 and Paper inside), see dev-server/docker-compose.yml for the one-time EULA and player name
cd stage/skippycraft/dev-server && SKIPPYCRAFT_EULA=TRUE MC_PLAYER=<name> docker compose up -d

# a theater, written as MineBench code with materials, into the world's data pack
python stage/skippycraft/export_mc.py --name tank --out stage/skippycraft/dev-server/data/world/datapacks \
  --block-m 0.05 --code 'box(0,0,0,7,0,7,"granite"); box(1,1,1,6,5,6,"glass"); box(2,1,2,5,5,5,"air"); box(2,1,2,5,2,5,"mercury"); block(3,3,3,"iron")'
docker exec skippycraft rcon-cli reload
# without a player online, place it from the console: load the chunks first
docker exec skippycraft rcon-cli "forceload add 0 0 31 31"
docker exec skippycraft rcon-cli "execute positioned 4 -60 4 run function skippycraft:tank"
```

Checked 2026-10-30 on Paper 26.3-231 (Minecraft 26.3 needs Java 25: the image tag is `java25`): the pack loads, the theater
rules run (26.x game-rule names), every block lands where it was written (`execute if block ...`), and `clear_tank` removes it.
The flat world's surface is y = -60. The server uses its 3 GB. A player name joins the whitelist with
`docker exec skippycraft rcon-cli "whitelist add <name>"`.

Looks are derived by sigma-ground (`field.interface.block.appearance`): colour, opacity through one block of the theater's
size, reflectance and smoothness. The exporter writes a resource pack into this PC's `.minecraft/resourcepacks/skippycraft`
that repaints one carrier block per material (concrete if opaque, stained glass if see-through) and adds LabPBR specular maps
(reflections need a shader pack; vanilla draws none). Enable it once in Options > Resource Packs; after a re-export press F3+T.
`--looks vanilla` uses hand-picked stock blocks and no pack. `skippycraft_looks.json` in the pack lists each material's look,
its sources and its gaps.

In the game: stand where the theater's low corner should be, `/function skippycraft:tank`; `/function skippycraft:clear_tank`
removes it. Minecraft only shows it: each material has a look (`export_mc.LOOKS`), chosen from blocks Minecraft leaves alone,
and the data pack switches off random ticks, fire, weather and mobs.
Tests: `python -m pytest tests/test_skippycraft_export.py`.

## Phase 1: a live theater

```bash
python stage/skippycraft/theater.py looks                          # one resource pack for every scene; F3+T in game
python stage/skippycraft/theater.py sink --at 4 -60 20 --verify    # gold falls into the mercury and sinks; iron floats
python stage/skippycraft/theater.py melt --at 24 -60 4 --verify    # lead melts on a 1500 K tungsten plate (vacuum) and spreads
```

`theater.py` ticks sigma-ground's `VoxelWorld` and sends only the changed blocks over RCON (127.0.0.1:42575, password from
`dev-server/data/server.properties`); a boss bar shows the world's own ledger (time, mass, energy, radiated, vented, changed
blocks). The first draw sets every cell from the theater's ground up, so a re-run clears what the last one left; Minecraft's own
ground below is never touched. `--verify` asks Minecraft about every cell afterwards. Tests: `tests/test_skippycraft_theater.py`
(against a fake server).

## Using it: buttons, no keyboard combos

```bash
python stage/skippycraft/theater.py serve
```

The service builds a control desk at each theater's viewing spot: a glowing platform to stand on and, in front of it at eye
level, three buttons with big labels: **Run**, **Stop**, and **Go to** the other theater's desk (a teleport). He spawns on the
sink desk. Desks sit outside every theater's grid (a run's first draw clears its whole grid). Theaters are built at 4x
resolution (`theater.RESOLUTION`; every written block is 4x4x4 blocks), fixed for each run: the tank is 32 blocks wide, the gold a 4x4x4 cube. Right-click a button: it sits on a command block that records the request in the
scoreboard, and the service (polling over RCON) runs that theater in front of you, with the boss bar as the readout. The world
spawn is beside the sink panel. The service also serves the resource pack (http://100.122.71.14:42566/skippycraft.zip, the
server's `RESOURCE_PACK`), so it arrives when he joins: accept it once, no F3+T. The service runs on skippy-pc for now (it must be
running for the buttons and the pack); Phase 5 makes it a kept-up service.

## Water with momentum: the wave machine (`slosh`)

The third desk (z = 88, "Go to slosh" from the melt desk) runs a glass tank 0.8 m long with an aluminium paddle at one end and
water 15 cm deep, at 4x (1.25 cm blocks, 129,024 water particles). The water is `dynamics.fluid.flip.Flip`: particles that carry
momentum through a staggered grid, made incompressible by a pressure solve each step, with air as empty cells (a real free
surface: waves, splashes). Density and viscosity from `block()`, gravity standard. The paddle is a moving wall
(`materia.voxel_fluid.FluidVoxelWorld`) driven at the tank's own first sloshing period (`fluid_surface.sloshing_angular_frequency`,
1.24 s), so each push adds to the last. **Slower** / **Faster** change the period by 10% (the chat says how far from resonance).
The boss bar shows the water's momentum, kinetic energy and the work the paddle has done.

Blocks go through the SkippyCraft plugin (`plugin/`, built in Docker by `bash stage/skippycraft/plugin/build.sh`, no JDK on the
PC): the service sends each tick's changed blocks as one binary frame to 127.0.0.1:42576 and the plugin sets them on the server
thread without block physics. RCON stays for commands (and is the fallback when the plugin is absent).

Speed (skippy-pc, numba installed): a 0.04 s frame takes ~0.3 s to compute, so it plays at about 1/8 of real speed.
The water is drawn as exactly its own volume of blocks, in the cells fullest on average over their neighbourhood; a
second, position-only solve keeps the particles evenly spread (FLIP bunches them, and a moving paddle shoves a layer at
a time), without touching any velocity.
Tests: `tests/test_flip.py` (still water stays still and the floor carries exactly its weight; a falling blob gains exactly
M·g·t; mass exact; the paddle does work).
