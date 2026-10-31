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
