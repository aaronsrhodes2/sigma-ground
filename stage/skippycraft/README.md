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
```

In the game: stand where the theater's low corner should be, `/function skippycraft:tank`; `/function skippycraft:clear_tank`
removes it. Minecraft only shows it: each material has a look (`export_mc.LOOKS`), chosen from blocks Minecraft leaves alone,
and the data pack switches off random ticks, fire, weather and mobs.
Tests: `python -m pytest tests/test_skippycraft_export.py`.
