# Mentat stage for Garry's Mod

Replays frozen sigma-ground experiments (Radiance trajectory bundles) inside Garry's Mod so
you can watch them and operate them. **Status: Phase 0 spike.** Plan: [`misc/GMOD_STAGE_PLAN.md`](../../misc/GMOD_STAGE_PLAN.md).

Nothing is simulated by the addon. A scenario is recorded sigma-ground output; REPLAY shows
it, RELEASE hands a body to Havok at its current pose and velocity, after which Havok (not
sigma-ground) moves it. The HUD says so.

## Install (once)

The addon lives in this repo and GMod sees it through a directory junction:

```bash
cmd //c mklink //J "D:\\Steam\\steamapps\\common\\GarrysMod\\garrysmod\\addons\\mentat-stage" "D:\\Aaron\\development\\sigma-ground\\stage\\gmod\\addon"
```

Remove it with `cmd //c rmdir "…\\addons\\mentat-stage"` (a junction rmdir never touches the target).

## Make a scenario

```bash
python -m sigma_ground.radiance.export_gmod fall_skillet          # writes addon/lua/mentat/scenarios/fall_skillet.lua
python -m sigma_ground.radiance.export_gmod fall_skillet --report-only   # validate only, write nothing
```

48 of the 87 trajectory bundles export today. The rest are refused with the reason (a body
made only of Voxel, Cone, Ellipsoid, Torus, Clipped, Rotated or Gear leaves). Nothing is
silently approximated: dropped CSG pieces are listed in the scenario's `notes` and printed on load.

## Run it

Start GMod (the 64-bit `gmod_win64.exe` or `gmod.exe`), load `gm_construct`, open the console:

| Command | Effect |
|---|---|
| `mentat_list` | list scenarios |
| `mentat_load <name> [x y z]` | spawn the bodies (default: 4.5 m ahead of you, floor under your feet) |
| `mentat_play` / `mentat_pause` | run / hold the replay clock |
| `mentat_rate <x>` | playback rate (default is the bundle's `suggested_rate`; `1` is real time) |
| `mentat_scrub <t>` | jump to a sim time |
| `mentat_release [label\|index\|all]` | hand bodies to Havok now |
| `mentat_reset` | back to frame 0, kinematic again |
| `mentat_clear` / `mentat_reload` | remove bodies / re-read regenerated scenario files |
| `mentat_status` | one line of state |

Grabbing a replaying body with the physgun releases it first (untested until used by hand).

## Headless verification run

`lua/mentat/autotest.lua` records what the engine actually does and `verify_autotest.py`
grades it with its own independent math:

```bash
echo '{"scenario":"fall_skillet","release_at":0.31,"rate":1.0}' > "D:/Steam/steamapps/common/GarrysMod/garrysmod/data/mentat/autotest.txt"
"D:/Steam/steamapps/common/GarrysMod/gmod_win64.exe" -console -condebug -windowed -w 1280 -h 720 +map gm_construct
python stage/gmod/verify_autotest.py fall_skillet
```

The game loads the map, loads the scenario on first spawn, replays, releases at 0.31 s, waits
for the body to rest, writes `data/mentat/autotest_fall_skillet.json` and quits.

## Offline tests (no game needed)

```bash
python -m pytest sigma_ground/radiance/tests/test_gmod_stage.py sigma_ground/radiance/tests/test_gmod_verify.py -q
```

`test_gmod_stage.py` runs the pure-Lua modules (`units`, `interp`, `shapes_core`) under LuaJIT
via `lupa` (optional, test-only: `pip install lupa`; the file skips cleanly without it) and
compares them to independent Python oracles. `test_gmod_verify.py` checks that the grader
passes a faithful synthetic run and fails seven broken ones.

## Conventions worth knowing

- **Up axis is per bundle** (53 viewer bundles are Y-up, 32 theater bundles are Z-up). The
  exporter reads it and cross-checks it against the motion; two bundles
  (`falling_aluminum_ball`, `falling_lead_ball`) declare Z-up but fall along Y and are flagged.
- **Entity origin = the body's pivot (centre of mass).** The viewer places a leaf at
  `pos + R·(local − pivot)`; shape centres are exported relative to the pivot.
- **Units:** 1 HU = 0.01905 m. Gravity is set to the scenario's value while one is loaded and
  restored on `mentat_clear`.
- **Velocity at release** is the slope of the bracketing recorded segment (the exact derivative
  of the linear interpolation the replay uses); past the end of the recording it continues the
  last segment, so a body that was cut off mid-fall keeps falling.
- `fall_skillet` stops 0.196 m above the floor, still moving at about 7.4 m/s. Replay ends
  with it hovering; `mentat_release` lets it land.

## Known gaps (Phase 0)

- Shapes: Sphere, Box, Cylinder only. CSG cavities are filled in (the skillet bundle is a
  solid 15 cm x 12 cm cylinder here), the 24-gon cylinder is 1.1 % under a true circle and the
  sphere polytope 3.6 % under a true sphere.
- No thermal colouring (Phase 3), no scaled stages (Phase 4), no live bridge (Phase 5).
- Angular velocity at release uses `PhysObj:SetAngleVelocityInstantaneous` with world-frame
  degrees per second. Whether that is the right frame and unit is unverified; it only matters
  for rotating bodies (the skillet does not rotate).
- Deformation does not exist in sigma-ground's trajectory contract (poses and temperature
  only), so no scenario can show a dent or a break.
