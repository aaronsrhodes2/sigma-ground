# Mentat stage for Garry's Mod

Replays frozen sigma-ground experiments (Radiance trajectory bundles) inside Garry's Mod so
you can watch them and operate them. **Status: Phase 0, verified in the real game.** Plan:
[`misc/GMOD_STAGE_PLAN.md`](../../misc/GMOD_STAGE_PLAN.md). Which Mentat tools each future
scenario covers: [`misc/GMOD_COVERAGE_PLAN.md`](../../misc/GMOD_COVERAGE_PLAN.md).

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
python -m sigma_ground.radiance.export_gmod fall_skillet                 # writes addon/lua/mentat/scenarios/fall_skillet.lua
python -m sigma_ground.radiance.export_gmod fall_skillet --report-only   # validate only, write nothing
python -m sigma_ground.radiance.export_gmod phase_e_ring_nbody --decimate 2   # thin frames to fit the 64 KB client limit
```

44 of the 86 trajectory bundles export today. The rest are refused with the reason (a body
made only of Voxel, Cone, Ellipsoid, Torus, Clipped, Rotated or Gear leaves), or by a guard:

- **Client file limit.** GMod will not send a Lua file over ~64 KB compressed to clients. The
  exporter refuses a scenario over 60 KB compressed (the four `phase_e_*` bundles) and tells you
  to `--decimate K`, or `--allow-large` for a server-only replay.
- **Map size.** Nine exportable bundles are kilometre-scale falls and cannot fit Source's
  ~600 m map at 1:1; the exporter warns and `mentat_load` refuses them (Phase 4 scaling).
- **Mass range.** GMod clamps a body to 0.1 to 50,000 kg; the exporter notes any body outside it.

Nothing is silently approximated: dropped CSG pieces, hull-versus-bundle inertia mismatches and
mass clamps are all listed in the scenario's `notes` and printed on load.

## Run it

Start GMod with `stage/gmod/gmod_ctl.sh start` (or from Steam), load `gm_construct`, open the console:

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

Grabbing a replaying body with the physgun releases it first, for admins only (untested by hand).
A released body falls in **vacuum**: Source's air drag and linear damping are switched off on it
(see the engine facts below). If the replay was shown in slow motion, release says so.

## Closing the game cleanly

GMod allows **one** instance, and it cannot be told to quit the obvious ways:

| Route | Result (measured 2026-10-10) |
|---|---|
| Lua `RunConsoleCommand("quit")` | blocked: "Command is blocked!" |
| Windows `WM_CLOSE` (`CloseMainWindow`) | ignored for 30 s and more |
| Windows `WM_SYSCOMMAND` / `SC_CLOSE` (what the title-bar X sends) | **works**: the whole process tree exits in 1 to 8 s, even while the game is still loading |

```bash
stage/gmod/gmod_ctl.sh status
stage/gmod/gmod_ctl.sh stop        # SC_CLOSE, waits for every gmod process, force-kills only if ignored
stage/gmod/gmod_ctl.sh restart
```

The script covers both `gmod.exe` and `gmod_win64.exe`, refuses to `start` while an instance
exists, and frees `console.log`. **Only one actor should drive the game at a time.** Two
processes launching it concurrently will kill each other's runs; run the autotests one at a time.

## Headless verification run

`lua/mentat/autotest.lua` records what the engine actually does and `verify_autotest.py` grades
it with its own independent math. The game is not closed by the test; the launcher does that.

```bash
GMD=/d/Steam/steamapps/common/GarrysMod/garrysmod/data/mentat
printf '%s' '{"scenario":"fall_skillet","release_at":0.31,"rate":1.0,"shots":true,"run_id":"r1"}' > "$GMD/autotest.txt"
stage/gmod/gmod_ctl.sh start
# wait for $GMD/autotest_fall_skillet.json (about 20 s after the map loads), then:
stage/gmod/gmod_ctl.sh stop
python stage/gmod/verify_autotest.py fall_skillet --run-id r1
```

Options in the config: `anchor_lift_hu` raises the stage so a low scenario has room to fall,
`shots` saves `data/mentat/shot_loaded.jpg`, `shot_mid.jpg`, `shot_landed.jpg` (aimed at the body).
For a rotating body add `--landing none`: `verify_autotest.py motor_spin --landing none`.

`verify_probe.py` analyses `probe.lua`, which measures how the engine's angular-velocity setters behave.

## Offline tests (no game needed)

```bash
python -m pytest sigma_ground/radiance/tests/test_gmod_stage.py sigma_ground/radiance/tests/test_gmod_verify.py sigma_ground/radiance/tests/test_gmod_coverage.py -q
```

`test_gmod_stage.py` runs the pure-Lua modules (`units`, `interp`, `shapes_core`) under LuaJIT
via `lupa` (optional, test-only: `pip install lupa`; the file skips cleanly without it) and
compares them to independent Python oracles. `test_gmod_verify.py` checks that the grader passes
a faithful synthetic run and fails the broken ones, including a wrong spin rate, axis and sense.
`test_gmod_coverage.py` checks every Mentat tool and simulation verb is accounted for.

## Engine facts measured in the real game (the wiki does not state these)

- **Angular velocity setters** (`SetAngleVelocity`, `SetAngleVelocityInstantaneous`, `AddAngleVelocity`)
  take **degrees per second about the body's LOCAL forward, left and up axes**, right-hand sign.
  Not a world-frame vector. `units.local_angvel_deg` does the projection. Verified end to end: a disc
  released at 4.006 rad/s kept spinning at 3.982 rad/s about the same axis.
- **Source slows every physics object twice**: air drag (about 9 % over a 3 m fall) and a **linear
  damping of exactly 0.1 per second**. Released bodies get both removed. Angular damping is kept: with
  it zeroed a landed cylinder tipped and rolled forever.
- **With both off, the engine integrates a fall exactly like semi-implicit Euler at its 15 ms tick**
  (agreement 5e-5 HU over a 125 HU drop). That differs from the textbook formula by g·t·τ/2, about
  1.4 % over a 0.75 s fall at 66.7 Hz. Any ENGINE test must account for that term.
- **A frozen physics object follows `SetPos`/`SetAngles` exactly** (1e-5 degrees, rotating body).
- **Collision skin** is about 0.41 HU: bodies rest that far above a surface.
- **A flat drop at ~7 m/s may topple** a 15 cm x 12 cm cylinder or may not. The landing pose is chaotic,
  so the grader accepts any stable pose of the shape at that pose's height.
- **`ceramic` is not a surface material** (index -1); `pottery` and `porcelain` are.

## Conventions worth knowing

- **Up axis is per bundle** (53 viewer bundles are Y-up, 32 theater bundles are Z-up). The
  exporter reads it and cross-checks it against the motion; two bundles
  (`falling_aluminum_ball`, `falling_lead_ball`) declare Z-up but fall along Y and are flagged.
- **Entity origin = the body's pivot (centre of mass).** The viewer places a leaf at
  `pos + R·(local − pivot)`; shape centres are exported relative to the pivot.
- **Units:** 1 HU = 0.01905 m (Mentat `convert_units(0.75 inch)` agrees). Gravity is set to the
  scenario's value while one is loaded and restored on `mentat_clear` or a map cleanup.
- **Velocity at release** is the slope of the bracketing recorded segment (the exact derivative
  of the linear interpolation the replay uses); past the end of the recording it continues the
  last segment, so a body that was cut off mid-fall keeps falling.
- `fall_skillet` stops 0.196 m above the floor, still moving at about 7.4 m/s. Replay ends
  with it hovering; `mentat_release` lets it land.
- A body that cannot be built (no geometry, invalid hull) is detected at load and refused. An
  entity that removes itself inside `Initialize` still passes `IsValid` until the next tick, so
  success is read from the `MentatReady` flag.

## Known gaps

- Shapes: Sphere, Box, Cylinder only. **CSG cavities are filled in.** The `fall_skillet` bundle is a
  hollow open vessel (4 mm wall) that GMod shows and simulates as a **solid** puck of the same mass; the
  exporter flags the 41 % inertia mismatch. The 24-gon cylinder is 1.1 % under a true circle and the
  sphere polytope 3.6 % under a true sphere.
- The physics object exists on the server only. Walking into a body without rubber-banding is untested.
- A replay body is moved by teleport, so it does not push props or the player.
- Regenerated scenario files reach clients only at map load; `mentat_reload` clears caches but a
  changed file needs a map restart.
- No thermal colouring (Phase 3), no scaled stages (Phase 4), no live bridge (Phase 5).
- Deformation does not exist in sigma-ground's trajectory contract (poses and temperature
  only), so no scenario can show a dent or a break.
