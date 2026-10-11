# Skippy as an in-world avatar

Approved 2026-10-11. A player calls Skippy and he appears next to them, hovers, follows, and his spoken answers come from where he
floats (3D, with proximity voice's range, wall and whisper rules). Built and tested offline; run live on 2026-10-11 (dedicated server + real client; see "Live pass" below).

## What it does

- **Call:** "Skippy, come here" / "come over" / "come closer" / "follow me" / "show yourself", typed in chat, spoken to the voice
  bridge, `holodeck come here` in the console, or `skippy_come`. Typos are tolerated ("cmoe here", "come hear"). A sentence that
  asks for anything else is never a call: "add a couch", "come up with a name", "move the couch here", "get me a couch".
- **Send away:** "Skippy, go away" / "dismiss" / "go home" / "leave me alone" / "vanish" / "get lost", or `skippy_go`.
- **Arrival:** a server-side bot player named `Skippy` is created once (`player.CreateNextBot`) and placed with `SetPos`
  instantly about 1.5 m to the caller's right (left if the right is a wall), slightly above eye height, turned to face them. A line
  from the caller's head to the spot must not hit the world, else the spot is pulled back 24 HU short of the wall.
- **Hover and follow** (every server tick): drifts (exponential, 4.5/s, at most 900 HU/s) to a point behind the owner's right
  shoulder, 16 HU above their eyes, with a 3 HU bob. Snaps (teleports) when the owner is more than 8 m (420 HU) away or moved
  300 HU in one tick. Looks where the owner looks; while he is speaking, looks at the owner. He waits if the owner is dead and
  leaves if the owner disconnects.
- **One shared Skippy:** he goes to whoever called last. Multiplayer splitting is out of scope.
- **His voice:** while he is out, `Holodeck.Say/Speak` use his position instead of the player's or the build site: the audio
  message carries his position, chat goes to people near him. On each client `avatar_client.lua` takes the playing clip
  (`SkippySpoke` hook), moves it to his position every frame and sets its volume to
  `skippy_volume * Core.gain(distance, range, edge)` where distance is eye-to-avatar, range is 1100 HU (`mentat_voice_range`), 35%
  of that when the person he serves is crouching (a whisper), and 55% of it through a wall (`mentat_voice_wall`). In `global` or
  `off` voice mode distance does not matter. With no avatar nothing touches a clip: Skippy sounds exactly as before.
- **Look:** the bot's player model is never drawn. Procedural geometry (`avatar/geometry.lua`, about 1100 triangles, no model or
  texture file): a beer can 0.5 m tall (silver, `#00FF88` label band, pull tab, two eyes and a mouth) wearing a giant dark bicorn
  (47 HU = 0.9 m point to point, amber `#FFAA00` braid, blue-white-red cockade). The label brightens, the mouth opens, a green
  dynamic light pulses and the hat leans toward whoever is talking (the nearest speaking player, else the owner) with the voice
  level (`chan:GetLevel()` when available, else a timed envelope).

## The bot

Holds a player slot. Weapons stripped, god mode, `MOVETYPE_NOCLIP`, not solid (nobody's aim or movement is blocked), no collision,
`SetNoTarget`. Its own bot AI is switched off: the `StartCommand` hook clears buttons and movement and pins the view angle.
`PlayerLoadout` and `PlayerShouldTakeDamage` are refused for it. Dismiss **hides** him by default (`SetNoDraw`, body kept: no
join/leave messages, no slot churn); `mentat_avatar_kick 1` kicks the bot instead, freeing the slot (he rejoins when called).

## Convars and commands

| name | default | meaning |
|---|---|---|
| `mentat_avatar_enabled` | 1 | 0 = the words go to the holodeck as before; nobody can call him |
| `mentat_avatar_kick` | 0 | 1 = going away kicks the bot |
| `mentat_avatar_debug` | 0 | 1 = per spoken line, per listener: distance, range, blocked, gain (server console) |
| `mentat_avatar_ack_voice` | 1 | 1 = his "Here I am." is spoken, 0 = text only |
| `mentat_avatar_draw`, `_light`, `_scale` (client) | 1, 1, 1 | draw him / his light / his size |
| `skippy_come`, `skippy_go`, `mentat_avatar_status` | | call, send away, report where he is and who hears him |

Lua: `Mentat.Avatar.Summon(ply)`, `.Dismiss()`, `.Get()`, `.IsActive()`, `.WhoHears([ply])` (array of
`{ply, nick, dist, range, blocked, gain, heard}`), `.Hearing(listener)`. Hooks (for the transporter effect):
`SkippyAvatarArrive(bot, from, to)` (`from` is nil on the first appearance, else where he was) and `SkippyAvatarLeave(bot, from)`.
A jump (called elsewhere, or a snap) fires Leave then Arrive.

## How it hooks in (nothing in holodeck.lua changed)

`avatar.lua` wraps `Holodeck.Handle` (puts the two commands in front of every path words take), `Holodeck.Say` and
`Holodeck.Speak` (position) after `holodeck.lua` has loaded; a 2-second timer re-wraps if a Lua refresh brings the plain functions
back. Files: `mentat/avatar.lua` (server), `mentat/avatar_client.lua` (client), `mentat/avatar/logic.lua` and
`mentat/avatar/geometry.lua` (shared, pure, unit-tested); loaded from `autorun/mentat_stage.lua`, which also lists them for
`AddCSLuaFile`. `package_addon.sh` copies the whole `lua/` tree, so it needs no change. Tests:
`sigma_ground/radiance/tests/test_gmod_avatar.py`.

## Live pass (2026-10-11): what the real game showed

How it was run: `GMOD_DS_MAXPLAYERS=4 python stage/gmod/holodeck/run_tool.py avatar_live --timeout 300` (dedicated server, no human: one bot plays
the caller through the same `Holodeck.Handle` entry the voice path uses, another is a second listener; tool `tools/avatar_live.lua`) and
`python stage/gmod/holodeck/avatar_shots.py` (the visible client for about 90 s, tool `tools/avatar_shots.lua`; pictures in
`holodeck/data/shots/avatar_*.jpg`). Console-only checks: `mentat_avatar_tone` plays a pulsing tone from where he hovers through the same
hook his voice takes and prints what the engine did with it (**use it to hear the pan**); `mentat_avatar_hide hat,label` (client) leaves
parts out of the drawing, for debugging.

| # | item | result |
|---|---|---|
| 1 | server starts, status "away; bot not created", no errors | VERIFIED (no Lua error in either console log over all runs) |
| 2 | "Skippy, come here" -> bot, ~1.5 m to the side, slightly above, arrival hook | VERIFIED. `player.CreateNextBot` works on the dedicated server and on a listen server (4 ms); named Skippy; 79 HU (1.5 m) to the right, 12 HU above the eyes, facing the caller; instant; Arrive fires once with `from = false`. His `PlayerSpawn` did not re-run `configure` on first creation (Summon configures him) |
| 3 | follows: walk/run/rise/teleport, wall on the right, back to a wall | VERIFIED. Walking (200 HU/s): steady 87 HU behind; at 800 HU/s (far beyond a real sprint) the lag peaks at 250 HU then settles near 178 (= v / 4.5, as designed); teleport: Leave+Arrive snap, 67 HU away on the next tick (also on the real client: 97 HU after 0.3 s); rise: steady 43 HU above the feet; wall on the right: he moves to the left side, never inside; back to a wall: he is 5 HU in front of it (not inside, but his 23 HU wide hat would clip it: cosmetic, not changed) |
| 4 | idle 60 s; kill; damage; void | VERIFIED after a FIX. Idle: no wandering (inside the 3 HU bob), no weapons, health 100, no respawns, stays MOVETYPE_NOCLIP. `TakeDamage(1e5)`: ignored (god). **FAILED, fixed:** `Player:Kill()` (also the console `kill`, a kill trigger) ignores god mode and the bot then stayed dead for ever (bots never respawn by themselves), pinned to his owner as a corpse; and after a respawn the engine had put him back to MOVETYPE_WALK. Now: `DoPlayerDeath` makes no ragdoll, `PlayerDeath` respawns him 0.1 s later, and every tick restores NOCLIP if the engine changed it (re-run: alive, noclip, beside the owner 0.5 s after `Kill()`). Void (`SetPos` 100000 HU down): back beside the owner within one tick. Owner dead: he waits where he is |
| 5 | client: model gone, can + hat, upright, size, front/side/5 m, inside out | VERIFIED after FIXES (see "The look") |
| 6 | voice from his position, debug gain lines | PARTLY. Verified: the per-listener lines match `Core.gain` (listener 100 HU: 1.0; 800: 0.77; 1000: 0.12; 1050: 0.03; 1100+: 0; crouched owner: range 385, 354 HU away -> 0.095; wall -> blocked, shorter range). On the real client the engine accepts everything the code does: `chan:SetPos` takes his position (read back), `chan:SetVolume` on a `"3d mono"` stream works (read back 0.77 at 800 HU, 0.12 at 1000 HU = `Core.gain` exactly), `Set3DFadeDistance` is accepted. **Only an ear can verify** that the sound pans and changes as you walk round him and that the engine's own distance fade is really flat: run `mentat_avatar_tone` while walking around him. A second human on another client was not possible |
| 7 | label brightens, mouth moves, hat leans | PARTLY. `chan:GetLevel()` works on a 3D stream (0.22 to 0.30 for the tone), so the real level drives the glow. Not photographed while speaking; "hat leans toward the speaker" needs a second talker: UNVERIFIED |
| 8 | `mentat_avatar_kick 1`; full server | VERIFIED. Kick frees the slot (3 -> 2 players), calling again re-creates him (`Dropped Skippy` in the log); on a full server (4/4) he says "the server has no free player slot for me" and nothing appears. "Player Skippy has joined the game" shows once, on first creation. Scoreboard row not looked at (UNVERIFIED) |
| 9 | `net.Send` to the bot | VERIFIED HARMLESS. `Holodeck.Say/Tell` audiences include the bot (`IsBot()` true); a direct `net.Send(bot)` returns without error and the console log has no line about it over many spoken lines. `holodeck.lua` was NOT changed |
| 10 | other words still work | VERIFIED. "add a couch" builds a couch while he is out, "add a lamp" while he is in; "go away" hides him (`nodraw`, bot kept, `IsActive()` false); "come with me" / "come here" bring him back |

### The look (item 5), honestly
Live, the bot's player model is never drawn and the mesh draws with no error (`draw_error` nil, ~1320 triangles). Three things were wrong and
are fixed (each has an offline test):
1. **The eyes were invisible.** The face sat at 0.97 R, inside the label band (1.012 R), so the label hid it. Moved in front of the label.
2. **The colour material does not write depth**, so the label (drawn after the face) painted over the eyes even once they stood proud, and
   every part's draw order mattered. The draw now forces depth writing on (`render.OverrideDepthEnable(true, true)`) and off again after.
3. **The hat** was a thin gold "banana" (amber along the whole front) with a cockade sunk into the felt. Now a chunkier crescent
   (half-depth 11, half-height 8 HU, centre 12 HU so the crown comes down over the can's top), dark blue felt `#222639` with a narrow amber
   lace along the top crest and lower edge and amber points, and a blue/white/red cockade standing clear of the felt. The label band is
   also a little shorter (more silver shows).

My read: a green-label silver can with a cheeky face wearing an enormous dark crescent hat with a gold-edged crest and a roundel. It reads
as "a can in a Napoleon-style bicorn" from the front and three-quarter views, and is recognisable as a can in a hat at 5 m. Side-on, the hat
is a pointed dome with a gold crest (a bishop's mitre more than a bicorn; that is what a bicorn looks like end-on). Weaknesses: the felt is
smooth and dark with no texture, and the can is more green than silver. Nothing is inside out. Pictures:
`stage/gmod/holodeck/data/shots/avatar_{front,close,side,threequarter,far5m,firstperson}.jpg` (the first-person one looks away from him:
he floats behind the right shoulder, outside the 90 degree view).

### Other things found
- `tool.lua` (the headless job runner) ran its job a second time, with the bot as "the player", when the avatar's bot joined a listen server
  (`PlayerInitialSpawn` fires for bots). It now runs once, for the first human only.
- `gmod_ds_ctl.sh` takes `GMOD_DS_MAXPLAYERS` (default 2); the live pass needs 4 (caller, second listener, Skippy, one filler).

### Still unverified (needs a human, ears or a second client)
- 3D panning and loudness by ear; that the engine's distance fade really is flat (the code sets 100000/200000); walls only lower the volume (no muffling).
- A second human at various distances (only the per-listener gain numbers were checked, with a bot as the second listener).
- The label, mouth and hat lean while he speaks and while someone talks into voice chat next to him; the dynamic light in the dark.
- The scoreboard row for Skippy; `PostDrawTranslucentRenderables` order against glass; the hidden body's shadow and target-ID name tag.
- Speech end to end through `holodeck_bridge` with the avatar out (no bridge was running): the position and gain plumbing was exercised, the real mp3 path was not.
