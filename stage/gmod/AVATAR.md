# Skippy as an in-world avatar

Approved 2026-10-11. A player calls Skippy and he appears next to them, hovers, follows, and his spoken answers come from where he
floats (3D, with proximity voice's range, wall and whisper rules). Built and tested offline; **never run in the real game yet**
(see "Not verified" and the live-pass task list below).

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

## Not verified (needs the real game)

1. **Bot creation.** `player.CreateNextBot("Skippy")` on a dedicated server, on a listen server, with `maxplayers` full; whether the
   bot's `PlayerSpawn` re-runs our `configure`; that `StartCommand` really silences a NextBot player (no wandering, jumping, respawn
   loops); that god mode and `SetNotSolid` hold after a death or a map trigger (`trigger_hurt`, void); that `SetPos` every tick on a
   noclip bot is smooth and not overridden by the engine's bot movement.
2. **net.Send to a bot.** `Holodeck.Say/Speak` choose their audience from `player.GetAll()`, which now includes the bot. Sending a
   net message to a bot is assumed to be ignored harmlessly; if it logs errors the audience builders in `holodeck.lua` need
   `and not p:IsBot()` (a follow-up edit, not made here).
3. **3D sound.** Whether `chan:SetPos` each frame moves the sound in 3D for a `"3d mono"` stream; whether
   `Set3DFadeDistance(100000, 200000)` really disables the engine's own distance fade (assumed: min = distance where fade starts);
   whether `chan:SetVolume` below 1 works on a 3D channel. **Walls:** GMod has no easy low-pass, so a wall only lowers volume (the
   55% range rule), it does not muffle.
4. **Level meter.** `IGModAudioChannel:GetLevel()` on a 3D mp3 stream; if it returns 0 the timed envelope is used (looks like talking,
   not synced to the syllables).
5. **Rendering.** `Mesh()`/`mesh.Begin` per part built once, `render.SetColorMaterial()` with vertex colours, `render.CullMode`
   none, `cam.PushModelMatrix(m, true)` nested (the multiply argument), `render.SetColorModulation` on the label, `DynamicLight`.
   Whether the mesh is lit as intended (shading is baked into vertex colours with a fixed light), whether the can/hat read at
   distance, whether the cockade z-fights the hat, whether `PostDrawTranslucentRenderables` order puts him behind or in front of
   glass. `PrePlayerDraw` returning true hides the bot's model (also: its shadow, and the target-ID name tag).
6. **Hat look.** Proportions are guessed (47 HU wide, points rising 10 HU, 9 HU deep): "a believable Napoleonic bicorn" is a human
   judgement. Tune `hat_station` in `geometry.lua`.
7. **The scoreboard** shows a player named Skippy while the bot exists, also when he is hidden.

## Live pass: task list

Run later by Aaron, on the dedicated server first (no window, no mouse grab), then once with a real client.

1. Start the server with the addon; `mentat_avatar_status` says `away; bot not created`. No errors in the console.
2. As a player: say "Skippy, come here". Expect a `Skippy` bot in `status`, no error from `CreateNextBot`, and the arrival hook
   in the log if you add a `hook.Add("SkippyAvatarArrive", ...)` print. Check he is about 1.5 m to the side and slightly above.
3. Walk, run, jump, noclip, climb a ladder: he follows smoothly. Teleport (`setpos`) far away: he is there next tick.
   Stand with your back against a wall and with the wall on your right: he must not end up inside it.
4. Watch the bot for 5 minutes idle: no wandering, no weapons, no damage, no respawn loop. Try `kill` on the bot
   (`kickid`/`bot_kill` are not needed), a `trigger_hurt`, and falling out of the map: he survives and stays with you.
5. Client: is the default model gone? Is the can + hat there, upright, the right size? Screenshot from the front, the side, 5 m away.
   Is anything inside out (flip `render.CullMode`)? Does the hat sit on the can?
6. Voice: with `holodeck_bridge` set, ask for something. The answer must come from him: turn your head, walk around him, hear it
   pan. Run `mentat_avatar_debug 1` and read the per-listener lines against what you hear: 2 m away (gain 1), at 800 HU (fading),
   behind a wall (shorter range), crouched (whisper: range 385). Second player on another client at various distances.
7. Does the label brighten and the mouth move while he talks, and the hat lean toward the speaker (talk into voice chat next to him)?
8. `mentat_avatar_kick 1`: send him away, call again; check the join/leave messages and that the slot is freed. Fill the server
   (`maxplayers`) and call: expect the polite refusal.
9. `net.Send` to the bot: look for console errors on every spoken line (see item 2 above).
10. "Skippy, add a couch" still builds a couch while he is out; "go away" hides him; his chat answers still reach the right people.
