# The cloud Garry's Mod server as our live mod server

Status as reported by the portal chat on 2026-10-10, twice (I have not been able to check it myself; see "Content" and "Not verified" below), and how we
use it for the Mentat stage mod (holodeck, proximity voice, Skippy).

## The three places the mod runs

| Place | What it is for | How it is reached |
|---|---|---|
| Home PC dedicated server, `D:\Aaron\servers\gmod-ds` | fast headless tests: engine measurements, live runs with real physics, anything that needs files | `stage/gmod/gmod_ds_ctl.sh`, `stage/gmod/holodeck/live.py` |
| The game on the home PC | photographs, and trying things as a player | `stage/gmod/gmod_ctl.sh`, `stage/gmod/holodeck/shots.py` |
| **Cloud server** (pop-up games VM), `35.233.145.176:27015` | live multiplayer: real players, real voice, the public mod | the portal's Game Servers page; `ops` workflow (`command=start`, `args=garrysmod`); `ops tail garrysmod` reads its console |

The cloud server is Sandbox on `gm_construct`, 16 players, anonymous (join by address), Docker image `ceifa/garrysmod:latest`
(Linux). The VM sleeps 30 minutes after the last player leaves and takes about 5 minutes to wake. Only `data/` and `addons/` persist
(`data/garrysmod/addons` is where a hand-placed addon goes); `cfg/` and `gamemodes/` edits are lost at restart. No RCON, no pinned
version.

## Getting our addon onto it

Three routes, in the order I recommend:

1. **Git at boot (recommended).** The repo is public (`aaronsrhodes2/sigma-ground`, branch `master`), so the VM needs no
   credentials. Before the server starts, a step on the VM puts the addon folder `stage/gmod/addon` at
   `data/garrysmod/addons/mentat-stage`:
   ```bash
   git clone --depth 1 --branch master --filter=blob:none --sparse https://github.com/aaronsrhodes2/sigma-ground.git /tmp/sg
   git -C /tmp/sg sparse-checkout set stage/gmod/addon
   rm -rf <data>/garrysmod/addons/mentat-stage && cp -r /tmp/sg/stage/gmod/addon <data>/garrysmod/addons/mentat-stage
   ```
   Then the dev loop is: edit here, `git push origin master`, restart the game server, test. Players receive the client Lua
   automatically (the addon sends its own client files); they download nothing else.
2. **A hand-placed archive.** `stage/gmod/package_addon.sh` builds `stage/gmod/dist/mentat-stage.gma` (about 0.5 MB, Lua only).
   Put it in `data/garrysmod/addons/`. Needs some way to get a file there (the portal has none today).
3. **Workshop.** Publish the addon and add it to the portal's Workshop collection; the server and players pull it at every start.
   That publishes under your Steam account, so it is your decision, not something to do automatically.

**Which version runs.** Following `master` is the development loop, and it means whoever can push to the repo (or to the GitHub
account) can run Lua on the server and send Lua to every player who joins. Before strangers are on the server, pin it: a full commit
id fetched by id cannot be moved. The deployment patch in `stage/gmod/deploy/skippy-tel-gmod-addon.patch` (not applied to SkippyTel;
see below) adds a portal setting, `GARRYSMOD_ADDON_REF` ("Mod version": blank = master, or a full 40-character commit id), read by
`gmod_addon.sh`. The pin lives in the portal behind the Captain's login, not in either repo, because a pin that the same agent
sessions can edit and auto-deploy protects nothing.

I have not built anything that downloads and runs code from GitHub while the server is up (a live hot-update). It would make the
loop seconds instead of minutes, but a server that executes whatever the repo says is a remote-code-execution path if that account or
repo is ever compromised. If you want it, it should be superadmin-triggered, pinned to a commit you name, and nothing else; say so and
I will build exactly that.

## What the addon does by itself on a server

- Prints one line at start: `[holodeck] N of M catalog models are available here, K kinds` (read it with `ops tail garrysmod`).
  The catalog was measured with Half-Life 2 and Counter-Strike: Source content. If a model is missing on the server the holodeck
  leaves that kind out instead of spawning an error model. `holodeck_status` (console) shows the details.
- `holodeck_content hl2` restricts it to the Half-Life 2 models (about 72 of 127), which every player has. The default `all` uses
  everything the server has; a player without Counter-Strike: Source would see error models for the rest. Set it before launch
  with `+holodeck_content hl2`, or at any time with `holodeck_content hl2` then `holodeck_reload` (superadmin).
- Chat: any sentence that starts with **Skippy** is a request (`Skippy, add three couches`). Everything else players say is left
  alone. Players with a live mic talk normally; only the wake word starts the holodeck.
- Proximity voice is native to the server. It is installed as the gamemode's own `PlayerCanHearPlayersVoice`, wrapped around the
  rule the gamemode already had, so a gag or mute addon (a hook that returns false) always wins and the game's own refusal still
  holds. About 21 m for a normal voice with a smooth fade, crouching whispers, holding `+mentat_shout` shouts (the server
  remembers a shout for 1.5 s at a time), walls muffle. `mentat_voice_mode global` turns it off (everyone hears everyone, flat),
  `off` hands voice back to the game's own rule; both also tell the clients not to fade voices. The mode is read once the same way
  everywhere (case and spaces ignored; anything that is not `proximity`, `global` or `off` means `proximity`, with one console
  warning) and the clients are sent that value. If another addon loads later and replaces the gamemode's voice rule, the rule is
  wrapped around it again within half a second (one console line says so). `mentat_voice_range` sets the
  reach, `mentat_voice_wall 0` stops walls shrinking it. A player can run `mentat_voice_fade 0` to opt out of the fade.
- Safety on a public server:
  - everyone's props belong to the player who made them (admins can change anything), and they are created as that player's props
    (`SetCreator`, and the CPPI owner when a prop-protection addon is present; a prop still waiting for its turn to appear when its
    builder leaves is made at that moment, so it is theirs too); the server's own spawn rules apply too: `PlayerSpawnObject` and
    `PlayerSpawnProp` are asked the way the engine asks them (registered hooks first, then the gamemode's own function, so Sandbox's
    `sbox_maxprops` check and a derived gamemode's jail or ban rules count), and an explicit `false` from either is a veto. A gamemode
    that has no such function forbids nothing, and when it has none the holodeck asks the player's `CheckLimit("props")` itself, once.
    Holodeck props are not counted against `sbox_maxprops` (`holodeck_max_per_player` and `holodeck_max_ents` limit them);
  - a rest between requests (`holodeck_cooldown`, 1 s, checked before anything else, longer after an expensive request), a cap per
    player (`holodeck_max_per_player`, 150) and overall (`holodeck_max_ents`, 400). Undo obeys the same caps. When the server is full,
    the props of players who left go first, then the oldest props of whoever holds more than a fair share; nobody can lock the
    holodeck by leaving. Props of players who left are taken down after `holodeck_orphan_minutes` (30; 0 = keep until the room is needed);
  - the planner runs on the server's main thread, so an ordinary player gets 0.05 s of planning per request, 40 things at most, and
    the server as a whole about a tenth of its time; admins get 0.25 s and 150 things;
  - a sentence may hold up to eight requests (the parser's own limit, one number); the clauses past it, and any text past 400
    characters, are named to the requester in a private line, and a request to remove, move or clear that was cut at 400
    characters is refused rather than carried out on half a sentence. "Another one" / "again" repeat the last thing added, even
    after a turn, move, remove or undo in between; undoing an add gives "it" back to the group before it;
  - when the library changes under standing props (`holodeck_content`, then `holodeck_reload`), a prop whose model left the library
    stays nameable and removable (marked legacy, with the variant it was built from). Undo never respawns a model the server no
    longer has (it says how many stayed gone) and never trusts a saved variant index;
  - the floor and ceiling probes see props somebody froze (a platform to stand on) as ground, never loose props, players, NPCs,
    ragdolls or the holodeck's own props; walls stay the map's brushes only;
  - Skippy's lines go to the requester and whoever is within earshot of where he stands (2600 HU), never to the whole server; they
    carry no words the player typed (an unknown word is shown only to the requester) and each viewer's own chat filter applies;
  - new props let players walk through until they are set free, and wait for anyone standing in the footprint to step out;
  - the older scenario commands (`mentat_*`) are admin-only.
- The bridge (agent, persona, voice) is rationed because every call costs a model run, a speech synthesis or credit: per player
  (a few calls a minute), for the whole server (`holodeck_bridge_per_min`, 24; `holodeck_voice_chars_per_min`, 1800) and in flight
  (`holodeck_bridge_inflight`, 4). When a limit is reached Skippy answers in plain text. The agent may only add things
  (`holodeck_agent_destructive 1` lets it also remove, move and clear). The bridge listens on port 8777
  (`holodeck_bridge http://127.0.0.1:8777`). `holodeck_bridge_token` is a secret and is deliberately not saved: the convar is not
  `FCVAR_ARCHIVE`, so it never reaches `server.vdf`; set it at every start (launch line or `server.cfg`). A token already written to
  an old `server.vdf` stays there until that line is deleted by hand. Spoken clips are sent a chunk at a time, at most about
  160 KB, mp3 only.

## What needs more before it works on the cloud server

- **Skippy's agent and voice** (answers in character, spoken, in 3D) need the Skippy bridge (`stage/gmod/holodeck/skippy_bridge.py`)
  reachable from the server. It runs next to the server and uses a local language model and speech; the cloud VM has neither, and
  the home PC cannot be reached from the VM. Until a bridge exists there, the holodeck answers in plain text and the parser (not the
  agent) understands the words. A bridge container on the VM (a hosted model and a voice) is the missing piece.
- **Speaking to Skippy by voice** works from any player's own PC (their microphone bridge, `skippy_ears.py`, talks to their local
  Whisper and sends the sentence to the server as text), so it does not need anything on the cloud VM.
- The home server needs `-allowlocalhttp` for the bridge on localhost (already in `gmod_ds_ctl.sh` and `gmod_ctl.sh`); the cloud VM
  does not unless a bridge runs on the same machine.

## The deployment patch (ready, not applied)

`stage/gmod/deploy/skippy-tel-gmod-addon.patch` is the whole of route 1 as a change to the SkippyTel repo, so nobody has to wait for
the portal chat to write it. It was checked against SkippyTel `origin/main` (`84f817c`, 2026-10-10): `git apply --check` is clean and
the 57 tests in `portal/tests/test_gmod_addon.py`, `test_garrysmod.py` and `test_fix_watchdog_startup.py` pass with it. It adds:

- `infra/gcp-games/gmod_addon.sh`, shipped in instance metadata, run at every boot and by `ops restart-game garrysmod` (stop, refresh
  the addon, start). Shallow sparse fetch of `stage/gmod/addon`, no `.git` left behind, file times untouched when nothing changed (the
  backup job fingerprints them), never stops the server from starting. Follows `master`, or the commit named by the new portal
  setting `GARRYSMOD_ADDON_REF`.
- `GARRYSMOD_EXTRA_ARGS` ("Extra server arguments", validated), so `+holodeck_content hl2` and the like can be set from the portal.
- the documentation lines, and tests (the script is run for real against a local repository).

Applying it changes infrastructure code that the next `terraform apply` ships to the live VM, so it waits for the Captain: in the
SkippyTel checkout, `git apply <this patch>`, run its tests, then deploy the way SkippyTel deploys.

## Content: the holodeck needs Half-Life 2 and Counter-Strike: Source models on the server

The portal chat's second report (2026-10-10, after the first) says it plainly: **the cloud server does not have those games' content
mounted the way the home server's `sourceengine` folder does.** The holodeck's catalog is 127 models from Half-Life 2 and
Counter-Strike: Source. `util.IsValidModel` is false on a server that lacks a model, and the holodeck then leaves that model out
(`library.lua`), so on the cloud server as it stands it will have few or none of its kinds. The `[holodeck] N of M catalog models are
available here` line at start says how many (`ops tail garrysmod`); `holodeck_status` shows what is missing.

What fixes it is a server-side mount of the two games' content (`mount.cfg` pointing at their folders), which needs the content on the
VM's disk. Steam's anonymous dedicated-server packages for Counter-Strike: Source (app 232330) and Half-Life 2: Deathmatch (app
232370) carry it; I have not verified that the image can mount them, or what that does to the persistent-disk size, so this is a
request for the portal chat, not something I have done. The players' side is separate: each client mounts its own copies of those
games (GMod does this for owned games), so a model the server has but a player lacks shows as an error model for that player;
`holodeck_content hl2` keeps to the Half-Life 2 models, which every GMod player has (72 of 127).

## Not verified (from the portal chat's reports)

The server name still shows the old "Skippy" until its next restart; no Steam login token, no RCON; Workshop download and
`sv_password` have never been exercised on the live server; the portal's per-game admin card cannot show a version for this game; the
image offers no way to pin the Garry's Mod version and re-updates from Steam at every start, and only `data/` and `addons/` persist
(`cfg/` and `gamemodes/` edits are lost at restart, and `server.cfg` is not editable from the portal yet). Verified by the portal chat:
the server answers Steam's server query from outside, and `READY garrysmod=1` came about 5 minutes after the VM started.

## Prompt for the portal chat

Paste this into the portal chat (it owns `portal/catalog.py` and `games/docker-compose.yml`):

> The Garry's Mod server on the VM is now our live development server for the Mentat stage mod. I need one thing added: **at each
> start of the `garrysmod` service, before srcds starts, put the addon from the public repo into the persistent addons folder.**
> Repo: `https://github.com/aaronsrhodes2/sigma-ground` (public, branch `master`), path `stage/gmod/addon`, target
> `data/garrysmod/addons/mentat-stage` (replace it each start so a push to `master` shows up on the next start; add a portal setting
> `GARRYSMOD_ADDON_REF`, blank = `master`, or a full 40-character commit id that pins exactly that version, fetched by id). Use a sparse, shallow
> clone, for example `git clone --depth 1 --branch master --filter=blob:none --sparse <url> /tmp/sg && git -C /tmp/sg sparse-checkout
> set stage/gmod/addon`, then copy the folder over the target. If the clone fails (no network, GitHub down) keep the previous copy and
> say so in the log; never block the server from starting. Then please add a **restart-only** operation (restart the game server
> without sleeping or waking the VM) so a code change can be tested in about a minute, and keep `ops tail garrysmod` working so I can
> read the console; the addon prints `[mentat] holodeck ready` and `[holodeck] N of M catalog models are available here` at start.
> Optional but useful: RCON on the VM's localhost only (password in the vault, never in chat), reachable through `ops`, so I can run
> `holodeck_status` and `holodeck_reload` without a restart. Do not open anything else to the internet. Report in one to three
> sentences what changed and how to trigger a restart.
>
> **Second thing, only after the first works:** the mod spawns models from Half-Life 2 and Counter-Strike: Source, and your report says
> the server does not have that content mounted. Please find out whether the image can mount it (for example SteamCMD's anonymous
> dedicated-server packages for Counter-Strike: Source, app 232330, and Half-Life 2: Deathmatch, app 232370, installed on the persistent
> data disk and listed in `garrysmod/cfg/mount.cfg`, which must also survive restarts since `cfg/` does not persist), what it costs in
> disk and boot time, and report. After it is mounted, `[holodeck] N of M catalog models are available here` in `ops tail garrysmod`
> should say 127 of 127. If it cannot be done, say why; the mod then runs with `+holodeck_content hl2` and whatever the image has.
