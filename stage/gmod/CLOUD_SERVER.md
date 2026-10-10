# The cloud Garry's Mod server as our live mod server

Status as reported by the portal chat on 2026-10-10 (I have not been able to check it myself; see "Not verified" below), and how we
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
- Proximity voice is native to the server (the `PlayerCanHearPlayersVoice` hook): about 21 m for a normal voice with a smooth fade,
  crouching whispers, holding `+mentat_shout` shouts, walls muffle. `mentat_voice_mode global` turns it off (everyone hears everyone),
  `off` hands voice back to the game's own rule. `mentat_voice_range` sets the reach.
- Safety on a public server: everyone's props belong to the player who made them (admins can change anything); a cooldown between
  requests (`holodeck_cooldown`, 1 s), a cap per player (`holodeck_max_per_player`, 150) and overall (`holodeck_max_ents`, 400); the
  older scenario commands (`mentat_*`) are admin-only.

## What needs more before it works on the cloud server

- **Skippy's agent and voice** (answers in character, spoken, in 3D) need the Skippy bridge (`stage/gmod/holodeck/skippy_bridge.py`)
  reachable from the server. It runs next to the server and uses a local language model and speech; the cloud VM has neither, and
  the home PC cannot be reached from the VM. Until a bridge exists there, the holodeck answers in plain text and the parser (not the
  agent) understands the words. A bridge container on the VM (a hosted model and a voice) is the missing piece.
- **Speaking to Skippy by voice** works from any player's own PC (their microphone bridge, `skippy_ears.py`, talks to their local
  Whisper and sends the sentence to the server as text), so it does not need anything on the cloud VM.
- The home server needs `-allowlocalhttp` for the bridge on localhost (already in `gmod_ds_ctl.sh` and `gmod_ctl.sh`); the cloud VM
  does not unless a bridge runs on the same machine.

## Not verified (from the portal chat's report)

The server name still shows the old "Skippy" until its next restart; no Steam login token, no RCON; Workshop download and
`sv_password` have never been exercised on the live server; whether the image's own Source content includes the Half-Life 2 and
Counter-Strike: Source models (the home server's `sourceengine` folder does). The `[holodeck] N of M` line will answer the last one
the first time the addon starts there.

## Prompt for the portal chat

Paste this into the portal chat (it owns `portal/catalog.py` and `games/docker-compose.yml`):

> The Garry's Mod server on the VM is now our live development server for the Mentat stage mod. I need one thing added: **at each
> start of the `garrysmod` service, before srcds starts, put the addon from the public repo into the persistent addons folder.**
> Repo: `https://github.com/aaronsrhodes2/sigma-ground` (public, branch `master`), path `stage/gmod/addon`, target
> `data/garrysmod/addons/mentat-stage` (replace it each start so a push to `master` shows up on the next start). Use a sparse, shallow
> clone, for example `git clone --depth 1 --branch master --filter=blob:none --sparse <url> /tmp/sg && git -C /tmp/sg sparse-checkout
> set stage/gmod/addon`, then copy the folder over the target. If the clone fails (no network, GitHub down) keep the previous copy and
> say so in the log; never block the server from starting. Then please add a **restart-only** operation (restart the game server
> without sleeping or waking the VM) so a code change can be tested in about a minute, and keep `ops tail garrysmod` working so I can
> read the console; the addon prints `[mentat] holodeck ready` and `[holodeck] N of M catalog models are available here` at start.
> Optional but useful: RCON on the VM's localhost only (password in the vault, never in chat), reachable through `ops`, so I can run
> `holodeck_status` and `holodeck_reload` without a restart. Do not open anything else to the internet. Report in one to three
> sentences what changed and how to trigger a restart.
