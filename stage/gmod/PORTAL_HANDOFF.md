# Handoff: put the Garry's Mod dedicated server on the makernet.live portal

> **Superseded (2026-10-10).** The portal chat chose a different route: a public server on the pop-up games VM, wired in like Valheim.
> That is now our live mod server; see [CLOUD_SERVER.md](CLOUD_SERVER.md). This file only matters if you also want the home-PC
> dedicated server (`D:\Aaron\servers\gmod-ds`) on the portal. For that, note SkippyTel's own service-registration protocol
> (`POST /services/register`, `X-Skippy-Token`, heartbeats; `services/flask-api/app/routes/service_registration.py`).

Paste the prompt below into a Claude Code session opened in the portal's codebase. `D:\Aaron\development\SkippyTel` has a `portal/` folder and is the likely home, but a search of the dev tree found no file that names makernet.live, so if the portal is somewhere else, open that instead.
Everything in "Facts" was read from the files and the running machine on 2026-10-10, not remembered.

---

## Prompt

You are adding one more service to the makernet.live portal: a Garry's Mod dedicated server that already exists on
this PC. Do not rebuild it, reinstall it or change the game. Your job is to make the portal able to show it, start it,
stop it and report on it, the way it already does for its other services. Follow the portal's own conventions (read
`PLATINUM_RULES.md` and `CLAUDE.md` in `D:\Aaron\development` first, in particular the deploy-and-verify rule, bash
first, and the rules for agents that can deploy). If a classifier or permission denial blocks you, report it to me
instead of working around it.

### Facts about the server (verified)

- Install: `D:\Aaron\servers\gmod-ds` (Steam app 4020, installed anonymously with SteamCMD; `srcds.exe` is a plain
  Windows process, not a container, so the Docker rules do not apply to it).
- The one supported way to run it is the control script in the sigma-ground repo:
  `D:\Aaron\development\sigma-ground\stage\gmod\gmod_ds_ctl.sh` with subcommands
  `setup | status | start [map] | stop | wait-ready`. Exit code 0 means ok. `status` prints
  `gmod-ds: running (N process(es))` or `gmod-ds: not running` (exit 1).
- It currently starts like this (read from the script): `srcds.exe -console -game garrysmod -condebug -insecure
  -nohltv -norestart -tickrate 66.666667 -port 27055 +maxplayers 2 +sv_lan 1 +sv_hibernate_think 1
  +sv_hibernate_when_empty 0 +map gm_flatgrass`. Defaults can be overridden with the environment variables
  `GMOD_DS_DIR`, `GMOD_DS_PORT` (27055) and `GMOD_DS_TICK`.
- It is a LAN-only, insecure (no VAC, no Steam login) server meant for headless physics experiments. It is not
  reachable from the internet today and nothing is forwarded. Making it playable by other people is a decision
  for me; list the options (Tailscale only, which fits our doctrine that all device traffic goes over the tailnet;
  or a public server, which needs a Steam game-server login token, `sv_lan 0`, a firewall rule for UDP and TCP on
  the port, and removing `-insecure`) and recommend one. Do not open anything to the internet without my say-so.
- It is stopped right now. Starting it is safe and takes about 30 to 60 seconds; the script's `wait-ready` returns
  once the Mentat addon announces itself in `D:\Aaron\servers\gmod-ds\garrysmod\console.log`
  (the line contains `[mentat`).
- Addon: `D:\Aaron\servers\gmod-ds\garrysmod\addons\mentat-stage` is a directory junction to
  `D:\Aaron\development\sigma-ground\stage\gmod\addon`, so editing the repo changes the server (restart to apply).
- Logs: `D:\Aaron\servers\gmod-ds\garrysmod\console.log` (deleted on every start). No rcon password is set; `server.cfg`
  is empty. If you need one, put it in the dev-root vault `D:\Aaron\development\.env` (PLATINUM §10), never in
  chat or in a committed file.
- The server holds no state worth saving; `stop` ends the process. A health probe that does not need rcon: the Source
  query protocol (`A2S_INFO`) over UDP on the server port, or simply `gmod_ds_ctl.sh status`.
- Players need Half-Life 2 and Counter-Strike: Source content mounted on their own machines. The holodeck feature
  now being built spawns real models from those two games. The server has the content through its `sourceengine`
  folder.
- Not built yet, coming from the sigma-ground side: the holodeck (ask for things in plain words and they appear,
  wake word "Skippy"), distance-based voice, and an in-game voice and text agent. When they land they add a small
  local bridge process next to the server and some convars; I will hand you a second note then. Design the portal
  entry so a "companion process" can be added later.

### What to deliver

1. A portal entry for the server, in whatever form the portal uses for services (read how SkippyTel's other
   services are registered, for example the status page at `http://localhost:3003/status`): name, status
   (running or stopped, map, uptime if cheap to get), start and stop controls behind the same login the portal
   already requires, and a link to the log tail.
2. Calls go through `gmod_ds_ctl.sh` run by bash (not by hand-built command lines), so there is one definition of
   how the server starts. The portal runs in Docker and the server runs on the Windows host: work out the narrowest
   safe bridge between them (the existing bridge pattern on port 9337 is the model, but note PLATINUM §19 and the
   2026-09-29 incident about an unauthenticated listener with every permission check skipped: whatever you add must
   be authenticated, bound to the narrowest interface, and offer fixed actions only, never an arbitrary command).
3. Tests that exercise the real thing: start, status, stop against the actual server, plus the authentication
   failure path. Verify live and report in one to three sentences: what changed, whether it is live, caveats.
4. Do not touch `D:\Aaron\development\sigma-ground\stage\gmod\` other than reading it; another session is working
   there and Garry's Mod allows only one game instance, so do not start the game client (`gmod_ctl.sh`).

Ask me only for the decisions marked as mine (exposure of the server). Everything else, decide and report.

---

## Notes for me (the Captain)

- The control script is the contract: portal code should call it and nothing else. If the start flags need to change
  for a public server, change them in the script in one place.
- `+maxplayers 2` is an experiments setting. A multiplayer holodeck wants a higher number; that is a one-line change
  in the script once exposure is decided.
