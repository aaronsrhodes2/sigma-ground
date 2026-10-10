#!/usr/bin/env bash
# Start, stop and inspect the Garry's Mod DEDICATED server used for headless engine experiments.
#
# A dedicated server has no window, no sound and no mouse: nothing to pop up or grab input. It is a separate
# install (Steam app 4020) from the game, so it can run while the game is closed and never competes with it.
#
#   stage/gmod/gmod_ds_ctl.sh setup           # link the Mentat addon into the server (idempotent)
#   stage/gmod/gmod_ds_ctl.sh status
#   stage/gmod/gmod_ds_ctl.sh start [map]     # default gm_flatgrass; minimized console, no window focus
#   stage/gmod/gmod_ds_ctl.sh stop
#   stage/gmod/gmod_ds_ctl.sh wait-ready
#
# Install the server once (about 6 GB, from Valve's anonymous SteamCMD depot):
#   steamcmd +force_install_dir D:/Aaron/servers/gmod-ds +login anonymous +app_update 4020 validate +quit
#
# Env: GMOD_DS_DIR (server install dir, default /d/Aaron/servers/gmod-ds), GMOD_DS_PORT (default 27055),
#      GMOD_DS_TICK (tickrate, default 66.666667 = the 15 ms tick the game itself runs at).
# Shell note (PLATINUM §20): bash first; nothing here needs PowerShell.
set -u
DS="${GMOD_DS_DIR:-/d/Aaron/servers/gmod-ds}"
PORT="${GMOD_DS_PORT:-27055}"
TICK="${GMOD_DS_TICK:-66.666667}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADDON_SRC="$HERE/addon"
ADDON_LINK="$DS/garrysmod/addons/mentat-stage"
LOG="$DS/garrysmod/console.log"
TIMEOUT="${GMOD_DS_STOP_TIMEOUT:-15}"

count() { tasklist //FI "IMAGENAME eq srcds.exe" //FO CSV //NH 2>/dev/null | grep -c '"srcds.exe"' || true; }

installed() { [ -f "$DS/srcds.exe" ] && [ -d "$DS/garrysmod" ]; }

setup() {
  if ! installed; then echo "gmod-ds: not installed at $DS (see the header of this script)" >&2; return 1; fi
  mkdir -p "$DS/garrysmod/addons" "$DS/garrysmod/data/mentat"
  if [ -e "$ADDON_LINK" ]; then echo "gmod-ds: addon already linked"; return 0; fi
  # a directory junction needs no administrator rights
  cmd //c mklink //J "$(cygpath -w "$ADDON_LINK")" "$(cygpath -w "$ADDON_SRC")" >/dev/null || { echo "gmod-ds: junction failed" >&2; return 1; }
  echo "gmod-ds: linked $ADDON_LINK -> $ADDON_SRC"
}

status() {
  local n; n=$(count)
  if [ "$n" -eq 0 ]; then echo "gmod-ds: not running"; return 1; fi
  echo "gmod-ds: running ($n process(es))"
}

stop() {
  local n s=0; n=$(count)
  if [ "$n" -eq 0 ]; then echo "gmod-ds: already stopped"; return 0; fi
  # The server holds no state worth saving, and its console takes input only from its own window, so there is no
  # gentler outside signal than ending the process. (A run with "quit" in experiment.txt exits by itself first.)
  taskkill //F //T //IM srcds.exe >/dev/null 2>&1
  while [ "$(count)" -gt 0 ] && [ "$s" -lt "$TIMEOUT" ]; do sleep 1; s=$((s + 1)); done
  if [ "$(count)" -eq 0 ]; then echo "gmod-ds: stopped"; return 0; fi
  echo "gmod-ds: FAILED to stop" >&2; return 1
}

start() {
  local map="${1:-gm_flatgrass}"
  installed || { echo "gmod-ds: not installed at $DS" >&2; return 1; }
  setup >/dev/null || return 1
  if [ "$(count)" -gt 0 ]; then echo "gmod-ds: already running. Run '$0 stop' first." >&2; return 1; fi
  if [ -e "$LOG" ] && ! rm -f "$LOG" 2>/dev/null; then echo "gmod-ds: console.log is locked; refusing to start" >&2; return 1; fi
  # -insecure/sv_lan: no Steam login or master-server contact. sv_hibernate_*: an EMPTY server must keep simulating
  # (by default it sleeps and ignores Think, which would freeze the experiment). -nohltv: no TV relay.
  ( cd "$DS" && cmd //c start "" //min srcds.exe -console -game garrysmod -condebug -insecure -nohltv -norestart \
      -tickrate "$TICK" -port "$PORT" +maxplayers 2 +sv_lan 1 +sv_hibernate_think 1 +sv_hibernate_when_empty 0 \
      +map "$map" >/dev/null 2>&1 )
  local s=0
  while [ "$(count)" -eq 0 ] && [ "$s" -lt 10 ]; do sleep 1; s=$((s + 1)); done
  if [ "$(count)" -gt 0 ]; then echo "gmod-ds: started on $map (port $PORT, tickrate $TICK)"; else echo "gmod-ds: failed to start" >&2; return 1; fi
}

wait_ready() {
  local s=0
  until grep -q "\[mentat" "$LOG" 2>/dev/null || [ "$s" -ge 120 ]; do sleep 2; s=$((s + 2)); done
  if [ "$s" -lt 120 ]; then echo "gmod-ds: ready after ${s}s"; else echo "gmod-ds: not ready after 120s" >&2; return 1; fi
}

case "${1:-status}" in
  setup) setup ;;
  status) status ;;
  stop) stop ;;
  start) shift; start "$@" ;;
  wait-ready) wait_ready ;;
  *) echo "usage: $0 setup|status|start [map]|stop|wait-ready" >&2; exit 2 ;;
esac
