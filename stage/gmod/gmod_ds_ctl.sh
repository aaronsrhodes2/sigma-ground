#!/usr/bin/env bash
# Start, stop and inspect the Garry's Mod DEDICATED server used for headless engine experiments.
#
# A dedicated server has no window, no sound and no mouse: nothing to pop up or grab input. It is a separate
# install (Steam app 4020) from the game, so it can run while the game is closed and never competes with it.
#
#   stage/gmod/gmod_ds_ctl.sh setup           # link the Mentat addon into the server (idempotent; re-links a stale link)
#   stage/gmod/gmod_ds_ctl.sh status
#   stage/gmod/gmod_ds_ctl.sh start [map]     # default gm_flatgrass; minimized console, no window focus
#   stage/gmod/gmod_ds_ctl.sh stop
#   stage/gmod/gmod_ds_ctl.sh wait-ready
#
# Install the server once (about 6 GB, from Valve's anonymous SteamCMD depot):
#   steamcmd +force_install_dir D:/Aaron/servers/gmod-ds +login anonymous +app_update 4020 validate +quit
#
# Our server is the srcds.exe started with `-port $GMOD_DS_PORT`: status and stop look at that, never at other Source servers
# (a TF2 or CS server on the same PC is neither "ours" nor ended by `stop`).
#
# Env: GMOD_DS_DIR (server install dir, default /d/Aaron/servers/gmod-ds), GMOD_DS_PORT (default 27055),
#      GMOD_DS_TICK (tickrate, default 66.666667 = the 15 ms tick the game itself runs at).
# Shell note (PLATINUM §20): bash first; PowerShell only to read a process's command line, which bash cannot do on Windows.
set -u
DS="${GMOD_DS_DIR:-/d/Aaron/servers/gmod-ds}"
PORT="${GMOD_DS_PORT:-27055}"
TICK="${GMOD_DS_TICK:-66.666667}"
IMAGE="${GMOD_DS_IMAGE:-srcds.exe}"          # the program's name; a test points it at a harmless stand-in
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADDON_SRC="$HERE/addon"
ADDON_LINK="$DS/garrysmod/addons/mentat-stage"
LOG="$DS/garrysmod/console.log"
TIMEOUT="${GMOD_DS_STOP_TIMEOUT:-15}"

# PIDs of OUR server (the process of that name whose command line has `-port <ours>`), one per line. Returns 1 when the
# process list could not be read, so callers never mistake "could not look" for "not running".
pids() {
  # nothing of that name at all: no need to start PowerShell
  tasklist //FI "IMAGENAME eq $IMAGE" //FO CSV //NH 2>/dev/null | grep -q "\"$IMAGE\"" || return 0
  local out
  out="$(powershell.exe -NoProfile -NonInteractive -Command \
    "try { Get-CimInstance Win32_Process -ErrorAction Stop | Where-Object { \$_.Name -eq '$IMAGE' -and \$_.CommandLine -match '(^|\\s)-port\\s+$PORT(\\s|\$)' } | ForEach-Object { \$_.ProcessId } } catch { exit 3 }" 2>/dev/null)" || return 1
  printf '%s\n' "$out" | tr -d '\r' | grep -E '^[0-9]+$' || true
}

count() { local p; p="$(pids)" || { echo "gmod-ds: cannot read the process list" >&2; echo 0; return 1; }; printf '%s\n' "$p" | grep -c '[0-9]' || true; }

installed() { [ -f "$DS/srcds.exe" ] && [ -d "$DS/garrysmod" ]; }

# lower-case forward-slash form of a path, resolved through links: two spellings of one place compare equal
canon() { local r; r="$(readlink -f "$1" 2>/dev/null)" || r="$1"; if command -v cygpath >/dev/null 2>&1; then r="$(cygpath -m "$r")"; fi; printf '%s\n' "${r,,}"; }

setup() {
  if ! installed; then echo "gmod-ds: not installed at $DS (see the header of this script)" >&2; return 1; fi
  mkdir -p "$DS/garrysmod/addons" "$DS/garrysmod/data/mentat"
  if [ -e "$ADDON_LINK" ] || [ -L "$ADDON_LINK" ]; then
    if [ -L "$ADDON_LINK" ]; then                                # a junction shows up as a link to Git Bash
      if [ "$(canon "$ADDON_LINK")" = "$(canon "$ADDON_SRC")" ]; then echo "gmod-ds: addon already linked"; return 0; fi
      # a link left by another checkout would make every run test THAT checkout's Lua: drop it (the rmdir removes the link only)
      echo "gmod-ds: the addon link points at $(readlink -f "$ADDON_LINK"), not here; re-linking"
      cmd //c rmdir "$(cygpath -w "$ADDON_LINK")" >/dev/null || { echo "gmod-ds: could not remove the old link" >&2; return 1; }
    else
      echo "gmod-ds: $ADDON_LINK exists and is not a link; move it away first" >&2; return 1
    fi
  fi
  # a directory junction needs no administrator rights
  cmd //c mklink //J "$(cygpath -w "$ADDON_LINK")" "$(cygpath -w "$ADDON_SRC")" >/dev/null || { echo "gmod-ds: junction failed" >&2; return 1; }
  echo "gmod-ds: linked $ADDON_LINK -> $ADDON_SRC"
}

status() {
  local n; n=$(count) || return 2                          # 2: could not tell (1 is "not running")
  if [ "$n" -eq 0 ]; then echo "gmod-ds: not running"; return 1; fi
  echo "gmod-ds: running ($n process(es) on port $PORT)"
}

stop() {
  local n s=0 p
  n=$(count) || { echo "gmod-ds: not stopping anything I cannot identify; end the server by hand" >&2; return 1; }
  if [ "$n" -eq 0 ]; then echo "gmod-ds: already stopped"; return 0; fi
  # The server holds no state worth saving, and its console takes input only from its own window, so there is no
  # gentler outside signal than ending the process. (A run with "quit" in experiment.txt exits by itself first.)
  for p in $(pids); do taskkill //F //T //PID "$p" >/dev/null 2>&1; done
  while [ "$(count)" -gt 0 ] && [ "$s" -lt "$TIMEOUT" ]; do sleep 1; s=$((s + 1)); done
  if [ "$(count)" -eq 0 ]; then echo "gmod-ds: stopped"; return 0; fi
  echo "gmod-ds: FAILED to stop" >&2; return 1
}

start() {
  local map="${1:-gm_flatgrass}"
  installed || { echo "gmod-ds: not installed at $DS" >&2; return 1; }
  setup >/dev/null || return 1
  local running; running="$(count)" || { echo "gmod-ds: cannot tell whether a server is already running; refusing to start" >&2; return 1; }
  if [ "$running" -gt 0 ]; then echo "gmod-ds: already running. Run '$0 stop' first." >&2; return 1; fi
  if [ -e "$LOG" ] && ! rm -f "$LOG" 2>/dev/null; then echo "gmod-ds: console.log is locked; refusing to start" >&2; return 1; fi
  # -insecure/sv_lan: no Steam login or master-server contact. sv_hibernate_*: an EMPTY server must keep simulating
  # (by default it sleeps and ignores Think, which would freeze the experiment). -nohltv: no TV relay.
  ( cd "$DS" && cmd //c start "" //min srcds.exe -console -game garrysmod -condebug -insecure -nohltv -norestart -allowlocalhttp \
      -tickrate "$TICK" -port "$PORT" +maxplayers "${GMOD_DS_MAXPLAYERS:-2}" +sv_lan 1 +sv_hibernate_think 1 +sv_hibernate_when_empty 0 \
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

# run the command only when executed; `source`d (by a test) it just defines the functions above
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  case "${1:-status}" in
    setup) setup ;;
    status) status ;;
    stop) stop ;;
    start) shift; start "$@" ;;
    wait-ready) wait_ready ;;
    *) echo "usage: $0 setup|status|start [map]|stop|wait-ready" >&2; exit 2 ;;
  esac
fi
