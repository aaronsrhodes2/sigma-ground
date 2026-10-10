#!/usr/bin/env bash
# Start, stop and inspect Garry's Mod without leaving a half-dead instance behind.
#
# GMod allows ONE instance, and it cannot be told to quit the obvious ways:
#   * Lua   RunConsoleCommand("quit")        -> "Command is blocked!"
#   * Win32 WM_CLOSE (CloseMainWindow)       -> ignored by the engine for 30 s+
#   * Win32 WM_SYSCOMMAND / SC_CLOSE         -> WORKS: the whole process tree (game plus its
#                                               Chromium helper processes) exits in under 10 s
# So `stop` posts SC_CLOSE, waits for EVERY gmod/gmod_win64 process to exit, and only then falls
# back to a window close and finally a forced tree kill (reported as such).
#
#   stage/gmod/gmod_ctl.sh status
#   stage/gmod/gmod_ctl.sh stop
#   stage/gmod/gmod_ctl.sh start [map]      # default gm_construct; refuses if an instance is already running
#   stage/gmod/gmod_ctl.sh restart [map]
#   stage/gmod/gmod_ctl.sh wait-ready       # until the addon announces itself in console.log
#
# Env: GMOD_DIR (install dir), GMOD_EXE (default gmod.exe, the 32-bit launcher Steam uses;
#      gmod_win64.exe for the 64-bit build), GMOD_STOP_TIMEOUT (seconds per rung, default 20).
# Shell note (PLATINUM §20): bash first; PowerShell only for the window message, which has no bash equivalent.
set -u
GMOD_DIR="${GMOD_DIR:-/d/Steam/steamapps/common/GarrysMod}"
GMOD_EXE="${GMOD_EXE:-gmod.exe}"
LOG="$GMOD_DIR/garrysmod/console.log"
TIMEOUT="${GMOD_STOP_TIMEOUT:-20}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

count() {
  local a b
  a=$(tasklist //FI "IMAGENAME eq gmod.exe" //FO CSV //NH 2>/dev/null | grep -c '"gmod.exe"' || true)
  b=$(tasklist //FI "IMAGENAME eq gmod_win64.exe" //FO CSV //NH 2>/dev/null | grep -c '"gmod_win64.exe"' || true)
  echo $((a + b))
}

wait_gone() {   # wait_gone <seconds>: returns 0 once no GMod process remains
  local s=0
  while [ "$(count)" -gt 0 ] && [ "$s" -lt "$1" ]; do sleep 1; s=$((s + 1)); done
  WAITED=$s
  [ "$(count)" -eq 0 ]
}

status() {
  local n; n=$(count)
  if [ "$n" -eq 0 ]; then echo "gmod: not running"; return 1; fi
  echo "gmod: running ($n process(es))"
  powershell.exe -NoProfile -Command \
    "Get-Process gmod, gmod_win64 -ErrorAction SilentlyContinue | Where-Object { \$_.MainWindowHandle -ne 0 } | ForEach-Object { 'window: ' + \$_.MainWindowTitle + ' (pid ' + \$_.Id + ', ' + \$_.ProcessName + ', responding=' + \$_.Responding + ')' }" 2>/dev/null | tr -d '\r'
  return 0
}

stop() {
  local n; n=$(count)
  if [ "$n" -eq 0 ]; then echo "gmod: already stopped"; return 0; fi
  echo "gmod: closing ($n process(es)): SC_CLOSE"
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$(cygpath -w "$HERE/gmod_close.ps1")" 2>/dev/null | tr -d '\r'
  if wait_gone "$TIMEOUT"; then echo "gmod: stopped cleanly in ${WAITED}s"; return 0; fi

  echo "gmod: ignored SC_CLOSE for ${TIMEOUT}s; trying a plain window close"
  powershell.exe -NoProfile -Command \
    "Get-Process gmod, gmod_win64 -ErrorAction SilentlyContinue | Where-Object { \$_.MainWindowHandle -ne 0 } | ForEach-Object { [void]\$_.CloseMainWindow() }" >/dev/null 2>&1
  if wait_gone 10; then echo "gmod: stopped after window close"; return 0; fi

  echo "gmod: still running; FORCING (process tree). This skips the engine's own shutdown."
  taskkill //F //T //IM gmod.exe >/dev/null 2>&1
  taskkill //F //T //IM gmod_win64.exe >/dev/null 2>&1
  if wait_gone 5; then echo "gmod: stopped by force"; return 0; fi
  echo "gmod: FAILED to stop; remaining:" >&2
  tasklist 2>&1 | grep -i gmod >&2
  return 1
}

start() {
  local map="${1:-gm_construct}"
  if [ "$(count)" -gt 0 ]; then
    echo "gmod: already running (one instance only). Run '$0 stop' first." >&2
    return 1
  fi
  if [ -e "$LOG" ] && ! rm -f "$LOG" 2>/dev/null; then
    echo "gmod: console.log is locked by something; refusing to start (a stale instance?)" >&2
    return 1
  fi
  ( cd "$GMOD_DIR" && ./"$GMOD_EXE" -console -condebug -novid -windowed -w 1280 -h 720 +map "$map" >/dev/null 2>&1 & )
  local s=0
  while [ "$(count)" -eq 0 ] && [ "$s" -lt 10 ]; do sleep 1; s=$((s + 1)); done
  if [ "$(count)" -gt 0 ]; then echo "gmod: started ($GMOD_EXE) on $map"; else echo "gmod: failed to start" >&2; return 1; fi
}

wait_ready() {   # the game is interactive once the clientside Lua has started and the map's console banner is written
  local s=0
  until grep -q "Redownloading all lightmaps\|\[mentat" "$LOG" 2>/dev/null || [ "$s" -ge 120 ]; do sleep 2; s=$((s + 2)); done
  if [ "$s" -lt 120 ]; then echo "gmod: ready after ${s}s"; else echo "gmod: not ready after 120s" >&2; return 1; fi
}

case "${1:-status}" in
  status) status ;;
  stop) stop ;;
  start) start "${2:-}" ;;
  restart) stop && start "${2:-}" ;;
  wait-ready) wait_ready ;;
  *) echo "usage: $0 status|stop|start [map]|restart [map]|wait-ready" >&2; exit 2 ;;
esac
