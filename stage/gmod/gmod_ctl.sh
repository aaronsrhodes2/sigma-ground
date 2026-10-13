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
#   stage/gmod/gmod_ctl.sh start [map] [--quiet]   # default gm_construct; --quiet = minimized, muted, no mouse capture
#   stage/gmod/gmod_ctl.sh restart [map]
#   stage/gmod/gmod_ctl.sh wait-ready       # until the addon announces itself in console.log
#
# Env: GMOD_EXTRA_ARGS (more game arguments, e.g. "+maxplayers 4"), GMOD_DIR (install dir), GMOD_EXE (default gmod.exe, the 32-bit launcher Steam uses;
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

# The game renders on the same GPU as any training run. A window starved of GPU time stops pumping messages while it loads the map and Windows shows the
# frozen "not responding" spinner (seen 2026-10-12 with the GPU at 100%). So a hands-off run WAITS until the GPU is calm; a played session only warns.
# GMOD_GPU_MAX = the busiest GPU utilisation (percent) a hands-off start tolerates (default 70; 0 = do not check), GMOD_GPU_WAIT = seconds to wait (default 1200).
gpu_util() {   # the GPU's utilisation in percent, averaged over three samples; empty when there is no NVIDIA GPU or nvidia-smi
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  local i s sum=0 n=0
  for i in 1 2 3; do
    s=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' %\r')
    case "$s" in ''|*[!0-9]*) ;; *) sum=$((sum + s)); n=$((n + 1)) ;; esac
    [ "$i" -lt 3 ] && sleep 1
  done
  [ "$n" -gt 0 ] && echo $((sum / n))
}

gpu_calm() {   # gpu_calm hands_off(1|0): 0 when the GPU is calm enough to start; 3 when a hands-off start gave up waiting
  local max="${GMOD_GPU_MAX:-70}" wait_s="${GMOD_GPU_WAIT:-1200}" waited=0 u
  [ "$max" -le 0 ] && return 0
  while :; do
    u=$(gpu_util)
    [ -z "$u" ] && return 0
    [ "$u" -le "$max" ] && return 0
    if [ "$1" != "1" ]; then echo "gmod: warning: the GPU is $u% busy (a training run?); the game may stutter or freeze while it loads" >&2; return 0; fi
    if [ "$waited" -ge "$wait_s" ]; then echo "gmod: the GPU stayed $u% busy for ${waited}s; not starting a hands-off run (GMOD_GPU_MAX=0 ignores this)" >&2; return 3; fi
    [ "$waited" -eq 0 ] && echo "gmod: the GPU is $u% busy; waiting for it to calm down before starting (up to ${wait_s}s)" >&2
    sleep 15; waited=$((waited + 15))
  done
}

start() {
  # start [map] [--quiet]   --quiet: minimized, muted, no mouse or joystick capture (for automated runs)
  local map="gm_construct" quiet=0 a
  for a in "$@"; do
    case "$a" in --quiet) quiet=1 ;; "") ;; *) map="$a" ;; esac
  done
  if [ "$(count)" -gt 0 ]; then
    echo "gmod: already running (one instance only). Run '$0 stop' first." >&2
    return 1
  fi
  # hands-off = a quiet run or a test run (GMOD_MOUSE unset/0): wait for a calm GPU; a played session only gets a warning
  if [ "$quiet" -eq 1 ] || [ "${GMOD_MOUSE:-0}" = "0" ]; then gpu_calm 1 || return $?; else gpu_calm 0; fi
  if [ -e "$LOG" ] && ! rm -f "$LOG" 2>/dev/null; then
    echo "gmod: console.log is locked by something; refusing to start (a stale instance?)" >&2
    return 1
  fi
  if [ "$quiet" -eq 1 ]; then
    # `start /min` launches it already minimized so it never takes focus; the flags stop it grabbing the mouse
    ( cd "$GMOD_DIR" && cmd //c start "" //min "$GMOD_EXE" -console -condebug -novid -allowlocalhttp -nosound -nomouse -nojoy -windowed -w 640 -h 360 +map "$map" ${GMOD_EXTRA_ARGS:-} >/dev/null 2>&1 )
  elif [ "${GMOD_MOUSE:-0}" = "1" ]; then
    # an interactive play session (holodeck/play.py sets GMOD_MOUSE=1): normal window, the engine owns the mouse
    ( cd "$GMOD_DIR" && ./"$GMOD_EXE" -console -condebug -novid -allowlocalhttp -windowed -w 1280 -h 720 +map "$map" ${GMOD_EXTRA_ARGS:-} >/dev/null 2>&1 & )
  elif [ "${GMOD_MOUSE:-0}" = "2" ]; then
    # a played session that must leave the mouse alone (holodeck/play.py --text-only): a normal visible window, but no mouse or joystick
    # input to the engine, so it cannot recentre the cursor while you work elsewhere. Look with the arrow keys, or type cl_mouseenable 1.
    ( cd "$GMOD_DIR" && ./"$GMOD_EXE" -console -condebug -novid -allowlocalhttp -nomouse -nojoy -windowed -w 1280 -h 720 +map "$map" ${GMOD_EXTRA_ARGS:-} >/dev/null 2>&1 & )
  else
    # a visible TEST run (screenshots): the window must render, but it must never take the mouse or the focus from whoever is at the PC:
    # no mouse and no joystick input, and the window is kept behind everything, unfocused (gmod_behind.ps1)
    ( cd "$GMOD_DIR" && ./"$GMOD_EXE" -console -condebug -novid -allowlocalhttp -nomouse -nojoy -windowed -w 1280 -h 720 +map "$map" ${GMOD_EXTRA_ARGS:-} >/dev/null 2>&1 & )
  fi
  if [ "$quiet" -eq 1 ]; then
    # keep the window minimized and unfocused while it loads (see gmod_hide.ps1); detached, ends on its own
    ( powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$(cygpath -w "$HERE/gmod_hide.ps1")" >/dev/null 2>&1 & )
  elif [ "${GMOD_MOUSE:-0}" = "0" ]; then
    ( powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$(cygpath -w "$HERE/gmod_behind.ps1")" >/dev/null 2>&1 & )
  fi
  local s=0
  while [ "$(count)" -eq 0 ] && [ "$s" -lt 10 ]; do sleep 1; s=$((s + 1)); done
  if [ "$(count)" -gt 0 ]; then echo "gmod: started ($GMOD_EXE) on $map$([ "$quiet" -eq 1 ] && echo ' (quiet)')"; else echo "gmod: failed to start" >&2; return 1; fi
}

wait_ready() {   # the game is interactive once the clientside Lua has started and the map's console banner is written
  local s=0
  until grep -q "Redownloading all lightmaps\|\[mentat" "$LOG" 2>/dev/null || [ "$s" -ge 120 ]; do sleep 2; s=$((s + 2)); done
  if [ "$s" -lt 120 ]; then echo "gmod: ready after ${s}s"; else echo "gmod: not ready after 120s" >&2; return 1; fi
}

case "${1:-status}" in
  status) status ;;
  stop) stop ;;
  start) shift; start "$@" ;;
  restart) shift; stop && start "$@" ;;
  wait-ready) wait_ready ;;
  *) echo "usage: $0 status|stop|start [map]|restart [map]|wait-ready" >&2; exit 2 ;;
esac
