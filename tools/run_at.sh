#!/usr/bin/env bash
# Run a command at the next HH:MM of local time, keeping Windows awake while waiting and running.
#
#   tools/run_at.sh 00:00 env DOCKER_OLLAMA=skippytel-ollama tools/run_router_sweep.sh
#
# Run it as a tracked background command so its exit (finished or failed) wakes the session. It prints
# when it will start and when it does, and exits with the command's status.
set -euo pipefail
(( $# >= 2 )) || { echo "usage: run_at.sh HH:MM command [args...]" >&2; exit 2; }
target="$1"; shift
now=$(date +%s)
t=$(date -d "today $target" +%s)
(( t > now )) || t=$(date -d "tomorrow $target" +%s)
printf '%s run_at: waiting until %s to run: %s\n' "$(date '+%F %T')" "$(date -d "@$t" '+%F %T')" "$*"

# ES_CONTINUOUS | ES_SYSTEM_REQUIRED for as long as the flag file exists (a per-process request).
flag="$(mktemp)"
trap 'rm -f "$flag"' EXIT
python -c "import ctypes, os, sys, time; ctypes.windll.kernel32.SetThreadExecutionState(0x80000001); [time.sleep(20) for _ in iter(lambda: os.path.exists(sys.argv[1]), False)]" "$flag" &

while (( $(date +%s) < t )); do sleep 20; done
printf '%s run_at: starting\n' "$(date '+%F %T')"
rc=0
"$@" || rc=$?
printf '%s run_at: finished with status %s\n' "$(date '+%F %T')" "$rc"
exit "$rc"
