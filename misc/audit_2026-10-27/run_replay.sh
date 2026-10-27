#!/usr/bin/env bash
# Replay parts unattended against a box that is recreated on every trunk push: wait until Mentat has been up 60 s, run ONE part, then re-ask
# the rows where /ask raised (QwenUnavailable: the shim was down) or the transport failed, after waiting again; merge; next part.
#   bash run_replay.sh OUT_DIR part1,part2,...           parts as audit_replay.py names them (replay_<suite>_<variant>)
set -u
OUT="$1"; PARTS="$2"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd /d/Aaron/development/sigma-ground || exit 1
wait_up() {
  local n=0
  until [ "$(curl -s --max-time 8 http://127.0.0.1:17350/health | python -c 'import sys,json; d=json.load(sys.stdin); print(bool(d.get("ready")) and d.get("uptime_s",0) >= 60)' 2>/dev/null)" = "True" ]; do
    n=$((n+1)); [ $n -gt 120 ] && { echo "gave up waiting for Mentat"; return 3; }; sleep 15
  done
}
spoiled() {   # ids of rows where nothing was decided: /ask raised, or the transport failed
  python -c "
import json,sys
d=json.load(open(sys.argv[1], encoding='utf-8'))
bad=[r['id'] for r in d['rows'] if (r.get('route') is None and r.get('router_error') is None and r.get('router_decline') is None and r.get('router_clarify') is None and r.get('value') is None)
     or (r.get('router_exc') or '').split(':')[0] in ('ReadError','ConnectError','HTTPStatusError','RemoteProtocolError','ConnectTimeout','ReadTimeout')]
print(','.join(bad))" "$1"
}
for part in $(echo "$PARTS" | tr ',' ' '); do
  suite=$(echo "$part" | cut -d_ -f2); variant=$(echo "$part" | cut -d_ -f3)
  f="$OUT/replay_${suite}_${variant}.json"
  wait_up || exit 3
  python "$S/audit_replay.py" "$OUT" "$part" > "$OUT.$part.log" 2>&1
  for attempt in 1 2 3; do
    ids=$(spoiled "$f")
    [ -z "$ids" ] && break
    echo "$part attempt $attempt: re-asking $(echo "$ids" | tr ',' '\n' | wc -l) spoiled rows"
    wait_up || exit 3
    REPLAY_IDS="$ids" python "$S/audit_replay.py" "$OUT/retry_$attempt" "$part" > "$OUT.$part.retry$attempt.log" 2>&1
    python "$S/merge_replay.py" "$f" "$OUT/retry_$attempt/replay_${suite}_${variant}.json"
  done
  python -c "
import json,sys
d=json.load(open(sys.argv[1], encoding='utf-8')); s=d['summary']; rows=d['rows']
left=[r['id'] for r in rows if r.get('route') is None and r.get('value') is None and r.get('router_error') is None]
print(json.dumps({'part': s['part'], 'n': len(rows), 'ok': sum(r['ok'] for r in rows), 'by_route': s.get('by_route'), 'undecided_left': len(left)}))" "$f"
done
echo "REPLAY DONE"
