#!/usr/bin/env bash
# One day of the improvement-distillation branch, unattended (misc/DISTILLATION_BRANCH_PLAN.md), run by the Windows scheduled task
# "sigma-ground distill daily" on skippy-pc (the wa-trickle service runs here, 127.0.0.1:3010):
#   1. score every distill-* suite from the trickle's export (yesterday's set has had its Wolfram and Mentat answers by now),
#      write the day files and misc/DISTILLATION_LEDGER.md, commit THOSE PATHS ONLY and push (the checkout is shared: never -A);
#   2. make today's new set, with the wild questions from Core's chat model through a short ssh tunnel, and queue it in the trickle.
#   bash tools/distill_daily.sh            # the real thing
#   bash tools/distill_daily.sh --dry      # score and make the set, but no commit, no push, no enqueue
# Log: local-cache/distill/daily.log. Exit codes: 0 ok, 2 the trickle is not reachable (nothing done), 3 a step failed (see the log).
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 3
DRY=""; [ "${1:-}" = "--dry" ] && DRY=1
TRICKLE="${WA_TRICKLE_URL:-http://127.0.0.1:3010}"
CORE="${SKIPPY_CORE_SSH:-aaron@192.168.0.242}"
TUNNEL_PORT="${DISTILL_TUNNEL_PORT:-17434}"
mkdir -p local-cache/distill
LOG="local-cache/distill/daily.log"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 SKIP_SERVER_TESTS=1
log() { printf '%s %s\n' "$(date +'%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG"; }

log "== distill daily start${DRY:+ (dry)}"
if ! curl -s -m 10 "$TRICKLE/status" >/dev/null; then
  log "trickle not reachable at $TRICKLE: nothing done"
  exit 2
fi

# 1. score what has come back, and record it
if python -X utf8 -m sigma_ground.mcp.benchmark.distill_score --export "$TRICKLE" >>"$LOG" 2>&1; then
  if [ -z "$DRY" ]; then
    git add sigma_ground/mcp/benchmark/results/distill/*.json misc/DISTILLATION_LEDGER.md 2>>"$LOG"
    if ! git diff --cached --quiet; then
      git -c user.name="distill daily" -c user.email="distill@skippy-pc" commit -q -m "Distillation: the day files and ledger as scored on $(date +%Y-%m-%d)" >>"$LOG" 2>&1 \
        && git push -q origin master >>"$LOG" 2>&1 && log "scored and pushed" || log "commit/push failed (see above)"
    else
      log "scored: nothing new to commit"
    fi
  else
    log "scored (dry: not committed)"
  fi
else
  log "scoring failed"
fi

# 2. today's set, with the wild questions through a tunnel to Core's chat model (closed when the script ends)
TUNNEL_PID=""
if ! curl -s -m 5 "http://127.0.0.1:$TUNNEL_PORT/api/tags" >/dev/null 2>&1; then
  ssh -N -o BatchMode=yes -o ConnectTimeout=10 -o ExitOnForwardFailure=yes -o HostKeyAlias=skippy-core \
      -L "127.0.0.1:$TUNNEL_PORT:127.0.0.1:11434" "$CORE" >>"$LOG" 2>&1 &
  TUNNEL_PID=$!
  sleep 4
fi
WILD=""
curl -s -m 5 "http://127.0.0.1:$TUNNEL_PORT/api/tags" >/dev/null 2>&1 || { WILD="--no-wild"; log "Core's chat model not reachable: no wild questions today"; }
if [ -n "$DRY" ]; then MODE="--out local-cache/distill/dry"; else MODE="--enqueue"; fi
if python -X utf8 tools/distill_day.py --url "$TRICKLE" --ollama-url "http://127.0.0.1:$TUNNEL_PORT" $WILD $MODE >>"$LOG" 2>&1; then
  if [ -n "$DRY" ]; then log "today's set made (dry: not queued)"; else log "today's set made and queued"; fi
  if [ -z "$DRY" ]; then
    git add sigma_ground/mcp/benchmark/results/distill/*.set.json 2>>"$LOG"
    git diff --cached --quiet || { git -c user.name="distill daily" -c user.email="distill@skippy-pc" commit -q -m "Distillation: the set asked on $(date +%Y-%m-%d)" >>"$LOG" 2>&1 && git push -q origin master >>"$LOG" 2>&1 && log "set committed and pushed" || log "set commit/push failed"; }
  fi
  RC=0
else
  log "making today's set failed"
  RC=3
fi
[ -n "$TUNNEL_PID" ] && kill "$TUNNEL_PID" 2>/dev/null
log "== distill daily end rc=$RC"
exit $RC
