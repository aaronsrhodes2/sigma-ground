#!/usr/bin/env bash
# Train and evaluate the router at several base-model sizes, same data and settings, so the sizes
# compare. Each size is one run of tools/run_router_v3.sh; the data stages are shared and run once
# (the first size does them).
#
#   tools/run_router_sweep.sh                      # SIZES="1.7b 4b 8b", one after another, one card
#   SIZES="4b 1.7b" tools/run_router_sweep.sh
#   TWO_CARDS=1 tools/run_router_sweep.sh          # two cards: sizes run in pairs, one per card
#
# With TWO_CARDS=1 the first size runs alone (it builds the data and needs the card to itself for the
# teacher), then the remaining sizes run two at a time: card 0 and card 1, each with its own Ollama
# port. Environment passes through (DOCKER_OLLAMA, TEACHER, BENCH_PROFILES, ...). The summary is
# local-cache/router_v3/sweep.md.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
# Smallest first: the whole pipeline (merge, GGUF, evaluation, benchmark) is proven on a run of under
# an hour before the 8b spends about three hours training.
SIZES="${SIZES:-1.7b 4b 8b}"
ROOT="$REPO/local-cache/router_v3"
mkdir -p "$ROOT"
rc=0

run_one() {   # run_one <size> <gpu> <port>
  SIZE="$1" TRAIN_GPU="$2" SERVE_GPU="$2" PORT="$3" "$HERE/run_router_v3.sh" || return $?
}

if [[ "${TWO_CARDS:-0}" == 1 ]]; then
  set -- $SIZES
  first=$1; shift
  run_one "$first" 0 11435 || rc=1
  while (( $# > 0 )); do
    a=$1; shift
    run_one "$a" 0 11435 &
    pa=$!
    if (( $# > 0 )); then
      b=$1; shift
      run_one "$b" 1 11436 &
      pb=$!
      wait "$pb" || rc=1
    fi
    wait "$pa" || rc=1
  done
else
  for s in $SIZES; do
    run_one "$s" "${TRAIN_GPU:-}" "${PORT:-11435}" || { rc=1; echo "size $s failed; continuing with the next" >&2; }
  done
fi

{
  echo "# Router v3 sweep ($(date '+%F %T'))"
  echo
  for s in $SIZES; do
    echo "## $s"
    [[ -s "$ROOT/$s/comparison.md" ]] && cat "$ROOT/$s/comparison.md" || echo "(no comparison: that size did not finish)"
    echo
  done
} > "$ROOT/sweep.md"
echo "sweep summary: $ROOT/sweep.md"
exit $rc
