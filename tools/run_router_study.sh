#!/usr/bin/env bash
# Mentat router fine-tune study, end to end and unattended (Git Bash on the
# Windows GPU box; training and export run in WSL2 Ubuntu).
#
#   dataset      router_dataset.py, --per-tool $PER_TOOL --variants $VARIANTS,
#                teacher $TEACHER on a dedicated single-slot Ollama
#   train_pilot  50 QLoRA steps: VRAM peak, steps/s, projected time. If the
#                Unsloth backend fails, the pilot and the full run fall back
#                to transformers+peft
#   train        the full run (2 epochs, recipe fixed in tools/train_router.py)
#   merge        LoRA folded into the bf16 base, shard by shard, on CPU
#   gguf         llama.cpp convert_hf_to_gguf (bf16) + llama-quantize Q4_K_M,
#                Modelfile (non-thinking Qwen template, temperature 0, ctx 4096)
#   create       ollama create mentat-router
#   smoke        the Schwarzschild plan, plus a check that Ollama renders the
#                exact prompt the model was trained on
#   eval_val     tools/eval_router_val.py over val.jsonl
#   benchmark    run_sigma_ground --router --model mentat-router (no index)
#   score        score.py, then tools/compare_router_runs.py: chain vs stock vs tuned
#
# Each finished stage leaves $STATE/<stage>.done; a rerun skips it, so after a
# failure fix the cause and run the script again. Delete a marker to redo
# that stage. Logs and stage times go to $LOGS.
#
#   tools/run_router_study.sh --dry-run     # preflight only, touches no GPU
#   tools/run_router_study.sh               # the whole study
#
# Environment knobs: PER_TOOL (40), VARIANTS (3), TEACHER (qwen2.5:14b),
# UNLOAD_MAIN=1 to unload whatever the main Ollama (11434) holds before each
# GPU stage, ROUTER_CACHE (local-cache/router). Exclusive stages (training,
# smoke, eval, benchmark) wait up to GPU_WAIT_MIN (45) minutes for the GPU to
# drop under GPU_BUSY_MIB (4000) in use, then stop with a message.
#
# Time: the teacher pass measured 0.34 s per rewrite attempt (pilot, 2,016
# attempts), so the defaults (about 24k attempts) take about 2.3 h before
# training starts. PER_TOOL=20 VARIANTS=2 is about 45 min. Training time is
# not measured yet; the train_pilot stage reports it.
#
# DOCKER_OLLAMA=skippytel-ollama stops that container before the first GPU stage
# and starts it again when the script exits (success or failure). MAX_TRAIN_H (6):
# the pilot's projected full-run time above this stops the study before training.
#
# Why a second Ollama on 11435: the main server runs OLLAMA_NUM_PARALLEL=4,
# and four 32k-token slots don't fit beside the model, which then runs half on
# CPU. The harness gets http://localhost:11435 because both baselines were run
# that way; anything else uses 127.0.0.1, since a new connection to
# "localhost" costs ~2 s on this machine.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO/.venv/Scripts/python"
CACHE="${ROUTER_CACHE:-$REPO/local-cache/router}"
STATE="$CACHE/state"
LOGS="$CACHE/logs"
DATA="$REPO/local-cache/router_dataset"
RESULTS="$REPO/sigma_ground/mcp/benchmark/results"
PER_TOOL="${PER_TOOL:-40}"
VARIANTS="${VARIANTS:-3}"
TEACHER="${TEACHER:-qwen2.5:14b}"
UNLOAD_MAIN="${UNLOAD_MAIN:-0}"
DOCKER_OLLAMA="${DOCKER_OLLAMA:-}"     # e.g. skippytel-ollama: stopped for the run, restarted after
MAX_TRAIN_H="${MAX_TRAIN_H:-6}"        # stop after the pilot if the full run is projected longer
GPU_BUSY_MIB="${GPU_BUSY_MIB:-4000}"   # MiB others may hold when an exclusive stage starts
GPU_WAIT_MIN="${GPU_WAIT_MIN:-45}"     # how long to wait for that before giving up
PORT=11435
API="http://127.0.0.1:$PORT"
HARNESS_URL="http://localhost:$PORT"
MODEL=mentat-router
DISTRO=Ubuntu-24.04
W_VENV='$HOME/venvs/mentat-router'
W_WORK='$HOME/router'
W_BASE='$HOME/models/Qwen3-8B'
W_LLAMA='$HOME/llama.cpp'
DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1

mkdir -p "$STATE" "$LOGS"
to_wsl() { echo "$1" | sed -E 's#^/([a-zA-Z])/#/mnt/\L\1/#'; }
W_REPO="$(to_wsl "$REPO")"
W_CACHE="$(to_wsl "$CACHE")"
W_DATA="$(to_wsl "$DATA")"

log() { printf '%s %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOGS/pipeline.log"; }
die() { log "FAILED: $*"; exit 1; }

# run a command inside WSL, in the training venv, from the repo
wsl_run() {
  MSYS_NO_PATHCONV=1 wsl.exe -d "$DISTRO" -- bash -lc "source $W_VENV/bin/activate && cd '$W_REPO' && $1" 2>&1 | tr -d '\0'
  return "${PIPESTATUS[0]}"
}

# ── dedicated Ollama ──────────────────────────────────────────────────────
OLLAMA_PID=""
ollama_up() {
  curl -s -m 3 "$API/api/version" >/dev/null && return 0
  OLLAMA_HOST="127.0.0.1:$PORT" OLLAMA_NUM_PARALLEL=1 OLLAMA_KEEP_ALIVE=30m \
    ollama serve >> "$LOGS/ollama_$PORT.log" 2>&1 &
  OLLAMA_PID=$!
  for _ in $(seq 1 60); do curl -s -m 2 "$API/api/version" >/dev/null && return 0; sleep 1; done
  die "Ollama on $PORT did not come up"
}
unload_all() {   # $1 = host:port
  local m
  for m in $(OLLAMA_HOST="$1" ollama ps 2>/dev/null | awk 'NR>1 {print $1}'); do
    OLLAMA_HOST="$1" ollama stop "$m" >/dev/null 2>&1 || true
  done
}
ollama_down() {
  curl -s -m 3 "$API/api/version" >/dev/null || return 0
  unload_all "127.0.0.1:$PORT"
  [[ -n "$OLLAMA_PID" ]] && kill "$OLLAMA_PID" 2>/dev/null || true
}
# A Docker Ollama (skippytel-ollama) keeps llava and qwen3:8b resident forever, about
# 6 GB of the 16 GB. With DOCKER_OLLAMA=<container> the script stops it before the
# first GPU stage and starts it again when the script ends, however it ends.
docker_ollama_stop() {
  [[ -n "$DOCKER_OLLAMA" ]] || return 0
  if [[ "$(docker inspect -f '{{.State.Running}}' "$DOCKER_OLLAMA" 2>/dev/null)" == true ]]; then
    log "  stopping Docker container $DOCKER_OLLAMA (restarted when the script ends)"
    docker stop "$DOCKER_OLLAMA" >/dev/null || die "could not stop $DOCKER_OLLAMA"
  fi
}
docker_ollama_restore() {
  [[ -n "$DOCKER_OLLAMA" ]] || return 0
  if [[ "$(docker inspect -f '{{.State.Running}}' "$DOCKER_OLLAMA" 2>/dev/null)" == false ]]; then
    docker start "$DOCKER_OLLAMA" >/dev/null 2>&1 && log "restarted Docker container $DOCKER_OLLAMA"
  fi
}
cleanup() { rm -f "$STATE/keepawake"; ollama_down; docker_ollama_restore; }
trap cleanup EXIT

# Before a GPU stage: record who holds the GPU; free it as allowed.
gpu_guard() {
  local f="$LOGS/gpu_$1.txt"
  { date; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
    echo "main 11434:"; ollama ps; echo "dedicated $PORT:"; OLLAMA_HOST="127.0.0.1:$PORT" ollama ps 2>/dev/null
    tasklist | grep -i -E "gmod|hl2" || true; } > "$f" 2>&1
  if [[ "$UNLOAD_MAIN" == 1 ]]; then unload_all "127.0.0.1:11434"; fi
  if [[ "$2" == exclusive ]]; then
    docker_ollama_stop
    unload_all "127.0.0.1:$PORT"
    # QLoRA on Qwen3-8B needs most of the 16 GB: wait for GMod and other
    # clients to let go rather than train into an out-of-memory error.
    local deadline=$(( $(date +%s) + GPU_WAIT_MIN * 60 )) used
    while :; do
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' \r')
      [[ "$used" -le "$GPU_BUSY_MIB" ]] && break
      [[ $(date +%s) -ge $deadline ]] && die "GPU still holds ${used} MiB after ${GPU_WAIT_MIN} min (limit ${GPU_BUSY_MIB}): close GMod and other Ollama clients, then rerun"
      log "  waiting for the GPU: ${used} MiB in use (limit ${GPU_BUSY_MIB}); main Ollama: $(ollama ps 2>/dev/null | awk 'NR>1 {printf "%s ", $1}')"
      sleep 30
    done
  fi
  local held
  held=$(ollama ps 2>/dev/null | awk 'NR>1 && $1 !~ /nomic-embed/ {print $1}' | tr '\n' ' ')
  [[ -n "$held" ]] && log "  note: main Ollama holds: $held(GPU shared; see $f)"
  return 0
}

stage() {   # stage <name> <function>
  local name=$1 fn=$2
  if [[ -f "$STATE/$name.done" ]]; then log "skip $name (done)"; return 0; fi
  log "start $name"
  local t0; t0=$(date +%s)
  "$fn" || die "$name (exit $?); see $LOGS/$name.log"
  local dt=$(( $(date +%s) - t0 ))
  echo "$dt" > "$STATE/$name.done"
  log "done  $name in $((dt / 60)) min $((dt % 60)) s"
}

# ── stages ────────────────────────────────────────────────────────────────
preflight() {
  local ok=1
  check() { if eval "$2" >/dev/null 2>&1; then log "  ok   $1"; else log "  MISSING $1"; ok=0; fi; }
  log "preflight (no GPU work): repo $REPO, cache $CACHE"
  log "  branch $(git -C "$REPO" branch --show-current), head $(git -C "$REPO" rev-parse --short HEAD)"
  check "repo venv + sigma_ground" "'$PY' -c 'import sigma_ground.mcp.benchmark.router_dataset, httpx'"
  check "baseline chain run" "test -s '$RESULTS/chain_qwen3_8b_run.json'"
  check "baseline stock router run" "test -s '$RESULTS/router_qwen3_8b_stock_run.json'"
  check "ollama models: $TEACHER, qwen3:8b, nomic-embed-text" \
    "ollama list | grep -q '^$TEACHER ' && ollama list | grep -q '^qwen3:8b ' && ollama list | grep -q '^nomic-embed-text'"
  check "WSL training venv (torch, transformers, peft, bitsandbytes, unsloth)" \
    "wsl_run 'python -c \"import torch, transformers, peft, bitsandbytes, accelerate\" && pip show -q unsloth'"
  check "WSL base model (5 shards on ext4)" "wsl_run 'test \$(ls $W_BASE/model-0000*-of-00005.safetensors | wc -l) -eq 5'"
  check "WSL llama.cpp converter + quantizer" \
    "wsl_run 'test -f $W_LLAMA/convert_hf_to_gguf.py && test -x $W_LLAMA/build/bin/llama-quantize'"
  check "WSL GPU visible (nvidia-smi only)" "MSYS_NO_PATHCONV=1 wsl.exe -d $DISTRO -- nvidia-smi -L"
  check "WSL disk >= 60 GB free" "wsl_run 'test \$(df -BG --output=avail \$HOME | tail -1 | tr -dc 0-9) -ge 60'"
  check "D: >= 20 GB free" "test \$(df -BG --output=avail '$CACHE' | tail -1 | tr -dc 0-9) -ge 20"
  log "  settings: PER_TOOL=$PER_TOOL VARIANTS=$VARIANTS TEACHER=$TEACHER UNLOAD_MAIN=$UNLOAD_MAIN"
  log "  main Ollama now: $(ollama ps 2>/dev/null | awk 'NR>1 {printf "%s ", $1}')"
  log "  stages done: $(cd "$STATE" && ls -- *.done 2>/dev/null | tr '\n' ' ')"
  [[ $ok == 1 ]] || die "preflight"
}

do_dataset() {
  ollama_up
  gpu_guard dataset exclusive
  "$PY" -u -m sigma_ground.mcp.benchmark.router_dataset --per-tool "$PER_TOOL" \
      --variants "$VARIANTS" --teacher-model "$TEACHER" --teacher-url "$API" --out "$DATA" 2>&1 \
    | while IFS= read -r l; do printf '%s %s\n' "$(date +%s)" "$l"; done > "$LOGS/dataset.log"
  [[ ${PIPESTATUS[0]} == 0 && -s "$DATA/manifest.json" ]] || return 1
  OLLAMA_HOST="127.0.0.1:$PORT" ollama stop "$TEACHER" >/dev/null 2>&1 || true
}

train_cmd() {   # train_cmd <out> [extra args]
  local out=$1; shift
  echo "python -u tools/train_router.py train --data '$W_DATA' --base $W_BASE --out $out $*"
}
do_train_pilot() {
  gpu_guard train_pilot exclusive
  if wsl_run "$(train_cmd "$W_WORK/pilot" --max-steps 50 --eval-steps 25)" > "$LOGS/train_pilot.log"; then
    echo "" > "$STATE/backend_flag"
  else
    log "  unsloth pilot failed; retrying with transformers+peft"
    wsl_run "$(train_cmd "$W_WORK/pilot" --max-steps 50 --eval-steps 25 --no-unsloth)" \
      > "$LOGS/train_pilot_peft.log" || return 1
    echo "--no-unsloth" > "$STATE/backend_flag"
  fi
  wsl_run "mkdir -p '$W_CACHE/pilot' && cp $W_WORK/pilot/train_report.json $W_WORK/pilot/log_history.json '$W_CACHE/pilot/'"
  local proj
  proj=$("$PY" -c "import json,sys; print(int(json.load(open(sys.argv[1]))['projected_full_run_s']))" \
         "$CACHE/pilot/train_report.json") || return 1
  log "  pilot projects the full training run at $((proj / 60)) min"
  (( proj <= MAX_TRAIN_H * 3600 )) || { log "  over MAX_TRAIN_H=${MAX_TRAIN_H} h: rerun with a larger MAX_TRAIN_H or a smaller dataset"; return 1; }
}
do_train() {
  gpu_guard train exclusive
  wsl_run "$(train_cmd "$W_WORK/run1" "$(cat "$STATE/backend_flag" 2>/dev/null)")" > "$LOGS/train.log" || return 1
  wsl_run "mkdir -p '$W_CACHE/run1' && cp -r $W_WORK/run1/adapter $W_WORK/run1/train_report.json $W_WORK/run1/log_history.json '$W_CACHE/run1/'"
}
do_merge() {
  wsl_run "python -u tools/train_router.py merge --base $W_BASE --adapter $W_WORK/run1/adapter --out $W_WORK/merged" \
    > "$LOGS/merge.log"
}
do_gguf() {
  wsl_run "set -e; python $W_LLAMA/convert_hf_to_gguf.py $W_WORK/merged --outtype bf16 --outfile $W_WORK/$MODEL-bf16.gguf
           $W_LLAMA/build/bin/llama-quantize $W_WORK/$MODEL-bf16.gguf $W_WORK/$MODEL-Q4_K_M.gguf Q4_K_M
           mkdir -p '$W_CACHE/gguf' && cp $W_WORK/$MODEL-Q4_K_M.gguf '$W_CACHE/gguf/'
           ls -la $W_WORK/*.gguf '$W_CACHE/gguf/'" > "$LOGS/gguf.log" || return 1
  "$PY" "$REPO/tools/train_router.py" modelfile --gguf "./$MODEL-Q4_K_M.gguf" \
    --out "$CACHE/gguf/Modelfile" >> "$LOGS/gguf.log"
}
do_create() {
  ollama_up
  (cd "$CACHE/gguf" && OLLAMA_HOST="127.0.0.1:$PORT" ollama create "$MODEL" -f Modelfile) > "$LOGS/create.log" 2>&1
}
do_smoke() {
  ollama_up
  gpu_guard smoke exclusive
  "$PY" "$REPO/tools/eval_router_val.py" --smoke --model "$MODEL" --ollama-url "$API" \
    --server-log "$LOGS/ollama_$PORT.log" > "$LOGS/smoke.log" 2>&1
}
do_eval_val() {
  ollama_up
  gpu_guard eval_val exclusive
  # Two request shapes, same records: "schema" is the harness's JSON-schema
  # format (Ollama sorts its keys, forcing "args" before "tool"); "free" sends
  # no format, the order the model was trained in. See Session 38.
  local f fmt
  for f in schema free; do
    fmt=$f; [[ $f == free ]] && fmt=none
    "$PY" -u "$REPO/tools/eval_router_val.py" --model "$MODEL" --ollama-url "$API" --data "$DATA" \
      --format "$fmt" --out "$CACHE/eval_val_${MODEL}_$f.json" > "$LOGS/eval_val_$f.log" 2>&1 || return 1
  done
}
do_benchmark() {
  ollama_up
  gpu_guard benchmark exclusive
  local h=(-m sigma_ground.mcp.benchmark.run_sigma_ground --ollama-url "$HARNESS_URL")
  # the tuned router as the harness defines router mode (schema format), then free
  "$PY" -u "${h[@]}" --router --model "$MODEL" \
    --output "$RESULTS/router_mentat_router_run.json" > "$LOGS/benchmark_schema.log" 2>&1 || return 1
  "$PY" -u "${h[@]}" --router --router-format free --model "$MODEL" \
    --output "$RESULTS/router_mentat_router_free_run.json" > "$LOGS/benchmark_free.log" 2>&1 || return 1
  # the stock model in the same free format, so both models meet both formats
  "$PY" -u "${h[@]}" --router --router-index --router-format free --model qwen3:8b \
    --output "$RESULTS/router_qwen3_8b_stock_free_run.json" > "$LOGS/benchmark_stock_free.log" 2>&1 || return 1
}
do_score() {
  "$PY" -m sigma_ground.mcp.benchmark.score > "$LOGS/score.log" 2>&1 || return 1
  "$PY" "$REPO/tools/compare_router_runs.py" \
    chain="$RESULTS/chain_qwen3_8b_run.json" \
    stock_schema="$RESULTS/router_qwen3_8b_stock_run.json" \
    stock_free="$RESULTS/router_qwen3_8b_stock_free_run.json" \
    tuned_schema="$RESULTS/router_mentat_router_run.json" \
    tuned_free="$RESULTS/router_mentat_router_free_run.json" > "$CACHE/comparison.md"
  cat "$CACHE/comparison.md" | tee -a "$LOGS/pipeline.log"
}

# Keep Windows awake while the study runs: a helper process holds the standard
# "system required" execution state (a request that ends with the process, not
# a power setting) until the sentinel file is removed by cleanup.
if [[ $DRY == 0 && ( "${OSTYPE:-}" == msys* || "${OSTYPE:-}" == cygwin* ) ]]; then
  touch "$STATE/keepawake"
  "$PY" -c "import ctypes,os,sys,time; ctypes.windll.kernel32.SetThreadExecutionState(0x80000001); [time.sleep(20) for _ in iter(lambda: os.path.exists(sys.argv[1]), False)]" \
    "$STATE/keepawake" &
fi

stage preflight preflight
rm -f "$STATE/preflight.done"          # always rerun
[[ $DRY == 1 ]] && { log "dry run: stopping after preflight"; exit 0; }
stage dataset     do_dataset
stage train_pilot do_train_pilot
stage train       do_train
stage merge       do_merge
stage gguf        do_gguf
stage create      do_create
stage smoke       do_smoke
stage eval_val    do_eval_val
stage benchmark   do_benchmark
stage score       do_score
log "study complete; stage seconds: $(cd "$STATE" && for f in *.done; do printf '%s=%s ' "${f%.done}" "$(cat "$f")"; done)"
