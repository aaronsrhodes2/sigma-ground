#!/usr/bin/env bash
# Mentat router v2, end to end and unattended (Git Bash on the Windows GPU box;
# training and export run in WSL2 Ubuntu). Plan: misc/ROUTER_V2_PLAN.md.
#
# CPU stages (no GPU):
#   recipes   router_recipes.py: multi-step chains and the weak families, every plan run on the real server
#   (preflight, --dry-run: checks only)
# GPU stages (need the whole GPU; DOCKER_OLLAMA=skippytel-ollama stops that container for the
# run and restarts it when the script ends):
#   dataset      build_router_v2.py: v1 records converted + recipe seeds + teacher rewrites of the seeds only
#   hardval      CPU, after dataset: verify the hand-written hard validation set, measure its similarity to training
#   train_pilot  50 QLoRA steps (Unsloth, PEFT fallback); stops the study if the projection is over MAX_TRAIN_H
#   train        2 epochs from the base model
#   merge, gguf, create    CPU merge, llama.cpp GGUF + Q4_K_M, ollama create $MODEL
#   smoke        the template check and one harness-shaped request
#   eval_val     validation, in the tool-first schema and in free generation
#   eval_hard    the hard validation set, both ways
#   benchmark    run_sigma_ground --router --router-plan v2, tool-first and free
#   score        score.py and tools/compare_router_runs.py: v1 and v2, stock and tuned
#
# Each finished stage leaves $STATE/<stage>.done; a rerun skips it. Delete a marker to redo it.
#   tools/run_router_v2.sh --dry-run
#   DOCKER_OLLAMA=skippytel-ollama tools/run_router_v2.sh
# Knobs: PER_RECIPE (70) VARIANTS (3) TEACHER (qwen2.5:14b) MAX_TRAIN_H (6) GPU_BUSY_MIB (4000)
# GPU_WAIT_MIN (45) UNLOAD_MAIN V1DATA.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO/.venv/Scripts/python"
CACHE="${ROUTER_CACHE:-$REPO/local-cache/router_v2}"
STATE="$CACHE/state"
LOGS="$CACHE/logs"
V1DATA="${V1DATA:-$REPO/local-cache/router_dataset}"
RECIPES="$REPO/local-cache/router_recipes"
DATA="$REPO/local-cache/router_dataset_v2"
HARD="$REPO/local-cache/router_hardval"
RESULTS="$REPO/sigma_ground/mcp/benchmark/results"
PER_RECIPE="${PER_RECIPE:-70}"
VARIANTS="${VARIANTS:-3}"
TEACHER="${TEACHER:-qwen2.5:14b}"
UNLOAD_MAIN="${UNLOAD_MAIN:-0}"
DOCKER_OLLAMA="${DOCKER_OLLAMA:-}"
MAX_TRAIN_H="${MAX_TRAIN_H:-6}"
GPU_BUSY_MIB="${GPU_BUSY_MIB:-4000}"
GPU_WAIT_MIN="${GPU_WAIT_MIN:-45}"
PORT=11435
API="http://127.0.0.1:$PORT"
HARNESS_URL="http://localhost:$PORT"        # localhost, as in the v1 runs, so latencies compare
MODEL=mentat-router-v2
DISTRO=Ubuntu-24.04
W_VENV='$HOME/venvs/mentat-router'
W_WORK='$HOME/router_v2'
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

# The CPU is shared: n-body work can hold 18 of its 20 threads. Training is GPU-bound, so cap
# the threads torch and the quantizer take in WSL (OMP/MKL 2, quantize 4) and leave the rest.
# To also keep the Windows-side jobs out of its way, start this script below normal priority:
#   powershell -Command "$p = Start-Process bash -ArgumentList 'tools/run_router_v2.sh' -PassThru -WindowStyle Hidden; $p.PriorityClass = 'BelowNormal'"
wsl_run() {
  MSYS_NO_PATHCONV=1 wsl.exe -d "$DISTRO" -- bash -lc "export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false; source $W_VENV/bin/activate && cd '$W_REPO' && $1" 2>&1 | tr -d '\0'
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
# A killed Ollama leaves its llama-server runner alive, holding VRAM. Kill runners whose parent is gone.
reap_orphans() {
  powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='llama-server.exe'\" | ForEach-Object { if (-not (Get-Process -Id \$_.ParentProcessId -ErrorAction SilentlyContinue)) { Stop-Process -Id \$_.ProcessId -Force; 'reaped orphan llama-server ' + \$_.ProcessId } }" 2>/dev/null | while read -r l; do log "  $l"; done
}
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
cleanup() { rm -f "$STATE/keepawake"; ollama_down; sleep 3; reap_orphans; docker_ollama_restore; }
trap cleanup EXIT

gpu_guard() {   # gpu_guard <stage> exclusive|shared
  local f="$LOGS/gpu_$1.txt"
  { date; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
    echo "main 11434:"; ollama ps; echo "dedicated $PORT:"; OLLAMA_HOST="127.0.0.1:$PORT" ollama ps 2>/dev/null
    tasklist | grep -i -E "gmod|hl2" || true; } > "$f" 2>&1
  if [[ "$UNLOAD_MAIN" == 1 ]]; then unload_all "127.0.0.1:11434"; fi
  if [[ "$2" == exclusive ]]; then
    docker_ollama_stop
    unload_all "127.0.0.1:$PORT"
    reap_orphans
    local deadline=$(( $(date +%s) + GPU_WAIT_MIN * 60 )) used
    while :; do
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' \r')
      [[ "$used" -le "$GPU_BUSY_MIB" ]] && break
      [[ $(date +%s) -ge $deadline ]] && die "GPU still holds ${used} MiB after ${GPU_WAIT_MIN} min (limit ${GPU_BUSY_MIB}): close GMod and other Ollama clients, then rerun"
      log "  waiting for the GPU: ${used} MiB in use (limit ${GPU_BUSY_MIB}); main Ollama: $(ollama ps 2>/dev/null | awk 'NR>1 {printf "%s ", $1}')"
      sleep 30
    done
  fi
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
  check "repo venv + sigma_ground" "'$PY' -c 'import sigma_ground.mcp.benchmark.router_recipes, httpx'"
  check "v1 dataset (train, val)" "test -s '$V1DATA/train.jsonl' && test -s '$V1DATA/val.jsonl'"
  check "hard validation spec" "test -s '$REPO/sigma_ground/mcp/benchmark/router_hardval.json'"
  check "baseline runs for the comparison" \
    "test -s '$RESULTS/chain_qwen3_8b_run.json' && test -s '$RESULTS/router_qwen3_8b_stock_free_run.json' && test -s '$RESULTS/router_mentat_router_free_run.json'"
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
  log "  settings: PER_RECIPE=$PER_RECIPE VARIANTS=$VARIANTS TEACHER=$TEACHER DOCKER_OLLAMA=${DOCKER_OLLAMA:-none}"
  log "  stages done: $(cd "$STATE" && ls -- *.done 2>/dev/null | tr '\n' ' ')"
  [[ $ok == 1 ]] || die "preflight"
}

do_recipes() {   # CPU only
  "$PY" -u -m sigma_ground.mcp.benchmark.router_recipes --per-recipe "$PER_RECIPE" --out "$RECIPES" \
    > "$LOGS/recipes.log" 2>&1
  [[ -s "$RECIPES/seeds.jsonl" ]]
}

do_dataset() {
  ollama_up
  gpu_guard dataset exclusive
  "$PY" -u "$REPO/tools/build_router_v2.py" --v1 "$V1DATA" --recipes "$RECIPES" --out "$DATA" \
    --teacher-model "$TEACHER" --teacher-url "$API" --variants "$VARIANTS" > "$LOGS/dataset.log" 2>&1
  [[ -s "$DATA/manifest.json" ]] || return 1
  OLLAMA_HOST="127.0.0.1:$PORT" ollama stop "$TEACHER" >/dev/null 2>&1 || true
}

do_hardval() {   # CPU only
  "$PY" -u "$REPO/tools/build_router_hardval.py" --train "$DATA/train.jsonl" --out "$HARD" > "$LOGS/hardval.log" 2>&1
  [[ -s "$HARD/hard.jsonl" ]]
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
           $W_LLAMA/build/bin/llama-quantize $W_WORK/$MODEL-bf16.gguf $W_WORK/$MODEL-Q4_K_M.gguf Q4_K_M 4
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
  "$PY" "$REPO/tools/eval_router_val.py" --smoke --model "$MODEL" --ollama-url "$API" --plan v2 \
    --format toolfirst --server-log "$LOGS/ollama_$PORT.log" > "$LOGS/smoke.log" 2>&1
}
eval_both() {   # eval_both <data dir> <split> <name>
  local f fmt
  for f in toolfirst free; do
    fmt=$f; [[ $f == free ]] && fmt=none
    "$PY" -u "$REPO/tools/eval_router_val.py" --model "$MODEL" --ollama-url "$API" --data "$1" --split "$2" \
      --plan v2 --format "$fmt" --out "$CACHE/eval_${3}_${MODEL}_$f.json" > "$LOGS/eval_${3}_$f.log" 2>&1 || return 1
  done
}
do_eval_val() { ollama_up; gpu_guard eval_val exclusive; eval_both "$DATA" val val; }
do_eval_hard() { ollama_up; gpu_guard eval_hard exclusive; eval_both "$HARD" hard hard; }
do_benchmark() {
  ollama_up
  gpu_guard benchmark exclusive
  local h=(-m sigma_ground.mcp.benchmark.run_sigma_ground --ollama-url "$HARNESS_URL" --router --router-plan v2 --model "$MODEL")
  "$PY" -u "${h[@]}" --router-format toolfirst \
    --output "$RESULTS/router_mentat_router_v2_toolfirst_run.json" > "$LOGS/benchmark_toolfirst.log" 2>&1 || return 1
  "$PY" -u "${h[@]}" --router-format free \
    --output "$RESULTS/router_mentat_router_v2_free_run.json" > "$LOGS/benchmark_free.log" 2>&1 || return 1
}
do_score() {
  "$PY" -m sigma_ground.mcp.benchmark.score > "$LOGS/score.log" 2>&1 || return 1
  "$PY" "$REPO/tools/compare_router_runs.py" \
    chain="$RESULTS/chain_qwen3_8b_run.json" \
    stock_free="$RESULTS/router_qwen3_8b_stock_free_run.json" \
    tuned_v1_free="$RESULTS/router_mentat_router_free_run.json" \
    tuned_v2_toolfirst="$RESULTS/router_mentat_router_v2_toolfirst_run.json" \
    tuned_v2_free="$RESULTS/router_mentat_router_v2_free_run.json" > "$CACHE/comparison.md"
  cat "$CACHE/comparison.md" | tee -a "$LOGS/pipeline.log"
}

# Keep Windows awake while the script runs (a per-process request that ends with it).
if [[ $DRY == 0 && ( "${OSTYPE:-}" == msys* || "${OSTYPE:-}" == cygwin* ) ]]; then
  touch "$STATE/keepawake"
  "$PY" -c "import ctypes,os,sys,time; ctypes.windll.kernel32.SetThreadExecutionState(0x80000001); [time.sleep(20) for _ in iter(lambda: os.path.exists(sys.argv[1]), False)]" \
    "$STATE/keepawake" &
fi

stage preflight preflight
rm -f "$STATE/preflight.done"          # always rerun
[[ $DRY == 1 ]] && { log "dry run: stopping after preflight"; exit 0; }
stage recipes     do_recipes
stage dataset     do_dataset
stage hardval     do_hardval
stage train_pilot do_train_pilot
stage train       do_train
stage merge       do_merge
stage gguf        do_gguf
stage create      do_create
stage smoke       do_smoke
stage eval_val    do_eval_val
stage eval_hard   do_eval_hard
stage benchmark   do_benchmark
stage score       do_score
log "router v2 complete; stage seconds: $(cd "$STATE" && for f in *.done; do printf '%s=%s ' "${f%.done}" "$(cat "$f")"; done)"
