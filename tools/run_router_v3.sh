#!/usr/bin/env bash
# Mentat router v3, end to end and unattended (Git Bash on the Windows GPU box; training and
# export run in WSL2 Ubuntu). Plan: misc/ROUTER_V3_PLAN.md. Same stages as run_router_v2.sh,
# generalised in four ways:
#   SIZE      one pipeline per base model (8b 4b 1.7b), same data and hyperparameters, so the sizes compare
#   v3 data   + hidden-tool, lookup ("$1.field"), contrast, drill and decline examples (misc/MENTAT_TWEAKS_AUDIT.md)
#   tweaks    every benchmark run states its tweak profile; a profile runs only if the data teaches
#             what that profile removes (tools/tweak_pairs.py), so a rule is never both removed and untaught
#   GPUs      TRAIN_GPU / SERVE_GPU pin the stages to a card (CUDA_VISIBLE_DEVICES) when there are two
#
# Shared stages (once, state in $ROOT/state):  recipes_v3 declines dataset hardval pairs ablate_v2
# Per-size stages (state in $ROOT/$SIZE/state): train_pilot train merge gguf create smoke eval_val
#                                               eval_hard baseline_stock benchmark score
#
#   SIZE=8b tools/run_router_v3.sh --dry-run          # preflight only, no GPU
#   SIZE=8b DOCKER_OLLAMA=skippytel-ollama tools/run_router_v3.sh
#   tools/run_router_sweep.sh                         # 8b, 4b, 1.7b in turn
# Knobs: SIZE (8b) PER_RECIPE (70) VARIANTS (3) TEACHER (qwen2.5:14b) MAX_TRAIN_H (6) GPU_BUSY_MIB (4000)
# GPU_WAIT_MIN (45) TRAIN_GPU SERVE_GPU PORT (11435) DOCKER_OLLAMA UNLOAD_MAIN ABLATE_V2 (1)
# BENCH_PROFILES ("legacy trained model-only") BENCH_PAR (3)
set -euo pipefail
export PYTHONIOENCODING=utf-8      # Windows consoles are cp1252; questions hold Ω, µ, °

FAMILY=v3
PLAN=v3
SIZE="${SIZE:-8b}"
case "$SIZE" in
  8b)   BASE=Qwen3-8B;   STOCK=qwen3:8b ;;
  4b)   BASE=Qwen3-4B;   STOCK=qwen3:4b ;;
  1.7b) BASE=Qwen3-1.7B; STOCK=qwen3:1.7b ;;
  *) echo "SIZE must be 8b, 4b or 1.7b (got '$SIZE')" >&2; exit 2 ;;
esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO/.venv/Scripts/python"
ROOT="${ROUTER_ROOT:-$REPO/local-cache/router_$FAMILY}"
DSTATE="$ROOT/state"                    # shared stages
CACHE="$ROOT/$SIZE"
STATE="$CACHE/state"                    # per-size stages
LOGS="$CACHE/logs"
V1DATA="${V1DATA:-$REPO/local-cache/router_dataset}"
V2DATA="$REPO/local-cache/router_dataset_v2"
RECIPES_V2="$REPO/local-cache/router_recipes"
RECIPES_V3="$REPO/local-cache/router_recipes_v3"
DECLINES="$REPO/local-cache/router_declines"
DATA="$REPO/local-cache/router_dataset_$FAMILY"
HARD="$REPO/local-cache/router_hardval_$FAMILY"
RESULTS="$REPO/sigma_ground/mcp/benchmark/results"
PER_RECIPE="${PER_RECIPE:-70}"
VARIANTS="${VARIANTS:-3}"
TEACHER="${TEACHER:-qwen2.5:14b}"
UNLOAD_MAIN="${UNLOAD_MAIN:-0}"
DOCKER_OLLAMA="${DOCKER_OLLAMA:-}"
MAX_TRAIN_H="${MAX_TRAIN_H:-6}"
EVAL_LIMIT="${EVAL_LIMIT:-800}"         # validation records used for the loss curve during training
GPU_BUSY_MIB="${GPU_BUSY_MIB:-4000}"
GPU_WAIT_MIN="${GPU_WAIT_MIN:-45}"
TRAIN_GPU="${TRAIN_GPU:-}"              # card index for training and merge; empty = every card
SERVE_GPU="${SERVE_GPU:-}"              # card index for the dedicated Ollama; empty = every card
ABLATE_V2="${ABLATE_V2:-1}"
BENCH_PROFILES="${BENCH_PROFILES:-legacy trained model-only}"
BENCH_PAR="${BENCH_PAR:-3}"
PORT="${PORT:-11435}"
API="http://127.0.0.1:$PORT"
HARNESS_URL="http://localhost:$PORT"        # localhost, as in the earlier runs, so latencies compare
MODEL="mentat-router-$FAMILY-$SIZE"
DISTRO=Ubuntu-24.04
W_VENV='$HOME/venvs/mentat-router'
W_WORK="\$HOME/router_${FAMILY}_$SIZE"
W_BASE="\$HOME/models/$BASE"
W_LLAMA='$HOME/llama.cpp'
DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1

mkdir -p "$DSTATE" "$STATE" "$LOGS"
to_wsl() { echo "$1" | sed -E 's#^/([a-zA-Z])/#/mnt/\L\1/#'; }
W_REPO="$(to_wsl "$REPO")"
W_CACHE="$(to_wsl "$CACHE")"
W_DATA="$(to_wsl "$DATA")"

log() { printf '%s [%s] %s\n' "$(date '+%F %T')" "$SIZE" "$*" | tee -a "$LOGS/pipeline.log"; }
die() { log "FAILED: $*"; exit 1; }

# The CPU is shared: n-body work can hold 18 of its 20 threads. Training is GPU-bound, so cap
# the threads torch and the quantizer take in WSL (OMP/MKL 2, quantize 4). Do NOT lower this
# script's priority: against 18 busy threads a below-normal process is starved.
# CUDA_VISIBLE_DEVICES is set only when a card was named.
wsl_cuda() { [[ -n "$TRAIN_GPU" ]] && echo "export CUDA_VISIBLE_DEVICES=$TRAIN_GPU;" || true; }
wsl_run() {
  MSYS_NO_PATHCONV=1 wsl.exe -d "$DISTRO" -- bash -lc "export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false; $(wsl_cuda) source $W_VENV/bin/activate && cd '$W_REPO' && $1" 2>&1 | tr -d '\0'
  return "${PIPESTATUS[0]}"
}

# ── dedicated Ollama ──────────────────────────────────────────────────────
OLLAMA_PID=""
ollama_up() {
  curl -s -m 3 "$API/api/version" >/dev/null && return 0
  # CUDA_VISIBLE_DEVICES only when a card was named: an empty value would hide every card.
  env ${SERVE_GPU:+CUDA_VISIBLE_DEVICES=$SERVE_GPU} OLLAMA_HOST="127.0.0.1:$PORT" OLLAMA_NUM_PARALLEL=1 \
    OLLAMA_KEEP_ALIVE=30m ollama serve >> "$LOGS/ollama_$PORT.log" 2>&1 &
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

gpu_used_mib() {   # the card being waited for, or the busiest one
  local idx="${TRAIN_GPU:-$SERVE_GPU}"
  if [[ -n "$idx" ]]; then nvidia-smi -i "$idx" --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' \r'
  else nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1 | tr -d ' \r'; fi
}
gpu_guard() {   # gpu_guard <stage> exclusive|shared
  local f="$LOGS/gpu_$1.txt"
  { date; nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader
    echo "main 11434:"; ollama ps; echo "dedicated $PORT:"; OLLAMA_HOST="127.0.0.1:$PORT" ollama ps 2>/dev/null
    tasklist | grep -i -E "gmod|hl2" || true; } > "$f" 2>&1
  if [[ "$UNLOAD_MAIN" == 1 ]]; then unload_all "127.0.0.1:11434"; fi
  if [[ "$2" == exclusive ]]; then
    docker_ollama_stop
    unload_all "127.0.0.1:$PORT"
    reap_orphans
    local deadline=$(( $(date +%s) + GPU_WAIT_MIN * 60 )) used
    while :; do
      used=$(gpu_used_mib)
      [[ "$used" -le "$GPU_BUSY_MIB" ]] && break
      [[ $(date +%s) -ge $deadline ]] && die "GPU still holds ${used} MiB after ${GPU_WAIT_MIN} min (limit ${GPU_BUSY_MIB}): close GMod and other Ollama clients, then rerun"
      log "  waiting for the GPU: ${used} MiB in use (limit ${GPU_BUSY_MIB}); main Ollama: $(ollama ps 2>/dev/null | awk 'NR>1 {printf "%s ", $1}')"
      sleep 30
    done
  fi
  return 0
}

stage_in() {   # stage_in <state dir> <name> <function>
  local sd=$1 name=$2 fn=$3
  if [[ -f "$sd/$name.done" ]]; then log "skip $name (done)"; return 0; fi
  log "start $name"
  local t0; t0=$(date +%s)
  "$fn" || die "$name (exit $?); see $LOGS/$name.log"
  local dt=$(( $(date +%s) - t0 ))
  echo "$dt" > "$sd/$name.done"
  log "done  $name in $((dt / 60)) min $((dt % 60)) s"
}
shared() { stage_in "$DSTATE" "$1" "$2"; }
stage()  { stage_in "$STATE" "$1" "$2"; }

# ── stages ────────────────────────────────────────────────────────────────
preflight() {
  local ok=1
  check() { if eval "$2" >/dev/null 2>&1; then log "  ok   $1"; else log "  MISSING $1"; ok=0; fi; }
  log "preflight (no GPU work): size $SIZE ($BASE, stock $STOCK), repo $REPO, cache $CACHE"
  log "  branch $(git -C "$REPO" branch --show-current), head $(git -C "$REPO" rev-parse --short HEAD)"
  check "repo venv + v3 recipes, declines, tweaks" \
    "'$PY' -c 'import httpx, sigma_ground.mcp.benchmark.router_recipes_v3, sigma_ground.mcp.benchmark.router_refusals, sigma_ground.mcp.benchmark.tweaks'"
  check "v1 dataset (train, val)" "test -s '$V1DATA/train.jsonl' && test -s '$V1DATA/val.jsonl'"
  check "v2 seeds + teacher-rewrite cache (reused: v2 seeds stay first)" \
    "test -s '$RECIPES_V2/seeds.jsonl' && test -s '$V2DATA/rewrites.jsonl'"
  check "hard validation spec" "test -s '$REPO/sigma_ground/mcp/benchmark/router_hardval.json'"
  check "baseline runs for the comparison" \
    "test -s '$RESULTS/chain_qwen3_8b_run.json' && test -s '$RESULTS/router_qwen3_8b_stock_free_run.json' && test -s '$RESULTS/router_mentat_router_v2_free_run.json'"
  check "ollama models: $TEACHER, $STOCK, nomic-embed-text, mentat-router-v2" \
    "ollama list | grep -q '^$TEACHER ' && ollama list | grep -q '^$STOCK ' && ollama list | grep -q '^nomic-embed-text' && ollama list | grep -q '^mentat-router-v2'"
  check "WSL training venv (torch, transformers, peft, bitsandbytes, unsloth)" \
    "wsl_run 'python -c \"import torch, transformers, peft, bitsandbytes, accelerate\" && pip show -q unsloth'"
  check "WSL base model $BASE (config, tokenizer, weights)" \
    "wsl_run 'test -s $W_BASE/config.json && test -s $W_BASE/tokenizer.json && ls $W_BASE/*.safetensors'"
  check "WSL llama.cpp converter + quantizer" \
    "wsl_run 'test -f $W_LLAMA/convert_hf_to_gguf.py && test -x $W_LLAMA/build/bin/llama-quantize'"
  check "WSL GPU visible (nvidia-smi only)" "MSYS_NO_PATHCONV=1 wsl.exe -d $DISTRO -- nvidia-smi -L"
  check "WSL disk >= 60 GB free" "wsl_run 'test \$(df -BG --output=avail \$HOME | tail -1 | tr -dc 0-9) -ge 60'"
  check "D: >= 20 GB free" "test \$(df -BG --output=avail '$CACHE' | tail -1 | tr -dc 0-9) -ge 20"
  log "  settings: PER_RECIPE=$PER_RECIPE VARIANTS=$VARIANTS TEACHER=$TEACHER DOCKER_OLLAMA=${DOCKER_OLLAMA:-none} TRAIN_GPU=${TRAIN_GPU:-all} SERVE_GPU=${SERVE_GPU:-all} PORT=$PORT"
  log "  benchmark profiles: $BENCH_PROFILES (each runs only if its tweaks are paired with training data)"
  log "  shared stages done: $(cd "$DSTATE" && ls -- *.done 2>/dev/null | tr '\n' ' ')"
  log "  $SIZE stages done: $(cd "$STATE" && ls -- *.done 2>/dev/null | tr '\n' ' ')"
  [[ $ok == 1 ]] || die "preflight"
}

# ── shared: the data ──
do_recipes_v3() {   # CPU only; the v2 seeds are kept as they are
  "$PY" -u -m sigma_ground.mcp.benchmark.router_recipes_v3 --per-recipe "$PER_RECIPE" --out "$RECIPES_V3" \
    > "$LOGS/recipes_v3.log" 2>&1
  [[ -s "$RECIPES_V3/seeds.jsonl" ]]
}
do_declines() {   # CPU only
  "$PY" -u -m sigma_ground.mcp.benchmark.router_refusals --out "$DECLINES" > "$LOGS/declines.log" 2>&1
  [[ -s "$DECLINES/seeds.jsonl" ]]
}
do_dataset() {
  ollama_up
  gpu_guard dataset exclusive
  mkdir -p "$DATA"
  # Seeds are read v2, v3 recipes, declines -- in that order -- so the v2 teacher rewrites (indexed by
  # seed) stay valid: copy them in. build_router_v2.py checks every cached rewrite against its seed.
  [[ -s "$DATA/rewrites.jsonl" ]] || cp "$V2DATA/rewrites.jsonl" "$DATA/rewrites.jsonl"
  "$PY" -u "$REPO/tools/build_router_v2.py" --plan "$PLAN" --v1 "$V1DATA" \
    --recipes "$RECIPES_V2" --recipes "$RECIPES_V3" --recipes "$DECLINES" --out "$DATA" \
    --teacher-model "$TEACHER" --teacher-url "$API" --variants "$VARIANTS" > "$LOGS/dataset.log" 2>&1
  [[ -s "$DATA/manifest.json" ]] || return 1
  OLLAMA_HOST="127.0.0.1:$PORT" ollama stop "$TEACHER" >/dev/null 2>&1 || true
}
do_hardval() {   # CPU only; the verified results of the v2 run are reused
  "$PY" -u "$REPO/tools/build_router_hardval.py" --plan "$PLAN" --train "$DATA/train.jsonl" --out "$HARD" \
    --cache "$REPO/local-cache/router_hardval/verified.json" > "$LOGS/hardval.log" 2>&1
  [[ -s "$HARD/hard.jsonl" ]]
}
do_pairs() {   # CPU only: which tweak profiles the data supports
  local p rc any=0
  for p in trained model-only; do
    rc=0
    "$PY" "$REPO/tools/tweak_pairs.py" --data "$DATA" --profile "$p" --json "$ROOT/pairs_$p.json" \
      > "$ROOT/pairs_$p.txt" 2>&1 || rc=$?
    cat "$ROOT/pairs_$p.txt" | tee -a "$LOGS/pipeline.log"
    if (( rc == 0 )); then : > "$ROOT/pairs_$p.ok"; else rm -f "$ROOT/pairs_$p.ok"; fi
  done
  return 0       # an unpaired profile is skipped later, loudly; it does not stop the training
}
profile_ok() {   # profile_ok <profile>
  [[ "$1" == legacy ]] && return 0
  [[ -f "$ROOT/pairs_$1.ok" ]]
}
do_ablate_v2() {   # the v2 model without its cascade: which groups did the cascade really carry?
  [[ "$ABLATE_V2" == 1 ]] || return 0
  ollama_up
  gpu_guard ablate_v2 shared
  local out="$RESULTS/router_mentat_router_v2_free_model-only_run.json"
  "$PY" -u -m sigma_ground.mcp.benchmark.run_sigma_ground --ollama-url "$HARNESS_URL" --router --router-plan v2 \
    --router-format free --model mentat-router-v2 --tweaks model-only --output "$out" > "$LOGS/ablate_v2.log" 2>&1 || return 1
  "$PY" -m sigma_ground.mcp.benchmark.score >> "$LOGS/ablate_v2.log" 2>&1 || true
  "$PY" "$REPO/tools/tweak_ablation.py" legacy="$RESULTS/router_mentat_router_v2_free_run.json" \
    model_only="$out" --json "$ROOT/ablation_v2.json" > "$ROOT/ablation_v2.md"
  tee -a "$LOGS/pipeline.log" < "$ROOT/ablation_v2.md"
}

# ── per size: train, export, evaluate ──
train_cmd() {   # train_cmd <out> [extra args]
  local out=$1; shift
  echo "python -u tools/train_router.py train --data '$W_DATA' --base $W_BASE --out $out --eval-limit $EVAL_LIMIT $*"
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
  # A GPU fault ("CUDA error: unknown error") kills the training process but not the checkpoints
  # (every ~330 steps), so retry from the newest one.
  local tries=0 resume rc=1
  while (( tries < 4 )); do
    resume=""
    wsl_run "ls -d $W_WORK/run1/checkpoints/checkpoint-* >/dev/null 2>&1" && resume="--resume"
    if wsl_run "$(train_cmd "$W_WORK/run1" "$(cat "$STATE/backend_flag" 2>/dev/null)" $resume)" >> "$LOGS/train.log"; then
      rc=0; break
    fi
    tries=$((tries + 1))
    log "  training stopped (attempt $tries of 4)"
    (( tries < 4 )) && { log "  resuming from the newest checkpoint in 90 s"; sleep 90; }
  done
  (( rc == 0 )) || return 1
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
  "$PY" "$REPO/tools/eval_router_val.py" --smoke --model "$MODEL" --ollama-url "$API" --plan "$PLAN" \
    --format toolfirst --server-log "$LOGS/ollama_$PORT.log" > "$LOGS/smoke.log" 2>&1
}
eval_both() {   # eval_both <data dir> <split> <name>
  local f fmt
  for f in toolfirst free; do
    fmt=$f; [[ $f == free ]] && fmt=none
    [[ -s "$CACHE/eval_${3}_${MODEL}_$f.json" ]] && { log "  eval_$3 $f already done"; continue; }
    "$PY" -u "$REPO/tools/eval_router_val.py" --model "$MODEL" --ollama-url "$API" --data "$1" --split "$2" \
      --plan "$PLAN" --format "$fmt" --out "$CACHE/eval_${3}_${MODEL}_$f.json" > "$LOGS/eval_${3}_$f.log" 2>&1 || return 1
  done
}
# The tuned model is small, so evaluation and the benchmark share the card with Skippy's Ollama
# ("shared": no container stop, no wait for the GPU to empty). Latencies are then indicative only.
do_eval_val() { ollama_up; gpu_guard eval_val shared; eval_both "$DATA" val val; }
do_eval_hard() { ollama_up; gpu_guard eval_hard shared; eval_both "$HARD" hard hard; }

# Benchmark runs. A run is: <label> <ollama model> <plan> <format> <profile> [--router-index]. Output
# goes to $RESULTS/router_<label>_<profile>_run.json. At most BENCH_PAR run at once (each starts its own
# Mentat server, whose first call costs minutes of CPU on a busy machine; the model is small).
bench_args=()
bench_pids=()
bench_fail=0
bench_launch() {   # bench_launch <label> <model> <plan> <format> <profile> [extra harness args]
  local label=$1 model=$2 plan=$3 fmt=$4 profile=$5; shift 5
  local out="$RESULTS/router_${label}_${profile}_run.json"
  if [[ -s "$out" ]]; then log "  benchmark $label / $profile already done"; return 0; fi
  log "  benchmark $label / $profile ($model, plan $plan, $fmt)"
  "$PY" -u -m sigma_ground.mcp.benchmark.run_sigma_ground --ollama-url "$HARNESS_URL" --router \
    --router-plan "$plan" --router-format "$fmt" --model "$model" --tweaks "$profile" "$@" \
    --output "$out" > "$LOGS/benchmark_${label}_${profile}.log" 2>&1 &
  bench_pids+=($!)
  if (( ${#bench_pids[@]} >= BENCH_PAR )); then
    wait "${bench_pids[0]}" || bench_fail=1
    bench_pids=("${bench_pids[@]:1}")
  fi
}
bench_wait() {
  local p
  for p in "${bench_pids[@]}"; do wait "$p" || bench_fail=1; done
  bench_pids=()
  return "$bench_fail"
}
do_baseline_stock() {   # the base model with no training, in the same profiles, for a fair comparison
  ollama_up
  gpu_guard baseline_stock shared
  local p
  for p in $BENCH_PROFILES; do
    profile_ok "$p" || { log "  SKIPPED stock / $p: its tweaks are not paired with training data (see $ROOT/pairs_$p.txt)"; continue; }
    bench_launch "stock_${SIZE}_free" "$STOCK" v1 free "$p" --router-index
  done
  bench_wait
}
do_benchmark() {
  ollama_up
  gpu_guard benchmark shared
  local p
  for p in $BENCH_PROFILES; do
    if ! profile_ok "$p"; then
      log "  SKIPPED profile $p: the training data does not teach what it removes (see $ROOT/pairs_$p.txt)"
      continue
    fi
    bench_launch "${MODEL//[:.]/_}_free" "$MODEL" "$PLAN" free "$p"
    [[ "$p" == trained ]] && bench_launch "${MODEL//[:.]/_}_toolfirst" "$MODEL" "$PLAN" toolfirst "$p"
  done
  bench_wait
}
do_score() {
  "$PY" -m sigma_ground.mcp.benchmark.score > "$LOGS/score.log" 2>&1 || return 1
  local args=(chain="$RESULTS/chain_qwen3_8b_run.json" stock_v1_8b="$RESULTS/router_qwen3_8b_stock_free_run.json"
              tuned_v2="$RESULTS/router_mentat_router_v2_free_run.json") p r label
  for p in $BENCH_PROFILES; do
    for r in "stock_${SIZE}_free:stock_$SIZE" "${MODEL//[:.]/_}_free:tuned_${SIZE}" "${MODEL//[:.]/_}_toolfirst:tuned_${SIZE}_toolfirst"; do
      label="${r#*:}_$p"
      [[ -s "$RESULTS/router_${r%%:*}_${p}_run.json" ]] && args+=("$label=$RESULTS/router_${r%%:*}_${p}_run.json")
    done
  done
  "$PY" "$REPO/tools/compare_router_runs.py" "${args[@]}" > "$CACHE/comparison.md"
  local l="$RESULTS/router_${MODEL//[:.]/_}_free_legacy_run.json" m="$RESULTS/router_${MODEL//[:.]/_}_free_model-only_run.json"
  if [[ -s "$l" && -s "$m" ]]; then
    "$PY" "$REPO/tools/tweak_ablation.py" legacy="$l" model_only="$m" --json "$CACHE/ablation.json" > "$CACHE/ablation.md"
    cat "$CACHE/ablation.md" >> "$CACHE/comparison.md"
  fi
  tee -a "$LOGS/pipeline.log" < "$CACHE/comparison.md"
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
shared recipes_v3   do_recipes_v3
shared declines     do_declines
shared dataset      do_dataset
shared hardval      do_hardval
shared pairs        do_pairs
shared ablate_v2    do_ablate_v2
stage  train_pilot  do_train_pilot
stage  train        do_train
stage  merge        do_merge
stage  gguf         do_gguf
stage  create       do_create
stage  smoke        do_smoke
stage  eval_val     do_eval_val
stage  eval_hard    do_eval_hard
stage  baseline_stock do_baseline_stock
stage  benchmark    do_benchmark
stage  score        do_score
log "router $FAMILY/$SIZE complete; stage seconds: $(cd "$STATE" && for f in *.done; do printf '%s=%s ' "${f%.done}" "$(cat "$f")"; done)"
