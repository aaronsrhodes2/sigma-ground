#!/usr/bin/env bash
# Train and evaluate a Mentat router on a Linux machine (the Bosgame M5, or WSL as a rehearsal), from the verified data bundle to the evaluated model.
# The Windows+WSL pipeline (tools/run_router_v3.sh) does the same stages; this one has no WSL calls, no teacher model and no Docker, starts from
# tools/data_bundle.py's zip, and takes the precision as a knob. Plan: misc/M5_TRAINING_PLAN.md.
#
#   tools/run_router_linux.sh --dry-run                      # print what would run
#   tools/run_router_linux.sh preflight data                 # check the machine, unpack and re-verify the data
#   SIZE=4b PRECISION=bf16 tools/run_router_linux.sh pilot   # 50 steps: the speed of THIS machine, and the projected full run
#   SIZE=4b PRECISION=bf16 tools/run_router_linux.sh         # every stage that is not done yet
#
# Stages (each writes $WORK/state/<stage>.done, so a rerun continues where it stopped):
#   preflight  python modules, accelerator (CUDA or ROCm), disk, base model, bundle, llama.cpp
#   data       verify the bundle, unpack it into local-cache/, regenerate the dataset on the CPU and compare the hashes
#   pilot      50 training steps; stops when the projected run is over MAX_TRAIN_H
#   train      the full run; resumes from the newest checkpoint if it stops (up to 4 tries)
#   merge      fold the adapter into the 16-bit base (CPU)
#   gguf       llama.cpp: bf16 GGUF, then Q4_K_M; writes the Modelfile
#   create     ollama create
#   smoke      one question through the model
#   eval       validation, hard validation, the held-out math suite and the convention suite
# Knobs (environment): SIZE 4b|8b|1.7b (4b)  PRECISION bf16|nf4 (bf16)  FAMILY v32  PLAN v4  BASE <HF model dir> (~/models/Qwen3-<size>)
#   RUN_NAME  WORK  BUNDLE  LLAMA_CPP (~/llama.cpp)  OLLAMA_URL (http://127.0.0.1:11434)  MAX_TRAIN_H (30)  EXTRA_TRAIN (e.g. --no-unsloth)  PY (python)
#   SKIP_GGUF=1 (train and merge only)  EVAL_LIMIT (800)
set -Eeuo pipefail
trap 'rc=$?; msg="$(date "+%F %T") FAILED in stage ${CURRENT_STAGE:-startup} (exit $rc): ${BASH_COMMAND}"; echo "$msg" >&2; [[ -d "${LOGS:-}" ]] && echo "$msg" >> "$LOGS/pipeline.log"; exit $rc' ERR
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1

FAMILY="${FAMILY:-v32}"
PLAN="${PLAN:-v4}"
SIZE="${SIZE:-4b}"
PRECISION="${PRECISION:-bf16}"
case "$SIZE" in
  8b) BASE_NAME=Qwen3-8B ;; 4b) BASE_NAME=Qwen3-4B ;; 1.7b) BASE_NAME=Qwen3-1.7B ;;
  *) echo "SIZE must be 8b, 4b or 1.7b (got '$SIZE')" >&2; exit 2 ;;
esac
case "$PRECISION" in bf16|nf4) ;; *) echo "PRECISION must be bf16 or nf4 (got '$PRECISION')" >&2; exit 2 ;; esac
PY="${PY:-python}"
BASE="${BASE:-$HOME/models/$BASE_NAME}"
RUN_NAME="${RUN_NAME:-mentat-router-$FAMILY-$SIZE-$PRECISION}"   # not NAME: WSL and some shells set NAME to the hostname
WORK="${WORK:-$REPO/local-cache/linux_runs/$RUN_NAME}"
STATE="$WORK/state"
LOGS="$WORK/logs"
BUNDLE="${BUNDLE:-$(ls -t "$REPO"/local-cache/bundles/router_"$FAMILY"_*.zip 2>/dev/null | head -n 1 || true)}"
DATA="$REPO/local-cache/router_dataset_$FAMILY"
HARD="$REPO/local-cache/router_hardval_$FAMILY"
MATH_SUITE="$REPO/local-cache/router_math_suite.json"
CONV_SUITE="$REPO/local-cache/router_conventions_suite.json"
LLAMA_CPP="${LLAMA_CPP:-$HOME/llama.cpp}"
OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"
MAX_TRAIN_H="${MAX_TRAIN_H:-30}"
EXTRA_TRAIN="${EXTRA_TRAIN:-}"
EVAL_LIMIT="${EVAL_LIMIT:-800}"
SKIP_GGUF="${SKIP_GGUF:-0}"
DRY=0
ALL_STAGES=(preflight data pilot train merge gguf create smoke eval)

log() { echo "$(date '+%F %T') $*" | tee -a "$LOGS/pipeline.log"; }
die() { echo "ERROR: $*" >&2; exit 1; }
run() { if (( DRY )); then echo "  [dry-run] $*"; else "$@"; fi; }
done_marker() { echo "$STATE/$1.done"; }

# ── stages ─────────────────────────────────────────────────────────────────────────────────────────
do_preflight() {
  local missing=() m
  $PY -c "import sigma_ground, httpx" 2>/dev/null || die "sigma_ground or httpx cannot be imported by $PY (pip install -e '.[mcp,benchmark]' in this environment)"
  # every package the project declares for the MCP layer, not a hand-picked few: a missing one does not always fail loudly (the dataset build once
  # changed silently without `periodictable`; the rebuild check below would have caught it, but this says why)
  missing=($($PY - <<'EOF'
import importlib.metadata as md, re, tomllib
extras = tomllib.load(open("pyproject.toml", "rb"))["project"]["optional-dependencies"]
need = set()
for spec in extras["mcp"]:
    name = re.split(r"[<>=!~ ]", spec, maxsplit=1)[0]
    try:
        md.version(name)
    except md.PackageNotFoundError:
        need.add(name)
print(" ".join(sorted(need)))
EOF
))
  (( ${#missing[@]} == 0 )) || die "packages the project declares (extra 'mcp') are missing for $PY: ${missing[*]}  (pip install -e '.[mcp,benchmark]')"
  if [[ " ${STAGE_LIST[*]} " =~ " pilot " || " ${STAGE_LIST[*]} " =~ " train " || " ${STAGE_LIST[*]} " =~ " merge " ]]; then
    for m in torch transformers peft accelerate; do $PY -c "import $m" 2>/dev/null || missing+=("$m"); done
    (( ${#missing[@]} == 0 )) || die "training modules missing for $PY: ${missing[*]}  (PyTorch for this accelerator, then pip install transformers peft accelerate)"
  fi
  log "accelerator: $($PY - <<'EOF'
import torch
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"{p.name}, {p.total_memory / 2**30:.1f} GiB visible to torch, cuda={torch.version.cuda}, hip={getattr(torch.version, 'hip', None)}")
else:
    print("NONE visible to torch (training would run on the CPU)")
EOF
)"
  [[ -n "$BUNDLE" && -f "$BUNDLE" ]] || die "no data bundle (BUNDLE=...; expected local-cache/bundles/router_${FAMILY}_*.zip, made by tools/data_bundle.py)"
  [[ -f "$BASE/tokenizer.json" ]] || die "base model not found at BASE=$BASE (needs the Hugging Face files of $BASE_NAME)"
  local free_gb; free_gb=$(df -P --block-size=1G "$WORK" 2>/dev/null | awk 'NR==2{print $4}' || echo 0)
  (( ${free_gb:-0} >= 60 )) || log "WARNING: only ${free_gb:-?} GiB free where $WORK lives; a run needs about 60 (merged weights, two GGUF files, checkpoints)"
  if [[ "$SKIP_GGUF" != 1 ]]; then
    [[ -f "$LLAMA_CPP/convert_hf_to_gguf.py" ]] || die "llama.cpp not found at LLAMA_CPP=$LLAMA_CPP (git clone and build it, or SKIP_GGUF=1)"
    [[ -x "$LLAMA_CPP/build/bin/llama-quantize" ]] || die "llama-quantize is not built at $LLAMA_CPP/build/bin (cmake -B build && cmake --build build -t llama-quantize)"
  fi
  curl -fsS -m 5 "$OLLAMA_URL/api/version" >/dev/null 2>&1 || log "WARNING: no Ollama at $OLLAMA_URL: the create, smoke and eval stages need one"
  log "bundle: $BUNDLE; base: $BASE; run: $RUN_NAME (plan $PLAN, $PRECISION); work dir: $WORK"
}
do_data() {
  $PY tools/data_bundle.py verify "$BUNDLE"
  $PY tools/data_bundle.py extract "$BUNDLE" --to "$REPO/local-cache"
  $PY tools/data_bundle.py rebuild-check "$BUNDLE"
  # the lint belongs on what the model sees (train.jsonl, val.jsonl: canonical names, quantities the server's reader accepts), not on the raw seeds:
  # those are inputs of the build and still use the old parameter names until it relabels them
  $PY tools/tool_census.py lint --canonical-only --data "$DATA"
}
train_args() {   # train_args <out> [more]
  local out=$1; shift
  echo "$PY -u tools/train_router.py train --data $DATA --base $BASE --out $out --precision $PRECISION --eval-limit $EVAL_LIMIT $EXTRA_TRAIN $*"
}
do_pilot() {
  $(train_args "$WORK/pilot" --max-steps 50 --eval-steps 25) > "$LOGS/pilot.log" 2>&1 || { tail -n 25 "$LOGS/pilot.log" >&2; return 1; }
  local secs; secs=$($PY -c "import json,sys; r=json.load(open(sys.argv[1])); print(int(r['projected_full_run_s']))" "$WORK/pilot/train_report.json")
  $PY - "$WORK/pilot/train_report.json" <<'EOF' | tee -a "$LOGS/pipeline.log"
import json, sys
r = json.load(open(sys.argv[1]))
print(f"pilot: {r['steps_trained']} steps, median step {r['median_step_s']} s, projected full run {r['projected_full_run_s'] / 3600:.1f} h "
      f"({r['steps_full_run']} steps), peak memory {r.get('vram_peak_reserved_gib')} GiB, precision {r.get('precision')}, optimiser {r.get('optim')}, accelerator {r.get('accelerator')}")
EOF
  (( secs <= MAX_TRAIN_H * 3600 )) || die "the pilot projects $((secs / 3600)) h, over MAX_TRAIN_H=$MAX_TRAIN_H: raise it, or pick a smaller SIZE or PRECISION=nf4"
}
do_train() {
  local tries=0 resume
  while (( tries < 4 )); do
    resume=""; compgen -G "$WORK/run1/checkpoints/checkpoint-*" >/dev/null && resume="--resume"
    if $(train_args "$WORK/run1" $resume) >> "$LOGS/train.log" 2>&1; then return 0; fi
    tries=$((tries + 1)); log "training stopped (attempt $tries of 4); the checkpoints are kept"
    (( tries < 4 )) && { log "resuming from the newest checkpoint in 60 s"; sleep 60; }
  done
  die "training did not finish in 4 attempts: see $LOGS/train.log"
}
do_merge() { $PY -u tools/train_router.py merge --base "$BASE" --adapter "$WORK/run1/adapter" --out "$WORK/merged" > "$LOGS/merge.log" 2>&1; }
do_gguf() {
  [[ "$SKIP_GGUF" == 1 ]] && { log "SKIP_GGUF=1: no GGUF"; return 0; }
  mkdir -p "$WORK/gguf"
  $PY "$LLAMA_CPP/convert_hf_to_gguf.py" "$WORK/merged" --outtype bf16 --outfile "$WORK/gguf/$RUN_NAME-bf16.gguf" > "$LOGS/gguf.log" 2>&1
  "$LLAMA_CPP/build/bin/llama-quantize" "$WORK/gguf/$RUN_NAME-bf16.gguf" "$WORK/gguf/$RUN_NAME-Q4_K_M.gguf" Q4_K_M 4 >> "$LOGS/gguf.log" 2>&1
  $PY tools/train_router.py modelfile --gguf "./$RUN_NAME-Q4_K_M.gguf" --out "$WORK/gguf/Modelfile" >> "$LOGS/gguf.log" 2>&1
  ls -la "$WORK/gguf" | tee -a "$LOGS/gguf.log"
}
ollama_host() { echo "${OLLAMA_URL#http://}"; }
do_create() { (cd "$WORK/gguf" && OLLAMA_HOST="$(ollama_host)" ollama create "$RUN_NAME" -f Modelfile) > "$LOGS/create.log" 2>&1; }
do_smoke() { $PY tools/eval_router_val.py --smoke --model "$RUN_NAME" --ollama-url "$OLLAMA_URL" --plan "$PLAN" --format toolfirst > "$LOGS/smoke.log" 2>&1; }
eval_both() {   # eval_both <data dir> <split> <label>
  local f fmt
  for f in toolfirst free; do
    fmt=$f; [[ $f == free ]] && fmt=none
    [[ -s "$WORK/eval_${3}_$f.json" ]] && { log "  eval_$3 $f already done"; continue; }
    $PY -u tools/eval_router_val.py --model "$RUN_NAME" --ollama-url "$OLLAMA_URL" --data "$1" --split "$2" --plan "$PLAN" --format "$fmt" \
      --out "$WORK/eval_${3}_$f.json" > "$LOGS/eval_${3}_$f.log" 2>&1
  done
}
do_eval() {
  eval_both "$DATA" val val
  eval_both "$HARD" hard hard
  local s
  for s in math conv; do
    local suite="$MATH_SUITE"; [[ $s == conv ]] && suite="$CONV_SUITE"
    [[ -s "$suite" ]] || { log "  no $s suite at $suite: skipped"; continue; }
    [[ -s "$WORK/eval_${s}.json" ]] || $PY -u tools/eval_math_suite.py --model "$RUN_NAME" --ollama-url "$OLLAMA_URL" --plan "$PLAN" --suite "$suite" \
      --out "$WORK/eval_${s}.json" > "$LOGS/eval_${s}.log" 2>&1 || log "  eval_$s did not complete: see $LOGS/eval_${s}.log"
    tail -n 12 "$LOGS/eval_${s}.log" 2>/dev/null | tee -a "$LOGS/pipeline.log"
  done
}

# ── driver ─────────────────────────────────────────────────────────────────────────────────────────
STAGE_LIST=()
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) [[ " ${ALL_STAGES[*]} " =~ " $a " ]] || die "unknown stage '$a' (stages: ${ALL_STAGES[*]})"; STAGE_LIST+=("$a") ;;
  esac
done
(( ${#STAGE_LIST[@]} )) || STAGE_LIST=("${ALL_STAGES[@]}")
mkdir -p "$STATE" "$LOGS"
log "run $RUN_NAME: stages ${STAGE_LIST[*]}$( (( DRY )) && echo ' (dry run)')"
for s in "${STAGE_LIST[@]}"; do
  if [[ -f "$(done_marker "$s")" && "$s" != preflight ]]; then log "== $s: already done"; continue; fi
  log "== $s"
  if (( DRY )); then echo "  [dry-run] stage $s"; continue; fi
  t0=$(date +%s)
  CURRENT_STAGE=$s
  "do_$s"                  # bare, so `set -e` is in force inside the stage: a failing command stops it (the ERR trap says where)
  echo "$(( $(date +%s) - t0 )) s" > "$(done_marker "$s")"
  log "== $s done in $(( $(date +%s) - t0 )) s"
done
log "finished: $WORK"
