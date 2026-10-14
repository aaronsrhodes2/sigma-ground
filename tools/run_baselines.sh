#!/usr/bin/env bash
# The contender plan's local baselines (misc/BASELINES.md): each model answers the SAME stratified sample of the held-out math suite
# with no tools, scored by the router's scorer. Runs one model after another on the dedicated Ollama whose model store is on D:
# (port 11436, started as: OLLAMA_HOST=127.0.0.1:11436 OLLAMA_MODELS=<D: folder> ollama serve), so nothing is written to C:.
#
#   bash tools/run_baselines.sh                       # P1 is run separately first; this does phi4-mini-reasoning, qwen3:4b, qwen3:8b
#   N=100 SEED=7 MODELS="phi4-mini-reasoning:0.8 qwen3:4b:0.6" bash tools/run_baselines.sh
#
# MODELS is "name:temperature" pairs (the temperatures are the model cards' recommended sampling for thinking mode). Resumable: a model
# whose result file exists is skipped.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
N="${N:-100}"; SEED="${SEED:-7}"; URL="${URL:-http://127.0.0.1:11436}"
MODELS="${MODELS:-phi4-mini-reasoning:0.8 qwen3:4b:0.6 qwen3:8b:0.6}"
export PYTHONIOENCODING=utf-8
curl -fsS -m 5 "$URL/api/version" >/dev/null || { echo "no Ollama at $URL" >&2; exit 2; }
for spec in $MODELS; do
  temp="${spec##*:}"; model="${spec%:*}"
  safe="${model//[:\/]/_}"
  out="local-cache/direct_${safe}_${N}.json"
  if [[ -s "$out" ]]; then echo "$(date '+%F %T') $model: already done ($out)"; continue; fi
  echo "$(date '+%F %T') $model: $N stratified questions, temperature $temp"
  python -X utf8 -u tools/eval_direct_answer.py --model "$model" --ollama-url "$URL" --stratified "$N" --seed "$SEED" \
    --temperature "$temp" --num-predict 8000 --out "$out" > "local-cache/direct_${safe}_${N}.log" 2>&1 \
    || echo "$(date '+%F %T') $model: FAILED (see local-cache/direct_${safe}_${N}.log)"
  OLLAMA_HOST="${URL#http://}" ollama stop "$model" >/dev/null 2>&1 || true      # free the VRAM for the next model
done
echo "$(date '+%F %T') baselines complete"
