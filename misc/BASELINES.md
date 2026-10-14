# Baselines — models we measure the contender against (CONTENDER_PLAN gate 4)

Question each answers: how far does a model get on the held-out math suite (329 questions the router was never trained on) **by itself, with no
tools**, compared with the router + Mentat route (v3 4B: 194/329 right answers, arguments 160/329, tools 310/329)? And, later, how much do they gain
when Mentat is their calculator. Scorer: `tools/eval_math_suite.py::same_answer`, unchanged; runner: `tools/eval_direct_answer.py`.
The Captain approved the downloads on 2026-10-13 (free public weights; licence checked first).

| model | what it is | licence | file / size | where it is |
|---|---|---|---|---|
| **P1-30B-A3B** (PRIME-RL) | open physics-olympiad model: Qwen3-30B-A3B-Thinking-2507 + multi-stage RL on physics competitions; IPhO 2025 silver at this size (the 235B flagship won gold; [paper](https://arxiv.org/abs/2511.13612)) | Apache-2.0 (`PRIME-RL/P1-30B-A3B`) | community GGUF `mradermacher/P1-30B-A3B-GGUF`, `Q4_K_M`, 18,556,686,432 bytes, sha256 `26da6175…37ad4` (verified by `tools/fetch_baseline.sh`) | `local-cache/baselines/P1-30B-A3B/` on D: |
| **Phi-4-mini-reasoning** (Microsoft) | 3.8B dense reasoning model for math ([paper](https://arxiv.org/abs/2504.21233): MATH-500 94.6, AIME24 57.5 as reported by its authors) | MIT (`microsoft/Phi-4-mini-reasoning`) | Ollama library `phi4-mini-reasoning`, 3.2 GB | host Ollama store (C:) |
| stock **Qwen3-4B / 8B** (thinking) | the base models of our routers | Apache-2.0 | already in Ollama (`qwen3:4b`, `qwen3:8b`) | host Ollama store |
| frontier ceiling | Sonnet driving Mentat through the API | — | — | **not started**: needs the spending cap the Captain names (decision 6) |

## How they will be run

* **Not before the v3.1 training run has released the GPU** (the router pipeline holds it from 00:00 until about 06:00; `local-cache/router_v31/4b/state/score.done`).
* Phi-4-mini-reasoning and Qwen3: the dedicated single-slot Ollama on port 11435 (`OLLAMA_NUM_PARALLEL=1`), `skippytel-ollama` stopped meanwhile.
* **P1-30B-A3B is not imported into Ollama**: the model store is on C: with 20 GB free and the file is 17 GB. It runs with `llama-server` (llama.cpp in WSL,
  `~/llama.cpp/build/bin/`) reading the GGUF from `/mnt/d/...`, MoE experts partly on the CPU if 16 GB of VRAM is not enough; the runner's `--openai` flag
  talks to it. First measure its speed on 20 questions; then a stratified 100 (`--stratified 100 --seed 7`, the same questions for every model), or all 329 if fast.
* Reasoning models write long traces: `--num-predict 6000`; the final line must be `ANSWER: …` (or `\boxed{…}`); a missing answer is counted as a miss and reported.
* Each run is recorded with tokens and seconds per question; results go to `local-cache/direct_<model>.json` and the numbers into this file.

## Results

(none yet — measured numbers only, appended here when the runs are done)
