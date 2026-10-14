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

* Run after the v3.1 training run released the GPU (2026-10-14 06:20 on). `skippytel-ollama` is stopped while they run (the models need the whole 16 GB).
* **One Ollama for all of them, with its model store on D:** `OLLAMA_HOST=127.0.0.1:11436 OLLAMA_MODELS=<local-cache/ollama-baselines> OLLAMA_NUM_PARALLEL=1 ollama serve`. The main Ollama's store is on C:,
  which had 7 GB free after the 2026-10-14 disk-full incident, and P1's file is 17 GB. P1 was imported from its GGUF with `local-cache/baselines/P1-30B-A3B/Modelfile` (a plain ChatML template that
  ends the prompt with `<think>`, as Qwen3-Thinking-2507's own template does); Phi-4-mini-reasoning, `qwen3:4b` and `qwen3:8b` were pulled into the same store. The 30B MoE does not fit 16 GB:
  Ollama puts what fits on the GPU and the rest on the CPU (about 70 tokens per second).
* Sampling is each model card's thinking-mode recommendation (P1 0.6, Phi 0.8, Qwen3 0.6), seed 7, `--num-predict 8000`; the prompt asks for a last line `ANSWER: …` and the extractor also
  accepts `\boxed{…}`; a missing answer counts as a miss and is reported. The queue is `tools/run_baselines.sh`.
* Each run records tokens and seconds per question; results go to `local-cache/direct_<model>_<N>.json` and the numbers into this file.

## Results (measured; the scorer is the router's, unchanged)

Same 100 questions for every model: a stratified sample of the 329-question held-out math suite (`--stratified 100 --seed 7`, spread over the 21 tool kinds).
Strict = `eval_math_suite.same_answer` against the oracle's answer (sympy conventions). Seconds are per question on this PC (RTX 4070 Ti SUPER 16 GB).

| system | right (strict) | median seconds | tokens per question | notes |
|---|---|---|---|---|
| v3 4B router + Mentat | 68 / 100 | about 0.5 (whole 329-question run took 155 s for v3.1; includes the tool) | one short plan | the router writes one tool call; Mentat computes |
| **v3.1 4B router + Mentat** (master tools) | **97 / 100** | about 0.5 | one short plan | misses: 2 ODEs, 1 determinant (written \|A\|, read as a norm) |
| **P1-30B-A3B alone, no tools** (Q4_K_M, 18 GB, thinking, temperature 0.6) | **80 / 100** | 24 | 1,705 (median) | 2 questions with no final answer; paired against v3.1 4B: only P1 right 0, only router right 17 |
| Phi-4-mini-reasoning alone | _running_ | | | |
| Qwen3-4B thinking alone | _queued_ | | | |
| Qwen3-8B thinking alone | _queued_ | | | |

### What the strict number hides — P1's 20 misses read by hand (my judgement; the scorer is not changed)

| kind | count | examples |
|---|---|---|
| a correct answer in another convention | 3 Fourier transforms | P1 used e^(-ikx) without the 2π of sympy's convention (`sqrt(pi)*exp(-k^2/4)` for e^(-x²), the oracle says `sqrt(pi)*exp(-pi^2 k^2)`) |
| the same general solution, constants named differently or swapped | 5 ODEs | `C*exp(6*x)` for `C1*exp(6*x)`; `C1*exp(3x)+C2*exp(-3x)` for the swapped pair; `C1 cos(2t) + C2 sin(2t)` where the question named no variable |
| equivalent but not identical to the oracle's form | 3 | `abs(t)` for `sqrt(t**2)` (right for real t); `(4x+1)^4/16` for an antiderivative differing by a constant; `e^2` for the list `[exp(2)]` |
| "order n" read differently | 3 series | P1 keeps the x^n term, the oracle stops before it |
| a set of eigenvalues without their multiplicity | 1 | `[-4, 1]` for `[1, -4, -4]` |
| said "no real solutions" where the oracle lists the complex pair | 1 | 8n²-8n+3 |
| genuinely wrong | 2 | a determinant with the wrong sign (12 for -12); a curl with a stray `/r` |
| no final answer | 2 | one factorisation of an irreducible quadratic, one Fourier transform |

So P1's 80 is a floor: counting every case I judged equivalent (12 of the 20) it is about 92 of 100, 95 if the three "order n" readings are accepted too. The router's 97 is strict *against its own engine's conventions* (it calls sympy, which is also the oracle), so the two are
not symmetric; what the comparison does show is **cost and reliability**: one tool call at half a second, the answer exact and in the oracle's form, from a 2.5 GB model, against a 30B-parameter
model thinking for 24 s and 1,700 tokens and still disagreeing with the conventions on a fifth of the questions. It does not show that the router generalises beyond the problem families it was
trained on (the suite's wordings are held out, its families are not) — P1 was trained on none of them, which is the other asymmetry. Speed of P1 here: about 70 tokens per second with the MoE
partly on the CPU.
