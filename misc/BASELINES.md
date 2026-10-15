# Baselines — models we measure the contender against (CONTENDER_PLAN gate 4)

Question each answers: how far does a model get on the held-out math suite (329 questions the router was never trained on) **by itself, with no
tools**, compared with the router + Mentat route (v3 4B: 194/329 right answers, arguments 160/329, tools 310/329)? And, later, how much do they gain
when Mentat is their calculator. Scorer: `tools/eval_math_suite.py::same_answer`, unchanged; runner: `tools/eval_direct_answer.py`; report: `tools/baselines_report.py`.
The Captain approved the downloads on 2026-10-13 (free public weights; licence checked first).

| model | what it is | licence | file / size | where it is |
|---|---|---|---|---|
| **P1-30B-A3B** (PRIME-RL) | open physics-olympiad model: Qwen3-30B-A3B-Thinking-2507 + multi-stage RL on physics competitions; IPhO 2025 silver at this size (the 235B flagship won gold; [paper](https://arxiv.org/abs/2511.13612)) | Apache-2.0 (`PRIME-RL/P1-30B-A3B`) | community GGUF `mradermacher/P1-30B-A3B-GGUF`, `Q4_K_M`, 18,556,686,432 bytes, sha256 `26da6175…37ad4` (verified by `tools/fetch_baseline.sh`) | `local-cache/baselines/P1-30B-A3B/` on D:, imported into the D: Ollama store |
| **Phi-4-mini-reasoning** (Microsoft) | 3.8B dense reasoning model for math ([paper](https://arxiv.org/abs/2504.21233): MATH-500 94.6, AIME24 57.5 as reported by its authors) | MIT (`microsoft/Phi-4-mini-reasoning`) | Ollama library `phi4-mini-reasoning`, 3.2 GB | D: Ollama store (`local-cache/ollama-baselines`) |
| stock **Qwen3-4B / 8B** (thinking) | the base models of our routers | Apache-2.0 | Ollama `qwen3:4b`, `qwen3:8b` | D: Ollama store |
| frontier ceiling | Sonnet driving Mentat through the API | — | — | **not started**: needs the spending cap the Captain names (decision 6) |

## How they were run

* **One Ollama for all of them, with its model store on D:** `OLLAMA_HOST=127.0.0.1:11436 OLLAMA_MODELS=<local-cache/ollama-baselines> OLLAMA_NUM_PARALLEL=1 ollama serve` (log: `local-cache/ollama_11436_baselines.log`).
  The main Ollama's store is on C:, which had 7 GB free after the 2026-10-14 disk-full incident, and P1's file is 17 GB. P1 was imported from its GGUF with
  `local-cache/baselines/P1-30B-A3B/Modelfile` (a plain ChatML template that ends the prompt with `<think>`, as Qwen3-Thinking-2507's own template does). The 30B MoE does not fit 16 GB: Ollama puts
  what fits on the GPU and the rest on the CPU (about 70 tokens per second); `skippytel-ollama` was stopped for that run. The three small models ran with it up.
* Sampling is each model card's thinking-mode recommendation (P1 0.6, Phi 0.8, Qwen3 0.6), seed 7, `--num-predict 8000`, a 16,384-token context; the prompt asks for a last line `ANSWER: …`
  and the extractor also accepts `\boxed{…}`; a missing answer counts as a miss and is reported. The queue is `tools/run_baselines.sh`; results go to `local-cache/direct_<model>_100.json`.
* The sample: a stratified 100 of the 329 questions (`--stratified 100 --seed 7`, spread over the 21 tool kinds), the same for every model.
* Elapsed: P1 0.9 h of answering; Phi-4-mini 55 min; Qwen3-4B 54 min; Qwen3-8B 0.8 h of answering but 6.8 h on the clock (a hung server, below).

## Results (measured; the scorer is the router's, unchanged)

Strict = `eval_math_suite.same_answer` against the oracle's answer (sympy's conventions). Lenient = strict + the misses I read as **the same answer in another form or convention**, one verdict per
question in `misc/baselines_adjudication.json`. "+ convention" also counts the Taylor-"order n" readings and Phi's factoring over the reals, which are defensible but different. Both extra numbers are my
judgement; the strict one is the machine's.

| system | strict | lenient | + convention | median s | median tokens | the rest of its misses |
|---|---|---|---|---|---|---|
| v3 4B router + Mentat | 68 / 100 | | | about 0.5 | one short plan | |
| **v3.1 4B router + Mentat** (master tools) | **97 / 100** | | | about 0.5 | one short plan | 2 ODEs, 1 determinant (written \|A\|, read as a norm) |
| **Qwen3-4B** thinking, no tools | **81 / 100** | **94** | 97 | 26.6 | 2,023 | 1 wrong, 1 eigenvalues without multiplicity, 1 no final answer |
| **Qwen3-8B** thinking, no tools | **81 / 100** | 93 | 95 | 8.9 | 877 | 4 wrong, 1 eigenvalues without multiplicity |
| P1-30B-A3B, no tools (Q4_K_M, 18 GB) | 80 / 100 | 91 | 94 | 24.3 | 1,705 | 3 wrong, 1 eigenvalues without multiplicity, 2 no final answer |
| Phi-4-mini-reasoning, no tools | 67 / 100 (64 as first scored) | 87 | 90 | 19.0 | 1,147 | 6 wrong, 4 no final answer |

Paired against the v3.1 router on the same 100 questions (two-sided exact sign test on the questions where they disagree):

| model | strict: only router / only model | p | lenient: only router / only model | p |
|---|---|---|---|---|
| Qwen3-4B | 17 / 1 | 0.0001 | 6 / 3 | 0.51 |
| Qwen3-8B | 16 / 0 | < 0.0001 | 6 / 2 | 0.29 |
| P1-30B-A3B | 17 / 0 | < 0.0001 | 8 / 2 | 0.11 |
| Phi-4-mini-reasoning | 30 / 0 | < 0.0001 | 12 / 2 | 0.013 |

## What the strict number hides — every model's misses read by hand (my judgement; the scorer is not changed)

| kind of miss | P1 | Phi | Qwen3-4B | Qwen3-8B |
|---|---|---|---|---|
| Fourier transform right in the angular-frequency convention (`sqrt(pi)*exp(-k^2/4)`; sympy's oracle uses e^(-2πikx)); each checked against the oracle's F(k/2π) | 3 | 3 | 4 | 3 |
| ODE: the same solution family, constants named or ordered differently (`C` for `C1`, sin/cos constants swapped) | 5 | 5 | 5 | 5 |
| `\|t\|` for `sqrt(t²)`, an antiderivative that differs by a constant, a scalar for a one-element list, the factors written as a list | 3 | 4 | 4 | 3 |
| the right answer in an odd wrapper (`[[…]]`, `<…>`, a unicode fraction slash, `isqrt`) | 0 | 8 | 0 | 1 |
| *(counted only in "+ convention")* Taylor "order n": keeps the term the oracle stops before | 3 | 2 | 3 | 2 |
| *(counted only in "+ convention")* factored over the reals where the oracle stays over the rationals | 0 | 1 | 0 | 0 |
| eigenvalues without their multiplicity | 1 | 0 | 1 | 1 |
| no final answer at the 8,000-token cap | 2 | 4 | 1 | 0 |
| wrong value or sign (determinants, curls, a Fourier prefactor, garbled series) | 3 | 6 | 1 | 4 |
| total misses | 20 | 33 | 19 | 19 |

Fourteen questions are missed strictly by all four models. They are the Fourier, ODE-constant, series-order, antiderivative-constant, `|t|` and one-element-list questions (`0025 0084 0102 0155 0160 0164 0195 0201
0202 0284 0288 0289 0291 0328`): in the main the oracle's conventions, with a few real slips on top (0201 for two models, 0160 for two).

## Incidents and corrections

* **Qwen3-8B: a hung server scored twelve questions as misses.** From 18:57 on 2026-10-14 the 8B's runner stopped answering; every request after that came back HTTP 500 after exactly 30 minutes (the client saw
  twelve `ReadTimeout`s, rows 16–27) until 01:25. Those twelve were never answered, so they are not the model's misses. They were asked again with the same sample and settings
  (`--retry-errors`, `--timeout 900`; rows marked `retried`): nine right, no errors. First scoring 72/100 (with the twelve as misses), now **81/100**. The other 88 rows are the original answers, untouched.
* **Phi-4-mini: my extractor mangled `\boxed{…}`.** Phi writes `ANSWER: \boxed{-39}`; `latex_to_plain` turned that into the literal text `boxed(-39)`, which the scorer then read as garbage. Four correct or
  nearly correct answers (0020, 0222, 0240, and 0284 with an ODE constant) were scored as misses. Fixed in `extract_answer` (with tests); the stored rows carry only the mangled answer, so `baselines_report.py`
  unwraps it before scoring (new runs also store the model's whole reply, `text`, so an answer can be re-extracted without asking again). Phi 64 → 67 strict. No other model's rows had the wrapper.
* **P1's earlier hand-read was off by one.** The first write-up said "about 92 of 100, 95 with the order-n readings"; counting by question it is 91 and 94 (the eigenvalues-without-multiplicity case is incomplete,
  not equivalent). The per-question verdicts above replace that paragraph.
* A parser gap seen in passing: Phi wrote `2⁄s^3` with the unicode fraction slash U+2044, which `mathparse` does not read. Not changed here.

## What we can learn from the comparison

* **A 4B general thinking model ties a 30B physics-RL model on this math** (81 vs 80 strict, 94 vs 91 lenient). Whatever P1's physics training bought, it did not show up in these questions. Qwen3-8B is no better than 4B
  (81 and 93) but thinks less: a 9 s median and 877 tokens against 27 s and 2,023.
* **Conventions, not mathematics, are the largest single gap between strict and lenient, and the same questions cost every model.** 11 to 13 of each model's hundred are lost to the oracle's choice of Fourier
  convention, ODE constant names, antiderivative constant or `|t|` (Phi loses another 8 to odd wrappers, which is format), and 2 or 3 more to the series-order reading. For the contender this is a specification problem: an answer should carry its convention (Fourier: which one; ODE: how the constants
  are named; series: what "order n" kept), and the tools should take it as an argument. A Wolfram head-to-head has the same problem (`misc/WA_TRICKLE.md`: the ODE constants and the `+1/16` antiderivative).
* **The router's edge is speed and exactness more than raw capability.** Its 97 against the lenient 87–94 is not significant for P1 (p 0.11), Qwen3-4B (0.51) or Qwen3-8B (0.29) at 100 questions, and it is
  significant for Phi (0.013). What the router has is 0.5 s instead of 9–27 s, answers in the oracle's own conventions, and no 8,000-token blow-ups; it also trained on these question families, which the baselines
  did not. This sample cannot say the router is the better mathematician.
* **Where the models really fail:** sign errors in 3×3 determinants (P1 1, Qwen3-8B 2), curls (Phi 3, P1 1), a Fourier prefactor, and reasoning that never converges (Phi 4 of its misses hit the token cap).

## Still to do

* **Mentat as the models' calculator** (the same questions, tools on): how much each model gains when Mentat does the arithmetic (W6).
* **Frontier ceiling** (Sonnet through the API): needs a hard spending cap the Captain names first.
