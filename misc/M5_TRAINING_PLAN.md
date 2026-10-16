# Training on the Bosgame M5 (Ryzen AI Max+ 395, 128 GB) — what is ready, what to run first, what to decide

Written 2026-10-15, updated 2026-10-16 with the Captain's answers to the six decisions and the Linux rehearsal. Local model projects and SkippyTel are moving to a Bosgame M5. The data work is done and verified on this PC; nothing here has run on the M5 itself. **The data (plan v4), its checks and its bugs are in `misc/ROUTER_V32_PLAN.md`; this file is about the machine.**

## Update 2026-10-16 in five lines

* The decisions are answered and built (table at the end): quantities quoted and answers in the question's own units with SI in parentheses, conventions explicit, parameters renamed, **Linux**, and this PC stays as the fast-small-model runner for development and gaming.
* The data is **v3.2**, `local-cache/bundles/router_v32_2026-10-15.zip` (3.5 MiB). It regenerates on the CPU with no teacher to the same bytes on Windows (Python 3.13) and Linux (Python 3.12): `train.jsonl 676538b6…`, `val.jsonl 1d2d1fcc…`.
* `tools/run_router_linux.sh` runs every stage (preflight, data, pilot, train, merge, gguf, create, smoke, eval). **Executed for real:** preflight, data and pilot under Linux (WSL, this PC's GPU), merge and gguf under Linux on the pilot's adapter (merged model, bf16 GGUF 4.07 GB, Q4_K_M 1.28 GB, Modelfile; 3 min 49 s for the 1.7B), create and smoke against a separate Ollama (after 50 steps of v4 training the model already answered `"mass_kg": "2 solar masses"`). **Not executed:** the full `train` stage (the resume loop is copied from the proven Windows pipeline) and `eval` (hours: validation, hard validation, math and convention suites). The rehearsal found defects in the build, all fixed (`ROUTER_V32_PLAN.md`).
* **A real GPU pilot of the new path** (`--precision bf16`, plain PEFT, no bitsandbytes, no Unsloth): Qwen3-1.7B, 50 steps, batch 4×4, median **1.57 s per step**, projected full run **2.3 h**, 11.1 GiB. The old recipe (4-bit through Unsloth) measured 0.87 s per step for the same size and 1.97–2.0 s for the 4B (2.4–2.8 h) on this card. So plain bf16 costs about 1.8× per step here; the M5's own pilot decides the rest.
* Not tested anywhere: ROCm, and Unsloth on AMD. Unsloth also failed to start reliably inside WSL today (it died during import, silently); the plain path is the one that is proven.

## A correction first

Earlier today I answered as if the new machine were "very big and fast". The M5 is a **capacity** machine, not a speed machine:

| | this PC (RTX 4070 Ti SUPER, 16 GB) | Bosgame M5 |
|---|---|---|
| memory the model can use | 16 GB | 128 GB shared; up to about 96 GB can be given to the GPU ([Guru3D](https://www.guru3d.com/review/review-bosgame-m5-ai-mini-desktop-ryzen-ai-395/) ran it at 64 GB) |
| memory bandwidth (spec sheets) | about 672 GB/s | about 256 GB/s (LPDDR5X; 8000 or 8533 MT/s depending on the source) |
| Qwen3-8B generating | **99 tokens/s measured here** (Q4_K_M, Ollama log) | about 26 tokens/s reported for the Q8_0 file on Llama.cpp ([OpenBenchmarking](https://mail.openbenchmarking.org/result/2511200-NE-STRIXHALO33&export=html)); a Q4 file would be faster, not 4x |
| a 35B mixture-of-experts model (3B active) | does not fit | about 46–53 tokens/s reported for a Q8_0 file in one community run |

The M5 numbers are as reported by search results (I read the summaries, not every page), so treat them as indicative; the first pilot below replaces them with measurements.

Decoding a dense model is limited by memory bandwidth, so small models will run **slower** on the M5 than on this card, while models that cannot run here at all become possible. For training the same holds: the M5 removes out-of-memory problems; it does not make a 4B fine-tune faster. How slow is unknown until the pilot in the checklist runs (the 4B v3.1 run took 187 minutes here; I would not assume less).

## What the room is good for

* **Training without quantization.** An 8B base in bfloat16 is about 16 GB, a 32B about 64 GB. The study's recipe (4-bit NF4 base, 8-bit optimizer) exists to fit 16 GB, and it needs bitsandbytes, whose support on this chip I could not confirm ([Unsloth on AMD](https://unsloth.ai/docs/get-started/install/amd/amd.md), [a Strix Halo write-up](https://sleepingrobots.com/dreams/unsloth-studio-strix-halo/)). So `train_router.py` now has `--precision bf16`: unquantized base, LoRA, plain AdamW, no bitsandbytes anywhere. The default is still `nf4`, the v3/v3.1 recipe.
* **Several models resident at once**: the router, SkippyTel's Qwen, a large judge, an embedding model.
* **A much stronger teacher** for the paraphrase step that writes the training questions' wordings (today `qwen2.5:14b`; `TEACHER` is a knob). Wording diversity is what the held-out suite tests.
* **Not a bigger deployed router.** A plan is a few dozen tokens and the router's job is to be fast; a 14B dense router would decode at roughly a third of a 4B's speed on this bandwidth (estimate from the bandwidth, not measured).

## What is done now

| item | result |
|---|---|
| labels against today's tools | `tool_census lint` on the v3.1 train and val data plus the five seed folders: **61,392 tool calls, 232 signatures, no issues**. The contract work was additive, so the labels have not drifted. |
| the dataset can be rebuilt with no teacher, no GPU, no WSL | rebuilt from the final rewrite cache with a dead teacher URL in 38 s: `train.jsonl`, `val.jsonl` and `rewrites.jsonl` are **byte-identical** (sha256 `7635a8a3…`, `ff057e65…`, `ab492719…`). |
| one portable file | (v3.1; superseded by `router_v32_2026-10-15.zip`, same size) `local-cache/bundles/router_v31_2026-10-15.zip`, **3.5 MiB** (103 MB raw: train 69.4 MiB, rebuild 33.5 MiB, reference 0.4 MiB; 31 files hashed, repo commit and rebuild command recorded). `verify` OK; `rebuild-check` from the extracted zip OK in 37 s. Tool: `tools/data_bundle.py` (7 tests, including tamper detection). |
| training script | `--precision bf16` (8 tests). A CPU smoke run in WSL on data rebuilt from the bundle, with a tiny random Qwen3 and the real tokenizer, ran to the end (3 steps, eval loss 11.96 = ln of the vocabulary, as a random model should give). A real GPU pilot of the plain bf16 path ran on 2026-10-16 (update at the top). **Not tested: ROCm, or Unsloth on AMD.** |
| the data's known holes (`tool_census coverage`) | 56 of 545 parameters appear in no training example (4 of them required, e.g. `parse_quantity.quantity_string`); units written as quantities (`"5 cm"`) are never taught; no label carries a convention. These need the decisions below, not more of the same data. |

The zip sits in `local-cache/` (gitignored): copy it to the M5 by any means, then `python tools/data_bundle.py verify <zip>` there.

## Day-one checklist on the M5 (about half an hour to know the speed)

1. `git clone` the repo, `python tools/data_bundle.py extract <zip> --to local-cache`, then `python tools/data_bundle.py rebuild-check <zip>`: expect three IDENTICAL lines (CPU only).
2. Training environment: PyTorch for gfx1151 (AMD's index or the TheRock nightlies; the stock PyTorch wheels did not include gfx1151 in the reports I found), transformers, peft; Unsloth only if its AMD path installs cleanly ([AMD's note](https://www.amd.com/en/developer/resources/technical-articles/2026/train-and-run-models-on-amd-gpus-with-unsloth.html) says it does; [an earlier report](https://zenn.dev/robustonian/articles/strix-halo_full_finetuning?locale=en) said it did not). Check `torch.cuda.is_available()` and `torch.version.hip`. Give the GPU its memory in the BIOS.
3. The pilot that decides the plan: `SIZE=4b PRECISION=bf16 EXTRA_TRAIN=--no-unsloth tools/run_router_linux.sh preflight pilot` (then the same with `SIZE=8b`, and without `EXTRA_TRAIN` to see whether Unsloth works), and read the line it prints: median step, projected hours, peak memory, optimiser, accelerator (`hip` is set on ROCm). On this card the same size took 1.57 s per step in bf16. The runner stops if the projection is over `MAX_TRAIN_H` (30 by default).
4. The rest of the stages are in the same script: merge on the CPU, a Vulkan build of llama.cpp for the GGUF (Vulkan was at least as fast as ROCm for generation in the [Phoronix](https://www.phoronix.com/review/lemonade-vulkan-rocm/2) and OpenBenchmarking runs), `ollama create`, smoke, and the evaluations on the validation split, the hard validation, the math suite and the convention suite. Compare with `tools/baselines_report.py` and the numbers in `misc/BASELINES.md`.
5. `tools/run_router_v3.sh` stays as the Windows-plus-WSL pipeline for this PC; `tools/run_router_linux.sh` is the one for the M5. Install the project with `pip install -e '.[mcp,benchmark]'` first: the preflight lists any declared package that is missing (a missing `periodictable` once changed the dataset silently).

## The six decisions (answered by the Captain, 2026-10-15) and what to train

| decision | answer | consequence |
|---|---|---|
| 1. quantities | "Return the answer to match the measurement system used, but put the scientific/metric standard in (parentheses) like '1 yard (0.914m)'" | built: `display` on results, quoted quantities in the labels (plan v4) |
| 2. conventions | "yes if the physics tests still pass" | built; the full test suite compared with a baseline (`ROUTER_V32_PLAN.md`) |
| 3. renames | "sure" | built, old names still accepted |
| 4. sizes | "you know better than I" | **4B in `nf4` and in `bf16`** (so the recipe change is measured, not assumed) and the **8B in `bf16`**; nothing larger as the deployed router. The extra room goes to a stronger teacher for wordings and to a judge model, not to a bigger router. |
| 5. operating system | Linux | `tools/run_router_linux.sh` |
| 6. this PC | "the fast-small-model runner, as well as dev and gaming" | the 4B router and small models are served from it (about 100 tokens/s for an 8B on this card); the M5 hosts the big models and SkippyTel; nothing to build until the M5 is up |

The three training runs, in order: **4B bf16** (the pilot first), **4B nf4** (the old recipe, for the comparison), **8B bf16**. Judge them on the math suite, the convention suite and the hard validation, against the v3.1 4B's numbers.

Sources: [Guru3D review](https://www.guru3d.com/review/review-bosgame-m5-ai-mini-desktop-ryzen-ai-395/), [Notebookcheck](https://www.notebookcheck.net/Mini-PC-with-AMD-Strix-Halo-and-128-GB-RAM-AMD-Ryzen-AI-Max-395-and-AMD-Radeon-8060S-impress-in-Bosgame-M5.1088558.0.html), [AMD on Unsloth](https://www.amd.com/en/developer/resources/technical-articles/2026/train-and-run-models-on-amd-gpus-with-unsloth.html), [Unsloth AMD install](https://unsloth.ai/docs/get-started/install/amd/amd.md), [Strix Halo Unsloth write-up](https://sleepingrobots.com/dreams/unsloth-studio-strix-halo/), [Strix Halo fine-tuning notes](https://zenn.dev/robustonian/articles/strix-halo_full_finetuning?locale=en), [Phoronix ROCm vs Vulkan](https://www.phoronix.com/review/lemonade-vulkan-rocm/2), OpenBenchmarking runs ([Nov 2025](https://mail.openbenchmarking.org/result/2511200-NE-STRIXHALO33&export=html), [Feb 2026](https://mail.openbenchmarking.org/result/2602150-NE-STRIXHALO14&export=html)).
