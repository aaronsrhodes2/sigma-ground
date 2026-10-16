# Training on the Bosgame M5 (Ryzen AI Max+ 395, 128 GB) — what is ready, what to run first, what to decide

Written 2026-10-15. Local model projects and SkippyTel are moving to a Bosgame M5. The data work below is done and verified on this PC; nothing here has run on the M5 itself.

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
| one portable file | `local-cache/bundles/router_v31_2026-10-15.zip`, **3.5 MiB** (103 MB raw: train 69.4 MiB, rebuild 33.5 MiB, reference 0.4 MiB; 31 files hashed, repo commit and rebuild command recorded). `verify` OK; `rebuild-check` from the extracted zip OK in 37 s. Tool: `tools/data_bundle.py` (7 tests, including tamper detection). |
| training script | `--precision bf16` (8 tests). A CPU smoke run in WSL on data rebuilt from the bundle, with a tiny random Qwen3 and the real tokenizer, ran to the end (3 steps, eval loss 11.96 = ln of the vocabulary, as a random model should give). **Not tested: any GPU path, ROCm, or Unsloth on AMD.** |
| the data's known holes (`tool_census coverage`) | 56 of 545 parameters appear in no training example (4 of them required, e.g. `parse_quantity.quantity_string`); units written as quantities (`"5 cm"`) are never taught; no label carries a convention. These need the decisions below, not more of the same data. |

The zip sits in `local-cache/` (gitignored): copy it to the M5 by any means, then `python tools/data_bundle.py verify <zip>` there.

## Day-one checklist on the M5 (about half an hour to know the speed)

1. `git clone` the repo, `python tools/data_bundle.py extract <zip> --to local-cache`, then `python tools/data_bundle.py rebuild-check <zip>`: expect three IDENTICAL lines (CPU only).
2. Training environment: PyTorch for gfx1151 (AMD's index or the TheRock nightlies; the stock PyTorch wheels did not include gfx1151 in the reports I found), transformers, peft; Unsloth only if its AMD path installs cleanly ([AMD's note](https://www.amd.com/en/developer/resources/technical-articles/2026/train-and-run-models-on-amd-gpus-with-unsloth.html) says it does; [an earlier report](https://zenn.dev/robustonian/articles/strix-halo_full_finetuning?locale=en) said it did not). Check `torch.cuda.is_available()` and `torch.version.hip`. Give the GPU its memory in the BIOS.
3. The pilot that decides the plan: `python tools/train_router.py train --precision bf16 --data local-cache/router_dataset_v31 --base <Qwen3-4B dir> --out local-cache/m5_pilot --max-steps 50 --eval-limit 200`, then read `train_report.json`: `projected_full_run_s`, `vram_peak_*`, `accelerator`. Run it again with `--no-unsloth` to compare.
4. Merge on the CPU, build the GGUF with a Vulkan build of llama.cpp (Vulkan was at least as fast as ROCm for generation in the [Phoronix](https://www.phoronix.com/review/lemonade-vulkan-rocm/2) and OpenBenchmarking runs), `ollama create`, then the unchanged evaluation scripts and `tools/baselines_report.py` for the comparison.
5. `tools/run_router_v3.sh` is Windows plus WSL specific. Once the M5's operating system is known I will write a plain-Python runner for the same stages.

## Decisions that change labels (make them before the real run, because they change the data)

1. **Teach quantities.** The router's system prompt tells it to "convert every input to the unit its parameter name states", so it does unit arithmetic in its head. The contract now accepts `"5 cm"` and Mentat converts. Recommendation: relabel numeric unit parameters as quantity strings in the question's own units, as a separate run to compare, and let the benchmark's argument check convert with pint before comparing. This removes a source of error rather than adding data.
2. **Conventions.** Every no-tool model lost 11–13 of 100 questions to the oracle's conventions (`misc/BASELINES.md`). Recommendation: make Fourier convention, ODE constant naming and series order explicit in the tool results (and as arguments), keep sympy's as the default so existing labels stand, and add training examples where the question names the other convention.
3. **Renames** (`misc/CONTRACT_RENAMES.md`). Apply now with aliases, then relabel the data mechanically, or defer to a later round. Applying them before the run costs one relabel; after, a second training run.
4. **Sizes.** My recommendation: train the 4B in both `nf4` and `bf16` (so the recipe change is measured, not assumed), and the 8B in `bf16`; none larger as the deployed router. Use the rest of the room for the teacher and judge.
5. **The M5's operating system** (Windows 11 or Linux). AMD's training stack is better documented on Linux; SkippyTel's Docker and Tailscale setup runs on either.
6. **The 4070 PC**: kept as a second GPU box (it trains and decodes small models faster than the M5) or retired.

Sources: [Guru3D review](https://www.guru3d.com/review/review-bosgame-m5-ai-mini-desktop-ryzen-ai-395/), [Notebookcheck](https://www.notebookcheck.net/Mini-PC-with-AMD-Strix-Halo-and-128-GB-RAM-AMD-Ryzen-AI-Max-395-and-AMD-Radeon-8060S-impress-in-Bosgame-M5.1088558.0.html), [AMD on Unsloth](https://www.amd.com/en/developer/resources/technical-articles/2026/train-and-run-models-on-amd-gpus-with-unsloth.html), [Unsloth AMD install](https://unsloth.ai/docs/get-started/install/amd/amd.md), [Strix Halo Unsloth write-up](https://sleepingrobots.com/dreams/unsloth-studio-strix-halo/), [Strix Halo fine-tuning notes](https://zenn.dev/robustonian/articles/strix-halo_full_finetuning?locale=en), [Phoronix ROCm vs Vulkan](https://www.phoronix.com/review/lemonade-vulkan-rocm/2), OpenBenchmarking runs ([Nov 2025](https://mail.openbenchmarking.org/result/2511200-NE-STRIXHALO33&export=html), [Feb 2026](https://mail.openbenchmarking.org/result/2602150-NE-STRIXHALO14&export=html)).
