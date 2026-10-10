# sigma-ground physics MCP — 3-way benchmark

A reproducible head-to-head test of three physics-Q&A systems:

| System | What it is | Strengths | Weaknesses |
|---|---|---|---|
| **sigma_ground + Qwen** | Local Qwen 2.5:7b orchestrating sigma-ground MCP tools | Transparent provenance, free, local, "fitted due to incompetence" honesty | Limited library breadth |
| **Wolfram Alpha** | Mathematica-backed computational engine via free-tier API | Massive topical breadth, symbolic-numeric integration, plots | No transparent provenance, free tier rate-limited |
| **Gemini 2.5 Pro** | Google's flagship LLM via paid API | Excellent reasoning, ~$1-5 for all 150 Qs, multimodal | No tools by default; hallucinations possible |

## Corpus

150 questions spanning a 4-year physics-major curriculum:

| Domain | Year | Count |
|---|---|---:|
| classical_mechanics_intro    | 1   | 15 |
| electromagnetism_intro       | 1-2 | 15 |
| waves_optics                 | 2   | 12 |
| thermodynamics_statmech      | 2-3 | 12 |
| modern_physics               | 2-3 | 12 |
| quantum_mechanics            | 3   | 12 |
| classical_mechanics_advanced | 3   | 10 |
| electrodynamics_advanced     | 3-4 | 10 |
| general_relativity           | 4   | 10 |
| cosmology                    | 4   |  8 |
| astrophysics                 | 4   | 12 |
| atomic_molecular             | 3-4 |  8 |
| nuclear_physics              | 3-4 |  7 |
| mathematical_methods         | 1-4 |  7 |

All questions phrased in natural English; numerical answers with tolerance bands; ground truth cited from textbooks (Halliday/Griffiths/Schutz/Hecht/Carroll-Ostlie) and standard databases (NIST, CODATA, IUPAC, NASA, Gaia DR3).

## Run all three systems

### 1. Install benchmark deps

```bash
cd D:\Aaron\development\sigma-ground
pip install -e ".[mcp,benchmark]"
```

### 2. Set up API keys

Put your keys in `D:\Aaron\development\.env` (the dev-root shared `.env`).
The runners auto-load this file via `python-dotenv` — no manual sourcing
needed. See `D:\Aaron\development\.env.reference.md` for the variable
catalog. Required for the benchmark:

```
GEMINI_API_KEY=...           # from https://aistudio.google.com/
WOLFRAM_ALPHA_APP_ID=...     # from https://developer.wolframalpha.com/ (free)
```

(`run_sigma_ground.py` needs no API keys; it uses local Ollama.)

### 3. Regenerate corpus (if needed)

```bash
python -m sigma_ground.mcp.benchmark.corpus
# Writes questions.json + ground_truth.json
```

### 4. Run each backend (parallel-safe)

**sigma_ground + Qwen** (~75 min runtime on 4070 Ti 16 GB):
```bash
ollama serve &           # if not running
ollama pull qwen2.5:7b   # if not already
python -m sigma_ground.mcp.benchmark.run_sigma_ground \
    --model qwen2.5:7b \
    --output sigma_ground/mcp/benchmark/results/sigma_ground_run.json
```

**Wolfram Alpha** (~2 days, rate-limited free tier):
```bash
python -m sigma_ground.mcp.benchmark.run_wolfram \
    --output sigma_ground/mcp/benchmark/results/wolfram_run.json \
    --pace-per-day 90 --pause-s 2
# Resumable -- run again the next day to continue
```

**Gemini 2.5 Pro** (~8 min, $1-5 total):
```bash
python -m sigma_ground.mcp.benchmark.run_gemini \
    --model gemini-2.5-pro \
    --output sigma_ground/mcp/benchmark/results/gemini_run.json
```

All runners are **resumable** — they skip questions already in their output file. Safe to ctrl-C and restart.

**How a Qwen-path question ends (`--finalize`, since 2026-10-10).** By default
(`chain`) the harness, not the model, ends each question. The first tool call
that returns a value ends it, unless the model's next call is a *different*
tool that takes that value as an input (a lookup feeding a computation, such
as `solar_system_body` → `schwarzschild_radius`). Anything else the model does
next (a re-call, an unrelated tool, a text answer) ends the question on the
last result. The harness writes the `ANSWER:` line from that tool result,
with its source and the call path. The record gets `finalized_from_tool`,
`finalize_reason` and `chain_length`. `--finalize model` restores the old loop
for A/B against earlier runs. Why: replaying the logged runs showed qwen2.5:7b
reached max turns on 63/64 Qwen-path questions by re-calling the right tool.
Under `chain` it scores 52→53 correct with 1→61 grounded, and qwen3:8b scores
50→50 with 27→61. Caveat: grounded means a tool produced the number, not that
it was the right tool. Two open-ended adversarial questions now get a
confident wrong-tool number where the old loop timed out and was flagged.

### 4. Score all three

```bash
python -m sigma_ground.mcp.benchmark.score
# Reads results/*_run.json, writes results/*_scored.json
# Prints per-system + per-domain summary
```

### 5. Generate the comparison report

Comparison-report scaffolding is in `comparison_report_template.md`. Fill in numbers from the scored.json files. Output goes in `misc/mcp_benchmark_results_<DATE>.md`.

## Result file schemas

### `<system>_run.json`

```json
[
  {
    "id": "mech_intro_001",
    "system": "sigma_ground",
    "model": "qwen2.5:7b",
    "answer_text": "<full LLM/Wolfram output>",
    "extracted_value": 1.43,
    "extracted_units": "s",
    "tool_calls": [{"name": "free_fall_time", "args": {...}, "result_text": "..."}],
    "elapsed_s": 12.3,
    "tokens_in": 250, "tokens_out": 80, "cost_usd": 0.0011
  },
  ...
]
```

### `<system>_scored.json`

```json
{
  "summary": {
    "overall_accuracy_pct": 87.3,
    "overall_correct": 131,
    "overall_total": 150,
    "by_domain": {
      "classical_mechanics_intro": {"correct": 14, "total": 15, "pct": 93.3},
      ...
    }
  },
  "rows": [...]
}
```

## What the benchmark measures (and what it doesn't)

**Measures:**
- Numerical accuracy (within stated tolerance)
- Per-domain coverage gaps
- Latency per question
- Cost per question (paid APIs only)
- For sigma_ground: tool-call discipline, "fitted due to incompetence" rate

**Does NOT measure:**
- Conversational quality (subjective; do separate user studies)
- Visualizations / plots (Wolfram does these; the others don't)
- Multimodal (image-based questions)
- Robustness to adversarial phrasing
- Long-form derivations / proofs

The "fitted due to incompetence" rate from sigma_ground is itself a useful output — every flagged answer becomes a library development ticket.

## Router fine-tune dataset (`router_dataset.py`)

Training data for a Qwen **router** fine-tune. A router reads a question and
emits one JSON plan; Mentat runs the plan and writes the answer, which is the
`--finalize chain` contract above. So every training target is a tool call,
a `request_clarification` call, or a decline. None of them is a number.

```bash
python -m sigma_ground.mcp.benchmark.router_dataset --per-tool 40          # templates only
python -m sigma_ground.mcp.benchmark.router_dataset --per-tool 40 \
    --teacher-model qwen2.5:32b --variants 3                              # + natural rewrites
```

Output goes to `local-cache/router_dataset/` (gitignored):
- `train.jsonl` / `val.jsonl`: chat records, each a system prompt, the
  question, and the JSON plan, plus `meta`.
- `manifest.json`: tool-surface hash, per-tool counts, tools with no examples
  and why, and drop counts with sample questions.

**How a label stays true to its question**
- Inputs are sampled from the parameter *name*. `wavelength_m` can be written
  as "550 nm", and the label is recomputed from the number actually shown
  (5.5e-07 m).
- String keys (materials, acids, metals, bodies) come from asking each tool:
  given a bad key, most of them list the valid ones.
- Every gold call is run through the real MCP server. Only calls that return
  a value are kept.
- If a deterministic classifier would send the question to a *different*
  tool, the example is dropped as ambiguous.
- Anything resembling a benchmark question is dropped, by text overlap or by
  same tool with the same answer or numbers. The corpora here stay the test
  set.
- Negatives: removing a required input gives a `request_clarification`
  example. Everyday and metaphor questions give a decline.
- Rewrites (`--teacher-model`, Ollama) are kept only if every number, unit
  (still next to its number) and named key survives, and the rewrite passes
  the classifier and benchmark checks again. A seed and its rewrites always
  land in the same split.

Template questions read stiffly ("Mass of 857 g and velocity of 0.694 km/s:
how much energy is it carrying?"). The teacher pass is what makes them
sound like people.

**Last full run** (2026-10-10, templates only, `--per-tool 40`, 2.6 min,
tool-surface hash `762bbee3725389bf`): 8,013 examples. That is 7,695 tool
calls, 289 clarifications and 29 declines, split 7,239 train / 774 val, and
covering 207 of the 217 tools the model sees. `manifest.json` gives the reason
for each of the 9 missing tools:
- **Free-text input** the generator can't invent: `parse_quantity`,
  `simulate`, `solve_ode`.
- **Every call fails**: `procedure_black_hole_profile` (see Session 37 in the
  session log), and `playground_apply` / `playground_simulate`, which need an
  open scene.
- **Fixed answer that equals a benchmark question's**, dropped by the guard:
  `age_of_universe`, `eta_value_report`, `mond_a0_constant`.

Flags:
- `--no-guard` keeps benchmark look-alikes. Use it for a router you will
  deploy, and never quote benchmark scores from that model.
- `--server-mem-gb` (default 6, POSIX only) caps the spawned server's memory,
  so a tool that runs away on a sampled input fails instead of getting the
  run OOM-killed.
- `--tools a,b,c` builds a subset.
